"""Retrieve the five most similar index entries for the coverage questions.

The question vectors and index vectors must already exist under
``embeddings/<model>/<mode>/``.  The output has one row per question, model,
mode, and rank, including chunk metadata and embedding/retrieval timings.

Two indexes to search, chosen with ``--index``:

  corpus   the raw chunk text, ``corpus.npy`` (default)
  fact     one vector per extracted fact, ``fact.npy``, keyed <chunk_id>#<fact_id>

With the fact index a question is scored against every individual fact, and the
hits are then folded back to chunks: a chunk's score is the best score among its
facts (its peak), chunks are ranked by peak, and each row reports the peak fact in
``fact_id`` / ``fact_text`` plus ``n_facts_top50``, how many of that chunk's facts
sit in the question's 50 best facts.  Comparing the two runs on the same questions
shows whether searching over extracted facts finds the right chunk more often
than searching over the chunk itself.

Examples::

    python src/retrieve_coverage_top5.py
    python src/retrieve_coverage_top5.py --index fact
    python src/retrieve_coverage_top5.py --index fact --coverage-name bo_sung \
        --coverage-file data/coverage_top1_top4_top5_bo_sung_60_cau.xlsx
    python src/retrieve_coverage_top5.py --models qwen3_0.6b vn_embedding \
        --modes no_instruct instruct --output results/coverage_top5.csv
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from fact_xlsx import chunk_of, load_facts


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
EMBEDDINGS = os.path.join(ROOT, "embeddings")
DEFAULT_OUTPUT = os.path.join(ROOT, "results", "coverage_top5.csv")
FACT_XLSX = os.path.join(DATA, "chunk_va_fact_500_bai.xlsx")


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_questions(path):
    """Return question text keyed by the IDs written by embed_local.py."""
    try:
        import openpyxl
    except ImportError:
        print("warning: openpyxl is not installed; question text will use question IDs")
        return {}

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet_name = "Comparison" if "Comparison" in workbook.sheetnames else workbook.sheetnames[0]
    rows = list(workbook[sheet_name].iter_rows(values_only=True))
    workbook.close()
    if not rows:
        raise SystemExit(f"no rows found in {path}")

    header = [str(value).strip().lower() if value is not None else "" for value in rows[0]]
    try:
        case_col = header.index("case")
        question_col = header.index("question")
    except ValueError as exc:
        raise SystemExit("coverage workbook must contain Case and Question columns") from exc

    questions = {}
    for row in rows[1:]:
        if not any(row):
            continue
        case_id = str(row[case_col]).strip() if row[case_col] is not None else ""
        question = str(row[question_col]).strip() if row[question_col] is not None else ""
        if case_id:
            questions[case_id] = question
    return questions


def load_fact_lookup(path):
    """fact key -> (fact_id, sentence), matching ids_fact.json."""
    facts = load_facts(path)
    if not facts:
        print(f"warning: no facts read from {path}; fact_text will be blank")
        return {}
    return {f["id"]: (f["fact_id"] or "", f["text"]) for f in facts}


def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def get_manifest(model_dir, mode):
    candidates = (
        os.path.join(model_dir, "manifest.json"),
        os.path.join(model_dir, mode, "manifest.json"),
    )
    for path in candidates:
        if os.path.exists(path):
            return load_json(path), path
    return {}, ""


def embedding_timing(manifest, mode, input_name):
    details = manifest.get("modes", {}).get(mode, {})
    corpus = details.get("corpus", {})
    questions = details.get(input_name, {})
    return {
        "embedding_corpus_seconds": corpus.get("encode_seconds", ""),
        "embedding_questions_seconds": questions.get("encode_seconds", ""),
        "embedding_corpus_items_per_second": corpus.get("items_per_second", ""),
        "embedding_questions_items_per_second": questions.get("items_per_second", ""),
        "embedding_question_prompt": questions.get("prompt", ""),
    }


def retrieve_cpu(questions, corpus, topk):
    scores = questions @ corpus.T
    top_indices = np.argpartition(-scores, topk - 1, axis=1)[:, :topk]
    ordered = np.empty_like(top_indices)
    ordered_scores = np.empty_like(top_indices, dtype=np.float32)
    for row in range(scores.shape[0]):
        indices = top_indices[row]
        order = np.argsort(-scores[row, indices], kind="stable")
        ordered[row] = indices[order]
        ordered_scores[row] = scores[row, indices[order]]
    return ordered, ordered_scores


def retrieve_gpu(questions, corpus, topk, batch_size):
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("CUDA retrieval requires PyTorch; use --device cpu otherwise") from exc

    corpus_tensor = torch.from_numpy(corpus).cuda()
    all_indices, all_scores = [], []
    for start in range(0, len(questions), batch_size):
        query_tensor = torch.from_numpy(questions[start:start + batch_size]).cuda()
        scores = query_tensor @ corpus_tensor.T
        values, indices = torch.topk(scores, k=topk, dim=1, largest=True, sorted=True)
        all_indices.append(indices.cpu().numpy())
        all_scores.append(values.cpu().numpy())
    return np.concatenate(all_indices), np.concatenate(all_scores)


def load_pair(model, mode, input_name, index="corpus"):
    """Load (index vectors, question vectors, index ids, question ids).

    index="fact" searches fact.npy instead of corpus.npy; both carry chunk ids.
    """
    directory = os.path.join(EMBEDDINGS, model, mode)
    index_npy = "corpus.npy" if index == "corpus" else f"{index}.npy"
    index_ids = "ids.json" if index == "corpus" else f"ids_{index}.json"
    paths = {
        "corpus": os.path.join(directory, index_npy),
        "questions": os.path.join(directory, f"{input_name}.npy"),
        "corpus_ids": os.path.join(directory, index_ids),
        "question_ids": os.path.join(directory, f"ids_{input_name}.json"),
    }
    missing = [name for name, path in paths.items() if not os.path.exists(path)]
    if missing:
        return None, ", ".join(missing)

    corpus = normalize(np.asarray(np.load(paths["corpus"]), dtype=np.float32))
    questions = normalize(np.asarray(np.load(paths["questions"]), dtype=np.float32))
    corpus_ids = load_json(paths["corpus_ids"])
    question_ids = load_json(paths["question_ids"])
    if corpus.ndim != 2 or questions.ndim != 2 or corpus.shape[1] != questions.shape[1]:
        raise ValueError(f"{model}/{mode}: incompatible vector shapes {corpus.shape} and {questions.shape}")
    if len(corpus_ids) != len(corpus) or len(question_ids) != len(questions):
        raise ValueError(f"{model}/{mode}: vector count does not match its ID file")
    return (corpus, questions, corpus_ids, question_ids), ""


def resolve_device(requested):
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        if requested == "cuda":
            raise SystemExit("--device cuda requires PyTorch")
        return "cpu"
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise SystemExit("--device cuda was requested but CUDA is unavailable")
        return "cpu"
    return "cuda"


FACT_POOL = 50      # facts considered per question before folding to chunks


def retrieve_pair(pair, device, topk, batch_size, index="corpus"):
    """Top-k index entries per question.

    For the corpus index that is simply the k best chunks. For the fact index the
    k best *chunks* are found by taking the FACT_POOL best facts, grouping them by
    the chunk each fact came from, scoring a chunk by its best fact (peak), and
    ranking chunks by peak. Each hit then carries the peak fact and how many of
    the chunk's facts were in the pool.
    """
    corpus, questions, corpus_ids, question_ids = pair
    started = time.perf_counter()
    if index != "fact":
        actual_topk = min(topk, len(corpus_ids))
        if device == "cuda":
            indices, scores = retrieve_gpu(questions, corpus, actual_topk, batch_size)
        else:
            indices, scores = retrieve_cpu(questions, corpus, actual_topk)
        hits = [[(corpus_ids[int(i)], float(sc), None, 0)
                 for i, sc in zip(indices[q], scores[q])]
                for q in range(len(question_ids))]
        return hits, question_ids, time.perf_counter() - started, actual_topk

    pool = min(max(FACT_POOL, topk), len(corpus_ids))
    if device == "cuda":
        indices, scores = retrieve_gpu(questions, corpus, pool, batch_size)
    else:
        indices, scores = retrieve_cpu(questions, corpus, pool)
    hits = []
    for q in range(len(question_ids)):
        best, count = {}, {}
        for i, sc in zip(indices[q], scores[q]):          # already best-first
            key = corpus_ids[int(i)]
            chunk = chunk_of(key)
            count[chunk] = count.get(chunk, 0) + 1
            if chunk not in best:
                best[chunk] = (float(sc), key)             # first seen = peak
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:topk]
        hits.append([(chunk, sc, key, count[chunk]) for chunk, (sc, key) in ranked])
    return hits, question_ids, time.perf_counter() - started, topk


def write_pair_rows(writer, model, mode, pair_result, question_text, corpus_rows,
                    manifest, manifest_path, device, input_name, index="corpus",
                    fact_lookup=None):
    hits, question_ids, retrieval_seconds, topk = pair_result
    timing = embedding_timing(manifest, mode, input_name)
    fact_lookup = fact_lookup or {}
    for question_row, question_id in enumerate(question_ids):
        for rank, (chunk_id, score, fact_key, n_facts) in enumerate(hits[question_row], 1):
            chunk = corpus_rows.get(chunk_id, {})
            fact_id, fact_text = fact_lookup.get(fact_key, ("", "")) if fact_key else ("", "")
            writer.writerow({
                "model": model,
                "mode": mode,
                "question_id": question_id,
                "question": question_text.get(question_id, question_id),
                "rank": rank,
                "index": index,
                "chunk_id": chunk_id,
                "score": f"{float(score):.8f}",
                "fact_id": fact_id,
                "fact_text": fact_text,
                "n_facts_top50": n_facts if index == "fact" else "",
                "title": chunk.get("title", ""),
                "text": chunk.get("text", ""),
                "doc_id": chunk.get("doc_id", ""),
                "chunk_index": chunk.get("chunk_index", ""),
                "businesses": json.dumps(chunk.get("businesses", []), ensure_ascii=False),
                "embedding_manifest": manifest_path,
                **timing,
                "retrieval_device": device,
                "retrieval_seconds": f"{retrieval_seconds:.6f}",
            })
    return sum(len(h) for h in hits), len(question_ids), retrieval_seconds, topk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+",
                        default=["qwen3_0.6b", "qwen3_vl_2b", "vn_embedding"])
    parser.add_argument("--modes", nargs="+", default=["no_instruct", "instruct"])
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--coverage-file", default=os.path.join(
                        DATA, "coverage_top1_top4_top5.xlsx"))
    parser.add_argument("--coverage-name", default="coverage",
                        help="use bo_sung for coverage_questions_bo_sung.npy")
    parser.add_argument("--index", choices=("corpus", "fact"), default="corpus",
                        help="search the raw chunks (corpus.npy) or the per-chunk "
                             "fact blocks (fact.npy); both are keyed by chunk id")
    parser.add_argument("--fact-file", default=FACT_XLSX,
                        help="workbook the fact_text column is read from")
    args = parser.parse_args()
    if args.topk < 1 or args.batch_size < 1:
        parser.error("--topk and --batch-size must be positive")

    device = resolve_device(args.device)

    input_name = "coverage_questions" if args.coverage_name == "coverage" \
        else f"coverage_questions_{args.coverage_name}"
    question_text = load_questions(args.coverage_file)
    fact_lookup = load_fact_lookup(args.fact_file) if args.index == "fact" else {}

    # Derive the output name from what was searched, so a bo_sung run or a fact-
    # index run cannot silently overwrite the default corpus run.
    if args.output == DEFAULT_OUTPUT:
        stem = f"coverage_top{args.topk}"
        if args.coverage_name != "coverage":
            stem += f"_{args.coverage_name}"
        if args.index != "corpus":
            stem += f"_{args.index}"
        args.output = os.path.join(ROOT, "results", stem + ".csv")
    corpus_rows = {row["chunk_id"]: row for row in (
        json.loads(line) for line in open(os.path.join(DATA, "corpus.jsonl"), encoding="utf-8")
    )}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fieldnames = [
        "model", "mode", "question_id", "question", "rank", "index", "chunk_id",
        "score", "fact_id", "fact_text", "n_facts_top50",
        "title", "text", "doc_id", "chunk_index", "businesses", "embedding_manifest",
        "embedding_corpus_seconds", "embedding_questions_seconds",
        "embedding_corpus_items_per_second", "embedding_questions_items_per_second",
        "embedding_question_prompt", "retrieval_device", "retrieval_seconds",
    ]
    total_rows = 0
    with open(args.output, "w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for model in args.models:
            model_dir = os.path.join(EMBEDDINGS, model)
            manifest, manifest_path = get_manifest(model_dir, args.modes[0])
            for mode in args.modes:
                pair, missing = load_pair(model, mode, input_name, args.index)
                if pair is None:
                    print(f"skip {model}/{mode}: missing {missing}")
                    continue
                pair_result = retrieve_pair(pair, device, args.topk, args.batch_size,
                                            args.index)
                rows, question_count, retrieval_seconds, topk = write_pair_rows(
                    writer, model, mode, pair_result, question_text, corpus_rows,
                    manifest, manifest_path, device, input_name, args.index, fact_lookup)
                total_rows += rows
                unit = "facts, folded to chunks" if args.index == "fact" else "chunks"
                print(f"{model}/{mode}: {question_count} questions over "
                      f"{len(pair[2])} {unit}, top-{topk}, "
                      f"{retrieval_seconds:.3f}s on {device}")
    print(f"wrote {total_rows} rows to {args.output}")


if __name__ == "__main__":
    main()