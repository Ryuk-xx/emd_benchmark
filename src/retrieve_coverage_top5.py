"""Retrieve the five most similar corpus chunks for coverage questions.

The coverage question vectors and corpus vectors must already exist under
``embeddings/<model>/<mode>/``.  The output has one row per question, model,
mode, and rank, including chunk metadata and embedding/retrieval timings.

Examples::

    python src/retrieve_coverage_top5.py
    python src/retrieve_coverage_top5.py --device cpu
    python src/retrieve_coverage_top5.py --models qwen3_0.6b vn_embedding \
        --modes no_instruct instruct --output results/coverage_top5.csv
"""
import argparse
import csv
import json
import os
import time

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
EMBEDDINGS = os.path.join(ROOT, "embeddings")
DEFAULT_OUTPUT = os.path.join(ROOT, "results", "coverage_top5.csv")


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


def embedding_timing(manifest, mode):
    details = manifest.get("modes", {}).get(mode, {})
    corpus = details.get("corpus", {})
    questions = details.get("coverage_questions", {})
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


def load_pair(model, mode):
    directory = os.path.join(EMBEDDINGS, model, mode)
    paths = {
        "corpus": os.path.join(directory, "corpus.npy"),
        "questions": os.path.join(directory, "coverage_questions.npy"),
        "corpus_ids": os.path.join(directory, "ids.json"),
        "question_ids": os.path.join(directory, "ids_coverage_questions.json"),
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


def retrieve_pair(pair, device, topk, batch_size):
    corpus, questions, corpus_ids, question_ids = pair
    actual_topk = min(topk, len(corpus_ids))
    started = time.perf_counter()
    if device == "cuda":
        indices, scores = retrieve_gpu(questions, corpus, actual_topk, batch_size)
    else:
        indices, scores = retrieve_cpu(questions, corpus, actual_topk)
    return (corpus_ids, question_ids, indices, scores,
            time.perf_counter() - started, actual_topk)


def write_pair_rows(writer, model, mode, pair_result, question_text, corpus_rows,
                    manifest, manifest_path, device):
    corpus_ids, question_ids, indices, scores, retrieval_seconds, topk = pair_result
    timing = embedding_timing(manifest, mode)
    for question_row, question_id in enumerate(question_ids):
        for rank, (index, score) in enumerate(
                zip(indices[question_row], scores[question_row]), 1):
            chunk_id = corpus_ids[int(index)]
            chunk = corpus_rows.get(chunk_id, {})
            writer.writerow({
                "model": model,
                "mode": mode,
                "question_id": question_id,
                "question": question_text.get(question_id, question_id),
                "rank": rank,
                "chunk_id": chunk_id,
                "score": f"{float(score):.8f}",
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
    return len(question_ids) * topk, len(question_ids), retrieval_seconds, topk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["qwen3_0.6b", "vn_embedding"])
    parser.add_argument("--modes", nargs="+", default=["no_instruct", "instruct"])
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.topk < 1 or args.batch_size < 1:
        parser.error("--topk and --batch-size must be positive")

    device = resolve_device(args.device)

    questions_path = os.path.join(DATA, "coverage_top1_top4_top5.xlsx")
    question_text = load_questions(questions_path)
    corpus_rows = {row["chunk_id"]: row for row in (
        json.loads(line) for line in open(os.path.join(DATA, "corpus.jsonl"), encoding="utf-8")
    )}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fieldnames = [
        "model", "mode", "question_id", "question", "rank", "chunk_id", "score",
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
                pair, missing = load_pair(model, mode)
                if pair is None:
                    print(f"skip {model}/{mode}: missing {missing}")
                    continue
                pair_result = retrieve_pair(pair, device, args.topk, args.batch_size)
                rows, question_count, retrieval_seconds, topk = write_pair_rows(
                    writer, model, mode, pair_result, question_text, corpus_rows,
                    manifest, manifest_path, device)
                total_rows += rows
                print(f"{model}/{mode}: {question_count} questions, top-{topk}, "
                      f"{retrieval_seconds:.3f}s on {device}")
    print(f"wrote {total_rows} rows to {args.output}")


if __name__ == "__main__":
    main()