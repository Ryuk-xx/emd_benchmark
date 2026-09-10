"""Embed the benchmark inputs with the two local open-source models.

Weights download automatically on first run into ./models/ (gitignored) and are
reused afterwards, so the second run starts encoding immediately. Vectors and a
timing log land in ./embeddings/<model>/.

  python src/embed_local.py                      # both models, every input present
  python src/embed_local.py --model qwen3_0.6b   # one model
  python src/embed_local.py --input corpus       # one input
  python src/embed_local.py --batch-size 8       # if VRAM is tight

Inputs it picks up automatically, skipping whatever is absent:
  data/corpus.jsonl                          the 2,038 chunks being searched
  data/queries.jsonl                         golden-set questions
  data/embedding_calibration_testcases.csv   similarity-calibration pairs
  data/coverage_top1_top4_top5.xlsx          Comparison sheet, Question + Expected
"""
import argparse
import csv
import hashlib
import json
import os
import time
import unicodedata

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_CACHE = os.path.join(ROOT, "models")

# Hugging Face must be pointed at the local cache before transformers is imported.
os.makedirs(MODEL_CACHE, exist_ok=True)
os.environ.setdefault("HF_HOME", MODEL_CACHE)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", MODEL_CACHE)

import torch                                            # noqa: E402
from sentence_transformers import SentenceTransformer   # noqa: E402

# Qwen3-Embedding is instruction-tuned: this string is prepended to queries only,
# never to documents, and changing it changes every query vector. Keep it fixed
# for the whole benchmark.
QWEN_TASK = ("Given a Vietnamese customer-service question about telecom services, "
             "retrieve the passages that answer it")

MODELS = {
    "vn_embedding": {
        "hf_id": "AITeamVN/Vietnamese_Embedding",
        "max_seq_length": 2048,
        "query_prompt": None,          # BGE-M3 lineage: no instruction prefix
        "doc_prompt": None,
    },
    "qwen3_0.6b": {
        "hf_id": "Qwen/Qwen3-Embedding-0.6B",
        "max_seq_length": 2048,        # corpus max is 1,201 tokens, so nothing is cut
        "query_prompt": f"Instruct: {QWEN_TASK}\nQuery: ",
        "doc_prompt": None,            # documents are embedded bare
    },
}


def sha(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def nfc(t):
    """The corpus was exported as NFC; every other input must match, or the same
    Vietnamese word tokenizes two different ways."""
    return unicodedata.normalize("NFC", str(t)).strip()


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_coverage_xlsx(path):
    """Read the Comparison sheet and return (case_ids, questions, expected).

    Questions and expected answers are returned separately because they are not the
    same kind of text: a question is a query and takes Qwen3's instruction prefix,
    an expected answer is a statement and must not.
    """
    try:
        import openpyxl
    except ImportError:
        print("  coverage xlsx found but openpyxl is not installed "
              "(pip install openpyxl) - skipping")
        return None

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = "Comparison" if "Comparison" in wb.sheetnames else wb.sheetnames[0]
    rows = list(wb[sheet].iter_rows(values_only=True))
    wb.close()
    if not rows:
        return None

    # Match the header case-insensitively; the file capitalises these.
    header = [str(c).strip().lower() if c is not None else "" for c in rows[0]]
    try:
        i_case = header.index("case")
        i_q = header.index("question")
        i_e = header.index("expected")
    except ValueError:
        print(f"  {os.path.basename(path)}: sheet '{sheet}' has no "
              f"case/question/expected columns - skipping")
        return None

    ids, questions, expected = [], [], []
    for r in rows[1:]:
        if not any(r):
            continue
        q = nfc(r[i_q]) if r[i_q] else ""
        e = nfc(r[i_e]) if r[i_e] else ""
        if not (q and e):
            continue
        ids.append(str(r[i_case]).strip() if r[i_case] else f"row{len(ids)}")
        questions.append(q)
        expected.append(e)
    return ids, questions, expected


def discover_inputs():
    """Collect whichever inputs exist. 'kind' decides whether a prompt is applied."""
    d = os.path.join(ROOT, "data")
    found = {}

    corpus_p = os.path.join(d, "corpus.jsonl")
    ids_p = os.path.join(d, "bench_ids.json")
    if os.path.exists(corpus_p):
        rows = load_jsonl(corpus_p)
        by_id = {r["chunk_id"]: r for r in rows}
        if os.path.exists(ids_p):
            # bench_ids.json fixes the row order shared by every model's matrix;
            # evaluate.py asserts it, so honour it rather than the file order.
            with open(ids_p, encoding="utf-8") as f:
                order = json.load(f)
            missing = [c for c in order if c not in by_id]
            if missing:
                raise SystemExit(f"{len(missing)} ids in bench_ids.json missing from corpus.jsonl")
            rows = [by_id[c] for c in order]
            ids = order
        else:
            ids = [r["chunk_id"] for r in rows]
        found["corpus"] = {"ids": ids, "texts": [r["text"] for r in rows], "kind": "doc"}

    queries_p = os.path.join(d, "queries.jsonl")
    if os.path.exists(queries_p):
        rows = load_jsonl(queries_p)
        found["queries"] = {"ids": [r["query_id"] for r in rows],
                            "texts": [r["text"] for r in rows], "kind": "query"}

    calib_p = os.path.join(d, "embedding_calibration_testcases.csv")
    if os.path.exists(calib_p):
        with open(calib_p, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        ids, texts = [], []
        for r in rows:
            # Both sides of a pair are plain statements, so both are encoded as
            # documents; a query prompt on one side would skew the cosine.
            ids += [f"{r['pair_id']}__a", f"{r['pair_id']}__b"]
            texts += [nfc(r["text_a"]), nfc(r["text_b"])]
        found["calibration"] = {"ids": ids, "texts": texts, "kind": "doc"}

    # Excel leaves a ~$ lock file behind while the workbook is open; skip it.
    cov_p = os.path.join(d, "coverage_top1_top4_top5.xlsx")
    if os.path.exists(cov_p) and not os.path.basename(cov_p).startswith("~$"):
        got = load_coverage_xlsx(cov_p)
        if got:
            case_ids, questions, expected = got
            # Split by kind: the question is a query, the expected answer is not.
            # Both keep the same Case order, so row i lines up across the two files.
            found["coverage_questions"] = {
                "ids": case_ids, "texts": questions, "kind": "query"}
            found["coverage_expected"] = {
                "ids": case_ids, "texts": expected, "kind": "doc"}

    return found


def encode_timed(model, texts, prompt, batch_size, device):
    """Encode in timed batches so the log carries a throughput distribution."""
    out, times = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        v = model.encode(batch, batch_size=batch_size, prompt=prompt,
                         normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=False)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        out.append(v)
        done = min(i + batch_size, len(texts))
        if (i // batch_size) % 10 == 0 or done == len(texts):
            rate = done / sum(times)
            print(f"     {done:6d}/{len(texts)}  {rate:7.1f} items/s  "
                  f"eta {(len(texts) - done) / rate:5.0f}s", flush=True)
    return np.vstack(out).astype(np.float32), times


def measure_latency(model, texts, prompt, device, n=20):
    """Single-item latency after warm-up: the production query path, which batch
    throughput does not predict."""
    for t in texts[:5]:
        model.encode([t], prompt=prompt, normalize_embeddings=True)
    lat = []
    for t in texts[:n]:
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.encode([t], prompt=prompt, normalize_embeddings=True)
        if device == "cuda":
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    return lat


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="all", choices=["all", *MODELS])
    ap.add_argument("--input", default="all",
                    choices=["all", "corpus", "queries", "calibration",
                             "coverage_questions", "coverage_expected", "coverage"],
                    help="'coverage' means both coverage_questions and coverage_expected")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--fp32", dest="fp16", action="store_false", default=True,
                    help="full precision; slower and needs more VRAM")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    print(f"device      : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"precision   : {'fp16' if args.fp16 else 'fp32'}, batch {args.batch_size}")
    print(f"weight cache: {MODEL_CACHE}")
    if device == "cpu":
        print("  no GPU detected - this will work but take much longer")

    ALL_INPUTS = ("corpus", "queries", "calibration",
                  "coverage_questions", "coverage_expected")

    available = discover_inputs()
    if args.input == "all":
        inputs = available
    elif args.input == "coverage":
        inputs = {k: v for k, v in available.items() if k.startswith("coverage_")}
    else:
        inputs = {k: v for k, v in available.items() if k == args.input}
    if not inputs:
        raise SystemExit(f"nothing to embed for --input {args.input} "
                         f"(found: {', '.join(available) or 'none'})")

    print("\ninputs found:")
    for name, d in inputs.items():
        chars = sum(len(t) for t in d["texts"])
        print(f"  {name:20s} {len(d['texts']):6d} items  {chars:,} chars  "
              f"[{d['kind']}]")
    for name in ALL_INPUTS:
        if name not in inputs:
            print(f"  {name:20s} (absent, skipped)")

    todo = list(MODELS) if args.model == "all" else [args.model]
    log = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "device": device,
           "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
           "precision": "fp16" if args.fp16 else "fp32",
           "batch_size": args.batch_size, "qwen_task_description": QWEN_TASK,
           "models": {}}

    for model_name in todo:
        cfg = MODELS[model_name]
        print(f"\n{'=' * 70}\n{model_name}  ({cfg['hf_id']})\n{'=' * 70}")
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        print("  loading (first run downloads the weights)...", flush=True)
        model = SentenceTransformer(
            cfg["hf_id"], device=device, cache_folder=MODEL_CACHE,
            model_kwargs={"torch_dtype": torch.float16} if args.fp16 else {},
        )
        model.max_seq_length = cfg["max_seq_length"]
        load_s = round(time.perf_counter() - t0, 2)
        dim = model.get_sentence_embedding_dimension()
        print(f"  ready in {load_s}s, dim={dim}")

        out_dir = os.path.join(ROOT, "embeddings", model_name)
        os.makedirs(out_dir, exist_ok=True)
        entry = {"hf_id": cfg["hf_id"], "dim": dim, "load_seconds": load_s,
                 "max_seq_length": cfg["max_seq_length"], "inputs": {}}

        for input_name, data in inputs.items():
            prompt = cfg["query_prompt"] if data["kind"] == "query" else cfg["doc_prompt"]
            texts, ids = data["texts"], data["ids"]
            print(f"  -- {input_name}: {len(texts)} items, prompt={prompt!r}")

            # Count with this model's own tokenizer to prove nothing was truncated.
            n_tok = [len(model.tokenizer.encode(t, add_special_tokens=True)) for t in texts]
            truncated = int(sum(n > cfg["max_seq_length"] for n in n_tok))
            print(f"     tokens: total {sum(n_tok):,}  max {max(n_tok)}  "
                  f"truncated: {truncated}")

            X, batch_times = encode_timed(model, texts, prompt, args.batch_size, device)
            total = sum(batch_times)
            lat = measure_latency(model, texts, prompt, device)

            np.save(os.path.join(out_dir, f"{input_name}.npy"), X)
            id_file = "ids.json" if input_name == "corpus" else f"ids_{input_name}.json"
            with open(os.path.join(out_dir, id_file), "w", encoding="utf-8") as f:
                json.dump(ids, f, ensure_ascii=False)

            entry["inputs"][input_name] = {
                "n": len(texts), "shape": list(X.shape), "prompt": prompt,
                "tokens_total": int(sum(n_tok)), "tokens_max": int(max(n_tok)),
                "tokens_truncated": truncated,
                "encode_seconds": round(total, 2),
                "items_per_second": round(len(texts) / total, 1),
                "tokens_per_second": round(sum(n_tok) / total, 1),
                "batch_seconds": {
                    "n_batches": len(batch_times),
                    "mean": round(float(np.mean(batch_times)), 4),
                    "p50": round(float(np.percentile(batch_times, 50)), 4),
                    "p95": round(float(np.percentile(batch_times, 95)), 4),
                    "first": round(batch_times[0], 4),
                },
                "single_item_latency_ms": {
                    "p50": round(float(np.percentile(lat, 50)), 1),
                    "p95": round(float(np.percentile(lat, 95)), 1),
                },
                "text_sha256": {i: sha(t) for i, t in zip(ids, texts)},
            }
            print(f"     done {X.shape} in {total:.1f}s "
                  f"({len(texts) / total:.1f} items/s), "
                  f"1-item p50 {np.percentile(lat, 50):.0f}ms "
                  f"p95 {np.percentile(lat, 95):.0f}ms")

        if device == "cuda":
            entry["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            print(f"  peak VRAM {entry['peak_vram_gb']} GB")

        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"model": model_name, **entry, "dtype": "float32",
                       "already_l2_normalized": True}, f, ensure_ascii=False)

        log["models"][model_name] = entry
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    log["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    log_path = os.path.join(ROOT, "results", "embedding_timing.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print(f"{'model':<16}{'input':<14}{'n':>7}{'sec':>9}{'items/s':>10}"
          f"{'p50 ms':>9}{'p95 ms':>9}")
    for m, e in log["models"].items():
        for i, s in e["inputs"].items():
            print(f"{m:<16}{i:<14}{s['n']:>7}{s['encode_seconds']:>9}"
                  f"{s['items_per_second']:>10}"
                  f"{s['single_item_latency_ms']['p50']:>9}"
                  f"{s['single_item_latency_ms']['p95']:>9}")
    print(f"\ntiming log -> {os.path.relpath(log_path, ROOT)}")

    bad = [(m, i) for m, e in log["models"].items()
           for i, s in e["inputs"].items() if s["tokens_truncated"]]
    if bad:
        print("\nWARNING truncated inputs:", bad)
    if "queries" in inputs and "qwen3_0.6b" in log["models"]:
        p = log["models"]["qwen3_0.6b"]["inputs"]["queries"]["prompt"]
        print("qwen3 query prompt: " + (repr(p) if p else
              "MISSING - scores would understate this model, re-run"))


if __name__ == "__main__":
    main()
