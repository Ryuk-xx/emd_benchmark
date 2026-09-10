"""Embed the benchmark corpus (or queries) with the local open-source models.

Run this on the GPU machine. It needs only this file plus the data/ directory.

  python src/embed_local.py --model vn_embedding --input corpus
  python src/embed_local.py --model qwen3_0.6b  --input corpus

Both models are encoded through sentence-transformers so that pooling, padding side
and normalization follow each model's own configuration rather than our guesswork.
The one thing that is NOT automatic is Qwen3's query instruction: it is applied to
queries only, never to documents, and the exact string is recorded in the manifest.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Qwen3-Embedding is instruction-tuned: the task description materially changes the
# query vector. Keep it fixed across the whole benchmark and record it.
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
        "max_seq_length": 2048,        # corpus max is 1201 tokens; 2048 is ample
        "query_prompt": f"Instruct: {QWEN_TASK}\nQuery: ",
        "doc_prompt": None,            # documents are embedded bare
    },
}


def sha(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--input", default="corpus", choices=["corpus", "queries"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--fp32", dest="fp16", action="store_false")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = MODELS[args.model]
    src = os.path.join(ROOT, "data", f"{args.input}.jsonl")
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]

    if args.input == "corpus":
        # bench_ids.json fixes the row order shared by every model's matrix.
        order = json.load(open(os.path.join(ROOT, "data", "bench_ids.json"), encoding="utf-8"))
        by_id = {r["chunk_id"]: r for r in rows}
        rows = [by_id[c] for c in order]
        ids = order
        prompt = cfg["doc_prompt"]
    else:
        ids = [r["query_id"] for r in rows]
        prompt = cfg["query_prompt"]

    texts = [r["text"] for r in rows]
    print(f"{args.model}: {len(texts)} {args.input} on {args.device} "
          f"({'fp16' if args.fp16 else 'fp32'}), prompt={prompt!r}")

    model = SentenceTransformer(
        cfg["hf_id"], device=args.device,
        model_kwargs={"torch_dtype": torch.float16} if args.fp16 else {},
    )
    model.max_seq_length = cfg["max_seq_length"]

    t0 = time.perf_counter()
    X = model.encode(
        texts, batch_size=args.batch_size, prompt=prompt,
        normalize_embeddings=True,          # cosine == dot product downstream
        convert_to_numpy=True, show_progress_bar=True,
    ).astype(np.float32)
    dt = time.perf_counter() - t0

    # Single-item latency matters for the query path in production; measure it separately.
    warm = texts[: min(20, len(texts))]
    model.encode(warm[:5], prompt=prompt, normalize_embeddings=True)   # warm-up
    lat = []
    for t in warm:
        s = time.perf_counter()
        model.encode([t], prompt=prompt, normalize_embeddings=True)
        lat.append((time.perf_counter() - s) * 1000)

    out = os.path.join(ROOT, "embeddings", args.model)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, f"{args.input}.npy"), X)
    json.dump(ids, open(os.path.join(out, f"ids_{args.input}.json"), "w"), ensure_ascii=False)

    man_path = os.path.join(out, "manifest.json")
    man = json.load(open(man_path, encoding="utf-8")) if os.path.exists(man_path) else {}
    man.update({
        "model": args.model, "hf_id": cfg["hf_id"], "dim": int(X.shape[1]),
        "max_seq_length": cfg["max_seq_length"], "dtype": "float32",
        "already_l2_normalized": True,
        "qwen_task_description": QWEN_TASK if args.model.startswith("qwen") else None,
    })
    man[args.input] = {
        "n": int(X.shape[0]),
        "prompt": prompt,
        "batch_size": args.batch_size,
        "precision": "fp16" if args.fp16 else "fp32",
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "encode_seconds": round(dt, 2),
        "items_per_second": round(len(texts) / dt, 1),
        "single_item_latency_ms": {
            "p50": round(float(np.percentile(lat, 50)), 1),
            "p95": round(float(np.percentile(lat, 95)), 1),
        },
        "text_sha256": {i: sha(t) for i, t in zip(ids, texts)},
    }
    json.dump(man, open(man_path, "w"), ensure_ascii=False)

    print(f"  -> {X.shape} in {dt:.1f}s ({len(texts)/dt:.1f}/s), "
          f"single-item p50 {np.percentile(lat,50):.0f}ms p95 {np.percentile(lat,95):.0f}ms")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
