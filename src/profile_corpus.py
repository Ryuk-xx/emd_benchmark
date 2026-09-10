"""Profile the exported corpus: token budgets, model context limits, duplicates, junk."""
import collections
import json
import os
import unicodedata

import numpy as np
import tiktoken

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
enc = tiktoken.get_encoding("cl100k_base")

# Context limits that matter for this benchmark.
LIMITS = {"ada-002 / 3-large (8191)": 8191, "Vietnamese_Embedding (2048)": 2048}

rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "corpus.jsonl"), encoding="utf-8")]
print(f"chunks: {len(rows)}")

ids = [r["chunk_id"] for r in rows]
dup_ids = [k for k, v in collections.Counter(ids).items() if v > 1]
texts = [r["text"] for r in rows]
dup_text = [k for k, v in collections.Counter(r["sha256"] for r in rows).items() if v > 1]
print(f"chunk_id collisions: {len(dup_ids)}   duplicate texts (by sha256): {len(dup_text)}")

toks = np.array([len(enc.encode(t)) for t in texts])
chars = np.array([r["n_chars"] for r in rows])
print(f"\ntokens  total={toks.sum():,}  mean={toks.mean():.0f}  "
      f"p50={np.percentile(toks,50):.0f} p90={np.percentile(toks,90):.0f} "
      f"p99={np.percentile(toks,99):.0f} max={toks.max():,}")
print(f"chars/token ratio: {chars.sum()/toks.sum():.2f}")

print("\n-- chunks exceeding model context --")
for name, lim in LIMITS.items():
    over = int((toks > lim).sum())
    lost = int(np.clip(toks - lim, 0, None).sum())
    print(f"  {name:32s} {over:6d} chunks ({over/len(toks)*100:5.2f}%)  "
          f"{lost:,} tokens truncated")

print("\n-- junk / degenerate chunks --")
for label, mask in [
    ("empty after strip", np.array([not t.strip() for t in texts])),
    ("< 20 chars", chars < 20),
    ("< 50 chars", chars < 50),
    ("< 10 tokens", toks < 10),
]:
    print(f"  {label:20s} {int(mask.sum()):6d}")

print("\n-- unicode form --")
nfc = sum(1 for t in texts if unicodedata.is_normalized("NFC", t))
print(f"  already NFC: {nfc}/{len(texts)}  (export normalized the rest)")

print("\n-- re-embedding cost, full corpus (USD) --")
# Published list prices per 1M input tokens.
for model, price in [("text-embedding-3-large", 0.13), ("text-embedding-3-small", 0.02),
                     ("text-embedding-ada-002", 0.10)]:
    billable = int(np.clip(toks, 0, 8191).sum())
    print(f"  {model:24s} {billable:12,d} tok  ->  ${billable/1e6*price:6.2f}")

print("\n-- coverage of existing embeddings --")
for name in ("ada002", "text3large"):
    m = json.load(open(os.path.join(ROOT, "embeddings", name, "manifest.json"), encoding="utf-8"))
    print(f"  {name:12s} n={m['n']:6d} dim={m['dim']:5d} normalized={m['already_l2_normalized']}")

# Does the stored embedding actually correspond to the exported text?
sha_by_id = {r["chunk_id"]: r["sha256"] for r in rows}
for name in ("ada002", "text3large"):
    m = json.load(open(os.path.join(ROOT, "embeddings", name, "manifest.json"), encoding="utf-8"))
    mismatch = sum(1 for cid, h in m["text_sha256"].items() if sha_by_id.get(cid) != h)
    print(f"  {name:12s} manifest sha vs corpus sha mismatches: {mismatch}")
