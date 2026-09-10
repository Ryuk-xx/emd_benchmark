"""Keep only chunks embedded by BOTH models, and align every matrix to one id order.

The full export stays in data/corpus_full.jsonl; the benchmark set is written to
data/corpus.jsonl. Row i of every embeddings/<model>/corpus.npy then refers to the
same chunk_id, which is what makes a per-query paired comparison valid.
"""
import json
import os
import shutil

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D, E = os.path.join(ROOT, "data"), os.path.join(ROOT, "embeddings")
MODELS = ["ada002", "text3large"]

full_path = os.path.join(D, "corpus_full.jsonl")
if not os.path.exists(full_path):
    shutil.move(os.path.join(D, "corpus.jsonl"), full_path)

rows = [json.loads(l) for l in open(full_path, encoding="utf-8")]
print(f"full corpus: {len(rows)} chunks")

# Intersect the id sets of every model we already have vectors for.
have = {m: set(json.load(open(os.path.join(E, m, "ids.json"), encoding="utf-8"))) for m in MODELS}
for m in MODELS:
    print(f"  {m:12s} {len(have[m]):6d}")
keep = set.intersection(*have.values())
print(f"\nembedded by ALL {len(MODELS)} models: {len(keep)}")
print(f"dropped (single-model only): {len(rows) - len(keep)}")

# One canonical order, stable and independent of any model's export order.
kept = sorted((r for r in rows if r["chunk_id"] in keep),
              key=lambda r: (int(r["doc_id"]), int(r["chunk_index"])))
order = [r["chunk_id"] for r in kept]
pos = {c: i for i, c in enumerate(order)}

with open(os.path.join(D, "corpus.jsonl"), "w", encoding="utf-8") as f:
    for r in kept:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

for m in MODELS:
    d = os.path.join(E, m)
    X = np.load(os.path.join(d, "corpus.npy"))
    ids = json.load(open(os.path.join(d, "ids.json"), encoding="utf-8"))
    idx = np.array([i for i, c in enumerate(ids) if c in keep])
    idx = idx[np.argsort([pos[ids[i]] for i in idx])]      # reorder to canonical
    Xk = X[idx]
    got = [ids[i] for i in idx]
    assert got == order, f"{m}: alignment failed"

    if not os.path.exists(os.path.join(d, "corpus_full.npy")):
        shutil.move(os.path.join(d, "corpus.npy"), os.path.join(d, "corpus_full.npy"))
        shutil.move(os.path.join(d, "ids.json"), os.path.join(d, "ids_full.json"))
    np.save(os.path.join(d, "corpus.npy"), Xk)
    json.dump(order, open(os.path.join(d, "ids.json"), "w"), ensure_ascii=False)

    man_path = os.path.join(d, "manifest.json")
    man = json.load(open(man_path, encoding="utf-8"))
    man["n"] = int(Xk.shape[0])
    man["filtered"] = "chunks embedded by all models only"
    man["text_sha256"] = {c: man["text_sha256"][c] for c in order}
    json.dump(man, open(man_path, "w"), ensure_ascii=False)
    print(f"  {m:12s} {X.shape} -> {Xk.shape}  aligned OK")

json.dump(order, open(os.path.join(D, "bench_ids.json"), "w"), ensure_ascii=False)
print(f"\ncanonical order -> data/bench_ids.json ({len(order)} ids)")
