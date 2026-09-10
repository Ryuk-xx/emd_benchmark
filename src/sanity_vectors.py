"""Sanity-check the exported ada-002 vectors without needing an API key.

If the stored vectors really encode these chunks, a chunk's nearest neighbour
should land in the same document far more often than chance.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(0)

rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "corpus.jsonl"), encoding="utf-8")]
doc_of = {r["chunk_id"]: r["doc_id"] for r in rows}

X = np.load(os.path.join(ROOT, "embeddings", "ada002", "corpus.npy"))
ids = json.load(open(os.path.join(ROOT, "embeddings", "ada002", "ids.json"), encoding="utf-8"))
docs = np.array([doc_of[i] for i in ids])
print(f"matrix {X.shape}, norms in [{np.linalg.norm(X,axis=1).min():.4f}, "
      f"{np.linalg.norm(X,axis=1).max():.4f}]")

sample = rng.choice(len(ids), size=2000, replace=False)
S = X[sample] @ X.T                      # cosine, vectors already unit-norm
S[np.arange(len(sample)), sample] = -2   # exclude self
nn = S.argmax(axis=1)
same_doc = (docs[nn] == docs[sample]).mean()

# Chance baseline: probability a random other chunk shares the document.
_, counts = np.unique(docs, return_counts=True)
chance = ((counts * (counts - 1)).sum()) / (len(docs) * (len(docs) - 1))
print(f"\nnearest-neighbour lands in same doc: {same_doc:.1%}  (chance {chance:.2%})")

top = np.sort(S, axis=1)[:, -1]
print(f"top-1 cosine: p05={np.percentile(top,5):.3f} p50={np.percentile(top,50):.3f} "
      f"p95={np.percentile(top,95):.3f}")

off = S[S > -2]
print(f"all-pairs cosine spread: p01={np.percentile(off,1):.3f} "
      f"p50={np.percentile(off,50):.3f} p99={np.percentile(off,99):.3f}")
print("  (ada-002 is known to compress cosine into a narrow band; "
      "this is why absolute thresholds do not transfer between models)")

dupe_vec = int((top > 0.9999).sum())
print(f"\nnear-identical vectors in sample: {dupe_vec}")
