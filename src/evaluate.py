"""Score every model on the golden set and write the comparison table.

Design choices that keep the comparison honest:
  * exact (flat) search, never ANN, so no recall is lost to the index
  * cosine on L2-normalized vectors, which for these matrices is a plain dot product
  * one shared row order (data/bench_ids.json) so per-query results are paired
  * BM25 and an RRF hybrid always in the table, because a dense model that cannot
    beat BM25 on this corpus is not worth deploying
  * bootstrap confidence intervals, and paired bootstrap for model-vs-model deltas,
    since a one-point gap over a few hundred queries is usually noise

  python src/evaluate.py --models ada002 text3large vn_embedding qwen3_0.6b
"""
import argparse
import collections
import json
import math
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D, E, R = (os.path.join(ROOT, x) for x in ("data", "embeddings", "results"))
KS = (1, 3, 5, 10, 20, 50, 100)


# ---------------------------------------------------------------- data loading

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def load_inputs():
    corpus = load_jsonl(os.path.join(D, "corpus.jsonl"))
    order = json.load(open(os.path.join(D, "bench_ids.json"), encoding="utf-8"))
    queries = load_jsonl(os.path.join(D, "queries.jsonl"))
    qrels = collections.defaultdict(set)
    for line in open(os.path.join(D, "qrels.tsv"), encoding="utf-8"):
        qid, cid, rel = line.rstrip("\n").split("\t")
        if int(rel) > 0:
            qrels[qid].add(cid)
    return corpus, order, queries, qrels


# ---------------------------------------------------------------- metrics

def per_query_metrics(ranked_ids, gold):
    """ranked_ids: chunk ids best-first. gold: set of relevant chunk ids."""
    rel = [1 if c in gold else 0 for c in ranked_ids]
    out = {}
    for k in KS:
        out[f"recall@{k}"] = min(sum(rel[:k]), len(gold)) / len(gold)
    rr = 0.0
    for i, r in enumerate(rel[:10], 1):
        if r:
            rr = 1.0 / i
            break
    out["mrr@10"] = rr
    dcg = sum(r / math.log2(i + 1) for i, r in enumerate(rel[:10], 1))
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold), 10) + 1))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    return out


def boot_ci(vals, n=1000, seed=0):
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_boot(a, b, n=1000, seed=0):
    """CI of mean(a) - mean(b), resampling queries jointly so the pairing is kept."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
    return float(a.mean() - b.mean()), float(lo), float(hi), bool(lo > 0 or hi < 0)


# ---------------------------------------------------------------- retrievers

def dense_ranking(model, order, queries, topk):
    """Returns {query_id: [chunk_id, ...]} or None when the model's vectors are missing."""
    d = os.path.join(E, model)
    cp, qp = os.path.join(d, "corpus.npy"), os.path.join(d, "queries.npy")
    if not (os.path.exists(cp) and os.path.exists(qp)):
        return None
    C = np.load(cp)
    Q = np.load(qp)
    qids = json.load(open(os.path.join(d, "ids_queries.json"), encoding="utf-8"))
    cids = json.load(open(os.path.join(d, "ids.json"), encoding="utf-8"))
    assert cids == order, f"{model}: corpus vectors are not in bench_ids order"

    # Vectors are stored L2-normalized, so a dot product is the cosine.
    C = C / np.clip(np.linalg.norm(C, axis=1, keepdims=True), 1e-12, None)
    Q = Q / np.clip(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12, None)

    want = {q["query_id"] for q in queries}
    out, scores = {}, {}
    k = min(topk, C.shape[0])
    for i in range(0, len(qids), 512):
        S = Q[i:i + 512] @ C.T
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        for row, qid in enumerate(qids[i:i + 512]):
            if qid not in want:
                continue
            top = part[row][np.argsort(-S[row, part[row]])]
            out[qid] = [order[j] for j in top]
            scores[qid] = S[row, top]
    return out, scores


class BM25:
    """Plain BM25 over whitespace/word tokens. No word segmentation: the models in
    this benchmark use sentencepiece, so segmenting here would compare unlike things."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.toks = [re.findall(r"\w+", d.lower(), flags=re.UNICODE) for d in docs]
        self.len = np.array([len(t) for t in self.toks], dtype=float)
        self.avg = self.len.mean() if len(self.len) else 0.0
        self.tf, df = [], collections.Counter()
        for t in self.toks:
            c = collections.Counter(t)
            self.tf.append(c)
            df.update(c.keys())
        N = len(docs)
        self.idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.post = collections.defaultdict(list)
        for i, c in enumerate(self.tf):
            for w, f in c.items():
                self.post[w].append((i, f))

    def score(self, query):
        s = np.zeros(len(self.toks))
        for w in re.findall(r"\w+", query.lower(), flags=re.UNICODE):
            if w not in self.post:
                continue
            idf = self.idf[w]
            for i, f in self.post[w]:
                denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                s[i] += idf * f * (self.k1 + 1) / denom
        return s


def bm25_ranking(corpus, order, queries, topk):
    by_id = {r["chunk_id"]: r for r in corpus}
    bm = BM25([by_id[c]["text"] for c in order])
    out, scores = {}, {}
    k = min(topk, len(order))
    for q in queries:
        s = bm.score(q["text"])
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        out[q["query_id"]] = [order[j] for j in top]
        scores[q["query_id"]] = s[top]
    return out, scores


def rrf(rankings, k=60):
    """Reciprocal-rank fusion over several {qid: [chunk_id,...]} rankings."""
    fused = {}
    for qid in rankings[0]:
        acc = collections.defaultdict(float)
        for r in rankings:
            for rank, cid in enumerate(r.get(qid, []), 1):
                acc[cid] += 1.0 / (k + rank)
        fused[qid] = [c for c, _ in sorted(acc.items(), key=lambda kv: -kv[1])]
    return fused


# ---------------------------------------------------------------- reporting

def summarize(name, ranking, queries, qrels):
    rows = {}
    for q in queries:
        qid = q["query_id"]
        if qid not in ranking or not qrels.get(qid):
            continue
        rows[qid] = per_query_metrics(ranking[qid], qrels[qid])
    return {"model": name, "per_query": rows}


def table(results, queries, metric, subset=None, label=""):
    qmeta = {q["query_id"]: q for q in queries}
    sel = [q for q in qmeta if subset is None or subset(qmeta[q])]
    print(f"\n### {metric}{(' | ' + label) if label else ''}  (n={len(sel)} queries)")
    print(f"{'model':<22}{'mean':>8}{'95% CI':>20}")
    ranked = []
    for r in results:
        vals = [r["per_query"][q][metric] for q in sel if q in r["per_query"]]
        if not vals:
            continue
        m, lo, hi = boot_ci(vals)
        ranked.append((m, r["model"], lo, hi))
    for m, name, lo, hi in sorted(ranked, reverse=True):
        print(f"{name:<22}{m:>8.4f}{f'[{lo:.4f}, {hi:.4f}]':>20}")
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["ada002", "text3large", "vn_embedding", "qwen3_0.6b"])
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--primary", default="ndcg@10")
    args = ap.parse_args()

    corpus, order, queries, qrels = load_inputs()
    print(f"corpus {len(order)} chunks | queries {len(queries)} | judged {len(qrels)}")

    rankings, results = {}, []
    for m in args.models:
        got = dense_ranking(m, order, queries, args.topk)
        if got is None:
            print(f"  {m}: no vectors yet, skipped")
            continue
        rankings[m] = got[0]
        results.append(summarize(m, got[0], queries, qrels))
        print(f"  {m}: ranked")

    rankings["bm25"] = bm25_ranking(corpus, order, queries, args.topk)[0]
    results.append(summarize("bm25", rankings["bm25"], queries, qrels))
    print("  bm25: ranked")

    for m in list(rankings):
        if m != "bm25":
            name = f"hybrid({m}+bm25)"
            rankings[name] = rrf([rankings[m], rankings["bm25"]])
            results.append(summarize(name, rankings[name], queries, qrels))

    orig = lambda q: q.get("variant") == "orig"
    table(results, queries, args.primary, orig, "original queries")
    for k in (1, 5, 10, 50):
        table(results, queries, f"recall@{k}", orig, "original queries")

    print("\n\n## Robustness: change vs original queries")
    variants = sorted({q.get("variant") for q in queries} - {"orig"})
    base = {}
    qmeta = {q["query_id"]: q for q in queries}
    for r in results:
        vals = [v[args.primary] for q, v in r["per_query"].items() if orig(qmeta[q])]
        base[r["model"]] = np.mean(vals) if vals else float("nan")
    w = max(13, max(len(v) for v in variants) + 2)
    header = f"{'model':<22}" + "".join(f"{v:>{w}}" for v in variants)
    print(header)
    for r in sorted(results, key=lambda r: -base[r["model"]]):
        line = f"{r['model']:<22}"
        for v in variants:
            vals = [x[args.primary] for q, x in r["per_query"].items()
                    if qmeta[q].get("variant") == v]
            line += (f"{(np.mean(vals) / base[r['model']] - 1) * 100:>{w-1}.1f}%"
                     if vals else f"{'-':>{w}}")
        print(line)

    print("\n\n## Paired comparison on original queries (primary metric)")
    sel = [q["query_id"] for q in queries if orig(q)]
    dense = [r for r in results if not r["model"].startswith("hybrid")]
    best = max(dense, key=lambda r: base[r["model"]])
    for r in dense:
        if r["model"] == best["model"]:
            continue
        common = [q for q in sel if q in best["per_query"] and q in r["per_query"]]
        d, lo, hi, sig = paired_boot([best["per_query"][q][args.primary] for q in common],
                                     [r["per_query"][q][args.primary] for q in common])
        verdict = "significant" if sig else "NOT significant"
        print(f"  {best['model']} - {r['model']:<18} {d:+.4f}  "
              f"[{lo:+.4f}, {hi:+.4f}]  {verdict}")

    print("\n\n## Breakdown by business")
    for biz in sorted({q.get("business") for q in queries if orig(q)}):
        table(results, queries, args.primary,
              lambda q, b=biz: orig(q) and q.get("business") == b, biz)

    os.makedirs(R, exist_ok=True)
    json.dump({r["model"]: r["per_query"] for r in results},
              open(os.path.join(R, "per_query.json"), "w"), ensure_ascii=False)
    print(f"\nper-query scores -> results/per_query.json")


if __name__ == "__main__":
    main()
