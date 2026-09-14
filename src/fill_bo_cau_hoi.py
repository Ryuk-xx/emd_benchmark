"""Fill the `embedding` and `top5` sheets of data/bo_cau_hoi_681cbdeb_60.xlsx from
the local-model vectors.

Needs `embeddings/<model>/<mode>/bo_cau_hoi.npy` (from `embed_local.py --input
bo_cau_hoi`) and the corpus vectors for the same model/mode.

  python src/fill_bo_cau_hoi.py                  # writes in place, keeps a .bak.xlsx
  python src/fill_bo_cau_hoi.py --out results/bo_cau_hoi_filled.xlsx
  python src/fill_bo_cau_hoi.py --dry-run        # report only, touch nothing

Sheet conventions follow the rows the workbook already holds for ada / 3-large:

  embedding  one row per (question, config). A vector is split into parts of 512
             floats, each part a JSON list in phan_1..phan_N, with so_phan = N.
  top5       one row per (question, config, rank 1..5) against the 2,038-chunk
             corpus. chunk_id is the Neo4j element id (what the workbook calls
             chunk_id), doc_id and chunk_idx come from corpus.jsonl, text_300 is
             the first 300 characters of the chunk. khop is left blank.

A config is <model>[_<mode>]: qwen3-0.6b_instruct, qwen3-0.6b_no_instruct,
qwen3-vl-2b_instruct, qwen3-vl-2b_no_instruct, vn-embedding. Rows for these
configs are replaced on every run, so the script is safe to re-run; rows for other
configs (ada, 3large, fact10) are never touched.
"""
import argparse
import json
import os
import shutil

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D, E = os.path.join(ROOT, "data"), os.path.join(ROOT, "embeddings")
WORKBOOK = os.path.join(D, "bo_cau_hoi_681cbdeb_60.xlsx")
PART = 512                                   # floats per cell, as the ada/3large rows do
INPUT = "bo_cau_hoi"

# Internal name -> the name written to the workbook. Models with one mode get no
# suffix, matching how text-embedding-ada-002 / -3-large appear there.
CONFIGS = [
    ("qwen3_0.6b", "no_instruct", "qwen3-0.6b_no_instruct"),
    ("qwen3_0.6b", "instruct", "qwen3-0.6b_instruct"),
    ("qwen3_vl_2b", "no_instruct", "qwen3-vl-2b_no_instruct"),
    ("qwen3_vl_2b", "instruct", "qwen3-vl-2b_instruct"),
    ("vn_embedding", "no_instruct", "vn-embedding"),
]
OURS = {c[2] for c in CONFIGS}


def load_vectors(model, mode):
    d = os.path.join(E, model, mode)
    need = [os.path.join(d, f) for f in
            (f"{INPUT}.npy", f"ids_{INPUT}.json", "corpus.npy", "ids.json")]
    if not all(os.path.exists(p) for p in need):
        return None
    Q = np.load(need[0]).astype(np.float32)
    with open(need[1], encoding="utf-8") as f:
        qids = json.load(f)
    C = np.load(need[2]).astype(np.float32)
    with open(need[3], encoding="utf-8") as f:
        cids = json.load(f)
    assert len(qids) == len(Q) and len(cids) == len(C), f"{model}/{mode}: id/vector mismatch"
    Q /= np.clip(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12, None)
    C /= np.clip(np.linalg.norm(C, axis=1, keepdims=True), 1e-12, None)
    return Q, qids, C, cids


def split_parts(vec):
    return [json.dumps([float(x) for x in vec[i:i + PART]])
            for i in range(0, len(vec), PART)]


def rel(p):
    """relpath that does not blow up when p sits on another drive."""
    try:
        return os.path.relpath(p, ROOT)
    except ValueError:
        return p


def sheet_rows(ws):
    rows = list(ws.iter_rows(values_only=True))
    return list(rows[0]), [list(r) for r in rows[1:] if any(v is not None for v in r)]


def rewrite(ws, header, rows):
    """Replace the sheet body below the header, keeping the header cells as they are."""
    if ws.max_row > 1:
        ws.delete_rows(2, ws.max_row - 1)
    for r in rows:
        ws.append(r)
    assert [c.value for c in ws[1]][:len(header)] == header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", default=WORKBOOK)
    ap.add_argument("--out", default=None, help="write here instead of in place")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import openpyxl

    # --- questions, in sheet order -------------------------------------------
    wb = openpyxl.load_workbook(args.workbook)
    qh, qrows = sheet_rows(wb["cau_hoi"])
    i_stt, i_q = qh.index("stt"), qh.index("cau_hoi")
    questions = {str(r[i_stt]): r[i_q] for r in qrows}
    stt_order = [str(r[i_stt]) for r in qrows]
    print(f"{len(questions)} questions in sheet cau_hoi")

    # --- corpus metadata by chunk_id (doc_id::idx) ---------------------------
    corpus = {}
    with open(os.path.join(D, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            corpus[r["chunk_id"]] = r

    # --- compute -------------------------------------------------------------
    emb_rows, top_rows, done = [], [], []
    for model, mode, name in CONFIGS:
        got = load_vectors(model, mode)
        if got is None:
            print(f"  {name:26s} no vectors ({model}/{mode}/{INPUT}.npy), skipped")
            continue
        Q, qids, C, cids = got
        missing = [s for s in stt_order if s not in qids]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} questions have no vector "
                             f"(stt {missing[:5]}); re-run embed_local.py --input {INPUT}")
        qpos = {s: i for i, s in enumerate(qids)}
        S = Q @ C.T                                       # cosine, unit vectors
        k = min(args.topk, C.shape[0])
        for stt in stt_order:
            q = Q[qpos[stt]]
            parts = split_parts(q)
            emb_rows.append([int(stt), questions[stt], name, int(len(q)), len(parts), *parts])
            row = S[qpos[stt]]
            top = np.argpartition(-row, k - 1)[:k]
            top = top[np.argsort(-row[top])]
            for rank, j in enumerate(top, 1):
                ch = corpus[cids[j]]
                top_rows.append([int(stt), questions[stt], name, rank,
                                 round(float(row[j]), 4), ch["element_id"],
                                 str(ch["doc_id"]), int(ch["chunk_index"]), None,
                                 (ch["text"] or "")[:300]])
        done.append(name)
        print(f"  {name:26s} dim {len(q):4d} -> {len(parts)} parts; top-{k} over {len(cids)} chunks")

    if not done:
        raise SystemExit("nothing to fill: no config has bo_cau_hoi vectors yet")

    # --- merge into the sheets ----------------------------------------------
    eh, erows = sheet_rows(wb["embedding"])
    i_model = eh.index("model")
    n_parts_cols = sum(1 for h in eh if str(h).startswith("phan_"))
    max_parts = max(r[4] for r in emb_rows)
    if max_parts > n_parts_cols:
        raise SystemExit(f"embedding sheet has {n_parts_cols} phan_ columns but a vector "
                         f"needs {max_parts}; add columns phan_{n_parts_cols + 1}.. first")
    kept = [r for r in erows if r[i_model] not in OURS]
    replaced = len(erows) - len(kept)
    # pad our rows to the sheet's width so shorter vectors leave trailing cells empty
    width = len(eh)
    ours = [r + [None] * (width - len(r)) for r in emb_rows]
    model_order = {}
    for r in kept + ours:
        model_order.setdefault(r[i_model], len(model_order))
    merged = sorted(kept + ours, key=lambda r: (int(r[0]), model_order[r[i_model]]))

    th, trows = sheet_rows(wb["top5"])
    i_cfg, i_rank = th.index("cau_hinh"), th.index("hang")
    tkept = [r for r in trows if r[i_cfg] not in OURS]
    treplaced = len(trows) - len(tkept)
    cfg_order = {}
    for r in tkept + top_rows:
        cfg_order.setdefault(r[i_cfg], len(cfg_order))
    tmerged = sorted(tkept + top_rows,
                     key=lambda r: (int(r[0]), cfg_order[r[i_cfg]], int(r[i_rank])))

    print(f"\nembedding: {len(kept)} rows kept, {replaced} replaced, "
          f"{len(ours)} written -> {len(merged)} total")
    print(f"top5     : {len(tkept)} rows kept, {treplaced} replaced, "
          f"{len(top_rows)} written -> {len(tmerged)} total")
    print(f"configs  : {', '.join(done)}")

    if args.dry_run:
        print("\ndry run - workbook untouched")
        return

    out = args.out or args.workbook
    if out == args.workbook:
        bak = os.path.splitext(args.workbook)[0] + ".bak.xlsx"
        if not os.path.exists(bak):
            shutil.copyfile(args.workbook, bak)
            print(f"\nbackup -> {rel(bak)}")
    rewrite(wb["embedding"], eh, merged)
    rewrite(wb["top5"], th, tmerged)
    wb.save(out)
    print(f"saved  -> {rel(out)}")


if __name__ == "__main__":
    main()
