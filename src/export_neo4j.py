"""Export chunk corpus + existing OpenAI embeddings from Neo4j for the embedding benchmark.

Produces, under data/ and embeddings/:
  data/corpus.jsonl          one row per chunk: ids, text, metadata, sha256 of text
  embeddings/ada002/{corpus.npy,ids.json,manifest.json}
  embeddings/text3large/{corpus.npy,ids.json,manifest.json}

The manifest records the sha256 of every embedded text so that any later
re-embedding can be proven to have run on byte-identical input.
"""
import argparse
import hashlib
import json
import os
import unicodedata

import numpy as np
from neo4j import GraphDatabase

URI = os.environ.get("NEO4J_URI")
USER = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD")
if not (URI and PASSWORD):
    raise SystemExit("set NEO4J_URI and NEO4J_PASSWORD (see .env.example)")
AUTH = (USER, PASSWORD)

AI_ID = "681cbdeb-d0d5-4441-9507-f6f7fa5cb580"
LABEL = f"{AI_ID.replace('-', '_')}_chunking_ai"

META_KEYS = [
    "doc_id", "chunk_index", "title", "chunk_strategy", "businesses", "business_ids",
    "ingest_run_id", "extraction_profile", "meta_selection_tier", "meta_rag_priority",
    "meta_answer_scope", "meta_lifecycle_status", "meta_content_availability",
    "meta_citation_status", "meta_start_date", "meta_end_date", "disabled",
    "sub_chunk_ids",
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def jsonable(v):
    """Neo4j temporal/spatial values are not JSON-serializable; render them as ISO strings."""
    if hasattr(v, "iso_format"):
        return v.iso_format()
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=500, help="nodes fetched per query")
    ap.add_argument("--limit", type=int, default=0, help="stop after N chunks (0 = all)")
    args = ap.parse_args()

    corpus_path = os.path.join(ROOT, "data", "corpus.jsonl")
    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)

    ada_vecs, ada_ids, ada_sha = [], [], []
    lg_vecs, lg_ids, lg_sha = [], [], []
    n = 0
    nfc_changed = 0

    query = f"""
    MATCH (n:`{LABEL}`)
    RETURN elementId(n) AS eid, n.text AS text, n.embedding AS ada,
           n.embedding_3large AS lg,
           {', '.join(f'n.{k} AS {k}' for k in META_KEYS)}
    ORDER BY n.doc_id, n.chunk_index
    SKIP $skip LIMIT $lim
    """

    driver = GraphDatabase.driver(URI, auth=AUTH)
    with driver, open(corpus_path, "w", encoding="utf-8") as out:
        driver.verify_connectivity()
        skip = 0
        while True:
            with driver.session(database="neo4j") as s:
                # execute_read marks the tx read-only; the server rejects any write.
                rows = s.execute_read(
                    lambda tx: list(tx.run(query, skip=skip, lim=args.batch)))
            if not rows:
                break
            for r in rows:
                raw = r["text"] or ""
                # NFC is the invariant every model in this benchmark will see.
                text = unicodedata.normalize("NFC", raw)
                if text != raw:
                    nfc_changed += 1
                h = sha(text)
                cid = f"{r['doc_id']}::{r['chunk_index']}"

                rec = {"chunk_id": cid, "element_id": r["eid"], "text": text,
                       "n_chars": len(text), "sha256": h}
                rec.update({k: jsonable(r[k]) for k in META_KEYS})
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if r["ada"] is not None:
                    ada_vecs.append(np.asarray(r["ada"], dtype=np.float32))
                    ada_ids.append(cid)
                    ada_sha.append(h)
                if r["lg"] is not None:
                    lg_vecs.append(np.asarray(r["lg"], dtype=np.float32))
                    lg_ids.append(cid)
                    lg_sha.append(h)
                n += 1
            skip += args.batch
            print(f"  {n} chunks... (ada {len(ada_ids)}, 3large {len(lg_ids)})", flush=True)
            if args.limit and n >= args.limit:
                break

    for name, vecs, ids, shas in (
        ("ada002", ada_vecs, ada_ids, ada_sha),
        ("text3large", lg_vecs, lg_ids, lg_sha),
    ):
        if not vecs:
            continue
        d = os.path.join(ROOT, "embeddings", name, "no_instruct")
        os.makedirs(d, exist_ok=True)
        arr = np.vstack(vecs)
        np.save(os.path.join(d, "corpus.npy"), arr)
        json.dump(ids, open(os.path.join(d, "ids.json"), "w"), ensure_ascii=False)
        norms = np.linalg.norm(arr, axis=1)
        manifest = {
            "model": name,
            "source": f"neo4j:{LABEL}",
            "n": int(arr.shape[0]),
            "dim": int(arr.shape[1]),
            "dtype": "float32",
            "already_l2_normalized": bool(np.allclose(norms, 1.0, atol=1e-3)),
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
            "text_sha256": dict(zip(ids, shas)),
        }
        json.dump(manifest, open(os.path.join(ROOT, "embeddings", name, "manifest.json"), "w"),
                  ensure_ascii=False)
        print(f"{name}: {arr.shape} -> {d}  normalized={manifest['already_l2_normalized']}")

    print(f"\ncorpus.jsonl: {n} chunks (NFC rewrote {nfc_changed})")


if __name__ == "__main__":
    main()
