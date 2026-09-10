"""Export the Facts asserted by the benchmark chunks — the seed for golden-set queries."""
import json
import os
import unicodedata

from neo4j import GraphDatabase

URI = os.environ.get("NEO4J_URI")
USER = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD")
if not (URI and PASSWORD):
    raise SystemExit("set NEO4J_URI and NEO4J_PASSWORD (see .env.example)")
AUTH = (USER, PASSWORD)
AI_ID = "681cbdeb-d0d5-4441-9507-f6f7fa5cb580"
P = AI_ID.replace("-", "_")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

keep = set(json.load(open(os.path.join(ROOT, "data", "bench_ids.json"), encoding="utf-8")))

Q = f"""
MATCH (c:`{P}_chunking_ai`)-[:ASSERTS]->(f:`{P}_fact_ai`)
WHERE c.embedding_3large IS NOT NULL
RETURN f.fact_id AS fact_id, f.text AS text, f.certainty AS certainty,
       f.polarity AS polarity, f.attribute AS attribute,
       f.subject_name AS subject, f.object_name AS object,
       f.condition_name AS condition, f.mau_thuan AS conflict,
       c.doc_id AS doc_id, c.chunk_index AS chunk_index, c.title AS title
"""

out_path = os.path.join(ROOT, "data", "facts.jsonl")
n = skipped = 0
driver = GraphDatabase.driver(URI, auth=AUTH)
with driver, driver.session(database="neo4j") as s, open(out_path, "w", encoding="utf-8") as out:
    # execute_read marks the tx read-only; the server rejects any write.
    for r in s.execute_read(lambda tx: list(tx.run(Q))):
        cid = f"{r['doc_id']}::{r['chunk_index']}"
        if cid not in keep:          # stay inside the filtered benchmark set
            skipped += 1
            continue
        rec = {k: r[k] for k in
               ("fact_id", "certainty", "polarity", "attribute", "subject",
                "object", "condition", "conflict", "title")}
        rec["text"] = unicodedata.normalize("NFC", r["text"] or "")
        rec["chunk_id"] = cid
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1

print(f"facts -> data/facts.jsonl: {n}  (skipped outside benchmark set: {skipped})")
