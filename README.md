# Vietnamese embedding benchmark

Comparing four embedding models on a Viettel telecom RAG corpus:

| model | dim | where it runs |
|---|---|---|
| `ada002` — text-embedding-ada-002 | 1536 | already embedded, pulled from Neo4j |
| `text3large` — text-embedding-3-large | 3072 | already embedded, pulled from Neo4j |
| `vn_embedding` — AITeamVN/Vietnamese_Embedding | 1024 | GPU machine |
| `qwen3_0.6b` — Qwen/Qwen3-Embedding-0.6B | 1024 | GPU machine |

## Rule: Neo4j is read-only

The graph at `192.168.20.38:7687` is a live RAG store and must never be written to.
`export_neo4j.py` and `export_facts.py` go through `session.execute_read(...)`, so the
server itself rejects any write with `Neo.ClientError.Statement.AccessMode`.
All filtering happens on the exported copies under `data/`.

## Corpus

`data/corpus.jsonl` holds the **2,038 chunks embedded by both OpenAI models**, the only
set on which all four models can be compared on identical input. The other 35,572 chunks
had ada-002 vectors only; they stay in `data/corpus_full.jsonl` and are excluded from
scoring. 502 documents, 1.14M cl100k tokens, max 1,201 tokens per chunk — under every
model's context limit, so nothing is truncated.

`data/bench_ids.json` fixes the row order shared by every `corpus.npy`. Row *i* is the
same chunk in every model's matrix; `evaluate.py` asserts this. That is what makes the
per-query paired comparison valid.

## Steps

**1. Export from Neo4j** (done; re-run only if the graph changes)

```bash
python src/export_neo4j.py          # -> data/corpus_full.jsonl, embeddings/{ada002,text3large}/
python src/filter_dual.py           # -> data/corpus.jsonl (2,038), aligned matrices
python src/export_facts.py          # -> data/facts.jsonl (19,529 golden-set seeds)
```

**2. Build the golden set** (needs an OpenAI-compatible endpoint)

```bash
export OPENAI_API_KEY=...           # OPENAI_BASE_URL too, for Azure or a local model
python src/build_goldenset.py generate --n 600
# review data/queries_raw.jsonl by hand before continuing
python src/build_goldenset.py variants   # -> data/queries.jsonl, data/qrels.tsv
```

`generate` writes one realistic Vietnamese question per sampled fact, then drops any
question that leaks its answer's wording. This filter is not optional: measured on real
data, **91% of a raw fact's words already appear verbatim in its source chunk** (p50
containment 0.91), and 70% of raw facts would be rejected outright. Questions that echo
the chunk are found by every model and by BM25, so they measure nothing.

`variants` then expands each question into `nodiacritic`, `lowercase`, `abbrev`, `typo`
and `nodiacritic_abbrev` — how Vietnamese users actually type.

**3. Embed with the local models** — on the GPU machine

`data/` is gitignored — it holds internal documents — so clone the code and move the data
across separately. The embedding step needs these files, 4.5 MB in total:

```
data/corpus.jsonl                          2,038 chunks to embed
data/bench_ids.json                        the row order every matrix must follow
data/queries.jsonl                         once the golden set exists
data/embedding_calibration_testcases.csv   optional, similarity-calibration pairs
data/coverage_top1_top4_top5.xlsx          optional, 60 Question/Expected cases
```

Everything under `data/` is picked up automatically; anything absent is reported and
skipped, so you can add files over time and re-run. Each input becomes
`embeddings/<model>/<input>.npy`:

| input | source | items | encoded as |
|---|---|---|---|
| `corpus` | `corpus.jsonl`, ordered by `bench_ids.json` | 2,038 | document |
| `queries` | `queries.jsonl` | golden set | **query** |
| `calibration` | CSV, `text_a` + `text_b` per pair | 40 | document |
| `coverage_questions` | xlsx `Comparison`, `Question` column | 60 | **query** |
| `coverage_expected` | xlsx `Comparison`, `Expected` column | 60 | document |

The coverage sheet is split in two on purpose. A question is a query and takes Qwen3's
instruction prefix; an expected answer is a statement and must not, or the two sides of
the same case are not comparable. Both files keep the `Case` order, so row *i* is the
same case in each.

Then, on the GPU machine:

```bash
git clone -b dev https://github.com/Ryuk-xx/emd_benchmark.git
cd emd_benchmark
mkdir -p data && cp /path/to/corpus.jsonl /path/to/bench_ids.json data/

pip install torch --index-url https://download.pytorch.org/whl/cu121   # match its CUDA
pip install -r requirements-gpu.txt

python src/embed_local.py
```

That one command does everything: it downloads both models' weights into `models/` on
first run (reused afterwards), encodes every input it finds under `data/`, and writes
vectors plus timings to `embeddings/<model>/`.

```bash
python src/embed_local.py --model qwen3_0.6b   # just one model
python src/embed_local.py --input corpus       # just one input
python src/embed_local.py --batch-size 8       # if VRAM is tight
python src/embed_local.py --fp32               # full precision
```

Both models fit in ~1.2 GB VRAM at fp16, and the corpus is only 1.14M tokens, so each
pass takes minutes. Every run writes `embeddings/<model>/manifest.json` and
`results/embedding_timing.json` with encode time, items/s, tokens/s, per-batch
percentiles, single-item p50/p95 latency, peak VRAM and the exact prompt used, and it
flags any input that had to be truncated.

Copy `embeddings/vn_embedding/` and `embeddings/qwen3_0.6b/` back here to score.

> **Qwen3 needs its query instruction.** `embed_local.py` applies
> `Instruct: {task}\nQuery: ` to queries and nothing to documents, and records the exact
> task string in the manifest. Getting this wrong is the single most common way to
> under-report Qwen3 by several points, so do not paste the prefix in by hand.

**4. Score**

```bash
python src/evaluate.py --models ada002 text3large vn_embedding qwen3_0.6b
```

## What the report shows

- **nDCG@10, Recall@{1,5,10,50}, MRR@10** with bootstrap 95% CIs.
  If a reranker sits downstream, read Recall@50, not nDCG@10.
- **BM25 and an RRF hybrid** in every table. A dense model that cannot beat BM25 on this
  corpus is not worth deploying, and the hybrid usually beats either alone.
- **Robustness**: percentage change per query variant. For Vietnamese this often decides
  the choice more than the headline score does.
- **Paired bootstrap** against the leading model, labelling each gap significant or not.
  A one-point difference over a few hundred queries is usually noise.
- **Per-business breakdown**, because a global mean hides a model that wins on `Gói cước`
  and loses on `Hướng dẫn xử lý lỗi`.

## Known limits of this benchmark

- **2,038 chunks is a small index.** Recall@50 will saturate near 1.0 and carry no signal;
  R@1 and MRR stay discriminative. Rankings here may not hold on the full 37,610-chunk
  corpus. To check that, embed the full corpus with the two local models and score them
  against ada-002 — `text3large` cannot join that run, as only 5.4% of the full corpus
  has 3-large vectors. Closing that gap costs about $2.70 in API calls.
- **Queries are LLM-written from extracted facts**, not real user traffic. The leakage
  filter removes the worst of the wording bias but not the distribution bias: real users
  ask shorter, vaguer questions. Replace with query logs when they exist.
- **ada-002 cosine is compressed** — p01 0.719 / p50 0.824 / p99 0.899 across the whole
  corpus. Rankings are comparable between models; absolute score thresholds are not.
