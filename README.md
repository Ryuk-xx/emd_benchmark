# Vietnamese embedding benchmark

Comparing four embedding models on a Viettel telecom RAG corpus:

| model | dim | where it runs |
|---|---|---|
| `ada002` — text-embedding-ada-002 | 1536 | already embedded, pulled from Neo4j |
| `text3large` — text-embedding-3-large | 3072 | already embedded, pulled from Neo4j |
| `vn_embedding` — AITeamVN/Vietnamese_Embedding | 1024 | GPU machine |
| `qwen3_0.6b` — Qwen/Qwen3-Embedding-0.6B | 1024 | GPU machine |
| `qwen3_vl_2b` — Qwen/Qwen3-VL-Embedding-2B | 2048 | GPU machine |

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
skipped, so you can add files over time and re-run.

### Two instruction modes

Every run produces both, so the instruction prefix's effect is measured rather than
assumed:

| mode | what it does |
|---|---|
| `no_instruct` | every text encoded bare |
| `instruct` | query-kind inputs get the model's instruction prefix |

Only query-kind inputs differ, since neither model defines a document-side prefix.
When a mode pair would produce identical vectors the encode runs once and the result is
written to both, reported as `(copied)`. Qwen3's prefix is its documented
`Instruct: {task}\nQuery: `; Vietnamese_Embedding descends from BGE-M3, which was not
trained with instructions, so its prefix is ours and `instruct` mode there is an
experiment expected to be neutral at best.

### Storage layout

```
embeddings/
  <model>/
    manifest.json              dims, load time, timings, prompts, per-text sha256
    no_instruct/
      corpus.npy                 ids.json              ordered by bench_ids.json
      queries.npy                ids_queries.json
      calibration_a.npy          ids_calibration_a.json
      calibration_b.npy          ids_calibration_b.json
      coverage_questions.npy     ids_coverage_questions.json
      coverage_expected.npy      ids_coverage_expected.json
    instruct/
      ... same file names ...
```

Each two-column source becomes two matrices that share an id order, never one
interleaved matrix, so comparing the two sides of a case or pair is a row-wise dot
product: `(A * B).sum(axis=1)`.

`ada002` and `text3large` have a `no_instruct/` directory only: the OpenAI embedding
API takes no instruction. Every matrix is float32 and L2-normalized, so cosine is a
plain dot product, and row *i* is the same item across every model and mode.

| input | source | items | encoded as |
|---|---|---|---|
| `corpus` | `corpus.jsonl`, ordered by `bench_ids.json` | 2,038 | document |
| `queries` | `queries.jsonl` | golden set | **query** |
| `calibration_a` | CSV, `text_a` column | 20 | **query** |
| `calibration_b` | CSV, `text_b` column | 20 | **query** |
| `coverage_questions` | xlsx `Comparison`, `Question` column | 60 | **query** |
| `coverage_expected` | xlsx `Comparison`, `Expected` column | 60 | document |

The coverage sheet is split in two on purpose. A question is a query and takes the
instruction prefix; an expected answer is a statement and must not, or the two sides of
the same case are not comparable. Both files keep the `Case` order, so row *i* is the
same case in each.

Copy `embeddings/vn_embedding/` and `embeddings/qwen3_0.6b/` back here to score.

**4. Score**

Models are named `<model>/<mode>`; a bare name means `no_instruct`. The default list
scores both modes of each local model, so the instruction's effect appears as two rows
in the same table.

```bash
python src/evaluate.py
python src/evaluate.py --models ada002 qwen3_0.6b/instruct qwen3_0.6b/no_instruct
```

**5. Score the calibration pairs**

```bash
python src/score_calibration.py            # -> results/calibration_report.xlsx
```

Both sides of a pair get the same treatment within a mode: bare in `no_instruct`,
prefixed in `instruct`. This is a symmetric similarity test, not retrieval. Prefixing
one side only would offset every cosine in the instruct column by however far the
prefix moves a vector, and that artefact would swamp the discrimination being measured;
it would also stop the T0 sanity pair reading 1.0. Kept symmetric, T0 is 1.0 in both
modes and the two columns are directly comparable.

The workbook has four sheets: `per_pair` (cosine per pair per model/mode, beside the
texts), `by_tier` (means down the ladder), `diagnostics` and `instruct_effect`.

`diagnostics` carries the number that decides deployability, **gap_T1_minus_T2**. T1
pairs are the same fact reworded; T2 pairs are the same document with a *different
attribute* (issue date vs effective date). A model whose T1 and T2 means sit on top of
each other will confidently return the wrong date, and no threshold can separate them.
Gaps under 0.05 and a broken ladder are highlighted in red.

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
