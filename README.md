# Vietnamese embedding benchmark

Comparing five embedding models on a Viettel telecom RAG corpus:

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
set on which every model can be compared on identical input. The other 35,572 chunks
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
data/chunk_va_fact_500_bai.xlsx            optional, facts per chunk (1,862 rows)
data/bo_cau_hoi_681cbdeb_60.xlsx           optional, 60 questions with gold chunks
```

Everything under `data/` is picked up automatically; anything absent is reported and
skipped, so you can add files over time and re-run.

### Instruction modes

Both modes are produced where a model has an instruction, so the prefix's effect is
measured rather than assumed:

| mode | what it does |
|---|---|
| `no_instruct` | every text encoded bare |
| `instruct` | query-kind inputs get the model's instruction prefix |

Only query-kind inputs differ, since no model here defines a document-side prefix.
When a mode pair would produce identical vectors the encode runs once and the result is
written to both, reported as `(copied)`.

Each model takes its own prefix shape, and they are not interchangeable:

| model | `instruct` prefix | note |
|---|---|---|
| `qwen3_0.6b` | `Instruct: {task}\nQuery: ` | its documented template |
| `qwen3_vl_2b` | a plain instruction sentence | **no** `Instruct:`/`Query:` wrapper |
| `vn_embedding` | none | BGE-M3 lineage, never trained with instructions |

Two things to carry into the results. `vn_embedding` defines no instruction, so its
`instruct` mode is skipped rather than writing a duplicate of the bare vectors under a
name implying otherwise — it has a `no_instruct/` directory only. And `qwen3_vl_2b`
wraps *every* input in a default `Represent the user's input.` system prompt, so its
`no_instruct` column means "model default instruction", not "no instruction": it is the
one model whose two modes are not bare-versus-prefixed, and its `no_instruct` is not
directly comparable to the others'.

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
| `fact` | xlsx `Chunk và fact`, `CÁC FACT SINH RA TỪ CHUNK NÀY (f.text)` | 1,862 | document |
| `bo_cau_hoi` | xlsx `cau_hoi`, `cau_hoi` column, keyed by `stt` | 60 | **query** |

The coverage sheet is split in two on purpose. A question is a query and takes the
instruction prefix; an expected answer is a statement and must not, or the two sides of
the same case are not comparable. Both files keep the `Case` order, so row *i* is the
same case in each.

`fact` is one row per chunk holding every fact extracted from it, keyed
`<doc_id>::<chunk>` — the corpus's own chunk id — and written in `bench_ids` order, so
`ids_fact.json` joins straight to `corpus.npy`. It is an **alternative index**: coverage
questions are searched against the fact blocks instead of the raw chunk text
(`retrieve_coverage_top5.py --index fact`), and a hit still reports the chunk the block
stands for. That puts it on the document side, so like `corpus` it is encoded bare in
both modes and only the question side changes with the prefix.

Fact blocks are long: p50 ≈ 1k tokens, max ≈ 9k. They ask for an **8192-token window**
rather than the default (4096 for the Qwen models, 2048 for `vn_embedding`), and each
model clamps that to its own cap — 32k for both
Qwen models, so nothing is cut; 2048 for `vn_embedding`, whose model card stops there,
so roughly a fifth of its fact rows are truncated. The manifest records the window and
the truncated count per input; read them before comparing `vn_embedding` on this input.
Long windows also shrink the batch to a quarter of `--batch-size` to stay inside VRAM.

Copy `embeddings/vn_embedding/`, `embeddings/qwen3_0.6b/` and `embeddings/qwen3_vl_2b/`
back here to score.

**4. Score**

Models are named `<model>/<mode>`; a bare name means `no_instruct`. The default list
scores both modes of every model that has two, so the instruction's effect appears as
two rows in the same table.

```bash
python src/evaluate.py
python src/evaluate.py --models ada002 qwen3_0.6b/instruct qwen3_0.6b/no_instruct
```

**5b. Fill the question workbook**

```bash
python src/fill_bo_cau_hoi.py            # in place; keeps data/bo_cau_hoi_681cbdeb_60.bak.xlsx
python src/fill_bo_cau_hoi.py --dry-run  # report only
```

Writes the local models' vectors into sheet `embedding` (one row per question and
config, vectors split into 512-float JSON parts across `phan_1..phan_N`) and their
top-5 against the corpus into sheet `top5`, following the format the ada / 3-large rows
already use. `chunk_id` there is the Neo4j element id, as the workbook expects. Rows for
our configs are replaced on each run; ada, 3large and fact10 rows are never touched.

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

### Coverage top-5 retrieval

To retrieve the five highest-scoring chunks for every question in
`coverage_top1_top4_top5.xlsx`, run:

```bash
python src/retrieve_coverage_top5.py
```

The command processes `qwen3_0.6b`, `qwen3_vl_2b`, and `vn_embedding` in both modes,
prefers CUDA when PyTorch reports a GPU, and falls back to CPU otherwise. Results are written to
`results/coverage_top5.csv`, one row per question/model/mode/rank, with the chunk
ID, cosine score, chunk metadata, embedding timings from `manifest.json`, and
retrieval timing. To force CPU or choose a different output file:

```bash
python src/retrieve_coverage_top5.py --device cpu --output results/coverage_top5_cpu.csv
```

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
