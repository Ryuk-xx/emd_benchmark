# Corpus profile — AI instance 681cbdeb-d0d5-4441-9507-f6f7fa5cb580

Source: Neo4j 5.24.2 @ 192.168.20.38, label `681cbdeb_..._chunking_ai`
Exported: `data/corpus.jsonl` (37,610 rows), `embeddings/{ada002,text3large}/`

## Size
| | |
|---|---|
| chunks | 37,610 (7,898 documents) |
| tokens (cl100k) | 20,763,460 — mean 552, p50 458, p90 1,078, max 1,201 |
| chars | 44,642,932 — **2.15 chars/token** (Vietnamese tokenization penalty) |
| chunk strategy | general 24,332 / package 13,278 |

## Context limits — no confound
Max chunk is 1,201 tokens; the chunker caps around 1,200.
**Zero** chunks exceed Vietnamese_Embedding's 2,048 or OpenAI's 8,191.
No truncation will distort the comparison.

## Embedding coverage
| model | property | dim | coverage | L2-normalized |
|---|---|---|---|---|
| ada-002 | `embedding` | 1536 | 37,610 / 37,610 (100%) | yes |
| text-embedding-3-large | `embedding_3large` | 3072 | **2,038 / 37,610 (5.4%)** | yes |

The 3-large subset is not random: 1,616 `general` vs 422 `package`, skewed to `__legacy__`.

## Vector sanity (ada-002, 2,000-chunk sample)
- nearest neighbour lands in the same document **55.6%** of the time vs **0.07%** chance
  -> the stored vectors genuinely encode these chunks.
- cosine spread across all pairs: p01 0.719 / p50 0.824 / p99 0.899.
  An 18-point band over the whole corpus — ada-002's known compression.
  **Absolute similarity thresholds cannot be carried across models.**

## Data hygiene
| issue | count | note |
|---|---|---|
| chunk_id collisions | 0 | `doc_id::chunk_index` is a safe key |
| duplicate texts | 673 chunks in 218 groups | 455 removable |
| too short (<20 chars) | 216 | |
| separator-only | 90 | e.g. chunk `28280::92` = 59,004 hyphens |
| low entropy | 14 | |
| clean | 37,290 | |
| non-NFC on export | 1,626 (4.3%) | corpus mixed NFC and NFD |

## Re-embedding cost, full corpus
| model | tokens | USD |
|---|---|---|
| text-embedding-3-large | 20,763,460 | **$2.70** |
| text-embedding-ada-002 | 20,763,460 | $2.08 |
| text-embedding-3-small | 20,763,460 | $0.42 |
