"""Embed the benchmark inputs with the two local open-source models.

Weights download automatically on first run into ./models/ (gitignored) and are
reused afterwards. Vectors and timings land in ./embeddings/<model>/<mode>/.

  python src/embed_local.py                      # both models, both modes, all inputs
  python src/embed_local.py --model qwen3_0.6b   # one model
  python src/embed_local.py --mode instruct      # one mode
  python src/embed_local.py --input corpus       # one input
  python src/embed_local.py --batch-size 8       # if VRAM is tight

Two modes are always produced so the effect of the instruction prefix is measured
rather than assumed:

  no_instruct  every text encoded bare
  instruct     query-kind inputs get the model's instruction prefix

Only query-kind inputs differ between the modes, because neither model defines a
document-side prefix. When a mode pair would produce identical vectors the encode
runs once and the result is written to both, and the summary says so.

Inputs are discovered under data/, and anything absent is reported and skipped:
  corpus.jsonl                          the chunks being searched      [doc]
  queries.jsonl                         golden-set questions           [query]
  embedding_calibration_testcases.csv   similarity pairs, text_a       [query]
                                        similarity pairs, text_b       [query]
  coverage_top1_top4_top5.xlsx          Comparison sheet Question      [query]
                                        Comparison sheet Expected      [doc]
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import time
import unicodedata

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_CACHE = os.path.join(ROOT, "models")

# Hugging Face must be pointed at the local cache before transformers is imported.
os.makedirs(MODEL_CACHE, exist_ok=True)
os.environ.setdefault("HF_HOME", MODEL_CACHE)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", MODEL_CACHE)

import torch                                            # noqa: E402
from sentence_transformers import SentenceTransformer   # noqa: E402

MODES = ("no_instruct", "instruct")

# Qwen3-Embedding is instruction-tuned and this is its documented prompt shape.
QWEN_TASK = ("Given a Vietnamese customer-service question about telecom services, "
             "retrieve the passages that answer it")

MODELS = {
    "vn_embedding": {
        "hf_id": "AITeamVN/Vietnamese_Embedding",
        "max_seq_length": 2048,
        # This model descends from BGE-M3, which was never trained with instructions,
        # so it has no prefix to apply. query_instruction=None means the instruct mode
        # is skipped for it entirely rather than duplicating the bare vectors.
        "query_instruction": None,
        "doc_instruction": None,
    },
    "qwen3_0.6b": {
        "hf_id": "Qwen/Qwen3-Embedding-0.6B",
        "max_seq_length": 2048,        # corpus max is 1,201 tokens, so nothing is cut
        "query_instruction": f"Instruct: {QWEN_TASK}\nQuery: ",
        "doc_instruction": None,       # Qwen3 defines no document-side prefix
    },
    "qwen3_vl_2b": {
        "hf_id": "Qwen/Qwen3-VL-Embedding-2B",
        "max_seq_length": 2048,        # model supports 32k; the corpus never needs it
        # This model takes a plain instruction sentence, NOT the
        # "Instruct: ...\nQuery: " template that Qwen3-Embedding uses. It also wraps
        # every input in a default "Represent the user's input." system prompt, so
        # its no_instruct mode is "model default", not "no instruction at all" -
        # the one model here for which the two modes are not bare-vs-prefixed.
        "query_instruction": QWEN_TASK + ".",
        "doc_instruction": None,
    },
}


def sha(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def nfc(t):
    """The corpus was exported as NFC; every other input must match, or the same
    Vietnamese word tokenizes two different ways."""
    return unicodedata.normalize("NFC", str(t)).strip()


def resolve_prompt(cfg, kind, mode):
    """The prefix for one (input kind, mode) pair. None means encode bare."""
    if mode == "no_instruct":
        return None
    return cfg["query_instruction"] if kind == "query" else cfg["doc_instruction"]


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_coverage_xlsx(path):
    """Read the Comparison sheet and return (case_ids, questions, expected).

    Questions and expected answers are returned separately because they are not the
    same kind of text: a question is a query and takes the instruction prefix, an
    expected answer is a statement and must not.
    """
    try:
        import openpyxl
    except ImportError:
        print("  coverage xlsx found but openpyxl is not installed "
              "(pip install openpyxl) - skipping")
        return None

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = "Comparison" if "Comparison" in wb.sheetnames else wb.sheetnames[0]
    rows = list(wb[sheet].iter_rows(values_only=True))
    wb.close()
    if not rows:
        return None

    # Match the header case-insensitively; the file capitalises these.
    header = [str(c).strip().lower() if c is not None else "" for c in rows[0]]
    try:
        i_case, i_q, i_e = (header.index(k) for k in ("case", "question", "expected"))
    except ValueError:
        print(f"  {os.path.basename(path)}: sheet '{sheet}' has no "
              f"case/question/expected columns - skipping")
        return None

    ids, questions, expected = [], [], []
    for r in rows[1:]:
        if not any(r):
            continue
        q = nfc(r[i_q]) if r[i_q] else ""
        e = nfc(r[i_e]) if r[i_e] else ""
        if not (q and e):
            continue
        ids.append(str(r[i_case]).strip() if r[i_case] else f"row{len(ids)}")
        questions.append(q)
        expected.append(e)
    return ids, questions, expected


def discover_inputs(coverage_file=None, coverage_name="coverage"):
    """Collect whichever inputs exist. 'kind' decides whether a prompt applies."""
    d = os.path.join(ROOT, "data")
    found = {}

    corpus_p = os.path.join(d, "corpus.jsonl")
    ids_p = os.path.join(d, "bench_ids.json")
    if os.path.exists(corpus_p):
        rows = load_jsonl(corpus_p)
        by_id = {r["chunk_id"]: r for r in rows}
        if os.path.exists(ids_p):
            # bench_ids.json fixes the row order shared by every model's matrix;
            # evaluate.py asserts it, so honour it rather than the file order.
            with open(ids_p, encoding="utf-8") as f:
                order = json.load(f)
            missing = [c for c in order if c not in by_id]
            if missing:
                raise SystemExit(
                    f"{len(missing)} ids in bench_ids.json missing from corpus.jsonl")
            rows = [by_id[c] for c in order]
            ids = order
        else:
            ids = [r["chunk_id"] for r in rows]
        found["corpus"] = {"ids": ids, "texts": [nfc(r["text"]) for r in rows],
                           "kind": "doc"}

    queries_p = os.path.join(d, "queries.jsonl")
    if os.path.exists(queries_p):
        rows = load_jsonl(queries_p)
        found["queries"] = {"ids": [r["query_id"] for r in rows],
                            "texts": [nfc(r["text"]) for r in rows], "kind": "query"}

    calib_p = os.path.join(d, "embedding_calibration_testcases.csv")
    if os.path.exists(calib_p):
        with open(calib_p, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        # One file per column, sharing the pair_id order, so scoring a pair is a
        # row-wise dot product of the two matrices instead of de-interleaving one.
        #
        # This is a symmetric similarity test, not retrieval, so BOTH sides get the
        # same treatment in a given mode: bare in no_instruct, prefixed in instruct.
        # Prefixing only one side would offset every cosine in the instruct column
        # by however far the prefix moves a vector, and that artefact would swamp
        # the thing being measured. Keeping them symmetric also preserves the T0
        # sanity pair reading 1.0 in both modes.
        ids = [r["pair_id"] for r in rows]
        found["calibration_a"] = {"ids": ids, "kind": "query",
                                  "texts": [nfc(r["text_a"]) for r in rows]}
        found["calibration_b"] = {"ids": ids, "kind": "query",
                                  "texts": [nfc(r["text_b"]) for r in rows]}

    # Excel leaves a ~$ lock file behind while the workbook is open; skip it.
    cov_p = coverage_file or os.path.join(d, "coverage_top1_top4_top5.xlsx")
    if os.path.exists(cov_p) and not os.path.basename(cov_p).startswith("~$"):
        got = load_coverage_xlsx(cov_p)
        if got:
            case_ids, questions, expected = got
            # Split by kind, keeping the same Case order in both, so row i is the
            # same case in each file.
            suffix = "" if coverage_name == "coverage" else f"_{coverage_name}"
            found[f"coverage_questions{suffix}"] = {
                "ids": case_ids, "texts": questions, "kind": "query"}
            found[f"coverage_expected{suffix}"] = {
                "ids": case_ids, "texts": expected, "kind": "doc"}

    return found


def encode_timed(model, texts, prompt, batch_size, device):
    """Encode in timed batches so the log carries a throughput distribution."""
    out, times = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        v = model.encode(batch, batch_size=batch_size, prompt=prompt,
                         normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=False)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        out.append(v)
        done = min(i + batch_size, len(texts))
        if (i // batch_size) % 20 == 0 or done == len(texts):
            rate = done / sum(times)
            print(f"       {done:6d}/{len(texts)}  {rate:7.1f} items/s  "
                  f"eta {(len(texts) - done) / rate:5.0f}s", flush=True)
    return np.vstack(out).astype(np.float32), times


def measure_latency(model, texts, prompt, device, n=20):
    """Single-item latency after warm-up: the production query path, which batch
    throughput does not predict."""
    for t in texts[:5]:
        model.encode([t], prompt=prompt, normalize_embeddings=True)
    lat = []
    for t in texts[:n]:
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.encode([t], prompt=prompt, normalize_embeddings=True)
        if device == "cuda":
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    return lat


def write_vectors(out_dir, input_name, X, ids):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{input_name}.npy"), X)
    id_file = "ids.json" if input_name == "corpus" else f"ids_{input_name}.json"
    with open(os.path.join(out_dir, id_file), "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False)


def run_model(model_name, cfg, inputs, modes, args):
    """Load one model once and encode every (input, mode) it is asked for."""
    device = args.device
    print(f"\n{'=' * 74}\n{model_name}  ({cfg['hf_id']})\n{'=' * 74}")
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    print("  loading (first run downloads the weights)...", flush=True)
    model = SentenceTransformer(
        cfg["hf_id"], device=device, cache_folder=MODEL_CACHE,
        model_kwargs={"torch_dtype": torch.float16} if args.fp16 else {},
    )
    model.max_seq_length = cfg["max_seq_length"]
    entry = {"hf_id": cfg["hf_id"], "load_seconds": round(time.perf_counter() - t0, 2),
             "dim": model.get_sentence_embedding_dimension(),
             "max_seq_length": cfg["max_seq_length"],
             "query_instruction": cfg["query_instruction"],
             "doc_instruction": cfg["doc_instruction"], "modes": {}}
    print(f"  ready in {entry['load_seconds']}s, dim={entry['dim']}")

    for input_name, data in inputs.items():
        texts, ids, kind = data["texts"], data["ids"], data["kind"]
        n_tok = [len(model.tokenizer.encode(t, add_special_tokens=True)) for t in texts]
        truncated = int(sum(n > cfg["max_seq_length"] for n in n_tok))
        print(f"  -- {input_name} [{kind}]: {len(texts)} items, "
              f"{sum(n_tok):,} tokens, max {max(n_tok)}, truncated {truncated}")

        done_by_prompt = {}          # prompt -> (mode already encoded)
        for mode in modes:
            prompt = resolve_prompt(cfg, kind, mode)
            out_dir = os.path.join(ROOT, "embeddings", model_name, mode)

            if prompt in done_by_prompt:
                # Same prefix as a mode already encoded: reuse rather than pay twice.
                src_mode = done_by_prompt[prompt]
                src = os.path.join(ROOT, "embeddings", model_name, src_mode)
                os.makedirs(out_dir, exist_ok=True)
                for fn in os.listdir(src):
                    if fn.startswith(input_name) or fn == f"ids_{input_name}.json" \
                            or (input_name == "corpus" and fn == "ids.json"):
                        shutil.copyfile(os.path.join(src, fn), os.path.join(out_dir, fn))
                prev = entry["modes"][src_mode][input_name]
                entry["modes"].setdefault(mode, {})[input_name] = {
                    **prev, "identical_to_mode": src_mode}
                print(f"     {mode:12s} prompt={prompt!r} -> identical to "
                      f"{src_mode}, copied")
                continue

            print(f"     {mode:12s} prompt={prompt!r}")
            X, batch_times = encode_timed(model, texts, prompt, args.batch_size, device)
            total = sum(batch_times)
            lat = measure_latency(model, texts, prompt, device)
            write_vectors(out_dir, input_name, X, ids)

            entry["modes"].setdefault(mode, {})[input_name] = {
                "n": len(texts), "shape": list(X.shape), "kind": kind,
                "prompt": prompt,
                "tokens_total": int(sum(n_tok)), "tokens_max": int(max(n_tok)),
                "tokens_truncated": truncated,
                "encode_seconds": round(total, 2),
                "items_per_second": round(len(texts) / total, 1),
                "tokens_per_second": round(sum(n_tok) / total, 1),
                "batch_seconds": {
                    "n_batches": len(batch_times),
                    "mean": round(float(np.mean(batch_times)), 4),
                    "p50": round(float(np.percentile(batch_times, 50)), 4),
                    "p95": round(float(np.percentile(batch_times, 95)), 4),
                    "first": round(batch_times[0], 4),
                },
                "single_item_latency_ms": {
                    "p50": round(float(np.percentile(lat, 50)), 1),
                    "p95": round(float(np.percentile(lat, 95)), 1),
                },
                "text_sha256": {i: sha(t) for i, t in zip(ids, texts)},
            }
            done_by_prompt[prompt] = mode
            print(f"       done {X.shape} in {total:.1f}s "
                  f"({len(texts) / total:.1f} items/s), "
                  f"1-item p50 {np.percentile(lat, 50):.0f}ms "
                  f"p95 {np.percentile(lat, 95):.0f}ms")

    if device == "cuda":
        entry["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        print(f"  peak VRAM {entry['peak_vram_gb']} GB")

    man_dir = os.path.join(ROOT, "embeddings", model_name)
    os.makedirs(man_dir, exist_ok=True)
    man_path = os.path.join(man_dir, "manifest.json")

    # Merge per input, not per file: re-running one --input must not erase the
    # record of vectors written by an earlier run that are still on disk. The merge
    # goes only into the persisted manifest -- `entry` stays this run's own record,
    # so the run summary reports what actually just ran.
    persisted = {**entry, "modes": {k: dict(v) for k, v in entry["modes"].items()}}
    if os.path.exists(man_path):
        with open(man_path, encoding="utf-8") as f:
            old = json.load(f)
        for mode, per_input in old.get("modes", {}).items():
            merged = dict(per_input)
            merged.update(persisted["modes"].get(mode, {}))
            persisted["modes"][mode] = merged

    with open(man_path, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, **persisted, "dtype": "float32",
                   "already_l2_normalized": True}, f, ensure_ascii=False)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="all", choices=["all", *MODELS])
    ap.add_argument("--mode", default="all", choices=["all", *MODES])
    ap.add_argument("--input", default="all",
                    choices=["all", "corpus", "queries",
                             "calibration_a", "calibration_b", "calibration",
                             "coverage_questions", "coverage_expected", "coverage"],
                    help="'calibration' and 'coverage' each mean both of their halves")
    ap.add_argument("--coverage-file", default=None,
                    help="coverage workbook path; defaults to data/coverage_top1_top4_top5.xlsx")
    ap.add_argument("--coverage-name", default="coverage",
                    help="output prefix for a custom coverage workbook (default: coverage)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--fp32", dest="fp16", action="store_false", default=True,
                    help="full precision; slower and needs more VRAM")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"device      : {args.device}"
          + (f" ({torch.cuda.get_device_name(0)})" if args.device == "cuda" else ""))
    print(f"precision   : {'fp16' if args.fp16 else 'fp32'}, batch {args.batch_size}")
    print(f"weight cache: {MODEL_CACHE}")
    if args.device == "cpu":
        print("  no GPU detected - this will work but take much longer")

    available = discover_inputs(args.coverage_file, args.coverage_name)
    if args.input == "all":
        inputs = available
    elif args.input in ("coverage", "calibration"):
        inputs = {k: v for k, v in available.items()
                  if k.startswith(args.input + "_")}
    else:
        inputs = {k: v for k, v in available.items() if k == args.input}
    if not inputs:
        raise SystemExit(f"nothing to embed for --input {args.input} "
                         f"(found: {', '.join(available) or 'none'})")

    all_names = ("corpus", "queries", "calibration_a", "calibration_b",
                 "coverage_questions", "coverage_expected")
    print("\ninputs found:")
    for name, d in inputs.items():
        chars = sum(len(t) for t in d["texts"])
        print(f"  {name:20s} {len(d['texts']):6d} items  {chars:>9,} chars  [{d['kind']}]")
    for name in all_names:
        if name not in inputs:
            print(f"  {name:20s} (absent, skipped)")

    modes = list(MODES) if args.mode == "all" else [args.mode]
    todo = list(MODELS) if args.model == "all" else [args.model]
    print(f"\nmodes: {', '.join(modes)}")

    log = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "device": args.device,
           "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
           "precision": "fp16" if args.fp16 else "fp32",
           "batch_size": args.batch_size, "modes": modes,
           "qwen_task_description": QWEN_TASK, "models": {}}

    for model_name in todo:
        cfg = MODELS[model_name]
        # A model with no instruction of its own has nothing to put in instruct mode;
        # running it would just duplicate the bare vectors under a misleading name.
        model_modes = [m for m in modes
                       if m != "instruct" or cfg["query_instruction"]]
        if len(model_modes) < len(modes):
            print(f"\n{model_name}: no instruction defined, instruct mode skipped")
        if not model_modes:
            continue
        log["models"][model_name] = run_model(
            model_name, cfg, inputs, model_modes, args)

    log["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    log_path = os.path.join(ROOT, "results", "embedding_timing.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 74}")
    print(f"{'model':<15}{'mode':<13}{'input':<21}{'n':>6}{'sec':>8}"
          f"{'items/s':>9}{'p50ms':>7}")
    for m, e in log["models"].items():
        for mode, per_input in e["modes"].items():
            for i, s in per_input.items():
                tag = " (copied)" if s.get("identical_to_mode") else ""
                print(f"{m:<15}{mode:<13}{i:<21}{s['n']:>6}{s['encode_seconds']:>8}"
                      f"{s['items_per_second']:>9}"
                      f"{s['single_item_latency_ms']['p50']:>7}{tag}")
    print(f"\ntiming log -> {os.path.relpath(log_path, ROOT)}")

    bad = [(m, mode, i) for m, e in log["models"].items()
           for mode, pi in e["modes"].items()
           for i, s in pi.items() if s["tokens_truncated"]]
    if bad:
        print("\nWARNING truncated inputs:", bad)


if __name__ == "__main__":
    main()
