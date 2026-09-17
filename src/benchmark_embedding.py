"""Inference performance benchmark for the three local embedding models.

Measures speed and resource use only - no accuracy, no relevance judgements. Runs
every model in `no_instruct` mode (no instruction prefix on any input), so what is
timed is the raw encode path.

    python src/benchmark_embedding.py
    python src/benchmark_embedding.py --models qwen_0.6 --batch-sizes 1 8
    python src/benchmark_embedding.py --no-single   # batch mode only
    python src/benchmark_embedding.py --single-only --merge-into results/benchmark_results.csv
    python src/benchmark_embedding.py --merge-from results/benchmark_results_single.csv \
        --merge-into results/benchmark_results.csv  # merge an existing single run, no GPU
    python src/benchmark_embedding.py --dry-run     # print the grid, load nothing

Everything worth changing sits in the CONFIG block below.

Each row of benchmark_results.csv is one (model, dataset, max_length, batch_size)
configuration with status `ok`, `unsupported` (the model cannot take that
max_length), `OOM`, or `error`. batch_size is `single` for the single-request mode.

Two modes, sharing one timing loop:

  single  one request per sample, sent one after another over SINGLE_SAMPLES distinct
          samples - what an online service sees. Its percentiles describe the latency
          across the dataset's length distribution.
  batch   BENCH_RUNS timed runs at each batch size. batch_size=1 here is repeated
          runs over a rotating window of samples, not a pass over the dataset, so it
          is not the same measurement as single.

Two numbers describe the batch, and they are not the same thing:

  actual_token_count  real tokens in the batch after truncation to max_length.
                      tokens_per_sec is derived from this.
  padded_token_count  batch_size x the longest sequence in the batch, which is what
                      the GPU actually computes. When it is far above the real count,
                      the batch is mostly padding and samples_per_sec will look worse
                      than the model deserves.
"""
import argparse
import csv
import gc
import json
import os
import random
import statistics
import threading
import time
from collections import defaultdict

# ----------------------------------------------------------------- CONFIG
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = "results/perf_dataset"
DATASET_FILES = {                      # dataset name -> CSV under DATASET_DIR
    "short": "perf_dataset_short.csv",
    "medium": "perf_dataset_medium.csv",
    "long": "perf_dataset_long.csv",
}
OUTPUT_CSV = "results/benchmark_results.csv"
SINGLE_ONLY_CSV = "results/benchmark_results_single.csv"   # --single-only default output

# Weights download to ./models on first use, as in embed_local.py.
MODEL_CACHE = "models"

# `cap` is the longest sequence the model accepts; a configuration asking for more
# is recorded as unsupported rather than silently truncated to something else.
MODELS = {
    "vn_embedding": {"hf_id": "AITeamVN/Vietnamese_Embedding", "cap": 2048},
    "qwen_0.6": {"hf_id": "Qwen/Qwen3-Embedding-0.6B", "cap": 32768},
    "qwen_vl": {"hf_id": "Qwen/Qwen3-VL-Embedding-2B", "cap": 32768},
}

# (dataset, max_length) pairs to time. short is run twice on purpose: once at a
# window that fits it, once at 2048, which isolates the cost of an oversized window
# on short input.
CONFIGURATIONS = [
    ("short", 128),
    ("short", 2048),
    ("medium", 2048),
    ("long", 8192),
]

BATCH_SIZES = [1, 4, 16, 32, 64]
WARMUP_RUNS = 10                       # not timed
BENCH_RUNS = 40                        # timed

# Single mode: one request per sample, sequentially, over distinct samples.
SINGLE_MODE = True
SINGLE_WARMUP = 10                     # not timed
SINGLE_SAMPLES = None                  # None = every sample in the dataset

# The dataset CSVs are sorted by token count. Without shuffling, batch_size=1 only
# ever touches the shortest samples near the top of the file, which flatters it.
# Shuffled once per dataset with a fixed seed, so runs stay reproducible.
SHUFFLE_SAMPLES = True
SEED = 20260917
FP16 = True
NORMALIZE = True                       # as production does
CONVERT_TO_NUMPY = True                # includes the GPU->CPU copy in the latency

# A batch size that OOMs makes every larger one pointless; skipping them saves a lot
# of wall clock on the long dataset. They are still written out as OOM.
SKIP_LARGER_AFTER_OOM = True
# Rotate through the dataset so successive runs see different samples. The dataset is
# length-stratified, so p95/p99 then include real content-length variation. Set False
# to time one fixed batch and see system noise alone.
ROTATE_SAMPLES = True
GPU_SAMPLE_INTERVAL = 0.05             # seconds between utilisation/RAM samples
# ------------------------------------------------------------- END CONFIG

CSV_FIELDS = [
    "model", "dataset", "max_length", "batch_size", "status",
    "actual_token_count", "padded_token_count",
    "latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
    "samples_per_sec", "tokens_per_sec",
    "gpu_utilization", "vram_mb", "peak_vram_mb", "peak_vram_reserved_mb",
    "ram_mb", "peak_ram_mb", "runs", "error",
]


def abspath(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def rel(p):
    """relpath that does not blow up when p sits on another drive."""
    try:
        return os.path.relpath(p, ROOT)
    except ValueError:
        return p


def pct(values, q):
    """Percentile by nearest-rank; avoids numpy and behaves on tiny samples."""
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q / 100.0 * len(s) + 0.5)) - 1))
    return s[k]


# ------------------------------------------------------------- monitoring

class ResourceMonitor:
    """Samples GPU utilisation and process RSS on a thread while a loop runs."""

    def __init__(self, interval=GPU_SAMPLE_INTERVAL):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.gpu_util = []
        self.rss_mb = []
        self.handle = None
        self.psutil = None
        try:
            import psutil
            self.psutil = psutil.Process(os.getpid())
        except ImportError:
            pass
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.pynvml = None

    def _loop(self):
        while not self._stop.is_set():
            if self.handle is not None:
                try:
                    self.gpu_util.append(
                        self.pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu)
                except Exception:
                    pass
            if self.psutil is not None:
                try:
                    self.rss_mb.append(self.psutil.memory_info().rss / 1e6)
                except Exception:
                    pass
            self._stop.wait(self.interval)

    def start(self):
        self.gpu_util, self.rss_mb = [], []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return {
            "gpu_utilization": (round(statistics.mean(self.gpu_util), 1)
                                if self.gpu_util else None),
            "ram_mb": round(self.rss_mb[-1], 1) if self.rss_mb else None,
            "peak_ram_mb": round(max(self.rss_mb), 1) if self.rss_mb else None,
        }


# --------------------------------------------------------------- adapters

class SentenceTransformerAdapter:
    """Default adapter: load, set the window, count tokens, encode.

    All three models load through sentence-transformers, which applies each one's own
    pooling, padding side and normalisation. A model needing different handling gets
    its own subclass overriding `encode` or `count_tokens`; the timing loop never
    changes.
    """

    def __init__(self, name, cfg, device, fp16=FP16):
        self.name, self.cfg, self.device, self.fp16 = name, cfg, device, fp16
        self.model = None

    def load(self):
        import torch
        from sentence_transformers import SentenceTransformer
        kwargs = {"torch_dtype": torch.float16} if self.fp16 else {}
        self.model = SentenceTransformer(self.cfg["hf_id"], device=self.device,
                                         cache_folder=abspath(MODEL_CACHE),
                                         model_kwargs=kwargs)
        return self

    def supports(self, max_length):
        return max_length <= self.cfg["cap"]

    def set_max_length(self, max_length):
        self.model.max_seq_length = max_length

    def count_tokens(self, texts, max_length):
        """Real and padded token counts for one batch, after truncation."""
        enc = self.model.tokenizer(texts, truncation=True, max_length=max_length)
        lens = [len(x) for x in enc["input_ids"]]
        return sum(lens), max(lens) * len(lens)

    def encode(self, texts):
        return self.model.encode(texts, batch_size=len(texts), prompt=None,
                                 normalize_embeddings=NORMALIZE,
                                 convert_to_numpy=CONVERT_TO_NUMPY,
                                 show_progress_bar=False)

    def free(self):
        del self.model
        self.model = None


ADAPTERS = defaultdict(lambda: SentenceTransformerAdapter)


# ------------------------------------------------------------------ bench

def load_datasets(shuffle=SHUFFLE_SAMPLES):
    out = {}
    for name, fn in DATASET_FILES.items():
        path = abspath(os.path.join(DATASET_DIR, fn))
        if not os.path.exists(path):
            print(f"  dataset '{name}' not found at {path} - skipped")
            continue
        with open(path, encoding="utf-8-sig") as f:
            rows = [r for r in csv.DictReader(f) if r.get("text")]
        texts = [r["text"] for r in rows]
        if shuffle:
            random.Random(f"{SEED}:{name}").shuffle(texts)
        out[name] = texts
        print(f"  {name:<8} {len(texts):>4} samples  ({fn})"
              + ("  shuffled" if shuffle else ""))
    return out


def make_batch(texts, batch_size, run_index):
    """Batch `run_index` when rotating, else always the first batch."""
    if not ROTATE_SAMPLES:
        start = 0
    else:
        start = (run_index * batch_size) % len(texts)
    picked, i = [], start
    while len(picked) < batch_size:                 # wrap if the pool is small
        picked.append(texts[i % len(texts)])
        i += 1
    return picked


def is_oom(exc):
    """OOM surfaces as torch.cuda.OutOfMemoryError on new torch, RuntimeError on old."""
    import torch
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return "out of memory" in str(exc).lower()


def measure(adapter, warmup_batches, timed_batches, max_length, monitor, device):
    """The timing loop both modes share. Returns metric fields, or raises on OOM.

    Warm-up is excluded from latency and from the VRAM peak; each timed encode is
    bracketed by torch.cuda.synchronize so the GPU work is inside the measurement.
    """
    import torch

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    for batch in warmup_batches:                    # not timed
        adapter.encode(batch)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()        # exclude warm-up from the peak

    latencies, real_tokens, padded_tokens, n_samples = [], [], [], 0
    monitor.start()
    try:
        for batch in timed_batches:
            real, padded = adapter.count_tokens(batch, max_length)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            adapter.encode(batch)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
            real_tokens.append(real)
            padded_tokens.append(padded)
            n_samples += len(batch)
    finally:
        res = monitor.stop()

    total_s = sum(latencies) / 1000.0
    out = {
        "actual_token_count": round(statistics.mean(real_tokens), 1),
        "padded_token_count": round(statistics.mean(padded_tokens), 1),
        "latency_mean_ms": round(statistics.mean(latencies), 3),
        "latency_p50_ms": round(pct(latencies, 50), 3),
        "latency_p95_ms": round(pct(latencies, 95), 3),
        "latency_p99_ms": round(pct(latencies, 99), 3),
        # Totals over the timed encodes: for a fixed batch size this equals
        # batch_size / mean latency, and it stays correct for single mode.
        "samples_per_sec": round(n_samples / total_s, 2),
        "tokens_per_sec": round(sum(real_tokens) / total_s, 1),
        "runs": len(latencies),
        **res,
    }
    if device == "cuda":
        out["vram_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
        out["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        out["peak_vram_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 1e6, 1)
    return out


def _run(adapter, dataset, max_length, batch_size, monitor, device,
         warmup_batches, timed_batches):
    row = {"model": adapter.name, "dataset": dataset, "max_length": max_length,
           "batch_size": batch_size, "status": "ok", "runs": 0, "error": ""}
    if not adapter.supports(max_length):
        row["status"] = "unsupported"
        row["error"] = f"max_length {max_length} > model cap {adapter.cfg['cap']}"
        return row
    adapter.set_max_length(max_length)
    try:
        row.update(measure(adapter, warmup_batches, timed_batches, max_length,
                           monitor, device))
    except Exception as exc:                        # noqa: BLE001 - reported per row
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()
        row["status"] = "OOM" if is_oom(exc) else "error"
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return row


def run_configuration(adapter, texts, dataset, max_length, batch_size, monitor,
                      device, warmup=WARMUP_RUNS, runs=BENCH_RUNS):
    """Batch mode: `runs` timed batches of `batch_size` over a rotating window."""
    return _run(adapter, dataset, max_length, batch_size, monitor, device,
                [make_batch(texts, batch_size, i) for i in range(warmup)],
                [make_batch(texts, batch_size, warmup + i) for i in range(runs)])


def run_single(adapter, texts, dataset, max_length, monitor, device,
               warmup=SINGLE_WARMUP, samples=SINGLE_SAMPLES):
    """Single mode: one request per sample, sequentially, over distinct samples."""
    n = len(texts) if not samples else min(samples, len(texts))
    return _run(adapter, dataset, max_length, "single", monitor, device,
                [[t] for t in texts[:warmup]],
                [[t] for t in texts[:n]])


def merge_single_rows(single_rows, target_path):
    """Put single-mode rows into an existing results CSV, ahead of each group's batch rows.

    A group is (model, dataset, max_length). Any single row already in the target for a
    group being merged is replaced, so re-running is safe; batch rows are never touched.
    Groups the target does not have are appended at the end. The target is backed up
    once to <name>.bak.csv before the first write.
    """
    import shutil

    def key(r):
        return (str(r["model"]), str(r["dataset"]), str(r["max_length"]))

    with open(target_path, encoding="utf-8-sig") as f:
        target = list(csv.DictReader(f))
    new = {key(r): {k: r.get(k, "") for k in CSV_FIELDS}
           for r in single_rows if str(r["batch_size"]) == "single"}
    if not new:
        raise SystemExit("no single-mode rows to merge")

    merged, placed = [], set()
    for r in target:
        k = key(r)
        if k in new and k not in placed:
            merged.append(new[k])
            placed.add(k)
        if k in new and r["batch_size"] == "single":
            continue                                # replaced by the new single row
        merged.append(r)
    appended = [v for k, v in new.items() if k not in placed]
    merged += appended

    bak = os.path.splitext(target_path)[0] + ".bak.csv"
    if not os.path.exists(bak):
        shutil.copyfile(target_path, bak)
        print(f"backup -> {rel(bak)}")
    with open(target_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in merged:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    kept_batch = sum(1 for r in target if r["batch_size"] != "single")
    print(f"merged {len(new)} single rows into {rel(target_path)} "
          f"({kept_batch} batch rows kept, {len(appended)} single rows appended as new groups, "
          f"{len(merged)} rows total)")


def print_summary(rows):
    print(f"\n{'=' * 118}\nSUMMARY\n{'=' * 118}")
    print(f"{'model':<14}{'dataset':<8}{'maxlen':>7}{'bs':>7}{'status':>12}"
          f"{'tok/batch':>10}{'mean ms':>10}{'p95 ms':>9}{'p99 ms':>9}"
          f"{'samp/s':>12}{'tok/s':>13}{'GPU%':>6}{'peakVRAM':>10}")
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['model']:<14}{r['dataset']:<8}{r['max_length']:>7}"
                  f"{str(r['batch_size']):>7}{r['status']:>12}"
                  + (f"   {r['error'][:60]}" if r["error"] else ""))
            continue
        print(f"{r['model']:<14}{r['dataset']:<8}{r['max_length']:>7}{str(r['batch_size']):>7}"
              f"{r['status']:>12}{r['actual_token_count']:>10.0f}"
              f"{r['latency_mean_ms']:>10.1f}{r['latency_p95_ms']:>9.1f}"
              f"{r['latency_p99_ms']:>9.1f}{r['samples_per_sec']:>12.1f}"
              f"{r['tokens_per_sec']:>13.0f}"
              f"{(r.get('gpu_utilization') if r.get('gpu_utilization') is not None else -1):>6.0f}"
              f"{(r.get('peak_vram_mb') or 0):>10.0f}")

    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        print(f"\nbest throughput per model/dataset (samples/sec):")
        best = {}
        for r in ok:
            key = (r["model"], r["dataset"], r["max_length"])
            if key not in best or r["samples_per_sec"] > best[key]["samples_per_sec"]:
                best[key] = r
        for (m, d, L), r in sorted(best.items()):
            print(f"  {m:<14}{d:<8}maxlen={L:<6}bs={str(r['batch_size']):<7}"
                  f"{r['samples_per_sec']:>12.1f} samp/s  "
                  f"{r['tokens_per_sec']:>13.0f} tok/s")
    counts = defaultdict(int)
    for r in rows:
        counts[r["status"]] += 1
    print("\nconfigurations: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=BATCH_SIZES)
    ap.add_argument("--warmup", type=int, default=WARMUP_RUNS)
    ap.add_argument("--runs", type=int, default=BENCH_RUNS)
    ap.add_argument("--output", default=OUTPUT_CSV)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fp32", dest="fp16", action="store_false", default=FP16)
    ap.add_argument("--no-single", dest="single", action="store_false", default=SINGLE_MODE,
                    help="skip the single-request mode")
    ap.add_argument("--single-warmup", type=int, default=SINGLE_WARMUP)
    ap.add_argument("--single-samples", type=int, default=SINGLE_SAMPLES,
                    help="requests timed in single mode (default: whole dataset)")
    ap.add_argument("--no-shuffle", dest="shuffle", action="store_false",
                    default=SHUFFLE_SAMPLES, help="keep the dataset's sorted order")
    ap.add_argument("--single-only", action="store_true",
                    help=f"run only single mode; writes {SINGLE_ONLY_CSV} unless --output is given")
    ap.add_argument("--merge-into", default=None,
                    help="after the run, merge single rows into this existing results CSV")
    ap.add_argument("--merge-from", default=None,
                    help="skip benchmarking: merge single rows from this CSV into --merge-into")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.merge_from:
        if not args.merge_into:
            ap.error("--merge-from needs --merge-into")
        with open(abspath(args.merge_from), encoding="utf-8-sig") as f:
            merge_single_rows(list(csv.DictReader(f)), abspath(args.merge_into))
        return
    if args.single_only:
        args.single = True
        args.batch_sizes = []
        if args.output == OUTPUT_CSV:              # never overwrite the batch results
            args.output = SINGLE_ONLY_CSV

    print("datasets:")
    datasets = load_datasets(args.shuffle)
    configs = [(d, L) for d, L in CONFIGURATIONS if d in datasets]
    modes = (["single"] if args.single else []) + list(args.batch_sizes)
    grid = [(m, d, L, b) for m in args.models for d, L in configs for b in modes]
    print(f"\ngrid: {len(grid)} configurations "
          f"({len(args.models)} models x {len(configs)} dataset/max_length x "
          f"{len(modes)} modes: {', '.join(map(str, modes))})")
    if args.single:
        n_single = args.single_samples or "all"
        print(f"single: warmup {args.single_warmup} (untimed), {n_single} requests timed")
    if args.batch_sizes:
        print(f"batch:  warmup {args.warmup} (untimed), benchmark {args.runs} runs each")
    print(f"output: {args.output}"
          + (f"  (then merged into {args.merge_into})" if args.merge_into else ""))

    if args.dry_run:
        for m, d, L, b in grid:
            supported = L <= MODELS[m]["cap"]
            print(f"  {m:<14}{d:<8}maxlen={L:<6}bs={str(b):<7}"
                  f"{'' if supported else '-> unsupported'}")
        return

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    if device == "cpu":
        print("  no GPU - VRAM and GPU utilisation columns will be empty")
    monitor = ResourceMonitor()
    if monitor.handle is None:
        print("  pynvml unavailable - gpu_utilization will be empty "
              "(pip install nvidia-ml-py)")
    if monitor.psutil is None:
        print("  psutil unavailable - RAM columns will be empty (pip install psutil)")

    rows = []
    for name in args.models:
        cfg = MODELS[name]
        print(f"\n{'=' * 78}\n{name}  ({cfg['hf_id']})\n{'=' * 78}")
        adapter = ADAPTERS[name](name, cfg, device, args.fp16)
        t0 = time.perf_counter()
        try:
            adapter.load()
            print(f"  loaded in {time.perf_counter() - t0:.1f}s, "
                  f"dim={adapter.model.get_sentence_embedding_dimension()}")
        except Exception as exc:                    # noqa: BLE001
            print(f"  load failed: {exc}")
            for d, L in configs:
                for b in modes:
                    rows.append({"model": name, "dataset": d, "max_length": L,
                                 "batch_size": b, "status": "error", "runs": 0,
                                 "error": f"load failed: {exc}"[:300]})
            continue

        for d, L in configs:
            if args.single:
                r = run_single(adapter, datasets[d], d, L, monitor, device,
                               args.single_warmup, args.single_samples)
                rows.append(r)
                if r["status"] == "ok":
                    print(f"  {d:<8}maxlen={L:<6}single "
                          f"{r['latency_mean_ms']:>9.1f} ms/req  "
                          f"p95 {r['latency_p95_ms']:>8.1f}  p99 {r['latency_p99_ms']:>8.1f}  "
                          f"{r['samples_per_sec']:>8.1f} req/s  "
                          f"({r['runs']} requests)")
                else:
                    print(f"  {d:<8}maxlen={L:<6}single {r['status']:>12}  {r['error'][:70]}")
            oom_from = None
            for b in args.batch_sizes:
                if oom_from is not None and SKIP_LARGER_AFTER_OOM and b > oom_from:
                    rows.append({"model": name, "dataset": d, "max_length": L,
                                 "batch_size": b, "status": "OOM", "runs": 0,
                                 "error": f"skipped: batch {oom_from} already OOM"})
                    print(f"  {d:<8}maxlen={L:<6}bs={b:<4} OOM (skipped)")
                    continue
                r = run_configuration(adapter, datasets[d], d, L, b, monitor,
                                      device, args.warmup, args.runs)
                rows.append(r)
                if r["status"] == "ok":
                    print(f"  {d:<8}maxlen={L:<6}bs={b:<4}"
                          f"{r['latency_mean_ms']:>9.1f} ms  "
                          f"{r['samples_per_sec']:>8.1f} samp/s  "
                          f"{r['tokens_per_sec']:>8.0f} tok/s  "
                          f"peakVRAM {r.get('peak_vram_mb') or 0:.0f} MB")
                else:
                    print(f"  {d:<8}maxlen={L:<6}bs={b:<4}{r['status']:>12}  "
                          f"{r['error'][:70]}")
                    if r["status"] == "OOM":
                        oom_from = b

        adapter.free()
        gc.collect()
        if device == "cuda":                        # keep models from skewing each other
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    out = abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    print_summary(rows)
    print(f"\nwrote {len(rows)} rows -> {rel(out)}")
    if args.merge_into:
        merge_single_rows(rows, abspath(args.merge_into))

    meta = {"device": device, "fp16": args.fp16, "warmup_runs": args.warmup,
            "bench_runs": args.runs, "batch_sizes": args.batch_sizes,
            "single_mode": args.single, "single_warmup": args.single_warmup,
            "single_samples": args.single_samples or "all",
            "shuffle_samples": args.shuffle, "seed": SEED,
            "configurations": CONFIGURATIONS, "rotate_samples": ROTATE_SAMPLES,
            "normalize": NORMALIZE, "convert_to_numpy": CONVERT_TO_NUMPY,
            "models": {m: MODELS[m] for m in args.models}}
    with open(os.path.splitext(out)[0] + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
