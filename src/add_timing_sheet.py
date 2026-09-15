"""Add a `thong_ke_model` sheet to a workbook: per-model embedding statistics for
the corpus, fact and bo_cau_hoi inputs, read from embeddings/<model>/manifest.json.

  python src/add_timing_sheet.py --workbook "data/bo_cau_hoi_681cbdeb_60 (1).xlsx"
  python src/add_timing_sheet.py --workbook ... --sheet thong_ke_model --dry-run

Three tables, one per input, each with a row per model/mode that has that input.
Every number is what embed_local.py recorded when it ran: sequence window, batch
size, token counts from the model's own tokenizer, encode time, throughput, per-batch
timing percentiles, and single-item latency measured after warm-up. Peak VRAM and
model load time are recorded per model run, not per input, so they describe the most
recent run of that model and repeat across its rows.

The sheet is replaced if it already exists; every other sheet is left untouched.
"""
import argparse
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E = os.path.join(ROOT, "embeddings")

MODELS = [("vn_embedding", "vn-embedding"),
          ("qwen3_0.6b", "qwen3-0.6b"),
          ("qwen3_vl_2b", "qwen3-vl-2b")]
MODES = ["no_instruct", "instruct"]
# vn_embedding defines no instruction; an instruct entry in its manifest is a leftover.
SKIP = {("vn_embedding", "instruct")}
INPUTS = [("corpus", "CORPUS (2.038 chunk)"),
          ("fact", "FACT (19.529 fact)"),
          ("bo_cau_hoi", "BO_CAU_HOI (60 câu hỏi)")]

# Runs made before the manifest recorded window/batch used these defaults.
LEGACY_WINDOW = {"vn_embedding": 2048, "qwen3_0.6b": 2048, "qwen3_vl_2b": 2048}
LEGACY_BATCH = 32

HEADER = [
    "model", "mode", "số chiều", "số items", "kind", "prompt",
    "window (token)", "batch size",
    "tổng token", "token max", "số bị cắt",
    "encode (s)", "items/s", "tokens/s",
    "batch p50 (s)", "batch p95 (s)", "batch đầu (s)",
    "latency 1 item p50 (ms)", "latency 1 item p95 (ms)",
    "peak VRAM (GB)*", "load model (s)*", "ghi chú",
]


def load_manifest(model):
    p = os.path.join(E, model, "manifest.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def row_for(model, label, mode, man, inp):
    e = man.get("modes", {}).get(mode, {}).get(inp)
    if not e or not e.get("n"):
        return None
    note = []
    window = e.get("max_seq_length")
    batch = e.get("batch_size")
    if window is None:
        window = LEGACY_WINDOW[model]
        note.append("window/batch không ghi trong manifest (lượt chạy cũ), lấy mặc định lúc đó")
    if batch is None:
        batch = LEGACY_BATCH
    if e.get("identical_to_mode"):
        note.append(f"vector giống {e['identical_to_mode']} (copy, không encode lại)")
    if e.get("tokens_truncated"):
        note.append(f"{e['tokens_truncated']} items vượt window, bị cắt")
    b = e.get("batch_seconds", {})
    lat = e.get("single_item_latency_ms", {})
    return [
        label, mode, man.get("dim"), e.get("n"), e.get("kind"),
        e.get("prompt") or "",
        window, batch,
        e.get("tokens_total"), e.get("tokens_max"), e.get("tokens_truncated"),
        e.get("encode_seconds"), e.get("items_per_second"), e.get("tokens_per_second"),
        b.get("p50"), b.get("p95"), b.get("first"),
        lat.get("p50"), lat.get("p95"),
        man.get("peak_vram_gb"), man.get("load_seconds"),
        "; ".join(note),
    ]


def build_tables():
    tables = []
    for inp, title in INPUTS:
        rows = []
        for model, label in MODELS:
            man = load_manifest(model)
            if not man:
                continue
            for mode in MODES:
                if (model, mode) in SKIP:
                    continue
                r = row_for(model, label, mode, man, inp)
                if r:
                    rows.append(r)
        tables.append((title, rows))
    return tables


def write_sheet(wb, name, tables):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    if name in wb.sheetnames:
        del wb[name]
    ws = wb.create_sheet(name)
    bold = Font(bold=True)
    title_font = Font(bold=True, size=12)
    head_fill = PatternFill("solid", fgColor="DDEBF7")
    title_fill = PatternFill("solid", fgColor="BDD7EE")

    r = 1
    for title, rows in tables:
        ws.cell(r, 1, title).font = title_font
        for c in range(1, len(HEADER) + 1):
            ws.cell(r, c).fill = title_fill
        r += 1
        for c, h in enumerate(HEADER, 1):
            cell = ws.cell(r, c, h)
            cell.font, cell.fill = bold, head_fill
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        r += 1
        if not rows:
            ws.cell(r, 1, "(chưa có manifest cho input này)")
            r += 1
        for row in rows:
            for c, v in enumerate(row, 1):
                ws.cell(r, c, v)
            r += 1
        r += 1                                        # blank line between tables

    notes = [
        "* peak VRAM và load model ghi theo LƯỢT CHẠY gần nhất của model đó (không theo từng input), nên lặp lại trên các dòng của cùng model.",
        "Latency 1 item: đo riêng sau warm-up, encode từng câu một (đường query production), không suy từ throughput batch.",
        "Token đếm bằng tokenizer của chính model; 'số bị cắt' = số items dài hơn window.",
        "Nguồn: embeddings/<model>/manifest.json do src/embed_local.py ghi.",
    ]
    for n in notes:
        ws.cell(r, 1, n).font = Font(italic=True, color="555555")
        r += 1

    widths = {1: 16, 2: 12, 6: 40, 22: 60}
    for c in range(1, len(HEADER) + 1):
        ws.column_dimensions[get_column_letter(c)].width = widths.get(c, 13)
    ws.freeze_panes = "A1"
    return ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", required=True)
    ap.add_argument("--sheet", default="thong_ke_model")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import openpyxl

    tables = build_tables()
    for title, rows in tables:
        print(f"\n{title}")
        print(f"  {'model':<14}{'mode':<13}{'n':>7}{'win':>6}{'batch':>6}{'tok max':>9}"
              f"{'cắt':>5}{'enc s':>9}{'items/s':>9}{'p50 ms':>8}{'VRAM':>6}")
        for r in rows:
            print(f"  {r[0]:<14}{r[1]:<13}{r[3]:>7}{r[6]:>6}{r[7]:>6}{r[9]:>9}"
                  f"{r[10]:>5}{r[11]:>9}{r[12]:>9}{r[17]:>8}{str(r[19]):>6}")

    if args.dry_run:
        print("\ndry run - workbook untouched")
        return

    wb = openpyxl.load_workbook(args.workbook)
    out = args.out or args.workbook
    if out == args.workbook:
        bak = os.path.splitext(args.workbook)[0] + ".bak.xlsx"
        if not os.path.exists(bak):
            shutil.copyfile(args.workbook, bak)
            print(f"\nbackup -> {bak}")
    write_sheet(wb, args.sheet, tables)
    wb.save(out)
    print(f"sheet '{args.sheet}' -> {out}")


if __name__ == "__main__":
    main()
