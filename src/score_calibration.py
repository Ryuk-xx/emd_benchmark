"""Score the calibration pairs and write results/calibration_report.xlsx.

For every model/mode that has vectors, computes cosine(text_a, text_b) per pair and
checks whether the tier ladder comes out in the intended order.

  python src/score_calibration.py
  python src/score_calibration.py --models qwen3_0.6b vn_embedding

The number that decides deployability is the **T1 - T2 gap**: T1 pairs are the same
fact reworded, T2 pairs are the same document with a *different attribute* (issue date
vs effective date). A model whose T1 and T2 scores sit on top of each other will
happily return the wrong date, and no similarity threshold can separate the two.
"""
import argparse
import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D, E, R = (os.path.join(ROOT, x) for x in ("data", "embeddings", "results"))
CSV_PATH = os.path.join(D, "embedding_calibration_testcases.csv")

# The ladder the test set is designed around, from most to least similar.
TIER_ORDER = ["0_sanity_identical", "1_paraphrase", "5_vocab_gap",
              "2_same_entity_diff_attribute_DANGER", "3_same_topic_diff_entity",
              "4_unrelated"]


def load_pairs():
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_cosines(model, mode, pair_ids):
    """Cosine per pair for one model/mode, or None when its vectors are missing."""
    d = os.path.join(E, model, mode)
    pa, pb = os.path.join(d, "calibration_a.npy"), os.path.join(d, "calibration_b.npy")
    ia = os.path.join(d, "ids_calibration_a.json")
    ib = os.path.join(d, "ids_calibration_b.json")
    if not all(os.path.exists(p) for p in (pa, pb, ia, ib)):
        return None

    A, B = np.load(pa), np.load(pb)
    with open(ia, encoding="utf-8") as f:
        ids_a = json.load(f)
    with open(ib, encoding="utf-8") as f:
        ids_b = json.load(f)

    # The two sides are only comparable row-wise if both id lists agree, and the
    # report is only joinable to the CSV if they follow it.
    assert ids_a == ids_b, f"{model}/{mode}: calibration a/b id lists differ"
    assert ids_a == pair_ids, f"{model}/{mode}: ids do not match the CSV order"
    assert A.shape == B.shape, f"{model}/{mode}: {A.shape} vs {B.shape}"

    A = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    B = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-12, None)
    return (A * B).sum(axis=1)


def tier_means(tiers, cos):
    out = {}
    for t in sorted(set(tiers)):
        v = np.array([c for c, tt in zip(cos, tiers) if tt == t])
        out[t] = float(v.mean())
    return out


def diagnostics(tiers, cos):
    """Per model/mode summary. Everything here is a decision input, not decoration."""
    m = tier_means(tiers, cos)
    t1 = m.get("1_paraphrase")
    t2 = m.get("2_same_entity_diff_attribute_DANGER")
    t4 = m.get("4_unrelated")
    t5 = m.get("5_vocab_gap")

    t3 = m.get("3_same_topic_diff_entity")
    present = [t for t in TIER_ORDER if t in m]
    ladder_ok = all(m[a] >= m[b] for a, b in zip(present, present[1:]))

    def gt(x, y):
        return None if None in (x, y) else bool(x > y)

    d = {
        "sanity_T0": m.get("0_sanity_identical"),
        "T1_paraphrase": t1,
        "T5_vocab_gap": t5,
        "T2_same_doc_other_attribute": t2,
        "T3_same_topic_diff_entity": t3,
        "T4_unrelated": t4,
        # Each rung of the ladder checked on its own, so a failure names itself
        # instead of collapsing into one boolean.
        "ok_T1_above_T2": gt(t1, t2),
        # The rung that matters most in production: a differently-worded correct
        # passage must outrank a same-document wrong-attribute one.
        "ok_T5_above_T2": gt(t5, t2),
        "ok_T2_above_T3": gt(t2, t3),
        "ok_T3_above_T4": gt(t3, t4),
        # The decisive separation, and the same distance as a share of the model's
        # own usable range, which is what makes it comparable across models.
        "gap_T1_minus_T2": None if None in (t1, t2) else t1 - t2,
        "gap_T1_minus_T4": None if None in (t1, t4) else t1 - t4,
        "ladder_in_order": ladder_ok,
        "cos_min": float(np.min(cos)),
        "cos_max": float(np.max(cos)),
        "cos_spread": float(np.max(cos) - np.min(cos)),
    }
    if d["gap_T1_minus_T2"] is not None and d["cos_spread"] > 0:
        d["gap_T1_T2_as_pct_of_spread"] = 100 * d["gap_T1_minus_T2"] / d["cos_spread"]
    return d


def autosize(ws, limits):
    from openpyxl.utils import get_column_letter
    for i, col in enumerate(ws.columns, 1):
        width = max((len(str(c.value)) for c in col if c.value is not None), default=8)
        ws.column_dimensions[get_column_letter(i)].width = min(width + 2, limits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3_0.6b", "vn_embedding",
                                                    "ada002", "text3large"])
    ap.add_argument("--modes", nargs="+", default=["no_instruct", "instruct"])
    ap.add_argument("--out", default=os.path.join(R, "calibration_report.xlsx"))
    args = ap.parse_args()

    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill
    except ImportError:
        raise SystemExit("pip install openpyxl")

    pairs = load_pairs()
    pair_ids = [p["pair_id"] for p in pairs]
    tiers = [p["tier"] for p in pairs]
    print(f"{len(pairs)} calibration pairs from {os.path.basename(CSV_PATH)}")

    cols = {}                                    # "model/mode" -> cosine array
    for model in args.models:
        for mode in args.modes:
            cos = load_cosines(model, mode, pair_ids)
            if cos is None:
                print(f"  {model}/{mode}: no vectors, skipped")
                continue
            cols[f"{model}/{mode}"] = cos
            print(f"  {model}/{mode}: scored")

    if not cols:
        raise SystemExit("no model/mode has calibration vectors yet - "
                         "run src/embed_local.py first")

    wb = openpyxl.Workbook()
    bold = Font(bold=True)
    head_fill = PatternFill("solid", fgColor="DDDDDD")
    warn_fill = PatternFill("solid", fgColor="FFC7CE")

    # --- per pair -----------------------------------------------------------
    ws = wb.active
    ws.title = "per_pair"
    header = ["pair_id", "tier", "expected_relation", "risk_note",
              "text_a", "text_b"] + list(cols)
    ws.append(header)
    for i, p in enumerate(pairs):
        ws.append([p["pair_id"], p["tier"], p.get("expected_relation", ""),
                   p.get("risk_note", ""), p["text_a"], p["text_b"]]
                  + [round(float(c[i]), 4) for c in cols.values()])
    for c in ws[1]:
        c.font, c.fill = bold, head_fill
    ws.freeze_panes = "A2"
    autosize(ws, 55)

    # --- tier means ---------------------------------------------------------
    ws = wb.create_sheet("by_tier")
    ws.append(["tier", "n"] + list(cols))
    counts = {t: tiers.count(t) for t in set(tiers)}
    for t in [x for x in TIER_ORDER if x in counts] + \
             [x for x in sorted(counts) if x not in TIER_ORDER]:
        row = [t, counts[t]]
        for c in cols.values():
            v = np.array([x for x, tt in zip(c, tiers) if tt == t])
            row.append(round(float(v.mean()), 4))
        ws.append(row)
    for c in ws[1]:
        c.font, c.fill = bold, head_fill
    autosize(ws, 42)

    # --- diagnostics --------------------------------------------------------
    ws = wb.create_sheet("diagnostics")
    diags = {k: diagnostics(tiers, c) for k, c in cols.items()}
    keys = list(next(iter(diags.values())))
    ws.append(["metric"] + list(diags))
    for k in keys:
        row = [k]
        for d in diags.values():
            v = d.get(k)
            row.append(round(v, 4) if isinstance(v, float) else v)
        ws.append(row)
    for c in ws[1]:
        c.font, c.fill = bold, head_fill
    # Flag the two failure signatures that make a model undeployable here.
    for r in range(2, ws.max_row + 1):
        metric = ws.cell(r, 1).value
        for c in range(2, ws.max_column + 1):
            v = ws.cell(r, c).value
            if metric == "gap_T1_minus_T2" and isinstance(v, float) and v < 0.05:
                ws.cell(r, c).fill = warn_fill
            if str(metric).startswith("ok_") and v is False:
                ws.cell(r, c).fill = warn_fill
            if metric == "ladder_in_order" and v is False:
                ws.cell(r, c).fill = warn_fill
    autosize(ws, 42)

    # --- instruct effect ----------------------------------------------------
    models_both = [m for m in args.models
                   if f"{m}/instruct" in cols and f"{m}/no_instruct" in cols]
    if models_both:
        ws = wb.create_sheet("instruct_effect")
        ws.append(["pair_id", "tier"]
                  + [f"{m}: instruct - no_instruct" for m in models_both])
        for i, p in enumerate(pairs):
            ws.append([p["pair_id"], p["tier"]]
                      + [round(float(cols[f"{m}/instruct"][i]
                                     - cols[f"{m}/no_instruct"][i]), 4)
                         for m in models_both])
        ws.append([])
        ws.append(["MEAN", ""] + [round(float((cols[f"{m}/instruct"]
                                               - cols[f"{m}/no_instruct"]).mean()), 4)
                                  for m in models_both])
        for c in ws[1]:
            c.font, c.fill = bold, head_fill
        ws[f"A{ws.max_row}"].font = bold
        autosize(ws, 42)

    os.makedirs(R, exist_ok=True)
    wb.save(args.out)

    # --- CSV: the source file with a cosine column appended per model/mode ------
    # Row order and the original columns are preserved exactly, so this diffs
    # cleanly against the input and joins on pair_id.
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        src_fields = csv.DictReader(f).fieldnames
    score_fields = [f"cos_{k.replace('/', '_')}" for k in cols]
    csv_out = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(src_fields) + score_fields)
        w.writeheader()
        for i, p in enumerate(pairs):
            row = {k: p.get(k, "") for k in src_fields}
            for name, c in zip(score_fields, cols.values()):
                row[name] = f"{float(c[i]):.4f}"
            w.writerow(row)
    print(f"csv    -> {os.path.relpath(csv_out, ROOT)}")

    # --- console summary ----------------------------------------------------
    print(f"\n{'model/mode':<28}{'T0':>7}{'T1':>7}{'T5':>7}{'T2':>7}{'T3':>7}"
          f"{'T4':>7}{'T1-T2':>8}")
    for k, d in diags.items():
        print(f"{k:<28}{d['sanity_T0']:>7.3f}{d['T1_paraphrase']:>7.3f}"
              f"{d['T5_vocab_gap']:>7.3f}{d['T2_same_doc_other_attribute']:>7.3f}"
              f"{d['T3_same_topic_diff_entity']:>7.3f}{d['T4_unrelated']:>7.3f}"
              f"{d['gap_T1_minus_T2']:>8.3f}")

    print("\nrung checks (False = the ladder failed at that step)")
    rungs = [k for k in next(iter(diags.values())) if k.startswith("ok_")]
    print(f"{'model/mode':<28}" + "".join(f"{r[3:]:>18}" for r in rungs))
    for k, d in diags.items():
        print(f"{k:<28}" + "".join(f"{str(d[r]):>18}" for r in rungs))
    print(f"\nreport -> {os.path.relpath(args.out, ROOT)}")


if __name__ == "__main__":
    main()
