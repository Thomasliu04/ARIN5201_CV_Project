"""
Put several results CSVs from train_baseline.py side by side (Markdown table),
so resize vs crop (and different seeds / models) can be compared at a glance.

Usage:
    python scripts\\compare_results.py results_resize_b16.csv results_crop128_b16.csv
    python scripts\\compare_results.py results_*.csv --metric auc --out comparison_auc.md
Metrics: acc bacc auc ap f1 recall real_fpr
"""
import argparse
import csv
import glob
from pathlib import Path

GROUPS = [
    ("Held-out (seen generator)", lambda s: s in ("val", "test_seen")),
    ("Unseen generators", lambda s: s.startswith("test_unseen_")),
    ("Bias tests (seen)", lambda s: s.startswith("test_bias_") and "unseen_" not in s and
     any(k in s for k in ("jpeg444", "jpeg420", "size_", "real_down"))),
    ("Degradations (seen)", lambda s: s.startswith("test_bias_") and "unseen_" not in s),
    ("Bias tests (unseen)", lambda s: s.startswith("test_bias_unseen_") and
     any(k in s for k in ("jpeg444", "jpeg420", "size_", "real_down"))),
    ("Degradations (unseen)", lambda s: s.startswith("test_bias_unseen_")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--metric", default="acc")
    ap.add_argument("--out", default=None, help="also write the table to this .md file")
    args = ap.parse_args()

    files = [f for pat in args.files for f in sorted(glob.glob(pat))] or args.files
    runs = {}
    for f in files:
        with open(f, newline="") as fh:
            runs[Path(f).stem] = {r["set"]: r for r in csv.DictReader(fh)}
    sets = sorted({s for r in runs.values() for s in r})
    names = list(runs)

    lines = [f"Metric: **{args.metric}**", "",
             "| Test set | " + " | ".join(names) + (" | Δ (last − first) |" if len(names) > 1 else " |"),
             "|---|" + "---|" * len(names) + ("---|" if len(names) > 1 else "")]
    used = set()
    for title, pred in GROUPS:
        group = [s for s in sets if pred(s) and s not in used]
        if not group:
            continue
        used.update(group)
        lines.append(f"| **{title}** |" + " |" * len(names) + (" |" if len(names) > 1 else ""))
        for s in group:
            vals = []
            for n in names:
                v = runs[n].get(s, {}).get(args.metric)
                vals.append(float(v) if v not in (None, "", "nan") else None)
            cells = [f"{v * 100:.1f}" if v is not None else "–" for v in vals]
            delta = ""
            if len(names) > 1 and vals[0] is not None and vals[-1] is not None:
                delta = f" {(vals[-1] - vals[0]) * 100:+.1f} |"
            elif len(names) > 1:
                delta = " – |"
            lines.append(f"| {s.replace('test_bias_', '').replace('test_', '')} | " + " | ".join(cells) + " |" + delta)
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
