#!/usr/bin/env python3
"""Compare evaluated variants against a reference with a paired bootstrap on the test trajectories.

    python scripts/compare.py                       # every <out>/results/*_test.json against 'full'
    python scripts/compare.py --reference full --variants no_cbam lstm early_fusion

Reports the F1 difference, its 95% CI and a two-sided p-value for each pair, then Benjamini-Hochberg
adjusted p-values across all pairs. With 11 test trajectories, expect wide intervals.
Writes <out>/results/comparison.csv.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trimodal.config import DEFAULT_OUT, Paths  # noqa: E402
from trimodal.stats import benjamini_hochberg, paired_bootstrap  # noqa: E402


def load(results_dir, variant):
    with open(results_dir / f"{variant}_test.json") as f:
        res = json.load(f)
    return [r["label"] for r in res["per_trajectory"]], [r["pred"] for r in res["per_trajectory"]]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", default="full")
    p.add_argument("--variants", nargs="*", help="default: every evaluated variant except the reference")
    p.add_argument("--out", help=f"output folder (default {DEFAULT_OUT})")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--q", type=float, default=0.05, help="false-discovery rate for Benjamini-Hochberg")
    args = p.parse_args()

    results_dir = Paths.resolve(out=args.out).out / "results"
    variants = args.variants or sorted(f.name[:-len("_test.json")] for f in results_dir.glob("*_test.json")
                                       if f.name != f"{args.reference}_test.json")
    if not variants:
        sys.exit(f"error: nothing to compare in {results_dir} (run scripts/evaluate.py first)")
    y, ref = load(results_dir, args.reference)

    rows = []
    for v in variants:
        y_v, pred_v = load(results_dir, v)
        assert y_v == y, f"{v}: test labels differ from the reference"
        r = paired_bootstrap(y, ref, pred_v, n_boot=args.n_boot)
        rows.append({"reference": args.reference, "variant": v, **{k: round(x, 4) for k, x in r.items()}})
    for row, bh in zip(rows, benjamini_hochberg([r["p_value"] for r in rows], args.q)):
        row["p_adj_bh"], row["significant"] = round(bh["p_adj"], 4), bh["significant"]

    print(f"{'variant':<16}{'dF1 (ref - variant)':>21}{'95% CI':>18}{'p':>8}{'p (BH)':>9}")
    for r in rows:
        ci = f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]"
        star = "  *" if r["significant"] else ""
        print(f"{r['variant']:<16}{r['delta_f1']:>+21.3f}{ci:>18}{r['p_value']:>8.3f}{r['p_adj_bh']:>9.3f}{star}")
    path = results_dir / "comparison.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
