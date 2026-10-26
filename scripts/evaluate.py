#!/usr/bin/env python3
"""Evaluate a trained variant on the held-out test split.

    python scripts/evaluate.py --variant full                      # checkpoints from scripts/train.py
    python scripts/evaluate.py --variant full --runs path/to/runs  # any folder with seed_<k>/model_state.pt

The seeds' probabilities are averaged; the decision threshold is the median of thresholds calibrated
on k stratified folds of the train + validation trajectories; the test set is scored once.
Writes <out>/results/<variant>_test.json with metrics, a 95% bootstrap CI for F1 and per-trajectory
predictions (used by scripts/compare.py).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trimodal.cli import add_path_args, resolve_paths, write_json  # noqa: E402
from trimodal.config import VARIANTS, pick_device  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default="full", choices=sorted(VARIANTS))
    p.add_argument("--runs", help="folder holding seed_<k>/model_state.pt (default <out>/runs/<variant>)")
    p.add_argument("--n-boot", type=int, default=10_000)
    add_path_args(p)
    args = p.parse_args()

    paths = resolve_paths(args)
    device = pick_device(args.device)
    run_root = Path(args.runs) if args.runs else paths.out / "runs" / args.variant

    from trimodal.data import make_loaders
    from trimodal.evaluation import evaluate_ensemble, find_checkpoints
    from trimodal.stats import bootstrap_f1_ci

    ckpts = find_checkpoints(run_root)
    if not ckpts:
        sys.exit(f"error: no seed_<k>/model_state.pt found under {run_root}")
    print(f"{args.variant}: {len(ckpts)} checkpoints under {run_root}")

    loaders = make_loaders(paths, batch_size=args.batch_size)
    res = evaluate_ensemble(ckpts, args.variant, loaders, device)
    ci = bootstrap_f1_ci([r["label"] for r in res["per_trajectory"]], [r["pred"] for r in res["per_trajectory"]],
                         n_boot=args.n_boot)
    res["test"]["f1_ci95"] = [round(ci["ci_low"], 4), round(ci["ci_high"], 4)]

    t, thr = res["test"], res["threshold"]
    print(f"\nthreshold {thr['threshold']:.3f} (median of K={thr['K']} folds, spread {thr['spread']:.3f})")
    print(f"TEST  F1 {t['f1']:.4f}  [95% CI {t['f1_ci95'][0]:.2f}, {t['f1_ci95'][1]:.2f}]   "
          f"AUC {t['auc']:.4f}   (all-spoiled baseline F1 {t['trivial_all_spoiled_f1']:.4f})")
    print(f"      MAE {t['mae']:.4f} days   RMSE {t['rmse']:.4f}   R2 {t['r2']:.4f}")
    for r in res["per_trajectory"]:
        mark = "" if r["pred"] == r["label"] else "   <- wrong"
        print(f"   {r['fruit']:<18} label={r['label']} p={r['p_spoiled']:.3f} pred={r['pred']}{mark}")
    write_json(res, paths.out / "results" / f"{args.variant}_test.json")


if __name__ == "__main__":
    main()
