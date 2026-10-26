#!/usr/bin/env python3
"""Missing-modality sweep: hide a share of each modality at test time and measure F1.

    python scripts/robustness.py --variant full
    python scripts/robustness.py --variant full --split val --repeats 5 --levels 0,0.1,0.2,0.3,0.4,0.5

Patterns: rgb, ir, gas (one modality hidden) and joint (each hidden independently).
Writes <out>/results/<variant>_robustness_<split>.csv (every run) and ..._summary.csv (mean over
seeds and repeats).
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trimodal.cli import add_path_args, resolve_paths  # noqa: E402
from trimodal.config import VARIANTS, pick_device  # noqa: E402


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default="full", choices=sorted(VARIANTS))
    p.add_argument("--runs", help="folder holding seed_<k>/model_state.pt (default <out>/runs/<variant>)")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--levels", default="0,0.1,0.2,0.3,0.4,0.5")
    p.add_argument("--patterns", default="rgb,ir,gas,joint")
    add_path_args(p)
    args = p.parse_args()

    paths = resolve_paths(args)
    device = pick_device(args.device)
    run_root = Path(args.runs) if args.runs else paths.out / "runs" / args.variant

    from trimodal.data import make_loaders
    from trimodal.evaluation import find_checkpoints
    from trimodal.robustness import summarise, sweep

    ckpts = find_checkpoints(run_root)
    if not ckpts:
        sys.exit(f"error: no seed_<k>/model_state.pt found under {run_root}")
    loaders = make_loaders(paths, batch_size=args.batch_size)
    rows = sweep(ckpts, args.variant, loaders, device, split=args.split,
                 patterns=args.patterns.split(","), levels=[float(x) for x in args.levels.split(",")],
                 repeats=args.repeats)
    summary = summarise(rows)
    print(f"\n{'pattern':<7}" + "".join(f"{lv:>8}" for lv in sorted({r['level'] for r in summary})))
    for pat in args.patterns.split(","):
        print(f"{pat:<7}" + "".join(f"{r['f1_mean']:>8.3f}" for r in summary if r["pattern"] == pat))
    stem = paths.out / "results" / f"{args.variant}_robustness_{args.split}"
    write_csv(rows, stem.with_suffix(".csv"))
    write_csv(summary, Path(f"{stem}_summary.csv"))


if __name__ == "__main__":
    main()
