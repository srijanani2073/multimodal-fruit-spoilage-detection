#!/usr/bin/env python3
"""Train one model variant for one or more seeds.

    python scripts/train.py --variant full --seeds 0-9            # the headline model (notebook 04)
    python scripts/train.py --variant no_cbam --seeds 0-9 --ablation-recipe   # an ablation (notebook 05)
    python scripts/train.py --variant full --seeds 0 --epochs 2   # quick trial run

Checkpoints go to <out>/runs/<variant>/seed_<k>/ (model_state.pt, history.json, summary.json).
Variants: full, no_cbam, no_cross_attn, lstm, mean_impute, early_fusion, late_fusion.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trimodal.cli import add_path_args, resolve_paths, write_json  # noqa: E402
from trimodal.config import VARIANTS, TrainConfig, parse_seeds, pick_device  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default="full", choices=sorted(VARIANTS))
    p.add_argument("--seeds", default="0-9", help="e.g. 0-9, 0,3,5 or 4 (default 0-9)")
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--ablation-recipe", action="store_true",
                   help="as in the ablation notebook: both backbones at the lower rate, no stochastic depth")
    p.add_argument("--no-pretrained", action="store_true", help="random backbone weights (testing only)")
    p.add_argument("--skip-existing", action="store_true", help="skip seeds that already have a checkpoint")
    add_path_args(p)
    args = p.parse_args()

    paths = resolve_paths(args)
    device = pick_device(args.device)
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, pretrained=not args.no_pretrained,
                      slow_backbones="both" if args.ablation_recipe else "ir",
                      stochastic_depth=not args.ablation_recipe)

    from trimodal.data import make_loaders
    from trimodal.training import train_one_seed

    loaders = make_loaders(paths, batch_size=cfg.batch_size, augment_copies=cfg.augment_copies)
    ds = loaders["datasets"]
    print(f"variant={args.variant} device={device} | train {len(ds['train'])} trajectories "
          f"(x{cfg.augment_copies + 1} with augmentation), val {len(ds['val'])}, test {len(ds['test'])} (unused here)")

    out_root = paths.out / "runs" / args.variant
    summaries = []
    for seed in parse_seeds(args.seeds):
        run_dir = out_root / f"seed_{seed}"
        if args.skip_existing and (run_dir / "model_state.pt").exists():
            print(f"  seed {seed}: checkpoint exists, skipping")
            continue
        summaries.append(train_one_seed(args.variant, seed, loaders, cfg, run_dir, device))
        print(f"  seed {seed}: best val F1 {summaries[-1]['best_val_f1']:.4f} "
              f"at epoch {summaries[-1]['best_epoch']} ({summaries[-1]['minutes']} min)")

    if summaries:
        import numpy as np
        f1s = [s["best_val_f1"] for s in summaries]
        print(f"\n{args.variant}: validation F1 {np.mean(f1s):.4f} +- {np.std(f1s):.4f} over {len(f1s)} seeds")
        write_json({"variant": args.variant, "seeds": summaries}, out_root / "training_summary.json")


if __name__ == "__main__":
    main()
