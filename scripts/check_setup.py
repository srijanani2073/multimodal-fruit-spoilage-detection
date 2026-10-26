#!/usr/bin/env python3
"""Check that the environment, dataset and manifest are in place before training.

    python scripts/check_setup.py                 # check your TR-6 copy
    python scripts/check_setup.py --synthetic     # no dataset needed: build a tiny fake TR-6 and run on it

Prints the split sizes (expected 12 / 12 / 11 trajectories) and runs one forward and backward pass.
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    from trimodal.cli import add_path_args, resolve_paths

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_args(p)
    p.add_argument("--synthetic", action="store_true", help="generate a small fake dataset instead of using TR-6")
    p.add_argument("--no-pretrained", action="store_true", help="skip downloading ImageNet weights")
    args = p.parse_args()

    print("1. Python packages")
    import numpy, sklearn, torch, torchvision  # noqa: E401
    from trimodal.config import Paths, pick_device
    print(f"   python {sys.version.split()[0]}, torch {torch.__version__}, torchvision {torchvision.__version__}, "
          f"numpy {numpy.__version__}, scikit-learn {sklearn.__version__}")
    device = pick_device(args.device)
    print(f"   device: {device}")

    print("2. Data")
    if args.synthetic:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        from synthetic_tr6 import make_synthetic_tr6
        tmp = Path(tempfile.mkdtemp(prefix="tr6_synthetic_"))
        root, manifest = make_synthetic_tr6(tmp)
        paths = Paths(root, manifest, tmp / "outputs")
        print(f"   synthetic TR-6 written to {tmp}")
    else:
        paths = resolve_paths(args)
        print(f"   data root: {paths.data_root}\n   manifest:  {paths.manifest}")

    from trimodal.data import make_loaders
    loaders = make_loaders(paths, batch_size=args.batch_size)
    expected = {"train": (12, 73), "val": (12, 21), "test": (11, 18)}
    ok = True
    for split in ("train", "val", "test"):
        ds = loaders["datasets"][split]
        n_sp = sum(t["label"] for t in ds.trajectories)
        n_rgb = sum(d["rgb"] is not None for t in ds.trajectories for d in t["days"])
        n_ir = sum(d["ir"] is not None for t in ds.trajectories for d in t["days"])
        n_gas = sum(d["gas_avail"] for t in ds.trajectories for d in t["days"])
        note = ""
        if not args.synthetic:
            match = (len(ds), ds.n_days()) == expected[split]
            ok &= match
            note = "  ok" if match else f"  (expected {expected[split][0]} trajectories / {expected[split][1]} days)"
        print(f"   {split:<5} {len(ds):>3} trajectories ({n_sp} spoiled), {ds.n_days():>3} days; "
              f"days with sRGB {n_rgb}, thermal {n_ir}, gas {n_gas}{note}")
    if not args.synthetic and n_rgb == 0:
        print("   warning: no images matched in the test split - check --data-root")
        ok = False

    print("3. Model")
    from trimodal.config import TrainConfig
    from trimodal.model import TrimodalModel, UncertaintyWeightedLoss, count_parameters
    from trimodal.training import multitask_loss
    model = TrimodalModel("full", pretrained=not args.no_pretrained).to(device)
    print(f"   full model: {count_parameters(model) / 1e6:.2f} M parameters")
    crit = UncertaintyWeightedLoss().to(device)
    batch = next(iter(loaders["val"]))
    t0 = time.time()
    out = model(batch)
    loss, _ = multitask_loss(out, batch, crit, TrainConfig().class_weights, device)
    loss.backward()
    print(f"   forward + backward on one batch of {batch['label'].shape[0]}: loss {loss.item():.3f}, "
          f"{time.time() - t0:.1f} s")

    print("\nAll checks passed." if ok else "\nSome checks need attention (see above).")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
