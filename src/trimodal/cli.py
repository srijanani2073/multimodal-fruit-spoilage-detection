"""Shared command-line options for the scripts in ``scripts/``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Paths


def add_path_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("paths (default: environment variables, then repository defaults)")
    g.add_argument("--data-root", help="TR-6 dataset folder containing Classified/ and Normal/ [TR6_ROOT]")
    g.add_argument("--manifest", help="master manifest CSV [TR6_MANIFEST]")
    g.add_argument("--out", help="output folder for checkpoints and results [TRIMODAL_OUT, default outputs/]")
    p.add_argument("--device", default="auto", help="cpu, cuda, mps or auto (default)")
    p.add_argument("--batch-size", type=int, default=4)


def resolve_paths(args, need_images: bool = True) -> Paths:
    paths = Paths.resolve(args.data_root, args.manifest, args.out)
    problems = paths.check(need_images)
    if problems:
        for msg in problems:
            print(f"error: {msg}", file=sys.stderr)
        print("See INSTALL.md, section 'Get the dataset'.", file=sys.stderr)
        sys.exit(2)
    return paths


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"wrote {path}")
