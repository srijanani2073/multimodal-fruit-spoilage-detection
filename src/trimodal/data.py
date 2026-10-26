"""TR-6 data loading: manifest -> day-level trajectories -> padded batches.

Ported from notebook 04 (``TrimodalDataset`` + ``DayLevelSequenceDataset``).
For every (produce, label, day) the first available sRGB image and the first
available thermal image of the day are used, and the gas vector is the mean of
the day's available session readings. Days of one (produce, label) form a
trajectory whose label and regression target are those of its last day.
"""
from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from .config import FRUITS, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, LABELS, SESSIONS, Paths

IR_PATTERN = re.compile(r"(\d{8})_(\d{6})_([\d.]+)C_([\d.]+)C\.jpg$", re.IGNORECASE)
SRGB_PATTERN = re.compile(r"^(\d{8})_(\d{6})[^/]*\.jpg$", re.IGNORECASE)
SESSION_HOURS = {"morning": (5, 11), "afternoon": (12, 16), "evening": (16, 19)}
LABEL_FOLDERS = {"not_spoiled": "Not_spoiled", "spoiled": "Spoiled"}
IR_FOLDER_NAMES = ["IR_fusion_images", "IR_Fusion_images", "ir_fusion_images"]

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _session_of(hhmmss: str) -> Optional[str]:
    hour = int(hhmmss[:2])
    for name, (lo, hi) in SESSION_HOURS.items():
        if lo <= hour < hi:
            return name
    return None


def _ir_folder(base: Path) -> Path:
    for name in IR_FOLDER_NAMES:
        if (base / name).exists():
            return base / name
    return base / IR_FOLDER_NAMES[0]


def _group_by_session(folder: Path, pattern: re.Pattern) -> dict:
    """{date: {session: [sorted image paths]}}"""
    out: dict = defaultdict(lambda: defaultdict(list))
    if not folder.exists():
        return {}
    for f in sorted(folder.iterdir()):
        m = pattern.search(f.name)
        if not m:
            continue
        session = _session_of(m.group(2))
        if session:
            out[m.group(1)][session].append(f)
    return out


def build_image_index(data_root: Path) -> dict:
    """{(fruit, label, date, session): {'rgb': [...], 'ir': [...]}} for the whole dataset."""
    index: dict = {}
    for fruit in FRUITS:
        for label, folder in LABEL_FOLDERS.items():
            base = data_root / "Classified" / fruit / folder
            for d, sessions in _group_by_session(base / "sRGB_images", SRGB_PATTERN).items():
                for s, paths in sessions.items():
                    index.setdefault((fruit, label, d, s), {"rgb": [], "ir": []})["rgb"] = paths
            for d, sessions in _group_by_session(_ir_folder(base), IR_PATTERN).items():
                for s, paths in sessions.items():
                    index.setdefault((fruit, label, d, s), {"rgb": [], "ir": []})["ir"] = paths
    # In TR-6 the spoiled banana's thermal images are stored under Normal/Banana.
    spoiled_banana_dates = {d for (f, lb, d, _) in index if f == "Banana" and lb == "spoiled"}
    for d, sessions in _group_by_session(_ir_folder(data_root / "Normal" / "Banana"), IR_PATTERN).items():
        if d in spoiled_banana_dates:
            for s, paths in sessions.items():
                entry = index.setdefault(("Banana", "spoiled", d, s), {"rgb": [], "ir": []})
                if not entry["ir"]:
                    entry["ir"] = paths
    return index


def read_manifest(manifest: Path, split: Optional[str] = None) -> list[dict]:
    with open(manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("exclude", "").strip().lower() != "true"]
    if split:
        rows = [r for r in rows if r.get("split", "").strip().lower() == split.lower()]
    return rows


@lru_cache(maxsize=4096)
def load_image(path: str) -> torch.Tensor:
    return IMAGE_TRANSFORM(Image.open(path).convert("RGB"))


class TrajectoryDataset(Dataset):
    """Day-level trajectories for one split ('train', 'val', 'test') or a list of splits."""

    def __init__(self, paths: Optional[Paths], split: Optional[str], image_index: Optional[dict] = None,
                 trajectories: Optional[list] = None):
        if trajectories is not None:
            self.split, self.trajectories = split, list(trajectories)
            return
        self.split = split
        index = image_index if image_index is not None else build_image_index(paths.data_root)
        self.trajectories = self._build(read_manifest(paths.manifest, split), index)

    @classmethod
    def concat(cls, *datasets: "TrajectoryDataset") -> "TrajectoryDataset":
        """Trajectory lists side by side (a produce's train and val days stay separate trajectories)."""
        return cls(None, "+".join(d.split for d in datasets), trajectories=[t for d in datasets for t in d.trajectories])

    @staticmethod
    def _build(rows: list[dict], index: dict) -> list[dict]:
        day_groups: dict = defaultdict(list)
        for row in rows:
            fruit, label = row["fruit"], row["label"]
            for si, session in enumerate(SESSIONS):
                srgb = _float(row.get(f"srgb_{session}_count", 0)) > 0
                ir = _float(row.get(f"ir_{session}_count", 0)) > 0
                gas = row.get(f"methane_{session}_present", "").strip().upper() == "TRUE"
                if not (srgb or ir or gas):
                    continue
                imgs = index.get((fruit, label, row["actual_date"], session), {"rgb": [], "ir": []})
                if gas:
                    m, sd = _float(row.get(f"methane_{session}_ppm")), _float(row.get(f"methane_{session}_std"))
                    left, right = _float(row.get(f"methane_{session}_left")), _float(row.get(f"methane_{session}_right"))
                    gas_vec = [m, sd, left, right, abs(left - right), si / 2.0]
                else:
                    gas_vec = None
                day_groups[(fruit, LABELS[label], int(row["corrected_day_index"]))].append({
                    "si": si,
                    "rgb": imgs["rgb"][0] if srgb and imgs["rgb"] else None,
                    "ir": imgs["ir"][0] if ir and imgs["ir"] else None,
                    "gas": gas_vec,
                    "days_until": _float(row.get("days_until_spoilage", ""), -1.0),
                    "fruit_idx": FRUITS.index(fruit),
                })

        per_traj: dict = defaultdict(list)
        for (fruit, label, day_idx), sessions in day_groups.items():
            sessions.sort(key=lambda s: s["si"])
            rgb = next((s["rgb"] for s in sessions if s["rgb"] is not None), None)
            ir = next((s["ir"] for s in sessions if s["ir"] is not None), None)
            gas_vals = [s["gas"] for s in sessions if s["gas"] is not None]
            gas = torch.tensor(gas_vals, dtype=torch.float32).mean(dim=0) if gas_vals else torch.zeros(6)
            per_traj[(fruit, label)].append({
                "day_idx": day_idx, "rgb": str(rgb) if rgb else None, "ir": str(ir) if ir else None,
                "gas": gas, "gas_avail": bool(gas_vals), "label": label,
                "days_until": sessions[0]["days_until"], "fruit_idx": sessions[0]["fruit_idx"],
            })

        trajectories = []
        for (fruit, label), days in sorted(per_traj.items()):
            days.sort(key=lambda d: d["day_idx"])
            last_obs = None
            for d in days:
                d["delta"] = 0.0 if last_obs is None else float(d["day_idx"] - last_obs)
                if d["gas_avail"]:
                    last_obs = d["day_idx"]
            trajectories.append({"fruit": fruit, "label": days[-1]["label"], "days": days})
        return trajectories

    def __len__(self) -> int:
        return len(self.trajectories)

    def n_days(self) -> int:
        return sum(len(t["days"]) for t in self.trajectories)

    def __getitem__(self, idx: int) -> dict:
        days = self.trajectories[idx]["days"]
        blank = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        return {
            "rgb_seq": torch.stack([load_image(d["rgb"]) if d["rgb"] else blank for d in days]),
            "ir_seq": torch.stack([load_image(d["ir"]) if d["ir"] else blank for d in days]),
            "rgb_avail": torch.tensor([d["rgb"] is not None for d in days], dtype=torch.bool),
            "ir_avail": torch.tensor([d["ir"] is not None for d in days], dtype=torch.bool),
            "gas_seq": torch.stack([d["gas"] for d in days]),
            "gas_mask": torch.tensor([d["gas_avail"] for d in days], dtype=torch.float32),
            "gas_delta": torch.tensor([d["delta"] for d in days], dtype=torch.float32),
            "T": len(days),
            "fruit_idx": days[0]["fruit_idx"],
            "label": days[-1]["label"],
            "days_until": days[-1]["days_until"],
        }


class AugmentedDataset(Dataset):
    """Original trajectories plus ``n_copies`` horizontally flipped and rotated (+-15 deg) copies.

    The same deterministic augmentation is applied to every day of a copy.
    """

    def __init__(self, base: TrajectoryDataset, n_copies: int = 1):
        self.base = base
        self.index = [(i, c) for i in range(len(base)) for c in range(n_copies + 1)]
        self.trajectories = [base.trajectories[i] for i, _ in self.index]

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _augment(img: torch.Tensor, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        angle = torch.rand(1, generator=g).item() * 30.0 - 15.0
        return TF.rotate(torch.flip(img, dims=[-1]), angle)

    def __getitem__(self, idx: int) -> dict:
        traj_idx, copy_id = self.index[idx]
        item = self.base[traj_idx]
        if copy_id == 0:
            return item
        seed = traj_idx * 10_000 + copy_id * 100
        item = dict(item)
        item["rgb_seq"] = torch.stack([self._augment(item["rgb_seq"][t], seed + t) for t in range(item["T"])])
        item["ir_seq"] = torch.stack([self._augment(item["ir_seq"][t], seed + t) for t in range(item["T"])])
        return item


def collate(batch: list[dict]) -> dict:
    """Pad a list of trajectories to the longest one; ``pad_mask`` is True on padding."""
    T, B = max(b["T"] for b in batch), len(batch)
    out = {
        "rgb_seq": torch.zeros(B, T, 3, IMAGE_SIZE, IMAGE_SIZE),
        "ir_seq": torch.zeros(B, T, 3, IMAGE_SIZE, IMAGE_SIZE),
        "rgb_avail": torch.zeros(B, T, dtype=torch.bool),
        "ir_avail": torch.zeros(B, T, dtype=torch.bool),
        "gas_seq": torch.zeros(B, T, 6),
        "gas_mask": torch.zeros(B, T),
        "gas_delta": torch.zeros(B, T),
        "pad_mask": torch.ones(B, T, dtype=torch.bool),
        "fruit_idx": torch.zeros(B, dtype=torch.long),
        "label": torch.zeros(B, dtype=torch.long),
        "days_until": torch.zeros(B, dtype=torch.float32),
    }
    for i, b in enumerate(batch):
        t = b["T"]
        for k in ("rgb_seq", "ir_seq", "rgb_avail", "ir_avail", "gas_seq", "gas_mask", "gas_delta"):
            out[k][i, :t] = b[k]
        out["pad_mask"][i, :t] = False
        out["fruit_idx"][i] = b["fruit_idx"]
        out["label"][i] = b["label"]
        out["days_until"][i] = b["days_until"]
    return out


def balanced_sampler(dataset) -> WeightedRandomSampler:
    labels = [int(t["label"]) for t in dataset.trajectories]
    n = len(labels)
    n_sp, n_fr = max(sum(labels), 1), max(n - sum(labels), 1)
    weights = [n / (2 * n_sp) if l == 1 else n / (2 * n_fr) for l in labels]
    return WeightedRandomSampler(weights=weights, num_samples=n, replacement=True)


def make_loaders(paths: Paths, batch_size: int = 4, augment_copies: int = 1, num_workers: int = 0) -> dict:
    """All loaders used by training and evaluation.

    ``train`` is augmented and class-balanced; ``train_plain`` is used to calibrate
    per-epoch thresholds; ``calib`` (train + val) is the pool for the final
    k-fold median threshold. ``val`` and ``test`` are never augmented.
    """
    index = build_image_index(paths.data_root)
    ds = {s: TrajectoryDataset(paths, s, index) for s in ("train", "val", "test")}
    ds["calib"] = TrajectoryDataset.concat(ds["train"], ds["val"])
    aug = AugmentedDataset(ds["train"], augment_copies)

    def plain(d):
        return DataLoader(d, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers)

    return {
        "datasets": ds,
        "train": DataLoader(aug, batch_size=batch_size, sampler=balanced_sampler(aug), collate_fn=collate,
                            num_workers=num_workers),
        "train_plain": plain(ds["train"]),
        "val": plain(ds["val"]),
        "test": plain(ds["test"]),
        "calib": plain(ds["calib"]),
    }


def seed_everything(seed: int) -> None:
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
