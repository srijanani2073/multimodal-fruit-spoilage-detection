"""Paths, hyperparameters and model variants in one place.

Paths resolve in this order: an explicit argument, then an environment
variable, then a default inside the repository:

    TR6_ROOT        folder containing the TR-6 dataset (``Classified/``, ``Normal/``)
    TR6_MANIFEST    master manifest CSV with the train / val / test split
    TRIMODAL_OUT    where trained checkpoints and results are written
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "TR-6"
DEFAULT_MANIFEST = REPO_ROOT / "Code" / "Manifest" / "manifests" / "master_manifest.csv"
DEFAULT_OUT = REPO_ROOT / "outputs"

FRUITS = ["Banana", "Carrot", "Guava", "Indian_Gooseberry", "Mango", "Tomato"]
SESSIONS = ["morning", "afternoon", "evening"]
LABELS = {"not_spoiled": 0, "spoiled": 1}
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
THRESHOLD_GRID = [t / 100 for t in range(10, 91)]


@dataclass
class Paths:
    data_root: Path
    manifest: Path
    out: Path

    @classmethod
    def resolve(cls, data_root: Optional[str] = None, manifest: Optional[str] = None,
                out: Optional[str] = None) -> "Paths":
        def pick(value, env, default):
            raw = value or os.environ.get(env)
            return Path(raw).expanduser().resolve() if raw else default
        return cls(
            data_root=pick(data_root, "TR6_ROOT", DEFAULT_DATA_ROOT),
            manifest=pick(manifest, "TR6_MANIFEST", DEFAULT_MANIFEST),
            out=pick(out, "TRIMODAL_OUT", DEFAULT_OUT),
        )

    def check(self, need_images: bool = True) -> list[str]:
        """Return a list of human-readable problems (empty when everything is in place)."""
        problems = []
        if not self.manifest.is_file():
            problems.append(f"Manifest not found: {self.manifest}")
        if need_images and not (self.data_root / "Classified").is_dir():
            problems.append(
                f"TR-6 images not found: expected {self.data_root / 'Classified'} "
                "(set --data-root or the TR6_ROOT environment variable)")
        return problems


@dataclass
class TrainConfig:
    """Training recipe used for every reported model (thesis Table 4.1)."""
    epochs: int = 30
    batch_size: int = 4
    lr: float = 1e-4
    lr_thermal_backbone: float = 1e-5
    weight_decay: float = 1e-4
    class_weights: tuple = (1.0, 2.3)
    d_model: int = 64
    dropout: float = 0.5
    grad_clip: float = 1.0
    unfreeze_epochs: tuple = (6, 8, 12)   # last block, last two blocks, full backbone
    augment_copies: int = 1               # offline flip + rotation copies per training trajectory
    pretrained: bool = True
    # The ablation sweep (notebook 05) differs from the headline model (notebook 04) in two details,
    # applied to every variant including its own full-model rerun; ``--ablation-recipe`` in
    # scripts/train.py switches both:
    slow_backbones: str = "ir"        # "ir": only the thermal backbone at the lower rate; "both": sRGB too
    stochastic_depth: bool = True     # False: backbones built with stochastic_depth_prob=0

    def to_dict(self) -> dict:
        return asdict(self)


# The full model and the six alternatives from the ablation study (notebook 05).
VARIANTS: dict[str, dict] = {
    "full": dict(fusion="intermediate", use_cbam=True, use_cross_attn=True, temporal="transformer", gas="grud"),
    "no_cbam": dict(fusion="intermediate", use_cbam=False, use_cross_attn=True, temporal="transformer", gas="grud"),
    "no_cross_attn": dict(fusion="intermediate", use_cbam=True, use_cross_attn=False, temporal="transformer", gas="grud"),
    "lstm": dict(fusion="intermediate", use_cbam=True, use_cross_attn=True, temporal="lstm", gas="grud"),
    "mean_impute": dict(fusion="intermediate", use_cbam=True, use_cross_attn=True, temporal="transformer", gas="mean"),
    "early_fusion": dict(fusion="early", use_cbam=False, use_cross_attn=False, temporal="lstm", gas="mean"),
    "late_fusion": dict(fusion="late", use_cbam=False, use_cross_attn=False, temporal="lstm", gas="mean"),
}

VARIANT_LABELS = {
    "full": "Full model",
    "no_cbam": "No CBAM",
    "no_cross_attn": "No cross-modal attention",
    "lstm": "LSTM instead of Transformer",
    "mean_impute": "Mean imputation instead of GRU-D",
    "early_fusion": "Early fusion baseline",
    "late_fusion": "Late fusion baseline",
}


# Where the notebooks saved each variant's checkpoints (relative to runs/), for evaluating existing runs.
NOTEBOOK_RUNS = {
    "full": "04_fusion_model/model_a_winner",
    "no_cbam": "05_ablation_study/ablation_1_no_cbam",
    "no_cross_attn": "05_ablation_study/ablation_2_no_cross_attn",
    "lstm": "05_ablation_study/ablation_3_lstm_not_transformer",
    "mean_impute": "05_ablation_study/ablation_4_mean_impute_not_grud",
    "early_fusion": "05_ablation_study/baseline_a_early",
    "late_fusion": "05_ablation_study/baseline_b_late",
}


def pick_device(requested: Optional[str] = None) -> str:
    import torch
    if requested and requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_seeds(text: str) -> list[int]:
    """'0-9' -> [0..9], '0,3,5' -> [0, 3, 5], '4' -> [4]."""
    seeds: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        elif part:
            seeds.append(int(part))
    return seeds
