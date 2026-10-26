"""Missing-modality robustness sweep (notebook 08).

At test time a random fraction of the observed days has one modality (or, for ``joint``, each modality
independently) marked as missing, by flipping its availability mask. Each trained seed keeps the
threshold calibrated on its own training set. Unlike the notebook, the random generator for every
(variant, seed, pattern, level, repeat) cell is seeded with a stable CRC32 hash, so reruns give the
same numbers.
"""
from __future__ import annotations

import zlib

import numpy as np
import torch

from .model import load_checkpoint
from .training import calibrate_threshold, classification_metrics, predict

PATTERNS = ["rgb", "ir", "gas", "joint"]
LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
MASK_KEYS = {"rgb": ["rgb_avail"], "ir": ["ir_avail"], "gas": ["gas_mask"],
             "joint": ["rgb_avail", "ir_avail", "gas_mask"]}


def corruption_generator(*parts) -> torch.Generator:
    return torch.Generator().manual_seed(zlib.crc32("|".join(map(str, parts)).encode()) % (2 ** 31))


def corrupt_batch(batch: dict, pattern: str, rate: float, generator: torch.Generator) -> dict:
    """Mark each observed (non-padded) day's modality as missing with probability ``rate``."""
    if rate <= 0:
        return batch
    out = dict(batch)
    valid = ~batch["pad_mask"]
    for key in MASK_KEYS[pattern]:
        drop = (torch.rand(batch[key].shape, generator=generator) < rate) & valid
        mask = batch[key].clone()
        mask[drop] = 0 if mask.dtype != torch.bool else False
        out[key] = mask
    return out


def sweep(checkpoints: dict, variant: str, loaders: dict, device: str, split: str = "test",
          patterns=PATTERNS, levels=LEVELS, repeats: int = 5, log=print) -> list[dict]:
    rows = []
    for seed, path in sorted(checkpoints.items()):
        model = load_checkpoint(path, variant, device)
        tr = predict(model, loaders["train_plain"])
        thr = calibrate_threshold(tr["probs"].tolist(), tr["labels"])
        for pattern in patterns:
            for level in levels:
                for rep in range(repeats):
                    g = corruption_generator(variant, seed, pattern, level, rep)
                    pr = predict(model, loaders[split], lambda b: corrupt_batch(b, pattern, level, g))
                    m = classification_metrics(pr["labels"], pr["probs"], thr)
                    rows.append({"variant": variant, "seed": seed, "pattern": pattern, "level": level,
                                 "repeat": rep, "threshold": thr, "f1": round(m["f1"], 4),
                                 "auc": round(m["auc"], 4)})
        log(f"  [{variant}] seed {seed} done")
        del model
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    """Mean and std of F1 / AUC over seeds and repeats for every (variant, pattern, level)."""
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["variant"], r["pattern"], r["level"]), []).append(r)
    out = []
    for (v, p, lv), rs in sorted(groups.items()):
        f1s, aucs = np.array([r["f1"] for r in rs]), np.array([r["auc"] for r in rs])
        out.append({"variant": v, "pattern": p, "level": lv, "n": len(rs),
                    "f1_mean": round(float(f1s.mean()), 4), "f1_std": round(float(f1s.std()), 4),
                    "auc_mean": round(float(np.nanmean(aucs)), 4)})
    return out
