"""Training loop (one seed of one variant), matching notebooks 04 and 05."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .config import THRESHOLD_GRID, TrainConfig
from .data import seed_everything
from .model import TrimodalModel, UncertaintyWeightedLoss


# ----------------------------------------------------------------------------- metrics
def calibrate_threshold(scores, labels, grid=THRESHOLD_GRID) -> float:
    """Threshold on P(spoiled) that maximises F1 (first best on a 0.10-0.90 grid; 0.5 if one class)."""
    if len(set(labels)) < 2:
        return 0.5
    best_t, best_f1 = 0.5, 0.0
    for t in grid:
        f = f1_score(labels, [int(s >= t) for s in scores], average="binary", zero_division=0)
        if f > best_f1:
            best_t, best_f1 = t, f
    return best_t


def safe_auc(labels, scores) -> float:
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")


def classification_metrics(labels, scores, threshold) -> dict:
    preds = (np.asarray(scores) >= threshold).astype(int)
    return {
        "f1": float(f1_score(labels, preds, average="binary", zero_division=0)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "auc": safe_auc(labels, scores),
    }


def regression_metrics(days_true, days_pred) -> dict:
    out = {"mae": float(mean_absolute_error(days_true, days_pred)),
           "rmse": float(mean_squared_error(days_true, days_pred) ** 0.5)}
    try:
        out["r2"] = float(r2_score(days_true, days_pred))
    except ValueError:
        out["r2"] = float("nan")
    return out


# ----------------------------------------------------------------------------- loss and inference
def multitask_loss(out: dict, batch: dict, criterion: UncertaintyWeightedLoss, class_weights, device):
    labels = batch["label"].to(device)
    days = batch["days_until"].to(device)
    cls_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))(out["cls_logits"], labels)
    reg_loss = nn.SmoothL1Loss()(out["reg_output"].squeeze(1), days)

    rows = torch.arange(labels.shape[0], device=device)
    last = out["last_idx"]
    gas_last_avail = batch["gas_mask"].to(device)[rows, last].bool()
    gas_last = batch["gas_seq"].to(device)[rows, last]
    recon_loss = (nn.MSELoss()(out["gas_recon"][gas_last_avail], gas_last[gas_last_avail])
                  if gas_last_avail.any() else torch.zeros((), device=device))
    total = criterion([cls_loss, reg_loss, recon_loss])
    return total, {"cls_loss": cls_loss.item(), "reg_loss": reg_loss.item(), "recon_loss": recon_loss.item()}


@torch.no_grad()
def predict(model: TrimodalModel, loader, batch_transform: Optional[Callable] = None) -> dict:
    """P(spoiled), labels and days-until predictions for every trajectory in ``loader`` (in order)."""
    model.eval()
    probs, labels, days_true, days_pred = [], [], [], []
    for batch in loader:
        if batch_transform is not None:
            batch = batch_transform(batch)
        out = model(batch)
        probs.extend(torch.softmax(out["cls_logits"], dim=1)[:, 1].cpu().tolist())
        labels.extend(batch["label"].tolist())
        days_true.extend(batch["days_until"].tolist())
        days_pred.extend(out["reg_output"].squeeze(1).cpu().tolist())
    return {"probs": np.array(probs), "labels": labels, "days_true": days_true, "days_pred": np.array(days_pred)}


# ----------------------------------------------------------------------------- one run
def build_optimizer(model: TrimodalModel, criterion: UncertaintyWeightedLoss, cfg: TrainConfig):
    slow = list(model.ir_encoder.backbone.parameters())
    if cfg.slow_backbones == "both":
        slow += list(model.rgb_encoder.backbone.parameters())
    slow_ids = {id(p) for p in slow}
    rest = [p for p in model.parameters() if id(p) not in slow_ids]
    return AdamW([
        {"params": rest, "lr": cfg.lr},
        {"params": slow, "lr": cfg.lr_thermal_backbone},
        {"params": list(criterion.parameters()), "lr": cfg.lr},
    ], weight_decay=cfg.weight_decay)


def train_one_seed(variant: str, seed: int, loaders: dict, cfg: TrainConfig, run_dir: Path, device: str,
                   log: Callable[[str], None] = print) -> dict:
    """Train one seed; keep the epoch with the best validation F1.

    Each epoch the decision threshold is calibrated on the (un-augmented) training set and then applied
    to the validation set, so validation data never chooses its own threshold.
    Writes ``model_state.pt`` and ``history.json`` into ``run_dir``.
    """
    seed_everything(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    model = TrimodalModel(variant, d_model=cfg.d_model, dropout=cfg.dropout, pretrained=cfg.pretrained,
                          stochastic_depth=cfg.stochastic_depth).to(device)
    criterion = UncertaintyWeightedLoss(3).to(device)
    optimizer = build_optimizer(model, criterion, cfg)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    clip_params = list(model.parameters()) + list(criterion.parameters())
    e1, e2, e3 = cfg.unfreeze_epochs

    history, best_f1, best_state, best_epoch = [], -1.0, None, 0
    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        if epoch == e1:
            model.unfreeze_visual_last_n(1)
        elif epoch == e2:
            model.unfreeze_visual_last_n(2)
        elif epoch == e3:
            model.unfreeze_visual_full()

        model.train(); criterion.train()
        total, n_batches = 0.0, 0
        parts_sum = {"cls_loss": 0.0, "reg_loss": 0.0, "recon_loss": 0.0}
        for batch in loaders["train"]:
            out = model(batch)
            loss, parts = multitask_loss(out, batch, criterion, cfg.class_weights, device)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(clip_params, max_norm=cfg.grad_clip)
            optimizer.step()
            total += loss.item(); n_batches += 1
            for k in parts_sum:
                parts_sum[k] += parts[k]
        scheduler.step()

        train_pred = predict(model, loaders["train_plain"])
        thr = calibrate_threshold(train_pred["probs"].tolist(), train_pred["labels"])
        val_pred = predict(model, loaders["val"])
        val_cls = classification_metrics(val_pred["labels"], val_pred["probs"], thr)
        val_reg = regression_metrics(val_pred["days_true"], val_pred["days_pred"])

        is_best = val_cls["f1"] > best_f1
        if is_best:
            best_f1, best_epoch = val_cls["f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        sig = criterion.sigmas()
        nb = max(n_batches, 1)
        history.append({
            "epoch": epoch, "train_loss": round(total / nb, 4),
            **{k: round(v / nb, 4) for k, v in parts_sum.items()},
            "val_f1": round(val_cls["f1"], 4), "val_auc": round(val_cls["auc"], 4),
            "val_mae": round(val_reg["mae"], 4), "val_threshold": thr,
            "sigma_cls": sig[0], "sigma_reg": sig[1], "sigma_recon": sig[2],
            "lr": optimizer.param_groups[0]["lr"], "epoch_time_s": round(time.time() - t0, 1),
        })
        log(f"  [{variant} seed {seed}] epoch {epoch:2d}/{cfg.epochs}  loss={total / nb:.4f}  "
            f"val F1={val_cls['f1']:.3f} AUC={val_cls['auc']:.3f} MAE={val_reg['mae']:.2f}  thr={thr:.2f}"
            f"{'  *' if is_best else ''}")

    torch.save(best_state, run_dir / "model_state.pt")
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    summary = {"variant": variant, "seed": seed, "best_epoch": best_epoch, "best_val_f1": round(best_f1, 4),
               "minutes": round((time.time() - start) / 60, 1), "config": cfg.to_dict()}
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary
