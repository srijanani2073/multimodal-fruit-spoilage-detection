"""Seed-ensemble test evaluation with a k-fold median decision threshold (notebook 04, section 5)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from .model import load_checkpoint
from .training import calibrate_threshold, classification_metrics, predict, regression_metrics


def find_checkpoints(run_root: Path) -> dict[int, Path]:
    """{seed: path} for every ``seed_<k>/model_state.pt`` below ``run_root``.

    Works for the scripts' layout (``<variant>/seed_3/``) and the notebooks' layouts
    (``model_a_winner/seed_3/`` and ``<config>_seed3/seed_3/``).
    """
    run_root = Path(run_root)
    if run_root.is_dir():
        candidates = run_root.glob("**/seed_*/model_state.pt")
    else:  # notebook 05 layout: runs/05_ablation_study/<config>_seed<k>/seed_<k>/
        candidates = run_root.parent.glob(f"{run_root.name}_seed*/seed_*/model_state.pt")
    found = {}
    for p in sorted(candidates):
        try:
            found[int(p.parent.name.split("_")[-1])] = p
        except ValueError:
            continue
    return found


def ensemble_predictions(checkpoints: dict[int, Path], variant: str, loaders: dict, device: str,
                         splits=("calib", "test"), log=print) -> dict:
    """Per-seed predictions for each split, plus their unweighted average."""
    per_seed = {s: [] for s in splits}
    for seed, path in sorted(checkpoints.items()):
        log(f"  loading seed {seed}: {path}")
        model = load_checkpoint(path, variant, device)
        for s in splits:
            per_seed[s].append(predict(model, loaders[s]))
        del model
    out = {"seeds": sorted(checkpoints)}
    for s in splits:
        runs = per_seed[s]
        out[s] = {
            "probs": np.mean([r["probs"] for r in runs], axis=0),
            "days_pred": np.mean([r["days_pred"] for r in runs], axis=0),
            "labels": runs[0]["labels"], "days_true": runs[0]["days_true"],
            "per_seed_probs": [r["probs"] for r in runs],
        }
    return out


def kfold_median_threshold(scores, labels, max_folds: int = 6, random_state: int = 0) -> dict:
    """Calibrate a threshold on each fold's training part and take the median.

    K = max(2, min(max_folds, smaller class count)). The median of the fold thresholds is less sensitive
    to a few borderline trajectories than a single threshold fitted to the whole pool.
    """
    scores, labels = np.asarray(scores), np.asarray(labels)
    K = max(2, min(max_folds, int(min((labels == 0).sum(), (labels == 1).sum()))))
    folds = []
    for tr, ho in StratifiedKFold(n_splits=K, shuffle=True, random_state=random_state).split(scores, labels):
        thr = calibrate_threshold(scores[tr].tolist(), labels[tr].tolist())
        folds.append({"threshold": thr, "heldout_f1": float(f1_score(labels[ho], (scores[ho] >= thr).astype(int),
                                                                        zero_division=0))})
    ts = [f["threshold"] for f in folds]
    return {"K": K, "folds": folds, "threshold": float(np.median(ts)), "spread": float(max(ts) - min(ts)),
            "single_threshold": calibrate_threshold(scores.tolist(), labels.tolist())}


def evaluate_ensemble(checkpoints: dict[int, Path], variant: str, loaders: dict, device: str, log=print) -> dict:
    """Full test evaluation; returns a JSON-serialisable dict."""
    ens = ensemble_predictions(checkpoints, variant, loaders, device, log=log)
    cal, test = ens["calib"], ens["test"]
    thr = kfold_median_threshold(cal["probs"], cal["labels"])
    cls = classification_metrics(test["labels"], test["probs"], thr["threshold"])
    reg = regression_metrics(test["days_true"], test["days_pred"])
    trivial = classification_metrics(test["labels"], np.ones(len(test["labels"])), 0.5)["f1"]
    test_ds = loaders["datasets"]["test"]
    return {
        "variant": variant, "seeds": ens["seeds"], "n_seeds": len(ens["seeds"]),
        "threshold": thr,
        "test": {**{k: round(v, 4) for k, v in {**cls, **reg}.items()}, "trivial_all_spoiled_f1": round(trivial, 4)},
        "per_trajectory": [
            {"fruit": t["fruit"], "label": int(y), "p_spoiled": round(float(p), 4),
             "pred": int(p >= thr["threshold"]), "days_true": float(dt), "days_pred": round(float(dp), 3)}
            for t, y, p, dt, dp in zip(test_ds.trajectories, test["labels"], test["probs"],
                                       test["days_true"], test["days_pred"])
        ],
        "per_seed_test_probs": [[round(float(x), 4) for x in p] for p in test["per_seed_probs"]],
    }
