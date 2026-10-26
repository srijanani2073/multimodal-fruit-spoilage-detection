"""Uncertainty and significance on a small test set: bootstrap CIs, paired bootstrap, Benjamini-Hochberg."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def _f1(y, p):
    return f1_score(y, p, average="binary", zero_division=0)


def bootstrap_f1_ci(labels, preds, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Percentile bootstrap CI for F1, resampling test trajectories with replacement."""
    y, p = np.asarray(labels), np.asarray(preds)
    rng = np.random.default_rng(seed)
    stats = [_f1(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"f1": float(_f1(y, p)), "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot}


def paired_bootstrap(labels, preds_a, preds_b, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Difference F1(A) - F1(B) on the same trajectories, with a two-sided bootstrap p-value.

    The p-value is the share of resamples whose difference has the opposite sign to the observed one,
    doubled and capped at 1.
    """
    y, a, b = map(np.asarray, (labels, preds_a, preds_b))
    rng = np.random.default_rng(seed)
    obs = _f1(y, a) - _f1(y, b)
    diffs = np.array([_f1(y[i], a[i]) - _f1(y[i], b[i])
                      for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))])
    if obs == 0:
        p = 1.0
    else:
        p = min(1.0, 2 * float(np.mean(diffs <= 0) if obs > 0 else np.mean(diffs >= 0)))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"delta_f1": float(obs), "ci_low": float(lo), "ci_high": float(hi), "p_value": p}


def benjamini_hochberg(p_values, q: float = 0.05) -> list[dict]:
    """Benjamini-Hochberg adjusted p-values and which tests survive at false-discovery rate ``q``."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, p[i] * m / rank)
        adj[i] = running
    return [{"p": float(p[i]), "p_adj": float(adj[i]), "significant": bool(adj[i] <= q)} for i in range(m)]
