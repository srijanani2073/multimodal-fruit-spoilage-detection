"""Smoke tests on a synthetic TR-6: data pipeline, every model variant, one training epoch, evaluation.

    pytest -q tests/
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_tr6 import make_synthetic_tr6  # noqa: E402
from trimodal.config import VARIANTS, Paths, TrainConfig  # noqa: E402
from trimodal.data import make_loaders  # noqa: E402
from trimodal.model import TrimodalModel, dummy_batch  # noqa: E402


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("tr6")
    data_root, manifest = make_synthetic_tr6(root)
    paths = Paths(data_root, manifest, root / "outputs")
    return paths, make_loaders(paths, batch_size=4)


def test_trajectories(synthetic):
    _, loaders = synthetic
    ds = loaders["datasets"]
    assert len(ds["train"]) == 12 and len(ds["test"]) == 12
    assert len(ds["calib"]) == 24
    traj = ds["train"].trajectories[0]
    days = [d["day_idx"] for d in traj["days"]]
    assert days == sorted(days)


def test_collate_shapes(synthetic):
    _, loaders = synthetic
    b = next(iter(loaders["train"]))
    B, T = b["pad_mask"].shape
    assert b["rgb_seq"].shape == (B, T, 3, 224, 224)
    assert b["gas_seq"].shape == (B, T, 6)
    assert b["rgb_avail"].dtype == torch.bool and b["gas_mask"].dtype == torch.float32


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_variant_forward(variant):
    torch.manual_seed(0)
    model = TrimodalModel(variant, pretrained=False).eval()
    batch = dummy_batch(B=2, T=3)
    batch["pad_mask"][1, 2] = True     # second trajectory is one day shorter
    batch["rgb_avail"][0, 1] = False   # a missing image
    batch["gas_mask"][:, 0] = 0.0      # missing gas on day one
    with torch.no_grad():
        out = model(batch)
    assert out["cls_logits"].shape == (2, 2)
    assert out["reg_output"].shape == (2, 1)
    assert out["last_idx"].tolist() == [2, 1]
    assert torch.isfinite(out["cls_logits"]).all()


def test_state_dict_names_match_notebooks():
    keys = set(TrimodalModel("full", pretrained=False).state_dict())
    for k in ("gas_encoder.cell.x_mean", "fusion.attn.in_proj_weight", "temporal.pos_enc.pe",
              "rgb_encoder.cbam.channel_attn.mlp.0.weight", "gas_recon_head.3.weight"):
        assert k in keys, k


def test_train_and_evaluate(synthetic, tmp_path):
    from trimodal.evaluation import evaluate_ensemble, find_checkpoints
    from trimodal.training import train_one_seed

    paths, loaders = synthetic
    cfg = TrainConfig(epochs=1, pretrained=False, unfreeze_epochs=(1, 2, 3))
    for seed in (0, 1):
        s = train_one_seed("full", seed, loaders, cfg, tmp_path / "full" / f"seed_{seed}", "cpu", log=lambda _: None)
        assert (tmp_path / "full" / f"seed_{seed}" / "model_state.pt").exists()
        assert 0.0 <= s["best_val_f1"] <= 1.0
    ckpts = find_checkpoints(tmp_path / "full")
    assert sorted(ckpts) == [0, 1]
    res = evaluate_ensemble(ckpts, "full", loaders, "cpu", log=lambda _: None)
    assert 0.1 <= res["threshold"]["threshold"] <= 0.9
    assert len(res["per_trajectory"]) == 12


def test_robustness_is_deterministic(synthetic):
    from trimodal.robustness import corrupt_batch, corruption_generator

    _, loaders = synthetic
    b = next(iter(loaders["test"]))
    a1 = corrupt_batch(b, "joint", 0.5, corruption_generator("full", 0, "joint", 0.5, 0))
    a2 = corrupt_batch(b, "joint", 0.5, corruption_generator("full", 0, "joint", 0.5, 0))
    assert torch.equal(a1["rgb_avail"], a2["rgb_avail"]) and torch.equal(a1["gas_mask"], a2["gas_mask"])
    assert not (a1["rgb_avail"] & b["pad_mask"]).any()
    assert corrupt_batch(b, "rgb", 0.0, corruption_generator(1)) is b


def test_stats():
    from trimodal.stats import benjamini_hochberg, bootstrap_f1_ci, paired_bootstrap

    y = [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1]
    ci = bootstrap_f1_ci(y, y, n_boot=200)
    assert ci["f1"] == 1.0 and ci["ci_low"] == 1.0
    r = paired_bootstrap(y, y, [1] * len(y), n_boot=200)
    assert r["delta_f1"] > 0 and 0 <= r["p_value"] <= 1
    bh = benjamini_hochberg([0.01, 0.04, 0.03, 0.5])
    assert [x["significant"] for x in bh] == [True, False, False, False]
    assert np.allclose([x["p_adj"] for x in bh], [0.04, 0.16 / 3, 0.16 / 3, 0.5])
