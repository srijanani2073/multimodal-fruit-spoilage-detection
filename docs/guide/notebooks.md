# Notebook guide

The fourteen notebooks in [`Code/`](../../Code) are the research record: each experiment in the
thesis was run in one of them, and their outputs (histories, CSVs and figures, without model weights)
are committed under [`runs/`](../../runs). Each notebook declares its own data pipeline and model
classes, so any one of them runs on its own once TR-6, the manifest and (from notebook 05 on) the
notebook 04 checkpoints are in place.

Run them in numeric order from a clean state to reproduce the whole study. Set the path lines in the
first code cell first ([INSTALL.md §5](../../INSTALL.md#5-get-the-dataset)).

| # | Notebook | What it does | Writes to `runs/` | Script equivalent |
|---|---|---|---|---|
| 01 | `data_pipeline` | Builds the manifest from TR-6: session grouping, day indices, days-until-spoilage, the stratified 70 / 15 / 15 day-level split, and the loaders shared by every later notebook | `Code/Manifest/manifests/` | `trimodal.data` |
| 02 | `dataset_analysis` | Composition, missing sessions, class balance and per-produce target distributions | — | — |
| 03 | `encoders_baselines` | Builds and checks the CBAM visual encoders and the GRU-D gas encoder in isolation | `03_encoders_baselines/` | `trimodal.model` |
| 04 | `fusion_model` | The full trimodal model: 10 seeds, ensembling, k-fold median threshold and the headline test result | `04_fusion_model/model_a_winner/` | `train.py`, `evaluate.py` |
| 05 | `ablation_study` | Four component ablations and two fusion baselines, 10 seeds each, plus a rerun of the full model | `05_ablation_study/` | `train.py --ablation-recipe` |
| 06 | `bootstrap_significance` | Paired bootstrap tests for the ablation | `06_bootstrap_significance/` | `compare.py` |
| 07 | `backbone_benchmark` | Eight sRGB backbones substituted into the full model, including a frozen DINOv2 | `07_backbone_benchmark/` | — |
| 08 | `missing_data_robustness` | F1 under sRGB, thermal, gas and joint dropout (0 – 50 %) on validation and test, with significance tests | `08_missing_data_robustness/` | `robustness.py`, `compare.py` |
| 09 | `lofo_generalization` | Leave-one-fruit-out: train on five produce types, test on the sixth | `09_lofo_generalization/` | — |
| 10 | `explainability` | Grad-CAM on both visual encoders, SHAP on the gas features, cross-modal attention weights | `10_explainability/` | — |
| 11 | `uncertainty_quantification` | Bootstrap CIs for the test metrics and a six-fold retraining check of model stability | `11_uncertainty_quantification/` | `evaluate.py` (CI) |
| 12 | `regression_diagnosis` | Why the days-until-spoilage head is weak: R² by split, trivial baselines | `12_regression_diagnosis/` | — |
| 13 | `regression_target_triage_combined` | Three alternative regression targets (per-produce z-score, fraction of shelf life, log) | `13_regression_target_triage/` | — |
| 14 | `dmodel_dropout_sweep` | Re-sweep of embedding size and dropout; the defaults were kept | `14_dmodel_dropout_sweep/` | — |

## Folder conventions in `runs/`

- `seed_<k>/history.json`: one entry per epoch (losses, validation F1 / AUC / MAE, the threshold
  calibrated on the training set that epoch, the learned task weights σ, learning rate, time).
- Notebook 04 keeps seeds in `model_a_winner/seed_<k>/`; notebook 05 in `<config>_seed<k>/seed_<k>/`.
- `model_state.pt` files are git-ignored (about 130 MB each). Copy them back into the same folders to
  evaluate with the scripts, see [scripts/README.md](../../scripts/README.md#evaluate-the-notebooks-own-checkpoints).
