# Results

The numbers below are those reported in the thesis. Each table names the notebook that produced it;
the matching outputs are in [`runs/`](../../runs).

## Headline: held-out test set (notebook 04)

10-seed ensemble, 11 test trajectories, threshold 0.660.

| Metric | Value |
|---|---|
| F1 (spoiled) | **0.8000** (95 % CI [0.50, 1.00]) |
| AUC | 0.7333 |
| Accuracy | 0.727 (8 of 11) |
| Always-predict-spoiled F1 | 0.7059 |
| MAE / RMSE (days) | 0.9188 / 1.0593 |
| R² | 0.0439 |

Every spoiled test trajectory was detected; the three errors are not-spoiled trajectories (guava,
Indian gooseberry, tomato) called spoiled. Per-seed validation F1 was 0.8634 ± 0.0668. The
regression head performs close to predicting the mean (R² ≈ 0.04).

## Robustness to missing sensors (notebook 08)

Test-set F1 (mean over 10 seeds × 5 random repeats) as a growing share of days loses one sensor:

| Pattern | Model | 0 % | 10 % | 20 % | 30 % | 40 % | 50 % |
|---|---|---|---|---|---|---|---|
| sRGB missing | **Proposed** | 0.676 | 0.661 | 0.652 | 0.638 | 0.595 | 0.587 |
| | Early fusion | 0.655 | 0.653 | 0.648 | 0.638 | 0.629 | 0.626 |
| | Late fusion | 0.612 | 0.599 | 0.606 | 0.602 | 0.575 | 0.586 |
| Thermal missing | **Proposed** | 0.676 | 0.669 | 0.669 | 0.681 | 0.674 | 0.674 |
| | Early fusion | 0.655 | 0.651 | 0.645 | 0.635 | 0.619 | 0.622 |
| | Late fusion | 0.612 | 0.621 | 0.645 | 0.627 | 0.643 | 0.642 |
| Gas missing | **Proposed** | 0.676 | 0.679 | 0.673 | 0.676 | 0.683 | 0.676 |
| | Early fusion | 0.655 | 0.649 | 0.651 | 0.650 | 0.644 | 0.645 |
| | Late fusion | 0.612 | 0.602 | 0.603 | 0.607 | 0.606 | 0.594 |
| All three | **Proposed** | 0.676 | 0.670 | 0.651 | 0.622 | 0.608 | 0.604 |
| | Early fusion | 0.655 | 0.629 | 0.617 | 0.610 | 0.576 | 0.579 |
| | Late fusion | 0.612 | 0.613 | 0.596 | 0.612 | 0.608 | 0.608 |

These are per-seed F1 scores at each seed's own threshold, so they sit below the ensemble's 0.80.
At 50 % missingness the proposed model's advantage is statistically confirmed (paired bootstrap,
Benjamini–Hochberg across 29 comparisons) for gas-only dropout against both baselines on validation
and against late fusion on test, and for thermal-only dropout against both baselines on validation.
Losing the sRGB camera hurts the proposed model the most.

## Ablation (notebooks 05 and 06)

| Configuration | Single-seed val. F1 | Ensemble F1 [95 % CI] | ΔF1 vs. full [95 % CI], p |
|---|---|---|---|
| Full model | 0.8133 ± 0.0529 | 0.8000 [0.50, 1.00] | — |
| No CBAM | 0.8225 ± 0.0565 | 0.7500 [0.44, 0.95] | −0.050 [−0.17, 0.00], p = 0.711 |
| No cross-attention | 0.7672 ± 0.0691 | 0.6667 [0.40, 0.86] | −0.133 [−0.32, 0.00], p = 0.064 |
| LSTM instead of Transformer | 0.7315 ± 0.0432 | 0.8571 [0.60, 1.00] | +0.057 [−0.14, 0.27], p = 0.766 |
| Mean imputation instead of GRU-D | 0.8154 ± 0.0841 | 0.9091 [0.67, 1.00] | +0.109 [−0.19, 0.45], p = 0.514 |
| Early fusion | 0.7396 ± 0.0546 | 0.7059 [0.40, 0.91] | −0.094 [−0.25, 0.00], p = 0.230 |
| Late fusion | 0.7023 ± 0.0394 | 0.6667 [0.40, 0.86] | −0.133 [−0.32, 0.00], p = 0.064 |

No difference is statistically significant: at this dataset size the ablation can neither confirm nor
rule out any single component. The gap between two independent training runs of the full model
(0.86 vs 0.81 validation F1) is as large as most ablation differences.

## sRGB backbone benchmark (notebook 07)

Each backbone substituted into the complete model, 10 seeds, validation set.

| Backbone | Val. F1 | Params (M) | GFLOPs |
|---|---|---|---|
| **ConvNeXt-Tiny (chosen)** | 0.7912 ± 0.0739 | 32.47 | 39.03 |
| Swin-Tiny | 0.7887 ± 0.0510 | 32.17 | 27.14 |
| EfficientNet-B0 | 0.7701 ± 0.0433 | 8.83 | 6.63 |
| RegNetY-1.6GF | 0.7469 ± 0.0337 | 15.00 | 16.48 |
| DINOv2 ViT-S/14 (frozen) | 0.7467 ± 0.0305 | 26.63 | 47.52 |
| ResNet-18 | 0.7414 ± 0.0348 | 15.77 | 17.91 |
| MobileNetV3-Small | 0.7388 ± 0.0358 | 5.54 | 3.81 |
| EfficientNetV2-S | 0.7362 ± 0.0263 | 25.00 | 26.53 |

## Leave-one-fruit-out (notebook 09)

Trained on five produce types, tested on both trajectories of the sixth:

| Held-out | Ensemble correct (of 2) | Mean seed F1 |
|---|---|---|
| Banana | 1 | 0.633 |
| Carrot | 1 | 0.367 |
| Guava | 1 | 0.400 |
| Indian gooseberry | 1 | 0.167 |
| Mango | 1 | 0.500 |
| Tomato | 2 | 0.733 |

Only tomato is fully recognised: generalisation to an unseen produce type is weak with two specimens
per type.

## Limits

The test set is 11 trajectories, so confidence intervals are wide; the day-level split means test
days come from specimens seen in training; and the regression head is close to the mean predictor.
These are discussed in the thesis (Chapter 5).
