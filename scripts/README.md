# Reproducible scripts

The notebooks in [`Code/`](../Code) are the original research record. This folder repeats the core
pipeline as command-line programs on top of one shared package, [`src/trimodal`](../src/trimodal),
so a result can be rerun with one command and no notebook state.

| Script | What it does | Notebook it mirrors |
|---|---|---|
| `check_setup.py` | Checks packages, paths and split sizes, then runs one forward/backward pass | — |
| `train.py` | Trains one variant for one or more seeds | 04 (full model), 05 (ablations) |
| `evaluate.py` | Seed-ensemble test evaluation with a k-fold median threshold and a bootstrap CI | 04 §5, 06 |
| `robustness.py` | Hides sRGB / thermal / gas / all on a share of days and measures F1 | 08 |
| `compare.py` | Paired bootstrap ΔF1 between variants with Benjamini–Hochberg correction | 06 |
| `reproduce_all.sh` | Runs all of the above in order | — |

Every script prints its options with `--help`. Install first ([INSTALL.md](../INSTALL.md)).

## Quick check, no dataset needed

```bash
python scripts/check_setup.py --synthetic --no-pretrained
pytest -q tests/
```

`--synthetic` writes a tiny fake TR-6 with the real folder layout and manifest columns to a temporary
folder and runs on that. It proves the code works; it is not a result.

## Check your TR-6 copy

```bash
python scripts/check_setup.py --data-root ~/Desktop/Thesis/TR-6
```

Expected: `train 12 trajectories, 73 days`, `val 12, 21`, `test 11, 18`, each marked `ok`.

## Train

```bash
python scripts/train.py --variant full --seeds 0-9                    # the headline model
python scripts/train.py --variant no_cbam --seeds 0-9 --ablation-recipe
python scripts/train.py --variant full --seeds 0 --epochs 2           # a short trial run
```

Variants: `full`, `no_cbam`, `no_cross_attn`, `lstm`, `mean_impute`, `early_fusion`, `late_fusion`.

Each seed writes `outputs/runs/<variant>/seed_<k>/` with `model_state.pt` (the epoch with the best
validation F1), `history.json` (per-epoch losses, metrics, threshold and task weights) and
`summary.json`. `--skip-existing` resumes an interrupted run.

**Two training recipes.** The headline model (notebook 04) trains only the thermal backbone at the
lower learning rate and keeps torchvision's default stochastic depth. The ablation sweep (notebook 05)
trained both backbones at the lower rate with stochastic depth off, for every configuration it
compared. `--ablation-recipe` selects the second, so ablation variants are compared under the same
recipe as in the thesis.

## Evaluate on the test split

```bash
python scripts/evaluate.py --variant full
```

1. Each seed predicts P(spoiled) for the train + validation trajectories and the test trajectories;
   the seeds are averaged.
2. The train + validation pool is split into K stratified folds (K = smaller class count, at most 6).
   A threshold is calibrated on each fold's training part; the **median** is used.
3. The test set is scored once at that threshold.

Output: `outputs/results/<variant>_test.json` with F1, AUC, MAE, RMSE, R², a 95 % bootstrap CI for
F1, the F1 of always predicting "spoiled", the threshold details, and one line per test trajectory.

### Evaluate the notebooks' own checkpoints

The model code loads checkpoints saved by the notebooks unchanged (the parameter names match, and
the tests check it). Point `--runs` at the notebook's run folder:

```bash
python scripts/evaluate.py --variant full    --runs runs/04_fusion_model/model_a_winner
python scripts/evaluate.py --variant no_cbam --runs runs/05_ablation_study/ablation_1_no_cbam
```

| Variant | Notebook run folder |
|---|---|
| `full` | `runs/04_fusion_model/model_a_winner` |
| `no_cbam` | `runs/05_ablation_study/ablation_1_no_cbam` |
| `no_cross_attn` | `runs/05_ablation_study/ablation_2_no_cross_attn` |
| `lstm` | `runs/05_ablation_study/ablation_3_lstm_not_transformer` |
| `mean_impute` | `runs/05_ablation_study/ablation_4_mean_impute_not_grud` |
| `early_fusion` | `runs/05_ablation_study/baseline_a_early` |
| `late_fusion` | `runs/05_ablation_study/baseline_b_late` |

Model weights are not stored in git (they are about 130 MB per seed), so these folders hold only the
training histories until the checkpoints are copied in.

## Robustness to missing sensors

```bash
python scripts/robustness.py --variant full --split test
python scripts/robustness.py --variant early_fusion --split val --repeats 5
```

For each seed, pattern (`rgb`, `ir`, `gas`, `joint`), level (0 – 50 %) and repeat, a random share of
the observed days has that modality's availability mask switched off. Each seed keeps the threshold
calibrated on its own training set. Writes every run to `..._robustness_<split>.csv` and the
mean over seeds and repeats to `..._summary.csv`.

The notebook seeded each corruption with Python's `hash()`, which changes between interpreter
sessions; the script uses a CRC32 of the same key, so its numbers repeat exactly.

## Compare variants

```bash
python scripts/compare.py --reference full
```

Paired bootstrap over the test trajectories (10 000 resamples): F1 difference, 95 % CI, two-sided
p-value, and Benjamini–Hochberg adjusted p-values across all comparisons.

## Run everything

```bash
bash scripts/reproduce_all.sh
SEEDS=0-2 bash scripts/reproduce_all.sh     # fewer seeds
```

Seventy training runs at 30 epochs each; on a laptop this is an overnight job or longer.

## Exact numbers

GPU kernels are not bit-for-bit deterministic, so a retrained model gives close but not identical
numbers: the thesis itself reports two independent runs of the same model at a per-seed validation F1
of 0.8634 ± 0.0668 and 0.8133 ± 0.0529. With 11 test trajectories a single trajectory changes test F1
by roughly 0.1, so retrained test results should be read against the bootstrap interval, not the point
value. Evaluating the original checkpoints uses the same ensemble and threshold procedure as
notebook 04, so it should give the reported test numbers up to floating-point differences between
devices.
