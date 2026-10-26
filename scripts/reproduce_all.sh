#!/usr/bin/env bash
# Reproduce the core results end to end: the headline model, the six ablation variants,
# test evaluation for each, the sensor-dropout sweep and the paired significance tests.
#
#   bash scripts/reproduce_all.sh                 # everything, 10 seeds each
#   SEEDS=0-2 bash scripts/reproduce_all.sh       # fewer seeds for a quicker pass
#
# Paths come from --data-root/--manifest/--out defaults or the TR6_ROOT, TR6_MANIFEST and
# TRIMODAL_OUT environment variables (see INSTALL.md). Seeds that already have a checkpoint are
# skipped, so the script can be stopped and restarted.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-0-9}"
PY="${PYTHON:-python}"
ABLATIONS="no_cbam no_cross_attn lstm mean_impute early_fusion late_fusion"

echo "== 0. setup check"
$PY scripts/check_setup.py

echo "== 1. headline model (notebook 04 recipe)"
$PY scripts/train.py --variant full --seeds "$SEEDS" --skip-existing
$PY scripts/evaluate.py --variant full

echo "== 2. ablation variants (notebook 05 recipe)"
for v in $ABLATIONS; do
  $PY scripts/train.py --variant "$v" --seeds "$SEEDS" --skip-existing --ablation-recipe
  $PY scripts/evaluate.py --variant "$v"
done

echo "== 3. missing-modality robustness on the test split"
for v in full early_fusion late_fusion; do
  $PY scripts/robustness.py --variant "$v" --split test
done

echo "== 4. paired bootstrap against the full model"
$PY scripts/compare.py --reference full

echo "Done. Results are in ${TRIMODAL_OUT:-outputs}/results/"
