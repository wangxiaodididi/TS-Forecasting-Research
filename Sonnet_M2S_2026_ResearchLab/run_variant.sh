#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VARIANT="${1:-}"
MODE="${2:-val}"  # val = skip test; test = train and evaluate test
LR="${LR:-0.0001}"
SEED="${SEED:-42}"
GPU="${GPU:-0}"

case "$VARIANT" in
  scaler_only|mvca_only|both_fixes) ;;
  *) echo "Usage: bash run_variant.sh {scaler_only|mvca_only|both_fixes} [val|test]"; exit 2 ;;
esac
case "$MODE" in val|test) ;; *) echo "MODE must be val or test"; exit 2 ;; esac

REPO="$ROOT/official/variants/$VARIANT"
if [[ ! -d "$REPO" ]]; then
  echo "Variant not found. Run: python prepare_variants.py"
  exit 3
fi

# Keep the original run's architecture and hyperparameters for one-factor
# comparisons. The paper-aligned configuration is a separate later experiment.
D_MODEL=64
N_ATOMS=16
ALPHA=0.5
REVIN=0

mkdir -p "$ROOT/results/controlled"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/results/controlled/${VARIANT}_${MODE}_seed${SEED}_${STAMP}.log"

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$MODE" == "val" ]]; then
  export SONNET_SKIP_TEST=1
else
  export SONNET_SKIP_TEST=0
fi

set +e
CUDA_VISIBLE_DEVICES="$GPU" python scripts/run_experiment.py \
  model=sonnet \
  dataset=exp_data_config/energy \
  exp=energy \
  seed="$SEED" \
  exp.seq_length=24 \
  exp.pred_length=24 \
  exp.learning_rate="$LR" \
  model.model_params.d_model="$D_MODEL" \
  model.model_params.n_atoms="$N_ATOMS" \
  model.model_params.alpha="$ALPHA" \
  model.model_params.revin="$REVIN" \
  2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

echo "LOG=$LOG"
echo "VARIANT=$VARIANT MODE=$MODE SEED=$SEED LR=$LR"
exit "$STATUS"
