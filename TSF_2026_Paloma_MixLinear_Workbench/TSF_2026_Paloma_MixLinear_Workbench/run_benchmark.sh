#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-$(command -v python)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/GTR_ResearchWorkbench_Modular/dataset}"
PROFILE="${PROFILE:-matched96}"
MODELS="${MODELS:-tstruct timebase}"
DATASETS="${DATASETS:-ETTh1 ETTm1}"
HORIZONS="${HORIZONS:-96 336}"
RESUME="${RESUME:-1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python not found. Activate your existing .venv or set PYTHON=/path/to/python" >&2
  exit 2
fi
if [[ ! -f "$DATA_ROOT/ETTh1.csv" || ! -f "$DATA_ROOT/ETTm1.csv" ]]; then
  echo "[ERROR] Expected ETTh1.csv and ETTm1.csv under DATA_ROOT=$DATA_ROOT" >&2
  echo "Set DATA_ROOT=/absolute/path/to/dataset and retry." >&2
  exit 2
fi

cd "$ROOT"
"$PYTHON_BIN" test_patcher.py
"$PYTHON_BIN" prepare_sources.py --root "$ROOT/third_party"
"$PYTHON_BIN" patch_sources.py --root "$ROOT/third_party"
"$PYTHON_BIN" check_environment.py
"$PYTHON_BIN" smoke_test.py --root "$ROOT/third_party" --profile "$PROFILE"
"$PYTHON_BIN" run_benchmark.py \
  --profile "$PROFILE" \
  --models $MODELS \
  --datasets $DATASETS \
  --horizons $HORIZONS \
  --seed 2024 \
  --data-root "$DATA_ROOT" \
  --resume "$RESUME"
