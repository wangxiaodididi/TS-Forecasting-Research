#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-energy}"
if [[ "$DATASET" != energy && "$DATASET" != etth1 ]]; then
  echo 'Usage: bash 02_small_realdata.sh energy|etth1' >&2; exit 2
fi
PY="${PYTHON_EXEC:-python}"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$ROOT/official/Sonnet${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/runs"
"$PY" "$ROOT/scripts/smoke_realdata.py" --dataset "$DATASET" --steps 4 2>&1 | tee "$ROOT/runs/02_small_${DATASET}.log"
