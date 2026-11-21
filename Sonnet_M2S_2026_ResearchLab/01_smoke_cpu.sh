#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_EXEC:-python}"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$ROOT/official/Sonnet${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/runs"
"$PY" "$ROOT/scripts/smoke_model.py" --steps 2 --seq-len 48 --pred-len 24 --n-vars 7 --batch-size 2 2>&1 | tee "$ROOT/runs/01_smoke_cpu.log"
