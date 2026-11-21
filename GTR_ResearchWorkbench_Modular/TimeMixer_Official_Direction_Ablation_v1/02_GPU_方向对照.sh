#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-$ROOT/../.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python
# Optional env FAST=1: only ETTm1-96 original+reverse, one seed (first official check).
if [[ "${FAST:-0}" == "1" ]]; then
    DATASETS=(ETTm1); HORIZONS=(96); SEEDS=(2024)
else
    DATASETS=(ETTm1 Weather); HORIZONS=(96 192); SEEDS=(2024 2025)
fi
for dataset in "${DATASETS[@]}"; do
    for h in "${HORIZONS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for variant in original reverse; do
                "$PYTHON" runner.py train --dataset "$dataset" --horizon "$h" --seed "$seed" --variant "$variant" --device gpu
            done
            "$PYTHON" summary.py
        done
    done
done
