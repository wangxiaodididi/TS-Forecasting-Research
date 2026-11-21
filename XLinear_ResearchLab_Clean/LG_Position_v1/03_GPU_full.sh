#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
PYTHON="${PYTHON:-python}"
"$PYTHON" LG_Position_v1/scripts/install.py
DATASETS="${DATASETS:-ETTh1}"
HORIZONS="${HORIZONS:-96,192}"
SEEDS="${SEEDS:-2025,2026}"
VARIANTS="${VARIANTS:-official,lg_short,lg_position}"
export PYTHON
bash run.sh full --datasets "$DATASETS" --horizons "$HORIZONS" --seeds "$SEEDS" --variants "$VARIANTS"
