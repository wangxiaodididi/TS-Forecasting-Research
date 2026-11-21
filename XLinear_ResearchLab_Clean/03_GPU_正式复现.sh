#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$(readlink -f "$0")")"
PYTHON="${PYTHON:-python}"
"$PYTHON" ops/install_lg.py
DATASETS="${DATASETS:-ETTh1}"
HORIZONS="${HORIZONS:-96}"
SEEDS="${SEEDS:-2025}"
VARIANTS="${VARIANTS:-official,lg_linear,lg_short,lg_dual}"
export PYTHON
bash run.sh full --datasets "$DATASETS" --horizons "$HORIZONS" --seeds "$SEEDS" --variants "$VARIANTS"
