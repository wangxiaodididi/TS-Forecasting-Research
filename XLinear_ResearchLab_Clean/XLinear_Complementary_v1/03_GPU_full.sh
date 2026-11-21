#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
PYTHON="${PYTHON:-python}"
"$PYTHON" XLinear_Complementary_v1/scripts/install.py
DATASETS="${DATASETS:-ETTh1}"
HORIZONS="${HORIZONS:-96,192}"
SEEDS="${SEEDS:-2025,2026}"
VARIANTS="${VARIANTS:-official,lg_short,lg_tmix_r16,lg_trend_w24}"
export PYTHON
bash run.sh full --datasets "$DATASETS" --horizons "$HORIZONS" --seeds "$SEEDS" --variants "$VARIANTS"
