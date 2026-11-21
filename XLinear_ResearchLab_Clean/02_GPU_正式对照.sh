#!/usr/bin/env bash
# MODE=pilot for 2-epoch engineering smoke; MODE=full for official-protocol training.
set -Eeuo pipefail
cd "$(dirname "$(readlink -f "$0")")"
PYTHON="${PYTHON:-python}"
"$PYTHON" ops/install_hcf.py
MODE="${MODE:-full}"
case "$MODE" in pilot|full) ;; *) echo "MODE must be pilot or full" >&2; exit 2;; esac
DATASETS="${DATASETS:-ETTh1}"
HORIZONS="${HORIZONS:-96}"
SEEDS="${SEEDS:-2025}"
VARIANTS="${VARIANTS:-official,hcf_static,hcf_shared,hcf}"
export PYTHON
bash run.sh "$MODE" --datasets "$DATASETS" --horizons "$HORIZONS" --seeds "$SEEDS" --variants "$VARIANTS"
