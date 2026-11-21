#!/usr/bin/env bash
# Runs a tiny chronological CPU exercise on pinned XLinear, NOT paper reproduction.
set -Eeuo pipefail
cd "$(dirname "$(readlink -f "$0")")"
PYTHON="${PYTHON:-python}"
"$PYTHON" ops/install_lg.py
CUDA_VISIBLE_DEVICES='' "$PYTHON" ops/cpu_mini.py "$@"
