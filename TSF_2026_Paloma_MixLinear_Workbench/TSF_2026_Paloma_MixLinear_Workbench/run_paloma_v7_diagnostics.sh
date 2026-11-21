#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python}"
echo "[PYTHON] $($PYTHON_BIN -c 'import sys;print(sys.executable)')"
"$PYTHON_BIN" paloma_v7_diagnose.py --workspace "$(pwd)" "$@"
