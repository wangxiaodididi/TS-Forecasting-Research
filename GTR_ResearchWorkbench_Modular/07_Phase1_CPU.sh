#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
./.venv/bin/python -m tests.check_model_cpu
./.venv/bin/python -m phase1.launch start --dataset "${1:-ETTm1}" --horizon "${2:-96}" --seed "${3:-2024}" --mode smoke
