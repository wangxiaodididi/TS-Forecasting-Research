#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# No training-parameter overrides; original official baseline is reused if fully verified.
./.venv/bin/python -m phase1.launch start --dataset "${1:-ETTm1}" --horizon "${2:-96}" --seed "${3:-2024}" --mode paper
