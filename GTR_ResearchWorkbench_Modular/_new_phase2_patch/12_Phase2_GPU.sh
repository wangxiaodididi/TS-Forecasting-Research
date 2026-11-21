#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
./.venv/bin/python -m phase2.cli start --dataset "${1:-Weather}" --horizon "${2:-96}" --seed "${3:-2024}" --mode paper
