#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
"${PWD}/.venv/bin/python" -m lab.cli launch --dataset "${1:-ETTm1}" --model "${2:-GTR}" --horizon "${3:-96}" --seed "${4:-2024}" --mode paper
