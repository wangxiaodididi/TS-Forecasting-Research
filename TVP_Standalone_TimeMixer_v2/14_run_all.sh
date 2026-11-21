#!/usr/bin/env bash
set -euo pipefail
# Run from standalone project root after copying v3 scripts here.
python 13_cpu_checks.py
python 08_validation_matrix.py --project-root . --tasks 'ETTm1:96:2024,ETTm1:192:2024,ETTh1:96:2024' --device gpu --train-missing
python 09_results_summary.py
