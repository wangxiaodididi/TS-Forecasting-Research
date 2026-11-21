#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"
python -m unittest discover -s tests -v
python -m tvp_lab.proxy_smoke --csv data/ETTh1.csv --outdir results/proxy --horizon 96
python -m tvp_lab.evaluate --val results/proxy/val.npz --test results/proxy/test.npz --outdir results/proxy/compare
printf '\n[DONE] CPU proxy smoke results: results/proxy/compare/metrics.csv\n'
