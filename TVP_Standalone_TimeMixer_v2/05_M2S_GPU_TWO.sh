#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
"$PYTHON" -m m2s.verify
"$PYTHON" -m m2s.run_grid --datasets Hubei CAISO \
  --variants target_only multivariate_joint --device cuda \
  --epochs 20 --patience 5 --batch-size 32
