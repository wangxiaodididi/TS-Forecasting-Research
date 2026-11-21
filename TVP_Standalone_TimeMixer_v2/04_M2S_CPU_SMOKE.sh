#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
"$PYTHON" -m m2s.verify
if [ ! -f training/workspace/official/models/TimeMixer.py ]; then
  echo '[ERROR] Missing official source: run python training/prepare.py once, then retry' >&2
  exit 2
fi
"$PYTHON" -m m2s.run_grid --datasets Hubei CAISO --first-horizon-only \
  --variants target_only multivariate_joint --device cpu \
  --epochs 1 --patience 1 --batch-size 8 --max-train-windows 16 \
  --max-eval-windows 8
