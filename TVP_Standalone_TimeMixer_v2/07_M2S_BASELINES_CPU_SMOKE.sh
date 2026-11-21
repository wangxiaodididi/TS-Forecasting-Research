#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
"$PYTHON" -m m2s.verify
"$PYTHON" -m m2s_baselines.run_grid \
  --datasets Hubei CAISO --first-horizon-only \
  --variants last_value seasonal_naive dlinear_target joint_linear gru_joint tcn_joint \
  --device cpu --epochs 1 --patience 1 --batch-size 8 \
  --max-train-windows 16 --max-eval-windows 8
