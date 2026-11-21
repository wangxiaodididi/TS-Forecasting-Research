#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
"$PYTHON" -m m2s.verify
"$PYTHON" -m m2s_baselines.run_grid --datasets Hubei CAISO \
  --variants last_value seasonal_naive dlinear_target joint_linear gru_joint lstm_joint tcn_joint \
  --device cuda --epochs 20 --patience 5 --batch-size 32
