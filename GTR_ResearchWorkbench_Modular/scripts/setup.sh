#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Reuse the current AutoDL PyTorch/CUDA, but isolate workbench packages and the
# NumPy<2 compatibility pin required by the upstream np.Inf usage.
if [[ ! -x .venv/bin/python ]]; then
  python -m venv --system-site-packages .venv
fi
PY=.venv/bin/python
"$PY" -m pip install -r requirements_workbench.txt
"$PY" -m lab.cli install-official
"$PY" -m pip install -r vendor/GTR/requirements.txt
"$PY" -m lab.cli verify-official
"$PY" - <<'PY'
import numpy, torch
assert int(numpy.__version__.split('.')[0]) < 2, 'NumPy<2 required for official utils/tools.py np.Inf'
print('NumPy:', numpy.__version__, '| torch:',torch.__version__, '| CUDA:', torch.cuda.is_available())
PY
printf 'Install complete. Open dashboard via bash 03_打开可视化面板.sh and forward port 8501.\n'
