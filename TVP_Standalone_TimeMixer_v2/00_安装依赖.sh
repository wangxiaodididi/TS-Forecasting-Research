#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
"$PYTHON" -m pip install "numpy>=1.24" "pandas>=2" "scikit-learn" "einops" "reformer-pytorch" "sktime" "matplotlib"
"$PYTHON" -c "import torch;print('PyTorch',torch.__version__,'CUDA',torch.cuda.is_available())"
