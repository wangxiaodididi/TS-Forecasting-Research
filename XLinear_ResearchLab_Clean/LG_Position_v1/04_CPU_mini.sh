#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
PYTHON="${PYTHON:-python}"
"$PYTHON" LG_Position_v1/scripts/install.py
if [[ ! -f ops/cpu_mini.py ]]; then
    echo 'LG v1 ops/cpu_mini.py is required. Please unzip original LG v1 first.' >&2
    exit 1
fi
"$PYTHON" ops/cpu_mini.py --variants official,lg_short,lg_position --epochs 2 --train-windows 512 --val-windows 256
