#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
PYTHON="${PYTHON:-python}"
"$PYTHON" LG_Position_v1/scripts/install.py
"$PYTHON" -m unittest discover -s LG_Position_v1/tests -v
