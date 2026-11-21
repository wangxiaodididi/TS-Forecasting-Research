#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
PYTHON="${PYTHON:-python}"
"$PYTHON" XLinear_Complementary_v1/scripts/install.py
"$PYTHON" -m unittest discover -s XLinear_Complementary_v1/tests -p 'test_*.py' -v
