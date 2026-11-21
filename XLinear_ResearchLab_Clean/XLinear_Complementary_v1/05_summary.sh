#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
if [[ $# -lt 1 ]]; then echo 'Usage: bash XLinear_Complementary_v1/05_summary.sh runs/<run_id>' >&2; exit 1; fi
PYTHON="${PYTHON:-python}"
"$PYTHON" XLinear_Complementary_v1/scripts/summarize.py "$1"
