#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
BATCH="${1:-20261115_235624_bef5fe}"
if [[ ! -f "phase3_reports/$BATCH/REPORT.txt" ]]; then
  echo "No report found. Run 16_Phase3_Weather_Analyze.sh first." >&2
  exit 1
fi
cat "phase3_reports/$BATCH/REPORT.txt"
