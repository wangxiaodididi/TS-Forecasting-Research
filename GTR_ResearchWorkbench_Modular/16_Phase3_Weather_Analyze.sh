#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Use best checkpoints, not best training epoch's displayed test loss.
./.venv/bin/python -m phase3.cli --dataset Weather --horizon 96 --batch "${1:-20261115_235624_bef5fe}" --limit "${2:-1024}" --device auto
