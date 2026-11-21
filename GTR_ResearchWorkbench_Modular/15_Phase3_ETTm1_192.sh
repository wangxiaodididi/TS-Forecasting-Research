#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Fresh full comparison; uses existing Phase2 runner and official locked configs.
# Experiment runs in the existing worker, monitor via 13_Phase2_Status.sh.
bash 12_Phase2_GPU.sh ETTm1 192 2024
