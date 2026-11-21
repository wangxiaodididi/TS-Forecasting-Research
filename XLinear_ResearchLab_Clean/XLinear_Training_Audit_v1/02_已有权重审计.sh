#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 1 ]; then echo "用法: bash XLinear_Training_Audit_v1/02_已有权重审计.sh <runs目录名称> [最大验证批次数，0为完整验证]"; exit 2; fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" "$SCRIPT_DIR/scripts/audit.py" manifest --run "$1"
"$PYTHON_BIN" "$SCRIPT_DIR/scripts/audit.py" checkpoint --run "$1" --max-batches "${2:-8}"
