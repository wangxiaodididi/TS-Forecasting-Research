#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"
echo '准备对照：ETTh1 96；Official/cc_static/cc_raw/cc_components；2 seeds'
echo '项目配置应与历史正式实验配置一致。此脚本不会改你的YAML或注册表。'
bash run.sh plan --datasets ETTh1 --horizons 96 --seeds 2025,2026 --variants official,cc_static,cc_raw,cc_components
if [ "${1:-}" != '--execute' ];then
  echo '只生成了计划。确认配置后传 --execute 启动 GPU 正式训练。'
  exit 0
fi
bash run.sh full --datasets ETTh1 --horizons 96 --seeds 2025,2026 --variants official,cc_static,cc_raw,cc_components
