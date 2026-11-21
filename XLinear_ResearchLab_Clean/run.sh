#!/usr/bin/env bash
# 唯一命令入口：统一配置在 configs/experiment.yaml。支持从 VS Code Remote SSH 直接运行。
set -Eeuo pipefail
cd "$(dirname "$(readlink -f "$0")")"
PYTHON="${PYTHON:-python}"
MODE="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
  init)
    echo '[1/4] 检查当前 Python 与 CUDA PyTorch（不会重装 torch）'
    "$PYTHON" - <<'PYTORCH_CHECK'
import sys
print('Python:',sys.executable)
try:
 import torch
 print('PyTorch:',torch.__version__)
 print('CUDA:',torch.cuda.is_available())
 if torch.cuda.is_available(): print('GPU:',torch.cuda.get_device_name(0))
 else: print('提醒：当前 Python 没有 CUDA，后续 full/pilot 会拒绝启动。请切换 AutoDL CUDA Python 环境。')
except ImportError:
 print('当前 Python 未安装 PyTorch。请使用已有 CUDA PyTorch 环境，再运行 init。')
PYTORCH_CHECK
    echo '[2/4] 安装基础依赖（不包含 PyTorch）'
    "$PYTHON" -m pip install -r requirements.txt
    echo '[3/4] 安装网页可视化依赖'
    "$PYTHON" -m pip install -r requirements-ui.txt
    echo '[4/4] 下载并校验作者固定版本源码，解压公开数据'
    "$PYTHON" -m src.runtime.bootstrap
    echo '初始化完成。下一步：bash run.sh check && bash run.sh plan' ;;
  setup)      "$PYTHON" -m src.runtime.bootstrap "$@" ;;
  check)      "$PYTHON" -m tests.selftest "$@" ;;
  baseline)   "$PYTHON" -u -m src.training.runner --mode full --datasets ETTh1 --horizons 96 --variants official --seeds 2025 "$@" ;;
  plan)       "$PYTHON" -m src.training.runner --mode full --plan "$@" ;;
  pilot)      "$PYTHON" -u -m src.training.runner --mode pilot "$@" ;;
  full)       "$PYTHON" -u -m src.training.runner --mode full "$@" ;;
  full-bg)
    mkdir -p ops
    LOG="ops/background_$(date +%Y%m%d_%H%M%S).log"
    nohup "$PYTHON" -u -m src.training.runner --mode full "$@" > "$LOG" 2>&1 < /dev/null &
    pid=$!
    echo "$pid" > "${LOG%.log}.pid"
    echo "后台任务已启动：PID=$pid；日志=$LOG"
    echo "查看日志：tail -f $LOG" ;;
  status)     "$PYTHON" -m src.training.status "$@" ;;
  diagnose)   "$PYTHON" -m src.training.diagnostics "$@" ;;
  probe)      "$PYTHON" -m src.training.mechanism_probe "$@" ;;
  dashboard)
    if ! "$PYTHON" -c 'import streamlit' 2>/dev/null; then
      echo '可视化依赖尚未安装。执行：python -m pip install -r requirements-ui.txt'; exit 1
    fi
    "$PYTHON" -m streamlit run src/dashboard/app.py --server.address 127.0.0.1 --server.port 8501 "$@" ;;
  help|*)
    cat <<'USAGE'
XLinear Research Lab — 唯一操作入口
  bash run.sh init               # 首次安装依赖并准备官方源码/数据（无需旧项目）
  bash run.sh setup              # 仅准备/校验官方源码和公开数据
  bash run.sh setup --reuse-upstream <OLD_PATH> # 复制旧项目已经下载的作者源码
  bash run.sh check              # CPU 模型结构与梯度测试
  bash run.sh baseline           # ETTh1-H96 作者配置 FP32 正式原版复现
  bash run.sh plan               # 查看 GPU 全量实验计划，不训练
  bash run.sh pilot              # 短轮数 GPU 工程测试（前台）
  bash run.sh full               # 官方协议完整 GPU 训练（前台）
  bash run.sh full-bg            # 完整 GPU 训练（后台）
  bash run.sh status             # 查看进度与结果目录
  bash run.sh diagnose --run RUN_ID [--job JOB_ID] # 完成后离线预测/门控/梯度分析
  bash run.sh probe [--run RUN_ID] [--splits val] # 离线反事实/频率/TGM诊断，不训练
  bash run.sh dashboard          # 3秒自动刷新网页，VS Code 转发 8501 端口
命令可附加参数：--datasets ETTh1,Weather --horizons 96 --variants official,dyn_token --seeds 2025,2026
所有主参数统一配置于 configs/experiment.yaml。不要同时运行 full 和 full-bg。
USAGE
    ;;
esac
