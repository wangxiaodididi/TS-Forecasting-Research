# 从零恢复：完全不依赖旧项目

本压缩包自带：Python 研究源码、真实公开数据压缩包、统一配置、网页看板和训练/诊断程序。作者官方 XLinear 固定版本代码首次执行 `bash run.sh init` 在线获取并进行 Git blob 校验。若服务器网络不可访问 GitHub codeload，需先修复网络；不会静默使用假官方代码。

## 安装

```bash
cd /root/autodl-tmp
unzip XLinear_ResearchLab_Full_Rebuild_v4.zip
cd XLinear_ResearchLab_Clean
bash run.sh init
bash run.sh check
bash run.sh plan
```

## 运行正式官方复现

```bash
bash run.sh baseline
```

ETTh1, L=96, H=96，作者官方配置与随机种子2025；GPU FP32。它会生成新 `runs/` 目录、新 checkpoint 和真实 MSE/MAE。旧 checkpoint 不在本压缩包，之前的 0.369467 是已完成的历史结果，无法从丢失的文件中直接找回。

## 新模块研究

`bash run.sh pilot`：两 Epoch 的工程试运行，非论文指标。`bash run.sh full`：三模型一随机种子，固定作者训练协议（如需增加种子，修改 `experiment.seeds`）。`bash run.sh status`：查看结果。`bash run.sh diagnose --run RUN_ID`：生成门控、梯度和逐预测步误差。

## 实时网页

另开终端执行 `bash run.sh dashboard`；VS Code 的 PORTS 转发 8501，本机浏览器打开 `http://localhost:8501`。Train/Val 每 Epoch 结束更新，Batch Loss 在训练过程中定期更新，GPU 监控外部采样。网页读取日志，并不重跑模型。

## Python 环境

沿用 AutoDL 已安装的 GPU PyTorch，安装器不会修改 torch。若当前 Python 的 `torch.cuda.is_available()` 为 False，先切换为服务器 CUDA 环境，不能通过改可视化脚本解决。

## 模型改造入口

- 改参数：`configs/experiment.yaml`
- 改/新增网络模块：`src/models/extensions/`
- 注册变体：`src/models/registry.py`
- 观察 Train/Val/GPU：`src/dashboard/app.py`
- 诊断接口：`src/training/diagnostics.py`
- 固定官方训练/数据/优化器：`vendor/XLinear/`（自动获取并哈希校验，不直接修改）
