# TimeMixer 官方架构：跨尺度方向反事实实验（v1）

## 本轮只检验一个问题

> 原始 TimeMixer 的 **Seasonal 细→粗、Trend 粗→细** 传递是否在不同预测期限下始终合适？

原始版本 `TimeMixer` 与反方向版本 `TimeMixerReverse` 保留相同的 PDM 层数、多尺度平均下采样、分解方法、归一化、Embedding、FMM 预测头及优化器；只交换两条 PDM 分支的信息传递方向。反方向：Seasonal 粗→细，Trend 细→粗。

这是**官方 TimeMixer 的配对消融**，不是提出新模型，也不是上一阶段的简化代理网络。

- 官方仓库：https://github.com/kwuking/TimeMixer
- 固定源码 commit：`e24610583b36fdd8c76cc17a8df4e65759a5f460`（2025-10-05），Apache-2.0
- 已核查官方 `models/TimeMixer.py`、`run.py`、`exp/exp_basic.py` 和 ETTm1 / Weather 的 `*_unify.sh`。
- 本补丁不会修改 `GTR_ResearchWorkbench_Modular` 的原始源码或运行结果；新工作区在本目录下 `workspace/official/`。

### 作者的正式设置（本研究仅选 H=96、192）

| 参数 | ETTm1 | Weather |
|---|---|---|
| `seq_len` | 96 | 96 |
| `e_layers` | 2 | 3 |
| `down_sampling_layers` | 3 | 3 |
| `down_sampling_window` | 2 | 2 |
| `d_model`, `d_ff` | 16, 32 | 16, 32 |
| `learning_rate` | 0.01 | 0.01 |
| `batch_size` | 16 | 128 |
| `train_epochs` | 100（官方 run.py 默认） | 20 |
| `patience` | 15（官方 run.py 默认） | 10 |
| 特征数 | 7 | 21 |
| `num_workers` | 本补丁默认 4（数据加载并行度）| 同左 |

> 作者 `run.py` 把随机种子写死为 2021，本实验在 **隔离副本** 中改为环境变量 `TIMEMIXER_SEED`，以便在 2024/2025 两个种子下配对比较。为避免测试集被用于开发，隔离副本关闭训练期间每轮 Test Loss 计算，也关闭训练后自动测试。这是评估流程的透明差异，不影响模型/验证损失及训练超参数。

## 使用说明（AutoDL，VS Code 远程终端）

先把 `TimeMixer_Official_Direction_Ablation_v1.zip` 上传到你原有项目目录，执行：

```bash
cd /root/autodl-tmp/GTR_ResearchWorkbench_Modular
unzip -n TimeMixer_Official_Direction_Ablation_v1.zip
cd TimeMixer_Official_Direction_Ablation_v1

# 先准备固定版本官方源码，程序下载后检查源码 blob SHA
../.venv/bin/python prepare.py
# 如果 GitHub 下载不通，PC 下载：
# https://codeload.github.com/kwuking/TimeMixer/zip/e24610583b36fdd8c76cc17a8df4e65759a5f460
# 上传后运行：../.venv/bin/python prepare.py --archive /path/to/downloaded.zip
# 或复制本地已有官方同一 commit：../.venv/bin/python prepare.py --source /path/to/original/TimeMixer
```

**Python 依赖**：建议沿用现有 `.venv`（torch、numpy、pandas、scipy、scikit-learn、einops、reformer-pytorch、matplotlib）；如果官方 import 报缺包，针对缺少的包补装即可，**不要直接按官方较旧 requirements.txt 降级 PyTorch**。TimeMixer 官方还可能要求 `sktime` 等包，依据实际报错安装。

准备成功后，先执行真正的官方网络 CPU 自检：

```bash
../.venv/bin/python tests/test_official_integration.py
```

预期看到 ETTm1/Weather × H=96/192 的 **4 行 `[PASS]`**，证明两个模型完整前向反向通过，并且参数量相同。如果失败，先修复环境，不启动 GPU 正式训练。

**最快的正式预检**：只运行 `ETTm1-96`、Seed 2024 的官方原方向和反方向：

```bash
FAST=1 PYTHON=../.venv/bin/python bash 02_GPU_方向对照.sh
```

完整矩阵：两数据集 × 两步长 × 两种方向 × 两随机种子，共 16 次原版架构的完整训练（可能需要较长 GPU 时间）；顺序执行，不同时启动 GPU 进程：

```bash
bash 02_GPU_方向对照.sh
```

单次运行（只用官方完整模型）：

```bash
../.venv/bin/python runner.py train --dataset Weather --horizon 192 --seed 2024 --variant reverse --device gpu
```

查看验证结果：

```bash
../.venv/bin/python summary.py
```

`runs/` 保留每个模型的逐 Epoch 日志；`validation_summary.csv` 给出每次最佳 Validation MSE 与 Epoch。程序自动跳过已经有训练日志的组合，出错时检查 `runs/.../train.log`；需要从头重跑单次时添加 `--force`。

如需将数据集路径直接指定（比如数据不在 `GTR_ResearchWorkbench_Modular/dataset` 下），给 `runner.py` 添加 `--data-root /your/dataset`。它会识别根目录、`ETT-small/` 或 `weather/` 下的 CSV。

## 只看验证集，测试集后置

官方 `exp_long_term_forecasting.py` 原本会在每一轮训练后计算 `Test Loss`，容易让开发者在选择模型前接触测试指标。本补丁的隔离版本将其替换为 `nan`，所有模型都按照相同修改执行。正式测试需要在验证决策锁定后通过 `runner.py test ...` 单独执行，且无需重新训练。

## 终止条件（先定规则，避免循环）

采用平均验证 MSE 作为主指标，同时报告每个数据集/种子/预测长度。仅当出现以下一致规律时，才进入下一轮创新：

1. `H=96`：反方向在 4 个数据集×种子组合中至少 **3 个**验证 MSE 更低；
2. `H=192`：原方向在 4 个组合中至少 **3 个**验证 MSE 更低；
3. 两个预测期限的平均相对变化有实质量级（建议至少约 **0.5%**），且不存在单一异常实验主导。

若上述条件不满足，**停止“预测期限决定跨尺度方向”研究，不开发新的方向门控或反复追参**。即使满足，也只能视作开展正式多种子/多数据集验证的前提，尚不能作为论文定论。

## 本地已验证 / 尚未验证

- 已核对 GitHub 官方固定源码和超参数；`tests/test_local_structure.py` 已在当前 CPU 环境通过参数量、形状和梯度测试（使用等价接口的轻量 stub，不是官方整网）。
- `prepare.py` 含官方三个关键文件的 Git blob SHA 版本校验，不会错误使用最新不兼容源码。
- **当前环境无法下载 GitHub 归档，无法直接运行官方完整网络与 GPU 训练。因此真实官方网络 CPU 整体 smoke、验证 MSE 必须在 AutoDL 上运行。** 请不要将上轮简化模型的结果当成本实验结果。

## 文件结构

- `prepare.py`：下载/导入指定 commit，克隆至隔离区、登记模型、关闭 test 泄漏。
- `model_code/TimeMixerReverse.py`：官方 TimeMixer 子类，仅交换 PDM 季节/趋势混合方向。
- `runner.py`：官方训练与最终测试单组运行。
- `summary.py`：只读取验证日志，按实验输出最佳验证 MSE。
- `tests/`：本地轻量结构测试及服务器官方完整架构检查。
- `01_CPU_结构自检.sh`、`02_GPU_方向对照.sh`、`03_查看验证结果.sh`：运行入口。

