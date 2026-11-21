# TStruct + TimeBase 母体复现工作包 v1

本包用于在现有 Paloma 工作台中，对两个完整时间序列预测网络进行独立训练与验证集比较：

- **TStruct**：NeurIPS 2026，官方仓库 MIT License，固定源码 commit `5b9eb393a092c14d28d08395b63de809accd115a`。
- **TimeBase**：ICML 2025 Spotlight，官方仓库 MIT License，固定源码 commit `369b330f3d77371fcc7e8c75c808d01330c40899`。

官方仓库及论文：
- TStruct paper: https://arxiv.org/abs/2610.17864
- TStruct code: https://github.com/yangzeli/TStruct
- TimeBase paper: https://proceedings.mlr.press/v267/huang25az.html
- TimeBase code: https://github.com/hqh0728/TimeBase

## 重要：两个输入长度口径

两个官方 ETT 脚本均采用 `seq_len=720`。但此前 Paloma 基准采用 `lookback=96`。为避免把不同输入长度的结果直接混在一起，本包提供两种 profile：

### `matched96`（默认，优先运行）
- ETTh1、ETTm1；预测步长 H=96、336；历史输入 L=96。
- 与 Paloma 既有实验保持相同数据集、预测步长、输入长度、标准 ETT 划分、M 特征和训练集 StandardScaler。
- TStruct 的官方模型结构保留，但将 patch size 固定为 8，使 96 个历史点形成 12 个 patch。**这是为 L=96 公平比较所作的输入配置适配，不是作者原始 L=720 配置。**
- TimeBase 保留官方周期长度：ETTh1 `period_len=24`，ETTm1 `period_len=4`；保留作者脚本对应的 basis 数、正交权重、学习率和训练轮数。

### `paper720`（可选）
- 同样两个数据集、H=96/336，但 L=720。
- TStruct 使用作者 ETT 脚本的 patch size 与其余数据集/预测步长配置；TimeBase 使用作者 ETT 脚本的 period、basis、正交权重、学习率、batch size 等配置。
- 此 profile 采用固定 seed 2024，并且仅报告验证集，不评估测试集。因此它是“作者输入长度及主要超参数配置下的验证复现”，不是对论文测试指标的逐项复刻。

## 测试集保护

上游两个训练脚本原本会在每个 epoch 读取并评估 test split。本包会对固定 commit 的工作副本做最小补丁：
1. 训练和早停只使用 train/validation；
2. 不在训练过程中加载 test split；
3. 训练结束后只计算 best checkpoint 的 validation MSE/MAE；
4. 额外将 TimeBase 上游 `np.Inf` 改为 `np.inf`，以兼容 NumPy 2.x；
5. 不调用上游 `exp.test()`。

因此本包不会使用测试集挑选模型。`summary.csv` 中的结果是验证集指标，不能与论文报告的测试集指标直接比较。

两个仓库的 MIT License 文件随源码下载保留。补丁后的源文件 SHA256、源码 commit、完整配置和日志路径会写入结果汇总。

## 运行环境

建议使用你现有的 AutoDL `.venv`（PyTorch 2.x、CUDA 可用）。依赖：`torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `einops`。

默认数据目录：
`/root/autodl-tmp/GTR_ResearchWorkbench_Modular/dataset`

其中必须存在 `ETTh1.csv` 和 `ETTm1.csv`。可通过 `DATA_ROOT` 环境变量覆盖。

## 一键运行

在 Paloma 工作台内层目录（建议与 `experiment.py` 同级）解压本包，然后：

```bash
unzip -o Paloma_MotherModel_Benchmark_v1.zip -d .
bash run_benchmark.sh
```

默认运行 `matched96` 的 8 个任务：2 个模型 × 2 个数据集 × 2 个预测步长。

可选运行作者输入长度配置：

```bash
PROFILE=paper720 bash run_benchmark.sh
```

两种 profile 都运行：

```bash
PROFILE=both bash run_benchmark.sh
```

指定数据集或预测步长：

```bash
DATASETS="ETTh1" HORIZONS="96 336" bash run_benchmark.sh
```

仅运行一个模型：

```bash
MODELS="tstruct" bash run_benchmark.sh
MODELS="timebase" bash run_benchmark.sh
```

## 结果位置

所有文件写入：

`runs/tstruct_timebase_benchmark_v1/`

主要结果：
- `summary.csv`：每个任务的验证集 MSE、MAE、运行时间、源码 commit、补丁 SHA256、参数配置。`matched96` 额外与 v7 已核验的 Original Paloma 验证集指标比较，给出 `mse_gain_pct_vs_paloma` / `mae_gain_pct_vs_paloma`；`paper720` 不与 L96 Paloma 比较。
- `logs/`：每个任务的完整训练日志。
- `checkpoints/`：每个任务的 best validation checkpoint。
- `third_party/download_manifest.json`：源码版本、许可证及上游模型文件哈希。
- `third_party/patch_manifest.json`：验证集专用补丁前后 SHA256。
- `smoke_test.log`：模型前向/反向检查结果。

程序支持断点续跑：已有 `status=ok` 的任务会自动跳过。若需强制重跑，先备份结果，再删除对应 summary 记录或使用 `RESUME=0`。

## 方法解读

### TStruct
输入经过 sample normalization 与 patch embedding，得到 `[B,C,N,D]`。Temporal Structure Encoder 学习 K 个共享的 `N×N` 时间依赖矩阵，并通过变量静态先验 `channel_query` 与实例动态权重 `instance_query` 组合；组合后的时间结构对 patch 表示进行加权聚合。PCAHead 将共享线性预测映射与按变量适配的低秩映射相加。

### TimeBase
将历史序列按周期长度 P 重排为 `[B,C,P,N_x]`，在每个周期位置上将历史周期段数 `N_x` 映射到低维 basis，再映射到未来周期段数 `N_y`，最后反归一化并还原为未来序列。它不使用 Transformer 注意力，核心是周期分段与低维线性基映射。

## 本轮比较原则

- 先比较验证集，不根据测试集选模型。
- 统一 `features=M`、7 个变量、ETT 官方时间切分与训练集标准化。
- 固定 seed 2024，单种子结果只用于筛选候选，不宣称统计显著。
- `matched96` 的 Paloma 验证集基准来自本项目 v7：ETTh1-H96 0.655625/0.525773，ETTh1-H336 1.292065/0.747915，ETTm1-H96 0.352285/0.377593，ETTm1-H336 0.630800/0.513685（MSE/MAE）。
- `matched96` 与 `paper720` 分开汇报，不混合平均。
- 只有当模型在多个任务上表现出稳定的验证集优势，才进入下一阶段多随机种子复验与结构拆解。
