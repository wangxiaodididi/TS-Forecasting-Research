# LG-Embedding XLinear v2：正式扩展复现与结果汇总

本包**不改变神经网络结构**、不覆盖任何 `src/` 模型训练文件，不进行超参数搜索；复用你已装好的 LG-Embedding v1 和 XLinear_ResearchLab_Clean 官方训练器。

## 固定训练矩阵（21次新增任务）

| 分组 | 数据 | 预测长度 | Seeds | 变体 | 新任务数 |
|---|---|---:|---|---|---:|
| A | ETTh1 | 192 | 2025,2026 | official,lg_short,lg_dual | 6 |
| B | Weather | 96 | 2025,2026 | official,lg_short,lg_dual | 6 |
| C | ETTh1 | 96 | 2026,2027,2028 | official,lg_short,lg_dual | 9 |

此前 `runs/20261116_220929_full_feb4a7` 的 ETTh1-96 Seed2025 结果只用于汇总，**不重复训练**（如果服务器中存在该目录）。

官方配置取现有 `configs/experiment.yaml`：ETTh1-96使用`d_model=128,t_ff=512,batch=128,lr=0.0005,epochs<=30,patience=10`；ETTh1-192使用`t_ff=1024,head_dropout=0.3,patience=5`；Weather-96使用`d_model=256,t_ff=256,batch=32,epochs<=20,patience=10`。训练脚本本身读取这些参数，不在本补丁写死。

## 在服务器执行

将ZIP解压到 `/root/autodl-tmp/XLinear_ResearchLab_Clean` 根目录（`LG_Replication_v2/` 文件夹位于根目录下）。先进入项目并激活**此前XLinear正式运行成功的Python环境**：

```bash
cd /root/autodl-tmp/XLinear_ResearchLab_Clean
unzip -o XLinear_LG_Replication_v2.zip -d .
export PYTHON=$(command -v python)
bash LG_Replication_v2/01_预览实验矩阵.sh
```

需要直接执行全部21次正式训练：

```bash
bash LG_Replication_v2/02_正式扩展实验.sh
```

建议分组进行，分别产生独立 `runs/` 目录，并在每组完成后写入 `LG_Replication_v2/state/new_runs.jsonl`：

```bash
bash LG_Replication_v2/02_正式扩展实验.sh --groups ett_192
bash LG_Replication_v2/02_正式扩展实验.sh --groups weather_96
bash LG_Replication_v2/02_正式扩展实验.sh --groups ett_96_seeds
```

运行程序是**同步执行**的：当前终端在训练完成前不会结束；没有启动后台任务。现有GPU运行器继续使用原有文件锁，防止重复GPU训练。已经完成的实验组在再次调用时自动跳过，`--force`则强制重跑整个组，慎用。

全部完成后汇总：

```bash
bash LG_Replication_v2/03_汇总.sh
```

结果：
- `LG_Replication_v2/outputs/all_selected_results.csv`：完整选择矩阵与来源。
- `LG_Replication_v2/outputs/paired_deltas.csv`：同数据集、预测期限、种子下的MSE/MAE改进率。
- `LG_Replication_v2/outputs/group_summary.csv`：每个数据/期限的各指标均值、赢过Official的种子数。
- `LG_Replication_v2/outputs/report.md`：中文对照报告，包含缺失任务提示。

如果训练尚未完全结束，希望只看已经**整组完成**的配对，可用：

```bash
bash LG_Replication_v2/03_汇总.sh --allow-partial
```

注意：尚未整组完成的组不会被纳入临时汇总。完整汇总默认必须含历史Seed2025和新增21次训练；否则报出缺失任务，不编造结果。如果历史运行目录名称不同，可以指定 `--reference-run <已存在run目录名>`。

## 严格比较要求

- MSE与MAE都要报告，不以单一MSE改善宣称全面优势。
- LG-Short与LG-Dual为之前已经确定的结构，这一轮不修改核大小、通道数、学习率或其他超参数。
- 使用完全一致的预测期限、数据划分、seed和训练器。代码强制验证相同seed的profile、源码commit、数据SHA256和精度一致。
- 先前的ETTh1-96 Seed2025已经参与过选择，所以本次新增的2026–2028是更重要的稳定性证据；这不是全新独立测试集。
- Weather和ETTh1分别比较，不直接平均两数据集的原始MSE。

## 尚未在本环境执行的部分

本包可以在CPU上验证计划和结果汇总逻辑，但没有对AutoDL远程GPU的直接访问，因此不会声称21次训练已经完成。服务器运行后，以生成的CSV与训练日志为准。
