# TimeMixer PDM Audit v1 — 源码、训练输入与已训练隐藏表示的结构诊断

本项目是独立 `TVP_Standalone_TimeMixer_v2` 的**只读诊断补丁**，用于决定是否值得修改 PDM。所有文件名为 ASCII。不会修改训练权重，不会重新训练 TimeMixer，不会使用测试集做结构选择。

## 运行

把压缩包**直接解压到原有 `TVP_Standalone_TimeMixer_v2` 项目根目录**（不是建在下一级目录）：

```bash
unzip -o TimeMixer_PDM_Audit_v1.zip -d .
bash 19_run_pdm_audit.sh
```

如果想针对已有的 ETTm1-192 权重检查：

```bash
DATASET=ETTm1 HORIZON=192 AUDIT_WINDOWS=192 bash 19_run_pdm_audit.sh
```

如果只需训练输入层面的 CPU 分解检查（不依赖模型权重）：

```bash
python 17_pdm_data_audit.py --data-dir data --out results/pdm_audit
```

单独使用完整训练模型进行检查：

```bash
python 18_pdm_checkpoint_audit.py --dataset ETTm1 --horizon 96 --seed 2024 \
  --max-windows 192 --device cpu --out results/pdm_audit
```

默认定位以下真实文件（不依赖旧项目）：

- 官方源码 `training/workspace/official/models/TimeMixer.py`
- 原版训练参数 `training/runs/ETTm1/h96/seed2024/original/train_config.json`
- 训练权重 `training/checkpoints/**/checkpoint.pth`，要求唯一匹配 ETTm1, TimeMixer, H=96, seed 2024
- 已有数据 `data/ETTm1.csv` 和 `data/ETTh1.csv`

如路径不同，`18_pdm_checkpoint_audit.py` 支持 `--official-root`、`--checkpoint`、`--config-json`、`--data-dir`，均可显式指定。

## 输出文件

- `results/pdm_audit/source_audit.json`：核对固定 commit 和模型 blob SHA，检查 PDM 已有残差、不同尺度模块、规范化路径。
- `results/pdm_audit/input_decomposition.csv`：训练集真实输入数据上的分解代理指标（不是 PDM 隐藏表示）。
- `results/pdm_audit/pdm_hidden_stats.csv`：已训练 PDM 每层、每尺度的分解比例、混合增量与残差更新幅度。
- `results/pdm_audit/frozen_interventions.csv`：在验证集冻结模型，仅移除 seasonal mixing 或 trend mixing 的**敏感性测试**。
- `results/pdm_audit/checkpoint_audit_manifest.json`：来源、版本、数据集、预测期限和窗口范围。

**注意：** `frozen_interventions.csv` 不是已重新训练的 ablation。临时切断训练好的模块会导致分布变化，其验证 MSE 不能直接表明完整模型训练后表现。

## 第一轮已在提供的 ETTm1 / ETTh1 数据上完成的训练输入代理实验

训练段：ETTm1 前 34560 个观测，ETTh1 前 8640 个观测；在各自训练段中抽取 384 个完整96步输入。使用训练段均值/方差标准化后进行平均池化，精确复现官方移动平均的边界重复填充。

固定核大小 k=25，序列尺度长度分别为96、48、24、12：

- 96步：约6.5%的卷积窗口权重来自边界填充；
- 48步：约13.0%；
- 24步：约26.0%；
- 12步：约52.0%；并且所有位置都受边界填充影响。

替代核定义 `25, 13, 7, 3` 只是“保持近似原始时间跨度”的**对照假设**，不保证比固定25更合理。当前不运行任何“改造模型”的正式预测实验。

## 源码级明确事实

官方固定 commit `e24610583b36fdd8c76cc17a8df4e65759a5f460`；原模型文件 git blob SHA `ad1847df86646848cdfecab568eb49f7099ef1ce`。

- PDM 层内对每个尺度分解为 seasonal/trend，Seasonal 从细尺度向粗尺度混合，Trend 从粗尺度向细尺度混合。
- CI=1 时 PDM 输出是 `ori + out_cross_layer(out_season+out_trend)`，已有隐藏特征残差，不应再添加形式相同的补丁。
- 该版本源码定义了 `layer_norm` 和 `dropout`，但当前 `PastDecomposableMixing.forward` 不调用它们：这是代码事实，不等于性能缺陷。
- FMM `predict_layers` 对不同尺度已采用不同的时间映射，因此“完全缺乏尺度专属参数”并不成立。

## 需要做什么决定？

必须把**输入代理事实**和**已训练隐藏表示证据**结合：若粗尺度分解确有异常且其模块在冻结验证中确有可辨认影响，才进入小范围、固定协议的分解核对照训练。若证据不支持，停止这条 PDM 改造路线，不继续盲目替换。

论文参考：官方 [TimeMixer ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/a7ac8a21e5a27e7ab31a5f42a0117bdb-Abstract-Conference.html)，源码 https://github.com/kwuking/TimeMixer/tree/e24610583b36fdd8c76cc17a8df4e65759a5f460。
