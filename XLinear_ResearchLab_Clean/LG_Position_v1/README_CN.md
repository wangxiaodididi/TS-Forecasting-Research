# LG-Position：XLinear局部编码顺序保留改造（v1）

## 研究依据

已完成的LG-Short用共享的短卷积 `Conv1d(k=3, out_channels=2)` 提取局部形态，然后通过8个**有序**时间区间的均值和最大值形成32维统计向量。它实际上已经保留了区间之间的粗位置，**没有**显式保留每个区间内峰值/形态的发生先后。

本版只改变一个非参数操作：

- `lg_short`：每个有序区间提取 `(mean, maximum)`；
- `lg_position`：每个有序区间再等分前/后两半，提取 `(mean, second_half_mean - first_half_mean)`。

两者相同的地方：输入按原官方XLinear归一化；全局 `Linear(96,D)` 不动；卷积核与通道数不变；区间数8不变；残差Linear输入宽度与输出宽度相同；输出仍为 `global_embedding + residual`；原TGM、VGM及Head不变。额外参数量相等，ETTh1-96、D=128时均为4,232（相对Official）。

原LG-Short的非对称卷积核仍可能隐式学到方向信息；这里改变的是池化摘要能否显式保留前后变化。LG-Position仅保留**区间内部的前后方向**，不是无损保存所有时间点位置。把均值/最大值替换为均值/斜率，也意味着可能损失峰值强度。效果优劣必须由正式训练判断。

## 接入步骤（无需替换已有LG版本）

在AutoDL项目根目录解压此ZIP（**保留原来的LG-Embedding v1文件**）：

```bash
cd /root/autodl-tmp/XLinear_ResearchLab_Clean
unzip -o XLinear_LG_Position_v1.zip -d .
```

### 1. CPU结构自检，不使用GPU

```bash
bash LG_Position_v1/01_CPU_check.sh
```

检查包括：按区间反转信号，LG-Position能够区别前后方向；参数量与LG-Short相同；两者参数在相同随机种子下初值一致；零初始化后与Official一致；新增分支反向传播；若服务器已有官方源码，进一步检查官方模型的前向兼容性。不存在vendor源码时会明确显示SKIP官方集成测试。

### 2. 查看正式运行计划

```bash
bash LG_Position_v1/02_GPU_plan.sh
```

### 3. 正式GPU训练（同一训练协议）

```bash
bash LG_Position_v1/03_GPU_full.sh
```

默认配置：`ETTh1`、`H=96,192`、`Seeds=2025,2026`、`Official,lg_short,lg_position` 共12项训练；官方项目保存新的 `runs/<run_id>/results/all_results.csv`。

可指定不同实验组合，例如Weather-96：

```bash
DATASETS=Weather HORIZONS=96 SEEDS=2025,2026 \
  bash LG_Position_v1/03_GPU_full.sh
```

如需仅跑96步：

```bash
HORIZONS=96 SEEDS=2025,2026,2027,2028 \
  bash LG_Position_v1/03_GPU_full.sh
```

### 4. 读取结果、同种子配对汇总

```bash
python LG_Position_v1/scripts/summarize.py \
  runs/<实际运行目录>/results/all_results.csv
```

输出至 `LG_Position_v1/outputs/<run_id>/`，包含 `paired.csv`、`summary.csv`、`report.md`，计算MSE与MAE、两种模型的胜出种子数，并拒绝不完整配对。

### 选做：CPU小样本端到端训练

服务器已经有官方源码和公开数据时：

```bash
bash LG_Position_v1/04_CPU_mini.sh
```

该脚本调用原LG v1的 `ops/cpu_mini.py`，**仅使用训练与验证区间**做2轮小规模工程测试，不用于论文评价。

## 安装安全性与运行环境

- `LG_Position_v1/scripts/install.py` 只为新变体追加独立注册段，不覆盖LG原有实现；首次修改前自动备份 `src/models/registry.py`，重复执行不会重复追加。
- 新代码是 `src/models/extensions/position_local_embedding.py`，依赖原有 `local_global_embedding.py`。不修改YAML、原作者源码、旧checkpoint及已有训练结果。
- 指定此前官方复现用的解释器：`PYTHON=/path/to/python bash LG_Position_v1/03_GPU_full.sh`。
- 安装注册后需要撤销：`python LG_Position_v1/scripts/install.py --remove`（不删除新模型源文件）。
- 正式实验依赖原项目 `run.sh`、`configs/experiment.yaml` 和作者固定版本源码。不要复制到新的独立空目录运行。

## 当前结论的边界

至本次打包，已完成本地CPU结构测试、项目注册和训练计划检查；**没有真实AutoDL GPU MSE/MAE**。旧ETTh1/Weather测试数据已多轮参与方案开发，所以这些测试结果后续只能作为开发性比较；如需投稿级别的无偏模型选择评价，应冻结方案后使用独立未参与设计的数据或时间段。
