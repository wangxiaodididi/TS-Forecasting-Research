# XLinear Complementary v1 — 在LG-Short上改造TGM与趋势外推

## 研究动机

基于已完成实验：LG-Short在ETTh1-96及ETTh1-192提供MSE收益，但MAE通常未优于Official；LG-Dual在部分情形表现不稳定。这里不再次改动Embedding。仅研究两个**功能不重叠**的部位：

1. **TGM的低秩特征混合**：原TGM是x*sigmoid(MLP(x))，以逐坐标缩放为主。新增的低秩GELU变换在原TGM输出上做残差更新，使不同时间隐藏坐标可相互组合。保持原TGM、VGM、Head不变。
2. **近期趋势外推路径**：全局标准化和线性Head未显式为最近两个历史片段提供独立、直达未来预测的线性漂移通道。依据最近片段均值与前一片段均值构造水平/斜率特征，用2×H的零初始化权重给预测输出提供残差。不是CSRA的固定周期残差，也不是HCF的动态特征权重。

两者都是已有思想的轻量组合，不宣称方法原创或性能必然改善。它们针对模型结构上的不同路径。

## 接口与模型

- `official`：原始XLinear，无改造
- `lg_short`：上一轮有效的局部编码基线
- `lg_tmix_r16`：LG-Short + TGM低秩特征变换（rank=16）
- `lg_trend_w24`：LG-Short + 最近24步趋势残差
- `lg_combo_r16_w24`：同时使用以上两项，**只有单模块证据支持时才建议跑**
- `lg_tmix_r8` / `lg_trend_w12`：用于后续验证集指导的小范围参数检验，**不要按测试集择优**

所有新模块最后一层权重零初始化，初始输出与Official严格相同（LG-Short最后线性层也为零）。初始化的额外参数与Official的随机数流隔离。模块仅支持`features=M`。

## 使用前准备

需要你**现有**的 `/root/autodl-tmp/XLinear_ResearchLab_Clean` 项目，且已安装LG-Embedding v1（`lg_short`能够正常训练）。不需要另外建立新项目或重装PyTorch。正式训练沿用 `configs/experiment.yaml` 的原论文参数、train/val/test划分和评价脚本。`vendor`源码不更改。

在根目录解压压缩包，得到文件夹`XLinear_Complementary_v1`。

```bash
cd /root/autodl-tmp/XLinear_ResearchLab_Clean
unzip -o XLinear_Complementary_v1.zip -d .
bash XLinear_Complementary_v1/01_CPU_check.sh
bash XLinear_Complementary_v1/02_plan.sh
```

先独立比较两个模块，默认16次正式训练（ETTh1-96/192×2025/2026×4变体）：

```bash
bash XLinear_Complementary_v1/03_GPU_full.sh
```

希望先减小任务量，只跑ETTh1-96，Seed 2025：

```bash
HORIZONS=96 SEEDS=2025 bash XLinear_Complementary_v1/03_GPU_full.sh
```

若希望测试组合结构：

```bash
bash XLinear_Complementary_v1/04_GPU_combo.sh
```

优化参数前，建议先看验证集表现，再比较更小的rank和更短的recent_window；预设变体如下：

```bash
VARIANTS=official,lg_short,lg_tmix_r8,lg_tmix_r16,lg_trend_w12,lg_trend_w24 HORIZONS=96 SEEDS=2025 bash XLinear_Complementary_v1/03_GPU_full.sh
```

**注意：上面仍然会产生测试结果。不可根据测试集选择最优参数或事后声称无偏增益；参数选择应在验证集先固定后，再汇报未参与选择的测试结果。** 已多次查阅历史测试集，因此后续稿件还需要独立验证/新数据来支撑未经选择的泛化声明。

训练完成后，在日志末尾寻找 `结果目录` 或查看 `runs/` 新建的文件夹，比如：

```bash
bash XLinear_Complementary_v1/05_summary.sh runs/你的新运行目录
```

得到 `XLinear_Complementary_v1/outputs/paired.csv` 和 `summary.csv`；只报告同seed、同数据与预测长度的配对差异。

## 本地测试与实际训练的界限

本地只测试初始化一致性、反向梯度、形状兼容性及脚本计划。没有访问你的AutoDL GPU，**并未获得任何新MSE或MAE**。

## 参考与设计依据

- XLinear官方源码：`Zaiwen/XLinear/models/XLinear.py`，原始TGM为逐坐标乘性门控。
- 近期轻量时间预测研究强调局部与不同时间尺度特征的互补，但是不等于本实验组合已有效。
- 前次CSRA、HCF未获得稳定增量，所以这里不用周期残差，也不重复动态融合Head。

## ETTh1-96模型额外参数量（与此前Official参数301113对齐）

| 变体 | 参数数 | 说明 |
|---|---:|---|
| Official | 301,113 | 原论文主干 |
| LG-Short | 305,345 | +4,232 |
| LG-Short + TGM-r16 | 314,321 | 比LG-Short多8,976 |
| LG-Short + Trend-w24 | 305,537 | 比LG-Short多192 |
| LG-Short + 两者 | 314,513 | TGM-r16加趋势直达路径 |

上述数量依据原主干一致的CPU模型结构计算；实际服务器以checkpoint参数统计为准。
