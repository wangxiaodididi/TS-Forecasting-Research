# Paloma v2：相位调制定向替换实验

本扩展基于已跑通的 Paloma（NeurIPS 2026）官方源码，在**不修改上游目录**的前提下复制一份私有模型，并只替换相位角 `theta` 的生成方式。保持周期 Q、幅度分支、输入直连、标准训练脚本和数据划分不变。

## 原因与假设

基于 ETTh1 (L96/H96) 原模型 MSE 0.360570 / MAE 0.378699 的消融，删去相位分支 MSE 上升 0.456%，但 MAE 下降 0.130%，因此静态周期相位旋转可能不是所有样本都适合。**这是待验证的假设，不是已有发现。**

## 三个配置

- `original`：**完全保留**官方 PhaseModulation 以及官方全部初始参数（严格逐元素验证）。
- `bounded_phase`：将旋转角绝对范围从 π 缩小至 π/2，保持参数量不变；检验是否需要更强的方向约束。
- `content_phase`：在现有 `phase_to_angle(phase)` 上加入 `Linear(residual)` 条件修正，权重全部零初始化；训练初始输出与官方一致，并且所有原模型参数的初始值保持一致；检验样本特异残差能否改善固定条件相位的限制。

这些是**本地研究性替换件**，不是对某篇论文模块的完整复现，更不能事先宣称效果更好。

## 安装

将 ZIP 上传到含有 `experiment.py`、`run_full.sh`、`sources/Paloma` 的**工作台根目录**，运行：

```bash
unzip -o Paloma_Phase_Replacement_v2.zip -d .
ls -l experiment.py paloma_phase_replacement.py run_paloma_phase_replacement.sh
```

在原来的 `.venv` 终端下：

```bash
# 1. 不用 GPU；验证原官方模型与扩展 original 严格一致、三个变体梯度正常
bash run_paloma_phase_replacement.sh --mode smoke --datasets ETTh1 --horizons 96,336

# 2. 只训练两种新增替换件。默认会尝试读取 v1 的 original 作为对照。
bash run_paloma_phase_replacement.sh --mode full --datasets ETTh1 --horizons 96 \
  --variants bounded_phase,content_phase --profile paper

# 3. 有初步信号后再到另一数据集 ETTm1 验证
bash run_paloma_phase_replacement.sh --mode full --datasets ETTm1 --horizons 96 \
  --variants original,bounded_phase,content_phase --profile paper
```

## 结果

每个变体保存在 `runs/paloma_phase_replacement/paper/<数据集>/h<步长>/<变体>/`，含 `train.log` 和 `metadata.json`。

结果汇总：`runs/paloma_phase_replacement/phase_replacement_summary.csv`。

`mse_gain_pct_vs_original` / `mae_gain_pct_vs_original` 为**正数表示性能改善**；本扩展优先读本轮 `original`，否则尝试读取 `runs/paloma_ablation/paper/.../original/metadata.json`。比较来源在 `baseline_source` 中明确标出，不会冒充同次运行。不同数据集或不同 H 的 baseline 不交叉引用。

## 研究注意事项

1. 初步试验只用于筛选方案，正式论文需要固定候选、独立验证及多随机种子；不能持续按测试集结果挑最好的设置。
2. 修改 `phase_fusion` 时只引入一个额外的零初始化矩阵，不更换整个 Backbone。
3. 运行之前查看 `sources/Paloma` 已按原始 SHA 固定；未安装时先执行 `bash setup_sources.sh`。
4. 如果此前工作台目录解压成双层，必须进入包含 `experiment.py` 的**内层**运行和解压。
5. 原始训练脚本会输出最终测试 MSE 和 MAE，但仍可能在每个训练 epoch 日志输出测试 loss；本扩展没有更改该行为。正式选模应只用验证集。
