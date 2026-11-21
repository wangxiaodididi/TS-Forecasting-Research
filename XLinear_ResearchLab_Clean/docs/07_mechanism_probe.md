# XLinear 机制诊断 v1：在现有项目内加功能，不增加另一个研究项目

本包只新增 `src/training/mechanism_probe.py`、`ops/install_probe.py` 和本说明。
**不会替换 `run.sh`、官方源码、训练器，也不会覆盖 checkpoint 或旧 CSV。**

## 安装

在 AutoDL 服务器 `/root/autodl-tmp/XLinear_ResearchLab_Clean/` 中：

```bash
unzip -o XLinear_Mechanism_Probe_v1.zip
python ops/install_probe.py
bash run.sh probe --run 20261115_190651_full_d2e123 --splits val
```

也可不修改 `run.sh`，直接用 `python -m src.training.mechanism_probe --run 20261115_190651_full_d2e123 --splits val`。
`--run` 可省略，默认自动选取最新且含 Official 对照的完整实验。
如 CPU 运算较慢，默认自动使用 CUDA；可手动 `--device cpu`。`--max-batches 2` 只用于工程测试。

## 输出路径

`runs/<本次run>/analysis/mechanism_probe/ETTh1_H96_s2025/`

- `REPORT.md`：中文结论说明 + 数值报告
- `sample_level.csv`：逐预测窗口误差、FFT 频率特征、TGM/VGM 门控、Token 反事实结果
- `counterfactual_summary.csv`：开启/关闭新增 Token 分支的 MSE 和预测变化幅度
- `frequency_groups.csv`：慢速/中频/高速峰值样本的各模型误差，保证模型间使用同一组窗口
- `gate_frequency_correlations.csv`：频率特征与潜在门控的 Spearman ρ；**不做独立样本显著性声明**
- `plots/*.png`：上述结果的简洁图表

## 分析口径

1. 分析以官方数据集 scaler 处理后的原始历史输入为基础，所有频率指标都不使用未来信息。
2. 验证集数据顺序被改为 `shuffle=False`（只读推理），且保持官方测试的 `drop_last=True` 对齐。
3. `mse_all` 由所有预测变量计算，`mse_ot` 为数据的最后一列；不混为同一个评价指标。
4. 在同一 **已经训练好的** `dyn_token`/`stat_token` 权重中，把新增 Token 分支的 `gain_logit` 临时设为零。它是训练后局部干预，不是删掉分支后重新训练的消融。
5. 将历史 96 步的去趋势频谱按峰值 FFT k=1–2（slow）、k=3–7（middle）、k>=8（fast）分组。这只是研究诊断，不代表统计检验已经确认机制。
6. 只有历史窗口至少覆盖 24 小时时，才输出 `daily_energy_share`；如 Weather (10min ×96) 不足以辨识24小时周期，就不作该解释。
7. 单组窗口高度重叠，不能当作独立样本进行显著性检验。优先在验证集诊断并提出假设；不要反复使用测试集挑选结构。
8. 如需检查测试集一致性：`bash run.sh probe --splits test`。程序会把 checkpoint 的重推预测与原训练器保存的 `pred.npy` 逐批核验，不一致就报错停止。

## 注意事项

- 需要作者官方源码、公开数据和当时训练的 `checkpoint.pth`。不能从本补丁恢复已删除的权重。
- 不修改官方原模型，也不占用训练锁；运行期间如有 GPU 训练，建议指定 `--device cpu` 或在训练完成后执行。
- 分析全部已有的同数据集、预测长度、随机种子下含 Official 的对照模型；不会跨实验条件拼接数据。
- 此诊断只为定位改造方向，不把频率特征和门控的相关性直接解释为模型缺陷或因果效应。
