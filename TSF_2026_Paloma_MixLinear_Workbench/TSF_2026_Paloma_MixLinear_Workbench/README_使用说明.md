# Paloma v8：基于验证集瓶颈的定向实验

本扩展包安装在 **原有内层工作台**（已经存在 `experiment.py`、`sources/Paloma`、`runs/paloma_functional_splice` 的目录），无需重新安装依赖、无需重复训练原始模型。所用 PyTorch 使用原 `.venv` 环境。

## 为什么不再继续盲目缝合

v7 的四任务验证集诊断表明：关闭 Input Skip 使 MSE 恶化 24.73%–98.80%，而关闭 Hidden 使 MSE 恶化 0.43%–13.83%；历史趋势强的窗口误差较高；各任务第4预测段/第1预测段误差比为 2.164–2.600；所谓高模板失配窗口没有一致的更高 MSE。**这些只是描述性证据，不证明因果瓶颈。**

另外，核对官方 `exp/exp_main.py`：训练损失是随预测距离按 `h^-alpha` 衰减的 WeightedLoss（MAE）；`vali()` 同样接受此 `WeightedLoss`，因此官方早停不是按最终 MSE。v8 对比训练目标时保留了**原始加权验证准则**用于早停，隔离“训练目标变化”。

## 本轮四种独立验证实验

| 变体 | 对应功能 | 训练损失 | 目的 |
| --- | --- | --- | --- |
| `uniform_l1` | 原始网络完全不变 | 均匀 MAE | 检查距离衰减训练目标是否影响平均/远期表现 |
| `trend_skip` | Input Skip 后方增加历史趋势残差修正 | 原作者 Weighted MAE | 检查趋势强时的预测残差是否可修正 |
| `structure_skip` | 借鉴 TStruct 实例条件结构选择，混合局部水平/趋势/周期参考预测 | 原作者 Weighted MAE | 检查输入直连是否需要更丰富的未来结构表示 |
| `trend_skip_uniform` | 同 `trend_skip`，独立改变训练损失 | 均匀 MAE | 检查训练目标与结构是否相互作用 |

注意：`trend_skip` 是定向、轻量、独立设计的趋势残差适配；`structure_skip` **只借鉴 TStruct 的实例条件化多结构思想**，不是论文中的原始 TSEncoder 或 PCAHead，不能在论文中标注为完整复现。

全部模块只基于可观察历史窗口工作，不使用未来标签。新增残差分支初始化为0，初始预测与官方原模型相同。官方 `models/Paloma.py` 只复制到本轮独立 `runs/paloma_validation_guided_v8/private_source/Paloma` 后追加兼容改动，原目录和旧检查点不变。

## 一键执行

将压缩包内容解压至 `experiment.py` 所在工作台目录：

```bash
unzip -o Paloma_Validation_Guided_v8.zip -d .
bash run_paloma_v8_all.sh
```

脚本依次计算**验证集** Last-Value 和 Seasonal-Naive 两种参照方法的分预测距离MSE，检查原始 Paloma 哈希与新增模型的前/反向，然后在 GPU 上执行：ETTh1、ETTm1 × H96、H336 × 4个变体，共 **16 组训练**。固定作者官方训练设置（含 batch size、学习率、dropout、epoch、seed2024）；除两个 `uniform_l1` 模型以外训练损失不变。程序自动跳过已完成任务，若某项失败需查看对应 `train.log`；重新运行默认跳过已成功项。

如果只想先看无需训练的季节性基准与预测距离变化：

```bash
python paloma_v8.py --mode diagnose
```

若只执行 CPU 确定性初始化、前向/反向检查：

```bash
bash run_paloma_v8_smoke.sh
```

若只恢复汇总 CSV：

```bash
bash run_paloma_v8_collect.sh
```

仅运行 ETTh1-H96 的四组实验：

```bash
bash run_paloma_v8_all.sh --datasets ETTh1 --horizons 96
```

## 数据、报告与评估协议

- 验证集基准：`runs/paloma_validation_guided_v8/v8_validation_naive_by_quartile.csv`；两种无需训练的参照用于判断绝对远期误差增加是不是普遍现象。
- 训练汇总：`runs/paloma_validation_guided_v8/v8_validation_summary.csv`。**全部 `mse`/`mae` 字段为验证集结果**，不读取测试标签作为新方案的评估依据。
- Original 验证集对照从已有 `runs/paloma_path_diagnostics_v7/all_tasks_summary.csv` 中读取 `split=val` 的 `original_recalculated_mse`。该文件缺失时依然可运行，但 `*_val_gain_pct` 留空，不能用测试 MSE 顶替。
- 每个训练目录包含 `metadata.json`、`train.log`、`checkpoints/`。
- 本轮私有 `exp_main.py` 修改了官方训练代码中两个 `flag='test'` 数据读取位置，让每轮所谓 `Test Loss` 的读数只是验证集的镜像；最终预测指标也从验证集计算。训练脚本不访问测试集。**训练与早停的官方指标仍使用原始 Weighted Loss；改造损失仅影响训练反向传播**。
- v7 给出的远期/近期比值，尚未证明模型对远期缺少相对技能；v8 要先比较 naive 误差趋势。后续是否测试固定的新模型，请基于验证集结果决定，并明确此前历次测试比较的选择偏差。

## 复现校验与限制

- 必须与上游 Paloma `def3dfc612bd6554be8f2ed7d5174848d6a37119` 对应的 `models/Paloma.py` Git Blob `03dc7c6d23751cde6acaca77c803117c89702006` 完全一致，否则直接拒绝修改。
- 本地已执行5项代码单元测试以及两个预测长度、两种周期设置的合成张量检查。未在本环境执行上游官方完整训练，需在 AutoDL `smoke` 后运行。
- `uniform_l1` 是训练目标对照，不是模型结构创新。任何改造表现是否支持论文贡献必须由独立后续实验检验，不能仅根据已反复观察的测试集取得的最优结果作结论。
