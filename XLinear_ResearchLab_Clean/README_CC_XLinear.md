# CC-XLinear — 时间成分条件化的变量信息交互（局部补丁）

## 研究问题

时间结构不同的变量不一定应当在同一信息通道上被同等利用。假定变量之间在趋势、日周期及短期残差方面的相似性具有差异，能否利用这些**历史可观察的时间成分**，为 XLinear 的 VGM 提供样本条件化的跨变量信息路由？

**本模块是待验证的渐进式结构改造，不声称“首次提出条件化交互”。** 基于已有工作（例如 DLinear、CycleNet、xPatch）的成熟分解思想，创新点候选是将时间成分一致性映射到 XLinear 原有的跨变量传递路径；是否成立取决于实验结果。

## 具体结构

XLinear 官方主干不变：

```
X[B,L,C] -> 官方 RevIN -> 官方历史嵌入和 Global Token -> 原 TGM
                                                       |
                                             原 TGM 的 Global 路径 H
                                                       |
历史 X -> 历史成分处理 -> 源-目标成分相似度 S -------> P_ij
                                                       |
                          H' = H + 0.5 (P-U) H  <------+
                                                       |
                        原 VGM([原 Embedding, H']) -> 原 Head
```

- 只修改 TGM 到 VGM **之间**传递的信息，不改两个原门控模块及其参数。
- 低频趋势：历史窗口中的 25 点滑动平均，再压缩到 12 维，比较时间形状。
- 日周期：去趋势后按 24 小时对齐求平均相位模板；用周期内的相位形状与多日一致性形成加权相似性。
- 局部残差：最近一天偏离平均相位模板的变化，比较近期非周期信息。
- 相似性是源-目标**两两**形成的；对角元素排除，不会把自变量的信息传递假装成跨变量依赖。
- 路由使用 `softmax(peer_logits + a_trend*S_trend + a_cycle*S_cycle + a_irregular*S_irregular)`。
- `U` 是每行的均匀其他变量路由：`(P-U)H` 形式保证**所有新增参数为零时，初始输出与官方模型相同**，并且可学习系数的梯度在第一步即可产生。
- 所有处理只使用每个预测起点之前96步历史，且在作者的 RevIN 归一化之后。不会读取未来值或修改训练集。

## 配对实验与消融

| 名称 | 作用 | 额外参数 |
|---|---|---:|
| `official` | 完整作者模型，论文参数复现基线 | 0 |
| `cc_static` | 学习**与输入无关**的跨变量路由；近似容量对照 | 49 个实际参与前向的参数 + 3 个保留系数 |
| `cc_raw` | 输入条件化路由，但不做时间成分分离；比较分解是否必要 | 52 |
| `cc_components` | 趋势/日周期/局部残差三类历史成分条件化路由 | 52 |

`cc_static` 中保留的3个系数不参与前向，因而**有效**训练参数比 `cc_components` 少3个；含原主干在内的名义参数量则一致。报告时不可称三个改造版所有参数都有效使用。

**正式选择流程：**先看验证损失与对应模型选择；测试指标只报告预先固定的对照，不可反复用测试集选路径和参数。如果没有实质改善，报告失败，不将单次微小波动包装为稳定优势。

## 安装（VS Code Remote SSH 的远程终端）

将 `XLinear_CC_XLinear_Existing_Project_Patch.zip` 上传到当前项目根目录，执行：

```bash
cd /root/autodl-tmp/XLinear_ResearchLab_Clean
unzip -o XLinear_CC_XLinear_Existing_Project_Patch.zip -d .
python ops/install_cc_xlinear.py
python -m tests.test_cc_xlinear_smoke
```

安装器**只修改** `src/models/registry.py`，并备份为 `src/models/registry_before_cc_xlinear.py.bak`。不会覆盖 CSRA、Token、官方源码、训练入口或旧实验目录。

## 运行

```bash
# 检查官方参数和实验任务（不训练）
bash run.sh plan --datasets ETTh1 --horizons 96 --seeds 2025 --variants official,cc_static,cc_raw,cc_components

# 先进行 2 Epoch 工程测试（不用于学术结果）
bash run.sh pilot --datasets ETTh1 --horizons 96 --seeds 2025 --variants official,cc_static,cc_raw,cc_components

# 之后正式按论文训练协议对照（重新训练配对的官方基线；不覆盖旧结果）
bash run.sh full --datasets ETTh1 --horizons 96 --seeds 2025 --variants official,cc_static,cc_raw,cc_components
```

结果：`runs/本次任务/results/paired_comparison.csv` 与 `all_results.csv`。实时训练曲线仍使用已有 `bash run.sh dashboard`。

## 解释图（训练完成后，可选，只读，不重训）

```bash
python ops/explain_cc_xlinear.py --run 实际训练结果目录名称 --variant cc_components --split val
```

输出到 `runs/.../jobs/ETTh1_H96_cc_components_s2025/explanation/`：
- `routing_mean.png`: 目标变量对其他源变量的平均路由权重。
- `routing_variability.png`: 路由在不同验证窗口的波动程度。
- `routing.csv`: 两两变量路由系数及相对均匀参考值的差异。
- `summary.json`: 三类成分系数、矩阵偏离均匀路由的程度。

权重只是网络的信息路由系数，不能解释为因果重要性或显著性。

## 范围限制和预期

- v1 **仅支持 ETTh1 / 多变量 M / 96 步历史**；不会把小时24步周期错误套用于 ETTm1 或 Weather。
- 训练主线、正式模型原始参数、Adam、验证早停、指标计算均使用现有官方训练器。
- 新设计的作用在 VGM 的**输入信息源**，不是额外的预测输出头。
- CPU 自检只能证明形状、梯度、输出一致性，不能证明精度优势。RTX 5090 正式训练必须在你的服务器执行。
