# Paloma 模块拆解及候选功能迁移（v3）

## A. 原始计算路径（Paloma 2026）

源代码：[TSTS13/Paloma/models/Paloma.py](https://github.com/TSTS13/Paloma/blob/main/models/Paloma.py)。固定源版本：`def3dfc612bd6554be8f2ed7d5174848d6a37119`。

1. 标准化：`x[B,L,C] -> z[B,L,C]`，均值和方差沿时间维、每个样本变量独立统计。
2. 转置：`z[B,C,L]`。
3. 周期表：`Q`、`phase`、`amplitude` 均为 `[period,C]`。索引由 `cycle_index` 和时间位置决定，生成 `[B,C,L]`。
4. 残差：`R=z-Q`。
5. Phase：两个 Linear(L,D) 投影、正交化、周期角控制的三角混合、LayerNorm + shortcut，输出 `[B,C,D]`。
6. Amplitude：Linear(L,D) 内容表示，周期参数生成 scale/shift，输出 `[B,C,D]`。
7. 融合：`hidden=Linear+GELU(phase_impact + amplitude_impact)`。
8. Skip：`input_proj(z)`；与 hidden 相加，经过 dropout 和 Linear(D,H)，最后反标准化。

原模型无显式跨变量注意力或矩阵乘法；所有 Linear 在 C 维共享参数。`query_proj` 定义但没进入 forward。

## B. TQNet-inspired channel branch

来源：[ACAT-SCUT/TQNet/models/TQNet.py](https://github.com/ACAT-SCUT/TQNet/blob/master/models/TQNet.py)。

- TQNet 中的 `channelAggregator = MultiheadAttention(embed_dim=seq_len, num_heads=4, batch_first=True, dropout=0.5)`；`query` 使用 learnable TemporalQuery，`key/value` 使用输入序列。
- 本实验采用单独的可学习周期查询表 `temporal_query[period,C]`（不共享 Paloma Q）、将历史序列看作 C 个长度为 L 的 token。
- 接口 `[B,C,L] -> [B,C,L]`；改动只位于 input skip，得到 `input_proj(z + gate*TQ(z))`，不改 Phase / Amplitude。
- 风险：ETTh1 变量数仅 7，注意力可能增加不必要的参数并过拟合。

## C. OLinear-inspired normalized channel matrix

来源：[jackyue1994/OLinear/layers/Transformer_EncDec.py](https://github.com/jackyue1994/OLinear/blob/master/layers/Transformer_EncDec.py)，类 `LinearEncoder`。

- 该类沿变量 token 维使用 `A = normalize(softplus(weight_mat),p=1,dim=-1)`，令 `A @ value` 实现显式跨变量混合。
- 本实验独立实现一个更小的研究性版本：`A[C,C] @ value_proj(z)`，输出 `[B,C,L]`，仍然注入 input skip。
- 与 TQ 使用**同一插入点**，因此是同功能竞争零件而非首轮叠加组合。
- 这里的 `olinear_channel` 并不是 `newLinear.py`，后者是另一个线性层；也不是原模型完整 NormLin。
- 风险：固定的全局变量混合关系无法表示随时间变化的相关性。

## D. MixLinear-inspired local residual filtering

来源：[aitianma/MixLinear/models/MixLinear.py](https://github.com/aitianma/MixLinear/blob/main/models/MixLinear.py)。

- MixLinear 在时域与频域分支之前使用 Conv1d 提取局部变化，并和原输入相加。
- 本实验仅迁移“局部滤波+残差连接”思想：`R' = R + gate*Conv1d(R)`，使用序列长度保持不变的奇数卷积核（25）。
- 接口 `[B,C,L] -> [B,C,L]`，只影响 Phase 和 Amplitude 读取的残差，不改变 Q、Phase 参数表和 input skip。
- 风险：原相位调制需要逐位置保留周期信息；卷积若过度平滑可能削弱尖峰与相位结构。

## E. 首轮判断

不因一次 0.01% MSE 改善宣称有效。四组模型均使用同一个 Paloma 作者参数表，先对 ETTh1-H96 比较 MSE、MAE、训练耗时与参数量；如果趋势明显，再检查 ETTh1-H336 与 ETTm1。TQ 和 OLinear 竞争同一个功能，若后者相近优先较小的模型。LocalConv 与两者互补，但必须等单模块试验后才讨论组合。

原始总参数约 431,032；TQ = 468,449；OLinear Channel = 440,394；LocalConv = 431,058（ETTh1 L96,H96,C7）。
