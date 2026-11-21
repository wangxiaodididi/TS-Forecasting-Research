# Paloma v4：位置与模块的实证区分

## 源码信息流

`X [B,L,C]` → 实例标准化 → `Xc [B,C,L]` → 索引 `Q, phase, amplitude` → `R=Xc-Q` → `PhaseModulation(R,phase)` + `ConditionalFusion(R,amplitude)` → `Linear(D,D)+GELU` → `+ input_proj(Xc)` → `Linear(D,H)` → 反标准化。

基准 P=Paloma，L=96，C=7，D=256，H∈{96,336}。`x`、`Q` 使用 RevIN 后同一表示空间。原 Paloma 没有显式跨变量混合，原始融合是两个分支直接相加。

## 1. olinear_residual（跨变量信息进入周期残差）

`R'=R+γ M_96(R)`，其中 `M_F(R)=row_normalize(softplus(A))@Linear_F(R)`，输入输出均为 `[B,C,F]`。

**意图**：变量共同扰动是否对 Phase/Amplitude 的残差表征有增益？

**风险**：周期模板 Q 是逐变量的；跨变量混合后两个分支共用修改后的 R，可能破坏原始 Q 对单变量残差的语义。

## 2. olinear_hidden（跨变量信息进入潜在特征）

`H=GELU(W(Phase(R)+Amplitude(R)))`，再 `H'=H+γM_256(H)`。

**意图**：保留原始周期残差的语义，直到两个分支编码后再跨变量融合。

**风险**：这个操作比 96维输入 Mixer 参数更多，需要结合成本和提升幅度判断。

## 3. lagged_residual（显式因果时滞）

每个目标变量 `i` 根据源变量 `j` 在滞后 `k∈{0,1,2,4,8}` 的残差加权：

`Z_i(t) = ∑_{k,j} α_(i,k,j) R_j(t-k)`，`α` 由训练得到的 softmax 权重定义。`R_j(t-k)` 只使用历史窗口内在 t 或之前的数据，不访问未来。

`R'=R+γZ`，`γ(0)=0`。

**意图**：相较仅同步混合的 C×C 矩阵，该模块能利用有限的历史滞后信息。

**风险**：固定滞后集合并不自动等于识别 Granger 因果；5个滞后不是经过统计验证的最佳值，不能作为因果解释。

## 4. adaptive_fusion（相位/幅度分支的样本条件融合）

`P=PhaseModulation(R,phase)`，`A=ConditionalFusion(R,amplitude)`。令 `g=tanh_like(MLP([P,A]))=2 sigmoid(MLP([P,A]))-1`，

`Z=(P+A)+g ⊙ (P-A)`。

低秩 MLP 的最后一层置零，因此 `g(0)=0` 时 `Z=P+A` 与原始前向完全相同。

**意图**：原模型只能固定相加两条分支，现在对不同样本和变量产生不同的特征融合权重；与近期的条件特征融合思想对应，但属于独立实现。

**风险**：前两轮单分支消融增益较小，融合门也可能过拟合，尤其 ETTh1 只有7个变量。

## 验证判据

同协议、同数据、同 seed 的复现基准比较 MSE 和 MAE。只在开发阶段 ETTh1-H96 筛选，随后将确定的结构扩展到其他任务，不把测试集最好一次的结果称为独立验证。记录参数数、训练耗时与失败的变体。比较时应区分模型变化（真实结构调整）和训练协议变化（不能同时调整）。
