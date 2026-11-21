# 代码模块与 XLinear Forward 逐步解读

## 1. 原始多变量分支（仅 features=M）

输入形状 `[B,L,C]`，L=历史长度 96，C=变量数（ETTh1=7、Weather=21）。

1. `Model.forcast_multi`：如 `usenorm=1`，逐样本、逐变量计算历史窗口均值/标准差并标准化。**只用历史输入**。
2. 转置成 `[B,C,L]`，交给原作者 `Forcast_multi`。
3. `projection: Linear(L,D) + Dropout`：逐变量将整段历史映射为 `[B,C,D]`。
4. `glob_token`：原作者可学习参数 `[1,C,D]`；扩展为 `[B,C,D]`。
5. `en_attention (TGM)`：拼接 `[emb,token]` 得到 `[B,C,2D]`，门控输出 `endogenous[D] + global_time[D]`。
6. `ex_attention (VGM)`：在变量维拼接 `emb` 和 `global_time` 得到 `[B,2C,D]`，转为 `[B,D,2C]` 后用变量门控，抽取全局信息 `[B,D,C]`。
7. 拼接原 TGM 特征与 VGM 输出得到 `[B,C,2D]`。
8. `head: Dropout + Linear(2D,H)`：输出 `[B,C,H]`，转置为 `[B,H,C]`。
9. `Model.forcast_multi` 再按原历史窗口统计量反标准化。

此模型并非 Transformer Attention。作者 `Gating_Block` 实际为 `Linear -> ReLU -> Dropout -> Linear -> Sigmoid` 与输入的**逐元素乘积**。

## 2. 现有两个模块具体改在哪里

仅替换第4步的全局 token：

```text
Global Token_new = Original Global Token + tanh(gain) × context_projection(recent_history)
```

- `dyn_token`：最近 K 个历史观测点映射到 D 维。
- `stat_token`：最近 K 期的均值、总体标准差与首末差映射到 D 维。
- `gain_logit` 初始化为 0：模型初始预测严格等于原始 XLinear，便于公平对照；新增分支学习过程不会因此保证最终性能提升。
- `context_ratio` 在统一 YAML 文件设置；默认 0.25，即 L=96 时 K=24。

重要：TGM、VGM、原始 Linear Embedding、Prediction Head、损失 MSE、训练器不做结构变更。

## 3. 优先阅读顺序

1. `vendor/XLinear/models/XLinear.py` —— 作者真正的 `Model`、`Forcast_multi`、`Gating_Block`，请先读 forward。
2. `src/models/extensions/global_token.py` —— 我们新增的结构及维度注释。
3. `src/models/registry.py` —— 模型注册与 RNG 隔离。
4. `src/training/official_entry.py` —— 将变体以进程内方式接入官方 Model。
5. `vendor/XLinear/exp/exp_main.py` —— 官方 optimizer / early stopping / test。
6. `src/training/runner.py` —— 不修改算法的任务编排、日志与输出隔离。
