# 06. 后续拆解与修改 XLinear 的工作流程

## 最重要的原则

每次只改变**一个信息传递机制**，不要将时间门控、变量门控、预测头、损失函数一起修改。原始官方源码位于 `vendor/XLinear/` 并有哈希校验，保留只读；新模块位于 `src/models/extensions/`。

原始多变量分支从 `[B,L,C]` 输入，经历史窗口归一化，转成 `[B,C,L]`，用时间投影形成 `[B,C,D]`，再执行：

1. **Global Token**：提供可共享的模型表征，原版为可学习参数（不随样本变化）。
2. **Temporal Gating Module (TGM)**：`en_attention`，让历史编码与共享 Token 信息进行特征筛选。
3. **Variate-wise Gating Module (VGM)**：`ex_attention`，变量之间的门控交互；其输入会进行跨变量维度拼接。
4. **预测头**：由每个变量的 `2D` 表征映射到未来 H 步。

当前 `dyn_token` / `stat_token` 只改变 Global Token 的构造方式，原 TGM、VGM、Head 没有替换。两版通过初始 gain=0 对齐官方初始前向输出，且使用相同优化配置。

## 推荐的代码位置

| 研究问题 | 要检查的源码/图 |
|---|---|
| 为什么长预测长度改造效果减弱？ | `diagnostics/horizon_errors.csv`、预测样例 |
| 模型是否过拟合？ | `metrics/epochs.csv` 中验证误差最低点和训练误差变化 |
| 某次实验是否梯度异常？ | Batch Loss 曲线，`module_diagnostics.json` 单批 gradient L2 |
| TGM/VGM 门是否饱和？ | `module_diagnostics.json` 的权重均值、直方图及极端权重占比 |
| 改造是否只对某个随机种子有效？ | `results/paired_comparison.csv` 和 `summary.csv` |
| 要增加新模块？ | 新建 `src/models/extensions/*.py`，在 `src/models/registry.py` 注册，配置 `experiment.variants` |

新增变体建议使用可切换的 `enable` 或归零初始化来验证“旧模型输出 == 新模型初始化输出”，然后跑 `bash run.sh check` 和小样本预测。仅改变模型，不改变训练器参数。**不能根据正式测试集结果反复选择最好的模型再当作未经选择的结果报告。**

梯度与门控快照是帮助理解 forward 和模块设计的工具，不提供经济学或物理机制解释。模型结构创新仍需要提出可检验假设，并有稳健实证支持。
