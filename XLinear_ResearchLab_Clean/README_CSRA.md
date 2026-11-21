# CSRA 单模块改造 —— 原项目局部补丁

**目的：直接进入模型改造与训练，不继续追加诊断。**

- 原始 XLinear 的 Embedding、Global Token、TGM、VGM、Head 完全不变。
- 新增 `csra` 仅在模型输出端执行可学习的、按变量/预测段分配的周期残差修正。
- 只针对 **ETTh1、96步历史、24小时周期**；预测长度可以是96或192，但须有官方训练参数。
- 严格使用当前 `run.sh` 调用的**原论文训练器与参数**，避免再次出现基线不一致。
- 修正系数全为0时，新增模型初始预测与 Official 完全一致；参数可从第一轮反传开始学习。
- 三种基函数：最后1天周期模板与预测差、最近4天平均模板与预测差、前后两天同相位变化。
- 每个变量和每个24步预测块分别学习三个系数，用 `tanh` 限定幅度。
- 不对未来真实值进行特征构建；不更改训练集、验证集和测试集的划分。

## 安装（在 AutoDL VS Code 远程终端）

上传 `XLinear_CSRA_Existing_Project_Patch.zip` 至
`/root/autodl-tmp/XLinear_ResearchLab_Clean/`，在项目根目录执行：

```bash
unzip -o XLinear_CSRA_Existing_Project_Patch.zip -d .
python ops/install_csra.py
python -m tests.test_csra_smoke
```

## 实验命令

```bash
# 只查看计划
bash run.sh plan --variants official,csra --datasets ETTh1 --horizons 96 --seeds 2025

# 2 Epoch 工程测试（不能用于学术比较）
bash run.sh pilot --variants official,csra --datasets ETTh1 --horizons 96 --seeds 2025

# 官方完整协议 GPU 配对比较
bash run.sh full --variants official,csra --datasets ETTh1 --horizons 96 --seeds 2025
```

保持旧实验文件夹不动。`full` 会重新训练一个 official 配对基线，保证同轮比较协议统一；不会覆盖先前结果。

结果在新 `runs/年月日_full_xxxx/results/all_results.csv`、`paired_comparison.csv` 及网页仪表板。重点检查 CSRA 与 Official 的 MSE、MAE；**若无改进则如实记录，并考虑其他单模块方案**，不要用测试集反复筛参。

## 代码位置

- 只增加 `src/models/extensions/seasonal_residual.py`
- 仅注册一处 `src/models/registry.py`（安装器自动备份为 `registry_before_csra.py.bak`）
- 不修改 `run.sh`、`runner.py`、官方源码或现有模型权重。

这是工程原型；没有在用户的 AutoDL RTX 5090 上执行正式训练，不能预设其预测收益。
