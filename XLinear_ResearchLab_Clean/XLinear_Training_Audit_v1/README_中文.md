# XLinear 改造与训练审计 v1（2026-11-16）

目的：不创造新模块，先检验 CC-XLinear 的结构接入、正式模型初始等价性、训练协议、已学习路由的实际贡献。**当前包为只读审计**，只在项目根目录创建 `audit_outputs`，不覆盖 `registry.py`、checkpoint、原代码、已有 runs。

## 解压与使用

将本 ZIP 上传至已经装有 CC-XLinear 变体的远端 `XLinear_ResearchLab_Clean` 项目根目录，原地解压：

```bash
cd /root/autodl-tmp/XLinear_ResearchLab_Clean
unzip -o XLinear_Training_Audit_v1.zip
# 下面的 python 必须是你原正式实验用的同一个解释器（示例按实际环境替换）
PYTHON_BIN=python bash XLinear_Training_Audit_v1/01_CPU_审计.sh
```

先看 `environment_*.json` 中 `source_integrity=PASS` 和全部变体已注册。结构审计只运行 CPU，不需要 GPU。它使用**官方真实 XLinear backbone**比较零初始化输出；如果作者源文件或注册表与原项目不符，会明确报错，不自动安装或修改。

已训练的 CC checkpoint 若仍在 `runs/` 下，直接只读审计：

```bash
# 具体名字以服务器 runs 目录为准；不要原样复制下行占位符
bash XLinear_Training_Audit_v1/02_已有权重审计.sh 你的运行目录名 8
# 要运行整个验证集而非前8个batch时：
bash XLinear_Training_Audit_v1/02_已有权重审计.sh 你的运行目录名 0
```

结果保存在 `audit_outputs/`，包括：
- `manifest_*.json`：当前每个 job 使用的解释器、参数和 checkpoint 存在情况，避免环境不一致。
- `structural_*.json`：官方模型真实初始等价性、新增参数梯度、路由归一化。
- `checkpoint_*.json`：已训练模型在验证样本上，正常/禁用新增路由/使用批次平均路由时的MSE，路由偏离均匀权重的幅度，隐藏状态相对更新，新增参数梯度与系数。
- `historical_*.csv`：先前截图结果按配对随机种子重新汇总。

由于模拟主干的单元测试不能替代官方模型，**必须运行 `01_CPU_审计.sh` 中的真实backbone结构审计**。如果服务器尚无 checkpoint，只执行第一步，不要把缺权重解释成结构失效。

## 判读标准

1. `initial_equivalent=false`：先修实现问题，不训练。
2. `peer_grad_l2`、`coeff_grad_l2` 为0或None：检查路由前向与梯度链路；`cc_static` 的 `coeff_grad_l2=None` 本来就是预期结果。
3. `mean_relative_hidden_update` 极低且 `mse_normal`≈`mse_without_new_route`：新增路径可能几乎未被使用；这是**局部诊断，不等于重训消融**。
4. `mse_normal`与`mse_with_batch_average_route`几乎相同：在所检验证窗口中，样本条件化路由可能不是关键；该干预并非可部署预测算法。
5. 正常输出明显变化但MSE不改善：怀疑新增路由引入噪声、目标不对齐或与VGM功能重叠，应检验假设而不是盲目调参。

**重要**：诊断仅使用 `val`，不使用测试集选择超参数；默认前8个验证批次用于快速检测。正式论文结论需要完整验证+固定配置测试，以及训练独立消融；不得把训练后关闭分支的结果解释为从零消融实验。

## 完整重跑（仅当服务器原权重无法定位或要对比修复训练协议时）

```bash
bash XLinear_Training_Audit_v1/03_正式训练_备用.sh
# 检查计划无误后：
bash XLinear_Training_Audit_v1/03_正式训练_备用.sh --execute
```

注意：这是**备选完整GPU对照**，不会自动启动。先审计已有权重，避免不必要的计算。脚本严格调用现有 `run.sh` 训练入口，按已有YAML规则执行。

## 已识别的具体风险

- `XLinear_CC_XLinear_Existing_Project_Patch.zip` 原始安装器的注册逻辑固定 ETTh1、96步历史和周期24；但最新服务器代码已有人为适配Weather和192预测长度，**不要覆盖你现有的 registry.py**。
- Weather为10分钟数据，24采样步=4小时而不是1天；如果Weather改造仍用24，则不能解释为“日周期”成分。
- `src/training/diagnostics.py` 是早期Token扩展的诊断：对CC变体读取 `gain_logit` 不成立，不应当直接用作CC的正式诊断。
- 旧结果只有截图汇总+交接包，最新模型checkpoint不在交接包中；没有执行真实预训练模型的验证损失干预。
- CPU或假主干前向通过不代表正式训练优化收敛。
