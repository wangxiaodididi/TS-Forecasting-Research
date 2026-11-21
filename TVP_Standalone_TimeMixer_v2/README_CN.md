# TimeMixer + TVP: 跨任务验证与预测头改造（v3）

**用途：** 给已有独立 `TVP_Standalone_TimeMixer_v2` 工程追加批量验证与预测头实验。此包是**补丁，不是新工程**，不会自动删除旧日志、权重或结果。

文件名和路径全部为英文（ASCII）。`data/weather.csv` 是可选的第三数据集；已含在补丁内。所有脚本放在**已有 v2 工程根目录**，与 `tvp_lab/`, `training/`, `results/` 同级。解压时保留 `code/`、`data/` 子目录。

## 一、安装（在旧项目根目录执行）

```bash
# 将 TVP_NextStage_v3.zip 上传到已有 v2 项目根目录；终端已经在该根目录
unzip -o TVP_NextStage_v3.zip -d .
python 13_cpu_checks.py
```

不要执行 `unzip -j`，因为预测头源文件需要位于 `code/` 子目录。

## 二、一次完成 TVP 的必要验证

**已有权重，仅复用且补全分析：**

```bash
python 08_validation_matrix.py --tasks 'ETTm1:96:2024' --device gpu
python 09_results_summary.py
```

**扩展到其他任务（缺失原版权重则自动训练）：**

```bash
python 08_validation_matrix.py --tasks 'ETTm1:96:2024,ETTm1:192:2024,ETTh1:96:2024,Weather:96:2024' --device gpu --train-missing
python 09_results_summary.py
```

或者一键批量验证前三项（默认不跑 Weather）：

```bash
bash 14_run_all.sh
```

包括：原 TimeMixer、原全尺度 TVP 的**统一初始化重算**、选择性时变尺度 TVP、分尺度 q TVP。模型选择仅用验证集，测试集只测一次；明确区分旧版 `tvp_online` warm start 与新的严格配对结果。

主要结果：

```text
results/tvp_validation_matrix/
  matrix_metrics.csv
  matrix_details.json
  decision_notes.md
```

每项还会生成：

```text
results/official_{dataset}_h{H}_seed{seed}/compare/structural_trials/
  structural_metrics.csv
  structural_manifest.json
  structural_predictions.npz
  validation_candidates.csv
```

验证重点：跨数据集、跨预测期限、原版与统一初值 TVP、选中尺度、各变量及各预测区间增益，以及重叠预测的探索性 HAC 检验。**固定测试期不根据结果重新挑参数；不要把一次小增益宣称为显著。**

## 三、下一位置：真正改造预测头（不再改 FMM）

原版 TimeMixer 结构是 `PDM → predict_layers → 各尺度预测求和(FMM)`。当前 TVP 是冻结网络后的在线后处理。

这次新增的 `TimeMixerResidualHead` 仅修改 `predict_layers` 的时间映射，保留 PDM、FMM。每个原始线性层加入一个**零初始化**的秩 r 残差映射：

- `linear`：低秩线性残差，属于参数化/优化对照（等效线性表示，不是全新的非线性能力）；
- `nonlinear`：中间使用 GELU，检验是否需要非线性预测头。

新增模型与原版**独立从头训练**，不把前述 TVP 参数直接移植到这一步。训练和提取严格对齐原官方的任务配置与测试样本。

**先做 CPU 结构接入测试：**

```bash
python 15_official_head_smoke.py
```

**再训练新预测头（先只跑一项 ETTm1-96）：**

```bash
python 11_train_head.py --dataset ETTm1 --horizon 96 --seed 2024 --mode nonlinear --rank 8
python 12_extract_head.py --trial ETTm1_h96_seed2024_nonlinear_rank8
```

如果非线性头优于原模型，再跑其线性容量对照：

```bash
python 11_train_head.py --dataset ETTm1 --horizon 96 --seed 2024 --mode linear --rank 8
python 12_extract_head.py --trial ETTm1_h96_seed2024_linear_rank8
```

实验保存在：

```text
results/head_trials/ETTm1_h96_seed2024_nonlinear_rank8/
  train.log
  train_config.json
  head_result.json
  extracted/{val,test}.npz
```

`head_result.json` 包含相同测试目标、样本规模和原始预测的严格对齐验证。若不对齐则报错，**不输出性能结论**。

## 四、边界和研究决策

1. v3 本地已通过 CPU 组件测试、TVP 延迟更新和选择结果不读取测试标签的验证；尚未在本地执行 AutoDL GPU 的完整新任务训练。
2. 此包的 TimeMixer 预测头是新的位置，不是把 Selective TVP 再叠进去。因此即使表现较好，先对比原模型、线性残差、非线性残差，再考虑两种改造联合。
3. 参考分组：ETTm1-96 (已做)、ETTm1-192、ETTh1-96；Weather-96 作为进一步验证。每一项使用各自独立训练的原始权重。
4. **不能把旧版 `tvp_online_v2_reference` 与本轮统一初值模型的性能差归因于选择性时变。**
5. 对小于1%的改进，需要同时关注 MAE 与 MSE、不同数据集符号是否一致、统计不确定性与参数/计算成本，不进行测试集择参。
6. 这里的公开数据多步预测用于验证模块迁移性；后续碳价格预测应重新定义日度目标、训练协议与基准，不能直接声称在碳市场上已有效。

## 五、排错

- 不是 GPU 训练环境：先运行 `python 13_cpu_checks.py`；完整训练需要 CUDA。
- `No verified checkpoint`：对应任务尚未训练，使用 `--train-missing`。
- `Official source missing`：旧工程先运行 `02_full_experiment.sh` 完成源码准备。
- `Missing weather.csv`：确认解压后的 `data/weather.csv` 存在。
- `Original vs modified output differs`：先处理官方源码或配置不匹配问题，不运行 GPU。
- `Head vs baseline test targets differ`：不能报告 MSE 比较；核对数据集切分/数据加载器和训练配置。
