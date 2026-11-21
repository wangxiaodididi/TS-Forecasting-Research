# GTR 第一轮：保留核心检索模块的结构消融

## 目标
已复现原始 GTR / ETTm1 / H=96 / Seed 2024：Test MSE=0.30702537、MAE=0.35145476（用户服务器结果；并非此次补丁产生的结果）。论文对应 Test MSE=0.305、MAE=0.349。

本补丁**不改动** `vendor/GTR`、`lab`、`configs`、`dataset`、`runs`、`.venv` 中任何已有文件。运行时直接调用已有 `lab.runner`，在每个实验目录中产生独立官方源码快照与改造模型。

## 四个候选模型
| 研究模型 | 变化 | 意义 |
| --- | --- | --- |
| GTRLinearResidual | 96→H 的可学习线性校正 | 新增预测参数的控制组 |
| GTRHaarHigh | 一层 Haar 高通细节→H | 高频信号是否可预测 |
| GTRHaarLow | 一层 Haar 低通近似→H | 较低频变动是否提供补充 |
| GTRHaarDual | 低频+高频分别映射→H | 分解后联合表征的控制组 |

> 方法学警告：对偶数输入长度，两条无约束线性 Haar 分支合并，与完整的线性残差头具有**相同线性表达能力**，二者参数量也相同。若出现性能差异，需进一步讨论优化路径/参数化/正则化，不能认定“双频段模型理论表达力更强”。

## 安装（只新增文件）
将 `GTR_Phase1_SafePatch.zip` 上传至项目根目录，用 `unzip -n GTR_Phase1_SafePatch.zip` 解压（无额外父目录）。可在解压前先用 `unzip -l` 检查，绝对不要再用 `unzip -o` 覆盖整个原项目。

## 工作流程
**CPU 自检和 4 个小样本试验（非论文复现）：**

```bash
bash 07_Phase1_CPU.sh ETTm1 96 2024
```

**官方同协议 GPU 实验（4 个模型依次运行，避免 GPU 争抢）：**

```bash
bash 08_Phase1_GPU.sh ETTm1 96 2024
```

如果没有找到匹配的已完成官方基线，它将拒绝启动，不会自动采用假数据或不公平的结果。你的官方基线实验 ID 为 `20261115_220842_809d65d`，可以指定：

```bash
./.venv/bin/python -m phase1.launch start --dataset ETTm1 --horizon 96 --seed 2024 --mode paper --baseline-run 20261115_220842_809d65d
```

**查看进度与验证结果：**

```bash
bash 09_Phase1_Status.sh
```

`bash 03_打开可视化面板.sh` 的 Streamlit 左侧导航会多一个 `Phase1 Ablation` 页面，5 秒自动刷新。模型逐个训练；原工作台的单次任务页面继续查看实时 Train/Validation Loss、Batch Loss 和 GPU 监控。

**锁定模型后才查看最终测试结果：**

```bash
bash 10_Phase1_Final_Results.sh
```

注意：原作者 `exp.train()` 内每轮仍打印 Test Loss，`run.py` 训练后自动执行一次 Test；本补丁没有修改官方训练流程，故技术上无法隔离测试信息。研究时应根据验证集选择模型，不依赖测试集指标确定结构/参数。

## 数据导出
每次实验：`runs/<run_id>/config.json`、`loss.csv`、`console.log`、`status.json`、`source/`、`provenance.json`。

批次：`phase1_batches/<batch_id>/state.json`、`orchestrator.log`、`VALIDATION_ONLY.csv`；运行 `--include-test` 才导出 `full_metrics_AFTER_MODEL_LOCK.csv`。

## 当前证据和限制
此补丁包含运行器和展示代码，**没有**在你的 AutoDL GPU 上运行四组正式实验，也没有得到任何改造模型 MSE/MAE。禁止将 CPU smoke 或训练集 Loss 当作论文对比结果。

## 未来改动
修改 `extensions/models/GTRSignalCore.py` 后，现有实验不受影响，下一次新建任务才会生成新的源代码快照。更换模型结构时建议更换名称/配置，避免结果误归类。
