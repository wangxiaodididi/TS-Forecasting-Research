# TVP × TimeMixer 独立工程 v2

这个版本专门解决“整个工程只在 `TVP_Standalone_TimeMixer_v2/` 中运行”的要求。
**不依赖 GTR_ResearchWorkbench_Modular、XLinear_ResearchLab_Clean 或此前的任何目录。**

## 两个主要入口

**入口一：CPU 小范围测试**（使用 ETTh1 真实 CSV 和代理预测器，仅验证实验程序）：

```bash
cd TVP_Standalone_TimeMixer_v2
bash 01_CPU_小样本测试.sh
```

**入口二：真正的官方 TimeMixer + TVP 复现实验（需要 GPU）**：

```bash
cd TVP_Standalone_TimeMixer_v2
bash 00_安装依赖.sh     # 如果 Python 环境还没有装依赖，先运行一次
bash 02_完整复现实验.sh
```

第二个脚本的执行流程是：
1. 检查 Python / PyTorch / CUDA 环境、ETTm1.csv 数据；
2. 下载并校验 TimeMixer GitHub 固定 commit 的官方源码（存到本目录 `training/workspace/official/`）；
3. 没有官方训练权重时，对 ETTm1-96（seed=2024）训练原版 TimeMixer；
4. 使用验证集和测试集提取官方 FMM 多尺度预测贡献；
5. 在**相同冻结 backbone**上比较 `original`, `static`, `input_gate`, `tvp_online`；
6. 输出 `results/official_ETTm1_h96_seed2024/compare/metrics.csv`。

注意：**v2 压缩包未附带原作者已训练的权重，也不假装已经完成了 GPU 训练。** 正式基准必须在服务器训练后产生。
如果 GitHub 直连下载不通：在 PC 浏览器下载官方固定版本 ZIP
`https://codeload.github.com/kwuking/TimeMixer/zip/e24610583b36fdd8c76cc17a8df4e65759a5f460`
上传该 ZIP 到此独立目录，例如 `TimeMixer_source.zip`，然后执行：

```bash
OFFICIAL_ZIP="$PWD/TimeMixer_source.zip" bash 02_完整复现实验.sh
```

## 详细设置

所有目录都相对于当前工程计算，而不是写死以前的用户路径。

- PyTorch：沿用用户服务器现有 CUDA 版本，不要因旧项目 requirements 降级 torch。
- 官方源码固定在 commit `e24610583b36fdd8c76cc17a8df4e65759a5f460`，并验证三个关键文件的 git blob SHA1。
- 仅用于适配训练流程的改动是：随机种子可配置，并关闭训练期间使用测试集的逻辑；**TimeMixer 模型架构完全保留**。
- `DATASET=ETTm1` (v2 只正式配置该数据集)；`H=96` 或 `192`; `SEED=2024`；`DEVICE=gpu`；`MAX_WINDOWS=0` 表示完整验证/测试。
- 测试期 TVP 更新需等待整个 H 步标签成熟；`MAX_WINDOWS>0` 是真实网络小范围抽取，但不是完整论文复现。
- 训练 checkpoint 存在 `training/checkpoints/`，训练日志在 `training/runs/`。
- 官方训练和提取按同一 `train_config.json` 和 batch_size=16 配置。提取器使用官方数据提供器和时间顺序。
- `03_复用已有权重.sh` 可选；只有已知 checkpoint 和对应源码配置时使用。它不应作为单独文件夹首次运行的必要步骤。

## 方法定位

`tvp_online` 是训练完成后对多尺度预测贡献进行的低维、基于已成熟标签的在线校准；目前不是端到端 TVP 网络，也没有得到证实的预测提升。CPU 代理测试通过只说明代码路径有效，不代表原版 TimeMixer 已跑通。

## 运行失败时

训练详情：`training/runs/ETTm1/h96/seed2024/original/train.log`；
源码下载出错：尝试 `OFFICIAL_ZIP` 路线；
依赖缺失：运行 `00_安装依赖.sh`；
GPU 不可用：检查 AutoDL CUDA 环境，不会静默切换成 CPU 训练。

README 的路径均相对于**当前独立工程**，不再指向旧工作台。
