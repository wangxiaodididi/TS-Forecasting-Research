# GTR Modular Research Workbench v1.0

**目的**：在不改动 ICLR 2026 官方 GTR 周期矩阵 Q、相位检索、2D 融合卷积与原 MLP 的前提下，进行可复现的信号处理扩展实验。

本压缩包附带 ETTh1、ETTm1、Weather、Electricity 四套公开数据。**为了确保官方源码完全相同，首次安装必须下载官方 GitHub 固定提交**；`vendor/GTR/` 初始为空，并非已包含原始源码。若服务器已下载作者仓库，可以通过 `./.venv/bin/python -m lab.cli adopt-official /旧仓库目录` 导入：只接受校验通过的原版提交。

## 最快开始（AutoDL + VS Code Remote SSH）

1. 在服务器上用 VS Code **打开整个项目文件夹**，不要单独打开模型文件。
2. 运行 `bash 00_安装依赖.sh`，使用当前 AutoDL Python/PyTorch 环境，使用 `.venv` 隔离安装，复用当前 Python 环境的 PyTorch/CUDA，并固定 `numpy<2` 兼容官方源码。
3. 运行 `bash 03_打开可视化面板.sh`，在 VS Code `PORTS` 中转发 **8501** 到本地，在浏览器打开 `http://localhost:8501`。
4. 选择 **ETTm1 / GTR / 96 / CPU 小样本**，启动；确认训练/验证曲线和权重后再运行论文模式。

## 运行入口（独立后台任务）

```bash
bash 01_CPU小样本测试.sh ETTm1 GTR 96
bash 02_完整论文配置.sh ETTm1 GTR 96 2024
bash 02_完整论文配置.sh ETTm1 GTRHaarHigh 96 2024
bash 04_查看实验列表.sh
./.venv/bin/python -m lab.cli stop <实验编号>
```

`02_完整论文配置.sh` 不代表所有改造模型都是“官方论文模型”，只代表沿用同一训练超参数和数据协议。**严格官方基线**必须是 `--model GTR --mode paper`。

## 代码结构 / 修改位置

```text
GTR_ResearchWorkbench_Modular/
  00_安装依赖.sh / 01_CPU小样本测试.sh / 02_完整论文配置.sh
  03_打开可视化面板.sh / 04_查看实验列表.sh
  configs/
    official/          # 作者脚本逐数据集锁定参数，基线不得修改
    experiments/       # 改造模型的实验参数，默认复用官方超参
  vendor/GTR/          # 官方仓库固定提交，仅在安装时创建；永远不直接修改
  extensions/models/   # 自定义模型：高频、低频、双频、线性残差，保留原始 GTR
  extensions/diagnostics/ # 后续可增补信号质量和误差分析
  lab/
    config.py          # 参数验证、模型与数据集参数统一组装
    source.py          # 锁定作者 SHA、校验、建立每次实验的源码快照
    extend.py          # 仅在实验快照内注册新模型/小样本化
    runner.py          # 独立训练任务进程管理
    monitor.py         # 解析作者实时 Train/Val/Test Loss 输出、GPU 采集
    dashboard.py       # 网页交互、实时曲线、日志和实验对比
    inspect.py         # 参数量、模块名、forward I/O 诊断
    explain.py         # 校正贡献可复核 CSV
    cli.py             # 命令行统一入口
  dataset/             # 公开数据集，不包含 ERCOT
  runs/<编号>/         # 每次独立的 config、status、metrics、完整源码与 checkpoints
  tests/               # 本地单元测试
  docs/                # 研究模块拆解与修改手册
```

## 关键原则

- 严格基线：使用作者提供的 `run.py`、`exp/exp_main.py`、`models/GTR.py`、`data_provider`，用 `scripts/GTR/*.sh` 的值组装参数。配置只覆盖作者显式填写的键，余下遵循官方 `run.py` 默认值。
- 原训练器每个 Epoch 打印真实 `Train Loss` 和 `Vali Loss`，工作台只解析标准输出。验证损失**不会**被伪装为每个 Batch 都在更新；Train Batch Loss 则按原论文训练器 100 个 Batch 打印一次。
- 官方实验会每轮同时计算 test loss（作者源行为）；这里只存档，不应据此调参。测试集最终指标应在固定方案后报告。
- CPU 小样本模式将 train/val/test 数据集的每个子集截取为最多 64 条样本，1 Epoch，batch=8，且强制 CPU。只是功能检查，**与官方论文结果不可比较**。
- 训练日志、GPU、源代码快照、验证最优 Epoch、配置和完整输出自动保存在 `runs/`；断开 VS Code 或浏览器不会终止服务器中运行的子进程，停止 AutoDL 实例则会终止。
- 源码快照**创建后与工作区分离**；修改 `extensions/` 只作用于以后启动的新任务。新模型通过 `lab/extend.py` 注册，不能直接编辑 `vendor/GTR/`。
- 所有可解释性结果是预测分支贡献（描述性），不是经济因果解释。

## 模型研究/解释命令

```bash
./.venv/bin/python -m lab.inspect --model GTRHaarHigh --dataset ETTm1 --horizon 96
./.venv/bin/python -m lab.explain <训练编号> --split val --limit 8 --channel 0
```

`explain` 读取当前实验快照内的最优 checkpoint，并从原始 data_provider 读取验证数据，保存含原始 GTR 预测、低频/高频校正和真实值的 CSV。建议固定验证集选模后才解读 test。

## 论文协议注意

- GTR 作者主仓库固定提交：`d94161906151bdc0c94f4d21c2b0ca356aeb3135`（可在 `lab/source.py` 更换，但所有版本要记录并重新审核）。
- ETTh1 论文脚本列 6 个种子（2024–2029），ETTm1/Weather/Electricity 脚本只有 2024；本系统不会悄悄多加种子。
- 使用相同硬件、数据、seed、作者训练代码和超参，不代表不同 PyTorch/CUDA 的运行结果一定逐位一致；需保留环境版本。
- 论文基线输出真实训练日志与最终指标，不包含本交付包任何声称已得到更低预测误差的数据。

原始论文： https://openreview.net/forum?id=QUJBPSfyui  
官方仓库： https://github.com/macovaseas/GTR  
许可证：官方 Apache-2.0，以 vendor/GTR/LICENSE 为准。

## 运行时兼容性

作者官方 `utils/tools.py` 中使用 `np.Inf`，NumPy 2.x 将报错。安装脚本在独立 `.venv` 中固定 `numpy>=1.24,<2`，不修改原始训练器；如果环境缺乏 venv，需先补齐。

## 额外理解工具

- `./.venv/bin/python -m extensions.diagnostics.data_signal_audit --dataset ETTm1`：训练段周期相关和小波高频能量诊断；
- `./.venv/bin/python -m lab.inspect --model GTRHaarHigh --dataset ETTm1`：统计模型参数量、模块及前向输出形状；
- `./.venv/bin/python -m lab.explain <RUN_ID> --split val --channel 0`：从最佳 checkpoint 导出逐步预测贡献；
- VS Code `Terminal → Run Task`：可直接选择安装、CPU 检查、官方训练或启动面板，无须重复输入命令。
