# XLinear Research Lab · v4 从零恢复完整工程

**适用于旧项目和权重全部丢失的情况。** 这里包含我们的完整研究源代码、4 组公开基准 CSV 数据压缩包、统一配置、官方训练调用、Global Token 改造、训练日志、实时 Loss/GPU 可视化和离线门控诊断。**唯一需要首次联网获取的是作者 XLinear 固定版本官方源码**；`bash run.sh init` 会自动下载、校验哈希并准备数据，不依赖任何旧目录。此前服务器训练权重和日志不在此包内，不能凭空恢复。

## 从零安装（复制执行）

将 ZIP 上传到服务器的 `/root/autodl-tmp/`，确保 ZIP 与待创建的项目目录处于同一级。若存在同名项目，请先备份/改名，勿直接 `unzip -o` 覆盖。

```bash
cd /root/autodl-tmp
unzip XLinear_ResearchLab_Full_Rebuild_v4.zip
cd XLinear_ResearchLab_Clean
bash run.sh init                 # 第一次安装依赖 + 下载/校验固定版本官方源码 + 解压公开数据
bash run.sh check                # CPU 核心结构和梯度自检
bash run.sh plan                 # 打印当前实验计划，确认后再训练
bash run.sh baseline             # 可选：先复现论文 ETTh1-96 official，单个随机种子2025
bash run.sh pilot                # 新模块的短 Epoch GPU 工程测试
bash run.sh full                 # 官方参数完整 GPU 对照 official/dyn_token/stat_token
```

**训练时另开 VS Code 远程终端**，执行 `bash run.sh dashboard`，在 Ports 转发 `8501` 至本机，访问 http://localhost:8501。它只监控，不启动或停止训练。需要持续后台训练则使用 `bash run.sh full-bg`，不要与 `full` 同时运行。修改唯一参数文件 `configs/experiment.yaml`，扩展位置 `src/models/extensions/` 和注册文件 `src/models/registry.py`。

## 原论文对照结果与可恢复边界

用户先前在 AutoDL 服务器通过官方原始训练器取得 **ETTh1-96 MSE 0.3694673777，MAE 0.3933632672，seed2025**。这些数字是先前真实运行记录，**不是本 v4 自带的 checkpoint**。重新运行时因运行环境差异，可能不会逐位相同。官方源码固定版本为 `8398e656e945da84a79e8b2f64af8f3f68842e19`，模型 Git blob 为 `9913ba6837a7c8658283636ab7f5727e07f2f709`。本项目的 official 和改造模型共享原始 train/data/optimizer/metrics 训练链路，改造部分是新增可审计模块，尚不代表已获稳定精度增益。

---

## 既有功能详解（v3 主体保留）

本项目用于拆解 AAAI 2026 XLinear 并开展**保留主干的局部模块改造**，支持 RTX 5090、VS Code Remote SSH、统一 YAML 配置、实时 Train/Validation Loss 及离线可解释性诊断。全项目使用英文文件名，源码和文档中文注释。

**一个训练入口：`run.sh`；一份参数：`configs/experiment.yaml`；一套官方训练流程。** 不再往旧项目叠加 `paper_reference`、`paper_variants` 或其他训练器补丁。旧版项目只保留为备份。

## 一、功能与文件结构

```text
XLinear_ResearchLab_Clean/
├── run.sh                         # 唯一终端命令入口
├── configs/experiment.yaml        # 数据集、预测长度、论文参数、随机种子、监控参数
├── src/
│   ├── core/config.py             # 读取配置、创建受控实验矩阵
│   ├── runtime/bootstrap.py       # 官方源码下载/字节校验、公开数据准备
│   ├── models/
│   │   ├── registry.py            # official / dyn_token / stat_token 注册表
│   │   └── extensions/global_token.py   # 目前唯一改造模块
│   ├── training/
│   │   ├── official_entry.py      # 复用作者训练器，仅内存加批次日志（可关）
│   │   ├── runner.py              # GPU 任务执行、进度、运行隔离和安全停止
│   │   ├── telemetry.py           # Epoch/Batch/GPU 实时指标采集
│   │   ├── diagnostics.py         # 测试误差分解、门控和梯度离线分析
│   │   ├── results.py             # 多模型多种子配对比较
│   │   └── status.py              # 查看任务状态
│   └── dashboard/app.py           # 一个 Streamlit 网页，集中所有可视化
├── tests/selftest.py              # 官方输出初始对齐及模块梯度测试
├── tests/test_monitor.py          # 实时监控注入与 CSV 解析自检
├── docs/                          # 论文流程、配置、模块改造、监控、迁移说明
├── data_archives/public_datasets_4.zip  # 已有 ETTh1/ETTm1/Weather/Electricity
├── vendor/XLinear/                # 首次 setup 下载/从旧备份复制，经固定哈希校验
├── runs/                          # 所有训练、模型权重、曲线和诊断结果
└── ops/                           # 后台日志与 GPU 锁
```

注意：本 ZIP **不包含作者完整 GitHub 源码**，首次 `setup` 需要从旧项目已验证版本复制或下载固定 commit；模型主干不允许在 `vendor` 里直接修改。

## 二、首次部署到 AutoDL

**以下旧项目迁移步骤仅供有旧备份时参考；旧目录已删除的情况直接使用上方从零安装即可。** 在 VS Code 远程终端执行以下命令，假设新版 ZIP 上传到 `/root/autodl-tmp/`：

```bash
cd /root/autodl-tmp
BACKUP="XLinear_ResearchLab_Backup_$(date +%Y%m%d_%H%M%S)"
mv XLinear_ResearchLab_Clean "$BACKUP"
unzip XLinear_ResearchLab_Clean_Monitor_v3.zip
cd XLinear_ResearchLab_Clean
python -m pip install -r requirements.txt
python -m pip install -r requirements-ui.txt

# 如果旧项目已有 vendor/XLinear，优先复用：
if [ -f "../$BACKUP/vendor/XLinear/run_longExp.py" ]; then
  bash run.sh setup --reuse-upstream "../$BACKUP/vendor/XLinear"
elif [ -f "../$BACKUP/XLinear_Paper_Reference_Patch_ASCII/paper_reference/upstream_XLinear/run_longExp.py" ]; then
  bash run.sh setup --reuse-upstream "../$BACKUP/XLinear_Paper_Reference_Patch_ASCII/paper_reference/upstream_XLinear"
else
  bash run.sh setup
fi

bash run.sh check
python -m tests.test_monitor
bash run.sh plan
```

整段代码可在**同一个终端**依次运行。旧项目重命名备份，不删除旧 `runs/`、正式权重和 ETTh1-96 的 MSE 0.369467 原始复现资料。

首次 GPU 小测试：

```bash
bash run.sh pilot
```

正式训练：

```bash
bash run.sh full               # 前台：tqdm Batch 进度 + 每 Epoch 训练/验证 MSE
# 或者：bash run.sh full-bg     # 后台：不会因为关闭 VS Code 终端而自动停止
```

**不要同时启动多个训练入口**。程序有单项目 GPU 文件锁，但无法阻止你在其他项目开启的训练。

## 三、训练可视化怎么打开？

在**另一个 VS Code 终端**执行：

```bash
bash run.sh dashboard
```

将 VS Code 底部 `PORTS` 的 `8501` 转发至本地，打开 `http://localhost:8501`。打开/关闭网页不会重新训练，也不会停止后台 GPU 进程。

看板包含：

1. **实时 Epoch 曲线**：Train MSE 与 Validation MSE 同图、最优验证 Epoch、过拟合差距。
2. **Batch 进度与算力**：每10 Batch 记录一次 Train Batch Loss，GPU 利用率/显存/功耗每5秒采样。
3. **原版与改造比较**：按相同数据集、H、随机种子配对计算 MSE/MAE 改善。
4. **预测与门控诊断**：未来步长误差、真实值和预测值、TGM/VGM Sigmoid 门控分布、单批梯度范数（需训练结束后单独诊断）。
5. **模型结构说明、日志与 CSV 导出**。

**验证集损失只能每个 Epoch 完成完整验证后更新；不是每个训练 Batch 更新。** Batch Loss 仅为训练批次误差。正式测试集不用于早停和参数选择。

## 四、离线诊断（训练后）

打开网页，选择需要诊断的模型，页面会显示可复制的 `diagnose` 命令。也可以直接执行：

```bash
bash run.sh status
bash run.sh diagnose --run 20261115_123456_full_abcdef
# 仅诊断一个作业：
bash run.sh diagnose --run 20261115_123456_full_abcdef --job ETTh1_H96_dyn_token_s2025
```

`diagnose` 加载已经保存的权重与作者保存的测试预测，重新读取测试真值验证 MSE 一致；使用一个验证 Batch 进行门控分布与梯度探针，**不进行梯度更新**，不改变任何权重。单批梯度和门控不能解释为变量因果重要性。

## 五、如何修改 / 扩展模型

参数统一在 `configs/experiment.yaml`，其中 `monitoring.batch_interval` 控制实时训练批次采样（设置为 `0` 则在内存中不注入任何批次日志）。

新算子放入 `src/models/extensions/`，并在 `src/models/registry.py` 注册，在 YAML `experiment.variants` 增加名字。训练器不必复制、改名或修改。每次修改后：

```bash
bash run.sh check
bash run.sh pilot
# 验证合理后，再运行 full 进行相同设置下的配对对照
```

已成功复现的 ETTh1-96 原作者结果 **MSE=0.369467 / MAE=0.393363** 来自用户先前服务器的独立任务，并非这个 ZIP 预先包含的运行结果。这个版本已通过离线代码、监控注入及解析测试；**尚未在 RTX 5090 上执行 GPU 复现实验**。首次更新应再跑一组官方基线，确认监控没有带来实质差异。

详细文档：`docs/01_model_flow.md`、`docs/02_training_config.md`、`docs/03_extending_models.md`、`docs/04_migration_guide.md`、`docs/05_visual_monitoring.md`、`docs/06_module_research.md`。
