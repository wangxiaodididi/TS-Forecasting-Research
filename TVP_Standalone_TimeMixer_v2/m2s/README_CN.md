# M→S 数据接入原有 TVP_Standalone_TimeMixer_v2 工程

## 放置方法（数据/入口覆盖式导入，不影响旧文件）

把 ZIP 上传到 `/root/autodl-tmp/`：

```bash
cd /root/autodl-tmp
unzip -o TVP_TimeMixer_M2S_Overlay_v1.zip -d .
cd TVP_Standalone_TimeMixer_v2
PYTHON=.venv/bin/python  # 若虚拟环境位于项目内，可改用绝对路径
python -m m2s.verify
bash 04_M2S_CPU_SMOKE.sh
```

假如旧工程使用的 Python 环境已激活，直接执行 `bash 04_M2S_CPU_SMOKE.sh`。
正式 GPU：`bash 05_M2S_GPU_TWO.sh` 或 `bash 06_M2S_GPU_ALL.sh`。
使用 `PYTHON=/path/to/python bash 04_M2S_CPU_SMOKE.sh` 指定解释器。

## 重要说明

- 使用旧项目 `training/workspace/official/models/TimeMixer.py` 里的**官方 TimeMixer.Model**，不会覆盖、重置或替换旧网络、权重、日志或 TVP 代码。
- 如该官方源码不存在，先在旧项目中执行 `python training/prepare.py`。旧项目 `training/prepare.py` 自带固定 commit 检查；无网络时按旧项目 README 使用原始官方源码 ZIP。
- 六组训练 CSV 位于旧工程的 `data/{Hubei,Guangdong,CAISO,PGCB,NESO,OPSD}.csv`。
- 预测目标是最后一列 `OT`，湖北、广东为**碳市场收盘价**。模型输入含 `OT` 历史及其他原始观测变量；**只预测未来 OT**。
- 没有添加收益、滚动、技术指标、日历编码、人工滞后等派生预测变量。历史窗口切片与规范化不增加输入通道。
- 对有缺失值的辅助变量，运行时使用训练期拟合的均值代入；原始 CSV 空值不改变，不创建缺失指示列；目标 OT 不填补。
- `target_only`：仅历史 OT，使用官方 TimeMixer 的通道独立配置。
- `multivariate_joint`：全部历史输入变量，使用官方 TimeMixer 通道相关配置。官方模型原版输出全部 C 通道，但训练损失、预测保存和评价**仅使用最后的 OT 通道**，属于单目标监督 M→S 任务。
- `multivariate_independent`（可选对照）：全部变量输入但通道独立，仅 OT 监督。
- 不把旧项目 ET​​Tm1 7通道 M→M 的模型权重用于新数据，新任务从头训练，旧权重原样保留。
- 不调用旧 `training/runner.py` 的 `--features M` 路径。新入口在自己的 `m2s/` 数据加载器中严格读取窗口索引，并单独训练官方模型。
- 仅使用已提供的固定 train/val/test 窗口，数据不连续时不跨段拼窗；仅用 train 期拟合预处理参数。测试集在验证集选出最佳权重后评估，不用于训练期选模。
- `results/m2s/smoke/` 和 `results/m2s/full/` 严格分开；`summary.csv` 在每个子实验后更新；日志在 `results/m2s/{mode}/logs/`。

## 数据参数

| 数据集 | 输入真实观测变量数（含 OT） | L | H |
|---|---:|---:|---|
| Hubei | 18 | 60 | 1, 5 |
| Guangdong | 16 | 60 | 1, 5 |
| CAISO | 6 | 168 | 24, 48 |
| PGCB | 9 | 168 | 24, 48 |
| NESO | 18 | 336 | 48, 96 |
| OPSD | 7 | 168 | 24, 48 |

## 手动运行单组

```bash
python -m m2s.train --dataset Hubei --horizon 1 --variant multivariate_joint --device cpu --epochs 3 --batch-size 8 --max-train-windows 64 --max-eval-windows 16
```

## 环境依赖

沿用原来 `.venv`（PyTorch、numpy、pandas）。缺依赖才执行 `python -m pip install numpy pandas`，不要重新安装或降级你已配置的 CUDA PyTorch。

## 验证限制

数据格式、窗口时间顺序、模型调用接口和训练路径已离线核查。官方网络训练必须在具有原始 `training/workspace/official/` 的 AutoDL 环境实际执行；交付包不包含旧服务器上的已训练权重，也不宣称已经完成 GPU 复现。
