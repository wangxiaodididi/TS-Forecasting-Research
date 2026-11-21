# 05. 实时训练可视化与结果诊断（v3）

## 1. 你会看到哪些图？

所有页面均为**真实训练数据**，读取 `runs/<任务>/` 下的文件，不生成示例数据、不启动第二个训练进程。

| 视图 | 何时更新 | 输入数据 | 研究用途 |
|---|---|---|---|
| Epoch Train MSE 与 Validation MSE | 每个 Epoch 结束，网页约每3秒刷新 | `metrics/epochs.csv` | 收敛、过拟合、早停检查 |
| Train/Validation 差距 | 每个 Epoch | 同上 | 判断训练拟合是否开始偏离验证性能 |
| Batch Train Loss | 每 `monitoring.batch_interval` 批 | `metrics/batches.csv` | 查发散、loss 跳变、学习率异常 |
| GPU 利用率、显存 | 每5秒（可调整） | `metrics/gpu.csv` | 性能和显存瓶颈诊断 |
| 模型总体比较 | 每个模型跑完并产生最终 MSE/MAE | `results/*.csv` | 逐种子公平配对比较 |
| 不同未来预测步的误差 | 离线 `diagnose` 后 | `diagnostics/horizon_errors.csv` | 判断短、中、长跨度误差变化 |
| 预测 vs 实际 | 离线 `diagnose` 后 | `diagnostics/prediction_sample.csv` | 找系统性偏差和预测滞后 |
| TGM/VGM Sigmoid 门控直方图 | 离线 `diagnose` 后 | `diagnostics/module_diagnostics.json` | 判断门控饱和/失效，诊断结构 |
| 模块梯度 L2 | 离线 `diagnose` 后 | 同上 | 单批反向传播的梯度敏感性探针 |

**Val Loss 只能在整个验证集完成一次前向计算后得到，因此是每 Epoch 更新，不能在每个训练 Batch 后获得完整、真实的 Val Loss。** 若要求每个 Batch 都重新计算验证集会显著增加耗时并改变实验运行条件，此版不这样做。

## 2. 启动方式

启动训练（终端 A）：

```bash
bash run.sh full
# 或后台：bash run.sh full-bg
```

启动监控（终端 B，仅首次安装网页依赖）：

```bash
python -m pip install -r requirements-ui.txt
bash run.sh dashboard
```

在 VS Code Remote SSH 底部 `PORTS` 转发 `8501`，本地浏览器访问 `http://localhost:8501`。网页自动每3秒刷新，可以切换不同 run、数据集、预测长度、模型和随机种子。关闭浏览器不会停止 GPU 后台训练。

只在终端看进度：

```bash
bash run.sh status
ls -td runs/* | head -n 1
# 找到目录名后查看其中 jobs/<作业>/train.log
```

完整训练**前台**运行会自动显示 tqdm Batch 进度条（Python 环境安装了 requirements.txt 后）。后台模式不写动态进度条，而写实时 CSV 和普通日志，以免日志污染。

## 3. 参数只在一个文件里修改

文件：`configs/experiment.yaml`。主要选项：

```yaml
monitoring:
  batch_interval: 10             # 10 批记录一次，0 = 不注入批次监控
  gpu_interval_seconds: 5        # 保存 GPU/显存/功耗
  dashboard_refresh_seconds: 3   # 网页自动刷新秒数，修改后重启网页生效
```

采样越密，日志写入开销越大；默认每10 Batch 一个点适合观察曲线，不会显著影响 GPU 计算。Epoch Loss 与官方原版完全相同，监控只增加 stdout 采样。

## 4. 训练结束后怎样做深入分析？

```bash
bash run.sh diagnose --run 20261115_123456_full_abcdef
# 只分析某个模型：
bash run.sh diagnose --run 20261115_123456_full_abcdef --job ETTh1_H96_dyn_token_s2025
```

输出诊断结果保存在相应作业的 `diagnostics/`，网页“预测与门控诊断”标签页自动显示。`diagnose` 不更新权重，不重新训练，不使用测试集来选择模型。

诊断解释注意：TGM/VGM 门控分布可以证明哪些门处于接近0/1的饱和状态，**不能当作对真实世界变量的因果解释**；梯度范数来自一个验证 Batch 的反向传播，只是局部敏感性，不代表整个训练过程。完整模型选择仍需配对随机种子、相同参数和独立测试。

## 5. 官方一致性说明

`src/training/official_entry.py` 在内存中从作者已校验的 `Exp_Main.train` 源码复制训练函数，仅增加 `print(LAB_BATCH...)`，不修改 loss、optim、scheduler、dataset 或调用次序。将 `batch_interval` 设为 `0` 可禁用所有额外批次日志，得到与作者原版完全相同的 train() 实现。网页与 GPU 监控只读实验文件，且在父进程采样 GPU。`vendor/XLinear/` 的文件受到 Git SHA 校验，不能直接修改。

**验证建议：** 初次升级先 `bash run.sh check && python -m tests.test_monitor && bash run.sh pilot`，再跑一组官方 ETTh1-96，核对 MSE 是否仍接近此前已复现的 0.369467。此数值只作为已知参考，GPU 环境不同可能产生小波动。
