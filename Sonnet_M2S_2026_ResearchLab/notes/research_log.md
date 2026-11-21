# 研究记录（2026-11-18）

- 完成：核验 AAAI 2026 正式论文表 1（Sonnet：ENER H24=0.3621 / H48=0.4120；ETTh1 H96=0.2145 / H192=0.2330；指标为第 H 步 MAE）。
- 完成：核验官方源码（`Model`、`AdaptiveWavelet`、`CoherenceAttention`、`KoopmanLayer`、`CustomDartsDataset`、Lightning 评价、Hydra 模型和实验 YAML），确认任务为 M→S。
- 发现：变量注意力 `einsum` 为对角迹缩放的等价表达式；已在本地 PyTorch 2.10 CPU 对代数恒等式运行验证，最大绝对误差 0.0（该数值**不是**官方整网 forward 验证）。
- 发现：官方 `_scaling` 的 `dataset_split` 比例拟合边界与数字划分边界不一致。要进行官方原样复现和修正协议的区分。
- 完成：制作原版代码下载、合成数据 forward/backward、真实数据少量 batch smoke、原始 GPU 训练、源码审计、指标记录入口。
- 尚未完成：在 AutoDL 真实下载依赖/数据及运行整网。没有声称成功复现或改进准确率。
- 下一关口：远端运行 `00_setup.sh`、`01_smoke_cpu.sh`、`02_small_realdata.sh energy`；根据**真实日志**确认问题，再启动 H24 GPU 全量复现。
- 许可：setup.py 写了 MIT 分类器，但独立 LICENSE 未查到。暂以不再分发官方代码的方式工作，进一步确认修改后的分发许可。
