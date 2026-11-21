# LG-Embedding：XLinear局部—全局互补历史编码，v1

## 模型结构

**保留**：官方 XLinear 的输入标准化、`projection` 中的全局线性映射、Global Token、TGM、VGM、Head、输出反标准化。

**仅改造**：`Forcast_multi.projection`，把原来 `emb = original_linear(x)` 改为 `emb = original_linear(x) + LocalPattern(x)`。

- `lg_linear`：新增一个零初始化的普通线性残差映射 `Linear(L,D)`，用作**额外容量对照**，不能与主方案视为参数量匹配。
- `lg_short`：通道独立的 `Conv1d(1,2,k=3) → GELU → 局部区间均值/最大池化 → Linear → D`。
- `lg_dual`：与上一种相同，但同时使用 `k=3` 和 `k=9` 两个局部感受野，作为主候选。

局部卷积共享于所有变量（channel-independent），全局线性投影的原有权重和计算路径不变。各尺度分别提取短局部形态，并将其按固定顺序的8个区间汇总后映射至D维。这个模块提供明确的局部非线性归纳偏置，不等同于证明原线性投影无法隐式学习局部模式。

新增分支的最后线性层的权重和偏置初始化为0；因此初始化时模型输出必须与原版 **逐元素完全相等**。优化开始后才学习增量残差；在首次反向传播中，最后一层先获得梯度，卷积层梯度会在其后更新阶段出现。

## 它可能解决什么问题？

原始单次 `Linear(L,D)` 对每个通道的全部历史进行整体投影。局部异常、突变、短时动量等可以被原架构隐式近似，但没有单独的短窗卷积特征提取和分段池化结构。提出的局部残差希望给TGM送入保留局部非线性形状的补充表征。

**研究假设**：如果多变量时间序列含局部非线性形态，`lg_dual` 在官方协议、公开基准、相同种子下可能带来可重复预测增益。不是一个现成结果。

## 在AutoDL现有XLinear_ResearchLab_Clean项目中使用

把 ZIP 解压到**该项目根目录**。三个脚本在项目根目录，而非独立子目录。必须使用此前完成官方复现的同一个Python环境。

```
cd /root/autodl-tmp/XLinear_ResearchLab_Clean
unzip -o XLinear_LGEmbed_v1.zip -d .
bash 01_CPU_结构测试.sh
```

此脚本不使用GPU。对当前项目中的 `vendor/XLinear/models/XLinear.py` 核对Git blob SHA (`9913ba6837a7c8658283636ab7f5727e07f2f709`)；若不存在则明确输出SKIP，不会把模拟模型检查伪装成官方测试。

完成结构检查后，可执行仅用于工程检查的两轮试跑（需要GPU）：

```
MODE=pilot bash 02_GPU_小范围试跑.sh
```

实际脚本默认即可，无需设置MODE；`MODE=pilot` 只是说明。正式训练：

```
bash 03_GPU_正式复现.sh
```

设置更多种子或预测长度：

```
SEEDS=2025,2026 HORIZONS=96,192 bash 03_GPU_正式复现.sh
```

默认同一实验中包含 `official,lg_linear,lg_short,lg_dual`，复用现有官方训练入口、优化器、early stopping、数据划分、训练配置、FP32、metrics；不修改官方模型源码，不覆盖原有checkpoint。

这两个GPU脚本读取 `DATASETS/HORIZONS/SEEDS/VARIANTS/PYTHON` 环境变量。只有在项目 `configs/experiment.yaml` 配置了对应数据集及预测长度时才支持；没有配置就会明确报错。

## 安装与卸载

`python ops/install_lg.py` 仅向 `src/models/registry.py` **追加带标记的注册段**，并预先备份。重复执行不会修改；已有的HCF/CC注册可继续存在。解除注册：`python ops/install_lg.py --remove`。由于该操作也备份当前注册表，原文件不会无备份覆盖。

## 审稿与新颖性界限

这不是宣称“第一种局部—全局编码”。ModernTCN、PatchTST以及2026年的PatchFusionMLP/GCMNet等已有局部和多尺度特征学习思想。与它们相比，本方案的可研究点仅是：在XLinear保留原轻量门控主干的前提下，零初始化地引入较小的局部非线性残差。如果无法在完整训练中获得可靠增量，不应单独宣称方法创新。

主候选 `lg_dual`；消融为 `lg_short`；容量对照 `lg_linear`。建议先ETTh1-96，后Weather-96和ETTh1-192，之后确认是否有必要扩展96/192/336/720。正式模型比较使用多个随机种子和配对差异，不以CPU smoke结果或测试集择参。

## 补充：不使用GPU的ETTh1小样本训练

先运行CPU结构测试，再在已经完成 `bash run.sh setup` 的服务器项目下：

```
bash 02_CPU_小样本试验.sh
```

默认取原官方ETTh1训练时间段内512个均匀抽样窗口和验证时间段内最前面256个窗口，只运行2个epoch。数据归一化仅拟合训练时间段，测试集保持完全未使用。这个CPU工程试跑用于判断端到端训练通不通，**不是作者30-epoch早停/大训练集的官方复现结果**，也不用于筛选更优模型或投稿数值。
