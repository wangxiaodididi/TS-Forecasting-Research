# 新增模型改造模块：不改官方训练器

假设以后要改造 `TGM` 或 `VGM` 内部的信息组织，但**保留该模块在原 Backbone 中的语义位置**。

1. 在 `src/models/extensions/` 下新建 `my_gate.py`，实现 `nn.Module`。其 `__init__(original_backbone, configs, ratio)` 应保存原模块的原始参数和必要属性。
2. 扩展模块的 `forward(x)` 要维持 `[B,C,L] -> [B,C,2D]`，保证 Head 完全不变。若替换的不是整个 backbone，也可让 wrapper 调用原始 `en_attention`、`ex_attention` 并仅对一个输入变换。
3. 在 `src/models/registry.py` 中添加一个工厂函数，例如：

```python
from src.models.extensions.my_gate import MyGate

def _my_gate(backbone, configs, ratio):
    return MyGate(backbone, configs, ratio)

VARIANTS['my_gate'] = _my_gate
```

4. 在 `configs/experiment.yaml` 的 `experiment.variants` 中加入 `my_gate`，例如 `[official, dyn_token, my_gate]`。
5. 执行 `bash run.sh check`（目前检查现有两个 Token；新增模块应同时在 `tests/selftest.py` 增加自己的检查）、`bash run.sh plan`、`bash run.sh pilot`。
6. 如果验证集表现合理，再运行 `bash run.sh full`。保留失败结果、参数量和配对随机种子结果。

强烈建议一次只改变一个功能部位，并保留 `official` 原始模型。对于变量交互新算子，优先验证通道维度与内存复杂度，尤其是高维 Electricity 数据。

## 目前三个模型的代码对应关系

- `official`：没有新层，完全按原作者训练。
- `dyn_token`：`global_token.py` 读取最近 K 个历史点，动态调整 Global Token。
- `stat_token`：同一文件，用均值、标准差、首末差压缩最近 K 个点。

未经独立验证不能称作原论文精度更好。现有 ETTh1-96 的官方复现单次结果约 0.369467 MSE，新的改造实验仍须在服务器上运行后才有新结果。
