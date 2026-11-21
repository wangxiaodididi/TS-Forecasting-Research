"""历史条件化 Global Token；原 XLinear 的投影层、TGM、VGM、Head 均原样保留。

输入 [B,C,L]；输出 [B,C,2D]（与官方多变量 backbone 完全一致）。
这属于可解释的局部结构对照，并非已证明的新颖方法或稳定增益。
"""
import torch
from torch import nn

class HistoricalGlobalToken(nn.Module):
    def __init__(self, official_backbone: nn.Module, seq_len: int, variant: str, ratio: float):
        super().__init__()
        if variant not in ('dyn_token','stat_token'):
            raise ValueError(f'不支持的历史 Token: {variant}')
        for name in ('projection','glob_token','en_attention','ex_attention','d_model'):
            if not hasattr(official_backbone,name):
                raise RuntimeError(f'原始 backbone 缺失 {name}，请先核对官方源码版本')
        self.original = official_backbone
        self.variant = variant
        self.context_length=max(2,min(seq_len,round(seq_len*ratio)))
        d=official_backbone.d_model
        self.context_projection=nn.Linear(self.context_length if variant=='dyn_token' else 3,d,bias=False)
        self.gain_logit=nn.Parameter(torch.tensor(0.0))  # 初始预测与官方原版严格一致

    def forward(self,x):
        b,c,l=x.shape
        core=self.original
        d=core.d_model
        emb=core.projection(x)
        recent=x[:,:,-self.context_length:]
        if self.variant=='dyn_token':
            context=recent
        else:
            context=torch.stack((recent.mean(dim=-1),recent.std(dim=-1,unbiased=False),
                                  recent[:,:,-1]-recent[:,:,0]),dim=-1)
        token=core.glob_token.expand(b,-1,-1)+torch.tanh(self.gain_logit)*self.context_projection(context)
        enc=core.en_attention(torch.cat((emb,token),dim=-1))
        endogenous,global_t=enc[:,:,:d],enc[:,:,d:]
        cross=core.ex_attention(torch.cat((emb,global_t),dim=1).transpose(1,2))
        glob=cross[:,:,c:].transpose(1,2)
        return torch.cat((endogenous,glob),dim=-1)
