"""CPU 结构自检；不消耗 CUDA，也不需要原来的 checkpoint。

检查：与官方初始严格对齐、梯度能流入三个可学习时间成分系数、跨变量
权重随样本变化、静态对照不随样本变化、拒绝泄露未来真实值和错误输入维度。
"""
from types import SimpleNamespace
import torch
from torch import nn
from src.models.registry import build_variant_model

class FakeGate(nn.Module):
    def __init__(self, n: int, hidden: int):
        super().__init__()
        self.weight=nn.Sequential(nn.Linear(n,hidden),nn.ReLU(),nn.Dropout(0.0),
                                  nn.Linear(hidden,n),nn.Sigmoid())
    def forward(self,x):return x*self.weight(x)

class FakeBackbone(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.d_model=c.d_model;self.channel=c.enc_in
        self.projection=nn.Sequential(nn.Linear(c.seq_len,c.d_model),nn.Dropout(0.0))
        self.glob_token=nn.Parameter(torch.ones(1,c.enc_in,c.d_model))
        self.en_attention=FakeGate(2*c.d_model,c.t_ff)
        self.ex_attention=FakeGate(2*c.enc_in,c.c_ff)
    def forward(self,x):
        b,c,_=x.shape;d=self.d_model
        emb=self.projection(x)
        temporal=self.en_attention(torch.cat((emb,self.glob_token.expand(b,-1,-1)),dim=-1))
        orig,glob=temporal[:,:,:d],temporal[:,:,d:]
        cross=self.ex_attention(torch.cat((emb,glob),dim=1).transpose(1,2))
        return torch.cat((orig,cross[:,:,c:].transpose(1,2)),dim=-1)

class FakeOfficial(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.backbone=FakeBackbone(config)
        self.head=nn.Linear(2*config.d_model,config.pred_len)
    def forward(self,x):return self.head(self.backbone(x.transpose(1,2))).transpose(1,2)


def main():
    cfg=SimpleNamespace(seq_len=96,pred_len=96,enc_in=7,d_model=32,t_ff=96,c_ff=7,
                        features='M',data='ETTh1')
    torch.manual_seed(2025)
    initial_state=torch.random.get_rng_state()
    original=FakeOfficial(cfg)
    x=torch.randn(6,96,7)
    y=original(x)
    results=[]
    for mode in ('cc_components','cc_raw','cc_static'):
        torch.random.set_rng_state(initial_state)
        Model=build_variant_model(FakeOfficial,mode,0.25)
        model=Model(cfg)
        pred=model(x)
        assert torch.equal(pred,y),(mode,'初始预测必须与原模型逐元素完全一致')
        assert model.backbone.original.en_attention is not None
        assert model.backbone.original.ex_attention is not None
        routes,_=model.backbone._routing(x.transpose(1,2))
        assert routes.shape==(6,7,7)
        assert torch.all(routes.diagonal(dim1=1,dim2=2)==0)
        assert torch.allclose(routes.sum(dim=-1),torch.ones_like(routes.sum(dim=-1)))
        assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in original.parameters())+52
        model.zero_grad(set_to_none=True)
        loss=(pred-torch.randn_like(pred)).square().mean()
        loss.backward()
        assert model.backbone.peer_logits.grad is not None
        assert model.backbone.peer_logits.grad.abs().sum()>0, mode
        if mode!='cc_static':
            assert model.backbone.component_weights.grad is not None
            assert model.backbone.component_weights.grad.abs().sum()>0, mode
        else:
            assert model.backbone.component_weights.grad is None
        assert model.backbone.original.en_attention.weight[0].weight.grad is not None
        if mode!='cc_static':
            with torch.no_grad():model.backbone.component_weights[:]=torch.tensor([0.8,1.0,0.8])
            r,_=model.backbone._routing(x.transpose(1,2))
            assert not torch.allclose(r[0],r[1]),mode
        else:
            with torch.no_grad():model.backbone.peer_logits[0,1]=0.3
            r,_=model.backbone._routing(x.transpose(1,2))
            assert torch.equal(r[0],r[1]),mode
        with torch.no_grad():
            changed=model(x)
            assert not torch.allclose(changed,y),mode
        results.append((mode,float(loss.detach()),int(sum(p.numel() for p in model.parameters()))))
    for mode,loss,count in results:
        print(f'PASS {mode}: CPU forward/backward, loss={loss:.6f}, params={count}')
    print('CC-XLINEAR CPU TEST PASS')

if __name__=='__main__':main()
