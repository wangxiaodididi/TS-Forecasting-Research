"""轻量 CPU 检查：初始与官方预测严格一致、系数有梯度、变量与未来段均可学习。"""
import torch
from types import SimpleNamespace
from src.models.extensions.seasonal_residual import SeasonalResidualAdapter
from src.models.registry import build_variant_model

class ExampleOfficial(torch.nn.Module):
    """接口兼容官方 Model 的 CPU 假模型；不冒充官方源码复现。"""
    def __init__(self, c):
        super().__init__()
        self.layer = torch.nn.Linear(c.seq_len, c.pred_len)
    def forward(self, x):
        return self.layer(x.transpose(1,2)).transpose(1,2)


def main():
    torch.manual_seed(2025)
    conf=SimpleNamespace(seq_len=96,pred_len=96,enc_in=7,features='M',data='ETTh1')
    a=ExampleOfficial(conf)
    b_cls=build_variant_model(ExampleOfficial,'csra',ratio=0.25)
    torch.manual_seed(2025)
    b=b_cls(conf)
    x=torch.randn(8,96,7)
    p=a(x)
    q=b(x)
    assert torch.equal(p,q), '零初始化未与官方输出完全一致'
    assert b.csra.coefficients.shape==(7,4,3)
    target=torch.randn_like(q)
    loss=(q-target).square().mean()
    loss.backward()
    assert b.csra.coefficients.grad is not None
    assert b.csra.coefficients.grad.abs().sum()>0
    assert b.layer.weight.grad is not None
    b.csra.coefficients.data[0,0,0]=0.2
    y=b(x)
    assert not torch.equal(p,y), '修改系数后预测没有改变'
    print('CSRA CPU TEST PASS: output identity / nonzero gradients / selective correction')

if __name__=='__main__':main()
