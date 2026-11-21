"""CPU 结构一致性、局部改造梯度、配置矩阵及数据校验测试。不会启动 GPU。"""
import sys
from types import SimpleNamespace
import torch
from src.core.config import read_config,make_jobs
from src.models.registry import build_variant_model
from src.runtime.bootstrap import UPSTREAM,verify_source,prepare_datasets


def test_model(XLinear,cfg,dataset,horizon,variant):
    p=cfg['profiles'][dataset][str(horizon)]
    args=SimpleNamespace(features='M',seq_len=96,pred_len=horizon,enc_in=p['channels'],
        d_model=p['d_model'],t_ff=p['t_ff'],c_ff=p['c_ff'],usenorm=1,
        t_dropout=p['t_dropout'],c_dropout=p['c_dropout'],
        head_dropout=p['head_dropout'],embed_dropout=p['embed_dropout'])
    torch.manual_seed(2025)
    ref=XLinear.Model(args).cpu()
    torch.manual_seed(2025)
    newcls=build_variant_model(XLinear.Model,variant,cfg['runtime']['context_ratio'])
    mod=newcls(args).cpu()
    ref.eval();mod.eval()
    torch.manual_seed(1337)
    x=torch.randn(2,96,p['channels'])
    with torch.no_grad():
        a,b=ref(x),mod(x)
    delta=(a-b).abs().max().item()
    assert delta<1e-7,(dataset,horizon,variant,delta)
    assert a.shape == b.shape == (2,horizon,p['channels'])
    opt=torch.optim.Adam(mod.parameters(),lr=0.0005)
    for i in range(2):
        opt.zero_grad()
        loss=(mod(x)-torch.randn_like(a)).square().mean()
        loss.backward();opt.step()
    assert mod.backbone.context_projection.weight.grad is not None
    assert torch.isfinite(mod.backbone.context_projection.weight.grad).all()
    n0=sum(x.numel() for x in ref.parameters())
    n1=sum(x.numel() for x in mod.parameters())
    print(f'PASS {dataset} H{horizon} {variant}: 初始一致 max_err={delta:.1g}, 参数量={n0}->{n1}')


def main():
    cfg=read_config()
    print('配置实验数:',len(make_jobs(cfg,'full')))
    verify_source(cfg)
    hashes=prepare_datasets(cfg)
    sys.path.insert(0,str(UPSTREAM))
    from models import XLinear
    for ds,h in [('ETTh1',96),('Weather',96)]:
        for variant in ('dyn_token','stat_token'):
            test_model(XLinear,cfg,ds,h,variant)
    print('PASS: 所有 CPU 自检已通过。该测试不构成 GPU 预测精度证据。')

if __name__=='__main__':main()
