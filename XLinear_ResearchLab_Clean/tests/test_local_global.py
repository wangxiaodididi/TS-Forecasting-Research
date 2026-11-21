"""CPU structural tests. Actual pinned official source required for official-integration test."""
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from src.models.registry import build_variant_model

ROOT = Path(__file__).resolve().parents[1]

def config(h):
    return SimpleNamespace(seq_len=96, pred_len=h, d_model=32, enc_in=7,
        t_ff=64, c_ff=7, usenorm=1, embed_dropout=0.0,
        head_dropout=0.0, t_dropout=0.0, c_dropout=0.0, features='M')


class ReferenceXLinear(nn.Module):
    """Minimal mock of upstream architecture for testing wrapper injection only."""
    def __init__(self, cfg):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.projection = nn.Sequential(nn.Linear(cfg.seq_len,cfg.d_model),nn.Dropout(cfg.embed_dropout))
        self.backbone.tgm = nn.Sequential(nn.Linear(cfg.d_model,cfg.d_model), nn.Sigmoid())
        self.backbone.vgm = nn.Sequential(nn.Linear(cfg.d_model,cfg.d_model), nn.Sigmoid())
        self.head = nn.Sequential(nn.Dropout(cfg.head_dropout), nn.Linear(2*cfg.d_model,cfg.pred_len))
    def forward(self,x):
        emb=self.backbone.projection(x.permute(0,2,1))
        return self.head(torch.cat([emb*self.backbone.tgm(emb),emb*self.backbone.vgm(emb)],dim=-1)).permute(0,2,1)


def evaluate(cls):
    for h in (96,192,336,720):
        cfg=config(h)
        torch.manual_seed(23)
        ref=cls(cfg).eval()
        x=torch.randn(2,96,7)
        for v in ('lg_linear','lg_short','lg_dual'):
            torch.manual_seed(23)
            altered=build_variant_model(cls,v,0.25)(cfg).eval()
            with torch.no_grad():
                a=ref(x);b=altered(x)
            assert a.shape==b.shape==(2,h,7)
            assert torch.equal(a,b),(v,h,float((a-b).abs().max()))
            altered.train()
            altered.zero_grad(set_to_none=True)
            altered(x).square().mean().backward()
            p=altered.backbone.projection.residual.weight
            assert p.grad is not None and float(p.grad.abs().sum()) > 0,(v,h)
            print('PASS',v,'H',h,'identity, shape, residual gradient, added params',altered.backbone.projection.extra_parameters())


def main():
    torch.set_num_threads(2)
    evaluate(ReferenceXLinear)
    file=ROOT/'vendor/XLinear/models/XLinear.py'
    if not file.is_file():
        print('SKIP real official integration: pinned vendor/XLinear is absent')
        return
    data=file.read_bytes()
    sha=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
    assert sha=='9913ba6837a7c8658283636ab7f5727e07f2f709',f'Upstream model git blob mismatch: {sha}'
    sys.path.insert(0,str(ROOT/'vendor/XLinear'))
    from models.XLinear import Model
    # Official model test with realistic ETTh1 dimensions; do not train.
    for h in (96,192,336,720):
        cfg=config(h);cfg.d_model=128;cfg.t_ff=512
        torch.manual_seed(23)
        original=Model(cfg).eval()
        x=torch.randn(2,96,7)
        for v in ('lg_linear','lg_short','lg_dual'):
            torch.manual_seed(23)
            modified=build_variant_model(Model,v,0.25)(cfg).eval()
            with torch.no_grad():
                a=original(x);b=modified(x)
            assert torch.equal(a,b),(v,h,float((a-b).abs().max()))
            modified.train();modified.zero_grad(set_to_none=True)
            modified(x).square().mean().backward()
            assert modified.backbone.projection.residual.weight.grad is not None
            assert modified.backbone.projection.residual.weight.grad.abs().sum() > 0
            print('PASS pinned official XLinear',v,'H',h)
    print('PASS ALL pinned official integration tests')

if __name__=='__main__':
    main()
