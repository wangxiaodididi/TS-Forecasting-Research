"""Offline test: counterfactual switches direction and keeps parameter count.

This uses torch-only stubs for official submodules; full official integration test
is separate and must be run after prepare.py on the AutoDL machine.
"""
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace
import torch
from torch import nn

PATH=Path(__file__).resolve().parents[1]/'model_code'/'TimeMixerReverse.py'

class Season(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.ops=nn.ModuleList([nn.Linear(c.seq_len//2**i,c.seq_len//2**(i+1)) for i in range(c.down_sampling_layers)])
    def forward(self,ls):
        out=[ls[0]]
        for i,op in enumerate(self.ops):
            out.append(ls[i+1]+op(out[-1]))
        return out

class Trend(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.ops=nn.ModuleList([nn.Linear(c.seq_len//2**(i+1),c.seq_len//2**i) for i in range(c.down_sampling_layers)])
    def forward(self,ls):
        out=list(ls)
        for i in reversed(range(len(ls)-1)):
            out[i]=out[i]+self.ops[i](out[i+1])
        return out

class PDM(nn.Module):
    def __init__(self,c):
        super().__init__();self.mixing_multi_scale_season=Season(c);self.mixing_multi_scale_trend=Trend(c)
    def forward(self,inputs):
        ss=self.mixing_multi_scale_season(inputs)
        tt=self.mixing_multi_scale_trend(inputs)
        return [a+b for a,b in zip(ss,tt)]

class Original(nn.Module):
    def __init__(self,c):
        super().__init__();self.pdm_blocks=nn.ModuleList([PDM(c) for _ in range(c.e_layers)])
    def forward(self,x):
        for p in self.pdm_blocks:x=p(x)
        return x

pkg=types.ModuleType('models');pkg.__path__=[]
mod=types.ModuleType('models.TimeMixer');mod.MultiScaleSeasonMixing=Season;mod.MultiScaleTrendMixing=Trend;mod.PastDecomposableMixing=PDM;mod.Model=Original
pkg.TimeMixer=mod
sys.modules['models']=pkg;sys.modules['models.TimeMixer']=mod
spec=importlib.util.spec_from_file_location('models.TimeMixerReverse',str(PATH))
r=importlib.util.module_from_spec(spec);sys.modules[spec.name]=r;spec.loader.exec_module(r)

for e_layers in (1,2,3):
    c=SimpleNamespace(seq_len=96,down_sampling_layers=3,e_layers=e_layers)
    torch.manual_seed(123)
    a=Original(c)
    torch.manual_seed(123)
    b=r.Model(c)
    # Pairwise-identical parameter initialization with branch assignment swapped.
    for orig_block,reverse_block in zip(a.pdm_blocks,b.pdm_blocks):
        for pa0,pb0 in zip(orig_block.mixing_multi_scale_season.parameters(),reverse_block.mixing_multi_scale_trend.parameters()):
            assert torch.equal(pa0,pb0)
        for pa0,pb0 in zip(orig_block.mixing_multi_scale_trend.parameters(),reverse_block.mixing_multi_scale_season.parameters()):
            assert torch.equal(pa0,pb0)
    pa=sum(p.numel() for p in a.parameters());pb=sum(p.numel() for p in b.parameters())
    assert pa==pb,(pa,pb)
    assert isinstance(b.pdm_blocks[0].mixing_multi_scale_season,Trend)
    assert isinstance(b.pdm_blocks[0].mixing_multi_scale_trend,Season)
    xx=[torch.randn(2,4,96//2**i,requires_grad=True) for i in range(4)]
    y=b(xx);assert [tuple(z.shape) for z in y]==[tuple(z.shape) for z in xx]
    sum(z.square().mean() for z in y).backward()
    assert all(p.grad is not None for p in b.parameters())
    print('[PASS] layers=',e_layers,'parameters=',pa,'full gradients=True')
