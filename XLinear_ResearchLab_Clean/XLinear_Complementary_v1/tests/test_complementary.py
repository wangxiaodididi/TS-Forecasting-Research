"""CPU engineering tests. If pinned upstream is available, uses it; otherwise
uses an explicitly labeled interface mimic, not a paper reproduction.
"""
import importlib.util
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.models.registry import build_variant_model, available_variants
from src.models.extensions.complementary_blocks_v1 import RecentTrendResidual,TemporalFeatureAdapter


class FakeGating(nn.Module):
    def __init__(self, n, hidden):
        super().__init__()
        self.weight=nn.Sequential(nn.Linear(n,hidden),nn.ReLU(),nn.Linear(hidden,n),nn.Sigmoid())
    def forward(self,x):return x*self.weight(x)

class FakeBackbone(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        d,c=cfg.d_model,cfg.enc_in
        self.d_model=d
        self.channel=c
        self.projection=nn.Sequential(nn.Linear(cfg.seq_len,d),nn.Dropout(0))
        self.glob_token=nn.Parameter(torch.ones(1,c,d))
        self.en_attention=FakeGating(2*d,cfg.t_ff)
        self.ex_attention=FakeGating(2*c,cfg.c_ff)
    def forward(self,x):
        b,c,l=x.shape;d=self.d_model
        emb=self.projection(x)
        glob=self.glob_token.repeat(b,1,1)
        tem=self.en_attention(torch.cat([emb,glob],dim=-1))
        cross=self.ex_attention(torch.cat([emb,tem[:,:,d:]],dim=1).transpose(1,2))
        return torch.cat([tem[:,:,:d],cross[:,:,c:].transpose(1,2)],dim=-1)

class FakeOriginal(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.norm=cfg.usenorm
        self.backbone=FakeBackbone(cfg)
        self.head=nn.Sequential(nn.Dropout(0),nn.Linear(2*cfg.d_model,cfg.pred_len))
    def forward(self,x):
        if self.norm:
            mu=x.mean(dim=1,keepdim=True).detach()
            std=torch.sqrt(x.var(dim=1,keepdim=True,unbiased=False)+1e-5)
            x=(x-mu)/std
        out=self.head(self.backbone(x.permute(0,2,1))).permute(0,2,1)
        if self.norm:out=out*std+mu
        return out

def model_class():
    path=ROOT/'vendor/XLinear/models/XLinear.py'
    if not path.is_file():
        print('NOTE: pinned official upstream missing; tests run against a structure/interface mimic')
        return FakeOriginal
    content=path.read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\x00'+content).hexdigest()
    if blob!='9913ba6837a7c8658283636ab7f5727e07f2f709':
        raise RuntimeError(f'Official model file changed (git blob={blob}); refuse pinned-source check')
    spec=importlib.util.spec_from_file_location('xlinear_pinned_for_comp_tests',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod.Model

class ComplementaryChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.base=model_class()
        self.cfg=SimpleNamespace(seq_len=96,pred_len=96,d_model=16,enc_in=7,
            t_ff=32,c_ff=7,usenorm=1,features='M',embed_dropout=0.,
            head_dropout=0.,t_dropout=0.,c_dropout=0.)
    def test_registered(self):
        for name in ('official','lg_short','lg_tmix_r16','lg_tmix_r8',
                     'lg_trend_w24','lg_trend_w12','lg_combo_r16_w24'):
            self.assertIn(name,available_variants())
    def test_input_output_equality_gradients(self):
        for h in [96,192,336,720]:
            self.cfg.pred_len=h
            x=torch.randn(2,96,7)
            torch.manual_seed(999)
            baseline=self.base(self.cfg).eval()
            with torch.no_grad():reference=baseline(x)
            for name in ('lg_short','lg_tmix_r16','lg_tmix_r8','lg_trend_w24','lg_trend_w12','lg_combo_r16_w24'):
                torch.manual_seed(999)
                variant=build_variant_model(self.base,name,.25)(self.cfg).eval()
                with torch.no_grad():out=variant(x)
                self.assertEqual(tuple(out.shape),(2,h,7))
                self.assertTrue(torch.equal(reference,out),f'{name}/H{h} did not start identical')
                variant.zero_grad(set_to_none=True)
                out=variant(x)
                loss=(out**2).mean()
                loss.backward()
                if 'tmix' in name or 'combo' in name:
                    adapter=variant.backbone.en_attention
                    self.assertIsNotNone(adapter.up.weight.grad)
                    self.assertTrue(adapter.up.weight.grad.abs().sum().item()>0)
                if 'trend' in name or 'combo' in name:
                    self.assertIsNotNone(variant.recent_trend_residual.horizon_weight.grad)
                    self.assertGreater(variant.recent_trend_residual.horizon_weight.grad.abs().sum().item(),0)
    def test_nonzero_modifications_change_predictions(self):
        for name in ('lg_tmix_r16','lg_trend_w24','lg_combo_r16_w24'):
            torch.manual_seed(1)
            variant=build_variant_model(self.base,name,.25)(self.cfg).eval()
            x=torch.randn(3,96,7)
            with torch.no_grad():
                y0=variant(x)
                if 'tmix' in name or 'combo' in name:
                    variant.backbone.en_attention.up.weight.fill_(.1)
                if 'trend' in name or 'combo' in name:
                    variant.recent_trend_residual.horizon_weight.fill_(.04)
                y1=variant(x)
            self.assertGreater((y0-y1).abs().max().item(),1e-6)
    def test_module_safety(self):
        m=RecentTrendResidual(pred_len=96,recent_window=24)
        with self.assertRaises(ValueError):m(torch.randn(2,40,7))
        x=torch.randn(2,96,7)
        self.assertEqual(tuple(m(x).shape),(2,96,7))
        self.assertEqual(m(x).abs().sum().item(),0)

if __name__=='__main__':unittest.main()
