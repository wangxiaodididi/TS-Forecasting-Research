"""Independent CPU tests; inject a minimal official-interface GTR stub.

This validates the extension semantics without pretending to reproduce the full
GTR optimizer/data pipeline on the local machine.
"""
import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'extensions'))


class FakeGTRBase(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.seq_len=cfg.seq_len
        self.cycle_len=cfg.cycle
        self.Q=nn.Parameter(torch.zeros(cfg.cycle,cfg.enc_in))
        # Q reference exactly has a learned Linear(T,T) in upstream.
        self.GTR=nn.Module()
        self.GTR.linear=nn.Linear(cfg.seq_len,cfg.seq_len)
        self.input_proj=nn.Linear(cfg.seq_len,cfg.pred_len)

    def forward(self,x,cycle_index):
        ind=(cycle_index.reshape(-1,1)+torch.arange(self.seq_len)[None,:])%self.cycle_len
        q=self.Q[ind].transpose(1,2)
        pred=self.input_proj(x.transpose(1,2)+self.GTR.linear(q)).transpose(1,2)
        return pred

fake=types.ModuleType('models')
mod=types.ModuleType('models.GTR')
mod.Model=FakeGTRBase
fake.GTR=mod
sys.modules['models']=fake
sys.modules['models.GTR']=mod
from models import GTR
from models import GTR as GTR_unused
from models import GTR as _
# Register extension modules under 'models' import namespace.
fake.__path__=[str(ROOT/'extensions/models')]
Core=importlib.import_module('models.GTRPhaseCore')
Simple=importlib.import_module('models.GTRPhaseInnovation').Model
Adaptive=importlib.import_module('models.GTRPhaseAdaptive').Model


class Phase2ModelsTest(unittest.TestCase):
    def setUp(self):
        self.cfg=SimpleNamespace(seq_len=96,pred_len=192,enc_in=7,cycle=96,use_revin=1)
        self.x=torch.randn(3,96,7)
        self.cycle=torch.tensor([0,10,23])

    def test_zero_init_equals_core(self):
        for cls in [Simple,Adaptive]:
            obj=cls(self.cfg)
            a=obj(self.x,self.cycle)
            b=obj.base(self.x,self.cycle)
            self.assertEqual(tuple(a.shape),(3,192,7))
            self.assertTrue(torch.allclose(a,b,atol=1e-7,rtol=1e-7))

    def test_backward_and_gtr_Q_remains_trainable(self):
        for cls in [Simple,Adaptive]:
            obj=cls(self.cfg)
            obj(self.x,self.cycle).square().mean().backward()
            self.assertTrue(torch.isfinite(obj.innovation_head.weight.grad).all())
            self.assertIsNotNone(obj.base.Q.grad)
            self.assertTrue(torch.isfinite(obj.base.Q.grad).all())

    def test_explain_identity_and_finite(self):
        for cls in [Simple,Adaptive]:
            obj=cls(self.cfg)
            with torch.no_grad():o=obj.explain(self.x,self.cycle)
            self.assertTrue(torch.allclose(o['prediction'],o['gtr_prediction']+o['innovation_correction']))
            for k in ('phase_fit_r2','correction_gain','innovation_energy'):
                self.assertEqual(tuple(o[k].shape),(3,7))
                self.assertTrue(torch.isfinite(o[k]).all())
            self.assertTrue((o['phase_fit_r2']>=0).all())
            self.assertTrue((o['phase_fit_r2']<=1).all())

    def test_reference_fit(self):
        torch.manual_seed(123)
        q=torch.randn(2,5,96)
        x=2*q
        e,fit,r2=Core.phase_calibrated_innovation(x,q)
        self.assertGreater(r2.min().item(),0.99)
        self.assertLess(e.square().mean().item(),1e-5)
        const=torch.zeros_like(q)
        e,fit,r2=Core.phase_calibrated_innovation(x,const)
        self.assertAlmostEqual(r2.mean().item(),0.0,places=6)
        self.assertTrue(torch.isfinite(e).all())

    def test_adaptive_gain_is_bounded_and_begins_at_one(self):
        m=Adaptive(self.cfg)
        with torch.no_grad():o=m.explain(self.x,self.cycle)
        self.assertTrue(torch.allclose(o['correction_gain'],torch.ones_like(o['correction_gain'])))
        m.gain_head.bias.data.fill_(1.)
        with torch.no_grad():o=m.explain(self.x,self.cycle)
        self.assertTrue((o['correction_gain']>0.5).all() and (o['correction_gain']<1.5).all())

    def test_phase_reference_changes_innovation_after_training(self):
        m=Simple(self.cfg)
        m.innovation_head.weight.data.uniform_(-0.02,.02)
        with torch.no_grad():
            # One nonzero cyclic shape and another; fixed local history.
            m.base.Q.copy_(torch.randn_like(m.base.Q))
            a=m.explain(self.x,self.cycle)['innovation_correction'].clone()
            m.base.Q.copy_(torch.randn_like(m.base.Q))
            b=m.explain(self.x,self.cycle)['innovation_correction'].clone()
        self.assertGreater((a-b).abs().max().item(),1e-7)

if __name__=='__main__':unittest.main()
