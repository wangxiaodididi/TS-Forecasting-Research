#!/usr/bin/env python3
"""CPU smoke tests for low-rank temporal heads and strict delayed-label TVP."""
import sys,unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'code'))
from residual_head import ResidualTemporalProjection
sys.path.insert(0,str(P))
from importlib import import_module
struct=import_module('06_tvp_structural_trials')

class ValidationTests(unittest.TestCase):
    def test_head_initial_identity_gradients(self):
        for mode in ('linear','nonlinear'):
            torch.manual_seed(42)
            base=nn.Linear(96,96)
            x=torch.randn(2,7,96)
            wrapped=ResidualTemporalProjection(base,rank=8,mode=mode)
            self.assertTrue(torch.allclose(base(x),wrapped(x),atol=1.e-7))
            optimizer=torch.optim.SGD(wrapped.parameters(),lr=.1)
            before=wrapped(x).detach().clone()
            loss=(wrapped(x)-torch.zeros_like(wrapped(x))).square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(wrapped.expand.weight.grad).all().item())
            optimizer.step()
            self.assertFalse(torch.allclose(before,wrapped(x)))
    def test_tvp_no_future_label_update(self):
        n,h,c,s=160,12,3,4
        r=np.random.default_rng(123)
        b=r.normal(size=(n,h,c));p=r.normal(size=(n,h,c,s));y=b+r.normal(size=b.shape)
        init=np.zeros(s)
        base,_,updates=struct.online_predict(b,p,y,init,[0,2],np.full(s,1e-4),h)
        altered=y.copy();altered[24]+=1000
        changed,_,u2=struct.online_predict(b,p,altered,init,[0,2],np.full(s,1e-4),h)
        self.assertTrue(np.array_equal(base[:24+h],changed[:24+h]))
        self.assertGreater(np.max(np.abs(base[24+h:]-changed[24+h:])),1e-6)
        self.assertEqual(u2,n-h)
    def test_validation_selection_independent_of_test_labels(self):
        val=P/'_test_val.npz';test=P/'_test_test.npz';out=P/'_test_out'
        r=np.random.default_rng(7);h=12;c=2;s=3
        vbase=r.normal(size=(60,h,c));vparts=r.normal(size=(60,h,c,s));vy=vbase+r.normal(scale=.5,size=vbase.shape)
        tbase=r.normal(size=(50,h,c));tparts=r.normal(size=(50,h,c,s));ty=tbase+r.normal(scale=.5,size=tbase.shape)
        try:
            np.savez(val,base=vbase,parts=vparts,y=vy)
            np.savez(test,base=tbase,parts=tparts,y=ty)
            _,manifest1=struct.run(val,test,out,base_q=1.e-4)
            np.savez(test,base=tbase,parts=tparts,y=ty+10)
            _,manifest2=struct.run(val,test,out,base_q=1.e-4)
            self.assertEqual(manifest1['chosen'],manifest2['chosen'])
        finally:
            import shutil
            for f in (val,test):f.unlink(missing_ok=True)
            shutil.rmtree(out,ignore_errors=True)
if __name__=='__main__':unittest.main(verbosity=2)
