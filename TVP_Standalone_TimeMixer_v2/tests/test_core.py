import unittest,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tvp_lab.online import fit_static,replay,RecursiveTVP
from tvp_lab.extract_official import timemixer_parts,xlinear_parts

class FakeNorm(nn.Module):
    def forward(self,x,typ):
        assert typ=='denorm'
        return x*2.+3.
class MockMixer(nn.Module):
    def __init__(self):
        super().__init__();self.normalize_layers=nn.ModuleList([FakeNorm()])
    def future_multi_mixing(self,b,enc,x):
        return [b*.3,b*.7]
    def forward(self,x,m,dec,md):
        pieces=self.future_multi_mixing(x,[],[])
        return self.normalize_layers[0](sum(pieces),'denorm')
class MockBackbone(nn.Module):
    def forward(self,x):
        return torch.cat((x,x*2.),dim=-1)
class MockLinear(nn.Module):
    def __init__(self):
        super().__init__();self.feature='M';self.norm=False;self.backbone=MockBackbone();self.head=nn.Sequential(nn.Dropout(0.),nn.Linear(8,3))
    def forward(self,x):
        return self.head(self.backbone(x)).permute(0,2,1)

class Smoke(unittest.TestCase):
    def test_identity_time_mixer(self):
        x=torch.rand(3,4,5)
        p,c=timemixer_parts(MockMixer(),x,None,None)
        self.assertEqual(c.shape,(3,4,5,2))
        self.assertTrue(torch.allclose(p,3.+c.sum(-1),atol=1e-6))
    def test_xlinear_parts(self):
        m=MockLinear();x=torch.randn(3,2,4)
        p,c=xlinear_parts(m,x)
        expected=m.head[-1].bias.view(1,3,1)+c.sum(-1)
        self.assertTrue(torch.allclose(p,expected,atol=1e-6))
    def test_tvp_delayed_labels(self):
        rng=np.random.default_rng(1);n=36;H=5
        parts=rng.normal(size=(n,H,2,3));base=rng.normal(size=(n,H,2));y=base+rng.normal(size=(n,H,2))*.1
        theta=fit_static(base[:20],parts[:20],y[:20])
        before,_,log=replay(base,parts,y,theta,H)
        alt=y.copy();alt[0]+=1000
        after,_,_=replay(base,parts,alt,theta,H)
        self.assertTrue(np.allclose(before[:H],after[:H]))
        self.assertFalse(np.allclose(before[H+1],after[H+1]))
        self.assertTrue(all(i-j>=H for i,j in log))
    def test_zero_delta_identity(self):
        p=np.ones((2,3,4));c=np.random.randn(2,3,4,3)
        self.assertTrue(np.allclose(RecursiveTVP(3).forecast(p,c),p))

if __name__=='__main__':unittest.main()
