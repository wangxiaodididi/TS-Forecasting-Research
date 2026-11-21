import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import paloma_v8 as v8

class BaseDummy:
    def __init__(self,c):
        import torch
        from torch import nn
        # Note, patched module inherits nn.Module in test below
        super().__init__()
        self.seq_len=c.seq_len;self.pred_len=c.pred_len
        self.cycle_len=c.cycle;self.enc_in=c.enc_in;self.d_model=c.d_model
        self.use_revin=c.use_revin;self.core=nn.Linear(c.seq_len,c.pred_len)
    def forward(self,x,idx):
        if self.use_revin:
            mean=x.mean(1,keepdim=True);std=(x.var(1,keepdim=True)+1e-5).sqrt();z=(x-mean)/std
        else:mean=0;std=1;z=x
        y=self.core(z.permute(0,2,1)).permute(0,2,1)
        return y*std+mean

class V8Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        class Dummy(torch.nn.Module,BaseDummy):
            def __init__(self,c):
                torch.nn.Module.__init__(self)
                BaseDummy.__init__(self,c)
            def forward(self,x,idx):return BaseDummy.forward(self,x,idx)
        cls.stub=types.ModuleType('v8_stub')
        cls.stub.Model=Dummy
        cls.C=SimpleNamespace(seq_len=96,pred_len=96,enc_in=7,cycle=24,d_model=256,dropout=.5,use_revin=1,individual=0)

    def test_patch_forward_backward(self):
        import torch
        v8.smoke_module(self.stub)

    def test_shapes_both_horizons_and_cycles(self):
        import torch
        for h in (96,336):
            for p in (24,96):
                c=SimpleNamespace(**{**self.C.__dict__,'pred_len':h,'cycle':p})
                x=torch.randn(3,96,7);idx=torch.arange(3)
                os.environ[v8.ENV_VARIANT]='original';torch.manual_seed(1337)
                original=self.stub.Model(c); expected=original(x,idx)
                for kind in ('trend_skip','structure_skip'):
                    os.environ[v8.ENV_VARIANT]=kind;torch.manual_seed(1337)
                    model=self.stub.Model(c)
                    y=model(x,idx)
                    self.assertEqual(y.shape,(3,h,7))
                    self.assertTrue(torch.equal(y,expected))
                    self.assertTrue(torch.isfinite(y).all())
        os.environ.pop(v8.ENV_VARIANT,None)

    def test_parse_and_collect(self):
        with tempfile.TemporaryDirectory() as td:
            w=Path(td);f=w/'runs/paloma_path_diagnostics_v7/all_tasks_summary.csv';f.parent.mkdir(parents=True)
            f.write_text('dataset,horizon,split,original_recalculated_mse,original_recalculated_mae,match_status\nETTh1,96,val,0.65,0.52,validation_diagnostics\nETTh1,96,test,0.36,0.37,matched\n',encoding='utf-8')
            out=w/'runs/paloma_validation_guided_v8';m=out/'paper/ETTh1/h96/trend_skip/metadata.json';m.parent.mkdir(parents=True)
            m.write_text(json.dumps(dict(dataset='ETTh1',horizon=96,variant='trend_skip',status='ok',returncode=0,mse=0.60,mae=0.50)),encoding='utf-8')
            outfile=v8.collect(out,w)
            s=outfile.read_text(encoding='utf-8-sig')
            self.assertIn('7.692308',s)
            self.assertNotIn('0.36',s)

    def test_loss_patch_isolates_validation(self):
        import torch
        ns={'WeightedLoss':type('WeightedLoss',(),{'__call__':lambda self,p,t:(p-t).square().mean()}),
            'Exp_Main':type('Exp_Main',(),{'vali':lambda self,d,l,criterion:criterion(torch.tensor([2.]),torch.tensor([0.]))})}
        exec(v8.LOSS_PATCH,ns)
        c=ns['WeightedLoss']();e=ns['Exp_Main']()
        os.environ[v8.ENV_LOSS]='uniform_l1'
        self.assertAlmostEqual(c(torch.tensor([2.]),torch.tensor([0.])).item(),2.)
        self.assertAlmostEqual(e.vali(None,None,c).item(),4.)
        os.environ.pop(v8.ENV_LOSS,None)

    def test_refuses_unpinned_sources(self):
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/'upstream';(src/'models').mkdir(parents=True)
            (src/'models/Paloma.py').write_text('print("not official")',encoding='utf-8')
            with self.assertRaises(RuntimeError):v8.patch_private(src,Path(td)/'private')

if __name__=='__main__':unittest.main()
