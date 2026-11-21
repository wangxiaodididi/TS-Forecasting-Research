"""CPU checks for diagnostic path equivalence, gradient flow and artifact outputs."""
from pathlib import Path
import importlib.util
import tempfile
import unittest
import numpy as np
import torch
from torch import nn

FILE=Path(__file__).resolve().parents[1]/'paloma_v7_diagnose.py'
spec=importlib.util.spec_from_file_location('paloma_v7_diagnose', FILE)
v7=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v7)

class FakePaloma(nn.Module):
    """Mathematically mirrors the official forward interface for the diagnostic unit tests."""
    def __init__(self,seq_len=12,pred_len=8,enc_in=7,d_model=10,cycle=4):
        super().__init__()
        self.seq_len=seq_len;self.pred_len=pred_len;self.enc_in=enc_in
        self.cycle_len=cycle;self.d_model=d_model;self.use_revin=True
        self.Q=nn.Parameter(torch.randn(cycle,enc_in))
        self.phase=nn.Parameter(torch.randn(cycle,enc_in))
        self.amplitude=nn.Parameter(torch.randn(cycle,enc_in))
        self.phase_fusion=nn.Linear(seq_len,d_model)
        self.amplitude_fusion=nn.Linear(seq_len,d_model)
        self.input_proj=nn.Linear(seq_len,d_model)
        self.model=nn.Sequential(nn.Linear(d_model,d_model),nn.GELU())
        self.output_proj=nn.Sequential(nn.Dropout(0.5),nn.Linear(d_model,pred_len))
    def forward(self,x,cycle_index):
        mean=x.mean(1,keepdim=True);var=x.var(1,keepdim=True)+1e-5
        xn=(x-mean)/var.sqrt();x_in=xn.transpose(1,2)
        idx=(cycle_index.view(-1,1)+torch.arange(self.seq_len).view(1,-1))%self.cycle_len
        r=x_in-self.Q[idx].transpose(1,2)
        hidden=self.model(self.phase_fusion(r)+self.amplitude_fusion(r))
        out=self.output_proj(hidden+self.input_proj(x_in)).transpose(1,2)
        return out*var.sqrt()+mean

# Override path-only behavior of fake module for the test. Fake uses nn.Linear with one arg.
class AdaptedFake(FakePaloma):
    def __init__(self):
        super().__init__()
        class Wrap(nn.Module):
            def __init__(self,linear):super().__init__();self.linear=linear
            def forward(self,residual,phase):return self.linear(residual)
        self.phase_fusion=Wrap(self.phase_fusion)
        self.amplitude_fusion=Wrap(self.amplitude_fusion)
    def forward(self,x,cycle_index):
        mean=x.mean(1,keepdim=True);var=x.var(1,keepdim=True)+1e-5
        xn=(x-mean)/var.sqrt();x_in=xn.transpose(1,2)
        idx=(cycle_index.view(-1,1)+torch.arange(self.seq_len).view(1,-1))%self.cycle_len
        r=x_in-self.Q[idx].transpose(1,2)
        p=self.phase_fusion(r, self.phase[idx].transpose(1,2))
        a=self.amplitude_fusion(r, self.amplitude[idx].transpose(1,2))
        hidden=self.model(p+a)
        out=self.output_proj(hidden+self.input_proj(x_in)).transpose(1,2)
        return out*var.sqrt()+mean

class TestV7(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.model=AdaptedFake().eval()
        self.x=torch.randn(4,12,7)
        self.cycles=torch.tensor([0,1,2,3])
        self.y=torch.randn(4,8,7)
    def test_exact_forward_and_path_gradients(self):
        preds,ex=v7.path_forward(self.model,self.x,self.cycles,check=True)
        self.assertTrue(torch.equal(preds['full'],self.model(self.x,self.cycles)))
        self.assertEqual(len(preds),6)
        self.assertEqual(tuple(preds['no_phase'].shape),(4,8,7))
        err=(preds['full']-self.y).square().mean()
        grads=torch.autograd.grad(err,[ex['phase'],ex['amplitude'],ex['skip'],ex['hidden']])
        self.assertTrue(all(g.isfinite().all().item() for g in grads))
        self.assertTrue(all(g.abs().sum().item()>0 for g in grads))
    def test_features_and_thresholds(self):
        loader=[(self.x,self.y,None,None,self.cycles)]
        vals=v7.threshold_from_validation(loader,self.model,torch.device('cpu'))
        self.assertEqual(set(vals),set(v7.CONDITIONS))
        for lo,hi in vals.values():self.assertLessEqual(lo,hi)
    def test_diagnostic_outputs(self):
        loader=[(self.x,self.y,None,None,self.cycles)]
        with tempfile.TemporaryDirectory() as d:
            meta={'mse':0.123,'mae':0.2}
            s=v7.finish_task('ETTh1',8,self.model,meta,Path(d)/'old.json',Path(d)/'checkpoint.pth',loader,loader,
                Path(d)/'result',torch.device('cpu'),batch_limit=0,grad_batches=1,split='val')
            self.assertEqual(s['n_windows'],4)
            self.assertEqual(s['match_status'],'validation_diagnostics')
            names=['path_interventions.csv','conditional_error.csv','conditional_path_interventions.csv',
                'forecast_horizon_bins.csv','channel_error.csv','gradient_sensitivity.csv','window_level.csv','manifest.json']
            for name in names:self.assertTrue((Path(d)/'result'/name).is_file(),name)
            import pandas as pd
            inter=pd.read_csv(Path(d)/'result'/'path_interventions.csv')
            self.assertEqual(len(inter),6)
            self.assertAlmostEqual(inter.iloc[0]['delta_mse_vs_full'],0.,places=6)

if __name__=='__main__':unittest.main()
