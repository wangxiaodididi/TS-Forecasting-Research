"""Standalone shape/gradient/checkpoint-free tests for m2s_baselines."""
import tempfile
from pathlib import Path
import csv
import unittest
import torch
from m2s_baselines.models import BENCHMARKS, NO_TRAIN, TARGET_ONLY, make_model
from m2s_baselines.merge import merge

class BaselineTests(unittest.TestCase):
    def test_shapes_and_gradients(self):
        for name in BENCHMARKS:
            c=1 if name in TARGET_ONLY else 4
            m=make_model(name,c,60,5,5)
            x=torch.randn(3,60,c)
            pred=m(x)
            self.assertEqual(tuple(pred.shape),(3,5,1),name)
            self.assertTrue(torch.isfinite(pred).all().item(),name)
            if name not in NO_TRAIN:
                pred.square().mean().backward()
                self.assertTrue(any(p.grad is not None for p in m.parameters()),name)
    def test_target_is_last_and_naive(self):
        x=torch.tensor([[[1.,10.],[2.,20.],[3.,30.],[4.,40.],[5.,50.]]])
        y=make_model('last_value',2,5,3,5)(x)
        self.assertEqual(y.squeeze(-1).tolist(),[[50.,50.,50.]])
        z=make_model('seasonal_naive',2,5,4,3)(x)
        self.assertEqual(z.squeeze(-1).tolist(),[[30.,40.,50.,30.]])
    def test_original_summary_merge(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            a=root/'results'/'m2s'/'full'/'summary.csv';a.parent.mkdir(parents=True)
            b=root/'results'/'m2s_baselines'/'full'/'summary.csv';b.parent.mkdir(parents=True)
            headers=['dataset','L','H','variant','seed','status','mse','mae','rmse']
            for f,rows in [(a,[['Hubei','60','1','target_only','2024','ok','10','2','3'],['Hubei','60','1','multivariate_joint','2024','ok','8','1.8','2.8']]),
                          (b,[['Hubei','60','1','joint_linear','2024','ok','7','1.7','2.64']])]:
                with f.open('w',newline='') as o:
                    w=csv.writer(o);w.writerow(headers);w.writerows(rows)
            n,path=merge(root)
            self.assertEqual(n,3)
            with open(path,newline='') as f: rows=list(csv.DictReader(f))
            self.assertEqual(rows[0]['model'],'joint_linear')
            self.assertEqual(rows[0]['pct_mse_improvement_vs_target_only_tm'],'30.0')
            self.assertEqual(rows[0]['pct_mse_improvement_vs_joint_tm'],'12.5')

if __name__=='__main__':unittest.main()
