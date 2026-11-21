"""Offline tests of stage-1 scheduler, comparability, and validation-only summary."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from lab.config import config_for
from phase1 import core

class Phase1Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name);data=root/'dataset';data.mkdir();(data/'ETTm1.csv').write_text('fixture')
        runs=root/'runs';runs.mkdir();batches=root/'batches';batches.mkdir()
        for attr,value in [('DATA',data),('RUNS',runs),('BATCHES',batches)]:
            p=patch.object(core,attr,value);p.start();self.addCleanup(p.stop)
        p=patch.object(core,'verify',return_value=True);p.start();self.addCleanup(p.stop)
    def baseline(self,bad_hash=False):
        p=core.RUNS/'example_baseline';p.mkdir()
        (p/'config.json').write_text(json.dumps(config_for('ETTm1','GTR',96,2024,'paper')))
        (p/'status.json').write_text(json.dumps({'state':'completed','epoch':10,'final_test_mse':.307,'final_test_mae':.351}))
        h='bad' if bad_hash else core.sha256(core.DATA/'ETTm1.csv')
        (p/'environment.json').write_text(json.dumps({'dataset_sha256':h}))
        (p/'loss.csv').write_text('epoch,train_loss,val_loss\n1,0.4,0.5\n2,0.3,0.39\n3,0.2,0.41\n')
        return p.name
    def test_baseline_matches_protocol(self):
        rid=self.baseline()
        self.assertEqual(core.preflight('ETTm1',96,2024,'paper',rid).name,rid)
    def test_wrong_data_hash_aborts(self):
        rid=self.baseline(bad_hash=True)
        with self.assertRaisesRegex(ValueError,'SHA256'):core.preflight('ETTm1',96,2024,'paper',rid)
    def test_fair_hyperparameters(self):
        expected=config_for('ETTm1','GTR',96,2024,'paper')
        for model in core.EXTENSIONS:
            c=config_for('ETTm1',model,96,2024,'paper')
            for k in ('learning_rate','batch_size','dropout','train_epochs','patience','seq_len','pred_len','cycle','random_seed'):
                self.assertEqual(c[k],expected[k])
    def test_sequential_jobs_and_best_validation_epoch(self):
        self.baseline()
        b=core.new_batch('ETTm1',96,2024,'paper');seen=[]
        def mock_train(path):
            p=Path(path);seen.append(json.loads((p/'config.json').read_text())['model'])
            core.atomic_json(p/'status.json',{'state':'completed','epoch':3,'final_test_mse':.33,'final_test_mae':.37})
            (p/'loss.csv').write_text('epoch,train_loss,val_loss\n1,0.4,0.5\n2,0.3,0.35\n3,0.2,0.39\n')
        with patch.object(core,'run_worker',side_effect=mock_train):self.assertEqual(core.worker(b),0)
        self.assertEqual(seen,list(core.EXTENSIONS))
        _,_,rows,p=core.summarize(b.name)
        self.assertEqual(len(rows),5);self.assertEqual(rows[1]['best_epoch'],2)
        self.assertNotIn('test_mse',p.read_text())
        _,_,rows,p=core.summarize(b.name,include_test=True)
        self.assertEqual(rows[1]['test_mse'],.33)
    def test_dual_haar_and_linear_have_equal_function_class(self):
        # Two unconstrained Haar projections are exactly a change of linear basis.
        import math
        import torch
        torch.manual_seed(2)
        x=torch.randn(3,96)
        wlow=torch.randn(96,48)
        whigh=torch.randn(96,48)
        a=(x[:,0::2]+x[:,1::2])/math.sqrt(2)
        d=(x[:,0::2]-x[:,1::2])/math.sqrt(2)
        lhs=a@wlow.T+d@whigh.T
        w=torch.empty(96,96)
        w[:,0::2]=(wlow+whigh)/math.sqrt(2)
        w[:,1::2]=(wlow-whigh)/math.sqrt(2)
        self.assertTrue(torch.allclose(lhs,x@w.T,atol=1e-5))

    def test_failed_run_stops_next_job(self):
        b=core.new_batch('ETTm1',96,2024,'smoke');seen=[]
        def failed(path):
            seen.append(1);core.atomic_json(Path(path)/'status.json',{'state':'failed','epoch':0})
        with patch.object(core,'run_worker',side_effect=failed):self.assertEqual(core.worker(b),1)
        self.assertEqual(len(seen),1)

if __name__=='__main__':unittest.main()
