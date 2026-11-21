"""Offline tests: official flags, parsing, safe source patch and model extension operations."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from lab.config import config_for,command_args
from lab.monitor import parse_line
from lab.extend import install_snapshot_extension,enable_smoke_subset
from lab.source import git_blob

class ConfigTests(unittest.TestCase):
    def test_ettm1_matches_author(self):
        c=config_for('ETTm1','GTR',96,2024,'paper')
        self.assertEqual((c['seq_len'],c['cycle'],c['batch_size'],c['learning_rate'],c['dropout'],c['train_epochs']),
                         (96,96,256,.001,.5,30))
        args=command_args(c)
        for key,value in [('model','GTR'),('data','ETTm1'),('enc_in','7'),('pred_len','96'),('random_seed','2024')]:
            self.assertEqual(args[args.index('--'+key)+1],value)
        self.assertNotIn('--num_workers',args)
        self.assertNotIn('--d_model',args)
        self.assertNotIn('--lradj',args)

    def test_electricity_does_not_force_dropout(self):
        args=command_args(config_for('Electricity','GTR',192,2024,'paper'))
        self.assertNotIn('--dropout',args)
        self.assertEqual(args[args.index('--learning_rate')+1],'0.003')

    def test_ett_seeds(self):
        self.assertEqual(config_for('ETTh1','GTR',96,2029,'paper')['random_seed'],2029)
        with self.assertRaises(ValueError):config_for('ETTm1','GTR',96,2029,'paper')

    def test_baseline_locked(self):
        with self.assertRaises(ValueError):config_for('ETTm1','GTR',96,2024,'paper',overrides={'learning_rate':0.2})

    def test_smoke_not_paper(self):
        c=config_for('ETTm1','GTR',96,2024,'smoke')
        self.assertFalse(c['paper_exact'])
        self.assertEqual((c['train_epochs'],c['batch_size'],c['num_workers']),(1,8,0))

class MonitorTests(unittest.TestCase):
    def test_epoch(self):
        q=parse_line('Epoch: 1, Steps: 42 | Train Loss: 0.1532314 Vali Loss: 0.2094198 Test Loss: 0.3064491')
        self.assertEqual(q['type'],'epoch')
        self.assertAlmostEqual(q['val_loss'],0.2094198)
    def test_batch(self):
        q=parse_line('iters: 100, epoch: 1 | loss: 0.9134500')
        self.assertEqual(q['type'],'batch')
    def test_git_blob(self):
        import hashlib
        expected=hashlib.sha1(b'blob 4\x00test').hexdigest()
        self.assertEqual(git_blob(b'test'),expected)

class PatchTests(unittest.TestCase):
    def test_extensions_only_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'models').mkdir();(root/'exp').mkdir();(root/'data_provider').mkdir()
            exp="from utils.tools import x\nmodel_dict = {'GTR': GTR,}\n"
            original=root/'exp'/'exp_main.py';original.write_text(exp)
            data=root/'data_provider'/'data_factory.py';data.write_text('    print(flag, len(data_set))\n')
            install_snapshot_extension(root)
            self.assertIn('GTRHaarHigh',original.read_text())
            self.assertEqual((root/'models'/'GTR.py').exists(),False)
            enable_smoke_subset(root)
            self.assertIn('Subset(',data.read_text())

if __name__=='__main__':unittest.main()
