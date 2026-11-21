"""No GPU required. Run: python -m unittest discover -s tests -p test_context_fusion_v4.py -v"""
import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from paloma_context_fusion_v4 import (
    VARIANTS, OFFICIAL_BLOB, git_blob,
    prepare_private_source, module_from_file, smoke, collect_results, parse_args,
)

SOURCE=Path(os.environ.get('PALOMA_TEST_SOURCE',str(ROOT/'sources'/'Paloma')))
DATA=Path(os.environ.get('PALOMA_TEST_ETT','/root/autodl-tmp/GTR_ResearchWorkbench_Modular/dataset/ETTh1.csv'))

class ContextFusionV4Tests(unittest.TestCase):
    def test_parser(self):
        p=parse_args(['--mode','smoke','--horizons','96,336','--variants','olinear_hidden,lagged_residual'])
        self.assertEqual(p.horizons,[96,336])
        self.assertEqual(p.variants,['olinear_hidden','lagged_residual'])
        self.assertTrue('original' not in parse_args([]).variants)
        with self.assertRaises(SystemExit):
            parse_args(['--variants','not_valid'])

    def test_recovered_metrics_and_v3_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            refdir=root/'v3'
            ref=refdir/'paper'/'ETTh1'/'h96'/'original'/'metadata.json'
            ref.parent.mkdir(parents=True)
            protocol={'model':'Paloma','dataset':'ETTh1','horizon':96,'lookback':96,
                      'profile':'paper','seed':2024,'status':'ok','source_commit':'abc',
                      'training_epochs':30}
            ref.write_text(json.dumps({**protocol,'variant':'original','mse':.36,'mae':.38}))
            newroot=root/'v4'
            new=newroot/'paper'/'ETTh1'/'h96'/'olinear_residual'/'metadata.json'
            new.parent.mkdir(parents=True)
            new.write_text(json.dumps({**protocol,'variant':'olinear_residual','mse':.35,'mae':.37}))
            csvfile=collect_results(newroot,refdir)
            with csvfile.open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),1)
            self.assertAlmostEqual(float(rows[0]['mse_gain_pct_vs_original']),(.36-.35)/.36*100)
            self.assertAlmostEqual(float(rows[0]['mae_gain_pct_vs_original']),(.38-.37)/.38*100)
            # Rebuilding never duplicates rows
            collect_results(newroot,refdir)
            with csvfile.open(encoding='utf-8-sig') as f: self.assertEqual(len(list(csv.DictReader(f))),1)
            new.write_text(json.dumps({**protocol,'variant':'olinear_residual','mse':.35,'mae':.37,'seed':2023}))
            with collect_results(newroot,refdir).open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
            self.assertEqual(rows[0]['mse_gain_pct_vs_original'],'')

    def test_official_source_and_smoke(self):
        if not (SOURCE/'models'/'Paloma.py').is_file() or not DATA.is_file():
            self.skipTest('Official model or ETT file unavailable; set PALOMA_TEST_SOURCE and PALOMA_TEST_ETT')
        self.assertEqual(git_blob((SOURCE/'models'/'Paloma.py').read_bytes()),OFFICIAL_BLOB)
        with tempfile.TemporaryDirectory() as d:
            patched=Path(d)/'private'
            m=prepare_private_source(SOURCE,patched)
            self.assertEqual(m['version'],4)
            self.assertEqual(prepare_private_source(SOURCE,patched),m)
            smoke(SOURCE,patched,'ETTh1',96,list(VARIANTS),DATA)
            smoke(SOURCE,patched,'ETTh1',336,list(VARIANTS),DATA)
            self._check_causal_lags(patched)

    def _check_causal_lags(self, source):
        import torch
        os.environ['PALOMA_CONTEXT_FUSION_V4']='lagged_residual'
        try:
            from types import SimpleNamespace
            model=module_from_file(source/'models'/'Paloma.py','context_v4_check_causal')
            cfg=SimpleNamespace(seq_len=96,pred_len=96,enc_in=7,cycle=24,d_model=256,
                                dropout=.5,use_revin=1,individual=0)
            net=model.Model(cfg)
            mixer=net.context_module
            a=torch.randn(2,7,96)
            b=a.clone(); b[...,48:]+=100.0
            with torch.no_grad():
                ya=mixer(a)
                yb=mixer(b)
            self.assertTrue(torch.equal(ya[...,:48],yb[...,:48]))
        finally:
            os.environ.pop('PALOMA_CONTEXT_FUSION_V4',None)

if __name__=='__main__': unittest.main()
