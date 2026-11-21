"""Portable smoke and bookkeeping tests; run at workbench root.

Official source and real ETT CSV are optional for unit-test discovery, but
required by the run_paloma_functional_splice.sh --mode smoke entrypoint.
"""
import contextlib
import csv
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from paloma_functional_splice import (
    OFFICIAL_BLOB, VARIANTS, collect_results, git_blob,
    parse_args, prepare_private_source, smoke
)

SOURCE=Path(os.environ.get('PALOMA_TEST_SOURCE',str(ROOT/'sources'/'Paloma')))
DATA_DIR=Path(os.environ.get('DATA_DIR','/root/autodl-tmp/GTR_ResearchWorkbench_Modular/dataset'))
ETT=Path(os.environ.get('PALOMA_TEST_ETT',str(DATA_DIR/'ETTh1.csv')))

class FunctionalSpliceTests(unittest.TestCase):
    def test_parser(self):
        a=parse_args(['--mode','smoke','--variants','original,tq_channel','--horizons','96,336'])
        self.assertEqual(a.horizons,[96,336])
        self.assertEqual(a.variants,['original','tq_channel'])
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            parse_args(['--variants','not_a_variant'])
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            parse_args(['--horizons','720'])

    def test_result_comparison_and_no_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)
            for variant,mse in (('original',.36),('tq_channel',.35)):
                p=base/'paper'/'ETTh1'/'h96'/variant
                p.mkdir(parents=True)
                (p/'metadata.json').write_text(json.dumps({
                    'variant':variant,'model':'Paloma','dataset':'ETTh1',
                    'horizon':96,'lookback':96,'seed':2024,'status':'ok',
                    'training_epochs':30,'mse':mse,'mae':.4
                }),encoding='utf-8')
            output=collect_results(base)
            with output.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
            result=next(x for x in rows if x['variant']=='tq_channel')
            self.assertAlmostEqual(float(result['mse_gain_pct_vs_original']),100*(.36-.35)/.36)
            original=base/'paper'/'ETTh1'/'h96'/'original'/'metadata.json'
            original.unlink()
            collect_results(base)
            with output.open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
            self.assertEqual(rows[0]['mse_gain_pct_vs_original'],'')

    def test_official_source_and_smoke_96(self):
        f=SOURCE/'models'/'Paloma.py'
        if not f.exists() or not ETT.is_file():
            self.skipTest('Official source or ETT test data missing')
        self.assertEqual(git_blob(f.read_bytes()),OFFICIAL_BLOB)
        with tempfile.TemporaryDirectory() as temp:
            dest=Path(temp)/'private'
            m=prepare_private_source(SOURCE,dest)
            self.assertEqual(m['version'],3)
            self.assertEqual(prepare_private_source(SOURCE,dest),m)
            smoke(SOURCE,dest,'ETTh1',96,list(VARIANTS),ETT)

    def test_official_source_and_smoke_336(self):
        f=SOURCE/'models'/'Paloma.py'
        if not f.exists() or not ETT.is_file():
            self.skipTest('Official source or ETT test data missing')
        with tempfile.TemporaryDirectory() as temp:
            dest=Path(temp)/'private'
            prepare_private_source(SOURCE,dest)
            smoke(SOURCE,dest,'ETTh1',336,list(VARIANTS),ETT)

if __name__=='__main__': unittest.main()
