from pathlib import Path
import csv
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import paloma_v5_replace as v5

class TestV5Launcher(unittest.TestCase):
    def test_variant_count(self):
        self.assertEqual(len(v5.VARIANTS),6)
        self.assertEqual(len(v5.DESCRIPTIONS),6)
        self.assertEqual(len(v5.CITATIONS),6)

    def test_git_blob_implementation(self):
        self.assertEqual(v5.git_blob(b'hello\n'),'ce013625030ba8dba906f756967f9e9ca394464a')

    def test_cli_default_24_jobs(self):
        a=v5.parse_args(['--mode','full','--dry-run'])
        self.assertEqual(len(a.datasets)*len(a.horizons)*len(a.variants),24)
        self.assertEqual(a.datasets,['ETTh1','ETTm1'])
        self.assertEqual(a.horizons,[96,336])

    def test_summary_collects_all_24_and_real_baselines(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); out=root/'v5'; old=root/'v3'
            for ds in ('ETTh1','ETTm1'):
                for h in (96,336):
                    base=old/'paper'/ds/f'h{h}'/'original'
                    base.mkdir(parents=True)
                    baseline={'status':'ok','dataset':ds,'horizon':h,'lookback':96,
                              'seed':2024,'profile':'paper','training_epochs':30,
                              'source_commit':v5.SHAS['Paloma'],'mse':0.4,'mae':0.35}
                    (base/'metadata.json').write_text(json.dumps(baseline))
                    for name in v5.VARIANTS:
                        path=out/'paper'/ds/f'h{h}'/name
                        path.mkdir(parents=True)
                        meta={**baseline,'model':'Paloma','variant':name,
                              'returncode':0,'mse':0.39,'mae':0.34,
                              'source_commit':v5.SHAS['Paloma']}
                        (path/'metadata.json').write_text(json.dumps(meta))
            csv_path=v5.collect_results(out,[out,old])
            with csv_path.open(encoding='utf-8-sig') as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),24)
            self.assertEqual({r['variant'] for r in rows},set(v5.VARIANTS))
            self.assertTrue(all(float(r['mse_gain_pct_vs_original'])==2.5 for r in rows))
            self.assertTrue(all(r['baseline_metadata'] for r in rows))
            # Re-collect must not truncate to a most recent batch.
            with csv_path.open(encoding='utf-8-sig') as reopened:
                self.assertEqual(len(list(csv.DictReader(reopened))),24)

    def test_metadata_mismatch_must_not_suggest_valid_improvement(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); out=root/'v5'; old=root/'v3'
            base=old/'paper'/'ETTh1'/'h96'/'original'
            base.mkdir(parents=True)
            (base/'metadata.json').write_text(json.dumps({
                'status':'ok','dataset':'ETTh1','horizon':96,'lookback':96,
                'profile':'paper','seed':2025,'training_epochs':30,
                'source_commit':v5.SHAS['Paloma'],'mse':0.4,'mae':0.35}))
            job=out/'paper'/'ETTh1'/'h96'/'segment_head'
            job.mkdir(parents=True)
            (job/'metadata.json').write_text(json.dumps({
                'status':'ok','model':'Paloma','variant':'segment_head',
                'dataset':'ETTh1','horizon':96,'lookback':96,'seed':2024,
                'profile':'paper','training_epochs':30,
                'source_commit':v5.SHAS['Paloma'],'mse':0.39,'mae':0.34}))
            path=v5.collect_results(out,[old])
            with path.open(encoding='utf-8-sig') as f: row=next(csv.DictReader(f))
            self.assertEqual(row['mse_gain_pct_vs_original'],'')
            self.assertEqual(row['baseline_metadata'],'')

if __name__=='__main__': unittest.main()
