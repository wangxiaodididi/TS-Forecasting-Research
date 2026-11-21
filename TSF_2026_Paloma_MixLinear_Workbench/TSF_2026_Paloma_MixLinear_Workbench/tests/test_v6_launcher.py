"""No-network unit tests for the v6 launcher and summary reconstruction.
Run from the workspace root: python -m unittest discover -s tests -p test_v6_launcher.py
"""
import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paloma_v6_routes import (ENV, VARIANTS, PATCH, VERSION, collect_results,
                              find_baseline, git_blob, main, prepare_private_source)


class V6LauncherTests(unittest.TestCase):
    def test_three_distinct_routes(self):
        self.assertEqual(len(VARIANTS),3)
        self.assertEqual(len(set(VARIANTS)),3)
        self.assertIn('MultiheadAttention',PATCH)
        self.assertIn('branch_aware_future',PATCH)
        self.assertIn('Model.forward=_v6_forward',PATCH)

    def test_no_dry_run_side_effects(self):
        with tempfile.TemporaryDirectory() as t:
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                main(['--mode','full','--dry-run','--datasets','ETTh1,ETTm1',
                      '--horizons','96,336','--out-dir',str(Path(t)/'results'),
                      '--data-dir',str(Path(t)/'dataset')])
            self.assertEqual(buf.getvalue().count('[DRY]'),12)
            self.assertFalse((Path(t)/'results').exists())

    def test_collect_all_jobs_no_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            out=root/'v6'
            base=root/'v3'
            for ds in ('ETTh1','ETTm1'):
                for horizon in (96,336):
                    expected={
                        'model':'Paloma','dataset':ds,'horizon':horizon,
                        'lookback':96,'profile':'paper','source_commit':'fixed',
                        'training_epochs':30,'seed':2024,'status':'ok',
                        'mse':0.45 if horizon==336 else 0.36,'mae':0.4,
                        'variant':'original','returncode':0,
                    }
                    target=base/'paper'/ds/f'h{horizon}'/'original'
                    target.mkdir(parents=True)
                    (target/'metadata.json').write_text(json.dumps(expected))
                    for vi,v in enumerate(VARIANTS):
                        job=dict(expected,variant=v,mse=expected['mse']-0.001*(vi+1))
                        folder=out/'paper'/ds/f'h{horizon}'/v
                        folder.mkdir(parents=True)
                        (folder/'metadata.json').write_text(json.dumps(job))
            with contextlib.redirect_stdout(io.StringIO()):
                file=collect_results(out,[out,base])
            with file.open('r',encoding='utf-8-sig',newline='') as f:
                records=list(csv.DictReader(f))
            self.assertEqual(len(records),12)
            self.assertEqual({x['dataset'] for x in records},{'ETTh1','ETTm1'})
            self.assertEqual({x['horizon'] for x in records},{'96','336'})
            self.assertTrue(all(float(r['mse_gain_pct_vs_original'])>0 for r in records))
            # Second reconstruction restores all 12 rows, not only latest task.
            with contextlib.redirect_stdout(io.StringIO()):collect_results(out,[out,base])
            with file.open('r',encoding='utf-8-sig',newline='') as f:
                self.assertEqual(sum(1 for _ in csv.DictReader(f)),12)

    def test_empty_baseline_keeps_blank_gain(self):
        row={'status':'ok','dataset':'ETTh1','horizon':96,'profile':'paper',
             'lookback':96,'source_commit':'a','training_epochs':30,'seed':2024}
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(find_baseline(row,[Path(t)]),({},None))

    def test_source_guard_rejects_unknown(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            official=root/'upstream'
            (official/'models').mkdir(parents=True)
            (official/'models'/'Paloma.py').write_text('class Model: pass\n')
            (official/'run.py').write_text('')
            with self.assertRaisesRegex(RuntimeError,'Unexpected upstream'):
                prepare_private_source(official,root/'private')
            self.assertFalse((root/'private').exists())

    def test_source_guard_does_not_modify_original(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);upstream=root/'upstream'
            (upstream/'models').mkdir(parents=True)
            code=b'class Model: pass\n'
            (upstream/'models'/'Paloma.py').write_bytes(code)
            (upstream/'run.py').write_text('')
            import paloma_v6_routes as runner
            with patch.object(runner,'OFFICIAL_BLOB',git_blob(code)):
                first=prepare_private_source(upstream,root/'private')
                again=prepare_private_source(upstream,root/'private')
                self.assertEqual(first,again)
                self.assertEqual((upstream/'models'/'Paloma.py').read_bytes(),code)
                self.assertIn(b'PALOMA_V6_INFORMATION_ROUTES',
                    (root/'private'/'models'/'Paloma.py').read_bytes())
                (root/'private'/'models'/'Paloma.py').write_bytes(b'corrupted')
                with self.assertRaisesRegex(RuntimeError,'different'):
                    prepare_private_source(upstream,root/'private')

if __name__=='__main__':unittest.main()
