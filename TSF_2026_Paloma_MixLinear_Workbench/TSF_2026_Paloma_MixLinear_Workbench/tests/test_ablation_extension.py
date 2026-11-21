import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paloma_ablation import (VARIANTS, ABLATED_FORWARD, MARKER, digest,
                            git_blob_digest, save_table, collect_summary, build_private_source)
from experiment import build_params

class AblationExtensionTests(unittest.TestCase):
    def test_variant_names_unique(self):
        self.assertEqual(len(VARIANTS), len(set(VARIANTS)))
        self.assertEqual(VARIANTS[0], 'original')

    def test_patch_has_original_forward_preservation(self):
        self.assertIn('_ReferenceForward = Model.forward', ABLATED_FORWARD)
        self.assertIn('return _ReferenceForward(self, x, cycle_index)', ABLATED_FORWARD)
        self.assertIn('Model.forward = _ablation_forward', ABLATED_FORWARD)
        self.assertIn(MARKER, ABLATED_FORWARD)
        compile(ABLATED_FORWARD, 'patch', 'exec')

    def test_source_digest(self):
        expected = hashlib.sha1(b'blob 4\x00test').hexdigest()
        self.assertEqual(git_blob_digest(b'test'), expected)

    def test_reject_unpinned_upstream_source(self):
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'upstream'
            (src/'models').mkdir(parents=True)
            (src/'run.py').write_text('',encoding='utf-8')
            (src/'models'/'Paloma.py').write_text('fake model',encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError,'Unexpected Paloma source blob'):
                build_private_source(src,Path(d)/'private')
            self.assertFalse((Path(d)/'private').exists())

    def test_author_params_unmodified(self):
        for ds,h in [('ETTh1',96),('ETTm1',336)]:
            cmd,meta = build_params('Paloma',ds,h,'paper',0,0,
                                    Path('/tmp') / (ds + '.csv'),Path('/tmp/out'))
            self.assertEqual(meta['lookback'],96)
            self.assertIn('--lossfun_alpha',cmd)
            self.assertEqual(cmd[cmd.index('--random_seed')+1],'2024')

    def test_summary_degradation_sign(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'summary.csv'
            r=[dict(model='Paloma',variant='original',dataset='ETTh1',horizon=96,profile='paper',status='ok',mse=.4,mae=.5),
               dict(model='Paloma',variant='no_phase',dataset='ETTh1',horizon=96,profile='paper',status='ok',mse=.44,mae=.55)]
            save_table(r,path)
            self.assertAlmostEqual(r[1]['mse_degradation_pct_vs_original'],10.0)
            self.assertAlmostEqual(r[1]['mae_degradation_pct_vs_original'],10.0)
            self.assertTrue(path.exists())

if __name__=='__main__':
    unittest.main()
