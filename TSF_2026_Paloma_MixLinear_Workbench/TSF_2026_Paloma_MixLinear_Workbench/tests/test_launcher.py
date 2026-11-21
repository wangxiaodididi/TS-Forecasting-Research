import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import build_params, parse_metrics


class LauncherTests(unittest.TestCase):
    def command(self, model, ds, horizon, profile):
        cmd, meta = build_params(model,ds,horizon,profile,0,0,
                                Path('/tmp/ETT-small')/f'{ds}.csv',Path('/tmp/out'))
        return cmd,meta

    def flag(self,cmd,name):
        return cmd[cmd.index(name)+1]

    def test_paper_profile_uses_official_lookbacks(self):
        for ds in ['ETTh1','ETTm1']:
            for h in [96,336]:
                for m,expected in [('Paloma','96'),('MixLinear','720')]:
                    cmd,meta=self.command(m,ds,h,'paper')
                    self.assertEqual(self.flag(cmd,'--seq_len'),expected)
                    self.assertEqual(meta['provenance'],'as_author_script')

    def test_aligned_profile_has_same_lookback(self):
        for m in ['Paloma','MixLinear']:
            for ds in ['ETTh1','ETTm1']:
                for h in [96,336]:
                    cmd,_=self.command(m,ds,h,'aligned')
                    self.assertEqual(self.flag(cmd,'--seq_len'),'96')
                    self.assertEqual(self.flag(cmd,'--pred_len'),str(h))

    def test_lpf_within_number_of_segments(self):
        for profile in ['paper','aligned']:
            for ds in ['ETTh1','ETTm1']:
                for h in [96,336]:
                    cmd,_=self.command('MixLinear',ds,h,profile)
                    segments=int(self.flag(cmd,'--seq_len'))//int(self.flag(cmd,'--period_len'))
                    self.assertLessEqual(int(self.flag(cmd,'--lpf')),segments)

    def test_ettm1_author_settings(self):
        cmd,_=self.command('MixLinear','ETTm1',336,'paper')
        self.assertEqual(self.flag(cmd,'--period_len'),'2')
        self.assertEqual(self.flag(cmd,'--lpf'),'144')
        self.assertEqual(self.flag(cmd,'--train_epochs'),'15')

    def test_metric_parser(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'train.log'
            p.write_text('Epoch 3 mse:0.481, mae:0.4\nmse:0.320, mae:0.301\n')
            self.assertEqual(parse_metrics(p),(0.320,0.301))

if __name__=='__main__':
    unittest.main()
