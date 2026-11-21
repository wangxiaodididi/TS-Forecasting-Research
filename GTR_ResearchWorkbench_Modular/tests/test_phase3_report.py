import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from phase3.report import MODELS, report, _bootstrap_paired, _terciles_by_channel

class TestReport(unittest.TestCase):
    def _fixture(self,root):
        ids=[0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290]
        rng=np.random.default_rng(20)
        meta=dict(dataset='Weather',horizon=96,seed=2024,seq_len=96,channels=2,
                  channel_names=['A','B'],source_dataset_sha256='test-hash',sample_indices=ids,
                  samples=len(ids),mode='paper',split='val')
        for num,model in enumerate(MODELS):
            rows=[]
            for i in ids:
                for c in (0,1):
                    base=.25+.01*c+.01*np.sin(i)
                    rows.append(dict(model=model,sample_index=i,channel=c,
                         mse=base+num*.01,mae=.2+num*.01,
                         relative_change=.1+c*.05+i*.001,relative_drift=.1+c*.04+i*.002,
                         phase_fit_r2=(i+1)/300 if 'Phase' in model else np.nan))
            pd.DataFrame(rows).to_csv(root/f'{model}.csv',index=False)
            (root/f'{model}.json').write_text(json.dumps(meta),encoding='utf-8')

    def test_generated_tables(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self._fixture(root)
            rows=report(root)
            self.assertEqual(len(rows),3)
            self.assertAlmostEqual(rows[0]['delta_mse'],.01,places=8)
            self.assertTrue((root/'by_channel.csv').is_file())
            self.assertTrue((root/'by_condition.csv').is_file())
            self.assertEqual(len(pd.read_csv(root/'by_channel.csv')),6)
            self.assertIn('GTRPhaseAdaptive', pd.read_csv(root/'by_condition.csv')['model'].values)

    def test_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self._fixture(root)
            p=root/'GTRPhaseAdaptive.json';v=json.loads(p.read_text());v['seed']=2025
            p.write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError,'mismatched seed'):
                report(root)

    def test_bootstrap_and_bins(self):
        x=np.linspace(-.1,.1,100)
        l,h,b=_bootstrap_paired(x,np.arange(100)*30,96,96,50)
        self.assertLess(l,h)
        self.assertGreaterEqual(b,2)
        df=pd.DataFrame({'channel':[0]*9+[1]*9,'v':list(range(9))+list(range(9,18))})
        labels=_terciles_by_channel(df,'v')
        self.assertEqual(set(labels),{'Low','Medium','High'})

if __name__=='__main__':unittest.main()
