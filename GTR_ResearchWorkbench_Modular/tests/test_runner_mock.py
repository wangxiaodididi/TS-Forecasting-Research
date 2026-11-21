"""Integration test with a synthetic command-line trainer, not an official GTR run."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from lab.config import config_for
from lab.runner import make_run,run_worker
import lab.source as source
import lab.runner as runner

class IntegrationTests(unittest.TestCase):
    def test_stream_to_status_and_epoch_csv(self):
        with tempfile.TemporaryDirectory() as d:
            tmp=Path(d);vendor=tmp/'vendor';vendor.mkdir()
            data=tmp/'dataset';data.mkdir();(data/'ETTm1.csv').write_text('date,OT\n1,2\n')
            (vendor/'exp').mkdir()
            (vendor/'exp'/'exp_main.py').write_text('# fixture')
            (vendor/'models').mkdir()
            (vendor/'models'/'GTR.py').write_text('# fixture')
            (vendor/'data_provider').mkdir()
            (vendor/'data_provider'/'data_factory.py').write_text('    print(flag, len(data_set))\n')
            (vendor/'run.py').write_text('''import time
print('iters: 100, epoch: 1 | loss: 0.2000000',flush=True)
print('Epoch: 1, Steps: 100 | Train Loss: 0.2300000 Vali Loss: 0.2700000 Test Loss: 0.2900000',flush=True)
print('mse:0.2900000, mae:0.4000000',flush=True)
''')
            run=tmp/'runs'/'testjob';run.parent.mkdir()
            cfg=config_for('ETTm1','GTR',96,2024,'smoke')
            make_run(run,cfg)
            with patch.object(source,'VENDOR',vendor),patch.object(source,'DATA',data),patch.object(source,'verify',lambda p:True),patch.object(runner,'DATA',data):
                done=run_worker(run)
            self.assertTrue(done)
            state=json.loads((run/'status.json').read_text())
            self.assertEqual(state['state'],'completed')
            self.assertEqual(state['epoch'],1)
            self.assertAlmostEqual(state['val_loss'],0.27)
            self.assertTrue((run/'loss.csv').exists())
            self.assertAlmostEqual(state['final_test_mse'],0.29)
if __name__=='__main__':unittest.main()
