#!/usr/bin/env python3
import tempfile
from pathlib import Path
import patch_sources

T_RUN='''fix_seed = 2026
if __name__ == '__main__':
    if args.is_training:
        for ii in range(args.itr):
            exp.train(setting)
            if not args.train_only:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)

'''
T_EXP='''    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')
        for epoch in range(2):
            if not self.args.train_only:
                self.model.eval()
                vali_loss, vali_mae = self.vali(vali_data, vali_loader, mse_criterion, phase='val')
                test_loss, test_mae = self.vali(test_data, test_loader, mse_criterion, phase='test')
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

                if (epoch+1) % 5 == 0:
                    train_mse, train_mae = self.vali(train_data, train_loader, mse_criterion, phase='train')
                    print("Train_mse: {0:.7f} Train_mae: {1:.7f} Vali MAE: {2:.7f} Test MAE: {3:.7f}".format(
                                train_mse, train_mae, vali_mae, test_mae))

                self.model.train()
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
       
        return self.model
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
'''
B_RUN='''fix_seed_list = range(2023, 2033)
if args.is_training:
    for ii in range(args.itr):
        exp.train(setting)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
else:
    exp.test(setting, test=1)
'''
B_EXP='''    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        for epoch in range(2):
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))# ,map_location="cuda:0"))

        print(f"Final Max Memory (MB): {max_memory}")
        return self.model
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
'''
with tempfile.TemporaryDirectory() as td:
    root=Path(td); t=root/'TStruct'; (t/'exp').mkdir(parents=True)
    (t/'run_longExp.py').write_text(T_RUN); (t/'exp/exp_main.py').write_text(T_EXP)
    patch_sources.patch_tstruct(t)
    assert 'fix_seed = 2024' in (t/'run_longExp.py').read_text()
    assert 'test_data, test_loader' not in (t/'exp/exp_main.py').read_text().split('def test(')[0]
    assert 'FINAL_VAL_MSE=' in (t/'exp/exp_main.py').read_text()
    assert 'exp.test(setting)' not in (t/'run_longExp.py').read_text()
    b=root/'TimeBase'; (b/'exp').mkdir(parents=True); (b/'utils').mkdir(parents=True)
    (b/'run_longExp.py').write_text(B_RUN); (b/'exp/exp_main.py').write_text(B_EXP); (b/'utils/tools.py').write_text('import numpy as np\nval = np.Inf\n')
    patch_sources.patch_timebase(b)
    assert 'fix_seed_list = [2024]' in (b/'run_longExp.py').read_text()
    train_part=(b/'exp/exp_main.py').read_text().split('def test(')[0]
    assert 'test_data, test_loader' not in train_part
    assert 'FINAL_VAL_MSE=' in train_part and 'FINAL_VAL_MAE=' in train_part
    assert 'exp.test(setting)' not in (b/'run_longExp.py').read_text()
    assert 'np.inf' in (b/'utils/tools.py').read_text()
print('PATCHER_TEST_OK TStruct+TimeBase validation-only patches')
