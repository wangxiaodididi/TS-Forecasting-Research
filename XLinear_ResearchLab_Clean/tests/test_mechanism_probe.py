"""不依赖作者下载/数据的离线函数级检查；运行 python -m tests.test_mechanism_probe。"""
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch
from torch import nn
from src.training.mechanism_probe import (spectral_features, disable_token_branch,
                                         summarize, frequency_tables, correlations,
                                         csv_write, report_text)


def main():
    # 24小时周期显著的历史信号（96个小时，FFT bin=4）。
    t=np.arange(96)
    x=np.sin(2*np.pi*t/24)[None,:]
    x=np.repeat(x,4,axis=0)
    spec=spectral_features(x,1)
    assert set(spec['peak_band'])=={'middle'}, spec['peak_band']
    assert np.allclose(spec['peak_period_hours'],24)
    assert np.all(spec['daily_energy_share']>0.50)
    # 对10分钟数据，16小时窗口不得冒充24小时周期。
    weather=spectral_features(x,1/6)
    assert np.isnan(weather['daily_energy_share']).all()

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.gain_logit=nn.Parameter(torch.tensor(-0.2))
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.backbone=Backbone()
    model=Model();before=float(model.backbone.gain_logit)
    with disable_token_branch(model):
        assert float(model.backbone.gain_logit)==0.0
    assert float(model.backbone.gain_logit)==before

    required=['split','variant','sample_index','mse_all','mse_ot','mse_token_off',
              'counterfactual_gain','prediction_abs_change','token_relative_norm',
              'tgm_gate_shift','vgm_gate_shift','peak_band','slow_energy_share',
              'middle_energy_share','fast_energy_share','daily_energy_share',
              'normalized_trend_slope','tgm_gate_low_frac','tgm_gate_high_frac',
              'tgm_gate_mean','vgm_gate_mean']
    data=[]
    for variant in ('official','dyn_token'):
        for i in range(10):
            row={k:0.1 for k in required}
            row.update(split='val',variant=variant,sample_index=i,
                       peak_band='middle' if i<5 else 'slow',mse_all=.2 if variant=='official' else .18,
                       mse_ot=.3, mse_token_off=.19 if variant!='official' else np.nan,
                       counterfactual_gain=.01 if variant!='official' else np.nan,
                       middle_energy_share=i/20,slow_energy_share=1-i/20,
                       daily_energy_share=.2,tgm_gate_low_frac=i/20,tgm_gate_high_frac=.2)
            data.append(row)
    a=summarize(data);b=frequency_tables(data);c=correlations(data)
    assert len(a)==2 and len(b)==4 and len(c)>0
    assert abs(next(z for z in a if z['variant']=='dyn_token')['counterfactual_gain']-.01)<1e-9
    with TemporaryDirectory() as d:
        root=Path(d)
        csv_write(root/'sample.csv',data,list(data[0]))
        assert len((root/'sample.csv').read_text(encoding='utf-8-sig').splitlines())==21
        text=report_text(Path('dummy'),['val'],a,b,c,data,1,0)
        assert '反事实' in text and '0.180000' in text
    print('PASS: spectral, intervention restoration, grouping, summary, CSV, report')

if __name__=='__main__':main()
