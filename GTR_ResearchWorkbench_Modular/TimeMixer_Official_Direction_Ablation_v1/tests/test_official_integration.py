"""Full official-source CPU smoke: run only after prepare.py."""
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'workspace'/'official'
if not (SRC/'models'/'TimeMixer.py').is_file():
    print('[SKIP] Official repository not yet installed. Execute python prepare.py on AutoDL first.')
    raise SystemExit(0)
sys.path.insert(0,str(SRC))
import torch
from models import TimeMixer,TimeMixerReverse

for dataset,channels,e_layers in [('ETTm1',7,2),('Weather',21,3)]:
    for horizon in (96,192):
        c=SimpleNamespace(
            task_name='long_term_forecast',seq_len=96,label_len=0,pred_len=horizon,
            enc_in=channels,dec_in=channels,c_out=channels,d_model=16,d_ff=32,
            e_layers=e_layers,dropout=0.1,channel_independence=1,decomp_method='moving_avg',
            moving_avg=25,top_k=5,down_sampling_layers=3,down_sampling_window=2,
            down_sampling_method='avg',use_norm=1,use_future_temporal_feature=0,
            embed='timeF',freq='h',
        )
        models=[TimeMixer.Model(c),TimeMixerReverse.Model(c)]
        par=[sum(p.numel() for p in m.parameters()) for m in models]
        assert par[0]==par[1],(dataset,horizon,par)
        for m in models:
            m.train()
            x=torch.randn(2,96,channels)
            y=m(x,None,None,None)
            assert y.shape==(2,horizon,channels),y.shape
            y.square().mean().backward()
            # Verify PDM scale-path gradients (not all unused optional branches).
            seasonal=m.pdm_blocks[0].mixing_multi_scale_season
            trend=m.pdm_blocks[0].mixing_multi_scale_trend
            assert all(p.grad is not None for p in seasonal.parameters())
            assert all(p.grad is not None for p in trend.parameters())
        print(f'[PASS] {dataset}-{horizon}, params={par[0]}, both full forward/backward OK')
