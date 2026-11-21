"""REAL ETTh1 input CSV + simple seasonal-repeat proxy forecasters.
This is NOT an implementation or a trained benchmark of TimeMixer/XLinear.
It only checks extraction/evaluation I/O, chronological labels, and online updates.
"""
from pathlib import Path
import numpy as np
import pandas as pd


def generate(file,out,horizon=96,seq_len=96,n_val=340,n_test=320):
    x=pd.read_csv(file).drop(columns=['date']).to_numpy(dtype=np.float32)
    # Official ETTh1 time boundaries: 12*30*24 train, 4*30*24 validation, 4*30*24 test.
    tr_end=12*30*24;v_end=tr_end+4*30*24;t_end=v_end+4*30*24
    if t_end>len(x):raise ValueError('ETTh1 file shorter than the official split boundaries')
    mu=x[:tr_end].mean(0);sd=x[:tr_end].std(0)+1e-6;x=(x-mu)/sd
    def split(begin,stop,maxn):
        n=min(maxn,stop-begin-horizon+1);y=[];parts=[]
        for t in range(begin,begin+n):
            hist=x[t-seq_len:t]
            # Three proxies using past periodic template only: 24, 48, 96 hours.
            scales=[hist[-k:][np.arange(horizon)%k] / 3 for k in (24,48,96)]
            parts.append(np.stack(scales,-1));y.append(x[t:t+horizon])
        p=np.asarray(parts,dtype=np.float32);y=np.asarray(y,dtype=np.float32)
        return {'base':p.sum(-1),'parts':p,'y':y}
    v=split(tr_end,v_end,n_val);t=split(v_end,t_end,n_test)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'val.npz',**v);np.savez_compressed(out/'test.npz',**t)
    print(f'[ETTh1 proxy smoke, NOT TimeMixer] val={len(v["y"])}, test={len(t["y"])}, H={horizon}, channels={t["y"].shape[-1]}')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--csv',default='data/ETTh1.csv');p.add_argument('--outdir',default='results/proxy')
    p.add_argument('--horizon',type=int,default=96)
    a=p.parse_args();generate(a.csv,a.outdir,a.horizon)
