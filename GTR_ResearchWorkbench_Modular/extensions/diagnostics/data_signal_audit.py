"""Training-only signal diagnostics. No held-out labels are inspected.

Usage: ./.venv/bin/python -m extensions.diagnostics.data_signal_audit --dataset ETTm1
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
from lab.config import config_for
from lab.paths import ROOT,DATA


def audit(dataset):
    cfg=config_for(dataset,'GTR',96,2024,'paper')
    path=DATA/cfg['data_path']
    frame=pd.read_csv(path)
    # Follow official ETT split definitions; custom datasets split first 70%.
    if dataset=='ETTh1':n=12*30*24
    elif dataset=='ETTm1':n=12*30*24*4
    else:n=int(len(frame)*.7)
    x=frame.iloc[:n,1:].apply(pd.to_numeric,errors='coerce')
    cycle=cfg['cycle']
    rows=[]
    for name in x.columns:
        a=x[name].to_numpy(dtype=np.float64)
        missing=~np.isfinite(a)
        a=a[~missing]
        if len(a)<cycle+2:continue
        a=(a-a.mean())/(a.std()+1e-8)
        t=len(a)//2*2
        low=(a[:t:2]+a[1:t:2])/np.sqrt(2)
        high=(a[:t:2]-a[1:t:2])/np.sqrt(2)
        ratio=float((high**2).sum()/((high**2+low**2).sum()+1e-10))
        lag_corr=float(np.corrcoef(a[:-cycle],a[cycle:])[0,1])
        rows.append({'variable':name,'train_samples':int(len(a)),'missing_rate':float(missing.mean()),
                     'cycle_lag_correlation':lag_corr,'haar_high_energy_ratio':ratio})
    out=ROOT/'runs'/'data_audits';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/f'{dataset}_train_audit.csv',index=False)
    summary={'dataset':dataset,'samples_total':len(frame),'train_samples':n,'cycle':cycle,'n_features':len(rows),
             'median_cycle_lag_corr':float(np.median([r['cycle_lag_correlation'] for r in rows])),
             'median_haar_high_ratio':float(np.median([r['haar_high_energy_ratio'] for r in rows])),
             'csv':str(out/f'{dataset}_train_audit.csv')}
    (out/f'{dataset}_train_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['ETTh1','ETTm1','Weather','Electricity'],default='ETTm1')
    a=p.parse_args();print(json.dumps(audit(a.dataset),indent=2,ensure_ascii=False))
if __name__=='__main__':main()
