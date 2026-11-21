#!/usr/bin/env python3
"""Non-destructive merge of the existing official TimeMixer and new baselines.
No stale M->M results mixed with M->S. All results must share 2024/splits/windows.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

COLS=['dataset','L','H','model','family','seed','mse','mae','rmse',
      'pct_mse_improvement_vs_target_only_tm','pct_mse_improvement_vs_joint_tm']


def read(path,family):
    if not path.is_file():return []
    with path.open(encoding='utf8',newline='') as f:
        rows=list(csv.DictReader(f))
    collected={}
    for r in rows:
        if r.get('status')!='ok':continue
        try:
            key=(r['dataset'],int(r['L']),int(r['H']),r['variant'],int(r['seed']))
            row=dict(dataset=key[0],L=key[1],H=key[2],model=key[3],family=family,
                     seed=key[4],mse=float(r['mse']),mae=float(r['mae']),rmse=float(r['rmse']))
            collected[key]=row
        except (ValueError,KeyError):
            continue
    return list(collected.values())


def merge(root,mode='full'):
    ours=read(root/'results'/'m2s_baselines'/mode/'summary.csv','reference_baseline')
    old=read(root/'results'/'m2s'/mode/'summary.csv','official_TimeMixer')
    rows=ours+old
    comparison={}
    for r in rows:
        k=(r['dataset'],r['L'],r['H'],r['seed'])
        if r['family']=='official_TimeMixer':
            comparison[(k,r['model'])]=r['mse']
    for r in rows:
        k=(r['dataset'],r['L'],r['H'],r['seed'])
        for name,label in [('target_only','pct_mse_improvement_vs_target_only_tm'),
                           ('multivariate_joint','pct_mse_improvement_vs_joint_tm')]:
            reference=comparison.get((k,name))
            r[label]=round(100*(reference-r['mse'])/reference,5) if reference and reference>0 else ''
    rows.sort(key=lambda r:(r['dataset'],r['L'],r['H'],r['seed'],r['mse']))
    out=root/'results'/'m2s_benchmarks'/mode
    out.mkdir(parents=True,exist_ok=True)
    with (out/'leaderboard.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=COLS);w.writeheader();w.writerows(rows)
    return len(rows),str(out/'leaderboard.csv')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--mode',choices=['full','smoke'],default='full')
    a=p.parse_args()
    print('[MERGED]',merge(a.root,a.mode))
