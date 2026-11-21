#!/usr/bin/env python3
"""Summarize a single run's completed per-job JSON, preserving exact seeds.

No test-based hyperparameter optimizer here. Missing pairs are explicitly listed.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parent
OUT=HERE/'outputs'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('run_dir',help='runs/<id> or path to the run directory')
    args=p.parse_args()
    rd=Path(args.run_dir).resolve()
    if not rd.is_dir() or not (rd/'manifest.json').exists():
        raise SystemExit(f'Run directory/manifest missing: {rd}')
    rows=[]
    for f in sorted((rd/'jobs').glob('*/result.json')):
        rec=json.loads(f.read_text(encoding='utf-8'))
        if rec.get('status')=='completed' and all(k in rec for k in ('mse','mae','dataset','horizon','seed','variant')):
            rows.append(rec)
    if not rows:raise SystemExit('No completed jobs found; refusing to fabricate metrics')
    d={(r['dataset'],int(r['horizon']),int(r['seed']),r['variant']):r for r in rows}
    paired=[];miss=[]
    for key,r in d.items():
        dataset,horizon,seed,variant=key
        if variant in ('official','lg_short'):continue
        for reference in ('official','lg_short'):
            k=(dataset,horizon,seed,reference)
            if k not in d:
                miss.append((*key,reference));continue
            b=d[k]
            paired.append(dict(dataset=dataset,horizon=horizon,seed=seed,variant=variant,
                reference=reference,mse=float(r['mse']),mae=float(r['mae']),
                reference_mse=float(b['mse']),reference_mae=float(b['mae']),
                mse_gain_pct=100*(float(b['mse'])-float(r['mse']))/float(b['mse']),
                mae_gain_pct=100*(float(b['mae'])-float(r['mae']))/float(b['mae'])))
    OUT.mkdir(parents=True,exist_ok=True)
    columns=['dataset','horizon','seed','variant','reference','mse','mae','reference_mse','reference_mae','mse_gain_pct','mae_gain_pct']
    with (OUT/'paired.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(paired)
    from collections import defaultdict
    groups=defaultdict(list)
    for r in paired:groups[(r['dataset'],r['horizon'],r['variant'],r['reference'])].append(r)
    summary=[]
    for key,g in sorted(groups.items()):
        summary.append(dict(dataset=key[0],horizon=key[1],variant=key[2],reference=key[3],
            seeds=len(g),mean_mse_gain_pct=mean(v['mse_gain_pct'] for v in g),
            mean_mae_gain_pct=mean(v['mae_gain_pct'] for v in g),
            mse_wins=sum(v['mse_gain_pct']>0 for v in g),mae_wins=sum(v['mae_gain_pct']>0 for v in g)))
    if summary:
        with (OUT/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    if miss:
        with (OUT/'missing_pairs.json').open('w',encoding='utf-8') as f:json.dump(miss,f,indent=2,ensure_ascii=False)
    print(f'Completed jobs={len(rows)}; paired comparisons={len(paired)}; missing comparisons={len(miss)}')
    for r in summary:
        print(f"{r['dataset']}-{r['horizon']} {r['variant']} vs {r['reference']} "
              f"n={r['seeds']} MSE gain={r['mean_mse_gain_pct']:+.3f}% MAE gain={r['mean_mae_gain_pct']:+.3f}% "
              f"MSE wins={r['mse_wins']} MAE wins={r['mae_wins']}")
    print('Outputs:',OUT)

if __name__=='__main__':main()
