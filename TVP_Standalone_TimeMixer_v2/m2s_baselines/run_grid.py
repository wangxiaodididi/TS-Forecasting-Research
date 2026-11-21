#!/usr/bin/env python3
"""Run baseline suite and immediately update merged leaderboard after each run."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
from m2s.data import CONFIGS
from m2s_baselines.models import BENCHMARKS
from m2s_baselines.merge import merge

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--datasets',nargs='+',choices=list(CONFIGS),default=list(CONFIGS))
    p.add_argument('--variants',nargs='+',choices=BENCHMARKS,
        default=['last_value','seasonal_naive','dlinear_target','joint_linear','gru_joint','lstm_joint','tcn_joint'])
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--max-train-windows',type=int,default=0)
    p.add_argument('--max-eval-windows',type=int,default=0)
    p.add_argument('--first-horizon-only',action='store_true')
    p.add_argument('--seed',type=int,default=2024)
    p.add_argument('--force',action='store_true')
    a=p.parse_args()
    mode='smoke' if a.max_train_windows or a.max_eval_windows else 'full'
    out=ROOT/'results'/'m2s_baselines'/mode
    out.mkdir(parents=True,exist_ok=True)
    summary=out/'summary.csv';recs=[];failed=[]
    for d in a.datasets:
        L,hs=CONFIGS[d]
        for h in (hs[:1] if a.first_horizon_only else hs):
            for v in a.variants:
                cmd=[sys.executable,'-u','-m','m2s_baselines.train','--dataset',d,
                     '--horizon',str(h),'--variant',v,'--device',a.device,
                     '--epochs',str(a.epochs),'--patience',str(a.patience),
                     '--batch-size',str(a.batch_size),'--lr',str(a.lr),
                     '--max-train-windows',str(a.max_train_windows),
                     '--max-eval-windows',str(a.max_eval_windows),'--seed',str(a.seed)]
                if a.force:cmd.append('--force')
                log=out/'logs'/f'{d}_L{L}_H{h}_{v}_s{a.seed}.log'
                log.parent.mkdir(parents=True,exist_ok=True)
                print('[RUN]',d,h,v,flush=True)
                st=time.monotonic()
                with log.open('w',encoding='utf8') as f:
                    f.write('COMMAND: '+' '.join(cmd)+'\n');f.flush()
                    proc=subprocess.Popen(cmd,cwd=ROOT,text=True,
                                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=1)
                    for line in proc.stdout:
                        f.write(line)
                        if line.startswith(('epoch=','[TEST]','[SAVED]')) or 'Traceback' in line:
                            print(line.rstrip(),flush=True)
                    code=proc.wait()
                row=dict(dataset=d,L=L,H=h,variant=v,seed=a.seed,
                         status='ok' if code==0 else 'failed',
                         seconds=round(time.monotonic()-st,2),mse='',mae='',rmse='',
                         log=str(log.relative_to(ROOT)))
                metrics=out/d/f'h{h}'/v/f's{a.seed}'/'metrics.json'
                if code==0 and metrics.exists():
                    obj=json.loads(metrics.read_text(encoding='utf8'))
                    row.update(mse=obj['test']['mse'],mae=obj['test']['mae'],
                               rmse=obj['test']['rmse'])
                else:
                    failed.append(row)
                    print('[FAILED]',str(log),flush=True)
                recs.append(row)
                with summary.open('w',newline='',encoding='utf8') as f:
                    w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(recs)
                merge(ROOT,mode)
    print('[SUMMARY]',summary,flush=True)
    print('[LEADERBOARD]',ROOT/'results'/'m2s_benchmarks'/mode/'leaderboard.csv',flush=True)
    if failed:raise SystemExit(f'{len(failed)} experiments failed; see logs')
if __name__=='__main__':main()
