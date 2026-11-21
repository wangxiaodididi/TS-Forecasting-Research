"""Control Phase-1 experiments: start, status, validation comparison, unlocked test."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from .core import new_batch,worker,resolve_batch,summarize,load_json,read_loss,EXTENSIONS
from lab.paths import ROOT,RUNS


def main():
    p=argparse.ArgumentParser(description='GTR phase-1. Runs are sequential to avoid GPU contention.')
    sub=p.add_subparsers(dest='action',required=True)
    s=sub.add_parser('start');s.add_argument('--dataset',default='ETTm1',choices=['ETTm1','ETTh1','Weather','Electricity'])
    s.add_argument('--horizon',type=int,default=96,choices=[96,192,336,720]);s.add_argument('--seed',type=int,default=2024)
    s.add_argument('--mode',choices=['smoke','paper'],default='smoke');s.add_argument('--baseline-run',default=None)
    t=sub.add_parser('worker');t.add_argument('batch')
    s=sub.add_parser('status');s.add_argument('--batch',default=None)
    s=sub.add_parser('summary');s.add_argument('--batch',default=None);s.add_argument('--include-test',action='store_true')
    a=p.parse_args()
    if a.action=='start':
        batch=new_batch(a.dataset,a.horizon,a.seed,a.mode,a.baseline_run)
        log=(batch/'orchestrator.log').open('a',encoding='utf-8')
        try:
            proc=subprocess.Popen([sys.executable,'-u','-m','phase1.launch','worker',batch.name],cwd=ROOT,
                                  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        finally:log.close()
        # Worker writes its own PID; avoid racing concurrent state.json updates.
        print('Batch started:',batch.name)
        print('Models:',', '.join(EXTENSIONS))
        print('Orchestrator log:',batch/'orchestrator.log')
        print('Refresh Streamlit > Phase1 Ablation page (or run `python -m phase1.launch status`).')
    elif a.action=='worker':
        pth=resolve_batch(a.batch);sys.exit(worker(pth))
    elif a.action=='status':
        batch=resolve_batch(a.batch);state=load_json(batch/'state.json')
        print('Batch:',batch.name,'| status:',state.get('phase'),'| current:',state.get('current_model','none'))
        for j in state.get('jobs',[]):
            path=RUNS/j['run_id']/'status.json'
            q=load_json(path) if path.exists() else {}
            val=read_loss(j['run_id']);best=min((v['val_loss'] for v in val),default=None)
            print(f'{j["model"]:20s} {q.get("state","queued"):10s} epoch={q.get("epoch",0):>2} best_val={best} run={j["run_id"]}')
        if state.get('reason'):print('Reason:',state['reason'])
    elif a.action=='summary':
        batch,state,rows,dest=summarize(a.batch,include_test=a.include_test)
        print('Batch',batch.name, '|',state['phase'])
        for r in rows:
            print('{model:19s} {status:10s} best_val_mse={best_val_mse} best_epoch={best_epoch}'.format(**{k:r.get(k) for k in ['model','status','best_val_mse','best_epoch']}))
        print('CSV:',dest)
        if a.include_test:print('WARNING: final-test metrics shown; do not use for choosing variants or hyperparameters')

if __name__=='__main__':main()
