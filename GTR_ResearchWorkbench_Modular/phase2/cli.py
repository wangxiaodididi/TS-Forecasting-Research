"""Launch GTR Phase2 comparison and report validation metrics only."""
import argparse
import subprocess
import sys
from lab.paths import ROOT, RUNS
from .batch import (MODELS,make_batch,resolve_batch,worker,
                    validation_rows,summarize_validation)
from .batch import load_json,read_loss


def main():
    p=argparse.ArgumentParser(description='GTR phase2: phase-calibrated local innovations')
    sp=p.add_subparsers(dest='action',required=True)
    start=sp.add_parser('start')
    start.add_argument('--dataset',choices=['ETTm1','ETTh1','Weather','Electricity'],default='Weather')
    start.add_argument('--horizon',type=int,choices=[96,192,336,720],default=96)
    start.add_argument('--seed',type=int,default=2024)
    start.add_argument('--mode',choices=['smoke','paper'],default='smoke')
    w=sp.add_parser('worker');w.add_argument('id')
    stat=sp.add_parser('status');stat.add_argument('--batch')
    summ=sp.add_parser('validation');summ.add_argument('--batch')
    a=p.parse_args()
    if a.action=='start':
        batch=make_batch(a.dataset,a.horizon,a.seed,a.mode)
        f=(batch/'orchestrator.log').open('a',encoding='utf-8')
        try:
            proc=subprocess.Popen([sys.executable,'-u','-m','phase2.cli','worker',batch.name],
                                  cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
                                  start_new_session=True,close_fds=True)
        finally:f.close()
        print('Batch ID:',batch.name)
        print('Models:',','.join(MODELS))
        print('Log:',batch/'orchestrator.log')
        print('Check: python -m phase2.cli status')
    elif a.action=='worker':
        sys.exit(worker(resolve_batch(a.id)))
    elif a.action=='status':
        batch=resolve_batch(a.batch);state=load_json(batch/'state.json')
        print('Batch',batch.name,'status:',state['phase'],'active:',state.get('current_model','none'))
        for r in validation_rows(batch):
            print('{model:24s} {status:12s} best_val={best_val_mse} epoch={best_epoch} run={run_id}'.format(**r))
        if state.get('reason'):print('Reason:',state['reason'])
    elif a.action=='validation':
        batch,out,dest=summarize_validation(a.batch)
        for row in out:print(row)
        print('Validation CSV:',dest)
        print('Test MSE/MAE intentionally NOT shown. Use validation for model selection.')

if __name__=='__main__':main()
