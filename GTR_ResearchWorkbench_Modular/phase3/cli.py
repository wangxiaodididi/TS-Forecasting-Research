"""Orchestrate per-checkpoint validation predictions in clean subprocesses."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from lab.paths import ROOT

MODELS=('GTR','GTRLinearResidual','GTRPhaseInnovation','GTRPhaseAdaptive')


def find_batch(name=None,dataset='Weather',horizon=96):
    home=ROOT/'phase2_batches'
    if name:
        candidates=[home/name]
    else:
        candidates=sorted(home.glob('*/state.json'),reverse=True)
        candidates=[p.parent for p in candidates]
    for folder in candidates:
        try:
            state=json.loads((folder/'state.json').read_text(encoding='utf-8'))
        except (FileNotFoundError,json.JSONDecodeError):continue
        if state.get('dataset')!=dataset or int(state.get('horizon',-1))!=horizon:
            continue
        if state.get('phase')!='completed' or state.get('mode')!='paper':
            continue
        run_map={job['model']:job['run_id'] for job in state.get('jobs',[])}
        if state.get('baseline_run'):
            run_map['GTR']=state['baseline_run']
        if all(m in run_map for m in MODELS):return folder,run_map,state
    raise FileNotFoundError(f'No completed paper Phase2 batch {name or "latest"} for {dataset} H{horizon}')


def analyse(batch=None,dataset='Weather',horizon=96,limit=1024,device='auto'):
    folder,runs,state=find_batch(batch,dataset,horizon)
    out=ROOT/'phase3_reports'/folder.name
    out.mkdir(parents=True,exist_ok=True)
    first_meta=out/'GTR.json'
    for model in MODELS:
        p=out/f'{model}.csv'
        cmd=[sys.executable,'-u','-m','phase3.predict','--run-id',runs[model],
             '--out',str(p),'--limit',str(limit),'--device',device]
        if model != 'GTR':
            if not first_meta.exists():raise FileNotFoundError(first_meta)
            cmd.extend(['--indices-json',str(out/'indices.json')])
        print('Running:',model,runs[model],flush=True)
        subprocess.run(cmd,cwd=ROOT,check=True)
        if model=='GTR':
            meta=json.loads(first_meta.read_text(encoding='utf-8'))
            (out/'indices.json').write_text(json.dumps(meta['sample_indices']),encoding='utf-8')
    subprocess.run([sys.executable,'-m','phase3.report',str(out)],cwd=ROOT,check=True)
    print('RESULT DIR:',out)
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--batch')
    p.add_argument('--dataset',default='Weather',choices=['Weather','ETTm1','ETTh1','Electricity'])
    p.add_argument('--horizon',type=int,default=96)
    p.add_argument('--limit',type=int,default=1024,help='0 = all validation windows')
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    a=p.parse_args()
    analyse(a.batch,a.dataset,a.horizon,a.limit,a.device)

if __name__=='__main__':main()
