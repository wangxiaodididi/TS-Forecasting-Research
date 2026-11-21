"""Phase 1: sequential, detached model trials without altering upstream source."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from lab.paths import ROOT, RUNS, DATA, VENDOR
from lab.config import config_for, command_args
from lab.source import verify
from lab.runner import make_run, run_worker

BATCHES = ROOT / 'phase1_batches'
EXTENSIONS = ('GTRLinearResidual', 'GTRHaarHigh', 'GTRHaarLow', 'GTRHaarDual')
BASELINE_METRICS = {'MSE': 0.305, 'MAE': 0.349}  # ETTm1 H96 from ICLR 2026 table (not training losses)


def atomic_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(tmp,path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):h.update(block)
    return h.hexdigest()


def compatible_baseline(runid, dataset, horizon, seed):
    path=RUNS/runid
    if not (path/'config.json').is_file() or not (path/'status.json').is_file():
        raise ValueError(f'Missing baseline config or status: {path}')
    cfg=load_json(path/'config.json')
    status=load_json(path/'status.json')
    expected=config_for(dataset,'GTR',horizon,seed,'paper')
    if status.get('state')!='completed':
        raise ValueError(f'Baseline {runid} not completed (state={status.get("state")})')
    if not cfg.get('paper_exact',False):
        raise ValueError(f'Baseline {runid} was not an exact official paper run')
    for k in ['dataset','model','pred_len','seq_len','cycle','enc_in','batch_size','learning_rate','dropout','train_epochs','patience','random_seed']:
        if cfg.get(k)!=expected.get(k):
            raise ValueError(f'Baseline {runid} config mismatch for {k}: {cfg.get(k)} != {expected.get(k)}')
    env_path=path/'environment.json'
    if env_path.is_file():
        old_hash=load_json(env_path).get('dataset_sha256')
        if old_hash and old_hash!=sha256(DATA/expected['data_path']):
            raise ValueError('Dataset SHA256 differs from baseline. Abort unfair comparison.')
    return path


def find_baseline(dataset,horizon,seed):
    if not RUNS.exists():return None
    for item in sorted(RUNS.glob('*/status.json'),reverse=True):
        try:
            compatible_baseline(item.parent.name,dataset,horizon,seed)
            return item.parent.name
        except (ValueError,KeyError,FileNotFoundError,TypeError,PermissionError):continue
    return None


def preflight(dataset, horizon, seed, mode, baseline_run=None):
    verify(VENDOR)
    config_for(dataset,'GTR',horizon,seed,mode)
    baseline_path=None
    if mode=='paper':
        runid=baseline_run or find_baseline(dataset,horizon,seed)
        if not runid:
            raise ValueError('No matching completed official baseline found. Run official GTR first; or specify --baseline-run.')
        baseline_path=compatible_baseline(runid,dataset,horizon,seed)
    for model in EXTENSIONS:
        cfg=config_for(dataset,model,horizon,seed,mode)
        if cfg['overrides']:
            raise ValueError(f'Model {model} has parameter overrides: {cfg["overrides"]}. Clear configs/experiments YAML to compare fairly.')
        official=config_for(dataset,'GTR',horizon,seed,mode)
        x=command_args(cfg);y=command_args(official)
        def as_dict(items):return dict(zip((v[2:] for v in items[::2]), items[1::2]))
        d1=as_dict(x);d0=as_dict(y)
        for key in d0:
            if key not in ('model','model_id') and d1.get(key)!=d0[key]:
                raise ValueError(f'{model} != official setting at {key}')
        for name in [model,'GTRSignalCore']:
            if not (ROOT/'extensions'/'models'/f'{name}.py').is_file():
                raise FileNotFoundError(f'Missing extension: extensions/models/{name}.py')
    path=DATA/config_for(dataset,'GTR',horizon,seed,mode)['data_path']
    if not path.exists():raise FileNotFoundError(path)
    return baseline_path


def new_batch(dataset,horizon,seed,mode,baseline_run=None):
    baseline=preflight(dataset,horizon,seed,mode,baseline_run)
    BATCHES.mkdir(parents=True,exist_ok=True)
    bid=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:6]
    base={'id':bid,'dataset':dataset,'horizon':horizon,'seed':seed,'mode':mode,
          'created_at':datetime.now().isoformat(),'baseline_run':baseline.name if baseline else None,
          'models':list(EXTENSIONS),'phase':'queued','jobs':[],
          'note':'Selection uses validation only. Official GTR trainer reports test metrics automatically; do not select on them.'}
    batch=BATCHES/bid;batch.mkdir()
    atomic_json(batch/'state.json',base)
    return batch


def worker(batch_dir):
    batch=Path(batch_dir);state=load_json(batch/'state.json')
    state['phase']='running';state['worker_pid']=os.getpid();atomic_json(batch/'state.json',state)
    try:
        for model in state['models']:
            # In each run the original GTR source remains unchanged; extensions are copied only into a source snapshot.
            cfg=config_for(state['dataset'],model,state['horizon'],state['seed'],state['mode'])
            rid=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:7]
            rp=make_run(RUNS/rid,cfg)
            state['current_model']=model
            state['current_run_id']=rid
            state['jobs'].append({'model':model,'run_id':rid})
            atomic_json(batch/'state.json',state)
            print(f'\n=== START {model}: {rid} ===',flush=True)
            run_worker(rp)
            s=load_json(rp/'status.json')
            print(f'=== FINISH {model}: {s.get("state")}, best validation recorded in loss.csv ===',flush=True)
            if s.get('state')!='completed':
                state['phase']='failed';state['reason']=f'{model} job {rid} ended as {s.get("state")}';
                atomic_json(batch/'state.json',state)
                print(state['reason'],flush=True)
                return 1
        state['phase']='completed';state['finished_at']=datetime.now().isoformat()
        state.pop('current_model',None)
        atomic_json(batch/'state.json',state)
        print('=== ALL FOUR EXTENSIONS COMPLETED ===',flush=True)
        return 0
    except Exception as exc:
        import traceback
        state['phase']='failed';state['reason']=f'{type(exc).__name__}: {exc}'
        atomic_json(batch/'state.json',state)
        traceback.print_exc()
        return 1


def read_loss(run_id):
    p=RUNS/run_id/'loss.csv';rows=[]
    if p.is_file():
        with p.open(newline='',encoding='utf-8') as f:
            for r in csv.DictReader(f):
                try:rows.append({'epoch':int(r['epoch']),'train_loss':float(r['train_loss']),'val_loss':float(r['val_loss'])})
                except (ValueError,TypeError,KeyError):continue
    return rows


def summarize(batch_id=None, include_test=False):
    batch=resolve_batch(batch_id);state=load_json(batch/'state.json')
    candidates=[]
    if state.get('baseline_run'):
        candidates.append(('GTR',state['baseline_run']))
    candidates.extend((j['model'],j['run_id']) for j in state.get('jobs',[]))
    out=[]
    for model,rid in candidates:
        try:
            r=RUNS/rid;s=load_json(r/'status.json');vals=read_loss(rid)
            best=min(vals,key=lambda row:row['val_loss']) if vals else {}
            row={'model':model,'run_id':rid,'status':s.get('state','unknown'),
                 'best_val_mse':best.get('val_loss'),'best_epoch':best.get('epoch'),
                 'train_mse_at_best':next((v['train_loss'] for v in vals if v['epoch']==best.get('epoch')),None),
                 'last_epoch':s.get('epoch'),'duration_sec':s.get('duration_sec')}
            if include_test:row.update(test_mse=s.get('final_test_mse'),test_mae=s.get('final_test_mae'))
            out.append(row)
        except (OSError,KeyError,ValueError) as exc:
            out.append({'model':model,'run_id':rid,'status':f'parse error: {exc}'})
    filename='full_metrics_AFTER_MODEL_LOCK.csv' if include_test else 'VALIDATION_ONLY.csv'
    dest=batch/filename
    fields=['model','run_id','status','best_val_mse','best_epoch','train_mse_at_best','last_epoch','duration_sec']
    if include_test:fields+=['test_mse','test_mae']
    with dest.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(out)
    return batch,state,out,dest


def resolve_batch(batch_id=None):
    if batch_id:
        p=BATCHES/batch_id
        if not (p/'state.json').is_file():raise FileNotFoundError(p/'state.json')
        return p
    dirs=sorted(p.parent for p in BATCHES.glob('*/state.json')) if BATCHES.exists() else []
    if not dirs:raise FileNotFoundError('No Phase-1 batches. Start a batch first.')
    return dirs[-1]
