"""Sequential validation-first Phase2, using the existing verified experiment runner.

All runs write to the existing /runs folders; nothing is overwritten.  The
original GTR source and its official training hyperparameters stay unchanged.
"""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime
from uuid import uuid4

from lab.paths import ROOT, RUNS, DATA, VENDOR
from lab.config import config_for, command_args
from lab.source import verify
from lab.runner import make_run, run_worker

def atomic_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(tmp,path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_loss(run_id):
    p=RUNS/run_id/'loss.csv';rows=[]
    if p.is_file():
        with p.open(newline='',encoding='utf-8') as f:
            for r in csv.DictReader(f):
                try:
                    rows.append({'epoch':int(r['epoch']),
                                 'train_loss':float(r['train_loss']),
                                 'val_loss':float(r['val_loss'])})
                except (ValueError,TypeError,KeyError):pass
    return rows


def file_sha256(path):
    hasher=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            hasher.update(block)
    return hasher.hexdigest()


def compatible_baseline(run_id,dataset,horizon,seed):
    p=RUNS/run_id
    if not (p/'config.json').is_file() or not (p/'status.json').is_file():
        raise ValueError('Baseline config or status missing')
    cfg=load_json(p/'config.json');status=load_json(p/'status.json')
    expected=config_for(dataset,'GTR',horizon,seed,'paper')
    if status.get('state')!='completed' or not cfg.get('paper_exact',False):
        raise ValueError('Not a completed exact official baseline')
    for k in ['dataset','model','pred_len','seq_len','cycle','enc_in','batch_size',
              'learning_rate','dropout','train_epochs','patience','random_seed']:
        if cfg.get(k)!=expected.get(k):
            raise ValueError(f'Baseline differs at {k}')
    env=p/'environment.json'
    if env.is_file():
        prior=load_json(env).get('dataset_sha256')
        if prior and prior!=file_sha256(DATA/expected['data_path']):
            raise ValueError('Dataset hash changed since baseline')
    return p


def find_baseline(dataset,horizon,seed):
    if not RUNS.exists():return None
    for status in sorted(RUNS.glob('*/status.json'),reverse=True):
        try:
            compatible_baseline(status.parent.name,dataset,horizon,seed)
            return status.parent.name
        except (OSError,ValueError,KeyError,TypeError):continue
    return None


BATCHES = ROOT/'phase2_batches'
MODELS = ('GTR', 'GTRLinearResidual', 'GTRPhaseInnovation', 'GTRPhaseAdaptive')


def model_config_match(model, dataset, horizon, seed, mode):
    baseline = config_for(dataset, 'GTR', horizon, seed, mode)
    cfg = config_for(dataset, model, horizon, seed, mode)
    if cfg.get('overrides'):
        raise RuntimeError(f'Unexpected experiment parameter override for {model}: {cfg["overrides"]}')
    a = dict(zip(command_args(baseline)[::2], command_args(baseline)[1::2]))
    b = dict(zip(command_args(cfg)[::2], command_args(cfg)[1::2]))
    for k in a:
        if k not in ('--model',) and a[k]!=b[k]:
            raise RuntimeError(f'Config differs for {model}: {k} {a[k]} vs {b[k]}')
    return cfg


def preflight(dataset, horizon, seed, mode):
    verify(VENDOR)
    expected=config_for(dataset,'GTR',horizon,seed,mode)
    if not (DATA/expected['data_path']).is_file():
        raise FileNotFoundError(DATA/expected['data_path'])
    if mode not in ('smoke','paper'):raise ValueError('mode must be smoke or paper')
    for model in MODELS:
        model_config_match(model,dataset,horizon,seed,mode)
    for name in ('GTRPhaseCore','GTRPhaseInnovation','GTRPhaseAdaptive','GTRLinearResidual'):
        if not (ROOT/'extensions/models'/f'{name}.py').is_file():
            raise FileNotFoundError(f'Missing extension: {name}')
    return expected


def make_batch(dataset='Weather',horizon=96,seed=2024,mode='smoke'):
    preflight(dataset,horizon,seed,mode)
    baseline = find_baseline(dataset,horizon,seed) if mode=='paper' else None
    if baseline:
        compatible_baseline(baseline,dataset,horizon,seed)
    bid=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:6]
    dest=BATCHES/bid;dest.mkdir(parents=True,exist_ok=False)
    state={'id':bid,'dataset':dataset,'horizon':horizon,'seed':seed,'mode':mode,
           'baseline_run':baseline,'phase':'queued','jobs':[],
           'models':list(MODELS),
           'source':'official GTR ICLR 2026 pinned source; model cores unmodified',
           'selection_protocol':'Compare best validation only; test data are generated by the unmodified official trainer and MUST NOT be used for model selection.',
           'description':'Phase-calibrated local innovation relative to learned global phase reference'}
    atomic_json(dest/'state.json',state)
    return dest


def resolve_batch(name=None):
    if name:
        dest=BATCHES/name
        if not (dest/'state.json').is_file():raise FileNotFoundError(dest/'state.json')
        return dest
    dirs=sorted(x.parent for x in BATCHES.glob('*/state.json')) if BATCHES.exists() else []
    if not dirs:raise FileNotFoundError('No Phase2 batch has been launched')
    return dirs[-1]


def worker(batchdir):
    dest=Path(batchdir)
    state=load_json(dest/'state.json')
    state['phase']='running';state['worker_pid']=os.getpid()
    atomic_json(dest/'state.json',state)
    try:
        models = MODELS[1:] if state.get('baseline_run') else MODELS
        for model in models:
            cfg=model_config_match(model,state['dataset'],state['horizon'],state['seed'],state['mode'])
            rid=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:7]
            run_path=make_run(RUNS/rid,cfg)
            state['current_model']=model;state['current_run_id']=rid
            state['jobs'].append({'model':model,'run_id':rid})
            atomic_json(dest/'state.json',state)
            print(f'[PHASE2] Starting {model}: {rid}',flush=True)
            run_worker(run_path)
            result=load_json(run_path/'status.json')
            print(f'[PHASE2] {model} finished: {result.get("state")}',flush=True)
            if result.get('state')!='completed':
                state['phase']='failed';state['reason']=f'Run {rid} failed ({result.get("state")})'
                atomic_json(dest/'state.json',state)
                return 1
        state['phase']='completed';state.pop('current_model',None)
        state['finished_at']=datetime.now().isoformat()
        atomic_json(dest/'state.json',state)
        print('[PHASE2] All experiments completed',flush=True)
        return 0
    except Exception as exc:
        import traceback
        state['phase']='failed';state['reason']=f'{type(exc).__name__}: {exc}'
        atomic_json(dest/'state.json',state)
        traceback.print_exc()
        return 1


def validation_rows(batch):
    state=load_json(Path(batch)/'state.json')
    trials=[]
    if state.get('baseline_run'):
        trials.append(('GTR',state['baseline_run']))
    trials.extend((j['model'],j['run_id']) for j in state.get('jobs',[]))
    out=[]
    for model,rid in trials:
        status_path=RUNS/rid/'status.json'
        s=load_json(status_path) if status_path.is_file() else {}
        curves=read_loss(rid)
        best=min(curves,key=lambda r:r['val_loss']) if curves else {}
        out.append({'model':model,'run_id':rid,'status':s.get('state','unknown'),
                    'best_val_mse':best.get('val_loss'),'best_epoch':best.get('epoch'),
                    'train_mse_at_best':best.get('train_loss'),'last_epoch':s.get('epoch'),
                    'duration_sec':s.get('duration_sec')})
    return out


def summarize_validation(name=None):
    batch=resolve_batch(name)
    out=validation_rows(batch)
    fields=['model','run_id','status','best_val_mse','best_epoch','train_mse_at_best','last_epoch','duration_sec']
    dest=batch/'VALIDATION_ONLY.csv'
    with dest.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
    return batch,out,dest
