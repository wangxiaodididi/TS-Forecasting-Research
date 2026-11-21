"""Checkpoint-based prediction export on the same deterministic validation windows.

Run in a separate subprocess PER MODEL because each run has an independent
(source+checkpoint) snapshot and Python caches `models` after first import.
The official upstream GTR and existing experiments are never modified.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def _hash(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _names(cfg, root):
    import pandas as pd
    names = pd.read_csv(root / 'dataset' / cfg['data_path'], nrows=0).columns.tolist()
    if 'date' not in names:
        raise ValueError('Expected date column in source dataset')
    if cfg['features'] == 'M':
        if cfg['data'] == 'custom':
            target = cfg.get('target', 'OT')
            if target not in names:
                raise ValueError(f'Custom dataset target {target!r} missing')
            return [x for x in names if x not in ('date', target)] + [target]
        return [x for x in names if x != 'date']
    raise ValueError('This diagnostic supports the M/multivariate protocol only')


def _indices(total, limit):
    import numpy as np
    if total < 1: raise ValueError('Empty validation split')
    if limit <= 0 or limit >= total:
        return list(range(total))
    if limit < 3:
        raise ValueError('Use at least 3 validation windows, or 0 for all')
    indexes = np.linspace(0, total - 1, num=limit, dtype=int).tolist()
    if len(set(indexes)) != len(indexes):
        raise ValueError('Duplicate sample index unexpectedly generated')
    return indexes


def export_one(root, run_id, output, limit=1024, device='auto', batch_size=32, existing_indices=None):
    import numpy as np
    import torch
    from lab.paths import RUNS

    root=Path(root).resolve()
    run=RUNS/run_id
    cfg=json.loads((run/'config.json').read_text(encoding='utf-8'))
    status=json.loads((run/'status.json').read_text(encoding='utf-8'))
    if status.get('state') != 'completed':
        raise RuntimeError(f'{run_id} not completed: {status.get("state")}')
    if cfg.get('mode') != 'paper':
        raise ValueError('Only full paper-mode experiments are eligible')
    snapshot=run/'source'
    checkpoint=list(snapshot.glob('checkpoints/*/checkpoint.pth'))
    if len(checkpoint) != 1:
        raise FileNotFoundError(f'Expected ONE best checkpoint, found {len(checkpoint)} for {run_id}')
    # Current working tree has no /models; all model imports must come from
    # this immutable snapshot. We launch a new process for every run.
    sys.path.insert(0,str(snapshot.resolve()))
    from data_provider.data_factory import data_provider
    module=importlib.import_module('models.'+cfg['model'])
    defaults = dict(label_len=0,embed='timeF',freq='h',target='OT',num_workers=0,
                    d_model=512,use_revin=1,individual=0,dropout=0.0,batch_size=32)
    args=SimpleNamespace(**{**defaults,**cfg,'num_workers':0})
    model=module.Model(args).float()
    try:
        state=torch.load(checkpoint[0],map_location='cpu',weights_only=True)
    except TypeError:
        state=torch.load(checkpoint[0],map_location='cpu')
    model.load_state_dict(state,strict=True)
    if device == 'auto': device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    model=model.to(device).eval()

    data,_=data_provider(args,'val')
    total=len(data)
    if existing_indices is None:
        indices=_indices(total,int(limit))
    else:
        indices=json.loads(Path(existing_indices).read_text(encoding='utf-8'))
        if not isinstance(indices,list) or not indices or any(not isinstance(i,int) or i<0 or i>=total for i in indices):
            raise ValueError('Invalid/mismatched validation index list')
    names=_names(cfg,root)
    channels=int(cfg['enc_in'])
    if len(names)!=channels:
        raise ValueError(f'Variable count mismatch: CSV has {len(names)}; enc_in={channels}')
    source_csv=root/'dataset'/cfg['data_path']
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fieldnames=['model','run_id','sample_index','channel','channel_name','mse','mae',
                'relative_change','relative_drift','phase_fit_r2','correction_rms','correction_gain']
    nrows=0
    with output.open('w',newline='',encoding='utf-8') as f, torch.no_grad():
        writer=csv.DictWriter(f,fieldnames=fieldnames)
        writer.writeheader()
        for pos in range(0,len(indices),int(batch_size)):
            choice=indices[pos:pos+int(batch_size)]
            examples=[data[i] for i in choice]
            x=np.stack([np.asarray(s[0],dtype=np.float32) for s in examples],axis=0)
            y=np.stack([np.asarray(s[1],dtype=np.float32)[-cfg['pred_len']:] for s in examples],axis=0)
            cycle=np.array([int(s[4]) for s in examples],dtype=np.int64)
            with torch.no_grad():
                xt=torch.from_numpy(x).to(device)
                ct=torch.as_tensor(cycle,device=device,dtype=torch.long)
                if cfg['model'] != 'GTR' and hasattr(model,'explain'):
                    components=model.explain(xt,ct)
                    pred=components['prediction']
                    if 'innovation_correction' in components:
                        correction=components['innovation_correction']
                    elif 'linear_correction' in components:
                        correction=components['linear_correction']
                    else:
                        correction=None
                    r2=components.get('phase_fit_r2')
                    gain=components.get('correction_gain')
                else:
                    pred=model(xt,ct);correction=None;r2=None;gain=None
                if pred.shape != (len(choice),int(cfg['pred_len']),channels):
                    raise RuntimeError(f'Output shape mismatch {pred.shape}')
                pr=pred.detach().float().cpu().numpy()
                corr=(correction.detach().float().square().mean(dim=1).sqrt().cpu().numpy()
                      if correction is not None else None)
                r2=(r2.detach().float().cpu().numpy() if r2 is not None else None)
                gain=(gain.detach().float().cpu().numpy() if gain is not None else None)
            mse=np.mean((pr-y)**2,axis=1)
            mae=np.mean(np.abs(pr-y),axis=1)
            std=np.std(x,axis=1,ddof=1).clip(min=1e-4)
            change=np.sqrt(np.mean(np.diff(x,axis=1)**2,axis=1))/std
            k=max(2,int(cfg['seq_len'])//4)
            drift=np.abs(x[:,-k:,:].mean(axis=1)-x[:,:k,:].mean(axis=1))/std
            for b,s in enumerate(choice):
                for c in range(channels):
                    writer.writerow(dict(model=cfg['model'],run_id=run_id,
                       sample_index=s,channel=c,channel_name=names[c],
                       mse=f'{float(mse[b,c]):.10g}',mae=f'{float(mae[b,c]):.10g}',
                       relative_change=f'{float(change[b,c]):.10g}',
                       relative_drift=f'{float(drift[b,c]):.10g}',
                       phase_fit_r2=f'{float(r2[b,c]):.10g}' if r2 is not None else '',
                       correction_rms=f'{float(corr[b,c]):.10g}' if corr is not None else '',
                       correction_gain=f'{float(gain[b,c]):.10g}' if gain is not None else ''))
                    nrows+=1
            if (pos//batch_size)%10==0 or pos+batch_size>=len(indices):
                print(f'{run_id}: validation windows {min(pos+batch_size,len(indices))}/{len(indices)}',flush=True)
    meta=dict(run_id=run_id,model=cfg['model'],dataset=cfg['dataset'],horizon=int(cfg['pred_len']),
         seed=int(cfg['random_seed']),seq_len=int(cfg['seq_len']),channels=channels,
         channel_names=names,source_dataset_sha256=_hash(source_csv),validation_length=total,
         sample_indices=indices,split='val',samples=len(indices),rows=nrows,
         mode=cfg['mode'],checkpoint=str(checkpoint[0]),device=device,
         note='Deterministic validation windows; MSE is not necessarily identical to shuffled drop_last official val_loss')
    output.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved',output, 'rows',nrows,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-id',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--limit',type=int,default=1024)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--indices-json')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    if a.batch_size < 1: p.error('--batch-size must be positive')
    export_one(root,a.run_id,a.out,a.limit,a.device,a.batch_size,a.indices_json)

if __name__=='__main__':main()
