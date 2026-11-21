#!/usr/bin/env python3
"""Train reference models with the identical m2s windows/splits/scalers as TimeMixer.

Never loads test targets during training or early stopping. Uses original units
for reported metrics. No modification of existing official models/checkpoints.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from m2s.data import load, CONFIGS
from m2s_baselines.models import BENCHMARKS, NO_TRAIN, TARGET_ONLY, make_model

ROOT = Path(__file__).resolve().parents[1]
PERIODS = {'Hubei': 5, 'Guangdong': 5, 'CAISO': 24,
           'PGCB': 24, 'NESO': 48, 'OPSD': 24}


def arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=list(CONFIGS),required=True)
    p.add_argument('--horizon',type=int,required=True)
    p.add_argument('--variant',choices=BENCHMARKS,required=True)
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--seed',type=int,default=2024)
    p.add_argument('--max-train-windows',type=int,default=0)
    p.add_argument('--max-eval-windows',type=int,default=0)
    p.add_argument('--force',action='store_true')
    return p.parse_args()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends,'cudnn'):
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False


def evaluate(model, loader, device, mu, sd, pred_path=None):
    model.eval()
    sq=0.0; ab=0.0; n=0; samples=[]
    with torch.no_grad():
        for xx,yy in loader:
            pred=model(xx.to(device))
            yy=yy.to(device)
            if pred.shape != yy.shape:
                raise ValueError(f'Prediction shape mismatch {pred.shape} != {yy.shape}')
            if not torch.isfinite(pred).all():raise FloatingPointError('Nonfinite prediction')
            diff=pred-yy
            sq+=float(diff.square().sum().item())
            ab+=float(diff.abs().sum().item())
            n+=yy.numel()
            if pred_path and len(samples)<4:
                samples.append((pred.detach().cpu().numpy(),yy.detach().cpu().numpy()))
    if not n:raise RuntimeError('No evaluation windows')
    mse=sq/n;mae=ab/n
    result=dict(mse=mse*sd*sd,mae=mae*sd,rmse=mse**0.5*sd,
                mse_scaled=mse,mae_scaled=mae,n_targets=n,n_windows=len(loader.dataset))
    if pred_path is not None:
        with open(pred_path,'w',newline='',encoding='utf8') as f:
            w=csv.writer(f);w.writerow(['eval_window_index','step','prediction','actual'])
            row=0
            for ps,ys in samples:
                for j in range(ps.shape[0]):
                    for h in range(ps.shape[1]):
                        w.writerow([row,h+1,float(ps[j,h,0]*sd+mu),float(ys[j,h,0]*sd+mu)])
                    row+=1
    return result


def main():
    a=arguments()
    if a.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device=torch.device('cuda' if (a.device=='cuda' or (a.device=='auto' and torch.cuda.is_available())) else 'cpu')
    if device.type=='cpu':torch.set_num_threads(min(4,os.cpu_count() or 1))
    seed_all(a.seed)
    d=load(ROOT,a.dataset,a.horizon)
    target_only=a.variant in TARGET_ONLY
    channels=1 if target_only else len(d.columns)
    model=make_model(a.variant,channels,d.input_len,d.horizon,PERIODS[a.dataset]).to(device)
    mode='smoke' if (a.max_train_windows or a.max_eval_windows) else 'full'
    out=ROOT/'results'/'m2s_baselines'/mode/a.dataset/f'h{a.horizon}'/a.variant/f's{a.seed}'
    out.mkdir(parents=True,exist_ok=True)
    meta={**vars(a),'model_name':a.variant,'channels':channels,'target':'OT',
          'input_length':d.input_len,'horizon':d.horizon,'dataset_columns':d.columns,
          'split_source':f'm2s/window_indices/{a.dataset}_L{d.input_len}_H{d.horizon}.csv',
          'train_end':d.train_end,'train_mean':d.mean.tolist(),'train_std':d.std.tolist(),
          'no_future_covariates':True,'no_engineered_input_features':True,
          'architecture_source':'reference implementation in m2s_baselines/models.py'}
    cfg=out/'config.json';mfile=out/'metrics.json'
    if cfg.exists() and mfile.exists() and not a.force:
        old=json.loads(cfg.read_text(encoding='utf8'))
        if old==meta:
            print('[SKIP identical config]',mfile,flush=True);return
    cfg.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
    loaders={split:d.loader(split,batch_size=a.batch_size,
              max_windows=(a.max_train_windows if split=='train' else a.max_eval_windows),
              target_only=target_only,seed=a.seed,workers=0)
             for split in ('train','val','test')}
    sd=float(d.std[-1]);mu=float(d.mean[-1])
    t0=time.monotonic(); best_epoch=0; hist=[]
    print(f'[START] {a.dataset} L={d.input_len} H={a.horizon} C={channels} variant={a.variant} device={device}',flush=True)
    if a.variant not in NO_TRAIN:
        opt=torch.optim.Adam(model.parameters(),lr=a.lr)
        best=float('inf');bad=0
        for epoch in range(1,a.epochs+1):
            model.train();total=0.0;ct=0
            for xx,yy in loaders['train']:
                xx=xx.to(device); yy=yy.to(device)
                opt.zero_grad(set_to_none=True)
                pred=model(xx)
                if pred.shape!=yy.shape:raise ValueError(f'{pred.shape} vs {yy.shape}')
                loss=(pred-yy).square().mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step()
                total+=float(loss.item())*xx.shape[0];ct+=xx.shape[0]
            val=evaluate(model,loaders['val'],device,mu,sd)
            v=val['mse_scaled'];hist.append((epoch,total/max(ct,1),v))
            print(f'epoch={epoch:03d} train={total/max(ct,1):.6f} val={v:.6f}',flush=True)
            if v<best-1e-9:
                best=v;best_epoch=epoch;bad=0
                torch.save({'epoch':epoch,'model':model.state_dict()},out/'best.pt')
            else:
                bad+=1
            if bad>=a.patience:
                print('[EARLY STOP]',flush=True);break
        if not hist:raise RuntimeError('No training epochs')
        with open(out/'history.csv','w',newline='',encoding='utf8') as f:
            w=csv.writer(f);w.writerow(['epoch','train_mse_scaled','val_mse_scaled']);w.writerows(hist)
        ckpt=torch.load(out/'best.pt',map_location=device,weights_only=True)
        model.load_state_dict(ckpt['model'])
    val=evaluate(model,loaders['val'],device,mu,sd)
    test=evaluate(model,loaders['test'],device,mu,sd,pred_path=out/'test_predictions_sample.csv')
    summary={'dataset':a.dataset,'horizon':a.horizon,'variant':a.variant,
             'mode':mode,'input_channels':channels,'best_epoch':best_epoch,'seed':a.seed,
             'device':str(device),'val':val,'test':test,
             'elapsed_seconds':round(time.monotonic()-t0,2)}
    mfile.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print('[TEST]',json.dumps(test,ensure_ascii=False),flush=True)
    print('[SAVED]',mfile,flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
