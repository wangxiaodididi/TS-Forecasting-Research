#!/usr/bin/env python3
"""M->S trainer using the pinned OFFICIAL TimeMixer.Model, not a reimplementation.
No modification of official source or previous training results.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from m2s.data import load,CONFIGS

ROOT=Path(__file__).resolve().parents[1]
OFFICIAL=ROOT/'training'/'workspace'/'official'

VARIANTS=('target_only','multivariate_joint','multivariate_independent')

def arguments():
    p=argparse.ArgumentParser(description='Run pinned official TimeMixer M->S on historical real variables')
    p.add_argument('--dataset',required=True,choices=sorted(CONFIGS))
    p.add_argument('--horizon',type=int,required=True)
    p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--seed',type=int,default=2024)
    p.add_argument('--d-model',type=int,default=16)
    p.add_argument('--d-ff',type=int,default=32)
    p.add_argument('--layers',type=int,default=2)
    p.add_argument('--moving-avg',type=int,default=25)
    p.add_argument('--max-train-windows',type=int,default=0)
    p.add_argument('--max-eval-windows',type=int,default=0)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--force',action='store_true')
    return p.parse_args()


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends,'cudnn'):
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False


def make_official_model(channels,seq_len,pred_len,variant,a):
    model_path=OFFICIAL/'models'/'TimeMixer.py'
    if not model_path.is_file():
        raise FileNotFoundError(
            'Pinned official TimeMixer source missing: '+str(model_path)+
            '. In original project run: python training/prepare.py. '
            'Existing checkpoints are not modified.')
    sys.path.insert(0,str(OFFICIAL))
    from models.TimeMixer import Model
    # Both joint and independent modes MUST have c_out=enc_in in this pinned version:
    # joint uses a raw residual with enc_in channels; independent reshapes B*N.
    ci=0 if variant=='multivariate_joint' else 1
    config=SimpleNamespace(
        task_name='long_term_forecast',seq_len=seq_len,label_len=0,pred_len=pred_len,
        channel_independence=ci,enc_in=channels,dec_in=channels,c_out=channels,
        e_layers=a.layers,d_model=a.d_model,d_ff=a.d_ff,dropout=0.1,
        embed='timeF',freq='h',moving_avg=a.moving_avg,top_k=5,
        decomp_method='moving_avg',down_sampling_layers=3,
        down_sampling_window=2,down_sampling_method='avg',
        use_future_temporal_feature=False,use_norm=1, num_class=1,
    )
    net=Model(config)
    return net,config


def selected_prediction(net,x,variant):
    # Official output is [B,H,C] with OT in last position.
    # Only OT is used as the future target in loss/metrics.
    y=net(x,None,None,None)
    return y[:,:,-1:]


def evaluate(net,loader,device,variant,mu,sd,save_csv=None):
    net.eval()
    sq=ab=0.0;n=0;examples=[]
    with torch.no_grad():
        for xx,yy in loader:
            xx=xx.to(device);yy=yy.to(device)
            pred=selected_prediction(net,xx,variant)
            if pred.shape!=yy.shape:raise ValueError(f'Output {pred.shape} != target {yy.shape}')
            if not torch.isfinite(pred).all():raise FloatingPointError('nonfinite prediction')
            delta=pred-yy
            sq+=float(delta.square().sum().item())
            ab+=float(delta.abs().sum().item())
            n+=yy.numel()
            if save_csv and len(examples)<4:
                examples.append((pred.detach().cpu().numpy(),yy.detach().cpu().numpy()))
    scaled_mse=sq/n;scaled_mae=ab/n
    metrics=dict(mse=scaled_mse*sd*sd,mae=scaled_mae*sd,rmse=(scaled_mse**0.5)*sd,
                 mse_scaled=scaled_mse,mae_scaled=scaled_mae,n_targets=n,n_windows=len(loader.dataset))
    if save_csv:
        with open(save_csv,'w',newline='',encoding='utf-8') as f:
            w=csv.writer(f)
            w.writerow(['eval_window_index','step','prediction','actual'])
            idx=0
            for preds,ys in examples:
                for j in range(len(preds)):
                    for h in range(preds.shape[1]):
                        w.writerow([idx,h+1,float(preds[j,h,0]*sd+mu),float(ys[j,h,0]*sd+mu)])
                    idx+=1
    return metrics


def main():
    a=arguments()
    if a.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Use CPU smoke, not silent fallback.')
    device=torch.device('cuda' if a.device=='cuda' or (a.device=='auto' and torch.cuda.is_available()) else 'cpu')
    if device.type=='cpu':torch.set_num_threads(min(4,os.cpu_count() or 1))
    seed_all(a.seed)
    data=load(ROOT,a.dataset,a.horizon)
    is_target_only=a.variant=='target_only'
    n_chan=1 if is_target_only else len(data.columns)
    net,config=make_official_model(n_chan,data.input_len,data.horizon,a.variant,a)
    net=net.to(device)
    if a.max_train_windows or a.max_eval_windows:
        mode='smoke'
    else:mode='full'
    out=ROOT/'results'/'m2s'/mode/a.dataset/f'h{a.horizon}'/a.variant/f's{a.seed}'
    out.mkdir(parents=True,exist_ok=True)
    config_json={**vars(a),'data_columns':data.columns,'num_channels':n_chan,
                 'forecast_target':'OT','target_is_last_input_column':True,
                 'source':'pinned official TimeMixer architecture, existing original project',
                 'official_source_path':str(OFFICIAL/'models'/'TimeMixer.py'),
                 'original_official_git_commit':'e24610583b36fdd8c76cc17a8df4e65759a5f460',
                 'selected_device':str(device),'data_root':str(ROOT/'data'),
                 'preprocessing':'train-period mean/std; missing auxiliary input -> training mean; no new predictors',
                 'missing_auxiliaries':data.missing_count,'scaler_mean':data.mean.tolist(),
                 'scaler_std':data.std.tolist(),'train_period_last_row':data.train_end,
                 'full_window_counts':{x:len(w) for x,w in data.windows.items()},
                 'time_features':'none; input contains only observed CSV variables'}
    # Skip only if exact params match, unlike original runner's log-based skip.
    result_path=out/'metrics.json'; config_path=out/'config.json'
    if result_path.exists() and config_path.exists() and not a.force:
        old=json.loads(config_path.read_text(encoding='utf8'))
        if old==config_json:
            print('[SKIP identical configuration]',result_path,flush=True);return
        else:
            print('[RETRAIN configuration changed]',out,flush=True)
    config_path.write_text(json.dumps(config_json,ensure_ascii=False,indent=2),encoding='utf8')
    print('[START]',a.dataset,'H',a.horizon,'variant',a.variant,'L',data.input_len,
          'C',n_chan,'device',device,'model=OFFICIAL_TimeMixer',flush=True)
    st=time.monotonic()
    loaders={name:data.loader(name,batch_size=a.batch_size,
              max_windows=(a.max_train_windows if name=='train' else a.max_eval_windows),
              seed=a.seed,target_only=is_target_only,workers=a.workers)
             for name in ('train','val','test')}
    opt=torch.optim.Adam(net.parameters(),lr=a.lr)
    best=float('inf');best_epoch=0;bad=0;history=[]
    for epoch in range(1,a.epochs+1):
        net.train();total=0.0;cnt=0
        for xx,yy in loaders['train']:
            xx=xx.to(device);yy=yy.to(device)
            opt.zero_grad(set_to_none=True)
            prediction=selected_prediction(net,xx,a.variant)
            if prediction.shape!=yy.shape:
                raise ValueError(f'Mismatch: pred {prediction.shape}, target {yy.shape}')
            loss=(prediction-yy).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
            opt.step()
            total+=float(loss.item())*len(xx);cnt+=len(xx)
        val=evaluate(net,loaders['val'],device,a.variant,float(data.mean[-1]),float(data.std[-1]))
        v=val['mse_scaled']
        history.append({'epoch':epoch,'train_loss_scaled':total/cnt,'val_mse_scaled':v})
        print(f'epoch={epoch:03d} train={total/cnt:.6f} val={v:.6f}',flush=True)
        if v<best-1e-9:
            best=v;best_epoch=epoch;bad=0
            torch.save({'epoch':epoch,'model':net.state_dict()},out/'best.pt')
        else:
            bad+=1
        if bad>=a.patience:
            print('[EARLY STOP]',epoch,flush=True);break
    with (out/'history.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    cp=torch.load(out/'best.pt',map_location=device,weights_only=True)
    net.load_state_dict(cp['model'])
    val=evaluate(net,loaders['val'],device,a.variant,float(data.mean[-1]),float(data.std[-1]))
    test=evaluate(net,loaders['test'],device,a.variant,float(data.mean[-1]),float(data.std[-1]),save_csv=out/'test_predictions_sample.csv')
    summary={'dataset':a.dataset,'horizon':a.horizon,'variant':a.variant,'mode':mode,
             'input_channels':n_chan,'best_epoch':best_epoch,'seed':a.seed,
             'device':str(device),'val':val,'test':test,
             'elapsed_seconds':round(time.monotonic()-st,2)}
    result_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf8')
    print('[TEST]',json.dumps(test,ensure_ascii=False),flush=True)
    print('[SAVED]',result_path,flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        import traceback;traceback.print_exc();sys.exit(1)
