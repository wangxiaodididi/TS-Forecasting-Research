#!/usr/bin/env python3
"""Post-training path attribution and conditional-error audit for official Paloma.

No model training, no fitting on test labels. Uses the official ETT Dataset classes,
original v3 trained checkpoints and source code, and exact Paloma forward flow.
This is a descriptive intervention audit, NOT causal attribution or performance improvement.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
UPSTREAM_SHA = 'def3dfc612bd6554be8f2ed7d5174848d6a37119'
ORIGINAL_BLOB_SHA = '03dc7c6d23751cde6acaca77c803117c89702006'
VARIABLES = ['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
PROFILES = {
    ('ETTh1',96): (24,0.5), ('ETTh1',336): (24,0.5),
    ('ETTm1',96): (96,0.4), ('ETTm1',336): (96,0.5),
}
CONDITIONS = ('historical_volatility','historical_trend','template_mismatch')
COUNTERFACTUALS = ('full','no_input_skip','no_hidden','no_phase','no_amplitude','no_phase_amplitude')


def read_args(argv=None):
    ap=argparse.ArgumentParser(description='Diagnose pre-trained Original Paloma; no retraining')
    ap.add_argument('--workspace',default=str(ROOT))
    ap.add_argument('--datasets',default='ETTh1,ETTm1')
    ap.add_argument('--horizons',default='96,336')
    ap.add_argument('--data-dir',default='/root/autodl-tmp/GTR_ResearchWorkbench_Modular/dataset')
    ap.add_argument('--source-dir',default='')
    ap.add_argument('--baseline-dir',default='')
    ap.add_argument('--out-dir',default='')
    ap.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    ap.add_argument('--batch-size',type=int,default=64)
    ap.add_argument('--max-batches',type=int,default=0,help='0=whole official test set; limited runs only for smoke')
    ap.add_argument('--gradient-batches',type=int,default=3,help='0=skip; MSE gradients computed on validation batches by default')
    ap.add_argument('--split',choices=['both','val','test'],default='both',help='both runs validation diagnostics and separate test parity check')
    ap.add_argument('--allow-baseline-mismatch',action='store_true')
    a=ap.parse_args(argv)
    a.workspace=Path(a.workspace).expanduser().resolve()
    a.source_dir=(Path(a.source_dir).expanduser().resolve() if a.source_dir
                  else a.workspace/'sources'/'Paloma')
    a.baseline_dir=(Path(a.baseline_dir).expanduser().resolve() if a.baseline_dir
                    else a.workspace/'runs'/'paloma_functional_splice')
    a.out_dir=(Path(a.out_dir).expanduser().resolve() if a.out_dir
               else a.workspace/'runs'/'paloma_path_diagnostics_v7')
    a.data_dir=Path(a.data_dir).expanduser().resolve()
    a.datasets=[t.strip() for t in a.datasets.split(',') if t.strip()]
    a.horizons=[int(t) for t in a.horizons.split(',') if t.strip()]
    if not a.datasets or any(d not in ('ETTh1','ETTm1') for d in a.datasets):ap.error('datasets must be ETTh1,ETTm1')
    if not a.horizons or any(h not in (96,336) for h in a.horizons):ap.error('horizons must be 96,336')
    if a.batch_size<1 or a.max_batches<0 or a.gradient_batches<0:ap.error('invalid batch settings')
    return a


def load_model_class(source_dir):
    f=source_dir/'models'/'Paloma.py'
    if not f.is_file():raise FileNotFoundError(f'Pinned Original Paloma model not found: {f}. Run bash setup_sources.sh')
    raw=f.read_bytes()
    blob=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
    if blob!=ORIGINAL_BLOB_SHA:
        raise RuntimeError(f'Original Paloma source has changed, expected Git blob {ORIGINAL_BLOB_SHA}, got {blob}: {f}')
    spec=importlib.util.spec_from_file_location('v7_official_paloma_readonly',f)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod.Model


def original_metadata(baseline_dir,dataset,horizon):
    meta=baseline_dir/'paper'/dataset/f'h{horizon}'/'original'/'metadata.json'
    if not meta.is_file():
        raise FileNotFoundError(f'Original checkpoint metadata missing: {meta}; run v3 Original or pass --baseline-dir')
    d=json.loads(meta.read_text(encoding='utf-8'))
    if d.get('status')!='ok' or d.get('model') not in ('Paloma',None):
        raise RuntimeError(f'Original metadata not successful Paloma run: {meta}')
    if d.get('variant') not in ('original',None):raise RuntimeError('Not an Original Paloma checkpoint: '+str(meta))
    if d.get('dataset')!=dataset or int(d.get('horizon',0))!=horizon:
        raise RuntimeError('Mismatched task in baseline metadata: '+str(meta))
    if int(d.get('lookback',96))!=96 or d.get('profile')!='paper':
        raise RuntimeError('Only L96 paper-profile original checkpoint is compatible')
    if d.get('source_commit',UPSTREAM_SHA)!=UPSTREAM_SHA:
        raise RuntimeError('Different upstream Paloma source commit')
    path=meta.parent/'checkpoints'
    candidates=sorted(path.rglob('checkpoint.pth')) if path.exists() else []
    if len(candidates)!=1:
        raise RuntimeError(f'Expected exactly 1 checkpoint.pth under {path}; found {len(candidates)}: {candidates[:6]}')
    return d,meta,candidates[0]


def config_for(dataset,horizon,meta):
    cycle,dropout=PROFILES[(dataset,horizon)]
    return SimpleNamespace(seq_len=96,pred_len=horizon,enc_in=7,cycle=cycle,d_model=256,
        dropout=dropout,use_revin=1,individual=0)


def load_checkpoint(model,path,device):
    # The .pth must come from the user's own existing trusted Paloma training.
    try: state=torch.load(path,map_location='cpu',weights_only=True)
    except TypeError: state=torch.load(path,map_location='cpu')
    if isinstance(state,dict) and 'state_dict' in state:state=state['state_dict']
    if not isinstance(state,dict):raise ValueError(f'Invalid checkpoint type: {type(state)}')
    model.load_state_dict(state,strict=True)
    return model.to(device).eval()


def make_loaders(source_dir,data_dir,dataset,horizon,batch_size):
    sys.path.insert(0,str(source_dir))
    try:
        from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute
    finally:
        sys.path.pop(0)
    dataset_class=Dataset_ETT_hour if dataset=='ETTh1' else Dataset_ETT_minute
    cycle=PROFILES[(dataset,horizon)][0]
    file=data_dir/f'{dataset}.csv'
    if not file.exists():raise FileNotFoundError(f'ETT dataset missing: {file}')
    kwargs=dict(root_path=str(data_dir),data_path=file.name,size=[96,0,horizon],features='M',
        target='OT',scale=True,timeenc=1,freq='h' if dataset=='ETTh1' else 't',cycle=cycle)
    vali=dataset_class(flag='val',**kwargs)
    test=dataset_class(flag='test',**kwargs)
    if len(test)<1 or len(vali)<1:raise ValueError('Empty official ETT split')
    return (DataLoader(vali,batch_size=batch_size,shuffle=False,num_workers=0,drop_last=False),
            DataLoader(test,batch_size=batch_size,shuffle=False,num_workers=0,drop_last=False),file)


def path_forward(model,x,cycle,check=False):
    """Reproduce the pinned Paloma forward and evaluate fixed-weight interventions."""
    if model.use_revin:
        seq_mean=x.mean(dim=1,keepdim=True)
        seq_var=x.var(dim=1,keepdim=True)+1e-5  # match upstream torch.var default
        scale=seq_var.sqrt()
        xn=(x-seq_mean)/scale
    else:
        seq_mean=torch.zeros_like(x[:,:1,:]);scale=torch.ones_like(seq_mean);xn=x
    _,_,C=x.shape
    x_input=xn.permute(0,2,1)
    gather=(cycle.reshape(-1,1)+torch.arange(model.seq_len,device=cycle.device).reshape(1,-1))%model.cycle_len
    q=model.Q[gather].permute(0,2,1)
    phase=model.phase[gather].permute(0,2,1)
    amp=model.amplitude[gather].permute(0,2,1)
    residual=x_input-q
    ph=model.phase_fusion(residual,phase)
    am=model.amplitude_fusion(residual,amp)
    skip=model.input_proj(x_input)
    hidden=model.model(ph+am)
    head=model.output_proj
    def decode(feat):return head(feat).permute(0,2,1)*scale+seq_mean
    zeros=torch.zeros_like(skip)
    out={
        'full':decode(hidden+skip),
        'no_input_skip':decode(hidden),
        'no_hidden':decode(skip),
        'no_phase':decode(model.model(am)+skip),
        'no_amplitude':decode(model.model(ph)+skip),
        'no_phase_amplitude':decode(model.model(torch.zeros_like(ph))+skip),
    }
    # Exact linear contribution to output variation, excluding head bias and restoring mean.
    W=head[-1].weight
    skip_signal=torch.nn.functional.linear(skip,W,bias=None)*scale.permute(0,2,1)
    hidden_signal=torch.nn.functional.linear(hidden,W,bias=None)*scale.permute(0,2,1)
    extras={'x_norm':xn,'q':q,'residual':residual,'phase':ph,'amplitude':am,
            'skip':skip,'hidden':hidden,'skip_signal':skip_signal,'hidden_signal':hidden_signal}
    if check:
        official=model(x,cycle)
        mx=(official-out['full']).abs().max().item()
        if mx>1e-5:raise RuntimeError(f'Manual forward mismatches official model: max abs delta={mx:.3g}')
    return out,extras


def historical_features(extras):
    xn=extras['x_norm'].transpose(1,2)
    n=xn.shape[1];q=max(1,n//4)
    return {
        'historical_volatility':(xn[:,1:,:]-xn[:,:-1,:]).abs().mean((1,2)),
        'historical_trend':(xn[:,-q:,:].mean(1)-xn[:,:q,:].mean(1)).abs().mean(1),
        'template_mismatch':extras['residual'].square().mean((1,2)).sqrt(),
    }


def threshold_from_validation(loader,model,device):
    samples=defaultdict(list)
    with torch.no_grad():
        for x,_,_,_,cyc in loader:
            x=x.float().to(device);cyc=cyc.long().to(device)
            _,extra=path_forward(model,x,cyc)
            for k,v in historical_features(extra).items():samples[k].append(v.cpu().numpy())
    return {k:tuple(float(z) for z in np.quantile(np.concatenate(v),[1/3,2/3]))
            for k,v in samples.items()}


def safe_pearson(a,b):
    if len(a)<3 or np.std(a)==0 or np.std(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])


def dataframe_csv(rows,path,columns=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows and columns is None:return
    df=pd.DataFrame(rows,columns=columns)
    df.to_csv(path,index=False,encoding='utf-8-sig')


def finish_task(dataset,horizon,model,meta,metadata_file,checkpoint,vali_loader,analysis_loader,out,device,
                batch_limit=0,grad_batches=3,allow_mismatch=False,split='val'):
    thresh=threshold_from_validation(vali_loader,model,device)
    out.mkdir(parents=True,exist_ok=True)
    total_n=0
    sum_sq=defaultdict(float);sum_abs=defaultdict(float)
    ns=0
    n_channels=model.enc_in
    channel_sq=np.zeros(n_channels);channel_abs=np.zeros(n_channels)
    segment_n=np.zeros(4,dtype=int);segment_sq=np.zeros(4);segment_abs=np.zeros(4)
    per_cf=[];group_rows=[];gradient_rows=[];window_rows=[];power=[]
    for batch_id,(xb,yb,_,_,cyc) in enumerate(analysis_loader):
        if batch_limit and batch_id>=batch_limit:break
        x=xb.float().to(device);y=yb.float().to(device);c=cyc.long().to(device)
        b=x.shape[0];h=y.shape[1]
        with torch.no_grad():
            outputs,ex=path_forward(model,x,c,check=(batch_id==0))
            features=historical_features(ex)
            sq_full=(outputs['full']-y).square()
            ab_full=(outputs['full']-y).abs()
            channel_sq+=sq_full.sum((0,1)).cpu().numpy()
            channel_abs+=ab_full.sum((0,1)).cpu().numpy()
            ns+=b*h
            for seg,idx in enumerate(np.array_split(np.arange(h),4)):
                segment_n[seg]+=b*len(idx)*n_channels
                segment_sq[seg]+=sq_full[:,idx,:].sum().item()
                segment_abs[seg]+=ab_full[:,idx,:].sum().item()
            for name,pred in outputs.items():
                er=pred-y
                sum_sq[name]+=er.square().sum().item()
                sum_abs[name]+=er.abs().sum().item()
                diff=(pred-outputs['full'])
                power.append({'batch':batch_id,'counterfactual':name,
                    'delta_rms':float(diff.square().mean().sqrt()),
                    'prediction_variation_rms':float(outputs['full'].var(unbiased=False).sqrt())})
                mse_batch=er.square().mean((1,2)).cpu().numpy()
                mae_batch=er.abs().mean((1,2)).cpu().numpy()
                per_cf.extend({'dataset':dataset,'horizon':horizon,'window_id':total_n+i,
                    'counterfactual':name,'mse':float(mse_batch[i]),'mae':float(mae_batch[i])}
                    for i in range(b))
            sq_full_per=sq_full.mean((1,2)).cpu().numpy()
            ab_full_per=ab_full.mean((1,2)).cpu().numpy()
            sq_ot_per=sq_full[:,:,-1].mean(1).cpu().numpy()
            ab_ot_per=ab_full[:,:,-1].mean(1).cpu().numpy()
            skip_rms=ex['skip_signal'].square().mean((1,2)).sqrt().cpu().numpy()
            hidden_rms=ex['hidden_signal'].square().mean((1,2)).sqrt().cpu().numpy()
            feature_array={cond:z.cpu().numpy() for cond,z in features.items()}
            for i in range(b):
                v={'dataset':dataset,'horizon':horizon,'window_id':total_n+i,
                   'full_mse':float(sq_full_per[i]),'full_mae':float(ab_full_per[i]),
                   'ot_mse':float(sq_ot_per[i]),'ot_mae':float(ab_ot_per[i]),
                   'skip_signal_rms':float(skip_rms[i]),
                   'hidden_signal_rms':float(hidden_rms[i])}
                for cond,values in feature_array.items():
                    value=float(values[i]);lo,hi=thresh[cond]
                    v[cond]=value
                    v[cond+'_group']='low' if value<=lo else 'high' if value>hi else 'medium'
                window_rows.append(v)
        total_n+=b
        if batch_id<grad_batches:
            # Re-execute with grad enabled, *no parameter update*; the loss uses the same full prediction.
            outputs,ex=path_forward(model,x,c)
            loss=(outputs['full']-y).square().mean()
            zs=[ex['skip'],ex['phase'],ex['amplitude'],ex['hidden']]
            grads=torch.autograd.grad(loss,zs,allow_unused=False)
            for name,feat,g in zip(['input_skip','phase','amplitude','hidden'],zs,grads):
                gradient_rows.append({'dataset':dataset,'horizon':horizon,'batch':batch_id,
                    'path':name,'loss_mse':float(loss.detach()),
                    'activation_rms':float(feat.detach().square().mean().sqrt()),
                    'gradient_rms':float(g.detach().square().mean().sqrt()),
                    'gradient_x_activation_abs':float((g.detach()*feat.detach()).abs().mean())})
    if not total_n:raise ValueError('No test samples evaluated')
    base_mse=sum_sq['full']/(total_n*horizon*n_channels)
    base_mae=sum_abs['full']/(total_n*horizon*n_channels)
    expected=meta.get('mse');diff=abs(base_mse-float(expected)) if isinstance(expected,(int,float)) else None
    expected_mae=meta.get('mae')
    diff_mae=abs(base_mae-float(expected_mae)) if isinstance(expected_mae,(int,float)) else None
    full_test=(batch_limit==0)
    status=('validation_diagnostics' if split=='val' else
            'matched' if full_test and diff is not None and diff_mae is not None and max(diff,diff_mae)<=0.0001 else
            'mismatch' if full_test and diff is not None and diff_mae is not None else
            'partial' if not full_test else 'not_available')
    if status=='mismatch' and not allow_mismatch:
        raise RuntimeError(f'Original checkpoint metric mismatch {dataset} H{horizon}: recalculated={base_mse:.7f} '
            f'metadata={expected:.7f} (MSE abs difference {diff:.7f}, MAE abs difference {diff_mae:.7f}). See README; use --allow-baseline-mismatch only after investigation.')
    for name in COUNTERFACTUALS:
        mse=sum_sq[name]/(total_n*horizon*n_channels)
        mae=sum_abs[name]/(total_n*horizon*n_channels)
        pp=[p for p in power if p['counterfactual']==name]
        per_cf_record={'dataset':dataset,'horizon':horizon,'intervention':name,
            'mse':mse,'mae':mae,'delta_mse_vs_full':mse-base_mse,
            'mse_change_pct_vs_full':100*(mse-base_mse)/base_mse,
            'prediction_delta_rms_mean':float(np.mean([q['delta_rms'] for q in pp]))}
        group_rows.append(per_cf_record)
    feature_df=pd.DataFrame(window_rows)
    for cond in CONDITIONS:
        for label in ('low','medium','high'):
            frame=feature_df[feature_df[cond+'_group']==label]
            item={'dataset':dataset,'horizon':horizon,'condition':cond,'group':label,
                  'n_windows':int(len(frame)),'mse':float(frame.full_mse.mean()) if len(frame) else None,
                  'mae':float(frame.full_mae.mean()) if len(frame) else None,
                  'ot_mse':float(frame.ot_mse.mean()) if len(frame) else None,
                  'threshold_low':thresh[cond][0],'threshold_high':thresh[cond][1]}
            group_rows.append({'_condition_row':item})
    bucket_rows=[{'dataset':dataset,'horizon':horizon,'bin':i+1,
                  'steps':len(np.array_split(np.arange(horizon),4)[i]),
                  'mse':segment_sq[i]/segment_n[i],'mae':segment_abs[i]/segment_n[i]}
                  for i in range(4)]
    channels=[{'dataset':dataset,'horizon':horizon,'variable':ch,
               'mse':channel_sq[i]/ns,'mae':channel_abs[i]/ns}
              for i,ch in enumerate(VARIABLES)]
    interventions=[z for z in group_rows if '_condition_row' not in z]
    groups=[z['_condition_row'] for z in group_rows if '_condition_row' in z]
    dataframe_csv(interventions,out/'path_interventions.csv')
    dataframe_csv(groups,out/'conditional_error.csv')
    dataframe_csv(bucket_rows,out/'forecast_horizon_bins.csv')
    dataframe_csv(channels,out/'channel_error.csv')
    dataframe_csv(gradient_rows,out/'gradient_sensitivity.csv',columns=['dataset','horizon','batch','path','loss_mse','activation_rms','gradient_rms','gradient_x_activation_abs'])
    dataframe_csv(window_rows,out/'window_level.csv')
    # Conditional value of each path, not just conditional error for the full model.
    cf_df=pd.DataFrame(per_cf)
    conditional_interventions=[]
    for cond in CONDITIONS:
        tmp=cf_df.merge(feature_df[['window_id',cond+'_group']],on='window_id',how='inner')
        for (label,intervention),sub in tmp.groupby([cond+'_group','counterfactual']):
            base=tmp[(tmp[cond+'_group']==label)&(tmp.counterfactual=='full')]
            bm=float(base.mse.mean())
            conditional_interventions.append({'dataset':dataset,'horizon':horizon,'condition':cond,'group':label,
                'intervention':intervention,'n_windows':len(sub),'mse':float(sub.mse.mean()),
                'mse_change_pct_vs_full':100*(float(sub.mse.mean())-bm)/bm if bm>0 else None})
    dataframe_csv(conditional_interventions,out/'conditional_path_interventions.csv')
    correlations=[]
    for c in CONDITIONS:
        for er in ('full_mse','full_mae','ot_mse'):
            rank=feature_df[[c,er]].rank()
            correlations.append({'dataset':dataset,'horizon':horizon,'condition':c,'error_metric':er,
                 'spearman_rank_correlation':safe_pearson(rank[c].to_numpy(),rank[er].to_numpy())})
    dataframe_csv(correlations,out/'condition_rank_correlations.csv')
    summary={'dataset':dataset,'horizon':horizon,'split':split,'n_windows':total_n,'original_recalculated_mse':base_mse,
          'original_recalculated_mae':base_mae,'original_metadata_mse':expected,
          'original_metadata_mae':meta.get('mae'),'baseline_mse_abs_diff':diff,'baseline_mae_abs_diff':diff_mae,
          'match_status':status,'baseline_metadata':str(metadata_file),'original_checkpoint':str(checkpoint),
          'device':str(device),'max_batches':batch_limit,'val_thresholds':thresh,'source_commit':UPSTREAM_SHA}
    (out/'manifest.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'[DONE] {dataset} H{horizon} split={split} n={total_n} MSE={base_mse:.6f} match={status}',flush=True)
    return summary


def write_overview(summaries,out_dir):
    """Human-readable validation diagnosis; test metrics used only for consistency checking."""
    lines=['# Paloma v7 路径与条件误差诊断（自动生成）','',
        '**说明：只分析已训练好的 Original Paloma。此文件不是新模型性能结果。**','',
        '主要判断依据是验证集。测试集结果仅用于核对官方重现精度，不作为继续挑模块的依据。',
        '路径关闭是固定权重干预，不是重新训练消融，也不能解释为因果或可加性贡献。','']
    for s in summaries:
        if s['split']=='test':
            lines.append(f"- 测试集核对 {s['dataset']} H{s['horizon']}: MSE={s['original_recalculated_mse']:.6f}; "
                         f"先前元数据 MSE={s.get('original_metadata_mse')}; 校验状态={s['match_status']}")
            continue
        ds=s['dataset'];h=s['horizon'];p=out_dir/ds/f'h{h}'/'val'
        lines.extend(['',f'## {ds} · H{h}（验证集）','',
          f"窗口数 {s['n_windows']}；MSE={s['original_recalculated_mse']:.6f}，MAE={s['original_recalculated_mae']:.6f}。",''])
        inter=pd.read_csv(p/'path_interventions.csv')
        lines.append('固定权重关闭路径的 MSE 相对变化（正数表示关闭后更差）：')
        for row in inter.to_dict('records'):
            if row['intervention']=='full':continue
            lines.append(f"- {row['intervention']}: {row['mse_change_pct_vs_full']:+.2f}%")
        groups=pd.read_csv(p/'conditional_error.csv')
        lines.extend(['','不同历史条件下，验证集高/低组的平均 MSE 比值：'])
        for cond in CONDITIONS:
            g=groups[groups.condition==cond]
            lo=g[g.group=='low'].mse.mean();hi=g[g.group=='high'].mse.mean()
            ratio=(hi/lo) if lo>0 else float('nan')
            lines.append(f'- {cond}: 高组/低组={ratio:.3f} (不是显著性检验)')
        bins=pd.read_csv(p/'forecast_horizon_bins.csv')
        if len(bins)>=4:
            x=bins.iloc[0].mse;y=bins.iloc[-1].mse
            lines.append(f'- 远期第4段/近期第1段 MSE 比值：{y/x:.3f}' if x>0 else '- 分段比值无法计算')
        window=pd.read_csv(p/'window_level.csv')
        lines.append(f"- 输入直连与隐含分支最终输出信号 RMS（窗口平均值）：{window.skip_signal_rms.mean():.4f} / {window.hidden_signal_rms.mean():.4f}。")
        lines.append('  注意：二者可能相互抵消，RMS 不等于贡献比例。')
    lines.extend(['','## 解读边界','',
        '1. 如果关闭 Input Skip 后误差增加，表明在当前已训练权重下该路径敏感，但不能说明应当删除或保留某种新模块。',
        '2. 如果高模板失配组误差更高，只能说明相关性；它与波动等其他历史特征可能同时变化。',
        '3. 如果后段预测误差更大，需要先判断误差是否具有随预测距离稳定增加的趋势，再选择针对性结构。',
        '4. 根据验证集诊断选择下一轮模块。测试集已经在先前多轮使用，因此任何后续测试成绩均需注明存在反复选择偏差。',''])
    (out_dir/'v7_diagnostic_overview.md').write_text('\n'.join(lines),encoding='utf-8')


def main(argv=None):
    a=read_args(argv)
    device=torch.device('cuda' if a.device=='cuda' or a.device=='auto' and torch.cuda.is_available() else 'cpu')
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA device not available')
    print('[MODE] Read existing Original Paloma checkpoints, DO NOT train or modify weights',flush=True)
    print('[DEVICE]',device,flush=True)
    cls=load_model_class(a.source_dir)
    all_summaries=[]
    for ds in a.datasets:
        for h in a.horizons:
            print(f'[TASK] {ds} H{h} -- locating matching original checkpoint',flush=True)
            meta,metadata_file,checkpoint=original_metadata(a.baseline_dir,ds,h)
            mdl=cls(config_for(ds,h,meta))
            load_checkpoint(mdl,checkpoint,device)
            val_loader,test_loader,_=make_loaders(a.source_dir,a.data_dir,ds,h,a.batch_size)
            for split in (['val','test'] if a.split=='both' else [a.split]):
                loader=val_loader if split=='val' else test_loader
                # Gradients on validation only to avoid tuning to held-out test labels.
                ngrad=a.gradient_batches if split=='val' else 0
                result=finish_task(ds,h,mdl,meta,metadata_file,checkpoint,val_loader,loader,
                    a.out_dir/ds/f'h{h}'/split,device,a.max_batches,ngrad,a.allow_baseline_mismatch,split=split)
                all_summaries.append(result)
                dataframe_csv(all_summaries,a.out_dir/'all_tasks_summary.csv')
    write_overview(all_summaries,a.out_dir)
    (a.out_dir/'run_manifest.json').write_text(json.dumps({
        'generated_utc':datetime.now(timezone.utc).isoformat(),'tasks':len(all_summaries),
        'original_models_only':True,'weights_modified':False,'diagnostic_split':a.split,
        'notes':'Counterfactual path ablations are fixed-weight descriptive interventions, not retrained ablations or causal attributions.'},indent=2),encoding='utf-8')
    print(f'[SAVED] {a.out_dir}/all_tasks_summary.csv',flush=True)

if __name__=='__main__':main()
