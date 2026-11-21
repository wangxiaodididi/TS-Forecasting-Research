#!/usr/bin/env python3
"""Extract val/test forecasts for an independently trained head variant.

This does not apply any TVP adaptation; test set only used for evaluation.
"""
import argparse,hashlib,json,os,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

def read_config(path):
    from tvp_lab.extract_official import make_config
    return make_config(path)

def run(root,trial,device='cuda'):
    root=Path(root).resolve();folder=root/'results/head_trials'/trial
    meta=json.loads((folder/'train_config.json').read_text('utf8'))
    if meta.get('exit_code')!=0:raise RuntimeError('Training has not finished successfully')
    src=root/'training/workspace/official'
    sys.path.insert(0,str(root));sys.path.insert(0,str(src))
    import torch
    from tvp_lab.extract_official import export_one
    from models import TimeMixerResidualHead
    cfg=read_config(folder/'train_config.json')
    dset=meta['dataset'];f={'ETTm1':'ETTm1.csv','ETTh1':'ETTh1.csv','Weather':'weather.csv'}[dset]
    defaults={'label_len':0,'e_layers':2,'d_model':16,'d_ff':32,'moving_avg':25,'down_sampling_layers':3,
         'down_sampling_window':2,'down_sampling_method':'avg','decomp_method':'moving_avg','top_k':5,
         'dropout':.1,'embed':'timeF','freq':'h','channel_independence':1,'use_norm':1,
         'use_future_temporal_feature':0,'factor':1,'data':dset,'num_workers':0,
         'features':'M','target':'OT','task_name':'long_term_forecast','batch_size':16}
    cfg={**defaults,**cfg,'root_path':str((root/'data').resolve())+os.sep,'data_path':f,'num_workers':0,'batch_size':16}
    env={'TIMEMIXER_HEAD_MODE':meta['mode'],'TIMEMIXER_HEAD_RANK':str(meta['rank']),'TIMEMIXER_HEAD_SCALE':str(meta['scale'])}
    os.environ.update(env)
    model=TimeMixerResidualHead.Model(SimpleNamespace(**cfg))
    state=torch.load(meta['checkpoint'],map_location='cpu',weights_only=False)
    if isinstance(state,dict) and 'state_dict' in state:state=state['state_dict']
    model.load_state_dict(state,strict=True)
    model=model.to(device).eval()
    out=folder/'extracted';out.mkdir(parents=True,exist_ok=True)
    for flag in ('val','test'):
        export_one(model,'timemixer',str(src),cfg,flag,str(out/f'{flag}.npz'),
                  max_windows=0,batch_size=16,device=device)
    with np.load(out/'test.npz') as z:
        y=z['y'];pred=z['base'];mse=float(np.mean((y-pred)**2));mae=float(np.mean(abs(y-pred)))
    baseline_file=root/'results'/f'official_{dset}_h{meta["horizon"]}_seed{meta["seed"]}'/'extracted/test.npz'
    if not baseline_file.is_file():raise FileNotFoundError('Matched official baseline extraction required: '+str(baseline_file))
    with np.load(baseline_file) as z:
        by=z['y'];bp=z['base']
        if by.shape!=y.shape or not np.allclose(by,y,atol=1.e-5,rtol=0):
            raise RuntimeError('Head vs baseline test targets differ; fix extraction protocol before comparison')
        bmse=float(np.mean((by-bp)**2));bmae=float(np.mean(abs(by-bp)))
    result={'trial':trial,'dataset':dset,'H':meta['horizon'],'baseline_mse':bmse,'baseline_mae':bmae,
        'head_mse':mse,'head_mae':mae,'mse_gain_pct':100*(bmse-mse)/bmse,'mae_gain_pct':100*(bmae-mae)/bmae,
        'protocol':'independent_end_to_end_head_training; no posthoc TVP; matched test samples'}
    (folder/'head_result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--project-root',default='.');p.add_argument('--trial',required=True)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    a=p.parse_args();run(a.project_root,a.trial,a.device)
if __name__=='__main__':main()
