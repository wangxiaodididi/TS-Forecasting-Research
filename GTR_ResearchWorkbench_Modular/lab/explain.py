"""Explain held-out validation windows from a trained extension checkpoint.

Do NOT use validation explanations to tune based on the held-out test set.
Outputs are descriptive additive component contributions, not causal attributions.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from .paths import RUNS


def export_explanation(run_id, split='val', limit=8, channel=0):
    if split not in ('train','val','test'):raise ValueError('Unexpected split')
    p=RUNS/run_id
    cfg=json.loads((p/'config.json').read_text(encoding='utf-8'))
    if cfg['model']=='GTR':raise ValueError('Original GTR has no correction branch')
    if cfg['mode']=='smoke':print('WARNING: smoke model was trained on truncated splits')
    src=p/'source'
    ckpts=list(src.glob('checkpoints/*/checkpoint.pth'))
    if len(ckpts)!=1:raise FileNotFoundError(f'Expected exactly one checkpoint, found {len(ckpts)}')
    # Insert per-run code snapshot. Never import the mutable working extension.
    sys.path.insert(0,str(src))
    from data_provider.data_factory import data_provider
    from models import GTRHaarHigh, GTRHaarLow, GTRHaarDual, GTRLinearResidual
    models={'GTRHaarHigh':GTRHaarHigh,'GTRHaarLow':GTRHaarLow,'GTRHaarDual':GTRHaarDual,'GTRLinearResidual':GTRLinearResidual}
    defaults={'label_len':0,'embed':'timeF','freq':'h','target':'OT','num_workers':0,'d_model':512,
              'use_revin':1,'individual':0,'dropout':0.0,'batch_size':8}
    defaults.update(cfg)
    args=SimpleNamespace(**defaults)
    network=models[cfg['model']].Model(args).float()
    try:weights=torch.load(ckpts[0],map_location='cpu',weights_only=True)
    except TypeError:weights=torch.load(ckpts[0],map_location='cpu')
    network.load_state_dict(weights);network.eval()
    data,_=data_provider(args,split)
    limit=min(int(limit),len(data))
    if channel<0 or channel>=cfg['enc_in']:raise ValueError('Channel out of bounds')
    rows=[]
    with torch.no_grad():
        for i in range(limit):
            sample=data[i]
            x=torch.as_tensor(sample[0],dtype=torch.float32).unsqueeze(0)
            y=torch.as_tensor(sample[1],dtype=torch.float32)
            cycle=torch.as_tensor(sample[4],dtype=torch.int64).reshape(1)
            comps=network.explain(x,cycle)
            for t in range(cfg['pred_len']):
                row={'sample_index':i,'forecast_step':t+1,'channel':channel,'target':float(y[-cfg['pred_len']+t,channel]),
                     'prediction':float(comps['prediction'][0,t,channel]),
                     'gtr_prediction':float(comps['gtr_prediction'][0,t,channel]),
                     'low_correction':float(comps['low_correction'][0,t,channel]),
                     'high_correction':float(comps['high_correction'][0,t,channel]),
                     'linear_correction':float(comps['linear_correction'][0,t,channel])}
                energy=comps['high_energy_ratio']
                row['high_energy_ratio']=float(energy[0,channel]) if energy is not None else ''
                rows.append(row)
    out=p/f'explanation_{split}_channel{channel}.csv'
    with out.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_id');parser.add_argument('--split',default='val',choices=['train','val','test'])
    parser.add_argument('--limit',type=int,default=8);parser.add_argument('--channel',type=int,default=0)
    a=parser.parse_args();print(export_explanation(a.run_id,a.split,a.limit,a.channel))
if __name__=='__main__':main()
