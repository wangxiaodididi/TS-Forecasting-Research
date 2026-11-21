"""Export additive predictions and phase-fit interpretation on validation only.

Uses the immutable source snapshot and its best checkpoint.  The trained Q is
not itself a price/temperature/electricity profile; fit-R2 is a conditional
projection statistic after affine calibration, not causal attribution.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from lab.paths import RUNS


def export(run_id, limit=64, max_channels=21):
    folder=RUNS/run_id
    cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    if cfg['model'] not in ('GTRPhaseInnovation','GTRPhaseAdaptive'):
        raise ValueError('Explanation only implemented for the Phase2 innovation models')
    snap=folder/'source'; check=list(snap.glob('checkpoints/*/checkpoint.pth'))
    if len(check)!=1:raise FileNotFoundError('Expected exactly one checkpoint in the run source')
    sys.path.insert(0,str(snap))
    from data_provider.data_factory import data_provider
    from models import GTRPhaseInnovation,GTRPhaseAdaptive
    registry={'GTRPhaseInnovation':GTRPhaseInnovation,'GTRPhaseAdaptive':GTRPhaseAdaptive}
    # Values are upstream run.py defaults unless the experiment config supplies them.
    defaults=dict(label_len=0,embed='timeF',freq='h',target='OT',num_workers=0,
                  d_model=512,use_revin=1,individual=0,dropout=0.0,batch_size=8)
    args=SimpleNamespace(**{**defaults,**cfg})
    model=registry[cfg['model']].Model(args).float()
    try: state=torch.load(check[0],map_location='cpu',weights_only=True)
    except TypeError: state=torch.load(check[0],map_location='cpu')
    model.load_state_dict(state);model.eval()
    data,_=data_provider(args,'val')
    count=min(int(limit),len(data));channels=min(int(max_channels),int(cfg['enc_in']))
    if count<1 or channels<1:raise ValueError('No validation samples/channels selected')
    windows=[];details=[]
    with torch.no_grad():
        for i in range(count):
            sample=data[i]
            x=torch.as_tensor(sample[0],dtype=torch.float32).unsqueeze(0)
            y=torch.as_tensor(sample[1],dtype=torch.float32)[-cfg['pred_len']:]
            cycle=torch.as_tensor(sample[4],dtype=torch.long).reshape(1)
            o=model.explain(x,cycle)
            for c in range(channels):
                target=y[:,c];base=o['gtr_prediction'][0,:,c]
                actual=o['prediction'][0,:,c];corr=o['innovation_correction'][0,:,c]
                row=dict(sample=i,channel=c,
                         phase_fit_r2=float(o['phase_fit_r2'][0,c]),
                         phase_mismatch=float(o['phase_mismatch'][0,c]),
                         correction_gain=float(o['correction_gain'][0,c]),
                         correction_rms=float(corr.square().mean().sqrt()),
                         base_mse=float((base-target).square().mean()),
                         proposed_mse=float((actual-target).square().mean()))
                row['delta_mse']=row['proposed_mse']-row['base_mse']
                windows.append(row)
                if i<min(4,count):
                    for h in range(cfg['pred_len']):
                        details.append(dict(sample=i,channel=c,horizon_step=h+1,
                                            target=float(target[h]),gtr_prediction=float(base[h]),
                                            innovation_correction=float(corr[h]),
                                            prediction=float(actual[h])))
    out1=folder/'phase2_validation_windows.csv'
    out2=folder/'phase2_validation_predictions.csv'
    for path,records in [(out1,windows),(out2,details)]:
        if records:
            with path.open('w',encoding='utf-8',newline='') as f:
                wr=csv.DictWriter(f,fieldnames=list(records[0]));wr.writeheader();wr.writerows(records)
    print('Windows:',out1,'Predictions:',out2)
    print('Phase fit is descriptive and cannot establish a causal mechanism.')
    return out1,out2

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('run_id');p.add_argument('--limit',type=int,default=64)
    p.add_argument('--max-channels',type=int,default=21)
    a=p.parse_args();export(a.run_id,a.limit,a.max_channels)
