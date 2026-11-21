#!/usr/bin/env python3
"""Inspect real TimeMixer TVP test predictions, without training or refitting.

Run from standalone project root:
    python 04_tvp_diagnostics.py
Optional:
    python 04_tvp_diagnostics.py --pred results/official_ETTm1_h96_seed2024/compare/predictions.npz

All tests operate on chronological forecast origins. HAC uses at least H-1 lags
because origins overlap, and is an approximate predictive accuracy comparison,
not a complete test of the statistical validity of post-hoc adaptation.
"""
import argparse, csv, json, math
from pathlib import Path
import numpy as np

VARIANTS=['original','static','input_gate','tvp_online']

def norm_p(z):
    return math.erfc(abs(float(z))/math.sqrt(2))

def hac_test(d, lags):
    d=np.asarray(d,dtype=np.float64).reshape(-1)
    n=len(d)
    if n<max(30,2*lags):
        return {'n':n, 'mean_loss_gain':float(d.mean()), 'z':None,'p':None,'se':None}
    x=d-d.mean()
    g0=float(x@x/n)
    s=g0
    L=min(lags,n-1)
    for k in range(1,L+1):
        cov=float(x[k:]@x[:-k]/n)
        s+=2*(1-k/(L+1))*cov
    se=math.sqrt(max(s,0)/n)
    z=float(d.mean()/se) if se>0 else float('nan')
    return {'n':n,'mean_loss_gain':float(d.mean()),'se':se,'z':z,'p':norm_p(z) if math.isfinite(z) else None}

def make_json(obj):
    if isinstance(obj,dict):return {k:make_json(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [make_json(x) for x in obj]
    if isinstance(obj,np.generic):return obj.item()
    if isinstance(obj,float) and not math.isfinite(obj):return None
    return obj

def run(path, outdir=None):
    path=Path(path)
    z=np.load(path)
    for k in ['y',*VARIANTS]:
        if k not in z.files:raise ValueError(f'Missing {k} in {path}')
    y=z['y'].astype(np.float64)
    pred={k:z[k].astype(np.float64) for k in VARIANTS}
    n,h,c=y.shape
    for k,v in pred.items():
        if v.shape!=y.shape: raise ValueError(f'{k} has incorrect shape {v.shape}')
    m={k:{'mse':float(np.mean((v-y)**2)), 'mae':float(np.mean(np.abs(v-y)))} for k,v in pred.items()}
    loss={k:((v-y)**2).mean(axis=(1,2)) for k,v in pred.items()}
    mae={k:np.abs(v-y).mean(axis=(1,2)) for k,v in pred.items()}
    summary={'path':str(path),'shape':{'n_origins':n,'horizon':h,'channels':c},'overall':m,'comparisons':{},'phases':{},'by_channel':[], 'horizon_blocks':[]}
    for name in ['original','static','input_gate']:
        d=loss[name]-loss['tvp_online']
        da=mae[name]-mae['tvp_online']
        summary['comparisons']['tvp_vs_'+name]={
            'mse_gain_pct':100*float(d.mean())/float(loss[name].mean()),
            'mae_gain_pct':100*float(da.mean())/float(mae[name].mean()),
            'horizon_lag_hac':hac_test(d,h-1),
            'twice_horizon_lag_hac':hac_test(d,2*h-1),
            'better_origin_fraction':float((d>0).mean()),
        }
    for phase,sl in [('no_test_label_matured',slice(0,min(h,n))),('matured_test_labels_available',slice(min(h,n),None))]:
        if len(loss['original'][sl])==0:continue
        summary['phases'][phase]={
            'n':len(loss['original'][sl]),
            'tvp_mse_gain_pct_vs_original':100*float((loss['original'][sl]-loss['tvp_online'][sl]).mean())/float(loss['original'][sl].mean()),
            'tvp_mae_gain_pct_vs_original':100*float((mae['original'][sl]-mae['tvp_online'][sl]).mean())/float(mae['original'][sl].mean())
        }
    for j in range(c):
        base=float(np.mean((pred['original'][:,:,j]-y[:,:,j])**2))
        tv=float(np.mean((pred['tvp_online'][:,:,j]-y[:,:,j])**2))
        summary['by_channel'].append({'channel':j,'original_mse':base,'tvp_mse':tv,'mse_gain_pct':100*(base-tv)/base})
    for a,b in [(0,min(24,h)),(24,min(48,h)),(48,min(72,h)),(72,min(96,h))]:
        if b<=a:continue
        bas=float(np.mean((pred['original'][:,a:b,:]-y[:,a:b,:])**2))
        tv=float(np.mean((pred['tvp_online'][:,a:b,:]-y[:,a:b,:])**2))
        summary['horizon_blocks'].append({'horizon_steps':f'{a+1}-{b}','original_mse':bas,'tvp_mse':tv,'mse_gain_pct':100*(bas-tv)/bas})
    outdir=Path(outdir) if outdir else path.parent/'tvp_diagnostics'
    outdir.mkdir(exist_ok=True,parents=True)
    (outdir/'diagnostics.json').write_text(json.dumps(make_json(summary),indent=2,ensure_ascii=False),encoding='utf-8')
    with (outdir/'by_channel.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=summary['by_channel'][0].keys());w.writeheader();w.writerows(summary['by_channel'])
    with (outdir/'by_horizon.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=summary['horizon_blocks'][0].keys());w.writeheader();w.writerows(summary['horizon_blocks'])
    print('== Report ==')
    print('Windows:',n,'horizon:',h,'channels:',c)
    for k in VARIANTS: print(f'{k:12} MSE={m[k]["mse"]:.7f} MAE={m[k]["mae"]:.7f}')
    for k,x in summary['comparisons'].items():
        print(k, f'MSE improvement {x["mse_gain_pct"]:.4f}%  MAE improvement {x["mae_gain_pct"]:.4f}%  HAC({h-1}) p={x["horizon_lag_hac"]["p"]}  HAC({2*h-1}) p={x["twice_horizon_lag_hac"]["p"]}')
    print('phase splits:',summary['phases'])
    print('channel gain %:',[round(x['mse_gain_pct'],3) for x in summary['by_channel']])
    print('horizon gain %:',[round(x['mse_gain_pct'],3) for x in summary['horizon_blocks']])
    print('Saved:',outdir/'diagnostics.json')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--pred',default='results/official_ETTm1_h96_seed2024/compare/predictions.npz')
    p.add_argument('--outdir',default='')
    a=p.parse_args()
    run(a.pred,a.outdir or None)
