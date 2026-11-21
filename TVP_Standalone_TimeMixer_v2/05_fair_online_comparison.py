#!/usr/bin/env python3
"""Time-consistent online controls for TimeMixer TVP v2.

Inputs must be chronological validation/test components extracted from one frozen
TimeMixer checkpoint. Online competitors see the SAME delayed test labels as TVP.
This script NEVER trains/backpropagates the backbone and NEVER tunes on test loss.

Run from TVP_Standalone_TimeMixer_v2 root:
    python 05_fair_online_comparison.py
"""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np

class RidgeStream:
    def __init__(self, n_feats:int, lam:float, forget:float=1.):
        if not (0<forget<=1):raise ValueError('forget must be in (0,1]')
        self.G=np.zeros((n_feats,n_feats),float);self.b=np.zeros(n_feats,float)
        self.n=0.;self.lam=lam;self.forget=forget
    def add(self,x,r):
        xx=np.asarray(x,float).reshape(-1,len(self.b))
        rr=np.asarray(r,float).reshape(-1)
        n=len(rr)
        self.G=self.G*self.forget+(xx.T@xx)/n
        self.b=self.b*self.forget+(xx.T@rr)/n
        self.n=self.n*self.forget+1
    def coef(self):
        return np.linalg.solve(self.G/self.n+self.lam*np.eye(len(self.b)),self.b/self.n)


def features(parts, gating=False, ref=None):
    if not gating:return parts
    gate=np.log1p(np.std(parts,axis=-1).mean(axis=(1,2)))
    mu,sd=ref
    z=((gate-mu)/sd)[:,None,None,None]
    return np.concatenate([parts,parts*z],axis=-1)


def run(val,test,pred,out,lam=5.,forget=.99):
    va=np.load(val);te=np.load(test);model=np.load(pred)
    vb,vp,vy=(np.asarray(va[k],float) for k in ['base','parts','y'])
    tb,tp,ty=(np.asarray(te[k],float) for k in ['base','parts','y'])
    if any(x.shape[:3]!=tb.shape for x in [tp,ty]):raise ValueError('Invalid test shape')
    if (vb.shape[1:]!=tb.shape[1:] or vp.shape[-1]!=tp.shape[-1]):raise ValueError('Validation/test incompatible')
    n,h,c=tb.shape
    if n<=h:raise ValueError('Need test length greater than full forecast horizon')
    if not np.allclose(model['original'],tb,rtol=0,atol=1.e-5):
        raise ValueError('Original forecasts do not match extracted test data')
    if not np.allclose(model['y'],ty,rtol=0,atol=1.e-5):
        raise ValueError('Ground truth does not match extracted test data')
    valgate=np.log1p(np.std(vp,axis=-1).mean(axis=(1,2)))
    ref=(float(valgate.mean()),float(valgate.std()+1.e-8))
    specs={
      'online_expanding_ridge':(False,1.),
      'online_forgetting_ridge':(False,forget),
      'online_expanding_gate':(True,1.),
      'online_forgetting_gate':(True,forget),
    }
    controls={}
    for name,(gate,decay) in specs.items():
        vfeat=features(vp,gate,ref);tfeat=features(tp,gate,ref)
        learner=RidgeStream(vfeat.shape[-1],lam,decay)
        # At first test forecast origin, all validation labels are mature.
        for i in range(len(vb)):
            learner.add(vfeat[i],vy[i]-vb[i])
        p=np.empty_like(tb)
        for i in range(n):
            j=i-h
            if j>=0:
                # The test outcome for origin j only matures at forecast origin j+H.
                learner.add(tfeat[j],ty[j]-tb[j])
            p[i]=tb[i]+np.einsum('hcs,s->hc',tfeat[i],learner.coef())
        controls[name]=p
    preds={name:model[name].astype(float) for name in ('original','static','input_gate','tvp_online')}
    preds.update(controls)
    benchmark=np.mean((tb-ty)**2)
    rows=[]
    for name,p in preds.items():
        mse=float(np.mean((p-ty)**2));mae=float(np.mean(np.abs(p-ty)))
        rows.append({'variant':name,'mse':mse,'mae':mae,'mse_improvement_pct':100*(benchmark-mse)/benchmark})
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'fair_online_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    np.savez_compressed(out/'fair_online_predictions.npz',y=ty,**preds)
    config={'data':str(test),'n_test_origins':n,'horizon':h,'channels':c,'ridge':lam,'forgetting_factor_fixed_in_advance':forget,
            'test_label_update_lag':h,'caution':'Forget-factor .99 is a fixed diagnostic choice, not validation-optimized. Online contenders use the same matured test information.'}
    (out/'fair_online_manifest.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    print('== Fair online controls ==')
    for r in rows:print(f"{r['variant']:25s} MSE={r['mse']:.7f} MAE={r['mae']:.7f} gain={r['mse_improvement_pct']:+.4f}%")
    print('Saved:',out/'fair_online_metrics.csv')
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser()
    base='results/official_ETTm1_h96_seed2024'
    p.add_argument('--val',default=f'{base}/extracted/val.npz')
    p.add_argument('--test',default=f'{base}/extracted/test.npz')
    p.add_argument('--pred',default=f'{base}/compare/predictions.npz')
    p.add_argument('--outdir',default=f'{base}/compare/fair_online')
    p.add_argument('--lambda-ridge',type=float,default=5.)
    p.add_argument('--forget',type=float,default=.99)
    a=p.parse_args()
    run(a.val,a.test,a.pred,a.outdir,a.lambda_ridge,a.forget)
