#!/usr/bin/env python3
"""Post-hoc TVP structural trials on frozen TimeMixer component exports.

This does NOT alter, train, or backpropagate the official TimeMixer backbone.
Inputs: chronological val/test files from tvp_lab.extract_official.
A forecast at test origin i uses test labels only for origins <= i-H.
The all-scales comparator in this experiment is recalibrated under one consistent
initialization protocol. The old v2 TVP score is reported separately for reference.
"""
from __future__ import annotations
import argparse, csv, itertools, json
from pathlib import Path
import numpy as np


def read_data(path):
    with np.load(path) as z:
        b,p,y=(np.asarray(z[k], dtype=np.float64) for k in ('base','parts','y'))
    if b.ndim!=3 or y.shape!=b.shape or p.ndim!=4 or p.shape[:3]!=b.shape or not np.isfinite(p).all():
        raise ValueError('Expected finite base/y [N,H,C] and parts [N,H,C,S]')
    return b,p,y


def ridge_init(b,p,y,ridge=5.0):
    x=p.reshape(-1,p.shape[-1]); r=(y-b).reshape(-1)
    gram=x.T@x/len(x)
    return np.linalg.solve(gram+ridge*np.eye(x.shape[-1]),x.T@r/len(x))


class DiagonalQTVP:
    """Random-walk TVP on active scale coefficients; inactive scales strictly fixed.

    The covariance is an algorithmic update regularizer, NOT a valid standard error.
    Each matured forecast origin contributes one batch-averaged observation.
    """
    def __init__(self, init, active, q_diag, measurement=1.0, p0=0.1, clip=0.8):
        init=np.asarray(init,dtype=float); self.S=len(init)
        self.active=np.asarray(active,dtype=int)
        self.fixed=init.copy();self.fixed[self.active]=0.
        self.d=init[self.active].copy()
        q=np.asarray(q_diag,dtype=float)
        if len(q)!=self.S or (q<0).any() or measurement<=0 or p0<=0:
            raise ValueError('Invalid Q or hyperparameters')
        self.q=q[self.active]
        self.P=np.eye(len(self.active))*p0
        self.measurement=float(measurement);self.clip=float(clip)
        self.n_updates=0
    def predict(self,b,p):
        p=np.asarray(p,float)
        return np.asarray(b,float)+np.einsum('...s,s->...',p,self.fixed)+np.einsum('...s,s->...',p[...,self.active],self.d)
    def update(self,b,p,y):
        x=np.asarray(p,float)[...,self.active].reshape(-1,len(self.active))
        r=(np.asarray(y,float)-np.asarray(b,float)-np.einsum('...s,s->...',p,self.fixed)).reshape(-1)
        prior=self.P+np.diag(self.q)
        pinv=np.linalg.inv(prior)
        n=len(r)
        posterior=np.linalg.inv(pinv + x.T@x/(self.measurement*n))
        self.d=posterior@(pinv@self.d+x.T@r/(self.measurement*n))
        self.d=np.clip(self.d,-self.clip,self.clip)
        self.P=(posterior+posterior.T)/2
        self.n_updates+=1


def online_predict(b,p,y,init,active,q_diag,h,skip_update_before=0):
    model=DiagonalQTVP(init,active,q_diag)
    out=np.empty_like(b); coef=np.empty((len(b),len(init)))
    for i in range(len(b)):
        matured=i-h
        if matured>=skip_update_before and matured>=0:
            model.update(b[matured],p[matured],y[matured])
        out[i]=model.predict(b[i],p[i])
        coef[i]=model.fixed
        coef[i,model.active]=model.d
    return out,coef,model.n_updates


def metrics(y,p):
    return float(np.mean((y-p)**2)),float(np.mean(np.abs(y-p)))


def candidate_defs(n_scales,base_q,energy):
    if n_scales<2:raise ValueError('Need at least two FMM component scales')
    all_scales=tuple(range(n_scales))
    # Selective: all one/two-active subsets only (bounded count for small S).
    sets=[(i,) for i in all_scales]
    sets += list(itertools.combinations(all_scales,2)) if n_scales<=6 else []
    profiles={
       'scale_profile_forward':np.linspace(.5,2.0,n_scales),
       'scale_profile_reverse':np.linspace(2.0,.5,n_scales),
       'scale_profile_energy_inverse':1/np.sqrt(np.maximum(energy,1.e-10)),
    }
    # Normalize mean Q multiplier=1; avoid drastic q jumps due to magnitude differences.
    for name,p in profiles.items():
        p=np.clip(p/(np.mean(p)+1e-12),.25,4.)
        profiles[name]=p
    out=[{'id':'all_scales_shared_q','family':'shared','active':all_scales,'q':np.ones(n_scales)*base_q}]
    out += [{'id':'selective_'+ '_'.join(str(j) for j in a),'family':'selective','active':a,'q':np.ones(n_scales)*base_q} for a in sets]
    out += [{'id':name,'family':'scale_specific','active':all_scales,'q':base_q*v} for name,v in profiles.items()]
    return out


def run(val,test,outdir,manifest=None,original_preds=None,ridge=5.0,base_q=None,split_fraction=0.6):
    vb,vp,vy=read_data(val);tb,tp,ty=read_data(test)
    if vb.shape[1:]!=tb.shape[1:] or vp.shape[-1]!=tp.shape[-1]:raise ValueError('Validation/test shapes mismatch')
    n,h,c=tb.shape;S=tp.shape[-1]
    if len(vb)<2*h+10 or n<h+1:raise ValueError('Need sufficient chronological validation/test origins')
    k=max(h+10,int(len(vb)*split_fraction)); l=k-h
    if k>=len(vb) or l<10:raise ValueError('Invalid validation split')
    params=json.loads(Path(manifest).read_text('utf8')) if manifest else {}
    if base_q is None:
        if 'selected_q' not in params:raise ValueError('Provide original compare/manifest.json or --q')
        base_q=float(params['selected_q'])
    if not np.isfinite(base_q) or base_q<0:raise ValueError('q must be finite and nonnegative')
    init= ridge_init(vb[:l],vp[:l],vy[:l],ridge)
    energy=np.mean(vp[:l]**2,axis=(0,1,2))
    candidates=candidate_defs(S,base_q,energy)
    scored=[]
    for cand in candidates:
        p,_,_=online_predict(vb,vp,vy,init,cand['active'],cand['q'],h,skip_update_before=l)
        mse,mae=metrics(vy[k:],p[k:])
        scored.append({'id':cand['id'],'family':cand['family'],'active':list(cand['active']),
                       'q_diag':[float(z) for z in cand['q']],'val_tail_mse':mse,'val_tail_mae':mae})
    chosen={family:min((s for s in scored if s['family']==family), key=lambda s:s['val_tail_mse']) for family in ['shared','selective','scale_specific']}
    # Refit static coefficient on ALL validation labels, all of which precede first
    # test origin by the provider split contract; do not feed those observations
    # again to the online Kalman update (prevents double-counting warm start).
    final_init=ridge_init(vb,vp,vy,ridge)
    preds={'original':tb}
    tracks={};updates={}
    for family,s in chosen.items():
        name={'shared':'tvp_shared_q_reinit','selective':'tvp_selective','scale_specific':'tvp_scale_specific'}[family]
        p,coefs,n_up=online_predict(tb,tp,ty,final_init,s['active'],s['q_diag'],h)
        preds[name]=p;tracks[name]=coefs;updates[name]=n_up
    if original_preds:
        with np.load(original_preds) as z:
            if np.array_equal(z['y'].shape,ty.shape) and np.allclose(z['y'],ty,atol=1.e-5,rtol=0) and np.allclose(z['original'],tb,atol=1.e-5,rtol=0):
                preds['tvp_online_v2_reference']=np.asarray(z['tvp_online'],float)
            else:raise ValueError('V2 reference predictions do not match test/target exports')
    bmse,_=metrics(ty,tb)
    rows=[]
    for name,p in preds.items():
        mse,mae=metrics(ty,p)
        rows.append({'variant':name,'mse':mse,'mae':mae,'mse_improvement_pct':100*(bmse-mse)/bmse})
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    with (out/'structural_metrics.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (out/'validation_candidates.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=['id','family','active','q_diag','val_tail_mse','val_tail_mae']);w.writeheader()
        for s in scored:w.writerow({**s,'active':json.dumps(s['active']),'q_diag':json.dumps(s['q_diag'])})
    np.savez_compressed(out/'structural_predictions.npz',y=ty,**preds,**{'state_'+k:v for k,v in tracks.items()})
    manifest_out={'protocol':'frozen_backbone_posthoc_TVP','validation_file':str(val),'test_file':str(test),
      'n_val':len(vb),'n_test':n,'H':h,'channels':c,'scales':S,'base_q_from_original_validation':base_q,
      'ridge':ridge,'val_tail_first_origin':k,'calibration_prefix_last_origin_exclusive':l,
      'chosen':chosen,'update_counts':updates,'test_update_first_origin':h,
      'selection_note':'Only validation-tail MSE ranks candidates; test labels never select variants.',
      'reinit_note':'All new variants are refit on full validation and never double-count validation observations. Old v2 TVP appears separately and differs in warm-start.',
      'limitation':'This tests post-hoc time-varying multi-scale calibration only; not end-to-end trained FMM.'}
    (out/'structural_manifest.json').write_text(json.dumps(manifest_out,indent=2,ensure_ascii=False),encoding='utf8')
    for row in rows:print(f"{row['variant']:28s} MSE={row['mse']:.7f} MAE={row['mae']:.7f} gain={row['mse_improvement_pct']:+.4f}%")
    for family,s in chosen.items():print('Validation choice:',family,s['id'],'val_tail_mse=',round(s['val_tail_mse'],8))
    print('Saved:',out)
    return rows,manifest_out

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    root='results/official_ETTm1_h96_seed2024'
    ap.add_argument('--val',default=f'{root}/extracted/val.npz')
    ap.add_argument('--test',default=f'{root}/extracted/test.npz')
    ap.add_argument('--manifest',default=f'{root}/compare/manifest.json')
    ap.add_argument('--pred',default=f'{root}/compare/predictions.npz')
    ap.add_argument('--outdir',default=f'{root}/compare/structural_trials')
    ap.add_argument('--q',type=float,default=None)
    ap.add_argument('--ridge',type=float,default=5.0)
    a=ap.parse_args()
    run(a.val,a.test,a.outdir,a.manifest,a.pred,a.ridge,a.q)
