"""Compare frozen official backbone vs static ridge, static conditioned gate, and online TVP.

No test labels are used for model selection; TVP q chosen on validation tail.
"""
import argparse,json,csv
from pathlib import Path
import numpy as np
from .online import fit_static,fit_gate,predict_gate,replay,warm_start,test_online,check_shapes


def load_npz(path):
    z=np.load(path)
    base=z['base'].astype('float64');parts=z['parts'].astype('float64');y=z['y'].astype('float64')
    check_shapes(base,parts,y)
    return base,parts,y


def metric(y,pred):
    return {'mse':float(np.mean((pred-y)**2)),'mae':float(np.mean(np.abs(pred-y)))}


def run(val_file,test_file,outdir,ridge=5.,q_values=(0.,1.e-5,1.e-4,1.e-3),measurement=1.,select_fraction=.6):
    vb,vp,vy=load_npz(val_file);tb,tp,ty=load_npz(test_file)
    if vb.shape[1:]!=tb.shape[1:] or vp.shape[3]!=tp.shape[3]:raise ValueError('Mismatched shapes')
    h=tb.shape[1]
    if len(vb)<max(h*2+10,40):raise ValueError('Validation requires >= 2*H + 10 windows for time-respecting TVP tuning')
    k=max(h+10,int(len(vb)*select_fraction))
    if k>=len(vb):raise ValueError('Validation portion too small; use longer input exports')
    # At validation origin k, only origins ending by k are fully observed.
    # Fit the initial calibration without any label extending into the tuning tail.
    initial=fit_static(vb[:k-h],vp[:k-h],vy[:k-h],ridge)
    ranking=[]
    for q in q_values:
        pred,states,updates=replay(vb,vp,vy,initial,h,q,measurement)
        ranking.append({'q':float(q),**metric(vy[k:],pred[k:])})
    q_best=min(ranking,key=lambda x:x['mse'])['q']
    fixed=fit_static(vb,vp,vy,ridge)
    gate=fit_gate(vb,vp,vy,ridge)
    state=warm_start(vb,vp,vy,initial,q_best,measurement)
    forecast_tv,states,updates=test_online(tb,tp,ty,state,h)
    preds={
        'original':tb,
        'static':tb+np.einsum('nhcs,s->nhc',tp,fixed),
        'input_gate':predict_gate(tb,tp,gate),
        'tvp_online':forecast_tv,
    }
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for key,pred in preds.items():
        m=metric(ty,pred)
        rows.append({'variant':key,**m,'relative_mse_improvement_pct':100*(metric(ty,tb)['mse']-m['mse'])/metric(ty,tb)['mse']})
    with open(out/'metrics.csv','w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    np.savez_compressed(out/'predictions.npz',y=ty,**preds,tvp_states=states)
    manifest={'validation_file':str(val_file),'test_file':str(test_file),'validation_n':len(vb),'test_n':len(tb),
              'forecast_horizon':h,'q_validation_trials':ranking,'selected_q':q_best,'static_delta':fixed.tolist(),
              'online_test_updates':len(updates),'test_label_first_eligible_origin':h,
              'caveat':'Results reflect only the supplied frozen forecaster; smoke proxy is NOT official TimeMixer/XLinear.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    return rows,manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--val',required=True);p.add_argument('--test',required=True)
    p.add_argument('--outdir',default='results/official');p.add_argument('--ridge',type=float,default=5.)
    a=p.parse_args()
    rows,m=run(a.val,a.test,a.outdir,a.ridge)
    print('selected_q',m['selected_q'],' test-online-updates:',m['online_test_updates'])
    for r in rows: print(r)
if __name__=='__main__':main()
