#!/usr/bin/env python3
"""Summarize all *existing* frozen-backbone paired TVP trials. Never fit on test data."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np

def metrics(y,pp):
    return float(np.mean((y-pp)**2)),float(np.mean(np.abs(y-pp)))

def hac_loss_difference(d,lag):
    d=np.asarray(d,float);n=len(d)
    # Exploratory only: few independent H-blocks for ETTm1 H=96.
    if n<max(40,2*lag):return None,None
    x=d-d.mean();v=float(x@x/n)
    L=min(lag,n-1)
    for k in range(1,L+1):v+=2*(1-k/(L+1))*float(x[k:]@x[:-k]/n)
    se=math.sqrt(max(v,0)/n)
    if se<=0:return None,None
    z=float(d.mean()/se)
    return z,math.erfc(abs(z)/math.sqrt(2))

def collect(root):
    root=Path(root)
    rows=[];details=[]
    for mpath in sorted(root.glob('official_*_h*_seed*/compare/structural_trials/structural_manifest.json')):
        folder=mpath.parent
        if not (folder/'structural_predictions.npz').is_file():continue
        meta=json.loads(mpath.read_text('utf8'))
        with np.load(folder/'structural_predictions.npz') as z:
            keys=set(z.files);y=z['y'];b=z['original']
            if not all(k in keys for k in ('tvp_selective','tvp_shared_q_reinit','tvp_scale_specific')):continue
            n,h,c=y.shape
            base_mse,base_mae=metrics(y,b)
            tname=folder.parts[-3]
            result={'task':tname,'n_test_origins':n,'horizon':h,'channels':c,'chosen_scales':meta['chosen']['selective']['active'],
                'validation_selected_q':meta['chosen']['selective']['q_diag'],
                'note':'All coefficients chosen on validation; no test-based re-ranking'}
            for name in ['tvp_shared_q_reinit','tvp_selective','tvp_scale_specific']:
                pred=z[name];mse,mae=metrics(y,pred)
                dl=((y-b)**2-(y-pred)**2).mean(axis=(1,2))
                zstat,pval=hac_loss_difference(dl,h-1)
                row={'task':tname,'variant':name,'n_origins':n,'H':h,'channels':c,
                    'mse':mse,'mae':mae,'gain_mse_pct':100*(base_mse-mse)/base_mse,
                    'gain_mae_pct':100*(base_mae-mae)/base_mae,
                    'better_origin_frac':float(np.mean(dl>0)),
                    'hac_lag_H_minus_1_z':zstat,'hac_lag_H_minus_1_p_exploratory':pval,
                    'chosen_active':json.dumps(meta['chosen']['selective']['active']) if name=='tvp_selective' else ''}
                rows.append(row)
                if name=='tvp_selective':
                    result.update({'mse_gain_pct':row['gain_mse_pct'], 'mae_gain_pct':row['gain_mae_pct'],
                        'better_origin_frac':row['better_origin_frac']})
                    result['channel_mse_gain_pct']=[float(100*(metrics(y[:,:,j],b[:,:,j])[0]-metrics(y[:,:,j],pred[:,:,j])[0])/metrics(y[:,:,j],b[:,:,j])[0]) for j in range(c)]
                    result['horizon_block_gain_pct']={}
                    for start in range(0,h,24):
                        stop=min(start+24,h)
                        mb=metrics(y[:,start:stop],b[:,start:stop])[0]
                        mt=metrics(y[:,start:stop],pred[:,start:stop])[0]
                        result['horizon_block_gain_pct'][f'{start+1}-{stop}']=100*(mb-mt)/mb
                    result['state_total_variation'] = float(np.abs(np.diff(z['state_tvp_selective'],axis=0)).sum())
            details.append(result)
    return rows,details

def write(rows,details,outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    if rows:
        with (out/'matrix_metrics.csv').open('w',encoding='utf8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'matrix_details.json').write_text(json.dumps(details,indent=2,ensure_ascii=False),encoding='utf8')
    with (out/'decision_notes.md').open('w',encoding='utf8') as f:
        f.write('# TVP multi-task validation (no final model selection using the test set)\n\n')
        for d in details:
            f.write(f'## {d["task"]}\n\nSelective active scales: {d["chosen_scales"]}; MSE gain {d["mse_gain_pct"]:+.4f}%; MAE gain {d["mae_gain_pct"]:+.4f}%; better origin fraction {d["better_origin_frac"]:.3f}.\n\n')
        f.write('Notes: horizon-overlapping origins reduce effective test size. HAC p-values are exploratory with short samples; do not claim statistical significance from one ETTm1 seed. The old V2 online run uses different warm start and should not be treated as a strict ablation. Other training seeds require separately trained checkpoints.\n')
    print('Summary:',out/'matrix_metrics.csv');print('Found tasks:',len(details))
    for d in details:print(d['task'],f"gain={d['mse_gain_pct']:+.3f}%",'active=',d['chosen_scales'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--results-root',default='results');p.add_argument('--outdir',default='results/tvp_validation_matrix')
    a=p.parse_args();r,d=collect(a.results_root)
    if not r:
        print('[NO DATA] No structural trials. Run 08_validation_matrix.py first.')
        raise SystemExit(2)
    write(r,d,a.outdir)
if __name__=='__main__':main()
