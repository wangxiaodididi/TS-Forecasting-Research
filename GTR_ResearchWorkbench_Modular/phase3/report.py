"""Paired, validation-only comparisons (not test-based model selection)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

MODELS=('GTR','GTRLinearResidual','GTRPhaseInnovation','GTRPhaseAdaptive')


def _bootstrap_paired(diff,original_indices,seq_len,horizon,reps=400,seed=2024):
    """Window-block bootstrap on horizon-averaged, channel-aggregated delta.

    Descriptive interval only: overlapping windows and validation-driven
    checkpoint selection preclude an independent confirmatory significance claim.
    """
    x=np.asarray(diff,dtype=float)
    if len(x)<15:return (float('nan'),float('nan'),0)
    ids=np.asarray(original_indices,dtype=int)
    typical_spacing=max(1.,float(np.median(np.diff(ids))))
    block=min(len(x),max(2,int(np.ceil((int(seq_len)+int(horizon))/typical_spacing))))
    # A handful of overlapping blocks cannot support a meaningful interval.
    if len(x) / block < 5:
        return float('nan'),float('nan'),int(block)
    rng=np.random.default_rng(seed)
    vals=np.empty(reps,dtype=float)
    for i in range(reps):
        starts=rng.integers(0,max(1,len(x)-block+1),size=int(np.ceil(len(x)/block)))
        sampled=np.concatenate([np.arange(s,s+block) for s in starts])[:len(x)]
        vals[i]=x[sampled].mean()
    return (float(np.quantile(vals,.025)),float(np.quantile(vals,.975)),int(block))


def _terciles_by_channel(df,feature):
    # Quantiles computed on validation only, within channel; descriptive, not
    # selected by the forecast error. All models share exactly these bins.
    label=pd.Series(index=df.index,dtype='object')
    for _,ind in df.groupby('channel').groups.items():
        values=df.loc[ind,feature].astype(float)
        try:
            ranked=values.rank(method='first')
            bins=pd.qcut(ranked,3,labels=['Low','Medium','High']).astype(str)
        except ValueError:
            bins=pd.Series(['Medium']*len(values),index=values.index)
        label.loc[ind]=bins.to_numpy()
    return label


def report(outdir):
    outdir=Path(outdir)
    bymodel={};metadata={}
    for model in MODELS:
        file=outdir/f'{model}.csv'
        if not file.exists():raise FileNotFoundError(file)
        bymodel[model]=pd.read_csv(file).sort_values(['sample_index','channel']).reset_index(drop=True)
        metadata[model]=json.loads(file.with_suffix('.json').read_text(encoding='utf-8'))
    ref=metadata['GTR']
    for m,meta in metadata.items():
        for k in ['dataset','horizon','seed','seq_len','channels','source_dataset_sha256','sample_indices','channel_names','mode','split']:
            if meta[k]!=ref[k]:raise ValueError(f'{m} has mismatched {k}')
        if not bymodel[m][['sample_index','channel']].equals(bymodel['GTR'][['sample_index','channel']]):
            raise ValueError(f'{m} validation sample/channel alignment failed')
    baseline=bymodel['GTR']
    metrics=[];summary=[];channel_rows=[];condition_rows=[]
    for m in MODELS:
        v=bymodel[m]
        metrics.append(dict(model=m,samples=ref['samples'],channels=ref['channels'],
                            subset_val_mse=v['mse'].mean(),subset_val_mae=v['mae'].mean(),
                            note='Deterministic diagnostic subset (not official shuffled best val)'))
        if m=='GTR':continue
        diff=v['mse'].to_numpy()-baseline['mse'].to_numpy()
        delta=pd.DataFrame(dict(sample_index=v.sample_index,channel=v.channel,
                                diff=diff))
        per_window=delta.groupby('sample_index',sort=True)['diff'].mean()
        low,high,block=_bootstrap_paired(per_window.to_numpy(),per_window.index.to_numpy(),
                                          ref['seq_len'],ref['horizon'])
        summary.append(dict(model=m,baseline_subset_mse=baseline.mse.mean(),
                            variant_subset_mse=v.mse.mean(),delta_mse=diff.mean(),
                            mse_improvement_pct=-100*diff.mean()/baseline.mse.mean(),
                            fraction_window_channel_improved=float((diff<0).mean()),
                            descriptive_block_boot_ci_low=low,descriptive_block_boot_ci_high=high,
                            bootstrap_block_in_sampled_windows=block))
        for c,g in delta.groupby('channel'):
            c=int(c);idx=g.index.to_numpy()
            b=baseline.loc[idx];x=v.loc[idx]
            channel_rows.append(dict(model=m,channel=c,channel_name=ref['channel_names'][c],
               base_mse=b.mse.mean(),variant_mse=x.mse.mean(),delta_mse=g['diff'].mean(),
               improvement_pct=100*(b.mse.mean()-x.mse.mean())/max(b.mse.mean(),1e-10),
               win_fraction=(g['diff']<0).mean(),sample_count=len(g)))
        # Use past-only features from baseline; shared across all model samples.
        for feature in ('relative_change','relative_drift'):
            binned=_terciles_by_channel(baseline,feature)
            for tier in ('Low','Medium','High'):
                take=binned==tier
                bm=baseline.loc[take,'mse'].mean();vm=v.loc[take,'mse'].mean()
                condition_rows.append(dict(model=m,condition=feature,tercile=tier,
                    n=int(take.sum()),baseline_mse=bm,variant_mse=vm,
                    delta_mse=vm-bm,improvement_pct=100*(bm-vm)/max(bm,1e-10),
                    win_fraction=float((diff[take.to_numpy()]<0).mean())))
        # Learned phase-fit R2 is available from Phase2 variants, not GTR;
        # it is a descriptive fitted latent representation.
        if v['phase_fit_r2'].notna().any():
            feature='phase_fit_r2'
            binned=_terciles_by_channel(v,feature)
            for tier in ('Low','Medium','High'):
                take=binned==tier
                bm=baseline.loc[take,'mse'].mean();vm=v.loc[take,'mse'].mean()
                condition_rows.append(dict(model=m,condition=feature,tercile=tier,
                    n=int(take.sum()),baseline_mse=bm,variant_mse=vm,
                    delta_mse=vm-bm,improvement_pct=100*(bm-vm)/max(bm,1e-10),
                    win_fraction=float((diff[take.to_numpy()]<0).mean())))
    for name,rows in [('model_metrics.csv',metrics),('paired_summary.csv',summary),
                      ('by_channel.csv',channel_rows),('by_condition.csv',condition_rows)]:
        pd.DataFrame(rows).to_csv(outdir/name,index=False,float_format='%.9g')
    report_text=[f"GTR Phase3 validation analysis: {ref['dataset']} H{ref['horizon']} seed {ref['seed']}",
      f"Same deterministic validation windows: {ref['samples']}; channels: {ref['channels']}",
      'NOTE: These MSE values may differ from official best_val_mse because official uses shuffled drop_last batches.',
      'CAUTION: Block bootstrap intervals are descriptive only; validation was involved in early stopping.',
      'CAUTION: Phase-fit R2 is a fitted latent representation, not verified ground-truth seasonality.','',
      pd.DataFrame(metrics).to_string(index=False,float_format=lambda x:f'{x:.6g}'),'\nPaired vs GTR:',
      pd.DataFrame(summary).to_string(index=False,float_format=lambda x:f'{x:.6g}'),'\nFiles: by_channel.csv, by_condition.csv']
    (outdir/'REPORT.txt').write_text('\n'.join(report_text)+'\n',encoding='utf-8')
    print('\n'.join(report_text))
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('outdir')
    a=p.parse_args();report(a.outdir)

if __name__=='__main__':main()
