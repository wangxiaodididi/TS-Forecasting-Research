#!/usr/bin/env python3
"""Paloma v8, validation-led forecast diagnostics and targeted route adaptations.

Copies a pinned upstream Paloma repo into a private run directory. Never edits
upstream sources/checkpoints. All selection metrics refer to validation only.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from experiment import (SHAS, build_params, find_data, validate_data,
                        split_names, get_cuda, run_job)

ORIGINAL_BLOB='03dc7c6d23751cde6acaca77c803117c89702006'
V='8.0'
ENV_VARIANT='PALOMA_V8_VARIANT'
ENV_LOSS='PALOMA_V8_LOSS'
VARIANTS={
 'uniform_l1':('original','uniform_l1','Training-objective control: unweighted MAE; architecture unchanged'),
 'trend_skip':('trend_skip','paper','Recent/medium-horizon trend-conditioned residual skip'),
 'structure_skip':('structure_skip','paper','Sample-conditioned level/trend/periodic forecast residual structure'),
 'trend_skip_uniform':('trend_skip','uniform_l1','Trend-conditioned skip with uniform-MAE training objective'),
}

PATCH=r'''
# --- PALOMA_V8_HORIZON_TREND ---
import os as _v8_os
import torch as _v8_t
import torch.nn as _v8_nn
import torch.nn.functional as _v8_F

class _V8TrendSkip(_v8_nn.Module):
    """Level/trend facts from observed history only, forecast-shaped residual.

    First output starts exactly at zero, so upstream Paloma's first forward is
    unchanged. Uses sample-normalized input, returns normalized forecast delta.
    """
    def __init__(self,horizon):
        super().__init__()
        self.horizon=horizon
        self.featurizer=_v8_nn.Sequential(_v8_nn.Linear(4,32),_v8_nn.GELU(),_v8_nn.Linear(32,32),_v8_nn.GELU())
        self.to_delta=_v8_nn.Linear(32,horizon)
        _v8_nn.init.zeros_(self.to_delta.weight)
        _v8_nn.init.zeros_(self.to_delta.bias)
    def forward(self,x): # x: [B,C,L] already normalized
        L=x.size(-1)
        q=max(4,L//8);k=max(8,L//4)
        recent=x[:,:,-q:].mean(-1)
        prev=x[:,:,-2*q:-q].mean(-1)
        start=x[:,:,:k].mean(-1)
        end=x[:,:,-k:].mean(-1)
        feat=_v8_t.stack((recent-prev,end-start, x[:,:,-1]-recent,
                          x[:,:,1:].sub(x[:,:,:-1]).square().mean(-1).sqrt()),dim=-1)
        return self.to_delta(self.featurizer(feat))

class _V8StructureSkip(_v8_nn.Module):
    """Three independently motivated forecast structures, conditioned on input.

    An independent, small mechanism adaptation; NOT the TStruct implementation.
    Inputs are normalized and all future values are computed from past inputs.
    """
    def __init__(self,horizon,period):
        super().__init__()
        self.horizon=horizon;self.period=period
        self.chooser=_v8_nn.Sequential(_v8_nn.Linear(4,24),_v8_nn.GELU(),_v8_nn.Linear(24,3))
        self.strength=_v8_nn.Parameter(_v8_t.zeros(3))
        self.output_proj=_v8_nn.Linear(horizon,horizon,bias=False)
        _v8_nn.init.eye_(self.output_proj.weight)
    def forward(self,x):
        B,C,L=x.shape;H=self.horizon;P=min(self.period,L)
        short=max(4,L//8);med=max(8,L//4)
        shorttrend=x[:,:,-short:].mean(-1)-x[:,:,-2*short:-short].mean(-1)
        medtrend=x[:,:,-med:].mean(-1)-x[:,:,:med].mean(-1)
        level=(x[:,:,-short:].mean(-1)-x.mean(-1))
        vol=x[:,:,1:].sub(x[:,:,:-1]).square().mean(-1).sqrt()
        feat=_v8_t.stack([shorttrend,medtrend,level,vol],dim=-1)
        weights=_v8_t.softmax(self.chooser(feat),dim=-1)
        hh=_v8_t.linspace(0,1,H,device=x.device,dtype=x.dtype).view(1,1,H)
        # centered level, saturating (not exploding) trend, centered repeat pattern
        structure0=level.unsqueeze(-1).expand(-1,-1,H)
        structure1=_v8_t.tanh(shorttrend).unsqueeze(-1)*hh
        pat=x[:,:,-P:]-x[:,:,-P:].mean(-1,keepdim=True)
        structure2=pat.repeat(1,1,(H+P-1)//P)[:,:,:H]
        basis=_v8_t.stack([structure0,structure1,structure2],dim=-1)
        combined=(basis*weights.unsqueeze(-2)*self.strength.view(1,1,1,3)).sum(-1)
        return self.output_proj(combined)

_V8OriginalInit=Model.__init__
_V8OriginalForward=Model.forward

def _v8_init(self,configs):
    _V8OriginalInit(self,configs)
    variant=_v8_os.environ.get('PALOMA_V8_VARIANT','original')
    if variant not in ('original','trend_skip','structure_skip'):
        raise ValueError('Unknown V8_VARIANT: '+variant)
    self.v8_variant=variant
    if variant=='original':return
    rng_state=_v8_t.get_rng_state()
    cuda_rng_state=_v8_t.cuda.get_rng_state_all() if _v8_t.cuda.is_initialized() else None
    try:
        self.v8_adapter=(_V8TrendSkip(self.pred_len) if variant=='trend_skip'
                         else _V8StructureSkip(self.pred_len,self.cycle_len))
    finally:
        _v8_t.set_rng_state(rng_state)
        if cuda_rng_state is not None:_v8_t.cuda.set_rng_state_all(cuda_rng_state)

def _v8_forward(self,x,cycle_index):
    y=_V8OriginalForward(self,x,cycle_index)
    if self.v8_variant=='original':return y
    if self.use_revin:
        mu=x.mean(dim=1,keepdim=True)
        std=(x.var(dim=1,keepdim=True)+1e-5).sqrt()
        xn=(x-mu)/std
    else:
        std=_v8_t.ones_like(x[:,:1,:]);xn=x
    delta=self.v8_adapter(xn.transpose(1,2)).transpose(1,2)
    return y+delta*std

Model.__init__=_v8_init
Model.forward=_v8_forward
'''

LOSS_PATCH=r'''
# --- PALOMA_V8_TRAIN_LOSS ---
import os as _v8_loss_os
import torch as _v8_loss_t
_V8_OriginalLoss=WeightedLoss.__call__
_V8_OriginalVali=Exp_Main.vali

def _v8_weighted_loss(self,pred,gt):
    mode=_v8_loss_os.environ.get('PALOMA_V8_LOSS','paper')
    if mode=='uniform_l1':
        return _v8_loss_t.abs(pred-gt).mean()
    return _V8_OriginalLoss(self,pred,gt)

def _v8_vali(self,data,loader,criterion):
    # Keep original author's weighted criterion for checkpoint selection in all
    # experiments; only the training objective is varied. The model is set to
    # eval mode inside _V8_OriginalVali, as in official Paloma.
    class _OriginalCriterion:
        def __call__(self,pred,gt):
            return _V8_OriginalLoss(criterion,pred,gt)
    return _V8_OriginalVali(self,data,loader,_OriginalCriterion())

WeightedLoss.__call__=_v8_weighted_loss
Exp_Main.vali=_v8_vali
'''

def blob_sha(path:Path)->str:
    raw=path.read_bytes()
    return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()

def patch_private(upstream:Path, private:Path):
    src=upstream/'models'/'Paloma.py'
    if not src.is_file():raise FileNotFoundError('Upstream Paloma missing: '+str(src))
    sha=blob_sha(src)
    if sha!=ORIGINAL_BLOB:
        raise RuntimeError(f'Pinned model differs from inspected source: {sha}, expected {ORIGINAL_BLOB}')
    if private.exists():
        manifest=private.parent/'manifest_v8.json'
        if manifest.exists():
            d=json.loads(manifest.read_text(encoding='utf-8'))
            p=private/'models'/'Paloma.py'
            e=private/'exp'/'exp_main.py'
            if p.is_file() and e.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==d['model_sha256'] and hashlib.sha256(e.read_bytes()).hexdigest()==d['training_sha256']:
                return d
        raise RuntimeError('Existing v8 private source differs; do not overwrite: '+str(private))
    private.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(upstream,private,ignore=shutil.ignore_patterns('.git','checkpoints','results','__pycache__'))
    model=private/'models'/'Paloma.py'
    model.write_text(model.read_text(encoding='utf-8')+'\n'+PATCH,encoding='utf-8')
    exp=private/'exp'/'exp_main.py'
    e=exp.read_text(encoding='utf-8')
    # Disable test-set reads during the training loop AND in post-training
    # evaluation. Official upstream reports Test Loss each epoch; this patched
    # experiment intentionally substitutes validation for test in both calls.
    needle="test_data, test_loader = self._get_data(flag='test')"
    if e.count(needle)!=2:raise RuntimeError('Unsupported exp_main.py: expected exactly two test-loader calls')
    e=e.replace(needle,"test_data, test_loader = self._get_data(flag='val')")
    e=e.replace("print('mse:{}, mae:{}'.format(mse, mae))", "print('V8_VALIDATION_METRICS mse:{}, mae:{}'.format(mse, mae))")
    e=e.replace("Test Loss: {4:.7f}", "Val Mirror Loss: {4:.7f}")
    e+='\n'+LOSS_PATCH
    exp.write_text(e,encoding='utf-8')
    d=dict(version=V,upstream_commit=SHAS['Paloma'],original_git_blob=sha,
           model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
           training_sha256=hashlib.sha256(exp.read_bytes()).hexdigest(),
           metric_split='validation',test_set_used=False)
    (private.parent/'manifest_v8.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    return d

def read_v7_validation(out_root:Path):
    path=out_root/'runs'/'paloma_path_diagnostics_v7'/'all_tasks_summary.csv'
    result={}
    if not path.exists():return result
    with path.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if row.get('split')!='val' or row.get('match_status') not in ('validation_diagnostics',None):continue
            try:
                key=row['dataset'],int(row['horizon'])
                result[key]=dict(mse=float(row['original_recalculated_mse']),mae=float(row['original_recalculated_mae']),source=str(path))
            except (ValueError,TypeError,KeyError):pass
    return result

def val_naive_diagnostics(src:Path,file:Path,ds:str,h:int,batch_size:int=256):
    """No test set, no learned parameters; validation horizon-reference controls."""
    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0,str(src))
    try:
        from data_provider.data_loader import Dataset_ETT_hour,Dataset_ETT_minute
    finally:sys.path.pop(0)
    cls=Dataset_ETT_hour if ds=='ETTh1' else Dataset_ETT_minute
    p=24 if ds=='ETTh1' else 96
    data=cls(str(file.parent),flag='val',size=[96,0,h],features='M',target='OT',
             scale=True,timeenc=1,freq='h' if ds=='ETTh1' else 't',cycle=p,data_path=file.name)
    loader=DataLoader(data,batch_size=batch_size,shuffle=False,num_workers=0)
    results={name:[0.,0.,0,0,0,0,0,0] for name in ('last_value','seasonal_naive')}
    denom=[0,0,0,0]
    with torch.no_grad():
        for x,y,_,_,_ in loader:
            x=x.float();y=y.float();b,_,c=y.shape
            seq={'last_value':x[:,-1:,:].expand(-1,h,-1),
                 'seasonal_naive':x[:,-p:,:].repeat(1,(h+p-1)//p,1)[:,:h,:]}
            for k,idx in enumerate(torch.tensor_split(torch.arange(h),4)):
                denom[k]+=b*c*len(idx)
                for name,z in seq.items():
                    e=z[:,idx,:]-y[:,idx,:]
                    results[name][2*k]+=float((e*e).sum())
                    results[name][2*k+1]+=float(e.abs().sum())
    rows=[]
    for name,vals in results.items():
        mse_all=sum(vals[::2])/sum(denom)
        mae_all=sum(vals[1::2])/sum(denom)
        for i in range(4):
            rows.append(dict(dataset=ds,horizon=h,model=name,split='val',quartile=i+1,
                mse=vals[2*i]/denom[i],mae=vals[2*i+1]/denom[i],
                overall_mse=mse_all,overall_mae=mae_all))
    return rows

def write_csv(path,rows,cols=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    cols=cols or list(rows[0])
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
    tmp.replace(path)

def collect(out_dir:Path,workspace:Path):
    baseline=read_v7_validation(workspace)
    rows=[]
    for m in sorted((out_dir/'paper').glob('*/h*/*/metadata.json')):
        try:d=json.loads(m.read_text(encoding='utf-8'))
        except (ValueError,OSError):continue
        if d.get('variant') not in VARIANTS:continue
        b=baseline.get((d['dataset'],int(d['horizon'])),{})
        r={k:d.get(k,'') for k in ('dataset','horizon','variant','status','returncode','mse','mae','wall_seconds','source_commit','patched_source_sha256','training_epochs','training_loss','source_inspiration')}
        r.update(metric_split='val',original_val_mse=b.get('mse',''),original_val_mae=b.get('mae',''),v7_baseline_file=b.get('source',''))
        for metric in ('mse','mae'):
            if isinstance(d.get(metric),(float,int)) and b.get(metric,0)>0:
                r[metric+'_val_gain_pct']=round(100*(b[metric]-d[metric])/b[metric],6)
            else:r[metric+'_val_gain_pct']=''
        rows.append(r)
    file=out_dir/'v8_validation_summary.csv';write_csv(file,rows)
    print(f'[COLLECT] {len(rows)} validation-only jobs -> {file}',flush=True)
    return file

def load_test_stub(file:Path):
    """Used by independent unit tests. Does not bypass production hash check."""
    spec=importlib.util.spec_from_file_location('_v8_test_official',file)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def smoke_module(stub_module):
    """Synthetic test with a test-only Paloma-compatible module, not a paper reproduction."""
    import torch
    import types
    src=stub_module.__dict__
    exec(PATCH,src)
    c=SimpleNamespace(seq_len=96,pred_len=96,enc_in=7,cycle=24,d_model=256,dropout=.5,use_revin=1,individual=0)
    x=torch.randn(2,96,7);cyc=torch.tensor([0,1]);target=torch.randn(2,96,7)
    os.environ[ENV_VARIANT]='original';torch.manual_seed(2024)
    base=stub_module.Model(c);y0=base(x,cyc)
    for v in ('trend_skip','structure_skip'):
        os.environ[ENV_VARIANT]=v;torch.manual_seed(2024)
        model=stub_module.Model(c)
        assert torch.equal(y0,model(x,cyc)),v
        assert all(torch.equal(z,model.state_dict()[k]) for k,z in base.state_dict().items())
        optim=torch.optim.Adam(model.parameters(),lr=0.01)
        for _ in range(3):
            optim.zero_grad();loss=(model(x,cyc)-target).square().mean();loss.backward();optim.step()
        optim.zero_grad();loss=(model(x,cyc)-target).square().mean();loss.backward()
        grads=[p.grad for p in model.v8_adapter.parameters() if p.grad is not None]
        assert len(grads)>0 and any(g.abs().sum().item()>0 for g in grads),v
        assert torch.isfinite(model(x,cyc)).all()
    os.environ.pop(ENV_VARIANT,None)
    print('[SMOKE] synthetic branch tests passed (NOT official metrics)',flush=True)

def parse_args(argv=None):
    p=argparse.ArgumentParser(description='Paloma v8 validation-driven interventions')
    p.add_argument('--mode',choices=['all','smoke','full','collect','diagnose'],default='all')
    p.add_argument('--datasets',default='ETTh1,ETTm1');p.add_argument('--horizons',default='96,336')
    p.add_argument('--variants',default=','.join(VARIANTS))
    p.add_argument('--workspace',default=str(ROOT));p.add_argument('--data-dir',default=os.environ.get('DATA_DIR','/root/autodl-tmp/GTR_ResearchWorkbench_Modular/dataset'))
    p.add_argument('--sources-dir',default='');p.add_argument('--out-dir',default='')
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--epochs',type=int,default=0)
    p.add_argument('--force',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--continue-on-error',action='store_true')
    a=p.parse_args(argv)
    a.workspace=Path(a.workspace).expanduser().resolve()
    a.sources_dir=Path(a.sources_dir).expanduser().resolve() if a.sources_dir else a.workspace/'sources'
    a.out_dir=Path(a.out_dir).expanduser().resolve() if a.out_dir else a.workspace/'runs'/'paloma_validation_guided_v8'
    a.datasets=split_names(a.datasets,{'ETTh1':1,'ETTm1':1})
    a.variants=split_names(a.variants,{k:1 for k in VARIANTS})
    a.horizons=[int(x) for x in a.horizons.split(',')]
    if not a.horizons or any(x not in (96,336) for x in a.horizons):p.error('horizons 96,336 only')
    return a

def main(argv=None):
    a=parse_args(argv)
    if a.mode=='collect':return collect(a.out_dir,a.workspace)
    upstream=a.sources_dir/'Paloma'
    if not upstream.is_dir():raise FileNotFoundError('Paloma sources missing: '+str(upstream))
    inputfiles={}
    for ds in a.datasets:
        f=find_data(a.data_dir,ds)
        if not a.dry_run:validate_data(f,ds)
        inputfiles[ds]=f
    if a.mode in ('diagnose','all') and not a.dry_run:
        rows=[]
        for ds in a.datasets:
            for h in a.horizons:
                print(f'[VALIDATION NAIVE] {ds} H{h}',flush=True)
                rows.extend(val_naive_diagnostics(upstream,inputfiles[ds],ds,h))
        write_csv(a.out_dir/'v8_validation_naive_by_quartile.csv',rows)
    if a.mode=='diagnose':return
    private=a.out_dir/'private_source'/'Paloma'
    if a.dry_run:
        for ds in a.datasets:
            for h in a.horizons:
                for v in a.variants:print('[DRY]',ds,h,v,flush=True)
        return
    manifest=patch_private(upstream,private)
    if a.mode=='smoke':
        # official source in private test rather than mocked source; no network.
        import torch
        import types
        f=private/'models'/'Paloma.py'
        c=SimpleNamespace(seq_len=96,pred_len=96,enc_in=7,cycle=24,d_model=256,dropout=.5,use_revin=1,individual=0)
        mod=load_test_stub(f)
        for ds in a.datasets:
            for h in a.horizons:
                cc=SimpleNamespace(**{**c.__dict__,'pred_len':h,'cycle':24 if ds=='ETTh1' else 96})
                x=torch.randn(2,96,7);cyc=torch.tensor([0,1]);ytarget=torch.randn(2,h,7)
                os.environ[ENV_VARIANT]='original';torch.manual_seed(2024)
                original=mod.Model(cc).eval();y0=original(x,cyc)
                for name in a.variants:
                    variant,loss,_=VARIANTS[name]
                    os.environ[ENV_VARIANT]=variant;torch.manual_seed(2024)
                    altered=mod.Model(cc).eval()
                    assert torch.equal(altered(x,cyc),y0),name
                    assert all(torch.equal(val,altered.state_dict()[k]) for k,val in original.state_dict().items())
                    altered.train();opt=torch.optim.Adam(altered.parameters(),lr=0.001)
                    for _ in range(3):
                        opt.zero_grad();z=altered(x,cyc);l=(z-ytarget).square().mean();l.backward();opt.step()
                    if variant!='original':
                        opt.zero_grad();l=(altered(x,cyc)-ytarget).square().mean();l.backward()
                        grad=sum(float(p.grad.abs().sum()) for p in altered.v8_adapter.parameters() if p.grad is not None)
                        assert grad>0,(name,grad)
                    print(f'[SMOKE_OK] {ds} H{h} {name}',flush=True)
        os.environ.pop(ENV_VARIANT,None)
        return
    if a.mode=='all':
        main(['--mode','smoke','--datasets',','.join(a.datasets),'--horizons',','.join(str(h) for h in a.horizons),
              '--variants',','.join(a.variants),'--workspace',str(a.workspace),'--data-dir',str(a.data_dir),
              '--sources-dir',str(a.sources_dir),'--out-dir',str(a.out_dir)])
    if not get_cuda():raise RuntimeError('GPU required for author training script; run --mode smoke first')
    failed=[]
    for ds in a.datasets:
        for h in a.horizons:
            for v in a.variants:
                name,loss,description=VARIANTS[v]
                dest=a.out_dir/'paper'/ds/f'h{h}'/v
                cmd,meta=build_params('Paloma',ds,h,'paper',a.epochs,a.gpu,inputfiles[ds],dest)
                cmd[cmd.index('--model_id')+1]+='_v8_'+v
                meta.update(variant=v,seed=2024,training_loss=loss,source_inspiration=description,
                            metric_split='val',patched_source_sha256=manifest['model_sha256'],
                            training_source_sha256=manifest['training_sha256'],
                            v8_test_held_out=True)
                existing=dest/'metadata.json'
                if existing.exists() and not a.force:
                    try:p=json.loads(existing.read_text(encoding='utf-8'))
                    except (ValueError,OSError):p={}
                    if(p.get('status')=='ok' and p.get('patched_source_sha256')==meta['patched_source_sha256']
                       and p.get('training_epochs')==meta['training_epochs'] and p.get('training_loss')==loss):
                        print('[SKIP]',ds,h,v,flush=True);continue
                os.environ[ENV_VARIANT]=name;os.environ[ENV_LOSS]=loss
                try:
                    result=run_job(cmd,private,dest,meta,a)
                finally:
                    os.environ.pop(ENV_VARIANT,None);os.environ.pop(ENV_LOSS,None)
                collect(a.out_dir,a.workspace)
                if result.get('status')!='ok':
                    failed.append(f'{ds}/H{h}/{v}')
                    if not a.continue_on_error:raise RuntimeError('Run failed; inspect '+str(dest/'train.log'))
    collect(a.out_dir,a.workspace)
    if failed:raise RuntimeError('Failures: '+', '.join(failed))
    print('[DONE] v8 validation-only experiment set complete',flush=True)

if __name__=='__main__':main()
