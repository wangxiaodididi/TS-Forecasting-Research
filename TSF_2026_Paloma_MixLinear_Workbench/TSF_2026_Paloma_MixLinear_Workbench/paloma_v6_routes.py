#!/usr/bin/env python3
"""Paloma v6: test token-wise branch communication and future-directed information reading.

Independent, zero-initialized mechanism adaptations. These are NOT exact replicas
of TimeMosaic, T3Time or TStruct. The upstream Paloma source is pinned and unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiment import SHAS, build_params, find_data, get_cuda, run_job, split_names, validate_data

OFFICIAL_BLOB = '03dc7c6d23751cde6acaca77c803117c89702006'
VERSION = '6.0'
ENV = 'PALOMA_V6_ROUTE'
VARIANTS = ('cross_branch_tokens','future_query','branch_aware_future')
ADDED_PARAMS = {
 'cross_branch_tokens': {96: 103168, 336: 103168},
 'future_query': {96: 47392, 336: 63312},
 'branch_aware_future': {96: 68320, 336: 84240},
}
DESCRIPTION = {
 'cross_branch_tokens':'Two-way phase/amplitude patch-token cross-attention, read out to residual hidden correction',
 'future_query':'Forecast-segment prompts cross-attend to ordered historical context patches, then correct forecasts',
 'branch_aware_future':'Future segment prompts independently read phase-conditioned and amplitude-conditioned historical tokens',
}
PROVENANCE = {
 'cross_branch_tokens': 'T3Time (AAAI 2026) cross-view alignment inspiration; independent minimal adaptation',
 'future_query': 'TimeMosaic (AAAI 2026) segment prompt decoding inspiration; independent cross-attention adaptation',
 'branch_aware_future': 'T3Time cross-view + TimeMosaic segmented prompts; independent architectural test',
}
REFERENCES = {
 'paloma': 'https://github.com/TSTS13/Paloma/blob/main/models/Paloma.py',
 'timemosaic':'https://github.com/BenchCouncil/TimeMosaic/blob/main/models/Mosaic.py',
 't3time':'https://github.com/monaf-chowdhury/T3Time/blob/main/models/T3Time.py',
 'tstruct':'https://github.com/yangzeli/TStruct/blob/main/models/TStruct.py',
}

# Appended to a verified, untouched pinned copy of the original Paloma.py.
# All added module code below is newly authored based on public conceptual mechanisms.
PATCH = r'''

# --- PALOMA_V6_INFORMATION_ROUTES ---
import os as _v6_os
import torch as _v6_torch
import torch.nn as _v6_nn
import torch.nn.functional as _v6_F


def _v6_make_patches(x, patch):
    # [B,C,L] -> [B*C,N,P], retaining strict chronological order
    b,c,l=x.shape
    if l%patch:
        raise ValueError(f'v6 history length {l} must divide patch {patch}')
    return x.reshape(b*c,l//patch,patch)

class _V6CrossBranchTokens(_v6_nn.Module):
    """Bidirectional branch token communication, not a scalar/gate fusion.

    Tokens represent original aligned, ordered time patches. The two attentions
    capture P<-A and A<-P. Final projection is zero initialized so the original
    Paloma forecast is exactly preserved at initialization.
    """
    def __init__(self, seq_len, d_model, patch=12, width=64):
        super().__init__()
        assert seq_len%patch==0
        self.patch=patch
        self.to_phase=_v6_nn.Linear(patch*2,width)
        self.to_amp=_v6_nn.Linear(patch*2,width)
        self.phase_latent=_v6_nn.Linear(d_model,width)
        self.amp_latent=_v6_nn.Linear(d_model,width)
        self.position=_v6_nn.Parameter(_v6_torch.zeros(1,seq_len//patch,width))
        self.p_read_a=_v6_nn.MultiheadAttention(width,4,batch_first=True)
        self.a_read_p=_v6_nn.MultiheadAttention(width,4,batch_first=True)
        self.layer_norm=_v6_nn.LayerNorm(width*2)
        self.to_delta=_v6_nn.Linear(width*2,d_model)
        _v6_nn.init.zeros_(self.to_delta.weight)
        _v6_nn.init.zeros_(self.to_delta.bias)

    def forward(self,residual,phase,amplitude,phase_impact,amplitude_impact):
        B,C,L=residual.shape
        rp=_v6_make_patches(residual,self.patch)
        pt=(self.to_phase(_v6_torch.cat([rp,_v6_make_patches(phase,self.patch)],dim=-1))
            +self.phase_latent(phase_impact.reshape(B*C,-1)).unsqueeze(1)+self.position)
        at=(self.to_amp(_v6_torch.cat([rp,_v6_make_patches(amplitude,self.patch)],dim=-1))
            +self.amp_latent(amplitude_impact.reshape(B*C,-1)).unsqueeze(1)+self.position)
        p_from_a,_=self.p_read_a(pt,at,at,need_weights=False)
        a_from_p,_=self.a_read_p(at,pt,pt,need_weights=False)
        joined=self.layer_norm(_v6_torch.cat([p_from_a,a_from_p],dim=-1))
        joined=joined.mean(dim=1)
        return self.to_delta(joined).reshape(B,C,-1)

class _V6FutureQuery(_v6_nn.Module):
    """Future queries retrieve history patches, not just independent segment heads.

    Output is a B,C,H correction; forecast horizon is never passed as target.
    Data are chronological 12-step patches [residual, original, Q, phase].
    """
    def __init__(self,seq_len,horizon,d_model,patch=12,seg=48,width=64):
        super().__init__()
        assert seq_len%patch==0
        self.patch=patch
        self.sizes=[min(seg,horizon-i) for i in range(0,horizon,seg)]
        self.memory=_v6_nn.Linear(patch*4,width)
        self.pos=_v6_nn.Parameter(_v6_torch.zeros(1,seq_len//patch,width))
        self.query=_v6_nn.Parameter(_v6_torch.randn(len(self.sizes),width)*0.02)
        self.context=_v6_nn.Linear(d_model,width)
        self.cross=_v6_nn.MultiheadAttention(width,4,batch_first=True)
        self.refine=_v6_nn.Sequential(_v6_nn.LayerNorm(width),_v6_nn.Linear(width,width),_v6_nn.GELU())
        self.seg_heads=_v6_nn.ModuleList([_v6_nn.Linear(width,n) for n in self.sizes])
        for h in self.seg_heads:
            _v6_nn.init.zeros_(h.weight)
            _v6_nn.init.zeros_(h.bias)

    def forward(self,features,residual,x_input,Q,phase):
        B,C,L=residual.shape
        patches=[_v6_make_patches(t,self.patch) for t in (residual,x_input,Q,phase)]
        memory=self.memory(_v6_torch.cat(patches,dim=-1))+self.pos
        bc=B*C
        query=self.query.unsqueeze(0).expand(bc,-1,-1)+self.context(features.reshape(bc,-1)).unsqueeze(1)
        received,_=self.cross(query,memory,memory,need_weights=False)
        refined=self.refine(received+query)
        pieces=[head(refined[:,i,:]) for i,head in enumerate(self.seg_heads)]
        return _v6_torch.cat(pieces,dim=-1).reshape(B,C,-1)

class _V6BranchAwareFuture(_v6_nn.Module):
    """Future segment tokens retrieve two distinct history representations.

    Attention is over temporal patches in each branch. This is NOT an exact
    TimeMosaic/T3Time implementation and does not use known future covariates.
    """
    def __init__(self,seq_len,horizon,d_model,patch=12,seg=48,width=64):
        super().__init__()
        assert seq_len%patch==0
        self.patch=patch
        self.sizes=[min(seg,horizon-i) for i in range(0,horizon,seg)]
        self.p_mem=_v6_nn.Linear(patch*2,width)
        self.a_mem=_v6_nn.Linear(patch*2,width)
        self.pos=_v6_nn.Parameter(_v6_torch.zeros(1,seq_len//patch,width))
        self.query=_v6_nn.Parameter(_v6_torch.randn(len(self.sizes),width)*0.02)
        self.context=_v6_nn.Linear(d_model,width)
        self.p_read=_v6_nn.MultiheadAttention(width,4,batch_first=True)
        self.a_read=_v6_nn.MultiheadAttention(width,4,batch_first=True)
        self.merge=_v6_nn.Sequential(_v6_nn.LayerNorm(width*2),
                                      _v6_nn.Linear(width*2,width),_v6_nn.GELU())
        self.heads=_v6_nn.ModuleList([_v6_nn.Linear(width,n) for n in self.sizes])
        for h in self.heads:
            _v6_nn.init.zeros_(h.weight)
            _v6_nn.init.zeros_(h.bias)

    def forward(self,features,residual,phase,amplitude):
        B,C,L=residual.shape
        rp=_v6_make_patches(residual,self.patch)
        pm=self.p_mem(_v6_torch.cat([rp,_v6_make_patches(phase,self.patch)],dim=-1))+self.pos
        am=self.a_mem(_v6_torch.cat([rp,_v6_make_patches(amplitude,self.patch)],dim=-1))+self.pos
        bc=B*C
        q=self.query.unsqueeze(0).expand(bc,-1,-1)+self.context(features.reshape(bc,-1)).unsqueeze(1)
        p,_=self.p_read(q,pm,pm,need_weights=False)
        a,_=self.a_read(q,am,am,need_weights=False)
        z=self.merge(_v6_torch.cat([p+q,a+q],dim=-1))
        return _v6_torch.cat([head(z[:,i,:]) for i,head in enumerate(self.heads)],dim=-1).reshape(B,C,-1)

_V6BaseInit=Model.__init__
_V6BaseForward=Model.forward

def _v6_init(self,configs):
    _V6BaseInit(self,configs)
    variant=_v6_os.environ.get('PALOMA_V6_ROUTE','original')
    if variant not in ('original','cross_branch_tokens','future_query','branch_aware_future'):
        raise ValueError('Unknown Paloma v6 variant: '+variant)
    self.v6_variant=variant
    if variant=='original': return
    state=_v6_torch.get_rng_state()
    cuda_state=_v6_torch.cuda.get_rng_state_all() if _v6_torch.cuda.is_initialized() else None
    try:
        if variant=='cross_branch_tokens':
            self.v6_module=_V6CrossBranchTokens(self.seq_len,self.d_model)
        elif variant=='future_query':
            self.v6_module=_V6FutureQuery(self.seq_len,self.pred_len,self.d_model)
        elif variant=='branch_aware_future':
            self.v6_module=_V6BranchAwareFuture(self.seq_len,self.pred_len,self.d_model)
    finally:
        _v6_torch.set_rng_state(state)
        if cuda_state is not None: _v6_torch.cuda.set_rng_state_all(cuda_state)

def _v6_forward(self,x,cycle_index):
    variant=self.v6_variant
    if variant=='original':
        return _V6BaseForward(self,x,cycle_index)
    if self.use_revin:
        seq_mean=_v6_torch.mean(x,dim=1,keepdim=True)
        seq_var=_v6_torch.var(x,dim=1,keepdim=True)+1e-5
        x=(x-seq_mean)/_v6_torch.sqrt(seq_var)
    x_input=x.permute(0,2,1)
    gather=(cycle_index.view(-1,1)+_v6_torch.arange(self.seq_len,device=cycle_index.device).view(1,-1))%self.cycle_len
    query_input=self.Q[gather].permute(0,2,1)
    phase=self.phase[gather].permute(0,2,1)
    amplitude=self.amplitude[gather].permute(0,2,1)
    residual=x_input-query_input
    phase_impact=self.phase_fusion(residual,phase)
    amplitude_impact=self.amplitude_fusion(residual,amplitude)
    merged=phase_impact+amplitude_impact
    if variant=='cross_branch_tokens':
        merged=merged+self.v6_module(residual,phase,amplitude,phase_impact,amplitude_impact)
    input_proj=self.input_proj(x_input)
    hidden=self.model(merged)
    features=hidden+input_proj
    output=self.output_proj(features)
    if variant=='future_query':
        output=output+self.v6_module(features,residual,x_input,query_input,phase)
    elif variant=='branch_aware_future':
        output=output+self.v6_module(features,residual,phase,amplitude)
    output=output.permute(0,2,1)
    if self.use_revin:
        output=output*_v6_torch.sqrt(seq_var)+seq_mean
    return output

Model.__init__=_v6_init
Model.forward=_v6_forward
# --- END PALOMA_V6_INFORMATION_ROUTES ---
'''


def git_blob(contents:bytes)->str:
    return hashlib.sha1(f'blob {len(contents)}\0'.encode('ascii')+contents).hexdigest()


def prepare_private_source(upstream:Path, private:Path)->dict:
    upstream,private=upstream.resolve(),private.resolve()
    if upstream==private or upstream in private.parents:
        raise ValueError('Private source must be outside untouched upstream source tree')
    orig=upstream/'models'/'Paloma.py'
    if not orig.is_file() or not (upstream/'run.py').is_file():
        raise FileNotFoundError('Official Paloma source missing; run bash setup_sources.sh')
    raw=orig.read_bytes()
    actual=git_blob(raw)
    if actual!=OFFICIAL_BLOB:
        raise RuntimeError(f'Unexpected upstream Paloma.py blob {actual}; expected {OFFICIAL_BLOB}. Source will not be modified.')
    # The patched file is deterministic across runs: exact original bytes + append.
    output_bytes=raw+PATCH.encode('utf-8')
    target_sha=hashlib.sha256(output_bytes).hexdigest()
    manifest_file=private/'v6_manifest.json'
    if private.exists():
        if not manifest_file.exists():
            raise RuntimeError(f'Existing unrecognized source directory: {private}')
        manifest=json.loads(manifest_file.read_text(encoding='utf-8'))
        file=private/'models'/'Paloma.py'
        if manifest.get('version')!=VERSION or not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=target_sha:
            raise RuntimeError(f'Existing private source is different: {private}. Manual inspection required.')
        return manifest
    private.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(upstream,private,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc','runs','checkpoints','results','test_results','logs'))
    (private/'models'/'Paloma.py').write_bytes(output_bytes)
    manifest={'version':VERSION,'source_commit':SHAS['Paloma'],'official_blob':OFFICIAL_BLOB,
              'patched_sha256':target_sha,'variants':list(VARIANTS),'references':REFERENCES,
              'note':'Independent, documented adaptations; official source untouched.'}
    manifest_file.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    return manifest


def import_file(path:Path,name:str):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(f'Unable to import {path}')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def smoke(upstream:Path,private:Path,ds:str,h:int,variants:list[str], data_file:Path|None=None):
    import numpy as np
    import torch
    c=SimpleNamespace(seq_len=96,pred_len=h,enc_in=7,
          cycle=24 if ds=='ETTh1' else 96,d_model=256,
          dropout=0.5 if ds=='ETTh1' or h==336 else 0.4,
          use_revin=1,individual=0)
    orig=import_file(upstream/'models'/'Paloma.py','paloma_upstream_v6')
    changed=import_file(private/'models'/'Paloma.py','paloma_changed_v6')
    if data_file is not None and data_file.exists():
        with data_file.open('r',encoding='utf-8-sig',newline='') as f:
            reader=csv.reader(f);next(reader)
            rows=[]
            for row in reader:
                rows.append([float(z) for z in row[1:8]])
                if len(rows)>=96+h+2:break
        arr=np.asarray(rows,dtype=np.float32)
        if len(arr)<96+h+2:raise ValueError('Insufficient rows for ETT window smoke')
        mean=arr[:96].mean(axis=0)
        std=arr[:96].std(axis=0)+1e-5
        arr=(arr-mean)/std
        x=torch.tensor(np.stack([arr[:96],arr[1:97]]))
        y=torch.tensor(np.stack([arr[96:96+h],arr[97:97+h]]))
    else:
        torch.manual_seed(2022)
        x=torch.randn(2,96,7)
        y=torch.randn(2,h,7)
    idx=torch.tensor([0,1],dtype=torch.long)
    torch.manual_seed(2024)
    base=orig.Model(c).eval()
    state=base.state_dict()
    os.environ[ENV]='original'
    torch.manual_seed(2024)
    check=changed.Model(c).eval()
    assert state.keys()==check.state_dict().keys()
    assert all(torch.equal(v,check.state_dict()[k]) for k,v in state.items())
    with torch.no_grad():
        baseline=base(x,idx)
        assert torch.equal(baseline,check(x,idx)), 'Original changed after patching'
    print(f'[CONTROL_OK] {ds} H{h} original predictions identical',flush=True)
    for variant in variants:
        os.environ[ENV]=variant
        torch.manual_seed(2024)
        model=changed.Model(c).eval()
        for k,v in state.items():
            assert torch.equal(v,model.state_dict()[k]),('baseline parameter mutated',k)
        with torch.no_grad():
            out=model(x,idx)
            assert torch.equal(out,baseline),(variant,(out-baseline).abs().max().item())
        model.train()
        yp=model(x,idx)
        assert yp.shape==(2,h,7)
        assert torch.isfinite(yp).all(),variant
        loss=(yp-y).square().mean()
        loss.backward()
        initial=sum(float(p.grad.abs().sum()) for p in model.v6_module.parameters() if p.grad is not None)
        assert initial>0,(variant,'no initial adapter gradient')
        opt=torch.optim.Adam(model.parameters(),lr=0.001)
        # Need 3 steps for zero-initialized readouts to expose upstream gradients.
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            train_loss=(model(x,idx)-y).square().mean()
            train_loss.backward()
            opt.step()
        # After several updates ensure inputs to final readout can receive gradient.
        opt.zero_grad(set_to_none=True)
        train_loss=(model(x,idx)-y).square().mean()
        train_loss.backward()
        live=sum(float(p.grad.abs().sum()) for n,p in model.v6_module.named_parameters() if p.grad is not None and not any(t in n for t in ('to_delta','seg_heads','heads.')))
        assert live>0,(variant,'upstream adapter parameters did not receive gradient')
        print(f'[SMOKE_OK] {variant} {ds} H{h} {tuple(yp.shape)} module_grad={initial:.3e} non_readout_grad={live:.3e}',flush=True)
    os.environ.pop(ENV,None)


def iter_runs(out_dir:Path):
    paper=out_dir/'paper'
    for meta in sorted(paper.glob('*'+'/h*'+'/*'+'/metadata.json')):
        if meta.parent.name not in VARIANTS:continue
        try:
            data=json.loads(meta.read_text(encoding='utf-8'))
            if data.get('variant') in VARIANTS:yield data,meta
        except (ValueError,OSError):print('[WARN] invalid metadata:',meta,flush=True)


def find_baseline(row:dict, paths:list[Path]):
    for p in paths:
        m=p/'paper'/row['dataset']/f"h{row['horizon']}"/'original'/'metadata.json'
        if not m.is_file():continue
        try:d=json.loads(m.read_text(encoding='utf-8'))
        except (ValueError,OSError):continue
        if (d.get('status')=='ok' and row.get('status')=='ok'
            and d.get('dataset')==row['dataset'] and int(d.get('horizon',0))==int(row['horizon'])
            and d.get('profile')==row.get('profile') and int(d.get('lookback',0))==int(row['lookback'])
            and d.get('source_commit')==row.get('source_commit')
            and d.get('training_epochs')==row.get('training_epochs')
            and int(d.get('seed',2024))==int(row.get('seed',2024))):
            if isinstance(d.get('mse'),(int,float)) and isinstance(d.get('mae'),(int,float)):
                return d,m
    return {},None


def collect_results(out_dir:Path,baseline_dirs:list[Path]):
    fields=['model','variant','dataset','horizon','lookback','profile','seed','status','returncode','mse','mae',
            'mse_gain_pct_vs_original','mae_gain_pct_vs_original','baseline_metadata',
            'wall_seconds','gpu_total_used_peak_mib','added_params','source_commit','patched_source_sha256','training_epochs','source_inspiration']
    rows=[]
    for item,source in iter_runs(out_dir):
        ref,refpath=find_baseline(item,baseline_dirs)
        item['baseline_metadata']=str(refpath) if refpath else ''
        for metric in ('mse','mae'):
            b=ref.get(metric);v=item.get(metric)
            item[f'{metric}_gain_pct_vs_original']=(round(100*(b-v)/b,6)
                 if isinstance(b,(int,float)) and b>0 and isinstance(v,(int,float)) else '')
        rows.append(item)
    out_dir.mkdir(parents=True,exist_ok=True)
    result=out_dir/'v6_all_summary.csv'
    tmp=out_dir/'v6_all_summary.csv.tmp'
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader();w.writerows(rows)
    tmp.replace(result)
    print(f'[COLLECT] {len(rows)} rows, {sum(r.get("status")=="ok" for r in rows)} successful -> {result}',flush=True)
    return result


def parse_args(argv=None):
    ap=argparse.ArgumentParser(description='Paloma v6 structural information path tests')
    ap.add_argument('--mode',choices=['all','smoke','full','collect'],default='all')
    ap.add_argument('--datasets',default='ETTh1,ETTm1')
    ap.add_argument('--horizons',default='96,336')
    ap.add_argument('--variants',default=','.join(VARIANTS))
    ap.add_argument('--profile',choices=['paper'],default='paper')
    ap.add_argument('--data-dir',default=os.environ.get('DATA_DIR','/root/autodl-tmp/GTR_ResearchWorkbench_Modular/dataset'))
    ap.add_argument('--sources-dir',default=os.environ.get('SOURCES_DIR',str(ROOT/'sources')))
    ap.add_argument('--out-dir',default=str(ROOT/'runs'/'paloma_information_routes_v6'))
    ap.add_argument('--baseline-dir',default=str(ROOT/'runs'/'paloma_functional_splice'))
    ap.add_argument('--gpu',type=int,default=0)
    ap.add_argument('--epochs',type=int,default=0)
    ap.add_argument('--force',action='store_true')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--continue-on-error',action='store_true')
    a=ap.parse_args(argv)
    try:
        a.datasets=split_names(a.datasets,{'ETTh1':1,'ETTm1':1})
        a.variants=split_names(a.variants,{k:1 for k in VARIANTS})
        a.horizons=[int(v) for v in a.horizons.split(',')]
    except ValueError as e:ap.error(str(e))
    if not a.horizons or any(h not in (96,336) for h in a.horizons):ap.error('Only horizons 96 and 336 are supported')
    if a.epochs<0:ap.error('--epochs must be >= 0')
    return a


def main(argv=None):
    a=parse_args(argv)
    out=Path(a.out_dir).expanduser().resolve()
    upstream=Path(a.sources_dir).expanduser().resolve()/'Paloma'
    private=out/'private_source'/'Paloma'
    base_dirs=[out,Path(a.baseline_dir).expanduser().resolve(),
        ROOT/'runs'/'paloma_replacements_v5',ROOT/'runs'/'paloma_context_fusion_v4',ROOT/'runs'/'paloma_phase_replacement']
    if a.mode=='collect':
        collect_results(out,base_dirs);return
    if a.dry_run:
        for ds in a.datasets:
            for h in a.horizons:
                for v in a.variants:
                    dest=out/'paper'/ds/f'h{h}'/v
                    cmd,_=build_params('Paloma',ds,h,'paper',a.epochs,a.gpu,Path(a.data_dir)/f'{ds}.csv',dest)
                    cmd[cmd.index('--model_id')+1]+='_'+v+'_v6'
                    print('[DRY]',ds,h,v,' '.join(cmd),flush=True)
        return
    manifest=prepare_private_source(upstream,private)
    files={}
    for ds in a.datasets:
        f=find_data(a.data_dir,ds)
        validate_data(f,ds)
        files[ds]=f
    if a.mode in ('all','smoke'):
        for ds in a.datasets:
            for h in a.horizons:
                smoke(upstream,private,ds,h,a.variants,files[ds])
        print('[SMOKE] all checks complete, no paper metrics computed',flush=True)
    if a.mode=='smoke':return
    if not get_cuda():
        raise RuntimeError('CUDA required for official training script checkpoint reload. Use --mode smoke on CPU.')
    total=len(a.datasets)*len(a.horizons)*len(a.variants)
    i=0;failed=[]
    for ds in a.datasets:
        for h in a.horizons:
            for v in a.variants:
                i+=1
                dest=out/'paper'/ds/f'h{h}'/v
                old=dest/'metadata.json'
                cmd,meta=build_params('Paloma',ds,h,'paper',a.epochs,a.gpu,files[ds],dest)
                cmd[cmd.index('--model_id')+1]+='_'+v+'_v6'
                meta.update(variant=v,seed=2024,source_inspiration=PROVENANCE[v],
                            added_params=ADDED_PARAMS[v][h],patched_source_sha256=manifest['patched_sha256'])
                if old.is_file() and not a.force:
                    try:prior=json.loads(old.read_text(encoding='utf-8'))
                    except (ValueError,OSError):prior={}
                    if (prior.get('status')=='ok' and prior.get('training_epochs')==meta['training_epochs']
                        and prior.get('patched_source_sha256')==meta['patched_source_sha256']
                        and prior.get('source_commit')==meta['source_commit']
                        and prior.get('seed')==meta['seed']):
                        print(f'[SKIP {i}/{total}] {ds} H{h} {v}',flush=True)
                        continue
                print(f'[JOB {i}/{total}] {ds} H{h} {v}',flush=True)
                previous=os.environ.get(ENV)
                os.environ[ENV]=v
                try:rec=run_job(cmd,private,dest,meta,a)
                finally:
                    if previous is None:os.environ.pop(ENV,None)
                    else:os.environ[ENV]=previous
                collect_results(out,base_dirs)
                if rec['status']!='ok':
                    failed.append(f'{ds} H{h} {v}')
                    if not a.continue_on_error:
                        raise RuntimeError(f'Failed: {ds} H{h} {v}; inspect {dest/"train.log"}')
    collect_results(out,base_dirs)
    if failed:raise RuntimeError('Failed jobs: '+', '.join(failed))
    print('[DONE] complete',flush=True)

if __name__=='__main__':
    main()
