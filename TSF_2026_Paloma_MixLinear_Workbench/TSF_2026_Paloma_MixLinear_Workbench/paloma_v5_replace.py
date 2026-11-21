#!/usr/bin/env python3
"""2025/26 paper-mechanism adaptations for the pinned official Paloma source.

Six mutually-exclusive changes; v5 has no dependencies on older extensions.
An unmodified copy of upstream Paloma is checked by Git blob before patching.
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

OFFICIAL_BLOB = "03dc7c6d23751cde6acaca77c803117c89702006"
VERSION = "5.0"
VARIANT_ENV = "PALOMA_REPLACEMENT_V5"
VARIANTS = (
    "pca_head", "segment_head", "timebase_residual", "phaseformer_router",
    "seer_soft_replace", "t3_alignment",
)
DESCRIPTIONS = {
    "pca_head": "TStruct PCAHead-inspired channel-weighted low-rank forecast correction",
    "segment_head": "TimeMosaic-inspired segment-conditioned nonlinear forecast correction",
    "timebase_residual": "TimeBase-inspired temporal-basis residual encoder (24-step subperiod)",
    "phaseformer_router": "PhaseFormer-inspired two-stage routing across phase-position tokens",
    "seer_soft_replace": "SEER-inspired soft patch quality/replacement without reordering",
    "t3_alignment": "T3Time-inspired cross-branch alignment across temporally defined patch tokens",
}
CITATIONS = {
    "pca_head": "https://github.com/yangzeli/TStruct/blob/main/models/TStruct.py",
    "segment_head": "https://github.com/BenchCouncil/TimeMosaic/blob/main/models/Mosaic.py",
    "timebase_residual": "https://github.com/hqh0728/TimeBase/blob/main/models/TimeBase.py",
    "phaseformer_router": "https://github.com/neumyor/PhaseFormer_TSL/blob/main/models/PhaseFormer.py",
    "seer_soft_replace": "https://github.com/decisionintelligence/SEER/blob/main/ts_benchmark/baselines/seer/models/seer_model.py",
    "t3_alignment": "https://github.com/monaf-chowdhury/T3Time/blob/main/layers/Cross_Modal_Align.py",
}

# Standalone append-only modification; never overwrite the official repository.
# The functions and classes below are ORIGINAL ADAPTATIONS of mechanism-level ideas,
# not copies of the other papers' implementations.
PATCH = r'''

# BEGIN PALOMA_REPLACEMENT_V5
import os as _v5_os
import torch as _v5_torch
import torch.nn as _v5_nn
import torch.nn.functional as _v5_F

class _V5PCAHead(_v5_nn.Module):
    """TStruct PCAHead-inspired mixture of channel-specific low-rank deltas.

    Baseline output_proj remains the shared base predictor; this is a residual
    correction with K low-rank bases and channel-dependent mixture weights.
    No patch-token encoder or exact TStruct head is claimed.
    """
    def __init__(self, dim, horizon, channels, rank=4, n_bases=4):
        super().__init__()
        self.low_A = _v5_nn.Parameter(_v5_torch.randn(n_bases, dim, rank) * 0.02)
        self.low_B = _v5_nn.Parameter(_v5_torch.zeros(n_bases, rank, horizon))
        self.channel_logits = _v5_nn.Parameter(_v5_torch.zeros(channels, n_bases))

    def forward(self, latent):
        z = _v5_torch.einsum('bcd,kdr->bckr',latent,self.low_A)
        z = _v5_torch.einsum('bckr,krh->bckh',z,self.low_B)
        coef = _v5_F.softmax(self.channel_logits,dim=-1)
        return _v5_torch.einsum('bckh,ck->bch',z,coef)

class _V5SegmentHead(_v5_nn.Module):
    """TimeMosaic-inspired segment-conditioned residual forecast decoder.

    Unlike trivial splitting Linear(D,H), segment embeddings enter a nonlinear
    hidden representation, and each segment has its own correction projection.
    """
    def __init__(self, dim, horizon, segment=48, bottleneck=32):
        super().__init__()
        self.lengths = [min(segment, horizon-i) for i in range(0,horizon,segment)]
        self.common = _v5_nn.Linear(dim,bottleneck)
        self.segment_embed = _v5_nn.Parameter(_v5_torch.randn(len(self.lengths),bottleneck)*0.02)
        self.heads = _v5_nn.ModuleList([_v5_nn.Linear(bottleneck,n) for n in self.lengths])
        for layer in self.heads:
            _v5_nn.init.zeros_(layer.weight)
            _v5_nn.init.zeros_(layer.bias)

    def forward(self,latent):
        h=self.common(latent)
        return _v5_torch.cat([
            head(_v5_F.gelu(h+self.segment_embed[i].view(1,1,-1)))
            for i,head in enumerate(self.heads)
        ],dim=-1)

class _V5TemporalBasis(_v5_nn.Module):
    """TimeBase-inspired low-rank projection over cycle repetitions.

    A 24-step SUBPERIOD is used for both ETT hour and minute series, leaving
    Paloma's author-provided cycle 24/96 unchanged. L=96 -> 4 repeats.
    """
    def __init__(self, length, subperiod=24, rank=2):
        super().__init__()
        if length%subperiod:
            raise ValueError(f'Length {length} not divisible by subperiod {subperiod}')
        self.length=length
        self.period=subperiod
        self.repeats=length//subperiod
        if self.repeats < 2:
            raise ValueError('Need at least two subperiod repetitions')
        self.to_basis=_v5_nn.Linear(self.repeats,rank,bias=False)
        self.from_basis=_v5_nn.Linear(rank,self.repeats,bias=False)
        _v5_nn.init.zeros_(self.from_basis.weight)

    def forward(self,residual):
        B,C,L=residual.shape
        x=residual.reshape(B,C,self.repeats,self.period).transpose(-1,-2)
        centered=x-x.mean(dim=-1,keepdim=True)
        delta=self.from_basis(self.to_basis(centered))
        return delta.transpose(-1,-2).reshape(B,C,L)

class _V5PhaseRouter(_v5_nn.Module):
    """PhaseFormer-inspired collect/distribute attention on ordered phase tokens.

    [B,C,L] -> [B*C,P,N] with P=24 phase positions and N=4 repeats.
    Tokens retain their phase-position order; no temporal reordering.
    """
    def __init__(self, length, subperiod=24, width=32, routers=4):
        super().__init__()
        if length%subperiod:
            raise ValueError('Phase routing needs divisible length')
        self.period=subperiod
        self.repeats=length//subperiod
        self.embedding=_v5_nn.Linear(self.repeats,width)
        self.routers=_v5_nn.Parameter(_v5_torch.randn(1,routers,width)*0.02)
        self.sender=_v5_nn.MultiheadAttention(width,4,batch_first=True)
        self.receiver=_v5_nn.MultiheadAttention(width,4,batch_first=True)
        self.proj=_v5_nn.Linear(width,self.repeats,bias=False)
        _v5_nn.init.zeros_(self.proj.weight)

    def forward(self,residual):
        B,C,L=residual.shape
        phase=residual.reshape(B*C,self.repeats,self.period).transpose(1,2)
        tokens=self.embedding(phase)
        routers=self.routers.expand(B*C,-1,-1)
        buffer,_=self.sender(routers,tokens,tokens,need_weights=False)
        rec,_=self.receiver(tokens,buffer,buffer,need_weights=False)
        return self.proj(rec).transpose(1,2).reshape(B,C,L)

class _V5SEERSoftReplacement(_v5_nn.Module):
    """SEER-inspired scoring & global reconstruction of ordered history patches.

    Soft differentiable adaptation, NOT SEER's exact thresholded replacement.
    Patches keep their original positions; target horizon never used.
    """
    def __init__(self,length,patch=16):
        super().__init__()
        if length%patch: raise ValueError('Need divisible sequence length for soft patch replacement')
        self.patch=patch
        self.scorer=_v5_nn.Sequential(_v5_nn.Linear(patch,16),_v5_nn.GELU(),_v5_nn.Linear(16,1))
        self.gate=_v5_nn.Parameter(_v5_torch.zeros(()))

    def forward(self,residual):
        B,C,L=residual.shape
        patches=residual.reshape(B,C,L//self.patch,self.patch)
        quality=_v5_torch.sigmoid(self.scorer(patches))
        global_patch=patches.mean(dim=2,keepdim=True)
        replacement=quality*patches+(1-quality)*global_patch
        return self.gate*(replacement-patches).reshape(B,C,L)

class _V5T3Alignment(_v5_nn.Module):
    """T3Time-inspired cross-branch alignment on semantic temporal patch tokens.

    Each phase/amplitude conditioning sequence uses 24-time-step blocks.
    Attention is over these ordered blocks, not arbitrary latent coordinates.
    """
    def __init__(self,length,latent,subperiod=24,token_dim=32):
        super().__init__()
        if length%subperiod: raise ValueError('Cross-branch patches require divisible history')
        self.patch=subperiod
        self.n=length//subperiod
        self.p_proj=_v5_nn.Linear(subperiod,token_dim)
        self.a_proj=_v5_nn.Linear(subperiod,token_dim)
        self.attn=_v5_nn.MultiheadAttention(token_dim,4,batch_first=True)
        self.output=_v5_nn.Linear(self.n*token_dim,latent)
        _v5_nn.init.zeros_(self.output.weight)
        _v5_nn.init.zeros_(self.output.bias)

    def forward(self,residual,phase,amplitude):
        B,C,L=residual.shape
        pt=(residual+phase).reshape(B*C,self.n,self.patch)
        at=(residual+amplitude).reshape(B*C,self.n,self.patch)
        pt=self.p_proj(pt)
        at=self.a_proj(at)
        aligned,_=self.attn(pt,at,at,need_weights=False)
        return self.output(aligned.reshape(B,C,-1))

_V5OriginalInit=Model.__init__
_V5OriginalForward=Model.forward

def _v5_init(self,configs):
    _V5OriginalInit(self,configs)
    name=_v5_os.environ.get('PALOMA_REPLACEMENT_V5','original')
    allowed=('original','pca_head','segment_head','timebase_residual',
             'phaseformer_router','seer_soft_replace','t3_alignment')
    if name not in allowed: raise ValueError('Invalid v5 variant: '+name)
    self.v5_variant=name
    if name=='original': return
    cpu_rng=_v5_torch.get_rng_state()
    cuda_rng=_v5_torch.cuda.get_rng_state_all() if _v5_torch.cuda.is_initialized() else None
    try:
        if name=='pca_head':
            self.v5_module=_V5PCAHead(self.d_model,self.pred_len,self.enc_in)
        elif name=='segment_head':
            self.v5_module=_V5SegmentHead(self.d_model,self.pred_len)
        elif name=='timebase_residual':
            self.v5_module=_V5TemporalBasis(self.seq_len)
        elif name=='phaseformer_router':
            self.v5_module=_V5PhaseRouter(self.seq_len)
        elif name=='seer_soft_replace':
            self.v5_module=_V5SEERSoftReplacement(self.seq_len)
        elif name=='t3_alignment':
            self.v5_module=_V5T3Alignment(self.seq_len,self.d_model)
    finally:
        _v5_torch.set_rng_state(cpu_rng)
        if cuda_rng is not None: _v5_torch.cuda.set_rng_state_all(cuda_rng)

def _v5_forward(self,x,cycle_index):
    v=self.v5_variant
    if v=='original': return _V5OriginalForward(self,x,cycle_index)
    if self.use_revin:
        seq_mean=_v5_torch.mean(x,dim=1,keepdim=True)
        seq_var=_v5_torch.var(x,dim=1,keepdim=True)+1e-5
        x=(x-seq_mean)/_v5_torch.sqrt(seq_var)
    x_input=x.permute(0,2,1)
    idx=(cycle_index.view(-1,1)+_v5_torch.arange(self.seq_len,device=cycle_index.device).view(1,-1))%self.cycle_len
    Q=self.Q[idx].permute(0,2,1)
    phase=self.phase[idx].permute(0,2,1)
    amplitude=self.amplitude[idx].permute(0,2,1)
    residual=x_input-Q
    if v in ('timebase_residual','phaseformer_router','seer_soft_replace'):
        residual=residual+self.v5_module(residual)
    p=self.phase_fusion(residual,phase)
    a=self.amplitude_fusion(residual,amplitude)
    merged=p+a
    if v=='t3_alignment':
        merged=merged+self.v5_module(residual,phase,amplitude)
    input_proj=self.input_proj(x_input)
    hidden=self.model(merged)
    features=hidden+input_proj
    output=self.output_proj(features)
    if v in ('pca_head','segment_head'):
        output=output+self.v5_module(features)
    output=output.permute(0,2,1)
    if self.use_revin:
        output=output*_v5_torch.sqrt(seq_var)+seq_mean
    return output

Model.__init__=_v5_init
Model.forward=_v5_forward
# END PALOMA_REPLACEMENT_V5
'''


def git_blob(contents: bytes) -> str:
    return hashlib.sha1(f'blob {len(contents)}\0'.encode() + contents).hexdigest()


def prepare_private_source(upstream: Path, private: Path) -> dict:
    upstream, private = upstream.resolve(), private.resolve()
    if upstream == private or upstream in private.parents:
        raise ValueError('The private repository must be separate from official source')
    original = upstream/'models'/'Paloma.py'
    if not original.is_file() or not (upstream/'run.py').is_file():
        raise FileNotFoundError('Official Paloma not found. Run bash setup_sources.sh once.')
    contents=original.read_bytes()
    observed=git_blob(contents)
    if observed != OFFICIAL_BLOB:
        raise RuntimeError(f'Unexpected Paloma.py Git blob: {observed}; expected {OFFICIAL_BLOB}. No patch applied.')
    digest=hashlib.sha256(contents+PATCH.encode('utf-8')).hexdigest()
    manifest_path=private/'splice_manifest.json'
    if private.exists():
        if not manifest_path.is_file():
            raise RuntimeError(f'Private source exists without manifest; inspect {private}')
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        path=private/'models'/'Paloma.py'
        if not path.exists() or manifest.get('version')!=VERSION or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Private v5 source failed checksum: {path}. Do not overwrite without inspection.')
        return manifest
    private.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(upstream,private,ignore=shutil.ignore_patterns(
        '.git','__pycache__','*.pyc','checkpoints','results','test_results','runs','logs'))
    with (private/'models'/'Paloma.py').open('a',encoding='utf-8') as f:
        f.write(PATCH)
    manifest={
        'version': VERSION, 'upstream_commit': SHAS['Paloma'], 'official_blob': OFFICIAL_BLOB,
        'official_sha256': hashlib.sha256(contents).hexdigest(), 'patched_sha256': digest,
        'variants': list(VARIANTS), 'source_attribution': CITATIONS,
        'note': 'Independent mechanism-level adaptations; the upstream repository is not changed',
    }
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    return manifest


def load_module(path: Path, module_name: str):
    spec=importlib.util.spec_from_file_location(module_name,path)
    if spec is None or spec.loader is None: raise RuntimeError(f'Cannot import {path}')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def smoke(upstream: Path, private: Path, ds: str, horizon: int, variants: list[str], data_file: Path|None):
    import numpy as np
    import torch
    cycle=24 if ds=='ETTh1' else 96
    dropout=0.5 if ds=='ETTh1' or horizon==336 else 0.4
    conf=SimpleNamespace(seq_len=96,pred_len=horizon,enc_in=7,cycle=cycle,d_model=256,
                         dropout=dropout,use_revin=1,individual=0)
    source=load_module(upstream/'models'/'Paloma.py','paloma_official_v5')
    changed=load_module(private/'models'/'Paloma.py','paloma_changed_v5')
    if data_file is not None:
        import pandas as pd
        df=pd.read_csv(data_file,usecols=list(range(1,8)),nrows=96+horizon+8)
        arr=df.to_numpy(dtype=np.float32)
        if len(arr)<96+horizon+2: raise ValueError('Insufficient rows for smoke')
        arr=(arr-arr.mean(axis=0))/(arr.std(axis=0)+1e-5)
        x=torch.tensor(np.stack((arr[0:96],arr[1:97])))
        y=torch.tensor(np.stack((arr[96:96+horizon],arr[97:97+horizon])))
    else:
        torch.manual_seed(2020)
        x=torch.randn(2,96,7)
        y=torch.randn(2,horizon,7)
    idx=torch.tensor([0,1],dtype=torch.long)
    torch.manual_seed(2024)
    baseline=source.Model(conf).eval()
    original_state=baseline.state_dict()
    os.environ[VARIANT_ENV]='original'
    torch.manual_seed(2024)
    copy=changed.Model(conf).eval()
    assert original_state.keys()==copy.state_dict().keys()
    for key,val in original_state.items():
        assert torch.equal(val,copy.state_dict()[key]),key
    with torch.no_grad():
        control=baseline(x,idx)
        assert torch.equal(control,copy(x,idx))
    print(f'[CONTROL] Original Paloma equals pinned source for {ds} H{horizon}',flush=True)
    for variant in variants:
        os.environ[VARIANT_ENV]=variant
        torch.manual_seed(2024)
        model=changed.Model(conf).eval()
        state=model.state_dict()
        assert all(key in state and torch.equal(val,state[key]) for key,val in original_state.items()),variant
        with torch.no_grad():
            yp=model(x,idx)
            assert torch.equal(control,yp),(variant, (control-yp).abs().max().item())
        model.train()
        output=model(x,idx)
        assert output.shape==(2,horizon,7), (variant,output.shape)
        assert torch.isfinite(output).all(),variant
        criterion=(output-y).square().mean()
        criterion.backward()
        valid=sum(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
        assert valid>0
        # At least one newly introduced parameter must be trainable immediately
        new_grad=sum(float(p.grad.abs().sum()) for p in model.v5_module.parameters() if p.grad is not None)
        assert new_grad>0,(variant,new_grad)
        # Three steps to ensure the zero-initialized adapter is not permanently inactive.
        optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
        for _ in range(3):
            optimizer.zero_grad()
            loss=(model(x,idx)-y).square().mean()
            loss.backward()
            optimizer.step()
        trainable=sum(float(p.grad.abs().sum()) for p in model.v5_module.parameters() if p.grad is not None)
        assert trainable>0
        print(f'[SMOKE_OK] {variant} {ds} H{horizon} shape={tuple(output.shape)} '
              f'module_grad={new_grad:.3e} post_update_grad={trainable:.3e}',flush=True)
    os.environ.pop(VARIANT_ENV,None)


def _metadata_records(out_dir:Path):
    for meta in sorted((out_dir/'paper').rglob('metadata.json')):
        if meta.parent.name not in VARIANTS: continue
        try:
            obj=json.loads(meta.read_text(encoding='utf-8'))
            if obj.get('variant') not in VARIANTS: continue
            yield obj,meta
        except (ValueError,OSError):
            print('[WARN] Invalid metadata:',meta,flush=True)


def _baseline_for(row:dict, paths:list[Path]):
    for root in paths:
        meta=root/'paper'/row['dataset']/f"h{row['horizon']}"/'original'/'metadata.json'
        if not meta.is_file(): continue
        try: ref=json.loads(meta.read_text(encoding='utf-8'))
        except (ValueError,OSError):continue
        matches=(ref.get('status')=='ok' and row.get('status')=='ok' and
                 ref.get('dataset')==row.get('dataset') and ref.get('horizon')==row.get('horizon') and
                 ref.get('lookback')==row.get('lookback') and ref.get('profile')==row.get('profile') and
                 ref.get('source_commit')==row.get('source_commit') and
                 ref.get('training_epochs')==row.get('training_epochs') and
                 ref.get('seed',2024)==row.get('seed',2024))
        if matches and all(isinstance(ref.get(k),(int,float)) for k in ('mse','mae')):
            return ref,meta
    return {},None


def collect_results(out_dir:Path, baselines:list[Path]):
    fields=['model','variant','dataset','horizon','lookback','profile','seed','status','returncode',
            'mse','mae','mse_gain_pct_vs_original','mae_gain_pct_vs_original',
            'baseline_metadata','wall_seconds','gpu_total_used_peak_mib','source_commit',
            'patched_source_sha256','training_epochs','mechanism','origin_url']
    rows=[]
    for obj,meta in _metadata_records(out_dir):
        base,path=_baseline_for(obj,baselines)
        obj['baseline_metadata']=str(path) if path else ''
        for metric in ('mse','mae'):
            b=base.get(metric)
            value=obj.get(metric)
            obj[f'{metric}_gain_pct_vs_original']=round(100*(b-value)/b,6) if (
                isinstance(b,(int,float)) and b>0 and isinstance(value,(int,float))) else ''
        obj['mechanism']=DESCRIPTIONS[obj['variant']]
        obj['origin_url']=CITATIONS[obj['variant']]
        rows.append(obj)
    # Always restore from all stored jobs; no "last invocation only" summaries.
    out_dir.mkdir(parents=True,exist_ok=True)
    file=out_dir/'v5_all_summary.csv'
    temp=out_dir/'v5_all_summary.csv.tmp'
    with temp.open('w',encoding='utf-8-sig',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(file)
    ok=sum(r.get('status')=='ok' for r in rows)
    print(f'[COLLECT] {len(rows)} entries, {ok} successful -> {file}',flush=True)
    return file


def parse_args(argv=None):
    ap=argparse.ArgumentParser(description='Paloma v5 one-command independent module-splice study')
    ap.add_argument('--mode',choices=['all','smoke','full','collect'],default='all')
    ap.add_argument('--datasets',default='ETTh1,ETTm1')
    ap.add_argument('--horizons',default='96,336')
    ap.add_argument('--variants',default=','.join(VARIANTS))
    ap.add_argument('--profile',choices=['paper'],default='paper')
    ap.add_argument('--data-dir',default=os.environ.get('DATA_DIR','/root/autodl-tmp/GTR_ResearchWorkbench_Modular/dataset'))
    ap.add_argument('--sources-dir',default=os.environ.get('SOURCES_DIR',str(ROOT/'sources')))
    ap.add_argument('--out-dir',default=str(ROOT/'runs'/'paloma_replacements_v5'))
    ap.add_argument('--baseline-dir',default=str(ROOT/'runs'/'paloma_functional_splice'))
    ap.add_argument('--gpu',type=int,default=0)
    ap.add_argument('--epochs',type=int,default=0,help='0 = original paper epoch count')
    ap.add_argument('--force',action='store_true',help='explicitly rerun completed runs')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--continue-on-error',action='store_true')
    args=ap.parse_args(argv)
    try:
        args.datasets=split_names(args.datasets,{'ETTh1':1,'ETTm1':1})
        args.variants=split_names(args.variants,{v:1 for v in VARIANTS})
    except ValueError as e: ap.error(str(e))
    try: args.horizons=[int(t) for t in args.horizons.split(',')]
    except ValueError:ap.error('Horizon must be comma-separated integers')
    if not args.horizons or any(h not in (96,336) for h in args.horizons):
        ap.error('Horizon limited to 96,336, verified official paper settings')
    if args.epochs<0: ap.error('--epochs must be >=0')
    return args


def main(argv=None):
    a=parse_args(argv)
    out=Path(a.out_dir).expanduser().resolve()
    upstream=Path(a.sources_dir).expanduser().resolve()/'Paloma'
    private=out/'private_source'/'Paloma'
    baseline_dirs=[out,Path(a.baseline_dir).expanduser().resolve(),
                   ROOT/'runs'/'paloma_context_fusion_v4',ROOT/'runs'/'paloma_phase_replacement']
    if a.mode=='collect':
        collect_results(out,baseline_dirs)
        return
    if a.dry_run:
        for ds in a.datasets:
            for h in a.horizons:
                for variant in a.variants:
                    dest=out/'paper'/ds/f'h{h}'/variant
                    cmd,_=build_params('Paloma',ds,h,'paper',a.epochs,a.gpu,Path(a.data_dir)/f'{ds}.csv',dest)
                    cmd[cmd.index('--model_id')+1]+='_'+variant+'_v5'
                    print('[DRY]',ds,h,variant,' '.join(cmd),flush=True)
        return
    m=prepare_private_source(upstream,private)
    # Important: smoke first checks exact equality to official model, using real ETT data.
    data_files={}
    for ds in a.datasets:
        p=find_data(a.data_dir,ds)
        validate_data(p,ds)
        data_files[ds]=p
    if a.mode in ('all','smoke'):
        for ds in a.datasets:
            for h in a.horizons:
                smoke(upstream,private,ds,h,a.variants,data_files[ds])
        print('[SMOKE] Completed; these are not paper metrics',flush=True)
    if a.mode=='smoke':return
    if not get_cuda():
        raise RuntimeError('Full training requires CUDA because official run.py reloads its checkpoint on GPU')
    total=len(a.datasets)*len(a.horizons)*len(a.variants)
    n=0
    failed=[]
    for ds in a.datasets:
        for h in a.horizons:
            for variant in a.variants:
                n+=1
                dest=out/'paper'/ds/f'h{h}'/variant
                old=dest/'metadata.json'
                cmd,meta=build_params('Paloma',ds,h,'paper',a.epochs,a.gpu,data_files[ds],dest)
                cmd[cmd.index('--model_id')+1]+='_'+variant+'_v5'
                meta.update(variant=variant,seed=2024,patched_source_sha256=m['patched_sha256'],
                            mechanism=DESCRIPTIONS[variant],origin_url=CITATIONS[variant])
                if old.is_file() and not a.force:
                    try: previous=json.loads(old.read_text(encoding='utf-8'))
                    except (ValueError,OSError):previous={}
                    if previous.get('status')=='ok' and previous.get('training_epochs')==meta['training_epochs'] and \
                       previous.get('patched_source_sha256')==m['patched_sha256'] and \
                       previous.get('source_commit')==meta['source_commit'] and \
                       previous.get('seed')==2024:
                        print(f'[SKIP {n}/{total}] {ds} H{h} {variant} already complete',flush=True)
                        continue
                print(f'[JOB {n}/{total}] {ds} H{h} {variant}',flush=True)
                old_mode=os.environ.get(VARIANT_ENV)
                os.environ[VARIANT_ENV]=variant
                try:
                    record=run_job(cmd,private,dest,meta,a)
                finally:
                    if old_mode is None:os.environ.pop(VARIANT_ENV,None)
                    else:os.environ[VARIANT_ENV]=old_mode
                collect_results(out,baseline_dirs)
                if record['status']!='ok':
                    failed.append(f'{ds} H{h} {variant}')
                    if not a.continue_on_error:
                        raise RuntimeError(f'Job failed: {ds} H{h} {variant}; inspect {dest/"train.log"}')
    print('[DONE]',collect_results(out,baseline_dirs),flush=True)
    if failed:
        raise RuntimeError(f'{len(failed)} jobs failed: '+', '.join(failed)+'. Other jobs were processed; inspect their individual logs.')


if __name__=='__main__':
    main()
