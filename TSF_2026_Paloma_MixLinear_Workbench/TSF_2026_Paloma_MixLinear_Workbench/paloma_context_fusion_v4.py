#!/usr/bin/env python3
"""Context/fusion targeted functional replacement experiments on Paloma (2026).

The official source is cloned as a private, hash-checked copy. Only the copy
is patched. Do not infer paper accuracy from smoke tests.
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
VERSION = 4
VARIANTS = ("original", "olinear_residual", "olinear_hidden", "lagged_residual", "adaptive_fusion")
VARIANT_ENV = "PALOMA_CONTEXT_FUSION_V4"

# Mechanism-specific append-only patch, applied to a byte-identical private copy
# of the pinned official model. No third-party OLinear or MixLinear source is copied.
PATCH = r'''

# PALOMA_CONTEXT_FUSION_V4
import os as _cf_os
import torch as _cf_torch
import torch.nn as _cf_nn
import torch.nn.functional as _cf_F

class _OLinearChannel(_cf_nn.Module):
    """OLinear LinearEncoder-inspired global CxC mixing with value projection.

    Independent implementation; not copied OLinear code. Assumes [B,C,F].
    """
    def __init__(self, width, channels):
        super().__init__()
        self.logits = _cf_nn.Parameter(4.0 * _cf_torch.eye(channels) - 3.0)
        self.value_proj = _cf_nn.Linear(width, width)

    def forward(self, x):
        w = _cf_F.softplus(self.logits)
        w = w / w.sum(-1,keepdim=True).clamp_min(1e-8)
        return _cf_torch.einsum('ij,bjf->bif',w,self.value_proj(x))


class _LaggedResidualMixer(_cf_nn.Module):
    """Independent causal cross-channel multi-lag mixing; current/history only.

    5 lag taps (0,1,2,4,8). Softmax normalizes across sources and lags
    per target channel, with no access to prediction-horizon observations.
    """
    def __init__(self, channels, lags=(0,1,2,4,8)):
        super().__init__()
        self.lags=tuple(lags)
        init=_cf_torch.full((len(lags),channels,channels),-5.0)
        init[0]=_cf_torch.eye(channels)*5.0-5.0
        self.logits=_cf_nn.Parameter(init)

    def forward(self, x):
        B,C,L=x.shape
        weights=_cf_torch.softmax(self.logits.permute(1,0,2).reshape(C,-1),dim=-1)
        weights=weights.reshape(C,len(self.lags),C).permute(1,0,2)
        output=_cf_torch.zeros_like(x)
        for k,lag in enumerate(self.lags):
            # Non-circular padding: at t, use x[t-lag] if observed, else zero.
            delayed=x if lag==0 else _cf_F.pad(x,(lag,0))[...,:L]
            output=output+_cf_torch.einsum('ij,bjl->bil',weights[k],delayed)
        return output


class _AdaptiveBranchFusion(_cf_nn.Module):
    """Sample-conditioned phase-vs-amplitude fusion, preserving sum at init.

    Two-input gate is a low-rank MLP whose last layer is zero-initialized;
    phase+amplitude is recovered bitwise on construction.
    """
    def __init__(self, d_model, bottleneck=24):
        super().__init__()
        self.gate_net=_cf_nn.Sequential(
            _cf_nn.Linear(2*d_model,bottleneck),
            _cf_nn.GELU(),
            _cf_nn.Linear(bottleneck,d_model))
        _cf_nn.init.zeros_(self.gate_net[-1].weight)
        _cf_nn.init.zeros_(self.gate_net[-1].bias)

    def forward(self,p,a):
        t=self.gate_net(_cf_torch.cat((p,a),dim=-1))
        # g=0 at initialization => recover p+a exactly
        g=2.0*_cf_torch.sigmoid(t)-1.0
        return (p+a) + g*(p-a)

_SourceInitContextV4=Model.__init__
_SourceForwardContextV4=Model.forward

def _context_init_v4(self, configs):
    _SourceInitContextV4(self,configs)
    mode=_cf_os.environ.get('PALOMA_CONTEXT_FUSION_V4','original')
    allowed=('original','olinear_residual','olinear_hidden','lagged_residual','adaptive_fusion')
    if mode not in allowed: raise ValueError('Unrecognized v4 variant '+str(mode))
    self.context_variant=mode
    if mode=='original': return
    cpu_state=_cf_torch.get_rng_state()
    cuda_state=_cf_torch.cuda.get_rng_state_all() if _cf_torch.cuda.is_initialized() else None
    try:
        if mode=='olinear_residual':
            self.context_module=_OLinearChannel(self.seq_len,self.enc_in)
        elif mode=='olinear_hidden':
            self.context_module=_OLinearChannel(self.d_model,self.enc_in)
        elif mode=='lagged_residual':
            self.context_module=_LaggedResidualMixer(self.enc_in)
        elif mode=='adaptive_fusion':
            self.context_module=_AdaptiveBranchFusion(self.d_model)
        # Zero init gives EXACT reference at initial forward for add-on variants.
        if mode!='adaptive_fusion':
            self.context_gate=_cf_nn.Parameter(_cf_torch.zeros(()))
    finally:
        _cf_torch.set_rng_state(cpu_state)
        if cuda_state is not None: _cf_torch.cuda.set_rng_state_all(cuda_state)


def _context_forward_v4(self, x, cycle_index):
    mode=self.context_variant
    if mode=='original': return _SourceForwardContextV4(self,x,cycle_index)

    if self.use_revin:
        seq_mean=_cf_torch.mean(x,dim=1,keepdim=True)
        seq_var=_cf_torch.var(x,dim=1,keepdim=True)+1e-5
        x=(x-seq_mean)/_cf_torch.sqrt(seq_var)
    x_input=x.permute(0,2,1)
    idx=(cycle_index.view(-1,1)+
         _cf_torch.arange(self.seq_len,device=cycle_index.device).view(1,-1))%self.cycle_len
    query_input=self.Q[idx].permute(0,2,1)
    phase=self.phase[idx].permute(0,2,1)
    amplitude=self.amplitude[idx].permute(0,2,1)
    residual=x_input-query_input

    if mode in ('olinear_residual','lagged_residual'):
        residual=residual+self.context_gate*self.context_module(residual)

    phase_impact=self.phase_fusion(residual,phase)
    amplitude_impact=self.amplitude_fusion(residual,amplitude)
    if mode=='adaptive_fusion':
        fused=self.context_module(phase_impact,amplitude_impact)
    else:
        fused=phase_impact+amplitude_impact
    input_proj=self.input_proj(x_input)
    hidden=self.model(fused)
    if mode=='olinear_hidden':
        hidden=hidden+self.context_gate*self.context_module(hidden)
    output=self.output_proj(hidden+input_proj).permute(0,2,1)
    if self.use_revin:
        output=output*_cf_torch.sqrt(seq_var)+seq_mean
    return output

Model.__init__=_context_init_v4
Model.forward=_context_forward_v4

'''


def git_blob(contents: bytes) -> str:
    return hashlib.sha1(f"blob {len(contents)}\0".encode() + contents).hexdigest()


def prepare_private_source(upstream: Path, private: Path) -> dict:
    upstream, private = upstream.resolve(), private.resolve()
    if upstream == private or upstream in private.parents:
        raise ValueError("Private source cannot be nested in upstream repository")
    src = upstream / 'models' / 'Paloma.py'
    if not src.is_file() or not (upstream / 'run.py').is_file():
        raise FileNotFoundError('Official sources missing. Run bash setup_sources.sh first.')
    official_bytes = src.read_bytes()
    observed = git_blob(official_bytes)
    if observed != OFFICIAL_BLOB:
        raise RuntimeError(f'Source Paloma.py changed (git blob {observed}); expected {OFFICIAL_BLOB}. Refuse to patch.')
    source_sha = hashlib.sha256(official_bytes).hexdigest()
    expected_patched_sha = hashlib.sha256(official_bytes + PATCH.encode('utf-8')).hexdigest()
    manifest_path = private / 'splice_manifest.json'
    if private.exists():
        if not manifest_path.exists():
            raise RuntimeError(f'Unrecognized private source directory: {private}')
        m = json.loads(manifest_path.read_text(encoding='utf-8'))
        patched = private / 'models' / 'Paloma.py'
        if m.get('version') != VERSION or m.get('source_sha256') != source_sha or not patched.is_file():
            raise RuntimeError('Private source out of date; remove runs/paloma_context_fusion_v4/private_source to rebuild')
        if hashlib.sha256(patched.read_bytes()).hexdigest() != expected_patched_sha or m.get('patched_sha256') != expected_patched_sha:
            raise RuntimeError('Private source was changed after installation')
        return m
    private.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(upstream, private, ignore=shutil.ignore_patterns(
        '.git', '__pycache__', '*.pyc', 'checkpoints', 'results', 'test_results', 'runs', 'logs'))
    patched = private / 'models' / 'Paloma.py'
    with patched.open('a', encoding='utf-8') as fp:
        fp.write(PATCH)
    m = {
        'version': VERSION,
        'source_commit': SHAS['Paloma'],
        'source_git_blob': OFFICIAL_BLOB,
        'source_sha256': source_sha,
        'patched_sha256': hashlib.sha256(patched.read_bytes()).hexdigest(),
        'variants': list(VARIANTS),
        'note': 'Independent experimental patch on private source; official repository untouched'
    }
    manifest_path.write_text(json.dumps(m, indent=2), encoding='utf-8')
    return m


def module_from_file(file: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import model: {file}')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def smoke(original_source: Path, private: Path, ds: str, h: int, variants: list[str], csv_file: Path):
    import numpy as np
    import pandas as pd
    import torch

    cfg = {
        'ETTh1': {'cycle':24, 'dropout':0.5},
        'ETTm1': {'cycle':96, 'dropout':0.4 if h == 96 else 0.5}
    }[ds]
    conf = SimpleNamespace(seq_len=96, pred_len=h, enc_in=7, cycle=cfg['cycle'],
                           dropout=cfg['dropout'], d_model=256, use_revin=1, individual=0)
    original = module_from_file(original_source/'models'/'Paloma.py', 'paloma_orig_context_v4')
    modified = module_from_file(private/'models'/'Paloma.py', 'paloma_patch_context_v4')
    data = pd.read_csv(csv_file, usecols=list(range(1,8)), nrows=96+h+8).to_numpy(np.float32)
    if len(data) < 96+h+2:
        raise ValueError(f'{csv_file}: insufficient real ETT rows for smoke')
    data = (data-data.mean(0)) / (data.std(0)+1e-6)
    X = torch.tensor(np.stack([data[i:i+96] for i in range(2)]))
    Y = torch.tensor(np.stack([data[i+96:i+96+h] for i in range(2)]))
    cycle_idx = torch.tensor([0,1],dtype=torch.long)

    torch.manual_seed(2024)
    baseline = original.Model(conf).eval()
    os.environ[VARIANT_ENV] = 'original'
    torch.manual_seed(2024)
    copy_original = modified.Model(conf).eval()
    a, b = baseline.state_dict(), copy_original.state_dict()
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    with torch.no_grad():
        y_ref = baseline(X, cycle_idx)
        assert torch.equal(y_ref, copy_original(X, cycle_idx))
    print('[CONTROL] Patched original == official model, exactly; all initial params and outputs', flush=True)
    for v in variants:
        os.environ[VARIANT_ENV] = v
        torch.manual_seed(2024)
        net = modified.Model(conf).eval()
        for k, value in a.items():
            assert torch.equal(net.state_dict()[k],value), ('initial baseline weight changed',v,k)
        with torch.no_grad():
            initial = net(X,cycle_idx)
            assert torch.equal(initial,y_ref), ('initial function changed',v, torch.max(torch.abs(initial-y_ref)).item())
        net.train()
        y = net(X,cycle_idx)
        assert tuple(y.shape) == tuple(Y.shape) and torch.isfinite(y).all()
        loss = (y-Y).square().mean()
        loss.backward()
        count = sum(p.grad is not None and torch.isfinite(p.grad).all().item() for p in net.parameters())
        assert count>0
        if v not in ('original','adaptive_fusion'):
            assert net.context_gate.grad is not None and torch.isfinite(net.context_gate.grad)
            assert net.context_gate.grad.abs()>0, 'residual gate must receive gradient'
        elif v=='adaptive_fusion':
            last=net.context_module.gate_net[-1]
            assert last.weight.grad is not None and torch.isfinite(last.weight.grad).all()
            assert last.weight.grad.abs().sum()>0, 'fusion gate must receive gradient'
        print(f'[SMOKE_OK] {v} {ds} L96 H{h} output={tuple(y.shape)} finite_grad_params={count}', flush=True)
    os.environ.pop(VARIANT_ENV, None)


def collect_results(out_dir: Path, baseline_dir: Path | None = None) -> Path:
    """Rebuild summary from disk; can recover after interruption or CSV overwrite.

    v4 original preferred; fallback to v3 original if metadata and protocol match.
    Never infer metrics for missing or failed runs.
    """
    rows=[]
    for metadata in sorted((out_dir/'paper').rglob('metadata.json')):
        if metadata.parent.name not in VARIANTS: continue
        try: rec=json.loads(metadata.read_text(encoding='utf-8'))
        except (ValueError, OSError): continue
        if rec.get('variant') in VARIANTS: rows.append(rec)
    for row in rows:
        original=out_dir/'paper'/row['dataset']/f"h{row['horizon']}"/'original'/'metadata.json'
        if not original.is_file() and baseline_dir is not None:
            candidate=baseline_dir/'paper'/row['dataset']/f"h{row['horizon']}"/'original'/'metadata.json'
            if candidate.is_file(): original=candidate
        try: ref=json.loads(original.read_text(encoding='utf-8')) if original.is_file() else {}
        except (OSError,ValueError): ref={}
        valid = ref.get('status')=='ok' and row.get('status')=='ok' and \
                ref.get('dataset')==row.get('dataset') and \
                ref.get('seed')==row.get('seed') and \
                ref.get('lookback')==row.get('lookback') and \
                ref.get('training_epochs')==row.get('training_epochs') and \
                ref.get('horizon')==row.get('horizon') and \
                ref.get('profile')==row.get('profile') and \
                ref.get('source_commit')==row.get('source_commit')
        row['baseline_metadata']=str(original) if valid else ''
        for m in ('mse','mae'):
            v=row.get(m)
            r=ref.get(m) if valid else None
            row[f'{m}_gain_pct_vs_original']=(100*(r-v)/r if isinstance(r,(int,float)) and r>0 and isinstance(v,(int,float)) else '')
    fields=['model','variant','dataset','horizon','lookback','profile','seed',
            'status','returncode','mse','mae','mse_gain_pct_vs_original',
            'mae_gain_pct_vs_original','wall_seconds','gpu_total_used_peak_mib',
            'baseline_metadata','source_commit','patched_source_sha256','training_epochs']
    path=out_dir/'context_fusion_summary.csv'
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w', newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader();w.writerows(rows)
    temp.replace(path)
    print(f'[COLLECT] records={len(rows)} path={path}',flush=True)
    return path


def parse_args(argv=None):
    ap=argparse.ArgumentParser(description='Paloma 2026 targeted context/fusion experiments')
    ap.add_argument('--mode',choices=('smoke','full','collect'),default='smoke')
    ap.add_argument('--datasets',default='ETTh1')
    ap.add_argument('--horizons',default='96')
    ap.add_argument('--variants',default=','.join(v for v in VARIANTS if v!='original'))
    ap.add_argument('--profile',choices=['paper'],default='paper')
    ap.add_argument('--data-dir',default=os.environ.get('DATA_DIR','/root/autodl-tmp/GTR_ResearchWorkbench_Modular/dataset'))
    ap.add_argument('--sources-dir',default=os.environ.get('SOURCES_DIR',str(ROOT/'sources')))
    ap.add_argument('--out-dir',default=str(ROOT/'runs'/'paloma_context_fusion_v4'))
    ap.add_argument('--epochs',type=int,default=0)
    ap.add_argument('--gpu',type=int,default=0)
    ap.add_argument('--force',action='store_true')
    ap.add_argument('--baseline-dir',default=str(ROOT/'runs'/'paloma_functional_splice'),
                    help='Read completed v3 original baselines; never retrain them by default')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--continue-on-error',action='store_true')
    a=ap.parse_args(argv)
    try:
        a.variants=split_names(a.variants,{v:1 for v in VARIANTS})
        a.datasets=split_names(a.datasets,{'ETTh1':1,'ETTm1':1})
    except ValueError as exc:
        ap.error(str(exc))
    try:
        a.horizons=[int(h.strip()) for h in a.horizons.split(',')]
    except ValueError: ap.error('Horizon must be 96 or 336')
    if not a.horizons or any(h not in [96,336] for h in a.horizons):
        ap.error('Only H=96 and H=336 are verified with official Paloma configs')
    if a.epochs<0:
        ap.error('--epochs must be nonnegative')
    return a


def main(argv=None):
    args=parse_args(argv)
    out_dir=Path(args.out_dir).expanduser().resolve()
    upstream=Path(args.sources_dir).expanduser().resolve()/'Paloma'
    private=out_dir/'private_source'/'Paloma'
    if args.mode=='collect':
        collect_results(out_dir,Path(args.baseline_dir).expanduser().resolve()); return
    if args.dry_run:
        for ds in args.datasets:
            for h in args.horizons:
                for v in args.variants:
                    dest=out_dir/'paper'/ds/f'h{h}'/v
                    fake=Path(args.data_dir)/f'{ds}.csv'
                    cmd,_=build_params('Paloma',ds,h,'paper',args.epochs,args.gpu,fake,dest)
                    cmd[cmd.index('--model_id')+1]+='_'+v+'_contextv4'
                    print('[DRY]',ds,h,v,VARIANT_ENV+'='+v, ' '.join(cmd))
        return
    manifest=prepare_private_source(upstream,private)
    if args.mode=='full' and not get_cuda():
        raise RuntimeError('Full Paloma training requires CUDA in official training script')
    for ds in args.datasets:
        data_file=find_data(args.data_dir,ds)
        validate_data(data_file,ds)
        for h in args.horizons:
            if args.mode=='smoke':
                smoke(upstream,private,ds,h,args.variants,data_file)
                continue
            for v in args.variants:
                dest=out_dir/'paper'/ds/f'h{h}'/v
                old=dest/'metadata.json'
                if old.exists() and not args.force:
                    record=json.loads(old.read_text(encoding='utf-8'))
                    if record.get('status')=='ok' and record.get('training_epochs')==(args.epochs or 30):
                        print('[SKIP]',ds,h,v, 'already complete',flush=True)
                        continue
                cmd,meta=build_params('Paloma',ds,h,'paper',args.epochs,args.gpu,data_file,dest)
                cmd[cmd.index('--model_id')+1]+='_'+v+'_contextv4'
                meta.update(variant=v,seed=2024,patched_source_sha256=manifest['patched_sha256'],
                            context_description={
                                'original':'Official Paloma source, no changes',
                                'olinear_residual':'OLinear-inspired CxC mixing on Q-subtracted residual',
                                'olinear_hidden':'OLinear-inspired CxC mixing on post-branch hidden feature',
                                'lagged_residual':'Independent causal lagged channel mixing on Q-subtracted residual',
                                'adaptive_fusion':'Sample-conditioned phase vs amplitude fusion (zero-identity init)'
                            }[v])
                previous=os.environ.get(VARIANT_ENV)
                os.environ[VARIANT_ENV]=v
                try:
                    result=run_job(cmd,private,dest,meta,args)
                finally:
                    if previous is None: os.environ.pop(VARIANT_ENV,None)
                    else: os.environ[VARIANT_ENV]=previous
                report=collect_results(out_dir,Path(args.baseline_dir).expanduser().resolve())
                print('[SUMMARY]',report,flush=True)
                if result['status']!='ok' and not args.continue_on_error:
                    raise RuntimeError(f'Failed {ds} H{h} {v}. Inspect {dest/"train.log"}')
    if args.mode=='full':
        print('[DONE]',collect_results(out_dir,Path(args.baseline_dir).expanduser().resolve()),flush=True)
    else:
        print('[SMOKE] all gradients/shapes checked; not paper reproduction',flush=True)


if __name__=='__main__':
    main()
