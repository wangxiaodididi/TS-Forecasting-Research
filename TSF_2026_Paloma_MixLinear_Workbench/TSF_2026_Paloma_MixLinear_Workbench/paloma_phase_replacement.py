#!/usr/bin/env python3
"""Targeted Paloma phase-rotation replacement experiments (NeurIPS 2026).

Add-on to TSF_2026_Paloma_MixLinear_Workbench. Never mutates the official repository.
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
from experiment import SHAS, build_params, find_data, validate_data, run_job, get_cuda, split_names

BLOB = '03dc7c6d23751cde6acaca77c803117c89702006'
PATCH_VERSION = 2
MARKER = '# PALOMA_PHASE_REPLACEMENT_V2'
VARIANTS = ('original', 'bounded_phase', 'content_phase')

APPEND_PATCH = r'''

# PALOMA_PHASE_REPLACEMENT_V2
import os as _phase_os
import torch as _phase_torch
import torch.nn as _phase_nn

_PalomaSourceModelInit = Model.__init__
_PalomaSourcePhaseForward = PhaseModulation.forward

def _paloma_phase_model_init(self, configs):
    _PalomaSourceModelInit(self, configs)
    mode = _phase_os.environ.get('PALOMA_PHASE_REPLACEMENT', 'original')
    if mode not in ('original', 'bounded_phase', 'content_phase'):
        raise ValueError('Unknown PALOMA_PHASE_REPLACEMENT: ' + repr(mode))
    if mode == 'content_phase':
        # Append after all original layers are initialized. Save RNG state to preserve
        # all original weights and the RNG stream used by the original model.
        _cpu_rng = _phase_torch.get_rng_state()
        extra = _phase_nn.Linear(self.seq_len, self.d_model, bias=False)
        _phase_nn.init.zeros_(extra.weight)
        self.phase_fusion.content_angle = extra
        _phase_torch.set_rng_state(_cpu_rng)


def _paloma_phase_forward(self, residual, phase):
    mode = _phase_os.environ.get('PALOMA_PHASE_REPLACEMENT', 'original')
    if mode == 'original':
        return _PalomaSourcePhaseForward(self, residual, phase)

    # The source implementation's branches, orthogonalization, projection,
    # normalization, and shortcut are preserved; only theta is changed.
    h_cos = self.cos_branch(residual)
    h_sin = self.sin_branch(residual)
    static_logit = self.phase_to_angle(phase)
    if mode == 'bounded_phase':
        # Limit max rotation from pi to pi/2, without adding parameters.
        theta = 0.5 * _phase_torch.pi * _phase_torch.tanh(static_logit)
    elif mode == 'content_phase':
        # Allow data-dependent correction to the phase-conditioned angle.
        # Zero init means the first forward pass exactly reproduces original.
        theta = _phase_torch.pi * _phase_torch.tanh(
            static_logit + self.content_angle(residual)
        )
    else:
        raise ValueError('Unknown phase replacement mode: ' + repr(mode))

    proj = (h_sin * h_cos).sum(dim=-1, keepdim=True) / (
        (h_cos * h_cos).sum(dim=-1, keepdim=True) + 1e-6)
    h_sin_ortho = h_sin - proj * h_cos
    phase_impact = _phase_torch.cos(theta) * h_cos + _phase_torch.sin(theta) * h_sin_ortho
    out = self.out_proj(phase_impact)
    out = self.norm(out + self.shortcut(residual))
    return out

Model.__init__ = _paloma_phase_model_init
PhaseModulation.forward = _paloma_phase_forward
'''


def git_blob(b: bytes) -> str:
    return hashlib.sha1(f'blob {len(b)}\0'.encode() + b).hexdigest()


def private_source(upstream: Path, destination: Path) -> dict:
    upstream, destination = upstream.resolve(), destination.resolve()
    if upstream == destination or upstream in destination.parents:
        raise ValueError('Private copy must not be within original source')
    orig = upstream / 'models' / 'Paloma.py'
    if not orig.is_file() or not (upstream / 'run.py').is_file():
        raise FileNotFoundError('Official Paloma source missing. Run bash setup_sources.sh first.')
    b = orig.read_bytes()
    if git_blob(b) != BLOB:
        raise RuntimeError(f'Official source hash mismatch: {git_blob(b)} (expected {BLOB})')
    sha = hashlib.sha256(b).hexdigest()
    manifest_path = destination / 'phase_replacement_manifest.json'
    if destination.exists():
        if not manifest_path.exists():
            raise RuntimeError('Private source directory exists but has no manifest')
        m = json.loads(manifest_path.read_text(encoding='utf-8'))
        patched = destination / 'models' / 'Paloma.py'
        if m.get('patch_version') != PATCH_VERSION or m.get('source_sha256') != sha or not patched.exists():
            raise RuntimeError('Stale private source; delete the private directory to rebuild')
        if hashlib.sha256(patched.read_bytes()).hexdigest() != m.get('patched_sha256'):
            raise RuntimeError('Private source changed after patching')
        return m
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(upstream, destination, ignore=shutil.ignore_patterns(
        '.git', '__pycache__', '*.pyc', 'checkpoints', 'results', 'test_results', 'runs', 'logs'
    ))
    patched = destination / 'models' / 'Paloma.py'
    with patched.open('a', encoding='utf-8') as f:
        f.write(APPEND_PATCH)
    m = {
        'patch_version': PATCH_VERSION, 'source_commit': SHAS['Paloma'],
        'source_sha256': sha, 'source_git_blob': BLOB,
        'patched_sha256': hashlib.sha256(patched.read_bytes()).hexdigest(),
        'variants': list(VARIANTS),
        'description': 'Standalone private copy; original source unchanged'
    }
    manifest_path.write_text(json.dumps(m, indent=2), encoding='utf-8')
    return m


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def batch_from_ett(file: Path, L: int, H: int):
    import numpy as np
    import torch
    import pandas as pd
    arr = pd.read_csv(file, usecols=list(range(1, 8)), nrows=L + H + 16).to_numpy(dtype='float32')
    if arr.shape[0] < L + H + 2:
        raise ValueError('Not enough rows for an ETTh1/ETTm1 smoke test')
    train = arr[:L + H]
    arr = (arr - train.mean(0)) / (train.std(0) + 1e-6)
    inputs = np.stack([arr[i:i+L] for i in range(2)])
    labels = np.stack([arr[i+L:i+L+H] for i in range(2)])
    return torch.from_numpy(inputs), torch.from_numpy(labels)


def smoke(upstream: Path, private_repo: Path, file: Path, ds: str, H: int, variants: list[str]):
    import torch
    opt = {
        'ETTh1': dict(cycle=24, dropout=.5),
        'ETTm1': dict(cycle=96, dropout=.4)
    }[ds]
    conf = SimpleNamespace(seq_len=96, pred_len=H, enc_in=7,
                           cycle=opt['cycle'], dropout=opt['dropout'],
                           d_model=256, use_revin=1, individual=0)
    src = load_module(upstream / 'models' / 'Paloma.py', 'paloma_upstream_phase_v2')
    patched = load_module(private_repo / 'models' / 'Paloma.py', 'paloma_private_phase_v2')
    x, target = batch_from_ett(file, 96, H)
    os.environ['PALOMA_PHASE_REPLACEMENT'] = 'original'
    torch.manual_seed(2024)
    upstream_model = src.Model(conf)
    torch.manual_seed(2024)
    private_original = patched.Model(conf)
    assert set(upstream_model.state_dict()) == set(private_original.state_dict())
    for k, a in upstream_model.state_dict().items():
        assert torch.equal(a, private_original.state_dict()[k]), ('initial weights changed', k)
    upstream_model.eval();private_original.eval()
    with torch.no_grad():
        y_ref = upstream_model(x, torch.tensor([0, 1]))
        y_base = private_original(x, torch.tensor([0, 1]))
        assert torch.equal(y_ref, y_base), 'patched original differs from official forward'
    print('[CONTROL] original == source (exact weights and exact forward)', flush=True)

    for v in variants:
        os.environ['PALOMA_PHASE_REPLACEMENT'] = v
        torch.manual_seed(2024)
        m = patched.Model(conf)
        # Same original parameters for all variants. content_phase adds one zero-init layer.
        for k, a in upstream_model.state_dict().items():
            assert torch.equal(a, m.state_dict()[k]), ('initial weights differ', v, k)
        if v == 'content_phase':
            assert torch.count_nonzero(m.phase_fusion.content_angle.weight) == 0
            m.eval()
            with torch.no_grad():
                z = m(x, torch.tensor([0, 1]))
                assert torch.allclose(z, y_ref, rtol=1e-7, atol=1e-7), \
                    'zero-initialized content conditioning must match baseline initially'
        m.train()
        y = m(x, torch.tensor([0, 1]))
        assert tuple(y.shape) == tuple(target.shape) and torch.isfinite(y).all()
        loss = (y-target).square().mean()
        loss.backward()
        grads = sum(1 for p in m.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
        if not grads:
            raise RuntimeError(v + ': no finite gradients')
        if v == 'content_phase':
            assert m.phase_fusion.content_angle.weight.grad is not None
            assert torch.isfinite(m.phase_fusion.content_angle.weight.grad).all()
        print(f'SMOKE_OK variant={v} {ds} L96 H{H} shape={tuple(y.shape)} '
              f'loss={loss.item():.6f} finite_grad_params={grads}', flush=True)
    os.environ.pop('PALOMA_PHASE_REPLACEMENT', None)


def summarize(out_dir: Path, baseline_dir: Path):
    rows = []
    for f in sorted(out_dir.glob('paper/*/h*/*/metadata.json')):
        try:
            item=json.loads(f.read_text(encoding='utf-8'))
            if 'variant' in item: rows.append(item)
        except (OSError, ValueError):
            continue
    for row in rows:
        data, horizon = row['dataset'], row['horizon']
        base_same = out_dir / 'paper' / data / f'h{horizon}' / 'original' / 'metadata.json'
        base_external = baseline_dir / 'paper' / data / f'h{horizon}' / 'original' / 'metadata.json'
        ref_path = base_same if base_same.exists() else base_external
        baseline = json.loads(ref_path.read_text()) if ref_path.exists() else None
        row['baseline_source'] = str(ref_path) if baseline and baseline.get('status') == 'ok' else ''
        for met in ('mse','mae'):
            base = baseline.get(met) if baseline and baseline.get('status') == 'ok' else None
            val = row.get(met)
            row[f'{met}_gain_pct_vs_original'] = (100*(base-val)/base if base and val is not None else None)
    fields = ['model','variant','dataset','horizon','lookback','profile','status','returncode',
              'mse','mae','mse_gain_pct_vs_original','mae_gain_pct_vs_original',
              'wall_seconds','gpu_total_used_peak_mib','baseline_source','source_commit',
              'phase_method','original_source_sha256']
    out_dir.mkdir(parents=True,exist_ok=True)
    file = out_dir/'phase_replacement_summary.csv'
    with file.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f, fieldnames=fields,extrasaction='ignore')
        w.writeheader();w.writerows(rows)
    return file


def main():
    ap=argparse.ArgumentParser(description='Paloma targeted phase replacement v2')
    ap.add_argument('--mode',choices=['smoke','full'],default='smoke')
    ap.add_argument('--variants',default='original,bounded_phase,content_phase')
    ap.add_argument('--datasets',default='ETTh1')
    ap.add_argument('--horizons',default='96')
    ap.add_argument('--profile',choices=['paper'],default='paper')
    ap.add_argument('--data-dir',default=os.environ.get('DATA_DIR','/root/autodl-tmp/GTR_ResearchWorkbench_Modular/dataset'))
    ap.add_argument('--sources-dir',default=os.environ.get('SOURCES_DIR',str(ROOT/'sources')))
    ap.add_argument('--out-dir',default=str(ROOT/'runs'/'paloma_phase_replacement'))
    ap.add_argument('--baseline-dir',default=str(ROOT/'runs'/'paloma_ablation'))
    ap.add_argument('--gpu',type=int,default=0)
    ap.add_argument('--epochs',type=int,default=0)
    ap.add_argument('--force',action='store_true')
    ap.add_argument('--continue-on-error',action='store_true')
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    datasets=split_names(args.datasets,{'ETTh1':0,'ETTm1':1})
    variants=split_names(args.variants,{x:1 for x in VARIANTS})
    horizons=[int(h.strip()) for h in args.horizons.split(',')]
    if not horizons or any(h not in (96,336) for h in horizons):
        ap.error('Only verified horizons 96,336 are supported')
    if args.epochs < 0:
        ap.error('epochs must be >= 0')
    upstream=Path(args.sources_dir).expanduser().resolve()/'Paloma'
    out_dir=Path(args.out_dir).expanduser().resolve()
    private_repo=out_dir/'private_source'/'Paloma'
    if args.dry_run:
        for d in datasets:
            for h in horizons:
                for v in variants:
                    dest=out_dir/'paper'/d/f'h{h}'/v
                    cmd,_=build_params('Paloma',d,h,'paper',args.epochs,args.gpu,
                          Path(args.data_dir)/f'{d}.csv',dest)
                    cmd[cmd.index('--model_id')+1]+='_'+v+'_phasev2'
                    print('[DRY]',f'PALOMA_PHASE_REPLACEMENT={v}',d,h,' '.join(cmd))
        return
    manifest=private_source(upstream,private_repo)
    print('[SOURCE] upstream untouched, original sha256='+manifest['source_sha256'],flush=True)
    if args.mode=='full' and not get_cuda():
        raise RuntimeError('Author trainer requires CUDA for checkpoint loading')
    for d in datasets:
        data_file=find_data(args.data_dir,d)
        validate_data(data_file,d)
        for h in horizons:
            if args.mode=='smoke':
                smoke(upstream,private_repo,data_file,d,h,variants)
                continue
            for v in variants:
                dest=out_dir/'paper'/d/f'h{h}'/v
                meta_path=dest/'metadata.json'
                if meta_path.exists() and not args.force:
                    previous=json.loads(meta_path.read_text(encoding='utf-8'))
                    if previous.get('status')=='ok':
                        print('[SKIP]',d,h,v,'already completed',flush=True)
                        continue
                cmd,meta=build_params('Paloma',d,h,'paper',args.epochs,args.gpu,data_file,dest)
                cmd[cmd.index('--model_id')+1]+='_'+v+'_phasev2'
                meta.update(variant=v,phase_method={'original':'official angle',
                             'bounded_phase':'pi/2 bounded static angle',
                             'content_phase':'static angle + zero initialized residual-conditioned correction'}[v],
                            original_source_sha256=manifest['source_sha256'],
                            private_source_sha256=manifest['patched_sha256'])
                old=os.environ.get('PALOMA_PHASE_REPLACEMENT')
                os.environ['PALOMA_PHASE_REPLACEMENT']=v
                try:
                    ret=run_job(cmd,private_repo,dest,meta,args)
                finally:
                    if old is None: os.environ.pop('PALOMA_PHASE_REPLACEMENT',None)
                    else: os.environ['PALOMA_PHASE_REPLACEMENT']=old
                summary=summarize(out_dir,Path(args.baseline_dir).expanduser().resolve())
                print('[SUMMARY]',summary,flush=True)
                if ret['status']!='ok' and not args.continue_on_error:
                    raise RuntimeError('Training failed: '+str(dest/'train.log'))
    if args.mode=='full':
        print('[DONE]',summarize(out_dir,Path(args.baseline_dir).expanduser().resolve()),flush=True)
    else:
        print('[SMOKE] Passed. No forecasting performance has been measured.',flush=True)

if __name__=='__main__':
    main()
