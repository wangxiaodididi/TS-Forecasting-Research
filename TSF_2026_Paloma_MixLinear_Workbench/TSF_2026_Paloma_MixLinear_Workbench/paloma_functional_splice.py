#!/usr/bin/env python3
"""Functional-module splice experiments on original Paloma (2026).

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
VERSION = 3
VARIANTS = ("original", "tq_channel", "olinear_channel", "localconv")
VARIANT_ENV = "PALOMA_FUNCTIONAL_SPLICE"

# Mechanism-specific append-only patch, applied to a byte-identical private copy
# of the pinned official model. No third-party OLinear or MixLinear source is copied.
PATCH = r'''

# PALOMA_FUNCTIONAL_SPLICE_V3
import os as _sp_os
import torch as _sp_torch
import torch.nn as _sp_nn
import torch.nn.functional as _sp_F

class _TQChannel(_sp_nn.Module):
    """TQNet-inspired query-key-value channel communication.

    Query is separate from Paloma's subtraction template Q. The source TQNet
    attends across channel tokens with temporal vectors as embedding.
    """
    def __init__(self, L, C, period):
        super().__init__()
        if L % 4:
            raise ValueError("TQ module requires seq_len divisible by four")
        self.temporal_query = _sp_nn.Parameter(_sp_torch.zeros(period, C))
        self.attn = _sp_nn.MultiheadAttention(L, num_heads=4, dropout=0.5, batch_first=True)

    def forward(self, x, cycle_index):
        idx = (cycle_index.view(-1, 1) +
               _sp_torch.arange(x.size(-1), device=x.device)[None, :]) % self.temporal_query.size(0)
        q = self.temporal_query[idx].permute(0, 2, 1)
        out, _ = self.attn(q, x, x, need_weights=False)
        return out


class _OLinearChannel(_sp_nn.Module):
    """Minimal independently implemented channel mixer inspired by OLinear.

    OLinear's LinearEncoder mixes channel-token values with a nonnegative,
    row-normalized CxC matrix. This is NOT newLinear.py nor the full NormLin.
    """
    def __init__(self, L, C):
        super().__init__()
        self.logits = _sp_nn.Parameter(4.0 * _sp_torch.eye(C) - 3.0)
        self.value_proj = _sp_nn.Linear(L, L)

    def forward(self, x):
        weights = _sp_F.softplus(self.logits)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        value = self.value_proj(x)
        return _sp_torch.einsum('ij,bjl->bil', weights, value)


class _LocalResidual(_sp_nn.Module):
    """MixLinear-inspired local 1D filtering, applied to Paloma residual only."""
    def __init__(self, kernel_size=25):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd for length preservation")
        self.conv = _sp_nn.Conv1d(1, 1, kernel_size=kernel_size,
                                  padding=kernel_size//2, bias=False)

    def forward(self, residual):
        B, C, L = residual.shape
        return self.conv(residual.reshape(B*C, 1, L)).reshape(B, C, L)


_SourcePalomaInitV3 = Model.__init__
_SourcePalomaForwardV3 = Model.forward

def _paloma_splice_init_v3(self, configs):
    _SourcePalomaInitV3(self, configs)
    mode = _sp_os.environ.get('PALOMA_FUNCTIONAL_SPLICE', 'original')
    if mode not in ('original','tq_channel','olinear_channel','localconv'):
        raise ValueError('Unknown functional splice variant: '+str(mode))
    self.splice_variant = mode
    if mode == 'original':
        return
    cpu_state = _sp_torch.get_rng_state()
    gpu_states = _sp_torch.cuda.get_rng_state_all() if _sp_torch.cuda.is_initialized() else None
    try:
        if mode == 'tq_channel':
            self.splice_module = _TQChannel(self.seq_len, self.enc_in, self.cycle_len)
        elif mode == 'olinear_channel':
            self.splice_module = _OLinearChannel(self.seq_len, self.enc_in)
        else:
            self.splice_module = _LocalResidual(25)
        # Zero at initialization: exact upstream function on first forward.
        self.splice_gate = _sp_nn.Parameter(_sp_torch.zeros(()))
    finally:
        _sp_torch.set_rng_state(cpu_state)
        if gpu_states is not None:
            _sp_torch.cuda.set_rng_state_all(gpu_states)


def _paloma_splice_forward_v3(self, x, cycle_index):
    mode = self.splice_variant
    if mode == 'original':
        return _SourcePalomaForwardV3(self, x, cycle_index)

    # Identical Paloma normalization, cycle indexing and branch ordering.
    if self.use_revin:
        seq_mean = _sp_torch.mean(x, dim=1, keepdim=True)
        seq_var = _sp_torch.var(x, dim=1, keepdim=True) + 1e-5
        x = (x - seq_mean) / _sp_torch.sqrt(seq_var)
    B, T, C = x.shape
    x_input = x.permute(0, 2, 1)
    idx = (cycle_index.view(-1, 1) +
           _sp_torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)) % self.cycle_len
    query_input = self.Q[idx].permute(0, 2, 1)
    phase = self.phase[idx].permute(0, 2, 1)
    amplitude = self.amplitude[idx].permute(0, 2, 1)
    residual = x_input - query_input

    if mode == 'localconv':
        residual = residual + self.splice_gate * self.splice_module(residual)

    phase_impact = self.phase_fusion(residual, phase)
    amplitude_impact = self.amplitude_fusion(residual, amplitude)

    if mode == 'tq_channel':
        x_skip = x_input + self.splice_gate * self.splice_module(x_input, cycle_index)
    elif mode == 'olinear_channel':
        x_skip = x_input + self.splice_gate * self.splice_module(x_input)
    else:
        x_skip = x_input

    input_proj = self.input_proj(x_skip)
    hidden = self.model(phase_impact + amplitude_impact)
    output = self.output_proj(hidden + input_proj)
    output = output.permute(0, 2, 1)
    if self.use_revin:
        output = output * _sp_torch.sqrt(seq_var) + seq_mean
    return output

Model.__init__ = _paloma_splice_init_v3
Model.forward = _paloma_splice_forward_v3
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
            raise RuntimeError('Private source out of date; remove runs/paloma_functional_splice/private_source to rebuild')
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
    original = module_from_file(original_source/'models'/'Paloma.py', 'paloma_orig_splice_v3')
    modified = module_from_file(private/'models'/'Paloma.py', 'paloma_patch_splice_v3')
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
        if v!='original':
            assert net.splice_gate.grad is not None and torch.isfinite(net.splice_gate.grad)
            assert net.splice_gate.grad.abs()>0, 'gate should receive initial gradient'
        print(f'[SMOKE_OK] {v} {ds} L96 H{h} output={tuple(y.shape)} finite_grad_params={count}', flush=True)
    os.environ.pop(VARIANT_ENV, None)


def collect_results(out_dir: Path) -> Path:
    rows=[]
    for metadata in sorted((out_dir/'paper').glob('*/h*/*/metadata.json')):
        rec=json.loads(metadata.read_text(encoding='utf-8'))
        if rec.get('variant') in VARIANTS:
            rows.append(rec)
    for row in rows:
        reference = (out_dir/'paper'/row['dataset']/f"h{row['horizon']}"/'original'/'metadata.json')
        original=json.loads(reference.read_text(encoding='utf-8')) if reference.is_file() else {}
        valid = original.get('status')=='ok' and row.get('status')=='ok' and \
                original.get('seed')==row.get('seed') and \
                original.get('lookback')==row.get('lookback') and \
                original.get('training_epochs')==row.get('training_epochs') and \
                original.get('horizon')==row.get('horizon')
        row['baseline_metadata'] = str(reference) if valid else ''
        for k in ('mse','mae'):
            value = row.get(k)
            baseline = original.get(k) if valid else None
            row[f'{k}_gain_pct_vs_original'] = (100 * (baseline-value) / baseline
                     if baseline and value is not None else '')
    fields = ['model','variant','dataset','horizon','lookback','profile','seed',
              'status','returncode','mse','mae','mse_gain_pct_vs_original',
              'mae_gain_pct_vs_original','wall_seconds','gpu_total_used_peak_mib',
              'baseline_metadata','source_commit','patched_source_sha256','training_epochs']
    path=out_dir/'functional_splice_summary.csv'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    return path


def parse_args(argv=None):
    ap=argparse.ArgumentParser(description='Paloma 2026 three-function splice experiments')
    ap.add_argument('--mode',choices=('smoke','full'),default='smoke')
    ap.add_argument('--datasets',default='ETTh1')
    ap.add_argument('--horizons',default='96')
    ap.add_argument('--variants',default=','.join(VARIANTS))
    ap.add_argument('--profile',choices=['paper'],default='paper')
    ap.add_argument('--data-dir',default=os.environ.get('DATA_DIR','/root/autodl-tmp/GTR_ResearchWorkbench_Modular/dataset'))
    ap.add_argument('--sources-dir',default=os.environ.get('SOURCES_DIR',str(ROOT/'sources')))
    ap.add_argument('--out-dir',default=str(ROOT/'runs'/'paloma_functional_splice'))
    ap.add_argument('--epochs',type=int,default=0)
    ap.add_argument('--gpu',type=int,default=0)
    ap.add_argument('--force',action='store_true')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--continue-on-error',action='store_true')
    a=ap.parse_args(argv)
    try:
        a.variants=split_names(a.variants,{v:1 for v in VARIANTS})
        a.datasets=split_names(a.datasets,{'ETTh1':1,'ETTm1':1})
    except ValueError as exc:
        ap.error(str(exc))
    a.horizons=[int(h.strip()) for h in a.horizons.split(',')]
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
    if args.dry_run:
        for ds in args.datasets:
            for h in args.horizons:
                for v in args.variants:
                    dest=out_dir/'paper'/ds/f'h{h}'/v
                    fake=Path(args.data_dir)/f'{ds}.csv'
                    cmd,_=build_params('Paloma',ds,h,'paper',args.epochs,args.gpu,fake,dest)
                    cmd[cmd.index('--model_id')+1]+='_'+v+'_splicev3'
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
                cmd[cmd.index('--model_id')+1]+='_'+v+'_splicev3'
                meta.update(variant=v,seed=2024,patched_source_sha256=manifest['patched_sha256'],
                            splice_description={
                                'original':'Official model with no injected module',
                                'tq_channel':'TQNet-inspired channel attention added to input skip',
                                'olinear_channel':'OLinear-LinearEncoder-inspired channel matrix added to input skip',
                                'localconv':'MixLinear-inspired local filter on template residual'
                            }[v])
                previous=os.environ.get(VARIANT_ENV)
                os.environ[VARIANT_ENV]=v
                try:
                    result=run_job(cmd,private,dest,meta,args)
                finally:
                    if previous is None: os.environ.pop(VARIANT_ENV,None)
                    else: os.environ[VARIANT_ENV]=previous
                report=collect_results(out_dir)
                print('[SUMMARY]',report,flush=True)
                if result['status']!='ok' and not args.continue_on_error:
                    raise RuntimeError(f'Failed {ds} H{h} {v}. Inspect {dest/"train.log"}')
    if args.mode=='full':
        print('[DONE]',collect_results(out_dir),flush=True)
    else:
        print('[SMOKE] all gradients/shapes checked; not paper reproduction',flush=True)


if __name__=='__main__':
    main()
