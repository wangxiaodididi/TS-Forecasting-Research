#!/usr/bin/env python3
"""Model-capability ablations of Paloma; uses the author's untouched trainer.

Deploy next to experiment.py in TSF_2026_Paloma_MixLinear_Workbench.
The upstream source is never edited in place; a private experiment copy is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiment import (
    SHAS, build_params, find_data, validate_data, run_job, get_cuda,
    split_names,
)

VARIANTS = (
    "original",
    "no_phase",
    "no_amplitude",
    "no_template",
    "no_cycle_condition",
    "no_input_skip",
    "no_phase_and_amplitude",
)
MARKER = "# PALOMA_ABLATION_EXTENSION_V1"
# Git blob hash of models/Paloma.py in the pinned repository snapshot.
EXPECTED_MODEL_BLOB = '03dc7c6d23751cde6acaca77c803117c89702006'

# The original forward is retained as _ReferenceForward and restored exactly
# in original mode. All variants retain identical state_dict keys.
ABLATED_FORWARD = r'''

# PALOMA_ABLATION_EXTENSION_V1
import os as _ablation_os
import torch as _ablation_torch

_ReferenceModel = Model
_ReferenceForward = Model.forward

def _ablation_forward(self, x, cycle_index):
    mode = _ablation_os.environ.get('PALOMA_ABLATION', 'original')
    allowed = ('original', 'no_phase', 'no_amplitude', 'no_template',
               'no_cycle_condition', 'no_input_skip',
               'no_phase_and_amplitude')
    if mode not in allowed:
        raise ValueError('Unknown PALOMA_ABLATION=' + repr(mode))
    if mode == 'original':
        return _ReferenceForward(self, x, cycle_index)

    # Follow the pinned upstream forward exactly, except for the
    # explicitly indicated branch deletion / conditioning deletion.
    if self.use_revin:
        seq_mean = _ablation_torch.mean(x, dim=1, keepdim=True)
        seq_var = _ablation_torch.var(x, dim=1, keepdim=True) + 1e-5
        x = (x - seq_mean) / _ablation_torch.sqrt(seq_var)

    B, T, C = x.shape
    x_input = x.permute(0, 2, 1)
    gather_index = (
        cycle_index.view(-1, 1)
        + _ablation_torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)
    ) % self.cycle_len
    query_input = self.Q[gather_index].permute(0, 2, 1)
    phase = self.phase[gather_index].permute(0, 2, 1)
    amplitude = self.amplitude[gather_index].permute(0, 2, 1)

    if mode in ('no_template', 'no_cycle_condition'):
        query_input = _ablation_torch.zeros_like(query_input)
    if mode == 'no_cycle_condition':
        phase = _ablation_torch.zeros_like(phase)
        amplitude = _ablation_torch.zeros_like(amplitude)

    residual = x_input - query_input
    phase_impact = self.phase_fusion(residual, phase)
    amplitude_impact = self.amplitude_fusion(residual, amplitude)
    if mode in ('no_phase', 'no_phase_and_amplitude'):
        phase_impact = _ablation_torch.zeros_like(phase_impact)
    if mode in ('no_amplitude', 'no_phase_and_amplitude'):
        amplitude_impact = _ablation_torch.zeros_like(amplitude_impact)

    input_proj = self.input_proj(x_input)
    hidden = self.model(phase_impact + amplitude_impact)
    if mode == 'no_input_skip':
        output = self.output_proj(hidden)
    else:
        output = self.output_proj(hidden + input_proj)
    output = output.permute(0, 2, 1)
    if self.use_revin:
        output = output * _ablation_torch.sqrt(seq_var) + seq_mean
    return output

Model.forward = _ablation_forward
'''


def digest(s: bytes) -> str:
    return hashlib.sha256(s).hexdigest()


def git_blob_digest(content: bytes) -> str:
    prefix = f'blob {len(content)}\0'.encode('ascii')
    return hashlib.sha1(prefix + content).hexdigest()


def build_private_source(upstream: Path, output: Path) -> dict:
    upstream = upstream.resolve()
    output = output.resolve()
    if not (upstream / 'models' / 'Paloma.py').is_file() or not (upstream / 'run.py').is_file():
        raise FileNotFoundError(f'Official Paloma source missing in {upstream}. First run bash setup_sources.sh')
    if upstream == output or upstream in output.parents:
        # Output as nested child would recursively copy itself.
        raise ValueError('Private source directory must not be nested in the upstream source directory')
    source_file = upstream / 'models' / 'Paloma.py'
    original = source_file.read_bytes()
    if MARKER.encode() in original:
        raise RuntimeError('The upstream source is already modified; restore the pinned official version')
    actual_blob = git_blob_digest(original)
    if actual_blob != EXPECTED_MODEL_BLOB:
        raise RuntimeError(f'Unexpected Paloma source blob {actual_blob}; expected {EXPECTED_MODEL_BLOB}. ' +
                           'Re-run setup_sources.sh with the pinned upstream snapshot')
    required = (
        'class Model(nn.Module):', 'def forward(self, x, cycle_index):',
        'query_input = self.Q[gather_index].permute(0, 2, 1)',
        'phase_impact = self.phase_fusion(residual, phase)',
        'amplitude_impact = self.amplitude_fusion(residual, amplitude)',
        'hidden = self.model(phase_impact + amplitude_impact)',
        'output = self.output_proj(hidden + input_proj)',
    )
    decoded = original.decode('utf-8')
    for fragment in required:
        if fragment not in decoded:
            raise RuntimeError(f'Unexpected upstream source structure: {fragment!r}; do not patch this version')
    orig_sha = digest(original)
    manifest_path = output / 'ablation_source_manifest.json'
    if output.exists():
        if not manifest_path.is_file():
            raise RuntimeError(f'Existing {output} lacks patch manifest; refusing to overwrite it')
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old.get('source_sha256') != orig_sha or old.get('patch_version') != 1:
            raise RuntimeError('Upstream changed or private patch version differs; remove the private copy yourself to rebuild')
        copied = (output / 'models' / 'Paloma.py').read_text(encoding='utf-8')
        if MARKER not in copied:
            raise RuntimeError('Private source patch missing marker')
        if digest((output / 'models' / 'Paloma.py').read_bytes()) != old.get('patched_sha256'):
            raise RuntimeError('Private source was changed after patch creation; refusing to use it')
        return old
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(upstream, output, ignore=shutil.ignore_patterns(
        '.git', '__pycache__', '*.pyc', 'checkpoints', 'results', 'test_results', 'runs', 'logs'
    ))
    patched = output / 'models' / 'Paloma.py'
    with patched.open('a', encoding='utf-8', newline='\n') as f:
        f.write(ABLATED_FORWARD)
    meta = {
        'patch_version': 1,
        'source_commit_expected': SHAS['Paloma'],
        'source_sha256': orig_sha,
        'source_blob_sha1': actual_blob,
        'patched_sha256': digest(patched.read_bytes()),
        'upstream': str(upstream),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'note': 'Local private research copy; original upstream remains untouched',
    }
    manifest_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    return meta


def load_private_model(repo: Path):
    module_path = repo / 'models' / 'Paloma.py'
    spec = importlib.util.spec_from_file_location('paloma_private_ablation', module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def real_ett_windows(data_path: Path, seq: int, horizon: int):
    import numpy as np
    import torch
    arr = np.loadtxt(data_path, delimiter=',', skiprows=1, usecols=range(1, 8), max_rows=seq+horizon+20, dtype=np.float32)
    if len(arr) < seq+horizon+2:
        raise RuntimeError('Not enough ETT data for smoke test')
    mu = arr[:seq+horizon].mean(axis=0)
    sd = arr[:seq+horizon].std(axis=0) + 1e-6
    norm = (arr - mu) / sd
    x = torch.tensor(np.stack([norm[i:i+seq] for i in range(2)]))
    y = torch.tensor(np.stack([norm[i+seq:i+seq+horizon] for i in range(2)]))
    return x, y


def run_smoke(repo: Path, dataset: str, data_path: Path, horizon: int, variants) -> None:
    import torch
    from types import SimpleNamespace
    config = {
        'ETTh1': dict(cycle=24, dropout=.5),
        'ETTm1': dict(cycle=96, dropout=.4),
    }[dataset]
    opts = SimpleNamespace(
        seq_len=96, pred_len=horizon, enc_in=7,
        cycle=config['cycle'], d_model=256,
        dropout=config['dropout'], use_revin=1, individual=0,
    )
    mod = load_private_model(repo)
    x, target = real_ett_windows(data_path, 96, horizon)
    # Strict equivalence of the original branch and untouched reference.
    torch.manual_seed(2024)
    reference = mod._ReferenceModel(opts)
    edited = mod.Model(opts)
    edited.load_state_dict(reference.state_dict())
    reference.eval(); edited.eval()
    with torch.no_grad():
        base = mod._ReferenceForward(reference, x, torch.tensor([0, 1]))
        os.environ['PALOMA_ABLATION'] = 'original'
        patched_base = edited(x, torch.tensor([0, 1]))
        assert torch.equal(base, patched_base), 'Original variant diverged from untouched original forward'
    print('[CONTROL] original forward is bitwise identical to source model on a real ETT window', flush=True)
    for variant in variants:
        os.environ['PALOMA_ABLATION'] = variant
        torch.manual_seed(2024)
        model = mod.Model(opts)
        y = model(x, torch.tensor([0, 1]))
        assert y.shape == target.shape, (variant, y.shape, target.shape)
        assert torch.isfinite(y).all(), f'{variant}: non-finite predictions'
        loss = (y-target).square().mean()
        loss.backward()
        n = sum(int(p.grad is not None and torch.isfinite(p.grad).all()) for p in model.parameters())
        assert n > 0, f'{variant}: no finite parameter gradients'
        print(f'SMOKE_OK variant={variant} {dataset} L96 H{horizon} shape={tuple(y.shape)} '
              f'loss={loss.item():.6f} finite_grad_params={n}', flush=True)
    os.environ.pop('PALOMA_ABLATION', None)


def argparser():
    p=argparse.ArgumentParser(description='Paloma targeted functional ablations')
    p.add_argument('--mode', choices=('smoke','full'), default='smoke')
    p.add_argument('--datasets', default='ETTh1')
    p.add_argument('--horizons', default='96')
    p.add_argument('--variants', default='original,no_phase,no_amplitude,no_template,no_cycle_condition,no_input_skip')
    p.add_argument('--data-dir', default=os.environ.get('DATA_DIR', '/root/autodl-tmp/GTR_ResearchWorkbench_Modular/dataset'))
    p.add_argument('--sources-dir', default=os.environ.get('SOURCES_DIR', str(ROOT / 'sources')))
    p.add_argument('--out-dir', default=str(ROOT / 'runs' / 'paloma_ablation'))
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--epochs', type=int, default=0, help='0 keeps paper hyperparameters')
    p.add_argument('--profile', choices=('paper','aligned'), default='paper')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--continue-on-error', action='store_true')
    p.add_argument('--force', action='store_true', help='rerun completed cases; original metrics retained in previous logs')
    return p.parse_args()


def save_table(records: list[dict], path: Path) -> None:
    fields = ['model','variant','dataset','horizon','lookback','profile','provenance','source_commit',
              'status','returncode','mse','mae','mse_degradation_pct_vs_original',
              'mae_degradation_pct_vs_original','wall_seconds','gpu_total_used_peak_mib']
    by_context = {(r['profile'],r['dataset'],r['horizon']):r for r in records
                  if r.get('variant')=='original' and r.get('status')=='ok'}
    for r in records:
        baseline = by_context.get((r['profile'],r['dataset'],r['horizon']))
        for metric in ('mse','mae'):
            key=f'{metric}_degradation_pct_vs_original'
            a, b = r.get(metric), baseline.get(metric) if baseline else None
            r[key] = (100.0*(a-b)/b if a is not None and b not in (None,0) else None)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader();w.writerows(records)


def collect_summary(outdir: Path) -> list[dict]:
    records=[]
    for f in sorted(outdir.glob('**/metadata.json')):
        try:
            meta=json.loads(f.read_text(encoding='utf-8'))
            if 'variant' in meta:
                records.append(meta)
        except (OSError,ValueError):
            print(f'[WARN] unreadable metadata: {f}', file=sys.stderr)
    return records


def main():
    args=argparser()
    datasets=split_names(args.datasets, {'ETTh1':0,'ETTm1':1})
    variants=split_names(args.variants, {x:1 for x in VARIANTS})
    horizons=[int(s.strip()) for s in args.horizons.split(',')]
    if any(h not in (96,336) for h in horizons) or not horizons:
        raise ValueError('Verified combinations: horizons 96,336 only')
    src=Path(args.sources_dir).expanduser().resolve()/'Paloma'
    out=Path(args.out_dir).expanduser().resolve()
    private_repo=out/'private_source'/'Paloma'
    if args.dry_run:
        for ds in datasets:
            for h in horizons:
                for v in variants:
                    path=out/args.profile/ds/f'h{h}'/v
                    cmd,_=build_params('Paloma',ds,h,args.profile,args.epochs,args.gpu,
                                       Path(args.data_dir)/f'{ds}.csv',path)
                    cmd[cmd.index('--model_id')+1] += '_' + v
                    print('[DRY]', f'PALOMA_ABLATION={v}', ds, h, ' '.join(cmd))
        return
    manifest=build_private_source(src, private_repo)
    print(f"[SOURCE] Original SHA256 {manifest['source_sha256'][:16]}...; upstream source unchanged",flush=True)
    if args.mode == 'full' and not get_cuda():
        raise RuntimeError('Full training requires CUDA because the author trainer hardcodes CUDA checkpoint loading')
    for ds in datasets:
        data_file=find_data(args.data_dir,ds)
        validate_data(data_file, ds)
        for h in horizons:
            if args.mode == 'smoke':
                run_smoke(private_repo,ds,data_file,h,variants)
                continue
            for variant in variants:
                dest=out/args.profile/ds/f'h{h}'/variant
                meta_file=dest/'metadata.json'
                if meta_file.exists() and not args.force:
                    previous=json.loads(meta_file.read_text(encoding='utf-8'))
                    if previous.get('status')=='ok':
                        print(f'[SKIP] existing completed case {ds} H{h} {variant}',flush=True)
                        continue
                cmd, meta=build_params('Paloma',ds,h,args.profile,args.epochs,args.gpu,data_file,dest)
                cmd[cmd.index('--model_id')+1] += '_' + variant
                meta['variant']=variant
                meta['private_source_sha256']=manifest['patched_sha256']
                meta['original_source_sha256']=manifest['source_sha256']
                            # run_job deliberately uses the upstream trainer unmodified.
                # Set os.environ temporarily so only this subprocess inherits it.
                old=os.environ.get('PALOMA_ABLATION')
                os.environ['PALOMA_ABLATION']=variant
                try:
                    result=run_job(cmd,private_repo,dest,meta,args)
                finally:
                    if old is None: os.environ.pop('PALOMA_ABLATION', None)
                    else: os.environ['PALOMA_ABLATION']=old
                save_table(collect_summary(out),out/'ablation_summary.csv')
                if result['status'] != 'ok' and not args.continue_on_error:
                    raise RuntimeError(f'Experiment failed: {dest / "train.log"}')
    if args.mode == 'full':
        save_table(collect_summary(out),out/'ablation_summary.csv')
        print(f'[DONE] {out / "ablation_summary.csv"}',flush=True)
    else:
        print('[SMOKE] completed; no held-out forecasting metrics were produced',flush=True)

if __name__=='__main__':
    main()
