#!/usr/bin/env python3
"""Recover Paloma functional splice v3 summary from all existing job files.

No training, no model imports, no GPU, no network. Preserves the old CSV as .bak.
Works when executed from the experiment workspace root or placed elsewhere with
--workspace /path/to/TSF_2026_Paloma_MixLinear_Workbench.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

VALID_VARIANTS = {"original", "olinear_channel", "tq_channel", "localconv"}
FIELDS = [
    'model', 'variant', 'dataset', 'horizon', 'lookback', 'profile', 'seed',
    'status', 'returncode', 'mse', 'mae',
    'mse_gain_pct_vs_original', 'mae_gain_pct_vs_original',
    'wall_seconds', 'gpu_total_used_peak_mib',
    'baseline_metadata', 'source_commit', 'patched_source_sha256',
    'training_epochs', 'recovery_source', 'job_directory'
]


def get_metrics(log: Path):
    if not log.is_file():
        return None
    s = log.read_text(encoding='utf-8', errors='replace')
    found = re.findall(r'\bmse\s*[:=]\s*([+\-\d.eE]+)\s*,?\s*\bmae\s*[:=]\s*([+\-\d.eE]+)', s, re.I)
    if not found:
        return None
    try:
        return float(found[-1][0]), float(found[-1][1])
    except ValueError:
        return None


def as_number(value):
    try:
        return float(value) if value is not None and value != '' else None
    except (TypeError, ValueError):
        return None


def extract_job(directory: Path, root: Path):
    rel = directory.relative_to(root)
    # Expected standard tree: paper/{dataset}/h{horizon}/{variant}
    parts = rel.parts
    if len(parts) != 4 or parts[0] != 'paper' or not re.fullmatch(r'h\d+', parts[2]):
        return None, f'Unrecognized directory layout: {rel}'
    _, ds, horizon, variant = parts
    if variant not in VALID_VARIANTS:
        return None, f'Unknown variant: {rel}'
    meta = directory / 'metadata.json'
    log = directory / 'train.log'
    record = {}
    source = []
    if meta.is_file():
        try:
            record = json.loads(meta.read_text(encoding='utf-8'))
            source.append('metadata')
        except (OSError, ValueError) as e:
            return None, f'Invalid metadata {meta}: {e}'
    measured = get_metrics(log)
    if log.exists():
        source.append('train.log')
    if not source:
        return None, f'No metadata or train.log: {rel}'

    row = dict(record)
    row.update(model='Paloma', variant=variant, dataset=ds,
               horizon=int(horizon[1:]), profile='paper',
               lookback=record.get('lookback', 96), seed=record.get('seed', 2024),
               job_directory=str(directory), recovery_source='+'.join(source))
    if measured is not None:
        old_mse, old_mae = as_number(record.get('mse')), as_number(record.get('mae'))
        if old_mse is not None and abs(old_mse - measured[0]) > 1e-6:
            print(f'[WARNING] metadata MSE disagrees with final train.log: {rel} ({old_mse} vs {measured[0]})')
        if old_mae is not None and abs(old_mae - measured[1]) > 1e-6:
            print(f'[WARNING] metadata MAE disagrees with final train.log: {rel} ({old_mae} vs {measured[1]})')
        row['mse'], row['mae'] = measured
    if as_number(row.get('mse')) is None or as_number(row.get('mae')) is None:
        row['status'] = 'incomplete_metrics'
    elif record.get('returncode') not in (None, 0):
        row['status'] = 'error'
    elif record.get('status') == 'error':
        row['status'] = 'error'
    elif meta.exists():
        row['status'] = record.get('status', 'recovered_metrics')
    else:
        row['status'] = 'recovered_from_log'
    if not meta.exists():
        row['returncode'] = ''  # logs alone cannot prove process exit code
    return row, None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', default='.', help='Directory containing experiment.py and runs/')
    p.add_argument('--out-dir', default='', help='Optional absolute/relative runs/paloma_functional_splice path')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    root = (Path(args.out_dir).expanduser().resolve() if args.out_dir
            else workspace / 'runs' / 'paloma_functional_splice')
    paper = root / 'paper'
    if not paper.is_dir():
        raise SystemExit(f'[ERROR] Not found: {paper}. Run from workspace root or use --workspace.')

    folders = sorted({p.parent for pattern in ('metadata.json', 'train.log')
                      for p in paper.rglob(pattern)})
    if not folders:
        raise SystemExit(f'[ERROR] No metadata.json/train.log found under {paper}')
    rows = []
    warnings = []
    for folder in folders:
        row, warning = extract_job(folder, root)
        if warning:
            warnings.append(warning)
        if row:
            rows.append(row)
    rows.sort(key=lambda r: (r['dataset'], r['horizon'],
                            ['original','olinear_channel','tq_channel','localconv'].index(r['variant'])))

    originals = {}
    for row in rows:
        if row['variant'] == 'original' and row['status'] in ('ok', 'recovered_metrics', 'recovered_from_log'):
            originals[(row['dataset'], row['horizon'], row['lookback'], row['seed'], row['profile'])] = row
    for row in rows:
        ref = originals.get((row['dataset'],row['horizon'],row['lookback'],row['seed'],row['profile']))
        row['baseline_metadata'] = str(Path(ref['job_directory'])/'metadata.json') if ref and (Path(ref['job_directory'])/'metadata.json').is_file() else ''
        for metric in ('mse','mae'):
            base = as_number(ref.get(metric)) if ref else None
            val = as_number(row.get(metric))
            row[f'{metric}_gain_pct_vs_original'] = round(100*(base-val)/base, 8) if base is not None and val is not None and base > 0 else ''
        if ref is not None and 'training_epochs' in row and 'training_epochs' in ref:
            if ref.get('training_epochs') != row.get('training_epochs'):
                warnings.append(f'Mismatched training_epochs: {row["dataset"]} H{row["horizon"]} {row["variant"]}')

    print(f'[SCAN] {paper}\n[FOUND] {len(rows)} job records, {len(warnings)} warnings')
    for r in rows:
        print('  ', r['dataset'], 'H'+str(r['horizon']), r['variant'], 'status='+str(r['status']),
              'MSE='+str(r.get('mse')), 'MAE='+str(r.get('mae')))
    for warning in warnings:
        print('[WARNING]', warning)
    if not rows:
        raise SystemExit('[ERROR] No supported jobs found; existing summary not touched')
    summary = root/'functional_splice_summary.csv'
    if args.dry_run:
        print('[DRY RUN] No files written. Would write:', summary)
        return
    if summary.exists():
        backup = root / ('functional_splice_summary.csv.bak')
        shutil.copy2(summary, backup)
        print('[BACKUP]', backup)
    tmp = root/'functional_splice_summary.csv.tmp'
    try:
        with tmp.open('w', newline='', encoding='utf-8-sig') as out:
            writer = csv.DictWriter(out, FIELDS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, summary)
    finally:
        if tmp.exists():
            tmp.unlink()
    print('[SAVED]', summary)
    print('[NOTE] Close/reopen summary CSV in VSCode if its editor shows an older snapshot.')


if __name__=='__main__':
    main()
