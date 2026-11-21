#!/usr/bin/env python3
"""Execute matched original/reverse official TimeMixer training; validation first."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'workspace' / 'official'
RUNS = ROOT / 'runs'

# Values taken from the two author's official '*_unify.sh' scripts, not searched.
DATASETS = {
  'ETTm1': dict(data='ETTm1', data_path='ETTm1.csv', enc_in=7, c_out=7, dec_in=7,
                e_layers=2, batch_size=16, train_epochs=100, patience=15,
                learning_rate=0.01, factor=1),
  'Weather': dict(data='custom', data_path='weather.csv', enc_in=21, c_out=21, dec_in=21,
                  e_layers=3, batch_size=128, train_epochs=20, patience=10,
                  learning_rate=0.01, factor=3),
}


def candidate_dirs():
    current = Path.cwd()
    parent = ROOT.parent
    return [ROOT.parent / 'data', current / 'data', parent / 'data', current / 'dataset']


def locate(dataset, dataroot):
    item = DATASETS[dataset]['data_path']
    bases = [Path(dataroot)] if dataroot else candidate_dirs()
    if dataroot:
        bases += candidate_dirs()
    for base in bases:
        for target in [base / item, base / 'ETT-small' / item, base / 'weather' / item]:
            if target.is_file():
                return str(target.parent.resolve()) + os.sep, item
    raise FileNotFoundError(f'Missing {item}; set --data-root to the directory containing it (checked {bases})')


def command(a):
    conf = DATASETS[a.dataset]
    data_root, data_path = locate(a.dataset, a.data_root)
    model = 'TimeMixer' if a.variant == 'original' else 'TimeMixerReverse'
    run_name = f'{a.dataset}_96_{a.horizon}_s{a.seed}'
    cmd = [sys.executable, '-u', 'run.py',
           '--task_name', 'long_term_forecast',
           '--is_training', '1' if a.action == 'train' else '0',
           '--model_id', run_name,
           '--model', model,
           '--root_path', data_root,
           '--data_path', data_path,
           '--data', conf['data'],
           '--features', 'M',
           '--seq_len', '96',
           '--label_len', '0',
           '--pred_len', str(a.horizon),
           '--e_layers', str(conf['e_layers']),
           '--enc_in', str(conf['enc_in']),
           '--dec_in', str(conf['dec_in']),
           '--c_out', str(conf['c_out']),
           '--d_model', '16',
           '--d_ff', '32',
           '--factor', str(conf['factor']),
           '--down_sampling_layers', '3',
           '--down_sampling_window', '2',
           '--down_sampling_method', 'avg',
           '--batch_size', str(conf['batch_size']),
           '--learning_rate', str(conf['learning_rate']),
           '--train_epochs', str(conf['train_epochs']),
           '--patience', str(conf['patience']),
           '--num_workers', str(a.num_workers),
           '--gpu', '0',
           '--itr', '1',
           '--des', 'DirectionStudy',
           '--comment', 'matched',
           '--checkpoints', str((ROOT / 'checkpoints').resolve()) + os.sep]
    return cmd


def dir_for(a):
    return RUNS / a.dataset / f'h{a.horizon}' / f'seed{a.seed}' / a.variant


def run(a):
    if not (SRC / 'models' / 'TimeMixer.py').is_file():
        raise RuntimeError('Run training/prepare.py first to create pinned official TimeMixer source.')
    cmd = command(a)
    folder = dir_for(a)
    folder.mkdir(parents=True, exist_ok=True)
    logfile = folder / ('train.log' if a.action == 'train' else 'test.log')
    if logfile.exists() and not a.force:
        print(f'[SKIP] {logfile} exists. Use --force to rerun.', flush=True)
        return 0
    env = dict(os.environ, TIMEMIXER_SEED=str(a.seed), TIMEMIXER_ENABLE_TEST='0')
    env['PYTHONUNBUFFERED'] = '1'
    if a.device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = '-1'
    elif a.device == 'gpu':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA not available, refusing to silently use CPU for GPU matrix')
    # --is_training 0 officially calls exp.test with best checkpoint; no retraining.
    meta = {'action':a.action,'dataset':a.dataset,'horizon':a.horizon,'seed':a.seed,
            'variant':a.variant,'device':a.device,'cmd':cmd,'data_root':a.data_root,
            'created_utc':datetime.now(timezone.utc).isoformat()}
    (folder / f'{a.action}_config.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    print('[START]', a.action, a.dataset, a.horizon, a.seed, a.variant, 'device=',a.device, flush=True)
    start = time.time()
    with logfile.open('w', encoding='utf8') as out:
        out.write('COMMAND: '+' '.join(cmd)+'\n');out.flush()
        p = subprocess.Popen(cmd, cwd=str(SRC), env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            out.write(line)
            if ('Epoch:' in line and 'Vali Loss:' in line) or ('mse:' in line) or ('Early stopping' in line) or ('Traceback' in line):
                print(line.rstrip(), flush=True)
        p.wait()
    meta['exit_code'] = p.returncode
    meta['elapsed_sec'] = round(time.time()-start,2)
    (folder / f'{a.action}_config.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    print('[END]', 'OK' if p.returncode == 0 else 'FAILED', meta['elapsed_sec'], 'sec', 'log:', logfile, flush=True)
    return p.returncode


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['train','test'])
    p.add_argument('--dataset', choices=sorted(DATASETS), required=True)
    p.add_argument('--horizon', type=int, choices=[96,192],required=True)
    p.add_argument('--seed',type=int,default=2024)
    p.add_argument('--variant',choices=['original','reverse'],required=True)
    p.add_argument('--device',choices=['cpu','gpu'],default='gpu')
    p.add_argument('--data-root',default='')
    p.add_argument('--num-workers',type=int,default=4)
    p.add_argument('--force',action='store_true')
    return p.parse_args()


if __name__=='__main__':
    try:
        raise SystemExit(run(parse_args()))
    except Exception as e:
        print('[ERROR]',e,file=sys.stderr)
        raise SystemExit(2)
