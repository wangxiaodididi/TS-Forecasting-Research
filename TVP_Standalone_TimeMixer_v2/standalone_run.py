#!/usr/bin/env python3
"""Self-contained launcher. No previous user project directories are referenced.

Workflow: obtain pinned official TimeMixer, train original if missing, extract
frozen FMM forecasts on ETTm1, compare TVP on full validation/test.
Not executed on GPU by the package author: must run on user's GPU server.
"""
import os,sys,subprocess,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
TRAIN=ROOT/'training'
DATA=ROOT/'data'
FAMILY='timemixer'
DATASET=os.environ.get('DATASET','ETTm1')
H=int(os.environ.get('H','96'))
SEED=int(os.environ.get('SEED','2024'))
DEVICE=os.environ.get('DEVICE','gpu').lower()
MAX_WINDOWS=int(os.environ.get('MAX_WINDOWS','0'))
assert DATASET in ('ETTm1',), 'v2 full runner is validated/configured for ETTm1 only'
assert H in (96,192), 'Use H=96 or 192'

def call(cmd,env=None,cwd=ROOT):
    print('[COMMAND]',' '.join(map(str,cmd)),flush=True)
    subprocess.run(list(map(str,cmd)),check=True,cwd=str(cwd),env=env)

def preflight():
    try:
        import torch,numpy,pandas,sklearn,einops,reformer_pytorch,sktime
    except ImportError as e:
        raise SystemExit(f'Missing Python dependency: {e}. Install deps using bash 00_安装依赖.sh')
    print('Torch',torch.__version__,'CUDA available:',torch.cuda.is_available(),flush=True)
    if DEVICE=='gpu' and not torch.cuda.is_available():
        raise SystemExit('DEVICE=gpu but CUDA unavailable. Check GPU and torch environment. No silent CPU fallback.')
    if not (DATA/'ETTm1.csv').exists():
        raise SystemExit('ETTm1.csv missing in project data/ directory')

def ensure_source():
    repo=TRAIN/'workspace/official'
    if (repo/'models/TimeMixer.py').is_file():
        print('[READY] Pinned official source',repo,flush=True)
        return repo
    if os.environ.get('OFFICIAL_ZIP'):
        call([sys.executable,TRAIN/'prepare.py','--archive',os.environ['OFFICIAL_ZIP']])
    else:
        call([sys.executable,TRAIN/'prepare.py'])
    if not (repo/'models/TimeMixer.py').exists():
        raise SystemExit('Official source setup failed.')
    return repo

def find_checkpoint():
    candidates=sorted((TRAIN/'checkpoints').glob('*/checkpoint.pth'))
    matched=[]
    for p in candidates:
        s=p.parent.name
        if DATASET in s and 'TimeMixer' in s and str(H) in s and f's{SEED}' in s:
            matched.append(p)
    if len(matched)>1:
        raise SystemExit(f'Multiple checkpoint candidates: {matched}; clean duplicates, then retry')
    return matched[0] if matched else None

def ensure_train():
    checkpoint=find_checkpoint()
    config=TRAIN/'runs'/DATASET/f'h{H}'/f'seed{SEED}'/'original'/'train_config.json'
    if checkpoint and config.exists():
        conf=json.loads(config.read_text())
        if conf.get('exit_code')==0:
            print('[READY] Existing trained checkpoint:',checkpoint,flush=True)
            return checkpoint,config
    print('[TRAIN] Original pinned official TimeMixer, dataset',DATASET,'H',H,'seed',SEED,flush=True)
    call([sys.executable,TRAIN/'runner.py','train','--dataset',DATASET,'--horizon',H,'--seed',SEED,
          '--variant','original','--device',DEVICE,'--data-root',DATA,'--num-workers',0,'--force'])
    checkpoint=find_checkpoint()
    if not checkpoint or not config.exists():
        raise SystemExit('Training completed but checkpoint or config not found; inspect training/runs')
    conf=json.loads(config.read_text())
    if conf.get('exit_code')!=0: raise SystemExit('Training not successful; see training/runs')
    print('[TRAINED]',checkpoint,flush=True)
    return checkpoint,config

def compare(checkpoint,config,repo):
    out=ROOT/'results'/f'official_{DATASET}_h{H}_seed{SEED}'
    extract=out/'extracted'
    cmd=[sys.executable,'-m','tvp_lab.extract_official',
           '--family','timemixer','--official-root',repo,'--config-json',config,
           '--checkpoint',checkpoint,'--data-root',DATA,'--data-file',f'{DATASET}.csv',
           '--device','cuda' if DEVICE=='gpu' else 'cpu',
           '--max-windows',MAX_WINDOWS,'--batch-size',16,'--output-dir',extract]
    call(cmd)
    call([sys.executable,'-m','tvp_lab.evaluate','--val',extract/'val.npz',
         '--test',extract/'test.npz','--outdir',out/'compare'])
    print('\n[FINISHED] Real TimeMixer comparison:',out/'compare/metrics.csv',flush=True)

def main():
    preflight()
    repo=ensure_source()
    cp,cfg=ensure_train()
    compare(cp,cfg,repo)
if __name__=='__main__':main()
