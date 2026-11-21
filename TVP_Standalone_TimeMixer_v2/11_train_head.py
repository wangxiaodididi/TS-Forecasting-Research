#!/usr/bin/env python3
"""Train one independent *prediction-head* modification of official TimeMixer.

First experiments: --mode linear vs --mode nonlinear; PDM, FMM are identical.
No TVP used in this training step; avoids stacked-attribution ambiguity.
"""
import argparse,os,sys,subprocess,json,hashlib
from pathlib import Path
from types import SimpleNamespace

PATCH=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def run(args):
    root=Path(args.project_root).resolve();sys.path.insert(0,str(root))
    from training import runner
    if args.dataset=='ETTh1':
        runner.DATASETS['ETTh1']=dict(data='ETTh1',data_path='ETTh1.csv',enc_in=7,c_out=7,dec_in=7,
            e_layers=2,batch_size=16,train_epochs=100,patience=15,learning_rate=.01,factor=1)
    if args.dataset not in runner.DATASETS:raise ValueError('Unsupported dataset')
    src=root/'training/workspace/official'
    if not (src/'models/TimeMixer.py').is_file():raise FileNotFoundError('Pinned official source not found. Run 02_full_experiment.sh first.')
    sys.path.insert(0,str(PATCH))
    from importlib import import_module
    installer=import_module('10_install_head_adapter')
    installer.patch(src,PATCH)
    a=SimpleNamespace(dataset=args.dataset,horizon=args.horizon,seed=args.seed,
        variant='original',action='train',device='gpu',data_root=str((root/'data').resolve()),num_workers=0)
    cmd=runner.command(a)
    def set_flag(flag,value):
        try:i=cmd.index(flag)
        except ValueError:raise RuntimeError('Runner changed flag '+flag)
        cmd[i+1]=str(value)
    set_flag('--model','TimeMixerResidualHead')
    set_flag('--des','TVPHeadTrial')
    set_flag('--comment',f'head_{args.mode}_r{args.rank}_s{args.scale}')
    checkpoint_dir=root/'training/head_experiments/checkpoints'
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    set_flag('--checkpoints',str(checkpoint_dir.resolve())+os.sep)
    trial=f'{args.dataset}_h{args.horizon}_seed{args.seed}_{args.mode}_rank{args.rank}'
    folder=root/'results/head_trials'/trial
    folder.mkdir(parents=True,exist_ok=True)
    config_path=folder/'train_config.json'
    existing=sorted(p for p in checkpoint_dir.glob('*/checkpoint.pth') if f'{args.dataset}_96_{args.horizon}_s{args.seed}' in p.parent.name and f'head_{args.mode}_r{args.rank}_s{args.scale}' in p.parent.name and '_TimeMixerResidualHead_' in p.parent.name)
    if len(existing)>1:raise RuntimeError('Multiple matching head checkpoints: '+str(existing))
    if existing and not args.force and config_path.exists():
        prior=json.loads(config_path.read_text('utf8'))
        if prior.get('exit_code')==0 and sha(existing[0])==prior.get('checkpoint_sha256'):
            print('[REUSE] verified head checkpoint:',existing[0]);return
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA not available: the official head training is GPU only')
    env=os.environ.copy();env.update(TIMEMIXER_SEED=str(args.seed),TIMEMIXER_ENABLE_TEST='0',
        TIMEMIXER_HEAD_MODE=args.mode,TIMEMIXER_HEAD_RANK=str(args.rank),TIMEMIXER_HEAD_SCALE=str(args.scale))
    meta={'dataset':args.dataset,'horizon':args.horizon,'seed':args.seed,'mode':args.mode,'rank':args.rank,
        'scale':args.scale,'cmd':cmd,'architecture_note':'PDM/FMM unchanged; only predict_layers altered'}
    config_path.write_text(json.dumps(meta,indent=2),encoding='utf8')
    with (folder/'train.log').open('w',encoding='utf8') as f:
        f.write('COMMAND: '+' '.join(cmd)+'\n');f.flush()
        rc=subprocess.run(cmd,cwd=src,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
    meta['exit_code']=rc
    checkpoint=sorted(p for p in checkpoint_dir.glob('*/checkpoint.pth') if f'{args.dataset}_96_{args.horizon}_s{args.seed}' in p.parent.name and f'head_{args.mode}_r{args.rank}_s{args.scale}' in p.parent.name and '_TimeMixerResidualHead_' in p.parent.name)
    if rc or len(checkpoint)!=1:
        meta['error']='See train.log; no valid checkpoint was confirmed'
        config_path.write_text(json.dumps(meta,indent=2),encoding='utf8')
        raise RuntimeError(f'Head training error rc={rc}, checkpoints={checkpoint}. Log: {folder/"train.log"}')
    meta['checkpoint']=str(checkpoint[0]);meta['checkpoint_sha256']=sha(checkpoint[0])
    config_path.write_text(json.dumps(meta,indent=2),encoding='utf8')
    print('[TRAINED]',trial,'checkpoint:',checkpoint[0]);print('Run 12_extract_head.py with --trial',trial)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project-root',default='.')
    p.add_argument('--dataset',choices=['ETTm1','ETTh1','Weather'],default='ETTm1')
    p.add_argument('--horizon',choices=[96,192],type=int,default=96)
    p.add_argument('--seed',type=int,default=2024)
    p.add_argument('--mode',choices=['linear','nonlinear'],default='nonlinear')
    p.add_argument('--rank',type=int,default=8)
    p.add_argument('--scale',type=float,default=.1)
    p.add_argument('--force',action='store_true')
    a=p.parse_args();run(a)
if __name__=='__main__':main()
