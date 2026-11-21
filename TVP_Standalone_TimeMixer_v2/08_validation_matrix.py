#!/usr/bin/env python3
"""Run fully matched TimeMixer+TVP experiments for named tasks.

Does not overwrite existing successful official baseline runs by default.
Uses current independent TimeMixer project, not a previous GTR workspace.
"""
import argparse, json, os, subprocess, sys, hashlib
from pathlib import Path

TASKS = [('ETTm1',96,2024),('ETTm1',192,2024),('ETTh1',96,2024),('Weather',96,2024)]
PROFILES = {
  # ETTm1 and Weather use values from project's training/runner.py.
  'ETTm1': {'data':'ETTm1','file':'ETTm1.csv','channels':7,'e_layers':2,'batch_size':16,'epochs':100,'patience':15,'factor':1},
  'ETTh1': {'data':'ETTh1','file':'ETTh1.csv','channels':7,'e_layers':2,'batch_size':16,'epochs':100,'patience':15,'factor':1},
  'Weather':{'data':'custom','file':'weather.csv','channels':21,'e_layers':3,'batch_size':128,'epochs':20,'patience':10,'factor':3},
}

def run_cmd(command, cwd, env=None, log=None):
    print('[RUN]', ' '.join(map(str,command)),flush=True)
    if log is None:
        subprocess.run(list(map(str,command)),cwd=cwd,env=env,check=True)
    else:
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w',encoding='utf8') as f:
            f.write('COMMAND: '+' '.join(map(str,command))+'\n');f.flush()
            p=subprocess.run(list(map(str,command)),cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
        if p.returncode:
            print('[ERROR] See:',log,flush=True)
            raise subprocess.CalledProcessError(p.returncode,command)
        print('[OK] log:',log,flush=True)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def run_task(root,dataset,h,seed,device,train_missing=True,force=False):
    from_path=root/'training/workspace/official'
    if not (from_path/'models/TimeMixer.py').exists():
        if not train_missing:raise FileNotFoundError('Official source missing. Use --train-missing.')
        run_cmd([sys.executable,'training/prepare.py'],root)
    d=PROFILES[dataset]; data=root/'data'/d['file']
    if not data.exists():raise FileNotFoundError(f'Missing dataset {data}; supply data/{d["file"]}.')
    out=root/'results'/f'official_{dataset}_h{h}_seed{seed}'
    extract=out/'extracted';compare=out/'compare'
    cfg=out/'baseline_train_config.json'
    src=from_path
    checkpoints=root/'training/checkpoints'
    cp_glob=[p for p in checkpoints.glob('*/checkpoint.pth') if f'{dataset}_96_{h}_s{seed}' in p.parent.name and '_TimeMixer_' in p.parent.name]
    # Use a valid checkpoint only if the corresponding project training record says it succeeded.
    legacy=root/'training/runs'/dataset/f'h{h}'/f'seed{seed}'/'original'/'train_config.json'
    cp=None
    if cfg.exists() and len(cp_glob)==1:
        prior=json.loads(cfg.read_text('utf8'))
        if prior.get('exit_code')==0 and prior.get('checkpoint_sha256')==sha(cp_glob[0]):cp=cp_glob[0]
    if cp is None and legacy.exists() and len(cp_glob)==1:
        prior=json.loads(legacy.read_text('utf8'))
        if prior.get('exit_code')==0:cp=cp_glob[0]
    if cp is None or force:
        if not train_missing:raise RuntimeError(f'No verified checkpoint for {dataset} h={h}. Rerun with --train-missing.')
        if device=='gpu':
            import torch
            if not torch.cuda.is_available():raise RuntimeError('Requested gpu but CUDA is not available')
        cmd=[sys.executable,'-u','run.py','--task_name','long_term_forecast','--is_training','1',
             '--model_id',f'{dataset}_96_{h}_s{seed}','--model','TimeMixer',
             '--data',d['data'],'--root_path',str(data.parent.resolve())+os.sep,'--data_path',data.name,
             '--features','M','--seq_len','96','--label_len','0','--pred_len',str(h),
             '--e_layers',str(d['e_layers']),'--enc_in',str(d['channels']),'--dec_in',str(d['channels']),
             '--c_out',str(d['channels']),'--d_model','16','--d_ff','32','--factor',str(d['factor']),
             '--down_sampling_layers','3','--down_sampling_window','2','--down_sampling_method','avg',
             '--batch_size',str(d['batch_size']),'--learning_rate','0.01',
             '--train_epochs',str(d['epochs']),'--patience',str(d['patience']),
             '--num_workers','0','--gpu','0','--itr','1','--des','DirectionStudy','--comment','matched',
             '--checkpoints',str(checkpoints.resolve())+os.sep]
        env=os.environ.copy();env['TIMEMIXER_SEED']=str(seed);env['TIMEMIXER_ENABLE_TEST']='0'
        log=out/'baseline_train.log'
        run_cmd(cmd,src,env=env,log=log)
        cp_glob=[p for p in checkpoints.glob('*/checkpoint.pth') if f'{dataset}_96_{h}_s{seed}' in p.parent.name and '_TimeMixer_' in p.parent.name]
        if len(cp_glob)!=1:raise RuntimeError(f'Expected exactly one checkpoint, found {cp_glob}')
        cp=cp_glob[0]
        cfg.parent.mkdir(parents=True,exist_ok=True)
        cfg.write_text(json.dumps({'cmd':cmd,'seed':seed,'device':device,'exit_code':0,'checkpoint_sha256':sha(cp),'data_sha256':sha(data),'training_protocol':'independent_v3_frozen_baseline'},indent=2))
        # Label extraction loader uses this exact command and checkpoint.
    else:
        print(f'[REUSE] {dataset} h{h} checkpoint: {cp}',flush=True)
        if not cfg.exists() and legacy.exists():cfg=legacy
    if not (extract/'val.npz').exists() or not (extract/'test.npz').exists() or force:
        command=[sys.executable,'-m','tvp_lab.extract_official',
                 '--family','timemixer','--official-root',str(src.resolve()),
                 '--config-json',str(cfg.resolve()),'--checkpoint',str(cp.resolve()),
                 '--data-root',str(data.parent.resolve()),'--data-file',data.name,
                 '--device','cuda' if device=='gpu' else 'cpu','--max-windows','0',
                 '--batch-size','16','--output-dir',str(extract.resolve())]
        run_cmd(command,root,log=out/'baseline_extract.log')
    if not (compare/'manifest.json').exists() or force:
        run_cmd([sys.executable,'-m','tvp_lab.evaluate','--val',extract/'val.npz',
                 '--test',extract/'test.npz','--outdir',compare],root,log=out/'baseline_tvp.log')
    struct=compare/'structural_trials'
    if not (struct/'structural_manifest.json').exists() or force:
        script=Path(__file__).resolve().parent/'06_tvp_structural_trials.py'
        run_cmd([sys.executable,script,'--val',extract/'val.npz','--test',extract/'test.npz',
                 '--manifest',compare/'manifest.json','--pred',compare/'predictions.npz',
                 '--outdir',struct],root,log=out/'selective_tvp.log')
    print('[COMPLETE]',dataset,h,seed,'result=',struct/'structural_metrics.csv',flush=True)
    return str(struct/'structural_metrics.csv')

def parse_tasks(value):
    if not value:return TASKS
    out=[]
    for part in value.split(','):
        toks=part.split(':')
        if len(toks)!=3:raise ValueError('Use dataset:horizon:seed, e.g. ETTm1:96:2024')
        ds,h,seed=toks[0],int(toks[1]),int(toks[2])
        if ds not in PROFILES or h not in (96,192):raise ValueError(f'Unsupported task: {part}')
        out.append((ds,h,seed))
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project-root',default='.',help='Standalone v2 root with training/, tvp_lab/, data/ and results/')
    p.add_argument('--tasks',default='',help='ETTm1:96:2024,ETTm1:192:2024,ETTh1:96:2024')
    p.add_argument('--device',choices=['gpu'],default='gpu',help='Full official training/extraction needs GPU; CPU tests use test script')
    p.add_argument('--train-missing',action='store_true',help='Allow official baseline training when checkpoint absent')
    p.add_argument('--force',action='store_true',help='Re-run existing experiments intentionally')
    a=p.parse_args();root=Path(a.project_root).resolve()
    if not (root/'tvp_lab/extract_official.py').exists():raise RuntimeError('Run in the standalone project root, not patch folder')
    errors=[]
    for ds,h,seed in parse_tasks(a.tasks):
        try:run_task(root,ds,h,seed,a.device,a.train_missing,a.force)
        except Exception as exc:
            print(f'[TASK FAILED] {ds} h={h} seed={seed} : {exc}',flush=True)
            errors.append(f'{ds}:h{h}:seed{seed}: {exc}')
    if errors:
        print('\nFAILED TASKS:\n'+'\n'.join(errors))
        raise SystemExit(1)
    print('[SUCCESS] all requested tasks finished')
if __name__=='__main__':main()
