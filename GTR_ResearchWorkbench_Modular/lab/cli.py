"""VS Code-friendly CLI: install, launch (detached), run (foreground), stop, status."""
import argparse
import json
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from .paths import ROOT, RUNS
from .config import config_for, DATASETS, MODEL_FILES
from .source import install_official, adopt_official, verify
from .runner import make_run, run_worker, update


def start(dataset='ETTm1',model='GTR',horizon=96,seed=2024,mode='smoke',detach=False):
    cfg=config_for(dataset,model,horizon,seed,mode)
    run_id=__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:7]
    RUNS.mkdir(parents=True,exist_ok=True)
    run_path=make_run(RUNS/run_id,cfg)
    if detach:
        p=subprocess.Popen([sys.executable,'-m','lab.cli','worker',run_id],cwd=ROOT,start_new_session=True,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        update(run_path,worker_pid=p.pid)
        print(f'Launched {run_id}; status: {run_path/"status.json"}')
    else:
        run_worker(run_path)
        print(f'Run {run_id}: {json.loads((run_path/"status.json").read_text())["state"]}')
    return run_id

def stop(run_id):
    if not run_id.isalnum() and not all(c.isalnum() or c=='_' for c in run_id):raise ValueError('Invalid run ID')
    path=RUNS/run_id
    p=path/'status.json'
    if not p.is_file():raise FileNotFoundError(p)
    status=json.loads(p.read_text())
    if status.get('state') not in ('running','stopping','queued'):
        print('Task already inactive');return
    update(path,state='stopping')
    pid=status.get('child_pid')
    if pid:
        try:
            # Child is started as a new process group; signal only that job.
            if os.getpgid(pid)==pid: os.killpg(pid,signal.SIGTERM)
        except ProcessLookupError:pass
    elif status.get('worker_pid'):
        try:os.kill(status['worker_pid'],signal.SIGTERM)
        except ProcessLookupError:pass
    print(f'Sent stop to {run_id}')


def main():
    p=argparse.ArgumentParser(description='GTR research lab; exact baseline flags come from author scripts.')
    sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('install-official'); ad=sub.add_parser('adopt-official');ad.add_argument('path')
    for op in ('launch','run'):
        s=sub.add_parser(op)
        s.add_argument('--dataset',choices=DATASETS,default='ETTm1')
        s.add_argument('--model',choices=MODEL_FILES,default='GTR')
        s.add_argument('--horizon',type=int,choices=[96,192,336,720],default=96)
        s.add_argument('--seed',type=int,default=2024)
        s.add_argument('--mode',choices=['paper','smoke'],default='smoke')
    w=sub.add_parser('worker');w.add_argument('id')
    s=sub.add_parser('stop');s.add_argument('id')
    sub.add_parser('status')
    sub.add_parser('verify-official')
    args=p.parse_args()
    if args.action=='install-official':print(install_official())
    elif args.action=='adopt-official':print(adopt_official(args.path))
    elif args.action=='verify-official':print('verified:',verify(ROOT/'vendor'/'GTR'))
    elif args.action in ('launch','run'):
        start(args.dataset,args.model,args.horizon,args.seed,args.mode,detach=args.action=='launch')
    elif args.action=='worker':
        if not args.id.replace('_','').isalnum():raise ValueError('Invalid run id')
        run_worker(RUNS/args.id)
    elif args.action=='stop':stop(args.id)
    elif args.action=='status':
        if not RUNS.exists():print('No runs yet');return
        for pth in sorted(RUNS.glob('*/status.json'),reverse=True)[:20]:
            q=json.loads(pth.read_text());print(pth.parent.name,q.get('state'),q.get('epoch'),q.get('val_loss'))
if __name__=='__main__':main()
