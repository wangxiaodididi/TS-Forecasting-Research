"""Run one pinned-source experiment and stream official training logs without altering optimizers."""
import hashlib
import json
import platform
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from .source import snapshot, UPSTREAM_COMMIT
from .config import command_args
from .monitor import parse_line, append_jsonl, append_epoch_row, poll_gpu
from .paths import DATA

def put_json(path,value):
    path=Path(path);p=path.with_suffix('.tmp')
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    os.replace(p,path)

def utc():return datetime.now(timezone.utc).isoformat()

def make_run(run_path,cfg):
    run_path=Path(run_path);run_path.mkdir(parents=True,exist_ok=False)
    put_json(run_path/'config.json',cfg)
    put_json(run_path/'status.json',{'state':'queued','created_at':utc(),'epoch':0,'model':cfg['model'],'dataset':cfg['dataset'],'mode':cfg['mode']})
    return run_path

def update(run_path,**values):
    file=run_path/'status.json';current=json.loads(file.read_text(encoding='utf-8')) if file.exists() else {}
    current.update(values);current['updated_at']=utc();put_json(file,current)

def gpu_watch(run_path, stop):
    while not stop.wait(4):
        for sample in poll_gpu():append_jsonl(run_path/'gpu.jsonl',sample)

def run_worker(run_path):
    started_monotonic=time.monotonic()
    run_path=Path(run_path).resolve()
    cfg=json.loads((run_path/'config.json').read_text(encoding='utf-8'))
    completed=False; child=None
    try:
        src=snapshot(run_path/'source',cfg['model'],cfg['mode'])
        if not (DATA/cfg['data_path']).is_file():
            raise FileNotFoundError(f"Dataset missing: {DATA/cfg['data_path']}")
        args=[sys.executable,'-u','run.py',*command_args(cfg)]
        environment={'python':sys.version,'platform':platform.platform(),'python_executable':sys.executable}
        try:
            import torch
            environment.update(torch_version=torch.__version__,cuda_version=torch.version.cuda,cuda_available=torch.cuda.is_available())
            if torch.cuda.is_available():environment['gpu_name']=torch.cuda.get_device_name(0)
        except Exception as exc:environment['torch_probe_error']=str(exc)
        environment['dataset_sha256']=hashlib.sha256((DATA/cfg['data_path']).read_bytes()).hexdigest()
        environment['source_sha256']={p:hashlib.sha256((src/p).read_bytes()).hexdigest() for p in ('run.py','exp/exp_main.py','models/GTR.py')}
        put_json(run_path/'environment.json',environment)
        put_json(run_path/'provenance.json',{'upstream_repository':'https://github.com/macovaseas/GTR','upstream_commit':UPSTREAM_COMMIT,
                    'immutable_official_baseline':cfg['model']=='GTR' and cfg['mode']=='paper',
                    'extension_patched_into_snapshot':cfg['model']!='GTR',
                    'smoke_modified_data_provider':cfg['mode']=='smoke',
                    'official_experiment_params':cfg,'command':args,'cwd':str(src)})
        update(run_path,state='running',started_at=utc(),command=args)
        env=os.environ.copy();env['PYTHONUNBUFFERED']='1'
        if cfg['mode']=='smoke':env['CUDA_VISIBLE_DEVICES']='' # force CPU without changing original PyTorch code
        stop=threading.Event()
        gpu_thread=threading.Thread(target=gpu_watch,args=(run_path,stop),daemon=True);gpu_thread.start()
        with (run_path/'console.log').open('w',encoding='utf-8',buffering=1) as fout:
            child=subprocess.Popen(args,cwd=src,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  text=True,encoding='utf-8',errors='replace',bufsize=1,start_new_session=True)
            update(run_path,child_pid=child.pid)
            for line in child.stdout:
                fout.write(line)
                event=parse_line(line)
                if event:
                    event['timestamp']=utc();append_jsonl(run_path/'events.jsonl',event)
                    if event['type']=='epoch':
                        append_epoch_row(run_path/'loss.csv',event)
                        update(run_path,epoch=event['epoch'],train_loss=event['train_loss'],val_loss=event['val_loss'],
                              steps=event['steps'],expected_epochs=cfg['train_epochs'])
                    if event['type']=='final_test':update(run_path,final_test_mse=event['test_mse'],final_test_mae=event['test_mae'])
        if child.stdout is not None:child.stdout.close()
        rc=child.wait()
        stop.set();gpu_thread.join(timeout=5)
        s=json.loads((run_path/'status.json').read_text(encoding='utf-8'))
        if s.get('state')=='stopping':update(run_path,state='stopped',exit_code=rc,finished_at=utc(),duration_sec=round(time.monotonic()-started_monotonic,2))
        elif rc==0:update(run_path,state='completed',exit_code=0,finished_at=utc(),duration_sec=round(time.monotonic()-started_monotonic,2))
        else:update(run_path,state='failed',exit_code=rc,finished_at=utc(),duration_sec=round(time.monotonic()-started_monotonic,2))
        completed=True
    except Exception as exc:
        with (run_path/'worker_error.log').open('a',encoding='utf-8') as f:traceback.print_exc(file=f)
        update(run_path,state='failed',error=str(exc),finished_at=utc(),duration_sec=round(time.monotonic()-started_monotonic,2))
    finally:
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    return completed
