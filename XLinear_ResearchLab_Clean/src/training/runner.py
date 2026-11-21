"""正式训练唯一主入口：复用作者原始训练器，模块扩展通过 registry 注入，监控仅日志输出。

运行方式：python -m src.training.runner --mode full
测试指标由作者官方 exp/exp_main.py 输出，采用 MSE/MAE 原评价协议。
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid
from src.core.config import ROOT,read_config,make_jobs
from src.runtime.bootstrap import UPSTREAM,verify_source,prepare_datasets
from src.training.results import atomic_json,refresh
from src.training.telemetry import MetricParser, GpuSampler

RX_RESULT=re.compile(r'mse:([0-9.eE+-]+),\s*mae:([0-9.eE+-]+)')
RX_EPOCH=re.compile(r'Epoch:\s*(\d+), Steps:\s*(\d+)\s*\|\s*Train Loss:')


def command_for(job:dict,dir:Path,cfg:dict,run_id:str):
    p=job['profile'];d=job['dataset'];h=job['horizon'];s=job['seed'];v=job['variant']
    model_id=f"lab_{run_id}_{d}_H{h}_{v}_s{s}"
    args=['--random_seed',str(s),'--is_training','1','--root_path',str(UPSTREAM/'dataset'),
          '--data_path',p['file'],'--model_id',model_id,'--model','XLinear','--data',p['data'],
          '--features','M','--seq_len','96','--pred_len',str(h),'--enc_in',str(p['channels']),
          '--d_model',str(p['d_model']),'--t_ff',str(p['t_ff']),'--c_ff',str(p['c_ff']),
          '--t_dropout',str(p['t_dropout']),'--c_dropout',str(p['c_dropout']),
          '--head_dropout',str(p['head_dropout']),'--embed_dropout',str(p['embed_dropout']),
          '--patience',str(p['patience']),'--train_epochs',str(p['epochs']),
          '--batch_size',str(p['batch_size']),'--learning_rate',str(p['learning_rate']),
          '--num_workers',str(cfg['runtime'].get('workers',10)),
          '--itr','1','--des','Exp','--lradj','type3','--usenorm','1',
          '--checkpoints',str(dir/'checkpoints')]
    # 所有模型统一经过薄封装：调用作者原版 run_longExp，不再为 official/variants 走两套不同训练主线。
    # monitor-interval 仅在内存增加 stdout 日志；设置为 0 可完全关闭监控代码注入。
    return [sys.executable,'-u',str(ROOT/'src/training/official_entry.py'),
            '--variant',v,'--upstream',str(UPSTREAM),
            '--context-ratio',str(cfg['runtime']['context_ratio']),
            '--monitor-batch-interval',str(cfg.get('monitoring',{}).get('batch_interval',10))]+args


def run(mode, cfg, jobs,run_dir):
    verify_source(cfg)
    checks=prepare_datasets(cfg)
    import torch
    if cfg['runtime']['device']!='cuda' or not torch.cuda.is_available():
        raise RuntimeError('GPU 正式训练要求当前 Python 环境可以使用 CUDA；不允许静默退回 CPU。')
    if cfg['runtime'].get('precision','fp32')!='fp32':
        raise ValueError('官方默认训练精度 FP32。本研究版本不接受未经审核的训练精度变化。')
    if run_dir.exists():raise FileExistsError(f'输出目录已存在: {run_dir}')
    run_dir.mkdir(parents=True)
    run_id=run_dir.name
    manifest=dict(mode=mode,source_commit=cfg['source']['commit'],
        model_git_blob=cfg['source']['model_git_blob'],
        dataset_sha256=checks,device=torch.cuda.get_device_name(0),torch=torch.__version__,
        python=sys.version,configuration=cfg,jobs=jobs,started_at=datetime.now().isoformat())
    atomic_json(run_dir/'manifest.json',manifest)
    progress=dict(status='running',pid=os.getpid(),total=len(jobs),completed=0,
                  current=None,epoch=0,batch=0,steps_per_epoch=0,train_loss=None,val_loss=None,
                  started_at=manifest['started_at'])
    atomic_json(run_dir/'progress.json',progress)
    try:
        for index,job in enumerate(jobs,1):
            job_id=f"{job['dataset']}_H{job['horizon']}_{job['variant']}_s{job['seed']}"
            folder=run_dir/'jobs'/job_id
            folder.mkdir(parents=True,exist_ok=False)
            # 上游相对输出路径被隔离到单次作业的 workspace；不污染 vendor 源码。
            workspace=folder/'workspace';workspace.mkdir()
            cmd=command_for(job,folder,cfg,run_id)
            atomic_json(folder/'command.json',dict(command=cmd,job=job))
            progress.update(current=job_id,index=index,epoch=0,batch=0,steps_per_epoch=0,
                            train_loss=None,val_loss=None,epoch_max=job['profile']['epochs'])
            atomic_json(run_dir/'progress.json',progress)
            print(f'\n[{index}/{len(jobs)}] {job_id} | epochs <= {job["profile"]["epochs"]}',flush=True)
            matches=[]
            # 前台终端提供 tqdm Batch 进度条；后台只写 CSV/日志和网页，不输出重复进度字符。
            bar=None
            try:
                from tqdm import tqdm
            except ImportError:
                tqdm=None
            bar_epoch=0
            bar_last=0
            metrics=MetricParser(folder)
            gpu=GpuSampler(folder,float(cfg.get('monitoring',{}).get('gpu_interval_seconds',5)))
            process=None
            try:
                gpu.start()
                with (folder/'train.log').open('w',encoding='utf-8',buffering=1) as logfile:
                    process=subprocess.Popen(cmd,cwd=workspace,stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,text=True,bufsize=1,errors='replace',start_new_session=True)
                    assert process.stdout is not None
                    for line in process.stdout:
                        logfile.write(line)
                        event=metrics.consume(line)
                        if event:
                            progress.update(epoch=metrics.latest_epoch,batch=metrics.latest_batch,
                                steps_per_epoch=metrics.steps_per_epoch,
                                train_loss=metrics.last_train,val_loss=metrics.last_val)
                            atomic_json(run_dir/'progress.json',progress)
                        if event=='batch' and sys.stdout.isatty() and tqdm is not None:
                            from src.training.telemetry import BATCH
                            match=BATCH.search(line)
                            if match:
                                ep,bidx,btotal=map(int,match.group(1,2,3))
                                if bar is None or bar_epoch!=ep:
                                    if bar is not None:bar.close()
                                    bar=tqdm(total=btotal,desc=f'{job_id} Epoch {ep}',leave=False,dynamic_ncols=True)
                                    bar_epoch=ep;bar_last=0
                                bar.update(max(0,bidx-bar_last));bar_last=bidx
                                bar.set_postfix(loss=f'{float(match.group(4)):.5f}')
                        if event == 'epoch' and bar is not None:
                            bar.close();bar=None
                        if event == 'epoch' or ('mse:' in line) or 'Traceback' in line or 'Early stopping' in line or 'Error' in line:
                            print(line.rstrip(),flush=True)
                        matches.extend(RX_RESULT.findall(line))
                    returncode=process.wait()
            except BaseException:
                if process is not None and process.poll() is None:
                    import signal
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL)
                        process.wait()
                raise
            finally:
                if bar is not None:bar.close()
                gpu.close()
                metrics.close()
            if returncode or not matches:
                atomic_json(folder/'result.json',dict(dataset=job['dataset'],horizon=job['horizon'],
                            seed=job['seed'],variant=job['variant'],status='failed',returncode=returncode))
                raise RuntimeError(f'训练异常 {job_id} exit={returncode}; 日志 {folder/"train.log"}')
            mse,mae=map(float,matches[-1]);
            # 从作者官方保存的最优 checkpoint 精确计算参数量。
            checkpoints=list((folder/'checkpoints').glob('**/checkpoint.pth'))
            if len(checkpoints)!=1:
                raise RuntimeError(f'{job_id}: 预期恰好1个作者原版 checkpoint，实际找到 {len(checkpoints)} 个')
            import torch
            state_dict=torch.load(checkpoints[0],map_location='cpu',weights_only=True)
            params=sum(t.numel() for t in state_dict.values() if hasattr(t,'numel'))
            result=dict(dataset=job['dataset'],horizon=job['horizon'],seed=job['seed'],
                variant=job['variant'],mse=mse,mae=mae,status='completed',
                job_dir=str(folder),params=params)
            atomic_json(folder/'result.json',result)
            progress['completed']=index
            progress['batch']=0
            atomic_json(run_dir/'progress.json',progress)
            refresh(run_dir)
            print(f'完成 {job_id}: MSE={mse:.6f} MAE={mae:.6f}',flush=True)
        progress['status']='completed'
    except KeyboardInterrupt:
        progress['status']='interrupted'
        raise
    except Exception:
        progress['status']='failed'
        raise
    finally:
        atomic_json(run_dir/'progress.json',progress)
        refresh(run_dir)
        print('结果目录:',run_dir/'results',flush=True)


def main():
    parser=argparse.ArgumentParser(description='XLinear 官方协议 + 可扩展模块对照')
    parser.add_argument('--mode',choices=['pilot','full'],default='full')
    parser.add_argument('--plan',action='store_true',help='只显示执行计划，不启动训练')
    parser.add_argument('--datasets',help='例如 ETTh1,Weather')
    parser.add_argument('--horizons',help='例如 96,192')
    parser.add_argument('--variants',help='例如 official,dyn_token')
    parser.add_argument('--seeds',help='例如 2025,2026,2027')
    args=parser.parse_args()
    cfg=read_config()
    kwargs={}
    for name in ('datasets','horizons','variants','seeds'):
        x=getattr(args,name)
        if x: kwargs[name]=[int(z) if name in ('horizons','seeds') else z for z in x.split(',')]
    jobs=make_jobs(cfg,args.mode,**kwargs)
    if args.plan:
        print(f'运行模式: {args.mode}; 共 {len(jobs)} 次训练')
        for i,j in enumerate(jobs,1):
            p=j['profile']
            print(f"[{i}] {j['dataset']}-H{j['horizon']} {j['variant']} seed{j['seed']}: "
                  f"d_model={p['d_model']} t_ff={p['t_ff']} c_ff={p['c_ff']} "
                  f"epochs={p['epochs']} batch={p['batch_size']}")
        return
    rid=f"{datetime.now():%Y%m%d_%H%M%S}_{args.mode}_{uuid.uuid4().hex[:6]}"
    target=ROOT/'runs'/rid
    # Linux 文件锁：防止同一项目内 full / full-bg / pilot 同时竞争一张 GPU。
    import fcntl
    lockfile=ROOT/'ops'/'gpu_training.lock'
    lockfile.parent.mkdir(parents=True,exist_ok=True)
    with lockfile.open('a+') as fd:
        try:
            fcntl.flock(fd.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('已有本项目 GPU 训练任务在运行；请用 bash run.sh status 查看，不要重复启动。') from exc
        run(args.mode,cfg,jobs,target)


if __name__=='__main__':main()
