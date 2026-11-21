"""从官方训练器 stdout 提取结构化实时指标，绝不改变优化过程。

每个作业独立保存 batches.csv、epochs.csv、gpu.csv，网页只读取这些文件。
官方训练器每个 Epoch 打印 Train/Vali/Test Loss；监控包装器额外打印只读 batch loss。
注意：Test Loss 不可用于早停或挑选模型，本面板默认不显示训练期 Test Loss。
"""
from __future__ import annotations
import csv
from datetime import datetime
import json
import re
import subprocess
import threading
from pathlib import Path

BATCH = re.compile(r'LAB_BATCH\s+epoch=(\d+)\s+batch=(\d+)\s+steps=(\d+)\s+loss=([\deE+.-]+)')
EPOCH = re.compile(r'Epoch:\s*(\d+),\s*Steps:\s*(\d+)\s*\|\s*Train Loss:\s*([\deE+.-]+)\s+Vali Loss:\s*([\deE+.-]+)\s+Test Loss:\s*([\deE+.-]+)')
LEARNING_RATE = re.compile(r'Updating learning rate to\s+([\deE+.-]+)')

def now():
    return datetime.now().isoformat(timespec='seconds')

class AppendCsv:
    def __init__(self, path:Path, fieldnames:list[str]):
        self.path=path
        path.parent.mkdir(parents=True,exist_ok=True)
        self.fd=path.open('w',encoding='utf-8',newline='',buffering=1)
        self.writer=csv.DictWriter(self.fd,fieldnames=fieldnames)
        self.writer.writeheader()
    def write(self, row):
        self.writer.writerow(row)
        self.fd.flush()
    def close(self):
        self.fd.close()

class MetricParser:
    def __init__(self, folder:Path):
        directory=folder/'metrics'
        self.batches=AppendCsv(directory/'batches.csv', ['time','epoch','batch','steps','progress','global_batch','train_batch_loss'])
        self.epochs=AppendCsv(directory/'epochs.csv', ['time','epoch','steps','train_loss','val_loss','test_loss'])
        self.events=(directory/'events.jsonl').open('w',encoding='utf-8',buffering=1)
        self.latest_epoch=0
        self.latest_batch=0
        self.steps_per_epoch=0
        self.last_train=None
        self.last_val=None
        self.learning_rate=None
    def consume(self,line:str):
        batch=BATCH.search(line)
        if batch:
            epoch,idx,total=map(int,batch.group(1,2,3));loss=float(batch.group(4))
            self.latest_epoch=max(self.latest_epoch,epoch-1)
            self.latest_batch=idx;self.steps_per_epoch=total
            self.batches.write({'time':now(),'epoch':epoch,'batch':idx,'steps':total,'progress':round(idx/total,6),
                'global_batch':(epoch-1)*total+idx,'train_batch_loss':loss})
            self.events.write(json.dumps({'type':'batch','epoch':epoch,'batch':idx,'steps':total,'loss':loss})+'\n')
            return 'batch'
        match=EPOCH.search(line)
        if match:
            epoch,steps=map(int,match.group(1,2));train,val,test=map(float,match.group(3,4,5))
            self.latest_epoch=epoch;self.latest_batch=0;self.steps_per_epoch=steps
            self.last_train=train;self.last_val=val
            self.epochs.write({'time':now(),'epoch':epoch,'steps':steps,
                               'train_loss':train,'val_loss':val,'test_loss':test})
            self.events.write(json.dumps({'type':'epoch','epoch':epoch,'train_loss':train,'val_loss':val,
                                          'test_loss':test})+'\n')
            return 'epoch'
        match=LEARNING_RATE.search(line)
        if match:
            self.learning_rate=float(match.group(1))
        return None
    def close(self):
        self.batches.close();self.epochs.close();self.events.close()

class GpuSampler:
    """nvidia-smi 外部只读采样；异常时静默跳过，不中断原论文训练。"""
    def __init__(self,folder:Path,interval:float=5):
        self.interval=max(2.0,float(interval))
        self.output=AppendCsv(folder/'metrics'/'gpu.csv',
            ['time','gpu_util_pct','vram_used_mib','vram_total_mib','power_w'])
        self._stop=threading.Event()
        self.thread=threading.Thread(target=self._loop,daemon=True)
    def _loop(self):
        while not self._stop.is_set():
            try:
                proc=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,power.draw',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=4)
                if proc.returncode==0 and proc.stdout.strip():
                    fields=[x.strip() for x in proc.stdout.strip().splitlines()[0].split(',')]
                    if len(fields)==4:
                        def val(i):
                            try:return float(fields[i])
                            except ValueError:return None
                        self.output.write(dict(time=now(),gpu_util_pct=val(0),vram_used_mib=val(1),
                                               vram_total_mib=val(2),power_w=val(3)))
            except (OSError,subprocess.TimeoutExpired):
                pass
            self._stop.wait(self.interval)
    def start(self):self.thread.start()
    def close(self):
        self._stop.set();self.thread.join(timeout=6)
        self.output.close()
