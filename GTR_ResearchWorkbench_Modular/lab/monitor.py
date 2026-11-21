"""Parse UNMODIFIED official GTR logs into machine-readable progress (no training changes)."""
import csv
import json
import re
import subprocess
import time
from pathlib import Path

EPOCH = re.compile(r'Epoch:\s*(\d+),\s*Steps:\s*(\d+)\s*\|\s*Train Loss:\s*([\d.eE+-]+)\s*Vali Loss:\s*([\d.eE+-]+)\s*Test Loss:\s*([\d.eE+-]+)')
BATCH = re.compile(r'iters:\s*(\d+),\s*epoch:\s*(\d+)\s*\|\s*loss:\s*([\d.eE+-]+)')
TEST_MAE_MSE = re.compile(r'mse:([\d.eE+-]+),\s*mae:([\d.eE+-]+)',re.IGNORECASE)
GPU_FIELDS=['timestamp','index','util_gpu_pct','memory_used_mb','memory_total_mb','temperature_c']

def append_jsonl(path, payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(payload,ensure_ascii=False)+'\n')

def parse_line(line):
    m=EPOCH.search(line)
    if m:
        epoch,steps,train,val,test=m.groups()
        return {'type':'epoch','epoch':int(epoch),'steps':int(steps),'train_loss':float(train),'val_loss':float(val),'test_loss_official_log_only':float(test)}
    m=BATCH.search(line)
    if m:
        step,epoch,loss=m.groups()
        return {'type':'batch','epoch':int(epoch),'batch':int(step),'batch_loss':float(loss)}
    m=TEST_MAE_MSE.search(line)
    if m:
        mse,mae=m.groups();return {'type':'final_test','test_mse':float(mse),'test_mae':float(mae)}
    if 'Early stopping' in line:
        return {'type':'early_stopping'}
    return None

def append_epoch_row(path,event):
    path=Path(path)
    first=not path.exists()
    with path.open('a',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['epoch','steps','train_loss','val_loss','test_loss_official_log_only'])
        if first:w.writeheader()
        w.writerow({k:event.get(k) for k in w.fieldnames})

def poll_gpu():
    """Read nvidia-smi once. Absence of GPU produces no fake metrics."""
    try:
        p=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
        if p.returncode: return []
        out=[]
        for line in p.stdout.splitlines():
            parts=[x.strip() for x in line.split(',')]
            if len(parts)!=5:continue
            try:
                out.append(dict(zip(GPU_FIELDS,[time.time(),int(parts[0]),float(parts[1]),float(parts[2]),float(parts[3]),float(parts[4])])))
            except ValueError:pass
        return out
    except (OSError,subprocess.TimeoutExpired):return []
