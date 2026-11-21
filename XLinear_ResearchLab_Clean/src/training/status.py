"""读取任务运行状态及结果目录；后台进程不需要打开网页就能查看。"""
from pathlib import Path
import json
import os
from src.core.config import ROOT


def main():
    runs=sorted((ROOT/'runs').glob('*/progress.json'),reverse=True)
    if not runs: print('当前没有新项目实验。执行 bash run.sh pilot 或 bash run.sh full。')
    for file in runs[:12]:
        info=json.loads(file.read_text(encoding='utf-8'))
        pid=info.get('pid')
        alive=False
        try:
            if pid:
                os.kill(int(pid),0)
                alive=True
        except (ProcessLookupError,PermissionError,ValueError): pass
        status=info.get('status')
        if status=='running' and not alive:status='stale (process exited)'
        current=info.get('current') or '-'
        total=info.get('total',0);done=info.get('completed',0)
        print(f'{file.parent.name}: {status}; jobs {done}/{total}; epoch {info.get("epoch",0)}/{info.get("epoch_max","?")}; current={current}; pid={pid}')
    logs=sorted((ROOT/'ops').glob('background_*.log'),reverse=True)
    if logs:print('最新后台启动日志:',logs[0])

if __name__=='__main__':main()
