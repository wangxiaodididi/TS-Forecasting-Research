#!/usr/bin/env python3
"""Install additive head-variant modules in a disposable pinned TimeMixer worktree.
It does not edit official models/TimeMixer.py or original experiment checkpoints.
"""
import argparse, shutil
from pathlib import Path

def patch(source_root,patch_root):
    src=Path(source_root).resolve();patch=Path(patch_root).resolve()
    basic=src/'exp/exp_basic.py'
    if not basic.exists():raise FileNotFoundError(f'Official repo missing: {basic}')
    for name in ('TimeMixerResidualHead.py','residual_head.py'):
        shutil.copy2(patch/'code'/name,src/'models'/name)
    txt=basic.read_text('utf8')
    if "'TimeMixerResidualHead': TimeMixerResidualHead" not in txt:
        key='from models import TimeMixer'
        if txt.count(key)!=1:raise ValueError('Official exp_basic.py import anchor changed; no changes applied')
        txt=txt.replace(key,key+', TimeMixerResidualHead',1)
        key="'TimeMixer': TimeMixer,"
        if txt.count(key)!=1:raise ValueError('Official model_dict changed; no changes applied')
        txt=txt.replace(key,key+"\n            'TimeMixerResidualHead': TimeMixerResidualHead,",1)
        basic.write_text(txt,encoding='utf8')
    print('[READY] Added model TimeMixerResidualHead, official TimeMixer.py remains untouched.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--official-root',default='training/workspace/official')
    a=p.parse_args();patch(a.official_root,Path(__file__).resolve().parent)
