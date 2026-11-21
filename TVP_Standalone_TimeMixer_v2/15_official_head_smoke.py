#!/usr/bin/env python3
"""Optional OFFICIAL model integration smoke: requires the prepared official tree."""
import os,sys,json
from pathlib import Path
from types import SimpleNamespace
import torch

root=Path.cwd();src=root/'training/workspace/official'
if not (src/'models/TimeMixer.py').is_file():
    raise SystemExit('Pinned TimeMixer source missing. Run 02_full_experiment.sh first.')
from importlib import import_module
installer=import_module('10_install_head_adapter')
installer.patch(src,Path(__file__).resolve().parent)
sys.path.insert(0,str(src))
from models import TimeMixer,TimeMixerResidualHead
config=json.loads((root/'examples/timemixer_ETTm1_96.json').read_text('utf8'))
config.update(task_name='long_term_forecast',label_len=0)
a=SimpleNamespace(**config)
for mode in ('linear','nonlinear'):
    os.environ.update(TIMEMIXER_HEAD_MODE=mode,TIMEMIXER_HEAD_RANK='8',TIMEMIXER_HEAD_SCALE='0.1')
    torch.manual_seed(55)
    orig=TimeMixer.Model(a).cpu().eval()
    torch.manual_seed(55)
    mod=TimeMixerResidualHead.Model(a).cpu().eval()
    s=orig.state_dict();ss=mod.state_dict()
    # Copy each original module, including original temporal projection into the wrapper base.
    for key,tensor in s.items():
        mapped=key
        if key.startswith('predict_layers.'):
            parts=key.split('.')
            mapped='.'.join(parts[:2]+['base']+parts[2:])
        if mapped not in ss:raise RuntimeError('Cannot align baseline state key: '+key)
        ss[mapped]=tensor
    mod.load_state_dict(ss,strict=True)
    x=torch.randn(2,96,7)
    with torch.no_grad():
        a0=orig(x,None,None,None)
        a1=mod(x,None,None,None)
    maximum=float((a0-a1).abs().max())
    if maximum>1.e-5:raise RuntimeError(f'Original vs modified output differs at initialization: {maximum}')
    print('[PASS] mode',mode,'shape',tuple(a1.shape),'initial max abs diff',maximum)
    mod.train()
    out=mod(x,None,None,None)
    loss=out.square().mean();loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in mod.parameters())
    print('[PASS] forward/backward',mode,'loss',float(loss.detach()))
