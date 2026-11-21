"""Understand model components and trainable parameter counts, no hidden model changes."""
import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from .paths import ROOT,VENDOR
from .config import config_for
from .source import verify


def describe(dataset='ETTm1',model='GTRHaarHigh',horizon=96):
    verify(VENDOR)
    cfg=config_for(dataset,model,horizon,2024,'paper')
    # Official GTR module import, no edits to vendor path.
    sys.path.insert(0,str(VENDOR))
    from models import GTR
    defaults={'d_model':512,'individual':0,'use_revin':1,'dropout':0.0}
    defaults.update(cfg)
    args=SimpleNamespace(**defaults)
    if model=='GTR':network=GTR.Model(args)
    else:
        from extensions.models.GTRSignalCore import SignalAugmentedModel
        mode={'GTRLinearResidual':'linear','GTRHaarHigh':'high','GTRHaarLow':'low','GTRHaarDual':'dual'}[model]
        network=SignalAugmentedModel(args,mode)
    trainable=sum(p.numel() for p in network.parameters() if p.requires_grad)
    by_module=[]
    for name,module in network.named_children():
        by_module.append({'module':name,'class':type(module).__name__,'trainable_parameters':sum(p.numel() for p in module.parameters() if p.requires_grad)})
    batch_x=torch.randn(2,cfg['seq_len'],cfg['enc_in'])
    cycle=torch.randint(0,cfg['cycle'],(2,))
    network.eval()
    with torch.no_grad():
        output=network(batch_x,cycle,return_components=True) if model!='GTR' else network(batch_x,cycle)
    shape=output['prediction'].shape if isinstance(output,dict) else output.shape
    return {'model':model,'dataset':dataset,'input_shape':list(batch_x.shape),'output_shape':list(shape),
            'total_trainable_parameters':trainable,'modules':by_module}


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='ETTm1');p.add_argument('--model',default='GTRHaarHigh');p.add_argument('--horizon',type=int,default=96)
    args=p.parse_args();print(json.dumps(describe(args.dataset,args.model,args.horizon),indent=2,ensure_ascii=False))
if __name__=='__main__':main()
