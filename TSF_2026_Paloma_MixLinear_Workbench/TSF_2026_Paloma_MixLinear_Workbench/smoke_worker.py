#!/usr/bin/env python3
import argparse
from types import SimpleNamespace
import torch

def run_tstruct(repo, dataset, horizon, seq_len, profile):
    import sys
    sys.path.insert(0, str(repo))
    from models import TStruct
    cfg_map = {
        ("ETTh1",96): dict(d_model=128,d_ff=128,dropout=0.3,rank=8,layers=1,k=4,head=0.1),
        ("ETTh1",336): dict(d_model=128,d_ff=128,dropout=0.3,rank=8,layers=1,k=4,head=0.1),
        ("ETTm1",96): dict(d_model=128,d_ff=128,dropout=0.3,rank=8,layers=3,k=5,head=0.2),
        ("ETTm1",336): dict(d_model=32,d_ff=32,dropout=0.4,rank=8,layers=1,k=5,head=0.2),
    }
    c=cfg_map[(dataset,horizon)]
    patch=8 if profile=="matched96" else (72 if dataset=="ETTm1" and horizon==336 else 48)
    args=SimpleNamespace(seq_len=seq_len,pred_len=horizon,enc_in=7,norm_type="standard",patch_size=patch,
        d_model=c["d_model"],dropout=c["dropout"],e_layers=c["layers"],d_ff=c["d_ff"],k_structure=c["k"],
        attn_dropout=0.3,rank=c["rank"],head_dropout=c["head"])
    model=TStruct.Model(args)
    x=torch.randn(2,seq_len,7,requires_grad=False)
    y=model(x)
    assert tuple(y.shape)==(2,horizon,7), (dataset,horizon,tuple(y.shape))
    loss=y.square().mean(); loss.backward()
    grads=sum(1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    assert grads>0
    print(f"SMOKE_OK model=TStruct dataset={dataset} H={horizon} L={seq_len} shape={tuple(y.shape)} grads={grads}")

def run_timebase(repo, dataset, horizon, seq_len):
    import sys
    sys.path.insert(0, str(repo))
    from models import TimeBase
    cfg_map = {
        ("ETTh1",96): dict(period=24,basis=6),
        ("ETTh1",336): dict(period=24,basis=6),
        ("ETTm1",96): dict(period=4,basis=18),
        ("ETTm1",336): dict(period=4,basis=20),
    }
    c=cfg_map[(dataset,horizon)]
    args=SimpleNamespace(seq_len=seq_len,pred_len=horizon,enc_in=7,period_len=c["period"],basis_num=c["basis"],
                         use_period_norm=1,use_orthogonal=1,individual=0)
    model=TimeBase.Model(args)
    x=torch.randn(2,seq_len,7)
    y,orth=model(x)
    assert tuple(y.shape)==(2,horizon,7), (dataset,horizon,tuple(y.shape))
    loss=y.square().mean()+0.01*orth
    loss.backward()
    grads=sum(1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    assert grads>0
    print(f"SMOKE_OK model=TimeBase dataset={dataset} H={horizon} L={seq_len} shape={tuple(y.shape)} grads={grads}")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--model",choices=["tstruct","timebase"],required=True); ap.add_argument("--repo",required=True); ap.add_argument("--dataset",choices=["ETTh1","ETTm1"],required=True); ap.add_argument("--horizon",type=int,choices=[96,336],required=True); ap.add_argument("--seq-len",type=int,required=True); ap.add_argument("--profile",required=True); a=ap.parse_args()
    torch.manual_seed(2024)
    if a.model=="tstruct": run_tstruct(a.repo,a.dataset,a.horizon,a.seq_len,a.profile)
    else: run_timebase(a.repo,a.dataset,a.horizon,a.seq_len)
