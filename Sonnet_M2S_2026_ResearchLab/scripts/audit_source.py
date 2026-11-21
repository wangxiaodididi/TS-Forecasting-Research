#!/usr/bin/env python3
"""Source-level audit; no paper reproduction claims. Writes evidence and diagnostics."""
import argparse
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = ROOT / 'official' / 'Sonnet'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',default=str(REPO))
    args=p.parse_args()
    base=pathlib.Path(args.repo)
    model=base/'sonnet/mts_model/models/Sonnet.py'
    loader=base/'sonnet/data/dartsDataloader.py'
    if not model.exists() or not loader.exists():
        sys.exit('Run bash 00_setup.sh first; official source is absent.')
    src=model.read_text()
    ds=loader.read_text()
    evidence={
      'model_path':str(model),'model_sha256':sha(model),
      'loader_path':str(loader),'loader_sha256':sha(loader),
      'coherence_trace_einsum':bool(re.search(r'einsum\(\s*["\']b\s+l\s+d,\s*c\s+c\s*->\s*b\s+l\s+d',src)),
      'scaler_default_70pct':bool(re.search(r'train_len\s*=\s*int\(len\(time_series\)\s*\*\s*dataset_split\[0\]\)',ds)),
      'separate_numeric_split':bool(re.search(r'_data_split_numeric\(',ds)),
      'paper_metric_is_horizon_only':True,
    }
    try:
        import torch
        torch.manual_seed(2026)
        z=torch.randn(2,5,12)
        mat=torch.softmax(torch.randn(6,6),dim=-1)
        product=torch.einsum('b l d, c c -> b l d',z,mat)
        trace_scaled=z*torch.trace(mat)
        evidence['trace_identity_max_abs_error']=float((product-trace_scaled).abs().max())
        evidence['trace_of_attention']=float(torch.trace(mat))
    except ImportError:
        evidence['trace_identity_max_abs_error']='PyTorch unavailable'
    out=ROOT/'results'/'audit_source.json'
    out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    print(json.dumps(evidence,ensure_ascii=False,indent=2))
    if evidence['coherence_trace_einsum']:
        print('ATTENTION AUDIT: original implementation reduces to TRACE SCALING, not matrix feature mixing.')
    if evidence['scaler_default_70pct'] and evidence['separate_numeric_split']:
        print('SCALER AUDIT: fit boundary may differ from numeric TRAIN/VAL cutoff; quantify and separate corrected protocol.')
    print('Audit report:',out)


if __name__=='__main__':
    main()
