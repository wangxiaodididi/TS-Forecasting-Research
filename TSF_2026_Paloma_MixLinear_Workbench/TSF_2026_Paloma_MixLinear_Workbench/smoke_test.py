#!/usr/bin/env python3
import argparse, subprocess, sys
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True); ap.add_argument("--profile",default="matched96"); a=ap.parse_args()
root=Path(a.root).resolve(); profile=a.profile
profiles=["matched96","paper720"] if profile=="both" else [profile]
for p in profiles:
    seq=96 if p=="matched96" else 720
    for model in ["tstruct","timebase"]:
        repo=root/("TStruct" if model=="tstruct" else "TimeBase")
        for ds in ["ETTh1","ETTm1"]:
            for h in [96,336]:
                cmd=[sys.executable,str(Path(__file__).with_name("smoke_worker.py")),"--model",model,"--repo",str(repo),"--dataset",ds,"--horizon",str(h),"--seq-len",str(seq),"--profile",p]
                subprocess.run(cmd,check=True)
print("[SMOKE] all source model shape/backward checks passed")
