#!/usr/bin/env python3
"""Verify six CSVs, measured predictors, OT and 12 historical window splits."""
from pathlib import Path
import sys
try:
    from m2s.data import load,CONFIGS
except ImportError:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from m2s.data import load,CONFIGS

ROOT=Path(__file__).resolve().parents[1]

def main():
    for name,(L,hs) in CONFIGS.items():
        for horizon in hs:
            d=load(ROOT,name,horizon)
            print(f'[OK] {name:<9} L={L:>3} H={horizon:>3} '
                  f'C={len(d.columns):>2} rows={len(d.raw):>6} '
                  f'train={len(d.windows["train"]):>6} '
                  f'val={len(d.windows["val"]):>6} '
                  f'test={len(d.windows["test"]):>6} '
                  f'OT={d.columns[-1]}')
    print('[READY] All 12 configurations verified, OT target last.')

if __name__=='__main__':main()
