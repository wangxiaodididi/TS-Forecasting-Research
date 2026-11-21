#!/usr/bin/env python3
import importlib, sys
mods = ["torch", "numpy", "pandas", "sklearn", "matplotlib", "einops"]
missing=[]
for m in mods:
    try:
        mod=importlib.import_module(m)
        print(f"[OK] {m} {getattr(mod,'__version__','')}")
    except Exception as e:
        missing.append((m,str(e)))
if missing:
    print("[ERROR] Missing or broken imports:")
    for m,e in missing: print(f"  {m}: {e}")
    sys.exit(2)
import torch
print(f"[TORCH] cuda_available={torch.cuda.is_available()} torch_cuda={torch.version.cuda}")
if not torch.cuda.is_available():
    print("[ERROR] CUDA is unavailable. This benchmark runner is configured for GPU training.")
    sys.exit(3)
print(f"[GPU] {torch.cuda.get_device_name(0)}")
