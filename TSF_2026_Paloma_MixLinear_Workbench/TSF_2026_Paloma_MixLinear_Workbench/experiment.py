#!/usr/bin/env python3
"""Experiment launcher for the untouched upstream Paloma/MixLinear models.

The official papers' configurations and aligned-input comparisons are distinct
experimental protocols. This script does not implement new forecasting models.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHAS = {
    "Paloma": "def3dfc612bd6554be8f2ed7d5174848d6a37119",
    "MixLinear": "42dbb98a5bbe64c13bc75b3cc07a9dc4acf20106",
}
MIN_ROWS = {"ETTh1": 14400, "ETTm1": 57600}


def args_parser():
    a = argparse.ArgumentParser(description="2026 forecasting models: CPU smoke / GPU training")
    a.add_argument("--mode", choices=["smoke", "full"], required=True)
    a.add_argument("--profile", choices=["paper", "aligned"], default="aligned",
                   help="paper: source config including source lookbacks; aligned: both use lookback 96")
    a.add_argument("--models", default="Paloma,MixLinear")
    a.add_argument("--datasets", default="ETTh1,ETTm1")
    a.add_argument("--horizons", default="96,336")
    a.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "/root/autodl-tmp/GTR_ResearchWorkbench_Modular/dataset"))
    a.add_argument("--sources-dir", default=os.environ.get("SOURCES_DIR", str(ROOT / "sources")))
    a.add_argument("--out-dir", default=str(ROOT / "runs"))
    a.add_argument("--gpu", type=int, default=0)
    a.add_argument("--epochs", type=int, default=0, help="override epochs (0 = official script value)")
    a.add_argument("--dry-run", action="store_true", help="print commands without downloading or training")
    a.add_argument("--no-download", action="store_true", help="expect upstream repositories to exist")
    a.add_argument("--continue-on-error", action="store_true", help="run next job if one fails")
    return a.parse_args()


def split_names(raw, allowed):
    names = [x.strip() for x in raw.split(",") if x.strip()]
    if not names or any(x not in allowed for x in names):
        raise ValueError(f"Expected comma-separated names from {sorted(allowed)}, got: {raw}")
    return list(dict.fromkeys(names))


def find_data(folder, ds):
    folder = Path(folder).expanduser().resolve()
    if folder.is_file() and folder.name == ds + ".csv":
        return folder
    for trial in (folder / f"{ds}.csv", folder / "ETT-small" / f"{ds}.csv",
                  folder / "dataset" / f"{ds}.csv", folder / "dataset" / "ETT-small" / f"{ds}.csv"):
        if trial.is_file():
            return trial
    if folder.exists():
        matches = list(folder.glob(f"**/{ds}.csv"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return sorted(matches, key=lambda f: len(str(f)))[0]
    raise FileNotFoundError(f"Cannot locate {ds}.csv under {folder}; pass --data-dir /path/to/ETT-small")


def validate_data(file, ds):
    with file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) != 8 or header[0] != "date":
            raise ValueError(f"Invalid {file.name} columns (expected date and seven variables): {header}")
        n = sum(1 for _ in reader)
    if n < MIN_ROWS[ds]:
        raise ValueError(f"{file.name} has {n} rows, expected >= {MIN_ROWS[ds]} for fixed ETT splits; shortened CSV is NOT valid")
    print(f"[DATA] {ds}: {file}, rows={n}, columns={header}", flush=True)


def ensure_sources(args, models):
    sources = Path(args.sources_dir).expanduser().resolve()
    missing = [x for x in models if not (sources / x / "models" / f"{x}.py").is_file()]
    if missing and not args.no_download:
        env = os.environ.copy()
        env["SOURCES_DIR"] = str(sources)
        subprocess.run(["bash", str(ROOT / "setup_sources.sh")], check=True, env=env)
    missing = [x for x in models if not (sources / x / "models" / f"{x}.py").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing upstream model sources: {missing}; run bash setup_sources.sh")
    return sources


PROBE = r'''
import sys, types, torch, csv, os
import numpy as np
from models import __MODEL__
name, horizon, csv_file = sys.argv[1], int(sys.argv[2]), sys.argv[3]
config = types.SimpleNamespace(seq_len=96, pred_len=horizon, enc_in=7,
                               cycle=24, d_model=256, dropout=0.5,
                               use_revin=1, individual=0,
                               period_len=24, lpf=1, alpha=0.95)
torch.manual_seed(2023)
if csv_file and os.path.isfile(csv_file):
    with open(csv_file, newline='',encoding='utf-8-sig') as fd:
        reader=csv.reader(fd)
        header=next(reader)
        rows=[]
        for row in reader:
            if len(rows) >= 500:
                break
            rows.append([float(v) for v in row[1:8]])
    raw=np.asarray(rows,dtype=np.float32)
    if len(raw) < 96+horizon+2: raise ValueError('Not enough rows for real-window smoke')
    mu=raw[:min(len(raw),450)].mean(0)
    std=raw[:min(len(raw),450)].std(0)+1e-6
    raw=(raw-mu)/std
    x=torch.from_numpy(np.stack([raw[i:i+96] for i in range(2)]))
    target=torch.from_numpy(np.stack([raw[i+96:i+96+horizon] for i in range(2)]))
    mode='real-ETT-window'
else:
    x = torch.randn(2,96,7)
    target = torch.randn(2,horizon,7)
    mode='synthetic'
if name == 'Paloma':
    model = __MODEL__.Model(config)
    y = model(x, torch.tensor([0, 1]))
else:
    model = __MODEL__.Model(config)
    y = model(x)
assert tuple(y.shape) == (2,horizon,7), (name, y.shape)
assert torch.isfinite(y).all(), 'nonfinite forward'
loss = (y-target).square().mean()
loss.backward()
grad_count = sum(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
assert grad_count > 0, 'No finite gradients'
print(f'SMOKE_OK model={name} mode={mode} shape={tuple(y.shape)} loss={loss.item():.6f} grads={grad_count}')
'''


def build_params(model, ds, horizon, profile, epochs, gpu, csv_file, out_dir):
    aligned = profile == "aligned"
    seq = 96 if model == "Paloma" or aligned else 720
    if model == "Paloma":
        opts = {
            ("ETTh1", 96): dict(cycle=24, d_model=256, dropout=0.5, batch=32, lr=0.0005, alpha=0.5),
            ("ETTh1", 336): dict(cycle=24, d_model=256, dropout=0.5, batch=32, lr=0.0007, alpha=0.5),
            ("ETTm1", 96): dict(cycle=96, d_model=256, dropout=0.4, batch=64, lr=0.0005, alpha=0.7),
            ("ETTm1", 336): dict(cycle=96, d_model=256, dropout=0.5, batch=128, lr=0.001, alpha=0.5),
        }
        if (ds,horizon) not in opts:
            raise ValueError(f"No verified Paloma author script config for {ds}/{horizon}")
        c = opts[(ds,horizon)]
        cmd = [sys.executable, "-u", "run.py", "--is_training", "1",
               "--root_path", str(csv_file.parent), "--data_path", csv_file.name,
               "--model_id", f"{ds}_{seq}_{horizon}_{profile}", "--model", "Paloma",
               "--data", ds, "--features", "M", "--seq_len", str(seq),
               "--pred_len", str(horizon), "--enc_in", "7", "--cycle", str(c["cycle"]),
               "--d_model", str(c["d_model"]), "--dropout", str(c["dropout"]),
               "--train_epochs", str(epochs or 30), "--batch_size", str(c["batch"]),
               "--patience", "5", "--gpu", str(gpu), "--learning_rate", str(c["lr"]),
               "--lossfun_alpha", str(c["alpha"]), "--loss_mode", "L1", "--itr", "1",
               "--random_seed", "2024" if profile == "paper" else "2023",
               "--checkpoints", str(out_dir / "checkpoints")]
        desc = "as_author_script" if profile == "paper" else "aligned_input_window_author_optimization"
    else:
        if ds == "ETTh1":
            period = 24
            c = {96: dict(lpf=1,alpha=.95,batch=256,lr=.03),
                 336: dict(lpf=2,alpha=.99,batch=256,lr=.03)}.get(horizon)
            default_epochs = 30
        else:
            period = 2
            c = {96: dict(lpf=144,alpha=.99,batch=64,lr=.005),
                 336: dict(lpf=144,alpha=.99,batch=64,lr=.005)}.get(horizon)
            default_epochs = 15
        if c is None:
            raise ValueError(f"No verified MixLinear author script config for {ds}/{horizon}")
        # Source's FFT domain accesses lpf bins along the n_segment axis.
        # Lookback matched runs necessarily alter ETTm1's source lpf=144.
        lpf = min(c["lpf"], seq // period)
        if seq % period:
            raise ValueError(f"MixLinear requires seq_len divisible by period_len: {seq}/{period}")
        cmd = [sys.executable, "-u", "run_longExp.py", "--is_training", "1",
               "--root_path", str(csv_file.parent), "--data_path", csv_file.name,
               "--model_id", f"{ds}_{seq}_{horizon}_{profile}", "--model", "MixLinear",
               "--data", ds, "--features", "M", "--seq_len", str(seq),
               "--pred_len", str(horizon), "--period_len", str(period),
               "--enc_in", "7", "--train_epochs", str(epochs or default_epochs),
               "--patience", "10", "--alpha", str(c["alpha"]), "--lpf", str(lpf),
               "--gpu", str(gpu), "--itr", "1", "--batch_size", str(c["batch"]),
               "--learning_rate", str(c["lr"]), "--num_workers", "0",
               "--des", profile, "--checkpoints", str(out_dir / "checkpoints")]
        desc = "as_author_script" if profile == "paper" else "aligned_input_window_lpf_clamped_if_needed"
    return cmd, {"model":model,"dataset":ds,"horizon":horizon,"lookback":seq,
                 "profile":profile,"provenance":desc,"source_commit":SHAS[model],
                 "training_epochs":epochs or default_epochs if model=="MixLinear" else epochs or 30}


def get_cuda():
    try:
        p = subprocess.run([sys.executable,"-c","import torch; print(torch.cuda.is_available())"],
                           capture_output=True,text=True,check=True)
        return p.stdout.strip().endswith("True")
    except (subprocess.CalledProcessError, OSError):
        return False


def gpu_mem(gpu_index):
    if not shutil.which("nvidia-smi"):
        return None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True,text=True,timeout=8,check=True)
        return int(r.stdout.splitlines()[gpu_index].strip().split()[0])
    except (ValueError, IndexError, OSError, subprocess.SubprocessError):
        return None


def parse_metrics(logfile):
    s = logfile.read_text(errors="replace")
    patterns = [r"mse\s*[:=]\s*([0-9.eE+-]+)\s*,?\s*mae\s*[:=]\s*([0-9.eE+-]+)",
                r"mse:([0-9.eE+-]+), mae:([0-9.eE+-]+)"]
    for p in patterns:
        matches = re.findall(p, s, flags=re.I)
        if matches:
            return float(matches[-1][0]), float(matches[-1][1])
    return None, None


def run_job(cmd, repo, out, metadata, args):
    out.mkdir(parents=True, exist_ok=True)
    logfile = out / "train.log"
    metadata["command"] = cmd
    metadata["source_directory"] = str(repo)
    metadata["started_utc"] = datetime.now(timezone.utc).isoformat()
    (out / "metadata.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False), encoding="utf-8")
    print(f"[RUN] {metadata['model']} {metadata['dataset']} H{metadata['horizon']} (L{metadata['lookback']})", flush=True)
    print("      ", " ".join(cmd), flush=True)
    print(f"[LOG] {logfile}", flush=True)
    t0=time.monotonic()
    peak = None
    with logfile.open("w",encoding="utf-8") as log:
        proc = subprocess.Popen(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            peak_now = gpu_mem(args.gpu)
            if peak_now is not None:
                peak = max(peak or 0,peak_now)
            time.sleep(2)
        returncode=proc.wait()
    elapsed=time.monotonic()-t0
    mse,mae=parse_metrics(logfile)
    metadata.update(status="ok" if returncode==0 and mse is not None else "error",
                    returncode=returncode, wall_seconds=round(elapsed,2),
                    mse=mse,mae=mae, gpu_total_used_peak_mib=peak,
                    gpu_memory_note="Peak is overall GPU memory used, not this process's allocation",
                    finished_utc=datetime.now(timezone.utc).isoformat())
    (out / "metadata.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False), encoding="utf-8")
    print(f"[RESULT] status={metadata['status']} MSE={mse} MAE={mae} sec={elapsed:.1f}",flush=True)
    return metadata


def smoke(args, sources, models, datasets, horizons):
    for model in models:
        repo=sources/model
        for ds in datasets:
            try:
                csv_file = find_data(args.data_dir,ds)
                if not args.dry_run:
                    validate_data(csv_file,ds)
            except FileNotFoundError:
                csv_file = None
                print(f"[NOTE] No {ds} CSV; synthetic tensor will be used for the CPU smoke",flush=True)
            for h in horizons:
                code=PROBE.replace("__MODEL__",model)
                cmd=[sys.executable,"-c",code,model,str(h),str(csv_file or "")]
                print(f"[SMOKE] {model}/{ds} L96 H{h} B2 C7 | CPU forward+backward",flush=True)
                if args.dry_run: continue
                env=os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"]=""
                r=subprocess.run(cmd,cwd=repo,env=env,capture_output=True,text=True)
                print(r.stdout[-4000:],flush=True)
                if r.returncode:
                    print(r.stderr[-5000:],file=sys.stderr,flush=True)
                    raise RuntimeError(f"{model}/{ds} H{h}: CPU forward/backward failed")
    print("[SMOKE] Forward/backward checks complete; NO paper metrics were reproduced",flush=True)


def main():
    args=args_parser()
    models=split_names(args.models,SHAS)
    datasets=split_names(args.datasets,MIN_ROWS)
    horizons=[int(x.strip()) for x in args.horizons.split(",")]
    if not horizons or any(h not in (96,336) for h in horizons):
        raise ValueError("This audited experiment package supports H=96,336")
    out=Path(args.out_dir).expanduser().resolve()
    sources=Path(args.sources_dir).expanduser().resolve()
    if not args.dry_run:
        sources=ensure_sources(args,models)
    if args.mode=="smoke":
        smoke(args,sources,models,datasets,horizons)
        return
    if not args.dry_run and not get_cuda():
        raise RuntimeError("Full run uses the upstream scripts, which hardcode CUDA checkpoint loading. Run it on the AutoDL GPU server. CPU-only uses run_smoke.sh.")
    records=[]
    for model in models:
        for ds in datasets:
            file = (find_data(args.data_dir,ds) if not args.dry_run else Path(args.data_dir)/f"{ds}.csv")
            if not args.dry_run:
                validate_data(file,ds)
            for horizon in horizons:
                dest=out/args.profile/model/ds/f"h{horizon}"
                cmd,meta=build_params(model,ds,horizon,args.profile,args.epochs,args.gpu,file,dest)
                if args.dry_run:
                    print(f"[DRY] {model}/{ds}/H{horizon}: {' '.join(cmd)}")
                    continue
                result=run_job(cmd,sources/model,dest,meta,args)
                records.append(result)
                out.mkdir(parents=True,exist_ok=True)
                with (out/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
                    fields=["model","dataset","horizon","lookback","profile","provenance","source_commit","status","returncode","mse","mae","wall_seconds","gpu_total_used_peak_mib"]
                    w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
                    w.writeheader()
                    w.writerows(records)
                if result["status"]!="ok" and not args.continue_on_error:
                    raise RuntimeError(f"Job failed. Inspect {dest/'train.log'}")
    if not args.dry_run: print(f"[DONE] Summary: {out/'summary.csv'}", flush=True)


if __name__=="__main__":
    main()
