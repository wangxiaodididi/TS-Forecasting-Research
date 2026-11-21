#!/usr/bin/env python3
import argparse, csv, json, os, re, subprocess, sys, time
from pathlib import Path

TSTRUCT_COMMIT = "5b9eb393a092c14d28d08395b63de809accd115a"
TIMEBASE_COMMIT = "369b330f3d77371fcc7e8c75c808d01330c40899"

# Author-script hyperparameters for the four ETT dataset/horizon combinations.
TSTRUCT_CFG = {
    ("ETTh1", 96):  dict(lr=0.002, bs=64, epochs=25, patience=10, d_model=128, d_ff=128, patch720=48, dropout=0.3, rank=8, huber=0.8, aux=0.2, layers=1, k=4, attn_drop=0.3, head_drop=0.1),
    ("ETTh1",336):  dict(lr=0.001, bs=64, epochs=25, patience=10, d_model=128, d_ff=128, patch720=48, dropout=0.3, rank=8, huber=0.8, aux=0.2, layers=1, k=4, attn_drop=0.3, head_drop=0.1),
    ("ETTm1", 96):  dict(lr=0.01,  bs=256,epochs=25, patience=10, d_model=128, d_ff=128, patch720=48, dropout=0.3, rank=8, huber=1.0, aux=0.0, layers=3, k=5, attn_drop=0.3, head_drop=0.2),
    ("ETTm1",336):  dict(lr=0.01,  bs=128,epochs=25, patience=10, d_model=32,  d_ff=32,  patch720=72, dropout=0.4, rank=8, huber=1.0, aux=0.0, layers=1, k=5, attn_drop=0.3, head_drop=0.2),
}
TIMEBASE_CFG = {
    ("ETTh1", 96): dict(lr=0.1,  bs=64,  epochs=30, patience=5, period=24, basis=6,  orth=0.16),
    ("ETTh1",336): dict(lr=0.4,  bs=256, epochs=30, patience=5, period=24, basis=6,  orth=0.08),
    ("ETTm1", 96): dict(lr=0.02, bs=512, epochs=30, patience=5, period=4,  basis=18, orth=0.04),
    ("ETTm1",336): dict(lr=0.02, bs=256, epochs=30, patience=5, period=4,  basis=20, orth=0.08),
}
PALOMA_VAL = {
    ("ETTh1",96):  {"mse":0.6556251963540164,"mae":0.5257728135085386},
    ("ETTh1",336): {"mse":1.292065443718008,"mae":0.747915324958675},
    ("ETTm1",96):  {"mse":0.3522848989000367,"mae":0.37759336585438075},
    ("ETTm1",336): {"mse":0.6307998125320374,"mae":0.513685171958675},
}
FIELDS = ["model","profile","dataset","horizon","lookback","seed","status","returncode","mse","mae","paloma_validation_mse","paloma_validation_mae","mse_gain_pct_vs_paloma","mae_gain_pct_vs_paloma","wall_seconds","source_commit","patched_source_sha256","max_epochs_configured","actual_epochs","gpu_peak_mib","log_path","checkpoint_root","config_json"]

def load_manifest(root: Path):
    patch_path = root / "third_party" / "patch_manifest.json"
    if not patch_path.exists():
        raise RuntimeError("Patched source manifest missing. Run prepare_sources.py and patch_sources.py first.")
    return json.loads(patch_path.read_text())

def get_patch_hash(manifest, model):
    item = manifest[model]
    if "after_sha256" in item:
        return item["after_sha256"].get("run_longExp.py", "") + ":" + item["after_sha256"].get("exp/exp_main.py", "")
    return item.get("run_sha256", "") + ":" + item.get("exp_sha256", "") + ":" + item.get("compat_sha256", "")

def build_tstruct_cmd(repo, data_root, out_root, dataset, horizon, lookback, profile, seed):
    cfg = TSTRUCT_CFG[(dataset, horizon)].copy()
    patch_size = 8 if profile == "matched96" else cfg["patch720"]
    freq = "h" if dataset == "ETTh1" else "t"
    cycle = 24 if dataset == "ETTh1" else 96
    model_id = f"TStruct_{dataset}_L{lookback}_H{horizon}_{profile}_s{seed}"
    ckpt_root = out_root / "checkpoints" / "TStruct"
    cmd = [
        sys.executable, "run_longExp.py",
        "--is_training", "1",
        "--root_path", str(data_root) + "/",
        "--data_path", f"{dataset}.csv",
        "--model_id", model_id,
        "--model", "TStruct",
        "--data", dataset,
        "--features", "M",
        "--target", "OT",
        "--freq", freq,
        "--des", f"v1_{profile}_s{seed}",
        "--seq_len", str(lookback),
        "--pred_len", str(horizon),
        "--enc_in", "7",
        "--itr", "1",
        "--batch_size", str(cfg["bs"]),
        "--learning_rate", str(cfg["lr"]),
        "--train_epochs", str(cfg["epochs"]),
        "--patience", str(cfg["patience"]),
        "--norm_type", "standard",
        "--d_model", str(cfg["d_model"]),
        "--d_ff", str(cfg["d_ff"]),
        "--patch_size", str(patch_size),
        "--stride", str(patch_size),
        "--dropout", str(cfg["dropout"]),
        "--rank", str(cfg["rank"]),
        "--e_layers", str(cfg["layers"]),
        "--k_structure", str(cfg["k"]),
        "--attn_dropout", str(cfg["attn_drop"]),
        "--head_dropout", str(cfg["head_drop"]),
        "--lradj", "type3",
        "--huber_lambda", str(cfg["huber"]),
        "--auxi_lambda", str(cfg["aux"]),
        "--cycle", str(cycle),
        "--checkpoints", str(ckpt_root),
        "--num_workers", "0",
        "--pct_start", "0.3",
        "--gpu", "0",
    ]
    config = {**cfg, "patch_size":patch_size,"stride":patch_size,"cycle":cycle,"freq":freq,"seq_len":lookback,"pred_len":horizon,"seed":seed}
    return cmd, ckpt_root, config

def build_timebase_cmd(repo, data_root, out_root, dataset, horizon, lookback, profile, seed):
    cfg = TIMEBASE_CFG[(dataset, horizon)].copy()
    freq = "h" if dataset == "ETTh1" else "t"
    model_id = f"TimeBase_{dataset}_L{lookback}_H{horizon}_{profile}_s{seed}"
    ckpt_root = out_root / "checkpoints" / "TimeBase"
    cmd = [
        sys.executable, "run_longExp.py",
        "--is_training", "1",
        "--root_path", str(data_root) + "/",
        "--data_path", f"{dataset}.csv",
        "--model_id", model_id,
        "--model", "LightTimeBaseTST",
        "--data", dataset,
        "--features", "M",
        "--target", "OT",
        "--freq", freq,
        "--checkpoints", str(ckpt_root),
        "--seq_len", str(lookback),
        "--pred_len", str(horizon),
        "--period_len", str(cfg["period"]),
        "--enc_in", "7",
        "--train_epochs", str(cfg["epochs"]),
        "--patience", str(cfg["patience"]),
        "--basis_num", str(cfg["basis"]),
        "--use_period_norm", "1",
        "--use_orthogonal", "1",
        "--orthogonal_weight", str(cfg["orth"]),
        "--gpu", "0",
        "--itr", "1",
        "--batch_size", str(cfg["bs"]),
        "--learning_rate", str(cfg["lr"]),
        "--des", f"v1_{profile}_s{seed}",
        "--loss", "mse",
        "--lradj", "type3",
        "--pct_start", "0.3",
        "--num_workers", "0",
    ]
    config = {**cfg,"freq":freq,"seq_len":lookback,"pred_len":horizon,"seed":seed,"use_period_norm":1,"use_orthogonal":1,"loss":"mse"}
    return cmd, ckpt_root, config

def read_rows(path):
    if not path.exists(): return []
    with path.open(newline="") as f: return list(csv.DictReader(f))

def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)

def parse_metric(log_text, name):
    m=re.findall(rf"{name}=([-+0-9.eE]+)",log_text)
    return float(m[-1]) if m else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--profile",choices=["matched96","paper720","both"],default="matched96")
    ap.add_argument("--models",nargs="+",choices=["tstruct","timebase"],default=["tstruct","timebase"])
    ap.add_argument("--datasets",nargs="+",choices=["ETTh1","ETTm1"],default=["ETTh1","ETTm1"])
    ap.add_argument("--horizons",nargs="+",type=int,choices=[96,336],default=[96,336])
    ap.add_argument("--seed",type=int,default=2024)
    ap.add_argument("--data-root",required=True)
    ap.add_argument("--resume",default="1")
    args=ap.parse_args()

    root=Path(__file__).resolve().parent
    data_root=Path(args.data_root).resolve()
    out_root=root/"runs"/"tstruct_timebase_benchmark_v1"
    log_root=out_root/"logs"
    summary_path=out_root/"summary.csv"
    out_root.mkdir(parents=True,exist_ok=True)
    manifest=load_manifest(root)
    profiles=["matched96","paper720"] if args.profile=="both" else [args.profile]
    lookbacks={"matched96":96,"paper720":720}
    rows=read_rows(summary_path)
    done={(r["model"],r["profile"],r["dataset"],int(r["horizon"]),int(r["lookback"]),int(r["seed"])) for r in rows if r.get("status")=="ok"}

    for profile in profiles:
        lookback=lookbacks[profile]
        for model_key in args.models:
            model="TStruct" if model_key=="tstruct" else "TimeBase"
            repo=root/"third_party"/model
            for dataset in args.datasets:
                for horizon in args.horizons:
                    key=(model,profile,dataset,horizon,lookback,args.seed)
                    if args.resume!="0" and key in done:
                        print(f"[SKIP] {key} already completed")
                        continue
                    if model=="TStruct":
                        cmd,ckpt_root,cfg=build_tstruct_cmd(repo,data_root,out_root,dataset,horizon,lookback,profile,args.seed)
                    else:
                        cmd,ckpt_root,cfg=build_timebase_cmd(repo,data_root,out_root,dataset,horizon,lookback,profile,args.seed)
                    task_log_dir=log_root/model/profile/dataset/f"h{horizon}"
                    task_log_dir.mkdir(parents=True,exist_ok=True)
                    log_path=task_log_dir/"train.log"
                    print("\n[RUN] "+" ".join(cmd))
                    start=time.time()
                    env=os.environ.copy()
                    env["PYTHONHASHSEED"]=str(args.seed)
                    env["OMP_NUM_THREADS"]=env.get("OMP_NUM_THREADS","8")
                    with log_path.open("w") as log:
                        proc=subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,text=True)
                    elapsed=time.time()-start
                    txt=log_path.read_text(errors="replace")
                    mse=parse_metric(txt,"FINAL_VAL_MSE")
                    mae=parse_metric(txt,"FINAL_VAL_MAE")
                    status="ok" if proc.returncode==0 and mse is not None and mae is not None else "failed"
                    commit=TSTRUCT_COMMIT if model=="TStruct" else TIMEBASE_COMMIT
                    patch_hash=get_patch_hash(manifest,model)
                    paloma = PALOMA_VAL.get((dataset,horizon), {}) if profile=="matched96" and lookback==96 else {}
                    paloma_mse = paloma.get("mse", "")
                    paloma_mae = paloma.get("mae", "")
                    mse_gain = ((paloma_mse-mse)/paloma_mse*100.0) if paloma_mse and mse is not None else ""
                    mae_gain = ((paloma_mae-mae)/paloma_mae*100.0) if paloma_mae and mae is not None else ""
                    row={
                        "model":model,"profile":profile,"dataset":dataset,"horizon":horizon,"lookback":lookback,"seed":args.seed,
                        "status":status,"returncode":proc.returncode,"mse":mse if mse is not None else "","mae":mae if mae is not None else "",
                        "paloma_validation_mse":paloma_mse,"paloma_validation_mae":paloma_mae,"mse_gain_pct_vs_paloma":round(mse_gain,6) if mse_gain!="" else "","mae_gain_pct_vs_paloma":round(mae_gain,6) if mae_gain!="" else "",
                        "wall_seconds":round(elapsed,2),"source_commit":commit,"patched_source_sha256":patch_hash,
                        "max_epochs_configured":cfg.get("epochs"),"actual_epochs":len(re.findall(r"Epoch: \d+, Steps:",txt)),"gpu_peak_mib":(float(re.findall(r"Final Max Memory \(MB\): ([0-9.]+)",txt)[-1]) if re.findall(r"Final Max Memory \(MB\): ([0-9.]+)",txt) else ""),"log_path":str(log_path),"checkpoint_root":str(ckpt_root),
                        "config_json":json.dumps(cfg,sort_keys=True)
                    }
                    rows=[r for r in rows if not (r.get("model")==model and r.get("profile")==profile and r.get("dataset")==dataset and int(r.get("horizon",-1))==horizon and int(r.get("lookback",-1))==lookback and int(r.get("seed",-1))==args.seed)]
                    rows.append(row)
                    write_rows(summary_path,rows)
                    print(f"[RESULT] {model} {profile} {dataset} H{horizon}: status={status} val_mse={mse} val_mae={mae} time={elapsed:.1f}s")
                    if status!="ok":
                        print(f"[WARN] Inspect log: {log_path}")
    print(f"\n[SUMMARY] {summary_path}")
    print("[NOTE] Validation only. No test split was loaded or evaluated by the patched training runners.")

if __name__=="__main__": main()
