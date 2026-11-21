#!/usr/bin/env python3
import argparse, hashlib, json, re
from pathlib import Path

COMMITS = {
    "TStruct": "5b9eb393a092c14d28d08395b63de809accd115a",
    "TimeBase": "369b330f3d77371fcc7e8c75c808d01330c40899",
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def replace_once(text, old, new, label):
    n=text.count(old)
    if n != 1:
        raise RuntimeError(f"Patch '{label}' expected exactly 1 match, found {n}")
    return text.replace(old,new,1)

def replace_first(text, old, new, label):
    n=text.count(old)
    if n < 1:
        raise RuntimeError(f"Patch '{label}' expected at least 1 match, found {n}")
    return text.replace(old,new,1)

def patch_tstruct(repo: Path):
    run=repo/"run_longExp.py"
    s=run.read_text()
    s=replace_once(s,"fix_seed = 2026","fix_seed = 2024","TStruct seed 2024")
    run.write_text(s)

    exp=repo/"exp/exp_main.py"
    s=exp.read_text()
    s=replace_once(s,"            test_data, test_loader = self._get_data(flag='test')\n","","TStruct do not load test split in train")
    s=replace_once(s,"                test_loss, test_mae = self.vali(test_data, test_loader, mse_criterion, phase='test')\n","","TStruct remove per-epoch test evaluation")
    pattern=(r'                print\("Epoch: \{0\}, Steps: \{1\} \| Train Loss: \{2:\.7f\} Vali Loss: \{3:\.7f\} Test Loss: \{4:\.7f\}"\.format\(\s*'
             r'epoch \+ 1, train_steps, train_loss, vali_loss, test_loss\)\)\s*'
             r'                if \(epoch\+1\) % 5 == 0:\s*.*?'
             r'                self\.model\.train\(\)')
    repl=('                print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} "\n'
          '                      f"Vali MSE: {vali_loss:.7f} Vali MAE: {vali_mae:.7f}")\n'
          '                self.model.train()')
    s2,n=re.subn(pattern,repl,s,count=1,flags=re.S)
    if n != 1: raise RuntimeError(f"TStruct per-epoch log patch expected 1 match, found {n}")
    s=s2
    old="""        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
       
        return self.model
"""
    new="""        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        final_val_mse, final_val_mae = self.vali(vali_data, vali_loader, mse_criterion, phase='val')
        print(f\"FINAL_VAL_MSE={final_val_mse:.10f}\")
        print(f\"FINAL_VAL_MAE={final_val_mae:.10f}\")
        return self.model
"""
    s=replace_once(s,old,new,"TStruct final validation metrics")
    exp.write_text(s)

    s=run.read_text()
    old="""            if not args.train_only:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)

"""
    s=replace_once(s,old,"","TStruct no final test call")
    run.write_text(s)

def patch_timebase(repo: Path):
    run=repo/"run_longExp.py"
    s=run.read_text()
    s=replace_once(s,"fix_seed_list = range(2023, 2033)","fix_seed_list = [2024]","TimeBase seed 2024")
    run.write_text(s)

    exp=repo/"exp/exp_main.py"
    s=exp.read_text()
    s=replace_first(s,"        test_data, test_loader = self._get_data(flag='test')\n","","TimeBase do not load test split in train")
    s=replace_once(s,"            test_loss = self.vali(test_data, test_loader, criterion)\n","","TimeBase remove per-epoch test evaluation")
    old="""            print(\"Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}\".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
"""
    new="""            print(\"Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}\".format(
                epoch + 1, train_steps, train_loss, vali_loss))
"""
    s=replace_once(s,old,new,"TimeBase validation-only epoch log")
    pattern=(r"        best_model_path = path \+ '/' \+ 'checkpoint\.pth'\n"
             r"        self\.model\.load_state_dict\(torch\.load\(best_model_path\)\).*?"
             r"        return self\.model\n")
    new_final="""        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        self.model.eval()
        total_sq = 0.0
        total_abs = 0.0
        total_count = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if self.use_orthogonal:
                    pred, _ = self.model(batch_x)
                else:
                    pred = self.model(batch_x)
                pred = pred[:, -self.args.pred_len:, :]
                true = batch_y[:, -self.args.pred_len:, :]
                err = pred - true
                total_sq += torch.square(err).sum().item()
                total_abs += torch.abs(err).sum().item()
                total_count += true.numel()
        final_val_mse = total_sq / max(total_count, 1)
        final_val_mae = total_abs / max(total_count, 1)
        print(f"FINAL_VAL_MSE={final_val_mse:.10f}")
        print(f"FINAL_VAL_MAE={final_val_mae:.10f}")
        self.model.train()
        print(f"Final Max Memory (MB): {max_memory}")
        return self.model
"""
    s2,n=re.subn(pattern,new_final,s,count=1,flags=re.S)
    if n != 1: raise RuntimeError(f"TimeBase final validation patch expected 1 match, found {n}")
    s=s2
    exp.write_text(s)

    compat=repo/"utils/tools.py"
    cs=compat.read_text()
    cs=replace_once(cs,"np.Inf","np.inf","NumPy 2 compatibility")
    compat.write_text(cs)

    s=run.read_text()
    old="""        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
"""
    s=replace_once(s,old,"","TimeBase no final test call")
    run.write_text(s)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    args=ap.parse_args()
    root=Path(args.root).resolve()
    manifest={}
    for name,commit in COMMITS.items():
        repo=root/name
        marker=repo/"SOURCE_COMMIT.txt"
        if not marker.exists() or marker.read_text().strip()!=commit:
            raise RuntimeError(f"Missing or incorrect pinned source marker for {name}: {repo}")
        patch_marker=repo/"PATCHED_FOR_VALIDATION_ONLY.txt"
        if not patch_marker.exists():
            files=[repo/"run_longExp.py",repo/"exp/exp_main.py"] + ([repo/"utils/tools.py"] if name=="TimeBase" else [])
            before={str(p.relative_to(repo)):sha(p) for p in files}
            if name=="TStruct": patch_tstruct(repo)
            else: patch_timebase(repo)
            after={str(p.relative_to(repo)):sha(p) for p in files}
            patch_marker.write_text("Fixed seed=2024; validation-only training/selection; no test split loaded or evaluated.\n")
            manifest[name]={"commit":commit,"before_sha256":before,"after_sha256":after}
        else:
            manifest[name]={"commit":commit,"already_patched":True,"run_sha256":sha(repo/"run_longExp.py"),"exp_sha256":sha(repo/"exp/exp_main.py"),"compat_sha256":sha(repo/"utils/tools.py") if name=="TimeBase" else ""}
    out=root/"patch_manifest.json"
    out.write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"[PATCH] validation-only patches ready: {out}")

if __name__=="__main__": main()
