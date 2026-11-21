#!/usr/bin/env python3
"""Create isolated Sonnet source variants for controlled implementation audits.

Run from the existing Sonnet_M2S_2026_ResearchLab project root:
    python prepare_variants.py

The script never edits official/Sonnet in place. It creates isolated copies under
official/variants/ and applies one controlled change per copy.
"""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "official" / "Sonnet"
OUT = SRC.parent / "variants"

if not SRC.is_dir():
    raise SystemExit(f"Cannot find official source: {SRC}")

IGNORE = shutil.ignore_patterns(
    ".git", "outputs", "multirun", "logs", "datasets", "__pycache__", "*.pyc", "*.bak*"
)

SCALER_OLD = "train_len = int(len(time_series) * dataset_split[0])"
SCALER_NEW = (
    "train_len = (\n"
    "                int(len(time_series) * dataset_split[0])\n"
    "                if do_dataset_split_ratio\n"
    "                else int(dataset_split_numeric[0])\n"
    "            )"
)

ATTN_OLD = '''        # Apply attention to values (v)
        out_time = time_attn.unsqueeze(2) * v

        out_time = rearrange(out_time, "b nh hd l -> b l (nh hd)")
        var_attn = F.softmax(self.var_attn, dim=-1)  # [C, C]
        out_var = torch.einsum(
            "b l d, c c -> b l d", out_time, var_attn
        )  # Mix variables
        out_var = rearrange(out_var, "b l (nh hd) -> b l nh hd", nh=self.d_k)

        # Residual connection + MLP
        out = out_var + self.var_mlp(out_var)
        out = self.out_proj(out)
        return out.permute(0, 2, 1, 3)  # [B, T, N, d_model]
'''

ATTN_NEW = '''        # Apply coherence weights to values, preserving the wavelet-head axis.
        # Shape: [B, K, d_model, T] -> [B, K, T, d_model].
        out_time = time_attn.unsqueeze(2) * v
        out_time = rearrange(out_time, "b nh hd l -> b nh l hd")

        # Paper-equation path: O_m = O_r + MLP(O_r), followed by W_out.
        # The published formulation does not contain an additional var_attn
        # matrix. This branch therefore removes the trace-only scaling found
        # in the released implementation and follows the stated MVCA equations.
        out = out_time + self.var_mlp(out_time)
        out = self.out_proj(out)
        return out  # [B, K, T, d_model]
'''

RUN_EXPERIMENT_PATCH_OLD = '''        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader, ckpt_path=cfg.exp.ckpt_name)
'''
RUN_EXPERIMENT_PATCH_NEW = '''        trainer.fit(model, train_loader, val_loader)

        # Persist validation-only diagnostics so the test set can remain sealed
        # during controlled comparisons and validation-based model selection.
        import json
        summary = {
            "seed": int(cfg.seed),
            "best_model_score": (
                float(checkpoint_callback.best_model_score.detach().cpu())
                if checkpoint_callback.best_model_score is not None else None
            ),
            "best_model_path": checkpoint_callback.best_model_path,
            "callback_metrics": {
                str(k): float(v.detach().cpu())
                for k, v in trainer.callback_metrics.items()
                if hasattr(v, "detach") and v.numel() == 1
            },
        }
        with open(os.path.join(output_dir, "controlled_val_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        if os.environ.get("SONNET_SKIP_TEST", "0") != "1":
            trainer.test(model, test_loader, ckpt_path=cfg.exp.ckpt_name)
        else:
            logger.info("SONNET_SKIP_TEST=1: test evaluation skipped by design")
'''


def copy_variant(name: str) -> Path:
    dst = OUT / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(SRC, dst, ignore=IGNORE)
    return dst


def patch_file(path: Path, old: str, new: str, label: str):
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match in {path}, found {count}")
    path.write_text(text.replace(old, new))


def prepare(name: str, scaler: bool, mvca: bool):
    dst = copy_variant(name)
    loader = dst / "sonnet" / "data" / "dartsDataloader.py"
    model = dst / "sonnet" / "mts_model" / "models" / "Sonnet.py"
    runner = dst / "scripts" / "run_experiment.py"
    lightning = dst / "sonnet" / "lightning" / "lightning_module.py"

    # Preserve the already-applied PyTorch 2.8 scheduler compatibility patch.
    ltxt = lightning.read_text()
    lightning.write_text(ltxt.replace(", verbose=True", ""))

    if scaler:
        patch_file(loader, SCALER_OLD, SCALER_NEW, "train-only scaler")
    if mvca:
        patch_file(model, ATTN_OLD, ATTN_NEW, "paper-equation MVCA")
        model_text = model.read_text()
        model_text = model_text.replace(
            "        self.var_attn = nn.Parameter(torch.eye(d_model))\n", ""
        )
        model.write_text(model_text)

    patch_file(runner, RUN_EXPERIMENT_PATCH_OLD, RUN_EXPERIMENT_PATCH_NEW, "validation-only runner")

    # Compile all project Python sources before a remote run.
    import compileall
    ok = compileall.compile_dir(str(dst), quiet=1, force=True)
    if not ok:
        raise RuntimeError(f"Python compile check failed for {name}")

    print(f"PREPARED {name}: scaler_train_only={scaler}, mvca_paper_equation={mvca}")
    print(f"  path={dst}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prepare("scaler_only", scaler=True, mvca=False)
    prepare("mvca_only", scaler=False, mvca=True)
    prepare("both_fixes", scaler=True, mvca=True)
    print("All controlled variants prepared. Original official/Sonnet was not modified.")


if __name__ == "__main__":
    main()
