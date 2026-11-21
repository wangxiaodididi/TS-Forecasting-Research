# Sonnet controlled implementation experiments (v1)

This is an overlay for the existing `Sonnet_M2S_2026_ResearchLab` project. Extract the ZIP from `~/autodl-tmp/` so these files merge into the existing project folder. It does not edit `official/Sonnet` in place.

## Prepare isolated variants

```bash
cd ~/autodl-tmp/Sonnet_M2S_2026_ResearchLab
python prepare_variants.py
```

It creates three copies under `official/variants/`:

- `scaler_only`: fit the global StandardScaler on the actual numeric training split (21,169 points), leaving released MVCA unchanged.
- `mvca_only`: replace the trace-only `var_attn` operation with the MVCA sequence described in the paper, leaving the released scaler behavior unchanged.
- `both_fixes`: apply both changes.

The original `official/Sonnet` directory is not modified.

## Run controlled validation experiments

All three variants retain the settings of the already completed run (d_model=64, K=16, alpha=0.5, RevIN=0, lr=1e-4, seed=42). These are one-factor diagnostic comparisons, not the paper-tuned configuration.

Run one at a time:

```bash
bash run_variant.sh scaler_only val
bash run_variant.sh mvca_only val
bash run_variant.sh both_fixes val
```

The default `val` mode skips test evaluation and writes `controlled_val_summary.json` in the Hydra output directory. Logs are saved in `results/controlled/`. Compare validation results, not test metrics.

After selecting a variant based on validation, evaluate its test set once:

```bash
bash run_variant.sh both_fixes test
```

Optional learning-rate override:

```bash
LR=0.001 bash run_variant.sh both_fixes val
```

## Paper-alignment items for the next stage

The paper appendix reports d=512/K, alpha candidates `{0, 0.1, 0.25, 0.75}`, and target-window normalisation. The released config uses d_model=64, K=16, alpha=0.5, and RevIN=0. The current code branches do not implement alpha=0.1. These discrepancies should be addressed in a separate protocol-alignment stage after the one-factor implementation effects are isolated.

The paper appendix also says global z-score statistics are fitted on training data, while the released loader fits on 70% of the full series even when numeric split lengths are supplied. The `scaler_only` patch corrects this mismatch.

The MVCA correction is an implementation alignment to the published equations, not a new proposed-model contribution.
