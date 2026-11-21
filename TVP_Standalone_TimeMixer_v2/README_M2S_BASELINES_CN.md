# M->S comparison baselines (overlay onto existing original TimeMixer project)

## Purpose

Existing `results/m2s/full/summary.csv` contains the **official pinned TimeMixer** in two input configurations:

- `target_only`: TimeMixer with OT historical input, output OT.
- `multivariate_joint`: TimeMixer with historical multivariate input, output OT.

These are *within-architecture input ablations*, not competing forecasting architectures.
This overlay adds seven reference baselines without modifying the original code, weights, data, or result files:

| Benchmark | Input | Architecture | Notes |
|---|---|---|---|
| last_value | OT only | Last observed OT repeated H times | Naive statistical control |
| seasonal_naive | OT only | Most recent seasonal cycle repeated | 5 trading sessions for carbon, 24 hours hourly, 48 half-hours NESO |
| dlinear_target | OT only | DLinear trend/season decomposition, dual linear heads | Local published-structure reimplementation; not packaged original author source |
| joint_linear | all historical | Flatten L*C then direct linear projection to H | Transparent cross-variate linear M->S benchmark |
| gru_joint | all historical | 1-layer GRU hidden=32 + linear H head | Local compact reference implementation |
| lstm_joint | all historical | 1-layer LSTM hidden=32 + linear H head | Local compact reference implementation |
| tcn_joint | all historical | causal dilated Conv1d hidden=32, dilation 1/2/4/8/16/32/64/128 | Local compact reference implementation |

There are no externally engineered input predictors. DLinear moving average is an *internal learned architecture operator*, not an added CSV feature.

These reference implementations enable a first rigorous multi-architecture comparison; for final research publication, add pinned original implementations of PatchTST/iTransformer/XLinear and perform seed/hyperparameter sensitivity evaluation. Do not label GRU/TCN as original authors' code.

## Install

Upload `TVP_TimeMixer_M2S_Baselines_Overlay_v1.zip` to `/root/autodl-tmp` and execute:

```bash
cd /root/autodl-tmp
unzip -o TVP_TimeMixer_M2S_Baselines_Overlay_v1.zip -d .
cd TVP_Standalone_TimeMixer_v2
bash 07_M2S_BASELINES_CPU_SMOKE.sh
bash 08_M2S_BASELINES_GPU_TWO.sh
bash 09_M2S_BASELINES_GPU_ALL.sh
bash 10_M2S_MERGE_LEADERBOARD.sh
```

Use your existing Python environment containing `torch`, `pandas`, `numpy`. No additional dependency. Existing official TimeMixer preparation unchanged.

If you want to run a single model:

```bash
python -u -m m2s_baselines.train --dataset Hubei --horizon 1 --variant gru_joint --device cuda --epochs 20 --patience 5 --batch-size 32
```

## Paths

- New individual results: `results/m2s_baselines/full/<dataset>/h<horizon>/<variant>/s2024/{config.json,metrics.json,best.pt,history.csv,test_predictions_sample.csv}`
- New baseline grid: `results/m2s_baselines/full/summary.csv`
- Combined official + baselines leaderboard: `results/m2s_benchmarks/full/leaderboard.csv`
- Training logs: `results/m2s_baselines/full/logs/`

The leaderboard compares only **same dataset, same L/H, same seed, same split**; differences in MSE are relative to TimeMixer target_only and TimeMixer multivariate_joint. MSE/MAE are in ORIGINAL target units (not cross-dataset comparable). Model selection on val; test run once after restoring best val checkpoint.

## Scientific limitations

- Full GPU run in AutoDL has not been executed in the authoring environment.
- Current default is one random seed (2024), maximum 20 epochs, early stop patience 5. Multiple-seed fixed-protocol verification is needed before making robust claims.
- Existing input records remain exactly as before; historical auxiliary missing cells are filled with TRAIN-only mean by existing `m2s.data`. Never fill the target `OT` or expose future auxiliary measurements.
- First compare benchmarks to TimeMixer, then test a PROPOSED architectural change using the *same* windows. `multivariate_joint` is still official TimeMixer and is NOT your proposed model.
