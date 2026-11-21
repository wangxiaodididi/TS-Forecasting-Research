# Technical audit notes

## MVCA source-to-paper mapping

The paper defines Q/K/V projections over the embedding dimension, FFT along the variable/feature dimension for each wavelet head, averages cross-/auto-spectral densities across frequency bins, normalises coherence across time, weights V elementwise, concatenates/retains wavelet heads, applies a two-layer MLP with residual, and applies the output projection.

The released source computes `einsum("b l d, c c -> b l d", out_time, var_attn)`. This sums only the diagonal of `var_attn` (its trace) and scales every feature by one scalar; it does not perform matrix feature mixing. The released `var_attn` is not part of the paper's described MVCA equations. The `mvca_only` variant removes that operation and retains the per-wavelet-head tensor layout for MLP/output projection.

## Scaler boundary

The released loader uses `int(len(series) * dataset_split[0])` even when `do_dataset_split_ratio=False` and numeric splits are supplied. For ENER, this is 70% of 35,064 = 24,544 points, while the numeric training split is 21,169. The difference (3,375 observations) overlaps the declared validation interval. The `scaler_only` patch uses the actual numeric train length.

## Protocol mismatches to address after one-factor tests

- Paper appendix: d=512/K; released config: d_model=64, K=16 (paper setting would be 32).
- Paper appendix alpha candidates: {0, 0.1, 0.25, 0.75}; released config alpha=0.5 and code branches only implement 0, 0.25, 0.5, 0.75, 1.
- Paper appendix: target-window variable normalisation; released config revin=0.
- Paper appendix: scaler fitted on training data; released loader's numeric-split path currently fits on first 70% of full series.
- Paper appendix early stopping describes validation error across t+1,...,t+H. Released validation callback monitors the last-horizon MSE through `val_loss`. This should be treated as another protocol discrepancy and tested separately, not changed simultaneously with the first three variants.
