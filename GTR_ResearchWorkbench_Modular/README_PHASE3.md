# GTR Phase 3: matched-window mechanism diagnostics and ETTm1-192

## Research problem

The Phase2 Weather-96 experiment **failed to outperform the unchanged GTR**: best validation losses GTR 0.3826478, LinearResidual 0.3867857, PhaseInnovation 0.3855864, PhaseAdaptive 0.3844036. We should **not** conclude that local–global inconsistency is the root cause, nor introduce more modules until comparing the models on identical validation examples.

We now test two descriptive hypotheses:

- **Complementarity vs redundancy:** Does an additional local forecast correction help on rapidly changing windows, or is it systematically redundant relative to learned global-cycle retrieval?
- **Condition dependence:** Do any such gains correlate with historical variability/drift or phase-reference fit, in a way consistent across variables and forecasting horizons?

These are hypotheses. The trained `Q` is latent; fitted phase R² is not a physically measured seasonality score, nor is the output correction a causal attribution.

## Contents (add-only)

- `15_Phase3_ETTm1_192.sh` — launch official-protocol Phase2 comparison for ETTm1-192 using the already installed Phase2 runner; **new experiments**, no files overwritten.
- `16_Phase3_Weather_Analyze.sh` — re-evaluate GTR, LinearResidual, PhaseInnovation, PhaseAdaptive from their **best checkpoints**, on the **same deterministic validation indices**.
- `17_Phase3_Analysis_Summary.sh` — display diagnostic summary.
- `phase3/predict.py` — one immutable run snapshot per clean subprocess, use same inputs and targets across models; compute sample × variable MSE and MAE from standardized predictions, historical relative change/drift, optional phase fit and correction energy.
- `phase3/report.py` — paired deltas to original GTR, per-variable analysis, within-variable terciles for change and drift, phase-fit terciles for phase variants, descriptive block-bootstrap intervals.
- `tests/test_phase3_report.py` — sanity tests for alignment and analysis.

**Does not alter:** official GTR source, `lab/config.py`, `lab/extend.py`, datasets, existing runs, checkpoints, or any model/optimizer implementation.

## Install

Upload the ZIP to your existing AutoDL project root, then:

```bash
cd /root/autodl-tmp/GTR_ResearchWorkbench_Modular
mkdir -p _phase3_upload
unzip -n GTR_Phase3_Diagnostic_ETTm1_192_v1.zip -d _phase3_upload
# Only copy NEW paths into the existing project; do not merge old model files.
cp -rn _phase3_upload/phase3 ./
cp -rn _phase3_upload/tests/test_phase3_report.py tests/
cp -n _phase3_upload/15_Phase3_ETTm1_192.sh .
cp -n _phase3_upload/16_Phase3_Weather_Analyze.sh .
cp -n _phase3_upload/17_Phase3_Analysis_Summary.sh .
chmod +x 15_Phase3_ETTm1_192.sh 16_Phase3_Weather_Analyze.sh 17_Phase3_Analysis_Summary.sh
./.venv/bin/python -m unittest discover -s tests -p test_phase3_report.py -v
```

If `cp -rn` is not supported on your system, copy `phase3/` and the three shell files by VS Code. All patch paths are **new** relative to Phase2.

## Step 1: Weather-96 diagnostic (does not retrain)

```bash
bash 16_Phase3_Weather_Analyze.sh 20261115_235624_bef5fe 1024
bash 17_Phase3_Analysis_Summary.sh 20261115_235624_bef5fe
```

`1024` samples are evenly spaced across the validation set (not hand-picked). Use `0` for all windows if necessary; that may require more evaluation time. The command uses available CUDA for inference; use `./.venv/bin/python -m phase3.cli --batch ... --dataset Weather --horizon 96 --limit 256 --device cpu` for a cheap check.

Output directory: `phase3_reports/20261115_235624_bef5fe/`

- `REPORT.txt` — validation-only headline diagnostics.
- `model_metrics.csv` — deterministic subset validation MSE/MAE (different evaluation convention from official shuffled/drop-last best-val losses).
- `paired_summary.csv` — matched-sample ΔMSE, win fraction, indicative moving-block-bootstrap intervals.
- `by_channel.csv` — variable-level matched comparisons.
- `by_condition.csv` — retrospective descriptive stratification by **past-only** `relative_change` and `relative_drift`, and the learned reference fit for phase models.
- `{model}.csv` — raw matched sample-variable records; `{model}.json` — checkpoint, dataset hash, sample indices and provenance.

**Sign:** `delta_mse = variant − GTR`; negative is favorable. `improvement_pct = 100*(GTR − variant)/GTR`; positive is favorable.

The validation dataset helped select the best checkpoints, so bootstrap intervals are **descriptive**, not final independent statistical tests. Sliding forecast windows overlap strongly: do not run an ordinary unpaired t-test across rows. The forecast values come from each separately trained model, so this analysis also cannot isolate *causal* effects of a single head without a controlled intervention.

## Step 2: ETTm1-192 cross-horizon check (training)

```bash
bash 15_Phase3_ETTm1_192.sh
bash 13_Phase2_Status.sh
bash 14_Phase2_Validation.sh
```

This script starts four official-protocol Phase2 variants sequentially with the pinned GTR code, same seed and training hyperparameters for that dataset. Launch once; do **not** run a second batch while the first worker is active. The Phase2 launcher starts the background process and prints the batch ID. When it is complete, you can run the same validation diagnostic for ETTm1-192 by passing its **new batch ID**:

```bash
./.venv/bin/python -m phase3.cli --batch NEW_BATCH_ID --dataset ETTm1 --horizon 192 --limit 1024
```

No early stopping or hyperparameters are modified by Phase3. Since ETTm1 was previously used for model ideas, this is still **development evidence**, not a new untouched final test. Reserve an uninspected dataset/horizon after the hypothesis and model design are fixed.

## Interpreting Phase3

- If the original GTR wins most channels/conditions in Weather, state clearly that the auxiliary branch is broadly redundant in this dataset.
- If improvements are concentrated in high-local-change or high-drift windows, test an architecture specifically addressing that condition **on validation**; do not automatically add a generic gate.
- If phase-fit R² shows no monotonic relation to gain, reject the current phase-fit proxy; it is not validated merely because its value can be plotted.
- If ETTm1-192 has a different conditional pattern, treat the pattern as hypothesis-generating, then design an appropriate solution and verify on an untouched setting.

## Requirements

Use the existing `.venv`; requires already installed `torch`, `numpy`, `pandas`, `scikit-learn` and original GTR dependencies. Full Weather-96 Phase2 run, including best checkpoints under `runs/<ID>/source/checkpoints/...`, must be available. Phase3 does not download datasets or contact your server from ChatGPT.
