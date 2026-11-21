# Phase 2 — Problem-driven GTR research (2026-11)

## Research question

**How can a GTR model use locally observed innovations that remain after aligning a learned global-phase reference to the current input window?**

The official GTR (Cao et al., ICLR 2026) learns a cycle embedding `Q`, retrieves it by absolute phase, fuses it with the local input via 2-D convolution, and predicts with an MLP. Our hypothesis is narrower: the **phase-aligned latent reference may capture a coherent part of the recent history, while the residual after past-only affine matching still contains complementary forecast signal**. It is a hypothesis to validate, not an established defect of the original GTR.

The Phase-1 experiment on ETTm1 H96 showed that GTRLinearResidual (best validation MSE 0.3716159) slightly beat the HaarDual branch (0.3719454) and GTR (0.3733875). Thus: a raw linear forecast correction can help; a one-level Haar transform has not demonstrated independent advantage. This result motivates a **different forecast input (global-phase-unexplained innovation)** instead of more frequency partitions.

## Architecture (core untouched)

```text
x ──[official GTR.Model: Q, phase retrieval, Conv2D, MLP, RevIN]── y_GTR
│
└──[normalize like upstream]── x_n ───────────┐
                                                ├─ past-only affine fit
   Q ──[official phase indexing + GTR.linear]── q ┘       ↓
                                                innovation e = x_n - fit(q)
                                                         ↓
                                            linear innovation forecasting head
                                                         ↓
                                            optional state-conditioned gain
                                                         ↓
final y = y_GTR + denormalized innovation correction
```

The side computation uses `with torch.no_grad()` for the learned reference, to prevent the correction path from learning an artificial Q. However, the original GTR still trains Q, retrieval and Conv2D normally from the original forecasting loss. The innovation head initializes at zero, therefore each new model initially equals the official GTR for the same shared weights.

## Mathematical specification

For sample b, variable c and window length T, normalize x with upstream RevIN convention. Let `q=Linear(Q[phase indices])` be the latent reference with an arbitrary scale and offset. Fit

`a = Cov_T(x_n,q) / (Var_T(q)+0.001);  q_fit = a*(q-mean_T(q)); e = (x_n - mean_T(x_n)) - q_fit`.

`phase_fit_R2 = clip(1 - mean(e²)/max(mean(centered(x_n)²), 1e-8), 0, 1)`.

`y_PCI = y_GTR + std(x) * W_e e` (each channel shares W_e, just like the original time-head).

Optional `PhaseAdaptive`: compute only from the past window (phase R², recent residual-difference energy and first/last-quarter residual drift):

`gain = 1 + 0.5*tanh(Linear([fit_R2, log(1+diff_energy), log(1+drift)]))`.

`y_PCA = y_GTR + gain * std(x) * W_e e`.

The gain layer contains only 4 parameters. No past-window future leakage. Both proposed extensions use the *same* official GTR core and the *same* author training parameters as the baseline.

## Essential comparisons (not a menu of unrelated ideas)

1. **GTR** — original author baseline, unchanged.
2. **GTRLinearResidual** — exact Phase-1 matched parameter control using raw normalized past history.
3. **GTRPhaseInnovation** — replace raw past input in the correction with phase-calibrated innovation.
4. **GTRPhaseAdaptive** — additionally test whether conditioning its correction on phase agreement gives extra value.

All four share the author train/val/test splits, identical source commit, lookback, forecast horizon, epochs, patience, seed, and optimizer schedule. Runs are sequential to avoid GPU interference.

Primary: best **validation** MSE (then MAE for context). Run extra seeds only **after locking a small set of candidates**; do not cherry pick test forecasts. Final test reporting requires a held-out setting not inspected for architecture decisions; ETTm1 H96 is *already exploratory* from Phase 1.

Start with `Weather H96` and `ETTm1 H192`, then lock one variant and check an untouched horizon/dataset. A previous ETTm1 H96 test improvement alone is insufficient novelty.

## Interpretation and falsification

Export per-window, per-channel phase-fit R² and additive correction. Predictions are *numerically* decomposable; this is not causal attribution.

Evidence in favor requires both:

- The phase innovation model outperforms raw linear residual on a validation setting under identical training protocol.
- The sample-level correction advantage varies systematically with the past-only phase-fit index (or drift), with a credible effect size beyond random seed noise.

Failure criteria: if the raw linear branch matches or beats phase-conditioned models across validations, the phase-mismatch thesis is not supported. Simplify to the linear control or change the research question, rather than fabricate an interpretation.

Caveat: GTR Q is a **learned latent** temporal embedding, not an observed seasonal signal. The fitted R² only quantifies agreement with the historical window after affine calibration, and no two independent full cycles are required inside the 96-step input.

## References / overlap

- Cao et al. (2026), *Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval*, ICLR 2026. https://proceedings.iclr.cc/paper_files/paper/2026/hash/7a3284ce53f620d244fca0c79d94c478-Abstract-Conference.html
- Wang et al. (2026), *AlignTime: Interperiodic phase alignment sampling for time-series forecasting*, Information Processing & Management. https://www.sciencedirect.com/science/article/pii/S0306457325002377
- Ding et al. (2026), *Global-local coherency contrastive learning for context-aware time series forecasting*, Knowledge-Based Systems. https://www.sciencedirect.com/science/article/pii/S0950705125017836

Phase alignment and local/global context modeling are not themselves new. The novelty claim must be specifically about **conditioning an auxiliary prediction on the part of history unexplained by the aligned learned reference** and must be supported by matched controls and validation behavior. No novelty claim is final at this stage.

## Run it

Extract the Phase2 ZIP into a separate folder. From the existing workbench root, run:

```bash
./.venv/bin/python /path/to/phase2_package/phase2/install.py --root . --dry-run
./.venv/bin/python /path/to/phase2_package/phase2/install.py --root .
./.venv/bin/python -m unittest discover -s tests -p test_phase2_model.py -v
bash 11_Phase2_CPU.sh Weather 96 2024
bash 13_Phase2_Status.sh
bash 12_Phase2_GPU.sh Weather 96 2024
bash 14_Phase2_Validation.sh
# OPTIONAL: explanatory CSV from an already completed Phase2 run ID:
./.venv/bin/python -m phase2.explain RUN_ID --limit 64
```

**The CLI initiates background jobs.** It does not run the user's RTX GPU from ChatGPT. Refresh Streamlit (old dashboard) to expose Phase2 page; its URL remains your existing local forwarded Streamlit port.
