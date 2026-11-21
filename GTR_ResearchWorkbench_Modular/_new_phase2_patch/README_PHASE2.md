# GTR Phase2: Problem-driven experiment patch

This is a **non-destructive patch** for the existing `GTR_ResearchWorkbench_Modular` project, which should already contain the Phase1 model file `GTRLinearResidual.py` and official GTR source. The Phase2 scheduler itself is independent of Phase1. Do not unpack this archive over existing project files. Unzip into a separate folder, then run `phase2/install.py`.

The new research designs: `GTRPhaseInnovation` and `GTRPhaseAdaptive`. They keep the original Q, global phase retrieval, 2D convolution and MLP **exactly** as upstream while using a past-only affine projection of learned phase reference to construct supplementary innovations. `GTRLinearResidual` is a matched baseline from Phase1.

## Installation (copy-paste)

```bash
cd /root/autodl-tmp/GTR_ResearchWorkbench_Modular
mkdir -p _new_phase2_patch
unzip -n GTR_Phase2_Problem_Driven_v1.zip -d _new_phase2_patch
./.venv/bin/python _new_phase2_patch/phase2/install.py --root . --dry-run
./.venv/bin/python _new_phase2_patch/phase2/install.py --root .
./.venv/bin/python -m unittest discover -s tests -p test_phase2_model.py -v
```

## Run and monitor

```bash
bash 11_Phase2_CPU.sh Weather 96 2024
bash 13_Phase2_Status.sh
# Only after CPU smoke succeeds:
bash 12_Phase2_GPU.sh Weather 96 2024
bash 13_Phase2_Status.sh
bash 14_Phase2_Validation.sh
```

If no matching official GTR benchmark is found for a dataset/horizon/seed, the batch automatically runs one first. Runs are sequential in background, and results are in `runs/` with snapshots. Streamlit sidebar will have the Phase2 page after restart/refresh.

For interpretation on a finished extended model:

```bash
./.venv/bin/python -m phase2.explain RUN_ID --limit 64
```

Read `docs/PHASE2_RESEARCH_DESIGN.md` for mechanism, validity limitations, hypothesis tests and future paper structure. The first validation setting is Weather-H96; ETTm1-H96 test results already appeared in the first phase and should not be used for method selection.

**No GPU results are included.** The patch includes CPU unit checks only. Full future results require running this on your AutoDL server.
