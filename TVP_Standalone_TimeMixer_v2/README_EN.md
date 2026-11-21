# TVP TimeMixer v3: validation matrix and temporal prediction head

Copy this patch **into the existing standalone TimeMixer v2 root**. Use `unzip -o TVP_NextStage_v3.zip -d .`, preserving `code/` and `data/`. Run `python 13_cpu_checks.py` first. See `README_CN.md` for the full workflow.

For original-model + TVP matched validation:

```sh
python 08_validation_matrix.py --tasks 'ETTm1:96:2024,ETTm1:192:2024,ETTh1:96:2024,Weather:96:2024' --train-missing
python 09_results_summary.py
```

For the next-module (nonlinear temporal-head) experiment:

```sh
python 15_official_head_smoke.py
python 11_train_head.py --dataset ETTm1 --horizon 96 --seed 2024 --mode nonlinear --rank 8
python 12_extract_head.py --trial ETTm1_h96_seed2024_nonlinear_rank8
```

This patch does not assert new GPU results. Results from already completed ETTh1 proxy CPU tests are not actual TimeMixer metrics. The nonlinear temporal-head variant independently retrains the backbone; it is not an online TVP postprocessor.
