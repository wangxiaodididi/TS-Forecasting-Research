LG-Position v1 — for existing XLinear_ResearchLab_Clean project
1) unzip -o XLinear_LG_Position_v1.zip -d .
2) bash LG_Position_v1/01_CPU_check.sh
3) bash LG_Position_v1/02_GPU_plan.sh
4) bash LG_Position_v1/03_GPU_full.sh
5) python LG_Position_v1/scripts/summarize.py runs/<run_id>/results/all_results.csv
Existing LG v1 required. No changes to official source. Full details: LG_Position_v1/README_CN.md
