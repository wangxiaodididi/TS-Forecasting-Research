#!/usr/bin/env python3
"""Read actual validation trajectories, no test-set selection."""
from pathlib import Path
import csv
import re
ROOT=Path(__file__).resolve().parent
pat = re.compile(r'Epoch:\s*(\d+),\s*Steps:\s*\d+\s*\|\s*Train Loss:\s*([0-9.e+-]+)\s+Vali Loss:\s*([0-9.e+-]+)')
rows=[]
for log in sorted((ROOT/'runs').glob('*/h*/seed*/*/train.log')):
    params=log.relative_to(ROOT/'runs').parts
    lines=log.read_text(errors='replace')
    matches=[(int(x),float(y),float(z)) for x,y,z in pat.findall(lines)]
    success = '[END]' in lines  # runner writes end to stdout not to log; use config instead
    conf=log.parent/'train_config.json'
    import json
    meta=json.loads(conf.read_text()) if conf.is_file() else {}
    best=min(matches,key=lambda t:t[2]) if matches else None
    rows.append(dict(dataset=params[0],horizon=int(params[1][1:]),seed=int(params[2][4:]),variant=params[3],
                     status='completed' if meta.get('exit_code')==0 and matches else 'incomplete',
                     best_val_mse=best[2] if best else '',best_epoch=best[0] if best else '',last_epoch=matches[-1][0] if matches else '',
                     elapsed_sec=meta.get('elapsed_sec',''),log=str(log)))
cols=['dataset','horizon','seed','variant','status','best_val_mse','best_epoch','last_epoch','elapsed_sec','log']
out=ROOT/'validation_summary.csv'
with out.open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
for row in rows:
    print('{dataset:8} {horizon:<4} {seed} {variant:8} {status:10} val={best_val_mse} best_ep={best_epoch}'.format(**row))
print('CSV:',out)
print('Model direction choice must use validation only. No test metrics are read.')
