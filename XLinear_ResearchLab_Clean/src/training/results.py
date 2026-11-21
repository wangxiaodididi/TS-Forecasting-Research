"""每组配对测试自动汇总；种子配对，无跨数据集/长度混用。"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean,stdev


def atomic_json(path:Path,data:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(path)


def csv_write(path:Path,cols:list[str],rows:list[dict]):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=cols)
        writer.writeheader();writer.writerows(rows)
    tmp.replace(path)


def refresh(run_dir:Path):
    records=[]
    for file in (run_dir/'jobs').glob('*/result.json'):
        data=json.loads(file.read_text(encoding='utf-8'))
        if data.get('status')=='completed':records.append(data)
    records.sort(key=lambda x:(x['dataset'],x['horizon'],x['seed'],x['variant']))
    output=run_dir/'results'
    csv_write(output/'all_results.csv',['dataset','horizon','seed','variant','mse','mae','status','job_dir','params'],records)
    lookup={(r['dataset'],r['horizon'],r['seed'],r['variant']):r for r in records}
    pairs=[]
    for r in records:
        if r['variant']=='official':continue
        base=lookup.get((r['dataset'],r['horizon'],r['seed'],'official'))
        if not base:continue
        pairs.append(dict(dataset=r['dataset'],horizon=r['horizon'],seed=r['seed'],variant=r['variant'],
            official_mse=base['mse'],variant_mse=r['mse'],
            improvement_pct_vs_official=100*(base['mse']-r['mse'])/base['mse'],
            official_mae=base['mae'],variant_mae=r['mae'],
            mae_improvement_pct=100*(base['mae']-r['mae'])/base['mae']))
    keys=['dataset','horizon','seed','variant','official_mse','variant_mse',
          'improvement_pct_vs_official','official_mae','variant_mae','mae_improvement_pct']
    csv_write(output/'paired_comparison.csv',keys,pairs)
    grouped=defaultdict(list)
    for row in pairs:grouped[(row['dataset'],row['horizon'],row['variant'])].append(row)
    summary=[]
    for (dataset,h,variant),rows in sorted(grouped.items()):
        g=[r['improvement_pct_vs_official'] for r in rows]
        summary.append(dict(dataset=dataset,horizon=h,variant=variant,
            paired_seeds=len(rows),mean_improvement_pct=mean(g),
            std_improvement_pct=stdev(g) if len(g)>1 else 0.0,
            positive_seeds=sum(x>0 for x in g),
            mean_original_mse=mean(r['official_mse'] for r in rows),
            mean_variant_mse=mean(r['variant_mse'] for r in rows)))
    csv_write(output/'summary.csv',[
        'dataset','horizon','variant','paired_seeds','mean_improvement_pct',
        'std_improvement_pct','positive_seeds','mean_original_mse','mean_variant_mse'],summary)
