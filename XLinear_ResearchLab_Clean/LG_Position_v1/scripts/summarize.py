#!/usr/bin/env python3
"""Strict same-seed comparison for a single official training run's all_results.csv.

Reads results only; never changes model, training settings, checkpoints, or test outcomes.
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
REQUIRED = {'official','lg_short','lg_position'}


def read_rows(input_file: Path):
    with input_file.open('r',encoding='utf-8-sig',newline='') as fh:
        rows=list(csv.DictReader(fh))
    if not rows:
        raise ValueError('empty CSV')
    present={(r['dataset'],int(r['horizon']),int(r['seed']),r['variant']) for r in rows}
    if len(rows)!=len(present):
        raise ValueError('duplicate dataset/horizon/seed/variant rows')
    jobs=defaultdict(dict)
    for r in rows:
        if r['variant'] not in REQUIRED:continue
        if 'status' in r and r['status'] not in ('completed','complete','success'):
            raise ValueError(f"incomplete job: {r}")
        k=(r['dataset'],int(r['horizon']),int(r['seed']))
        jobs[k][r['variant']]=dict(mse=float(r['mse']),mae=float(r['mae']),
                                  params=r.get('params',''))
    if not jobs:
        raise ValueError('CSV contains no official/short/position comparison jobs')
    for k,by in jobs.items():
        if set(by)!=REQUIRED:
            raise ValueError(f'Incomplete paired group {k}: {sorted(by)}')
    return jobs


def write_csv(file: Path, data: list[dict], headers: list[str]):
    file.parent.mkdir(parents=True,exist_ok=True)
    with file.open('w',encoding='utf-8-sig',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=headers,extrasaction='ignore')
        w.writeheader();w.writerows(data)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('csv',type=Path,help='runs/<run_id>/results/all_results.csv')
    args=ap.parse_args()
    source=args.csv.resolve()
    jobs=read_rows(source)
    summary=defaultdict(list)
    paired=[]
    for (ds,h,seed), v in sorted(jobs.items()):
        base=v['official']
        for variant in ('lg_short','lg_position'):
            change={m:(base[m]-v[variant][m])/base[m]*100 for m in ('mse','mae')}
            row=dict(dataset=ds,horizon=h,seed=seed,variant=variant,
                official_mse=base['mse'],variant_mse=v[variant]['mse'],mse_gain_pct=change['mse'],
                official_mae=base['mae'],variant_mae=v[variant]['mae'],mae_gain_pct=change['mae'],
                official_params=base['params'],variant_params=v[variant]['params'])
            paired.append(row);summary[(ds,h,variant)].append(row)
    out=[]
    for (ds,h,variant),r in sorted(summary.items()):
        out.append(dict(dataset=ds,horizon=h,variant=variant,n_seed=len(r),
            avg_official_mse=mean(x['official_mse'] for x in r),
            avg_variant_mse=mean(x['variant_mse'] for x in r),
            mean_mse_gain_pct=mean(x['mse_gain_pct'] for x in r),
            mse_positive_seeds=sum(x['mse_gain_pct']>0 for x in r),
            avg_official_mae=mean(x['official_mae'] for x in r),
            avg_variant_mae=mean(x['variant_mae'] for x in r),
            mean_mae_gain_pct=mean(x['mae_gain_pct'] for x in r),
            mae_positive_seeds=sum(x['mae_gain_pct']>0 for x in r)))
    dest=HERE/'outputs'/source.parents[1].name
    write_csv(dest/'paired.csv',paired,list(paired[0]))
    write_csv(dest/'summary.csv',out,list(out[0]))
    report=['# LG-Position 同种子对照', '',
            f'输入文件：`{source}`',
            '本表为已有测试集结果的开发性比较，不构成新的独立样本验证。',
            '',
            '| 数据集 | H | 模型 | 配对种子数 | MSE增益(%) | MSE胜出种子 | MAE增益(%) | MAE胜出种子 |',
            '|---|---:|---|---:|---:|---:|---:|---:|']
    for x in out:
        report.append(f"| {x['dataset']} | {x['horizon']} | {x['variant']} | {x['n_seed']} | {x['mean_mse_gain_pct']:+.3f} | {x['mse_positive_seeds']}/{x['n_seed']} | {x['mean_mae_gain_pct']:+.3f} | {x['mae_positive_seeds']}/{x['n_seed']} |")
    report.extend(['','比较要点：LG-Position与LG-Short参数量相同，区别是局部特征池化方式。',
      '不要把本轮重复使用的ETTh1/Weather测试集当作未参与模型开发的独立评价。'])
    (dest/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('\n'.join(report))
    print('\nFiles written:',dest)

if __name__=='__main__':main()
