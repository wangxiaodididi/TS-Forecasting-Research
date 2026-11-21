#!/usr/bin/env python3
"""Run fixed LG embedding replication groups and assemble paired official comparisons.
Read-only with respect to model/training code and existing runs.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
STATE = HERE / "state" / "new_runs.jsonl"
OUTPUT = HERE / "outputs"
VARIANTS = ("official", "lg_short", "lg_dual")
# No parameter search: original profiles from configs/experiment.yaml only.
GROUPS = {
    "ett_192": {"dataset": "ETTh1", "horizon": 192, "seeds": (2025, 2026)},
    "weather_96": {"dataset": "Weather", "horizon": 96, "seeds": (2025, 2026)},
    "ett_96_seeds": {"dataset": "ETTh1", "horizon": 96, "seeds": (2026, 2027, 2028)},
}
REFERENCE = "20261116_220929_full_feb4a7"


def now():
    return datetime.now(timezone.utc).isoformat()


def get_python():
    return os.environ.get("PYTHON", sys.executable)


def execute(cmd, *, log=None):
    print("COMMAND:", " ".join(map(str, cmd)), flush=True)
    if log is None:
        return subprocess.run(cmd, cwd=ROOT, check=False).returncode
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8') as fh:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, errors='replace')
        for ln in p.stdout:
            print(ln, end='', flush=True)
            fh.write(ln)
        return p.wait()


def args_for(g):
    spec = GROUPS[g]
    return ['--datasets', spec['dataset'], '--horizons', str(spec['horizon']),
            '--variants', ','.join(VARIANTS), '--seeds', ','.join(map(str, spec['seeds']))]


def read_state():
    rows = []
    if STATE.is_file():
        for line in STATE.read_text(encoding='utf-8').splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def register_run(row):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with STATE.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + '\n')


def inspect_project():
    req = [ROOT/'run.sh', ROOT/'configs/experiment.yaml', ROOT/'src/training/runner.py',
           ROOT/'src/models/registry.py', ROOT/'src/models/extensions/local_global_embedding.py']
    missing = [str(p.relative_to(ROOT)) for p in req if not p.is_file()]
    if missing:
        raise RuntimeError('请在已安装 LG v1 的 XLinear_ResearchLab_Clean 根目录解压。缺少：'+', '.join(missing))
    from src.core.config import read_config, make_jobs
    cfg = read_config()
    for group in GROUPS:
        spec = GROUPS[group]
        for seed in spec['seeds']:
            jobs = make_jobs(cfg,'full',datasets=[spec['dataset']], horizons=[spec['horizon']],
                             variants=list(VARIANTS),seeds=[seed])
            assert len(jobs)==len(VARIANTS)
    print('已通过注册/官方参数配置检查。')
    print('Python:', get_python())
    print('官方训练入口、LG模块、三个实验组均在项目中存在。')
    return cfg


def successful_runs_for(group):
    spec=GROUPS[group]
    expected = {(spec['dataset'],spec['horizon'],seed,v) for seed in spec['seeds'] for v in VARIANTS}
    out=[]
    for record in read_state():
        if record.get('group')!=group: continue
        rd = ROOT/'runs'/record['run_id']
        if not rd.is_dir(): continue
        completed=set()
        for f in rd.glob('jobs/*/result.json'):
            r=json.loads(f.read_text(encoding='utf-8'))
            if r.get('status')=='completed':
                completed.add((r['dataset'],int(r['horizon']),int(r['seed']),r['variant']))
        if expected.issubset(completed):
            out.append(rd)
    return out


def latest_run(before):
    runs = ROOT/'runs'
    return {p.name for p in runs.iterdir() if p.is_dir()} - before if runs.is_dir() else set()


def plan(groups):
    inspect_project()
    for g in groups:
        print(f'\n### {g}: {GROUPS[g]} / {len(GROUPS[g]["seeds"])*len(VARIANTS)} jobs')
        cmd=[get_python(), '-m','src.training.runner','--mode','full','--plan']+args_for(g)
        rc=execute(cmd)
        if rc: raise RuntimeError(f'计划失败 {g}: exit={rc}')
    print('\n计划共',sum(len(GROUPS[g]['seeds'])*len(VARIANTS) for g in groups),'次新训练。')
    print('历史ETTh1-96 Seed2025默认只读取，不重复训练。')


def run(groups,force=False):
    inspect_project()
    for g in groups:
        if successful_runs_for(g) and not force:
            print(f'\nSKIP {g}: 之前已有完整配对结果；如确需重跑，使用 --force')
            continue
        before=set(p.name for p in (ROOT/'runs').iterdir() if p.is_dir()) if (ROOT/'runs').exists() else set()
        logfile=HERE/'state'/f'{g}_{datetime.now():%Y%m%d_%H%M%S}.log'
        print(f'\n===== 正式训练 {g} | {GROUPS[g]} =====',flush=True)
        rc=execute(['bash','run.sh','full']+args_for(g),log=logfile)
        after=latest_run(before)
        # Not necessarily exactly one when external processes were creating dirs; filter by current training log.
        found=[]
        content=logfile.read_text(encoding='utf-8',errors='replace')
        for name in sorted(after):
            rd=ROOT/'runs'/name
            if name in content:
                found.append(name)
        if len(found)!=1:
            # In the absence of a successful run, still record an unambiguous run with the matching manifest.
            for name in sorted(after):
                mf=ROOT/'runs'/name/'manifest.json'
                if not mf.exists():continue
                doc=json.loads(mf.read_text(encoding='utf-8'))
                if len(doc.get('jobs',[]))==len(GROUPS[g]['seeds'])*len(VARIANTS) and all(
                    (j['dataset']==GROUPS[g]['dataset'] and int(j['horizon'])==GROUPS[g]['horizon'])
                    for j in doc['jobs']):
                    if name not in found:found.append(name)
        if len(found)!=1:
            raise RuntimeError(f'无法唯一识别本次训练输出目录 {g}: {sorted(after)}，请手动核对日志 {logfile}')
        register_run({'group':g,'run_id':found[0],'exit_code':rc,'recorded_at':now(), 'log':str(logfile.relative_to(ROOT))})
        print(f'已记录：{g} -> runs/{found[0]}, exit={rc}')
        if rc:
            print(f'训练中止，请查看 {logfile}，不能用残缺结果计算均值。',file=sys.stderr)
            raise SystemExit(rc)
    print('\n各组运行结束。汇总：bash LG_Replication_v2/03_汇总.sh')


def get_rows(run_dir):
    mf=run_dir/'manifest.json'
    if not mf.is_file():raise ValueError(f'缺少manifest: {mf}')
    meta=json.loads(mf.read_text(encoding='utf-8'))
    rows=[]
    for f in sorted((run_dir/'jobs').glob('*/result.json')):
        r=json.loads(f.read_text(encoding='utf-8'))
        if r.get('status')!='completed':continue
        r={**r,'source_run':run_dir.name, 'source_kind':'reference' if run_dir.name==REFERENCE else 'replication'}
        r['horizon']=int(r['horizon']);r['seed']=int(r['seed'])
        if 'mse' not in r or 'mae' not in r:continue
        rows.append(r)
    return meta,rows


def fingerprint(meta, job):
    c=meta.get('configuration',{})
    p=next((j.get('profile') for j in meta.get('jobs',[]) if j['dataset']==job['dataset'] and int(j['horizon'])==job['horizon']),None)
    if p is None: raise ValueError(f'manifest中缺少匹配profile: {job}')
    return (meta.get('source_commit'),meta.get('model_git_blob'),
            json.dumps(meta.get('dataset_sha256',{}),sort_keys=True),
            c.get('runtime',{}).get('precision'), c.get('runtime',{}).get('workers'),
            json.dumps(p,sort_keys=True))


def write_csv(path,rows,cols):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def collect(reference=REFERENCE,strict=True):
    inspect_project()
    refdir=ROOT/'runs'/reference
    if not refdir.is_dir():
        print(f'警告：原Seed2025历史结果目录不存在：{refdir}；不伪造基准。')
    candidates=[]
    if refdir.is_dir():candidates.append(refdir)
    for group in GROUPS:
        runs=successful_runs_for(group)
        if runs:
            if len(runs)>1:
                print(f'警告：{group}已有多个完整运行，仅取最新登记的一组，避免重复计数。')
            candidates.append(runs[-1])
        else:
            print(f'未完成整组：{group}')
    records={}
    fps={}
    duplicate=[]
    for rd in candidates:
        meta,allrows=get_rows(rd)
        for r in allrows:
            key=(r['dataset'],r['horizon'],r['seed'],r['variant'])
            if key[3] not in VARIANTS:continue
            # Take only previous ETTh1-96 seed2025 from reference; other experiments must be from new runs.
            if rd==refdir and key[:3]!=('ETTh1',96,2025):continue
            if key in records:
                duplicate.append(key);continue
            records[key]=r
            fps[key]=fingerprint(meta,r)
    if duplicate:raise ValueError('重复配对任务，结果不可无歧义汇总：'+str(duplicate))
    baseline_fingerprints={}
    for key,fp in fps.items():
        groupkey=key[:3]
        if groupkey in baseline_fingerprints and baseline_fingerprints[groupkey]!=fp:
            raise ValueError(f'同一seed对照的训练设置不一致：{groupkey}')
        baseline_fingerprints[groupkey]=fp
    # Dataset-wide source versions/fingerprints should match, apart from distinct H profiles.
    for dataset in ('ETTh1','Weather'):
        fingerprints={(fp[0],fp[1],fp[2],fp[3],fp[4]) for k,fp in fps.items() if k[0]==dataset}
        if len(fingerprints)>1:raise ValueError(f'{dataset}跨运行的官方源码/数据/精度/worker设置不一致。')
    expected=[]
    for key in VARIANTS: expected.append(('ETTh1',96,2025,key))
    for spec in GROUPS.values():
        for seed in spec['seeds']:
            for v in VARIANTS:expected.append((spec['dataset'],spec['horizon'],seed,v))
    missing=[x for x in expected if x not in records]
    print(f'已有{len(records)}/{len(expected)}条所需原始记录；缺少{len(missing)}条。')
    if missing:
        for x in missing:print('  MISSING',x)
        if strict:raise SystemExit('汇总未就绪：使用 --allow-partial 仅生成开发性汇总。')
    raw=[]
    for key in expected:
        if key not in records:continue
        r=records[key]
        raw.append({**{k:r.get(k) for k in ('dataset','horizon','seed','variant','mse','mae','params','source_run')},
                    'experiment_set': 'original_2025' if key[:3]==('ETTh1',96,2025) else 'new'})
    cols=['dataset','horizon','seed','variant','mse','mae','params','source_run','experiment_set']
    write_csv(OUTPUT/'all_selected_results.csv',raw,cols)
    paired=[]
    for dataset,horizon,seed,variant in expected:
        if variant=='official' or (dataset,horizon,seed,variant) not in records or (dataset,horizon,seed,'official') not in records:continue
        ref=records[(dataset,horizon,seed,'official')]; r=records[(dataset,horizon,seed,variant)]
        paired.append({'dataset':dataset,'horizon':horizon,'seed':seed,'variant':variant,
                       'official_mse':ref['mse'],'variant_mse':r['mse'],
                       'mse_gain_pct':(ref['mse']-r['mse'])/ref['mse']*100,
                       'official_mae':ref['mae'],'variant_mae':r['mae'],
                       'mae_gain_pct':(ref['mae']-r['mae'])/ref['mae']*100,
                       'params_official':ref.get('params'),'params_variant':r.get('params')})
    pcols=['dataset','horizon','seed','variant','official_mse','variant_mse','mse_gain_pct','official_mae','variant_mae','mae_gain_pct','params_official','params_variant']
    write_csv(OUTPUT/'paired_deltas.csv',paired,pcols)
    by=defaultdict(list)
    for p in paired: by[(p['dataset'],p['horizon'],p['variant'])].append(p)
    summary=[]
    for (d,h,v),rows in sorted(by.items()):
        summary.append({'dataset':d,'horizon':h,'variant':v,'paired_seeds':len(rows),
                        'mean_official_mse':mean(x['official_mse'] for x in rows),
                        'mean_variant_mse':mean(x['variant_mse'] for x in rows),
                        'mean_mse_gain_pct':mean(x['mse_gain_pct'] for x in rows),
                        'mse_improved_seeds':sum(x['mse_gain_pct']>0 for x in rows),
                        'mean_official_mae':mean(x['official_mae'] for x in rows),
                        'mean_variant_mae':mean(x['variant_mae'] for x in rows),
                        'mean_mae_gain_pct':mean(x['mae_gain_pct'] for x in rows),
                        'mae_improved_seeds':sum(x['mae_gain_pct']>0 for x in rows)})
    scols=['dataset','horizon','variant','paired_seeds','mean_official_mse','mean_variant_mse','mean_mse_gain_pct','mse_improved_seeds','mean_official_mae','mean_variant_mae','mean_mae_gain_pct','mae_improved_seeds']
    write_csv(OUTPUT/'group_summary.csv',summary,scols)
    report=['# LG-Embedding XLinear: 固定协议复现实验汇总','',f'生成日期：{now()}','',
            f'结果覆盖：{len(raw)}/{len(expected)} 条任务记录；已匹配 {len(paired)} 个模型-基准配对。','',
            '> 正向改善定义为 (Official误差-改造误差)/Official误差 ×100%。','',
            '| 数据集 | 预测步长 | 变体 | 配对种子数 | MSE平均改善 | MSE获胜种子 | MAE平均改善 | MAE获胜种子 |',
            '|---|---:|---|---:|---:|---:|---:|---:|']
    for x in summary:
        report.append(f"| {x['dataset']} | {x['horizon']} | {x['variant']} | {x['paired_seeds']} | {x['mean_mse_gain_pct']:+.3f}% | {x['mse_improved_seeds']}/{x['paired_seeds']} | {x['mean_mae_gain_pct']:+.3f}% | {x['mae_improved_seeds']}/{x['paired_seeds']} |")
    report+=['','## 解释边界','','- 平均改善率是同种子的改善百分比算术平均，不是均值MSE之比。',
             '- 仅为已有方案的固定协议复核，不根据本轮测试集结果修改结构或参数。',
             '- 2至4个种子用于重复性初判，不足以作为强统计显著性证据。',
             '- 若MSE改善但MAE下降，请明确报告指标权衡，不要声称全面优于官方。']
    if missing: report+=['','## 尚缺记录']+[f'- {x}' for x in missing]
    (OUTPUT/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('已生成：',OUTPUT/'all_selected_results.csv',OUTPUT/'paired_deltas.csv',OUTPUT/'group_summary.csv',OUTPUT/'report.md',sep='\n  ')


def main():
    p=argparse.ArgumentParser()
    subs=p.add_subparsers(dest='mode',required=True)
    for name in ('plan','run'):
        sp=subs.add_parser(name)
        sp.add_argument('--groups',nargs='+',choices=list(GROUPS),default=list(GROUPS))
        if name=='run':sp.add_argument('--force',action='store_true')
    sp=subs.add_parser('collect')
    sp.add_argument('--reference-run',default=REFERENCE)
    sp.add_argument('--allow-partial',action='store_true')
    args=p.parse_args()
    if args.mode=='plan':plan(args.groups)
    elif args.mode=='run':run(args.groups,args.force)
    else:collect(args.reference_run,strict=not args.allow_partial)

if __name__=='__main__':main()
