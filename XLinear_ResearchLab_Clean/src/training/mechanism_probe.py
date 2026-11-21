"""XLinear 现有 checkpoint 的机制诊断（只读，不训练）。

在同一确定性验证/测试样本上完成三项诊断：
1) Token 已训练分支 on/off（只将 gain_logit 临时置0）反事实；
2) 历史频率谱分组下的 Official/变体误差；
3) 频率特征与 TGM、VGM 门控的描述性相关。

使用既有 manifest、command、checkpoint、官方数据提供器，输出到
runs/<run>/analysis/mechanism_probe/；绝不写回 checkpoint 或训练结果。
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.core.config import ROOT
from src.runtime.bootstrap import UPSTREAM, prepare_datasets, verify_source
from src.models.registry import build_variant_model

SPLIT_LABELS = ('val', 'test')


def spectral_features(x: np.ndarray, interval_hours: float = 1.0) -> dict[str, np.ndarray]:
    """只使用预测时已知历史 OT，逐窗口线性去趋势 + Hann FFT。

    x: [B, L]，已经按训练段 scaler 标准化；不会读取未来标签。
    频带按 k=1,2 / k=3..7 / k>=8 定义，适合固定 L=96 的前期诊断。
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 16:
        raise ValueError('频谱输入必须是形如 [B,L] 且 L>=16 的历史矩阵')
    b, length = x.shape
    time = np.arange(length, dtype=np.float64)
    time -= time.mean()
    demeaned = x - np.mean(x, axis=1, keepdims=True)
    slope = demeaned @ time / np.dot(time, time)
    detrended = demeaned - slope[:, None] * time[None, :]
    window = np.hanning(length)
    spectrum = np.abs(np.fft.rfft(detrended * window, axis=1)) ** 2
    spectrum[:, 0] = 0
    # 常数窗口的谱功率全为零，统一标记为 flat，不伪称其具有某个周期。
    power = spectrum[:, 1:].sum(axis=1)
    safe = np.maximum(power, 1e-20)
    ks = np.arange(spectrum.shape[1])
    slow = spectrum[:, (ks >= 1) & (ks <= 2)].sum(axis=1) / safe
    middle = spectrum[:, (ks >= 3) & (ks <= 7)].sum(axis=1) / safe
    fast = spectrum[:, ks >= 8].sum(axis=1) / safe
    peak_k = spectrum[:, 1:].argmax(axis=1) + 1
    periods = length / peak_k
    bands = np.where(peak_k <= 2, 'slow', np.where(peak_k <= 7, 'middle', 'fast'))
    flat = power < 1e-14
    bands[flat] = 'flat'
    periods = periods.astype(float)
    periods[flat] = np.nan
    # 24h 特征只有覆盖至少24h历史时才计算；否则不作日周期解释。
    cover_hours = length * interval_hours
    daily = np.full(b, np.nan, dtype=float)
    if cover_hours >= 24:
        freq_per_hour = np.fft.rfftfreq(length, d=interval_hours)
        daily_band = (freq_per_hour >= 1/30) & (freq_per_hour <= 1/18)
        daily[:] = spectrum[:, daily_band].sum(axis=1) / safe
        daily[flat] = np.nan
    slow[flat] = np.nan
    middle[flat] = np.nan
    fast[flat] = np.nan
    return {'peak_band': bands, 'peak_period_steps': periods,
            'peak_period_hours': periods * interval_hours,
            'slow_energy_share': slow, 'middle_energy_share': middle,
            'fast_energy_share': fast, 'daily_energy_share': daily,
            'normalized_trend_slope': slope / (np.std(x, axis=1) + 1e-10)}


def _options(spec):
    p = spec['profile']; h = int(spec['horizon'])
    return SimpleNamespace(seq_len=96,pred_len=h,label_len=48,features='M',target='OT',
        d_model=int(p['d_model']),enc_in=int(p['channels']),
        t_ff=int(p['t_ff']),c_ff=int(p['c_ff']),usenorm=1,
        embed_dropout=float(p['embed_dropout']),head_dropout=float(p['head_dropout']),
        t_dropout=float(p['t_dropout']),c_dropout=float(p['c_dropout']),
        data=p['data'],root_path=str(UPSTREAM/'dataset'),data_path=p['file'],
        embed='timeF',freq='h',batch_size=int(p['batch_size']),num_workers=0)


def _core(model, variant):
    return model.backbone if variant == 'official' else model.backbone.original


class GateCapture:
    """只统计每个输入窗口的潜在门控，不改变输出。"""
    def __init__(self, core):
        self.data = {}
        def register(name):
            def fn(_module, _inputs, outputs):
                w = outputs.detach().float()
                axes = tuple(range(1, w.ndim))
                self.data[name + '_values'] = w.cpu().numpy()  # 仅保留当前 batch，用于逐元素门控反事实对照
                self.data[name + '_mean'] = w.mean(dim=axes).cpu().numpy()
                self.data[name + '_low'] = (w < 0.1).float().mean(dim=axes).cpu().numpy()
                self.data[name + '_high'] = (w > 0.9).float().mean(dim=axes).cpu().numpy()
                # 不保留完整中间张量，控制内存和显存。
            return fn
        self.handles = [core.en_attention.weight.register_forward_hook(register('tgm')),
                        core.ex_attention.weight.register_forward_hook(register('vgm'))]

    def clear(self):
        self.data = {}

    def close(self):
        for handle in self.handles: handle.remove()


@contextmanager
def disable_token_branch(model):
    """临时关闭已经训练的动态 Token 分支，退出时完整恢复原参数。"""
    gain = model.backbone.gain_logit
    with torch.no_grad():
        saved = gain.detach().clone()
        gain.zero_()
    try:
        yield
    finally:
        with torch.no_grad(): gain.copy_(saved)


def token_relative_norm(model, x):
    """实际注入的 Token 扰动范数 / 原始静态 Token 范数（逐窗口）。"""
    backbone = model.backbone
    with torch.no_grad():
        if model.norm:
            means = x.mean(dim=1, keepdim=True).detach()
            z = x - means
            std = torch.sqrt(torch.var(z, dim=1, keepdim=True, unbiased=False) + 1e-5)
            z = z / std
        else:
            z = x
        z = z.permute(0, 2, 1)
        recent = z[:, :, -backbone.context_length:]
        if backbone.variant == 'dyn_token':
            context = recent
        else:
            context = torch.stack((recent.mean(-1), recent.std(-1, unbiased=False),
                                   recent[:, :, -1] - recent[:, :, 0]), dim=-1)
        delta = torch.tanh(backbone.gain_logit) * backbone.context_projection(context)
        base = backbone.original.glob_token.expand_as(delta)
        return (delta.flatten(1).norm(dim=1) / (base.flatten(1).norm(dim=1) + 1e-12)).cpu().numpy()


def _device(option):
    if option == 'cpu': return torch.device('cpu')
    if option == 'cuda':
        if not torch.cuda.is_available(): raise RuntimeError('明确指定了 CUDA，但当前环境没有可用 GPU')
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_jobs(run: Path):
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    folders = sorted(x for x in (run/'jobs').iterdir()
                     if x.is_dir() and (x/'result.json').exists() and
                     json.loads((x/'result.json').read_text(encoding='utf-8')).get('status') in ('completed','complete'))
    if not folders: raise RuntimeError('没有已完成的模型作业')
    loaded = []
    for folder in folders:
        job = json.loads((folder/'command.json').read_text(encoding='utf-8'))['job']
        loaded.append((folder, job))
    # 仅分析相同 dataset / horizon / seed 下的配对，不混合不同条件。
    matched = {}
    for folder, spec in loaded:
        key = (spec['dataset'],int(spec['horizon']),int(spec['seed']))
        matched.setdefault(key, []).append((folder,spec))
    matched = {key: sorted(values, key=lambda z: (z[1]['variant'] != 'official',z[1]['variant']))
               for key,values in matched.items() if any(z[1]['variant']=='official' for z in values)}
    if not matched: raise RuntimeError('当前 run 中找不到含 Official 的完整对照组')
    return manifest['configuration'], matched


def _mean(vals):
    vals = np.asarray(vals, dtype=float)
    return float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else float('nan')


def _pearson_rank(a, b):
    """只返回描述性 Spearman 相关，不输出误导性的独立样本 p 值。"""
    a,b = np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if np.sum(valid)<6: return float('nan')
    # pandas 是官方数据读取的既有依赖。
    import pandas as pd
    u = pd.Series(a[valid]).rank(method='average').to_numpy(dtype=float)
    v = pd.Series(b[valid]).rank(method='average').to_numpy(dtype=float)
    if u.std()<1e-12 or v.std()<1e-12: return float('nan')
    return float(np.corrcoef(u,v)[0,1])


def csv_write(path, records, columns):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        writer.writeheader();writer.writerows(records)


def _load_model(folder, spec, ratio, device):
    from models import XLinear
    options = _options(spec)
    Model = build_variant_model(XLinear.Model, spec['variant'], ratio)
    model = Model(options)
    checkpoints = list((folder/'checkpoints').glob('**/checkpoint.pth'))
    if len(checkpoints) != 1: raise RuntimeError(f'无法定位唯一 checkpoint: {folder}; found={len(checkpoints)}')
    state = torch.load(checkpoints[0],map_location='cpu',weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, options


def _loader(options, split):
    # data_provider 的 val 采用 shuffle=True；诊断强制 shuffle=False，保证模型间样本严格对齐。
    from data_provider.data_factory import data_provider
    dataset, _ignored = data_provider(options, split)
    loader = DataLoader(dataset, batch_size=options.batch_size,
                        shuffle=False,num_workers=0,drop_last=True)
    return loader


def evaluate(folder, spec, split, ratio, device, max_batches, interval_hours):
    model,options = _load_model(folder,spec,ratio,device)
    loader = _loader(options,split)
    variant=spec['variant']
    capture=GateCapture(_core(model,variant))
    predictions_saved = []
    # 检查测试推理与原作者保存的 pred.npy 是否一致。
    pred_file_list=list((folder/'workspace'/'results').glob('*/pred.npy')) if split=='test' else []
    cache = np.load(pred_file_list[0],mmap_mode='r') if len(pred_file_list)==1 else None
    all_records = []
    checked_total=0
    try:
        with torch.inference_mode():
            for bi,(xb,yb,*_) in enumerate(loader):
                if max_batches and bi >= max_batches: break
                xb = xb.float().to(device)
                truth = yb[:,-options.pred_len:,:].float().numpy()
                capture.clear()
                pred = model(xb).float().cpu().numpy()
                gates_on={k:v.copy() for k,v in capture.data.items()}
                if pred.shape != truth.shape:
                    raise RuntimeError(f'{folder.name}: 预测维度 {pred.shape} != 真值 {truth.shape}')
                n=len(truth)
                if cache is not None:
                    old=np.asarray(cache[checked_total:checked_total+n]); maxdiff=float(np.max(np.abs(old-pred)))
                    if old.shape != pred.shape or maxdiff > 1e-3:
                        raise RuntimeError(f'{folder.name}: checkpoint 重推结果与官方 pred.npy 不一致: maxdiff={maxdiff}; 停止以免样本错位')
                checked_total+=n
                normal_sq=(pred-truth)**2
                normal_mse=normal_sq.mean(axis=(1,2))
                normal_ot=normal_sq[:,:,-1].mean(axis=1)
                altered_mse=np.full(n,np.nan)
                altered_ot=np.full(n,np.nan)
                pred_delta=np.full(n,np.nan)
                tgm_shift=np.full(n,np.nan)
                vgm_shift=np.full(n,np.nan)
                token_ratio=np.full(n,np.nan)
                if variant!='official':
                    token_ratio=token_relative_norm(model,xb)
                    capture.clear()
                    with disable_token_branch(model):
                        disabled_pred=model(xb).float().cpu().numpy()
                    off_gates={k:v.copy() for k,v in capture.data.items()}
                    ablated_sq=(disabled_pred-truth)**2
                    altered_mse=ablated_sq.mean(axis=(1,2))
                    altered_ot=ablated_sq[:,:,-1].mean(axis=1)
                    pred_delta=np.abs(pred-disabled_pred).mean(axis=(1,2))
                    tgm_shift=np.abs(gates_on['tgm_values']-off_gates['tgm_values']).mean(axis=(1,2))
                    vgm_shift=np.abs(gates_on['vgm_values']-off_gates['vgm_values']).mean(axis=(1,2))
                spectral = spectral_features(xb[:,:,-1].cpu().numpy(), interval_hours)
                for i in range(n):
                    record={'variant':variant,'split':split,'sample_index':checked_total-n+i,
                        'mse_all':float(normal_mse[i]),'mse_ot':float(normal_ot[i]),
                        'mse_token_off':float(altered_mse[i]),'mse_ot_token_off':float(altered_ot[i]),
                        'counterfactual_gain':float(altered_mse[i]-normal_mse[i]),
                        'prediction_abs_change':float(pred_delta[i]),
                        'token_relative_norm':float(token_ratio[i]),
                        'tgm_gate_shift':float(tgm_shift[i]),'vgm_gate_shift':float(vgm_shift[i]),
                        'tgm_gate_mean':float(gates_on['tgm_mean'][i]),
                        'tgm_gate_low_frac':float(gates_on['tgm_low'][i]),
                        'tgm_gate_high_frac':float(gates_on['tgm_high'][i]),
                        'vgm_gate_mean':float(gates_on['vgm_mean'][i]),
                        'vgm_gate_low_frac':float(gates_on['vgm_low'][i]),
                        'vgm_gate_high_frac':float(gates_on['vgm_high'][i])}
                    for k,values in spectral.items():
                        val=values[i]
                        record[k]=str(val) if k=='peak_band' else float(val)
                    all_records.append(record)
                if (bi+1)%10==0:
                    print(f'  {variant} {split}: {bi+1}/{len(loader)} batches, {checked_total} samples',flush=True)
    finally:
        capture.close()
        model.cpu()
        if device.type=='cuda': torch.cuda.empty_cache()
    if not all_records: raise RuntimeError(f'{folder}: 当前条件没有完整 Batch 可供诊断')
    if cache is not None and not max_batches and len(cache)!=checked_total:
        raise RuntimeError(f'{folder.name}: 官方 pred.npy 数量 {len(cache)} != 诊断 {checked_total}')
    return all_records


def summarize(records):
    from collections import defaultdict
    groups=defaultdict(list)
    for row in records: groups[(row['split'],row['variant'])].append(row)
    summary=[]
    for (split,variant),rs in sorted(groups.items()):
        summary.append({'split':split,'variant':variant,'samples':len(rs),
            'mse_all':_mean([r['mse_all'] for r in rs]),
            'mse_ot':_mean([r['mse_ot'] for r in rs]),
            'mse_token_off':_mean([r['mse_token_off'] for r in rs]),
            'counterfactual_gain':_mean([r['counterfactual_gain'] for r in rs]),
            'prediction_abs_change':_mean([r['prediction_abs_change'] for r in rs]),
            'token_relative_norm':_mean([r['token_relative_norm'] for r in rs]),
            'tgm_gate_shift':_mean([r['tgm_gate_shift'] for r in rs]),
            'vgm_gate_shift':_mean([r['vgm_gate_shift'] for r in rs])})
    return summary


def frequency_tables(records):
    """按官方模型对应样本频带分组；所有改造版使用同一窗口的类别。"""
    from collections import defaultdict
    baseline={(r['split'],r['sample_index']):r for r in records if r['variant']=='official'}
    groups=defaultdict(list)
    for r in records:
        key=(r['split'],r['sample_index'])
        if key not in baseline:continue
        groups[(r['split'], r['variant'], baseline[key]['peak_band'])].append((r,baseline[key]))
    table=[]
    for (split,variant,band),data in sorted(groups.items()):
        mse=_mean([r['mse_all'] for r,base in data])
        mse_ot=_mean([r['mse_ot'] for r,base in data])
        base_mse=_mean([base['mse_all'] for r,base in data])
        table.append({'split':split,'variant':variant,'peak_band':band,'samples':len(data),
            'mse_all':mse,'mse_ot':mse_ot,'official_mse':base_mse,
            'improvement_pct':100*(base_mse-mse)/base_mse if base_mse else float('nan')})
    return table


def correlations(records):
    from collections import defaultdict
    grouped=defaultdict(list)
    for r in records:grouped[(r['split'],r['variant'])].append(r)
    out=[]
    for (split,variant),rows in sorted(grouped.items()):
        for feat in ('slow_energy_share','middle_energy_share','fast_energy_share','daily_energy_share','normalized_trend_slope'):
            for gate in ('tgm_gate_low_frac','tgm_gate_high_frac','tgm_gate_mean','vgm_gate_mean'):
                out.append({'split':split,'variant':variant,'spectral_feature':feat,
                            'gate_feature':gate,'samples':len(rows),
                            'spearman_rho':_pearson_rank([r[feat] for r in rows],[r[gate] for r in rows])})
    return out


def save_plots(output, summary, groups, corrs, records):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('未安装 matplotlib，仅生成 CSV 和文字报告（无需影响数值诊断）',flush=True)
        return
    out=output/'plots';out.mkdir(exist_ok=True)
    # 每张图单独绘制，满足后续论文插图时的可编辑重绘基础。
    for split in sorted(set(z['split'] for z in summary)):
        data=[z for z in groups if z['split']==split and z['peak_band']!='flat']
        bands=[b for b in ('slow','middle','fast') if any(x['peak_band']==b for x in data)]
        names=sorted(set(x['variant'] for x in data),key=lambda v:(v!='official',v))
        if not bands:continue
        x=np.arange(len(bands));width=0.8/max(1,len(names))
        fig,ax=plt.subplots(figsize=(8,4.5))
        for j,name in enumerate(names):
            by={z['peak_band']:z['mse_all'] for z in data if z['variant']==name}
            ax.bar(x+(j-(len(names)-1)/2)*width,[by.get(b,np.nan) for b in bands],width,label=name)
        ax.set_xticks(x,bands);ax.set_ylabel('MSE (all channels)');ax.set_xlabel('History-window dominant FFT peak band')
        ax.set_title(f'Conditional forecast error | {split}');ax.legend();fig.tight_layout()
        fig.savefig(out/f'{split}_frequency_group_mse.png',dpi=180);plt.close(fig)
    for split in sorted(set(z['split'] for z in summary)):
        data=[z for z in summary if z['split']==split and z['variant']!='official']
        if data:
            fig,ax=plt.subplots(figsize=(7,4.0))
            ax.bar([r['variant'] for r in data],[r['counterfactual_gain'] for r in data])
            ax.axhline(0,color='gray',linewidth=1)
            ax.set_ylabel('MSE(token disabled) - MSE(token active)')
            ax.set_title(f'Trained Token branch intervention | {split}')
            fig.tight_layout();fig.savefig(out/f'{split}_token_intervention.png',dpi=180);plt.close(fig)
        only=[r for r in records if r['split']==split and r['variant']=='official' and
              np.isfinite(r['middle_energy_share'])]
        if len(only)>=5:
            fig,ax=plt.subplots(figsize=(7,4.0))
            stride=max(1,len(only)//1500);ss=only[::stride]
            ax.scatter([r['middle_energy_share'] for r in ss],
                       [r['tgm_gate_low_frac'] for r in ss],s=7,alpha=0.35)
            ax.set_xlabel('Mid-frequency energy share (FFT bins 3-7)')
            ax.set_ylabel('TGM gate fraction < 0.1')
            ax.set_title(f'Frequency / latent gate association | {split}')
            fig.tight_layout();fig.savefig(out/f'{split}_frequency_tgm_scatter.png',dpi=180);plt.close(fig)


def report_text(run, splits, summary, groups, corrs, records, interval_h, max_batches):
    lines=['# XLinear Mechanism Probe — 只读诊断报告','',
           f'- 实验目录：`{run.name}`',f'- 数据拆分：{", ".join(splits)}',
           '- 模型来源：每个 job 对应的官方训练最优 checkpoint；不重新训练。',
           '- 反事实：在同一已训练权重中，临时将 Token 新增分支系数置零，恢复后无权重写入。',
           '- 频率特征：仅基于每个预测窗口的历史 OT；线性去趋势、Hann 窗、rFFT。',
           '- 频带：slow 为 k=1..2；middle 为 k=3..7；fast 为 k>=8。k 是 96 步历史窗口的 FFT 索引。',
           f'- 每条历史采样间隔：{interval_h:g} 小时。24h 频带仅当历史覆盖 >=24 小时才报告。',
           '- 所有指标基于重叠滑窗，不做独立样本显著性声称；频率—门控相关是描述性相关，非因果。',
           '- 默认只分析验证集；如果用 `--splits test`，应视为探索性分析，不可反复根据测试结果选择模型。',
           '- 条件误差包含所有变量 MSE 与 OT 单变量 MSE；OT 只是最后一列，不是训练时唯一目标。',
           '- `counterfactual_gain > 0` 表示此训练权重下开启 Token 分支比关闭时 MSE 更低，**并不等同于训练 Official 的增量**。',
           '- 关闭分支属于分布外局部干预，不等同于移除模块后重训消融。','']
    if max_batches: lines.append(f'**注意：只运行前 {max_batches} 个 batch，结果不能视为完整验证/测试集合。**\n')
    lines+=['## 平均误差与 Token 干预','',
            '| Split | 模型 | 窗口数 | 原MSE | 关闭Token后MSE | 分支增益(MSE差) | 预测平均绝对变化 | Token相对范数 |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    def f(x):return f'{x:.6f}' if np.isfinite(x) else '—'
    for z in summary:
        lines.append(f"| {z['split']} | {z['variant']} | {z['samples']} | {f(z['mse_all'])} | {f(z['mse_token_off'])} | {f(z['counterfactual_gain'])} | {f(z['prediction_abs_change'])} | {f(z['token_relative_norm'])} |")
    lines+=['','## 不同历史频率峰值条件下的预测误差','',
            '| Split | 模型 | 频带 | 窗口数 | 所有变量MSE | OT MSE | 相对Official改善 |',
            '|---|---|---|---:|---:|---:|---:|']
    for z in groups:
        lines.append(f"| {z['split']} | {z['variant']} | {z['peak_band']} | {z['samples']} | {f(z['mse_all'])} | {f(z['mse_ot'])} | {z['improvement_pct']:.3f}% |")
    lines+=['','## 频率与门控相关（示例，完整数据见 CSV）','',
            '| Split | 模型 | 频率指标 | 门控指标 | Spearman ρ |','|---|---|---|---|---:|']
    for c in corrs:
        if c['spectral_feature'] in ('slow_energy_share','middle_energy_share') and c['gate_feature'] in ('tgm_gate_low_frac','tgm_gate_high_frac'):
            lines.append(f"| {c['split']} | {c['variant']} | {c['spectral_feature']} | {c['gate_feature']} | {f(c['spearman_rho'])} |")
    lines+=['','## 如何根据结果决定下一步','',
            '1. 如果关闭 Token 分支后预测几乎不变，说明该训练模型对新增分支的依赖较弱；再核对条件投影及上下游表示。',
            '2. 如果 Official 在某一频率分组上明显误差更大，并且这不是少数重叠窗口驱动，才进入频率条件化 TGM 实验。',
            '3. 如果频率指标和 TGM 选择系数缺乏关联，不代表 TGM 有缺陷；需要考虑趋势、局部变化或信息路径分析。',
            '4. 后续需要按连续时间块或不同随机种子确认稳定性，避免把重叠窗口当作独立证据。','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description='基于现有 checkpoint 的 XLinear Token/频谱/TGM 诊断')
    parser.add_argument('--run',default='latest',help='runs 目录名；默认 latest 自动选择最近一个包含 Official 的已完成实验')
    parser.add_argument('--splits',default='val',help='val / test / val,test，默认 val（用于机制探索）')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--max-batches',type=int,default=0,help='调试限制；0 表示遍历全部完整 batch')
    parser.add_argument('--interval-hours',type=float,default=None,help='单步采样间隔，ETTh1默认1h，Weather默认1/6h')
    args=parser.parse_args()
    splits=[s.strip() for s in args.splits.split(',')]
    if len(splits)!=len(set(splits)) or not set(splits).issubset(SPLIT_LABELS):
        raise ValueError('--splits 仅可为 val、test 或 val,test')
    if args.max_batches<0:raise ValueError('--max-batches 必须 >=0')
    runs=(ROOT/'runs').resolve()
    if args.run=='latest':
        candidates=sorted((d for d in runs.glob('*_full_*') if d.is_dir() and (d/'manifest.json').is_file()),reverse=True)
        matched=[]
        for path in candidates:
            try:
                _, groups=read_jobs(path)
                if any(len(g)>=2 for g in groups.values()):matched.append(path)
            except Exception: continue
        if not matched:raise FileNotFoundError('没有找到带 Official 和改造模型的已完成 full 实验')
        run=matched[0]
    else:
        run=(runs/args.run).resolve()
        if run.parent!=runs or not run.is_dir(): raise ValueError('只能选择 runs 下实际存在的实验目录')
    cfg, matched=read_jobs(run)
    verify_source(cfg);prepare_datasets(cfg)
    sys.path.insert(0,str(UPSTREAM))
    device=_device(args.device)
    print('诊断实验:',run.name,'| 设备:',device,'| 拆分:',','.join(splits),flush=True)
    output=run/'analysis'/'mechanism_probe';output.mkdir(parents=True,exist_ok=True)
    # 每个 dataset/horizon/seed 单独输出，严禁混合不同设置。
    for (dataset,horizon,seed),jobs in matched.items():
        current=output/f'{dataset}_H{horizon}_s{seed}';current.mkdir(exist_ok=True)
        interval=args.interval_hours if args.interval_hours is not None else {'ETTh1':1.,'ETTh2':1.,'ETTm1':.25,'ETTm2':.25,'Weather':1/6,'Electricity':1.}.get(dataset,1.)
        records=[]
        for split in splits:
            for folder,spec in jobs:
                print(f'  开始 {folder.name} split={split}',flush=True)
                records.extend(evaluate(folder,spec,split,float(cfg['runtime']['context_ratio']),device,args.max_batches,interval))
        summary=summarize(records)
        groups=frequency_tables(records)
        corrs=correlations(records)
        csv_write(current/'sample_level.csv',records,list(records[0]))
        csv_write(current/'counterfactual_summary.csv',summary,list(summary[0]))
        csv_write(current/'frequency_groups.csv',groups,list(groups[0]))
        csv_write(current/'gate_frequency_correlations.csv',corrs,list(corrs[0]))
        save_plots(current,summary,groups,corrs,records)
        (current/'REPORT.md').write_text(report_text(run,splits,summary,groups,corrs,records,interval,args.max_batches),encoding='utf-8')
        print('完成:',current/'REPORT.md',flush=True)
    print('全部诊断完成；原始 checkpoint 与训练结果未被修改。',flush=True)

if __name__=='__main__':main()
