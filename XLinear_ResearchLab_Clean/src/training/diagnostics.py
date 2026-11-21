"""训练完成后独立分析已保存的 checkpoint、预测序列、TGM/VGM 门控与梯度。

纯离线只读分析：不修改训练参数，不更新 checkpoint，不参与模型选择。
门控和梯度快照仅采用一个验证 Batch，属于局部诊断，不能代表全数据重要性或因果解释。
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from src.core.config import ROOT,read_config
from src.runtime.bootstrap import UPSTREAM,verify_source,prepare_datasets
from src.training.results import atomic_json,csv_write


def analyze_one(folder:Path,cfg:dict):
    verify_source(cfg);prepare_datasets(cfg)
    spec=json.loads((folder/'command.json').read_text(encoding='utf-8'))['job']
    p=spec['profile']; h=int(spec['horizon']);v=spec['variant']
    if not (folder/'result.json').is_file():
        print('跳过尚未完成的作业:',folder.name);return
    target=folder/'diagnostics';target.mkdir(exist_ok=True)
    sys.path.insert(0,str(UPSTREAM))
    from data_provider.data_factory import data_provider
    from models import XLinear
    from src.models.registry import build_variant_model
    options=SimpleNamespace(
        seq_len=96,pred_len=h,label_len=48,features='M',target='OT',
        d_model=int(p['d_model']),enc_in=int(p['channels']),t_ff=int(p['t_ff']),c_ff=int(p['c_ff']),
        usenorm=1,embed_dropout=float(p['embed_dropout']),head_dropout=float(p['head_dropout']),
        t_dropout=float(p['t_dropout']),c_dropout=float(p['c_dropout']),
        data=p['data'],root_path=str(UPSTREAM/'dataset'),data_path=p['file'],
        embed='timeF',freq='h',batch_size=int(p['batch_size']),num_workers=0)
    Model=build_variant_model(XLinear.Model,v,float(cfg['runtime']['context_ratio']))
    model=Model(options)
    paths=list((folder/'checkpoints').glob('**/checkpoint.pth'))
    if len(paths)!=1:raise RuntimeError(f'需要恰好一个训练权重: {folder}; found={len(paths)}')
    model.load_state_dict(torch.load(paths[0],map_location='cpu',weights_only=True),strict=True)
    model.eval()
    # 作者的 test() 会将完整预测数组保存在 workspace/results/.../pred.npy。
    pred_files=list((folder/'workspace'/'results').glob('*/pred.npy'))
    if len(pred_files)!=1:raise FileNotFoundError(f'找不到官方测试预测 pred.npy: {folder}')
    predictions=np.load(pred_files[0],mmap_mode='r')
    _,test_loader=data_provider(options,'test')
    n=0
    horizon_sq=np.zeros(h);horizon_abs=np.zeros(h)
    channel_sq=np.zeros(int(p['channels']));channel_abs=np.zeros(int(p['channels']))
    n_h=n_c=0
    first_rows=[]
    for _,batch_y,_,_ in test_loader:
        truth=batch_y[:,-h:,:].numpy()
        forecast=np.asarray(predictions[n:n+len(truth)])
        if len(forecast)!=len(truth):raise RuntimeError('pred.npy 与数据迭代数量不一致')
        delta=forecast-truth
        horizon_sq+=(delta**2).sum(axis=(0,2));horizon_abs+=np.abs(delta).sum(axis=(0,2))
        channel_sq+=(delta**2).sum(axis=(0,1));channel_abs+=np.abs(delta).sum(axis=(0,1))
        n_h+=delta.shape[0]*delta.shape[2];n_c+=delta.shape[0]*delta.shape[1]
        if not first_rows:
            channel=delta.shape[2]-1  # 论文数据 OT 目标为最后一列
            first_rows=[{'horizon_step':int(k+1),'actual':float(truth[0,k,channel]),
                         'prediction':float(forecast[0,k,channel])} for k in range(h)]
        n+=len(truth)
    if n!=len(predictions):raise RuntimeError(f'预测条数 {len(predictions)} 与真实标签条数 {n} 不同')
    calculated_mse=float(horizon_sq.sum()/(n_h*h))
    recorded=float(json.loads((folder/'result.json').read_text(encoding='utf-8'))['mse'])
    if abs(calculated_mse-recorded)>1e-4:
        raise RuntimeError(f'重算测试 MSE={calculated_mse} 和官方 MSE={recorded} 不一致；停止生成误导性诊断')
    csv_write(target/'horizon_errors.csv',['horizon_step','mse','mae'],[
        {'horizon_step':i+1,'mse':float(horizon_sq[i]/n_h),'mae':float(horizon_abs[i]/n_h)} for i in range(h)])
    csv_write(target/'channel_errors.csv',['channel_index','mse','mae'],[
        {'channel_index':i,'mse':float(channel_sq[i]/n_c),'mae':float(channel_abs[i]/n_c)}
        for i in range(int(p['channels']))])
    csv_write(target/'prediction_sample.csv',['horizon_step','actual','prediction'],first_rows)
    # Gate 诊断：权重模块的 forward 输出是 Sigmoid 门控值，范围0~1。
    gates={}
    core=model.backbone.original if v!='official' else model.backbone
    handles=[]
    def capture(name):
        def hook(_, __, value):
            weights=value.detach().float().flatten()
            if weights.numel()==0:return
            hist=torch.histc(weights,bins=10,min=0,max=1).int().tolist()
            gates[name]={'mean':float(weights.mean()),'std':float(weights.std(unbiased=False)),
               'fraction_below_0_1':float((weights<0.1).float().mean()),
               'fraction_above_0_9':float((weights>0.9).float().mean()),
               'histogram_counts':hist,'num_values':int(weights.numel())}
        return hook
    handles.append(core.en_attention.weight.register_forward_hook(capture('TGM_time_gate')))
    handles.append(core.ex_attention.weight.register_forward_hook(capture('VGM_variable_gate')))
    _,val_loader=data_provider(options,'val')
    batch_x,batch_y,_,_=next(iter(val_loader))
    batch_x=batch_x[:min(len(batch_x),4)].float()
    batch_y=batch_y[:len(batch_x),-h:,:].float()
    try:
        output=model(batch_x)
        # 单个验证 Batch 的梯度敏感性（仅诊断，不更新权重）。
        loss=torch.mean((output-batch_y)**2)
        model.zero_grad(set_to_none=True)
        loss.backward()
    finally:
        for handle in handles:handle.remove()
    groups=defaultdict(list)
    for name,param in model.named_parameters():
        if param.grad is None:continue
        if 'backbone.original.projection' in name or 'backbone.projection' in name:group='Embedding'
        elif '.en_attention.' in name:group='TGM'
        elif '.ex_attention.' in name:group='VGM'
        elif 'context_projection' in name or 'gain_logit' in name:group='Token_extension'
        elif name.startswith('head.'):group='Prediction_head'
        else:group='Other'
        groups[group].append(param.grad.detach().float().norm().item()**2)
    gradients={k:{'grad_l2':float(sum(v)**0.5),'parameter_tensors':len(v)} for k,v in groups.items()}
    token={}
    if v!='official':
        token={'gain_tanh':float(torch.tanh(model.backbone.gain_logit).detach()),
            'context_length':int(model.backbone.context_length)}
    atomic_json(target/'module_diagnostics.json',dict(
        variant=v,probe='first validation batch (at most 4 samples)',
        gate_stats=gates,gradient_norms=gradients,token=token,
        validation_probe_mse=float(loss.detach()),
        test_recomputed_mse=calculated_mse,official_test_mse=recorded,
        notes=['门控值仅说明选择系数，不等于变量因果重要性',
               '梯度为单批验证快照，不是训练全过程梯度',
               '诊断不修改模型权重，不参与超参数选择']))
    print('诊断完成:',target,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True,help='runs 下实验目录名，如 20261115_...')
    parser.add_argument('--job',help='只诊断指定作业；省略则处理该实验所有完成的作业')
    args=parser.parse_args()
    run=(ROOT/'runs'/args.run).resolve()
    if run.parent!=(ROOT/'runs').resolve() or not run.is_dir():raise ValueError('run 必须是 runs 下真实的一个实验文件夹')
    cfg=json.loads((run/'manifest.json').read_text(encoding='utf-8'))['configuration']
    jobs=[run/'jobs'/args.job] if args.job else sorted((run/'jobs').iterdir())
    for folder in jobs:
        if not folder.is_dir():raise FileNotFoundError(folder)
        analyze_one(folder,cfg)

if __name__=='__main__':main()
