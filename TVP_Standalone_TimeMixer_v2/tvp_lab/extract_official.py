"""Extract mathematically additive forecast contributions from an EXACT trained official backbone.

Does NOT modify official source, checkpoint or training pipeline.
For TimeMixer we hook future_multi_mixing and inverse-normalize differences.
For XLinear we hook backbone and split the original linear prediction head.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader


def timemixer_parts(model,x,x_mark,y_mark):
    captured=[]
    original=model.future_multi_mixing
    def capture(*args, **kwargs):
        output=original(*args, **kwargs)
        captured.extend(output)
        return output
    model.future_multi_mixing=capture
    try:
        # The original forecast signature is forward(x_enc, x_mark_enc, x_dec, x_mark_dec).
        pred=model(x,x_mark,None,y_mark)
        if isinstance(pred,tuple):pred=pred[0]
        if len(captured)<2:raise RuntimeError('No multiple TimeMixer FMM component predictions captured.')
        # Denorm is affine in each component at the SAME forecast origin.
        denorm=model.normalize_layers[0]
        zeros=denorm(torch.zeros_like(captured[0]),'denorm')
        parts=[denorm(v,'denorm')-zeros for v in captured]
        components=torch.stack(parts,dim=-1)
        reconstructed=zeros+components.sum(-1)
        error=float((reconstructed-pred).abs().max())
        if error>1.e-3:raise RuntimeError(f'FMM decomposition failed identity check: {error:.6g}')
        return pred,components
    finally:
        model.future_multi_mixing=original


def xlinear_parts(model,x):
    if getattr(model,'feature',None)!='M':raise ValueError('XLinear TVP adapter supports multivariate M only.')
    hidden=[]
    def hook(_module,_args,result):hidden.append(result)
    h=model.backbone.register_forward_hook(hook)
    try:
        pred=model(x)
        if len(hidden)!=1:raise RuntimeError('Expected exactly one XLinear backbone call')
        z=hidden[0]
        linear=model.head[-1]
        if not isinstance(linear,torch.nn.Linear) or linear.weight.shape[1]!=z.shape[-1]:
            raise RuntimeError('Unexpected XLinear prediction head: needs final Linear with 2*d_model inputs')
        d=z.shape[-1]//2
        if d*2!=z.shape[-1]:raise ValueError('Expected 2*d_model hidden features')
        p_t=torch.nn.functional.linear(z[:,:,:d],linear.weight[:,:d],bias=None).permute(0,2,1)
        p_v=torch.nn.functional.linear(z[:,:,d:],linear.weight[:,d:],bias=None).permute(0,2,1)
        if model.norm:
            stdev=torch.sqrt(x.var(dim=1,keepdim=True,unbiased=False)+1e-5)
            p_t=p_t*stdev; p_v=p_v*stdev
        parts=torch.stack([p_t,p_v],dim=-1)
        return pred,parts
    finally:
        h.remove()


def make_config(path):
    raw=json.loads(Path(path).read_text(encoding='utf8'))
    raw=raw.get('args',raw)
    if 'cmd' in raw:
        cmd=raw['cmd']
        result={};k=None
        for val in cmd:
            if str(val).startswith('--'):
                k=str(val)[2:].replace('-','_');result[k]=True
            elif k is not None:
                if val in ['True','False']:val=val=='True'
                else:
                    try:val=int(val)
                    except (ValueError,TypeError):
                        try:val=float(val)
                        except (ValueError,TypeError):pass
                result[k]=val;k=None
        raw=result
    if not isinstance(raw,dict): raise ValueError('Config must be JSON object or {"args":{...}}')
    return raw


def load_model(family,repo_root,config,checkpoint,device):
    sys.path.insert(0,str(Path(repo_root).resolve()))
    if family=='timemixer':
        from models import TimeMixer
        m=TimeMixer.Model(SimpleNamespace(**config))
        if not hasattr(m,'future_multi_mixing'):raise ValueError('TimeMixer official FMM is missing')
    elif family=='xlinear':
        from models import XLinear
        m=XLinear.Model(SimpleNamespace(**config))
        if not hasattr(m,'backbone') or not hasattr(m,'head'):raise ValueError('XLinear official head not found')
    else:raise ValueError(family)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if isinstance(state,dict) and 'state_dict' in state:state=state['state_dict']
    m.load_state_dict(state,strict=True)
    return m.to(device).eval()


def export_one(model,family,repo_root,config,flag,out,max_windows=0,batch_size=32,device='cpu'):
    sys.path.insert(0,str(Path(repo_root).resolve()))
    from data_provider.data_factory import data_provider
    args=SimpleNamespace(**{**{'batch_size':batch_size,'num_workers':0,'embed':'timeF',
                                  'freq':'h','features':'M','target':'OT','seasonal_patterns':'Monthly','drop_last':True},**config})
    dataset,_=data_provider(args,flag)
    # Never reuse provider's potentially shuffled train loader for time-ordered online evaluation.
    # Match the authors' evaluation window count (TimeMixer defaults drop_last=True;
    # XLinear hardcodes drop_last=True for both val/test). Preserve chronological order.
    dl=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0,drop_last=True)
    bases=[];parts=[];ys=[];total=0
    with torch.inference_mode():
        for x,y,x_mark,y_mark in dl:
            x=x.float().to(device); x_mark=x_mark.float().to(device); y_mark=y_mark.float().to(device)
            if family=='timemixer':pred,c=timemixer_parts(model,x,x_mark,y_mark)
            else:pred,c=xlinear_parts(model,x)
            y_true=y[:,-int(args.pred_len):,:].float().to(device)
            if pred.shape!=y_true.shape:raise RuntimeError(f'Output {pred.shape} vs true {y_true.shape}. Use features=M.')
            take=len(x) if not max_windows else min(len(x),max_windows-total)
            if take<=0:break
            bases.append(pred[:take].cpu().numpy());parts.append(c[:take].cpu().numpy());ys.append(y_true[:take].cpu().numpy())
            total+=take
            if max_windows and total>=max_windows:break
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,base=np.concatenate(bases).astype('float32'),
                        parts=np.concatenate(parts).astype('float32'),y=np.concatenate(ys).astype('float32'))
    print(f'[{flag}] exported {total} chronological forecast origins to {out}')


def main():
    p=argparse.ArgumentParser(description='Export frozen ORIGINAL backbone val/test FMM/head contributions.')
    p.add_argument('--family',choices=['timemixer','xlinear'],required=True)
    p.add_argument('--official-root',required=True)
    p.add_argument('--config-json',required=True,help='Full official training args JSON, or wrapper train_config.json containing cmd')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--data-file',default='ETTh1.csv')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--max-windows',type=int,default=0,help='0=all; smoke only if >0')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--output-dir',default='results/extracted')
    a=p.parse_args()
    config=make_config(a.config_json)
    config.update(root_path=str(Path(a.data_root).resolve())+'/',data_path=a.data_file,batch_size=a.batch_size,
                  num_workers=0,features='M')
    required=['task_name','seq_len','pred_len','enc_in','c_out']
    if any(k not in config for k in required):raise ValueError(f'Missing required model config keys: {[k for k in required if k not in config]}')
    if a.family=='timemixer':
        # Defaults used only when missing from official training args; user should verify training config.
        defaults={'label_len':0,'e_layers':2,'d_model':16,'d_ff':32,'moving_avg':25,'down_sampling_layers':3,
                  'down_sampling_window':2,'down_sampling_method':'avg','decomp_method':'moving_avg','top_k':5,
                  'dropout':.1,'embed':'timeF','freq':'h','channel_independence':1,'use_norm':1,
                  'use_future_temporal_feature':0,'factor':1,'data':'ETTh1'}
    else:
        defaults={'usenorm':1,'d_model':128,'t_ff':512,'c_ff':7,'embed_dropout':0.,'head_dropout':.2,
                  't_dropout':0.,'c_dropout':0.,'data':'ETTh1'}
    config={**defaults,**config}
    model=load_model(a.family,a.official_root,config,a.checkpoint,a.device)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'extraction_settings.json').write_text(json.dumps({'family':a.family,'repo_root':a.official_root,
        'checkpoint':a.checkpoint,'config':config,'max_windows':a.max_windows},indent=2,ensure_ascii=False))
    for flag in ['val','test']:
        export_one(model,a.family,a.official_root,config,flag,str(out/f'{flag}.npz'),a.max_windows,a.batch_size,a.device)
if __name__=='__main__':main()
