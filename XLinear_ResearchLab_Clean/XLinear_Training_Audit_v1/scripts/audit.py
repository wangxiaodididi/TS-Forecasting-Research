#!/usr/bin/env python3
"""Read-only XLinear baseline/CC audit; no training, no checkpoint edits.
Run from the installed audit directory inside existing XLinear_ResearchLab_Clean.
"""
from __future__ import annotations
import argparse, collections, csv, datetime, hashlib, json, os, platform, random, sys, traceback
from pathlib import Path
from types import SimpleNamespace

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parent
sys.path.insert(0, str(ROOT))


def output_path(name: str) -> Path:
    d = ROOT / 'audit_outputs'
    d.mkdir(exist_ok=True)
    return d / name


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def timestamp():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def audit_environment():
    import torch
    record = dict(root=str(ROOT), python_executable=sys.executable,
                  python_version=sys.version.split()[0], torch=torch.__version__,
                  cuda_available=torch.cuda.is_available(),
                  cuda_devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                  platform=platform.platform(),
                  registry=str(ROOT/'src/models/registry.py'),
                  source_exists=(ROOT/'vendor/XLinear/models/XLinear.py').is_file(),
                  dataset_exists=(ROOT/'vendor/XLinear/dataset/ETTh1.csv').is_file(),
                  torch_threads=torch.get_num_threads())
    from src.core.config import read_config
    from src.runtime.bootstrap import verify_source
    cfg=read_config()
    try:
        verify_source(cfg)
        record['source_integrity']='PASS'
    except Exception as e:
        record['source_integrity']='FAIL: '+str(e)
    from src.models.registry import available_variants
    record['variants']=list(available_variants())
    record['notes']=['工具不会训练模型，且不会修改 checkpoint',
                     'manifest记录的环境比当前环境更能代表历史训练条件',
                     '旧版CC安装器仅支持ETTh1历史96，Weather适配需核实服务器当前版本']
    f=output_path('environment_'+timestamp()+'.json')
    save_json(f,record)
    print(json.dumps(record,ensure_ascii=False,indent=2));print('REPORT:',f)
    return record


def settings_from_profile(dataset, horizon, p):
    return SimpleNamespace(
        seq_len=96, pred_len=int(horizon), label_len=48,
        enc_in=int(p['channels']), d_model=int(p['d_model']),
        t_ff=int(p['t_ff']), c_ff=int(p['c_ff']),
        t_dropout=float(p['t_dropout']), c_dropout=float(p['c_dropout']),
        head_dropout=float(p['head_dropout']), embed_dropout=float(p['embed_dropout']),
        usenorm=1, features='M',
        data=p['data'], data_path=p['file'],
        root_path=str(ROOT/'vendor'/'XLinear'/'dataset'), target='OT',
        embed='timeF', freq='h', batch_size=int(p['batch_size']), num_workers=0,
    )


def get_model(config,variant,ratio):
    from src.models.registry import build_variant_model
    from models import XLinear
    Model=build_variant_model(XLinear.Model,variant,ratio)
    return Model(config)


def check_structural():
    import torch
    from src.core.config import read_config
    cfg=read_config();ratio=float(cfg['runtime'].get('context_ratio',0.25))
    from src.runtime.bootstrap import verify_source
    verify_source(cfg)
    sys.path.insert(0,str(ROOT/'vendor/XLinear'))
    from src.models.registry import available_variants
    profiles=[('ETTh1',96)]
    result=[]
    for ds,h in profiles:
        c=settings_from_profile(ds,h,cfg['profiles'][ds][str(h)])
        torch.manual_seed(1723);state=torch.random.get_rng_state()
        official=get_model(c,'official',ratio).cpu().eval()
        x=torch.randn(3,c.seq_len,c.enc_in)
        y=torch.randn(3,c.pred_len,c.enc_in)
        with torch.no_grad():ref=official(x)
        count0=sum(p.numel() for p in official.parameters())
        for variant in ('cc_static','cc_raw','cc_components'):
            record=dict(dataset=ds,horizon=h,variant=variant)
            if variant not in available_variants():
                record.update(status='NOT_REGISTERED'); result.append(record); continue
            torch.random.set_rng_state(state)
            model=get_model(c,variant,ratio).cpu().eval()
            with torch.no_grad():pred=model(x)
            record['max_initial_output_absdiff']=float((pred-ref).abs().max())
            record['initial_equivalent']=bool(torch.allclose(pred,ref,atol=1e-6,rtol=1e-6))
            record['extra_param_count']=sum(p.numel() for p in model.parameters())-count0
            model.zero_grad(set_to_none=True)
            loss=(model(x)-y).square().mean();loss.backward()
            core=model.backbone
            record['peer_grad_l2']=None if core.peer_logits.grad is None else float(core.peer_logits.grad.norm())
            record['coeff_grad_l2']=None if core.component_weights.grad is None else float(core.component_weights.grad.norm())
            record['loss']=float(loss)
            with torch.no_grad():
                route,uniform=core._routing(x.transpose(1,2))
                record['initial_route_deviation']=float((route-uniform).abs().mean())
                record['route_row_sum_error']=float((route.sum(dim=-1)-1).abs().max())
            record['status']='PASS' if record['initial_equivalent'] and (record['peer_grad_l2'] or 0)>0 else 'CHECK'
            result.append(record)
    p=output_path('structural_'+timestamp()+'.json')
    save_json(p,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print('REPORT:',p)
    return result


def check_manifest(run):
    cfgfile=run/'manifest.json'
    if not cfgfile.exists(): raise FileNotFoundError(cfgfile)
    manifest=json.loads(cfgfile.read_text(encoding='utf-8'))
    jobs=[]
    for folder in sorted((run/'jobs').glob('*')):
        if not folder.is_dir(): continue
        command_file=folder/'command.json'; result_file=folder/'result.json'
        if not command_file.exists():continue
        command=json.loads(command_file.read_text(encoding='utf-8'))
        spec=command['job']; cmd=command.get('command',[])
        checkpoint=list((folder/'checkpoints').glob('**/checkpoint.pth'))
        args=cmd if isinstance(cmd,list) else str(cmd).split()
        kv={}
        for i,s in enumerate(args[:-1]):
            if str(s).startswith('--'): kv[str(s)]=str(args[i+1])
        result=json.loads(result_file.read_text(encoding='utf-8')) if result_file.exists() else {}
        jobs.append(dict(job=folder.name,variant=spec.get('variant'),horizon=spec.get('horizon'),seed=spec.get('seed'),
                         dataset=spec.get('dataset'), python_command=args[0] if args else None,
                         command_flag_values={k:kv.get(k) for k in ('--learning_rate','--batch_size','--train_epochs','--random_seed','--seq_len','--pred_len')},
                         checkpoints=len(checkpoint),status=result.get('status','unknown'),mse=result.get('mse'),
                         result_file=bool(result_file.exists())))
    record={'run':str(run),'original_python':manifest.get('python'),
            'original_torch':manifest.get('torch'), 'original_device':manifest.get('device'),
            'source_commit':manifest.get('source_commit'), 'jobs':jobs}
    out=output_path('manifest_'+run.name+'_'+timestamp()+'.json')
    save_json(out,record)
    print(json.dumps(record,ensure_ascii=False,indent=2));print('REPORT:',out)
    return manifest,jobs


def safe_load(path):
    import torch
    try: return torch.load(path,map_location='cpu',weights_only=True)
    except TypeError: return torch.load(path,map_location='cpu')


def check_checkpoints(run, max_batches=8, variant_filter=None):
    import torch
    from src.runtime.bootstrap import verify_source
    manifest, summaries=check_manifest(run)
    cfg=manifest['configuration']
    verify_source(cfg)
    sys.path.insert(0,str(ROOT/'vendor/XLinear'))
    from data_provider.data_factory import data_provider
    from src.models.registry import available_variants
    records=[];ratio=float(cfg['runtime'].get('context_ratio',0.25))
    for sj in summaries:
        v=sj['variant']
        if v not in ('cc_static','cc_raw','cc_components'):continue
        if variant_filter and v not in variant_filter:continue
        if v not in available_variants():
            records.append(dict(job=sj['job'],status='NOT_REGISTERED'));continue
        folder=run/'jobs'/sj['job']
        checkpoints=list((folder/'checkpoints').glob('**/checkpoint.pth'))
        if len(checkpoints)!=1:
            records.append(dict(job=sj['job'],status=f'CHECKPOINT_COUNT_{len(checkpoints)}'));continue
        job=json.loads((folder/'command.json').read_text(encoding='utf-8'))['job']
        c=settings_from_profile(job['dataset'],job['horizon'],job['profile'])
        try:
            model=get_model(c,v,ratio)
            state=safe_load(checkpoints[0]);model.load_state_dict(state,strict=True)
            model.cpu().eval();core=model.backbone
            _,loader=data_provider(c,'val')
            n=0; batches=0; sse=collections.defaultdict(float)
            route_dev_sum=0.;route_std_sum=0.;norm_ratio_sum=0.;gate_samples=0
            first_grad={}
            for x,truth,_,_ in loader:
                x=x.float().cpu();truth=truth[:,-c.pred_len:,:].float().cpu()
                # Capture *actual* normalized backbone input (RevIN may be applied by Model).
                captured={}
                def capture_backbone_input(module, inputs):
                    captured['Xroute']=inputs[0].detach()
                def capture_temporal_output(module, inputs, output):
                    captured['H']=output.detach()[:,:,core.original.d_model:]
                h1=core.register_forward_pre_hook(capture_backbone_input)
                h2=core.original.en_attention.register_forward_hook(capture_temporal_output)
                try:
                    with torch.no_grad():pred=model(x)
                finally:
                    h1.remove();h2.remove()
                with torch.no_grad():
                    P,U=core._routing(captured['Xroute'])
                    H=captured['H']
                    actual_scale=core.mix_scale
                    try:
                        core.mix_scale=0.
                        no_route=model(x)
                    finally:core.mix_scale=actual_scale
                    original_method=core._routing
                    meanP=P.mean(dim=0,keepdim=True).expand(P.shape[0],-1,-1).contiguous()
                    try:
                        core._routing=lambda __, pp=meanP, uu=U: (pp,uu)
                        mean_route_pred=model(x)
                    finally:core._routing=original_method
                    e=pred-truth
                    sse['normal']+=float(e.square().sum())
                    sse['neutral']+=float((no_route-truth).square().sum())
                    sse['batch_average_route']+=float((mean_route_pred-truth).square().sum())
                    sse['change_vs_neutral']+=float((pred-no_route).square().sum())
                    route_dev_sum+=float((P-U).abs().mean())*len(x)
                    route_std_sum+=float(P.std(dim=0,unbiased=False).mean())*len(x)
                    delta=actual_scale*torch.bmm(P-U,H)
                    norm_ratio_sum+=float(delta.norm()/H.norm().clamp_min(1e-12))*len(x)
                    gate_samples+=len(x)
                    n+=e.numel();batches+=1
                if batches==1:
                    model.zero_grad(set_to_none=True)
                    val_loss=(model(x)-truth).square().mean();val_loss.backward()
                    first_grad={'peer_grad_l2':float(core.peer_logits.grad.norm()) if core.peer_logits.grad is not None else None,
                                'components_grad_l2':float(core.component_weights.grad.norm()) if core.component_weights.grad is not None else None,
                                'peer_parameter_l2':float(core.peer_logits.detach().norm()),
                                'components_coeff':core.component_weights.detach().tolist()}
                    model.zero_grad(set_to_none=True)
                if max_batches>0 and batches>=max_batches:break
            if not n:raise ValueError('No validation samples')
            rec=dict(job=sj['job'],status='PASS',checkpoint=str(checkpoints[0]),
                     split='val',diagnostic_only=True,batches=batches,samples=gate_samples,
                     mse_normal=sse['normal']/n,mse_without_new_route=sse['neutral']/n,
                     mse_with_batch_average_route=sse['batch_average_route']/n,
                     prediction_rms_change_from_neutral=(sse['change_vs_neutral']/n)**0.5,
                     mean_abs_routing_deviation=route_dev_sum/gate_samples,
                     mean_route_variability=route_std_sum/gate_samples,
                     mean_relative_hidden_update=norm_ratio_sum/gate_samples,
                     grad=first_grad)
            records.append(rec)
            print('PASS',rec['job'],'mse_val:',round(rec['mse_normal'],7),'neutral:',round(rec['mse_without_new_route'],7),'relative_update:',round(rec['mean_relative_hidden_update'],6),flush=True)
        except Exception as exc:
            records.append(dict(job=sj['job'],status='ERROR',error=str(exc),traceback=traceback.format_exc()))
            print('ERROR',sj['job'],str(exc),flush=True)
    f=output_path('checkpoint_'+run.name+'_'+timestamp()+'.json')
    save_json(f,records)
    print('REPORT:',f)
    return records


def historical():
    f=AUDIT/'reference/historical_results.csv'
    with f.open(encoding='utf-8-sig') as handle:rows=list(csv.DictReader(handle))
    group=collections.defaultdict(dict)
    for r in rows:group[(r['scenario'],r['seed'])][r['variant']]=float(r['mse'])
    scenarios=sorted({a for a,b in group})
    lines=[]
    for s in scenarios:
        g=[x for (sc,seed),x in group.items() if sc==s]
        for variant in ('cc_static','cc_raw','cc_components'):
            gains=[100*(v['official']-v[variant])/v['official'] for v in g if variant in v and 'official' in v]
            if not gains:continue
            lines.append(dict(scenario=s,variant=variant,n_seeds=len(gains),mean_improvement_pct=sum(gains)/len(gains),positive_seeds=sum(x>0 for x in gains)))
        gains=[100*(v['cc_raw']-v['cc_components'])/v['cc_raw'] for v in g if 'cc_raw' in v and 'cc_components' in v]
        if gains:lines.append(dict(scenario=s,variant='cc_components_vs_cc_raw',n_seeds=len(gains),mean_improvement_pct=sum(gains)/len(gains),positive_seeds=sum(x>0 for x in gains)))
    f=output_path('historical_'+timestamp()+'.csv')
    with f.open('w',encoding='utf-8-sig',newline='') as out:
        w=csv.DictWriter(out,fieldnames=list(lines[0]));w.writeheader();w.writerows(lines)
    for l in lines:print(l)
    print('REPORT:',f)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['environment','structure','history','manifest','checkpoint'])
    ap.add_argument('--run',help='runs directory name (not a path outside the project)')
    ap.add_argument('--max-batches',type=int,default=8,help='0=complete validation; default 8 for quick check')
    ap.add_argument('--variants',default='cc_static,cc_raw,cc_components')
    args=ap.parse_args()
    if args.mode=='environment': audit_environment()
    elif args.mode=='structure': check_structural()
    elif args.mode=='history':historical()
    else:
        if not args.run:ap.error('manifest/checkpoint requires --run')
        run=(ROOT/'runs'/args.run).resolve()
        if run.parent!=(ROOT/'runs').resolve() or not run.is_dir():ap.error('Run must be a real direct child of runs/')
        if args.mode=='manifest': check_manifest(run)
        else: check_checkpoints(run,args.max_batches,set(args.variants.split(',')))

if __name__=='__main__': main()
