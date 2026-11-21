"""加载唯一的 YAML 配置、生成实验矩阵，并在训练前做严格校验。"""
from __future__ import annotations
import copy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT / 'configs' / 'experiment.yaml'


def read_config(path: Path = CONFIG_FILE) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    for key in ('source', 'runtime', 'experiment', 'pilot', 'profiles'):
        if key not in cfg: raise ValueError(f'配置缺失 {key}: {path}')
    return cfg


def profile(cfg: dict, dataset: str, horizon: int) -> dict:
    try:
        return copy.deepcopy(cfg['profiles'][dataset][str(int(horizon))])
    except KeyError as exc:
        raise ValueError(f'未找到官方实验参数: {dataset}, H={horizon}。先核对作者脚本再添加。') from exc


def make_jobs(cfg: dict, mode: str = 'full', *, datasets=None, horizons=None, variants=None, seeds=None):
    if mode not in ('pilot', 'full'):
        raise ValueError('mode 只接受 pilot/full')
    base = cfg['pilot'] if mode == 'pilot' else cfg['experiment']
    ds = datasets if datasets is not None else base['datasets']
    hs = horizons if horizons is not None else base['horizons']
    vs = variants if variants is not None else base['variants']
    ss = seeds if seeds is not None else base['seeds']
    if not (0 < float(cfg['runtime']['context_ratio']) <= 1):
        raise ValueError('context_ratio 应在 (0,1]')
    # 将已有模型注册表与配置校验绑定；不能 silently fallback 为 official。
    from src.models.registry import available_variants
    for v in vs:
        if v not in available_variants(): raise ValueError(f'未知模型变体: {v}; 可用: {available_variants()}')
    if 'official' not in vs: raise ValueError('配对实验必须包含 official 基线')
    if len(vs) != len(set(vs)) or len(ss) != len(set(ss)):
        raise ValueError('variants/seeds 不得出现重复项')
    jobs=[]
    for d in ds:
        for h in hs:
            p=profile(cfg,d,int(h))
            if mode=='pilot':p['epochs']=int(cfg['pilot']['epochs'])
            for s in ss:
                for v in vs:
                    jobs.append({'dataset':d,'horizon':int(h),'seed':int(s),'variant':v,'profile':p.copy()})
    return jobs
