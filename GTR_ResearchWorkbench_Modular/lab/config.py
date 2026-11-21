"""Author-script-matched, immutable baseline settings separated from research overrides."""
import yaml
from .paths import ROOT

MODEL_FILES = {
    'GTR': None,
    'GTRLinearResidual': 'linear_residual.yaml',
    'GTRHaarHigh': 'haar_high.yaml',
    'GTRHaarLow': 'haar_low.yaml',
    'GTRHaarDual': 'haar_dual.yaml',
    'GTRPhaseInnovation': 'phase_innovation.yaml',
    'GTRPhaseAdaptive': 'phase_adaptive.yaml',
}
DATASETS = {'ETTh1': 'etth1.yaml', 'ETTm1': 'ettm1.yaml', 'Weather': 'weather.yaml', 'Electricity': 'electricity.yaml'}

def config_for(dataset, model, horizon, seed, mode, overrides=None):
    if dataset not in DATASETS or model not in MODEL_FILES:
        raise ValueError(f"Unknown dataset/model: {dataset}/{model}")
    with (ROOT / 'configs' / 'official' / DATASETS[dataset]).open(encoding='utf-8') as f:
        original = yaml.safe_load(f)
    if int(horizon) not in original['pred_lens']:
        raise ValueError('Prediction horizon is not in the paper script (96,192,336,720)')
    if model == 'GTR' and mode == 'paper' and int(seed) not in original['official_seeds']:
        raise ValueError('For strict baseline, use the official script seeds')
    if mode not in ('paper', 'smoke'):
        raise ValueError('mode must be paper or smoke')
    selected = original.copy()
    selected['model'] = model
    selected['pred_len'] = int(horizon)
    selected['random_seed'] = int(seed)
    selected['mode'] = mode
    chosen_overrides = dict(overrides or {})
    if model == 'GTR' and mode == 'paper' and chosen_overrides:
        raise ValueError('Paper baseline parameters are locked; use research mode with an extension')
    if model != 'GTR':
        with (ROOT / 'configs' / 'experiments' / MODEL_FILES[model]).open(encoding='utf-8') as f:
            research = yaml.safe_load(f)
        chosen_overrides = {**(research.get('overrides') or {}), **chosen_overrides}
    if mode == 'smoke':
        selected.update(train_epochs=1, patience=1, batch_size=8, num_workers=0)
    if chosen_overrides:
        allowed = {'train_epochs','patience','batch_size','learning_rate','dropout','d_model','num_workers','lradj','use_revin'}
        unknown = set(chosen_overrides) - allowed
        if unknown:
            raise ValueError(f'Unapproved override(s): {sorted(unknown)}')
        selected.update(chosen_overrides)
    selected['overrides'] = chosen_overrides
    selected['model_id_base'] = {'Weather':'weather'}.get(dataset,dataset)
    selected['author_training_hparams'] = mode == 'paper' and not chosen_overrides
    selected['paper_exact'] = mode == 'paper' and model == 'GTR' and not chosen_overrides
    return selected

def command_args(cfg):
    """Pass ONLY flags explicitly present in official scripts; upstream defaults remain upstream."""
    flags = ['is_training','root_path','data_path','model_id','model','data',
             'features','seq_len','pred_len','enc_in','cycle','train_epochs','patience',
             'dropout','itr','batch_size','learning_rate','random_seed',
             'num_workers','d_model','lradj','use_revin']
    args = {'model_id': f"{cfg['model_id_base']}_{cfg['seq_len']}_{cfg['pred_len']}", 'is_training': 1, **cfg}
    result=[]
    for name in flags:
        if name in args and args[name] is not None:
            result.extend([f'--{name}',str(args[name])])
    return result
