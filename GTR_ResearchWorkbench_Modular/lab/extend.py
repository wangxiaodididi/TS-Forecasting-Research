"""Only run snapshots are patched. Never edit vendor/GTR itself."""
from pathlib import Path
import shutil
from .paths import ROOT
MODELS=['GTRLinearResidual','GTRHaarLow','GTRHaarHigh','GTRHaarDual','GTRPhaseInnovation','GTRPhaseAdaptive']

def install_snapshot_extension(snapshot):
    snapshot=Path(snapshot)
    for name in ['GTRSignalCore','GTRPhaseCore', *MODELS]:
        source=ROOT/'extensions'/'models'/f'{name}.py'
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source,snapshot/'models'/f'{name}.py')
    p=snapshot/'exp'/'exp_main.py'
    code=p.read_text(encoding='utf-8')
    anchor='from utils.tools import'
    if code.count(anchor)!=1: raise RuntimeError('Unexpected upstream import layout; do not patch')
    code=code.replace(anchor,'from models import '+', '.join(MODELS)+'\n'+anchor,1)
    anchor="'GTR': GTR,"
    if code.count(anchor)!=1: raise RuntimeError('Unexpected upstream registry layout; do not patch')
    code=code.replace(anchor,anchor+'\n'+''.join(f"            '{n}': {n},\n" for n in MODELS),1)
    p.write_text(code,encoding='utf-8')

def enable_smoke_subset(snapshot,max_samples=64):
    """Independent smoke-only data truncation, never used for paper mode."""
    p=Path(snapshot)/'data_provider'/'data_factory.py'
    code=p.read_text(encoding='utf-8')
    marker='    print(flag, len(data_set))'
    if code.count(marker)!=1: raise RuntimeError('Unexpected official data provider: smoke patch aborted')
    replacement="""    # SMOKE-ONLY: limit each split, preserving official dataset preprocessing.
    from torch.utils.data import Subset
    if flag in ('train','val','test'):
        limit = min(len(data_set), 64)
        data_set = Subset(data_set, list(range(limit)))
"""+marker
    p.write_text(code.replace(marker,replacement,1),encoding='utf-8')
