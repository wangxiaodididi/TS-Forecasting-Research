"""Idempotent, surgical installation into an EXISTING GTR Workbench.

Copies only *new* files, then inserts a few registry lines after exact anchors.
Does not touch original GTR, Phase1, training optimizer, dataset or runs.
"""
from __future__ import annotations
import argparse
import shutil
from datetime import datetime
from uuid import uuid4
from pathlib import Path

MODELS = ('GTRPhaseInnovation', 'GTRPhaseAdaptive')
NEWFILES = [
    'extensions/models/GTRPhaseCore.py',
    'extensions/models/GTRPhaseInnovation.py',
    'extensions/models/GTRPhaseAdaptive.py',
    'configs/experiments/phase_innovation.yaml',
    'configs/experiments/phase_adaptive.yaml',
    'phase2/__init__.py',
    'phase2/install.py',
    'phase2/batch.py',
    'phase2/cli.py',
    'phase2/explain.py',
    'lab/pages/02_Phase2_Innovation.py',
    'tests/test_phase2_model.py',
    'tests/test_phase2_install.py',
    'docs/PHASE2_RESEARCH_DESIGN.md',
    '11_Phase2_CPU.sh',
    '12_Phase2_GPU.sh',
    '13_Phase2_Status.sh',
    '14_Phase2_Validation.sh',
]

def install(root: Path, package: Path, dry_run=False):
    root,package=root.resolve(),package.resolve()
    fconfig=root/'lab/config.py'
    fextend=root/'lab/extend.py'
    if not fconfig.exists() or not fextend.exists():
        raise RuntimeError(f'Not the existing workbench root: missing {fconfig} or {fextend}')
    for rel in NEWFILES:
        if not (package/rel).is_file():
            raise FileNotFoundError(package/rel)
        dst=root/rel
        if dst.exists():
            if dst.read_bytes()!=(package/rel).read_bytes():
                raise FileExistsError(f'Installation would overwrite an edited file: {dst}')
    cfg=fconfig.read_text(encoding='utf-8')
    ext=fextend.read_text(encoding='utf-8')
    if "'GTRPhaseInnovation': 'phase_innovation.yaml'" not in cfg:
        anchor="    'GTRHaarDual': 'haar_dual.yaml',"
        if cfg.count(anchor)!=1:raise RuntimeError('Unexpected lab/config.py layout; aborting')
        cfg=cfg.replace(anchor,anchor+"\n    'GTRPhaseInnovation': 'phase_innovation.yaml',\n    'GTRPhaseAdaptive': 'phase_adaptive.yaml',",1)
    if "'GTRPhaseInnovation'" not in ext:
        anchor="MODELS=['GTRLinearResidual','GTRHaarLow','GTRHaarHigh','GTRHaarDual']"
        if ext.count(anchor)!=1:raise RuntimeError('Unexpected lab/extend.py model list; aborting')
        ext=ext.replace(anchor,"MODELS=['GTRLinearResidual','GTRHaarLow','GTRHaarHigh','GTRHaarDual','GTRPhaseInnovation','GTRPhaseAdaptive']",1)
        anchor="['GTRSignalCore', *MODELS]"
        if ext.count(anchor)!=1:raise RuntimeError('Unexpected lab/extend.py core list; aborting')
        ext=ext.replace(anchor,"['GTRSignalCore','GTRPhaseCore', *MODELS]",1)
    for name,text in [('lab/config.py',cfg),('lab/extend.py',ext)]:
        compile(text,str(root/name),'exec')
    if dry_run:
        print('DRY RUN OK: no existing modules or experiments will be overwritten')
        return
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:6]
    backup=root/'phase2_backups'/stamp
    backup.mkdir(parents=True,exist_ok=False)
    for name in ['lab/config.py','lab/extend.py']:
        shutil.copy2(root/name,backup/Path(name).name)
    for rel in NEWFILES:
        dst=root/rel
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copy2(package/rel,dst)
    for name,text in [('lab/config.py',cfg),('lab/extend.py',ext)]:
        (root/name).write_text(text,encoding='utf-8')
    print('Phase2 installed. Backup:',backup)
    print('Official source, previous model files, datasets and runs were not touched.')

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='.')
    ap.add_argument('--package',default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args()
    install(Path(a.root),Path(a.package),a.dry_run)
