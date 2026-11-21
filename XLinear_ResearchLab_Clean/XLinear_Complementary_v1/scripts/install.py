#!/usr/bin/env python3
"""Non-destructive installation in an existing LG-Short-enabled XLinear lab.

Never touches vendor/XLinear. Only copies one new module and appends a marked
registry block. The original registry is backed up atomically before changes.
"""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import tempfile

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parent
REG=ROOT/'src/models/registry.py'
DST=ROOT/'src/models/extensions/complementary_blocks_v1.py'
SRC=HERE/'module/complementary_blocks.py'
BEGIN='# >>> XLINEAR-COMPLEMENTARY-V1 BEGIN'
END='# <<< XLINEAR-COMPLEMENTARY-V1 END'

REGISTRY='''# >>> XLINEAR-COMPLEMENTARY-V1 BEGIN
from src.models.extensions.complementary_blocks_v1 import TemporalFeatureAdapter, RecentTrendResidual
_comp_previous_builder = build_variant_model
_COMP_CONFIGS = {
    'lg_tmix_r16': (16, None),
    'lg_tmix_r8': (8, None),
    'lg_trend_w24': (None, 24),
    'lg_trend_w12': (None, 12),
    'lg_combo_r16_w24': (16, 24),
}
for _comp_name in _COMP_CONFIGS:
    if _comp_name in VARIANTS:
        raise RuntimeError('Model variant conflict: ' + _comp_name)
    VARIANTS[_comp_name] = None

def build_variant_model(original_model_class, variant: str, ratio: float):
    if variant not in _COMP_CONFIGS:
        return _comp_previous_builder(original_model_class, variant, ratio)
    # Retain the original proven LG-Short module exactly; do not reimplement it.
    base_lg = _comp_previous_builder(original_model_class, 'lg_short', ratio)
    rank, recent_window = _COMP_CONFIGS[variant]

    class XLinearComplementary(base_lg):
        def __init__(self, configs):
            super().__init__(configs)
            if configs.features != 'M':
                raise ValueError('Only XLinear multivariate features=M is supported')
            _state = torch.random.get_rng_state()
            try:
                if rank is not None:
                    self.backbone.en_attention = TemporalFeatureAdapter(
                        self.backbone.en_attention, int(configs.d_model), rank=rank)
                if recent_window is not None:
                    if int(configs.seq_len) < 2 * recent_window:
                        raise ValueError('Not enough history for trend skip')
                    self.recent_trend_residual = RecentTrendResidual(
                        int(configs.pred_len), recent_window=recent_window)
            finally:
                torch.random.set_rng_state(_state)

        def forward(self, x_enc):
            y = super().forward(x_enc)
            if recent_window is not None:
                y = y + self.recent_trend_residual(x_enc, normalized=bool(self.norm))
            return y

    XLinearComplementary.__name__ = 'XLinear_' + variant
    return XLinearComplementary
# <<< XLINEAR-COMPLEMENTARY-V1 END
'''


def install(root=ROOT):
    reg=root/'src/models/registry.py'
    ext=root/'src/models/extensions/complementary_blocks_v1.py'
    if not reg.exists() or not (root/'run.sh').exists():
        raise RuntimeError('Run this from XLinear_ResearchLab_Clean project root')
    content=reg.read_text(encoding='utf-8')
    if 'lg_short' not in content or 'build_variant_model' not in content:
        raise RuntimeError('LG-Short registry is missing. Install LG-Embedding v1 first')
    for name in ('lg_tmix_r16','lg_trend_w24','lg_combo_r16_w24'):
        if name in content and BEGIN not in content:
            raise RuntimeError('Variant already exists outside this package: '+name)
    if BEGIN in content:
        if content.count(BEGIN)!=1 or END not in content:
            raise RuntimeError('Partial/corrupt marker in registry; refusing modification')
        if not ext.exists() or ext.read_bytes()!=SRC.read_bytes():
            raise RuntimeError('Installed module differs from this package; leave existing code unchanged')
        print('Already installed; nothing modified')
        return
    if ext.exists() and ext.read_bytes()!=SRC.read_bytes():
        raise RuntimeError('Another module with same filename exists; refusing overwrite')
    backup=reg.with_name('registry_before_complementary_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.bak')
    shutil.copy2(reg,backup)
    # atomic write with tempfile in same directory
    new=content.rstrip()+'\n\n'+REGISTRY
    temp=reg.with_name(reg.name+'.complementary.tmp')
    try:
        if not ext.exists():
            shutil.copy2(SRC,ext)
        temp.write_text(new,encoding='utf-8')
        compile(new,str(reg),'exec')
        temp.replace(reg)
    except BaseException:
        if temp.exists():temp.unlink()
        shutil.copy2(backup,reg)
        raise
    print('Installed:',reg)
    print('Backup:',backup)
    print('New module:',ext)

if __name__=='__main__':
    install()
