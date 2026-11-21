#!/usr/bin/env python3
"""Install additive LG-Position registry entry, without replacing LG v1 code."""
from pathlib import Path
from datetime import datetime
import argparse
import shutil

ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / 'src/models/registry.py'
EXT = ROOT / 'src/models/extensions/position_local_embedding.py'
LG = ROOT / 'src/models/extensions/local_global_embedding.py'
BEGIN = '# >>> LG-POSITION V1 BEGIN'
END = '# <<< LG-POSITION V1 END'
APPEND = '''# >>> LG-POSITION V1 BEGIN
from src.models.extensions.position_local_embedding import PositionLocalGlobalEmbedding
_LGP_previous_build_variant_model = build_variant_model
if 'lg_position' in VARIANTS:
    raise RuntimeError('Variant lg_position already registered before LG-Position v1 block')
VARIANTS['lg_position'] = None

def build_variant_model(original_model_class, variant: str, ratio: float):
    if variant != 'lg_position':
        return _LGP_previous_build_variant_model(original_model_class, variant, ratio)
    class XLinearLGPosition(original_model_class):
        def __init__(self, configs):
            super().__init__(configs)
            if configs.features != 'M':
                raise ValueError('lg_position supports only features=M')
            _lgp_rng_state = torch.random.get_rng_state()
            try:
                self.backbone.projection = PositionLocalGlobalEmbedding(
                    self.backbone.projection, int(configs.seq_len), int(configs.d_model))
            finally:
                torch.random.set_rng_state(_lgp_rng_state)
    XLinearLGPosition.__name__ = 'XLinear_lg_position'
    return XLinearLGPosition
# <<< LG-POSITION V1 END
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--remove', action='store_true')
    args = p.parse_args()
    if not REG.is_file():
        raise SystemExit('Please unzip into XLinear_ResearchLab_Clean root; registry absent')
    if not LG.is_file():
        raise SystemExit('LG v1 is required; local_global_embedding.py absent')
    if not EXT.is_file() and not args.remove:
        raise SystemExit('LG-Position extension source file is absent')
    old = REG.read_text(encoding='utf-8')
    if args.remove:
        if BEGIN not in old and END not in old:
            print('LG-Position is not installed')
            return
        if old.count(BEGIN) != 1 or old.count(END) != 1:
            raise SystemExit('Registry marker corruption; refuse removal')
        new = old[:old.index(BEGIN)] + old[old.index(END) + len(END):]
    else:
        if BEGIN in old and END in old and old.count(BEGIN) == old.count(END) == 1:
            print('LG-Position is already registered; no change')
            return
        if BEGIN in old or END in old:
            raise SystemExit('Partial registry marker; refuse installation')
        if '# >>> LG-EMBED V1 BEGIN' not in old:
            raise SystemExit('LG v1 registry injection not found; install LG v1 first')
        if 'import torch' not in old or 'def build_variant_model' not in old:
            raise SystemExit('Incompatible registry')
        new = old.rstrip() + '\n\n' + APPEND
    backup = REG.with_name('registry_before_LGPosition_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.bak')
    shutil.copy2(REG, backup)
    REG.write_text(new, encoding='utf-8')
    print('Updated registry:', REG)
    print('Backup:', backup)


if __name__ == '__main__':
    main()
