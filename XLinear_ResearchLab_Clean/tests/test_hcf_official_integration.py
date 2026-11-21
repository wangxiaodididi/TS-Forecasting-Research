"""Run against *actual* pinned upstream XLinear on the AutoDL project.

Skip explicitly when upstream is not present (never pretend to have tested it).
"""
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from src.models.registry import build_variant_model


def run():
    root = Path(__file__).resolve().parents[1]
    upstream = root / 'vendor' / 'XLinear'
    file = upstream / 'models' / 'XLinear.py'
    if not file.is_file():
        print('SKIP actual-upstream integration: missing', file)
        return
    from src.core.config import read_config
    from src.runtime.bootstrap import verify_source
    verify_source(read_config(), upstream)
    # isolate official module import to guarantee correct source resolution
    sys.path.insert(0, str(upstream))
    from models import XLinear
    torch.set_num_threads(2)
    for H in (96, 192, 336, 720):
        cfg = SimpleNamespace(
            seq_len=96, pred_len=H, d_model=128, enc_in=7,
            t_ff=512, c_ff=7, usenorm=1, embed_dropout=0.0,
            head_dropout=0.2, t_dropout=0.0, c_dropout=0.0,
            features='M')
        torch.manual_seed(27)
        baseline = XLinear.Model(cfg).eval()
        torch.manual_seed(27)
        revised = build_variant_model(XLinear.Model, 'hcf', 0.25)(cfg).eval()
        x = torch.randn(3, cfg.seq_len, cfg.enc_in)
        with torch.no_grad():
            ya, yb = baseline(x), revised(x)
        assert ya.shape == yb.shape == (3, H, 7)
        assert torch.allclose(ya, yb, atol=0, rtol=0), (H, float((ya-yb).abs().max()))
        print('PASS official source:', H, 'exact zero-gate parity')
        revised.train()
        revised.zero_grad(set_to_none=True)
        loss = revised(x).square().mean()
        loss.backward()
        p = revised.head.gate[-1].weight
        assert p.grad is not None and p.grad.abs().sum().item() > 0
        print('PASS official source:', H, 'new gate gradient')
    print('PASS pinned official XLinear integration')

if __name__ == '__main__':
    run()
