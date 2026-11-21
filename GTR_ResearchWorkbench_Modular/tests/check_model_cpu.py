"""After official install: test *real upstream GTR* and each extension on CPU.

Usage: python -m tests.check_model_cpu
No dataset, GPU, or official training required. Does not claim paper reproduction.
"""
import sys
from types import SimpleNamespace
from pathlib import Path
import torch
from lab.source import verify
from lab.paths import VENDOR

def main():
    verify(VENDOR)
    sys.path.insert(0,str(VENDOR))
    from models import GTR
    from extensions.models.GTRSignalCore import SignalAugmentedModel
    cfg=SimpleNamespace(seq_len=96,pred_len=96,enc_in=7,cycle=96,d_model=32,
                        dropout=.5,use_revin=1,individual=0)
    x=torch.randn(3,96,7)
    cycle=torch.tensor([0,20,60])
    for mode in ('linear','high','low','dual'):
        torch.manual_seed(2024)
        variant=SignalAugmentedModel(cfg,mode)
        reference=GTR.Model(cfg)
        reference.load_state_dict(variant.base.state_dict())
        variant.eval();reference.eval()
        with torch.no_grad():
            a=reference(x,cycle);b=variant(x,cycle)
        assert torch.allclose(a,b,rtol=0,atol=1e-6),(mode,(a-b).abs().max().item())
        variant.train();target=torch.randn(3,96,7)
        loss=(variant(x,cycle)-target).square().mean();loss.backward()
        active={name:float(p.grad.norm()) for name,p in variant.named_parameters() if p.grad is not None}
        assert any(name.startswith('base.GTR') and norm>0 for name,norm in active.items()), 'No GTR gradient'
        assert any((name.endswith('_head.weight')) and norm>0 for name,norm in active.items()), 'No correction gradient'
        with torch.no_grad():
            comp=variant.explain(x,cycle)
            residual=(comp['prediction']-(comp['gtr_prediction']+comp['high_correction']+comp['low_correction']+comp['linear_correction'])).abs().max()
            assert residual.item()<1e-6
        print(f'{mode:6s}: shape={tuple(b.shape)} loss={loss.item():.5f}, GTR_grad={active.get("base.GTR.conv2d.weight",float("nan")):.5f}; OK')
    print('Official core preserved; all 4 extension modes passed forward / backward / initial parity / additive outputs.')
if __name__=='__main__':main()
