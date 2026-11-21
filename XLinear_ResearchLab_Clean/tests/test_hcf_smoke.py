"""CPU structural test; no GPU and no benchmark claims."""
import torch
from torch import nn
from types import SimpleNamespace
from src.models.extensions.horizon_fusion import HorizonFusionHead
from src.models.registry import build_variant_model, available_variants


class SyntheticOfficial(nn.Module):
    """Shape-compatible test double, not the official XLinear architecture."""
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.encoder = nn.Linear(cfg.seq_len, 2*cfg.d_model)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(2*cfg.d_model, cfg.pred_len))
    def forward(self, x):
        return self.head(self.encoder(x.transpose(1, 2))).transpose(1, 2)


def run():
    torch.set_num_threads(2)
    for H in (96, 192, 336, 720):
        cfg = SimpleNamespace(seq_len=96, pred_len=H, d_model=32, features='M')
        torch.manual_seed(101)
        baseline = SyntheticOfficial(cfg).eval()
        x = torch.randn(3, 96, 7)
        base_out = baseline(x)
        for mode in ('hcf', 'hcf_static', 'hcf_shared'):
            torch.manual_seed(101)
            revised = build_variant_model(SyntheticOfficial, mode, 0.25)(cfg).eval()
            assert revised.head.added_parameter_count() > 0
            assert revised(x).shape == (3, H, 7)
            assert torch.allclose(base_out, revised(x), rtol=0, atol=0), (mode, H)
            revised.train()
            revised.zero_grad(set_to_none=True)
            target = torch.randn(3, H, 7)
            loss = (revised(x) - target).square().mean()
            loss.backward()
            train_param = revised.head.logit if mode == 'hcf_static' else revised.head.gate[-1].weight
            assert train_param.grad is not None and train_param.grad.abs().sum().item() > 0, mode
        print('PASS: H=',H,'shape / baseline parity / gradient')
    print('REGISTERED:', available_variants())
    print('CPU structural smoke test passed; no accuracy conclusion')

if __name__=='__main__':
    run()
