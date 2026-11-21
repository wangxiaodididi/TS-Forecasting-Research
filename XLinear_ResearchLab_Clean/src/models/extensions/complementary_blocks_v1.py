"""Complementary feature transformation + trend residual for XLinear.

These blocks are intentionally independent of LG-Short and the original model.
Only install.py composes them. Both branches start at exact identity.
"""
import torch
from torch import nn
from torch.nn import functional as F


class TemporalFeatureAdapter(nn.Module):
    """Retain original TGM; add a lightweight cross-feature low-rank update.

    Original TGM only applies elementwise gates to an existing feature vector.
    This addition can rotate/mix feature coordinates, without replacing the gate.
    """
    def __init__(self, original_tgm: nn.Module, d_model: int, rank: int = 16,
                 residual_scale: float = 0.25):
        super().__init__()
        if rank < 1 or residual_scale <= 0:
            raise ValueError('rank must be >=1 and residual_scale positive')
        self.original_tgm = original_tgm
        width = 2 * int(d_model)
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, rank)
        self.up = nn.Linear(rank, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.residual_scale = float(residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.original_tgm(x)
        update = self.up(F.gelu(self.down(self.norm(z))))
        return z + self.residual_scale * update


class RecentTrendResidual(nn.Module):
    """Use observed history to provide a recent-level and recent-drift bypass.

    It never observes future labels. Output is on original scale, exactly
    matching the XLinear denormalization convention when use_norm=True.
    """
    def __init__(self, pred_len: int, recent_window: int = 24):
        super().__init__()
        if pred_len < 1 or recent_window < 2:
            raise ValueError('pred_len>0 and recent_window>=2 required')
        self.pred_len = int(pred_len)
        self.recent_window = int(recent_window)
        self.horizon_weight = nn.Parameter(torch.zeros(2, pred_len))

    def forward(self, x_enc: torch.Tensor, normalized: bool = True) -> torch.Tensor:
        if x_enc.ndim != 3:
            raise ValueError('x_enc must have shape [B,L,C]')
        b, l, c = x_enc.shape
        if l < 2 * self.recent_window:
            raise ValueError(f'lookback {l} smaller than two recent windows {self.recent_window}')
        if normalized:
            mu = x_enc.mean(dim=1, keepdim=True).detach()
            sd = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            zn = (x_enc - mu) / sd
        else:
            zn = x_enc
            sd = None
        w = self.recent_window
        recent = zn[:, -w:, :].mean(dim=1)
        previous = zn[:, -2*w:-w, :].mean(dim=1)
        overall = zn.mean(dim=1)
        # B x C x 2: contrast with whole-window location and recent drift
        z = torch.stack([recent - overall, recent - previous], dim=-1)
        z = torch.tanh(z)  # constrain contribution under extreme local shifts
        residual = torch.einsum('bck,kh->bhc', z, self.horizon_weight)
        if normalized:
            residual = residual * sd
        return residual
