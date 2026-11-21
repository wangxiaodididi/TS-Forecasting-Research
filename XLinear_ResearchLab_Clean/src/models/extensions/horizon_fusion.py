"""Horizon-conditioned fusion for the official XLinear M-branch.

Does not edit TGM, VGM, embedding, or the original head's trainable weights.
The original head dropout is called once, and its exact linear prediction is
recovered when the new gate equals zero (the default initialization).
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class HorizonFusionHead(nn.Module):
    """Replace only the *wrapper* around original Dropout+Linear forecasting head.

    x: [batch, channels, 2*d_model] -> [batch, channels, pred_len].
    gamma_{b,c,g} is input-conditioned, shared by a consecutive horizon group.
    Delta_y = gamma * (linear_temporal - linear_cross).
    """

    def __init__(self, original_head: nn.Module, d_model: int,
                 pred_len: int, mode: str = 'hcf', groups: int = 4,
                 hidden: int = 16, max_gain: float = 0.5):
        super().__init__()
        if mode not in ('hcf', 'hcf_static', 'hcf_shared'):
            raise ValueError('mode must be hcf, hcf_static, or hcf_shared')
        if (not isinstance(original_head, nn.Sequential) or len(original_head) != 2
                or not isinstance(original_head[0], nn.Dropout)
                or not isinstance(original_head[1], nn.Linear)):
            raise TypeError('Expected official head: nn.Sequential(Dropout, Linear)')
        out_layer = original_head[1]
        if out_layer.in_features != 2 * d_model or out_layer.out_features != pred_len:
            raise ValueError('XLinear head size does not match 2*d_model -> pred_len')
        if not (0.0 < max_gain <= 1.0):
            raise ValueError('max_gain must be in (0,1]')
        if min(d_model, pred_len, groups, hidden) < 1:
            raise ValueError('dimension sizes must be positive')
        self.original_head = original_head
        self.d_model = int(d_model)
        self.pred_len = int(pred_len)
        self.mode = mode
        self.groups = min(int(groups), int(pred_len))
        self.max_gain = float(max_gain)
        ids = torch.div(torch.arange(pred_len) * self.groups, pred_len,
                        rounding_mode='floor').long()
        self.register_buffer('group_ids', ids, persistent=False)
        if mode in ('hcf', 'hcf_shared'):
            outputs = self.groups if mode == 'hcf' else 1
            self.gate = nn.Sequential(nn.Linear(2*d_model, int(hidden)),
                                      nn.GELU(), nn.Linear(int(hidden), outputs))
            # Identity starting point: the exact original head before learning.
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
        else:
            self.logit = nn.Parameter(torch.zeros(self.groups))
        self.capture_gate = False
        self.last_gate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 2*self.d_model:
            raise ValueError(f'expected [B,C,{2*self.d_model}], got {tuple(x.shape)}')
        dropped = self.original_head[0](x)  # dropout exactly once
        head = self.original_head[1]
        baseline = F.linear(dropped, head.weight, head.bias)
        if self.mode == 'hcf_static':
            gamma = self.max_gain * torch.tanh(self.logit).view(1, 1, self.groups)
        else:
            gamma = self.max_gain * torch.tanh(self.gate(dropped))
        if self.mode == 'hcf_shared':
            per_horizon_gamma = gamma.expand(dropped.shape[0], dropped.shape[1], self.pred_len)
        else:
            per_horizon_gamma = gamma.index_select(-1, self.group_ids)
        local = F.linear(dropped[..., :self.d_model], head.weight[:, :self.d_model], None)
        cross = F.linear(dropped[..., self.d_model:], head.weight[:, self.d_model:], None)
        if self.capture_gate:
            self.last_gate = per_horizon_gamma.detach()
        else:
            self.last_gate = None
        return baseline + per_horizon_gamma * (local - cross)

    def added_parameter_count(self) -> int:
        return sum(p.numel() for p in self.gate.parameters()) if hasattr(self, 'gate') else self.logit.numel()
