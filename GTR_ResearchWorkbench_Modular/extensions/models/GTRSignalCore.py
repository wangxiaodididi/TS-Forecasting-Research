"""Additive signal-correction wrapper for the official ICLR 2026 GTR.

Works alongside, never changes, the official models/GTR.py.
The core GTR retrieval and backbone are reused literally by importing its Model.

Signal branch: orthonormal one-level Haar decomposition of the *past-only*
normalized input. Low/high corrections are additive and inspectable.
"""
import math
import torch
import torch.nn as nn
from models import GTR


class SignalAugmentedModel(nn.Module):
    def __init__(self, configs, mode):
        super().__init__()
        if mode not in ("high", "low", "dual", "linear"):
            raise ValueError(f"Unknown branch mode: {mode}")
        self.base = GTR.Model(configs)  # Official GTR is unmodified.
        self.mode = mode
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.use_revin = bool(configs.use_revin)
        n = (self.seq_len + 1) // 2
        if mode in ("low", "dual"):
            self.low_head = nn.Linear(n, self.pred_len, bias=False)
            nn.init.zeros_(self.low_head.weight)
        if mode in ("high", "dual"):
            self.high_head = nn.Linear(n, self.pred_len, bias=False)
            nn.init.zeros_(self.high_head.weight)
        if mode == "linear":
            self.linear_head = nn.Linear(self.seq_len, self.pred_len, bias=False)
            nn.init.zeros_(self.linear_head.weight)

    def forward(self, x, cycle_index, return_components=False):
        # Official GTR handles Q, retrieval, 2D fusion, original MLP, RevIN.
        base_prediction = self.base(x, cycle_index)
        if self.use_revin:
            mu = x.mean(dim=1, keepdim=True)
            # Exact convention of upstream GTR.py (torch.var default correction=1)
            std = (x.var(dim=1, keepdim=True) + 1e-5).sqrt()
            xn = (x - mu) / std
        else:
            std = torch.ones_like(x[:, :1, :])
            xn = x

        low_contribution = torch.zeros_like(base_prediction)
        high_contribution = torch.zeros_like(base_prediction)
        linear_contribution = torch.zeros_like(base_prediction)
        energy_ratio = None

        if self.mode == "linear":
            linear_contribution = self.linear_head(xn.transpose(1, 2)).transpose(1, 2) * std
        else:
            # For odd input lengths, repeat the last observed value (not future).
            if xn.size(1) % 2:
                xn_pair = torch.cat([xn, xn[:, -1:, :]], dim=1)
            else:
                xn_pair = xn
            even, odd = xn_pair[:, 0::2, :], xn_pair[:, 1::2, :]
            low = (even + odd) / math.sqrt(2.0)
            high = (even - odd) / math.sqrt(2.0)
            if hasattr(self, 'low_head'):
                low_contribution = self.low_head(low.transpose(1, 2)).transpose(1, 2) * std
            if hasattr(self, 'high_head'):
                high_contribution = self.high_head(high.transpose(1, 2)).transpose(1, 2) * std
            # Descriptive local-detail energy share; not a learned causal attribution.
            energy_ratio = high.square().sum(dim=1) / (
                high.square().sum(dim=1) + low.square().sum(dim=1) + 1e-8
            )

        prediction = base_prediction + low_contribution + high_contribution + linear_contribution
        if return_components:
            return {
                'prediction': prediction,
                'gtr_prediction': base_prediction,
                'low_correction': low_contribution,
                'high_correction': high_contribution,
                'linear_correction': linear_contribution,
                'high_energy_ratio': energy_ratio,
            }
        return prediction

    def explain(self, x, cycle_index):
        """For diagnostics with gradients retained; call under no_grad to save memory."""
        return self.forward(x, cycle_index, return_components=True)
