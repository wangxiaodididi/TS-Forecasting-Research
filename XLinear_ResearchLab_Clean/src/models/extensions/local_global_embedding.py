"""Minimal local + global embedding extension for XLinear's M branch.

The original projection remains the global path. A channel-independent
local branch extracts nonlinear short motifs and pooled position summaries.
Zero-init of the fusion projection ensures exact equality at initialization.
This is a hypothesis, not an established performance improvement.
"""
import torch
from torch import nn
from torch.nn import functional as F


class LocalGlobalEmbedding(nn.Module):
    def __init__(self, original_projection: nn.Module, seq_len: int, d_model: int,
                 mode: str = 'lg_dual', channels: int = 2, bins: int = 8):
        super().__init__()
        if not (isinstance(original_projection, nn.Sequential)
                and len(original_projection) == 2
                and isinstance(original_projection[0], nn.Linear)
                and isinstance(original_projection[1], nn.Dropout)):
            raise TypeError('Expected original XLinear projection: Sequential(Linear, Dropout)')
        if original_projection[0].in_features != seq_len or original_projection[0].out_features != d_model:
            raise ValueError('XLinear original projection does not match dimensions')
        if mode not in ('lg_dual', 'lg_short', 'lg_linear'):
            raise ValueError('mode must be lg_dual, lg_short, lg_linear')
        self.global_projection = original_projection  # same pretrained/original params
        self.mode = mode
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        if mode == 'lg_linear':
            self.residual = nn.Linear(seq_len, d_model)
            nn.init.zeros_(self.residual.weight)
            nn.init.zeros_(self.residual.bias)
        else:
            self.short = nn.Conv1d(1, channels, 3, padding=1)
            if mode == 'lg_dual':
                self.long = nn.Conv1d(1, channels, 9, padding=4)
            n_scale = 2 if mode == 'lg_dual' else 1
            self.residual = nn.Linear(n_scale * channels * bins * 2, d_model)
            nn.init.zeros_(self.residual.weight)
            nn.init.zeros_(self.residual.bias)
            self.bins = int(bins)
        self.capture = False
        self.local_norm_cache = None

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.seq_len:
            raise ValueError(f'expected [B,C,{self.seq_len}], got {tuple(x.shape)}')
        global_emb = self.global_projection(x)
        if self.mode == 'lg_linear':
            delta = self.residual(x)
        else:
            b, c, l = x.shape
            t = x.reshape(b*c, 1, l)
            # Input is already normalized by the pinned upstream XLinear forward.
            # Shared local convs preserve ordering within each receptive field.
            parts = [F.gelu(self.short(t))]
            if self.mode == 'lg_dual':
                parts.append(F.gelu(self.long(t)))
            stats = []
            for p in parts:
                stats.extend([F.adaptive_avg_pool1d(p, self.bins),
                              F.adaptive_max_pool1d(p, self.bins)])
            pooled = torch.cat(stats, dim=1).flatten(1)
            delta = self.residual(pooled).reshape(b, c, self.d_model)
        self.local_norm_cache = (float(delta.detach().square().mean().sqrt())
                                 if self.capture else None)
        return global_emb + delta

    def extra_parameters(self):
        return sum(p.numel() for name, p in self.named_parameters()
                   if not name.startswith('global_projection.'))
