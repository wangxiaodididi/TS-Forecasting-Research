"""Position-aware refinement of the existing LG-Short XLinear embedding.

This module differs from LG-Short only in the nonparametric summary of
short-convolution features: 8 ordered bins, with mean and intra-bin direction
instead of mean and maximum. The conv, projection, parameters, initialization,
and original XLinear global path are otherwise the same.
"""

import torch
from torch.nn import functional as F
from src.models.extensions.local_global_embedding import LocalGlobalEmbedding


def position_pool(features: torch.Tensor, bins: int = 8) -> torch.Tensor:
    """[B*C, filters, L] -> [B*C, filters*2*bins] preserving within-bin direction.

    Split each ordered bin into an earlier and later half, average each half.
    Return both the mean and (later-earlier) difference, in channel-first order.
    This summary has exactly the same width as LG-Short's mean+maximum pooling.
    """
    if features.ndim != 3:
        raise ValueError('position_pool expects [N, filters, L]')
    if features.shape[-1] < bins * 2:
        raise ValueError('history length must be at least twice the number of bins')
    half = F.adaptive_avg_pool1d(features, bins * 2)
    half = half.reshape(features.shape[0], features.shape[1], bins, 2)
    mean = half.mean(dim=-1)
    direction = half[..., 1] - half[..., 0]
    return torch.cat((mean, direction), dim=1).flatten(1)


class PositionLocalGlobalEmbedding(LocalGlobalEmbedding):
    """Same parameters as LG-Short; only the pooling statistic changes."""

    def __init__(self, original_projection, seq_len: int, d_model: int,
                 channels: int = 2, bins: int = 8):
        if seq_len < bins * 2:
            raise ValueError('seq_len must be >= 2*bins')
        super().__init__(original_projection, seq_len, d_model,
                         mode='lg_short', channels=channels, bins=bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.seq_len:
            raise ValueError(f'expected [B,C,{self.seq_len}], got {tuple(x.shape)}')
        global_emb = self.global_projection(x)
        b, c, l = x.shape
        local = F.gelu(self.short(x.reshape(b * c, 1, l)))
        summary = position_pool(local, self.bins)
        delta = self.residual(summary).reshape(b, c, self.d_model)
        self.local_norm_cache = (float(delta.detach().square().mean().sqrt())
                                 if self.capture else None)
        return global_emb + delta
