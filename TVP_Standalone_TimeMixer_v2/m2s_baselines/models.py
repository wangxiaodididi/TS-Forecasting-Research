"""Reference benchmarks for M-to-S. No engineered input features.

Model names are literal, not claimed to be upstream published code copies.
DLinear follows the published decomposition + two-linear-head design for a single target.
GRU, LSTM, and TCN are compact, specified reference architectures.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class LastValue(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon
    def forward(self, x):
        return x[:, -1:, -1:].expand(-1, self.horizon, 1)


class SeasonalNaive(nn.Module):
    def __init__(self, horizon, period):
        super().__init__()
        self.horizon = horizon
        self.period = period
    def forward(self, x):
        pattern = x[:, -self.period:, -1]
        idx = torch.arange(self.horizon, device=x.device) % self.period
        return pattern.index_select(dim=1, index=idx).unsqueeze(-1)


class DLinearTarget(nn.Module):
    """DLinear with one input/output channel (target-only), MA kernel 25."""
    def __init__(self, seq_len, horizon, kernel=25):
        super().__init__()
        if kernel % 2 != 1:
            raise ValueError('DLinear moving average requires odd kernel')
        self.kernel = kernel
        self.seasonal = nn.Linear(seq_len, horizon)
        self.trend = nn.Linear(seq_len, horizon)
        # Original DLinear linear-head initialization.
        nn.init.constant_(self.seasonal.weight, 1. / seq_len)
        nn.init.constant_(self.trend.weight, 1. / seq_len)
    def forward(self, x):
        target = x[:, :, -1:].transpose(1, 2)    # [B, 1, L]
        pad = (self.kernel - 1) // 2
        smoothed = F.avg_pool1d(F.pad(target, (pad, pad), mode='replicate'), self.kernel, stride=1)
        season = target - smoothed
        pred = self.seasonal(season) + self.trend(smoothed)
        return pred.transpose(1, 2)


class JointLinear(nn.Module):
    """M->S direct linear regression using every historical input channel."""
    def __init__(self, seq_len, horizon, channels):
        super().__init__()
        self.readout = nn.Linear(seq_len * channels, horizon)
    def forward(self, x):
        return self.readout(x.flatten(start_dim=1)).unsqueeze(-1)


class RecurrentJoint(nn.Module):
    def __init__(self, kind, channels, horizon, hidden=32):
        super().__init__()
        klass = {'gru': nn.GRU, 'lstm': nn.LSTM}[kind]
        self.rnn = klass(input_size=channels, hidden_size=hidden, num_layers=1, batch_first=True)
        self.readout = nn.Linear(hidden, horizon)
    def forward(self, x):
        z, _ = self.rnn(x)
        return self.readout(z[:, -1, :]).unsqueeze(-1)


class CausalConvBlock(nn.Module):
    def __init__(self, channels, dilation, kernel=3):
        super().__init__()
        self.left = (kernel - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel, dilation=dilation)
        self.act = nn.GELU()
    def forward(self, x):
        # pad LEFT only: no future leakage within the historical input window.
        y = self.conv(F.pad(x, (self.left, 0)))
        return x + 0.25 * self.act(y)


class TCNJoint(nn.Module):
    def __init__(self, channels, horizon, hidden=32):
        super().__init__()
        self.entry = nn.Conv1d(channels, hidden, kernel_size=1)
        self.blocks = nn.Sequential(*[CausalConvBlock(hidden, d)
                                      for d in (1,2,4,8,16,32,64,128)])
        self.readout = nn.Linear(hidden, horizon)
    def forward(self, x):
        z = self.blocks(self.entry(x.transpose(1, 2)))
        return self.readout(z[:, :, -1]).unsqueeze(-1)


BENCHMARKS = (
    'last_value', 'seasonal_naive', 'dlinear_target', 'joint_linear',
    'gru_joint', 'lstm_joint', 'tcn_joint'
)

NO_TRAIN = {'last_value', 'seasonal_naive'}
TARGET_ONLY = {'last_value', 'seasonal_naive', 'dlinear_target'}


def make_model(name, channels, seq_len, horizon, period):
    if name == 'last_value': return LastValue(horizon)
    if name == 'seasonal_naive': return SeasonalNaive(horizon, period)
    if name == 'dlinear_target': return DLinearTarget(seq_len, horizon)
    if name == 'joint_linear': return JointLinear(seq_len, horizon, channels)
    if name == 'gru_joint': return RecurrentJoint('gru', channels, horizon)
    if name == 'lstm_joint': return RecurrentJoint('lstm', channels, horizon)
    if name == 'tcn_joint': return TCNJoint(channels, horizon)
    raise ValueError(f'Unknown benchmark: {name}')
