"""CSRA: 逐变量、逐周期预测段的轻量季节残差适配器。

仅使用预测起点前的历史序列；不修改官方 XLinear 主干。
输入:  history [B, L, C]；官方输出 forecast [B, H, C]
输出:  forecast + seasonal_correction [B, H, C]

三个可解释残差基函数：
  1. 最近24小时模板与官方预测的差
  2. 最近最多4天的平均周期模板与官方预测的差
  3. 最近两日同一小时的变化量
每个变量、每个24步预测块拥有3个系数，用 tanh 限制在 [-1,1]。
系数严格零初始化，首次前向与原模型逐元素一致。
"""
import math

import torch
from torch import nn


class SeasonalResidualAdapter(nn.Module):
    def __init__(self, channels: int, seq_len: int, pred_len: int, period: int = 24,
                 max_cycles: int = 4):
        super().__init__()
        self.channels = int(channels)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.period = int(period)
        self.max_cycles = int(max_cycles)
        if self.period <= 0 or self.seq_len < 2 * self.period:
            raise ValueError('CSRA需要至少两个完整历史周期；请检查采样频率和seq_len')
        self.n_cycles = min(self.max_cycles, self.seq_len // self.period)
        self.blocks = math.ceil(self.pred_len / self.period)
        self.coefficients = nn.Parameter(torch.zeros(self.channels, self.blocks, 3))

    def _features(self, history: torch.Tensor, original_forecast: torch.Tensor):
        if history.ndim != 3 or original_forecast.ndim != 3:
            raise ValueError('CSRA输入必须是 [batch,time,channels]')
        batch, lookback, channels = history.shape
        if lookback != self.seq_len or channels != self.channels:
            raise ValueError(f'历史数据维度不符: {tuple(history.shape)}，期望 L={self.seq_len} C={self.channels}')
        if original_forecast.shape != (batch, self.pred_len, channels):
            raise ValueError(f'官方预测维度不符: {tuple(original_forecast.shape)}')
        if history.device != original_forecast.device or history.dtype != original_forecast.dtype:
            raise ValueError('历史与预测张量须使用同一设备和 dtype')
        # 最近 n 个完整周期；不读取未来真实值。
        cycles = history[:, -self.n_cycles*self.period:, :].reshape(
            batch, self.n_cycles, self.period, channels)
        last = cycles[:, -1, :, :]
        older = cycles[:, -2, :, :]
        average = cycles.mean(dim=1)
        phase = torch.arange(self.pred_len, device=history.device) % self.period
        last_pattern = last.index_select(1, phase)
        mean_pattern = average.index_select(1, phase)
        change_pattern = (last - older).index_select(1, phase)
        return torch.stack((last_pattern - original_forecast,
                            mean_pattern - original_forecast,
                            change_pattern), dim=-1)

    def forward(self, history: torch.Tensor, official_forecast: torch.Tensor):
        features = self._features(history, official_forecast)
        block_index = (torch.arange(self.pred_len, device=history.device) // self.period)
        # [C, blocks, 3] -> [H, C, 3]
        weights = self.coefficients.tanh().permute(1, 0, 2).index_select(0, block_index)
        correction = (features * weights.unsqueeze(0)).sum(dim=-1)
        return official_forecast + correction

    @torch.no_grad()
    def learned_weights(self):
        """训练完成后可直接读取有物理含义的变量/预测块修正系数。"""
        return self.coefficients.tanh().detach().cpu()
