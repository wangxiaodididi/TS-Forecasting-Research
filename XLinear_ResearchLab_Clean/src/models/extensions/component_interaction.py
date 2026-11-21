"""CC-XLinear: component-conditioned peer routing *before* the original VGM.

研究问题：多变量序列的时间结构存在差异，统一的跨变量信息交互能否按趋势、
日周期和短期残差的一致性进行条件选择？

保持作者 2026 XLinear 的 projection、Global Token、TGM、VGM、head 原封不动；
改变的是输入 VGM 的 Global Token 路径：

  H = TGM([Embedding(X), GlobalToken]) 后半支，维度 [B,C,D]
  P_{ij}=softmax_j(B_{ij} + sum_k alpha_k S^k_{ij}(X)) (j != i)
  H' = H + 0.5 * (P - U) H,  U_{ij}=1/(C-1), j != i
  VGM([Embedding(X), H']) -> 原始 Head

P 和 U 是每行的跨变量分配；使用 P-U 保证零初始化完全退化为原模型，
并且在第一轮反向传播时，B、alpha 均可直接收到梯度。

三种实验变体：
- cc_components: 趋势、日周期（含稳定性）、局部残差的分量相似性
- cc_raw:       不分解的原始历史、近期历史、差分相似性；参数量完全相同
- cc_static:    只学习样本无关的同一跨变量交互矩阵；以同样参数张量作近似容量对照，
                其中 alpha 的 3 个系数在此版本不参与前向，需如实报告。

注意：本模块的相似性和路由权重只是信息路径统计，不应解释为变量因果影响。
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ComponentConditionedVGM(nn.Module):
    MODES = ('cc_components', 'cc_raw', 'cc_static')

    def __init__(self, official_backbone: nn.Module, seq_len: int, channels: int,
                 mode: str, period: int = 24):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f'不支持模式: {mode}')
        if int(channels) < 2:
            raise ValueError('跨变量路由至少需要两个变量')
        if seq_len < 2 * period:
            raise ValueError(f'历史长度 {seq_len} 不足两个完整周期 (period={period})')
        for name in ('projection', 'glob_token', 'en_attention', 'ex_attention', 'd_model', 'channel'):
            if not hasattr(official_backbone, name):
                raise RuntimeError(f'官方 backbone 缺少 {name}，请核实源码版本')
        if int(official_backbone.channel) != int(channels):
            raise ValueError('channel 数量与官方 backbone 不符')
        self.original = official_backbone
        self.seq_len = int(seq_len)
        self.channels = int(channels)
        self.period = int(period)
        self.mode = mode
        self.peer_logits = nn.Parameter(torch.zeros(channels, channels))
        self.component_weights = nn.Parameter(torch.zeros(3))
        self.mix_scale = 0.5   # 固定机制设计，不作为每次实验的寻优旋钮
        eye = torch.eye(channels, dtype=torch.bool)
        self.register_buffer('_diag_mask', eye, persistent=False)
        # 非持久性缓存：仅在显式解释模式下使用，不影响前向和参数状态。
        self.capture_routing = False
        self.last_routing: torch.Tensor | None = None

    @staticmethod
    def _similarity(signals: torch.Tensor) -> torch.Tensor:
        """signals [B,C,T] -> cosine similarity [B,C,C]；零能量 -> 零。"""
        centered = signals - signals.mean(dim=-1, keepdim=True)
        unit = F.normalize(centered, dim=-1, eps=1e-6)
        return unit @ unit.transpose(1, 2)

    def _component_scores(self, x: torch.Tensor) -> torch.Tensor:
        """样本仅使用当前历史窗口数据 [B,C,L]，不需要未来值或外部标签。"""
        b, c, l = x.shape
        period = self.period
        if self.mode == 'cc_raw':
            # 参数量/推理路径与分量版一致；但不执行趋势—周期—残差分离。
            raw_coarse = F.adaptive_avg_pool1d(x, 12)
            raw_recent = x[:, :, -period:]
            recent_diff = x[:, :, -period:] - x[:, :, -(period+1):-1]
            return torch.stack((self._similarity(raw_coarse),
                                self._similarity(raw_recent),
                                self._similarity(recent_diff)), dim=1)

        # 跨时间的滑动均值是每个独立预测窗口的历史处理；未来时点不可见。
        trend = F.avg_pool1d(F.pad(x, (period // 2, period // 2), mode='replicate'),
                             kernel_size=period+1, stride=1)
        if trend.shape[-1] != l:
            # 只允许偶数 period；ETTh1 的 period=24 走上述无裁剪路径。
            raise RuntimeError('趋势平滑长度异常，请使用偶数采样周期')
        detrended = x - trend
        n_cycles = l // period
        cycles = detrended[:, :, -n_cycles*period:].reshape(b, c, n_cycles, period)
        cycle_pattern = cycles.mean(dim=2)                     # [B,C,period]
        # 此处定义的是多日平均相位形状；不等同于假设未来必然重复该周期。
        cycle_variance = (cycles - cycle_pattern.unsqueeze(2)).square().mean(dim=(2,3))
        cycle_power = cycle_pattern.square().mean(dim=-1)
        reliability = (cycle_power/(cycle_power+cycle_variance+1e-6)).clamp(0,1)
        recent_irregular = cycles[:, :, -1, :] - cycle_pattern
        trend_shape = F.adaptive_avg_pool1d(trend, 12)
        trend_sim = self._similarity(trend_shape)
        cycle_sim = self._similarity(cycle_pattern)
        cycle_sim = cycle_sim * torch.sqrt(reliability.unsqueeze(2)*reliability.unsqueeze(1))
        irregular_sim = self._similarity(recent_irregular)
        return torch.stack((trend_sim, cycle_sim, irregular_sim), dim=1)

    def _routing(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, _ = x.shape
        logits = self.peer_logits.unsqueeze(0).expand(b, -1, -1)
        if self.mode != 'cc_static':
            scores = self._component_scores(x)
            # 权重初始全为 0；避免任何未经验证的频段系数假定。
            logits = logits + torch.einsum('k,bkij->bij', self.component_weights, scores)
        # 对角必须严格排除；避免将自身的信息重命名为变量交互。
        very_negative = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(self._diag_mask.unsqueeze(0), very_negative)
        routing = logits.softmax(dim=-1)
        # 与原模型等价的均匀跨变量参照路由。
        zero_logits = torch.zeros_like(logits).masked_fill(self._diag_mask.unsqueeze(0), very_negative)
        uniform = zero_logits.softmax(dim=-1)
        return routing, uniform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.channels, self.seq_len):
            raise ValueError(f'期望 X [B,{self.channels},{self.seq_len}]，实际 {tuple(x.shape)}')
        b, c, _ = x.shape
        core = self.original
        d = core.d_model
        emb = core.projection(x)
        global_token = core.glob_token.expand(b, -1, -1)
        temporal = core.en_attention(torch.cat((emb,global_token),dim=-1))
        orig_temporal, temporal_global = temporal[:, :, :d], temporal[:, :, d:]
        routing, uniform = self._routing(x)
        # 零初始化时 P == U，改造输出严格退化为官方 TGM+VGM。
        routed_global = temporal_global + self.mix_scale * torch.bmm(routing-uniform, temporal_global)
        cross = core.ex_attention(torch.cat((emb,routed_global),dim=1).transpose(1,2))
        variable_global = cross[:,:,c:].transpose(1,2)
        if self.capture_routing:
            self.last_routing = routing.detach()
        else:
            self.last_routing = None
        return torch.cat((orig_temporal,variable_global),dim=-1)

    @torch.no_grad()
    def interpretation(self) -> dict:
        """仅报告可观察到的参数状态；不声称可构成因果解释。"""
        return {
            'mode': self.mode,
            'component_names': ['trend','daily_cycle','local_irregular'] if self.mode=='cc_components'
                               else ['raw_coarse','raw_recent','raw_difference'] if self.mode=='cc_raw' else [],
            'component_coefficients': self.component_weights.detach().cpu().tolist(),
            'static_logit_magnitude': float(self.peer_logits.detach().abs().mean()),
            'new_parameter_count': sum(p.numel() for p in (self.peer_logits,self.component_weights)),
            'note': '路由系数只是模型内部信息分配，不代表因果关系或真实物理贡献。',
        }
