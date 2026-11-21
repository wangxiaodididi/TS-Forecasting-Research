"""Phase-calibrated innovation branch for official ICLR-2026 GTR.

Research question: does a local innovation relative to the retrieved global phase
reference have complementary predictive information?

- The official GTR.Model is used unchanged for the primary prediction.
- Its learnable Q and GTR.linear furnish a phase-indexed *latent* reference.
- A past-only per-window affine projection maps that reference into the
  normalized observed series; we forecast only the remaining innovation.
- The projection is detached from Q so the auxiliary head cannot alter the
  meaning of Q merely to make its own residual easier to forecast. Q still
  trains normally through the official GTR forward pass.
- Agreement R2 is a descriptive *reference-fit statistic*, not calibrated
  probability, forecast confidence, or a causal explanation.
"""
import torch
import torch.nn as nn
from models import GTR


def phase_calibrated_innovation(xn, reference, ridge=1e-3):
    """Past-only affine fit per sample/channel.

    xn, reference: [batch, channels, lookback], normalized xn.  Returns
    innovation, fitted_reference, bounded fit R2 in [0,1].  Reference shape
    matters, not its original scale/offset.  All operations differentiable
    w.r.t. xn; reference is detached by the caller.
    """
    if xn.shape != reference.shape or xn.ndim != 3:
        raise ValueError('Expected identical [B,C,T] input and reference shapes')
    if ridge <= 0:
        raise ValueError('ridge must be positive')
    xc = xn - xn.mean(dim=-1, keepdim=True)
    qc = reference - reference.mean(dim=-1, keepdim=True)
    qvar = qc.square().mean(dim=-1, keepdim=True)
    cov = (xc * qc).mean(dim=-1, keepdim=True)
    # Ridge shrinkage stabilizes Q while it is nearly constant early in training.
    coeff = cov / (qvar + float(ridge))
    fitted = coeff * qc
    innovation = xc - fitted
    orig_power = xc.square().mean(dim=-1)
    residual_power = innovation.square().mean(dim=-1)
    fit_r2 = (1.0 - residual_power / orig_power.clamp_min(1e-8)).clamp(0.0, 1.0)
    return innovation, fitted, fit_r2


class PhaseInnovationModel(nn.Module):
    def __init__(self, configs, adaptive=False):
        super().__init__()
        self.base = GTR.Model(configs)   # Exact upstream model, never rewritten.
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.use_revin = bool(configs.use_revin)
        self.adaptive = bool(adaptive)
        self.ridge = 1e-3  # fixed, not a paper hyperparameter search
        self.innovation_head = nn.Linear(self.seq_len, self.pred_len, bias=False)
        nn.init.zeros_(self.innovation_head.weight)
        if adaptive:
            # Shared across channels, only 4 trainable parameters.
            # Gain is 1 initially; adjustment stays within [0.5, 1.5].
            self.gain_head = nn.Linear(3, 1)
            nn.init.zeros_(self.gain_head.weight)
            nn.init.zeros_(self.gain_head.bias)

    def forward(self, x, cycle_index, return_components=False):
        # Original model performs its own RevIN, retrieval, fusion and MLP.
        baseline = self.base(x, cycle_index)
        if x.ndim != 3 or x.size(1) != self.seq_len:
            raise ValueError('Expected input [B,seq_len,C]')
        cycle_index = cycle_index.long().reshape(-1)
        if cycle_index.numel() != x.size(0):
            raise ValueError('One cycle index required per sample')
        if self.use_revin:
            mu = x.mean(dim=1, keepdim=True)
            # Matches official torch.var default (correction=1).
            scale = (x.var(dim=1, keepdim=True) + 1e-5).sqrt()
            xn = (x - mu) / scale
        else:
            scale = torch.ones_like(x[:, :1, :])
            xn = x
        xn_bc = xn.transpose(1, 2)  # [B,C,T]

        # Absolute phase indexing is IDENTICAL to official GTR.Model.forward.
        ind = (cycle_index.view(-1, 1) +
               torch.arange(self.seq_len, device=x.device).view(1, -1)) % self.base.cycle_len
        # Learned Q is latent, not a physical seasonal series. Fit it to the
        # current historical window before any agreement measurement.
        with torch.no_grad():
            q = self.base.Q[ind].permute(0, 2, 1)
            ref = self.base.GTR.linear(q)
        innovation, fitted, fit_r2 = phase_calibrated_innovation(xn_bc, ref, self.ridge)
        correction = self.innovation_head(innovation).transpose(1, 2) * scale
        gain = torch.ones_like(fit_r2)
        if self.adaptive:
            # All covariates use only observed past: fit, local innovation
            # differences, and first-vs-last-quarter drift. No future inputs.
            k = max(2, self.seq_len // 4)
            local_diff = innovation[..., -k:].diff(dim=-1)
            diff_energy = local_diff.square().mean(dim=-1)
            diff_feature = torch.log1p(diff_energy.clamp_min(0.0))
            drift = (innovation[..., -k:].mean(dim=-1) - innovation[..., :k].mean(dim=-1)).abs()
            feat = torch.stack([fit_r2, diff_feature, torch.log1p(drift)], dim=-1)
            gain = 1.0 + 0.5 * torch.tanh(self.gain_head(feat).squeeze(-1))
            correction = correction * gain.unsqueeze(1)
        output = baseline + correction
        if return_components:
            return dict(prediction=output,
                        gtr_prediction=baseline,
                        innovation_correction=correction,
                        phase_fit_r2=fit_r2,
                        phase_mismatch=1.0-fit_r2,
                        correction_gain=gain,
                        innovation_energy=innovation.square().mean(dim=-1),
                        reference_fit_energy=fitted.square().mean(dim=-1))
        return output

    def explain(self, x, cycle_index):
        return self.forward(x, cycle_index, return_components=True)
