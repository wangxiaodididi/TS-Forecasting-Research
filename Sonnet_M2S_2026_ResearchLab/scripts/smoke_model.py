#!/usr/bin/env python3
"""Synthetic CPU smoke of the UNMODIFIED official Sonnet model. Not a benchmark."""
import argparse, json, time
from types import SimpleNamespace
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=2)
    p.add_argument('--seq-len', type=int, default=48)
    p.add_argument('--pred-len', type=int, default=24)
    p.add_argument('--n-vars', type=int, default=7)
    p.add_argument('--batch-size', type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    from sonnet.mts_model.models.Sonnet import Model
    cfg = SimpleNamespace(
        enc_in=args.n_vars, c_out=1, seq_len=args.seq_len,
        pred_len=args.pred_len, d_model=16, n_atoms=4,
        revin=0, alpha=0.5, downsample_factors=[1],
        data_group='darts',
    )
    torch.manual_seed(2026)
    model = Model(cfg, datamodule=None, lr=0.001).cpu()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    started = time.perf_counter()
    for step in range(args.steps):
        x = torch.randn(args.batch_size, args.seq_len, args.n_vars)
        y = torch.randn(args.batch_size, args.pred_len, 1)
        optim.zero_grad(set_to_none=True)
        pred = model(x)
        assert tuple(pred.shape) == (args.batch_size, args.pred_len, 1), (tuple(pred.shape), args)
        assert torch.isfinite(pred).all(), 'Forward produced NaN / Inf'
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        grad = sum(float(p.grad.detach().norm().item()) for p in model.parameters() if p.grad is not None)
        assert grad > 0, 'No trainable gradients'
        optim.step()
        print(json.dumps({'step': step+1, 'mse': round(float(loss), 6),
                          'grad_l2_sum': round(grad, 6), 'output_shape': list(pred.shape)}))
    print('PASS synthetic CPU forward/backward, seconds=', round(time.perf_counter()-started,3))


if __name__ == '__main__':
    main()
