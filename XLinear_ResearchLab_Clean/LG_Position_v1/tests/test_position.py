"""CPU unit and compatibility tests: all non-GPU, no training benchmark claims."""
import hashlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from src.models.extensions.local_global_embedding import LocalGlobalEmbedding
from src.models.extensions.position_local_embedding import (PositionLocalGlobalEmbedding, position_pool)
from src.models.registry import build_variant_model

ROOT = Path(__file__).resolve().parents[2]


def cfg(h, d=32, n=7):
    return SimpleNamespace(seq_len=96, pred_len=h, d_model=d, enc_in=n,
        t_ff=64, c_ff=7, usenorm=1, embed_dropout=0., head_dropout=0.,
        t_dropout=0., c_dropout=0., features='M', data='ETTh1')


class MockOriginal(nn.Module):
    """Only checks wrapper injection; not a replacement for official XLinear."""
    def __init__(self, config):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.projection = nn.Sequential(nn.Linear(config.seq_len, config.d_model), nn.Dropout(0))
        self.backbone.tgm = nn.Linear(config.d_model, config.d_model)
        self.head = nn.Linear(config.d_model, config.pred_len)

    def forward(self, x):
        h = self.backbone.projection(x.transpose(1, 2))
        return self.head(self.backbone.tgm(h)).transpose(1, 2)


class TestPosition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_pooling_uses_intra_bin_order(self):
        # Reversing each bin keeps mean and max equal, but changes the direction.
        a = torch.arange(96., dtype=torch.float32).view(1, 1, 96)
        b = a.reshape(1, 1, 8, 12).flip(-1).reshape(1, 1, 96)
        self.assertTrue(torch.allclose(torch.nn.functional.adaptive_avg_pool1d(a, 8),
                                       torch.nn.functional.adaptive_avg_pool1d(b, 8)))
        self.assertTrue(torch.allclose(torch.nn.functional.adaptive_max_pool1d(a, 8),
                                       torch.nn.functional.adaptive_max_pool1d(b, 8)))
        pa, pb = position_pool(a), position_pool(b)
        self.assertFalse(torch.allclose(pa, pb))
        self.assertEqual(pa.shape, (1, 16))

    def test_parameter_count_and_exact_parameter_initialization(self):
        for h in (96,192,336,720):
            torch.manual_seed(51)
            short = build_variant_model(MockOriginal, 'lg_short', .25)(cfg(h))
            torch.manual_seed(51)
            position = build_variant_model(MockOriginal, 'lg_position', .25)(cfg(h))
            self.assertEqual(sum(p.numel() for p in short.parameters()),
                             sum(p.numel() for p in position.parameters()))
            self.assertEqual(short.backbone.projection.extra_parameters(),
                             position.backbone.projection.extra_parameters())
            self.assertEqual(short.backbone.projection.extra_parameters(), 1064)
            for key, v in short.state_dict().items():
                self.assertTrue(torch.equal(v, position.state_dict()[key]), key)

    def test_zero_initialization_matches_official_and_gradient(self):
        for h in (96,192,336,720):
            config = cfg(h)
            torch.manual_seed(23)
            official = MockOriginal(config).eval()
            torch.manual_seed(23)
            model = build_variant_model(MockOriginal, 'lg_position', .25)(config).eval()
            x = torch.randn(2,96,7)
            with torch.no_grad():
                a,b = official(x), model(x)
            self.assertEqual(a.shape, b.shape)
            self.assertTrue(torch.equal(a,b),f'init equality broken at h={h}')
            model.train(); opt=torch.optim.SGD(model.parameters(),lr=.1)
            opt.zero_grad(); model(x).square().mean().backward()
            weight=model.backbone.projection.residual.weight
            self.assertIsNotNone(weight.grad)
            self.assertGreater(weight.grad.abs().sum().item(), 0)
            opt.step()
            opt.zero_grad(); model(x).square().mean().backward()
            self.assertIsNotNone(model.backbone.projection.short.weight.grad)
            self.assertGreater(model.backbone.projection.short.weight.grad.abs().sum().item(), 0)

    def test_official_source_optional(self):
        file=ROOT/'vendor/XLinear/models/XLinear.py'
        if not file.is_file():
            self.skipTest('Official vendor source missing; mock integration tests only')
        raw=file.read_bytes()
        sha=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
        self.assertEqual(sha, '9913ba6837a7c8658283636ab7f5727e07f2f709')
        sys.path.insert(0, str(ROOT/'vendor/XLinear'))
        from models.XLinear import Model
        for h in (96,192,336,720):
            config=cfg(h,128)
            torch.manual_seed(72); original=Model(config).eval()
            torch.manual_seed(72); position=build_variant_model(Model, 'lg_position',.25)(config).eval()
            x=torch.randn(2,96,7)
            with torch.no_grad():
                ref=original(x)
                out=position(x)
            self.assertTrue(torch.equal(ref,out),f'official mismatch h={h}')
            self.assertEqual(tuple(out.shape),(2,h,7))


if __name__=='__main__':
    unittest.main()
