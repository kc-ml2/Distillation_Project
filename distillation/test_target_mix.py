import unittest
import torch
import torch.nn.functional as F

from distillation.target_mix import (
    ttm_gamma, teacher_soft_in_batch, teacher_soft_queue, mix_target,
)


def _norm(x):
    return F.normalize(x, dim=-1)


class TestTtmGamma(unittest.TestCase):
    def test_hold_region_is_one(self):
        self.assertEqual(ttm_gamma(0, 100, 600), 1.0)
        self.assertEqual(ttm_gamma(99, 100, 600), 1.0)
        self.assertEqual(ttm_gamma(100, 100, 600), 1.0)   # 감쇠 시작점 포함

    def test_linear_midpoint(self):
        # hold=100, end=600 → 감쇠구간 500. 중점 step=350 → γ=0.5
        self.assertAlmostEqual(ttm_gamma(350, 100, 600), 0.5, places=6)

    def test_decay_end_and_beyond_is_zero(self):
        self.assertEqual(ttm_gamma(600, 100, 600), 0.0)
        self.assertEqual(ttm_gamma(999, 100, 600), 0.0)

    def test_monotonic_non_increasing(self):
        vals = [ttm_gamma(s, 100, 600) for s in range(0, 700, 10)]
        self.assertTrue(all(a >= b for a, b in zip(vals, vals[1:])))


class TestTeacherSoftAndMix(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B, self.D, self.Q, self.tau = 4, 8, 6, 0.05

    def test_in_batch_shape_and_zero_pad(self):
        row, col = _norm(torch.randn(self.B, self.D)), _norm(torch.randn(self.B, self.D))
        n_cols = self.B + self.Q
        out = teacher_soft_in_batch(row, col, self.tau, n_cols)
        self.assertEqual(tuple(out.shape), (self.B, n_cols))
        self.assertTrue(torch.all(out[:, self.B:] == 0))            # queue 열 0
        self.assertTrue(torch.allclose(out.sum(dim=1), torch.ones(self.B), atol=1e-5))  # 행 합 1

    def test_queue_shape_and_rowsum(self):
        row = _norm(torch.randn(self.B, self.D))
        col_all = _norm(torch.randn(self.B + self.Q, self.D)).t()   # [D, B+Q]
        out = teacher_soft_queue(row, col_all, self.tau)
        self.assertEqual(tuple(out.shape), (self.B, self.B + self.Q))
        self.assertTrue(torch.allclose(out.sum(dim=1), torch.ones(self.B), atol=1e-5))

    def test_mix_target_rowsum_and_endpoints(self):
        N = self.B + self.Q
        onehot = torch.zeros(self.B, N); onehot[:, :self.B].fill_diagonal_(1)
        mom = F.softmax(torch.randn(self.B, N), dim=1)
        tea = F.softmax(torch.randn(self.B, N), dim=1)
        W = 0.4
        for g in (0.0, 0.3, 1.0):
            t = mix_target(onehot, mom, tea, g, W)
            self.assertTrue(torch.allclose(t.sum(dim=1), torch.ones(self.B), atol=1e-5))
            self.assertTrue(torch.all(t >= 0))
        # γ=1 → 티처만, γ=0 → momentum만 (soft 성분)
        t1 = mix_target(onehot, mom, tea, 1.0, W)
        self.assertTrue(torch.allclose(t1, (1 - W) * onehot + W * tea, atol=1e-6))
        t0 = mix_target(onehot, mom, tea, 0.0, W)
        self.assertTrue(torch.allclose(t0, (1 - W) * onehot + W * mom, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
