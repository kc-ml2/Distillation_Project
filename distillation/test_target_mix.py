import unittest
import torch
import torch.nn.functional as F

from distillation.target_mix import ttm_gamma


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


if __name__ == "__main__":
    unittest.main()
