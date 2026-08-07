import unittest
import torch
import torch.nn.functional as F

from distillation.losses import itc_distill_loss, lm_distill_loss, itm_matrix_kd_loss


def _norm(x):
    return F.normalize(x, dim=-1)


class TestItcDistillLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B, self.D, self.temp = 5, 8, 0.05

    def test_zero_when_teacher_equals_student(self):
        img = _norm(torch.randn(self.B, self.D)).requires_grad_(True)
        txt = _norm(torch.randn(self.B, self.D)).requires_grad_(True)
        loss = itc_distill_loss(img, txt, img.detach(), txt.detach(), self.temp)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_positive_when_teacher_differs(self):
        img_s = _norm(torch.randn(self.B, self.D)).requires_grad_(True)
        txt_s = _norm(torch.randn(self.B, self.D)).requires_grad_(True)
        img_t = _norm(torch.randn(self.B, self.D))
        txt_t = _norm(torch.randn(self.B, self.D))
        loss = itc_distill_loss(img_s, txt_s, img_t, txt_t, self.temp)
        self.assertGreater(loss.item(), 0.0)

    def test_gradient_flows_to_student_not_teacher(self):
        img_s = _norm(torch.randn(self.B, self.D)).requires_grad_(True)
        txt_s = _norm(torch.randn(self.B, self.D)).requires_grad_(True)
        img_t = _norm(torch.randn(self.B, self.D))   # no grad
        txt_t = _norm(torch.randn(self.B, self.D))
        loss = itc_distill_loss(img_s, txt_s, img_t, txt_t, self.temp)
        loss.backward()
        self.assertIsNotNone(img_s.grad)
        self.assertIsNotNone(txt_s.grad)
        self.assertFalse(img_t.requires_grad)

    def test_image_text_swap_symmetry(self):
        # swapping (image<->text) on both student and teacher swaps i2t<->t2i,
        # and since we average them the scalar must be identical.
        img_s = _norm(torch.randn(self.B, self.D))
        txt_s = _norm(torch.randn(self.B, self.D))
        img_t = _norm(torch.randn(self.B, self.D))
        txt_t = _norm(torch.randn(self.B, self.D))
        a = itc_distill_loss(img_s, txt_s, img_t, txt_t, self.temp)
        b = itc_distill_loss(txt_s, img_s, txt_t, img_t, self.temp)
        self.assertAlmostEqual(a.item(), b.item(), places=6)

    def test_returns_scalar(self):
        img = _norm(torch.randn(self.B, self.D))
        txt = _norm(torch.randn(self.B, self.D))
        loss = itc_distill_loss(img, txt, img, txt, self.temp)
        self.assertEqual(loss.dim(), 0)


class TestLmDistillLoss(unittest.TestCase):
    """token-level LM logit KD. shift 프레임: 위치 i 로짓이 토큰 i+1 예측,
    유효 마스크는 decoder_targets[:, 1:] != -100 (CE와 동일 집합)."""

    def setUp(self):
        torch.manual_seed(0)
        self.B, self.L, self.V, self.temp = 3, 8, 50, 2.0
        # 샘플별 유효 길이 4/6/8, 나머지는 -100 (pad)
        self.targets = torch.full((self.B, self.L), -100, dtype=torch.long)
        for b, n in enumerate((4, 6, 8)):
            self.targets[b, :n] = torch.randint(0, self.V, (n,))

    def test_zero_when_teacher_equals_student(self):
        logits = torch.randn(self.B, self.L, self.V)
        s = logits.clone().requires_grad_(True)
        loss = lm_distill_loss(s, logits, self.targets, temp=1.0)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_positive_when_teacher_differs(self):
        s = torch.randn(self.B, self.L, self.V, requires_grad=True)
        t = torch.randn(self.B, self.L, self.V)
        loss = lm_distill_loss(s, t, self.targets, self.temp)
        self.assertGreater(loss.item(), 0.0)

    def test_matches_manual_reference(self):
        # 독립 레퍼런스: 유효 위치를 루프로 골라 KL을 직접 합산
        torch.manual_seed(1)
        s = torch.randn(self.B, self.L, self.V)
        t = torch.randn(self.B, self.L, self.V)
        T = self.temp
        total, n = 0.0, 0
        for b in range(self.B):
            for j in range(self.L - 1):                      # 위치 j 로짓 → 토큰 j+1 예측
                if self.targets[b, j + 1].item() == -100:    # 다음 토큰이 pad면 제외
                    continue
                p = F.softmax(t[b, j] / T, dim=-1)
                logq = F.log_softmax(s[b, j] / T, dim=-1)
                total += (p * (p.log() - logq)).sum().item()
                n += 1
        expected = total / n * T ** 2
        actual = lm_distill_loss(s, t, self.targets, T).item()
        self.assertAlmostEqual(actual, expected, places=4)

    def test_pad_positions_do_not_affect_loss(self):
        s = torch.randn(self.B, self.L, self.V)
        t = torch.randn(self.B, self.L, self.V)
        base = lm_distill_loss(s, t, self.targets, self.temp)
        # 미사용 로짓 위치 = 마지막 위치(L-1) + "다음 토큰이 pad"인 위치
        unused = torch.cat([self.targets[:, 1:] == -100,
                            torch.ones(self.B, 1, dtype=torch.bool)], dim=1)  # [B, L]
        s2, t2 = s.clone(), t.clone()
        s2[unused] += 100.0
        t2[unused] -= 100.0
        perturbed = lm_distill_loss(s2, t2, self.targets, self.temp)
        self.assertAlmostEqual(base.item(), perturbed.item(), places=5)

    def test_gradient_flows_to_student_not_teacher(self):
        s = torch.randn(self.B, self.L, self.V, requires_grad=True)
        t = torch.randn(self.B, self.L, self.V, requires_grad=True)  # detach가 막아야 함
        loss = lm_distill_loss(s, t, self.targets, self.temp)
        loss.backward()
        self.assertIsNotNone(s.grad)
        self.assertIsNone(t.grad)

    def test_normalized_per_valid_token(self):
        # 모든 위치가 같은 (s_row, t_row) 분포 쌍이면 유효 토큰 수와 무관하게 loss 동일해야 함
        torch.manual_seed(2)
        s_row = torch.randn(self.V)
        t_row = torch.randn(self.V)

        def loss_with(n_valid):
            targets = torch.full((1, self.L), -100, dtype=torch.long)
            targets[0, :n_valid] = 1
            s = s_row.expand(1, self.L, self.V)
            t = t_row.expand(1, self.L, self.V)
            return lm_distill_loss(s, t, targets, self.temp).item()

        self.assertAlmostEqual(loss_with(3), loss_with(7), places=5)

    def test_vocab_mismatch_raises(self):
        s = torch.randn(self.B, self.L, self.V)
        t = torch.randn(self.B, self.L, self.V + 1)
        with self.assertRaises(AssertionError):
            lm_distill_loss(s, t, self.targets, self.temp)

    def test_returns_scalar(self):
        s = torch.randn(self.B, self.L, self.V)
        t = torch.randn(self.B, self.L, self.V)
        self.assertEqual(lm_distill_loss(s, t, self.targets, self.temp).dim(), 0)


class TestItmMatrixKdLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B = 5
        self.temp = 0.05

    def test_identical_is_zero(self):
        t = torch.randn(self.B, self.B, 2)
        s = t.clone().requires_grad_(True)
        self.assertLess(itm_matrix_kd_loss(s, t, "bidir", self.temp).item(), 1e-6)

    def test_forward_uses_only_z2_channel(self):
        base = torch.randn(self.B, self.B, 2)
        s, t = base.clone(), base.clone()
        s[:, :, 0] += torch.randn(self.B, self.B)          # z1(reverse)만 교란
        self.assertLess(itm_matrix_kd_loss(s, t, "forward", self.temp).item(), 1e-6)
        self.assertGreater(itm_matrix_kd_loss(s, t, "reverse", self.temp).item(), 1e-6)

    def test_reverse_uses_only_z1_channel(self):
        base = torch.randn(self.B, self.B, 2)
        s, t = base.clone(), base.clone()
        s[:, :, 1] += torch.randn(self.B, self.B)          # z2(forward)만 교란
        self.assertLess(itm_matrix_kd_loss(s, t, "reverse", self.temp).item(), 1e-6)
        self.assertGreater(itm_matrix_kd_loss(s, t, "forward", self.temp).item(), 1e-6)

    def test_gradient_flows_to_student_not_teacher(self):
        s = torch.randn(self.B, self.B, 2, requires_grad=True)
        t = torch.randn(self.B, self.B, 2, requires_grad=True)
        itm_matrix_kd_loss(s, t, "bidir", self.temp).backward()
        self.assertIsNotNone(s.grad)
        self.assertIsNone(t.grad)                          # 내부에서 detach

    def test_invalid_direction_raises(self):
        t = torch.randn(self.B, self.B, 2)
        with self.assertRaises(ValueError):
            itm_matrix_kd_loss(t, t, "sideways", self.temp)


if __name__ == "__main__":
    unittest.main()
