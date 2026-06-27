import unittest
import torch
import torch.nn.functional as F

from distillation.losses import itc_distill_loss


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


if __name__ == "__main__":
    unittest.main()
