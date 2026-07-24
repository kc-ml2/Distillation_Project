# distillation/test_losses_itm.py
import unittest
import torch
import torch.nn.functional as F
from distillation.losses import itm_target_mix_loss, itm_hinton_kd_loss


class TestItmTargetMixLoss(unittest.TestCase):
    def test_w0_equals_cross_entropy(self):
        vl = torch.tensor([[2.0, -1.0], [0.5, 0.5], [-1.0, 3.0]])
        labels = torch.tensor([0, 1, 1])
        teacher = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.1, 0.9]])
        got = itm_target_mix_loss(vl, labels, teacher, 0.0)
        self.assertTrue(torch.allclose(got, F.cross_entropy(vl, labels), atol=1e-6))

    def test_w1_teacher_onehot_equals_ce(self):
        labels = torch.tensor([0, 1])
        vl = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        teacher = F.one_hot(labels, 2).float()
        got = itm_target_mix_loss(vl, labels, teacher, 1.0)
        self.assertTrue(torch.allclose(got, F.cross_entropy(vl, labels), atol=1e-6))

    def test_matches_manual_soft_ce(self):
        vl = torch.tensor([[1.5, -0.5]])
        labels = torch.tensor([0])
        teacher = torch.tensor([[0.2, 0.8]])
        W = 0.4
        target = (1 - W) * torch.tensor([[1.0, 0.0]]) + W * teacher
        manual = -(target * F.log_softmax(vl, dim=1)).sum(1).mean()
        self.assertTrue(torch.allclose(itm_target_mix_loss(vl, labels, teacher, W), manual, atol=1e-6))

    def test_gradient_flows_to_logits(self):
        vl = torch.randn(3, 2, requires_grad=True)
        teacher = torch.softmax(torch.randn(3, 2), 1)
        labels = torch.tensor([0, 1, 0])
        itm_target_mix_loss(vl, labels, teacher, 0.4).backward()
        self.assertIsNotNone(vl.grad)
        self.assertTrue(torch.isfinite(vl.grad).all())


class TestItmHintonKdLoss(unittest.TestCase):
    def test_alpha0_equals_cross_entropy(self):
        vl = torch.tensor([[2.0, -1.0], [0.5, 0.5], [-1.0, 3.0]])
        labels = torch.tensor([0, 1, 1])
        teacher = torch.softmax(torch.randn(3, 2), dim=1)
        got = itm_hinton_kd_loss(vl, labels, teacher, alpha=0.0, temp=2.0)
        self.assertTrue(torch.allclose(got, F.cross_entropy(vl, labels), atol=1e-6))

    def test_matches_manual_2term(self):
        vl = torch.tensor([[1.5, -0.5], [0.2, 0.9]])
        labels = torch.tensor([0, 1])
        teacher = torch.tensor([[0.3, 0.7], [0.4, 0.6]])
        alpha, T = 0.4, 2.0
        hard = F.cross_entropy(vl, labels)
        logq = F.log_softmax(vl / T, dim=1)
        kl = (teacher * (teacher.log() - logq)).sum(1).mean() * (T ** 2)  # batchmean·T²
        expected = (1 - alpha) * hard + alpha * kl
        got = itm_hinton_kd_loss(vl, labels, teacher, alpha, T)
        self.assertTrue(torch.allclose(got, expected, atol=1e-6))

    def test_gradient_finite_to_logits_only(self):
        vl = torch.randn(4, 2, requires_grad=True)
        teacher = torch.softmax(torch.randn(4, 2), dim=1)   # no grad
        labels = torch.tensor([0, 1, 0, 1])
        itm_hinton_kd_loss(vl, labels, teacher, alpha=0.4, temp=2.0).backward()
        self.assertIsNotNone(vl.grad)
        self.assertTrue(torch.isfinite(vl.grad).all())
        self.assertFalse(teacher.requires_grad)

    def test_returns_scalar(self):
        vl = torch.randn(3, 2)
        teacher = torch.softmax(torch.randn(3, 2), dim=1)
        loss = itm_hinton_kd_loss(vl, torch.tensor([0, 1, 1]), teacher, 0.4, 2.0)
        self.assertEqual(loss.dim(), 0)


if __name__ == "__main__":
    unittest.main()
