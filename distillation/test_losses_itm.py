# distillation/test_losses_itm.py
import unittest
import torch
import torch.nn.functional as F
from distillation.losses import itm_target_mix_loss


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


if __name__ == "__main__":
    unittest.main()
