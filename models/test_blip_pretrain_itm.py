import unittest
import torch
from models.blip_pretrain import blip_pretrain
from distillation.online_teacher import OnlineTeacher


class TestForwardItmMix(unittest.TestCase):
    """CPU smoke: base student forward with/without itm_mix. Construction ~1-2 min."""

    @classmethod
    def setUpClass(cls):
        cls.model = blip_pretrain(image_size=224, vit="base", my_bert_size="base", queue_size=240)
        cls.model.eval()
        cls.image = torch.randn(2, 3, 224, 224)
        cls.caption = ["a green field", "a red car"]

    def _forward(self, **kw):
        return self.model(self.image, self.caption, alpha=0.0, update_train_state=False, **kw)

    def test_baseline_itm_none(self):
        loss_ita, loss_itm, loss_lm, kd_itc, kd_lm = self._forward()
        self.assertTrue(torch.isfinite(loss_itm))
        self.assertTrue(loss_itm.requires_grad)

    def test_student_neg_w0_is_plain_ce(self):
        itm_mix = {'neg_source': 'student', 'soft_weight': 0.0, 'temp': 1.0, 'sel_scale': 14.29}
        loss_itm = self._forward(itm_mix=itm_mix)[1]
        self.assertTrue(torch.isfinite(loss_itm))

    def test_student_neg_w_positive_with_teacher(self):
        teacher = OnlineTeacher(checkpoint="", image_size=224, vit="base",
                                bert="base", queue_size=240, keep=("itm",))
        itm_mix = {'neg_source': 'student', 'soft_weight': 0.4, 'temp': 1.0, 'sel_scale': 14.29}
        loss_itm = self._forward(online_teacher=teacher, itm_mix=itm_mix)[1]
        self.assertTrue(torch.isfinite(loss_itm))
        self.assertTrue(loss_itm.requires_grad)

    def test_teacher_neg_requires_feats(self):
        itm_mix = {'neg_source': 'teacher', 'soft_weight': 0.0, 'temp': 1.0, 'sel_scale': 63.9}
        with self.assertRaises(AssertionError):
            self._forward(itm_mix=itm_mix)   # teacher feats not provided


if __name__ == "__main__":
    unittest.main()
