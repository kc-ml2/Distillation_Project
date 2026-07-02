import unittest
import torch

from models.blip_pretrain import blip_pretrain


class TestForwardLmKd(unittest.TestCase):
    """KD 확장 forward의 CPU 스모크 (small BLIP base/base, 구성 ~1-2분).
    update_train_state=False로 momentum/queue 갱신 경로(분산 필요)를 우회한다."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = blip_pretrain(image_size=224, vit="base", my_bert_size="base",
                                  queue_size=240)
        cls.model.eval()
        cls.image = torch.randn(2, 3, 224, 224)
        cls.caption = ["a green field", "a red car"]

    def _teacher_payload(self):
        # 학생과 동일 규칙으로 만든 가짜 티처 로짓/입력 (forward의 assert를 통과해야 함)
        tok = self.model.tokenizer
        text = tok(self.caption, padding="max_length", truncation=True,
                   max_length=30, return_tensors="pt")
        dec_ids = text.input_ids.clone()
        dec_ids[:, 0] = tok.bos_token_id
        return torch.randn(2, 30, len(tok)), dec_ids

    def test_five_tuple_with_nones_when_no_teacher(self):
        out = self.model(self.image, self.caption, alpha=0.4, update_train_state=False)
        self.assertEqual(len(out), 5)
        loss_ita, loss_itm, loss_lm, loss_itc_kd, loss_lm_kd = out
        for l in (loss_ita, loss_itm, loss_lm):
            self.assertTrue(torch.isfinite(l).all())
        self.assertIsNone(loss_itc_kd)
        self.assertIsNone(loss_lm_kd)

    def test_lm_kd_computed_when_teacher_logits_given(self):
        t_logits, t_ids = self._teacher_payload()
        out = self.model(self.image, self.caption, alpha=0.4, update_train_state=False,
                         teacher_lm_logits=t_logits, teacher_lm_input_ids=t_ids,
                         lm_distill_temp=2.0)
        loss_lm_kd = out[4]
        self.assertIsNotNone(loss_lm_kd)
        self.assertTrue(torch.isfinite(loss_lm_kd))
        self.assertTrue(loss_lm_kd.requires_grad)   # 학생 로짓 경유 grad

    def test_mismatched_teacher_ids_raise(self):
        t_logits, t_ids = self._teacher_payload()
        bad = t_ids.clone()
        bad[0, 1] = (bad[0, 1] + 1) % 30000
        with self.assertRaises(AssertionError):
            self.model(self.image, self.caption, alpha=0.4, update_train_state=False,
                       teacher_lm_logits=t_logits, teacher_lm_input_ids=bad)


if __name__ == "__main__":
    unittest.main()
