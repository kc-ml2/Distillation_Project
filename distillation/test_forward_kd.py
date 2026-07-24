import unittest
import torch
import torch.nn.functional as F

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


class TestForwardItmHintonKd(unittest.TestCase):
    """variant='hinton_kd' forward 스모크 (mock 티처, neg_source=student로 티처-feat 배선 우회)."""

    class _MockTeacher:
        def itm_soft(self, image, enc_ids, att, neg_i, neg_t, temp, image_embeds=None):
            bs = image.size(0)
            return F.softmax(torch.randn(3 * bs, 2), dim=1)   # [3B, 2]

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = blip_pretrain(image_size=224, vit="base", my_bert_size="base", queue_size=240)
        cls.model.eval()
        cls.image = torch.randn(2, 3, 224, 224)
        cls.caption = ["a green field", "a red car"]

    def _itm_mix(self, variant):
        m = {'neg_source': 'student', 'soft_weight': 0.4, 'temp': 2.0,
             'sel_scale': 1.0, 'teacher_image_embeds': None}
        if variant is not None:
            m['variant'] = variant
        return m

    def _forward_loss_itm(self, variant, seed):
        # 동일 seed → 동일 neg 샘플(multinomial)·동일 mock teacher_soft(randn) → vl_output/teacher_soft 동일.
        # 그래서 두 forward의 차이는 오직 손실 함수(variant)뿐이다.
        torch.manual_seed(seed)
        out = self.model(self.image, self.caption, alpha=0.4, update_train_state=False,
                         online_teacher=self._MockTeacher(), itm_mix=self._itm_mix(variant))
        return out[1]

    def test_hinton_kd_finite_and_grad(self):
        loss_itm = self._forward_loss_itm('hinton_kd', seed=0)
        self.assertTrue(torch.isfinite(loss_itm).all())
        self.assertTrue(loss_itm.requires_grad)

    def test_hinton_differs_from_target_mix(self):
        # 분기가 실제로 동작해야만 통과 (fail-first): 배선 전엔 둘 다 target_mix라 동일 → 실패.
        l_hinton = self._forward_loss_itm('hinton_kd', seed=1)
        l_tmix = self._forward_loss_itm('target_mix', seed=1)
        self.assertFalse(torch.allclose(l_hinton, l_tmix, atol=1e-4))

    def test_missing_variant_equals_target_mix(self):
        # 레거시 불변: variant 미지정 == 'target_mix'
        l_none = self._forward_loss_itm(None, seed=2)
        l_tmix = self._forward_loss_itm('target_mix', seed=2)
        self.assertTrue(torch.allclose(l_none, l_tmix, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
