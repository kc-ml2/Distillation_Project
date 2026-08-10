import unittest
from distillation.distill_config import (
    derive_teacher_keep, need_teacher_image_embeds)


class TestDeriveTeacherKeep(unittest.TestCase):
    def test_baseline_empty(self):
        self.assertEqual(derive_teacher_keep({}), ())

    def test_itm_kd_alone(self):  # itm k=4 KD -> 'itm' only (negatives from student sim)
        cfg = {'itm': {'enabled': True}}
        self.assertEqual(derive_teacher_keep(cfg), ('itm',))

    def test_lm_alone(self):
        self.assertEqual(derive_teacher_keep({'lm': {'enabled': True}}), ('lm',))

    def test_itc_target_mix_alone_needs_itc(self):
        cfg = {'itc_target_mix': {'enabled': True, 'variant': 'queue', 'soft_weight': 0.4}}
        self.assertEqual(derive_teacher_keep(cfg), ('itc',))

    def test_itc_target_mix_combines_with_itm_kd(self):
        cfg = {'itc_target_mix': {'enabled': True, 'variant': 'queue', 'soft_weight': 0.4},
               'itm': {'enabled': True}}
        self.assertEqual(derive_teacher_keep(cfg), ('itc', 'itm'))

    def test_all_three_mechanisms_together(self):
        cfg = {'lm': {'enabled': True},
               'itc_target_mix': {'enabled': True, 'variant': 'queue', 'soft_weight': 0.4},
               'itm': {'enabled': True}}
        self.assertEqual(derive_teacher_keep(cfg), ('itc', 'itm', 'lm'))

    def test_disabled_flags_kept_out(self):
        cfg = {'lm': {'enabled': False}, 'itm': {'enabled': False},
               'itc_target_mix': {'enabled': False}}
        self.assertEqual(derive_teacher_keep(cfg), ())


class TestNeedTeacherImageEmbeds(unittest.TestCase):
    def test_all_false(self):
        self.assertFalse(need_teacher_image_embeds(False, False, False))

    def test_itc_alone(self):
        self.assertTrue(need_teacher_image_embeds(True, False, False))

    def test_lm_alone(self):
        self.assertTrue(need_teacher_image_embeds(False, True, False))

    def test_itm_kd_alone(self):
        self.assertTrue(need_teacher_image_embeds(False, False, True))

    def test_all_three(self):
        self.assertTrue(need_teacher_image_embeds(True, True, True))


if __name__ == "__main__":
    unittest.main()
