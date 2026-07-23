import unittest
from distillation.distill_config import derive_teacher_keep, validate_itm_mix_config


class TestDeriveTeacherKeep(unittest.TestCase):
    def test_baseline_empty(self):
        self.assertEqual(derive_teacher_keep({}), ())

    def test_arm_A_student_wpos(self):  # student neg + W>0 -> itm only
        cfg = {'itm_target_mix': {'enabled': True, 'neg_source': 'student', 'soft_weight': 0.4}}
        self.assertEqual(derive_teacher_keep(cfg), ('itm',))

    def test_arm_B_teacher_w0(self):    # teacher neg + W=0 -> itc only (selection feats)
        cfg = {'itm_target_mix': {'enabled': True, 'neg_source': 'teacher', 'soft_weight': 0.0}}
        self.assertEqual(derive_teacher_keep(cfg), ('itc',))

    def test_arm_C_teacher_wpos(self):  # teacher neg + W>0 -> itc + itm
        cfg = {'itm_target_mix': {'enabled': True, 'neg_source': 'teacher', 'soft_weight': 0.4}}
        self.assertEqual(derive_teacher_keep(cfg), ('itc', 'itm'))

    def test_combines_with_itc_lm(self):
        cfg = {'itc': {'enabled': True}, 'lm': {'enabled': True},
               'itm_target_mix': {'enabled': True, 'neg_source': 'student', 'soft_weight': 0.4}}
        self.assertEqual(derive_teacher_keep(cfg), ('itc', 'itm', 'lm'))

    def test_itc_target_mix_alone_needs_itc(self):
        cfg = {'itc_target_mix': {'enabled': True, 'variant': 'queue', 'soft_weight': 0.4}}
        self.assertEqual(derive_teacher_keep(cfg), ('itc',))

    def test_itc_target_mix_combines_with_itm_target_mix(self):
        cfg = {'itc_target_mix': {'enabled': True, 'variant': 'queue', 'soft_weight': 0.4},
               'itm_target_mix': {'enabled': True, 'neg_source': 'teacher', 'soft_weight': 0.4}}
        self.assertEqual(derive_teacher_keep(cfg), ('itc', 'itm'))

    def test_all_three_mechanisms_together(self):
        cfg = {'lm': {'enabled': True},
               'itc_target_mix': {'enabled': True, 'variant': 'queue', 'soft_weight': 0.4},
               'itm_target_mix': {'enabled': True, 'neg_source': 'teacher', 'soft_weight': 0.4}}
        self.assertEqual(derive_teacher_keep(cfg), ('itc', 'itm', 'lm'))


class TestValidateItmMixConfig(unittest.TestCase):
    def test_ok_passes(self):
        validate_itm_mix_config({'itm_target_mix': {'enabled': True, 'neg_source': 'teacher',
                                                    'soft_weight': 0.4, 'temp': 1.0,
                                                    'schedule': 'constant'}})

    def test_disabled_noop(self):
        validate_itm_mix_config({})  # no raise

    def test_bad_neg_source(self):
        with self.assertRaises(AssertionError):
            validate_itm_mix_config({'itm_target_mix': {'enabled': True, 'neg_source': 'foo',
                                                        'soft_weight': 0.4}})

    def test_bad_weight(self):
        with self.assertRaises(AssertionError):
            validate_itm_mix_config({'itm_target_mix': {'enabled': True, 'neg_source': 'teacher',
                                                        'soft_weight': 1.5}})

    def test_unsupported_schedule(self):
        with self.assertRaises(AssertionError):
            validate_itm_mix_config({'itm_target_mix': {'enabled': True, 'neg_source': 'teacher',
                                                        'soft_weight': 0.4, 'schedule': 'decay_to_floor'}})


if __name__ == "__main__":
    unittest.main()
