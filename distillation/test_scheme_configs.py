import unittest
import yaml
from distillation.distill_config import validate_itm_mix_config, derive_teacher_keep


def _distill(path):
    with open(path) as f:
        return yaml.safe_load(f)['distill']


class TestSchemeAConfig(unittest.TestCase):
    PATH = 'configs/pretrain_itm_schemeA_tempered.yaml'

    def test_validates(self):
        validate_itm_mix_config(_distill(self.PATH))            # no raise

    def test_itm_block_values(self):
        itm = _distill(self.PATH)['itm_target_mix']
        self.assertTrue(itm['enabled'])
        self.assertEqual(itm['neg_source'], 'teacher')
        self.assertEqual(float(itm['soft_weight']), 1.0)        # W=1 → 순수 tempered-teacher
        self.assertEqual(float(itm['temp']), 2.0)               # T=2
        self.assertEqual(itm.get('variant', 'target_mix'), 'target_mix')

    def test_teacher_keep(self):
        # teacher neg + soft_weight>0 → itc(선택) + itm(스코어)
        self.assertEqual(derive_teacher_keep(_distill(self.PATH)), ('itc', 'itm'))


class TestSchemeBConfig(unittest.TestCase):
    PATH = 'configs/pretrain_itm_schemeB_hinton.yaml'

    def test_validates(self):
        validate_itm_mix_config(_distill(self.PATH))

    def test_itm_block_values(self):
        itm = _distill(self.PATH)['itm_target_mix']
        self.assertTrue(itm['enabled'])
        self.assertEqual(itm['neg_source'], 'teacher')
        self.assertEqual(itm['variant'], 'hinton_kd')
        self.assertEqual(float(itm['temp']), 2.0)
        self.assertGreater(float(itm['soft_weight']), 0.0)      # α>0

    def test_teacher_keep(self):
        self.assertEqual(derive_teacher_keep(_distill(self.PATH)), ('itc', 'itm'))


if __name__ == "__main__":
    unittest.main()
