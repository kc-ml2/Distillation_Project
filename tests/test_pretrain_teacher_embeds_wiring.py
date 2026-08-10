import inspect
import unittest

import pretrain


class PretrainTeacherEmbedsWiringTest(unittest.TestCase):
    def test_encode_image_called_exactly_once_in_source(self):
        source = inspect.getsource(pretrain.train)
        self.assertEqual(source.count("online_teacher.encode_image("), 1)

    def test_encode_image_result_threaded_into_itc_and_lm(self):
        # whitespace/line-break independent: just confirm the same encode_image()
        # result feeds both calls (exactly 2 occurrences: itc_feats + lm_logits) and
        # that both call sites are present.
        source = inspect.getsource(pretrain.train)
        self.assertEqual(source.count("image_embeds=teacher_image_embeds"), 2)
        self.assertIn("online_teacher.itc_feats(", source)
        self.assertIn("online_teacher.lm_logits(", source)

    def test_encode_image_result_threaded_into_itm_mix_dict(self):
        source = inspect.getsource(pretrain.train)
        self.assertIn("'teacher_image_embeds': teacher_image_embeds", source)


if __name__ == "__main__":
    unittest.main()
