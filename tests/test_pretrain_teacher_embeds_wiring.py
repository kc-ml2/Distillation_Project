import inspect
import unittest

import pretrain


class PretrainTeacherEmbedsWiringTest(unittest.TestCase):
    def test_encode_image_called_exactly_once_in_source(self):
        source = inspect.getsource(pretrain.train)
        self.assertEqual(source.count("online_teacher.encode_image("), 1)

    def test_encode_image_result_threaded_into_itc_and_lm(self):
        # whitespace/line-break independent: confirm the same encode_image() result
        # feeds both teacher-feature helpers (>=2 occurrences: itc_feats + lm_logits;
        # the model forward adds more via teacher_image_embeds=) and both are present.
        source = inspect.getsource(pretrain.train)
        self.assertGreaterEqual(source.count("image_embeds=teacher_image_embeds"), 2)
        self.assertIn("online_teacher.itc_feats(", source)
        self.assertIn("online_teacher.lm_logits(", source)

    def test_encode_image_result_threaded_into_forward(self):
        # itm k=4 KD: the once-computed teacher embeds are passed to the student
        # forward, which reuses them for the teacher itm_matrix path (dedup restored).
        source = inspect.getsource(pretrain.train)
        self.assertIn("teacher_image_embeds=teacher_image_embeds", source)


if __name__ == "__main__":
    unittest.main()
