import inspect
import unittest

from models.blip_pretrain import BLIP_Pretrain


class TeacherEmbedsDedupWiringTest(unittest.TestCase):
    """Task 6: 티처 encode_image dedup이 pretrain.train → forward로 이동. forward가
    스텝당 encode_image를 1회 계산하고, 각 step이 image_embeds=teacher_image_embeds로
    재사용(itc_feats / lm_logits / itm_matrix)한다. 소스 문자열로 배선을 고정한다."""

    cls_src = inspect.getsource(BLIP_Pretrain)
    fwd_src = inspect.getsource(BLIP_Pretrain.forward)

    def test_encode_image_called_once_in_forward(self):
        self.assertEqual(self.fwd_src.count("online_teacher.encode_image("), 1)

    def test_embeds_threaded_into_teacher_calls(self):
        # itc_feats + lm_logits + itm_matrix(+gathered) 모두 동일 embeds 재사용(>=2).
        self.assertGreaterEqual(self.cls_src.count("image_embeds=teacher_image_embeds"), 2)
        self.assertIn("online_teacher.itc_feats(", self.cls_src)
        self.assertIn("online_teacher.lm_logits(", self.cls_src)

    def test_forward_passes_embeds_to_each_step(self):
        # 한 번 계산한 teacher_image_embeds를 세 step 호출에 모두 넘긴다.
        self.assertIn("teacher_image_embeds", self.fwd_src)


if __name__ == "__main__":
    unittest.main()
