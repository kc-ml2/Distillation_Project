import unittest
from types import SimpleNamespace
import torch
import torch.nn as nn

from distillation.itm_matrix import itm_bxb_logits


class FakeEncoder(nn.Module):
    """CLS[b] = [image-marker, text-marker] — (i,j) 조립 순서를 검증하기 위한 가짜."""
    def forward(self, input_ids, attention_mask, encoder_hidden_states,
                encoder_attention_mask, return_dict):
        B, L = input_ids.shape
        hidden = torch.zeros(B, L, 2)
        hidden[:, 0, 0] = encoder_hidden_states[:, 0, 0]   # image marker (row 내 상수)
        hidden[:, 0, 1] = input_ids[:, 1].float()          # text marker (col 따라 변화)
        return SimpleNamespace(last_hidden_state=hidden)


class TestItmBxbLogits(unittest.TestCase):
    def test_matrix_assembly_row_image_col_text(self):
        B = 4
        image_embeds = torch.zeros(B, 10, 8)
        for i in range(B):
            image_embeds[i, 0, 0] = i                      # image i 표식
        image_atts = torch.ones(B, 10, dtype=torch.long)
        input_ids = torch.zeros(B, 6, dtype=torch.long)
        for j in range(B):
            input_ids[j, 1] = j                            # text j 표식
        text_atts = torch.ones(B, 6, dtype=torch.long)

        logits = itm_bxb_logits(FakeEncoder(), nn.Identity(),
                                image_embeds, image_atts, input_ids, text_atts)

        self.assertEqual(tuple(logits.shape), (B, B, 2))
        self.assertEqual(logits.dtype, torch.float32)
        for i in range(B):
            for j in range(B):
                self.assertAlmostEqual(logits[i, j, 0].item(), i)   # 행 = image i
                self.assertAlmostEqual(logits[i, j, 1].item(), j)   # 열 = text j

    def test_itm_head_forced_fp32_even_under_autocast(self):
        """Verify itm_head stays fp32 even when outer autocast(bfloat16) is active."""
        B = 3
        image_embeds = torch.randn(B, 5, 8)  # FakeEncoder will output depth 2
        image_atts = torch.ones(B, 5, dtype=torch.long)
        input_ids = torch.zeros(B, 6, dtype=torch.long)
        text_atts = torch.ones(B, 6, dtype=torch.long)
        itm_head = nn.Linear(2, 2)  # autocast-sensitive matmul, unlike nn.Identity

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            logits = itm_bxb_logits(FakeEncoder(), itm_head,
                                    image_embeds, image_atts, input_ids, text_atts)

        self.assertEqual(logits.dtype, torch.float32)
        self.assertEqual(itm_head.weight.dtype, torch.float32)  # head params stayed fp32

    def test_checkpoint_matches_noncheckpoint_and_flows_grad(self):
        """checkpoint on/off 출력 동일(투명) + checkpoint 경로로 grad가 image_embeds까지 흐른다."""
        torch.manual_seed(0)
        B = 4
        image_embeds = torch.randn(B, 5, 4, requires_grad=True)
        image_atts = torch.ones(B, 5, dtype=torch.long)
        input_ids = torch.zeros(B, 6, dtype=torch.long)
        for j in range(B):
            input_ids[j, 1] = j
        text_atts = torch.ones(B, 6, dtype=torch.long)
        head = nn.Linear(2, 2)  # FakeEncoder는 depth-2 cls를 뱉으므로 2->2

        out_plain = itm_bxb_logits(FakeEncoder(), head, image_embeds, image_atts,
                                   input_ids, text_atts, use_checkpoint=False)
        out_ckpt = itm_bxb_logits(FakeEncoder(), head, image_embeds, image_atts,
                                  input_ids, text_atts, use_checkpoint=True)
        self.assertTrue(torch.allclose(out_plain, out_ckpt, atol=1e-5))

        out_ckpt.sum().backward()
        self.assertIsNotNone(image_embeds.grad)
        self.assertTrue(torch.isfinite(image_embeds.grad).all())
