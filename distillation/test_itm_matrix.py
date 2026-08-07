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
