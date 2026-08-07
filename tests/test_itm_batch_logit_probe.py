import csv as _csv
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

REPO = "/home/minwoo/Distillation_Project"
sys.path.insert(0, os.path.join(REPO, "critical_bugfix/2026-07-31_itm_batch_logit_raw"))

import probe_core as pc


class NegWeightsTest(unittest.TestCase):
    def test_diagonal_is_zeroed(self):
        sim = torch.randn(5, 5)
        w = pc.neg_weights(sim)
        self.assertTrue(torch.all(torch.diagonal(w) == 0))

    def test_offdiagonal_is_strictly_positive(self):
        # softmax + 1e-4 이므로 비대각은 절대 0이 되지 않는다 (multinomial 이 항상 성공)
        sim = torch.full((4, 4), -50.0)
        w = pc.neg_weights(sim)
        off = w[~torch.eye(4, dtype=torch.bool)]
        self.assertTrue(torch.all(off > 0))

    def test_does_not_mutate_input(self):
        sim = torch.randn(3, 3)
        before = sim.clone()
        pc.neg_weights(sim)
        self.assertTrue(torch.equal(sim, before))


class SampleNegIdxTest(unittest.TestCase):
    def test_never_samples_self(self):
        torch.manual_seed(0)
        sim = torch.randn(6, 6)
        w = pc.neg_weights(sim)
        for _ in range(20):
            idx = pc.sample_neg_idx(w)
            self.assertEqual(idx.shape, (6,))
            self.assertTrue(torch.all(idx != torch.arange(6)))

    def test_seed_reproducible(self):
        sim = torch.randn(8, 8)
        w = pc.neg_weights(sim)
        torch.manual_seed(123)
        a = pc.sample_neg_idx(w)
        torch.manual_seed(123)
        b = pc.sample_neg_idx(w)
        self.assertTrue(torch.equal(a, b))


class SplitBlocksTest(unittest.TestCase):
    def test_splits_into_three_equal_blocks_in_order(self):
        logits = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(12, 2)
        blocks = pc.split_blocks(logits)
        self.assertEqual(set(blocks), {"pos", "neg_img", "neg_txt"})
        self.assertTrue(torch.equal(blocks["pos"], logits[0:4]))
        self.assertTrue(torch.equal(blocks["neg_img"], logits[4:8]))
        self.assertTrue(torch.equal(blocks["neg_txt"], logits[8:12]))

    def test_rejects_non_multiple_of_three(self):
        with self.assertRaises(ValueError):
            pc.split_blocks(torch.zeros(11, 2))


class MakeFrameTest(unittest.TestCase):
    def test_derives_m_and_gap(self):
        logits = np.array([[1.0, 5.0], [2.0, -2.0]], dtype=np.float32)
        fr = pc.make_frame([0, 0], [0, 1], logits)
        np.testing.assert_allclose(fr["z1_nomatch"], [1.0, 2.0])
        np.testing.assert_allclose(fr["z2_match"], [5.0, -2.0])
        np.testing.assert_allclose(fr["gap"], [4.0, -4.0])   # z2 - z1
        np.testing.assert_allclose(fr["m"], [3.0, 0.0])      # (z1 + z2) / 2

    def test_keeps_index_columns(self):
        logits = np.zeros((3, 2), dtype=np.float32)
        fr = pc.make_frame([7, 7, 8], [0, 1, 0], logits)
        np.testing.assert_array_equal(fr["batch"], [7, 7, 8])
        np.testing.assert_array_equal(fr["i"], [0, 1, 0])


class ColumnStatsTest(unittest.TestCase):
    def test_uses_sample_variance_ddof1(self):
        logits = np.array([[0.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        fr = pc.make_frame([0, 0], [0, 1], logits)
        st = pc.column_stats(fr)
        # gap = [0, 2] → mean 1, ddof=1 분산 = 2.0
        self.assertAlmostEqual(st["gap"]["mean"], 1.0)
        self.assertAlmostEqual(st["gap"]["var"], 2.0)
        self.assertAlmostEqual(st["gap"]["std"], np.sqrt(2.0), places=6)

    def test_covers_all_four_columns(self):
        fr = pc.make_frame([0], [0], np.zeros((1, 2), dtype=np.float32))
        self.assertEqual(set(pc.column_stats(fr)), set(pc.COLUMNS))


class WriteBlockCsvTest(unittest.TestCase):
    def test_writes_rows_then_three_summary_rows(self):
        logits = np.array([[1.0, 3.0], [2.0, 6.0]], dtype=np.float32)
        fr = pc.make_frame([0, 0], [0, 1], logits)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "b.csv")
            pc.write_block_csv(p, fr)
            with open(p) as f:
                rows = list(_csv.reader(f))
        self.assertEqual(rows[0], ["batch", "i", "z1_nomatch", "z2_match", "m", "gap"])
        self.assertEqual(len(rows), 1 + 2 + 3)          # header + data + mean/var/std
        self.assertEqual(rows[-3][0], "mean")
        self.assertEqual(rows[-2][0], "var")
        self.assertEqual(rows[-1][0], "std")
        self.assertAlmostEqual(float(rows[-3][5]), 3.0)  # gap mean = (2+4)/2


class SummarizeBlocksTest(unittest.TestCase):
    def _frames(self, pos_gap, negimg_gap, negtxt_gap):
        """gap 만 지정하고 m=0 이 되도록 z1=-gap/2, z2=+gap/2 로 만든다."""
        fr = {}
        for name, gaps in (("pos", pos_gap), ("neg_img", negimg_gap), ("neg_txt", negtxt_gap)):
            g = np.asarray(gaps, dtype=np.float64)
            logits = np.stack([-g / 2, g / 2], axis=1)
            fr[name] = pc.make_frame([0] * len(g), list(range(len(g))), logits)
        return fr

    def test_delta_is_pos_mean_minus_pooled_neg_mean(self):
        frames = self._frames([4.0, 6.0], [1.0, 1.0], [3.0, 3.0])
        s = pc.summarize_blocks(frames)
        # pos mean = 5.0, pooled neg mean = (1+1+3+3)/4 = 2.0
        self.assertAlmostEqual(s["delta"], 3.0)

    def test_p_pos_from_delta_matches_closed_form(self):
        frames = self._frames([2.0, 2.0], [0.0, 0.0], [0.0, 0.0])
        s = pc.summarize_blocks(frames)
        expected = np.exp(2.0) / (np.exp(2.0) + 2)
        self.assertAlmostEqual(s["p_pos_merged_from_delta"], expected, places=6)

    def test_p_pos_measured_uses_actual_exponentials(self):
        # 모든 gap 이 동일하면 pos 는 정확히 1/3
        frames = self._frames([1.0, 1.0], [1.0, 1.0], [1.0, 1.0])
        s = pc.summarize_blocks(frames)
        self.assertAlmostEqual(s["p_pos_merged_measured"], 1.0 / 3.0, places=6)

    def test_measured_is_overflow_safe(self):
        frames = self._frames([900.0, 900.0], [0.0, 0.0], [0.0, 0.0])
        s = pc.summarize_blocks(frames)
        self.assertTrue(np.isfinite(s["p_pos_merged_measured"]))
        self.assertAlmostEqual(s["p_pos_merged_measured"], 1.0, places=6)

    def test_delta_pools_negatives_rather_than_averaging_block_means(self):
        # 블록 길이를 다르게 해 pooled 와 per-block-average 가 갈리게 한다.
        #   pooled  = (0+0+0+4)/4 = 1.0  → delta = 5.0 - 1.0 = 4.0   ← 올바른 정의
        #   per-blk = (0.0 + 4.0)/2 = 2.0 → delta = 5.0 - 2.0 = 3.0   ← 잘못된 정의
        frames = self._frames([5.0, 5.0], [0.0, 0.0, 0.0], [4.0])
        s = pc.summarize_blocks(frames)
        self.assertAlmostEqual(s["delta"], 4.0)

    def test_per_block_reports_sigma_only(self):
        frames = self._frames([4.0, 6.0], [1.0, 1.0], [3.0, 3.0])
        s = pc.summarize_blocks(frames)
        for name in pc.BLOCK_NAMES:
            self.assertIn("sigma_m", s[name])
            self.assertIn("sigma_gap", s[name])
            self.assertIn("stats", s[name])
            # 무차원 비율 지표는 의도적으로 내지 않는다 (스펙 §3.1)
            self.assertNotIn("var_m_over_var_z2", s[name])
        # 이 합성 데이터는 m ≡ 0 이므로 sigma_m = 0
        self.assertAlmostEqual(s["pos"]["sigma_m"], 0.0)


sys.path.insert(0, REPO)
import itm_batch_logit_probe as probe  # noqa: E402  (critical_bugfix 경로에서 임포트)


class _StubTokenizerOut(dict):
    """tokenizer(...) 결과 흉내 — .input_ids / .attention_mask / .to(device)"""
    def __init__(self, input_ids, attention_mask):
        super().__init__()
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def to(self, device):
        return self


class _StubTokenizer:
    enc_token_id = 999

    def __call__(self, caption, **kw):
        # caption[i] == "s<i>" → input_ids[i] = [0, i, 0, 0]. 두 번째 토큰이 샘플 식별자.
        ids = torch.tensor([[0, int(c[1:]), 0, 0] for c in caption], dtype=torch.long)
        return _StubTokenizerOut(ids, torch.ones_like(ids))


class _StubTextEncoder:
    """CLS 자리에 [텍스트 식별자, 이미지 식별자] 를 실어 반환.

    mode="text"(=ITC 경로) 호출이 받은 0번 토큰을 기록한다 — 그 자리에 [ENC] 가 오면
    text_feat 이 학습과 달라져 negative 샘플링 분포가 어긋나므로, 테스트가 이를 감시한다."""
    def __init__(self):
        self.text_mode_first_tokens = []

    def __call__(self, input_ids, attention_mask=None, encoder_hidden_states=None,
                 encoder_attention_mask=None, return_dict=True, mode=None):
        n = input_ids.size(0)
        if mode == "text":
            self.text_mode_first_tokens.append(input_ids[:, 0].clone())
            h = torch.zeros(n, 4, 8)
            h[:, 0, 0] = input_ids[:, 1].float()
            return type("O", (), {"last_hidden_state": h})()
        text_id = input_ids[:, 1].float()
        image_id = encoder_hidden_states[:, 0, 0]
        h = torch.zeros(n, 4, 8)
        h[:, 0, 0] = text_id
        h[:, 0, 1] = image_id
        return type("O", (), {"last_hidden_state": h})()


class _StubModel(torch.nn.Module):
    """visual_encoder 는 이미지 식별자를 채널 0 에 그대로 싣는다.
    itm_head 는 [[1,0,...],[0,1,...]] 이므로 z1 = 텍스트 식별자, z2 = 이미지 식별자."""
    def __init__(self, bs):
        super().__init__()
        self.tokenizer = _StubTokenizer()
        self.text_encoder = _StubTextEncoder()
        self.text_encoder_m = _StubTextEncoder()
        self.itm_head = torch.nn.Linear(8, 2, bias=True)
        with torch.no_grad():
            self.itm_head.weight.zero_()
            self.itm_head.weight[0, 0] = 1.0
            self.itm_head.weight[1, 1] = 1.0
            self.itm_head.bias.zero_()
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.0))
        self.vision_proj = torch.nn.Identity()
        self.vision_proj_m = torch.nn.Identity()
        self.text_proj = torch.nn.Identity()
        self.text_proj_m = torch.nn.Identity()
        self._bs = bs

    def _embed(self, image):
        n = image.size(0)
        h = torch.zeros(n, 5, 8)
        h[:, 0, 0] = torch.arange(n, dtype=torch.float32)  # 이미지 식별자 = 배치 위치
        return h

    def visual_encoder(self, image):
        return self._embed(image)

    def visual_encoder_m(self, image):
        return self._embed(image)


class ItmForwardAssemblyTest(unittest.TestCase):
    def setUp(self):
        torch.set_grad_enabled(False)
        self.bs = 4
        self.model = _StubModel(self.bs)
        self.image = torch.zeros(self.bs, 3, 8, 8)
        self.caption = [f"s{i}" for i in range(self.bs)]
        self.device = torch.device("cpu")

    def test_block_composition_matches_blip_pretrain(self):
        torch.manual_seed(0)
        logits, neg_img, neg_txt = probe.student_itm_logits(
            self.model, self.image, self.caption, self.device, amp=False)
        self.assertEqual(tuple(logits.shape), (3 * self.bs, 2))
        z_text, z_image = logits[:, 0], logits[:, 1]
        ar = torch.arange(self.bs, dtype=torch.float32)

        # ① pos: (img_i, txt_i)
        self.assertTrue(torch.equal(z_text[0:self.bs], ar))
        self.assertTrue(torch.equal(z_image[0:self.bs], ar))
        # ② neg_img: 텍스트는 원본, 이미지는 negative
        self.assertTrue(torch.equal(z_text[self.bs:2 * self.bs], ar))
        self.assertTrue(torch.equal(z_image[self.bs:2 * self.bs], neg_img.float()))
        # ③ neg_txt: 이미지는 원본, 텍스트는 negative
        self.assertTrue(torch.equal(z_text[2 * self.bs:3 * self.bs], neg_txt.float()))
        self.assertTrue(torch.equal(z_image[2 * self.bs:3 * self.bs], ar))

    def test_negatives_are_never_self(self):
        torch.manual_seed(1)
        _, neg_img, neg_txt = probe.student_itm_logits(
            self.model, self.image, self.caption, self.device, amp=False)
        ar = torch.arange(self.bs)
        self.assertTrue(torch.all(neg_img != ar))
        self.assertTrue(torch.all(neg_txt != ar))

    def test_teacher_reuses_given_negatives(self):
        neg_img = torch.tensor([1, 2, 3, 0])
        neg_txt = torch.tensor([3, 0, 1, 2])
        logits = probe.teacher_itm_logits(
            self.model, self.image, self.caption, neg_img, neg_txt, self.device, amp=False)
        z_text, z_image = logits[:, 0], logits[:, 1]
        self.assertTrue(torch.equal(z_image[self.bs:2 * self.bs], neg_img.float()))
        self.assertTrue(torch.equal(z_text[2 * self.bs:3 * self.bs], neg_txt.float()))

    def test_head_is_applied_in_fp32(self):
        torch.manual_seed(0)
        logits, _, _ = probe.student_itm_logits(
            self.model, self.image, self.caption, self.device, amp=False)
        self.assertEqual(logits.dtype, torch.float32)

    def test_itc_forward_uses_raw_cls_not_enc_token(self):
        """ITC(mode='text') 는 원본 [CLS] 시퀀스를 받아야 한다 — blip_pretrain.py:388, 404.
        [ENC] 치환은 ITM 경로(:436-437)에서만 일어난다. 여기에 enc_ids 를 넣으면
        text_feat/text_feat_m 이 통째로 달라져 negative 샘플링이 학습과 어긋난다."""
        torch.manual_seed(0)
        probe.student_itm_logits(self.model, self.image, self.caption, self.device, amp=False)
        for enc in (self.model.text_encoder, self.model.text_encoder_m):
            self.assertTrue(enc.text_mode_first_tokens, "mode='text' forward 가 호출되지 않았다")
            for toks in enc.text_mode_first_tokens:
                self.assertTrue(torch.all(toks != _StubTokenizer.enc_token_id))


class CollectFramesTest(unittest.TestCase):
    def test_concatenates_per_block_and_truncates(self):
        # 배치 2개 × B=2 → 블록당 4행, n_samples=3 이면 3행
        b = 2
        l0 = torch.arange(3 * b * 2, dtype=torch.float32).reshape(3 * b, 2)
        l1 = l0 + 100.0
        frames = probe.collect_frames([l0, l1], [0, 1], n_samples=3)
        self.assertEqual(set(frames), {"pos", "neg_img", "neg_txt"})
        for name in ("pos", "neg_img", "neg_txt"):
            self.assertEqual(len(frames[name]["gap"]), 3)
        # pos 블록: 배치0의 2행 + 배치1의 첫 행
        np.testing.assert_array_equal(frames["pos"]["batch"], [0, 0, 1])
        np.testing.assert_array_equal(frames["pos"]["i"], [0, 1, 0])
        np.testing.assert_allclose(frames["pos"]["z1_nomatch"], [0.0, 2.0, 100.0])

    def test_keeps_all_rows_when_n_samples_exceeds_available(self):
        b = 2
        l0 = torch.zeros(3 * b, 2)
        frames = probe.collect_frames([l0], [0], n_samples=999)
        self.assertEqual(len(frames["pos"]["gap"]), 2)


class EmitTest(unittest.TestCase):
    def test_writes_three_csvs_and_npz_and_returns_summary(self):
        b = 3
        logits = torch.randn(3 * b, 2)
        frames = probe.collect_frames([logits], [0], n_samples=b)
        with tempfile.TemporaryDirectory() as d:
            summary = probe.emit("student", frames, d)
            for name in ("pos", "neg_img", "neg_txt"):
                self.assertTrue(os.path.exists(os.path.join(d, "csv", f"student_{name}.csv")))
            self.assertTrue(os.path.exists(os.path.join(d, "npz", "student.npz")))
        self.assertIn("delta", summary)
        self.assertIn("pos", summary)


if __name__ == "__main__":
    unittest.main()
