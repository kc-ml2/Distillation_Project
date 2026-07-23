import unittest
import unittest.mock
import torch

from distillation.online_teacher import OnlineTeacher


class TestOnlineTeacher(unittest.TestCase):
    """CPU smoke. Constructs a small BLIP (base) as a stand-in teacher with no
    checkpoint (pretrained backbones only) and checks the ITC feature contract.
    Construction loads cached deit-base + bert-base weights; may take ~1-2 min."""

    @classmethod
    def setUpClass(cls):
        cls.teacher = OnlineTeacher(checkpoint="", image_size=224, vit="base", bert="base", queue_size=240)

    def test_freed_unused_submodules(self):
        m = self.teacher.model
        for attr in ("visual_encoder_m", "text_encoder_m", "vision_proj_m", "text_proj_m", "text_decoder", "itm_head"):
            self.assertIsNone(getattr(m, attr), f"{attr} should be freed")

    def test_params_frozen(self):
        self.assertTrue(all(not p.requires_grad for p in self.teacher.model.parameters()))

    def test_itc_feats_contract(self):
        image = torch.randn(2, 3, 224, 224)
        caption = ["a green field", "a red car"]
        img_feat, txt_feat = self.teacher.itc_feats(image, caption)
        self.assertEqual(tuple(img_feat.shape), (2, 256))
        self.assertEqual(tuple(txt_feat.shape), (2, 256))
        self.assertFalse(img_feat.requires_grad)
        self.assertFalse(txt_feat.requires_grad)
        # L2-normalized rows (bf16 tolerance)
        norms = img_feat.float().norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=5e-2))
        txt_norms = txt_feat.float().norm(dim=-1)
        self.assertTrue(torch.allclose(txt_norms, torch.ones_like(txt_norms), atol=5e-2))

    def test_lm_logits_raises_without_lm_keep(self):
        with self.assertRaises(RuntimeError):
            self.teacher.lm_logits(torch.randn(1, 3, 224, 224), ["a cat"])

    def test_itm_soft_raises_without_itm_keep(self):
        image = torch.randn(1, 3, 224, 224)
        ids = torch.zeros(1, 30, dtype=torch.long)
        with self.assertRaises(RuntimeError):
            self.teacher.itm_soft(image, ids, ids, [0], [0], temp=1.0)


class TestOnlineTeacherKeepLm(unittest.TestCase):
    """keep=('lm',): visual_encoder + text_decoder만 생존해야 한다."""

    @classmethod
    def setUpClass(cls):
        cls.teacher = OnlineTeacher(checkpoint="", image_size=224, vit="base",
                                    bert="base", queue_size=240, keep=("lm",))

    def test_kept_and_freed(self):
        m = self.teacher.model
        self.assertIsNotNone(m.visual_encoder)
        self.assertIsNotNone(m.text_decoder)
        for attr in ("text_encoder", "vision_proj", "text_proj",
                     "visual_encoder_m", "text_encoder_m", "vision_proj_m",
                     "text_proj_m", "itm_head"):
            self.assertIsNone(getattr(m, attr), f"{attr} should be freed")
        for buf in ("image_queue", "text_queue", "queue_ptr"):
            self.assertIsNone(getattr(m, buf), f"{buf} should be freed")

    def test_itc_feats_raises_without_itc_keep(self):
        with self.assertRaises(RuntimeError):
            self.teacher.itc_feats(torch.randn(1, 3, 224, 224), ["a cat"])

    def test_lm_logits_contract(self):
        image = torch.randn(2, 3, 224, 224)
        caption = ["a green field", "a red car"]
        logits, dec_ids = self.teacher.lm_logits(image, caption)
        vocab = len(self.teacher.tokenizer)          # 30524
        self.assertEqual(tuple(logits.shape), (2, 30, vocab))
        self.assertEqual(tuple(dec_ids.shape), (2, 30))
        self.assertFalse(logits.requires_grad)
        self.assertTrue(torch.isfinite(logits.float()).all())
        # 첫 토큰은 BOS로 치환되어야 함 (학생의 decoder_input_ids 규칙과 동일)
        self.assertTrue((dec_ids[:, 0] == self.teacher.tokenizer.bos_token_id).all())


class TestOnlineTeacherKeepBoth(unittest.TestCase):
    """keep=('itc','lm'): 두 경로의 합집합 생존, 학습 전용 장치는 여전히 해제."""

    @classmethod
    def setUpClass(cls):
        cls.teacher = OnlineTeacher(checkpoint="", image_size=224, vit="base",
                                    bert="base", queue_size=240, keep=("itc", "lm"))

    def test_union_kept(self):
        m = self.teacher.model
        for attr in ("visual_encoder", "text_encoder", "vision_proj",
                     "text_proj", "text_decoder"):
            self.assertIsNotNone(getattr(m, attr), f"{attr} should be kept")
        for attr in ("visual_encoder_m", "text_encoder_m", "vision_proj_m",
                     "text_proj_m", "itm_head"):
            self.assertIsNone(getattr(m, attr), f"{attr} should be freed")

    def test_unknown_keep_raises(self):
        with self.assertRaises(ValueError):
            OnlineTeacher(checkpoint="", image_size=224, vit="base",
                          bert="base", queue_size=240, keep=("bogus",))


class TestOnlineTeacherKeepItm(unittest.TestCase):
    """keep=('itm',): visual_encoder + text_encoder + itm_head 생존, 나머지 해제."""

    @classmethod
    def setUpClass(cls):
        cls.teacher = OnlineTeacher(checkpoint="", image_size=224, vit="base",
                                    bert="base", queue_size=240, keep=("itm",))

    def test_kept_and_freed(self):
        m = self.teacher.model
        for attr in ("visual_encoder", "text_encoder", "itm_head"):
            self.assertIsNotNone(getattr(m, attr), f"{attr} should be kept")
        for attr in ("vision_proj", "text_proj", "visual_encoder_m",
                     "text_encoder_m", "vision_proj_m", "text_proj_m", "text_decoder"):
            self.assertIsNone(getattr(m, attr), f"{attr} should be freed")

    def test_teacher_scale_default(self):
        # no checkpoint -> default temp 0.07 -> scale ~14.29
        self.assertAlmostEqual(self.teacher.teacher_temp, 0.07, places=5)
        self.assertAlmostEqual(self.teacher.teacher_scale, 1.0 / 0.07, places=3)

    def test_itm_soft_contract(self):
        B = 3
        image = torch.randn(B, 3, 224, 224)
        text = self.teacher.tokenizer(["a green field", "a red car", "a blue sky"],
                                      padding="max_length", truncation=True,
                                      max_length=30, return_tensors="pt")
        enc_ids = text.input_ids.clone()
        enc_ids[:, 0] = self.teacher.tokenizer.enc_token_id
        out = self.teacher.itm_soft(image, enc_ids, text.attention_mask,
                                    [1, 2, 0], [2, 0, 1], temp=1.0)
        self.assertEqual(tuple(out.shape), (3 * B, 2))
        self.assertFalse(out.requires_grad)
        self.assertTrue(torch.allclose(out.float().sum(1), torch.ones(3 * B), atol=1e-3))
        self.assertTrue(torch.isfinite(out.float()).all())

    def test_itm_soft_block_order_contract(self):
        """Regression guard, symmetric to models/test_blip_pretrain_itm.py's
        test_itm_block_order_contract: itm_soft hardcodes block order
        [pos B | neg-image B | neg-text B] independently from forward()'s
        identical hardcoded order — nothing else keeps them in sync. Spies on
        text_encoder to inspect the actual assembled 3B input tensors and
        asserts the block-order contract structurally: block1 (neg-image) must
        keep the ORIGINAL text but swap in a DIFFERENT image; block2 (neg-text)
        must swap in a DIFFERENT text but keep the ORIGINAL image."""
        B = 2
        image = torch.randn(B, 3, 224, 224)
        text = self.teacher.tokenizer(["a green field", "a red car"],
                                      padding="max_length", truncation=True,
                                      max_length=30, return_tensors="pt")
        enc_ids = text.input_ids.clone()
        enc_ids[:, 0] = self.teacher.tokenizer.enc_token_id
        neg_idx_img = [1, 0]   # explicit, no randomness needed
        neg_idx_txt = [1, 0]

        captured = {}
        real_forward = self.teacher.model.text_encoder.forward
        def spy(*args, **kwargs):
            captured['text_ids'] = args[0] if args else kwargs['input_ids']
            captured['image_embeds'] = kwargs['encoder_hidden_states']
            return real_forward(*args, **kwargs)
        with unittest.mock.patch.object(self.teacher.model.text_encoder, 'forward', side_effect=spy):
            self.teacher.itm_soft(image, enc_ids, text.attention_mask,
                                  neg_idx_img, neg_idx_txt, temp=1.0)

        text_ids = captured['text_ids']
        image_embeds = captured['image_embeds']
        pos_t, negimg_t, negtxt_t = text_ids[:B], text_ids[B:2*B], text_ids[2*B:3*B]
        pos_i, negimg_i, negtxt_i = image_embeds[:B], image_embeds[B:2*B], image_embeds[2*B:3*B]

        self.assertTrue(torch.equal(negimg_t, pos_t))          # neg-image: text unswapped
        self.assertFalse(torch.equal(negimg_i, pos_i))         # neg-image: image swapped
        self.assertFalse(torch.equal(negtxt_t, pos_t))         # neg-text: text swapped
        self.assertTrue(torch.equal(negtxt_i, pos_i))          # neg-text: image unswapped


class TestOnlineTeacherKeepAllThree(unittest.TestCase):
    """keep=('itc','itm','lm'): 3-way 합집합 생존 — 3개 증류 메커니즘이 동시에 켜지는
    병합 config(pretrain_itc_itm_lm_targetmix_smoke.yaml)가 요구하는 조합."""

    @classmethod
    def setUpClass(cls):
        cls.teacher = OnlineTeacher(checkpoint="", image_size=224, vit="base",
                                    bert="base", queue_size=240, keep=("itc", "itm", "lm"))

    def test_union_kept(self):
        m = self.teacher.model
        for attr in ("visual_encoder", "text_encoder", "vision_proj",
                     "text_proj", "text_decoder", "itm_head"):
            self.assertIsNotNone(getattr(m, attr), f"{attr} should be kept")
        for attr in ("visual_encoder_m", "text_encoder_m", "vision_proj_m", "text_proj_m"):
            self.assertIsNone(getattr(m, attr), f"{attr} should be freed")

    def test_all_three_entry_points_work(self):
        image = torch.randn(2, 3, 224, 224)
        caption = ["a green field", "a red car"]
        img_feat, txt_feat = self.teacher.itc_feats(image, caption)
        self.assertEqual(tuple(img_feat.shape), (2, 256))
        logits, dec_ids = self.teacher.lm_logits(image, caption)
        self.assertEqual(tuple(dec_ids.shape), (2, 30))
        enc_ids = txt_feat.new_zeros(2, 5, dtype=torch.long)  # dummy encoder-side ids
        atts = txt_feat.new_ones(2, 5, dtype=torch.long)
        soft = self.teacher.itm_soft(image, enc_ids, atts,
                                     neg_idx_img=[1, 0], neg_idx_txt=[1, 0], temp=1.0)
        self.assertEqual(tuple(soft.shape), (6, 2))


class TestOnlineTeacherLargeConstruction(unittest.TestCase):
    """실제 티처 아키텍처(vit='large')의 생성 회귀. timm 1.x에서 BLIP 원본의
    in21k load_custom_pretrained 경로가 깨지므로(DefaultCfg.get 부재), 티처는
    백본 사전 초기화를 건너뛰어야 한다 — 가중치는 어차피 BLIP 체크포인트가
    전량 덮는다. 느림: ViT-L 본체+모멘텀 생성 (~수 분, CPU)."""

    def test_construct_large_lm_teacher(self):
        teacher = OnlineTeacher(checkpoint="", image_size=224, vit="large",
                                bert="base", queue_size=240, keep=("lm",))
        m = teacher.model
        self.assertIsNotNone(m.visual_encoder)
        self.assertIsNotNone(m.text_decoder)
        self.assertIsNone(m.text_encoder)


if __name__ == "__main__":
    unittest.main()
