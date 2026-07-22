import unittest
import unittest.mock
import torch
import torch.nn.functional as F
from models.blip_pretrain import blip_pretrain
from distillation.online_teacher import OnlineTeacher


class TestForwardItmMix(unittest.TestCase):
    """CPU smoke: base student forward with/without itm_mix. Construction ~1-2 min."""

    @classmethod
    def setUpClass(cls):
        cls.model = blip_pretrain(image_size=224, vit="base", my_bert_size="base", queue_size=240)
        cls.model.eval()
        cls.image = torch.randn(2, 3, 224, 224)
        cls.caption = ["a green field", "a red car"]

    def _forward(self, **kw):
        return self.model(self.image, self.caption, alpha=0.0, update_train_state=False, **kw)

    def test_baseline_itm_none(self):
        loss_ita, loss_itm, loss_lm, kd_itc, kd_lm = self._forward()
        self.assertTrue(torch.isfinite(loss_itm))
        self.assertTrue(loss_itm.requires_grad)

    def test_student_neg_w0_is_plain_ce(self):
        # W=0 must reduce ITM loss to exactly F.cross_entropy(vl_output, itm_labels).
        # Spy on itm_head to grab the real student logits, rebuild itm_labels the
        # same way forward() does, and compare against plain CE.
        captured = {}
        real_forward = self.model.itm_head.forward

        def spy(x):
            out = real_forward(x)
            captured['vl_output'] = out.detach().clone()
            return out

        itm_mix = {'neg_source': 'student', 'soft_weight': 0.0, 'temp': 1.0, 'sel_scale': 14.29}
        with unittest.mock.patch.object(self.model.itm_head, 'forward', side_effect=spy):
            loss_itm = self._forward(itm_mix=itm_mix)[1]
        bs = self.image.size(0)
        vl_output = captured['vl_output']
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long),
                                torch.zeros(2 * bs, dtype=torch.long)], dim=0)
        self.assertTrue(torch.allclose(loss_itm, F.cross_entropy(vl_output, itm_labels), atol=1e-5))

    def test_student_neg_w_positive_with_teacher(self):
        teacher = OnlineTeacher(checkpoint="", image_size=224, vit="base",
                                bert="base", queue_size=240, keep=("itm",))
        itm_mix = {'neg_source': 'student', 'soft_weight': 0.4, 'temp': 1.0, 'sel_scale': 14.29}
        loss_itm = self._forward(online_teacher=teacher, itm_mix=itm_mix)[1]
        self.assertTrue(torch.isfinite(loss_itm))
        self.assertTrue(loss_itm.requires_grad)

    def test_teacher_neg_requires_feats(self):
        itm_mix = {'neg_source': 'teacher', 'soft_weight': 0.0, 'temp': 1.0, 'sel_scale': 63.9}
        with self.assertRaises(AssertionError):
            self._forward(itm_mix=itm_mix)   # teacher feats not provided

    def test_teacher_neg_source_selects_successfully(self):
        # Exercise the neg_source='teacher' selection path (t_img @ t_txt.t() * sel_scale
        # + fill_diagonal_) end-to-end. soft_weight=0.0 keeps it fast (plain CE, no
        # teacher itm_head forward), while still requiring real teacher feats.
        teacher = OnlineTeacher(checkpoint="", image_size=224, vit="base",
                                bert="base", queue_size=240, keep=("itc",))
        teacher_img_feat, teacher_text_feat = teacher.itc_feats(self.image, self.caption)
        itm_mix = {'neg_source': 'teacher', 'soft_weight': 0.0, 'temp': 1.0,
                   'sel_scale': teacher.teacher_scale}
        loss_itm = self._forward(itm_mix=itm_mix,
                                 teacher_img_feat=teacher_img_feat,
                                 teacher_text_feat=teacher_text_feat)[1]
        self.assertTrue(torch.isfinite(loss_itm))

    def test_teacher_soft_w1_reduces_to_teacher_ce(self):
        """With soft_weight=1.0, itm_target_mix_loss's target collapses to pure
        teacher_soft (the onehot component's weight becomes 1-W=0), so loss_itm
        must equal -(teacher_soft * log_softmax(vl_output)).sum(1).mean(). Stubs
        itm_soft to a fixed target and spies itm_head to capture vl_output, then
        hand-computes that loss. This guards the W>0 wiring: that forward feeds
        teacher_soft and vl_output row-aligned into itm_target_mix_loss and that
        the (1-W) coefficient is applied. NOTE: this does NOT guard the
        forward-vs-itm_soft 3B block-order sync — itm_soft is mocked here, so
        expected is computed from the captured vl_output and would match
        regardless of forward's block order. That contract is guarded instead by
        test_itm_block_order_contract below."""
        teacher = OnlineTeacher(checkpoint="", image_size=224, vit="base",
                                bert="base", queue_size=240, keep=("itm",))
        bs = self.image.size(0)
        fake_teacher_soft = torch.cat([
            torch.tensor([[0.0, 1.0]]).expand(bs, 2),
            torch.tensor([[1.0, 0.0]]).expand(bs, 2),
            torch.tensor([[0.9, 0.1]]).expand(bs, 2),
        ], dim=0)
        captured = {}
        real_forward = self.model.itm_head.forward

        def spy(x):
            out = real_forward(x)
            captured['vl_output'] = out.detach().clone()
            return out

        with unittest.mock.patch.object(teacher, 'itm_soft', return_value=fake_teacher_soft), \
             unittest.mock.patch.object(self.model.itm_head, 'forward', side_effect=spy):
            itm_mix = {'neg_source': 'student', 'soft_weight': 1.0, 'temp': 1.0, 'sel_scale': 14.29}
            loss_itm = self._forward(online_teacher=teacher, itm_mix=itm_mix)[1]
        vl_output = captured['vl_output']
        expected = -(fake_teacher_soft * F.log_softmax(vl_output, dim=1)).sum(1).mean()
        self.assertTrue(torch.allclose(loss_itm, expected, atol=1e-5))

    def test_itm_block_order_contract(self):
        """Regression guard for the row-order contract that forward() and
        OnlineTeacher.itm_soft each hardcode INDEPENDENTLY: the merged 3B ITM
        batch is [pos B | neg-image B | neg-text B]. If forward's block order
        ever drifts from this (e.g. the neg-image/neg-text blocks get swapped),
        teacher_soft[i] (which itm_soft builds assuming this order) would stop
        describing the same triplet as vl_output[i] — a silent training-signal
        corruption with no isfinite/label symptom.

        A loss-value comparison cannot catch this: any expected loss built from
        the CAPTURED vl_output inherits whatever order forward actually used, so
        it always matches. This instead inspects the real tensors forward
        assembles for its ITM text_encoder pass and asserts each block's
        identity structurally:
          - neg-image block keeps the pos TEXT, substitutes the IMAGE;
          - neg-text  block keeps the pos IMAGE, substitutes the TEXT.
        With bs=2 the sampled negatives are forced (the sole off-diagonal entry
        per row), so these assertions are deterministic. Verified to fail if the
        neg-image/neg-text blocks in forward() are swapped."""
        captured = {}
        real_forward = self.model.text_encoder.forward

        def spy(*args, **kwargs):
            # forward() calls text_encoder twice: the text-only ITC pass (no
            # cross-attention) and the ITM 3B pass. Capture only the latter,
            # identified by encoder_hidden_states (image embeddings) being present.
            if kwargs.get('encoder_hidden_states') is not None and 'text_ids_all' not in captured:
                captured['text_ids_all'] = args[0].detach().clone()
                captured['image_embeds_all'] = kwargs['encoder_hidden_states'].detach().clone()
            return real_forward(*args, **kwargs)

        with unittest.mock.patch.object(self.model.text_encoder, 'forward', side_effect=spy):
            self._forward()

        bs = self.image.size(0)
        tid = captured['text_ids_all']       # [3B, L]
        img = captured['image_embeds_all']   # [3B, P, D]
        self.assertEqual(tid.size(0), 3 * bs)
        pos_t, negimg_t, negtxt_t = tid[:bs], tid[bs:2 * bs], tid[2 * bs:3 * bs]
        pos_i, negimg_i, negtxt_i = img[:bs], img[bs:2 * bs], img[2 * bs:3 * bs]
        # Block 1 == neg-image: same text as pos, different (substituted) image.
        self.assertTrue(torch.equal(negimg_t, pos_t))
        self.assertFalse(torch.equal(negimg_i, pos_i))
        # Block 2 == neg-text: same image as pos, different (substituted) text.
        self.assertTrue(torch.equal(negtxt_i, pos_i))
        self.assertFalse(torch.equal(negtxt_t, pos_t))


if __name__ == "__main__":
    unittest.main()
