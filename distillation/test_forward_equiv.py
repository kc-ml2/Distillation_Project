"""Numerical-equivalence harness: pins BLIP_Pretrain.forward's 5 losses to golden
values so a downstream pure refactor can prove it changed nothing.

Runs on CPU in eval() (dropout off) with a deterministic seeded STUB teacher (no
real BLIP-large, no checkpoint) and update_train_state=False (no momentum/queue
mutation). The only student-side RNG is ITM neg-mining (torch.multinomial); it is
pinned by reseeding the global generator IMMEDIATELY before each forward, so runs
are bit-identical. A later refactor that reorders RNG-consuming ops in the student
will therefore make this test FAIL -- that is intended (it catches real changes).

Requires conda env kd_r4 (transformers 4.33.3); the base env cannot import the
model stack. Capture / re-capture goldens standalone (prints paste-ready dict):
    conda run -n kd_r4 python distillation/test_forward_equiv.py
Run the test:
    conda run -n kd_r4 python -m pytest distillation/test_forward_equiv.py -q
"""
import unittest

import torch
import torch.nn.functional as F

from models.blip_pretrain import blip_pretrain

# ---- fixed experiment knobs (changing ANY of these invalidates the golden) ----
B = 4
QUEUE_SIZE = 8            # multiple of B; queue never mutated (update_train_state=False)
ITM_TOPK = 4             # Phase-2 gathered ITM KD (k+1=5 candidates)
GAMMA = 0.5              # TTM teacher/momentum slot split
ALPHA = 0.4             # unused while TTM branch active; passed for signature
LM_TEMP = 2.0
ITM_TEMP = 0.05
ITM_DIR = "bidir"
IMAGE_SEED = 42
STUB_SEED = 7
FORWARD_SEED = 1234      # reseeded right before forward -> multinomial reproducible
CAPTIONS = ["a cat", "a dog", "two birds", "a red car"]
LOSS_KEYS = ("loss_ita", "loss_itm", "loss_lm", "loss_lm_kd", "loss_itm_kd")

# Captured on the UNMODIFIED forward at 215dc9a in kd_r4 (see __main__ capture path).
GOLDEN = {
    "loss_ita": 2.6987245082855225,
    "loss_itm": 0.6517201066017151,
    "loss_lm": 10.964111328125,
    "loss_lm_kd": 1.0696642398834229,
    "loss_itm_kd": 0.003918764181435108,
}
# Golden tolerance: run-to-run is bit-identical (test_run_to_run_stable, torch.equal).
# This atol/rtol only absorbs cross-process last-bit BLAS reduction-order noise; it is
# ~7 orders below any real logic change, so equivalence still catches genuine drift.
ATOL, RTOL = 1e-6, 1e-6


class StubTeacher:
    """Deterministic stand-in for OnlineTeacher. Each method returns fixed seeded
    tensors of the contract shapes (distillation/online_teacher.py + itm_matrix.py);
    the image_embeds= argument is accepted and IGNORED (the dedup no-op). Per-method
    fixed seed => every call returns the same tensor regardless of call order/count,
    so the teacher signal is a stable constant (test scaffolding, not code under test)."""

    def __init__(self, tokenizer, base_seed=STUB_SEED):
        self.tokenizer = tokenizer
        self.vocab = len(tokenizer)
        self.base_seed = base_seed

    def _randn(self, key, *shape):
        g = torch.Generator().manual_seed(self.base_seed + key)
        return torch.randn(*shape, generator=g)

    def encode_image(self, image):                             # [B, N, D] = student ViT out
        return self._randn(1, image.size(0), 197, 384)

    def itc_feats(self, image, caption, image_embeds=None):    # ([B,256],[B,256]) L2-norm
        img = F.normalize(self._randn(2, len(caption), 256), dim=-1)
        txt = F.normalize(self._randn(3, len(caption), 256), dim=-1)
        return img, txt

    def lm_logits(self, image, caption, image_embeds=None):    # ([B,L,V],[B,L]); ids per forward rule
        text = self.tokenizer(caption, padding="max_length", truncation=True,
                              max_length=30, return_tensors="pt")
        dec_ids = text.input_ids.clone()
        dec_ids[:, 0] = self.tokenizer.bos_token_id
        return self._randn(4, len(caption), 30, self.vocab), dec_ids

    def itm_matrix(self, image, caption, image_embeds=None):   # [B,B,2] (unused when itm_topk>0)
        B_ = len(caption)
        return self._randn(5, B_, B_, 2)

    def itm_matrix_gathered(self, image, caption, idx_i2t_full, idx_t2i_full, image_embeds=None):
        B_, M = idx_i2t_full.shape                             # M = k+1
        return self._randn(6, B_, M, 2), self._randn(7, B_, M, 2)


def build_model():
    torch.manual_seed(0)
    model = blip_pretrain(image_size=224, vit="small_reg", my_bert_size="minilm",
                          queue_size=QUEUE_SIZE, ttm_enabled=True, ttm_variant="queue",
                          ttm_temp=0.05, ttm_soft_weight=0.4, init_backbone_weights=False)
    model.eval()
    return model


def fixed_image():
    g = torch.Generator().manual_seed(IMAGE_SEED)
    return torch.randn(B, 3, 224, 224, generator=g)


def run_forward(model, image, stub):
    """One forward on the fixed batch with the stub teacher; returns {loss_name: tensor}.
    Exercises all three mechanisms' base + KD: ITC(+TTM), LM(+KD), ITM(+gathered KD)."""
    t_img_feat, t_txt_feat = stub.itc_feats(image, CAPTIONS)
    t_lm_logits, t_lm_ids = stub.lm_logits(image, CAPTIONS)
    t_embeds = stub.encode_image(image)
    torch.manual_seed(FORWARD_SEED)          # pin ITM neg-mining multinomial draws
    out = model(image, CAPTIONS, alpha=ALPHA, update_train_state=False,
                teacher_img_feat=t_img_feat, teacher_text_feat=t_txt_feat,
                teacher_lm_logits=t_lm_logits, teacher_lm_input_ids=t_lm_ids,
                lm_distill_temp=LM_TEMP, gamma=GAMMA,
                online_teacher=stub, teacher_image_embeds=t_embeds,
                itm_topk=ITM_TOPK, itm_distill_temp=ITM_TEMP,
                itm_distill_direction=ITM_DIR)
    return dict(zip(LOSS_KEYS, out))


class TestForwardEquiv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_model()
        cls.image = fixed_image()
        cls.stub = StubTeacher(cls.model.tokenizer)

    def test_all_five_losses_present_and_finite(self):
        """No mechanism path silently skipped: base + KD losses all non-None + finite."""
        out = run_forward(self.model, self.image, self.stub)
        for k in LOSS_KEYS:
            self.assertIsNotNone(out[k], f"{k} is None -- a mechanism path was skipped")
            self.assertTrue(torch.isfinite(out[k]).all(), f"{k} not finite")

    def test_run_to_run_stable(self):
        """Bit-identical across reruns (same seed reset) -> multinomial is the only RNG
        and it is pinned. This is the precondition for a trustworthy golden."""
        a = run_forward(self.model, self.image, self.stub)
        b = run_forward(self.model, self.image, self.stub)
        for k in LOSS_KEYS:
            self.assertTrue(torch.equal(a[k], b[k]),
                            f"{k} not bit-identical across reruns: {a[k].item()!r} vs {b[k].item()!r}")

    def test_matches_golden(self):
        out = run_forward(self.model, self.image, self.stub)
        for k in LOSS_KEYS:
            actual, gold = out[k].item(), GOLDEN[k]
            self.assertLessEqual(abs(actual - gold), ATOL + RTOL * abs(gold),
                                 f"{k}: {actual!r} != golden {gold!r}")


if __name__ == "__main__":
    # Capture path: prints a paste-ready GOLDEN dict + run-to-run stability flags.
    _model = build_model()
    _image = fixed_image()
    _stub = StubTeacher(_model.tokenizer)
    r1 = run_forward(_model, _image, _stub)
    r2 = run_forward(_model, _image, _stub)
    print("=== forward 5-loss golden capture (paste into GOLDEN) ===")
    for _k in LOSS_KEYS:
        print(f'    "{_k}": {r1[_k].item()!r},   # stable_rerun={torch.equal(r1[_k], r2[_k])}')
