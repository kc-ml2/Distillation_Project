"""Numerical-equivalence harness: pins BLIP_Pretrain.forward's 5 losses to golden
values so a downstream pure refactor can prove it changed nothing.

Two goldens:
  - TestForwardEquiv: update_train_state=False (eval path). No momentum update,
    no queue enqueue, logit_scale clamped by-value only.
  - TestForwardEquivTrainState: update_train_state=True (train path). Covers
    _momentum_update, queue/teacher-queue enqueue, and the in-place logit_scale
    clamp_ -- exactly what the refactor moves into _itc_step/_scale_housekeeping,
    and none of which the eval golden above touches. Student params are
    perturbed with seeded noise first so momentum actually lags the student
    (otherwise momentum == student at construction and the update is a no-op).

Both run on CPU in eval() (dropout off) with a deterministic seeded STUB teacher
(no real BLIP-large, no checkpoint). The only student-side RNG is ITM neg-mining
(torch.multinomial); it is pinned by reseeding the global generator IMMEDIATELY
before each forward, so runs are bit-identical. A later refactor that reorders
RNG-consuming ops in the student will therefore make these tests FAIL -- that is
intended (it catches real changes).

Requires conda env kd_r4 (transformers 4.33.3); the base env cannot import the
model stack. Capture / re-capture goldens standalone (prints paste-ready dicts;
must run as a module so `models` resolves against repo root, not script dir):
    conda run -n kd_r4 python -m distillation.test_forward_equiv
Run the test:
    conda run -n kd_r4 python -m pytest distillation/test_forward_equiv.py -q
"""
import unittest

import torch
import torch.distributed as dist
import torch.nn.functional as F

from models.blip_pretrain import blip_pretrain

# ---- fixed experiment knobs (changing ANY of these invalidates the golden) ----
B = 4
QUEUE_SIZE = 8            # multiple of B; mutated only by the train-state golden below
ITM_TOPK = 4             # Phase-2 gathered ITM KD (k+1=5 candidates)
GAMMA = 0.5              # TTM teacher/momentum slot split
ALPHA = 0.4             # unused while TTM branch active; passed for signature
LM_TEMP = 2.0
ITM_TEMP = 0.05
ITM_DIR = "bidir"
IMAGE_SEED = 42
STUB_SEED = 7
FORWARD_SEED = 1234      # reseeded right before forward -> multinomial reproducible
PERTURB_SEED = 99        # train-state golden only: nudges params so student != momentum
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

# Second golden: same fixed input + stub, update_train_state=True on a FRESH,
# seed-perturbed model (build_and_perturb_model). Exercises _momentum_update,
# queue enqueue, and the in-place logit_scale clamp_ -- none of which GOLDEN
# above touches. Captured the same way (see __main__).
GOLDEN_TRAIN = {
    "loss_ita": 2.8063817024230957,
    "loss_itm": 0.6480762958526611,
    "loss_lm": 10.720789909362793,
    "loss_lm_kd": 1.0469632148742676,
    "loss_itm_kd": 0.0038292997051030397,
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


def perturb_params(model, seed=PERTURB_SEED):
    """Nudges every param (student + momentum) with seeded noise so student !=
    momentum post-construction. Needed only for the train-state golden: momentum
    is copied from student at construction (copy_params), so with student ==
    momentum, _momentum_update's 0.995*m + 0.005*p == m would be a no-op."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))
    return model


def build_and_perturb_model():
    """Fresh model for the train-state golden -- never the eval test's shared
    instance, since update_train_state=True mutates momentum/queue/logit_scale
    in place. Same construction args as build_model(); see perturb_params."""
    return perturb_params(build_model())


def run_forward(model, image, stub, update_train_state=False):
    """One forward on the fixed batch with the stub teacher; returns {loss_name: tensor}.
    Exercises all three mechanisms' base + KD: ITC(+TTM), LM(+KD), ITM(+gathered KD).
    update_train_state=True additionally runs _momentum_update, queue enqueue, and
    the in-place logit_scale clamp_ (see TestForwardEquivTrainState)."""
    if update_train_state and not dist.is_initialized():
        # _dequeue_and_enqueue -> concat_all_gather calls dist.get_world_size(),
        # which raises unless a process group exists -- even for world_size=1 on
        # CPU. HashStore needs no networking/ports, so this is instant and inert.
        dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)
    # Phase 2: forward calls the teacher itself (encode_image/itc_feats/lm_logits/
    # itm_matrix_gathered) via online_teacher=stub; the stub's per-method seeds make
    # those calls order-independent, so the goldens are UNCHANGED from the Phase-1
    # pre-computed-tensor harness. The only student RNG (multinomial) is still pinned.
    torch.manual_seed(FORWARD_SEED)          # pin ITM neg-mining multinomial draws
    out = model(image, CAPTIONS, alpha=ALPHA, update_train_state=update_train_state,
                gamma=GAMMA, online_teacher=stub,
                lm_kd_enabled=True, itm_kd_enabled=True, lm_distill_temp=LM_TEMP,
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


class TestForwardEquivTrainState(unittest.TestCase):
    """update_train_state=True golden: covers _momentum_update, queue enqueue, and
    the in-place logit_scale clamp_, which TestForwardEquiv's update_train_state=False
    golden never runs. The refactor moves exactly these into _itc_step /
    _scale_housekeeping, so this is the coverage that catches it breaking.

    Every test below builds its OWN fresh model (never shared, never forwarded
    twice) -- update_train_state=True mutates momentum/queue/logit_scale in
    place, so a second forward on the same model would see already-advanced
    state rather than a repeat of the first."""

    def test_matches_golden(self):
        model = build_and_perturb_model()
        image = fixed_image()
        stub = StubTeacher(model.tokenizer)
        out = run_forward(model, image, stub, update_train_state=True)
        for k in LOSS_KEYS:
            self.assertIsNotNone(out[k], f"{k} is None -- a mechanism path was skipped")
            self.assertTrue(torch.isfinite(out[k]).all(), f"{k} not finite")
        for k in LOSS_KEYS:
            actual, gold = out[k].item(), GOLDEN_TRAIN[k]
            self.assertLessEqual(abs(actual - gold), ATOL + RTOL * abs(gold),
                                 f"{k}: {actual!r} != golden {gold!r}")
        # Proves the perturbation actually engaged momentum drift: if this ever
        # collapses onto the eval golden, _momentum_update silently became a
        # no-op again (e.g. the perturbation stopped taking effect).
        self.assertTrue(
            any(abs(out[k].item() - GOLDEN[k]) > ATOL + RTOL * abs(GOLDEN[k]) for k in LOSS_KEYS),
            "train-state losses match the eval golden -- momentum drift didn't engage")

    def test_fresh_model_run_to_run_stable(self):
        """Two INDEPENDENT fresh+perturbed models, each forwarded once, must match
        bit-for-bit -- construction, perturbation, and the multinomial pin are all
        seeded, so this is the train-state analogue of TestForwardEquiv's
        test_run_to_run_stable (which reruns forward on one never-mutated model;
        that trick doesn't apply here, see class docstring)."""
        image = fixed_image()
        a_model = build_and_perturb_model()
        b_model = build_and_perturb_model()
        a = run_forward(a_model, image, StubTeacher(a_model.tokenizer), update_train_state=True)
        b = run_forward(b_model, image, StubTeacher(b_model.tokenizer), update_train_state=True)
        for k in LOSS_KEYS:
            self.assertTrue(torch.equal(a[k], b[k]),
                            f"{k} not bit-identical across fresh models: {a[k].item()!r} vs {b[k].item()!r}")


if __name__ == "__main__":
    # Capture path: prints paste-ready GOLDEN / GOLDEN_TRAIN dicts + stability flags.
    _model = build_model()
    _image = fixed_image()
    _stub = StubTeacher(_model.tokenizer)
    r1 = run_forward(_model, _image, _stub)
    r2 = run_forward(_model, _image, _stub)
    print("=== forward 5-loss golden capture (paste into GOLDEN) ===")
    for _k in LOSS_KEYS:
        print(f'    "{_k}": {r1[_k].item()!r},   # stable_rerun={torch.equal(r1[_k], r2[_k])}')

    _tm1 = build_and_perturb_model()
    _tm2 = build_and_perturb_model()
    t1 = run_forward(_tm1, _image, StubTeacher(_tm1.tokenizer), update_train_state=True)
    t2 = run_forward(_tm2, _image, StubTeacher(_tm2.tokenizer), update_train_state=True)
    print("=== forward 5-loss TRAIN-STATE golden capture (paste into GOLDEN_TRAIN) ===")
    for _k in LOSS_KEYS:
        print(f'    "{_k}": {t1[_k].item()!r},   # stable_fresh_model={torch.equal(t1[_k], t2[_k])}')
