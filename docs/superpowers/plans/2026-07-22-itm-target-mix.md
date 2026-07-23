# ITM teacher-target-mix distillation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Distill the BLIP-large teacher's ITM (image-text matching) judgment into the student by mixing the teacher's soft match-probability into the ITM target, config-gated (OFF = current baseline).

**Architecture:** ITM negatives (mined per-sample) can be selected by student-sim (A/baseline) or teacher-sim (B/C); the 3B triplet (pos + neg-image + neg-text) is scored in one merged forward; the loss is a single soft-target CE `(1−W)·onehot + W·teacher_soft` (W=0 → plain CE). The teacher scores the *same* 3B triplets via a new `OnlineTeacher.itm_soft()` that reuses the student's sampled negative indices. All new behavior lives behind `distill.itm_target_mix.enabled`.

**Tech Stack:** PyTorch, HuggingFace BERT (med.py), BLIP (`models/blip_pretrain.py`), unittest (CPU smoke), YAML configs.

**Spec:** `docs/superpowers/specs/2026-07-22-itm-target-mix-design.md`

## Global Constraints

- **Regression-safe OFF path:** with `itm_mix=None`, forward must produce a finite ITM CE numerically equivalent to the current baseline (bit-identical not required; merged 3B forward is allowed).
- **Teacher is frozen / no-grad:** teacher outputs never carry gradient into the student; `teacher_soft` is detached.
- **Shared negative indices:** student and teacher must assemble the identical 3B triplet order `[pos B, neg-image B, neg-text B]` from the SAME sampled indices.
- **Tokenizer parity:** student and teacher use identical tokenizers (30524 vocab, same `enc_token_id`/`bos_token_id`); `itm_soft` consumes the student's `encoder_input_ids`/`attention_mask` directly (no re-tokenize).
- **Teacher build:** `vit='large'`, `bert='base'`, `init_backbone_weights=False` (timm-1.x incompat); weights come entirely from `model_large.pth`.
- **Teacher checkpoint path:** `/home/minwoo/Distillation_Project/output/official_pretrain_checkpoint/model_large.pth` (has `itm_head`, cross-attention, `temp=0.0157`; no `logit_scale` key).
- **Hyperparameters (spec §6):** W(`soft_weight`)=0.4, T(`temp`)=1.0, `schedule='constant'`, target applied to all 3B.
- **Python env:** `/home/minwoo/miniconda3/envs/kd_r4/bin/python`.
- **Branch:** implement on a dedicated branch `exp10_itm_target_mix` (create via `superpowers:using-git-worktrees` at execution start).

---

### Task 1: `itm_target_mix_loss` pure function

**Files:**
- Modify: `distillation/losses.py` (append function; imports `torch`, `torch.nn.functional as F` already present at top)
- Test: `distillation/test_losses_itm.py` (create)

**Interfaces:**
- Produces: `itm_target_mix_loss(vl_output, itm_labels, teacher_soft, soft_weight) -> Tensor` (scalar). `vl_output`: `[N,2]` student logits; `itm_labels`: `[N]` long in {0,1}; `teacher_soft`: `[N,2]` rows-sum-to-1; `soft_weight`: float in [0,1]. `W=0` reduces exactly to `F.cross_entropy(vl_output, itm_labels)`.

- [ ] **Step 1: Write the failing tests**

```python
# distillation/test_losses_itm.py
import unittest
import torch
import torch.nn.functional as F
from distillation.losses import itm_target_mix_loss


class TestItmTargetMixLoss(unittest.TestCase):
    def test_w0_equals_cross_entropy(self):
        vl = torch.tensor([[2.0, -1.0], [0.5, 0.5], [-1.0, 3.0]])
        labels = torch.tensor([0, 1, 1])
        teacher = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.1, 0.9]])
        got = itm_target_mix_loss(vl, labels, teacher, 0.0)
        self.assertTrue(torch.allclose(got, F.cross_entropy(vl, labels), atol=1e-6))

    def test_w1_teacher_onehot_equals_ce(self):
        labels = torch.tensor([0, 1])
        vl = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        teacher = F.one_hot(labels, 2).float()
        got = itm_target_mix_loss(vl, labels, teacher, 1.0)
        self.assertTrue(torch.allclose(got, F.cross_entropy(vl, labels), atol=1e-6))

    def test_matches_manual_soft_ce(self):
        vl = torch.tensor([[1.5, -0.5]])
        labels = torch.tensor([0])
        teacher = torch.tensor([[0.2, 0.8]])
        W = 0.4
        target = (1 - W) * torch.tensor([[1.0, 0.0]]) + W * teacher
        manual = -(target * F.log_softmax(vl, dim=1)).sum(1).mean()
        self.assertTrue(torch.allclose(itm_target_mix_loss(vl, labels, teacher, W), manual, atol=1e-6))

    def test_gradient_flows_to_logits(self):
        vl = torch.randn(3, 2, requires_grad=True)
        teacher = torch.softmax(torch.randn(3, 2), 1)
        labels = torch.tensor([0, 1, 0])
        itm_target_mix_loss(vl, labels, teacher, 0.4).backward()
        self.assertIsNotNone(vl.grad)
        self.assertTrue(torch.isfinite(vl.grad).all())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest distillation/test_losses_itm.py -v`
Expected: FAIL with `ImportError: cannot import name 'itm_target_mix_loss'`

- [ ] **Step 3: Implement the function**

Append to `distillation/losses.py`:

```python
def itm_target_mix_loss(vl_output, itm_labels, teacher_soft, soft_weight):
    """Soft-target CE for ITM: target = (1-W)*onehot(itm_labels) + W*teacher_soft.

    vl_output   : [N, 2] student ITM logits.
    itm_labels  : [N]    long class indices in {0=no-match, 1=match}.
    teacher_soft: [N, 2] teacher match distribution (rows sum to 1, no grad).
    soft_weight : W in [0, 1]. W=0 reduces exactly to F.cross_entropy(vl_output, itm_labels).
    """
    onehot = F.one_hot(itm_labels, num_classes=2).to(vl_output.dtype)
    target = (1.0 - soft_weight) * onehot + soft_weight * teacher_soft.to(vl_output.dtype)
    return -(target * F.log_softmax(vl_output, dim=1)).sum(dim=1).mean()
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest distillation/test_losses_itm.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add distillation/losses.py distillation/test_losses_itm.py
git commit -m "feat(itm-distill): itm_target_mix_loss soft-target CE"
```

---

### Task 2: `OnlineTeacher` — add `itm` keep path + teacher_scale

**Files:**
- Modify: `distillation/online_teacher.py` (`NEEDS`, `CRITICAL_PREFIXES`, `__init__`)
- Modify: `distillation/test_online_teacher.py` (fix stale test; add itm-keep test)

**Interfaces:**
- Consumes: nothing new.
- Produces: `OnlineTeacher(..., keep=('itm',))` keeps `visual_encoder`, `text_encoder`, `itm_head`; frees the rest. New attributes `self.teacher_temp` (float, from checkpoint `temp`, default 0.07) and `self.teacher_scale = 1/teacher_temp`.

- [ ] **Step 1: Write/adjust the failing tests**

In `distillation/test_online_teacher.py`, change the stale `test_unknown_keep_raises` (it uses `("itm",)`, which is about to become valid) and add a new keep test class:

```python
    def test_unknown_keep_raises(self):
        with self.assertRaises(ValueError):
            OnlineTeacher(checkpoint="", image_size=224, vit="base",
                          bert="base", queue_size=240, keep=("bogus",))
```

```python
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
```

- [ ] **Step 2: Run to verify new/changed tests fail**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest distillation/test_online_teacher.py::TestOnlineTeacherKeepItm -v`
Expected: FAIL — `ValueError: unknown keep paths: ['itm']` (itm not yet in NEEDS)

- [ ] **Step 3: Implement the itm keep path**

In `distillation/online_teacher.py`, extend the module-level dicts:

```python
NEEDS = {
    "itc": {"visual_encoder", "text_encoder", "vision_proj", "text_proj"},
    "lm": {"visual_encoder", "text_decoder"},
    "itm": {"visual_encoder", "text_encoder", "itm_head"},
}
```

```python
CRITICAL_PREFIXES = {
    "itc": ("visual_encoder.", "text_encoder.", "vision_proj.", "text_proj."),
    "lm": ("visual_encoder.", "text_decoder."),
    "itm": ("visual_encoder.", "text_encoder.", "itm_head."),
}
```

Read the teacher temperature from the raw checkpoint (the reparam'd model has no `temp` param, so it is only in `state`). Placement matters because `state` only exists inside the `if checkpoint:` block:

1. **Before** the `if checkpoint:` block (near the top of `__init__`, after `self.keep = ...`), set the default:

```python
        self.teacher_temp = 0.07   # overwritten from checkpoint below if present
```

2. **Inside** the `if checkpoint:` block, right after `state = ...` is resolved and before/after `model.load_state_dict(...)`:

```python
            # teacher's learned ITC temperature (for teacher-guided neg selection scale).
            # model_large.pth carries `temp` (=0.0157); the reparam'd model has none, so read it here.
            if "temp" in state:
                self.teacher_temp = float(state["temp"])
```

3. **After** the `if checkpoint:` block (before `model.eval()`), derive the scale from the finalized temp:

```python
        self.teacher_scale = 1.0 / self.teacher_temp
```

- [ ] **Step 4: Run to verify tests pass**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest distillation/test_online_teacher.py -v`
Expected: PASS (all classes incl. new TestOnlineTeacherKeepItm; TestOnlineTeacherKeepBoth.test_union_kept still passes because itm not in its keep)

- [ ] **Step 5: Commit**

```bash
git add distillation/online_teacher.py distillation/test_online_teacher.py
git commit -m "feat(itm-distill): OnlineTeacher itm keep path + teacher_scale"
```

---

### Task 3: `OnlineTeacher.itm_soft()` teacher scoring

**Files:**
- Modify: `distillation/online_teacher.py` (add method)
- Test: `distillation/test_online_teacher.py` (add to `TestOnlineTeacherKeepItm`, plus a raises-test)

**Interfaces:**
- Consumes: `self._require`, `self.model.{visual_encoder,text_encoder,itm_head}`.
- Produces: `itm_soft(image, enc_input_ids, attention_mask, neg_idx_img, neg_idx_txt, temp) -> Tensor[3B, 2]` — softmaxed teacher match distribution over `[pos B, neg-image B, neg-text B]`, no grad. `neg_idx_img`/`neg_idx_txt` are length-B sequences of ints.

- [ ] **Step 1: Write the failing tests**

Add to `TestOnlineTeacherKeepItm`:

```python
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
```

Add a raises-test to the `TestOnlineTeacher` class (keep=('itc',) default):

```python
    def test_itm_soft_raises_without_itm_keep(self):
        image = torch.randn(1, 3, 224, 224)
        ids = torch.zeros(1, 30, dtype=torch.long)
        with self.assertRaises(RuntimeError):
            self.teacher.itm_soft(image, ids, ids, [0], [0], temp=1.0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest distillation/test_online_teacher.py -k itm_soft -v`
Expected: FAIL — `AttributeError: 'OnlineTeacher' object has no attribute 'itm_soft'`

- [ ] **Step 3: Implement `itm_soft`**

Add to `OnlineTeacher` (mirrors student ITM assembly, order `[pos, neg-image, neg-text]`):

```python
    @torch.no_grad()
    def itm_soft(self, image, enc_input_ids, attention_mask,
                 neg_idx_img, neg_idx_txt, temp):
        """Teacher ITM match distribution over the SAME 3B triplets the student built.
        enc_input_ids/attention_mask come from the student (identical tokenizer, pos-0
        already set to enc_token_id). neg_idx_img/neg_idx_txt: length-B int sequences.
        Returns softmax(teacher_itm_logits / temp) as [3B, 2] (no grad)."""
        self._require("itm")
        device = image.device
        bs = image.size(0)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            image_embeds = self.model.visual_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
            img_neg = torch.stack([image_embeds[neg_idx_img[b]] for b in range(bs)])
            txt_neg = torch.stack([enc_input_ids[neg_idx_txt[b]] for b in range(bs)])
            txt_neg_atts = torch.stack([attention_mask[neg_idx_txt[b]] for b in range(bs)])
            text_ids_all = torch.cat([enc_input_ids, enc_input_ids, txt_neg], dim=0)
            text_atts_all = torch.cat([attention_mask, attention_mask, txt_neg_atts], dim=0)
            image_embeds_all = torch.cat([image_embeds, img_neg, image_embeds], dim=0)
            image_atts_all = torch.cat([image_atts, image_atts, image_atts], dim=0)
            out = self.model.text_encoder(text_ids_all,
                                          attention_mask=text_atts_all,
                                          encoder_hidden_states=image_embeds_all,
                                          encoder_attention_mask=image_atts_all,
                                          return_dict=True)
            logits = self.model.itm_head(out.last_hidden_state[:, 0, :]).float()
        return F.softmax(logits / temp, dim=1)
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest distillation/test_online_teacher.py -k itm -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add distillation/online_teacher.py distillation/test_online_teacher.py
git commit -m "feat(itm-distill): OnlineTeacher.itm_soft teacher 3B scoring"
```

---

### Task 4: `blip_pretrain.forward` — merged 3B ITM + neg_source + target-mix

**Files:**
- Modify: `models/blip_pretrain.py` (import line 22; forward signature 347-349; ITM block 435-489)
- Test: `models/test_blip_pretrain_itm.py` (create)

**Interfaces:**
- Consumes: `itm_target_mix_loss` (Task 1), `OnlineTeacher.itm_soft` (Task 3), `teacher_img_feat`/`teacher_text_feat` (existing forward kwargs).
- Produces: forward accepts `online_teacher=None, itm_mix=None`. `itm_mix` is `None` or a dict `{'neg_source': 'teacher'|'student', 'soft_weight': float, 'temp': float, 'sel_scale': float}`. Return tuple unchanged: `(loss_ita, loss_itm, loss_lm, loss_itc_kd, loss_lm_kd)`.

- [ ] **Step 1: Write the failing tests**

```python
# models/test_blip_pretrain_itm.py
import unittest
import torch
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
        itm_mix = {'neg_source': 'student', 'soft_weight': 0.0, 'temp': 1.0, 'sel_scale': 14.29}
        loss_itm = self._forward(itm_mix=itm_mix)[1]
        self.assertTrue(torch.isfinite(loss_itm))

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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest models/test_blip_pretrain_itm.py -v`
Expected: FAIL — `TypeError: forward() got an unexpected keyword argument 'itm_mix'`

- [ ] **Step 3a: Update import and signature**

`models/blip_pretrain.py` line 22 — add the new loss:

```python
from distillation.losses import itc_distill_loss, lm_distill_loss, itm_target_mix_loss
```

Forward signature (lines 347-349) — append two kwargs:

```python
    def forward(self, image, caption, alpha, update_train_state=None,
                teacher_img_feat=None, teacher_text_feat=None, distill_temp=0.05,
                teacher_lm_logits=None, teacher_lm_input_ids=None, lm_distill_temp=2.0,
                online_teacher=None, itm_mix=None):
```

- [ ] **Step 3b: Replace the ITM block (lines 435-489)**

Replace the entire `###== Image-text Matching ==###` block (current lines 435-489, from `encoder_input_ids = text.input_ids.clone()` through `loss_itm = F.cross_entropy(vl_output, itm_labels)`) with:

```python
        ###============== Image-text Matching ===================###
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
        bs = image.size(0)

        # negative-mining sampling weights: teacher-sim (itm_mix teacher) or student-sim.
        if itm_mix is not None and itm_mix['neg_source'] == 'teacher':
            assert teacher_img_feat is not None and teacher_text_feat is not None, \
                "itm_mix neg_source='teacher' requires teacher_img_feat/teacher_text_feat"
            with torch.no_grad():
                t_img = teacher_img_feat.to(image.device)
                t_txt = teacher_text_feat.to(image.device)
                s = itm_mix['sel_scale']
                weights_i2t = F.softmax(t_img @ t_txt.t() * s, dim=1) + 1e-4
                weights_t2i = F.softmax(t_txt @ t_img.t() * s, dim=1) + 1e-4
                weights_i2t.fill_diagonal_(0)
                weights_t2i.fill_diagonal_(0)
        else:
            with torch.no_grad():
                weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1) + 1e-4
                weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1) + 1e-4
                weights_t2i.fill_diagonal_(0)
                weights_i2t.fill_diagonal_(0)

        # draw one negative index per sample (neg image for each text, neg text for each image)
        neg_idx_img = [torch.multinomial(weights_t2i[b], 1).item() for b in range(bs)]
        neg_idx_txt = [torch.multinomial(weights_i2t[b], 1).item() for b in range(bs)]
        image_embeds_neg = torch.stack([image_embeds[neg_idx_img[b]] for b in range(bs)])
        text_ids_neg = torch.stack([encoder_input_ids[neg_idx_txt[b]] for b in range(bs)])
        text_atts_neg = torch.stack([text.attention_mask[neg_idx_txt[b]] for b in range(bs)])

        # one merged 3B forward: rows [pos B | neg-image B | neg-text B]
        text_ids_all = torch.cat([encoder_input_ids, encoder_input_ids, text_ids_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text.attention_mask, text_atts_neg], dim=0)
        image_embeds_all = torch.cat([image_embeds, image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts, image_atts], dim=0)
        output_all = self.text_encoder(text_ids_all,
                                       attention_mask=text_atts_all,
                                       encoder_hidden_states=image_embeds_all,
                                       encoder_attention_mask=image_atts_all,
                                       return_dict=True)
        vl_output = self.itm_head(output_all.last_hidden_state[:, 0, :])   # [3B, 2]
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long),
                                torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(image.device)

        # ITM loss: teacher target-mix (W>0) or plain CE.
        if itm_mix is not None and itm_mix['soft_weight'] > 0 and online_teacher is not None:
            teacher_soft = online_teacher.itm_soft(
                image, encoder_input_ids, text.attention_mask,
                neg_idx_img, neg_idx_txt, itm_mix['temp'])                 # [3B, 2], no grad
            loss_itm = itm_target_mix_loss(vl_output, itm_labels,
                                           teacher_soft.to(image.device),
                                           itm_mix['soft_weight'])
        else:
            loss_itm = F.cross_entropy(vl_output, itm_labels)
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest models/test_blip_pretrain_itm.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add models/blip_pretrain.py models/test_blip_pretrain_itm.py
git commit -m "feat(itm-distill): forward merged 3B ITM + neg_source + target-mix"
```

---

### Task 5: `distill_config` helpers — keep derivation + validation

**Files:**
- Create: `distillation/distill_config.py`
- Test: `distillation/test_distill_config.py`

**Interfaces:**
- Produces: `derive_teacher_keep(distill_cfg) -> tuple` (sorted subset of `('itc','itm','lm')`); `validate_itm_mix_config(distill_cfg) -> None` (raises `AssertionError` on invalid).

- [ ] **Step 1: Write the failing tests**

```python
# distillation/test_distill_config.py
import unittest
from distillation.distill_config import derive_teacher_keep, validate_itm_mix_config


class TestDeriveTeacherKeep(unittest.TestCase):
    def test_baseline_empty(self):
        self.assertEqual(derive_teacher_keep({}), ())

    def test_arm_A_student_wpos(self):  # student neg + W>0 -> itm only
        cfg = {'itm_target_mix': {'enabled': True, 'neg_source': 'student', 'soft_weight': 0.4}}
        self.assertEqual(derive_teacher_keep(cfg), ('itm',))

    def test_arm_B_teacher_w0(self):    # teacher neg + W=0 -> itc only (selection feats)
        cfg = {'itm_target_mix': {'enabled': True, 'neg_source': 'teacher', 'soft_weight': 0.0}}
        self.assertEqual(derive_teacher_keep(cfg), ('itc',))

    def test_arm_C_teacher_wpos(self):  # teacher neg + W>0 -> itc + itm
        cfg = {'itm_target_mix': {'enabled': True, 'neg_source': 'teacher', 'soft_weight': 0.4}}
        self.assertEqual(derive_teacher_keep(cfg), ('itc', 'itm'))

    def test_combines_with_itc_lm(self):
        cfg = {'itc': {'enabled': True}, 'lm': {'enabled': True},
               'itm_target_mix': {'enabled': True, 'neg_source': 'student', 'soft_weight': 0.4}}
        self.assertEqual(derive_teacher_keep(cfg), ('itc', 'itm', 'lm'))


class TestValidateItmMixConfig(unittest.TestCase):
    def test_ok_passes(self):
        validate_itm_mix_config({'itm_target_mix': {'enabled': True, 'neg_source': 'teacher',
                                                    'soft_weight': 0.4, 'temp': 1.0,
                                                    'schedule': 'constant'}})

    def test_disabled_noop(self):
        validate_itm_mix_config({})  # no raise

    def test_bad_neg_source(self):
        with self.assertRaises(AssertionError):
            validate_itm_mix_config({'itm_target_mix': {'enabled': True, 'neg_source': 'foo',
                                                        'soft_weight': 0.4}})

    def test_bad_weight(self):
        with self.assertRaises(AssertionError):
            validate_itm_mix_config({'itm_target_mix': {'enabled': True, 'neg_source': 'teacher',
                                                        'soft_weight': 1.5}})

    def test_unsupported_schedule(self):
        with self.assertRaises(AssertionError):
            validate_itm_mix_config({'itm_target_mix': {'enabled': True, 'neg_source': 'teacher',
                                                        'soft_weight': 0.4, 'schedule': 'decay_to_floor'}})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest distillation/test_distill_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'distillation.distill_config'`

- [ ] **Step 3: Implement**

```python
# distillation/distill_config.py
"""Teacher keep-set derivation and itm_target_mix config validation."""


def derive_teacher_keep(distill_cfg):
    """Which teacher submodule paths must be kept, given the distill config.
    itc/lm: their own enabled flags. itm_target_mix: 'itc' when neg_source=='teacher'
    (selection feats), 'itm' when soft_weight>0 (teacher scoring)."""
    keep = set()
    if distill_cfg.get('itc', {}).get('enabled', False):
        keep.add('itc')
    if distill_cfg.get('lm', {}).get('enabled', False):
        keep.add('lm')
    itm = distill_cfg.get('itm_target_mix', {})
    if itm.get('enabled', False):
        if itm.get('neg_source') == 'teacher':
            keep.add('itc')
        if float(itm.get('soft_weight', 0.0)) > 0.0:
            keep.add('itm')
    return tuple(sorted(keep))


def validate_itm_mix_config(distill_cfg):
    """Assert itm_target_mix config is well-formed (no-op when disabled/absent)."""
    itm = distill_cfg.get('itm_target_mix', {})
    if not itm.get('enabled', False):
        return
    ns = itm.get('neg_source')
    assert ns in ('teacher', 'student'), \
        f"itm_target_mix.neg_source must be 'teacher'|'student', got {ns!r}"
    w = float(itm.get('soft_weight', 0.0))
    assert 0.0 <= w <= 1.0, f"itm_target_mix.soft_weight must be in [0,1], got {w}"
    sched = itm.get('schedule', 'constant')
    assert sched == 'constant', \
        f"itm_target_mix.schedule only 'constant' implemented, got {sched!r}"
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest distillation/test_distill_config.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add distillation/distill_config.py distillation/test_distill_config.py
git commit -m "feat(itm-distill): derive_teacher_keep + validate_itm_mix_config"
```

---

### Task 6: `pretrain.py` wiring — parse, teacher build, loop, TB

**Files:**
- Modify: `pretrain.py` (teacher build ~399-400; `train()` parse ~92; train loop 120-146; TB ~190)

**Interfaces:**
- Consumes: `derive_teacher_keep`, `validate_itm_mix_config` (Task 5); `online_teacher.teacher_scale`, `online_teacher.itm_soft` (Tasks 2-3); forward `online_teacher`/`itm_mix` kwargs (Task 4).
- Produces: end-to-end training with `distill.itm_target_mix`.

- [ ] **Step 1: Teacher build — use derive_teacher_keep + validate**

Replace `pretrain.py` line 399-400:

```python
    distill_cfg = config.get('distill', {})
    teacher_keep = tuple(k for k in ('itc', 'lm') if distill_cfg.get(k, {}).get('enabled', False))
```

with:

```python
    from distillation.distill_config import derive_teacher_keep, validate_itm_mix_config
    distill_cfg = config.get('distill', {})
    validate_itm_mix_config(distill_cfg)
    teacher_keep = derive_teacher_keep(distill_cfg)
```

- [ ] **Step 2: `train()` — parse itm_target_mix config**

In `train()`, after the `distill_lm` parse block (after line 92), add:

```python
    distill_itm = config.get('distill', {}).get('itm_target_mix', {})
    itm_mix_enabled = distill_itm.get('enabled', False)
    itm_neg_source = distill_itm.get('neg_source', 'student')
    itm_soft_weight = float(distill_itm.get('soft_weight', 0.0))
    itm_teacher_temp = float(distill_itm.get('temp', 1.0))
    itm_sel_scale = online_teacher.teacher_scale if online_teacher is not None else 1.0
```

- [ ] **Step 3: Train loop — teacher feats + itm_mix dict + forward call**

Replace the teacher-ITC-feats block (lines 120-124) so teacher feats are also computed when itm neg_source is 'teacher':

```python
        # online teacher ITC feats: for ITC KD and/or teacher-guided ITM neg selection.
        need_teacher_itc = itc_kd_enabled or (itm_mix_enabled and itm_neg_source == 'teacher')
        if need_teacher_itc and online_teacher is not None:
            teacher_img_feat, teacher_text_feat = online_teacher.itc_feats(image, caption)
        else:
            teacher_img_feat = teacher_text_feat = None
```

Immediately before the `if device.type == "cuda":` forward dispatch (line ~139), build the itm_mix dict:

```python
        itm_mix = None
        if itm_mix_enabled:
            itm_mix = {'neg_source': itm_neg_source, 'soft_weight': itm_soft_weight,
                       'temp': itm_teacher_temp, 'sel_scale': itm_sel_scale}
        itm_online_teacher = online_teacher if itm_mix_enabled else None
```

Add `online_teacher=itm_online_teacher, itm_mix=itm_mix` to BOTH `model(...)` calls (the cuda branch at line 141-146 and the else branch at 153-158). Example for the cuda branch:

```python
                loss_ita, loss_itm, loss_lm, loss_itc_kd, loss_lm_kd = model(
                    image, caption, alpha=alpha,
                    teacher_img_feat=teacher_img_feat, teacher_text_feat=teacher_text_feat,
                    distill_temp=itc_kd_temp,
                    teacher_lm_logits=teacher_lm_logits, teacher_lm_input_ids=teacher_lm_ids,
                    lm_distill_temp=lm_kd_temp,
                    online_teacher=itm_online_teacher, itm_mix=itm_mix)
```

(Apply the identical two-kwarg addition to the else-branch `model(...)` at lines 153-158.)

- [ ] **Step 4: TB — log the teacher weight**

After line 190 (`writer.add_scalar("train/alpha", alpha, global_step)`), add:

```python
                if itm_mix_enabled:
                    writer.add_scalar("train/itm_teacher_weight", itm_soft_weight, global_step)
```

- [ ] **Step 5: Smoke-check the module imports and parses**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -c "import pretrain; from distillation.distill_config import derive_teacher_keep; print('import OK')"`
Expected: `import OK` (no syntax/import errors)

- [ ] **Step 6: Commit**

```bash
git add pretrain.py
git commit -m "feat(itm-distill): wire itm_target_mix into pretrain train loop"
```

---

### Task 7: Config files for arms A / B / C

**Files:**
- Create: `configs/pretrain_itm_A_studneg_soft.yaml`
- Create: `configs/pretrain_itm_B_teachneg_hard.yaml`
- Create: `configs/pretrain_itm_C_teachneg_soft.yaml`

**Interfaces:**
- Consumes: config schema (Task 5/6). Each is `pretrain_student.yaml` + an `itm_target_mix` block + a real teacher checkpoint + distinct `exp`/`output_dir`.

- [ ] **Step 1: Create arm C (teacher neg + teacher soft)**

Copy `configs/pretrain_student.yaml` to `configs/pretrain_itm_C_teachneg_soft.yaml`, then set:
- `exp: '10.C_itm_teachneg_soft'`
- `output_dir: '/home/minwoo/Distillation_Project/output/pt_itm_C_teachneg_soft'`
- `teacher.checkpoint: '/home/minwoo/Distillation_Project/output/official_pretrain_checkpoint/model_large.pth'`
- Replace the `distill:` block with:

```yaml
distill:
  itc:
    enabled: false
    weight: 1.0
    temp: 0.05
  itm_target_mix:
    enabled: true
    neg_source: teacher     # 'teacher' | 'student'
    soft_weight: 0.4        # W (0 = hard label). spec §6.2
    temp: 1.0               # T, teacher ITM softmax temp. spec §6.3
    schedule: constant      # 'constant' only (decay_to_floor = future)
```

- [ ] **Step 2: Create arm A (student neg + teacher soft)**

Copy arm C to `configs/pretrain_itm_A_studneg_soft.yaml`; change:
- `exp: '10.A_itm_studneg_soft'`
- `output_dir: '/home/minwoo/Distillation_Project/output/pt_itm_A_studneg_soft'`
- `distill.itm_target_mix.neg_source: student`

- [ ] **Step 3: Create arm B (teacher neg + hard label)**

Copy arm C to `configs/pretrain_itm_B_teachneg_hard.yaml`; change:
- `exp: '10.B_itm_teachneg_hard'`
- `output_dir: '/home/minwoo/Distillation_Project/output/pt_itm_B_teachneg_hard'`
- `distill.itm_target_mix.soft_weight: 0.0`

- [ ] **Step 4: Validate all three parse + derive expected keep**

Run:
```bash
/home/minwoo/miniconda3/envs/kd_r4/bin/python - <<'PY'
import yaml
from distillation.distill_config import derive_teacher_keep, validate_itm_mix_config
for f, exp in [('A_studneg_soft', ('itm',)), ('B_teachneg_hard', ('itc',)), ('C_teachneg_soft', ('itc','itm'))]:
    cfg = yaml.safe_load(open(f'configs/pretrain_itm_{f}.yaml'))
    validate_itm_mix_config(cfg['distill'])
    keep = derive_teacher_keep(cfg['distill'])
    assert keep == exp, f'{f}: keep {keep} != {exp}'
    print(f, 'OK keep=', keep)
PY
```
Expected: three `OK keep=` lines matching A→('itm',), B→('itc',), C→('itc','itm')

- [ ] **Step 5: Commit**

```bash
git add configs/pretrain_itm_A_studneg_soft.yaml configs/pretrain_itm_B_teachneg_hard.yaml configs/pretrain_itm_C_teachneg_soft.yaml
git commit -m "feat(itm-distill): configs for arms A/B/C"
```

---

### Task 8: Full suite + GPU smoke

**Files:** none (verification only)

- [ ] **Step 1: Run the full CPU test suite**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest distillation/ models/test_blip_pretrain_itm.py -v`
Expected: all PASS (losses, online_teacher incl. itm keep + itm_soft, distill_config, forward itm_mix)

- [ ] **Step 2: GPU smoke — real teacher, arm C, a few steps with gradient**

Write `scratchpad/smoke_itm_mix.py` (reuses the validated teacher-load path; runs student forward+backward with arm-C `itm_mix` on GPU for a handful of random batches) and run on one GPU:

```bash
CUDA_VISIBLE_DEVICES=1 /home/minwoo/miniconda3/envs/kd_r4/bin/python scratchpad/smoke_itm_mix.py
```

The smoke must: build a **large** teacher `OnlineTeacher(checkpoint='/home/minwoo/Distillation_Project/output/official_pretrain_checkpoint/model_large.pth', image_size=224, vit='large', bert='base', queue_size=57600, keep=('itc','itm'))` — **`queue_size=57600` is required**: `model_large.pth`'s `image_queue`/`text_queue` are shaped `[256, 57600]`, and `load_state_dict` raises a shape-mismatch `RuntimeError` (not silently skipped by `strict=False`) against any smaller `queue_size` (this matches `configs/pretrain_student.yaml`'s `queue_size: 57600`, the established value for this exact checkpoint); build a **base** student `blip_pretrain(image_size=224, vit='base', my_bert_size='base', queue_size=240)`; for ~5 random batches compute `teacher_img_feat/teacher_text_feat` via `itc_feats`, call `student(image, caption, alpha=0.4, update_train_state=True, teacher_img_feat=..., teacher_text_feat=..., online_teacher=teacher, itm_mix={'neg_source':'teacher','soft_weight':0.4,'temp':1.0,'sel_scale':teacher.teacher_scale})`, then `loss = loss_ita+loss_itm+loss_lm; loss.backward()`.
Expected: no crash, `loss_itm` finite and decreasing-ish across steps, `teacher.teacher_scale ≈ 63.9`.

- [ ] **Step 3: Final commit (smoke script)**

```bash
git add scratchpad/smoke_itm_mix.py
git commit -m "test(itm-distill): GPU smoke for arm-C target-mix"
```

(Note: `scratchpad/` may be gitignored; if `git add` reports it ignored, skip this commit — the smoke script is a throwaway verification artifact.)

---

## Notes for the executor

- **Order matters:** Tasks 1-3 and 5 are independent leaves; Task 4 depends on 1 & 3; Task 6 depends on 2,3,4,5; Task 7 depends on 5; Task 8 depends on all.
- **Slow tests:** base-model construction (`OnlineTeacher`, `blip_pretrain`) takes ~1-2 min each on CPU; the forward integration test builds two base models.
- **Do NOT** re-tokenize inside `itm_soft` — always consume the student's `encoder_input_ids`/`attention_mask` (Global Constraint: tokenizer parity).
- **Do NOT** add a separate OFF-path ITM code branch — the merged 3B forward with `itm_mix=None` IS the baseline path (Global Constraint: regression-safe OFF path).
