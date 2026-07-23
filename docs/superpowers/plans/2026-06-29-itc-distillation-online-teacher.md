# ITC Distillation — Online Teacher Pivot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the (abandoned) offline teacher cache with an online frozen BLIP-large teacher that runs live on the same augmented batch as the student, feeding the existing ITC distillation loss.

**Architecture:** Keep the already-built student-side pieces (`itc_distill_loss`, the forward's `loss_itc_kd`, the `distill.itc` config). Remove all offline-cache machinery and re-enable augmentation. Add an `OnlineTeacher` (frozen BLIP-large, ITC path only, replicated per GPU) and call it once per train step to produce teacher features, which flow into the unchanged student forward.

**Tech Stack:** PyTorch (bf16 autocast), stdlib `unittest`, existing BLIP codebase.

**Reference spec:** `docs/superpowers/specs/2026-06-29-itc-distillation-online-teacher.md` (supersedes the teacher-delivery half of the 2026-06-27 caching spec; student-side loss math/toggle semantics there still hold).

## Global Constraints

- **Python env:** `kd_r4` — `/home/minwoo/miniconda3/envs/kd_r4/bin/python`. Run all commands from the worktree root `/home/minwoo/Distillation_Project_itc_distill` (cwd may reset between shell calls — `cd` at the start of each).
- **Branch:** `itc_distill` (already checked out in this worktree; current HEAD `0f2cb19`). Do NOT create a branch. Modify forward with new commits.
- **Test framework:** stdlib `unittest` only (no pytest), co-located with code.
- **Distillation default OFF:** `distill.itc.enabled` stays `false` by default → with it off, training is the original 2-tuple path, byte-for-byte.
- **Augmentation REQUIRED ON:** `pretrain_train_aug: true` (the reason for this pivot; aug-off was a NO-GO).
- **Precision:** bf16 autocast for both the teacher forward and the KD sim (verified: the ITC-logit matmul is bf16 under autocast, softmax stays fp32). Do NOT force fp32.
- **Batch fixed at 40**; KD is in-batch local **40×40**, no all-gather, no GPU communication; DDP averages gradients across ranks.
- **Reused UNCHANGED — do not touch:** `distillation/losses.py` (`itc_distill_loss`), `models/blip_pretrain.py` forward (4-tuple return `(loss_ita, loss_itm, loss_lm, loss_itc_kd)`), `data/eval_validation_loss.py` (4-value unpack).
- **Teacher:** BLIP-large = ViT-L + BERT-base (`vit='large'`, `my_bert_size='base'`); `teacher.checkpoint` is user-supplied and only needed when distillation is enabled.

---

## File Structure

| File | Change |
|---|---|
| `distillation/teacher_cache.py` | **delete** |
| `distillation/test_teacher_cache.py` | **delete** |
| `build_teacher_cache.py` | **delete** |
| `data/test_pretrain_dataset_cache.py` | **delete** |
| `data/pretrain_dataset.py` | **revert** to pre-distillation (drop `teacher_cache`) |
| `data/__init__.py` | **revert** to pre-distillation (drop cache wiring, keep aug toggle) |
| `distillation/online_teacher.py` | **new** — `OnlineTeacher` |
| `distillation/test_online_teacher.py` | **new** — CPU smoke |
| `pretrain.py` | **modify** — build teacher in `main()`, call it in `train()` (2-tuple loop) |
| `configs/pretrain.yaml` | **modify** — `pretrain_train_aug: true`, drop `cache_dir` |

---

## Task 1: Remove offline-cache code + revert dataset/aug

**Files:**
- Delete: `distillation/teacher_cache.py`, `distillation/test_teacher_cache.py`, `build_teacher_cache.py`, `data/test_pretrain_dataset_cache.py`
- Revert: `data/pretrain_dataset.py`, `data/__init__.py`

**Interfaces:**
- Produces: `pretrain_dataset(ann_file, laion_path, img_root_coco, img_root_vg, transform)` (no `teacher_cache`); `__getitem__` returns `(image, caption)`. `create_dataset('pretrain', config)` returns a plain `pretrain_dataset` honoring `pretrain_train_aug`, with NO cache.

- [ ] **Step 1: Delete the cache files**

```bash
cd /home/minwoo/Distillation_Project_itc_distill
git rm distillation/teacher_cache.py distillation/test_teacher_cache.py build_teacher_cache.py data/test_pretrain_dataset_cache.py
```

- [ ] **Step 2: Revert the two data files to their pre-distillation state**

`b988045` (the branch base) already contains the `pretrain_train_aug` toggle in `data/__init__.py` and the original `pretrain_dataset.py` (no `teacher_cache`). Reverting these two files to `b988045` removes exactly the Task-4 cache additions while keeping the aug toggle:

```bash
cd /home/minwoo/Distillation_Project_itc_distill
git checkout b988045 -- data/pretrain_dataset.py data/__init__.py
```

- [ ] **Step 3: Verify no cache references remain and imports are clean**

```bash
cd /home/minwoo/Distillation_Project_itc_distill
grep -rn "teacher_cache\|TeacherCache\|build_cache\|dataset_signature" data/ distillation/ pretrain.py || echo "NO cache refs (good)"
/home/minwoo/miniconda3/envs/kd_r4/bin/python -c "import data, distillation; print('imports clean')"
/home/minwoo/miniconda3/envs/kd_r4/bin/python -m unittest distillation.test_losses -v
```
Expected: `NO cache refs (good)`, `imports clean`, and `distillation.test_losses` 5/5 PASS.
(Note: `data/__init__.py` still imports fine — its `TeacherCache` import line is gone after the revert. `models/blip_pretrain.py` is unchanged and still returns the 4-tuple; nothing references the cache.)

- [ ] **Step 4: Commit**

```bash
cd /home/minwoo/Distillation_Project_itc_distill
git add -A
git commit -m "refactor(distill): remove offline teacher cache, revert dataset+aug (pivot to online teacher)"
```

---

## Task 2: `OnlineTeacher` module

**Files:**
- Create: `distillation/online_teacher.py`
- Test: `distillation/test_online_teacher.py`

**Interfaces:**
- Produces: `OnlineTeacher(checkpoint, image_size, vit='large', bert='base', queue_size=240)` with `.to(device) -> self` and `@torch.no_grad() itc_feats(image, caption) -> (img_feat[B,256], txt_feat[B,256])` (L2-normalized). When `checkpoint` is falsy, no weights are loaded (pretrained backbones only) — used by the test.

- [ ] **Step 1: Write the failing test**

Create `distillation/test_online_teacher.py`:

```python
import unittest
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m unittest distillation.test_online_teacher -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'distillation.online_teacher'`.

- [ ] **Step 3: Write the implementation**

Create `distillation/online_teacher.py`:

```python
import torch
import torch.nn.functional as F

from models.blip_pretrain import blip_pretrain


class OnlineTeacher:
    """Frozen BLIP teacher served live in the train loop. Provides ITC features
    for the same augmented batch the student sees. Only the ITC path is kept;
    momentum encoders, decoder, ITM head and queues are freed after load.
    """

    def __init__(self, checkpoint, image_size, vit="large", bert="base", queue_size=240):
        model = blip_pretrain(image_size=image_size, vit=vit, my_bert_size=bert,
                              queue_size=queue_size)
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            msg = model.load_state_dict(state, strict=False)
            print("teacher load:", msg)

        # free submodules/buffers itc_feats never uses (~halves teacher memory)
        for attr in ("visual_encoder_m", "text_encoder_m", "vision_proj_m",
                     "text_proj_m", "text_decoder", "itm_head"):
            if hasattr(model, attr):
                setattr(model, attr, None)
        for buf in ("image_queue", "text_queue", "queue_ptr"):
            if hasattr(model, buf):
                setattr(model, buf, None)

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        self.model = model
        self.tokenizer = model.tokenizer

    def to(self, device):
        self.model.to(device)
        return self

    @torch.no_grad()
    def itc_feats(self, image, caption):
        device = image.device
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            image_embeds = self.model.visual_encoder(image)
            img_feat = F.normalize(self.model.vision_proj(image_embeds[:, 0, :]), dim=-1)
            text = self.tokenizer(caption, padding="max_length", truncation=True,
                                  max_length=30, return_tensors="pt").to(device)
            text_output = self.model.text_encoder(text.input_ids,
                                                  attention_mask=text.attention_mask,
                                                  return_dict=True, mode="text")
            txt_feat = F.normalize(self.model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
        return img_feat, txt_feat
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m unittest distillation.test_online_teacher -v`
Expected: PASS (3 tests). May take ~1-2 min for construction. If construction errors for an environment reason unrelated to the code (e.g. weight-download failure offline), report DONE_WITH_CONCERNS and fall back to `python -c "import ast; ast.parse(open('distillation/online_teacher.py').read()); print('parses')"`.

- [ ] **Step 5: Commit**

```bash
cd /home/minwoo/Distillation_Project_itc_distill
git add distillation/online_teacher.py distillation/test_online_teacher.py
git commit -m "feat(distill): OnlineTeacher (frozen BLIP-large, ITC path only, bf16 no_grad)"
```

---

## Task 3: Wire `pretrain.py` (build teacher in main, call it in train loop)

**Files:**
- Modify: `pretrain.py` (train signature ~70, loop 94-99 + 108, main 341-347 + 356)

**Interfaces:**
- Consumes: `OnlineTeacher` (Task 2), the unchanged `model.forward(..., teacher_img_feat, teacher_text_feat, distill_temp)`.
- Produces: training that, when `distill.itc.enabled`, computes teacher feats per step from a replicated `OnlineTeacher` and feeds them to the student forward.

- [ ] **Step 1: Add `online_teacher` to the `train()` signature**

Replace `pretrain.py:70`:
```python
def train(model, data_loader, optimizer, epoch, device, config, writer=None, val_loss_runner=None, collapse_counter=None, model_without_ddp=None, retrieval_val_runner=None): # writer추가 val loss runner 추가, collapse_counter추가, retrieval val runner 추가
```
with:
```python
def train(model, data_loader, optimizer, epoch, device, config, writer=None, val_loss_runner=None, collapse_counter=None, model_without_ddp=None, retrieval_val_runner=None, online_teacher=None): # online_teacher 추가
```

- [ ] **Step 2: Revert the batch loop to a 2-tuple and source teacher feats from the online teacher**

Replace the loop head (`pretrain.py:94-99`):
```python
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if len(batch) == 4:
            image, caption, teacher_img_feat, teacher_text_feat = batch
        else:
            image, caption = batch
            teacher_img_feat = teacher_text_feat = None
```
with:
```python
    for i, (image, caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
```

Then replace the device-move line (`pretrain.py:108`):
```python
        image = image.to(device,non_blocking=True)
```
with:
```python
        image = image.to(device,non_blocking=True)

        # online teacher: same augmented batch -> ITC features (bf16, no_grad). None when distill off.
        if itc_kd_enabled and online_teacher is not None:
            teacher_img_feat, teacher_text_feat = online_teacher.itc_feats(image, caption)
        else:
            teacher_img_feat = teacher_text_feat = None
```
(The forward calls at lines ~119-133 already consume `teacher_img_feat`/`teacher_text_feat` — leave them unchanged.)

- [ ] **Step 3: Build the teacher in `main()` and pass it to `train()`**

After the DDP-wrap block (`pretrain.py:341`, the line `model_without_ddp = model.module`) and before the `collapse_counter` block, insert:
```python

    #### online teacher (distillation) — replicate per rank, frozen, not DDP-wrapped ####
    online_teacher = None
    if config.get('distill', {}).get('itc', {}).get('enabled', False):
        from distillation.online_teacher import OnlineTeacher
        online_teacher = OnlineTeacher(
            checkpoint=config['teacher']['checkpoint'],
            image_size=config['image_size'],
            vit='large', bert='base',
            queue_size=config['queue_size'],
        ).to(device)
        print("[distill] online teacher loaded")
```

Then replace the `train(...)` call (`pretrain.py:356`):
```python
            train_stats = train(model, data_loader, optimizer, epoch, device, config, writer, val_loss_runner=val_loss_runner, collapse_counter=collapse_counter, model_without_ddp=model_without_ddp, retrieval_val_runner=retrieval_val_runner) # writer추가, collapse_counter추가, retrieval val runner 추가
```
with:
```python
            train_stats = train(model, data_loader, optimizer, epoch, device, config, writer, val_loss_runner=val_loss_runner, collapse_counter=collapse_counter, model_without_ddp=model_without_ddp, retrieval_val_runner=retrieval_val_runner, online_teacher=online_teacher) # online_teacher 추가
```

- [ ] **Step 4: Verify parse + tests + off-path**

```bash
cd /home/minwoo/Distillation_Project_itc_distill
/home/minwoo/miniconda3/envs/kd_r4/bin/python -c "import ast; ast.parse(open('pretrain.py').read()); print('pretrain.py parses')"
/home/minwoo/miniconda3/envs/kd_r4/bin/python -m unittest distillation.test_losses distillation.test_online_teacher -v
```
Expected: `pretrain.py parses`; tests PASS. (Off-path: with `distill.itc.enabled` absent/false, `online_teacher` stays `None`, the loop sets teacher feats to `None`, and the forward returns `loss_itc_kd=None` → original training. A full training smoke is out of scope here — GPUs are busy and it needs the real flow; it is covered by the end-to-end run in Task 4.)

- [ ] **Step 5: Commit**

```bash
cd /home/minwoo/Distillation_Project_itc_distill
git add pretrain.py
git commit -m "feat(distill): wire OnlineTeacher into pretrain (build in main, call per step, 2-tuple loop)"
```

---

## Task 4: Config (aug on) + full suite + end-to-end checklist

**Files:**
- Modify: `configs/pretrain.yaml`

- [ ] **Step 1: Re-enable augmentation and drop the now-unused cache_dir**

In `configs/pretrain.yaml`, set the aug toggle to true. Replace:
```yaml
pretrain_train_aug: false
```
with:
```yaml
pretrain_train_aug: true   # online teacher distillation requires augmentation ON (no-aug was a NO-GO)
```
And delete the `cache_dir` line from the `distill.itc` block (online teacher needs no cache):
```yaml
    cache_dir: '/home/minwoo/Distillation_Project/output/teacher_cache_blip_large'
```
(Leave `distill.itc.enabled: false`, `weight`, `temp`, and the `teacher` block as-is.)

- [ ] **Step 2: Verify config + full suite**

```bash
cd /home/minwoo/Distillation_Project_itc_distill
/home/minwoo/miniconda3/envs/kd_r4/bin/python -c "import yaml; c=yaml.safe_load(open('configs/pretrain.yaml')); assert c['pretrain_train_aug'] is True; assert c['distill']['itc']['enabled'] is False; assert 'cache_dir' not in c['distill']['itc']; print('config OK: aug on, distill off, no cache_dir')"
/home/minwoo/miniconda3/envs/kd_r4/bin/python -m unittest distillation.test_losses distillation.test_online_teacher -v
```
Expected: `config OK: aug on, distill off, no cache_dir`; tests PASS.

- [ ] **Step 3: Commit**

```bash
cd /home/minwoo/Distillation_Project_itc_distill
git add configs/pretrain.yaml
git commit -m "config(distill): pretrain_train_aug=true, drop cache_dir (online teacher)"
```

- [ ] **Step 4: End-to-end (manual, by the user)**

Run by the user once the BLIP-large checkpoint is available:
1. Set `teacher.checkpoint` to the trained BLIP-large weights; set `distill.itc.enabled: true`; set a distinct `output_dir` and `exp` (e.g. `6.itc_distill_online`).
2. Launch training as usual (4 GPUs, batch 40, aug on).
3. Confirm: `[distill] online teacher loaded` prints; `loss_train/itc_kd` appears in TensorBoard and is finite; per-GPU memory holds teacher (~2 GB) + student; step time reflects the added teacher forward.

---

## Self-Review

**Spec coverage:** §1 pivot rationale → plan intro; §2 remove/revert/reuse lists → Task 1 (+ reused files untouched); §3.1 OnlineTeacher (load, free submodules, eval/no_grad, itc_feats bf16) → Task 2; §3.2 replicate per rank, not DDP-wrapped → Task 3 Step 3; §3.3 2-tuple loop + per-step teacher call → Task 3 Steps 2; §4 batch-40/local-KD (unchanged loop math), bf16 (autocast in itc_feats + existing forward), aug on → Task 4 Step 1 + constraints; §5 cost → end-to-end note; §6 testing → Task 2 test + Task 3/4 verifies; §7 teacher checkpoint required → Task 3 build + Task 4 end-to-end; §9 branch modify-forward → all tasks commit on `itc_distill`.

**Placeholder scan:** no TBD/TODO; the only external value is `teacher.checkpoint` (user-supplied, exercised only when enabled).

**Type consistency:** `OnlineTeacher(checkpoint, image_size, vit, bert, queue_size)` + `.to()` + `itc_feats(image, caption) -> (img_feat, txt_feat)` identical in Task 2 (def/test) and Task 3 (construction + call). `online_teacher` param threads consistently through `train()` signature, body, and the `main()` call. The forward's `(loss_ita, loss_itm, loss_lm, loss_itc_kd)` 4-tuple is untouched and matches the existing loop unpack.
