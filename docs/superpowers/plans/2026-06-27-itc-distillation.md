# ITC Knowledge Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a toggleable ITC knowledge-distillation term that pulls an in-batch B×B similarity-matrix KL signal from a frozen BLIP-large teacher (served from an offline per-sample feature cache) into the existing BLIP pretraining.

**Architecture:** A frozen BLIP-large teacher is run once offline to cache per-sample 256-d ITC features (image + text) keyed by dataset row index. During student training the dataset returns the cached teacher features by index; the model forward reconstructs the teacher's in-batch B×B similarity matrix and matches it against the student's via a symmetric KL term, added to the existing `loss_ita + loss_itm + loss_lm`. The whole distillation path is gated by a config toggle that, when off, restores byte-for-byte original training.

**Tech Stack:** PyTorch, NumPy memmap (`.npy`), stdlib `unittest`, existing BLIP codebase (`models/blip_pretrain.py`, `data/`, `pretrain.py`).

**Reference spec:** `docs/superpowers/specs/2026-06-27-itc-distillation-design.md`

## Global Constraints

- **Python env:** `kd_r4` — `/home/minwoo/miniconda3/envs/kd_r4/bin/python`. Every `python` command below runs in this env. Run all commands from repo root `/home/minwoo/Distillation_Project`.
- **Test framework:** stdlib `unittest` only (no pytest in `kd_r4`), matching `claude_skills/monitoring/test_*.py`. Tests are co-located with code.
- **Branch:** all work on `dev/itc_distill` (branched from `dev`). A `5.no_aug` experiment is running from the main working tree — prefer a separate git worktree so its files are undisturbed.
- **Distillation default OFF:** `distill.itc.enabled` defaults to `false` in config so no unrelated run is ever affected.
- **Distillation runs require `pretrain_train_aug: false`** (already committed `b988045`) so the cached teacher view matches the student input view.
- **Features:** all ITC features are L2-normalized, 256-d. Teacher features are constants (`detach`).
- **Loss form:** same temperature `τ` for teacher and student; `KL(teacher ‖ student)`; `0.5*(i2t+t2i)*τ²`.
- **Cache:** fp16 on disk, upcast to fp32 at use; **dataset row index is the key**; a manifest signature guards against stale caches.
- **forward arity:** `BLIP_Pretrain.forward` return changes from 3 → 4 values. The only two call sites are `pretrain.py` (train loop) and `data/eval_validation_loss.py:297` (val) — both must be updated.

---

## File Structure

| File | Responsibility |
|---|---|
| `distillation/__init__.py` | package marker |
| `distillation/losses.py` | `itc_distill_loss` pure function (ITM/LM added later) |
| `distillation/teacher_cache.py` | manifest + memmap cache writer/loader (`write_cache`, `TeacherCache`, `build_cache`, `dataset_signature`) |
| `distillation/test_losses.py` | unittest for `itc_distill_loss` |
| `distillation/test_teacher_cache.py` | unittest for cache format + builder |
| `build_teacher_cache.py` | Phase-0 CLI: load BLIP-large, iterate dataset, write cache |
| `data/pretrain_dataset.py` (modify) | optional `teacher_cache`; return feats by index |
| `data/__init__.py` (modify) | wire cache into pretrain dataset when distill enabled |
| `models/blip_pretrain.py` (modify) | forward accepts teacher feats → `loss_itc_kd`; 4-tuple return |
| `pretrain.py` (modify) | handle 4-tuple batch, add weighted KD loss, TB logging |
| `data/eval_validation_loss.py` (modify) | absorb 4th return value |
| `configs/pretrain.yaml` (modify) | `distill` + `teacher` blocks |

---

## Task 1: Branch + `itc_distill_loss`

**Files:**
- Create: `distillation/__init__.py`, `distillation/losses.py`, `distillation/test_losses.py`

**Interfaces:**
- Produces: `itc_distill_loss(image_feat_s, text_feat_s, teacher_img_feat, teacher_txt_feat, temp) -> torch.Tensor` (0-dim scalar). Inputs are `[B, D]` L2-normalized; teacher inputs treated as constants.

- [ ] **Step 1: Create the branch (separate worktree recommended)**

```bash
# Option A — worktree (keeps the running 5.no_aug tree untouched):
git worktree add -b dev/itc_distill ../Distillation_Project_itc_distill dev
# then cd ../Distillation_Project_itc_distill for all further work
# Option B — in place:
# git checkout dev && git checkout -b dev/itc_distill
```

- [ ] **Step 2: Create the package marker**

Create `distillation/__init__.py`:

```python
```

(empty file)

- [ ] **Step 3: Write the failing test**

Create `distillation/test_losses.py`:

```python
import unittest
import torch
import torch.nn.functional as F

from distillation.losses import itc_distill_loss


def _norm(x):
    return F.normalize(x, dim=-1)


class TestItcDistillLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B, self.D, self.temp = 5, 8, 0.05

    def test_zero_when_teacher_equals_student(self):
        img = _norm(torch.randn(self.B, self.D)).requires_grad_(True)
        txt = _norm(torch.randn(self.B, self.D)).requires_grad_(True)
        loss = itc_distill_loss(img, txt, img.detach(), txt.detach(), self.temp)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_positive_when_teacher_differs(self):
        img_s = _norm(torch.randn(self.B, self.D)).requires_grad_(True)
        txt_s = _norm(torch.randn(self.B, self.D)).requires_grad_(True)
        img_t = _norm(torch.randn(self.B, self.D))
        txt_t = _norm(torch.randn(self.B, self.D))
        loss = itc_distill_loss(img_s, txt_s, img_t, txt_t, self.temp)
        self.assertGreater(loss.item(), 0.0)

    def test_gradient_flows_to_student_not_teacher(self):
        img_s = _norm(torch.randn(self.B, self.D)).requires_grad_(True)
        txt_s = _norm(torch.randn(self.B, self.D)).requires_grad_(True)
        img_t = _norm(torch.randn(self.B, self.D))   # no grad
        txt_t = _norm(torch.randn(self.B, self.D))
        loss = itc_distill_loss(img_s, txt_s, img_t, txt_t, self.temp)
        loss.backward()
        self.assertIsNotNone(img_s.grad)
        self.assertIsNotNone(txt_s.grad)
        self.assertFalse(img_t.requires_grad)

    def test_image_text_swap_symmetry(self):
        # swapping (image<->text) on both student and teacher swaps i2t<->t2i,
        # and since we average them the scalar must be identical.
        img_s = _norm(torch.randn(self.B, self.D))
        txt_s = _norm(torch.randn(self.B, self.D))
        img_t = _norm(torch.randn(self.B, self.D))
        txt_t = _norm(torch.randn(self.B, self.D))
        a = itc_distill_loss(img_s, txt_s, img_t, txt_t, self.temp)
        b = itc_distill_loss(txt_s, img_s, txt_t, img_t, self.temp)
        self.assertAlmostEqual(a.item(), b.item(), places=6)

    def test_returns_scalar(self):
        img = _norm(torch.randn(self.B, self.D))
        txt = _norm(torch.randn(self.B, self.D))
        loss = itc_distill_loss(img, txt, img, txt, self.temp)
        self.assertEqual(loss.dim(), 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python -m unittest distillation.test_losses -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'distillation.losses'` (or ImportError).

- [ ] **Step 5: Write minimal implementation**

Create `distillation/losses.py`:

```python
import torch
import torch.nn.functional as F


def itc_distill_loss(image_feat_s, text_feat_s, teacher_img_feat, teacher_txt_feat, temp):
    """In-batch B×B relational KD for the ITC signal.

    Args:
        image_feat_s, text_feat_s: student online features, [B, D], L2-normalized, require grad.
        teacher_img_feat, teacher_txt_feat: teacher features, [B, D], L2-normalized (constants).
        temp: distillation temperature τ (same for teacher and student).

    Returns:
        Scalar tensor = 0.5 * (KL_i2t + KL_t2i) * τ², with KL(teacher ‖ student).
    """
    teacher_img_feat = teacher_img_feat.detach()
    teacher_txt_feat = teacher_txt_feat.detach()

    s_s = (image_feat_s @ text_feat_s.t()) / temp          # [B, B] student
    s_t = (teacher_img_feat @ teacher_txt_feat.t()) / temp  # [B, B] teacher

    # image->text: rows = images, distribution over texts (dim=1)
    loss_i2t = F.kl_div(F.log_softmax(s_s, dim=1), F.softmax(s_t, dim=1), reduction="batchmean")
    # text->image: transpose
    loss_t2i = F.kl_div(F.log_softmax(s_s.t(), dim=1), F.softmax(s_t.t(), dim=1), reduction="batchmean")

    return 0.5 * (loss_i2t + loss_t2i) * (temp ** 2)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m unittest distillation.test_losses -v`
Expected: PASS (5 tests OK).

- [ ] **Step 7: Commit**

```bash
git add distillation/__init__.py distillation/losses.py distillation/test_losses.py
git commit -m "feat(distill): itc_distill_loss in-batch BxB relational KL"
```

---

## Task 2: Teacher cache format (manifest + memmap)

**Files:**
- Modify: `distillation/teacher_cache.py` (create)
- Test: `distillation/test_teacher_cache.py` (create)

**Interfaces:**
- Produces:
  - `dataset_signature(train_files, n) -> dict` — `{"train_files": [...], "N": n}`.
  - `write_cache(out_dir, img_feats, txt_feats, signature, dim, teacher_id) -> None` — writes fp16 `.npy` arrays + `manifest.json`.
  - `TeacherCache(cache_dir)` with `.manifest`, `.N`, `__len__()`, `get(index) -> (img_fp32[D], txt_fp32[D])`, `validate_against(n, train_files) -> None` (raises `ValueError` on mismatch).

- [ ] **Step 1: Write the failing test**

Create `distillation/test_teacher_cache.py`:

```python
import os
import tempfile
import unittest

import numpy as np
import torch

from distillation.teacher_cache import (
    dataset_signature, write_cache, TeacherCache,
)


class TestTeacherCacheFormat(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.N, self.D = 4, 8
        rng = np.random.default_rng(0)
        self.img = rng.standard_normal((self.N, self.D)).astype(np.float32)
        self.txt = rng.standard_normal((self.N, self.D)).astype(np.float32)
        self.sig = dataset_signature(["a_coco.json", "b_vg.json"], self.N)
        write_cache(self.tmp, self.img, self.txt, self.sig, self.D, teacher_id="dummy")

    def test_roundtrip_get(self):
        cache = TeacherCache(self.tmp)
        self.assertEqual(len(cache), self.N)
        img0, txt0 = cache.get(0)
        self.assertEqual(img0.dtype, torch.float32)
        self.assertEqual(tuple(img0.shape), (self.D,))
        # fp16 storage tolerance
        np.testing.assert_allclose(img0.numpy(), self.img[0], atol=1e-2)
        np.testing.assert_allclose(txt0.numpy(), self.txt[0], atol=1e-2)

    def test_validate_ok(self):
        cache = TeacherCache(self.tmp)
        cache.validate_against(self.N, ["a_coco.json", "b_vg.json"])  # no raise

    def test_validate_signature_mismatch_raises(self):
        cache = TeacherCache(self.tmp)
        with self.assertRaises(ValueError):
            cache.validate_against(self.N, ["a_coco.json"])  # different train_files
        with self.assertRaises(ValueError):
            cache.validate_against(self.N + 1, ["a_coco.json", "b_vg.json"])  # different N


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest distillation.test_teacher_cache -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'distillation.teacher_cache'`.

- [ ] **Step 3: Write minimal implementation**

Create `distillation/teacher_cache.py`:

```python
import json
import os

import numpy as np
import torch

MANIFEST_NAME = "manifest.json"
IMG_FEATS_NAME = "teacher_img_feats.npy"
TXT_FEATS_NAME = "teacher_txt_feats.npy"


def dataset_signature(train_files, n):
    """Stable signature of the dataset ordering used to key the cache."""
    return {"train_files": list(train_files), "N": int(n)}


def write_cache(out_dir, img_feats, txt_feats, signature, dim, teacher_id):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, IMG_FEATS_NAME), np.asarray(img_feats, dtype=np.float16))
    np.save(os.path.join(out_dir, TXT_FEATS_NAME), np.asarray(txt_feats, dtype=np.float16))
    manifest = {
        "N": int(np.asarray(img_feats).shape[0]),
        "dim": int(dim),
        "teacher_id": teacher_id,
        "signature": signature,
    }
    with open(os.path.join(out_dir, MANIFEST_NAME), "w") as f:
        json.dump(manifest, f, indent=2)


class TeacherCache:
    def __init__(self, cache_dir):
        with open(os.path.join(cache_dir, MANIFEST_NAME)) as f:
            self.manifest = json.load(f)
        self.img = np.load(os.path.join(cache_dir, IMG_FEATS_NAME), mmap_mode="r")
        self.txt = np.load(os.path.join(cache_dir, TXT_FEATS_NAME), mmap_mode="r")
        self.N = int(self.manifest["N"])

    def __len__(self):
        return self.N

    def get(self, index):
        img = torch.from_numpy(np.asarray(self.img[index], dtype=np.float32))
        txt = torch.from_numpy(np.asarray(self.txt[index], dtype=np.float32))
        return img, txt

    def validate_against(self, n, train_files):
        sig = self.manifest.get("signature", {})
        if int(self.manifest["N"]) != int(n):
            raise ValueError(
                f"Teacher cache N={self.manifest['N']} != dataset len {n}; cache is stale."
            )
        if sig.get("train_files") != list(train_files):
            raise ValueError(
                "Teacher cache train_files signature mismatch; dataset changed since caching."
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest distillation.test_teacher_cache -v`
Expected: PASS (3 tests OK).

- [ ] **Step 5: Commit**

```bash
git add distillation/teacher_cache.py distillation/test_teacher_cache.py
git commit -m "feat(distill): teacher feature cache format (manifest + fp16 memmap)"
```

---

## Task 3: Cache builder core + CLI

**Files:**
- Modify: `distillation/teacher_cache.py` (add `build_cache`)
- Modify: `distillation/test_teacher_cache.py` (add builder test)
- Create: `build_teacher_cache.py`

**Interfaces:**
- Produces: `build_cache(teacher_feat_fn, dataset, out_dir, dim, signature, teacher_id, batch_size=64) -> int` (returns N). `teacher_feat_fn(images[B,...], captions[list]) -> (img_feat[B,dim], txt_feat[B,dim])`, L2-normalized. Iterates `dataset` in **index order**, writes via `write_cache`.

- [ ] **Step 1: Write the failing test (append to `distillation/test_teacher_cache.py`)**

Add this class to `distillation/test_teacher_cache.py`:

```python
import torch.nn.functional as F
from distillation.teacher_cache import build_cache


class _TinyDataset:
    """Returns (image_tensor, caption) by index; deterministic."""
    def __init__(self, n, c, h, w):
        self.items = []
        g = torch.Generator().manual_seed(123)
        for i in range(n):
            self.items.append((torch.rand(c, h, w, generator=g), f"cap{i}"))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _dummy_feat_fn(dim):
    # deterministic per-sample features: project flattened image + caption-length signal
    def fn(images, captions):
        flat = images.flatten(1)                       # [B, C*H*W]
        proj = flat[:, :dim] if flat.shape[1] >= dim else F.pad(flat, (0, dim - flat.shape[1]))
        cap_sig = torch.tensor([[len(c)] for c in captions], dtype=torch.float32)
        img_f = F.normalize(proj, dim=-1)
        txt_f = F.normalize(proj + cap_sig, dim=-1)
        return img_f, txt_f
    return fn


class TestBuildCache(unittest.TestCase):
    def test_build_matches_teacher_per_index(self):
        tmp = tempfile.mkdtemp()
        N, D = 6, 8
        ds = _TinyDataset(N, 3, 4, 4)
        fn = _dummy_feat_fn(D)
        sig = dataset_signature(["x_coco.json"], N)
        n = build_cache(fn, ds, tmp, D, sig, teacher_id="dummy", batch_size=4)
        self.assertEqual(n, N)

        cache = TeacherCache(tmp)
        for i in range(N):
            img_i, txt_i = ds[i]
            exp_img, exp_txt = fn(img_i.unsqueeze(0), [txt_i])
            got_img, got_txt = cache.get(i)
            np.testing.assert_allclose(got_img.numpy(), exp_img[0].numpy(), atol=1e-2)
            np.testing.assert_allclose(got_txt.numpy(), exp_txt[0].numpy(), atol=1e-2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest distillation.test_teacher_cache.TestBuildCache -v`
Expected: FAIL with `ImportError: cannot import name 'build_cache'`.

- [ ] **Step 3: Implement `build_cache` (append to `distillation/teacher_cache.py`)**

```python
@torch.no_grad()
def build_cache(teacher_feat_fn, dataset, out_dir, dim, signature, teacher_id, batch_size=64):
    """Iterate `dataset` in index order, compute teacher features, write the cache.

    teacher_feat_fn(images, captions) -> (img_feat[B,dim], txt_feat[B,dim]), L2-normalized.
    Returns N.
    """
    n = len(dataset)
    img_feats = np.zeros((n, dim), dtype=np.float32)
    txt_feats = np.zeros((n, dim), dtype=np.float32)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        images = torch.stack([dataset[i][0] for i in range(start, stop)])
        captions = [dataset[i][1] for i in range(start, stop)]
        img_f, txt_f = teacher_feat_fn(images, captions)
        img_feats[start:stop] = img_f.detach().cpu().numpy()
        txt_feats[start:stop] = txt_f.detach().cpu().numpy()
    write_cache(out_dir, img_feats, txt_feats, signature, dim, teacher_id)
    return n
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest distillation.test_teacher_cache -v`
Expected: PASS (4 tests OK total).

- [ ] **Step 5: Write the Phase-0 CLI**

Create `build_teacher_cache.py`:

```python
"""Phase-0: run the frozen BLIP-large teacher over the pretrain dataset (index order,
deterministic transform) and cache per-sample 256-d ITC features.

Usage:
  python build_teacher_cache.py --config ./configs/pretrain.yaml [--batch_size 64] [--device cuda]
"""
import argparse
import ruamel.yaml as _yaml  # repo already uses ruamel/pyyaml; fall back below if needed

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from data.pretrain_dataset import pretrain_dataset
from distillation.teacher_cache import dataset_signature, build_cache


def load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def build_teacher(config, device):
    """Construct + load the frozen BLIP-large teacher.

    NOTE: this must match the user's trained BLIP-large checkpoint. Adjust vit/bert
    sizes here if the teacher differs from (vit='large', my_bert_size='base').
    """
    from models.blip_pretrain import blip_pretrain
    teacher = blip_pretrain(
        image_size=config["image_size"],
        vit="large",
        my_bert_size="base",
        queue_size=config["queue_size"],
    )
    ckpt = torch.load(config["teacher"]["checkpoint"], map_location="cpu")
    state = ckpt.get("model", ckpt)
    msg = teacher.load_state_dict(state, strict=False)
    print("teacher load:", msg)
    teacher.eval().to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def make_feat_fn(teacher, device):
    @torch.no_grad()
    def fn(images, captions):
        images = images.to(device)
        image_embeds = teacher.visual_encoder(images)
        img_f = F.normalize(teacher.vision_proj(image_embeds[:, 0, :]), dim=-1)
        text = teacher.tokenizer(captions, padding="max_length", truncation=True,
                                 max_length=30, return_tensors="pt").to(device)
        text_output = teacher.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                           return_dict=True, mode="text")
        txt_f = F.normalize(teacher.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
        return img_f, txt_f
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./configs/pretrain.yaml")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    config = load_yaml(args.config)
    device = torch.device(args.device)

    # deterministic transform (no random aug) — matches student input when pretrain_train_aug=false
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711))
    transform_test = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"]),
                          interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(), normalize,
    ])

    dataset = pretrain_dataset(
        ann_file=config["train_file"], laion_path=config["laion_path"],
        img_root_coco=config["image_root_coco"], img_root_vg=config["image_root_vg"],
        transform=transform_test,
    )
    n = len(dataset)
    sig = dataset_signature(config["train_file"], n)
    feat_fn = make_feat_fn(build_teacher(config, device), device)

    out_dir = config["distill"]["itc"]["cache_dir"]
    print(f"building teacher cache: N={n} -> {out_dir}")
    build_cache(feat_fn, dataset, out_dir, dim=256, signature=sig,
                teacher_id=config["teacher"]["checkpoint"], batch_size=args.batch_size)
    print("done")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add distillation/teacher_cache.py distillation/test_teacher_cache.py build_teacher_cache.py
git commit -m "feat(distill): teacher cache builder (build_cache) + Phase-0 CLI"
```

---

## Task 4: Dataset integration

**Files:**
- Modify: `data/pretrain_dataset.py` (constructor + `__getitem__`)
- Modify: `data/__init__.py` (`create_dataset` 'pretrain' branch)
- Test: `data/test_pretrain_dataset_cache.py` (create)

**Interfaces:**
- Consumes: `TeacherCache.get(index)`, `TeacherCache.validate_against(n, train_files)` (Task 2).
- Produces: `pretrain_dataset(..., teacher_cache=None)`; `__getitem__(i)` returns `(image, caption)` when no cache, else `(image, caption, img_feat_t, txt_feat_t)`.

- [ ] **Step 1: Write the failing test**

Create `data/test_pretrain_dataset_cache.py`:

```python
import json
import os
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from data.pretrain_dataset import pretrain_dataset
from distillation.teacher_cache import dataset_signature, write_cache, TeacherCache


class TestPretrainDatasetCache(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.coco_root = os.path.join(self.root, "coco"); os.makedirs(self.coco_root)
        Image.new("RGB", (16, 16), (123, 222, 64)).save(os.path.join(self.coco_root, "x.jpg"))
        ann = [
            {"image": "x.jpg", "caption": "a green square", "dataset_source": "coco"},
            {"image": "x.jpg", "caption": "another caption", "dataset_source": "coco"},
        ]
        self.ann_path = os.path.join(self.root, "tiny_coco.json")
        with open(self.ann_path, "w") as f:
            json.dump(ann, f)
        self.tf = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
        self.N, self.D = 2, 8

    def _make_cache(self):
        tmp = tempfile.mkdtemp()
        rng = np.random.default_rng(1)
        img = rng.standard_normal((self.N, self.D)).astype(np.float32)
        txt = rng.standard_normal((self.N, self.D)).astype(np.float32)
        sig = dataset_signature([self.ann_path], self.N)
        write_cache(tmp, img, txt, sig, self.D, teacher_id="dummy")
        return TeacherCache(tmp)

    def test_without_cache_returns_pair(self):
        ds = pretrain_dataset([self.ann_path], "", self.coco_root, self.coco_root, self.tf)
        out = ds[0]
        self.assertEqual(len(out), 2)

    def test_with_cache_returns_feats_by_index(self):
        cache = self._make_cache()
        ds = pretrain_dataset([self.ann_path], "", self.coco_root, self.coco_root, self.tf,
                              teacher_cache=cache)
        image, caption, img_t, txt_t = ds[1]
        exp_img, exp_txt = cache.get(1)
        self.assertTrue(torch.allclose(img_t, exp_img))
        self.assertTrue(torch.allclose(txt_t, exp_txt))

    def test_deterministic_image(self):
        ds = pretrain_dataset([self.ann_path], "", self.coco_root, self.coco_root, self.tf)
        self.assertTrue(torch.allclose(ds[0][0], ds[0][0]))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest data.test_pretrain_dataset_cache -v`
Expected: FAIL — `test_with_cache_returns_feats_by_index` errors (`__init__` got unexpected `teacher_cache`).

- [ ] **Step 3: Modify `data/pretrain_dataset.py`**

In `__init__`, change the signature and store the cache. Replace:
```python
    def __init__(self, ann_file, laion_path, img_root_coco, img_root_vg, transform):
```
with:
```python
    def __init__(self, ann_file, laion_path, img_root_coco, img_root_vg, transform, teacher_cache=None):
```
and, right after `self.transform = transform` (line ~54), add:
```python
        self.teacher_cache = teacher_cache
```

At the end of `__getitem__`, replace:
```python
        image = self.transform(image)
        caption = pre_caption(ann['caption'],30)
        
        return image, caption # 반출
```
with:
```python
        image = self.transform(image)
        caption = pre_caption(ann['caption'],30)

        if self.teacher_cache is not None:
            img_feat_t, txt_feat_t = self.teacher_cache.get(index)
            return image, caption, img_feat_t, txt_feat_t
        return image, caption # 반출
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest data.test_pretrain_dataset_cache -v`
Expected: PASS (3 tests OK).

- [ ] **Step 5: Wire the cache into `create_dataset` (`data/__init__.py`)**

Add the import near the top (after the existing `from data.pretrain_dataset import pretrain_dataset`):
```python
from distillation.teacher_cache import TeacherCache
```

Replace the `if dataset=='pretrain':` branch (currently building with `pretrain_transform`) with:
```python
    if dataset=='pretrain':
        # pretrain_train_aug=false면 학습 입력을 transform_test(결정적)로 고정 (캐싱 distillation용 control)
        use_train_aug = config.get('pretrain_train_aug', True)
        pretrain_transform = transform_train if use_train_aug else transform_test

        teacher_cache = None
        distill_itc = config.get('distill', {}).get('itc', {})
        if distill_itc.get('enabled', False):
            teacher_cache = TeacherCache(distill_itc['cache_dir'])

        dataset = pretrain_dataset(ann_file=config['train_file'],
                                   laion_path=config['laion_path'],
                                   img_root_coco=config['image_root_coco'],
                                   img_root_vg=config['image_root_vg'],
                                   transform=pretrain_transform,
                                   teacher_cache=teacher_cache)
        if teacher_cache is not None:
            teacher_cache.validate_against(len(dataset), config['train_file'])
        return dataset
```

- [ ] **Step 6: Re-run the dataset test (regression)**

Run: `python -m unittest data.test_pretrain_dataset_cache -v`
Expected: PASS (still 3 OK — `create_dataset` change does not affect these direct-construction tests).

- [ ] **Step 7: Commit**

```bash
git add data/pretrain_dataset.py data/__init__.py data/test_pretrain_dataset_cache.py
git commit -m "feat(distill): pretrain_dataset returns teacher feats by index; wire cache in create_dataset"
```

---

## Task 5: Model forward — `loss_itc_kd` + update call sites

**Files:**
- Modify: `models/blip_pretrain.py` (import, `forward` signature, KD term, return)
- Modify: `data/eval_validation_loss.py:297` (absorb 4th value)
- (Train-loop call sites handled in Task 6.)

**Interfaces:**
- Consumes: `itc_distill_loss` (Task 1).
- Produces: `BLIP_Pretrain.forward(image, caption, alpha, update_train_state=None, teacher_img_feat=None, teacher_text_feat=None, distill_temp=0.05) -> (loss_ita, loss_itm, loss_lm, loss_itc_kd)` where `loss_itc_kd` is `None` when no teacher feats are given.

- [ ] **Step 1: Add the import (`models/blip_pretrain.py`)**

After `from models.blip import create_vit, init_tokenizer, load_checkpoint` add:
```python
from distillation.losses import itc_distill_loss
```

- [ ] **Step 2: Extend the forward signature**

Replace:
```python
    def forward(self, image, caption, alpha, update_train_state=None):
```
with:
```python
    def forward(self, image, caption, alpha, update_train_state=None,
                teacher_img_feat=None, teacher_text_feat=None, distill_temp=0.05):
```

- [ ] **Step 3: Compute the KD term and extend the return**

Replace the final:
```python
        loss_lm = decoder_output.loss                
        return loss_ita, loss_itm, loss_lm
```
with:
```python
        loss_lm = decoder_output.loss

        # external-teacher ITC distillation (in-batch BxB relational KL); None when disabled.
        loss_itc_kd = None
        if teacher_img_feat is not None and teacher_text_feat is not None:
            loss_itc_kd = itc_distill_loss(
                image_feat, text_feat,
                teacher_img_feat.to(image.device), teacher_text_feat.to(image.device),
                distill_temp,
            )

        return loss_ita, loss_itm, loss_lm, loss_itc_kd
```

- [ ] **Step 4: Update the validation call site (`data/eval_validation_loss.py:297`)**

Change the unpack (line ~297) from:
```python
                    loss_ita, loss_itm, loss_lm = model(
```
so the 4th value is absorbed — i.e. the assignment target becomes:
```python
                    loss_ita, loss_itm, loss_lm, _loss_itc_kd = model(
```
(Leave the rest of that call unchanged; the val loader has no teacher cache so `_loss_itc_kd` is `None`.)

- [ ] **Step 5: Verify all `BLIP_Pretrain.forward` call sites are updated**

Run: `grep -rn "= model(" pretrain.py data/eval_validation_loss.py`
Expected: `eval_validation_loss.py` now unpacks 4 values; `pretrain.py` lines 110/114 still unpack 3 (fixed in Task 6). No other unpack-3 site for this model exists (verified: `train_retrieval.py`/`train_caption.py` use different models).

- [ ] **Step 6: Integration smoke (run on GPU if available)**

Create and run `scratch_smoke_forward.py` (temporary, not committed):
```python
import torch, torch.nn.functional as F
from models.blip_pretrain import blip_pretrain

m = blip_pretrain(image_size=224, vit="base", my_bert_size="base", queue_size=240).cuda().train()
img = torch.randn(2, 3, 224, 224).cuda()
caps = ["a cat on a mat", "a dog in a park"]

# distill OFF
out = m(img, caps, alpha=0.4)
assert len(out) == 4 and out[3] is None, out
# distill ON (synthetic teacher feats)
t_img = F.normalize(torch.randn(2, 256), dim=-1).cuda()
t_txt = F.normalize(torch.randn(2, 256), dim=-1).cuda()
out = m(img, caps, alpha=0.4, teacher_img_feat=t_img, teacher_text_feat=t_txt, distill_temp=0.05)
assert len(out) == 4 and torch.isfinite(out[3]), out
print("forward smoke OK:", float(out[3]))
```
Run: `python scratch_smoke_forward.py`
Expected: prints `forward smoke OK: <finite number>`. Then `rm scratch_smoke_forward.py`.
(If no GPU is available, drop `.cuda()` and run on CPU — slower but valid.)

- [ ] **Step 7: Commit**

```bash
git add models/blip_pretrain.py data/eval_validation_loss.py
git commit -m "feat(distill): forward returns loss_itc_kd; update val call site to 4-tuple"
```

---

## Task 6: Train-loop integration (`pretrain.py`)

**Files:**
- Modify: `pretrain.py` (batch unpack at :89, forward calls at :110/:114, loss sum, metric + TB logging)

**Interfaces:**
- Consumes: `BLIP_Pretrain.forward(...) -> (loss_ita, loss_itm, loss_lm, loss_itc_kd)` (Task 5); `config['distill']['itc']` (`enabled`, `weight`, `temp`).
- Produces: training that adds `weight * loss_itc_kd` to the total and logs `loss_train/itc_kd`.

- [ ] **Step 1: Read distill settings once at the top of `train(...)`**

After `model.train()` (line ~72), add:
```python
    distill_itc = config.get('distill', {}).get('itc', {})
    itc_kd_enabled = distill_itc.get('enabled', False)
    itc_kd_weight = float(distill_itc.get('weight', 1.0))
    itc_kd_temp = float(distill_itc.get('temp', 0.05))
```

- [ ] **Step 2: Make the batch loop accept both 2- and 4-tuples**

Replace line 89:
```python
    for i, (image, caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
```
with:
```python
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if len(batch) == 4:
            image, caption, teacher_img_feat, teacher_text_feat = batch
        else:
            image, caption = batch
            teacher_img_feat = teacher_text_feat = None
```

- [ ] **Step 3: Pass teacher feats into both forward calls and add the KD loss**

Replace the autocast/else block (lines ~107-115):
```python
        if device.type == "cuda": # .type로 수정해봄
            # print("autocast available")
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss_ita, loss_itm, loss_lm = model(image, caption, alpha = alpha)  
                loss = loss_ita + loss_itm + loss_lm
        else:
            # print("autocast failed")
            loss_ita, loss_itm, loss_lm = model(image, caption, alpha = alpha)  
            loss = loss_ita + loss_itm + loss_lm
```
with:
```python
        if device.type == "cuda":
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss_ita, loss_itm, loss_lm, loss_itc_kd = model(
                    image, caption, alpha=alpha,
                    teacher_img_feat=teacher_img_feat, teacher_text_feat=teacher_text_feat,
                    distill_temp=itc_kd_temp)
                loss = loss_ita + loss_itm + loss_lm
                if itc_kd_enabled and loss_itc_kd is not None:
                    loss = loss + itc_kd_weight * loss_itc_kd
        else:
            loss_ita, loss_itm, loss_lm, loss_itc_kd = model(
                image, caption, alpha=alpha,
                teacher_img_feat=teacher_img_feat, teacher_text_feat=teacher_text_feat,
                distill_temp=itc_kd_temp)
            loss = loss_ita + loss_itm + loss_lm
            if itc_kd_enabled and loss_itc_kd is not None:
                loss = loss + itc_kd_weight * loss_itc_kd
```

- [ ] **Step 4: Log the KD loss**

After the existing `metric_logger.update(loss_lm=loss_lm.item())` (line ~122), add:
```python
        if loss_itc_kd is not None:
            metric_logger.update(loss_itc_kd=loss_itc_kd.item())
```
And inside the `if writer is not None:` TB block (after `writer.add_scalar("loss_train/lm", ...)`, line ~131), add:
```python
                if loss_itc_kd is not None:
                    writer.add_scalar("loss_train/itc_kd", loss_itc_kd.item(), global_step)
```

- [ ] **Step 5: Regression — distillation OFF behaves exactly as before**

With the default config (`distill.itc.enabled` absent/false), run a short smoke: 2 steps of the real train loop on the existing data is heavy; instead do a syntax + toggle check:

Run: `python -c "import ast; ast.parse(open('pretrain.py').read()); print('pretrain.py parses')"`
Expected: `pretrain.py parses`.

Then a logic check that the 2-tuple path is taken when no cache: run `python -m unittest data.test_pretrain_dataset_cache -v` (PASS) confirms the dataset still returns 2-tuples when distillation is off, so the loop's `else` branch and original loss are used unchanged.

- [ ] **Step 6: Commit**

```bash
git add pretrain.py
git commit -m "feat(distill): train loop adds weighted itc_kd loss + TB logging, handles cache batches"
```

---

## Task 7: Config block + end-to-end verification

**Files:**
- Modify: `configs/pretrain.yaml` (add `distill` + `teacher`)

**Interfaces:**
- Consumes: all prior tasks.

- [ ] **Step 1: Add the config blocks (`configs/pretrain.yaml`)**

Append near the existing `exp` / `pretrain_train_aug` keys:
```yaml
# ====== knowledge distillation (default OFF — never affects unrelated runs) ======
distill:
  itc:
    enabled: false          # true로 켤 때 pretrain_train_aug 도 false 여야 함
    weight: 1.0             # λ_itc
    temp: 0.05              # τ, 증류 온도 (model logit_scale와 별개)
    cache_dir: '/home/minwoo/Distillation_Project/output/teacher_cache_blip_large'

teacher:
  arch: 'blip_large'
  checkpoint: '<REQUIRED: 학습된 BLIP-large 체크포인트 경로>'   # Phase-0 전에 채울 것
```

- [ ] **Step 2: Verify config parses and defaults are OFF**

Run:
```bash
python -c "import yaml; c=yaml.safe_load(open('configs/pretrain.yaml')); print('itc enabled =', c['distill']['itc']['enabled']); assert c['distill']['itc']['enabled'] is False"
```
Expected: `itc enabled = False`.

- [ ] **Step 3: Full suite green**

Run: `python -m unittest distillation.test_losses distillation.test_teacher_cache data.test_pretrain_dataset_cache -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add configs/pretrain.yaml
git commit -m "feat(distill): config blocks for itc distillation + teacher (default off)"
```

- [ ] **Step 5: End-to-end (manual, gated on teacher checkpoint + no-aug go/no-go)**

This is run by the user once the `5.no_aug` control confirms aug-off is acceptable and the BLIP-large checkpoint path is filled in:
1. Set `teacher.checkpoint`, confirm `pretrain_train_aug: false`.
2. Build the cache: `python build_teacher_cache.py --config ./configs/pretrain.yaml` → writes `output/teacher_cache_blip_large/{manifest.json,teacher_img_feats.npy,teacher_txt_feats.npy}`.
3. Sanity-check the cache: `python -c "from distillation.teacher_cache import TeacherCache; c=TeacherCache('output/teacher_cache_blip_large'); print('N=',len(c)); print(c.get(0)[0].shape)"`.
4. Set `distill.itc.enabled: true`, set a distinct `output_dir` and `exp` (e.g. `6.itc_distill`).
5. Launch training as usual; confirm `loss_train/itc_kd` appears in TensorBoard and total loss reflects it.

---

## Self-Review

**Spec coverage:** §2.1 relational KL → Task 1; §2.2 in-batch B×B → Task 1 loss; §2.3 offline cache → Tasks 2-3; §2.4 index key + manifest → Tasks 2-4; §2.5 aug-off → already committed (`b988045`), enforced via doc note in Task 7; §2.6 momentum/queue untouched → preserved (forward only adds a term, never touches queue); §2.7 additive + toggle → Tasks 6-7; §3.1 TeacherProvider seam → realized as cache+dataset path (Task 4) with note; §3.2 cache builder → Task 3; §3.3 dataset → Task 4; §3.4 loss fn → Task 1; §3.5 forward → Task 5; §3.6 train loop → Task 6; §5 math → Task 1 code; §6 tests → Tasks 1-4 unit tests + Task 5/6 smokes; §7 config → Task 7; §8 toggle semantics → Tasks 4/6 (enabled gate) + Task 7 default off.

**Placeholder scan:** the only `<REQUIRED ...>` is `teacher.checkpoint`, which is a genuine user-supplied value flagged in the spec; everything else is concrete code.

**Type consistency:** `itc_distill_loss(image_feat_s, text_feat_s, teacher_img_feat, teacher_txt_feat, temp)` used identically in Task 1 and Task 5; `TeacherCache.get`/`validate_against`/`dataset_signature(train_files, n)`/`write_cache`/`build_cache` signatures consistent across Tasks 2-4; forward 4-tuple `(loss_ita, loss_itm, loss_lm, loss_itc_kd)` consistent across Tasks 5-6 and the val call site.
