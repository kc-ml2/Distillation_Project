import os
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from distillation.teacher_cache import (
    dataset_signature, write_cache, TeacherCache, build_cache,
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


if __name__ == "__main__":
    unittest.main()
