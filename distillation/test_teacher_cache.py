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
