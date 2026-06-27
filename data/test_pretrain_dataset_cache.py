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
