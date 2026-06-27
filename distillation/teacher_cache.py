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
