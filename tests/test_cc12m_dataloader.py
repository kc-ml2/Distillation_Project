import inspect
import os
import unittest

import data as data_pkg
from data.pretrain_cc12m_webdataset import cc12m_webdataset


def test_shardshuffle_size_param_default_100():
    sig = inspect.signature(cc12m_webdataset.__init__)
    assert 'shardshuffle_size' in sig.parameters
    assert sig.parameters['shardshuffle_size'].default == 100


def test_no_literal_shardshuffle_true_in_source():
    src = inspect.getsource(cc12m_webdataset)
    assert 'shardshuffle=True' not in src
    assert 'shardshuffle=shardshuffle_size' in src


def test_cc12m_transform_respects_pretrain_train_aug():
    src = inspect.getsource(data_pkg.create_dataset)
    # cc12m 분기가 base와 동일하게 토글을 참조해야 한다(하드코딩 transform_train 금지)
    assert "config.get('pretrain_train_aug'" in src
    assert 'cc12m_transform' in src


def test_factory_returns_base_loader_when_no_cc12m():
    from data.pretrain_loader import build_pretrain_dataloader
    src = inspect.getsource(build_pretrain_dataloader)
    assert "if not config['cc12m_tar_path']:" in src
    assert 'return base_loader' in src


def test_factory_applies_with_epoch_unconditionally():
    from data.pretrain_loader import build_pretrain_dataloader
    src = inspect.getsource(build_pretrain_dataloader)
    # with_epoch는 num_tasks>1 게이트 없이 항상 적용(단일/DDP 일관)
    assert '.with_epoch(' in src
    assert 'num_tasks > 1' not in src


def test_factory_builds_combined_loader():
    from data.pretrain_loader import build_pretrain_dataloader
    src = inspect.getsource(build_pretrain_dataloader)
    assert 'CombinedLoader(' in src


def test_factory_docstring_has_operational_notes():
    from data.pretrain_loader import build_pretrain_dataloader
    doc = build_pretrain_dataloader.__doc__ or ''
    assert 'torchrun' in doc        # queue distill 분산 필수 note
    assert 'warmup_steps' in doc     # 에폭 길이 커플링 note


import pretrain


def test_pretrain_uses_factory():
    src = inspect.getsource(pretrain.main)
    assert 'build_pretrain_dataloader(' in src


def test_pretrain_has_no_inline_cc12m_wiring():
    src = inspect.getsource(pretrain.main)
    assert 'wds.WebLoader(' not in src
    assert 'CombinedLoader(' not in src


def test_create_loader_no_dead_cc12m_comment():
    src = inspect.getsource(data_pkg.create_loader)
    assert 'ddp_equalize' not in src
    assert '잘못된 데이터셋' not in src
    assert 'cc12m_loader = wds.WebLoader' not in src


CC12M_SHARD_DIR = "/home/minwoo/Distillation_Project/datasets/vision/cc12m/cc12m_dataset"


class CC12MBehavioralRegression(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
            import webdataset  # noqa: F401
            import torchvision  # noqa: F401
        except Exception:
            self.skipTest("torch/webdataset/torchvision not available")
        if not os.path.isdir(CC12M_SHARD_DIR) or not any(
            f.endswith('.tar') for f in os.listdir(CC12M_SHARD_DIR)
        ):
            self.skipTest("cc12m shards not present on this machine")

    def test_batch_format_and_with_epoch_cap(self):
        import torch
        import webdataset as wds
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode
        from data.pretrain_cc12m_webdataset import cc12m_webdataset

        norm = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711))
        tf = transforms.Compose([
            transforms.RandomResizedCrop(
                224, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            norm,
        ])
        ds = cc12m_webdataset(tar_root=CC12M_SHARD_DIR, transform=tf, batch_size=8)
        loader = wds.WebLoader(ds, batch_size=None, num_workers=2).with_epoch(3)

        n = 0
        for img, cap in loader:
            self.assertIsInstance(img, torch.Tensor)
            self.assertEqual(tuple(img.shape), (8, 3, 224, 224))
            self.assertTrue(img.dtype == torch.float32)
            self.assertIsInstance(cap, (list, tuple))
            self.assertEqual(len(cap), 8)
            n += 1
        self.assertEqual(n, 3)  # with_epoch(3) caps the epoch to 3 batches
