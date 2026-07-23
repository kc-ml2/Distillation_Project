import inspect

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
