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
