import inspect

from data.pretrain_cc12m_webdataset import cc12m_webdataset


def test_shardshuffle_size_param_default_100():
    sig = inspect.signature(cc12m_webdataset.__init__)
    assert 'shardshuffle_size' in sig.parameters
    assert sig.parameters['shardshuffle_size'].default == 100


def test_no_literal_shardshuffle_true_in_source():
    src = inspect.getsource(cc12m_webdataset)
    assert 'shardshuffle=True' not in src
    assert 'shardshuffle=shardshuffle_size' in src
