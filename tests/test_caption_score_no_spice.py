from data.utils import _score_no_spice


def test_score_no_spice_returns_cider_not_spice():
    gts = {1: [{'caption': 'a cat sitting on a mat'}]}
    res = {1: [{'caption': 'a cat sitting on a mat'}]}
    scores = _score_no_spice(gts, res)
    assert 'CIDEr' in scores
    assert 'Bleu_4' in scores
    assert 'METEOR' in scores
    assert 'ROUGE_L' in scores
    assert 'SPICE' not in scores
    assert scores['Bleu_1'] > 0.9  # identical hypothesis/reference
