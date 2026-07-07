import os
import pytest
from models.blip import blip_decoder

CKPT = "output/pt_smallreg_minilm_baseline/checkpoint_19.pth"


@pytest.mark.skipif(not os.path.exists(CKPT), reason="small_reg/minilm checkpoint not present")
def test_load_small_reg_minilm_checkpoint_no_keyerror():
    # blip_decoder asserts missing_keys == 0 internally; must not KeyError on pos_embed either.
    model = blip_decoder(pretrained=CKPT, image_size=224, vit='small_reg',
                         prompt='', my_bert_size='minilm')
    assert model is not None
