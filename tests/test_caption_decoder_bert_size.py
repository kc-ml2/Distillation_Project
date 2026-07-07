from models.blip import BLIP_Decoder


def test_minilm_decoder_hidden_size():
    model = BLIP_Decoder(vit='small_reg', image_size=224, prompt='', my_bert_size='minilm')
    assert model.text_decoder.config.hidden_size == 384
    # encoder_width must be overridden to the small_reg vision width (384), not the json's 768
    assert model.text_decoder.config.encoder_width == 384


def test_base_decoder_hidden_size_unchanged():
    model = BLIP_Decoder(vit='base', image_size=224, prompt='', my_bert_size='base')
    assert model.text_decoder.config.hidden_size == 768
