import types
import torch
from unittest import mock

from models.blip_pretrain import BLIP_Pretrain
from models.blip import init_tokenizer


def test_generate_returns_one_caption_per_image():
    tokenizer = init_tokenizer()

    def fake_decoder_generate(**kwargs):
        bsz = kwargs["input_ids"].size(0)
        # "a cat" 문장을 batch만큼 반환 (토큰 id 텐서)
        return tokenizer(["a cat"] * bsz, return_tensors="pt").input_ids

    fake = types.SimpleNamespace(
        visual_encoder=lambda img: torch.zeros(img.size(0), 5, 8),
        tokenizer=tokenizer,
        text_decoder=types.SimpleNamespace(generate=fake_decoder_generate),
    )

    image = torch.zeros(3, 3, 224, 224)
    caps = BLIP_Pretrain.generate(fake, image, num_beams=1)

    assert isinstance(caps, list)
    assert len(caps) == 3
    assert all(isinstance(c, str) for c in caps)
