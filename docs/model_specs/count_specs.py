#!/usr/bin/env python
"""BLIP-large(teacher) / BLIP-base / student(DINOv3-S+MiniLM) 스펙·파라미터 실측.

`configs/model_size.md` 와 `docs/model_specs/model_specs.html` 의 숫자를 재생성한다.

- `init_backbone_weights=False` 로 사전학습 백본 다운로드를 건너뛰고 구조만 만들어 센다
  (MiniLM/bert-base 텍스트 가중치는 캐시에서 로드되고, DINOv3는 timm 캐시에서 로드됨).
- tied 파라미터는 `data_ptr` 로 중복 제거한다. `tie_encoder_decoder_weights` 가
  embedding/cross-attention/FFN 을 인코더↔디코더에서 공유하므로, 단순 sum(p.numel()) 은
  이들을 2번 세서 과대계상한다.

실행:  python docs/model_specs/count_specs.py        # repo root 에서
출력:  docs/model_specs/specs.json  (+ stdout 표)
"""
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
sys.path.insert(0, REPO)

import transformers  # noqa: E402

transformers.logging.set_verbosity_error()
from models.blip_pretrain import BLIP_Pretrain  # noqa: E402

OUT = os.path.join(REPO, "docs/model_specs/specs.json")

CONFIGS = [
    ("teacher_blip_large", dict(vit="large", my_bert_size="base", image_size=224)),
    ("blip_base", dict(vit="base", my_bert_size="base", image_size=224)),
    ("student_dinov3s_minilm", dict(vit="small_reg", my_bert_size="minilm", image_size=224)),
]


def uniq(params):
    """data_ptr 로 dedupe 한 고유 파라미터 수."""
    seen, tot = set(), 0
    for p in params:
        if p.data_ptr() in seen:
            continue
        seen.add(p.data_ptr())
        tot += p.numel()
    return tot


def main():
    out = {}
    for name, kw in CONFIGS:
        print(f"\n########## {name}  {kw} ##########", flush=True)
        m = BLIP_Pretrain(embed_dim=256, queue_size=57600, init_backbone_weights=False, **kw)
        m.eval()
        L0 = m.text_encoder.encoder.layer[0]
        D0 = m.text_decoder.bert.encoder.layer[0]
        cfg = m.text_encoder.config
        n_layer = cfg.num_hidden_layers

        rec = dict(
            vit=kw["vit"], bert=kw["my_bert_size"], image_size=kw["image_size"],
            vision_width=m.vision_proj.in_features,
            text_width=m.text_proj.in_features,
            embed_dim=m.vision_proj.out_features,
            text_layers=n_layer, text_heads=cfg.num_attention_heads,
            text_intermediate=cfg.intermediate_size, vocab_size=cfg.vocab_size,
            max_pos=cfg.max_position_embeddings,
            cross_attn_kv_in=L0.crossattention.self.key.in_features,
            # --- 모듈별 (고유 파라미터) ---
            p_visual_encoder=uniq(m.visual_encoder.parameters()),
            p_text_encoder=uniq(m.text_encoder.parameters()),
            p_text_embeddings=uniq(m.text_encoder.embeddings.parameters()),
            p_selfattn_per_layer=uniq(L0.attention.parameters()),
            p_crossattn_per_layer=uniq(L0.crossattention.parameters()),
            p_ffn_per_layer=uniq(L0.intermediate.parameters()) + uniq(L0.output.parameters()),
            p_selfattn_enc_total=n_layer * uniq(L0.attention.parameters()),
            p_selfattn_dec_total=n_layer * uniq(D0.attention.parameters()),
            p_crossattn_total=n_layer * uniq(L0.crossattention.parameters()),
            p_ffn_total=n_layer * (uniq(L0.intermediate.parameters()) + uniq(L0.output.parameters())),
            p_lm_head_transform=uniq(m.text_decoder.cls.predictions.transform.parameters()),
            p_itm_head=uniq(m.itm_head.parameters()),
            p_vision_proj=uniq(m.vision_proj.parameters()),
            p_text_proj=uniq(m.text_proj.parameters()),
            # --- 합계 ---
            p_online_unique=uniq(
                list(m.visual_encoder.parameters()) + list(m.text_encoder.parameters())
                + list(m.text_decoder.parameters()) + list(m.itm_head.parameters())
                + list(m.vision_proj.parameters()) + list(m.text_proj.parameters())
            ),
            p_all_unique=uniq(m.parameters()),
            p_buffers=sum(b.numel() for b in m.buffers()),
        )
        with torch.no_grad():
            emb = m.visual_encoder(torch.zeros(1, 3, kw["image_size"], kw["image_size"]))
        rec["image_tokens"] = tuple(emb.shape)[1]
        rec["vision_out_dim"] = tuple(emb.shape)[2]

        # HTML 스택 차트가 쓰는 6개 카테고리 (합이 p_online_unique 가 되도록)
        cat = dict(
            vision=rec["p_visual_encoder"],
            emb=rec["p_text_embeddings"],
            selfattn=rec["p_selfattn_enc_total"] + rec["p_selfattn_dec_total"],
            cross=rec["p_crossattn_total"],
            ffn=rec["p_ffn_total"],
        )
        cat["other"] = rec["p_online_unique"] - sum(cat.values())
        rec["categories"] = cat

        out[name] = rec
        for k, v in rec.items():
            print(f"  {k:34s} {v}")
        del m

    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
