import torch


def itm_bxb_logits(text_encoder, itm_head, image_embeds, image_atts,
                   encoder_input_ids, text_atts):
    """전체 B×B ITM 매치-로짓 매트릭스. row i = image i vs 배치 전체 텍스트.

    반환 [B, B, 2] fp32. text_encoder는 호출자 autocast(bf16 가능)에서 돌지만
    itm_head는 autocast 밖 fp32로 강제(bf16이면 gap 오차 폭증 — 프로브 §7).
    row별 chunk(image i를 B 텍스트에 repeat)라 메모리는 배치당 B 시퀀스로 유계.
    enc_token_id는 호출자가 encoder_input_ids에 이미 설정한 상태로 받는다.
    """
    B = image_embeds.size(0)
    cls_rows = []
    for i in range(B):
        enc_hidden = image_embeds[i:i + 1].repeat(B, 1, 1)     # [B, L_img, D]
        enc_att = image_atts[i:i + 1].repeat(B, 1)             # [B, L_img]
        out = text_encoder(encoder_input_ids,
                           attention_mask=text_atts,
                           encoder_hidden_states=enc_hidden,
                           encoder_attention_mask=enc_att,
                           return_dict=True)
        cls_rows.append(out.last_hidden_state[:, 0, :])        # [B, D]
    cls = torch.stack(cls_rows, dim=0)                         # [B, B, D]
    with torch.autocast(device_type=image_embeds.device.type, enabled=False):
        logits = itm_head(cls.float())                         # [B, B, 2] fp32
    return logits
