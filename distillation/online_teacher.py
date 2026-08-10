import torch
import torch.nn.functional as F

from models.blip_pretrain import blip_pretrain

# 경로별 필요 서브모듈. momentum 4개·itm_head·큐는 학습 전용 장치라 어떤 keep에서도 해제.
NEEDS = {
    "itc": {"visual_encoder", "text_encoder", "vision_proj", "text_proj"},
    "lm": {"visual_encoder", "text_decoder"},
    "itm": {"visual_encoder", "text_encoder", "itm_head"},
}
ALL_SUBMODULES = {
    "visual_encoder", "text_encoder", "vision_proj", "text_proj", "text_decoder",
    "visual_encoder_m", "text_encoder_m", "vision_proj_m", "text_proj_m", "itm_head",
}
CRITICAL_PREFIXES = {
    "itc": ("visual_encoder.", "text_encoder.", "vision_proj.", "text_proj."),
    "lm": ("visual_encoder.", "text_decoder."),
    "itm": ("visual_encoder.", "text_encoder.", "itm_head."),
}


class OnlineTeacher:
    """Frozen BLIP teacher served live in the train loop, on the same augmented
    batch the student sees. keep이 지정한 경로의 서브모듈만 유지하고 나머지는
    로드 직후 해제해 티처 메모리를 최소화한다."""

    def __init__(self, checkpoint, image_size, vit="large", bert="base",
                 queue_size=240, keep=("itc",)):
        unknown = set(keep) - set(NEEDS)
        if unknown:
            raise ValueError(f"unknown keep paths: {sorted(unknown)} (choose from {sorted(NEEDS)})")
        self.keep = tuple(keep)
        self.teacher_temp = 0.07   # overwritten from checkpoint below if present

        model = blip_pretrain(image_size=image_size, vit=vit, my_bert_size=bert,
                              queue_size=queue_size,
                              # 티처 가중치는 아래 체크포인트 로드가 전량 결정 —
                              # 백본 사전 초기화(large의 timm-1.x 비호환 경로 포함)를 건너뜀
                              init_backbone_weights=False)
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            # teacher's learned ITC temperature (for teacher-guided neg selection scale).
            # model_large.pth carries `temp` (=0.0157); the reparam'd model has none, so read it here.
            if "temp" in state:
                self.teacher_temp = float(state["temp"])
            msg = model.load_state_dict(state, strict=False)
            print("teacher load:", msg)
            critical_prefixes = tuple(p for k in self.keep for p in CRITICAL_PREFIXES[k])
            critical_missing = [k for k in msg.missing_keys if k.startswith(critical_prefixes)]
            if critical_missing:
                raise RuntimeError(
                    f"Teacher checkpoint is missing {len(critical_missing)} critical keys "
                    f"for keep={self.keep} (architecture mismatch vs vit='{vit}', bert='{bert}'?). "
                    f"First few: {critical_missing[:5]}"
                )

        self.teacher_scale = 1.0 / self.teacher_temp

        # keep 합집합 외 서브모듈/버퍼 해제 (로드 후 → .to(device) 전이므로 GPU엔 안 올라감)
        needed = set().union(*(NEEDS[k] for k in self.keep))
        for attr in sorted(ALL_SUBMODULES - needed):
            setattr(model, attr, None)
        for buf in ("image_queue", "text_queue", "queue_ptr"):
            setattr(model, buf, None)

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        self.model = model
        self.tokenizer = model.tokenizer

    def to(self, device):
        self.model.to(device)
        return self

    def _require(self, path):
        if path not in self.keep:
            raise RuntimeError(
                f"OnlineTeacher was built with keep={self.keep}; '{path}' path is unavailable"
            )

    @torch.no_grad()
    def encode_image(self, image):
        """Compute teacher visual_encoder(image) fresh — NOT a cache, recomputes every
        call. Callers needing this for more than one of itc_feats/lm_logits/itm_soft in
        the same step should call this ONCE and pass the result via image_embeds= to
        each, to avoid redundant ViT forwards on the identical input."""
        device = image.device
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            return self.model.visual_encoder(image)

    @torch.no_grad()
    def itc_feats(self, image, caption, image_embeds=None):
        self._require("itc")
        device = image.device
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            if image_embeds is None:
                image_embeds = self.model.visual_encoder(image)
            img_feat = F.normalize(self.model.vision_proj(image_embeds[:, 0, :]), dim=-1)
            text = self.tokenizer(caption, padding="max_length", truncation=True,
                                  max_length=30, return_tensors="pt").to(device)
            text_output = self.model.text_encoder(text.input_ids,
                                                  attention_mask=text.attention_mask,
                                                  return_dict=True, mode="text")
            txt_feat = F.normalize(self.model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
        return img_feat, txt_feat

    @torch.no_grad()
    def lm_logits(self, image, caption, image_embeds=None):
        """Teacher-forced decoder logits for the same augmented batch.
        학생 forward의 LM 경로와 동일한 토크나이즈/BOS 규칙 — forward 쪽에서
        decoder_input_ids 일치를 assert하므로 규칙이 어긋나면 즉시 검출된다."""
        self._require("lm")
        device = image.device
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            if image_embeds is None:
                image_embeds = self.model.visual_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
            text = self.tokenizer(caption, padding="max_length", truncation=True,
                                  max_length=30, return_tensors="pt").to(device)
            decoder_input_ids = text.input_ids.clone()
            decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
            out = self.model.text_decoder(decoder_input_ids,
                                          attention_mask=text.attention_mask,
                                          encoder_hidden_states=image_embeds,
                                          encoder_attention_mask=image_atts,
                                          return_dict=True)   # labels 없음 → logits만
        return out.logits, decoder_input_ids

    @torch.no_grad()
    def itm_soft(self, image, enc_input_ids, attention_mask,
                 neg_idx_img, neg_idx_txt, temp, image_embeds=None):
        """Teacher ITM match distribution over the SAME 3B triplets the student built.
        enc_input_ids/attention_mask come from the student (identical tokenizer, pos-0
        already set to enc_token_id). neg_idx_img/neg_idx_txt: length-B int sequences.
        Returns softmax(teacher_itm_logits / temp) as [3B, 2] (no grad)."""
        self._require("itm")
        device = image.device
        bs = image.size(0)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            if image_embeds is None:
                image_embeds = self.model.visual_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
            img_neg = torch.stack([image_embeds[neg_idx_img[b]] for b in range(bs)])
            txt_neg = torch.stack([enc_input_ids[neg_idx_txt[b]] for b in range(bs)])
            txt_neg_atts = torch.stack([attention_mask[neg_idx_txt[b]] for b in range(bs)])
            text_ids_all = torch.cat([enc_input_ids, enc_input_ids, txt_neg], dim=0)
            text_atts_all = torch.cat([attention_mask, attention_mask, txt_neg_atts], dim=0)
            image_embeds_all = torch.cat([image_embeds, img_neg, image_embeds], dim=0)
            image_atts_all = torch.cat([image_atts, image_atts, image_atts], dim=0)
            out = self.model.text_encoder(text_ids_all,
                                          attention_mask=text_atts_all,
                                          encoder_hidden_states=image_embeds_all,
                                          encoder_attention_mask=image_atts_all,
                                          return_dict=True)
            logits = self.model.itm_head(out.last_hidden_state[:, 0, :]).float()
        return F.softmax(logits / temp, dim=1)
