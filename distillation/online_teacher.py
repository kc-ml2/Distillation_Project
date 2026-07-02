import torch
import torch.nn.functional as F

from models.blip_pretrain import blip_pretrain

# 경로별 필요 서브모듈. momentum 4개·itm_head·큐는 학습 전용 장치라 어떤 keep에서도 해제.
NEEDS = {
    "itc": {"visual_encoder", "text_encoder", "vision_proj", "text_proj"},
    "lm": {"visual_encoder", "text_decoder"},
}
ALL_SUBMODULES = {
    "visual_encoder", "text_encoder", "vision_proj", "text_proj", "text_decoder",
    "visual_encoder_m", "text_encoder_m", "vision_proj_m", "text_proj_m", "itm_head",
}
CRITICAL_PREFIXES = {
    "itc": ("visual_encoder.", "text_encoder.", "vision_proj.", "text_proj."),
    "lm": ("visual_encoder.", "text_decoder."),
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

        model = blip_pretrain(image_size=image_size, vit=vit, my_bert_size=bert,
                              queue_size=queue_size)
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
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
    def itc_feats(self, image, caption):
        self._require("itc")
        device = image.device
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            image_embeds = self.model.visual_encoder(image)
            img_feat = F.normalize(self.model.vision_proj(image_embeds[:, 0, :]), dim=-1)
            text = self.tokenizer(caption, padding="max_length", truncation=True,
                                  max_length=30, return_tensors="pt").to(device)
            text_output = self.model.text_encoder(text.input_ids,
                                                  attention_mask=text.attention_mask,
                                                  return_dict=True, mode="text")
            txt_feat = F.normalize(self.model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
        return img_feat, txt_feat
