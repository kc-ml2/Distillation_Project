import torch
import torch.nn.functional as F

from models.blip_pretrain import blip_pretrain


class OnlineTeacher:
    """Frozen BLIP teacher served live in the train loop. Provides ITC features
    for the same augmented batch the student sees. Only the ITC path is kept;
    momentum encoders, decoder, ITM head and queues are freed after load.
    """

    def __init__(self, checkpoint, image_size, vit="large", bert="base", queue_size=240):
        model = blip_pretrain(image_size=image_size, vit=vit, my_bert_size=bert,
                              queue_size=queue_size)
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            msg = model.load_state_dict(state, strict=False)
            print("teacher load:", msg)

        # free submodules/buffers itc_feats never uses (~halves teacher memory)
        for attr in ("visual_encoder_m", "text_encoder_m", "vision_proj_m",
                     "text_proj_m", "text_decoder", "itm_head"):
            if hasattr(model, attr):
                setattr(model, attr, None)
        for buf in ("image_queue", "text_queue", "queue_ptr"):
            if hasattr(model, buf):
                setattr(model, buf, None)

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        self.model = model
        self.tokenizer = model.tokenizer

    def to(self, device):
        self.model.to(device)
        return self

    @torch.no_grad()
    def itc_feats(self, image, caption):
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
