"""Phase-0: run the frozen BLIP-large teacher over the pretrain dataset (index order,
deterministic transform) and cache per-sample 256-d ITC features.

Usage:
  python build_teacher_cache.py --config ./configs/pretrain.yaml [--batch_size 64] [--device cuda]
"""
import argparse

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from data.pretrain_dataset import pretrain_dataset
from distillation.teacher_cache import dataset_signature, build_cache


def load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def build_teacher(config, device):
    """Construct + load the frozen BLIP-large teacher.

    NOTE: this must match the user's trained BLIP-large checkpoint. Adjust vit/bert
    sizes here if the teacher differs from (vit='large', my_bert_size='base').
    """
    from models.blip_pretrain import blip_pretrain
    teacher = blip_pretrain(
        image_size=config["image_size"],
        vit="large",
        my_bert_size="base",
        queue_size=config["queue_size"],
    )
    ckpt = torch.load(config["teacher"]["checkpoint"], map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    msg = teacher.load_state_dict(state, strict=False)
    print("teacher load:", msg)
    teacher.eval().to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def make_feat_fn(teacher, device):
    @torch.no_grad()
    def fn(images, captions):
        images = images.to(device)
        image_embeds = teacher.visual_encoder(images)
        img_f = F.normalize(teacher.vision_proj(image_embeds[:, 0, :]), dim=-1)
        text = teacher.tokenizer(captions, padding="max_length", truncation=True,
                                 max_length=30, return_tensors="pt").to(device)
        text_output = teacher.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                           return_dict=True, mode="text")
        txt_f = F.normalize(teacher.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
        return img_f, txt_f
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./configs/pretrain.yaml")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    config = load_yaml(args.config)
    device = torch.device(args.device)

    # deterministic transform (no random aug) — matches student input when pretrain_train_aug=false
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711))
    transform_test = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"]),
                          interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(), normalize,
    ])

    dataset = pretrain_dataset(
        ann_file=config["train_file"], laion_path=config["laion_path"],
        img_root_coco=config["image_root_coco"], img_root_vg=config["image_root_vg"],
        transform=transform_test,
    )
    n = len(dataset)
    sig = dataset_signature(config["train_file"], n)
    feat_fn = make_feat_fn(build_teacher(config, device), device)

    out_dir = config["distill"]["itc"]["cache_dir"]
    print(f"building teacher cache: N={n} -> {out_dir}")
    build_cache(feat_fn, dataset, out_dir, dim=256, signature=sig,
                teacher_id=config["teacher"]["checkpoint"], batch_size=args.batch_size)
    print("done")


if __name__ == "__main__":
    main()
