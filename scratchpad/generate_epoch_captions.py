"""Generate only missing COCO-val captions; scoring stays a separate CPU job."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from data import create_dataset
from models.blip import blip_decoder, load_checkpoint
from train_caption import evaluate


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "base": (
        ROOT / "output/pt_checkpoint_base_logitscale_nodecay",
        ROOT / "output/caption_zeroshot/base_nodecay_ep02/config.yaml",
        "base_nodecay",
    ),
    "lm": (
        ROOT / "output/pt_smallreg_minilm_lm_distill",
        ROOT / "output/caption_zeroshot/small_distill_ep02/config.yaml",
        "small_distill",
    ),
    "small": (
        ROOT / "output/pt_smallreg_minilm_baseline",
        ROOT / "output/caption_zeroshot/small_solo_ep02/config.yaml",
        "small_solo",
    ),
}


def missing_epochs(epochs, existing):
    return [epoch for epoch in epochs if epoch not in existing]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint_dir, template, output_prefix = MODELS[args.model]
    config = yaml.safe_load(template.read_text())
    existing = {
        epoch for epoch in range(20)
        if (ROOT / f"output/caption_zeroshot/{output_prefix}_ep{epoch:02d}/result/val_epoch0.json").exists()
    }
    epochs = missing_epochs(range(20), existing)
    print(f"{args.model}: generating epochs {epochs}", flush=True)

    _, val_dataset, _ = create_dataset("caption_coco", config)
    loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False,
                        num_workers=4, pin_memory=True)
    model = blip_decoder(image_size=config["image_size"], vit=config["vit"],
                         vit_grad_ckpt=False, vit_ckpt_layer=0,
                         prompt=config["prompt"], my_bert_size=config["my_bert_size"]).to(args.device)

    for epoch in epochs:
        checkpoint = checkpoint_dir / f"checkpoint_{epoch:02d}.pth"
        model, message = load_checkpoint(model, str(checkpoint))
        if message.missing_keys:
            print(f"epoch {epoch:02d}: {len(message.missing_keys)} unused/missing keys", flush=True)
        predictions = evaluate(model, loader, torch.device(args.device), config)
        result_dir = ROOT / f"output/caption_zeroshot/{output_prefix}_ep{epoch:02d}/result"
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "val_epoch0.json").write_text(json.dumps(predictions))
        print(f"epoch {epoch:02d}: saved {len(predictions)} captions", flush=True)


if __name__ == "__main__":
    main()
