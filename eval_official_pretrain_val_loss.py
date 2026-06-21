'''
Evaluate BLIP pretrain validation losses from an official pretrained checkpoint.

This script:
  - loads BLIP pretrain model
  - loads model weights only
  - does not create optimizer
  - does not train
  - runs COCO Karpathy validation loss runner
'''

import argparse
import os
import json
import datetime
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import yaml

from torch.utils.tensorboard import SummaryWriter

import utils
from models.blip_pretrain import blip_pretrain
from data import eval_validation_loss


def load_model_weights_only(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean_state_dict[k] = v

    msg = model.load_state_dict(clean_state_dict, strict=False)

    if utils.is_main_process():
        print(f"Loaded model weights from: {checkpoint_path}")
        print(msg)

    return model


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)
    cudnn.benchmark = True

    if utils.is_main_process():
        print("Creating official BLIP pretrain validation-loss evaluator")

    writer = None
    if args.tensorboard and utils.is_main_process():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_log_dir = os.path.join(args.output_dir, "tensorboard", f"{timestamp}__official_pretrain_val_loss")
        writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"[TensorBoard] log_dir: {tb_log_dir}")

    if utils.is_main_process():
        print("Creating validation loss runner")

    val_loss_runner = eval_validation_loss.build_pretrain_val_loss_runner(
        config=config,
        device=device,
        writer=writer,
    )

    if val_loss_runner is None:
        raise RuntimeError("val_loss_runner is None. Set val_loss_enabled: true in config.")

    if utils.is_main_process():
        print("Creating model")

    model = blip_pretrain(
        image_size=config["image_size"],
        vit=config["vit"],
        vit_grad_ckpt=config["vit_grad_ckpt"],
        vit_ckpt_layer=config["vit_ckpt_layer"],
        queue_size=config["queue_size"],
        my_bert_size=config["my_bert_size"],
    )

    model = model.to(device)

    if not args.checkpoint:
        raise ValueError("--checkpoint is required")

    model = load_model_weights_only(model, args.checkpoint)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    if utils.is_main_process():
        print("Start official checkpoint validation loss evaluation")

    # train_loader_len is only needed for alpha_mode=current.
    # For official checkpoint evaluation, alpha_mode=fixed or zero is cleaner.
    train_loader_len_for_alpha = int(config.get("official_eval_train_loader_len", 1))
    global_step = int(config.get("official_eval_global_step", 0))

    val_stats = val_loss_runner.run_epoch_end(
        model=model,
        epoch=0,
        global_step=global_step,
        train_loader_len=train_loader_len_for_alpha,
    )

    if utils.is_main_process():
        print("Official pretrain checkpoint validation loss stats:")
        print(json.dumps(val_stats, indent=2))

        out_path = os.path.join(args.output_dir, "official_pretrain_val_loss.json")
        with open(out_path, "w") as f:
            json.dump(val_stats, f, indent=2)
        print(f"Saved stats to: {out_path}")

    if args.distributed:
        dist.barrier()

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/pretrain.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--distributed", default=True, type=bool)
    parser.add_argument("--tensorboard", action="store_true")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = config.get("output_dir", "./output_official_val_loss")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args, config)