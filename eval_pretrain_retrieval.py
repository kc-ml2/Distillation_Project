'''
Evaluate BLIP pretrain retrieval metrics (ITC-only and ITM-rerank) from a
pretrain checkpoint.

This script:
  - loads BLIP pretrain model
  - loads model weights only
  - does not create optimizer
  - does not train
  - runs COCO Karpathy retrieval validation (R@1/R@5/R@10) for both the
    ITC-only and ITM-rerank tiers, on the requested split
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
from data import eval_validation_retrieval


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/pretrain.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--distributed", default=True, type=bool)
    parser.add_argument("--tensorboard", action="store_true")
    return parser


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)
    cudnn.benchmark = True

    if utils.is_main_process():
        print("Creating pretrain retrieval evaluator")

    writer = None
    if args.tensorboard and utils.is_main_process():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_log_dir = os.path.join(args.output_dir, "tensorboard", f"{timestamp}__pretrain_retrieval")
        writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"[TensorBoard] log_dir: {tb_log_dir}")

    # 단독 체크포인트 평가에서는 항상 두 티어 다 돌리고, split은 CLI 인자를 우선한다.
    config = dict(config)
    config["val_retrieval_enabled"] = True
    config["val_retrieval_split"] = args.split
    config["val_retrieval_itc_epoch_end"] = True
    config["val_retrieval_itm_epoch_end"] = True

    retrieval_val_runner = eval_validation_retrieval.build_pretrain_retrieval_val_runner(
        config=config,
        device=device,
        writer=writer,
    )

    if retrieval_val_runner is None:
        raise RuntimeError("retrieval_val_runner is None. Set val_retrieval_enabled: true in config.")

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

    model = utils.load_model_weights_only(model, args.checkpoint)

    if utils.is_main_process():
        print(f"Start checkpoint retrieval evaluation (split={args.split})")

    global_step = int(config.get("official_eval_global_step", 0))

    val_stats = retrieval_val_runner.run_epoch_end(
        model_without_ddp=model,
        epoch=0,
        global_step=global_step,
    )

    if utils.is_main_process():
        print("Checkpoint retrieval evaluation stats:")
        print(json.dumps(val_stats, indent=2))

        out_path = os.path.join(args.output_dir, f"pretrain_retrieval_{args.split}.json")
        with open(out_path, "w") as f:
            json.dump(val_stats, f, indent=2)
        print(f"Saved stats to: {out_path}")

    if args.distributed:
        dist.barrier()

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = config.get("output_dir", "./output_pretrain_retrieval")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args, config)
