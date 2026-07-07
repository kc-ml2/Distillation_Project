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

    if utils.is_main_process():
        print(f"[arch] vit={config['vit']} bert={config['my_bert_size']} "
              f"image_size={config['image_size']}")

    model = blip_pretrain(
        image_size=config["image_size"],
        vit=config["vit"],
        vit_grad_ckpt=config["vit_grad_ckpt"],
        vit_ckpt_layer=config["vit_ckpt_layer"],
        queue_size=config["queue_size"],
        my_bert_size=config["my_bert_size"],
        # --checkpoint로 직후 전량 덮으므로 백본 사전 초기화 불필요.
        # large(in21k) 사전초기화는 timm 1.x에서 깨져 있어 반드시 꺼야 로드된다.
        init_backbone_weights=False,
    )

    model = model.to(device)

    if not args.checkpoint:
        raise ValueError("--checkpoint is required")

    model = utils.load_model_weights_only(model, args.checkpoint)

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
    # arch 오버라이드 (없으면 config 값). 티처(large/base)를 학생 config 그대로 평가할 때 사용.
    parser.add_argument("--vit", default=None, help="override config vit (e.g. large)")
    parser.add_argument("--bert", default=None, help="override config my_bert_size (e.g. base)")
    parser.add_argument("--image_size", default=None, type=int, help="override config image_size")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--distributed", default=True, type=bool)
    parser.add_argument("--tensorboard", action="store_true")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # CLI arch 오버라이드를 config에 주입 → val runner(transform)와 모델 빌드가 같은 값을 공유
    if args.vit is not None:
        config["vit"] = args.vit
    if args.bert is not None:
        config["my_bert_size"] = args.bert
    if args.image_size is not None:
        config["image_size"] = args.image_size

    if args.output_dir is None:
        args.output_dir = config.get("output_dir", "./output_official_val_loss")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args, config)