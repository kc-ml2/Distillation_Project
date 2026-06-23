'''
Sweep COCO retrieval validation (ITC and/or ITM) across a range of saved
pretrain checkpoints (checkpoint_{epoch:02d}.pth), logging every checkpoint's
metrics to a single shared TensorBoard run so they render as continuous
per-metric lines across checkpoints, plus a combined JSON dump.
'''

import argparse
import datetime
import json
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import yaml

from torch.utils.tensorboard import SummaryWriter

import utils
import eval_validation_tool
from models.blip_pretrain import blip_pretrain
from data import eval_validation_retrieval


def checkpoint_path_for_epoch(checkpoint_dir, epoch):
    return os.path.join(checkpoint_dir, f"checkpoint_{epoch:02d}.pth")


def build_custom_scalars_layout(tag_prefix="checkpoint_sweep", include_baseline=False, baseline_name="official"):
    """txt_r1/5/10과 img_r1/5/10을 티어별로 따로 묶고, r_mean은 ITC/ITM을 한 차트에
    합쳐서 보여줌 - 7개 라인을 한 차트에 몰아넣는 대신 자연스러운 묶음으로 분리.
    include_baseline=True면 각 차트에 고정값(수평선)으로 깔리는 외부 기준 체크포인트
    (예: 오리지널 BLIP 공개 가중치) 라인을 같이 묶어서 한눈에 비교 가능하게 함."""

    def recall_tags(tier, metric_prefix):
        tags = [f"{tag_prefix}/{tier}/{metric_prefix}_r1",
                f"{tag_prefix}/{tier}/{metric_prefix}_r5",
                f"{tag_prefix}/{tier}/{metric_prefix}_r10"]
        if include_baseline:
            tags += [f"{tag_prefix}/{tier}/{baseline_name}_{metric_prefix}_r1",
                     f"{tag_prefix}/{tier}/{baseline_name}_{metric_prefix}_r5",
                     f"{tag_prefix}/{tier}/{baseline_name}_{metric_prefix}_r10"]
        return tags

    r_mean_tags = [f"{tag_prefix}/itc/r_mean", f"{tag_prefix}/itm/r_mean"]
    if include_baseline:
        r_mean_tags += [f"{tag_prefix}/itc/{baseline_name}_r_mean", f"{tag_prefix}/itm/{baseline_name}_r_mean"]

    return {
        "ITC": {
            "txt_recall": ["Multiline", recall_tags("itc", "txt")],
            "img_recall": ["Multiline", recall_tags("itc", "img")],
        },
        "ITM": {
            "txt_recall": ["Multiline", recall_tags("itm", "txt")],
            "img_recall": ["Multiline", recall_tags("itm", "img")],
        },
        "Summary": {
            "r_mean": ["Multiline", r_mean_tags],
        },
    }


def write_metrics_to_tensorboard(writer, tier, metrics, global_step, tag_prefix="checkpoint_sweep"):
    for key, value in metrics.items():
        writer.add_scalar(f"{tag_prefix}/{tier}/{key}", value, global_step)


def write_constant_baseline_to_tensorboard(writer, tier, metrics, epoch_range,
                                            tag_prefix="checkpoint_sweep", baseline_name="official"):
    """외부 기준 체크포인트(epoch 개념이 없는 단일 평가)를 sweep의 x축 전체에 같은 값으로
    찍어서 수평선으로 보이게 함."""
    for epoch in epoch_range:
        for key, value in metrics.items():
            writer.add_scalar(f"{tag_prefix}/{tier}/{baseline_name}_{key}", value, epoch)


def save_results(out_path, results):
    """기존 JSON과 tier 단위가 아니라 epoch/baseline 키 단위로 합쳐서 저장 (예: itc만 먼저 돌리고
    나중에 itm을 별도 프로세스로 돌리거나, baseline 하나만 추가할 때 같은 tier의 기존
    epoch 결과가 사라지지 않게)."""
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)

    for tier, tier_results in results.items():
        existing.setdefault(tier, {}).update(tier_results)

    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)


def evaluate_checkpoint(model, loader, device, config, checkpoint_path, tier):
    model = utils.load_model_weights_only(model, checkpoint_path)
    model.eval()

    if tier == "itc":
        scores_i2t, scores_t2i = eval_validation_tool.evaluate_retrieval_itc(model, loader, device, config)
    elif tier == "itm":
        scores_i2t, scores_t2i = eval_validation_tool.evaluate_retrieval_itm(model, loader, device, config)
    else:
        raise ValueError(f"Unknown tier: {tier}")

    return eval_validation_tool.itm_eval(
        scores_i2t, scores_t2i, loader.dataset.txt2img, loader.dataset.img2txt)


def run_sweep(model, loader, device, config, checkpoint_dir, start_epoch, end_epoch, tier,
              writer, results, tag_prefix="checkpoint_sweep"):
    if tier not in ("itc", "itm"):
        raise ValueError(f"Unknown tier: {tier}")

    for epoch in range(start_epoch, end_epoch + 1):
        ckpt_path = checkpoint_path_for_epoch(checkpoint_dir, epoch)
        metrics = evaluate_checkpoint(model, loader, device, config, ckpt_path, tier)

        if utils.is_main_process():
            print(f"[sweep] epoch={epoch} tier={tier} r_mean={metrics['r_mean']:.2f}")
            write_metrics_to_tensorboard(writer, tier, metrics, epoch, tag_prefix)

        results.setdefault(tier, {})[epoch] = metrics

    return results


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/pretrain.yaml")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--end_epoch", required=True, type=int)
    parser.add_argument("--tier", default="both", choices=["itc", "itm", "both"])
    parser.add_argument("--baseline_checkpoint", default=None,
                         help="epoch 개념이 없는 외부 기준 체크포인트(예: 오리지널 BLIP 공개 가중치). "
                              "한 번 평가해서 sweep 전체 x축에 수평선으로 깔아 비교용으로 보여줌.")
    parser.add_argument("--baseline_name", default="official")
    parser.add_argument("--skip_checkpoint_sweep", action="store_true",
                         help="checkpoint_dir의 epoch sweep은 건너뛰고 baseline_checkpoint만 기존 결과에 추가")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--distributed", default=True, type=bool)
    return parser


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)
    cudnn.benchmark = True

    config = dict(config)
    config["val_retrieval_enabled"] = True
    config["val_retrieval_split"] = args.split

    writer = None
    if utils.is_main_process():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_log_dir = os.path.join(args.output_dir, "tensorboard", f"{timestamp}__checkpoint_sweep")
        writer = SummaryWriter(log_dir=tb_log_dir)
        writer.add_custom_scalars(build_custom_scalars_layout(
            include_baseline=args.baseline_checkpoint is not None, baseline_name=args.baseline_name))
        print(f"[TensorBoard] log_dir: {tb_log_dir}")

    if utils.is_main_process():
        print("Creating model")

    model = blip_pretrain(
        image_size=config["image_size"], vit=config["vit"],
        vit_grad_ckpt=config["vit_grad_ckpt"], vit_ckpt_layer=config["vit_ckpt_layer"],
        queue_size=config["queue_size"], my_bert_size=config["my_bert_size"],
    ).to(device)

    loader = eval_validation_retrieval.build_coco_karpathy_retrieval_val_loader(config)

    tiers = ["itc", "itm"] if args.tier == "both" else [args.tier]

    results = {}
    if not args.skip_checkpoint_sweep:
        for tier in tiers:
            results = run_sweep(
                model=model, loader=loader, device=device, config=config,
                checkpoint_dir=args.checkpoint_dir, start_epoch=args.start_epoch, end_epoch=args.end_epoch,
                tier=tier, writer=writer, results=results,
            )

    if args.baseline_checkpoint is not None:
        epoch_range = range(args.start_epoch, args.end_epoch + 1)
        for tier in tiers:
            metrics = evaluate_checkpoint(model, loader, device, config, args.baseline_checkpoint, tier)

            if utils.is_main_process():
                print(f"[sweep] baseline={args.baseline_name} tier={tier} r_mean={metrics['r_mean']:.2f}")
                write_constant_baseline_to_tensorboard(
                    writer, tier, metrics, epoch_range, baseline_name=args.baseline_name)

            results.setdefault(tier, {})[args.baseline_name] = metrics

    if utils.is_main_process():
        out_path = os.path.join(args.output_dir, f"checkpoint_sweep_{args.split}.json")
        save_results(out_path, results)
        print(f"Saved sweep stats to: {out_path}")

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
