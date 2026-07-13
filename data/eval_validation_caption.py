import os
import json

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import utils
from data.coco_karpathy_dataset import coco_karpathy_caption_eval
from data.eval_validation_loss import build_val_loss_transform
from data.utils import coco_caption_eval


def parse_cpu_list(spec):
    """'0-31' 또는 '0-3,8,10-12' -> 정렬 불필요한 코어 집합. None/'' -> None."""
    if not spec:
        return None
    cores = set()
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-')
            cores.update(range(int(a), int(b) + 1))
        else:
            cores.add(int(part))
    return cores or None


class cpu_affinity:
    """채점 구간 동안 프로세스(및 SPICE의 java 자식)를 지정 코어에 제한하고
    종료 시 원복. cores=None 또는 플랫폼 미지원이면 no-op."""

    def __init__(self, cores):
        self.cores = cores
        self.orig = None

    def __enter__(self):
        if self.cores and hasattr(os, "sched_setaffinity"):
            self.orig = os.sched_getaffinity(0)
            os.sched_setaffinity(0, self.cores)
        return self

    def __exit__(self, *exc):
        if self.orig is not None:
            os.sched_setaffinity(0, self.orig)
        return False


def build_caption_val_loader(config):
    image_root = config["val_caption_image_root"]
    ann_root = config["val_caption_ann_root"]
    split = config.get("val_caption_split", "val")
    image_size = config["image_size"]
    batch_size = config.get("val_caption_batch_size", 32)
    num_workers = config.get("val_caption_num_workers", 4)

    transform = build_val_loss_transform(image_size)
    dataset = coco_karpathy_caption_eval(transform, image_root, ann_root, split)

    sampler = None
    if utils.is_dist_avail_and_initialized() and utils.get_world_size() > 1:
        sampler = DistributedSampler(dataset, num_replicas=utils.get_world_size(),
                                     rank=utils.get_rank(), shuffle=False)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        shuffle=False, num_workers=num_workers,
                        pin_memory=True, drop_last=False)
    return loader


class CaptionValRunner:
    def __init__(self, data_loader, device, config, writer=None):
        self.data_loader = data_loader
        self.device = torch.device(device)
        self.config = config
        self.writer = writer
        self.cpu_cores = parse_cpu_list(config.get("caption_score_cpu_list"))

    def val_caption_during_train(self, model_without_ddp, epoch, iteration, global_step):
        mid = self.config.get("val_caption_mid_interval_steps", 0)
        if mid and iteration == mid:
            return self._run(model_without_ddp, global_step, with_spice=False,
                             header=f"Val Caption Mid: [epoch {epoch} | step {iteration} | global {global_step}]")
        return {}

    def run_epoch_end(self, model_without_ddp, epoch, global_step):
        if not self.config.get("val_caption_epoch_end", True):
            return {}
        with_spice = self.config.get("val_caption_use_spice", True)
        return self._run(model_without_ddp, global_step, with_spice=with_spice,
                         header=f"Val Caption Epoch: [{epoch}]")

    def _run(self, model_without_ddp, global_step, with_spice, header):
        was_training = model_without_ddp.training
        model_without_ddp.eval()
        try:
            preds = self._generate(model_without_ddp)
            all_preds = self._gather(preds)
            if not utils.is_main_process():
                return {}
            metrics = self._score(all_preds, global_step, with_spice)
            msg = f"{header} CIDEr={metrics.get('CIDEr', float('nan')):.3f}"
            if with_spice and "SPICE" in metrics:
                msg += f" SPICE={metrics['SPICE']:.3f}"
            print(msg)
            self._write_tensorboard(metrics, global_step)
            return metrics
        finally:
            if was_training:
                model_without_ddp.train()
            else:
                model_without_ddp.eval()

    @torch.no_grad()
    def _generate(self, model_without_ddp):
        cfg = self.config
        result = []
        for image, image_id in self.data_loader:
            image = image.to(self.device)
            captions = model_without_ddp.generate(
                image, sample=False,
                num_beams=cfg.get("caption_num_beams", 3),
                max_length=cfg.get("caption_max_length", 20),
                min_length=cfg.get("caption_min_length", 5),
                prompt=cfg.get("caption_prompt", ""))
            for cap, iid in zip(captions, image_id):
                result.append({"image_id": int(iid), "caption": cap})
        return result

    def _gather(self, preds):
        if not (utils.is_dist_avail_and_initialized() and utils.get_world_size() > 1):
            return preds
        gathered = [None] * utils.get_world_size()
        dist.all_gather_object(gathered, preds)
        if not utils.is_main_process():
            return None
        merged, seen = [], set()
        for part in gathered:
            for item in part:
                if item["image_id"] not in seen:
                    seen.add(item["image_id"])
                    merged.append(item)
        return merged

    def _score(self, preds, global_step, with_spice):
        out_dir = self.config.get("output_dir", ".")
        res_file = os.path.join(out_dir, f"caption_val_step{global_step}.json")
        with open(res_file, "w") as f:
            json.dump(preds, f)
        with cpu_affinity(self.cpu_cores):
            coco_eval = coco_caption_eval(
                self.config["val_caption_gt_root"], res_file,
                self.config.get("val_caption_split", "val"), use_spice=with_spice)
        return coco_eval.eval

    def _write_tensorboard(self, metrics, global_step):
        if self.writer is None or not utils.is_main_process():
            return
        for key, value in metrics.items():
            self.writer.add_scalar(f"val_caption/{key}", value, global_step)


def build_pretrain_caption_val_runner(config, device, writer=None):
    if not config.get("val_caption_enabled", False):
        return None
    data_loader = build_caption_val_loader(config)
    return CaptionValRunner(data_loader, device, config, writer)
