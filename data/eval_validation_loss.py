import os
import json
import contextlib

from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision import transforms

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import utils


ImageFile.LOAD_TRUNCATED_IMAGES = True


class CocoKarpathyValLossDataset(Dataset):
    def __init__(self, ann_file, image_root, transform, caption_mode="first"):
        self.ann_file = ann_file
        self.image_root = image_root
        self.transform = transform
        self.caption_mode = caption_mode

        with open(ann_file, "r") as f:
            annotations = json.load(f)

        self.samples = self._build_samples(annotations)

    def _build_samples(self, annotations):
        samples = []

        for ann in annotations:
            image_rel_path = ann["image"]
            captions = ann["caption"]

            if isinstance(captions, str):
                captions = [captions]

            if self.caption_mode == "first":
                samples.append({ # 딕셔너리를 사용해서 패스 저장
                    "image": image_rel_path,
                    "caption": captions[0],
                })

            elif self.caption_mode == "all":
                for caption in captions:
                    samples.append({
                        "image": image_rel_path,
                        "caption": caption,
                    })

            else:
                raise ValueError(f"Unknown caption_mode: {self.caption_mode}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image_path = os.path.join(
            self.image_root,
            sample["image"].lstrip("/")
        )

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        caption = " ".join(sample["caption"].split())

        return image, caption


def val_loss_collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    captions = list(captions)
    return images, captions


def build_val_loss_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def build_coco_karpathy_val_loss_loader(config):
    ann_file = config["val_loss_file"]
    image_root = config.get("val_loss_image_root", config["image_root_coco"])

    image_size = config["image_size"]
    batch_size = config.get("val_loss_batch_size", config["batch_size"])
    num_workers = config.get("val_loss_num_workers", 4)
    caption_mode = config.get("val_loss_caption_mode", "first")
    drop_last = config.get("val_loss_drop_last", True)
    pin_memory = config.get("val_loss_pin_memory", True)

    transform = build_val_loss_transform(image_size)

    dataset = CocoKarpathyValLossDataset(
        ann_file=ann_file,
        image_root=image_root,
        transform=transform,
        caption_mode=caption_mode,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=utils.get_world_size(),
        rank=utils.get_rank(),
        shuffle=False,
        drop_last=drop_last,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=val_loss_collate_fn,
    )

    return loader


def compute_val_alpha(config, global_step, train_loader_len):
    alpha_mode = config.get("val_loss_alpha_mode", "current")
    base_alpha = float(config["alpha"])

    if alpha_mode == "fixed":
        return base_alpha

    if alpha_mode == "zero":
        return 0.0

    if alpha_mode != "current":
        raise ValueError(f"Unknown val_loss_alpha_mode: {alpha_mode}")

    if train_loader_len is None or train_loader_len <= 0:
        return base_alpha

    return base_alpha * min(1.0, global_step / (2.0 * train_loader_len))


def capture_torch_rng_state():
    state = {
        "cpu": torch.get_rng_state(),
        "cuda": None,
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def restore_torch_rng_state(state):
    torch.set_rng_state(state["cpu"])

    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class PretrainValLossRunner:
    def __init__(self, data_loader, device, config, writer=None):
        self.data_loader = data_loader
        self.device = torch.device(device)
        self.config = config
        self.writer = writer

        self.print_freq = config.get("val_loss_print_freq", 50)
        self.restore_rng = config.get("val_loss_restore_rng", True)
        self.use_amp = config.get("val_loss_amp", True)

    def val_loss_during_train( # 이건 좀 바꾸자고
        self,
        model,
        epoch,
        iteration,
        global_step,
        train_loader_len,
    ):
        interval = self.config.get("val_loss_interval_steps", 0)

        if interval is None or interval <= 0:
            return None

        if global_step <= 0:
            return None

        if global_step % interval != 0:
            return None

        max_batches = self.config.get("val_loss_max_batches", 20)

        return self.evaluate(
            model=model,
            epoch=epoch,
            global_step=global_step,
            train_loader_len=train_loader_len,
            max_batches=max_batches,
            header=f"Val Loss Step: [epoch {epoch} | step {iteration} | global {global_step}]",
        )

    def run_epoch_end(
        self,
        model,
        epoch,
        global_step,
        train_loader_len,
    ):
        if not self.config.get("val_loss_epoch_end", True):
            return {}

        max_batches = self.config.get("val_loss_epoch_max_batches", None)

        return self.evaluate(
            model=model,
            epoch=epoch,
            global_step=global_step,
            train_loader_len=train_loader_len,
            max_batches=max_batches,
            header=f"Val Loss Epoch: [{epoch}]",
        )

    @torch.no_grad()
    def evaluate(
        self,
        model,
        epoch,
        global_step,
        train_loader_len,
        max_batches=None,
        header="Val Loss:",
    ):
        if max_batches is not None and max_batches <= 0:
            return {}

        if hasattr(self.data_loader.sampler, "set_epoch"):
            self.data_loader.sampler.set_epoch(epoch)

        alpha = compute_val_alpha(
            config=self.config,
            global_step=global_step,
            train_loader_len=train_loader_len,
        )

        was_training = model.training
        rng_state = capture_torch_rng_state() if self.restore_rng else None

        model.eval()

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("loss_ita", utils.SmoothedValue(window_size=50, fmt="{value:.6f}"))
        metric_logger.add_meter("loss_itm", utils.SmoothedValue(window_size=50, fmt="{value:.6f}"))
        metric_logger.add_meter("loss_lm", utils.SmoothedValue(window_size=50, fmt="{value:.6f}"))
        metric_logger.add_meter("loss_retrieval", utils.SmoothedValue(window_size=50, fmt="{value:.6f}"))
        metric_logger.add_meter("loss_captioning", utils.SmoothedValue(window_size=50, fmt="{value:.6f}"))
        metric_logger.add_meter("loss_total", utils.SmoothedValue(window_size=50, fmt="{value:.6f}"))

        device_type = self.device.type
        amp_enabled = self.use_amp and device_type == "cuda"

        autocast_context = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if amp_enabled
            else contextlib.nullcontext()
        )

        try:
            for batch_idx, (image, caption) in enumerate(
                metric_logger.log_every(self.data_loader, self.print_freq, header)
            ):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                image = image.to(self.device, non_blocking=True)

                with autocast_context:
                    loss_ita, loss_itm, loss_lm, _loss_itc_kd = model(
                        image,
                        caption,
                        alpha=alpha,
                        update_train_state=False,
                    )

                    loss_retrieval = loss_ita + loss_itm
                    loss_captioning = loss_lm
                    loss_total = loss_ita + loss_itm + loss_lm

                metric_logger.update(loss_ita=loss_ita.item())
                metric_logger.update(loss_itm=loss_itm.item())
                metric_logger.update(loss_lm=loss_lm.item())
                metric_logger.update(loss_retrieval=loss_retrieval.item())
                metric_logger.update(loss_captioning=loss_captioning.item())
                metric_logger.update(loss_total=loss_total.item())

            metric_logger.synchronize_between_processes()

            stats = {
                name: meter.global_avg
                for name, meter in metric_logger.meters.items()
            }
            stats["alpha"] = alpha

            self._write_tensorboard(stats, global_step)

            if utils.is_main_process():
                print("Averaged validation stats:", metric_logger.global_avg())

            return {
                key: "{:.6f}".format(value)
                for key, value in stats.items()
            }

        finally:
            if was_training:
                model.train()
            else:
                model.eval()

            if rng_state is not None:
                restore_torch_rng_state(rng_state)

    def _write_tensorboard(self, stats, global_step):
        if self.writer is None:
            return

        if not utils.is_main_process():
            return

        self.writer.add_scalar("loss_val/ita", stats["loss_ita"], global_step)
        self.writer.add_scalar("loss_val/itm", stats["loss_itm"], global_step)
        self.writer.add_scalar("loss_val/lm", stats["loss_lm"], global_step)
        # self.writer.add_scalar("loss/val/retrieval", stats["loss_retrieval"], global_step)
        # self.writer.add_scalar("loss/val/captioning", stats["loss_captioning"], global_step)
        self.writer.add_scalar("loss_val/total", stats["loss_total"], global_step)
        self.writer.add_scalar("val/alpha", stats["alpha"], global_step)


def build_pretrain_val_loss_runner(config, device, writer=None):
    if not config.get("val_loss_enabled", False):
        return None

    data_loader = build_coco_karpathy_val_loss_loader(config)

    return PretrainValLossRunner(
        data_loader=data_loader,
        device=device,
        config=config,
        writer=writer,
    )