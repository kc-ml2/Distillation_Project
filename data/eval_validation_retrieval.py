import torch
from torch.utils.data import DataLoader

import utils
import eval_validation_tool
from data.coco_karpathy_dataset import coco_karpathy_retrieval_eval
from data.eval_validation_loss import build_val_loss_transform


def build_coco_karpathy_retrieval_val_loader(config):
    ann_root = config["val_retrieval_ann_root"]
    image_root = config.get("val_retrieval_image_root", config["image_root_coco"])
    split = config.get("val_retrieval_split", "val")

    image_size = config["image_size"]
    batch_size = config.get("val_retrieval_batch_size", config["batch_size"])
    num_workers = config.get("val_retrieval_num_workers", 4)

    transform = build_val_loss_transform(image_size)

    dataset = coco_karpathy_retrieval_eval(
        transform=transform,
        image_root=image_root,
        ann_root=ann_root,
        split=split,
    )

    # 모든 rank가 전체 데이터셋을 로드 (DistributedSampler 없음) - evaluate_retrieval_itm이
    # 내부적으로 rank별 샤딩을 직접 수행하기 때문에 train_retrieval.py의 val/test loader와
    # 동일한 패턴을 따른다.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return loader


class RetrievalValRunner:
    def __init__(self, data_loader, device, config, writer=None):
        self.data_loader = data_loader
        self.device = torch.device(device)
        self.config = config
        self.writer = writer

    def val_retrieval_during_train(self, model_without_ddp, epoch, iteration, global_step):
        if global_step <= 0:
            return {}

        results = {}

        itc_interval = self.config.get("val_retrieval_itc_interval_steps", 0)
        if itc_interval and itc_interval > 0 and global_step % itc_interval == 0:
            results["itc"] = self._run_tier(
                model_without_ddp, "itc", global_step,
                header=f"Val Retrieval ITC: [epoch {epoch} | step {iteration} | global {global_step}]",
            )

        # ITM은 비싸서(epoch당 ~17% tax 실측) global_step 배수로 반복하지 않고, iteration(에폭 로컬
        # step, 매 에폭 0부터 재시작)이 이 값과 같아지는 순간 - 즉 에폭당 정확히 1번만 돈다.
        itm_mid_step = self.config.get("val_retrieval_itm_interval_steps", 0)
        if itm_mid_step and itm_mid_step > 0 and iteration == itm_mid_step:
            results["itm"] = self._run_tier(
                model_without_ddp, "itm", global_step,
                header=f"Val Retrieval ITM Mid-epoch: [epoch {epoch} | step {iteration} | global {global_step}]",
            )

        return results

    def run_epoch_end(self, model_without_ddp, epoch, global_step):
        results = {}

        if self.config.get("val_retrieval_itc_epoch_end", True):
            results["itc"] = self._run_tier(
                model_without_ddp, "itc", global_step,
                header=f"Val Retrieval ITC Epoch: [{epoch}]",
            )

        if self.config.get("val_retrieval_itm_epoch_end", True):
            results["itm"] = self._run_tier(
                model_without_ddp, "itm", global_step,
                header=f"Val Retrieval ITM Epoch: [{epoch}]",
            )

        return results

    def _run_tier(self, model_without_ddp, tier, global_step, header):
        was_training = model_without_ddp.training
        model_without_ddp.eval()

        try:
            if tier == "itc":
                scores_i2t, scores_t2i = eval_validation_tool.evaluate_retrieval_itc(
                    model_without_ddp, self.data_loader, self.device, self.config,
                )
            elif tier == "itm":
                scores_i2t, scores_t2i = eval_validation_tool.evaluate_retrieval_itm(
                    model_without_ddp, self.data_loader, self.device, self.config,
                )
            else:
                raise ValueError(f"Unknown retrieval tier: {tier}")

            metrics = eval_validation_tool.itm_eval(
                scores_i2t, scores_t2i,
                self.data_loader.dataset.txt2img,
                self.data_loader.dataset.img2txt,
            )

            if utils.is_main_process():
                print(f"{header} r_mean={metrics['r_mean']:.2f}")

            self._write_tensorboard(tier, metrics, global_step)

            return metrics
        finally:
            if was_training:
                model_without_ddp.train()
            else:
                model_without_ddp.eval()

    def _write_tensorboard(self, tier, metrics, global_step):
        if self.writer is None or not utils.is_main_process():
            return

        for key, value in metrics.items():
            self.writer.add_scalar(f"val_retrieval_{tier}/{key}", value, global_step)


def build_pretrain_retrieval_val_runner(config, device, writer=None):
    if not config.get("val_retrieval_enabled", False):
        return None

    data_loader = build_coco_karpathy_retrieval_val_loader(config)

    return RetrievalValRunner(
        data_loader=data_loader,
        device=device,
        config=config,
        writer=writer,
    )
