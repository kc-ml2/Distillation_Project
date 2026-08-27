import os, time, yaml, torch
import torch.distributed as dist

from models.blip_pretrain import blip_pretrain
import data.eval_validation_caption as evc

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])

torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

config = yaml.safe_load(open("configs/pretrain.yaml"))
config["output_dir"] = "scratchpad"
config["val_caption_use_spice"] = True
device = torch.device(f"cuda:{local_rank}")

model = blip_pretrain(image_size=config["image_size"], vit=config["vit"],
                      vit_grad_ckpt=False, vit_ckpt_layer=0,
                      queue_size=config["queue_size"], my_bert_size=config["my_bert_size"],
                      init_backbone_weights=False)
sd = torch.load("output/pt_smallreg_minilm_baseline/checkpoint_19.pth", map_location="cpu", weights_only=False)
model.load_state_dict(sd.get("model", sd), strict=False)
model.to(device).eval()

runner = evc.build_pretrain_caption_val_runner(config, device, writer=None)

t0 = time.time()
metrics = runner.run_epoch_end(model, epoch=0, global_step=999998)
elapsed = time.time() - t0

if rank == 0:
    print("ELAPSED_SEC:", round(elapsed, 1))
    print("METRICS:", {k: round(float(v), 4) for k, v in metrics.items()})
    assert "CIDEr" in metrics and "SPICE" in metrics
    print("OK")

dist.barrier()
dist.destroy_process_group()
