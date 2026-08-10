"""GPU smoke for arm-C ITM target-mix distillation (Task 8, Step 2).

Exercises the real teacher checkpoint (model_large.pth, NOT the empty-checkpoint
stand-ins used by unit tests) end-to-end on GPU: real forward + backward with
gradients, arm C (neg_source='teacher', soft_weight=0.4 — the richest arm, using
both teacher-guided negative selection AND teacher 3B ITM soft scoring).

Run:
    CUDA_VISIBLE_DEVICES=1 python scratchpad/smoke_itm_mix.py

Landmine: model_large.pth's image_queue/text_queue are [256, 57600]; the teacher
MUST be built with queue_size=57600 or load_state_dict raises a shape RuntimeError
(strict=False does not skip shape mismatches on existing keys).
"""
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distillation.online_teacher import OnlineTeacher       # noqa: E402
from models.blip_pretrain import blip_pretrain              # noqa: E402

CKPT = "/home/minwoo/Distillation_Project/output/official_pretrain_checkpoint/model_large.pth"

CAPTIONS = [
    "a dog running along the sandy beach at sunset",
    "two people sitting at a wooden table drinking coffee",
    "a red car parked in front of a tall glass building",
    "a plate of food with vegetables rice and grilled chicken",
    "a young child playing with a bright colorful ball",
    "an old stone bridge crossing a quiet green river",
    "a small cat sleeping on a soft folded blanket",
    "a group of friends hiking up a rocky mountain trail",
    "a cup of hot coffee next to an open laptop",
    "several birds flying over a wide green field",
    "a busy city street with many cars and bright lights",
    "a woman reading a paperback book in a sunny park",
    "a wooden bowl of fresh fruit on a kitchen counter",
    "a small white boat floating on calm blue water",
    "a crowded train arriving at a large station platform",
    "a bright modern kitchen with clean white cabinets",
]


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — this smoke must run on GPU")

    device = torch.device("cuda")
    torch.cuda.set_device(0)
    print(f"visible device: {torch.cuda.get_device_name(0)} "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    torch.manual_seed(0)

    # The real train loop runs under DDP; update_train_state=True hits the momentum
    # queue enqueue path (concat_all_gather -> torch.distributed.get_world_size()),
    # which needs an initialized process group. Faithfully reproduce that with a
    # single-process (world_size=1) group so the enqueue path actually runs, rather
    # than disabling update_train_state (which the brief requires to stay True).
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    dist.init_process_group(backend="nccl", rank=0, world_size=1)

    print("=== building LARGE teacher (keep=('itc','itm'), queue_size=57600) ===")
    teacher = OnlineTeacher(
        checkpoint=CKPT,
        image_size=224, vit="large", bert="base",
        queue_size=57600, keep=("itc", "itm"),
    )
    teacher.to(device)
    print(f"teacher.teacher_scale = {teacher.teacher_scale:.4f}  (expected ~63.9, +/-1)")

    print("=== building BASE student (queue_size=240) ===")
    student = blip_pretrain(image_size=224, vit="base", my_bert_size="base", queue_size=240)
    student.to(device)
    student.train()

    B = 8                       # 240 % 8 == 0 (student queue enqueue assert)
    STEPS = 5
    itm_mix = {
        "neg_source": "teacher",
        "soft_weight": 0.4,
        "temp": 1.0,
        "sel_scale": teacher.teacher_scale,
    }

    all_finite = True
    itm_values = []
    for step in range(STEPS):
        image = torch.randn(B, 3, 224, 224, device=device)
        caption = CAPTIONS[:B]

        teacher_img_feat, teacher_text_feat = teacher.itc_feats(image, caption)

        loss_ita, loss_itm, loss_lm, loss_itc_kd, loss_lm_kd = student(
            image, caption, alpha=0.4, update_train_state=True,
            teacher_img_feat=teacher_img_feat, teacher_text_feat=teacher_text_feat,
            online_teacher=teacher, itm_mix=itm_mix,
        )
        loss = loss_ita + loss_itm + loss_lm
        loss.backward()

        itm_val = loss_itm.item()
        finite = bool(torch.isfinite(loss_itm))
        all_finite = all_finite and finite
        itm_values.append(itm_val)
        print(f"[step {step}] loss_itm={itm_val:.6f} finite={finite} | "
              f"ita={loss_ita.item():.4f} lm={loss_lm.item():.4f} total={loss.item():.4f}")

        student.zero_grad(set_to_none=True)

    print("=== smoke summary ===")
    print(f"loss_itm values: {[round(v, 6) for v in itm_values]}")
    print(f"all loss_itm finite: {all_finite}")
    print(f"teacher.teacher_scale = {teacher.teacher_scale:.4f}")
    scale_ok = abs(teacher.teacher_scale - 63.9) <= 1.0
    print(f"teacher_scale within 63.9 +/- 1: {scale_ok}")
    assert all_finite, "NON-FINITE loss_itm encountered — investigate before real run"
    assert scale_ok, "teacher_scale off — checkpoint temp not read as expected"
    print("SMOKE OK")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
