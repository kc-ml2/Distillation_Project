#!/usr/bin/env python
"""스킴 A/B 손실의 균형점을 수치로 검증 (합성 티처, tiny 최적화).

기대:
  스킴 A (itm_target_mix_loss, W=1.0, teacher tempered by T):  student gap → teacher_gap / T
  스킴 B (itm_hinton_kd_loss, α, T):                           student gap → teacher_gap (풀), 하드 CE floor로 약간 더 sharp

실행 (repo/worktree root, kd_r4):  python critical_bugfix/2026-07-24_itm_distill_sharpness/verify_scheme_equilibria.py
"""
import os, sys
sys.path.insert(0, os.getcwd())
import torch
import torch.nn.functional as F
from distillation.losses import itm_target_mix_loss
try:
    from distillation.losses import itm_hinton_kd_loss
    HAVE_B = True
except Exception:
    HAVE_B = False

torch.manual_seed(0)

# 합성 티처: match gap 7.8, distractor gap 4.5 (실측 티처 p50). z0=0 기준.
teacher_logits = torch.tensor([[0.0, 7.8], [0.0, 4.5]])
labels = torch.tensor([1, 0])                      # match=1, distractor=0
teacher_gap = (teacher_logits[:, 1] - teacher_logits[:, 0])


def optimize(loss_fn, steps=4000, lr=0.05):
    z = torch.zeros(2, 2, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss_fn(z).backward()
        opt.step()
    return (z[:, 1] - z[:, 0]).detach()


print(f"teacher gap        : match={teacher_gap[0]:.3f} distractor={teacher_gap[1]:.3f} "
      f"sep={ (teacher_gap[0]-teacher_gap[1]).item():.3f}")

# ---- 스킴 A: W=1.0, teacher tempered by T ----
for T in (2.0, 4.0):
    teacher_soft = F.softmax(teacher_logits / T, dim=1)
    gap = optimize(lambda z: itm_target_mix_loss(z, labels, teacher_soft, soft_weight=1.0))
    exp = teacher_gap / T
    print(f"[A] W=1 T={T}: student gap match={gap[0]:.3f} distr={gap[1]:.3f} "
          f"sep={(gap[0]-gap[1]).item():.3f}  (기대 gap≈teacher/T={exp[0]:.2f},{exp[1]:.2f}, sep≈{(exp[0]-exp[1]).item():.2f})")

# ---- 참고: arm C (W=0.4, T=1) 재현 ----
teacher_soft_T1 = F.softmax(teacher_logits / 1.0, dim=1)
gap_c = optimize(lambda z: itm_target_mix_loss(z, labels, teacher_soft_T1, soft_weight=0.4))
print(f"[armC] W=0.4 T=1: student gap match={gap_c[0]:.3f} distr={gap_c[1]:.3f} sep={(gap_c[0]-gap_c[1]).item():.3f}")

# ---- 스킴 B: 2텀 Hinton (구현되면) ----
if HAVE_B:
    for (alpha, T) in ((0.4, 2.0), (0.4, 4.0)):
        teacher_soft = F.softmax(teacher_logits / T, dim=1)
        gap = optimize(lambda z: itm_hinton_kd_loss(z, labels, teacher_soft, alpha=alpha, temp=T))
        print(f"[B] alpha={alpha} T={T}: student gap match={gap[0]:.3f} distr={gap[1]:.3f} "
              f"sep={(gap[0]-gap[1]).item():.3f}  (기대 gap→teacher, sep≳{ (teacher_gap[0]-teacher_gap[1]).item():.2f})")
else:
    print("[B] itm_hinton_kd_loss 미구현 — 스킴 B 구현 후 재실행")
