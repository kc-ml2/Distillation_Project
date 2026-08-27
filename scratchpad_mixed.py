"""Compute the entropy/effcand of the ACTUAL mixed ITC target
  target = (1-W)*onehot + W*(gamma*teacher + (1-gamma)*momentum),  W=0.4, gamma=0.7
and compare to student and teacher-alone. Also E[cos] under each (drives logit_scale grad).
Uses cached sims (instant, no model reload). Momentum ~= student (EMA proxy); onehot on the
image's first positive caption (single-diagonal proxy). Run: conda run -n kd_r4 python scratchpad_mixed.py
"""
import sys, math, os, torch, numpy as np, torch.nn.functional as F
sys.argv=['x']
from data.eval_validation_retrieval import build_coco_karpathy_retrieval_val_loader
CACHE=os.path.expanduser('~/.claude/jobs/d6026650/tmp/sim_cache.pt')
d=torch.load(CACHE); S_stu,S_tea,s_scale=d['S_stu'],d['S_tea'],d['s_scale']
N=S_stu.shape[0]
cfg={'val_retrieval_split':'val','val_retrieval_ann_root':'/home/minwoo/Distillation_Project/datasets/vision/coco/annotations/',
 'val_retrieval_image_root':'/home/minwoo/Distillation_Project/datasets/vision/coco/images/',
 'image_root_coco':'/home/minwoo/Distillation_Project/datasets/vision/coco/images/',
 'val_retrieval_batch_size':64,'batch_size':64,'image_size':224,'k_test':128}
ds=build_coco_karpathy_retrieval_val_loader(cfg).dataset
pos=[int(ds.img2txt[i][0]) for i in range(N)]  # single-positive proxy per image
W,gamma=0.4,0.7

def ent_ec(P):  # mean row entropy -> effcand
    H=(-(P*torch.log(P.clamp_min(1e-30))).sum(1)).mean().item(); return H, math.exp(H)
def ecos(P):    # E_P[cos] using the STUDENT cosines (what logit_scale grad sees)
    return (P*S_stu).sum(1).mean().item()

stu=F.softmax(S_stu*s_scale,dim=1)
print(f"{'dist':>26} {'H':>6} {'effcand':>9} {'E[cos]':>8} {'posmass':>8}")
def row(name,P):
    H,ec=ent_ec(P); pm=np.mean([P[i,pos[i]].item() for i in range(N)])
    print(f"{name:>26} {H:>6.2f} {ec:>9.1f} {ecos(P):>8.4f} {pm:>8.4f}")
row('STUDENT @ safe_scale', stu)
for tsc,lab in [(20.0,'ttm 0.05'),(38.77,'T* 0.0258'),(63.89,'native 0.0157')]:
    tea=F.softmax(S_tea*tsc,dim=1)
    oh=torch.zeros_like(tea);
    for i in range(N): oh[i,pos[i]]=1.0
    mixed=(1-W)*oh + W*(gamma*tea + (1-gamma)*stu)   # momentum:=stu proxy
    print(f"\n--- teacher scale {tsc} ({lab}) ---")
    row(f'  teacher-alone', tea)
    row(f'  MIXED target', mixed)
print("\n# The MIXED target is what the student's CE actually trains against.")
print("# Compare MIXED effcand & E[cos] to STUDENT: if mixed is SHARPER (lower effcand /")
print("#  higher E[cos]) than student, student is pulled to sharpen (no overshoot);")
print("#  the crater must then be about WHERE the diffuse teacher mass lands (hubs), not overshoot.")
