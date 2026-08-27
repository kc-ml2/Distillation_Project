"""Read-only 측정: BLIP-large 티처의 batch 내 ITC 유사도 간격 → τ 선택 근거.
돌고 있는 학습엔 전혀 손대지 않음. GPU0에 티처 inference-only(no_grad/bf16)로 얹음.

두 패스로 측정:
  - aug ON  : 학습/KD가 실제로 티처에 먹이는 증강 이미지 (min_scale 0.2 + RandomAugment)
  - aug OFF : 결정적 리사이즈만 (clean) — 증강 노이즈가 티처 신호를 얼마나 망가뜨리는지 대조
B×B 코사인 행렬을 (mode,N)별 .pt로 캐시 → τ 그리드만 바꿀 땐 재-forward 없이 즉시.

실행:
  cd /home/minwoo/Distillation_Project_itc_distill_only
  CUDA_VISIBLE_DEVICES=0 /home/minwoo/miniconda3/envs/kd_r4/bin/python \
    critical_bugfix/measure_teacher_sim.py
"""
import os, sys, math, copy, yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

WT = "/home/minwoo/Distillation_Project_itc_distill_only"
sys.path.insert(0, WT)
CFG = f"{WT}/configs/pretrain_itc_distill.yaml"
BUGDIR = "/home/minwoo/Distillation_Project/critical_bugfix/2026-07-07_teacher_sim_and_tau"
OUT_PNG = f"{BUGDIR}/teacher_sim_hist.png"
OUT_CSV = f"{BUGDIR}/tau_sweep.csv"
N_BATCHES = 100                     # ×B(40) = 4000 samples per mode
TAU_GRID = [round(0.020 + 0.001 * k, 3) for k in range(0, 21)]  # 0.020→0.040 step 0.001

with open(CFG) as f:
    config = yaml.safe_load(f)

from data import create_dataset
from distillation.online_teacher import OnlineTeacher

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_teacher = {"m": None}

def get_teacher():
    if _teacher["m"] is None:
        print("[teacher] building BLIP-large (frozen)…")
        _teacher["m"] = OnlineTeacher(checkpoint=config["teacher"]["checkpoint"],
                                      image_size=config["image_size"], vit="large", bert="base",
                                      queue_size=config["queue_size"], keep=("itc",)).to(device)
    return _teacher["m"]

def collect(mode):
    """mode: 'aug' or 'noaug'. returns list of [B,B] cosine matrices (cpu)."""
    cache = f"{BUGDIR}/sim_{mode}_n{N_BATCHES}.pt"
    if os.path.isfile(cache):
        S_list = torch.load(cache)
        print(f"[cache] {mode}: loaded {len(S_list)} matrices ← {cache}")
        return S_list
    cfg = copy.deepcopy(config)
    cfg["pretrain_train_aug"] = (mode == "aug")
    ds = create_dataset("pretrain", cfg, min_scale=0.2)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=4, drop_last=True)
    teacher = get_teacher()
    S_list, seen = [], 0
    for image, caption in loader:
        image = image.to(device)
        img_f, txt_f = teacher.itc_feats(image, caption)
        S_list.append((img_f.float() @ txt_f.float().t()).cpu())
        seen += 1
        if seen >= N_BATCHES:
            break
    torch.save(S_list, cache)
    print(f"[cache] {mode}: saved {len(S_list)} matrices → {cache}")
    return S_list

print(f"[device] {device}  B={config['batch_size']}  N_BATCHES={N_BATCHES} (→ {N_BATCHES*config['batch_size']} samples/mode)")

def cosine_stats(S_list):
    diag_all, off_all, phn_all, batch_delta = [], [], [], []
    for S in S_list:
        B = S.size(0); eye = torch.eye(B, dtype=torch.bool)
        d, o = S[eye], S[~eye]
        diag_all.append(d); off_all.append(o)
        hardneg = S.masked_fill(eye, float("-inf")).max(dim=1).values
        phn_all.append(d - hardneg)
        batch_delta.append((d.mean() - o.mean()).item())      # 배치별 Δ (튀는 정도)
    return (torch.cat(diag_all), torch.cat(off_all), torch.cat(phn_all), torch.tensor(batch_delta))

def report_cos(name, S_list):
    diag, off, phn, bd = cosine_stats(S_list)
    def line(nm, t):
        print(f"    {nm:22s} mean={t.mean():.4f} std={t.std():.4f} "
              f"[p5={t.quantile(0.05):.4f} p50={t.median():.4f} p95={t.quantile(0.95):.4f}]")
    print(f"\n=== [{name}] 코사인 분포 ({len(S_list)} batches, {diag.numel()} pos / {off.numel()} neg) ===")
    line("diagonal (positive)", diag); line("off-diag (neg)", off); line("pos - hardest_neg", phn)
    print(f"    Δ(mean_diag-mean_off) = {(diag.mean()-off.mean()):.4f}")
    print(f"    배치별 Δ: mean={bd.mean():.4f} std={bd.std():.4f}  "
          f"→ 상대변동(std/mean)={bd.std()/bd.mean():.2%}  (KD 신호가 배치마다 튀는 정도)")
    return diag, off

# ---- 두 패스 수집 ----
S_aug = collect("aug")
S_noaug = collect("noaug")

diag_aug, off_aug = report_cos("aug ON  (KD 실제 조건)", S_aug)
report_cos("aug OFF (clean)", S_noaug)

# ---- τ 정밀 스윕 (aug ON = KD 실제 조건 기준) ----
Bref = S_aug[0].size(0)
print(f"\n=== τ 정밀 스윕 [aug ON] (i2t, 행별 평균) | 학생 수렴온도≈0.023 앵커 | {len(S_aug)*Bref} samples ===")
print(f"  {'τ':>6}  {'max-prob':>9}  {'pos-prob':>9}  {'entropy':>8}  {'argmax=diag':>11}   해석")
rows = []
for tau in TAU_GRID:
    mps, pps, ents, accs = [], [], [], []
    for S in S_aug:
        B = S.size(0); eye = torch.eye(B, dtype=torch.bool)
        p = F.softmax(S / tau, dim=1)
        mps.append(p.max(dim=1).values); pps.append(p[eye])
        ents.append(-(p * (p + 1e-12).log()).sum(dim=1) / math.log(B))
        accs.append((p.argmax(dim=1) == torch.arange(B)).float())
    mp = torch.cat(mps).mean().item(); pp = torch.cat(pps).mean().item()
    en = torch.cat(ents).mean().item(); ac = torch.cat(accs).mean().item()
    rows.append((tau, mp, pp, en, ac))
    tag = "너무 뾰족" if pp > 0.7 else ("너무 평평" if pp < 0.2 else "★ soft+peaked")
    print(f"  {tau:>6.3f}  {mp:>9.3f}  {pp:>9.3f}  {en:>8.3f}  {ac:>11.3f}   {tag}")
print(f"\n  (uniform: 1/B = {1.0/Bref:.4f})  pos-prob=티처가 정답쌍에 주는 평균확률")

with open(OUT_CSV, "w") as f:
    f.write("tau,max_prob,pos_prob,norm_entropy,argmax_eq_diag\n")
    for r in rows:
        f.write("{:.3f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(*r))
print(f"[csv] saved → {OUT_CSV}")

# 히스토그램 (aug ON)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(off_aug.numpy(), bins=100, alpha=0.6, label="negatives", density=True, color="#888")
    ax.hist(diag_aug.numpy(), bins=60, alpha=0.7, label="positives", density=True, color="#e45756")
    ax.set_xlabel("teacher cosine similarity (aug ON)"); ax.set_ylabel("density")
    ax.set_title(f"BLIP-large teacher: pos vs neg  ({len(S_aug)*Bref} samples)"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=110)
    print(f"[hist] saved → {OUT_PNG}")
except Exception as e:
    print(f"[hist] skipped ({e})")
