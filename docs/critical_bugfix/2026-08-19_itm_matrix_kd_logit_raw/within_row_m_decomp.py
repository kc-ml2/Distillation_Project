#!/usr/bin/env python
"""KD가 실제로 보는 anchor×후보 격자 위에서 m의 행안/행간 분산분해.

질문: distilled 학생 σ(m)=0.199 가 (조각1) 행 간 오프셋인가, (조각2) 행 안 후보 변동인가?
  - 행 간 오프셋: KD softmax(행별)에서 상쇄 → 무해
  - 행 안 후보 변동: 안 상쇄 → KD 오염원

KD 재현(브랜치 itm_bxb_matrix_kd, blip_pretrain.py:530-541 과 동일):
  idx = [정답(col0) | topk(weights, k)]  → itm_pair_logits 로 [B,k+1,2] 채점.
  i2t: 이미지 i × 후보텍스트 k+1.   t2i: 텍스트 i × 후보이미지 k+1.
프로덕션 itm_pair_logits 를 그대로 호출(손코딩 X). 프로브의 모델로딩/sim·weights 재사용.

teacher 는 학생이 뽑은 그 인덱스 위에서 채점(training 의 itm_matrix_gathered 와 동일).
distilled/teacher 는 distilled 인덱스, baseline 은 자기 인덱스(자기 네거티브라 다름).
"""
import argparse, os, sys
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/minwoo/Distillation_Project"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "critical_bugfix/2026-07-31_itm_batch_logit_raw"))
sys.path.insert(0, HERE)

import itm_batch_logit_probe as probe          # noqa: E402  (모델로딩/loader/_tokenize/autocast)
from itm_matrix import itm_pair_logits          # noqa: E402  (프로덕션 pair 채점)
from models.blip_pretrain import LOGIT_SCALE_MAX, LOGIT_SCALE_MIN  # noqa: E402


def _gather_indices(model, image, caption, device, amp, k):
    """학생 sim→weights→[정답|topk] 인덱스. blip_pretrain.py:450-534 와 동일한 규약.
    반환 (image_embeds, image_atts, enc_ids, text_atts, idx_i2t, idx_t2i)."""
    raw_ids, enc_ids, text_atts = probe._tokenize(model, caption, device)
    B = image.size(0)
    ar = torch.arange(B, device=device)[:, None]
    with probe.autocast_ctx(device, amp):
        image_embeds = model.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
        image_feat = F.normalize(model.vision_proj(image_embeds[:, 0, :]), dim=-1)
        text_out = model.text_encoder(raw_ids, attention_mask=text_atts, return_dict=True, mode="text")
        text_feat = F.normalize(model.text_proj(text_out.last_hidden_state[:, 0, :]), dim=-1)
        image_embeds_m = model.visual_encoder_m(image)
        image_feat_m = F.normalize(model.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
        text_out_m = model.text_encoder_m(raw_ids, attention_mask=text_atts, return_dict=True, mode="text")
        text_feat_m = F.normalize(model.text_proj_m(text_out_m.last_hidden_state[:, 0, :]), dim=-1)
        safe = model.logit_scale.clamp(LOGIT_SCALE_MIN, LOGIT_SCALE_MAX).exp()
        sim_i2t = (image_feat @ text_feat_m.t()).float() * safe.float()
        sim_t2i = (text_feat @ image_feat_m.t()).float() * safe.float()
        w_i2t = probe.pc.neg_weights(sim_i2t)   # [B,B], diag 0 (정답 제외)
        w_t2i = probe.pc.neg_weights(sim_t2i)
        idx_i2t = torch.cat([ar, w_i2t.topk(k, dim=1).indices], dim=1)   # [B,k+1], col0=정답
        idx_t2i = torch.cat([ar, w_t2i.topk(k, dim=1).indices], dim=1)
    return image_embeds, image_atts, enc_ids, text_atts, idx_i2t, idx_t2i


def _score(model, image_embeds, image_atts, enc_ids, text_atts, idx_i2t, idx_t2i, device, amp):
    """주어진 인덱스 위에서 [B,k+1,2] 로짓 2방향. arM=anchor(자기자신) 반복."""
    B, k1 = idx_i2t.shape
    arM = torch.arange(B, device=device)[:, None].expand(B, k1)
    with probe.autocast_ctx(device, amp):
        s_i2t = itm_pair_logits(model.text_encoder, model.itm_head, image_embeds, image_atts,
                                enc_ids, text_atts, arM, idx_i2t.to(device))          # 이미지 anchor × 텍스트 후보
        s_t2i = itm_pair_logits(model.text_encoder, model.itm_head, image_embeds, image_atts,
                                enc_ids, text_atts, idx_t2i.to(device), arM)          # 텍스트 anchor × 이미지 후보
    return s_i2t.float().cpu(), s_t2i.float().cpu()


def teacher_score_on(model, image, caption, idx_i2t, idx_t2i, device, amp):
    """티처: 학생 인덱스 위에서 채점. 티처 자기 tokenizer/enc_ids/image_embeds 사용."""
    _, enc_ids, text_atts = probe._tokenize(model, caption, device)
    with probe.autocast_ctx(device, amp):
        image_embeds = model.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
    return _score(model, image_embeds, image_atts, enc_ids, text_atts, idx_i2t, idx_t2i, device, amp)


def decomp(grid):
    """[N,k+1,2] → m·gap 의 행안/행간 분산분해. ddof=0 (정확한 항등식)."""
    z1, z2 = grid[..., 0], grid[..., 1]
    out = {}
    for name, arr in (("m", 0.5 * (z1 + z2)), ("gap", z2 - z1)):
        row_mean = arr.mean(axis=1)                       # [N] 행 평균
        var_within = arr.var(axis=1, ddof=0).mean()       # 행 내 분산의 행평균
        var_between = row_mean.var(ddof=0)                 # 행 평균들의 분산
        var_total = arr.reshape(-1).var(ddof=0)
        out[name] = dict(within=float(var_within), between=float(var_between), total=float(var_total))
    r = 2.0 * np.sqrt(out["m"]["within"]) / np.sqrt(out["gap"]["within"])   # 오염비
    return out, float(r)


def run_model(tag, model, cache, device, amp, k, own_indices):
    """own_indices=True: 자기 sim 으로 인덱스 재생성(학생). False: 캐시된 인덱스 채점(티처)."""
    gi, gt = [], []
    for img_cpu, caption, cidx_i2t, cidx_t2i in cache:
        image = img_cpu.to(device)
        if own_indices:
            ie, ia, enc, ta, idx_i2t, idx_t2i = _gather_indices(model, image, caption, device, amp, k)
            s_i2t, s_t2i = _score(model, ie, ia, enc, ta, idx_i2t, idx_t2i, device, amp)
        else:
            s_i2t, s_t2i = teacher_score_on(model, image, caption, cidx_i2t, cidx_t2i, device, amp)
        gi.append(s_i2t.numpy()); gt.append(s_t2i.numpy())
    return np.concatenate(gi, 0), np.concatenate(gt, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rows", type=int, default=500)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import yaml
    torch.set_grad_enabled(False); torch.manual_seed(a.seed)
    device = torch.device(a.device)

    cfg_d = yaml.safe_load(open(f"{REPO}/output/pt_smallreg_minilm_itm_matrix_kd_configfix/config.yaml"))
    cfg_b = yaml.safe_load(open(f"{REPO}/output/pt_smallreg_minilm_baseline/config.yaml"))
    loader = probe._build_loader(cfg_d, a.num_workers)
    n_batches = -(-a.n_rows // cfg_d["batch_size"])
    print(f"[plan] k={a.k} n_batches={n_batches} n_rows={a.n_rows} device={device}")

    # 공통 이미지 배치 캐시 (세 모델 동일 배치)
    images = []
    for bi, (image, caption) in enumerate(loader):
        if bi >= n_batches:
            break
        images.append((image.cpu(), list(caption)))

    grids = {}   # tag -> (i2t[N,k+1,2], t2i)

    # 1) distilled: 자기 인덱스 + 채점, 인덱스는 티처가 재사용하도록 캐시
    print("[load] distilled ...", flush=True)
    dm = probe._load_student(cfg_d, f"{REPO}/output/pt_smallreg_minilm_itm_matrix_kd_configfix/checkpoint_19.pth", device)
    cache = []
    gi, gt = [], []
    for img_cpu, caption in images:
        image = img_cpu.to(device)
        ie, ia, enc, ta, idx_i2t, idx_t2i = _gather_indices(dm, image, caption, device, a.amp, a.k)
        s_i2t, s_t2i = _score(dm, ie, ia, enc, ta, idx_i2t, idx_t2i, device, a.amp)
        gi.append(s_i2t.numpy()); gt.append(s_t2i.numpy())
        cache.append((img_cpu, caption, idx_i2t.cpu(), idx_t2i.cpu()))
    grids["distilled"] = (np.concatenate(gi, 0), np.concatenate(gt, 0))
    del dm

    # 2) teacher: distilled 인덱스 위에서 채점
    print("[load] teacher ...", flush=True)
    tm = probe._load_teacher(cfg_d, f"{REPO}/output/official_pretrain_checkpoint/model_large.pth", device)
    grids["teacher"] = run_model("teacher", tm, cache, device, a.amp, a.k, own_indices=False)
    del tm

    # 3) baseline: 자기 인덱스 + 채점 (같은 이미지)
    print("[load] baseline ...", flush=True)
    bm = probe._load_student(cfg_b, f"{REPO}/output/pt_smallreg_minilm_baseline/checkpoint_19.pth", device)
    base_cache = [(img, cap, None, None) for img, cap in images]
    grids["baseline"] = run_model("baseline", bm, base_cache, device, a.amp, a.k, own_indices=True)
    del bm

    # ---- 분산분해 표 ----
    rows = []
    for tag in ("teacher", "baseline", "distilled"):
        for di, direction in enumerate(("i2t", "t2i")):
            g = grids[tag][di][:a.n_rows]
            d, r = decomp(g)
            rows.append({
                "model": tag, "direction": direction, "n_rows": len(g), "k+1": g.shape[1],
                "sig_within_m": np.sqrt(d["m"]["within"]), "sig_between_m": np.sqrt(d["m"]["between"]),
                "sig_total_m": np.sqrt(d["m"]["total"]),
                "frac_within_m": d["m"]["within"] / d["m"]["total"],
                "sig_within_gap": np.sqrt(d["gap"]["within"]),
                "contamination_r": r,
            })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n=== within/between-row m decomposition ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

    with pd.ExcelWriter(os.path.join(HERE, "withinrow_m_decomp.xlsx")) as xl:
        df.to_excel(xl, sheet_name="decomp", index=False)

    # ---- 그래프: frac_within_m + contamination_r ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    order = ["teacher", "baseline", "distilled"]
    col = {"teacher": "#d62728", "baseline": "#1f77b4", "distilled": "#2ca02c"}
    x = np.arange(len(order)); w = 0.35
    for di, direction in enumerate(("i2t", "t2i")):
        fw = [df[(df.model == m) & (df.direction == direction)]["frac_within_m"].iloc[0] for m in order]
        rr = [df[(df.model == m) & (df.direction == direction)]["contamination_r"].iloc[0] for m in order]
        axes[0].bar(x + (di - 0.5) * w, fw, w, label=direction,
                    color=[col[m] for m in order], alpha=0.6 if di else 1.0, edgecolor="k")
        axes[1].bar(x + (di - 0.5) * w, rr, w, label=direction,
                    color=[col[m] for m in order], alpha=0.6 if di else 1.0, edgecolor="k")
    axes[0].set_title("frac within-row of m variance  (what KD sees)"); axes[0].set_ylabel("var_within / var_total")
    axes[1].set_title("contamination r = 2 sig_within(m) / sig_within(gap)"); axes[1].axhline(1.0, color="r", ls="--", lw=1)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(order); ax.legend(title="direction")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "withinrow_m_decomp.png"), dpi=120)
    print("\nwrote withinrow_m_decomp.xlsx, withinrow_m_decomp.png")


if __name__ == "__main__":
    main()
