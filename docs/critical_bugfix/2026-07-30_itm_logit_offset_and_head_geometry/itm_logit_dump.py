#!/usr/bin/env python
"""ITM 로짓 원자료 덤프 + within-query 통계.

동기 — 기존 조사(2026-07-24 / 2026-07-27)의 프로브는 모두 **집계값만** 저장했다
(`{n, mean, p10, p50, p90}` / `{r1, r5, r10, mrr}`). 그래서 다음 질문에 답할 수 없다:

  "eval이 z1(match 로짓)으로 랭킹하는 게 gap(z1-z0) 랭킹과 왜 같은 결과를 주는가?"

z1 = z0 + gap 이므로, 두 랭킹의 일치도는 **쿼리 내부** 통계 세 개로 정해진다:
  sigma_0 = std(z0),  sigma_g = std(gap),  rho = corr(z0, gap)      (모두 within-query)
  => rho(z1, gap) = (rho*sigma_0 + sigma_g) / sqrt(sigma_0^2 + sigma_g^2 + 2*rho*sigma_0*sigma_g)

이 값이 1이면 두 랭킹이 완전 일치. 기존 기록은 쿼리를 합친 집계값이라 세 항 중 하나도 계산할 수 없다.

이 스크립트는 (a) 쿼리별 원자료를 npz로 남기고 (b) 위 통계를 계산한다. 헤드는 autocast 밖 fp32.

실행 (repo root, 여유 GPU):
  CUDA_VISIBLE_DEVICES=0 python critical_bugfix/2026-07-30_itm_logit_offset_and_head_geometry/itm_logit_dump.py
"""
import argparse
import json
import os
import sys

REPO = "/home/minwoo/Distillation_Project"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "critical_bugfix/2026-07-24_itm_distill_sharpness"))
os.chdir(REPO)

import numpy as np
import torch
import yaml
from scipy.stats import kendalltau

import utils
from itm_sharpness_probe import pick_ckpt
from models.blip_pretrain import blip_pretrain
from data.eval_validation_retrieval import build_coco_karpathy_retrieval_val_loader
from eval_validation_tool import _encode_retrieval_features, _retrieval_autocast_context

HERE = os.path.dirname(os.path.abspath(__file__))

# (tag, run_dir, epoch, vit, bert)  — run_dir=None 이면 티처(공식 BLIP-large 체크포인트)
TARGETS = [
    ("baseline@ep15", "pt_smallreg_minilm_baseline", 15, None, None),
    ("baseline@ep19", "pt_smallreg_minilm_baseline", 19, None, None),
    ("schemeA@ep15", "pt_itm_schemeA_tempered", 15, None, None),
    ("teacher_blip_large", None, None, "large", "base"),
]


# ---------------------------------------------------------------- 헤드 기하
def head_geometry(model):
    W = model.itm_head.weight.detach().float().cpu()
    b = model.itm_head.bias.detach().float().cpu()
    w0, w1 = W[0], W[1]
    wd, ws = w1 - w0, w1 + w0
    return dict(
        w0_norm=round(float(w0.norm()), 4), w1_norm=round(float(w1.norm()), 4),
        cos_w0_w1=round(float(torch.dot(w0, w1) / (w0.norm() * w1.norm())), 4),
        wd_norm=round(float(wd.norm()), 4), ws_norm=round(float(ws.norm()), 4),
        ws_over_wd=round(float(ws.norm() / wd.norm()), 4),
        b0=round(float(b[0]), 4), b1=round(float(b[1]), 4),
        b_d=round(float(b[1] - b[0]), 4), b_s=round(float(b[1] + b[0]), 4),
    )


# ---------------------------------------------------------------- 덤프
@torch.no_grad()
def dump_direction(model, sims_dir, image_feats, text_ids, text_atts, is_i2t,
                   img2txt, txt2img, device, cfg, topk, limit):
    """쿼리별 top-k 후보의 z0/z1/cos/is_gt 를 [Q,K] 배열로 반환. 헤드는 fp32."""
    n = min(limit, sims_dir.size(0))
    Z0 = np.zeros((n, topk), dtype=np.float32)
    Z1 = np.zeros((n, topk), dtype=np.float32)
    COS = np.zeros((n, topk), dtype=np.float32)
    GT = np.zeros((n, topk), dtype=bool)
    for i in range(n):
        topk_sim, topk_idx = sims_dir[i].topk(k=min(topk, sims_dir.size(1)))
        with _retrieval_autocast_context(device, cfg):
            if is_i2t:
                enc = image_feats[i].repeat(topk_idx.size(0), 1, 1).to(device)
                att = torch.ones(enc.size()[:-1], dtype=torch.long, device=device)
                out = model.text_encoder(text_ids[topk_idx], attention_mask=text_atts[topk_idx],
                                        encoder_hidden_states=enc, encoder_attention_mask=att,
                                        return_dict=True)
                gt = set(int(x) for x in img2txt[i])
            else:
                enc = image_feats[topk_idx.cpu()].to(device)
                att = torch.ones(enc.size()[:-1], dtype=torch.long, device=device)
                out = model.text_encoder(text_ids[i].repeat(topk_idx.size(0), 1),
                                        attention_mask=text_atts[i].repeat(topk_idx.size(0), 1),
                                        encoder_hidden_states=enc, encoder_attention_mask=att,
                                        return_dict=True)
                gt = {int(txt2img[i])}
        # ★ 헤드는 autocast 밖에서 fp32 — gap = z1 - z0 의 차분 정밀도 확보
        h = out.last_hidden_state[:, 0, :].float()
        logits = torch.nn.functional.linear(h, model.itm_head.weight.float(),
                                           model.itm_head.bias.float())
        Z0[i] = logits[:, 0].cpu().numpy()
        Z1[i] = logits[:, 1].cpu().numpy()
        COS[i] = topk_sim.float().cpu().numpy()
        GT[i] = np.array([int(j) in gt for j in topk_idx.cpu().tolist()])
    return Z0, Z1, COS, GT


# ---------------------------------------------------------------- 통계
def within_query_stats(Z0, Z1, COS, GT):
    """쿼리 내부 통계. 반환값은 쿼리별 배열 + 요약."""
    GAP = Z1 - Z0
    M = 0.5 * (Z1 + Z0)                      # 07-27 README 의 offset(h)
    s0 = Z0.std(axis=1, ddof=1)
    sg = GAP.std(axis=1, ddof=1)
    s1 = Z1.std(axis=1, ddof=1)
    sm = M.std(axis=1, ddof=1)
    scos = COS.std(axis=1, ddof=1)

    def rowcorr(A, B):
        Ac, Bc = A - A.mean(1, keepdims=True), B - B.mean(1, keepdims=True)
        den = np.sqrt((Ac ** 2).sum(1) * (Bc ** 2).sum(1))
        out = np.zeros(len(A))
        ok = den > 0
        out[ok] = (Ac * Bc).sum(1)[ok] / den[ok]
        return out

    rho_0g = rowcorr(Z0, GAP)
    rho_1g = rowcorr(Z1, GAP)               # 랭킹 일치도의 선형 대리지표
    # 예측식 검산: rho(z1,gap) = (rho*s0 + sg)/sqrt(s0^2+sg^2+2*rho*s0*sg)
    den = np.sqrt(s0 ** 2 + sg ** 2 + 2 * rho_0g * s0 * sg)
    rho_pred = np.where(den > 0, (rho_0g * s0 + sg) / np.maximum(den, 1e-12), np.nan)

    # 순위 일치 (z1 vs gap)
    tau = np.array([kendalltau(Z1[i], GAP[i]).statistic for i in range(len(Z1))])
    argmax_same = (Z1.argmax(1) == GAP.argmax(1))

    # within-query AUC / d' / rank
    def wq_auc(S):
        out = []
        for i in range(len(S)):
            pos, neg = S[i][GT[i]], S[i][~GT[i]]
            if len(pos) == 0 or len(neg) == 0:
                out.append(np.nan); continue
            out.append(float((pos[:, None] > neg[None, :]).mean()))
        return np.array(out)

    def wq_dprime(S):
        out = []
        for i in range(len(S)):
            pos, neg = S[i][GT[i]], S[i][~GT[i]]
            sd = neg.std(ddof=1)
            out.append(float((pos.mean() - neg.mean()) / sd) if len(pos) and sd > 0 else np.nan)
        return np.array(out)

    def rank_of_gt(S):
        out = []
        for i in range(len(S)):
            order = np.argsort(-S[i], kind="stable")
            hit = np.where(GT[i][order])[0]
            out.append(int(hit[0]) if len(hit) else S.shape[1])
        return np.array(out)

    scores = {"z1": Z1, "gap": GAP, "cos": COS, "z1+cos": Z1 + COS, "gap+cos": GAP + COS}
    ranks = {k: rank_of_gt(v) for k, v in scores.items()}

    def q(a):
        a = np.asarray(a, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return None
        return dict(mean=round(float(a.mean()), 4), p10=round(float(np.percentile(a, 10)), 4),
                    p50=round(float(np.percentile(a, 50)), 4), p90=round(float(np.percentile(a, 90)), 4))

    def rk(r):
        r = np.asarray(r)
        return dict(r1=round(100 * float((r < 1).mean()), 2), r5=round(100 * float((r < 5).mean()), 2),
                    r10=round(100 * float((r < 10).mean()), 2),
                    rmean=round(100 * float(((r < 1).mean() + (r < 5).mean() + (r < 10).mean()) / 3), 2))

    n_q = len(Z0)
    return dict(
        n_query=n_q, n_cand=int(Z0.shape[1]),
        # --- 핵심: 쿼리 내부 산포 ---
        std_z0=q(s0), std_gap=q(sg), std_z1=q(s1), std_offset_m=q(sm), std_cos=q(scos),
        ratio_std_z0_over_gap=q(s0 / np.maximum(sg, 1e-12)),
        ratio_std_m_over_gap=q(sm / np.maximum(sg, 1e-12)),
        corr_z0_gap=q(rho_0g), corr_z1_gap=q(rho_1g), corr_z1_gap_predicted=q(rho_pred),
        max_abs_pred_error=round(float(np.nanmax(np.abs(rho_1g - rho_pred))), 6),
        kendall_tau_z1_gap=q(tau),
        argmax_same_frac=round(100 * float(argmax_same.mean()), 2),
        # --- 랭킹 품질 (스케일 무관) ---
        wq_auc_gap=q(wq_auc(GAP)), wq_auc_z1=q(wq_auc(Z1)), wq_auc_cos=q(wq_auc(COS)),
        wq_dprime_gap=q(wq_dprime(GAP)),
        # --- 정답 순위 기반 R@k (기존 실험과 대조용) ---
        Rk={k: rk(v) for k, v in ranks.items()},
        # --- paired 부호검정 (z1 vs gap) ---
        paired_z1_vs_gap=dict(
            n_better=int((ranks["z1"] < ranks["gap"]).sum()),
            n_worse=int((ranks["z1"] > ranks["gap"]).sum()),
            n_tie=int((ranks["z1"] == ranks["gap"]).sum()),
        ),
        # --- offset 평균 (기존 집계값과 대조) ---
        mean_offset_m=round(float(M.mean()), 4),
        mean_gap_gt=round(float(GAP[GT].mean()), 4) if GT.any() else None,
        mean_gap_distractor=round(float(GAP[~GT].mean()), 4),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--topk", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(HERE, "itm_logit_stats.json"))
    ap.add_argument("--npz-dir", default=os.path.join(HERE, "npz"))
    ap.add_argument("--only", default=None, help="쉼표구분 tag 필터 (스모크테스트용)")
    args = ap.parse_args()

    targets = TARGETS
    if args.only:
        want = {t.strip() for t in args.only.split(",")}
        targets = [t for t in TARGETS if t[0] in want]

    # ★ _encode_retrieval_features 는 no_grad 를 스스로 걸지 않는다
    #   (호출부인 evaluate_retrieval_itm 이 @torch.no_grad() 였음).
    #   빼먹으면 ViT forward 5000장의 autograd 그래프가 쌓여 즉시 OOM.
    torch.set_grad_enabled(False)

    os.makedirs(args.npz_dir, exist_ok=True)
    device = torch.device(args.device)
    cfg = yaml.safe_load(open(os.path.join(REPO, "output/pt_itm_schemeA_tempered/config.yaml")))
    cfg["val_retrieval_enabled"] = True
    cfg["val_retrieval_split"] = "val"

    print("[build] loader ...", flush=True)
    loader = build_coco_karpathy_retrieval_val_loader(cfg)
    img2txt = loader.dataset.img2txt
    txt2img = getattr(loader.dataset, "txt2img", None)
    if txt2img is None:
        txt2img = {}
        for im, txts in enumerate(img2txt):
            for t in txts:
                txt2img[t] = im

    allres = {}
    for tag, run, ep, vit_ovr, bert_ovr in targets:
        if run is None:
            ck = os.path.join(REPO, "output/official_pretrain_checkpoint/model_large.pth")
            if not os.path.exists(ck):
                print(f"[skip] {tag}: no teacher checkpoint"); continue
            model = blip_pretrain(image_size=cfg["image_size"], vit=vit_ovr, vit_grad_ckpt=False,
                                  vit_ckpt_layer=0, queue_size=cfg["queue_size"],
                                  my_bert_size=bert_ovr, init_backbone_weights=False).to(device)
        else:
            ck = pick_ckpt(run, prefer_epoch=ep)
            if ck is None:
                print(f"[skip] {tag}: no checkpoint in {run}"); continue
            model = blip_pretrain(image_size=cfg["image_size"], vit=cfg["vit"],
                                  vit_grad_ckpt=cfg["vit_grad_ckpt"], vit_ckpt_layer=cfg["vit_ckpt_layer"],
                                  queue_size=cfg["queue_size"], my_bert_size=cfg["my_bert_size"]).to(device)
        print(f"[load] {tag} <- {os.path.basename(ck)}", flush=True)
        model = utils.load_model_weights_only(model, ck)
        model.eval()

        geom = head_geometry(model)
        print(f"  head: {geom}", flush=True)

        with _retrieval_autocast_context(device, cfg):
            sims, image_feats, text_ids, text_atts = _encode_retrieval_features(
                model, loader, device, need_patch_feats=True)

        rec = dict(tag=tag, checkpoint=os.path.basename(ck), head_geometry=geom)
        for dname, sd, isi2t in (("i2t", sims, True), ("t2i", sims.t().contiguous(), False)):
            print(f"  [{dname}] dumping ...", flush=True)
            Z0, Z1, COS, GT = dump_direction(model, sd, image_feats, text_ids, text_atts, isi2t,
                                             img2txt, txt2img, device, cfg, args.topk, args.limit)
            np.savez_compressed(os.path.join(args.npz_dir, f"{tag.replace('@','_')}_{dname}.npz"),
                                z0=Z0, z1=Z1, cos=COS, is_gt=GT)
            rec[dname] = within_query_stats(Z0, Z1, COS, GT)
            s = rec[dname]
            print(f"    std_z0 p50={s['std_z0']['p50']}  std_gap p50={s['std_gap']['p50']}  "
                  f"ratio p50={s['ratio_std_z0_over_gap']['p50']}  corr(z0,gap) p50={s['corr_z0_gap']['p50']}  "
                  f"corr(z1,gap) p50={s['corr_z1_gap']['p50']}  tau p50={s['kendall_tau_z1_gap']['p50']}  "
                  f"argmax_same={s['argmax_same_frac']}%", flush=True)
        allres[tag] = rec
        json.dump(allres, open(args.out, "w"), ensure_ascii=False, indent=2)
        del model, sims, image_feats
        torch.cuda.empty_cache()

    print(f"\nsaved: {args.out}\nnpz:   {args.npz_dir}")


if __name__ == "__main__":
    main()
