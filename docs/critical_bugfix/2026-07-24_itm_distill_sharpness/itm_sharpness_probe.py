#!/usr/bin/env python
"""ITM 헤드 sharpness 실측: 학생(arm C)·sharp 레퍼런스·티처(BLIP-large)의
ITM 로짓 gap(z1-z0)을 COCO val에서 '정답 vs 하드네거(ITC top-k distractor)'로 나눠 측정.

결정 질문: 티처가 하드네거에서 확신(gap 깊게 음수)하나, 애매(gap≈0)하나.
  - 티처 distractor gap 깊게 음수 → 헤드 증류 살릴 여지(양성 천장이 주범).
  - 티처 distractor gap ≈0        → 티처도 애매 → 헤드 증류 부적합 → soft는 ITC 경로로.
학생(arm C) gap이 baseline보다 압축 → 캡 실측 확정.

실행 (repo root, kd_v4, 여유 GPU):
  CUDA_VISIBLE_DEVICES=<free> python itm_sharpness_probe.py --limit 500 --topk 32
"""
import sys, os, glob, json, argparse
REPO = "/home/minwoo/Distillation_Project"
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np
import torch
import yaml

import utils
from models.blip_pretrain import blip_pretrain
from distillation.online_teacher import OnlineTeacher
from data.eval_validation_retrieval import build_coco_karpathy_retrieval_val_loader
from eval_validation_tool import _encode_retrieval_features, _retrieval_autocast_context


def pick_ckpt(run_dir, prefer_epoch=None):
    cks = sorted(glob.glob(os.path.join(REPO, "output", run_dir, "checkpoint_*.pth")))
    if not cks:
        return None
    if prefer_epoch is not None:
        want = os.path.join(REPO, "output", run_dir, f"checkpoint_{prefer_epoch:02d}.pth")
        if want in cks:
            return want
    return cks[-1]


@torch.no_grad()
def measure(model, loader, device, config, topk, n_query, tag):
    model.eval()
    with _retrieval_autocast_context(device, config):
        sims, image_feats, text_ids, text_atts = _encode_retrieval_features(
            model, loader, device, need_patch_feats=True)
    img2txt = loader.dataset.img2txt
    m_gap, d_gap, m_z1, d_z1 = [], [], [], []
    n = min(n_query, sims.size(0))
    for i in range(n):
        _, topk_idx = sims[i].topk(k=min(topk, sims.size(1)))
        enc = image_feats[i].repeat(topk_idx.size(0), 1, 1).to(device)
        att = torch.ones(enc.size()[:-1], dtype=torch.long, device=device)
        with _retrieval_autocast_context(device, config):
            out = model.text_encoder(text_ids[topk_idx], attention_mask=text_atts[topk_idx],
                                     encoder_hidden_states=enc, encoder_attention_mask=att,
                                     return_dict=True)
            logits = model.itm_head(out.last_hidden_state[:, 0, :]).float()  # [k,2]
        gap = (logits[:, 1] - logits[:, 0]).cpu().numpy()
        z1 = logits[:, 1].cpu().numpy()
        gt = set(img2txt[i])
        for r, j in enumerate(topk_idx.cpu().tolist()):
            (m_gap if j in gt else d_gap).append(gap[r])
            (m_z1 if j in gt else d_z1).append(z1[r])
    def stat(a):
        a = np.asarray(a, dtype=np.float64)
        if a.size == 0:
            return dict(n=0)
        return dict(n=int(a.size), mean=round(float(a.mean()), 3),
                    p10=round(float(np.percentile(a, 10)), 3),
                    p50=round(float(np.percentile(a, 50)), 3),
                    p90=round(float(np.percentile(a, 90)), 3))
    res = dict(tag=tag, match_gap=stat(m_gap), distractor_gap=stat(d_gap),
               match_z1=stat(m_z1), distractor_z1=stat(d_z1),
               separation_mean=round(float(np.mean(m_gap) - np.mean(d_gap)), 3) if m_gap and d_gap else None)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500, help="쿼리 이미지 수 (속도)")
    ap.add_argument("--topk", type=int, default=32, help="쿼리당 후보(하드네거) 수")
    ap.add_argument("--epoch", type=int, default=3, help="학생계열 epoch-matched 체크포인트")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(REPO, "output", "itm_sharpness_probe.json"))
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "itmC_config.yaml"),
                    help="itm_C config (arch + val 경로). dev 워킹트리엔 없어서 브랜치에서 뽑은 스크래치패드 사본 사용")
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = yaml.safe_load(open(args.config))
    cfg["val_retrieval_enabled"] = True
    cfg["val_retrieval_split"] = "val"
    loader = build_coco_karpathy_retrieval_val_loader(cfg)

    students = {
        "baseline_sharp_ref": "pt_smallreg_minilm_baseline",
        "lm_distill_sharp_ref": "pt_smallreg_minilm_lm_distill",
        "armC_itm_soft": "pt_itm_C_teachneg_soft",
    }
    results = []
    for tag, run in students.items():
        ck = pick_ckpt(run, prefer_epoch=args.epoch)
        if ck is None:
            print(f"[skip] {tag}: no checkpoint in {run}"); continue
        print(f"[load] {tag}  <- {os.path.basename(ck)}")
        model = blip_pretrain(image_size=cfg["image_size"], vit=cfg["vit"],
                              vit_grad_ckpt=cfg["vit_grad_ckpt"], vit_ckpt_layer=cfg["vit_ckpt_layer"],
                              queue_size=cfg["queue_size"], my_bert_size=cfg["my_bert_size"]).to(device)
        model = utils.load_model_weights_only(model, ck)
        r = measure(model, loader, device, cfg, args.topk, args.limit, f"{tag}@ep{args.epoch}")
        print(json.dumps(r, ensure_ascii=False)); results.append(r)
        del model; torch.cuda.empty_cache()

    # teacher (BLIP-large): blip_pretrain 직접 빌드(dev online_teacher엔 itm keep 없음).
    # init_backbone_weights=False로 large 백본 timm-init 우회, 가중치는 체크포인트가 전량 결정.
    teacher_ck = os.path.join(REPO, "output/official_pretrain_checkpoint/model_large.pth")
    if os.path.exists(teacher_ck):
        print("[load] teacher BLIP-large (direct blip_pretrain vit=large/bert=base)")
        tmodel = blip_pretrain(image_size=cfg["image_size"], vit="large",
                               vit_grad_ckpt=False, vit_ckpt_layer=0,
                               queue_size=cfg["queue_size"], my_bert_size="base",
                               init_backbone_weights=False).to(device)
        tmodel = utils.load_model_weights_only(tmodel, teacher_ck)
        r = measure(tmodel, loader, device, cfg, args.topk, args.limit, "teacher_blip_large")
        print(json.dumps(r, ensure_ascii=False)); results.append(r)
        del tmodel; torch.cuda.empty_cache()

    json.dump(results, open(args.out, "w"), ensure_ascii=False, indent=2)
    print("\n===== ITM sharpness 요약 (gap = z1 - z0) =====")
    print(f"{'model':26s} {'match_gap p50':>14s} {'distractor_gap p50':>19s} {'sep(mean)':>10s}")
    for r in results:
        mg = r["match_gap"].get("p50", "-"); dg = r["distractor_gap"].get("p50", "-")
        print(f"{r['tag'][:26]:26s} {str(mg):>14s} {str(dg):>19s} {str(r['separation_mean']):>10s}")
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
