#!/usr/bin/env python
"""ITM per-pair 이진 정답률 (= 학습 손실과 동일 타깃) — 유저 정의.
이미지 1개에 대해: 정답 캡션 -> match(gap>0)면 +1, 오답 캡션 -> no-match(gap<0)면 +1.

불균형(정답 1 : 오답 다수) 처리:
  - 클래스별로 따로 본다: TPR(정답쌍 맞힘률) / TNR(오답쌍 거부율)
  - 유저 '정답률' = balanced accuracy = (TPR + TNR)/2  (raw accuracy는 오답에 압도돼 무의미)
  - 네거티브를 난이도로 분리: HARD(ITC top-K, 배포형) vs EASY(랜덤, 학습 쉬운 네거티브 근사)
  - threshold=0(학습 argmax 경계)뿐 아니라 OPT(최적 임계 balanced acc)·AUC(임계 무관 마진)도 보고
    -> threshold-0에서 degenerate(둘 다 하드네거를 match로)인지, 마진(AUC)에서만 갈리는지 확인.

i2t(이미지->캡션) 방향. baseline@ep15 vs schemeA@ep15.
"""
import sys, os, json
REPO = "/home/minwoo/Distillation_Project"
sys.path.insert(0, os.path.join(REPO, "critical_bugfix/2026-07-24_itm_distill_sharpness"))
import numpy as np, yaml, torch
import itm_sharpness_probe as P
import utils
from models.blip_pretrain import blip_pretrain
from data.eval_validation_retrieval import build_coco_karpathy_retrieval_val_loader
from eval_validation_tool import _encode_retrieval_features, _retrieval_autocast_context

OUT = os.path.join(REPO, "critical_bugfix/2026-07-27_itm_eval_ranking_and_structure/binary_accuracy_results.json")
LIMIT, K_HARD, K_EASY, K_TOPEXCL = 500, 32, 16, 128
rng = np.random.default_rng(0)

cfg = yaml.safe_load(open(os.path.join(REPO, "output/pt_itm_schemeA_tempered/config.yaml")))
cfg["val_retrieval_enabled"] = True; cfg["val_retrieval_split"] = "val"
device = torch.device("cuda")
print("[build] loader ...", flush=True)
loader = build_coco_karpathy_retrieval_val_loader(cfg)
img2txt = loader.dataset.img2txt
n_text = len(loader.dataset.text)


def auc(pos, neg):
    pos, neg = np.sort(np.asarray(pos)), np.sort(np.asarray(neg))
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    idx = np.searchsorted(neg, pos, side="left")  # #neg strictly < each pos
    return float(idx.sum() / (len(pos) * len(neg)))


def opt_bal_acc(pos, neg):
    lo, hi = min(pos.min(), neg.min()), max(pos.max(), neg.max())
    ths = np.linspace(lo, hi, 200)
    best = 0.0
    for t in ths:
        tpr = np.mean(pos > t); tnr = np.mean(neg <= t)
        best = max(best, (tpr + tnr) / 2)
    return float(best)


@torch.no_grad()
def evaluate(model):
    model.eval()
    with _retrieval_autocast_context(device, cfg):
        sims, image_feats, text_ids, text_atts = _encode_retrieval_features(model, loader, device, need_patch_feats=True)
    g_true, g_hard, g_easy = [], [], []
    n = min(LIMIT, sims.size(0))
    for i in range(n):
        gt = list(img2txt[i])
        top = sims[i].topk(k=min(K_TOPEXCL, sims.size(1))).indices.cpu().numpy().tolist()
        gt_set, top_set = set(gt), set(top)
        hard = [j for j in top if j not in gt_set][:K_HARD]
        easy_pool = np.array([j for j in range(n_text) if j not in top_set and j not in gt_set])
        easy = rng.choice(easy_pool, size=min(K_EASY, len(easy_pool)), replace=False).tolist()
        cand = gt + hard + easy
        idx = torch.tensor(cand, device=text_ids.device)
        enc = image_feats[i].repeat(len(cand), 1, 1).to(device)
        att = torch.ones(enc.size()[:-1], dtype=torch.long, device=device)
        with _retrieval_autocast_context(device, cfg):
            out = model.text_encoder(text_ids[idx], attention_mask=text_atts[idx],
                                     encoder_hidden_states=enc, encoder_attention_mask=att, return_dict=True)
            logits = model.itm_head(out.last_hidden_state[:, 0, :]).float()
        gap = (logits[:, 1] - logits[:, 0]).cpu().numpy()
        a, b = len(gt), len(gt) + len(hard)
        g_true += gap[:a].tolist(); g_hard += gap[a:b].tolist(); g_easy += gap[b:].tolist()
    g_true, g_hard, g_easy = map(lambda x: np.asarray(x, dtype=np.float64), (g_true, g_hard, g_easy))
    TPR = float(np.mean(g_true > 0))
    TNR_h, TNR_e = float(np.mean(g_hard < 0)), float(np.mean(g_easy < 0))
    raw_hard = float((np.sum(g_true > 0) + np.sum(g_hard < 0)) / (len(g_true) + len(g_hard)))
    return dict(
        n=dict(true=len(g_true), hard=len(g_hard), easy=len(g_easy)),
        mean_gap=dict(true=round(g_true.mean(), 3), hard=round(g_hard.mean(), 3), easy=round(g_easy.mean(), 3)),
        TPR=round(100*TPR, 1), TNR_hard=round(100*TNR_h, 1), TNR_easy=round(100*TNR_e, 1),
        bal_acc_hard_th0=round(100*(TPR+TNR_h)/2, 1), bal_acc_easy_th0=round(100*(TPR+TNR_e)/2, 1),
        raw_acc_hard_th0_imbalanced=round(100*raw_hard, 1),
        bal_acc_hard_OPT=round(100*opt_bal_acc(g_true, g_hard), 1),
        bal_acc_easy_OPT=round(100*opt_bal_acc(g_true, g_easy), 1),
        AUC_hard=round(100*auc(g_true, g_hard), 1), AUC_easy=round(100*auc(g_true, g_easy), 1),
    )


targets = [("baseline@ep15", "pt_smallreg_minilm_baseline", 15),
           ("schemeA@ep15", "pt_itm_schemeA_tempered", 15)]
allres = {}
for tag, run, ep in targets:
    ck = P.pick_ckpt(run, prefer_epoch=ep)
    print(f"[load] {tag} <- {os.path.basename(ck)}", flush=True)
    model = blip_pretrain(image_size=cfg["image_size"], vit=cfg["vit"], vit_grad_ckpt=cfg["vit_grad_ckpt"],
                          vit_ckpt_layer=cfg["vit_ckpt_layer"], queue_size=cfg["queue_size"],
                          my_bert_size=cfg["my_bert_size"]).to(device)
    model = utils.load_model_weights_only(model, ck)
    allres[tag] = evaluate(model)
    print(tag, json.dumps(allres[tag], ensure_ascii=False), flush=True)
    del model; torch.cuda.empty_cache()

json.dump(allres, open(OUT, "w"), ensure_ascii=False, indent=2)
print(f"\n===== ITM 이진 정답률 (i2t, {LIMIT} imgs, HARD=top{K_HARD} / EASY=random{K_EASY}) =====", flush=True)
cols = ["TPR", "TNR_hard", "TNR_easy", "bal_acc_hard_th0", "bal_acc_easy_th0",
        "raw_acc_hard_th0_imbalanced", "bal_acc_hard_OPT", "AUC_hard", "AUC_easy"]
print(f"{'metric':30s}" + "".join(f"{t.split('@')[0]:>14s}" for t, _, _ in targets), flush=True)
for c in cols:
    print(f"{c:30s}" + "".join(f"{allres[t][c]:>14}" for t, _, _ in targets), flush=True)
print("\nsaved:", OUT, flush=True)
