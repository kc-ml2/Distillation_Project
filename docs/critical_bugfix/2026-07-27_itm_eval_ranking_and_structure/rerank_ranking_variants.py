#!/usr/bin/env python
"""ITM rerank 랭킹량 비교: 같은 ITC top-128을 여러 점수로 재정렬했을 때 R@1/5/10/MRR.

핵심 질문: 현재 eval은 랭킹을 z1(match 로짓 절대값)으로 매긴다(eval_validation_tool.py:163,
softmax 없음). 그런데 ITM 학습(softmax CE)은 gap=z1-z0만 최적화하고 z1의 절대 offset은
loss-unconstrained(shift-invariant)라 랭킹 관점에선 노이즈다. 그러면 gap(=softmax prob)으로
재정렬하면 R@k가 오를까? 오르면 'ITM 실패'의 일부는 헤드가 아니라 z1-랭킹이라는 eval 선택 탓.

variants (모두 동일한 cosine-top128 위에서 순서만 다르게):
  cos      : ITC cosine (= ITM 없이, top128 내 ITC 순서 = no-ITM 바닥)
  z1       : 순수 match 로짓 (offset 노이즈 포함)
  z1+cos   : 현행 eval 방식 (eval_validation_tool.py:164)
  gap      : z1-z0 (= logit of softmax prob; 학습이 실제 최적화한 양)
  gap+cos  : gap + cosine

실행: CUDA_VISIBLE_DEVICES=<free> python rerank_ranking_variants.py
"""
import sys, os, json
REPO = "/home/minwoo/Distillation_Project"
sys.path.insert(0, os.path.join(REPO, "critical_bugfix/2026-07-24_itm_distill_sharpness"))

import numpy as np, yaml, torch
import itm_sharpness_probe as P   # module-level: chdir(REPO)+path insert; gives pick_ckpt
import utils
from models.blip_pretrain import blip_pretrain
from data.eval_validation_retrieval import build_coco_karpathy_retrieval_val_loader
from eval_validation_tool import _encode_retrieval_features, _retrieval_autocast_context

OUT = os.path.join(REPO, "critical_bugfix/2026-07-27_itm_eval_ranking_and_structure/rerank_ranking_variants_results.json")
TOPK, LIMIT = 128, 500
VARIANTS = ["cos", "z1", "z1+cos", "gap", "gap+cos"]

cfg = yaml.safe_load(open(os.path.join(REPO, "output/pt_itm_schemeA_tempered/config.yaml")))
cfg["val_retrieval_enabled"] = True
cfg["val_retrieval_split"] = "val"
device = torch.device("cuda")
print("[build] loader ...", flush=True)
loader = build_coco_karpathy_retrieval_val_loader(cfg)
img2txt = loader.dataset.img2txt
txt2img = getattr(loader.dataset, "txt2img", None)
if txt2img is None:  # fallback: derive from img2txt
    txt2img = {}
    for im, txts in enumerate(img2txt):
        for t in txts:
            txt2img[t] = im


def rank_true(ordered_orig, gt_set):
    for r, j in enumerate(ordered_orig):
        if int(j) in gt_set:
            return r
    return TOPK  # not in top-k -> miss


def metr(ranks):
    r = np.asarray(ranks, dtype=np.float64)
    return dict(r1=round(100*np.mean(r < 1), 2), r5=round(100*np.mean(r < 5), 2),
                r10=round(100*np.mean(r < 10), 2), mrr=round(100*np.mean(1.0/(r+1)), 2),
                rmean=round(100*(np.mean(r < 1)+np.mean(r < 5)+np.mean(r < 10))/3, 2))


@torch.no_grad()
def run_direction(model, sims_dir, image_feats, text_ids, text_atts, is_i2t):
    ranks = {v: [] for v in VARIANTS}
    n = min(LIMIT, sims_dir.size(0))
    for i in range(n):
        topk_sim, topk_idx = sims_dir[i].topk(k=min(TOPK, sims_dir.size(1)))
        with _retrieval_autocast_context(device, cfg):
            if is_i2t:
                enc = image_feats[i].repeat(topk_idx.size(0), 1, 1).to(device)
                att = torch.ones(enc.size()[:-1], dtype=torch.long, device=device)
                out = model.text_encoder(text_ids[topk_idx], attention_mask=text_atts[topk_idx],
                                         encoder_hidden_states=enc, encoder_attention_mask=att, return_dict=True)
                gt = set(int(x) for x in img2txt[i])
            else:
                enc = image_feats[topk_idx.cpu()].to(device)
                att = torch.ones(enc.size()[:-1], dtype=torch.long, device=device)
                out = model.text_encoder(text_ids[i].repeat(topk_idx.size(0), 1),
                                         attention_mask=text_atts[i].repeat(topk_idx.size(0), 1),
                                         encoder_hidden_states=enc, encoder_attention_mask=att, return_dict=True)
                gt = {int(txt2img[i])}
            logits = model.itm_head(out.last_hidden_state[:, 0, :]).float()
        z1, z0 = logits[:, 1], logits[:, 0]
        gap = z1 - z0
        cos = topk_sim.to(z1.dtype)
        scores = {"cos": cos, "z1": z1, "z1+cos": z1 + cos, "gap": gap, "gap+cos": gap + cos}
        topk_orig = topk_idx.cpu().numpy()
        for v in VARIANTS:
            order = torch.argsort(scores[v], descending=True).cpu().numpy()
            ranks[v].append(rank_true(topk_orig[order], gt))
    return ranks


@torch.no_grad()
def evaluate(model):
    model.eval()
    with _retrieval_autocast_context(device, cfg):
        sims, image_feats, text_ids, text_atts = _encode_retrieval_features(model, loader, device, need_patch_feats=True)
    i2t = run_direction(model, sims, image_feats, text_ids, text_atts, True)
    t2i = run_direction(model, sims.t().contiguous(), image_feats, text_ids, text_atts, False)
    res = {}
    for v in VARIANTS:
        ti, ir = metr(i2t[v]), metr(t2i[v])
        res[v] = dict(i2t=ti, t2i=ir, r_mean=round((ti["rmean"] + ir["rmean"]) / 2, 2))
    return res


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
print(f"\n===== ITM rerank 랭킹방식별 r_mean (top-{TOPK}, {LIMIT} queries/direction) =====", flush=True)
hdr = f"{'variant':10s}" + "".join(f"{t.split('@')[0]:>16s}" for t, _, _ in targets)
print(hdr, flush=True)
for v in VARIANTS:
    print(f"{v:10s}" + "".join(f"{allres[t][v]['r_mean']:>16.2f}" for t, _, _ in targets), flush=True)
print("\nsaved:", OUT, flush=True)
