#!/usr/bin/env python
"""scheme A(ITM tempered soft, W=1.0/T=2) sharpness 프로브.
기존 itm_sharpness_probe.py의 measure/pick_ckpt를 재사용해 scheme A(ep3, ep15)와
baseline(ep15, 수렴 레퍼런스)를 같은 조건으로 재서 arm C(0.64)/baseline(1.15) 표와 비교.
학습과 공존: patch feats는 CPU, GPU는 모델+forward만 → 여유 GPU 1장에서 OK."""
import sys, os, json
REPO = "/home/minwoo/Distillation_Project"
PROBE_DIR = os.path.join(REPO, "critical_bugfix/2026-07-24_itm_distill_sharpness")
sys.path.insert(0, PROBE_DIR)

import yaml, torch
import itm_sharpness_probe as P   # module-level: chdir(REPO) + sys.path insert
import utils
from models.blip_pretrain import blip_pretrain
from data.eval_validation_retrieval import build_coco_karpathy_retrieval_val_loader

OUT = "/tmp/claude-1014/-home-minwoo-Distillation-Project/faaf594e-92a3-4870-8d7b-737d92e4eeea/scratchpad/schemeA_probe.json"
TOPK, LIMIT = 32, 500

cfg = yaml.safe_load(open(os.path.join(REPO, "output/pt_itm_schemeA_tempered/config.yaml")))
cfg["val_retrieval_enabled"] = True
cfg["val_retrieval_split"] = "val"
device = torch.device("cuda")
print("[build] coco karpathy retrieval val loader ...", flush=True)
loader = build_coco_karpathy_retrieval_val_loader(cfg)

targets = [
    ("schemeA@ep3",   "pt_itm_schemeA_tempered",     3),
    ("schemeA@ep15",  "pt_itm_schemeA_tempered",     15),
    ("baseline@ep15", "pt_smallreg_minilm_baseline", 15),
]
results = []
for tag, run, ep in targets:
    ck = P.pick_ckpt(run, prefer_epoch=ep)
    if ck is None:
        print(f"[skip] {tag}: no ckpt in {run}", flush=True); continue
    print(f"[load] {tag} <- {os.path.basename(ck)}", flush=True)
    model = blip_pretrain(image_size=cfg["image_size"], vit=cfg["vit"],
                          vit_grad_ckpt=cfg["vit_grad_ckpt"], vit_ckpt_layer=cfg["vit_ckpt_layer"],
                          queue_size=cfg["queue_size"], my_bert_size=cfg["my_bert_size"]).to(device)
    model = utils.load_model_weights_only(model, ck)
    r = P.measure(model, loader, device, cfg, TOPK, LIMIT, tag)
    print(json.dumps(r, ensure_ascii=False), flush=True)
    results.append(r)
    del model; torch.cuda.empty_cache()

json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)
print("\n===== scheme A sharpness (gap = z1 - z0), topk=%d limit=%d =====" % (TOPK, LIMIT), flush=True)
print(f"{'model':16s} {'match_gap p50':>14s} {'distr_gap p50':>14s} {'sep(mean)':>10s}", flush=True)
for r in results:
    mg = r["match_gap"].get("p50", "-"); dg = r["distractor_gap"].get("p50", "-")
    print(f"{r['tag'][:16]:16s} {str(mg):>14s} {str(dg):>14s} {str(r['separation_mean']):>10s}", flush=True)
print("\nref(기존 json @ep3): baseline sep 1.149 / armC sep 0.64 / teacher sep 3.40", flush=True)
print("saved:", OUT, flush=True)
