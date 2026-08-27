#!/usr/bin/env python
"""COCO Karpathy val 에서 ITC-only / ITM-rerank retrieval R@k 실측.

동기 — 학습 배치(3B) 프로브에서 티처가 라벨과 자주 어긋나는 것으로 나왔다
(pos 의 36.6% 를 no-match, neg 의 18.2% 를 match). 그것이 "티처 ITM 이 나쁘다"는 뜻인지,
아니면 라벨 쪽 문제(VG region description 90% + min_scale=0.2 crop)인지 가르려면
**실제 랭킹 태스크**에서 재봐야 한다.

기존 평가 경로를 그대로 호출한다. 재구현한 것은 없다:
  - data.eval_validation_retrieval.build_coco_karpathy_retrieval_val_loader
  - eval_validation_tool.evaluate_retrieval_itc / evaluate_retrieval_itm / itm_eval

GPU 4장이 학습 잡으로 점유돼 있어 CPU 로 돈다. 전량(5000 img × 25010 txt)은 CPU 로 불가능하므로
annotation JSON 을 앞 N 이미지로 잘라 임시 ann_root 에 두고, 데이터셋 클래스가 스스로
text/image/img2txt/txt2img 를 재구성하게 한다(자르는 로직을 복제하지 않는다).

⚠️ 서브셋 R@k 를 전량 R@k 나 BLIP 논문 수치와 직접 비교하면 안 된다 — 후보 풀이 작아질수록
   태스크가 쉬워져 값이 부풀려진다. **같은 N 에서 잰 모델끼리만** 비교가 성립한다.

실행 (repo root):
  CUDA_VISIBLE_DEVICES="" python critical_bugfix/2026-07-31_itm_batch_logit_raw/coco_retrieval_check.py \
      --arch teacher --n-images 100
"""
import argparse
import json
import os
import sys
import time

REPO = "/home/minwoo/Distillation_Project"
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch
import yaml

import utils
from models.blip_pretrain import blip_pretrain
from data.eval_validation_retrieval import build_coco_karpathy_retrieval_val_loader
from eval_validation_tool import (evaluate_retrieval_itc, evaluate_retrieval_itm, itm_eval)

HERE = os.path.dirname(os.path.abspath(__file__))


def subset_cfg(cfg, n_images, tmpdir):
    """annotation 을 앞 n_images 로 자른 임시 ann_root 를 가리키는 cfg 사본을 만든다.

    데이터셋 클래스(coco_karpathy_retrieval_eval)가 이 JSON 하나로부터 text/image/
    img2txt/txt2img 를 전부 재구성하므로, 인덱스 정합성을 손으로 맞출 필요가 없다.
    """
    src = os.path.join(cfg["val_retrieval_ann_root"], "coco_karpathy_val.json")
    ann = json.load(open(src))
    if n_images > 0:
        ann = ann[:n_images]
    os.makedirs(tmpdir, exist_ok=True)
    json.dump(ann, open(os.path.join(tmpdir, "coco_karpathy_val.json"), "w"))
    out = dict(cfg)
    out["val_retrieval_ann_root"] = tmpdir + os.sep
    out["val_retrieval_split"] = "val"
    return out, len(ann), sum(len(a["caption"]) for a in ann)


def load_model(arch, cfg, ckpt, device):
    if arch == "teacher":
        # init_backbone_weights=False: large 의 in21k 초기화 경로가 timm 1.x 에서 깨져 있고,
        # 체크포인트가 가중치를 전량 덮는다. queue_size 는 체크포인트 버퍼 모양과 맞아야 한다.
        model = blip_pretrain(image_size=cfg["image_size"], vit="large", vit_grad_ckpt=False,
                              vit_ckpt_layer=0, queue_size=cfg["queue_size"],
                              my_bert_size="base", init_backbone_weights=False)
    else:
        model = blip_pretrain(image_size=cfg["image_size"], vit=cfg["vit"],
                              vit_grad_ckpt=cfg["vit_grad_ckpt"], vit_ckpt_layer=cfg["vit_ckpt_layer"],
                              queue_size=cfg["queue_size"], my_bert_size=cfg["my_bert_size"])
    model = utils.load_model_weights_only(model, ckpt)
    return model.to(device).eval()


def show(tag, res):
    print(f"\n----- {tag} -----")
    print(f"  i2t (txt retrieval)  R@1={res['txt_r1']:.2f}  R@5={res['txt_r5']:.2f}  "
          f"R@10={res['txt_r10']:.2f}  mean={res['txt_r_mean']:.2f}")
    print(f"  t2i (img retrieval)  R@1={res['img_r1']:.2f}  R@5={res['img_r5']:.2f}  "
          f"R@10={res['img_r10']:.2f}  mean={res['img_r_mean']:.2f}")
    print(f"  r_mean = {res['r_mean']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="teacher", choices=["teacher", "student"])
    ap.add_argument("--ckpt", default=None, help="생략 시 arch 별 기본 체크포인트")
    ap.add_argument("--config", default=os.path.join(REPO, "output/pt_smallreg_minilm_baseline/config.yaml"))
    ap.add_argument("--n-images", type=int, default=100, help="0 이면 전량(5000)")
    ap.add_argument("--k-test", type=int, default=None, help="생략 시 config 의 k_test")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-itm", action="store_true", help="ITC-only 만 (속도 측정용)")
    ap.add_argument("--out", default=os.path.join(HERE, "coco_retrieval_check.json"))
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    cfg = yaml.safe_load(open(args.config))
    if args.k_test is not None:
        cfg["k_test"] = args.k_test

    ckpt = args.ckpt or (os.path.join(REPO, "output/official_pretrain_checkpoint/model_large.pth")
                         if args.arch == "teacher"
                         else os.path.join(REPO, "output/pt_smallreg_minilm_baseline/checkpoint_19.pth"))

    tmpdir = os.path.join(HERE, f"_ann_subset_{args.n_images}")
    cfg, n_img, n_txt = subset_cfg(cfg, args.n_images, tmpdir)
    print(f"[setup] arch={args.arch}  n_images={n_img}  n_texts={n_txt}  "
          f"k_test={cfg.get('k_test', 128)}  device={device}", flush=True)
    print(f"[setup] ckpt={os.path.basename(ckpt)}", flush=True)

    loader = build_coco_karpathy_retrieval_val_loader(cfg)
    model = load_model(args.arch, cfg, ckpt, device)

    txt2img, img2txt = loader.dataset.txt2img, loader.dataset.img2txt
    out = {"arch": args.arch, "checkpoint": os.path.basename(ckpt),
           "n_images": n_img, "n_texts": n_txt, "k_test": cfg.get("k_test", 128),
           "device": str(device)}

    t0 = time.time()
    i2t, t2i = evaluate_retrieval_itc(model, loader, device, cfg)
    out["itc"] = itm_eval(i2t, t2i, txt2img, img2txt)
    out["itc_seconds"] = round(time.time() - t0, 1)
    show("ITC-only (cosine)", out["itc"])
    print(f"  [{out['itc_seconds']}s]", flush=True)

    if not args.skip_itm:
        t0 = time.time()
        i2t, t2i = evaluate_retrieval_itm(model, loader, device, cfg)
        out["itm"] = itm_eval(i2t, t2i, txt2img, img2txt)
        out["itm_seconds"] = round(time.time() - t0, 1)
        show("ITM-rerank (z2 + cosine)", out["itm"])
        print(f"  [{out['itm_seconds']}s]", flush=True)
        d = out["itm"]["r_mean"] - out["itc"]["r_mean"]
        print(f"\n  ITM 재정렬 이득: r_mean {out['itc']['r_mean']:.2f} -> {out['itm']['r_mean']:.2f}  ({d:+.2f})")

    prev = {}
    if os.path.exists(args.out):
        prev = json.load(open(args.out))
    prev[f"{args.arch}_n{n_img}_k{cfg.get('k_test', 128)}"] = out
    json.dump(prev, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
