#!/usr/bin/env python
"""ITM 학습 배치 로짓 원자료 덤프 — 블록별 z1·z2·m·gap 표.

blip_pretrain.forward 의 ITM 경로(models/blip_pretrain.py:435-489)를 그대로 재현한다.
retrieval 후보가 아니라 학습 배치 3B(서로 다른 쌍)를 재는 것이 요점이다.

재현에서 놓치기 쉬운 세 가지:
  1) negative 샘플링은 momentum 피처를 쓴다. image_feat_all = cat([image_feat_m.t(), queue])
     이고 sim[:, :bs] 가 그 앞부분만 자르므로, 가중치는 text_feat @ image_feat_m.t() 에서 나온다.
     → 학생은 momentum 인코더 4종을 로드해야 한다(갱신은 하지 않는다).
  2) logit_scale 이 샘플링 분포를 바꾼다. safe_scale = clamp(logit_scale).exp() 를 그대로 쓴다.
  3) itm_head 는 autocast 밖 fp32. gap 이 두 로짓의 차라 bf16 이면 유효숫자가 깎인다.
"""
import argparse
import contextlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = "/home/minwoo/Distillation_Project"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.blip_pretrain import LOGIT_SCALE_MAX, LOGIT_SCALE_MIN  # noqa: E402

import probe_core as pc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_TEXT_LEN = 30


def autocast_ctx(device, amp):
    """학습(pretrain.py:140)과 동일한 bf16 autocast. CPU 경로에서는 fp32."""
    if amp and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _tokenize(model, caption, device):
    """(raw_ids, enc_ids, attention_mask) 반환.

    raw_ids : 원본 [CLS] 시작. **ITC 경로(mode='text')가 쓴다** — blip_pretrain.py:388, 404
    enc_ids : 0번을 [ENC] 로 치환. **ITM 경로만 쓴다** — blip_pretrain.py:436-437

    [ENC] 는 additional_special_tokens 로 추가된 별도 토큰이라 [CLS] 와 임베딩이 완전히 다르다.
    text_feat = normalize(text_proj(last_hidden_state[:, 0, :])) 가 바로 그 0번 위치라서,
    ITC forward 에 enc_ids 를 넣으면 text_feat/text_feat_m 이 통째로 달라지고
    → sim → negative 샘플링 분포가 학습과 어긋난다. 프로브의 존재 이유가 깨지는 지점이다.
    """
    text = model.tokenizer(caption, padding="max_length", truncation=True,
                           max_length=MAX_TEXT_LEN, return_tensors="pt").to(device)
    raw_ids = text.input_ids
    enc_ids = raw_ids.clone()
    enc_ids[:, 0] = model.tokenizer.enc_token_id
    return raw_ids, enc_ids, text.attention_mask


def itm_logits_from_negs(model, image_embeds, image_atts, enc_ids, text_atts,
                         neg_idx_img, neg_idx_txt):
    """blip_pretrain.py:441-485 와 동일한 3B 조립 + fp32 헤드.

    주의: text_ids_all 은 [원본, neg] 인데 image_embeds_all 은 [neg, 원본] 으로 순서가 반대다.
    그래서 블록 ②(=[B:2B])가 '원본 텍스트 + negative 이미지'가 된다.
    """
    output_pos = model.text_encoder(enc_ids, attention_mask=text_atts,
                                    encoder_hidden_states=image_embeds,
                                    encoder_attention_mask=image_atts, return_dict=True)

    image_embeds_neg = image_embeds[neg_idx_img]
    text_ids_neg = enc_ids[neg_idx_txt]
    text_atts_neg = text_atts[neg_idx_txt]

    text_ids_all = torch.cat([enc_ids, text_ids_neg], dim=0)
    text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0)
    image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
    image_atts_all = torch.cat([image_atts, image_atts], dim=0)

    output_neg = model.text_encoder(text_ids_all, attention_mask=text_atts_all,
                                    encoder_hidden_states=image_embeds_all,
                                    encoder_attention_mask=image_atts_all, return_dict=True)

    vl = torch.cat([output_pos.last_hidden_state[:, 0, :],
                    output_neg.last_hidden_state[:, 0, :]], dim=0)
    # ★ 헤드는 autocast 밖 fp32
    return F.linear(vl.float(), model.itm_head.weight.float(), model.itm_head.bias.float())


def student_itm_logits(model, image, caption, device, amp):
    """학생: negative 를 직접 뽑고 3B 로짓을 낸다. 반환 (logits, neg_idx_img, neg_idx_txt)."""
    raw_ids, enc_ids, text_atts = _tokenize(model, caption, device)
    with autocast_ctx(device, amp):
        image_embeds = model.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
        image_feat = F.normalize(model.vision_proj(image_embeds[:, 0, :]), dim=-1)
        # ★ ITC 경로는 raw_ids([CLS] 시작). enc_ids 를 쓰면 안 된다 — _tokenize docstring 참고
        text_out = model.text_encoder(raw_ids, attention_mask=text_atts,
                                      return_dict=True, mode="text")
        text_feat = F.normalize(model.text_proj(text_out.last_hidden_state[:, 0, :]), dim=-1)

        # negative 샘플링은 momentum 피처를 쓴다 (queue 부분은 [:, :bs] 슬라이싱으로 배제됨)
        image_embeds_m = model.visual_encoder_m(image)
        image_feat_m = F.normalize(model.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
        text_out_m = model.text_encoder_m(raw_ids, attention_mask=text_atts,
                                          return_dict=True, mode="text")
        text_feat_m = F.normalize(model.text_proj_m(text_out_m.last_hidden_state[:, 0, :]), dim=-1)

        safe_scale = model.logit_scale.clamp(LOGIT_SCALE_MIN, LOGIT_SCALE_MAX).exp()
        sim_i2t = (image_feat @ text_feat_m.t()).float() * safe_scale.float()
        sim_t2i = (text_feat @ image_feat_m.t()).float() * safe_scale.float()

        neg_idx_img = pc.sample_neg_idx(pc.neg_weights(sim_t2i)).to(device)
        neg_idx_txt = pc.sample_neg_idx(pc.neg_weights(sim_i2t)).to(device)

        logits = itm_logits_from_negs(model, image_embeds, image_atts, enc_ids, text_atts,
                                      neg_idx_img, neg_idx_txt)
    return logits.float(), neg_idx_img.cpu(), neg_idx_txt.cpu()


def teacher_itm_logits(model, image, caption, neg_idx_img, neg_idx_txt, device, amp):
    """티처: 학생이 뽑은 negative 를 그대로 재사용해 동일한 3B 쌍을 본다.

    ITM forward 만 하므로 raw_ids 는 쓰지 않는다 (ITC 경로가 없다)."""
    _, enc_ids, text_atts = _tokenize(model, caption, device)
    with autocast_ctx(device, amp):
        image_embeds = model.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
        logits = itm_logits_from_negs(model, image_embeds, image_atts, enc_ids, text_atts,
                                      neg_idx_img.to(device), neg_idx_txt.to(device))
    return logits.float()


def collect_frames(logits_list, batch_ids, n_samples):
    """배치별 [3B,2] 를 블록별로 이어붙이고 앞 n_samples 행만 남긴다."""
    acc = {name: {"batch": [], "i": [], "logits": []} for name in pc.BLOCK_NAMES}
    for logits, bid in zip(logits_list, batch_ids):
        blocks = pc.split_blocks(logits)
        for name in pc.BLOCK_NAMES:
            blk = blocks[name].detach().cpu().numpy()
            acc[name]["logits"].append(blk)
            acc[name]["batch"].extend([bid] * len(blk))
            acc[name]["i"].extend(range(len(blk)))
    frames = {}
    for name in pc.BLOCK_NAMES:
        arr = np.concatenate(acc[name]["logits"], axis=0)[:n_samples]
        frames[name] = pc.make_frame(acc[name]["batch"][:n_samples],
                                     acc[name]["i"][:n_samples], arr)
    return frames


def emit(tag, frames, out_dir):
    """CSV 3개 + npz 1개를 쓰고 요약을 반환한다."""
    csv_dir = os.path.join(out_dir, "csv")
    npz_dir = os.path.join(out_dir, "npz")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)
    payload = {}
    for name in pc.BLOCK_NAMES:
        pc.write_block_csv(os.path.join(csv_dir, f"{tag}_{name}.csv"), frames[name])
        for col in ("batch", "i", *pc.COLUMNS):
            payload[f"{name}__{col}"] = frames[name][col]
    np.savez_compressed(os.path.join(npz_dir, f"{tag}.npz"), **payload)
    return pc.summarize_blocks(frames)


def _print_summary(tag, summary):
    print(f"\n===== {tag} =====")
    print(f"delta (pos gap mean - pooled neg gap mean) = {summary['delta']:.4f}")
    print(f"P_pos if merged: measured={summary['p_pos_merged_measured']:.4f}  "
          f"from_delta={summary['p_pos_merged_from_delta']:.4f}")
    head = f"{'block':<9}{'n':>5}" + "".join(f"{c:>26}" for c in pc.COLUMNS)
    print(head)
    for name in pc.BLOCK_NAMES:
        st = summary[name]["stats"]
        cells = "".join(f"{st[c]['mean']:>13.4f}{st[c]['var']:>13.4f}" for c in pc.COLUMNS)
        print(f"{name:<9}{summary['n_per_block'][name]:>5}{cells}")
    print("           (각 열은 mean / var 순)")
    for name in pc.BLOCK_NAMES:
        s = summary[name]
        print(f"  {name:<9} sigma_m={s['sigma_m']:.4f}  sigma_gap={s['sigma_gap']:.4f}")


def _build_loader(cfg, num_workers):
    """sampler=None + is_trains=[True] → create_loader 가 shuffle=True, drop_last=True 로 만든다
    (data/__init__.py:102-104). RandomSampler 가 전역 RNG 를 쓰므로, 호출 전에 main 에서
    torch.manual_seed 를 걸어 두면 배치 순서가 재현된다."""
    from data import create_dataset, create_loader
    ds = create_dataset("pretrain", cfg, min_scale=0.2)
    return create_loader([ds], [None], batch_size=[cfg["batch_size"]],
                         num_workers=[num_workers], is_trains=[True], collate_fns=[None])[0]


def _load_student(cfg, ckpt, device):
    import utils
    from models.blip_pretrain import blip_pretrain
    model = blip_pretrain(image_size=cfg["image_size"], vit=cfg["vit"],
                          vit_grad_ckpt=cfg["vit_grad_ckpt"], vit_ckpt_layer=cfg["vit_ckpt_layer"],
                          queue_size=cfg["queue_size"], my_bert_size=cfg["my_bert_size"])
    model = utils.load_model_weights_only(model, ckpt)
    return model.to(device).eval()


def _load_teacher(cfg, ckpt, device):
    import utils
    from models.blip_pretrain import blip_pretrain
    # init_backbone_weights=False: large 의 in21k 초기화 경로가 timm 1.x 에서 깨져 있고,
    # 어차피 체크포인트가 가중치를 전량 덮는다.
    # queue_size=cfg["queue_size"]: 브리프 원안은 240 이었으나, 실제 model_large.pth 의
    # image_queue/text_queue 버퍼가 [256, 57600] 이라 240 으로 만들면 load_state_dict 가
    # size mismatch 로 즉시 실패한다(strict=False 는 shape mismatch 를 봐주지 않는다).
    # queue 는 파라미터가 아니라 buffer 이고 ITM-only 프로브는 이를 전혀 쓰지 않으므로
    # cfg 값으로 맞춰도 안전하다 — 자매 프로브(itm_sharpness_probe.py:121,
    # itm_logit_dump.py:259)도 동일하게 cfg["queue_size"] 를 쓴다.
    model = blip_pretrain(image_size=cfg["image_size"], vit="large", vit_grad_ckpt=False,
                          vit_ckpt_layer=0, queue_size=cfg["queue_size"], my_bert_size="base",
                          init_backbone_weights=False)
    model = utils.load_model_weights_only(model, ckpt)
    # 티처는 negative 를 뽑지 않으므로 momentum 인코더가 필요 없다 → 메모리 해제
    for attr in ("visual_encoder_m", "text_encoder_m", "vision_proj_m", "text_proj_m"):
        setattr(model, attr, None)
    return model.to(device).eval()


def main():
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO, "output/pt_smallreg_minilm_baseline/config.yaml"))
    ap.add_argument("--student-ckpt", default=os.path.join(REPO, "output/pt_smallreg_minilm_baseline/checkpoint_19.pth"))
    ap.add_argument("--teacher-ckpt", default=os.path.join(REPO, "output/official_pretrain_checkpoint/model_large.pth"))
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--amp", action="store_true", help="GPU 경로에서만 bf16 autocast")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--skip-teacher", action="store_true")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cfg = yaml.safe_load(open(args.config))

    loader = _build_loader(cfg, args.num_workers)
    n_batches = -(-args.n_samples // cfg["batch_size"])   # ceil
    print(f"[plan] batch_size={cfg['batch_size']}  n_batches={n_batches}  "
          f"n_samples={args.n_samples}  device={device}")

    # ---- 1) 학생: forward + 배치 캐시 (티처가 동일한 3B 쌍을 보도록)
    print("[load] student ...", flush=True)
    student = _load_student(cfg, args.student_ckpt, device)
    cache, s_logits, batch_ids = [], [], []
    for bi, (image, caption) in enumerate(loader):
        if bi >= n_batches:
            break
        image = image.to(device)
        logits, neg_img, neg_txt = student_itm_logits(student, image, caption, device, args.amp)
        s_logits.append(logits.cpu())
        batch_ids.append(bi)
        cache.append((image.cpu(), list(caption), neg_img, neg_txt))
        print(f"  [student] batch {bi + 1}/{n_batches}", flush=True)
    del student
    if device.type == "cuda":
        torch.cuda.empty_cache()

    summaries = {}
    frames_s = collect_frames(s_logits, batch_ids, args.n_samples)
    summaries["student"] = emit("student", frames_s, args.out)
    _print_summary("student", summaries["student"])

    # ---- 2) 티처: 캐시된 배치 + 학생이 뽑은 negative 재사용
    if not args.skip_teacher:
        print("[load] teacher ...", flush=True)
        teacher = _load_teacher(cfg, args.teacher_ckpt, device)
        t_logits = []
        for bi, (image, caption, neg_img, neg_txt) in enumerate(cache):
            logits = teacher_itm_logits(teacher, image.to(device), caption,
                                        neg_img, neg_txt, device, args.amp)
            t_logits.append(logits.cpu())
            print(f"  [teacher] batch {bi + 1}/{len(cache)}", flush=True)
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
        frames_t = collect_frames(t_logits, batch_ids, args.n_samples)
        summaries["teacher"] = emit("teacher", frames_t, args.out)
        _print_summary("teacher", summaries["teacher"])

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"\nsaved: {os.path.join(args.out, 'summary.json')}")


if __name__ == "__main__":
    main()
