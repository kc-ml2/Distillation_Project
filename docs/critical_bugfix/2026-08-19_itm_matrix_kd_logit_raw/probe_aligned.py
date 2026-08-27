#!/usr/bin/env python
"""행-정렬 3-모델 ITM 로짓 프로브 (teacher / student_baseline / student_itm_kxk).

시드 고정 단일 패스. student_itm_kxk(distilled)가 negative를 뽑고,
teacher·student_baseline 이 **그 동일 negative** 를 재사용 → 행 N = 세 모델 동일한 3B 쌍.
07-31 프로브 함수(_load_*, student_itm_logits, teacher_itm_logits, collect_frames, emit)를
그대로 재사용. teacher_itm_logits 는 '주어진 negative 로 채점'이라 baseline 채점에도 그대로 씀.

산출: aligned/csv/{tag}_{block}.csv, aligned/npz/{tag}.npz, aligned/summary.json
"""
import argparse, json, os, sys
REPO = "/home/minwoo/Distillation_Project"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "critical_bugfix/2026-07-31_itm_batch_logit_raw"))
import torch, yaml                                   # noqa: E402
import itm_batch_logit_probe as probe                # noqa: E402

KXK_CKPT = f"{REPO}/output/pt_smallreg_minilm_itm_matrix_kd_configfix/checkpoint_19.pth"
KXK_CFG  = f"{REPO}/output/pt_smallreg_minilm_itm_matrix_kd_configfix/config.yaml"
BASE_CKPT = f"{REPO}/output/pt_smallreg_minilm_baseline/checkpoint_19.pth"
BASE_CFG  = f"{REPO}/output/pt_smallreg_minilm_baseline/config.yaml"
TEACHER_CKPT = f"{REPO}/output/official_pretrain_checkpoint/model_large.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "aligned"))
    a = ap.parse_args()
    torch.set_grad_enabled(False); torch.manual_seed(a.seed)
    device = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)

    cfg_k = yaml.safe_load(open(KXK_CFG)); cfg_b = yaml.safe_load(open(BASE_CFG))
    loader = probe._build_loader(cfg_k, a.num_workers)
    n_batches = -(-a.n_samples // cfg_k["batch_size"])
    print(f"[plan] n_batches={n_batches} n_samples={a.n_samples} device={device} seed={a.seed}")

    # 공통 이미지/캡션 배치 (세 모델 동일)
    images = []
    for bi, (image, caption) in enumerate(loader):
        if bi >= n_batches:
            break
        images.append((image.cpu(), list(caption)))
    batch_ids = list(range(len(images)))

    # 1) student_itm_kxk: negative 샘플 + 로짓, negative 캐시
    print("[load] student_itm_kxk ...", flush=True)
    kxk = probe._load_student(cfg_k, KXK_CKPT, device)
    kxk_logits, cache = [], []
    for img, cap in images:
        logits, ni, nt = probe.student_itm_logits(kxk, img.to(device), cap, device, a.amp)
        kxk_logits.append(logits.cpu()); cache.append((img, cap, ni, nt))
    del kxk

    # 2) teacher: 동일 negative 재사용
    print("[load] teacher ...", flush=True)
    teacher = probe._load_teacher(cfg_k, TEACHER_CKPT, device)
    t_logits = [probe.teacher_itm_logits(teacher, img.to(device), cap, ni, nt, device, a.amp).cpu()
                for img, cap, ni, nt in cache]
    del teacher

    # 3) student_baseline: 동일 negative 재사용 (teacher_itm_logits = '주어진 neg 로 채점')
    print("[load] student_baseline ...", flush=True)
    base = probe._load_student(cfg_b, BASE_CKPT, device)
    b_logits = [probe.teacher_itm_logits(base, img.to(device), cap, ni, nt, device, a.amp).cpu()
                for img, cap, ni, nt in cache]
    del base

    summaries = {}
    for tag, L in (("teacher", t_logits), ("student_baseline", b_logits), ("student_itm_kxk", kxk_logits)):
        frames = probe.collect_frames(L, batch_ids, a.n_samples)
        summaries[tag] = probe.emit(tag, frames, a.out)
        probe._print_summary(tag, summaries[tag])
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print("\nsaved:", os.path.join(a.out, "summary.json"))


if __name__ == "__main__":
    main()
