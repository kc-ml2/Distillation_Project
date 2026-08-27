"""Phase 0-A: BLIP-large teacher in-batch similarity statistics.

- aug ON 학습 분포의 배치(b40)를 티처에 forward, 40x40 cosine sim 저장
- tau 그리드별 teacher softmax의 top-1/diag 질량, 엔트로피(유효 서포트) 표 출력
- same-image(코코 5캡션/VG 리전) 중복 질량 분리 측정
- 체크포인트의 teacher temp 파라미터 확인 (tau=0.022 출처 검증)
"""
import sys, os, math, argparse
import numpy as np
import torch
import yaml

WT = '/home/minwoo/Distillation_Project_itc_distill_only'
sys.path.insert(0, WT)
os.chdir(WT)  # med config 등 상대경로 대비

from torch.utils.data import DataLoader, Dataset
from data import create_dataset
from distillation.online_teacher import OnlineTeacher


class IndexedDS(Dataset):
    """(image, caption) -> (image, caption, index): 배치 내 same-image 중복 마스크용."""
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        img, cap = self.ds[i]
        return img, cap, i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-batches', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=40)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='/tmp/claude-1014/-home-minwoo-Distillation-Project/748b4c45-6cf3-46aa-8062-bb87cb9ae0e5/scratchpad/teacher_sims.npz')
    ap.add_argument('--no-aug', action='store_true', help='transform_test로 결정적 뷰 사용 (aug 영향 분리)')
    args = ap.parse_args()

    cfg_path = '/home/minwoo/Distillation_Project/output/pt_smallreg_minilm_itc_distill_tau0.022_droptau/config.yaml'
    config = yaml.safe_load(open(cfg_path))
    if args.no_aug:
        config['pretrain_train_aug'] = False
        print('[mode] no-aug (transform_test)')

    # teacher temp 파라미터 확인
    ckpt_path = config['teacher']['checkpoint']
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = state.get('model', state)
    t_temp = float(state['temp']) if 'temp' in state else None
    print(f"[ckpt] teacher learned temp = {t_temp}")
    del state

    ds = create_dataset('pretrain', config)
    print(f"dataset size: {len(ds)}")
    g = torch.Generator().manual_seed(args.seed)
    dl = DataLoader(IndexedDS(ds), batch_size=args.batch_size, shuffle=True,
                    num_workers=8, drop_last=True, generator=g, pin_memory=True)

    teacher = OnlineTeacher(checkpoint=ckpt_path, image_size=config['image_size'],
                            vit='large', bert='base', queue_size=config['queue_size'],
                            keep=('itc',)).to('cuda:0')

    anns = ds.annotation
    sims_all, samemask_all = [], []
    for bi, (image, caption, idx) in enumerate(dl):
        if bi >= args.n_batches:
            break
        img_f, txt_f = teacher.itc_feats(image.to('cuda:0', non_blocking=True), list(caption))
        sims = (img_f.float() @ txt_f.float().t()).cpu().numpy()  # [B,B]
        img_ids = [anns[int(i)]['dataset_source'] + '/' + anns[int(i)]['image'] for i in idx]
        same = np.array([[a == b for b in img_ids] for a in img_ids])  # [B,B] same source image
        sims_all.append(sims)
        samemask_all.append(same)
        if (bi + 1) % 50 == 0:
            print(f"  {bi+1}/{args.n_batches} batches")

    S = np.stack(sims_all)          # [N,B,B]
    M = np.stack(samemask_all)      # [N,B,B] True=same underlying image
    np.savez_compressed(args.out, sims=S, same=M, teacher_temp=t_temp)
    print(f"saved {args.out}  sims={S.shape}")

    analyze(S, M, t_temp)


def analyze(S, M, t_temp):
    N, B, _ = S.shape
    eye = np.eye(B, dtype=bool)
    diag = S[:, eye].reshape(N, B)                     # [N,B] positive sims
    off_mask = ~eye
    dup_off = M & off_mask[None, :, :]                  # same-image but not the pair itself

    print("\n===== cosine sim 분포 =====")
    offs = S[:, off_mask].ravel()
    dups = S[dup_off] if dup_off.any() else np.array([np.nan])
    q = lambda a, p: float(np.percentile(a, p))
    print(f"pos(diag):  mean={diag.mean():.4f} std={diag.std():.4f} p5={q(diag,5):.4f} p50={q(diag,50):.4f} p95={q(diag,95):.4f}")
    print(f"neg(off):   mean={offs.mean():.4f} std={offs.std():.4f} p50={q(offs,50):.4f} p95={q(offs,95):.4f} p99={q(offs,99):.4f} max={offs.max():.4f}")
    print(f"same-image off-diag: count/batch={dup_off.sum()/N:.2f} mean={np.nanmean(dups):.4f} p95={np.nanpercentile(dups,95):.4f}")

    for name, sim in (('i2t', S), ('t2i', np.transpose(S, (0, 2, 1)))):
        d = sim[:, eye].reshape(N, B)
        max_off = np.where(off_mask[None], sim, -np.inf).max(axis=2)   # [N,B]
        margin = d - max_off
        top1_err = float((margin < 0).mean())
        print(f"[{name}] margin(diag-max_off): mean={margin.mean():.4f} p5={np.percentile(margin,5):.4f} "
              f"p50={np.percentile(margin,50):.4f} | teacher in-batch top1 오답률={top1_err*100:.2f}%")

    print("\n===== tau별 teacher softmax (i2t 행 기준, N*B rows) =====")
    header = f"{'tau':>6} {'diag질량':>9} {'top1질량':>9} {'top5질량':>9} {'entropy':>8} {'유효서포트':>9} {'dup질량':>8}"
    print(header)
    for tau in (0.022, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3):
        logits = S / tau
        logits -= logits.max(axis=2, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=2, keepdims=True)               # [N,B,B] row dist
        diag_m = P[:, eye].mean()
        sortP = np.sort(P, axis=2)[:, :, ::-1]
        top1 = sortP[:, :, 0].mean()
        top5 = sortP[:, :, :5].sum(axis=2).mean()
        ent = float((-(P * np.log(np.clip(P, 1e-12, None))).sum(axis=2)).mean())
        dupm = float(np.where(dup_off, P, 0).sum(axis=2).mean())
        print(f"{tau:>6} {diag_m:>9.4f} {top1:>9.4f} {top5:>9.4f} {ent:>8.4f} {math.exp(ent):>9.2f} {dupm:>8.4f}")

    if t_temp is not None:
        print(f"\n(참고) teacher 체크포인트 temp={t_temp:.4f}")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'reanalyze':
        d = np.load(sys.argv[2])
        analyze(d['sims'], d['same'], float(d['teacher_temp']))
    else:
        main()
