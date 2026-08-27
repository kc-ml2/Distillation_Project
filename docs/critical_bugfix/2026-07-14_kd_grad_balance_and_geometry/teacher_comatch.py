"""Phase 0-D: 티처 in-batch top-1 오답의 co-match 육안 판정용 몽타주.

가설(§1 미검증): 티처 top-1 오답(31~41%)의 상당수는 '정당한 co-match'
 — 티처가 고른 off-diagonal 캡션 j*가 query 이미지 i를 실제로 기술하는 경우.
이걸 눈으로 판정하려고 (이미지 i, GT 캡션 i, 티처픽 캡션 j*, 이미지 j*)를 렌더한다.

- clean 뷰(transform_test)로 forward — aug crop 노이즈를 배제하고 '의미적' 랭킹만 본다.
- i2t 행 기준(이미지가 query). 티처가 대각선 대신 j*를 1등으로 올린 행만 수집.
- 티처가 가장 '확신'한 오답(sim_j* − sim_i 큰 순)을 우선 — soft-KD 타깃에 가장 큰 영향.
- COCO/VG source 태그 병기 (VG 짧은 phrase는 co-match 흔함 예상, COCO는 진짜 혼동 여부 판정).
"""
import sys, os, argparse, textwrap
import numpy as np
import torch
import yaml
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WT = '/home/minwoo/Distillation_Project_itc_distill_only'
sys.path.insert(0, WT)
os.chdir(WT)

from torch.utils.data import DataLoader, Dataset
from data import create_dataset
from data.utils import pre_caption
from distillation.online_teacher import OnlineTeacher

DEV = 'cuda:0'
OUTDIR = '/tmp/claude-1014/-home-minwoo-Distillation-Project/748b4c45-6cf3-46aa-8062-bb87cb9ae0e5/scratchpad'


class IndexedDS(Dataset):
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        img, cap = self.ds[i]
        return img, cap, i


def raw_image(ds, idx):
    """전송 전 PIL 이미지 (VG는 region crop 적용, transform은 미적용) — 디스플레이용."""
    ann = ds.annotation[idx]
    root = ds.dataset_root_dict[ann['dataset_source']]
    img = Image.open(os.path.join(root, ann['image'])).convert('RGB')
    if ann['dataset_source'] == 'vg':
        w, h = img.size
        x0 = min(max(ann['x'], 0), w); y0 = min(max(ann['y'], 0), h)
        x1 = min(max(ann['x'] + ann['width'], 0), w); y1 = min(max(ann['y'] + ann['height'], 0), h)
        if x1 > x0 and y1 > y0:
            img = img.crop((x0, y0, x1, y1))
    return img, ann['dataset_source'], pre_caption(ann['caption'], 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-batches', type=int, default=40)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--per-fig', type=int, default=8)
    args = ap.parse_args()

    cfg = '/home/minwoo/Distillation_Project/output/pt_smallreg_minilm_itc_distill_tau0.022_droptau/config.yaml'
    config = yaml.safe_load(open(cfg))
    config['pretrain_train_aug'] = False   # clean 뷰

    ds = create_dataset('pretrain', config)
    g = torch.Generator().manual_seed(args.seed)
    dl = DataLoader(IndexedDS(ds), batch_size=40, shuffle=True, num_workers=8,
                    drop_last=True, generator=g)
    teacher = OnlineTeacher(checkpoint=config['teacher']['checkpoint'], image_size=config['image_size'],
                            vit='large', bert='base', queue_size=config['queue_size'], keep=('itc',)).to(DEV)

    errors = []   # (margin, idx_i, idx_j, sim_i, sim_j)
    for bi, (image, caption, idx) in enumerate(dl):
        if bi >= args.n_batches:
            break
        img_f, txt_f = teacher.itc_feats(image.to(DEV), list(caption))
        sims = (img_f.float() @ txt_f.float().t()).cpu().numpy()   # [B,B] i2t
        idx = idx.numpy()
        B = sims.shape[0]
        for i in range(B):
            row = sims[i]
            jstar = int(row.argmax())
            if jstar != i:   # top-1 오답
                errors.append((float(row[jstar] - row[i]), int(idx[i]), int(idx[jstar]),
                               float(row[i]), float(row[jstar])))
    print(f"수집된 i2t top-1 오답: {len(errors)}건 / {args.n_batches*40}행")

    # source 분해 (오답이 VG에 몰리는가 COCO에도 있는가)
    from collections import Counter
    src_i = Counter(ds.annotation[e[1]]['dataset_source'] for e in errors)
    print(f"오답 query 이미지 source 분포: {dict(src_i)}")

    errors.sort(key=lambda e: -e[0])   # 티처가 가장 확신한 오답 우선
    # 상위 확신 오답 + 무작위 오답 섞어서 대표성
    rng = np.random.RandomState(0)
    top = errors[:args.per_fig]
    rest_idx = rng.choice(len(errors), size=min(args.per_fig, len(errors)), replace=False)
    rand = [errors[k] for k in rest_idx]

    for tag, cases in (('top_confident', top), ('random', rand)):
        render(ds, cases, f'{OUTDIR}/comatch_{tag}.png', tag)


def render(ds, cases, path, tag):
    n = len(cases)
    fig, axes = plt.subplots(n, 2, figsize=(13, 2.6 * n))
    if n == 1:
        axes = axes[None, :]
    for r, (margin, ii, jj, si, sj) in enumerate(cases):
        img_i, src_i, cap_i = raw_image(ds, ii)
        img_j, src_j, cap_j = raw_image(ds, jj)
        axes[r, 0].imshow(img_i); axes[r, 0].axis('off')
        axes[r, 0].set_title(f'[{r}] query IMG_i ({src_i})  sim_correct={si:.3f}', fontsize=9)
        axes[r, 1].imshow(img_j); axes[r, 1].axis('off')
        axes[r, 1].set_title(f'IMG_j* (티처픽 캡션 출처, {src_j})  sim_picked={sj:.3f}  Δ={margin:.3f}', fontsize=9)
        txt = (f"GT(정답, img_i용):  {textwrap.fill(cap_i, 70)}\n"
               f"티처 1등 픽(img_j*의 캡션): {textwrap.fill(cap_j, 70)}\n"
               f"→ 판정: 이 픽 캡션이 왼쪽 img_i도 맞게 기술하나?")
        axes[r, 0].text(0, -0.28, txt, transform=axes[r, 0].transAxes, fontsize=8.5,
                        va='top', family='monospace')
    plt.suptitle(f'Teacher i2t top-1 오답 co-match 판정 — {tag}', fontsize=12, y=0.997)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(path, dpi=95, bbox_inches='tight')
    plt.close()
    print(f"saved {path}")


if __name__ == '__main__':
    main()
