"""co-match 후속: source별(COCO/VG) i2t top-1 오답률 + VG region 크기별 오답률.
가설: 티처는 COCO(전체 이미지+완결 캡션)엔 신뢰 가능, VG(짧은 phrase+작은 region)엔 노이즈."""
import sys, os
import numpy as np
import torch, yaml

WT = '/home/minwoo/Distillation_Project_itc_distill_only'
sys.path.insert(0, WT); os.chdir(WT)
from torch.utils.data import DataLoader, Dataset
from data import create_dataset
from distillation.online_teacher import OnlineTeacher

DEV = 'cuda:0'
class IndexedDS(Dataset):
    def __init__(s, ds): s.ds = ds
    def __len__(s): return len(s.ds)
    def __getitem__(s, i):
        img, cap = s.ds[i]; return img, cap, i

cfg = '/home/minwoo/Distillation_Project/output/pt_smallreg_minilm_itc_distill_tau0.022_droptau/config.yaml'
config = yaml.safe_load(open(cfg)); config['pretrain_train_aug'] = False
ds = create_dataset('pretrain', config)
g = torch.Generator().manual_seed(7)
dl = DataLoader(IndexedDS(ds), batch_size=40, shuffle=True, num_workers=8, drop_last=True, generator=g)
teacher = OnlineTeacher(checkpoint=config['teacher']['checkpoint'], image_size=config['image_size'],
                        vit='large', bert='base', queue_size=config['queue_size'], keep=('itc',)).to(DEV)

from collections import Counter
tot = Counter(); err = Counter()
# VG region area (px²) 버킷별
area_tot = Counter(); area_err = Counter()
def area_bucket(ann):
    if ann['dataset_source'] != 'vg': return None
    a = ann.get('width', 0) * ann.get('height', 0)
    for lo, name in [(0, '<32²'), (32**2, '32-64²'), (64**2, '64-128²'), (128**2, '128-256²'), (256**2, '≥256²')]:
        if a >= lo: b = name
    return b

for bi, (image, caption, idx) in enumerate(dl):
    if bi >= 60: break
    img_f, txt_f = teacher.itc_feats(image.to(DEV), list(caption))
    sims = (img_f.float() @ txt_f.float().t()).cpu().numpy()
    idx = idx.numpy(); B = sims.shape[0]
    for i in range(B):
        ann = ds.annotation[int(idx[i])]; src = ann['dataset_source']
        wrong = int(sims[i].argmax()) != i
        tot[src] += 1; err[src] += wrong
        b = area_bucket(ann)
        if b: area_tot[b] += 1; area_err[b] += wrong

print("\n=== source별 i2t top-1 오답률 ===")
for s in ('coco', 'vg'):
    if tot[s]: print(f"{s}: {err[s]}/{tot[s]} = {100*err[s]/tot[s]:.1f}%")
print("\n=== VG region 크기별 오답률 (작을수록 degenerate) ===")
for b in ['<32²', '32-64²', '64-128²', '128-256²', '≥256²']:
    if area_tot[b]: print(f"{b:>10}: {area_err[b]:>4}/{area_tot[b]:<4} = {100*area_err[b]/area_tot[b]:.1f}%")
