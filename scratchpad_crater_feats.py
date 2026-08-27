"""Characterize the holdhalf txt_r1 crater in FEATURE space (fp32, bf16 already ruled out).
For ckpt 13(pre)/14(crater)/15(recover): measure r1 + collapse metrics on student
retrieval features. Answers "what does 'mangled' mean" — text collapse vs image vs
cross-modal ranking reshuffle. Run: conda run -n kd_r4 python scratchpad_crater_feats.py
"""
import sys, time, torch, numpy as np, torch.nn.functional as F
sys.argv = ['x']
from models.blip_pretrain import blip_pretrain
from data.eval_validation_retrieval import build_coco_karpathy_retrieval_val_loader
import eval_validation_tool

BASE = '/home/minwoo/Distillation_Project/output/pt_ttm_queue_lm_holdhalf'
CKPTS = [12, 13, 19]  # ckpt_N = TB ep(N+1): 12=pre-crater(ep13~49), 13=crater(ep14~42.6), 19=recovered peak(ep20~52.6)
cfg = {
  'val_retrieval_split':'val',
  'val_retrieval_ann_root':'/home/minwoo/Distillation_Project/datasets/vision/coco/annotations/',
  'val_retrieval_image_root':'/home/minwoo/Distillation_Project/datasets/vision/coco/images/',
  'image_root_coco':'/home/minwoo/Distillation_Project/datasets/vision/coco/images/',
  'val_retrieval_batch_size':64, 'batch_size':64, 'image_size':224, 'k_test':128,
}
device = torch.device('cpu')
print('threads', torch.get_num_threads())
loader = build_coco_karpathy_retrieval_val_loader(cfg)
ds = loader.dataset
print(f'val: {len(ds.image)} images, {len(ds.text)} texts')

@torch.no_grad()
def extract(model):
    txt=[]
    for i in range(0, len(ds.text), 256):
        t_in = model.tokenizer(ds.text[i:i+256], padding='max_length', truncation=True, max_length=35, return_tensors='pt').to(device)
        o = model.text_encoder(t_in.input_ids, attention_mask=t_in.attention_mask, mode='text')
        te = F.normalize(model.text_proj(o.last_hidden_state[:,0,:]), dim=-1)
        txt.append(te.float())
    img=[]
    for image, _ in loader:
        ie = model.visual_encoder(image.to(device))
        ie = F.normalize(model.vision_proj(ie[:,0,:]), dim=-1)
        img.append(ie.float())
    return torch.cat(img,0), torch.cat(txt,0)

def eff_rank(X):
    # participation ratio of covariance eigenvalues = (Σλ)²/Σλ²   (max = D)
    C = (X.t() @ X) / X.shape[0]
    lam = torch.linalg.eigvalsh(C).clamp(min=0)
    return (lam.sum()**2 / (lam**2).sum()).item()

def mean_cos(X):
    # exact mean pairwise cosine for L2-normalized X: (||Σx||²-N)/(N(N-1))
    N = X.shape[0]; s = X.sum(0)
    return ((s@s).item() - N) / (N*(N-1))

def align_unif(img_e, txt_e):
    # alignment: mean ||img - matched txt||² over positive pairs; uniformity on sampled txt
    pos=[]
    for im, tlist in ds.img2txt.items():
        for t in tlist:
            pos.append((int(im), int(t)))
    pos = np.array(pos)
    a = (img_e[pos[:,0]] - txt_e[pos[:,1]]).pow(2).sum(1).mean().item()
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, txt_e.shape[0], (8000,2), generator=g)
    sq = (txt_e[idx[:,0]] - txt_e[idx[:,1]]).pow(2).sum(1)
    u = torch.log(torch.exp(-2*sq).mean()).item()
    return a, u

rows=[]
for c in CKPTS:
    ck = f'{BASE}/checkpoint_{c}.pth'
    print(f'\n=== ckpt_{c} ==='); t=time.time()
    model = blip_pretrain(image_size=224, vit='small_reg', my_bert_size='minilm', init_backbone_weights=False)
    sd = torch.load(ck, map_location='cpu', weights_only=False)['model']
    model.load_state_dict(sd, strict=False); model.eval().to(device)
    ie, te = extract(model)
    s = (ie @ te.t()).numpy()
    m = eval_validation_tool.itm_eval(s, s.T, ds.txt2img, ds.img2txt)
    a, u = align_unif(ie, te)
    rows.append((c, m['txt_r1'], m['img_r1'], m['r_mean'],
                 eff_rank(te), eff_rank(ie), mean_cos(te), mean_cos(ie), a, u))
    print(f'  done {time.time()-t:.0f}s  txt_r1={m["txt_r1"]:.2f} img_r1={m["img_r1"]:.2f}')

print('\n############ CRATER FEATURE CHARACTERIZATION (fp32) ############')
h = ['ckpt','txt_r1','img_r1','r_mean','txt_effR','img_effR','txt_mcos','img_mcos','align','unif']
print(' '.join(f'{x:>9}' for x in h))
for r in rows:
    print(f'{r[0]:>9} '+' '.join(f'{v:>9.3f}' for v in r[1:]))
print('# txt_effR/img_effR: effective dim (max 256; ↓=collapse). mcos: mean pairwise cos (↑=clustered).')
print('# align: ||img-txt_pos||² (↓=better). unif: log E exp(-2d²) (↓=more uniform/spread).')
