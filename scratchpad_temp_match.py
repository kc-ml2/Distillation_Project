"""How sharp is the TEACHER vs the STUDENT? -> pick the distillation temperature.
Measures i2t similarity-distribution entropy (=sharpness) + teacher target quality
(argmax-vs-GT agreement, positive mass, hubness) on the COCO val retrieval set, fp32.
Student = crater ckpt_13 (safe_scale=exp(logit_scale)=56.75). Teacher = frozen BLIP-large
(native temp 0.0157 -> scale 63.7; ttm uses temp 0.05 -> scale 20).
Answers: (1) is the teacher intrinsically sharper/softer than the student, and by how much
(scale s* that entropy-matches teacher to student); (2) at ttm 0.05 is the teacher target too
soft (spread to hubs) or is its top-1 just wrong -> which fix direction (sharpen vs distrust).
Run: conda run -n kd_r4 python scratchpad_temp_match.py
Caveat: candidate pool = 25010 val texts (~ vs 57760 train queue); relative teacher-vs-student
comparison on the SAME pool is exact; absolute T* shifts slightly with pool size.
"""
import sys, time, math, torch, numpy as np, torch.nn.functional as F
sys.argv = ['x']
from models.blip_pretrain import blip_pretrain
from data.eval_validation_retrieval import build_coco_karpathy_retrieval_val_loader
from distillation.online_teacher import OnlineTeacher

STU = '/home/minwoo/Distillation_Project/output/pt_ttm_queue_lm_holdhalf/checkpoint_13.pth'
TEA = '/home/minwoo/Distillation_Project/output/official_pretrain_checkpoint/model_large.pth'
N_IMG = 800  # subset of val images used as i2t query rows (all 25010 texts are candidates)
cfg = {
  'val_retrieval_split':'val',
  'val_retrieval_ann_root':'/home/minwoo/Distillation_Project/datasets/vision/coco/annotations/',
  'val_retrieval_image_root':'/home/minwoo/Distillation_Project/datasets/vision/coco/images/',
  'image_root_coco':'/home/minwoo/Distillation_Project/datasets/vision/coco/images/',
  'val_retrieval_batch_size':64, 'batch_size':64, 'image_size':224, 'k_test':128,
}
dev = torch.device('cpu')
loader = build_coco_karpathy_retrieval_val_loader(cfg); ds = loader.dataset
print(f'val: {len(ds.image)} imgs, {len(ds.text)} texts; using {N_IMG} query imgs')
# positives for each image row: text indices in ds.img2txt
pos = [set(int(t) for t in ds.img2txt[i]) for i in range(N_IMG)]

@torch.no_grad()
def enc_text(model, proj, max_len):
    out=[]
    for i in range(0, len(ds.text), 256):
        t = model.tokenizer(ds.text[i:i+256], padding='max_length', truncation=True,
                            max_length=max_len, return_tensors='pt').to(dev)
        o = model.text_encoder(t.input_ids, attention_mask=t.attention_mask, mode='text')
        out.append(F.normalize(proj(o.last_hidden_state[:,0,:]), dim=-1).float())
    return torch.cat(out,0)

@torch.no_grad()
def enc_img(model, proj, n):
    out=[]; got=0
    for image,_ in loader:
        ie = model.visual_encoder(image.to(dev))
        out.append(F.normalize(proj(ie[:,0,:]), dim=-1).float()); got+=image.shape[0]
        if got>=n: break
    return torch.cat(out,0)[:n]

print('== student ckpt_13 =='); t=time.time()
sm = blip_pretrain(image_size=224, vit='small_reg', my_bert_size='minilm', init_backbone_weights=False)
sm.load_state_dict(torch.load(STU,map_location='cpu',weights_only=False)['model'], strict=False); sm.eval()
s_scale = math.exp(sm.logit_scale.item()); print(f'  student safe_scale = {s_scale:.2f}')
s_img = enc_img(sm, sm.vision_proj, N_IMG); s_txt = enc_text(sm, sm.text_proj, 35)
print(f'  done {time.time()-t:.0f}s')

print('== teacher BLIP-large =='); t=time.time()
tea = OnlineTeacher(checkpoint=TEA, image_size=224, vit='large', bert='base', queue_size=57600, keep=('itc',)).to('cpu')
t_scale_native = tea.teacher_scale; print(f'  teacher native scale = {t_scale_native:.2f} (temp {tea.teacher_temp})')
t_img = enc_img(tea.model, tea.model.vision_proj, N_IMG); t_txt = enc_text(tea.model, tea.model.text_proj, 30)
print(f'  done {time.time()-t:.0f}s')

S_stu = s_img @ s_txt.t()   # [N_IMG, 25010] cosine
S_tea = t_img @ t_txt.t()

def stats(S, scale):
    P = F.softmax(S*scale, dim=1)
    logP = torch.log(P.clamp_min(1e-30))
    H = (-(P*logP).sum(1)).mean().item()          # mean row entropy (nats)
    posm = np.mean([P[r, list(pos[r])].sum().item() for r in range(N_IMG)])
    top1 = P.argmax(1).numpy()
    agree = np.mean([top1[r] in pos[r] for r in range(N_IMG)])
    # hubness: how concentrated are the top-1 texts across queries
    vals,cnts = np.unique(top1, return_counts=True)
    order = np.argsort(-cnts)
    hub10 = cnts[order[:10]].sum()/N_IMG          # share of queries whose top1 is a top-10 hub
    return H, math.exp(H), posm, agree*100, len(vals), hub10*100

def row(label, S, scale):
    H,eff,pm,ag,ndist,hub = stats(S,scale)
    print(f'{label:34} s={scale:6.2f} T={1/scale:6.4f}  H={H:5.2f} effcand={eff:8.1f} '
          f'posmass={pm:5.3f} top1_GT={ag:5.1f}% ndistinct={ndist:4d} hub10={hub:5.1f}%')
    return H

print('\n############ SHARPNESS + TEACHER-TARGET QUALITY (i2t, val 25010 cands) ############')
Hs = row('STUDENT @ safe_scale', S_stu, s_scale)
print('  -- teacher at various scales --')
row('TEACHER @ student-scale(56.75)', S_tea, s_scale)      # same scale -> intrinsic sharpness diff
row('TEACHER @ native(0.0157)',       S_tea, t_scale_native)
row('TEACHER @ ttm(0.05)',            S_tea, 20.0)
# find s* where teacher entropy == student entropy
grid = np.array([8,10,12,15,20,25,30,40,50,56.75,63.7,80,100,140,200], float)
Ht = np.array([stats(S_tea, s)[0] for s in grid])
# interpolate s* for Ht==Hs (Ht decreases as s grows)
s_star = float(np.interp(-Hs, -Ht[::-1], grid[::-1])) if Ht.min()<=Hs<=Ht.max() else float('nan')
print(f'  -- entropy-match: student H={Hs:.2f}; teacher matches at s* = {s_star:.2f} (T*={1/s_star:.4f}) --')
if not math.isnan(s_star): row('TEACHER @ s* (entropy-matched)', S_tea, s_star)
print('\n# INTERPRET: compare TEACHER@student-scale vs STUDENT (same s) -> who has bigger cosine gaps.')
print('# s* > 56.75 => teacher intrinsically SOFTER (needs higher scale to match); s* < 56.75 => SHARPER.')
print('# ttm(0.05,s=20): if posmass low & hub10 high => target spreads to hubs (fix=SHARPEN toward s*).')
print('#                 if top1_GT low => teacher top-1 often WRONG (fix=DISTRUST teacher / lower gamma, temp wont help).')
