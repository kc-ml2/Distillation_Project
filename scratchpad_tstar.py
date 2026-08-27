"""Find the teacher temperature T* where teacher effcand == student effcand (484):
the entropy-match / crossover threshold. Fixes the earlier np.interp bug and prints a
fine scale sweep. Saves S_stu/S_tea so future sweeps are instant.
Run: conda run -n kd_r4 python scratchpad_tstar.py
"""
import sys, time, math, os, torch, numpy as np, torch.nn.functional as F
sys.argv = ['x']
from models.blip_pretrain import blip_pretrain
from data.eval_validation_retrieval import build_coco_karpathy_retrieval_val_loader
from distillation.online_teacher import OnlineTeacher

STU='/home/minwoo/Distillation_Project/output/pt_ttm_queue_lm_holdhalf/checkpoint_13.pth'
TEA='/home/minwoo/Distillation_Project/output/official_pretrain_checkpoint/model_large.pth'
CACHE=os.path.expanduser('~/.claude/jobs/d6026650/tmp/sim_cache.pt')
N_IMG=800
cfg={'val_retrieval_split':'val','val_retrieval_ann_root':'/home/minwoo/Distillation_Project/datasets/vision/coco/annotations/',
 'val_retrieval_image_root':'/home/minwoo/Distillation_Project/datasets/vision/coco/images/',
 'image_root_coco':'/home/minwoo/Distillation_Project/datasets/vision/coco/images/',
 'val_retrieval_batch_size':64,'batch_size':64,'image_size':224,'k_test':128}
dev=torch.device('cpu')

def effcand(S, scale):
    P=F.softmax(S*scale,dim=1); H=(-(P*torch.log(P.clamp_min(1e-30))).sum(1)).mean().item()
    return math.exp(H), H

if os.path.exists(CACHE):
    d=torch.load(CACHE); S_stu,S_tea,s_scale=d['S_stu'],d['S_tea'],d['s_scale']
    print(f'loaded cache; student safe_scale={s_scale:.2f}')
else:
    loader=build_coco_karpathy_retrieval_val_loader(cfg); ds=loader.dataset
    @torch.no_grad()
    def enc_text(m,proj,ml):
        o=[]
        for i in range(0,len(ds.text),256):
            t=m.tokenizer(ds.text[i:i+256],padding='max_length',truncation=True,max_length=ml,return_tensors='pt').to(dev)
            e=m.text_encoder(t.input_ids,attention_mask=t.attention_mask,mode='text')
            o.append(F.normalize(proj(e.last_hidden_state[:,0,:]),dim=-1).float())
        return torch.cat(o,0)
    @torch.no_grad()
    def enc_img(m,proj,n):
        o=[];g=0
        for im,_ in loader:
            e=m.visual_encoder(im.to(dev)); o.append(F.normalize(proj(e[:,0,:]),dim=-1).float()); g+=im.shape[0]
            if g>=n: break
        return torch.cat(o,0)[:n]
    print('student...'); t=time.time()
    sm=blip_pretrain(image_size=224,vit='small_reg',my_bert_size='minilm',init_backbone_weights=False)
    sm.load_state_dict(torch.load(STU,map_location='cpu',weights_only=False)['model'],strict=False); sm.eval()
    s_scale=math.exp(sm.logit_scale.item())
    S_stu=enc_img(sm,sm.vision_proj,N_IMG)@enc_text(sm,sm.text_proj,35).t(); print(f'  {time.time()-t:.0f}s')
    print('teacher...'); t=time.time()
    tea=OnlineTeacher(checkpoint=TEA,image_size=224,vit='large',bert='base',queue_size=57600,keep=('itc',)).to('cpu')
    S_tea=enc_img(tea.model,tea.model.vision_proj,N_IMG)@enc_text(tea.model,tea.model.text_proj,30).t(); print(f'  {time.time()-t:.0f}s')
    torch.save({'S_stu':S_stu,'S_tea':S_tea,'s_scale':s_scale},CACHE)

stu_ec,_=effcand(S_stu,s_scale)
print(f'\nSTUDENT effcand @ safe_scale {s_scale:.2f} = {stu_ec:.1f}  (this is the target to match)')
print(f'\n{"scale":>7} {"T=1/scale":>10} {"teacher effcand":>16}')
grid=np.array([20,25,30,33,35,37,40,42,45,50,56.75,63.89],float)
ecs=[]
for s in grid:
    ec,_=effcand(S_tea,s); ecs.append(ec)
    mark=' <-- ttm 0.05' if abs(s-20)<.1 else (' <-- student scale' if abs(s-56.75)<.1 else (' <-- teacher native 0.0157' if abs(s-63.89)<1 else ''))
    print(f'{s:>7.2f} {1/s:>10.4f} {ec:>16.1f}{mark}')
ecs=np.array(ecs)
# teacher effcand DEcreases as scale increases; find scale where teacher effcand == stu_ec (484)
# interp scale as fn of effcand -> need effcand ascending: reverse the (monotone) arrays
order=np.argsort(ecs)
s_star=float(np.interp(stu_ec, ecs[order], grid[order]))
print(f'\n>>> T*: teacher effcand == student {stu_ec:.0f} at scale s* = {s_star:.2f}  => ttm_temp T* = {1/s_star:.4f}')
print(f'    ttm 0.05 (scale20, effcand~11000) is ABOVE T* = softer than student = CRATER zone.')
print(f'    native 0.0157 (scale64, effcand~25) is BELOW T* = sharper than student = SAFE zone.')
