"""Decisive test: is the holdhalf ep14 txt_r1 crater a bf16-precision artifact?
Load cratered checkpoint on CPU, extract student retrieval features in fp32 AND
bf16, rank in fp32 AND bf16 -> 2x2 R@1. If fp32 recovers txt_r1 (~50 vs bf16 ~42),
bf16 is confirmed. Run: conda run -n kd_r4 python scratchpad_bf16_probe.py
"""
import sys, time, torch, numpy as np, torch.nn.functional as F
sys.argv = ['x']  # guard argparse-y imports
from models.blip_pretrain import blip_pretrain
from data.eval_validation_retrieval import build_coco_karpathy_retrieval_val_loader
import eval_validation_tool

CKPT = '/home/minwoo/Distillation_Project/output/pt_ttm_queue_lm_holdhalf/checkpoint_14.pth'
cfg = {
  'val_retrieval_split':'val',
  'val_retrieval_ann_root':'/home/minwoo/Distillation_Project/datasets/vision/coco/annotations/',
  'val_retrieval_image_root':'/home/minwoo/Distillation_Project/datasets/vision/coco/images/',
  'image_root_coco':'/home/minwoo/Distillation_Project/datasets/vision/coco/images/',
  'val_retrieval_batch_size':64, 'batch_size':64, 'image_size':224, 'k_test':128,
}
device = torch.device('cpu')
torch.set_num_threads(torch.get_num_threads())
print('threads', torch.get_num_threads())

print('build model on CPU + load ckpt_14 ...'); t=time.time()
model = blip_pretrain(image_size=224, vit='small_reg', my_bert_size='minilm', init_backbone_weights=False)
sd = torch.load(CKPT, map_location='cpu', weights_only=False)['model']
msg = model.load_state_dict(sd, strict=False)
print('  missing', len(msg.missing_keys), 'unexpected', len(msg.unexpected_keys), f'({time.time()-t:.0f}s)')
model.eval().to(device)
loader = build_coco_karpathy_retrieval_val_loader(cfg)
ds = loader.dataset
print(f'val: {len(ds.image)} images, {len(ds.text)} texts')

@torch.no_grad()
def feats(precision):
    ctx = torch.autocast(device_type='cpu', dtype=torch.bfloat16) if precision=='bf16' else torch.autocast('cpu', enabled=False)
    txt=[]
    with ctx:
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

def r1(img_e, txt_e, sim_prec):
    if sim_prec=='bf16':
        s = (img_e.bfloat16() @ txt_e.bfloat16().t()).float()
    else:
        s = img_e @ txt_e.t()
    m = eval_validation_tool.itm_eval(s.numpy(), s.t().numpy(), ds.txt2img, ds.img2txt)
    return m['txt_r1'], m['img_r1'], m['r_mean']

print('\n=== extracting features (fp32) ...'); t=time.time()
i32,t32 = feats('fp32'); print(f'  done {time.time()-t:.0f}s')
print('=== extracting features (bf16 autocast) ...'); t=time.time()
i16,t16 = feats('bf16'); print(f'  done {time.time()-t:.0f}s')

print('\n############ RESULT 2x2 (ckpt_14, TB bf16-logged txt_r1~42) ############')
print(f"{'feat':>6} {'sim':>6} {'txt_r1':>7} {'img_r1':>7} {'r_mean':>7}")
for fe,(ie,te) in [('fp32',(i32,t32)),('bf16',(i16,t16))]:
    for se in ['fp32','bf16']:
        a,b,c = r1(ie,te,se)
        print(f"{fe:>6} {se:>6} {a:>7.2f} {b:>7.2f} {c:>7.2f}")
print('# fp32/fp32 = clean ; bf16/bf16 = training-eval condition (should ~match TB ~42)')
