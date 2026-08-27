"""Phase 0-C: 학생 임베딩 기하 비교 (baseline ep19 vs droptau ep19, clean val 뷰).

가설 검증: KD(τ=0.022, λ=1)가 학생 cosine 마진을 티처의 압축된 스케일로
끌어내렸고(작은 마진), ITC는 logit_scale 인플레로 보상했다 → retrieval 손상.
"""
import sys, os
import numpy as np
import torch
import yaml

WT = '/home/minwoo/Distillation_Project_itc_distill_only'
sys.path.insert(0, WT)
os.chdir(WT)

from torch.utils.data import DataLoader
from data import create_dataset
from models.blip_pretrain import blip_pretrain
import torch.nn.functional as F

DEV = 'cuda:0'
CFG = '/home/minwoo/Distillation_Project/output/pt_smallreg_minilm_itc_distill_tau0.022_droptau/config.yaml'
MODELS = {
    'baseline_ep19': '/home/minwoo/Distillation_Project/output/pt_smallreg_minilm_baseline/checkpoint_19.pth',
    'droptau_ep19': '/home/minwoo/Distillation_Project/output/pt_smallreg_minilm_itc_distill_tau0.022_droptau/checkpoint_19.pth',
}
N_BATCH = 50


@torch.no_grad()
def feats(model, image, caption):
    image_embeds = model.visual_encoder(image)
    img_f = F.normalize(model.vision_proj(image_embeds[:, 0, :]), dim=-1)
    text = model.tokenizer(caption, padding='max_length', truncation=True,
                           max_length=30, return_tensors='pt').to(DEV)
    to = model.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                            return_dict=True, mode='text')
    txt_f = F.normalize(model.text_proj(to.last_hidden_state[:, 0, :]), dim=-1)
    return img_f, txt_f


def main():
    import json
    config = yaml.safe_load(open(CFG))
    # karpathy val은 caption이 리스트(5개) → val_loss_caption_mode:first와 동일하게 첫 캡션만 플랫화
    ann = json.load(open(config['val_loss_file']))
    flat = [{'image': a['image'], 'caption': a['caption'][0] if isinstance(a['caption'], list) else a['caption']}
            for a in ann]
    flat_path = '/tmp/claude-1014/-home-minwoo-Distillation-Project/748b4c45-6cf3-46aa-8062-bb87cb9ae0e5/scratchpad/coco_karpathy_val_flat.json'
    json.dump(flat, open(flat_path, 'w'))
    config['train_file'] = [flat_path]                # 'coco' 포함 파일명 → source 태깅 OK
    config['pretrain_train_aug'] = False              # clean 뷰
    ds = create_dataset('pretrain', config)
    print(f"val pairs: {len(ds)}")
    g = torch.Generator().manual_seed(42)
    dl = DataLoader(ds, batch_size=40, shuffle=True, num_workers=8, drop_last=True, generator=g)
    batches = []
    for image, caption in dl:
        batches.append((image, list(caption)))
        if len(batches) >= N_BATCH:
            break

    model = blip_pretrain(image_size=config['image_size'], vit=config['vit'],
                          vit_grad_ckpt=False, vit_ckpt_layer=0,
                          queue_size=config['queue_size'], my_bert_size=config['my_bert_size'])
    for name, path in MODELS.items():
        ck = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'], strict=False)
        model.to(DEV).eval()
        scale = model.logit_scale.exp().item()

        sims_all = []
        for image, caption in batches:
            i_f, t_f = feats(model, image.to(DEV), caption)
            sims_all.append((i_f.float() @ t_f.float().t()).cpu().numpy())
        S = np.stack(sims_all)
        model.cpu()

        N, B, _ = S.shape
        eye = np.eye(B, dtype=bool)
        diag = S[:, eye].reshape(N, B)
        offs = S[:, ~eye]
        print(f"\n===== {name} (logit_scale={scale:.1f}) =====")
        print(f"pos: mean={diag.mean():.4f} std={diag.std():.4f}   "
              f"neg: mean={offs.mean():.4f} std={offs.std():.4f}")
        for nm, sim in (('i2t', S), ('t2i', np.transpose(S, (0, 2, 1)))):
            d = sim[:, eye].reshape(N, B)
            mo = np.where(~eye[None], sim, -np.inf).max(axis=2)
            mg = d - mo
            print(f"[{nm}] margin: mean={mg.mean():.4f} p5={np.percentile(mg,5):.4f} "
                  f"p50={np.percentile(mg,50):.4f} | top1 오답률={(mg<0).mean()*100:.2f}%")
        print(f"raw scale margin*logit_scale (i2t mean): {((diag - np.where(~eye[None],S,-np.inf).max(axis=2)).mean()*scale):.3f}")


if __name__ == '__main__':
    main()
