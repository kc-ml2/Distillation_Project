"""Phase 0-B: L_kd(tau) vs L_ita 그래디언트 크기 비율 + 방향 충돌(cosine) 실측.

- exp7 baseline 학생 ckpt(ep00/04/10/19)에 고정 배치 8개
- 같은 forward 그래프에서 L_ita와 L_kd(tau)를 각각 backward
- 공유 파라미터(visual_encoder, vision_proj, text_encoder, text_proj) grad로
  ratio = |g_kd|/|g_ita|, cos(g_kd, g_ita) 산출
- d L_ita / d logit_scale 부호도 기록 (scale 상승 압력 확인)
"""
import sys, os
import numpy as np
import torch
import torch.nn.functional as F
import yaml

WT = '/home/minwoo/Distillation_Project_itc_distill_only'
sys.path.insert(0, WT)
os.chdir(WT)

from torch.utils.data import DataLoader
from data import create_dataset
from distillation.online_teacher import OnlineTeacher
from models.blip_pretrain import blip_pretrain, LOGIT_SCALE_MIN, LOGIT_SCALE_MAX


def itc_distill_loss(image_feat_s, text_feat_s, ti, tt, temp):
    """drop-τ² 버전 고정 (워크트리 losses.py는 keep-τ² arm 상태라 import 안 함)."""
    s_s = (image_feat_s @ text_feat_s.t()) / temp
    s_t = (ti @ tt.t()) / temp
    l_i2t = F.kl_div(F.log_softmax(s_s, dim=1), F.softmax(s_t, dim=1), reduction="batchmean")
    l_t2i = F.kl_div(F.log_softmax(s_s.t(), dim=1), F.softmax(s_t.t(), dim=1), reduction="batchmean")
    return 0.5 * (l_i2t + l_t2i)

DEV = 'cuda:0'
TAUS = (0.022, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3)
CKPTS = ['checkpoint_00', 'checkpoint_04', 'checkpoint_10', 'checkpoint_19']
BASE_RUN = '/home/minwoo/Distillation_Project/output/pt_smallreg_minilm_baseline'
CFG = '/home/minwoo/Distillation_Project/output/pt_smallreg_minilm_itc_distill_tau0.022_droptau/config.yaml'
N_BATCH, ALPHA = 8, 0.4


def ita_loss_and_feats(model, image, caption):
    """branch forward의 ITC 경로 재현 (queue/momentum 갱신 없음)."""
    safe_scale = model.logit_scale.clamp(LOGIT_SCALE_MIN, LOGIT_SCALE_MAX).exp()
    image_embeds = model.visual_encoder(image)
    image_feat = F.normalize(model.vision_proj(image_embeds[:, 0, :]), dim=-1)
    text = model.tokenizer(caption, padding='max_length', truncation=True,
                           max_length=30, return_tensors='pt').to(DEV)
    text_output = model.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                     return_dict=True, mode='text')
    text_feat = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

    with torch.no_grad():
        image_embeds_m = model.visual_encoder_m(image)
        image_feat_m = F.normalize(model.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
        image_feat_all = torch.cat([image_feat_m.t(), model.image_queue.clone().detach()], dim=1)
        text_output_m = model.text_encoder_m(text.input_ids, attention_mask=text.attention_mask,
                                             return_dict=True, mode='text')
        text_feat_m = F.normalize(model.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
        text_feat_all = torch.cat([text_feat_m.t(), model.text_queue.clone().detach()], dim=1)
        sim_i2t_m = image_feat_m @ text_feat_all * safe_scale
        sim_t2i_m = text_feat_m @ image_feat_all * safe_scale
        sim_targets = torch.zeros_like(sim_i2t_m)
        sim_targets.fill_diagonal_(1)
        t_i2t = ALPHA * F.softmax(sim_i2t_m, dim=1) + (1 - ALPHA) * sim_targets
        t_t2i = ALPHA * F.softmax(sim_t2i_m, dim=1) + (1 - ALPHA) * sim_targets

    sim_i2t = image_feat @ text_feat_all * safe_scale
    sim_t2i = text_feat @ image_feat_all * safe_scale
    loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * t_i2t, dim=1).mean()
    loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * t_t2i, dim=1).mean()
    return (loss_i2t + loss_t2i) / 2, image_feat, text_feat


def shared_params(model):
    ps = []
    for m in (model.visual_encoder, model.vision_proj, model.text_encoder, model.text_proj):
        ps += [p for p in m.parameters() if p.requires_grad]
    return ps


def grab(ps):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().flatten()
                      for p in ps])


def main():
    config = yaml.safe_load(open(CFG))
    ds = create_dataset('pretrain', config)
    g = torch.Generator().manual_seed(123)
    dl = DataLoader(ds, batch_size=config['batch_size'], shuffle=True, num_workers=4,
                    drop_last=True, generator=g)
    batches = []
    for image, caption in dl:
        batches.append((image, list(caption)))
        if len(batches) >= N_BATCH:
            break
    del dl

    teacher = OnlineTeacher(checkpoint=config['teacher']['checkpoint'],
                            image_size=config['image_size'], vit='large', bert='base',
                            queue_size=config['queue_size'], keep=('itc',)).to(DEV)
    t_feats = []
    for image, caption in batches:
        ti, tt = teacher.itc_feats(image.to(DEV), caption)
        t_feats.append((ti.float(), tt.float()))

    model = blip_pretrain(image_size=config['image_size'], vit=config['vit'],
                          vit_grad_ckpt=config['vit_grad_ckpt'],
                          vit_ckpt_layer=config['vit_ckpt_layer'],
                          queue_size=config['queue_size'],
                          my_bert_size=config['my_bert_size'])

    results = {}
    for name in CKPTS:
        ck = torch.load(f'{BASE_RUN}/{name}.pth', map_location='cpu', weights_only=False)
        missing, unexpected = model.load_state_dict(ck['model'], strict=False)
        assert not missing, f'missing keys: {missing[:5]}'
        model.to(DEV).train()
        ps = shared_params(model)
        scale = model.logit_scale.detach().exp().item()

        rows = {tau: [] for tau in TAUS}
        ita_norms, ls_grads, ita_vals = [], [], []
        torch.manual_seed(7)  # dropout 고정(배치 간 비교 안정화)
        for (image, caption), (ti, tt) in zip(batches, t_feats):
            image = image.to(DEV)
            loss_ita, img_f, txt_f = ita_loss_and_feats(model, image, caption)

            model.zero_grad(set_to_none=True)
            loss_ita.backward(retain_graph=True)
            g_ita = grab(ps)
            ls_grads.append(float(model.logit_scale.grad.detach()))
            ita_norms.append(float(g_ita.norm()))
            ita_vals.append(float(loss_ita))

            for tau in TAUS:
                loss_kd = itc_distill_loss(img_f, txt_f, ti, tt, tau)
                model.zero_grad(set_to_none=True)
                loss_kd.backward(retain_graph=True)
                g_kd = grab(ps)
                rows[tau].append((float(g_kd.norm()),
                                  float(F.cosine_similarity(g_kd, g_ita, dim=0)),
                                  float(loss_kd)))
            model.zero_grad(set_to_none=True)
            del img_f, txt_f, loss_ita
        model.cpu()

        print(f"\n===== {name} (logit_scale={scale:.1f}) =====")
        print(f"|g_ita| = {np.mean(ita_norms):.4f} ± {np.std(ita_norms):.4f}   "
              f"L_ita = {np.mean(ita_vals):.3f}   dL_ita/d(logit_scale_raw) = {np.mean(ls_grads):+.5f}")
        print(f"{'tau':>6} {'|g_kd|':>10} {'ratio':>8} {'cos(g_kd,g_ita)':>16} {'L_kd':>8}")
        for tau in TAUS:
            arr = np.array(rows[tau])
            print(f"{tau:>6} {arr[:,0].mean():>10.4f} {arr[:,0].mean()/np.mean(ita_norms):>8.3f} "
                  f"{arr[:,1].mean():>10.3f}±{arr[:,1].std():.3f} {arr[:,2].mean():>8.4f}")
        results[name] = {str(tau): rows[tau] for tau in TAUS}

    np.save('/tmp/claude-1014/-home-minwoo-Distillation-Project/748b4c45-6cf3-46aa-8062-bb87cb9ae0e5/scratchpad/grad_probe_results.npy',
            results, allow_pickle=True)


if __name__ == '__main__':
    main()
