# custom code
# evaluate caption / retrieval validation metrics
# usually for pretraining

# two options
# 1. caption metric from val set -> coco
# 2. retrival metric from val set -> coco
# all validation functions from coco dataset

import contextlib
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import time
import datetime
import utils


def _retrieval_autocast_context(device, config):
    """val_loss_runner와 동일한 게이팅: val_retrieval_amp(기본 true) and device.type=='cuda'일 때만
    bf16 autocast, 그 외엔 nullcontext."""
    amp_enabled = config.get('val_retrieval_amp', True) and device.type == 'cuda'
    if amp_enabled:
        return torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()

# =================================================================================
# 1. Captioning 평가 모듈
# =================================================================================
@torch.no_grad()
def evaluate_caption(model, data_loader, device, config):
    print("\nCaption generation 평가 시작...")
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 50

    result = []
    # 주의: pretrain 로더와 달리 caption 로더는 image, image_id를 반환한다고 가정
    for image, image_id in metric_logger.log_every(data_loader, print_freq, header): 
        image = image.to(device)       
        
        captions = model.generate(image, sample=False, 
                                  num_beams=config.get('num_beams', 3), 
                                  max_length=config.get('max_length', 20), 
                                  min_length=config.get('min_length', 5))
        
        for caption, img_id in zip(captions, image_id):
            result.append({"image_id": img_id.item(), "caption": caption})
  
    return result

# =================================================================================
# 2. Retrieval 평가 및 지표(Recall) 계산 모듈
# =================================================================================
def _encode_retrieval_features(model, data_loader, device, need_patch_feats):
    """Shared by both retrieval tiers: extracts text/image embeddings and the
    full cosine-similarity matrix. Image patch features (needed only for ITM
    cross-attention rerank) are skipped entirely when need_patch_feats=False,
    which is what makes the ITC-only tier cheap."""
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []

    # Text 특징 추출
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')
        text_embed = model.text_proj(text_output.last_hidden_state[:,0,:])
        text_embed = F.normalize(text_embed, dim=-1)
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    text_ids[:,0] = model.tokenizer.enc_token_id

    image_feats = [] if need_patch_feats else None
    image_embeds = []

    # Image 특징 추출
    for image, img_id in data_loader:
        image = image.to(device)
        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat[:,0,:])
        image_embed = F.normalize(image_embed, dim=-1)

        if need_patch_feats:
            image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)

    if need_patch_feats:
        image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    # 코사인 유사도 행렬 계산
    sims_matrix = image_embeds @ text_embeds.t()

    return sims_matrix, image_feats, text_ids, text_atts


@torch.no_grad()
def evaluate_retrieval_itc(model, data_loader, device, config):
    """ITC-only tier: cosine-similarity ranking, no ITM head calls. Cheap
    enough to run frequently during pretraining (~10-30s for full COCO val)."""
    print('\nComputing features for ITC-only Retrieval evaluation...')
    start_time = time.time()

    with _retrieval_autocast_context(device, config):
        sims_matrix, _, _, _ = _encode_retrieval_features(
            model, data_loader, device, need_patch_feats=False)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('ITC-only Retrieval Evaluation time {}'.format(total_time_str))

    score_matrix_i2t = sims_matrix
    score_matrix_t2i = sims_matrix.t()

    return score_matrix_i2t.float().cpu().numpy(), score_matrix_t2i.float().cpu().numpy()


@torch.no_grad()
def evaluate_retrieval_itm(model, data_loader, device, config):
    """ITM-rerank tier: ITC top-k candidates re-scored with the cross-attention
    ITM head. This is the BLIP-paper-style R@1/5/10 definition, but much more
    expensive than evaluate_retrieval_itc (~2-5min for full COCO val)."""
    print('\nComputing features for Retrieval evaluation...')
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Retrieval Evaluation:'
    start_time = time.time()

    with _retrieval_autocast_context(device, config):
        sims_matrix, image_feats, text_ids, text_atts = _encode_retrieval_features(
            model, data_loader, device, need_patch_feats=True)

    texts = data_loader.dataset.text
    score_matrix_i2t = torch.full((len(data_loader.dataset.image), len(texts)), -100.0).to(device)

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    # I2T 평가
    with _retrieval_autocast_context(device, config):
        for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
            topk_sim, topk_idx = sims.topk(k=config.get('k_test', 128), dim=0)
            encoder_output = image_feats[start+i].repeat(config.get('k_test', 128), 1, 1).to(device)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
            output = model.text_encoder(text_ids[topk_idx],
                                        attention_mask = text_atts[topk_idx],
                                        encoder_hidden_states = encoder_output,
                                        encoder_attention_mask = encoder_att,
                                        return_dict = True)
            score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
            score_matrix_i2t[start+i, topk_idx] = (score + topk_sim).to(score_matrix_i2t.dtype)

    sims_matrix_t = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)

    step = sims_matrix_t.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix_t.size(0), start + step)

    # T2I 평가
    with _retrieval_autocast_context(device, config):
        for i, sims in enumerate(metric_logger.log_every(sims_matrix_t[start:end], 50, header)):
            topk_sim, topk_idx = sims.topk(k=config.get('k_test', 128), dim=0)
            encoder_output = image_feats[topk_idx.cpu()].to(device)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
            output = model.text_encoder(text_ids[start+i].repeat(config.get('k_test', 128), 1),
                                        attention_mask = text_atts[start+i].repeat(config.get('k_test', 128), 1),
                                        encoder_hidden_states = encoder_output,
                                        encoder_attention_mask = encoder_att,
                                        return_dict = True)
            score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
            score_matrix_t2i[start+i, topk_idx] = (score + topk_sim).to(score_matrix_t2i.dtype)

    if utils.is_dist_avail_and_initialized():
        dist.barrier()
        dist.all_reduce(score_matrix_i2t, op=dist.ReduceOp.SUM)
        dist.all_reduce(score_matrix_t2i, op=dist.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Retrieval Evaluation time {}'.format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

# itm은 뭐지? 그냥 위의 저거에서 고르는건가? 왜 이름을 itm eval이라 지었나
@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    ranks = np.zeros(scores_t2i.shape[0])
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    return {'txt_r1': tr1, 'txt_r5': tr5, 'txt_r10': tr10, 'txt_r_mean': tr_mean,
            'img_r1': ir1, 'img_r5': ir5, 'img_r10': ir10, 'img_r_mean': ir_mean,
            'r_mean': r_mean}