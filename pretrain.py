'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import sys
# import ruamel.yaml as yaml
# from ruamel.yaml import YAML
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from data.combined_loader import CombinedLoader
import webdataset as wds # 웹데이터셋용으로 추가



from models.blip_pretrain import blip_pretrain
import utils
from utils import warmup_lr_schedule, step_lr_schedule
from data import create_dataset, create_sampler, create_loader # init에 있는 놈들임
#### 수정부분 시작: validation loss 모듈 import ####
from data import eval_validation_loss
#### 수정부분 끝 ####
#### 수정부분 시작: retrieval validation 모듈 import ####
from data import eval_validation_retrieval
#### 수정부분 끝 ####
#### 수정부분 시작: caption validation 모듈 import ####
from data import eval_validation_caption
#### 수정부분 끝 ####

#### tensorboard scalars ####
from torch.utils.tensorboard import SummaryWriter

#### 수정부분 시작: temp 런어웨이 ablation용 자동 종료 트리거 (실험 1~4 끝나면 제거) ####
# 이전 run(exp=baseline_lrlow_amp)에서 logit_scale(1/temp)이 epoch 4.3~6 사이에
# 38~58 수준의 정상 플래토에서 클램프 상한(1000)까지 폭주한 걸 확인함.
# 정상 범위보다 훨씬 위인 700을 10000 step 연속으로 넘기면 "사실상 붕괴 확정"으로 보고 종료.
COLLAPSE_LOGIT_SCALE_THRESHOLD = 700.0
COLLAPSE_SUSTAINED_STEPS = 10000

class CollapseDetected(Exception):
    pass
#### 수정부분 끝 ####

# 텐서보드 이름을 제작해주는 함수. 컨피그만 있으면 알아서 만들 것임
def make_tb_run_name(config):
    tb_option_dict = {
        # "experiment": "test_tensorboard",
        "exp": config.get("exp", "unnamed"),   # config의 exp 키로 실험마다 지정 (하드코딩 제거)
        "mode": "pretrain",
        "vit": config["vit"],
        "bert": config["my_bert_size"],
        "batch_size": f"bs{config['batch_size']}x{utils.get_world_size()}eff{config['batch_size'] * utils.get_world_size()}",
        "dataset": config["used_dataset"], # 주의! 야멜안바꾸면 잘못입력됨.
    }
    # distill 활성 시 run name에 kd 태그 (예: kd=lm_w1.0T2.0) — w/T는 고정 하이퍼파라미터라
    # 시계열 로깅 대신 이름+config.yaml 덤프로 기록
    distill_cfg = config.get("distill", {})
    kd_parts = [f"{k}_w{distill_cfg[k].get('weight', 1.0)}T{distill_cfg[k].get('temp')}"
                for k in ("itc", "lm") if distill_cfg.get(k, {}).get("enabled", False)]
    if kd_parts:
        tb_option_dict["kd"] = "+".join(kd_parts)
    return "__".join(f"{k}={v}" for k, v in tb_option_dict.items())

def train(model, data_loader, optimizer, epoch, device, config, writer=None, val_loss_runner=None, collapse_counter=None, model_without_ddp=None, retrieval_val_runner=None, caption_val_runner=None, online_teacher=None): # online_teacher 추가
    # train
    model.train()

    distill_itc = config.get('distill', {}).get('itc', {})
    itc_kd_enabled = distill_itc.get('enabled', False)
    itc_kd_weight = float(distill_itc.get('weight', 1.0))
    itc_kd_temp = float(distill_itc.get('temp', 0.05))

    distill_lm = config.get('distill', {}).get('lm', {})
    lm_kd_enabled = distill_lm.get('enabled', False)
    lm_kd_weight = float(distill_lm.get('weight', 1.0))
    lm_kd_temp = float(distill_lm.get('temp', 2.0))

    distill_ttm = config.get('distill', {}).get('itc_target_mix', {})
    ttm_enabled = distill_ttm.get('enabled', False)
    assert not (itc_kd_enabled and ttm_enabled), \
        "distill.itc(별도 KD)와 distill.itc_target_mix 동시 사용 금지"
    ttm_soft_weight = float(distill_ttm.get('soft_weight', 0.4))
    ttm_hold_steps = int(distill_ttm.get('hold_epochs', 2) * len(data_loader))
    ttm_decay_end_steps = int(distill_ttm.get('decay_end_epochs', 12) * len(data_loader))

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))    
    metric_logger.add_meter('loss_lm', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   

    if config['laion_path']:
        data_loader.dataset.reload_laion(epoch)
    
    data_loader.sampler.set_epoch(epoch)


    for i, (image, caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # 글로벌 스텝 추가
        global_step = epoch * len(data_loader) + i
        
        if epoch==0:
            warmup_lr_schedule(optimizer, i, config['warmup_steps'], config['warmup_lr'], config['init_lr'])
            
        optimizer.zero_grad()
        
        image = image.to(device,non_blocking=True)

        # online teacher: same augmented batch -> ITC features (bf16, no_grad). None when distill off.
        if (itc_kd_enabled or ttm_enabled) and online_teacher is not None:
            teacher_img_feat, teacher_text_feat = online_teacher.itc_feats(image, caption)
        else:
            teacher_img_feat = teacher_text_feat = None

        # online teacher: same augmented batch -> LM decoder logits (bf16, no_grad). None when distill off.
        if lm_kd_enabled and online_teacher is not None:
            teacher_lm_logits, teacher_lm_ids = online_teacher.lm_logits(image, caption)
        else:
            teacher_lm_logits = teacher_lm_ids = None

        # ramp up alpha in the first 2 epochs
        alpha = config['alpha']*min(1,(epoch*len(data_loader)+i)/(2*len(data_loader)))

        from distillation.target_mix import ttm_gamma
        gamma = ttm_gamma(global_step, ttm_hold_steps, ttm_decay_end_steps) if ttm_enabled else None

        # loss_ita, loss_itm, loss_lm = model(image, caption, alpha = alpha)
        # loss = loss_ita + loss_itm + loss_lm  
        # bp 16 mixed precision
        # if device == "cuda": # 이 부분 수정할 예정
        if device.type == "cuda":
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss_ita, loss_itm, loss_lm, loss_itc_kd, loss_lm_kd = model(
                    image, caption, alpha=alpha,
                    teacher_img_feat=teacher_img_feat, teacher_text_feat=teacher_text_feat,
                    distill_temp=itc_kd_temp,
                    teacher_lm_logits=teacher_lm_logits, teacher_lm_input_ids=teacher_lm_ids,
                    lm_distill_temp=lm_kd_temp,
                    gamma=gamma)
                loss = loss_ita + loss_itm + loss_lm
                if itc_kd_enabled and loss_itc_kd is not None:
                    loss = loss + itc_kd_weight * loss_itc_kd
                if lm_kd_enabled and loss_lm_kd is not None:
                    loss = loss + lm_kd_weight * loss_lm_kd
        else:
            loss_ita, loss_itm, loss_lm, loss_itc_kd, loss_lm_kd = model(
                image, caption, alpha=alpha,
                teacher_img_feat=teacher_img_feat, teacher_text_feat=teacher_text_feat,
                distill_temp=itc_kd_temp,
                teacher_lm_logits=teacher_lm_logits, teacher_lm_input_ids=teacher_lm_ids,
                lm_distill_temp=lm_kd_temp,
                gamma=gamma)
            loss = loss_ita + loss_itm + loss_lm
            if itc_kd_enabled and loss_itc_kd is not None:
                loss = loss + itc_kd_weight * loss_itc_kd
            if lm_kd_enabled and loss_lm_kd is not None:
                loss = loss + lm_kd_weight * loss_lm_kd

        loss.backward()
        optimizer.step()    

        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_lm=loss_lm.item())
        if loss_itc_kd is not None:
            metric_logger.update(loss_itc_kd=loss_itc_kd.item())
        if loss_lm_kd is not None:
            metric_logger.update(loss_lm_kd=loss_lm_kd.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # 로컬 기록 이후에 텐서보드 입력, config에 tb_log_interval을 적기
        # 주의: i, epoch은 모든 rank에서 동일하므로 이 게이트는 writer 유무와 무관하게 전체 rank가 동일하게 통과/스킵함
        if global_step % config["tb_train_log_interval"] == 0:
            if writer is not None:
                writer.add_scalar("loss_train/ita", loss_ita.item(), global_step)
                writer.add_scalar("loss_train/itm", loss_itm.item(), global_step)
                writer.add_scalar("loss_train/lm", loss_lm.item(), global_step)
                if loss_itc_kd is not None:
                    writer.add_scalar("loss_train/itc_kd", loss_itc_kd.item(), global_step)
                if loss_lm_kd is not None:
                    writer.add_scalar("loss_train/lm_kd", loss_lm_kd.item(), global_step)
                writer.add_scalar("loss_train/total", loss.item(), global_step)
                writer.add_scalar("optim/lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("train/alpha", alpha, global_step)
                if ttm_enabled:
                    writer.add_scalar("train/gamma", gamma, global_step)
                    writer.add_scalar("train/beta", ttm_soft_weight * gamma, global_step)
            # 이후 벨리데이션 로그도 여기다 적기

            #### 수정부분 시작: 실험 4 - logit_scale(곱하기 reparam) 기록 ####
            # model.logit_scale도 DDP gradient all-reduce로 모든 rank에서 bit-identical하게 유지되므로
            # writer 유무와 상관없이 모든 rank가 독립적으로 계산해도 동일한 시점에 동일한 결론에 도달함
            model_for_log = model.module if hasattr(model, "module") else model
            logit_scale_raw = model_for_log.logit_scale.detach().item() # log space의 raw parameter
            logit_scale_value = model_for_log.logit_scale.detach().exp().item() # 실제 loss에 곱해지는 scale

            if writer is not None:
                writer.add_scalar("model/logit_scale_raw", logit_scale_raw, global_step)
                writer.add_scalar("model/logit_scale", logit_scale_value, global_step)
            #### 수정부분 끝 ####

            #### 수정부분 시작: temp 런어웨이 ablation용 자동 종료 트리거 (실험 1~4 끝나면 제거) ####
            if collapse_counter is not None:
                if logit_scale_value > COLLAPSE_LOGIT_SCALE_THRESHOLD:
                    collapse_counter['sustained_steps'] += config["tb_train_log_interval"]
                else:
                    collapse_counter['sustained_steps'] = 0

                if collapse_counter['sustained_steps'] >= COLLAPSE_SUSTAINED_STEPS:
                    raise CollapseDetected(
                        f"logit_scale={logit_scale_value:.2f}가 {collapse_counter['sustained_steps']} step 연속으로 "
                        f"{COLLAPSE_LOGIT_SCALE_THRESHOLD} 초과 (epoch={epoch}, global_step={global_step})"
                    )
            #### 수정부분 끝 ####

        #### 수정부분 시작: train 도중 validation loss optional 실행 ####
        if val_loss_runner is not None:
            val_loss_runner.val_loss_during_train(
                model=model,
                epoch=epoch,
                iteration=i,
                global_step=global_step,
                train_loader_len=len(data_loader),
            )
        #### 수정부분 끝 ####

        #### 수정부분 시작: train 도중 retrieval validation optional 실행 (ITC는 자주, ITM rerank는 드물게) ####
        if retrieval_val_runner is not None:
            retrieval_val_runner.val_retrieval_during_train(
                model_without_ddp=model_without_ddp,
                epoch=epoch,
                iteration=i,
                global_step=global_step,
            )
        #### 수정부분 끝 ####

        #### 수정부분 시작: train 도중 caption validation optional 실행 (경량 지표만, ITM mid와 동일 시점) ####
        if caption_val_runner is not None:
            caption_val_runner.val_caption_during_train(
                model_without_ddp=model_without_ddp,
                epoch=epoch,
                iteration=i,
                global_step=global_step,
            )
        #### 수정부분 끝 ####


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


def main(args, config): # configs.pretrain.yaml
    utils.init_distributed_mode(args)    
    device = torch.device(args.device)

    # 디스트리뷰트 이후에 라이터 삽입
    writer = None
    if utils.is_main_process():
        run_name = make_tb_run_name(config) # 컨피그 바탕으로 런네임 제작
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        tb_log_dir = os.path.join(
            args.output_dir,
            "tensorboard",
            f"{timestamp}__{run_name}"
        )

        writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"[TensorBoard] log_dir: {tb_log_dir}")
    
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset") # base_dataset으로 이름 변경
    base_datasets = [create_dataset('pretrain', config, min_scale=0.2)] # 이미지 루트 인자 추가
    print('number of training samples: %d'%len(base_datasets[0]))

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()            
    samplers = create_sampler(base_datasets, [True], num_tasks, global_rank) 

    base_loader = create_loader(base_datasets,samplers,batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]      

    #### cc12m dataset ####
    if config['cc12m_tar_path']:
        print("Creating cc12m dataset")
        cc12m_datasets = create_dataset('pretrain_cc12m_webdataset', config, min_scale=0.2) # 여기서 이미 배치가 완성되서 나감
        # 여긴 넘버를 적을 수 없음. 스킵.
        cc12m_loader = wds.WebLoader(
            cc12m_datasets,
            batch_size=None, # 데이터셋에서 이미 배치 완성함
            num_workers=4,
            pin_memory=True
        )
        ### ddp equlizer for stable webdataloading ####
        if num_tasks > 1:
            ratio = config["cc12m_ratio"]
            batches_per_gpu = len(base_loader) * ratio
            print(f"하나의 gpu가 소모하는 배치 양: {batches_per_gpu}")
            cc12m_loader = cc12m_loader.with_epoch(batches_per_gpu) # ddp 메서드 대신 하나의 gpu가 소모하는 양 체크

        data_loader = CombinedLoader(
            loader_map=base_loader,
            loader_iterable=cc12m_loader,
            ratio=config['cc12m_ratio']
        )
    else: # 패스를 빼주면 원래대로 가도록 수정함.
        data_loader = base_loader
    
    #### 수정부분 시작: validation loss runner 생성 ####
    # 여기서 COCO Karpathy validation json을 rank별로 한 번 읽고,
    # 이후 train 중간 / epoch 종료 validation에서 재사용한다.
    val_loss_runner = eval_validation_loss.build_pretrain_val_loss_runner( # 러너 생성
        config=config,
        device=device,
        writer=writer,
    )
    #### 수정부분 끝 ####

    #### 수정부분 시작: retrieval validation runner 생성 (momentum 없이 student 인코더만으로 COCO val retrieval 평가) ####
    retrieval_val_runner = eval_validation_retrieval.build_pretrain_retrieval_val_runner(
        config=config,
        device=device,
        writer=writer,
    )
    #### 수정부분 끝 ####

    #### 수정부분 시작: caption validation runner 생성 (student generate로 COCO val 캡션 채점) ####
    caption_val_runner = eval_validation_caption.build_pretrain_caption_val_runner(
        config=config,
        device=device,
        writer=writer,
    )
    #### 수정부분 끝 ####




    #### Model #### 
    print("Creating model")
    ttm_cfg = config.get('distill', {}).get('itc_target_mix', {})
    model = blip_pretrain(image_size=config['image_size'],
                          vit=config['vit'],
                          vit_grad_ckpt=config['vit_grad_ckpt'],
                          vit_ckpt_layer=config['vit_ckpt_layer'],
                          queue_size=config['queue_size'],
                          my_bert_size=config['my_bert_size'],
                          ttm_enabled=ttm_cfg.get('enabled', False),
                          ttm_variant=ttm_cfg.get('variant', 'in_batch'),
                          ttm_temp=float(ttm_cfg.get('temp', 0.05)),
                          ttm_soft_weight=float(ttm_cfg.get('soft_weight', 0.4)))

    model = model.to(device)   

    #### 수정부분 시작: 실험 4 - logit_scale reparam + weight decay 대상에서 제외 ####
    # optimizer 생성은 DDP wrap 이전이라 named_parameters() 이름이 'logit_scale' 그대로 나옴 (module. 접두사 없음)
    no_decay_param_names = {'logit_scale'}
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if name in no_decay_param_names:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': config['weight_decay']},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=config['init_lr'],
    )
    #### 수정부분 끝 ####
    
    start_epoch = 0
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']    
        model.load_state_dict(state_dict)    
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']+1                
        print('resume checkpoint from %s'%args.checkpoint)    
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # 만약 ddp가 설정됨 -> without ddp와 ddp를 분리시킴. model은 ddp설정된거, w/o ddp는 일반 블립 하나

    #### online teacher (distillation) — replicate per rank, frozen, not DDP-wrapped ####
    online_teacher = None
    distill_cfg = config.get('distill', {})
    ttm_on = distill_cfg.get('itc_target_mix', {}).get('enabled', False)
    teacher_keep = tuple(k for k in ('itc', 'lm')
                         if distill_cfg.get(k, {}).get('enabled', False)
                         or (k == 'itc' and ttm_on))
    if teacher_keep:
        ckpt_path = config.get('teacher', {}).get('checkpoint', '')
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"distill enabled (keep={teacher_keep}) but teacher.checkpoint not found: '{ckpt_path}'"
            )
        from distillation.online_teacher import OnlineTeacher
        online_teacher = OnlineTeacher(
            checkpoint=ckpt_path,
            image_size=config['image_size'],
            vit='large', bert='base',
            queue_size=config['queue_size'],
            keep=teacher_keep,
        ).to(device)
        print(f"[distill] online teacher loaded (keep={teacher_keep})")

    #### 수정부분 시작: temp 런어웨이 ablation용 자동 종료 트리거 - epoch 경계 넘어서도 연속 카운트 유지 (실험 1~4 끝나면 제거) ####
    collapse_counter = {'sustained_steps': 0}
    #### 수정부분 끝 ####

    try: # writer 자동종료를 위해서
        print("Start training")
        start_time = time.time()
        for epoch in range(start_epoch, config['max_epoch']):

            step_lr_schedule(optimizer, epoch, config['init_lr'], config['min_lr'], config['lr_decay_rate'])

            train_stats = train(model, data_loader, optimizer, epoch, device, config, writer, val_loss_runner=val_loss_runner, collapse_counter=collapse_counter, model_without_ddp=model_without_ddp, retrieval_val_runner=retrieval_val_runner, caption_val_runner=caption_val_runner, online_teacher=online_teacher) # online_teacher 추가
            
            #### 수정부분 시작: epoch 종료 validation loss 실행 ####
            val_stats = {}

            if val_loss_runner is not None:
                epoch_end_global_step = (epoch + 1) * len(data_loader)

                val_stats = val_loss_runner.run_epoch_end(
                    model=model,
                    epoch=epoch,
                    global_step=epoch_end_global_step,
                    train_loader_len=len(data_loader),
                )
            #### 수정부분 끝 ####

            #### 수정부분 시작: epoch 종료 retrieval validation 실행 ####
            if retrieval_val_runner is not None:
                epoch_end_global_step = (epoch + 1) * len(data_loader)

                retrieval_val_runner.run_epoch_end(
                    model_without_ddp=model_without_ddp,
                    epoch=epoch,
                    global_step=epoch_end_global_step,
                )
            #### 수정부분 끝 ####

            #### 수정부분 시작: epoch 종료 caption validation 실행 (경량 + SPICE) ####
            if caption_val_runner is not None:
                epoch_end_global_step = (epoch + 1) * len(data_loader)

                caption_val_runner.run_epoch_end(
                    model_without_ddp=model_without_ddp,
                    epoch=epoch,
                    global_step=epoch_end_global_step,
                )
            #### 수정부분 끝 ####

            if utils.is_main_process():
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,
                            }                     
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  
                
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            dist.barrier()        
                    
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    #### 수정부분 시작: temp 런어웨이 ablation용 자동 종료 트리거 - 감지 시 체크포인트 남기고 종료 (실험 1~4 끝나면 제거) ####
    except CollapseDetected as e:
        print(f"[COLLAPSE TRIGGER] {e}")
        if utils.is_main_process():
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_collapsed.pth'))
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps({'collapsed': str(e)}) + "\n")
        sys.exit(99)
    #### 수정부분 끝 ####

    finally:
        if writer is not None:
            writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/pretrain.yaml') # 세팅을 바꿔놨음
    # parser.add_argument('--output_dir', default='')   # 여기서 계속 문제가 생기네
    parser.add_argument('--checkpoint', default='')    
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    # config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader) depreciated in yaml
    with open(args.config, 'r') as f: # pyYAML사용
        config = yaml.safe_load(f)
    args.output_dir = config["output_dir"]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True) # 있으면 경고내도록
    
    # yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))   dep in yaml 
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    print(f"output_dir: {args.output_dir}")    
    
    main(args, config)