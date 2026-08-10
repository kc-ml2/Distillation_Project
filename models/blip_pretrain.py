'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer
import transformers
# transformers.logging.set_verbosity_error()
transformers.logging.set_verbosity_warning() # 에러와 경고까지는 보여줌?
logger = transformers.logging.get_logger(__name__) # added? for warning at bottom logger

import math
import torch
from torch import nn
import torch.nn.functional as F
from custom_functions.dinov3_encoder import DINOv3_Wrapper # custom function으로 이동한 후에 임포트

from models.blip import create_vit, init_tokenizer, load_checkpoint
from distillation.losses import lm_distill_loss, itm_target_mix_loss

#### 수정부분 시작: 실험 4 - temp 나누기 대신 logit_scale 곱하기로 reparam ####
# 기존 temp clamp 범위 (0.001, 0.5) -> effective scale(1/temp) 범위는 (2, 1000).
# reparam 자체의 효과만 보기 위해 exp(logit_scale)의 허용 범위를 동일하게 (2, 1000)으로 맞춤.
LOGIT_SCALE_MIN = math.log(2)
LOGIT_SCALE_MAX = math.log(1000)
#### 수정부분 끝 ####

class BLIP_Pretrain(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/bert_config.json',
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                    
                 embed_dim = 256,     
                 queue_size = 57600,
                 momentum = 0.995,
                 ttm_enabled = False,
                 ttm_variant = 'in_batch',
                 ttm_temp = 0.05,
                 ttm_soft_weight = 0.4,

                 # add option for language model
                 my_bert_size = "base", # default = bert (original)
                 med_bert_medium_config = 'configs/bert_medium_config.json',
                 med_bert_MiniLM_config = 'configs/bert_minilm_config.json', # med라고 적어놨지만 med가 아닌 일반 버트컨피그임

                 # False면 base(deit)/large(in21k) 백본 사전 초기화를 건너뜀.
                 # 생성 직후 BLIP 체크포인트로 전량 덮는 경우용(온라인 티처 / 체크포인트 평가).
                 # large의 in21k 경로는 timm 1.x에서 깨져 있어 large를 로드하려면 필수.
                 # small 계열은 timm.create_model(pretrained=True) 내장이라 이 플래그와 무관.
                 init_backbone_weights = True
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        if vit=='base':
            self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer, 0)
            if init_backbone_weights:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                    map_location="cpu", check_hash=True)
                state_dict = checkpoint["model"]
                msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
        elif vit=='large':
            self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer, 0)
            if init_backbone_weights:
                # timm 0.4.x 시절 API — timm 1.x에서는 default_cfgs 항목이 DefaultCfg라 깨짐.
                # 체크포인트를 로드하는 쪽(온라인 티처 / 평가)은 init_backbone_weights=False로 우회.
                from timm.models.helpers import load_custom_pretrained
                from timm.models.vision_transformer import default_cfgs
                load_custom_pretrained(self.visual_encoder,default_cfgs['vit_large_patch16_224_in21k'])
        
        elif vit=='small': # model for small vit, and 
            # model: vit_small_patch16_dinov3
            import timm
            self.visual_encoder = timm.create_model(
                model_name='vit_small_patch16_dinov3.lvd1689m',
                pretrained=True,
                img_size=224,
                num_classes=0,
                global_pool='',
                num_reg_tokens=0 # do not have reg tokens
            )
        
        # add vit small with register tokens option (i don't know how skipping register token is working well?)
        elif vit=='small_reg':
            import timm
            raw_model = timm.create_model(
                model_name='vit_small_patch16_dinov3.lvd1689m',
                pretrained=True,
                img_size=224,
                num_classes=0,
                global_pool='',
                num_reg_tokens=4 #
            )
            self.visual_encoder = DINOv3_Wrapper(base_model=raw_model, model_register_tokens=4)

        elif vit=='small_plus':
            import timm
            self.visual_encoder = timm.create_model(
                model_name="vit_small_plus_patch16_dinov3.lvd1689m",
                pretrained=True,
                img_size=224,
                num_classes=0,
                global_pool='', # output
                num_reg_tokens=0 # do not have reg tokens
            )

        elif vit=='small_plus_reg': 
            import timm
            raw_model = timm.create_model(
                model_name="vit_small_plus_patch16_dinov3.lvd1689m",
                pretrained=True,
                img_size=224,
                num_classes=0,
                global_pool='', # output
                num_reg_tokens=4 # default =4
            )
            self.visual_encoder = DINOv3_Wrapper(base_model=raw_model, model_register_tokens=4)
        
        # additional conditions for small vit
        if vit in ['small', 'small_plus', 'small_reg', 'small_plus_reg']:
            # embed_dim -> vision projection output
            # embed_dim = 384 # needs to be checked. it is raw output for model
            embed_dim = 256 # output layer 384 of vit to 256 projection bert also projected to 256 same.
            
            # model making method is different: small's vs base and large
            vision_width = 384 # hard coded for these model
        print(f"모델 비전 width 변수: {vision_width}")
        #========================================================================================================
        
        ## do we use bert or other generation model?
        # ok for experiment: fix same bert with only different config of smaller model
        # set bert_small_config.json
        # config initialize method changed

        # 모델 체급별 정보 관리 딕셔너리
        model_specs = {
            "minilm": {
                "hub_id": "microsoft/MiniLM-L12-H384-uncased",
                "config": med_bert_MiniLM_config
            },
            "medium": {
                "hub_id": "google/bert_uncased_L-8_H-512_A-8",
                "config": med_bert_medium_config
            },
            "base": {
                "hub_id": "bert-base-uncased",
                "config": med_config
            }
        } # 저렇게 변수를 저장해놔도 되는구나 흠흠
        if my_bert_size in model_specs:
            spec = model_specs[my_bert_size]
            self.tokenizer = init_tokenizer() # 토크나이저 초기화
            encoder_config = BertConfig.from_json_file(spec["config"]) # 컨피그에 맞게 가져옴
            encoder_config.encoder_width = vision_width # 수정 # 이부분은 한번 더 맞춰줘야 크로소 모달이 잘 작동한다 - 디코더 커플링을하면서 컨피그를 수정을 안해놓으니깐 오류가 나네

            self.text_encoder = BertModel.from_pretrained(
                pretrained_model_name_or_path=spec["hub_id"], 
                config=encoder_config, 
                add_pooling_layer=False
            )

            self.text_encoder.resize_token_embeddings(len(self.tokenizer)) # 컨피그로 만든 임베딩 토큰 수가 다르니깐 다시 하는 것
            text_width = self.text_encoder.config.hidden_size # 768 default 모델 내부의 고유한 벡터 차원.
            print("encoder loading step finish")
            # print(self.text_encoder) # 
        else:
            raise ValueError(f"Unknown bert size: {my_bert_size}")
        print(f"bert size: {my_bert_size}")
        assert my_bert_size in ["base", "medium", "minilm"], "bert size must be base, medium, minilm" # 버트를 만들고 나서

        # ====================== depreciated ==============
        '''
        if my_bert_size == 'base':
            self.tokenizer = init_tokenizer()   # -> blip.py file
            encoder_config = BertConfig.from_json_file(med_config) # configs.bert_config.json, only vocab size is different
            encoder_config.encoder_width = vision_width # when used as med - vit output is important - default = 768
            # encoder_width = 외부 출력값을 받아들일 때 즉, 비전을 받아들일 때 이 width를 쓴다
            self.text_encoder = BertModel.from_pretrained(
                'bert-base-uncased',
                config=encoder_config,
                add_pooling_layer=False
            )
            self.text_encoder.resize_token_embeddings(len(self.tokenizer)) # 컨피그로 만든 임베딩 토큰 수가 다르니깐 다시 하는 것
            text_width = self.text_encoder.config.hidden_size # 768 default 모델 내부의 고유한 벡터 차원.
            med_config = med_config # 자기 자신-base 모델 사용
        
        # feat addition: small bert model import code
        elif my_bert_size == 'medium':
            self.tokenizer = init_tokenizer()   # additional two special token
            encoder_config = BertConfig.from_json_file(med_bert_medium_config) # small model config

            # encoder_width 는 우리가 따로 달은 컨피그임. 따라서 밑에서 바뀌게 됨
            encoder_config.encoder_width = vision_width # 이건 자동으로 바꾸게 될 거고 384로
            self.text_encoder = BertModel.from_pretrained(
                'google/bert_uncased_L-8_H-512_A-8',
                config=encoder_config,
                add_pooling_layer=False
            )
            # https://huggingface.co/google/bert_uncased_L-8_H-512_A-8
            self.text_encoder.resize_token_embeddings(len(self.tokenizer)) # 컨피그로 만든 임베딩 토큰 수가 다르니깐 다시 하는 것.
            text_width = self.text_encoder.config.hidden_size # 512
            med_config = med_bert_medium_config # 밑에서부터는 변경해서 들어가게
        
        elif my_bert_size == "MiniLM":
            self.tokenizer = init_tokenizer()
            encoder_config = BertConfig.from_json_file(med_bert_MiniLM_config)
            encoder_config.encoder_width = vision_width
            self.text_encoder = BertModel.from_pretrained(
                'microsoft/MiniLM-L12-H384-uncased',
                config=encoder_config,
                add_pooling_layer=False
            )
            # https://huggingface.co/microsoft/MiniLM-L12-H384-uncased
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
            text_width = self.text_encoder.config.hidden_size # 384
            med_config = med_bert_MiniLM_config # decoder와 momentum은 이제 이 컨피그를 보고 제작함
        '''
        # ================= depreciated ================

        # itc loss part
        self.vision_proj = nn.Linear(vision_width, embed_dim) # 256 in default 
        self.text_proj = nn.Linear(text_width, embed_dim) # 256 in default? why 256 size? - 애초에 contrastive learning을 256으로함
        # 이건 itc loss 를 구하기 위해서 줄인 듯 하다.

        # itm loss head
        self.itm_head = nn.Linear(text_width, 2) # binary 
        
        
        # create momentum encoders  == shadow model for stable learning in itm?
        ## do we have to maintain momentum encoder?
        # 아하 여기서 create_vit와 text_encoder_m이 생성되는구나
        # 미리 컨피그는 다 따놨으니 알아서 될 듯 함.
        self.visual_encoder_m, vision_width = create_vit(vit,image_size)              
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(config=encoder_config, add_pooling_layer=False)      
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]       
        self.copy_params() # copy paras function -> self.model_pairs loop -> copy and grad off

        # 아래 두 코드는 timm import에서만 작동함
        # print("Online Dim:", self.visual_encoder.model.embed_dim)
        # print("Momentum Dim:", self.visual_encoder_m.model.embed_dim)
        # # 모델 생성 직후 테스트기
        # print("Checking Layer 0 Cross-Attention:")
        # layer0_ca = self.text_encoder.encoder.layer[0].crossattention
        # print(f"Layer 0 CrossAttention exists: {layer0_ca is not None}")
        # print(f"Layer 0 CrossAttention Query weight shape: {layer0_ca.self.query.weight.shape}")

        # # 가중치가 랜덤인지(평균이 0에 가깝고 표준편차가 작음) 확인
        # print(f"Layer 0 Mean: {layer0_ca.self.query.weight.mean().item():.4f}")

        # create the queue
        ## momentum encoder requirements: queue size 57600 at __init__
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long)) # queue pointer?

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0) # norm?
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        #### teacher-target-mixing (exp9) — config-gated, 기본 OFF ####
        self.ttm_enabled = ttm_enabled
        self.ttm_variant = ttm_variant
        self.ttm_temp = ttm_temp
        self.ttm_soft_weight = ttm_soft_weight
        if ttm_enabled and ttm_variant == 'queue':
            # momentum 큐와 동형인 티처 큐 (frozen이라 드리프트 없음)
            self.register_buffer("teacher_image_queue", torch.randn(embed_dim, queue_size))
            self.register_buffer("teacher_text_queue", torch.randn(embed_dim, queue_size))
            self.teacher_image_queue = nn.functional.normalize(self.teacher_image_queue, dim=0)
            self.teacher_text_queue = nn.functional.normalize(self.teacher_text_queue, dim=0)

        self.queue_size = queue_size # 57600
        self.momentum = momentum # 0.995
        #### 수정부분 시작: 실험 4 - temp 나누기 대신 logit_scale 곱하기로 reparam ####
        # exp(logit_scale_init) = 1/0.07 ≈ 14.2857로, 기존 temp=0.07 시작점과 동일한 effective scale에서 출발
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        #### 수정부분 끝 ####
        # momentum은 안바꿔도 되더라 그런데 디코더는 손을 좀 봐야겠음.

        ## ==========================decoder line=====================
        # 디코더만 학습시킬까 생각했는데 컨피그만 잘 주면 아예 윗부분도 다 재사용할 수 있겠는데?
        # create the decoder -> go to med.py file
        # 미리 컨피그랑 다 바꾸어놓 지 않았네
        # 수정된 model_spec을 사용해서 디코더도 한번에 생성해 보자

        if my_bert_size in model_specs:
            spec = model_specs[my_bert_size]
            decoder_config = BertConfig.from_json_file(spec["config"])
            decoder_config.encoder_width = vision_width # 수정
            self.text_decoder = BertLMHeadModel.from_pretrained(
                pretrained_model_name_or_path=spec["hub_id"],
                config=decoder_config
            )
            self.text_decoder.resize_token_embeddings(len(self.tokenizer)) # 위에서 이미 선언함
            tie_encoder_decoder_weights(self.text_encoder,self.text_decoder.bert,'','/attention') # 디코더 묶기
        
        # ========== depreciated =========
        # decoder_config = BertConfig.from_json_file(med_config)
        # decoder_config.encoder_width = vision_width      # 비전 인코더의 아웃풋 출력 = small은 384임  이 부분은 쓸때마다 바뀌기 때문인듯하다.
        # self.text_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased',config=decoder_config)    
        # self.text_decoder.resize_token_embeddings(len(self.tokenizer)) 
        # tie_encoder_decoder_weights(self.text_encoder,self.text_decoder.bert,'','/attention')
        # =========== depreciated ========
        

    def generate(self, image, sample=False, num_beams=3, max_length=20, min_length=5,
                 top_p=0.9, repetition_penalty=1.0, prompt=''):
        """캡션 생성 (BLIP_Decoder.generate 미러링). pretrain 모델의
        visual_encoder + text_decoder + tokenizer를 그대로 사용, prompt는 인자."""
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}

        prompts = [prompt] * image.size(0)
        input_ids = self.tokenizer(prompts, return_tensors="pt").input_ids.to(image.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        if sample:
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  do_sample=True,
                                                  top_p=top_p,
                                                  num_return_sequences=1,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,
                                                  repetition_penalty=1.1,
                                                  **model_kwargs)
        else:
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  num_beams=num_beams,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,
                                                  repetition_penalty=repetition_penalty,
                                                  **model_kwargs)

        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(prompt):])
        return captions

    def forward(self, image, caption, alpha, update_train_state=None,
                teacher_img_feat=None, teacher_text_feat=None,
                teacher_lm_logits=None, teacher_lm_input_ids=None, lm_distill_temp=2.0,
                gamma=None, online_teacher=None, itm_mix=None):
    #### 수정부분 시작: validation-safe forward option 추가 ####
        """
        update_train_state:
            - None  : self.training 값을 따라감
                    train mode면 True, eval mode면 False
            - True  : train forward처럼 내부 학습 state 갱신 허용
                    self.logit_scale clamp_, momentum encoder update, queue enqueue 수행
            - False : validation forward처럼 loss만 계산
                    self.logit_scale 직접 변경, momentum update, queue enqueue 금지

        주의:
            model.eval()은 dropout/batchnorm 동작만 바꾸며,
            forward 안의 self.logit_scale.clamp_(), _momentum_update(),
            _dequeue_and_enqueue()를 자동으로 막아주지는 않는다.
        """

        if update_train_state is None:
            update_train_state = self.training

        #### 수정부분 시작: 실험 4 - temp 나누기 대신 logit_scale 곱하기로 reparam ####
        # self.logit_scale은 contrastive loss에 곱해지는 scale의 log값 (exp(logit_scale)가 실제 scale).
        # train에서는 기존 BLIP처럼 in-place clamp로 parameter 범위를 유지한다.
        # validation에서는 parameter를 직접 바꾸지 않고 계산용 clamped value만 사용한다.
        if update_train_state:
            with torch.no_grad():
                self.logit_scale.clamp_(LOGIT_SCALE_MIN, LOGIT_SCALE_MAX)
            safe_scale = self.logit_scale.exp()
        else:
            safe_scale = self.logit_scale.clamp(LOGIT_SCALE_MIN, LOGIT_SCALE_MAX).exp()
        #### 수정부분 끝 ####
    #### 수정부분 끝 ####
        
        image_embeds = self.visual_encoder(image) # 임베딩 벡터 따와
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device) # 어텐션 마스크 준비물
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)          # 흠 이놈은 뭐지? vision_proj에 0번 토큰 = cls를 준다
        
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=30, # 토크나이저가 쪼개서 줌
                              return_tensors="pt").to(image.device)  
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask, # return dict?이건 뭐지?
                                        return_dict = True, mode = 'text')            # 아웃풋도 텍스트로만 나가게 하고
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)                 # 노말라이즈는 국룰인가보네
        # 이건 ITC 로스를 구하는 코드구나.
             
        # get momentum features
        with torch.no_grad():
            #### 수정부분 시작: validation에서는 momentum encoder 갱신 금지 ####
            if update_train_state: # T / F 만 존재
                self._momentum_update()
            #### 수정부분 끝 ####

            image_embeds_m = self.visual_encoder_m(image) 
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  
            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)                   
            
            text_output_m = self.text_encoder_m(text.input_ids, attention_mask = text.attention_mask,                      
                                                return_dict = True, mode = 'text')    
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

            #### 수정부분 시작: 실험 4 - temp 나누기 대신 logit_scale 곱하기 ####
            sim_i2t_m = image_feat_m @ text_feat_all * safe_scale
            sim_t2i_m = text_feat_m @ image_feat_all * safe_scale
            #### 수정부분 끝 ####

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)

            if self.ttm_enabled and gamma is not None and teacher_img_feat is not None:
                from distillation.target_mix import (
                    teacher_soft_in_batch, teacher_soft_queue, mix_target)
                n_cols = sim_i2t_m.shape[1]
                mom_i2t = F.softmax(sim_i2t_m, dim=1)
                mom_t2i = F.softmax(sim_t2i_m, dim=1)
                ti = teacher_img_feat.to(image.device).float()
                tt = teacher_text_feat.to(image.device).float()
                if self.ttm_variant == 'queue':
                    t_img_all = torch.cat([ti.t(), self.teacher_image_queue.clone().detach()], dim=1)
                    t_txt_all = torch.cat([tt.t(), self.teacher_text_queue.clone().detach()], dim=1)
                    teacher_i2t = teacher_soft_queue(ti, t_txt_all, self.ttm_temp)
                    teacher_t2i = teacher_soft_queue(tt, t_img_all, self.ttm_temp)
                else:  # in_batch (D)
                    teacher_i2t = teacher_soft_in_batch(ti, tt, self.ttm_temp, n_cols)
                    teacher_t2i = teacher_soft_in_batch(tt, ti, self.ttm_temp, n_cols)
                sim_i2t_targets = mix_target(sim_targets, mom_i2t, teacher_i2t, gamma, self.ttm_soft_weight)
                sim_t2i_targets = mix_target(sim_targets, mom_t2i, teacher_t2i, gamma, self.ttm_soft_weight)
            else:
                sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        #### 수정부분 시작: 실험 4 - temp 나누기 대신 logit_scale 곱하기 ####
        sim_i2t = image_feat @ text_feat_all * safe_scale
        sim_t2i = text_feat @ image_feat_all * safe_scale
        #### 수정부분 끝 ####
                             
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

        loss_ita = (loss_i2t+loss_t2i)/2 # itc는 평균내서 보는구나 그런데 이상하네 왜 왜 image - text text - image가 다른 거지?
    
        #### 수정부분 시작: validation에서는 queue 업데이트 금지 ####
        if update_train_state:
            self._dequeue_and_enqueue(image_feat_m, text_feat_m, teacher_img_feat, teacher_text_feat)
        #### 수정부분 끝 ####  

        ###============== Image-text Matching ===================###
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
        bs = image.size(0)

        # negative-mining sampling weights: teacher-sim (itm_mix teacher) or student-sim.
        if itm_mix is not None and itm_mix['neg_source'] == 'teacher':
            assert teacher_img_feat is not None and teacher_text_feat is not None, \
                "itm_mix neg_source='teacher' requires teacher_img_feat/teacher_text_feat"
            with torch.no_grad():
                t_img = teacher_img_feat.to(image.device)
                t_txt = teacher_text_feat.to(image.device)
                s = itm_mix['sel_scale']
                weights_i2t = F.softmax(t_img @ t_txt.t() * s, dim=1) + 1e-4
                weights_t2i = F.softmax(t_txt @ t_img.t() * s, dim=1) + 1e-4
                weights_i2t.fill_diagonal_(0)
                weights_t2i.fill_diagonal_(0)
        else:
            with torch.no_grad():
                weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1) + 1e-4
                weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1) + 1e-4
                weights_t2i.fill_diagonal_(0)
                weights_i2t.fill_diagonal_(0)

        # draw one negative index per sample (neg image for each text, neg text for each image)
        neg_idx_img = [torch.multinomial(weights_t2i[b], 1).item() for b in range(bs)]
        neg_idx_txt = [torch.multinomial(weights_i2t[b], 1).item() for b in range(bs)]
        image_embeds_neg = torch.stack([image_embeds[neg_idx_img[b]] for b in range(bs)])
        text_ids_neg = torch.stack([encoder_input_ids[neg_idx_txt[b]] for b in range(bs)])
        text_atts_neg = torch.stack([text.attention_mask[neg_idx_txt[b]] for b in range(bs)])

        # one merged 3B forward: rows [pos B | neg-image B | neg-text B]
        text_ids_all = torch.cat([encoder_input_ids, encoder_input_ids, text_ids_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text.attention_mask, text_atts_neg], dim=0)
        image_embeds_all = torch.cat([image_embeds, image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts, image_atts], dim=0)
        output_all = self.text_encoder(text_ids_all,
                                       attention_mask=text_atts_all,
                                       encoder_hidden_states=image_embeds_all,
                                       encoder_attention_mask=image_atts_all,
                                       return_dict=True)
        vl_output = self.itm_head(output_all.last_hidden_state[:, 0, :])   # [3B, 2]
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long),
                                torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(image.device)

        # ITM loss: teacher target-mix (W>0) or plain CE.
        if itm_mix is not None and itm_mix['soft_weight'] > 0 and online_teacher is not None:
            teacher_soft = online_teacher.itm_soft(
                image, encoder_input_ids, text.attention_mask,
                neg_idx_img, neg_idx_txt, itm_mix['temp'],
                image_embeds=itm_mix.get('teacher_image_embeds'))          # [3B, 2], no grad
            loss_itm = itm_target_mix_loss(vl_output, itm_labels,
                                           teacher_soft.to(image.device),
                                           itm_mix['soft_weight'])
        else:
            loss_itm = F.cross_entropy(vl_output, itm_labels)
        
        ##================= LM ========================##     
        decoder_input_ids = text.input_ids.clone()      # 얘는 어떻게 생겼길래? input_ids가? 입력 토큰처럼 생겼나?
        decoder_input_ids[:,0] = self.tokenizer.bos_token_id # batch,0번을 bos_token_id로 바꾼다
        decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100) # 패딩은 -100으로 채우기

        decoder_output = self.text_decoder(decoder_input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,                  
                                           labels = decoder_targets,
                                           return_dict = True,   
                                          )   
          
        loss_lm = decoder_output.loss

        # external-teacher LM logit distillation (token-level, teacher-forced); None when disabled.
        loss_lm_kd = None
        if teacher_lm_logits is not None:
            if teacher_lm_input_ids is not None:
                assert torch.equal(teacher_lm_input_ids, decoder_input_ids), \
                    "teacher/student decoder input mismatch (tokenizer drift?)"
            loss_lm_kd = lm_distill_loss(decoder_output.logits,
                                         teacher_lm_logits.to(image.device),
                                         decoder_targets, lm_distill_temp)

        return loss_ita, loss_itm, loss_lm, loss_lm_kd
 


    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat,
                             teacher_img_feat=None, teacher_text_feat=None):
        from distillation.target_mix import enqueue_all
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        batch_size = image_feats.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        pairs = [(self.image_queue, image_feats.T), (self.text_queue, text_feats.T)]
        if self.ttm_enabled and self.ttm_variant == 'queue' and teacher_img_feat is not None:
            t_img = concat_all_gather(teacher_img_feat.float())
            t_txt = concat_all_gather(teacher_text_feat.float())
            pairs += [(self.teacher_image_queue, t_img.T), (self.teacher_text_queue, t_txt.T)]
        self.queue_ptr[0] = enqueue_all(pairs, ptr, batch_size, self.queue_size)


def blip_pretrain(**kwargs):
    model = BLIP_Pretrain(**kwargs)
    return model 


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output     


from typing import List
def tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key:str):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        logger.info(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        uninitialized_encoder_weights: List[str],
        skip_key: str,
        depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias                
            print(module_name+' is tied')    
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                        encoder_modules
                    ) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key)  

