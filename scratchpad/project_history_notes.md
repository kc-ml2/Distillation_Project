# Distillation Project — 개발 히스토리 수집 노트

> 2026-07-16 작성. HTML/Markdown 히스토리 문서 작성을 위한 원자료.
> 출처: git log(전 브랜치 118커밋, 2026-05-12 ~ 2026-07-15), docs/superpowers/ 설계문서,
> critical_bugfix/ 진단 기록, output/ 실험 런 목록, 메모리 파일.

---

## 0. 프로젝트 개요

- **베이스**: Salesforce BLIP 공식 코드 (initial commit `15fe4f7`, 2026-05-12)
- **목표**: 소형 학생 모델(DINOv3 small ViT + MiniLM 텍스트) 구축 → BLIP-large 티처로부터 지식 증류
- **데이터**: COCO + Visual Genome (초기엔 CC12M도 시도)
- **평가축**: COCO retrieval (r_mean, rank 기반) + 캡션 메트릭 (BLEU/METEOR/ROUGE/CIDEr/SPICE)

## 1. 시기별 개발 타임라인

### Phase A — 학생 모델 아키텍처 셋업 (05-12 ~ 05-19, 브랜치: test_vit_output_dim, dev_text)
- `15fe4f7` BLIP 베이스 코드 임포트
- DINOv3 wrapper + register token 제어 (`236d79b`, `87c70ab`) — ViT 출력 차원 테스트
- blip_pretrain에 small/small_plus ViT 옵션 (`18fcab6`, `55f535b`)
- BERT 교체 가능화: minilm/medium config 추가, `my_bert_size` 옵션 (`c96b2c8`, `e38d6d7`, `0eb4ffa`)
- Lite-BERT distillation 코드 반입 (`ce77f57`)
- configs 소문자 통일, pretrain/dataset 경로 정비, loss curve 코드 (`337e70c`)

### Phase B — 소형 모델 pretrain 가동 (05-27 ~ 06-08, 브랜치: dev_small_vit)
- minilm + dinov3 조합 완성 (`e280d4c`), decoder encoder_width=vision_width (`bebe8fe`)
- CC12M pretrain 시도 (`d00543c`, `95bc146`)
- retrieval 코드 리팩터 (`626752e`), TB train 메트릭 (`705133b`), val loss ita/itm/lm (`b59aa82`)
- `d5726b1` (06-08) **"temperature scalar added - problem on converge"** ← temp collapse 최초 기록

### Phase C — 데이터셋 확장: COCO+VG (06-03, 브랜치: dev_vg_add)
- vg_captioning 함수 + create_dataset img root (`e37b101`), dataset control 폴더 (`326803a`)

### Phase D — Temperature collapse 진단 & ablation (06-15 ~ 06-23)
- gamma 셋업/lr 수정 (`294e823`, `1fa3fe5`, `c5f7a22`)
- **4실험 ablation 플랜** (메모리 project_temp_collapse_ablation):
  - exp 1.vg_crop: VG 이미지를 region bbox로 크롭 (`b6a0429`, `4cd2924`)
  - exp 2.no_decay_temp: contrastive temp를 weight decay에서 제외 (`d059128`)
  - exp 3: skip
  - exp 4.logit_scale_no_decay: temp를 logit_scale로 재파라미터화 + decay 제외 (`d0a9b33`)
- collapse 자동 정지 트리거 (`2beaed8`, `30a5209`)
- 결론(런 이름 기준): base 모델 66.2~66.7 도달, exp5-lrup은 logit_scale 110.9로 retrieval 최고
  → "높은 scale 자체가 병" 가설 기각

### Phase E — Retrieval validation 인프라 (06-21 ~ 06-23)
- 설계: `docs/superpowers/specs/2026-06-21-pretrain-retrieval-validation-design.md`
- RetrievalValRunner (COCO 5k, ITC-only + ITM-rerank 티어 분리) (`13049fe`, `614c98b`)
- pretrain.py 배선 (`95f7bde`), standalone eval 스크립트 (`d8b84eb`)
- CUDA device/dtype 버그픽스 + bf16 autocast (`263b021`)
- 체크포인트 sweep + TB 비교 툴링 (`be08f23`), ITM 미드에폭 1회로 조정 (`f270c8d`)
- 케이던스: ITC 1000step마다 + ITM 에폭당 1회

### Phase F — ITC distillation v1: offline teacher cache (06-27 ~ 06-28)
- 설계: `2026-06-27-itc-distillation-design.md`
- itc_distill_loss: in-batch B×B relational KL (`e9ec273`)
- teacher feature cache (manifest + fp16 memmap) + build_cache CLI (`d3ef234`, `cc89780`)
- dataset이 teacher feats 반환, train loop에 weighted itc_kd (`e7391f2`, `b5b937f`)
- pretrain image aug 끄기 토글 + config-driven exp tag (`b988045`) — no-aug(exp5)가 go/no-go 게이트

### Phase G — Online teacher로 피벗 (06-29)
- 설계: `2026-06-29-itc-distillation-online-teacher.md`
- offline cache 제거 (`9650845`), OnlineTeacher(frozen BLIP-large, ITC path, bf16 no_grad) (`bef69b5`, `6fc96df`)
- 피벗 이유: aug ON 학습과 캐시(무-aug) 불일치 → 온라인 티처로 전환, aug 복원 (`f5dc810`)

### Phase H — LM logit distillation (07-02, 브랜치: lm_distill)
- 설계: `2026-07-02-lm-logit-distillation-design.md`
- lm_distill_loss: token-level KD (shift frame, pad mask, T²) (`2bd7817`)
- OnlineTeacher.lm_logits (teacher-forced decoder) (`313a529`), keep arg 리팩터 (`6a63e25`)
- train loop 배선 + TB + run-name tag (`05a577f`), exp 8.lm_distill config (`9b35c8a`)
- `8e6f07c` timm 1.x 티처 크래시 픽스 (legacy backbone init 스킵) — 이후 dev에 FF 머지
- **결과**: ep20 완주, zero-shot 캡션 A/B: distill > solo CIDEr +2.5% / SPICE +2.0% (전 에폭), SPICE는 ep10 포화

### Phase I — Caption zero-shot eval 툴링 (07-03 ~ 07-07, 브랜치: claude/caption-minilm-eval)
- 설계: `2026-07-03-caption-minilm-zeroshot-eval-design.md`
- 캡션 디코더 BERT config를 my_bert_size로 선택 (`5483617`), med_config 파라미터 정리 (`e8e588e`)
- DINOv3 pos_embed interpolation 가드 (`eefc2a3`)
- use_spice=False 경로 (`bbc7b52`), config-driven eval base/large/minilm (`82a6c74`)
- generate 파라미터 (num_beams 등) (`125be06`), METEOR/ROUGE 계약 테스트 (`f1e0648`)
- base(amp/nodecay) eval + checkpoint arch override (`dfbe304`)
- `8547ed8` **main에 머지** (07-07) — 현재 main tip

### Phase J — ITC-distill-only 실험 + τ 진단 (07-06 ~ 07-14)
- `88ea5d4` single-file pretrain.yaml 제어 + τ=0.022 셋업
- **critical_bugfix 2026-07-07 (×τ² 계수)**: itc_kd loss가 다른 loss 대비 ~2000× 작음 발견.
  원인 = Hinton ×T² 관례의 무비판적 이식 (τ=0.05 → τ²=0.0025). KL 자체는 1.36 nats로 정상.
- **티처 유사도 실측** (measure_teacher_sim.py, 100배치×B40):
  - pos 0.42 / neg 0.31, margin(pos−hardneg) aug ON 0.016 vs OFF 0.030 — aug가 마진 절반 훼손
  - argmax=diag 58.7% = 티처가 옮길 수 있는 신호 상한
  - τ 스윕 → **τ=0.022 선정** (학생 수렴온도 0.023 앵커)
- exp 6.1 (droptau: τ=0.022, ×τ² 제거, λ=1) / 6.2 (keeptau2: no-op 대조군) 완주
- 결과: 6.1 ep1 r_mean 40.2 (baseline 21.9, +18 — 티처 신호 가치 증명)였으나 ep20 59.5 vs baseline 63.4 (**−3.9**)
- **critical_bugfix 2026-07-14 (grad 균형·기하)**: KD가 ITC 대비 2.6~6.7× grad로 압도 →
  margin 0.082→0.054 압축, ITC는 logit_scale 인플레(59→88)로 상쇄 (margin×scale 4.84 vs 4.77 불변).
  순위 재배열이 retrieval 손상. val ITC loss는 scale 오염 지표 → 런 간 비교는 retrieval로만.
- 병행: VG min_scale 0.5 실험 (`cbe9d30`, extra_exp/vg_transform_min0.5, base 모델)

### Phase K — Caption in-loop metric eval (07-13)
- 설계: `2026-07-13-caption-metric-eval-design.md`
- BLIP_Pretrain.generate() (`ee61bb6`), epoch-elapsed wall time 로그 (`2326127`)
- cpu_affinity ctx (`89a5550`), CaptionValRunner: DDP gen/gather + coco_caption_eval + TB (`0f9cc0a`)
- train loop 배선 (`889dec5`), rank0 non-fatal (`2237836`)
- `de9ee01` **dev에 머지** (07-13)
- SPICE 실측: 32코어=64코어=~65s → 32코어 캡

### Phase L — Phase 1 (armT/armS) + exp9 teacher-target-mix (07-14 ~ 07-15, 현재)
- `fe23681` (dev tip): ×τ 보정 + Phase 1 config — **armT** (τ=0.022, λ=0.1) / **armS** (τ=0.05, λ=0.25),
  caption-val + CPU 분리
- 설계: `2026-07-14-teacher-target-mix-design.md`
- **exp9 (브랜치 exp9_teacher_target_mix, 9커밋)**: 별도 KD loss 대신 티처를 ITC soft-target에 혼합
  → grad 충돌 원천 제거. γ 스케줄 (2ep hold + 12ep 선형 감쇠) (`02f4399`), 타깃 빌더 in_batch/queue (`01b60e5`),
  페어 enqueue (`dab1da0`, `875f8e9`), pretrain 배선 + TB beta/gamma (`d339a1e`),
  variant 9.1(in_batch)/9.2(queue) config (`e140c71`), γ=0 티처 forward 스킵 + assert (`3c24ec9`, `69a8107`)
- 상태: 단위테스트 9/9 + CPU 스모크 + 리뷰 SHIP READY, 스페어 서버 GPU 스모크 → 본런 대기
- Phase 1 armT/armS 승인 대기

## 2. 브랜치 지도

| 브랜치 | tip | 분기점(vs dev) | 상태 |
|---|---|---|---|
| main | `8547ed8` (07-07) | dev의 조상 | caption eval 머지까지 반영 |
| **dev** | `fe23681` (07-14) | — | 메인 개발선 |
| exp9_teacher_target_mix | `69a8107` (07-15) | fe23681 | origin push됨, GPU 스모크 대기 |
| extra_exp/vg_transform_min0.5 | `cbe9d30` (07-06) | 8e6f07c | VG min_scale 0.5 실험 |
| claude/caption-minilm-eval | `dfbe304` (07-07) | main에 머지됨 | 완료 |
| origin/lm_distill | `8e6f07c` (07-02) | dev에 FF 머지됨 | 완료 |
| origin/test_vit_output_dim | `87c70ab` (05-13) | 초기 | DINOv3 wrapper 실험 |
| origin/dev_text | `93b0792` (05-14) | 초기 | BERT 교체 작업 |
| origin/dev_small_vit | `95bc146` (06-02) | 초기 | small ViT + CC12M |
| origin/dev_vg_add | `fa5723f` (06-03) | 초기 | VG 데이터셋 추가 |

태그: 없음. 총 118커밋, 저자 minwoo → MINWOO CHOI (06-17 전환).

## 3. 실험 런 카탈로그 (output/)

- `pretrain_coco_vg_baseline` — 초기 baseline
- `pt_checkpoint_base` + 변형: `_amp`, `_logitscale_nodecay`(exp4), `_lrup`(exp5-lrup, retrieval 최고 66.2~66.7),
  `_noaug`(exp5 게이트), `_vg_min0.5`
- `pt_smallreg_minilm_baseline` — 학생 solo (r_mean 63.4 @ ep20)
- `pt_smallreg_minilm_lm_distill` — exp8 (CIDEr +2.5%)
- `pt_smallreg_minilm_itc_distill_tau0.022_droptau` / `_keeptau2` — exp 6.1/6.2
- `pt_smallreg_minilm_itc_distill_armT_t0.022` / `_armS_t0.05` — Phase 1
- 기타: `caption_zeroshot`, `official_pretrain_*`, `teacher_val_loss`, `Caption_coco`, `mps_log`

## 4. 문서/인프라 자산

- **설계 스펙 7건 + 실행 플랜 7건**: `docs/superpowers/{specs,plans}/` —
  retrieval-val(06-21), itc-distill offline(06-27), online-teacher(06-29), lm-distill(07-02),
  caption-zeroshot(07-03), caption-metric-eval(07-13), teacher-target-mix(07-14)
- **critical_bugfix/ 2건**: 07-07 (×τ² 무력화 + τ 실측 선정), 07-14 (KD grad 압도 + margin×scale 불변 진단)
- **모니터링 툴킷**: `claude_skills/monitoring/` — TB collapse 감시 스크립트(check_pretrain_collapse.py + 테스트),
  런별 watch 셸(minilm/lm_distill/logitscale_nodecay/vg_crop), ntfy push(phase1), 3h cron
- 변경량 상위: models/ (+4231), data/ (+2706), configs/ (75회 터치), pretrain.py (30회), distillation/ (31회), tests/ (+877)

## 5. 핵심 기술 서사 (문서의 뼈대 후보)

1. **BLIP 축소**: base → DINOv3-small + MiniLM 학생 구축
2. **수렴 위기와 해결**: temperature collapse → 4-ablation → logit_scale reparam + no-decay
3. **평가 인프라 선행 구축**: retrieval val → caption zero-shot → in-loop caption 메트릭 (측정 없이는 실험 없음)
4. **증류 시행착오의 나선**: offline cache → online teacher → ×τ² 버그 → τ 실측 → grad 압도 진단 →
   (현재) λ 보정 arm vs 타깃 혼합(exp9) 두 갈래
5. **성과**: LM 증류 CIDEr +2.5% 확정; ITC 증류는 ep1 +18 신호 증명 후 크기 관리 단계
