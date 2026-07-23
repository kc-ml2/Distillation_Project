# 경량 Vision-Language 모델 지식 증류: 기술 보고서

> **프로젝트:** BLIP 기반 경량 VLM 사전학습 및 BLIP-large 티처 지식 증류
> **기간:** 2026-05-12 ~ 2026-07-15 (진행 중) · **작성일:** 2026-07-16
> **베이스:** Salesforce BLIP (공식 구현 fork)
> 시간순 개발 맥락은 자매 문서 [development-timeline.md](development-timeline.md) 참조.

---

## 요약 (Executive Summary)

BLIP-base 대비 **30% 파라미터(73.8M)의 경량 학생 모델**(DINOv3 ViT-S + MiniLM)을 구축하고,
frozen BLIP-large(469.4M) 티처로부터 지식 증류를 수행했다. 주요 성과:

1. **LM logit 증류로 zero-shot 캡션 성능 확정 개선** — 동일 학생 solo 대비
   **CIDEr +2.5% (1.076→1.103), SPICE +2.0% (0.203→0.207)**, 전 에폭에서 일관 우위.
   학생 모델은 base(247.2M)의 30% 파라미터로 **CIDEr 96% 달성**.
2. **학습 안정성 문제(temperature collapse)의 근본 해결** — 원인을 gradient 증폭 구조로
   특정하고 CLIP식 logit_scale 재파라미터화 + weight-decay 제외로 해소. 통제 ablation으로 검증.
3. **ITC(contrastive) 증류의 실패 원인 정량 규명** — Hinton ×T² 보정의 적용 한계를
   gradient 실측(crossover)으로 입증하고, KD-ITC gradient 불균형(6.7×)과 기하 압축-상쇄
   (margin×scale 불변)를 진단. 이를 반영한 2차 실험(×τ 보정 + λ 정합, target-mixing) 진행 중.
4. **재현 가능한 평가 인프라** — 학습 중 상시 retrieval/caption 지표 평가, scale-오염 없는
   지표 선택 원칙(rank 기반), 체크포인트 sweep, 자동 collapse 감시 체계.

---

## 1. 문제 정의와 목표

| 항목 | 내용 |
|---|---|
| 목표 | 경량 VLM이 대형 티처의 성능에 근접하도록 사전학습 단계에서 지식 증류 |
| 학생 | DINOv3 ViT-S (reg tokens, 21.6M) + MiniLM 계열 텍스트 인코더/디코더 (52.3M) — 캡션 모델 기준 73.8M |
| 티처 | 공식 BLIP-large pretrain 체크포인트 (469.4M), frozen |
| 데이터 | COCO(566K 캡션) + Visual Genome region caption(5.4M) — VG가 mixture의 ~90% |
| 평가 | COCO 5k retrieval (R@1/5/10 → r_mean), COCO Karpathy test zero-shot 캡션 (BLEU/METEOR/ROUGE/CIDEr/SPICE) |
| 성공 기준 | 증류 학생 > solo 학생 (필수), base 모델(66.2+) 근접 (목표) |

**학습 레시피 (전 실험 공통, 1-변수 통제):** batch 40×4GPU (eff 160), lr 1e-5 (cosine),
warmup 37,000 steps, 20 epochs, image 224², queue 57,600, AMP(bf16), weight decay 0.05.

## 2. 시스템 아키텍처

### 2.1 학생 모델

BLIP의 3-loss 구조(ITC/ITM/LM)를 유지하되 양쪽 백본을 교체 가능하게 재설계:

- **비전:** timm DINOv3 wrapper — register token 제어, `global_pool=''` 출력 차원 정합,
  DINOv3 체크포인트의 pos_embed interpolation 가드. `vit: small_reg` 옵션.
- **텍스트:** `my_bert_size` 옵션으로 BERT-base/medium/MiniLM 선택 — encoder/decoder config 쌍
  (`configs/{bert,med}_{minilm,medium}_config.json`), decoder `encoder_width=vision_width` 연결.
- 캡션 디코더도 동일 옵션으로 선택되어 pretrain 모델에서 zero-shot 생성 가능 (`BLIP_Pretrain.generate()`).

### 2.2 모델 크기와 캡션 성능 (zero-shot, COCO Karpathy test)

| 모델 | 파라미터 | CIDEr | SPICE |
|---|---|---|---|
| BLIP-large (티처) | 469.4M | 1.236 | 0.227 |
| BLIP-base (official) | 247.2M | 1.150 | 0.219 |
| base 재학습 (amp) | 247.2M | 1.120 | 0.210 |
| base 재학습 (nodecay) | 247.2M | 1.113 | 0.211 |
| **student + LM 증류** | **73.8M** | **1.103** | **0.207** |
| student solo | 73.8M | 1.076 | 0.203 |

## 3. 학습 안정화: temperature collapse

### 3.1 증상과 원인 분석

baseline 런(`exp=baseline_lrlow_amp`)이 epoch 0.5~4 안정 plateau(logit_scale 38~58) 후
epoch ~4.3부터 폭주, epoch 6에 클램프 상한(scale ~1000 = temp 0.001) 고정, `loss_val/ita` 동반 악화.

- weight decay의 temp 드리프트 기여를 계산하면 ~1.5%/epoch — 관측된 30~40% 변동을 설명 불가.
- 원본 BLIP은 `sim / temp` (temp 학습 파라미터, [0.001, 0.5] 클램프) 구조 — temp가 바닥에
  접근할수록 **1/temp² gradient 증폭**이 걸리는 양성 피드백 구조로 판단.
- 독립 요인으로 **VG 데이터 버그** 발견: region caption이 bbox crop이 아닌 전체 이미지를 사용
  (mixture의 90%가 오염). bbox crop + bounds-clamp로 수정.

### 3.2 통제 ablation과 해결

| exp | 조작 | 목적 |
|---|---|---|
| 1.vg_crop | VG bbox fix만 | 데이터 요인 분리 |
| 2.no_decay_temp | + temp를 decay 제외 | decay 단독 가설 검증 |
| 3 (생략) | reparam만 | 수학적 분석으로 대체, GPU 절약 |
| 4.logit_scale_no_decay | **CLIP식 reparam + decay 제외** | 결합 해법 |

exp4: `logit_scale = nn.Parameter(log(1/0.07))`, `sim × exp(logit_scale)`, **log-space 클램프로
유효 범위 [2, 1000] 동일 유지** (파라미터화만 변경, 동역학 범위는 불변 — 교란 변수 차단).
결과 exp4 구성이 표준 레시피로 정착, base r_mean 66.20 (best 66.83).

**반증 사례 기록:** exp5-lrup은 logit_scale 110.9에 도달하고도 retrieval 최고(67.97)
→ "높은 scale 자체가 병리"라는 가설 기각. scale은 증상이며, 문제는 저-temp 구간의 gradient 구조.

부속 안전장치: 학습 중 collapse 자동 정지 트리거(logit_scale>700 지속 10,000스텝) +
TB 이벤트 파일 직독 방식의 외부 감시(cron + push 알림).

## 4. 평가 인프라

실험 진행보다 측정 체계를 먼저 구축하는 원칙으로 3층 평가를 마련했다.

| 층 | 도구 | 케이던스 | 근거 |
|---|---|---|---|
| retrieval | `RetrievalValRunner` (COCO 5k, ITC-only + ITM-rerank 2티어) | ITC 1000스텝 / ITM 에폭 2회 | ITM rerank는 비싸 저빈도 |
| caption (in-loop) | `CaptionValRunner` (DDP 샤딩 생성→gather/dedup→coco_caption_eval) | 경량지표 에폭 2회, SPICE 에폭 1회 | SPICE 62s/회 (32코어 캡) |
| caption (offline) | `train_caption.py --evaluate` config-driven + SPICE 오프라인 재계산 | 체크포인트 sweep | 에폭 궤적 A/B |

**지표 선택 원칙 (Phase 0 진단에서 확립):** retrieval은 rank 기반이라 유사도의 단조 변환
(logit_scale 곱 포함)에 불변. 반면 val ITC loss는 softmax(sim×scale) CE라 **scale이 다른 두 런 간
비교 불가** — "loss는 내려가는데 retrieval은 나쁘다"는 모순이 아니라 scale 오염.
⇒ **런 간 비교와 모델 선택은 retrieval(및 생성 지표)로만 수행한다.**

공학적 하드닝: rank0 채점 예외의 non-fatal 처리(체크포인트 유실 방지, collective 이후라 DDP-safe),
SPICE 채점의 CPU affinity 제한(MPS 동시 2잡 스래싱 방지, 32C=64C 실측 동속), DDP gather/dedup
정합성 검증(single vs 2-GPU CIDEr 동일치 확인).

## 5. 지식 증류

### 5.1 LM logit 증류 (exp8) — 확정 성과

**방법:** frozen 티처가 매 스텝 동일 증강 배치에 teacher-forced 디코더 로짓 [B, 30, 30524] 생성 →
token-level KL (shift frame 정렬, pad mask, T=2.0, ×T²) → `total + w·loss_lm_kd` (w=1.0).

**결과 (§2.2 표):** CIDEr +2.5%, SPICE +2.0%, **전 평가 에폭(2·5·7·10·13·16·19)에서 distill > solo**.

**심층 분석:**
- **headroom 측정:** 티처의 val LM CE = 3.732 vs 학생 3.760 — 차이 ~0.03 nats.
  티처가 caption-finetune이 아닌 pretrain ckpt이므로 CE 지표로는 증류 효과가 원리적으로
  나타날 수 없음을 사전 규명 → 평가를 downstream 생성 지표로 설계.
- **SPICE 분해:** 전 모델 CIDEr는 ep16 피크 후 소폭 하락, SPICE(의미)는 ep10(ITC val 최저점)에서
  포화. 단 **solo의 SPICE는 ep10 이후 완전 플래토(0.203)인 반면 distill만 계속 상승(→0.207)** —
  증류가 유창성이 아닌 **의미(scene-graph) 내용**을 추가 전달한다는 증거.
- 성공 요인 (사후 가설): (1) teacher-forced에서 티처 top-1 ≈ 정답 토큰이라 CE와 타깃 비충돌,
  (2) T=2 > 1은 Hinton softening regime이라 ×T² 보정이 유효, (3) logit_scale 같은 공유 도피
  노브가 없음 — §5.2의 ITC 증류와 정확히 대비되는 조건.

### 5.2 ITC (contrastive) 증류 — 실패의 정량 규명

**방법:** 학생·티처의 L2-normalized feature로 in-batch B×B cosine 유사도 행렬 →
공통 온도 τ의 softmax 분포 간 양방향 KL (relational KD). offline feature 캐시로 시작했으나
aug 불일치로 online teacher(bf16, no_grad)로 피벗.

#### (a) ×τ² 계수 버그 — Hinton 보정의 적용 한계

초기 런에서 `loss_itc_kd ≈ 0.0034` (ita 7.73의 1/2250). 원인은 loss의 `×τ²` 계수:
Hinton(2015)의 ×T² 보정은 **고온 극한(T ≫ 로짓 차)에서 softmax 선형화로 grad ~ 1/T²가 되는
regime에서만 유도**된다. 본 설정은 τ=0.05 ≪ 1 (sharpening)이라 grad ~ 1/τ regime이고,
×τ²는 순 τ배 = 신호를 죽인다. **gradient probe로 crossover를 직접 실측:**

| τ | 0.022 | 0.05 | 0.1 | 0.15 | 0.3 |
|---|---|---|---|---|---|
| \|g_kd\|·τ (1/τ regime이면 상수) | 0.95 | 1.15 | 0.98 | 0.70 | 0.35 |
| \|g_kd\|·τ² (Hinton regime이면 상수) | 0.021 | 0.057 | 0.098 | **0.105** | **0.104** |

τ≥0.15에서만 Hinton 보정 성립, τ≤0.05는 1/τ regime → **본 설정의 올바른 정규화는 ×τ**.

#### (b) τ 선정 — 티처 유사도 실측 기반

티처를 실제 학습 배치에 forward하여 4,000 pos / 156,000 neg 통계 측정:
- 티처 임베딩이 좁은 콘([0.2, 0.6])에 압축: pos 0.42 / neg 0.31, per-row margin(pos−hardest neg)
  aug ON **0.016** / clean 0.030 — 증강이 positive 유사도만 선택적으로 훼손.
- **argmax=diag 58.7%** — 티처가 옮길 수 있는 top-1 신호의 상한. 오답의 대부분(31~41%p)은
  티처·데이터 고유(캡션 모호성, VG degenerate region), aug 기여 ~6%p.
- τ 그리드: 정보가 살아있는 창은 **0.022~0.07** (τ≥0.1은 유효서포트 34+ ≈ 균등, 정보 소멸).
  τ=0.022: diag질량 0.51/서포트 4.2 (CE-관계정보 경계), τ=0.05: 0.23/19.4 (티처 랭킹 전달).

#### (c) exp 6.1/6.2 결과와 Phase 0 진단

| 런 | 구성 | ep1 r_mean | ep20 r_mean | logit_scale@20 |
|---|---|---|---|---|
| baseline (exp7) | 증류 없음 | 21.9 | **63.4** | 59.1 |
| 6.1 droptau | τ=0.022, ×τ² 제거, λ=1 | **40.2** | 59.5 | 88.1 |
| 6.2 keeptau2 | ×τ² 유지 | 16.0 | 62.7 | 65.7 |

- 6.1의 **ep1 +18.3은 티처 신호의 실효 가치를 증명** — 문제는 신호가 아니라 크기 관리.
- 6.2는 유효 ratio 0.003의 no-op 대조군 (baseline과 −0.7 = 노이즈).
- 6.1의 최종 **−3.9 원인 3종 실측** (`critical_bugfix/2026-07-14`):
  1. **gradient 불균형:** λ=1에서 |g_kd|/|g_ita| = 6.66(ep0) → 2.56(ep19) — KD가 전 구간 압도.
     cos(g_kd, g_ita) 전 조합 음수(−0.06~−0.34): 순 갱신의 ITC 방향 성분 1+6.66×(−0.24) ≈ **−0.6**,
     즉 초반 학습이 주 손실 gradient를 거슬러 진행.
  2. **기하 압축과 상쇄:** KD가 학생 margin을 0.082→0.054로 압축(티처의 압축된 콘 강제),
     ITC가 logit_scale 59→88 인플레로 정확히 상쇄 — **margin×scale 4.84 vs 4.77 (불변)**.
     이 줄다리기에 동반된 순위 재배열이 retrieval을 손상.
  3. **지표 오염:** 6.1의 val ITC loss 하락은 scale 오염 — 개선의 증거가 아님 (§4 원칙의 출처).

### 5.3 진행 중: 2차 실험 (2026-07-14 ~ )

**Phase 1 — ×τ 보정 + λ 크기 정합 (dev `fe23681`, 07-14부터 가동 중):**
- loss에 ×τ 반환 → 두 τ의 |g_kd|가 자동 정합 → **단일 공유 λ=2.5** (목표 ep0 ratio ~0.4,
  6.1의 6.66 대비 16× 약한 압력). 명시적 anneal 없음 — λ 고정만으로 ratio가 학습 중 ~3× 자연 감쇠.
- **Arm T** (τ=0.022, sharp): 순 갱신 ITC 방향 보존 +0.91 / **Arm S** (τ=0.05, soft): +0.85.
  "같은 압력, 다른 sharpness" — τ 1-변수 통제.
- 게이트: ep20 r_mean ≥63.4 (solo 초과 필수), 목표 66+. 미달 시 Phase 1.5 후보 = gradient surgery
  (PCGrad류 충돌 성분 사영 제거, 비용 ~1.4×).

**exp9 — teacher-target-mixing (브랜치 `exp9_teacher_target_mix`, GPU 스모크 대기):**
- 접근 전환: 별도 KD loss(두 gradient의 벡터 합 타협) 대신 **티처를 ITC soft-target에 혼합**하여
  손실을 단일화 — gradient 충돌이 정의상 소멸.
- `target = 0.6·one-hot + 0.4·[γ·teacher_soft + (1−γ)·momentum_soft]`, 양방향 대칭.
- γ 스케줄: 2ep 홀드(γ=1, 초기 불안정 구간 scaffold) → ep12까지 선형 감쇠 → 이후 baseline 동일.
- variant: 9.1 in_batch(B×B, queue열 0패딩) / 9.2 queue(frozen 티처 feature 큐, momentum 큐와
  페어 enqueue로 열 정렬). γ=0 이후 티처 forward 스킵(불변식 assert) — 런당 ~1일 절약.
- 검증: 단위테스트 9/9, OFF 분기 baseline bit-identical, CPU 스모크, 전 브랜치 리뷰 통과.

**부속 실험 — VG min_scale 0.5 (완료, 무승부):** VG는 이미 region-crop된 이미지라 공통
RandomResizedCrop(0.2)이 과하다는 가설 → VG 전용 min_scale 0.5. 결과 exp4 대비 66.34 vs 66.20
(노이즈 범위). 함의: VG 문제의 지렛대는 crop 강도가 아니라 **region 크기/내용 선택**
(VG<32²px가 63%, 티처 co-match 오답 VG 35.8% vs COCO 5.4%) → Phase 2 content-gating 후보의 근거.

## 6. 정량 결과 종합

### 6.1 Retrieval (COCO 5k, ITC r_mean @ ep20)

| 런 | r_mean | 비고 |
|---|---|---|
| base exp5-lrup | 66.74 (best 67.97) | 전체 최고 (레시피 상이) |
| base exp4 (표준 레시피) | 66.20 (best 66.83) | 학생의 목표선 |
| base vg_min0.5 | 66.34 | exp4와 무승부 |
| student + LM 증류 (exp8) | ≈64.2 | |
| **student solo (exp7)** | **63.4** | 증류 실험의 필수 게이트 |
| student + ITC KD 6.2 (no-op) | 62.7 | 대조군 |
| student + ITC KD 6.1 (λ=1) | 59.5 | 실패 사례 — §5.2(c) 진단 |
| Phase 1 armT / armS | (가동 중) | 게이트 ≥63.4, 목표 66+ |

### 6.2 Zero-shot 캡션 (COCO Karpathy test, ep20 매칭 A/B)

§2.2 표 참조. 핵심: **LM 증류 학생(73.8M) CIDEr 1.103 = base official(247.2M)의 96%**,
distill > solo 전 에폭 성립, SPICE 궤적에서 의미 내용 전달 확인.

## 7. 결론 및 향후 과제

**확립된 것:**
1. 경량 학생(30% 파라미터)이 LM logit 증류로 base급 캡션 성능의 96%에 도달 — 증류 효과는
   CE가 아닌 downstream 생성 지표(특히 SPICE 궤적)에서 검증해야 함.
2. contrastive 증류에서 Hinton ×T² 보정은 sharpening regime(τ<1)에 적용 불가 —
   올바른 정규화(×τ)와 gradient norm 정합(λ)을 실측으로 도출하는 방법론 확립.
3. 티처 신호의 가치(ep1 +18)와 한계(top-1 정확도 58.7%, 좁은 유사도 콘)를 분리 정량화.
4. scale-불변 지표만으로 런을 비교하는 평가 원칙.

**진행 중 (go/no-go 게이트 명시):**
- Phase 1 (armT/armS): ep20 r_mean ≥63.4 필수, 66+ 목표. 실패 시 gradient surgery(Phase 1.5).
- exp9 (target-mixing): gradient 충돌의 구조적 제거가 λ 튜닝보다 나은지 검증. C(queue)/D(in_batch) 대조.

**후속 후보 (Phase 2):**
- content-gating KD — VG degenerate region 다운웨이트/제외 (COCO 가중 또는 region-area 기준).
- caption-finetuned 티처로 LM 증류 headroom 확장 (해상도 224/384 trade-off 검토 필요).
- LM-KD 성공 요인의 probe 검증 — cos(∇L_lm_kd, ∇L_lm) 측정 (미실시).

---

## 부록 A — 재현성

| 항목 | 위치 |
|---|---|
| 표준 레시피 config | `configs/pretrain.yaml` (단일 파일 제어, `88ea5d4`) |
| Phase 1 config | `configs/pretrain_itc_distill_{armT,armS}.yaml` (dev `fe23681`) |
| exp9 config | `configs/pretrain_ttm_{inbatch,queue}.yaml` (`exp9_teacher_target_mix`) |
| 학습 명령 | `CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=295xx pretrain.py --config <cfg>` |
| 티처 ckpt | `output/official_pretrain_checkpoint/model_large.pth` (missing `logit_scale`/unexpected `temp` 경고는 정상 — reparam 차이) |
| τ/grad/기하 측정 | `critical_bugfix/2026-07-{07,14}_*/` — 스크립트·캐시(.pt/.npz)·재현 명령 포함 |
| 진단 주의사항 | 폐기된 itc_distill_only 워크트리의 losses.py는 keep-τ² 로컬 수정 상태였음 — Phase 1 전 원복 완료 확인됨 |

## 부록 B — 실험 명명 체계

`exp` 태그 (config) ↔ `output/` 런 디렉터리 ↔ TB run name(`__exp=...__kd=...` 서픽스) 3중 대응.
주요 계보: 1.vg_crop / 2.no_decay_temp / 4.logit_scale_no_decay / 5.noaug·lrup / 6.itc_distill(6.1, 6.2) /
7.student_baseline / 8.lm_distill / Phase 1(armT/armS) / 9.teacher_target_mix(9.1, 9.2).
