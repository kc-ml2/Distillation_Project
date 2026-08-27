# 모델 크기·파라미터 회계 (실측)

**작성:** 2026-07-30 | **재생성:** `python docs/model_specs/count_specs.py` (GPU 불필요, `init_backbone_weights=False`로 다운로드 생략)

모든 숫자는 `models/blip_pretrain.py:BLIP_Pretrain`를 실제로 인스턴스화해 센 값이다.
**tied 파라미터는 `data_ptr` 로 중복 제거**했으므로 "고유 파라미터" 기준이다
(`tie_encoder_decoder_weights`가 인코더↔디코더의 embedding/cross-attention/FFN을 공유하므로,
단순 `sum(p.numel())`은 이들을 2번 세서 과대계상된다).

---

## 1. 세 가지 구성 요약

| | **teacher: BLIP-large** | **BLIP-base (원본)** | **student: DINOv3-S + MiniLM** |
|---|---|---|---|
| `vit` | `large` (ViT-L/16) | `base` (ViT-B/16, DeiT init) | `small_reg` (DINOv3 ViT-S/16 + 4 reg) |
| `my_bert_size` | `base` (bert-base-uncased) | `base` | `minilm` (MiniLM-L12-H384) |
| `image_size` | 224 | 224 | 224 |
| vision_width | 1024 | 768 | **384** |
| text_width | 768 | 768 | **384** |
| embed_dim (ITC) | 256 | 256 | 256 |
| text layers / heads | 12 / 12 | 12 / 12 | 12 / 12 |
| text intermediate | 3072 | 3072 | **1536** |
| vocab_size | 30524 | 30524 | 30524 |
| cross-attn K/V in_features | 1024 | 768 | **384** |
| 이미지 토큰 수 (모델 출력) | 197 | 197 | 197 |
| **online 고유 파라미터** | **474,729,022** | **252,441,918** | **69,387,198** |
| momentum+queue 포함 전체 | 920,467,007 | 475,892,799 | 131,488,959 |
| buffer (queue 2×256×57600 등) | 29,492,737 | 29,492,737 | 29,492,769 |

**압축비 (online 기준):** student / BLIP-large = **14.6%** (6.84× 감소), student / BLIP-base = 27.5%.
- vision만: 21.59M / 303.30M = **7.1%** (14.05× 감소)
- text encoder만: 40.32M / 141.98M = **28.4%** (3.52× 감소)

> **주의:** `online 고유`는 추론(retrieval/caption)에 실제로 쓰이는 파라미터.
> `전체`는 pretrain 시 메모리에 상주하는 값(momentum visual/text encoder + proj 사본 포함).
> queue는 파라미터가 아니라 buffer이며 `queue_size=57600 × embed_dim=256 × 2개 = 29,491,200`.

---

## 2. 모듈별 분해

### 2.1 student (DINOv3 ViT-S/16 + MiniLM-L12-H384) — online 69,387,198

| 모듈 | 파라미터 | 비고 |
|---|---|---|
| `visual_encoder` (DINOv3 ViT-S/16 + 4 reg token) | 21,586,944 | timm `vit_small_patch16_dinov3.lvd1689m` |
| `text_encoder` (fused, 12층) | **40,317,696** | 아래 세분화 |
| ├ `embeddings` (30524×384 + pos 512×384 + LN) | 11,918,592 | |
| ├ self-attention × 12 (592,128/층) | 7,105,536 | MiniLM 사전학습 로드 |
| ├ **cross-attention × 12 (592,128/층)** | **7,105,536** | ⚠️ **전부 랜덤 초기화** |
| └ FFN × 12 (1,182,336/층) | 14,188,032 | MiniLM 사전학습 로드 |
| `text_decoder` 추가분 (self-attn 7,105,536 + LM head transform 148,608 + bias 30,524) | 7,284,668 | 나머지는 인코더와 tie |
| `itm_head` = `Linear(384, 2)` | **770** | 랜덤 초기화 |
| `vision_proj` `Linear(384,256)` + `text_proj` `Linear(384,256)` | 98,560 + 98,560 | 랜덤 초기화 |
| `logit_scale` | 1 | `log(1/0.07)` 초기화, weight-decay 제외 |
| 검산 | 21,586,944 + 40,317,696 + 7,284,668 + 770 + 197,120 + 1 = **69,387,199** | (±1: logit_scale) |

### 2.2 teacher BLIP-large — online 474,729,022

| 모듈 | 파라미터 |
|---|---|
| `visual_encoder` ViT-L/16 (24층, 1024, 16 heads) | 303,301,632 |
| `text_encoder` (bert-base + cross-attn 12층) | 141,977,088 |
| ├ embeddings (30524×768 + pos) | 23,837,184 |
| ├ self-attn × 12 (2,363,904/층) | 28,366,848 |
| ├ **cross-attn × 12 (2,757,120/층 — K/V가 1024→768)** | 33,085,440 |
| └ FFN × 12 (4,723,968/층) | 56,687,616 |
| `text_decoder` 추가분 (self-attn 28,366,848 + LM transform 592,128 + bias 30,524) | 28,989,500 |
| `itm_head` = `Linear(768, 2)` | 1,538 |
| `vision_proj` `Linear(1024,256)` + `text_proj` `Linear(768,256)` | 262,400 + 196,864 |

체크포인트: `output/official_pretrain_checkpoint/model_large.pth`
(공식 Salesforce BLIP-large pretrain, 129M 이미지. 1,560 텐서 / 1,087,042,940 파라미터 — momentum + queue 포함값)

### 2.3 BLIP-base — online 252,441,918

`visual_encoder` ViT-B/16 = 85,798,656 / `text_encoder` = 137,258,496 (cross-attn 총 28,366,848) /
`text_decoder` 추가분 = 28,989,500 / `itm_head` = 1,538 / proj = 196,864 × 2.

---

## 3. 사전학습 로드 vs 랜덤 초기화 (⚠️ MiniLM 특유의 불리함)

`from_pretrained(..., output_loading_info=True)`의 `missing_keys` 실측:

| 로드 대상 | 랜덤 초기화되는 텐서 | 버려지는 체크포인트 텐서 |
|---|---|---|
| MiniLM → `BertModel` (ITM 인코더) | **120개, 전부 crossattention** (7,105,536) | `pooler.*`, `token_type_embeddings` |
| MiniLM → `BertLMHeadModel` (캡션 디코더) | 120 crossattention + **`cls.predictions.transform.{dense,LayerNorm}` + `cls.predictions.bias`** (148,608 + 30,524) | 동일 |
| bert-base → `BertLMHeadModel` (대조군) | **120 crossattention만** — MLM 헤드는 전부 사전학습값 로드 | `cls.seq_relationship.*`, `token_type_embeddings` |

- fused encoder의 **17.6% (7,105,536 / 40,317,696)** 가 랜덤 시작. bert-base 학생도 동일 비율(20.7%)이라 이건 BLIP 설계상 불가피.
- **MiniLM만 랜덤인 부분:** LM head의 `transform`(384→384 dense + LayerNorm)과 output bias가 사전학습값을 못 받는다. bert-base는 받는다.
  MiniLM 체크포인트에 `cls.*` 키가 아예 없기 때문(→ `configs/corpus_and_pretraining.md` §3).

  **✅ 실측 (2026-07-30) — 이건 유의미한 손해가 아니다.** 같은 배치(COCO val 32장)·같은 몸통에서 head 초기화만 바꿔 초기 `loss_lm` 측정:

  | head 초기화 | MiniLM 학생 | bert-base 학생 |
  |---|---|---|
  | 체크포인트/HF 기본 그대로 | 10.39 ~ 11.32 (시드 의존) | 10.97 (사전학습 transform) |
  | `dense = I` (항등 초기화) | **11.90 — 랜덤보다 나쁘다** | 10.96 (차이 0.005) |
  | `dense = I` + GELU 제거 | 11.51 | 10.74 |
  | `dense = 0` (h를 무시) | **10.33 = ln(30524) 정확히** | **8.63** |

  읽는 법: **두 모델 다 초기 상태에서는 `h`를 아예 무시하는 편이 낫다**(10.33 < 10.39, 8.63 < 10.97).
  즉 `transform`의 초기값은 병목이 아니고, **항등 초기화는 오히려 해롭다** —
  항등은 "`h`의 좌표계가 출력 임베딩 `E`의 행과 정렬돼 있다"를 가정하는데,
  MiniLM은 MLM을 학습한 적이 없어 그 정렬이 애초에 없다(bert-base조차 causal+cross-attn 하의 `h`에는 정렬이 깨져 있다).
  ※ 경로에 residual/skip이 없으므로(`models/med.py:511-514`) 랜덤 `dense`는 신호 전부를 랜덤 행렬로 통과시킨다.
    skip이 있었다면 HF 기본 랜덤 초기화가 무해했을 것이다.

- **✅ bert-base가 실제로 공짜로 얻는 것은 `transform`이 아니라 output bias다.** 위 표의 `dense=0` 행 차이(8.63 vs 10.33 = **1.70 nats**)가 전부 bias/LayerNorm β에 인코딩된 unigram 빈도다.
  → **더 나은 개입: `cls.predictions.bias`를 학습 캡션의 log-unigram으로 초기화** (bert-base 흉내내기보다 낫다):

  | | 초기 `loss_lm` |
  |---|---|
  | 현재 (랜덤 transform, bias=0) | 10.39 |
  | 랜덤 transform + **bias = log unigram** | **6.87 (−3.5 nats)** |
  | `dense=0` + bias = log unigram | 6.22 |
  | (참고) COCO 캡션 unigram 엔트로피 | 5.33 |

  ⚠️ 단, bias는 30,524개 파라미터에 직접 gradient가 걸려 수백 스텝에 수렴한다.
  **초기 수렴 속도 이득은 확실하나 최종 CIDEr을 바꾸는지는 미검증** — A/B 필요.
  (측정 한계: 배치 1개(32장), 랜덤 transform은 시드에 따라 10.39~11.32로 흔들린다.)
- `token_type_embeddings`가 버려지는 건 정상: BLIP `models/med.py:52-63`의 `BertEmbeddings`는 segment 임베딩을 갖지 않는다
  (segment 대신 `[ENC]`/`[DEC]` 토큰으로 모드를 구분).

---

## 4. `itm_head`의 유효 자유도 = 385 (명목 770)

$z_1 = w_1^\top h + b_1,\ z_0 = w_0^\top h + b_0$, softmax CE는 shift-invariant:

$$p(\text{match}) = \sigma\big(\underbrace{(w_1-w_0)}_{w_d}{}^\top h + \underbrace{(b_1-b_0)}_{b_d}\big)$$

- 손실이 구속하는 성분: $w_d \in \mathbb{R}^{384},\ b_d$ → **385개**
- 손실이 구속하지 않는 성분: $w_s = w_1+w_0,\ b_s = b_1+b_0$ → 385개.
  `weight_decay=0.05`만 작용(`pretrain.py:363-374`, `logit_scale`만 제외)하므로 **느리게 0으로 감쇠**.
- eval은 $z_1 = \tfrac{1}{2}(w_d + w_s)^\top h + \tfrac{1}{2}(b_d+b_s)$ 를 점수로 쓴다
  (`eval_validation_tool.py:163`) → $w_s$가 0에 가까울수록 `z1` 랭킹 = `gap` 랭킹.

**체크포인트 실측** ($\|w_s\|/\|w_d\|$ 가 작을수록 "z1 = gap/2" 에 가깝다):

| 모델 | $\|w_0\|$ | $\|w_1\|$ | $\cos(w_0,w_1)$ | $\|w_d\|$ | $\|w_s\|$ | $\|w_s\|/\|w_d\|$ |
|---|---|---|---|---|---|---|
| **teacher BLIP-large** | 1.6674 | 1.6732 | **−0.9971** | **3.3382** | 0.1269 | **0.038** |
| student baseline ep0 | 0.5998 | 0.5842 | −0.0710 | 0.8665 | 0.8070 | 0.931 |
| student baseline ep3 | 0.6474 | 0.6229 | −0.2648 | 1.0103 | 0.7704 | 0.763 |
| student baseline ep15 | 0.7241 | 0.6959 | −0.5079 | 1.2331 | 0.7048 | 0.572 |
| student baseline ep19 | 0.7327 | 0.7065 | −0.5337 | 1.2604 | 0.6953 | 0.552 |
| schemeA ep15 | 0.5406 | 0.5237 | −0.1238 | 0.7979 | 0.7046 | 0.883 |
| schemeA ep19 | 0.5395 | 0.5253 | −0.1483 | 0.8069 | 0.6949 | 0.861 |

**읽는 법**
- 티처는 완전 수렴해 $w_1 \approx -w_0$ (cos −0.997) → `z1`은 `gap/2`와 사실상 동일. offset 성분이 물리적으로 없다.
- 학생은 20 에폭으로 부족해 $w_s$가 아직 $w_d$의 55~86% 크기로 남아 있다.
  → `z1` 랭킹과 `gap` 랭킹이 **원리적으로는** 갈릴 수 있다.
  단, 2026-07-30 실측에서 **실제로는 거의 갈리지 않는다**(쿼리 내 $h$의 변동이 $w_s$와 거의 직교).
  Kendall τ(z1, gap) = 0.945~0.998, offset 산포/gap 산포 = 0.15~4.6%.
  → `critical_bugfix/2026-07-30_itm_logit_offset_and_head_geometry/`
- **$\|w_d\|$는 학습 진행에 따라 커진다** — 하드라벨은 키우고(0.867→1.260), soft 타깃은 그 압력을 끄고(0.828→0.807),
  티처는 3.338까지 자랐다. `separation`(match_gap − distractor_gap) $= w_d^\top(h_{\rm match}-h_{\rm distr})$ 은
  **$\|w_d\|$에 비례하므로 모델 간 원값 비교가 성립하지 않는다.**
- ⚠️ **정정 (2026-07-30):** 위 문제의 해법으로 이 문서가 처음 제안했던 "$\|w_d\|$로 나누기"는 **틀렸다.**
  그건 헤드 스케일만 제거하고 **특징 스케일($\|h\|$: 768차원 티처 vs 384차원 학생, LayerNorm gain 차이)** 을 남기는 부분 정규화라,
  "학생이 티처보다 잘 분리한다"는 잘못된 결론을 냈다.
  올바른 스케일 무관 지표는 **within-query $d'$** $=\dfrac{\overline{\rm gap}_{\rm 정답}-\overline{\rm gap}_{\rm distractor}}{{\rm std}({\rm gap}_{\rm distractor})}$ 이며,
  분모가 gap과 같은 단위라 헤드·특징 스케일이 함께 소거된다. 실측:

  | 지표 | teacher | baseline@ep15 | schemeA@ep15 | 순서 |
  |---|---|---|---|---|
  | separation 원값 | 3.399 | 1.670 | 0.620 | T > B > A |
  | ~~sep / ‖w_d‖~~ (폐기) | ~~1.02~~ | ~~1.35~~ | ~~0.78~~ | ~~B > T > A~~ ← **오류** |
  | **within-query d′ (i2t)** | **2.195** | **1.492** | **1.083** | T > B > A |
  | **within-query d′ (t2i)** | **3.055** | **2.354** | **1.621** | T > B > A |
  | R@1 (z1+cos, i2t) | 72.4 | 58.2 | 50.0 | T > B > A |

  → 원 서사("티처가 더 sharp")는 **유지된다.** 다만 배율은 원값 비교(3.399/1.670 = 2.04)보다 작다(d′ 2.195/1.492 = **1.47**).

관련 조사: `critical_bugfix/2026-07-30_itm_logit_offset_and_head_geometry/` (헤드 기하 + 로짓 원자료),
`critical_bugfix/2026-07-24_itm_distill_sharpness/`, `critical_bugfix/2026-07-27_itm_eval_ranking_and_structure/`
