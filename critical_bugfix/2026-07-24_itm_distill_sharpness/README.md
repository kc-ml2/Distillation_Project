# ITM 증류가 retrieval을 뒤집은 근본원인 + sharpness 실측 + 스킴 A/B

**날짜:** 2026-07-24
**대상:** exp10 (ITM distillation, arm A/B/C) · ViT-small_reg + MiniLM · coco_vg · 티처 BLIP-large(`model_large.pth`)
**결론:** arm C(ITM soft target-mix, W=0.4/T=1)에서 `val_retrieval_itm`이 `itc` 아래로 처박히는 건 **코드 버그가 아니라, 오보정된 티처 ITM을 확률-믹싱으로 증류해 헤드의 match-vs-hardneg separation을 붕괴시킨 것.** 실측으로 확증. 해결책으로 스킴 A(티처만 temper한 soft-target, config-only)·스킴 B(2텀 Hinton KD, 새 손실)를 설계.

---

## 1. 증상

- exp10.C(itm_teachneg_soft) 학습에서 `val_retrieval_itc`는 정상 상승(모든 런 중 최고), **그러나 `val_retrieval_itm`이 itc보다 계속 아래**.
- 정상 BLIP에선 ITM rerank가 ITC보다 **+6~11 좋아야** 함(다른 모든 로컬 런에서 확인).

**전 로컬 런의 itm − itc 갭 (최종):**

| 런 | itc | itm | **itm − itc** |
|---|---|---|---|
| exp7 student baseline | 63.4 | 70.9 | **+7.5** |
| exp8 lm_distill | 64.4 | 71.6 | **+7.2** |
| exp9.2 ttm_queue | 64.5 | 71.1 | **+6.6** |
| exp6 itc_distill(various) | 58~63 | 68~70 | **+6~+11** |
| **exp10.C itm_teachneg_soft** | 57.5 | 55.7 | **−1.2** ⚠️ |

→ **ITM 증류 런에서만** rerank가 도움이 아니라 해악. 유일 이상치.

---

## 2. 오진 배제 (가설 반증)

1. **"티처가 평가에 개입"(H1/H2) — 반증.** `RetrievalValRunner` + `eval_validation_tool.evaluate_retrieval_itc/itm` 전 경로가 dev↔branch **byte-동일**이며 teacher 참조 **0개**. 평가 러너는 teacher 인자를 받지도 않음. → 좋은 itc는 착시 아님, 나쁜 itm은 티처가 어려운 문제를 주입한 게 아님.
2. **"허우적" 겉보기 — 측정주기 착시.** itc는 1000 step마다(164점), itm은 에폭당 1번(8점). 초반 TB에서 itm이 저점에 점 몇 개만 찍혀 바닥처럼 보임.
3. **라벨 반전 — 없음.** 3B `[pos B | neg-img B | neg-txt B]`, `itm_labels=[1]*B+[0]*2B`, 평가 `itm_head[:,1]`=match. 학습↔평가 인덱스 일치.
4. **토크나이저 미스매치 — 없음.** 학생·티처 둘 다 `init_tokenizer()`(동일 bert-base-uncased 어휘+enc_token). 학생 MiniLM은 `resize_token_embeddings`로 정합. `itm_soft`가 학생 ID를 티처에 먹여도 정상 → teacher_soft는 정당한 값.

---

## 3. 메커니즘 — 실측 확증 (`itm_sharpness_probe.py`)

COCO val에서 쿼리별 ITC top-k(하드네거)에 각 모델의 ITM 헤드를 돌려 **로짓 gap `z1−z0`** 을 정답 vs distractor로 분리 측정. epoch-matched(ep3), 티처는 full.

| 모델 | match_gap p50 | distractor_gap p50 | **separation(mean)** |
|---|---|---|---|
| baseline (하드, ep3) | 4.52 | 3.41 | **1.15** |
| lm_distill (하드, ep3) | 4.38 | 3.23 | **1.16** |
| **arm C (soft, ep3)** | 4.45 | **3.83** ↑ | **0.64** ← 최악 |
| **teacher (BLIP-large)** | **7.80** | 4.50 | **3.40** ← 최고 |

**읽기:**
- arm C의 **match는 안 죽음**(4.45 ≈ baseline). 손상은 **distractor를 더 높게(3.83 > 3.41) 매겨** — soft 타깃이 "하드네거도 좀 match다"라고 가르쳐 **거부를 약화** → separation **1.15 → 0.64 (~45%↓)**.
- **티처는 훌륭한 랭커**(separation 3.4, match를 7.8로 압도적으로 밀어올림). 그러나 **distractor gap +4.5 = σ=0.989** — 즉 **하드네거(진짜 non-pair)를 99% match로 오판**. 티처는 *랭킹*은 잘하지만 이진 판정 *보정*은 나쁨.

**→ 근본원인:** rerank는 헤드의 절대 로짓(z1)에 지배됨. 오보정된 티처를 확률-믹싱으로 증류하면 하드네거 타깃이 올라가(`0.6·0 + 0.4·0.989 = 0.396`) 헤드의 거부가 약화 → separation 붕괴 → itm < itc. **코드 버그 아님, 목적함수↔지표(+티처 보정) 상충.**

---

## 4. 이론 정리 (오늘 도출한 것)

### 4.1 로짓은 확률이 아니라 log-odds
`softmax([z0,z1])[1]=p ⟺ z1−z0 = logit(p)`. p=0.99→gap 4.6, p=1(하드라벨)→+∞. 그래서 하드라벨 헤드는 로짓이 ±6~10까지 커짐(멈춤 없이 sharpen). soft 타깃은 gap을 `logit(target_p)`에 **유한하게 캡**.

### 4.2 확률 믹싱 vs 2텀 KD — T=1에선 동일
- target-mix: `∇z = σ(z) − [(1−W)y + W·p_t]`
- 2텀 KD(T=1): `∇z = (1−α)(σ(z)−y) + α(σ(z)−p_t) = σ(z) − [(1−α)y + α·p_t]`
- **W=α면 두 gradient가 모든 점에서 동일 → arm C = "2텀 KD의 T=1 특수경우".** 그냥 손실을 쪼개는 걸론 안 바뀜. **레버는 온도 T.**

### 4.3 온도 T의 진짜 역할 = gradient 조건화 (정보/탈포화 아님)
- separation은 softmax 확률에 온전히 인코딩됨(σ(7.8) vs σ(4.5)의 차이가 곧 3.3 로짓 separation). "포화라서 정보가 사라진다"는 **틀린 표현**.
- 진짜 문제: T=1에서 match 타깃(0.9996)이 σ 꼬리라 **z를 올려도 확률이 안 변함 → gradient≈0 → 학생이 큰 갭을 못 올리고 정체**(arm C가 separation 못 키운 이유).
- temper(÷T)는 작동점을 중앙(기울기 max)으로 옮겨 **gradient를 되살림**. "작은 갭→약한 gradient"는 오해 — **약한 gradient는 포화 꼬리 때문**이지 갭 크기 때문이 아님.

### 4.4 두 스킴의 균형점
- **스킴 A (티처만 temper, 학생 plain CE):** 균형 `z = u/T`. **갭이 T배 줄어 통제 가능.** T=2 → match 3.9/distractor 2.25 → **separation ~1.65**.
- **스킴 B (대칭 Hinton, 둘 다 /T + ×T²):** 균형 `z = u`(풀 티처 갭 3.4). 고온 극한 = 로짓 L2 매칭. 용량 한계·easy 예제 로짓 낭비가 부담. sweet spot T≈2~4.

### 4.5 왜 무작정 고온이 안 좋나
- 고온 Hinton gradient → `(z−u)/4` (로짓 오차 비례, **작은 차이=작은 gradient**, `×T²`가 폭발 방지). "조금만 달라도 강한 gradient"는 아님.
- 저온: 확률 매칭 → 확신-일치하는 easy 예제 자동 무시, **경계에 집중.**
- 고온: 로짓 매칭 → **easy 예제 극단 로짓까지 학생이 재현하도록 강요 → 작은 학생 용량 낭비.** 그래서 T는 단조롭게 좋지 않고 최적점 존재.

---

## 5. 해결책 — 스킴 A / B

핵심 교훈: **오보정된 티처 ITM을 이진-거부 헤드에 어떤 방식으로든 그대로 넣으면 거부 보정이 망가진다.** 목표는 *separation(랭킹)*을 통제해서 올리는 것.

### 스킴 A — 티처만 temper한 soft-target KD  ★config-only, 새 코드 0
- `itm_target_mix_loss`를 **W=1.0, temp=2.0**로 사용 → target = `softmax(teacher_logits/2)`, 학생 plain CE → 균형 `z = u/2`.
- 이유: arm C의 병(하드라벨↔티처 정면충돌 = target-mix)을 없애고, 통제된 modest 갭(sep~1.65)을 **도달 쉬운 타깃**으로 준다.
- config: `distill.itm_target_mix: {enabled: true, neg_source: teacher, soft_weight: 1.0, temp: 2.0, schedule: constant}` — `validate_itm_mix_config`가 W∈[0,1], temp 무제약이라 그대로 통과.

### 스킴 B — 2텀 Hinton KD  (새 손실)
```
loss_itm = (1−α)·CE(z, hard_label) + α·T²·KL( σ(u/T) ‖ σ(z/T) )
```
- 하드 CE(거부 앵커, 안 꺼지는 sharpening) + tempered 티처 KL(랭킹 지식). 균형 `z→u`(풀 티처 sep), gradient 건강.
- `distillation/losses.py`에 `itm_hinton_kd_loss` 신규 + config `variant: 'hinton_kd'` 분기 + `validate` 확장 + 단위 테스트(TDD).

### 실험 계획
1. **A 먼저**(config만) → 몇 에폭 → `itm_sharpness_probe.py`로 separation + val_retrieval_itm을 arm B(하드)·arm C(사망)와 비교.
2. **B 추가** → 동일 비교. A(sep~1.65 통제) vs B(sep→3.4 시도) 중 작은 학생이 실제로 뭘 realize하는지가 판정.
3. 게이트: **itm − itc > 0 회복**(정상 rerank) + itc 유지.

---

## 6. 파일
- `itm_sharpness_probe.py` — sharpness 실측 스크립트(학생 계열 + 티처, gap 분포).
- `itm_sharpness_results.jsonl` — ep3 baseline/lm/armC + 티처 실측 결과.
- 관련 아티팩트(개념 정리): claude.ai artifact "ITM 증류 조사 정리".

## 7. 한 줄 교훈
**티처 ITM = 좋은 랭커, 나쁜 이진-보정.** 확률-믹싱(arm C, T=1)은 이 티처를 증류하는 최악의 방법(오보정 수입 + gradient 포화). 하드라벨을 타깃에 녹이지 말고 — soft를 통제된 온도로 별도 전달(스킴 A) 또는 하드 CE를 별도 항으로 유지(스킴 B). 궁극적으로 랭킹 지식은 ITC 경로가 가장 자연스럽다(exp9).
