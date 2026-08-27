# 2026-07-14 — ITC-KD 그래디언트 균형·기하 압축 진단 (Phase 0)

**상위 맥락:** [`../2026-07-07_teacher_sim_and_tau/`](../2026-07-07_teacher_sim_and_tau/README.md) — τ=0.022 선정 근거와
"×τ² 제거 시 **λ로 grad-norm 정합 필수**"라는 Open item. 본 문서는 그 Open item이 이행되지 않은 채(λ=1.0)
실행된 exp 6.1/6.2의 결과를 사후 진단하고, Phase 1의 (τ, λ)를 실측으로 결정한 기록이다.
(2026-07-14 논의로 정밀화: 시프트 불변성, Hinton ×τ² 유도·적용 한계 실측, 그래디언트 방향 문제, 대안 로스 계열.)

**질문:** 왜 6.1(τ=0.022, drop-τ², λ=1)은 val ITC loss는 잘 내리면서 retrieval은 baseline보다 −3.9 나쁜가?

**한 줄 결론:** KD가 ITC 대비 **2.6~6.7× 그래디언트로 전 구간 압도**하며 학생의 행 내부 유사도 갭을 티처의 압축된
스케일로 강제(margin 0.082→0.054)했고, ITC는 logit_scale 인플레(59→88)로 유효 마진을 정확히 원상 복구
(**margin×scale 4.84 vs 4.77 — 동일**)했다. 이 줄다리기에 동반된 **순위 재배열**이 retrieval을 손상시켰다.
val ITC loss 하락은 scale 오염된 지표라 개선의 증거가 아니다.

---

## 0. 배경 — 런 결과 (TB, 재현: `epoch_table.py`)

val_retrieval_itc/r_mean (COCO val 5k, ITC 단독, rank 기반):

| ep | 1 | 4 | 6 | 10 | 14 | 20 | logit_scale@20 |
|---|---|---|---|---|---|---|---|
| baseline (7) | 21.9 | 53.3 | 57.8 | 61.1 | 62.9 | **63.4** | 59.1 |
| 6.1 droptau | **40.2** | 53.8 | 57.1 | 59.2 | 59.7 | 59.5 | 88.1 |
| 6.2 keeptau2 | 16.0 | 50.4 | 55.7 | 60.7 | 61.4 | 62.7 | 65.7 |

읽는 법:
- **6.1의 ep1 +18은 티처 신호의 실효 가치 증명** — 문제는 신호가 아니라 이후의 크기 관리.
- **6.1의 loss_train/itc_kd가 ep4 0.46 → ep20 0.59로 상승 반전** — ITC가 기하를 끌고 가고 KD가 저항하는
  줄다리기의 직접 신호. 이 패턴이 다음 런에서 재발하면 λ가 여전히 크다는 뜻.
- **6.2는 no-op 대조군**: ×τ²=0.022²=4.84e-4가 loss·grad를 ~2000× 축소(§2에서 실측 확인).
  baseline과의 차이(−0.7)는 노이즈 수준.
- 목표 사다리: 바닥 = student solo 63.4 (동일 사이즈 — 반드시 초과), 목표 = base 모델 66.2~66.7
  (exp4/exp5-lrup). **exp5-lrup은 logit_scale 110.9로 retrieval 최고** → "높은 scale 자체가 병"이라는
  가설은 기각. scale은 증상이다.

### 지표 읽기의 대전제 — retrieval은 rank 기반·scale 불변, CE는 아니다

retrieval 평가 절차: 쿼리 캡션 하나에 대해 5,000개 이미지 전부와 코사인 유사도 계산 → 내림차순 정렬 →
정답 이미지가 상위 1/5/10위 안인가(R@1/5/10). **유사도의 '값'이 아니라 '순서'만 쓴다.** 모든 유사도에
양수 상수를 곱하거나(=logit_scale) 더해도 순서는 불변 → 지표 불변. 반면 loss_val/ita는
softmax(sim×logit_scale)의 확률값 CE라서 scale이 크면 (순위가 맞는 다수 행에서) 값이 그냥 작아진다.
⇒ **scale이 다른 두 런의 val ITC loss는 비교 불가. 런 간 비교·모델 선택은 retrieval로만 한다.**
"loss는 내려가는데 retrieval은 나쁘다"는 관측은 이래서 모순이 아니다.

---

## 1. 측정 A — 티처 in-batch 유사도 통계 (`teacher_stats.py`)

BLIP-large 티처를 학습과 동일한 pretrain 배치(B=40, aug ON) 200개에 forward → 40×40 코사인 행렬 저장
(`teacher_sims.npz`). `--no-aug`로 clean 뷰 200배치도 별도 저장(`teacher_sims_noaug.npz`).
07-07 측정(100배치)의 재확인 + t2i 방향·τ 상단 그리드 확장.

### 각 통계의 의미

| 통계 | 정의 | 무엇을 말해주나 |
|---|---|---|
| pos(diag) | 정답쌍 코사인 | 티처 임베딩의 절대 위치. 0.434±0.089 |
| neg(off) | 무작위 음성 코사인 | 배경 유사도 수준. 0.314±0.057 — 전체가 [0.2, 0.6] 좁은 콘에 압축 |
| margin | 행별 diag − max(off) | 배치 내 판별 여유. **mean i2t 0.024 / t2i 0.002** |
| top-1 오답률 | margin<0 행 비율 | 티처가 40개 중 정답쌍을 1등으로 못 놓는 비율 |
| diag질량(τ) | softmax(sims/τ) 행에서 정답 확률 평균 | KD 타깃이 "정답 맞히기"에 쓰는 몫 — CE스러움의 정도 |
| top1질량(τ) | 행 최댓값 평균 | 타깃 sharpness (정답 여부 무관) |
| entropy / 유효서포트 | H, exp(H) | 확률이 실질적으로 퍼진 후보 수. 40이면 균등=정보 없음 |
| dup질량 | same-image 비대각 확률 합 | COCO 5캡션/VG region 중복 효과 — **0.01/batch로 무시 가능** |

### 왜 margin을 τ와 비교하는가 — 시프트 불변성

softmax는 행 내 상수 이동에 불변이다:

    p_j = exp(z_j/τ) / Σ_k exp(z_k/τ) = 1 / Σ_k exp((z_k − z_j)/τ)

마지막 형태에서 분포는 **로짓 차이 (z_k − z_j)/τ에만** 의존한다. 정답 로짓의 절대 크기(0.43/τ ≈ 20)는
소거되고, 유일한 무차원량은 (차이)/τ = margin/τ다. 실측 검산: margin/τ = 0.024/0.022 ≈ 1.1 →
1등:2등 odds ≈ e^1.1 ≈ 3:1 → 측정된 top1질량 0.61·유효서포트 4.2와 정합. "로짓 크기 대비"라는 직관이
유효한 유일한 형태는 크기=스프레드(neg std 0.057)로 읽는 것이고, 그건 τ 그리드의 엔트로피가 이미 반영한다
(스프레드/τ가 서포트를 결정).

### 결과와 해석

aug ON: 오답률 i2t 37.1% / t2i 46.9%. clean: 31.3% / 41.0%.
- **오답의 대부분(31~41%p)은 티처+데이터 고유, aug 기여는 ~6%p뿐.**
- **[미검증 추론]** 이 오답의 상당 부분은 "정당한 co-match" — 배치 내 비대응 쌍 (이미지 i, 캡션 j)인데
  캡션 j가 이미지 i를 사실상 참으로 기술하는 경우(예: 배치에 화장실 사진 2장 + "a bathroom with a white
  toilet") — 일 것이다. 근거는 간접적(BLIP-large가 5k retrieval에선 강함)이며 **직접 검증 안 됨**.
  검증 방법: 오답 행 표본 30~50개의 (이미지, 정답 캡션, 티처 1등 캡션)을 시각화. 티처 1등이 명백히
  틀린 경우가 다수면 이 해석은 기각되고 KD 타깃의 노이즈 비중↑ → λ를 더 낮출 근거가 된다.
- t2i가 항상 더 나쁜 이유: 한 캡션(특히 VG region의 짧은 구)에 맞는 이미지가 여럿이라 i2t보다 모호성이 큼.

τ 그리드 (aug ON, i2t 행):

| τ | diag질량 | top1질량 | 유효서포트 |
|---|---|---|---|
| 0.022 | 0.51 | 0.61 | 4.2 |
| 0.05 | 0.23 | 0.26 | 19.4 |
| 0.07 | 0.14 | 0.15 | 28.5 |
| 0.1 | 0.09 | 0.09 | 34.5 |
| ≥0.15 | ≤0.06 | ≤0.06 | 37.7+ (≈균등) |

- **τ 창의 양쪽 벽**: 타깃이 one-hot이면 KD는 라벨 CE와 동일해져 증류의 존재 이유가 없고(하한),
  균등이면 정보가 없다(상한). 0.022는 one-hot이 아니지만(서포트 4.2) diag질량 0.51로
  "절반은 CE, 절반은 관계 정보"인 경계 지대 — Arm T vs S가 이 경계의 어느 쪽이 나은지 검증.
- **정보가 살아있는 τ 창은 0.022~0.07.** τ≥0.1은 티처 코사인 스프레드(std 0.057)가 눌려 있어 즉시 균등화.
- τ는 "티처 분포를 읽는 배율"(정보 배분)이면서 grad 1/τ 증폭기(크기)를 겸한다.
  **정보는 τ로, 크기는 λ로 분리** — 단 이 분리는 Hinton 보정(§2)이 아니라 grad 실측으로 한다.
- 부수 확인: 티처 ckpt의 학습된 temp = **0.0157** (τ=0.022의 출처 아님 — 0.022는 07-07의
  pos-prob 0.46 근거 + 학생 수렴온도 0.023 앵커로 선정된 값).

---

## 2. 측정 B — grad probe (`grad_probe.py`, `grad_probe_results.npy`)

baseline(exp7) ckpt ep00/04/10/19 각각에서, 고정 배치 8개에 대해 **같은 forward 그래프**로부터
L_ita와 L_kd(τ)를 따로 backward하여 공유 파라미터(visual_encoder, vision_proj, text_encoder, text_proj)의
grad를 비교. 같은 그래프에서 뽑으므로 dropout·배치 조건이 동일해 순수하게 로스 차이만 반영된다.

### 각 통계의 의미

| 통계 | 정의 | 무엇을 말해주나 |
|---|---|---|
| \|g_ita\| | L_ita grad의 L2 norm | ITC의 기본 압력. **6.4 → 9.0으로 훈련 중 증가** |
| \|g_kd(τ)\| | L_kd grad norm (λ=1, drop-τ²) | KD의 압력. τ 의존성은 아래 crossover 참조 |
| ratio | \|g_kd\|/\|g_ita\| | λ=1일 때 KD의 상대 압력. **λ×ratio가 실제 균형** |
| cos(g_kd, g_ita) | 두 grad 벡터의 코사인 | 방향 정렬. **전 조합 음수(−0.06~−0.34)** — 수백만 차원에서 무작위면 ≈0이므로, 일관된 −0.2는 작지만 체계적인 둔각(정반대 아님) |
| dL_ita/d(logit_scale_raw) | scale raw 파라미터에 대한 ITC grad | **전 ckpt 음수** = scale 상승 압력 상시 존재 (아래) |

**"종속"의 정의:** λ·|g_kd|/|g_ita| < 1. 파라미터 갱신에서 ITC가 주(主) 손실, KD가 보조가 되게 하는
grad-norm 조건이다. 근거: 행의 ~40%(top-1 오답률)에서 KD 타깃("1등은 대각선 아님")과 ITC 타깃("대각선이
1등")이 동시 만족 불가 → 타협점은 grad 크기 비가 결정 → 평가 기준(정답쌍 랭킹 = retrieval)은 ITC 타깃
쪽 세계관이므로 ITC가 주여야 한다. (정리가 아닌 휴리스틱 — 최적 비율은 Phase 1이 답할 경험적 질문.)

**왜 ITC는 scale을 키우고 싶어하나:** CE = −log p(pos)에서 순위가 이미 맞는 행(마진>0)은 scale↑로
p(pos)→1이 되어 loss가 공짜로 내려간다. 마진<0 행은 반대로 커진다. 다수 행이 맞는 상태에선 총합이
"키워라" → dL/d(scale)<0 상시. 평형 = 맞는 행의 한계 이득과 틀린 행의 한계 손해가 상쇄되는 지점
(baseline 59, KD가 마진을 눌러놓은 6.1은 88). temp collapse 때 본 것과 같은 힘.

### Hinton ×τ² — 유도와 적용 한계 (실측 crossover)

유도 (Hinton, Vinyals, Dean 2015 §2.1): ∂L/∂z_i = (q_i − p_i)/τ. **고온 극한**(τ ≫ 로짓 차이)에서
softmax가 선형화되어 q_i ≈ 1/N + (z_i − z̄)/(Nτ) → ∂L/∂z_i ≈ (z_{s,i} − z_{t,i} − 평균차)/(Nτ²).
grad가 1/τ²로 죽으므로 ×τ²를 곱해 τ 선택과 무관하게 O(1)로 유지하자는 것. **선형화 regime에서만 유도된다.**
반대 극한(τ ≪ 차이)에선 (q−p)가 포화(유계)라 grad ~ 1/τ이고, ×τ²는 net τ배 = grad를 죽인다.

probe 데이터(ckpt00)가 crossover를 직접 보여준다:

| τ | 0.022 | 0.05 | 0.07 | 0.1 | 0.15 | 0.2 | 0.3 |
|---|---|---|---|---|---|---|---|
| \|g_kd\|·τ (1/τ regime이면 상수) | 0.95 | 1.15 | 1.15 | 0.98 | 0.70 | 0.53 | 0.35 |
| \|g_kd\|·τ² (Hinton regime이면 상수) | 0.021 | 0.057 | 0.080 | 0.098 | **0.105** | **0.105** | **0.104** |

τ≥0.15(분포 ≈ 균등 = 선형화 구간)에서 |g_kd|·τ²가 상수 — Hinton 보정이 정확히 성립. τ≤0.05에선
|g_kd|·τ가 상수에 가까움 — 1/τ regime. ⇒ **τ=0.022에서 ×τ²는 유도 근거가 없고(실측으로도 기각),
keeptau2가 no-op이 된 이유가 이것이다.**

### ×τ 보정 채택 — 우리 regime의 올바른 정규화 (2026-07-14 논의)

crossover의 실행 결론: **saturation regime(τ≤0.1)의 올바른 그래디언트 정규화는 loss에 ×τ**다
(×τ²는 고온 전용, ×1은 무보정). grad ∝ 1/τ이므로 ×τ가 이를 상쇄해 |g_kd|를 τ에 무관하게 만든다.
0.022도 0.05도 이 구간이라 **둘 다 ×τ**. 채택 방식: `itc_distill_loss`가 `× temp`를 반환
(drop-τ² 대비 한 줄) → 두 arm이 **단일 공유 λ로 grad 크기를 자동 정합**한다.

**함정 — ×τ 후의 절대값 ~1.0은 ITC와 1:1이 아니다:**

    |g_kd|·τ ≈ 0.95~1.15 (절대 norm)  vs  |g_ita| ≈ 6.45 (같은 단위)
    ⇒ ratio(λ=1) = 1.0 / 6.45 ≈ 0.15,   λ 1 올릴 때마다 ratio +0.15

진짜 1:1은 λ≈6.5에서. 목표 ratio는 λ_shared로 정한다(§7). 정합성 확인: drop-τ² + 손튜닝
λ(T 0.10 / S 0.25)의 비율 0.40 ≈ τ비율 0.44 → **손값이 이미 ×τ를 우연히 근사**했음(원리를 명시화한 것).
잔차: |g_kd|·τ 완전 평평 아님(0.95~1.15, ~20%) + 훈련 중 감쇠(아래) → "동일 자릿수 정합".

### 핵심 결과 — ratio(λ=1)

| τ | ep0 | ep4 | ep10 | ep19 |
|---|---|---|---|---|
| **0.022 (6.1 실황)** | **6.66** | **4.19** | **3.58** | **2.56** |
| 0.05 | 3.56 | 2.08 | 1.37 | 0.89 |
| 0.07 | 2.54 | 1.28 | 0.75 | 0.49 |
| 0.1 | 1.51 | 0.63 | 0.37 | 0.24 |

- keeptau2의 유효 ratio = 6.66×4.84e-4 ≈ **0.003** → no-op 실측 확인. 기존 두 런은
  "압도(≥2.6)"와 "무력(0.003)"만 밟았고 **중간 지대(0.2~1)는 미답**.
- λ 고정만으로 ratio가 훈련 중 ~3× 자연 감쇠(|g_ita|↑, |g_kd|↓). 이 자연 감쇠를 Phase 1 스케줄로 채택
  (명시적 anneal 없음) — 근거는 §7 스케줄 논의.

### 방향 문제 — ratio 튜닝은 "같은 방향" 가정이 아니다

방향이 같다면 λ는 거의 안 중요하다(같은 방향의 재스케일). 비율이 중요한 이유가 방향이 다르기 때문:
갱신은 벡터 합 g_ita + λ·g_kd이고 λ가 두 방향 사이의 타협점을 정한다. 순 갱신의 ITC 방향 성분:

    ĝ_ita · (g_ita + λ_eff·g_kd) / |g_ita| = 1 + ratio·cos     (ratio = λ_eff·|g_kd|/|g_ita|)

- **6.1 (λ=1, ep0):** 1 + 6.66×(−0.24) = **−0.6** → 초반 순 갱신이 주 손실 grad와 음의 정렬
  (문자 그대로 ITC를 거슬러 걸음).
- **Arm T (×τ, λ_shared=2.5 → ep0 ratio 0.37):** 1 + 0.37×(−0.24) = **+0.91** → ITC 진행 91% 보존.
- **Arm S (×τ, λ_shared=2.5 → ep0 ratio 0.45):** 1 + 0.45×(−0.34) = **+0.85** → S는 cos가 더 음수라 보존 약간 낮음.

그래도 cos<0이 지속되는 한 λ만으로 충분한가는 열린 문제 — **Phase 1.5 후보: gradient surgery
(PCGrad류: cos<0일 때 g_kd에서 g_ita 충돌 성분을 사영 제거).** 비용 스텝당 backward 2회(~1.3–1.5×).
Phase 1에서 λ만으로 게이트를 못 넘으면 올린다(지금 넣으면 변인 2개).

**LM-KD는 왜 잘 됐나 (가설):** (1) teacher-forced에서 티처 top-1 토큰 ≈ 정답 토큰 → CE와 타깃 비충돌,
(2) T=2는 softening regime(위 표 오른쪽)이라 ×T² 보정 유효(실제 유지), (3) logit_scale 같은 공유 도피
노브가 없어 싸움의 흔적이 남을 곳이 없음. 검증 가능: 같은 probe로 cos(∇L_lm_kd, ∇L_lm) 측정(미실시).

### 함정 기록 (재현 시 주의)

1. **워크트리 `Distillation_Project_itc_distill_only`의 `losses.py`는 keep-τ²(6.2 arm) 로컬 수정 상태.**
   1차 probe가 이를 import해 |g_kd|·L_kd가 τ²배 오염됐다(cos만 스케일 불변이라 유효).
   본 스크립트는 drop-τ² KD를 내장해 재실행한 결과다. **Phase 1 런 전 원복 필수** — 안 하면 no-op 재현.
2. baseline 궤적 위에서의 측정이므로, 실제 KD 런에서는 학생이 티처에 가까워져 ratio가 이보다 낮게 형성될
   것 → 여기서 고른 λ는 보수적 상한으로 읽는다.
3. alpha=0.4 고정(실제 초기 램프와 다름), dropout은 train mode(고정 seed).

---

## 3. 측정 C — 학생 기하 비교 (`student_geom.py`)

측정 A와 같은 통계를 학생 두 모델(baseline ep19 vs 6.1 droptau ep19)에 적용. clean COCO val 5000쌍
(caption first, transform_test — retrieval 평가와 동일 조건).

| | baseline (7) | 6.1 droptau | 읽는 법 |
|---|---|---|---|
| pos cosine | 0.432±0.038 | 0.254±0.027 | 절대 수준 하락 — 어느 로스의 직접 효과도 아님(아래) |
| neg cosine | 0.213±0.057 | 0.096±0.043 | 〃 |
| margin (i2t) | 0.082 | 0.054 | **행 내부 갭 34% 압축 = KD의 직접 효과** (티처 갭 0.024~0.030 쪽으로) |
| logit_scale | 59.1 | 88.1 | ITC의 보상 |
| **margin×scale** | **4.84** | **4.77** | **동일** — ITC CE가 요구하는 유효 로짓 마진이 상수 유지. 압축↔인플레 상쇄 평형의 증거 |
| top-1 오답 (i2t/t2i) | 7.35/7.60% | 8.90/7.90% | 순위 품질 실손상 — retrieval −3.9와 정합 |

### 절대 수준(0.43→0.25)의 원인 귀속 — 시프트 불변성의 귀결

KD도 ITC도 행별 softmax 로스라서(§6) **둘 다 행 전체에 상수를 더하는 방향엔 grad가 정확히 0**이다:
softmax(z + c·1) = softmax(z). "이미지 i가 모든 캡션과 일괄적으로 0.1씩 덜 유사해지는" 변화는 어느 loss
값도 안 바꾼다. 따라서:
- **KD는 유사도의 절대 수준을 밀거나 당길 능력이 원천적으로 없다** — KD가 통제하는 건 행 내부의 차이뿐.
  "티처 pos는 0.43인데 왜 학생이 0.25로 갔지? KD가 이상하다"로 읽으면 안 되는 이유.
- 절대 수준은 어느 로스도 직접 정하지 않는 **부산물 평형**이다: CE가 음성을 밀어내는 uniformity 압력이
  구면 위에서 특징을 퍼뜨리면 배경 유사도가 내려가고, weight decay·임베딩 이방성 등 로스 밖 힘이
  그 자유 방향의 표류를 정한다.
- KD의 책임 범위는 **차이의 압축(margin 0.082→0.054)까지**다.

### margin×scale 동일성은 "진단"이지 "손상"이 아니다

균일·단조 압축이었다면 순서가 불변이라 retrieval은 무해했을 것이고 scale이 보정해줬을 것이다.
실제 손상은 압축량이 아니라 **압축에 동반된 순위 재배열**: KD가 티처의 (aug 뷰 기준, per-view 노이즈
std 0.067 ≫ mean 0.016) 행별 순위로 학생을 재배열한 것 (top-1 오답 7.35→8.90%, retrieval −3.9).
⇒ 통제할 노브는 "압축 지점"이 아니라 **재배열의 양 = λ**다.

---

## 4. 측정 D — 티처 오답의 co-match 판정 (`teacher_comatch.py`, `comatch_rate.py`)

§1의 미검증 추론("티처 top-1 오답의 상당수는 정당한 co-match")을 검증. clean 뷰 i2t top-1 오답을
(query 이미지 i, GT 캡션, 티처가 고른 캡션 j*, 이미지 j*) 몽타주로 렌더해 육안 판정 +
source·region 크기별 오답률 측정.

### 결과 1 — 오답은 압도적으로 VG·degenerate-region 현상

| source | i2t top-1 오답률 |  | VG region area | 오답률 |
|---|---|---|---|---|
| COCO (전체이미지+완결캡션) | **5.4%** (12/221) |  | <32² px | **63.1%** |
| VG (region+짧은 phrase) | **35.8%** (781/2179) |  | 32–64² | 43.3% |
|  |  |  | 64–128² | 32.2% |
|  |  |  | 128–256² | 18.6% |
|  |  |  | ≥256² | **15.8%** |

⇒ **티처는 내용이 풍부한 이미지엔 신뢰 가능한 랭커(COCO 5.4%, 큰 VG region 15.8%), 작은 VG 크롭엔
사실상 무작위(63%).** in-batch "오답"의 대부분은 의미 혼동이 아니라 **degenerate region 아티팩트**다
([[vg-min-scale-experiment]]의 "min_scale 0.2 크롭이 객체를 잘라냄"과 같은 뿌리).

### 결과 2 — 육안 판정: 정당한 co-match와 노이즈의 혼합

몽타주(`comatch_top_confident.png`, `comatch_random.png`, top-8 확신오답 + 무작위 8) 판정:
- **정당한 co-match 존재**: "black topping on pizza" ↔ "a piece of pizza"(둘 다 피자),
  셀러리 ↔ "the leaves are green", "sandals with socks" ↔ "bare feet"(둘 다 발).
- **그러나 상당수는 (a) 초-일반 속성 문구**("the leaves are green" / "the background is colored tan" /
  "soft focus caused by camera aperture" — 수많은 이미지에 다 맞아 판별력 없음) **또는
  (b) degenerate 크롭 혼동**(인도 연석→"cow under a tree", 양초→"bus", 어두운 타일→"man on phone").
- **가장 확신한 오답(Δ 0.15~0.20)일수록 (b) 노이즈 쪽.** near-tie(Δ≈0)에 (a)·정당 co-match가 섞임.

### 함의

- **T vs S는 육안으로 못 가린다 → 둘 다 실험(계획대로).** 프라이어는 **T(sharp)가 안전** 쪽:
  soft(S)는 이 혼합 꼬리(정당한 거친 co-match + VG degenerate 노이즈)를 다 전달하므로 리스크 실재.
- **Phase 2 정제 후보(신규, 잘 동기화됨)**: KD 타깃을 **내용 풍부도로 게이팅** — COCO 가중 /
  VG region area로 weight 스케일 / 작은 region KD 제외. 63% 노이즈원(작은 VG)을 직접 제거. §6.2와 결합 가능.

---

## 5. 종합 서사

1. τ=0.022 (유효서포트 4.2, 정보 배분 자체는 07-07 근거로 타당) + drop-τ² + **λ=1** → KD 압력이 ITC의 2.6~6.7×.
   ep0에서 순 갱신의 ITC 정렬이 **−0.6** — 주 손실을 거슬러 시작.
2. KD는 학생의 행 내부 갭을 티처의 압축 스케일로 강제 (margin 0.082→0.054). 방향 충돌은 구조적
   (cos<0): KD 압축 vs ITC 확장.
3. ITC는 남은 자유도인 logit_scale로 보상 (59→88, margin×scale 상수 4.8).
4. 초기(랜덤 학생)에는 티처 신호가 순이득 (ep1 r_mean +18) — 학생이 배울수록 신호는 소진되고
   구속만 남아 ep4부터 KD loss 상승 반전, retrieval은 baseline에 역전·정체.
5. val ITC loss는 scale 오염으로 계속 하락 — 가짜 개선 신호.

**배치 40 문제에 대한 판단:** in-batch 40이 작은 것은 사실이나(무작위·easy 음성 39개), τ가 유효 창 밖이거나
λ가 불균형인 상태에서는 배치를 키워도 같은 병리가 재현된다. 서포트 확장(all_gather 160 또는 teacher-feature
queue)은 (τ, λ) 균형이 확보된 뒤의 Phase 2 옵션으로 보류.

---

## 6. ITC와 ITC-KD는 같은 함수족 — 대안 로스 계열 (2026-07-14 논의)

두 로스를 나란히 쓰면:

| | 로짓 | 후보 집합 | 온도 | 타깃 | 로스 |
|---|---|---|---|---|---|
| ITC | scale·(v_i·t_j) | 배치 40 + queue 57600 | 학습되는 logit_scale | (1−α)·one-hot + α·softmax(momentum) | CE |
| ITC-KD | (v_i·t_j)/τ | 배치 40 | 고정 τ | softmax(티처 sims/τ) | KL |

KL(p‖q) = CE(p, q) − H(p)이고 H(p)는 학생과 무관한 상수 → **학생 grad 기준으로 KL = soft-target CE.**
즉 둘 다 "코사인 행 → 온도 → softmax → (soft) 타깃과 CE"라는 동일 구조이고, ITC-KD는
"타깃이 티처이고 후보가 40개뿐이며 온도가 고정인 ITC"다. 같은 지렛대(행 내부 유사도 차이)를 다른 타깃으로
당기기 때문에 정면으로 싸울 수 있는 것이고(§2 cos<0), 아래 통합(6.2)이 자연스러운 이유이기도 하다.

### 6.1 sim-MSE (직접 유사도 매칭) — 후보로만 기록

"로짓 KD는 로짓이 비정규라 softmax가 필요하다"는 일반론은 **LM 로짓(무계·스케일 임의)엔 맞지만, ITC의
'로짓'은 L2 정규화 특징의 코사인이라 이미 [−1,1]로 정규화**돼 있다. 따라서 직접 매칭이 잘 정의된다:
L = ‖S_s − S_t‖²_F (similarity-preserving KD / RKD 계열).

| | KL(softmax) — Phase 1 채택 | sim-MSE |
|---|---|---|
| 시프트 불변 | O (절대 수준 못 건드림) | X — 절대 수준까지 고정하지만, 그 방향은 ITC 무관심 자유 방향이라 싸움 안 남 |
| grad 형태 | (q−p)/τ — 포화, 상위 후보 집중 | (s_s−s_t) — 스프링, 전 쌍 균등 가중 |
| τ | 필요 (이중 역할) | 불필요 — τ 논쟁 소멸 |
| 전달 정보 | softmax 상위권 상대 구조 | 유사도 구조 전체 (easy 음성 간 거리 포함) |
| 핵심 긴장 | 갭 압축 vs ITC 확장 — 존재 | **동일하게 존재** (λ 필요, grad probe 재실측으로 산출 가능) |

⇒ 균형 문제를 회피하는 게 아니라 가중 프로파일을 바꾸는 선택. Phase 1은 KL 유지(λ 캘리브레이션이 그 기준),
sim-MSE는 백업 arm 후보.

### 6.2 티처-타깃-혼합 (통합형) — Phase 2 본선 후보 (2026-07-14 사용자 합의)

KD를 별도 로스로 두지 않고 **ITC의 soft-target 자리에 티처를 넣는다**: BLIP의
targets = α·softmax(momentum) + (1−α)·one-hot에서 momentum 자리를 티처로 교체/혼합.
- 로스가 하나 → **grad 방향 충돌이 정의상 소멸**, 후보 집합도 queue까지 자동 통일.
- 선결 조건: queue 항목들의 티처 특징 = **teacher-feature queue** (Phase 2로 보류했던 서포트 확장과 동일 재료).

---

## 7. Phase 1 설계 (이 진단의 산출, 2026-07-14 확정)

**로스 = τ·KL (×τ 보정, §2), 단일 공유 λ_shared = 2.5** (목표 ep0 ratio ~0.4 — 6.1의 6.66 대비 16× 약).
두 arm은 **grad 크기가 ×τ로 자동 정합**되어 유일한 변인이 **타깃 sharpness(τ)** 하나. config 변경은
`itc.temp` + output_dir + exp 태그뿐(weight=2.5 공통):

| arm | τ (temp) | λ_shared | ratio ep0→ep19 | ITC정렬 ep0 | 타깃 성격 (측정 A) |
|---|---|---|---|---|---|
| **T** (sharp) | 0.022 | 2.5 | 0.37 → 0.14 | +0.91 | diag질량 0.51·서포트 4.2 — ITC와 가까움. **안전하나 marginal 정보 적음** |
| **S** (soft) | 0.05 | 2.5 | 0.45 → 0.11 | +0.85 | diag질량 0.23·서포트 19 — 티처 랭킹 전달. **정보 많으나 VG 노이즈 포함**(§4) |

**T vs S가 실제로 묻는 것:** "티처의 세밀한 음성 랭킹을 배울 가치가 있는가, 상위 몇 개만 믿을 것인가."
= §4 co-match의 실험적 판정. co-match 육안 판정이 혼합(정당+노이즈)이라 못 가림 → **둘 다 실행 확정**.

**스케줄 — 자연 감쇠 채택 (명시적 anneal 없음):**
- 근거: KD는 초기에 돕고(6.1 ep1 +18) 후반에 해친다(학생 ep19 in-batch 오답 7% ≪ 티처 37% → 후반엔
  학생이 티처보다 나음). 고정 λ가 ratio를 40%→12%로 수동 front-load하므로 이 자연 감쇠로 충분하다는 가설.
- 결정적 근거: **6.1의 후반 손해는 ratio 2.6에서 났고, Phase 1 후반 ratio는 0.12로 20× 작다** → 후반
  손해 메커니즘이 안 켜질 가능성이 큼. 명시적 anneal은 τ 효과와 섞이므로 지금 넣지 않음.

- 코드: drop-τ²(`4448a7e`)에 ×τ 한 줄 추가. **워크트리 keep-τ² 로컬 수정 원복 선행**(패치 박제됨:
  `../2026-07-07_teacher_sim_and_tau/exp6.2_keeptau2.patch`).
- 런칭: 두 arm 코드 동일(config만 차이) → **워크트리 1개 + config 2개**로 충분(동시 실행 시 `--master_port` 분리).
  **4GPU·eff160 유지 필수**(2GPU 분할 시 ITC eff batch 변인 추가로 통제 깨짐).
- 성공 게이트: ep1 r_mean ≥ 30(신호 생존) / ep10 ≥ 61.1 / **ep20 ≥ 63.4(student solo, 필수)**, 목표 66+ (base).
- 감시: loss_train/itc_kd 중반 이후 상승 반전(줄다리기 재발), logit_scale의 baseline(59) 대비 이탈 폭.
- **Phase 1.5 (조건부):** 후반 손해 재현 시 명시적 λ anneal; 게이트 미달 + cos<0 지속 시 PCGrad(§2).
- **Phase 2 (보류):** §6.2 티처-타깃-혼합 + teacher-feature queue / all_gather 서포트 확장 /
  **내용 풍부도 게이팅 KD**(§4: COCO 가중·VG region-area 스케일 — 63% 노이즈원 제거).

### 미실시 검증 항목 (선택)

1. ~~티처 오답 co-match 시각화~~ → **완료(§4)**: 오답은 VG degenerate-region 현상, 정당 co-match와 노이즈 혼합.
2. cos(∇L_lm_kd, ∇L_lm) probe → LM-KD 성공 요인 가설 검증 (§2).
3. sim-MSE grad probe → §6.1 백업 arm의 λ 산출.

---

## 파일

- `teacher_stats.py` — 측정 A. 재현:
  `cd /home/minwoo/Distillation_Project_itc_distill_only && CUDA_VISIBLE_DEVICES=0 /home/minwoo/miniconda3/envs/kd_r4/bin/python teacher_stats.py [--no-aug]`
  (출력 npz 경로는 `--out`으로 지정; 캐시 재분석은 `teacher_stats.py reanalyze <npz>`)
- `teacher_sims.npz` / `teacher_sims_noaug.npz` — 200배치 40×40 코사인 행렬 + same-image 마스크 캐시
- `grad_probe.py` — 측정 B (drop-τ² KD 내장). `grad_probe_results.npy` — (ckpt × τ × batch) 원시 결과
- `student_geom.py` — 측정 C (val json 플랫화 포함)
- `teacher_comatch.py` — 측정 D 몽타주 생성. `comatch_top_confident.png` / `comatch_random.png` — 육안 판정 figure
- `comatch_rate.py` — 측정 D source·region-크기별 오답률 카운터
- `epoch_table.py` — §0 TB 에폭 표 재생성
- (참고) keeptau2 arm 코드 스냅샷: `../2026-07-07_teacher_sim_and_tau/exp6.2_keeptau2.patch`
