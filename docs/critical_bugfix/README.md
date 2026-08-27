# Critical bugfix: ITC-distillation의 τ² 계수가 KD 신호를 사실상 무력화

**날짜:** 2026-07-07
**대상:** `distillation/losses.py::itc_distill_loss` (worktree `extra_exp/itc_distill_only`, config `configs/pretrain_itc_distill.yaml`, exp `6.itc_distill`)
**증상:** TensorBoard `loss_train/itc_kd` ≈ 0.003 (다른 loss 대비 ~2000×작음) → ITC 증류가 학습에 거의 기여하지 못함.

---

## 1. 관측

학습 중 TB 스칼라 (step ~436):

| loss | mean | itc_kd 대비 |
|---|---|---|
| `ita` | 7.73 | ×2250 |
| `lm` | 5.25 | ×1530 |
| `itm` | 0.63 | ×185 |
| **`itc_kd`** | **0.0034** | 1 |
| `total` | 13.6 | — |

weight=1.0로 더해도 total에 0.005 기여 → 무시 수준.

## 2. 원인: `× τ²` 계수 (Hinton 관례의 무비판적 이식)

`itc_distill_loss` 리턴값:

```
0.5 * (KL_i2t + KL_t2i) * τ²      # τ = 0.05  →  τ² = 0.0025
```

- 관측 0.0034를 τ²로 되나누면 **원래 KL ≈ 1.36 nats** — 신호 자체는 작지 않다. `τ²=0.0025` 곱이 로스를 ~2000× 눌렀을 뿐.
- 이 `× T²`는 **Hinton(2015) KD의 관례**이고, 그 정당성은 **T > 1 (softening)** 전제 위에서만 성립한다.

### Hinton T² 논리

소프트 타깃 `p=softmax(z/T)`, 학생 `q=softmax(v/T)`. 학생 logit에 대한 gradient:

```
∂L_soft/∂v = (1/T)(q − p)      # 고온 극한에서 ~ 1/T² 스케일
```

소프트 타깃 gradient가 `1/T²`로 줄어드니, 하드 타깃(order 1)과 균형을 맞추려 **× T²**를 곱해 order 1로 되돌린다(온도 불변성). **T² 포함 시 실제 gradient:**

```
∂(T²·L)/∂v = T·(q − p)
```

- **T > 1 (원래 용도):** (q−p)도 작아지고 앞의 T와 상쇄 → order 1, 의도대로.
- **우리 τ = 0.05 (< 1, sharpening):** gradient ∝ **0.05·(q−p)** → 의도적으로 tiny. **전제 위반.**

즉 τ<1에서 `τ²`는 보정이 아니라 **gradient를 τ배로 죽이는 역효과**. τ²를 빼면 `(1/τ)(q−p)=20(q−p)`로 반대로 과해지므로, 어느 쪽이든 **weight로 스케일을 잡아야** 한다.

## 3. "Adam이 작은 loss를 알아서 키워주지 않나?" — 반은 맞고 반은 틀림

- ✅ **단독 loss라면** 맞다. Adam은 상수배 불변(`L→cL` ⇒ `m→cm, v→c²v, m/√v` 불변). KD가 유일한 목적이면 τ² 곱해도 스텝 동일.
- ❌ **합산되면 안 된다.** `total = ita+itm+lm + w·itc_kd`를 backward → **파라미터별 합쳐진 gradient**에 Adam이 적용된다. Adam은 항별로 정규화하지 않음. tiny한 itc_kd gradient는 큰 ita/lm gradient에 더해지는 순간 ~0.05% 섭동으로 **묻히고**, 공유 파라미터(학생 인코더)에서 Adam이 못 살린다.

## 4. "itm은 바이너리라 값이 작은데 왜 등가중 합인가?" (원본 BLIP)

`loss = loss_ita + loss_itm + loss_lm` (ALBEF/BLIP 원본). **로스 스칼라 크기 ≠ 중요도(=gradient 크기).**

| loss | 값이 그런 이유 | gradient |
|---|---|---|
| `ita`≈7.7 | queue 57,600 음성 CE, baseline≈ln(57600)≈10.96 | order-1 (logit_scale로 스케일) |
| `lm`≈5 | vocab CE, baseline≈ln(V) | order-1 |
| `itm`≈0.6 | 2-class CE, 상한 ln2≈0.69 | **order-1** (p≈0.5~0.7 ⇒ (p−y)≈0.3~0.5) |

itm은 **값만 작고 gradient는 건강**하다. 각 loss는 내부 정규화가 달라 자릿수가 다를 뿐, 대체로 **다른 헤드**에 작용하므로 등가중 합이 통한다. **itc_kd만 값도 gradient도 둘 다 작아서** 이 논리의 예외 → weight 보정(또는 τ² 제거)이 필수.

## 5. 티처 유사도 실측 — τ를 "높이는" 방향은 틀렸다

τ를 감이 아니라 데이터로 정하기 위해 BLIP-large 티처를 학습과 동일한 배치에 forward하여 B×B 코사인 측정. **상세 측정·표·결정: [`2026-07-07_teacher_sim_and_tau/`](2026-07-07_teacher_sim_and_tau/) 하부폴더 참조.** (아래는 요약; 초기 1600샘플 → itc가 튀어 4000샘플/모드로 상향.)

| 항목 | aug ON (KD 실제) | aug OFF (clean) |
|---|---|---|
| positive(대각) mean | 0.4225 | 0.4485 |
| negative(비대각) mean | 0.3123 | 0.3177 |
| **pos − hardest_neg** mean | **0.0156** | 0.0302 |
| Δ (간격) | 0.1102 | 0.1308 |

τ 정밀 스윕(aug ON, 학생 수렴온도 0.023 앵커) 발췌 — pos-prob = 티처가 정답쌍에 주는 확률:

| τ | max-prob | pos-prob | entropy | 해석 |
|---|---|---|---|---|
| **0.022** | 0.571 | **0.460** | 0.425 | **다음 실험 목표값** |
| 0.023 | 0.554 | 0.450 | 0.446 | 학생 수렴온도 |
| **0.050** | ~0.27 | ~0.27 | — | **현재 실험값** |
| 0.10 | 0.087 | — | 0.964 | 거의 uniform → 신호 소멸 |

**결론:**
- 티처 코사인이 **[0.22, 0.55]에 압축**돼 pos/neg가 크게 겹침(`teacher_sim_hist.png`). argmax=diag 0.587(τ 무관) = 티처 batch top-1 정확도 58.7%.
- 압축 때문에 **τ=0.1만 가도 softmax가 uniform** → **τ를 올리면 dark knowledge 소멸.** 초기 직관("τ↑로 soft KL")은 **데이터로 반박됨.**
- 0.020–0.040 전 구간이 매끈한 soft+peaked 고원. **τ=0.022에서 pos-prob 0.46 = 정오답 정보 배분 이상적.**
- 옮길 신호는 positive peak가 아니라 음성 soft 랭킹(Δ≈0.11)뿐 → ITC-KD 기대효과 제한적일 수 있음(리스크).
- (aug가 positive만 −0.026 끌어내려 마진 절반으로 훼손 — **관측 기록만; 설계상 aug 유지 결정.** 하부폴더 "결정" 참조.)

## 6. 결정 & 권장 수정 (2026-07-07)

**결정(사용자):** 현재 τ=0.05 실험 그대로 진행 / **다음 실험 τ=0.022**(정오답 정보 배분 위해 측정 후 결정) / aug 유지·aug-off 미고려·min_scale 0.2는 별도 서브실험.

**τ=0.022 실험에 반드시 동반(미적용):** τ 값과 KD 크기는 직교 — τ만 0.05→0.022로 낮추면 `×τ²`가 5× 작아져 KD가 더 약해짐(로스 ~절반). 따라서
1. **`× τ²` 계수 제거** — τ<1에서 gradient 죽이는 원인. 제거 후 gradient ∝ `(1/τ)(q−p)` 정상 스케일.
2. **λ_itc(weight)로 grad-norm을 ita/lm과 정합** (값 크기비 아닌 gradient 기준).
3. 재시작 필요 — 현재 run 판단 후 반영.

## 참고
- 상세 측정/코드/결과: [`2026-07-07_teacher_sim_and_tau/`](2026-07-07_teacher_sim_and_tau/)
  (`measure_teacher_sim.py`, `tau_sweep.csv`, `teacher_sim_hist.png`, `console_output.txt`, `sim_{aug,noaug}_n100.pt`)

## 후속 (2026-07-14)

exp 6.1(drop-τ², τ=0.022, **λ=1**)/6.2(keep-τ²) 완주 결과의 사후 진단 —
위 "반드시 동반 2번(λ 정합)"이 미이행된 채 실행된 6.1에서 KD가 ITC를 2.6~6.7× 압도,
학생 마진 34% 압축 + logit_scale 인플레(59→88, margin×scale 불변)로 retrieval −3.9.
grad-norm 비율·방향 충돌 실측과 Phase 1 (τ, λ) 결정:
[`2026-07-14_kd_grad_balance_and_geometry/`](2026-07-14_kd_grad_balance_and_geometry/README.md)
