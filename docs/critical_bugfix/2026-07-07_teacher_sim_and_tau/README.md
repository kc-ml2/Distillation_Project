# 2026-07-07 — 티처 ITC 유사도 측정 & τ 결정

**상위 맥락:** [`../README.md`](../README.md) — ITC-distillation의 `×τ²` 계수 버그(KD 신호 ~2000× 무력화) 분석.
이 하부폴더는 그 후속으로, **"τ를 얼마로 둘지"를 감이 아니라 티처 유사도 실측으로 정한 기록**이다.

**대상 실험:** worktree `extra_exp/itc_distill_only`, config `configs/pretrain_itc_distill.yaml`, exp `6.itc_distill`
(small_reg ViT + MiniLM 학생 ← frozen BLIP-large 티처, ITC-only KD)

---

## 측정 방법

`measure_teacher_sim.py` (read-only, 돌고 있는 학습에 무간섭, GPU0 inference-only/bf16):
- BLIP-large 티처(`online_teacher.itc_feats`)를 학습과 동일한 pretrain 배치에 forward → B×B 코사인 유사도 행렬.
- 대각 = 정답쌍(positive), 비대각 = 음성(negative).
- **두 패스**: `aug ON`(KD 실제 조건, min_scale 0.2 크롭 + RandomAugment) vs `aug OFF`(clean, 결정적 리사이즈).
- 각 **100 batches × B40 = 4000 pos / 156000 neg** (초기 1600샘플은 itc가 튀어 부족 → 4000으로 상향).
- B×B 행렬을 `.pt`로 캐시 → τ 그리드만 바꿀 땐 재-forward 없이 즉시 재계산.

재현:
```bash
cd /home/minwoo/Distillation_Project_itc_distill_only
CUDA_VISIBLE_DEVICES=0 /home/minwoo/miniconda3/envs/kd_r4/bin/python \
  critical_bugfix/2026-07-07_teacher_sim_and_tau/measure_teacher_sim.py
```

---

## 결과 1 — 코사인 분포 & 안정성

| 지표 | aug ON (KD 실제) | aug OFF (clean) |
|---|---|---|
| positive(대각) mean | 0.4225 | 0.4485 |
| negative(비대각) mean | 0.3123 | 0.3177 |
| **pos − hardest_neg** mean | **0.0156** | **0.0302** |
| Δ (mean_diag − mean_off) | 0.1102 | 0.1308 |
| 배치간 Δ std (CV) | 0.0108 (9.8%) | 0.0115 (8.8%) |

- **집계 통계 확정**: Δ = 0.110 ± ~0.001 (4000샘플). 배치 간 변동 CV ~10%.
- **튀는 근원은 per-sample**: `pos−hardest_neg` std 0.067 ≫ mean 0.016 (4배) — 개별 쌍은 부호까지 뒤집힘. **표본을 늘려도 안 줄어드는 고유 노이즈**(itc 로스 spikiness의 원인).
- **argmax=diag = 0.587** (τ 무관) → BLIP-large가 batch top-1을 58.7%만 맞힘. ITC-KD가 옮길 수 있는 신호의 상한 지표.

## 결과 2 — 증강이 티처 신호를 훼손 (관측만, 아래 "결정" 참조)

- aug가 **positive 유사도만 −0.026 끌어내리고 negative는 불변** → 판별 마진 절반(pos−hardneg 0.016 vs clean 0.030), Δ −19%.
- 원인: `min_scale=0.2` 크롭이 캡션 속 객체를 잘라낼 수 있음 + RandomAugment 왜곡.

## 결과 3 — τ 정밀 스윕 (aug ON, 학생 수렴온도 0.023 앵커)

`tau_sweep.csv` 전체. 발췌:

| τ | max-prob | **pos-prob** | entropy | argmax=diag |
|---|---|---|---|---|
| 0.020 | 0.606 | 0.481 | 0.382 | 0.587 |
| **0.022** | **0.571** | **0.460** | 0.425 | 0.587 |
| 0.023 | 0.554 | 0.450 | 0.446 | 0.587 |
| 0.025 | 0.521 | 0.429 | 0.487 | 0.587 |
| 0.030 | 0.444 | 0.375 | 0.580 | 0.587 |
| **0.050**(현재) | ~0.27 | ~0.27 | — | 0.587 |
| 0.040 | 0.321 | 0.278 | 0.727 | 0.587 |

- **0.020–0.040 전 구간이 매끈한 "soft+peaked" 고원** — one-hot(pos-prob>0.7)도 uniform(0.025)도 없음. 진짜 절벽(uniform)은 τ≥0.1.
- **τ=0.022에서 pos-prob 0.46** = 티처가 정답쌍에 46%, 나머지 54%를 음성에 구조 있게 배분 → **정오답 정보 배분이 이상적**.
- 초기 직관("τ를 높여 soft KL")은 **반박됨**: 티처 코사인이 [0.22,0.55]로 압축돼 τ 올리면 즉시 uniform.

---

## 결정 (2026-07-07, 사용자)

1. **aug 유지 (clean-teacher 도입 안 함).** 근거: 티처가 훼손된 이미지를 보면 **훼손된 채로 출력을 따라하는 게 맞다.** 온라인 티처라 이는 동시학습이 아니라 **별개 학습에 가까운 성격**. → 결과 2는 관측으로만 기록, 설계 변경 없음.
2. **`min_scale=0.2`의 손해**는 다른 컴퓨터에서 별도 서브실험으로 측정 중 — **본 실험 판단엔 크게 반영하지 않음.**
3. **aug-off는 현재 전혀 고려하지 않음.**
4. **현재 돌아가는 실험(τ=0.05)은 그대로 계속 진행.**
5. **다음 실험: τ = 0.022 로 낮춤.** 근거 = **"정오답에 대한 적절한 정보 배분"을 위해 위 측정(pos-prob 0.46, 학생 수렴온도 0.023 앵커) 후 결정.**

## Open item — 다음 실험에 반드시 동반해야 할 것

**τ 값과 KD 크기(τ²·weight)는 직교.** τ를 0.05→0.022로 **낮추기만** 하면 `×τ²`가 0.0025→0.00048로 ~5× 작아져 **KD 로스/그래디언트가 지금보다 더 약해짐**(초기 근사 로스 ~절반). 따라서 τ=0.022 실험은
- **`×τ²` 계수 제거** (상위 README 참조 — τ<1에서 gradient 죽이는 원인) +
- **λ_itc(weight)로 grad-norm을 ita/lm과 정합**
을 **반드시 함께** 적용해야 τ 변경 효과가 학습에 실린다. (아직 코드 미적용.)

---

## 파일
- `measure_teacher_sim.py` — 측정 코드
- `console_output.txt` — 실행 전체 로그
- `tau_sweep.csv` — τ 0.020–0.040 스윕 (max-prob/pos-prob/entropy/argmax=diag)
- `teacher_sim_hist.png` — pos vs neg 코사인 분포 (aug ON, 4000샘플)
- `sim_aug_n100.pt`, `sim_noaug_n100.pt` — B×B 코사인 행렬 캐시 (재분석용)
