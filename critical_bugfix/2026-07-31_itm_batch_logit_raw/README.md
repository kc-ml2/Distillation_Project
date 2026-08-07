# ITM 학습 배치 로짓 원자료 — 블록별 z1·z2·m·gap 실측

**날짜:** 2026-07-31
**대상:** pretrain **학습 배치 3B**(retrieval 후보 아님) — 학생 baseline@ep19 + 티처 BLIP-large
**설계:** `docs/superpowers/specs/2026-07-31-itm-batch-logit-probe-design.md`
**산출물:** `itm_batch_logit_probe.py`, `probe_core.py`, `csv/*.csv`(6개), `npz/*.npz`, `summary.json`

**세 줄 결론:**

1. **축 B는 결정됐다 — 블록 분리 필수.** Δ = 4.12(티처) / 4.98(학생). 3B 통합 softmax는 확률 질량의
   **96~99%를 positive가 독식**한다(P_pos 실측 0.9878 / 0.9568). 그 질량은 하드라벨이 이미 주는 정보다.
2. **티처의 m은 사실상 상수다.** σ(m) = 0.006~0.007, 500샘플 전체 범위가 폭 0.04. 같은 구간에서 gap은
   −8.4 ~ +7.5로 움직인다. 학생은 σ(m)이 **9배**(0.056~0.063)지만 절대값으로는 여전히 작다.
3. **티처는 라벨과 자주 불일치한다** — positive의 **36.6%**를 no-match로, negative의 **18.2%**를 match로 본다
   (학생은 22.8% / 7.4%). pair-wise target-mix가 학생을 망가뜨린 이유가 여기서 직접 보인다.

---

## 1. 측정 조건

| 항목 | 값 |
|---|---|
| 학생 | `output/pt_smallreg_minilm_baseline/checkpoint_19.pth` (DINOv3 ViT-S/16 + MiniLM) |
| 티처 | `output/official_pretrain_checkpoint/model_large.pth` (BLIP-large) |
| 데이터 | pretrain train split (COCO+VG), `min_scale=0.2`, **학습과 동일한 train transform** |
| 배치 | `batch_size=40` → 3B=120, 13배치 수집 후 블록당 앞 **500행** |
| device | CPU / fp32 (GPU 4장이 학습 잡으로 점유 중) |
| seed | 0 |

**티처는 학생이 뽑은 negative 인덱스를 그대로 재사용**해 완전히 동일한 3B 쌍을 본다
(학생 forward 시 `(image, caption, neg_idx_img, neg_idx_txt)`를 캐시 → 학생 해제 → 티처 로드 → 캐시 replay).
augmentation이 매 `__getitem__`마다 랜덤이므로 이미지 텐서 자체를 캐시해야 비교가 성립한다.

블록 정의 (`models/blip_pretrain.py:471-489`와 동일):

| 블록 | 구성 | 라벨 |
|---|---|---|
| ① `pos` | (img_i, txt_i) | 1 |
| ② `neg_img` | (**neg**\_img_i, txt_i) — 텍스트가 원본 | 0 |
| ③ `neg_txt` | (img_i, **neg**\_txt_i) — 이미지가 원본 | 0 |

## 2. 표 6개 (mean / var)

### 티처 (BLIP-large)

| 블록 | z1 mean | z1 var | z2 mean | z2 var | m mean | m var | gap mean | gap var |
|---|---|---|---|---|---|---|---|---|
| pos | −0.764 | 3.435 | 0.755 | 3.476 | −0.0044 | 0.000051 | 1.520 | 13.821 |
| neg_img | 1.228 | 2.461 | −1.246 | 2.492 | −0.0093 | 0.000041 | −2.474 | 9.906 |
| neg_txt | 1.353 | 2.512 | −1.372 | 2.542 | −0.0095 | 0.000037 | −2.724 | 10.108 |

### 학생 (baseline@ep19)

| 블록 | z1 mean | z1 var | z2 mean | z2 var | m mean | m var | gap mean | gap var |
|---|---|---|---|---|---|---|---|---|
| pos | −0.743 | 1.074 | 0.830 | 1.126 | 0.0435 | 0.003947 | 1.573 | 4.383 |
| neg_img | 1.651 | 1.772 | −1.582 | 1.787 | 0.0344 | 0.003134 | −3.232 | 7.106 |
| neg_txt | 1.832 | 1.874 | −1.749 | 1.902 | 0.0413 | 0.003298 | −3.581 | 7.538 |

전체 원자료는 `csv/{student,teacher}_{pos,neg_img,neg_txt}.csv` (각 500행 + mean/var/std 3행).

## 3. ★ §8.1 판정 — 블록 분리 필수

$$P_{pos} = \frac{\sum e^{g_{pos}}}{\sum e^{g_{all}}}$$

| | Δ = mean(gap_pos) − mean(gap_neg, pooled) | P_pos 실측 | P_pos 근사식 |
|---|---|---|---|
| 학생 | **4.980** | **0.9568** | 0.9864 |
| 티처 | **4.119** | **0.9878** | 0.9685 |

스펙 §8.1의 판정선(Δ≲1 통합 가능 / Δ≳2 분리 필수)에 비추어 **Δ = 4.1~5.0은 분리 필수 구간**이다.

통합 softmax를 쓰면 negative 1000개 전체가 확률 질량의 1.2~4.3%를 나눠 갖게 되고, KL gradient의 대부분이
"positive와 negative를 더 벌려라"에 실린다 — 하드라벨 CE가 이미 하는 일이다. 얻으려던
"같은 라벨 안에서의 상대 순위"에는 거의 gradient가 가지 않는다.

→ **pos B-way / neg_img B-way / neg_txt B-way 세 분포로 분리하고 KL 3개.**

## 4. ★ §8.2 관찰 — m 상수성

| | σ(m) | m min ~ max | m p1 ~ p99 |
|---|---|---|---|
| 티처 pos | **0.0072** | −0.0289 ~ 0.0141 | −0.0207 ~ 0.0103 |
| 티처 neg_img | 0.0064 | −0.0255 ~ 0.0072 | −0.0231 ~ 0.0028 |
| 티처 neg_txt | 0.0061 | −0.0263 ~ 0.0062 | −0.0226 ~ 0.0029 |
| 학생 pos | **0.0628** | −0.1108 ~ 0.2452 | −0.0819 ~ 0.2050 |
| 학생 neg_img | 0.0560 | −0.1110 ~ 0.2199 | −0.0866 ~ 0.1782 |
| 학생 neg_txt | 0.0574 | −0.1292 ~ 0.2199 | −0.0813 ~ 0.1831 |

**티처는 500샘플 전체에서 m이 폭 0.04 안에 갇혀 있다.** 같은 구간에서 gap은 −8.4 ~ +7.5로 움직인다.
티처가 `z2` 단독으로 재정렬할 수 있는 이유가 이것이다.

학생은 σ(m)이 티처의 **약 9배**, 범위 폭 0.38이다. 다만 절대값으로는 gap 산포(2.09~2.75)의 2~3% 수준이다.

**축이 바뀌어도 결론이 바뀌지 않았다.** 설계 §2는 "선행 조사의 within-query 측정이 cross-sample에서
유지된다는 보장이 없다"고 우려했지만, 실측값은 거의 같다 (within-query 티처 0.22% / 학생 2.05%
vs cross-sample 티처 0.19% / 학생 2.1~3.0%). `w_s` 방향과 h 변동 방향의 준직교성은 배치 축에서도 유지된다.

## 5. §8.3 관찰 — 블록별 σ(gap)이 모델 간 반대 방향

| | pos | neg_img | neg_txt |
|---|---|---|---|
| 티처 | **3.718** | 3.147 | 3.179 |
| 학생 | **2.094** | 2.666 | 2.746 |

티처는 pos가 18% 크고, 학생은 neg가 31% 크다. 블록마다 온도를 따로 걸 근거가 된다.
티처의 σ(gap)이 전반적으로 1.2~1.8배 크다는 점도 스케일 보정이 필요함을 시사한다.

## 6. ★ 부수 발견 — 티처는 라벨과 훨씬 자주 불일치한다

| | P(gap_pos > gap_neg) | pos인데 gap<0 | neg인데 gap>0 |
|---|---|---|---|
| **티처** | 0.798 | **36.6%** | **18.2%** |
| **학생** | 0.930 | 22.8% | 7.4% |

**티처는 positive의 36.6%를 "no-match"로, negative의 18.2%를 "match"로 판단한다.**
학생보다 훨씬 자주 라벨과 어긋난다.

이는 pair-wise 2-class 증류(arm C / scheme A 계열)가 왜 학생을 망가뜨렸는지를 직접 설명한다 —
쌍 단위로 보면 티처 신호의 상당 부분이 하드라벨과 정면으로 충돌하고, 그 충돌을 확률 믹싱으로 녹이면
학생은 "판단 보류"를 학습한다. 반면 **분포 단위(블록 내 상대 순위)로 보면 그 충돌이 사라진다** —
순위는 라벨과 경쟁하지 않기 때문이다. 배치 단위 증류의 근거가 여기서 나온다.

## 7. 알려진 한계 (설계 결정으로 파킹됨)

- **`itm_head`가 `autocast_ctx` 블록 안에서 호출된다.** 이번 측정은 CPU + `--amp` 미지정이라
  `nullcontext`가 반환되어 **영향 없음**(전 구간 fp32). 다만 `--device cuda --amp`로 재실행하려면
  **반드시 먼저 고쳐야 한다** — `F.linear`는 autocast 대상이라 입력을 `.float()`해도 bf16으로 강등되고,
  gap 오차가 3e-7 → 2.6e-3으로 약 8000배 커진다. `sigma_gap`이 직접 오염된다.
- `test_head_is_applied_in_fp32`는 항상 통과한다(`student_itm_logits`가 `logits.float()`로 승격하므로).
  위 항목과 함께 고쳐야 의미가 있다.
- stub이 대칭(`sim_i2t ≡ sim_t2i`)이라 `weights_t2i`/`weights_i2t` 스왑과 momentum→online 스왑을
  회귀 테스트로 잡지 못한다. 구현 자체는 프로덕션 대조로 정확함이 확인됐다(리뷰 수동 검증).

## 8. 재현

```bash
cd /home/minwoo/Distillation_Project && CUDA_VISIBLE_DEVICES="" \
  /home/minwoo/miniconda3/envs/kd_r4/bin/python \
  critical_bugfix/2026-07-31_itm_batch_logit_raw/itm_batch_logit_probe.py \
  --n-samples 500 --device cpu --num-workers 16
```

**`CUDA_VISIBLE_DEVICES=""`는 필수다.** `data/__init__.py:113`의 `create_loader`가 `pin_memory=True`를
하드코딩하는데, GPU가 학습 잡으로 포화돼 있으면 `--device cpu`로 돌려도 pin-memory 스레드의
CUDA 할당이 OOM 난다. 학습 코드는 읽기 전용이므로 호출 측에서 GPU를 숨긴다.

단위 테스트: `python -m pytest tests/test_itm_batch_logit_probe.py -v` (26개)

## 9. 파일

| 파일 | 내용 |
|---|---|
| `probe_core.py` | 순수 함수 — negative 가중치, 블록 분해, 프레임/통계, CSV, 요약 지표. 모델·데이터셋을 모른다 |
| `itm_batch_logit_probe.py` | 모델 로드, ITM 3B 조립, CLI, 순차 로딩 |
| `csv/{tag}_{block}.csv` | 표 6개. 500행 + mean/var/std |
| `npz/{tag}.npz` | 재분석용 원자료 |
| `summary.json` | §3~§5 수치의 출처 |
