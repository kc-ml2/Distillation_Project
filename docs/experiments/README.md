# 실험 네이밍 규칙 + 구이름↔신이름 별칭

**작성:** 2026-07-31
**적용 범위:** `tensorboard_output/` 폴더명. ITM 계열(exp10.*)만 소급 적용했고, 이후 실험은 아래 규칙을 따른다.
**미적용:** `output/` 디렉토리명, 이미 생성된 TB run 디렉토리명, `configs/` 파일명 — 전부 구이름 그대로다.

---

## 1. 왜 바꿨나

ITM 실험에 **A/B/C가 두 세트** 있었다.

- `arm A/B/C` — 2026-07-22 설계. neg_source × hard/soft 조합
- `스킴 A/B` — 2026-07-24 설계. arm C 실패 후의 대안

이름이 충돌할 뿐 아니라, **`arm C`와 `스킴 A`는 사실 같은 손실 함수**(`itm_target_mix_loss`)를 W/T만 바꿔 쓴 것인데 이름이 "arm"과 "scheme"으로 갈려 서로 다른 로직처럼 보였다.

새 규칙은 **문자를 손실 함수(코드 경로)에 묶어서** 이 문제를 없앤다. 같은 함수를 쓰면 같은 문자다.

---

## 2. 네이밍 규칙

```
exp<N>[.<L><k>]_<신호>_<방식>
```

| 필드 | 의미 | 규칙 |
|---|---|---|
| `N` | 실험 계열 번호 | append-only. 한번 쓴 번호는 재사용 금지 |
| `L` | 손실 함수 문자 | **그 계열 안에서만 정의.** 같은 코드 경로 = 같은 문자 |
| `k` | 하이퍼 변형 번호 | 1부터. 손실은 같고 W/λ/T/neg만 다르면 증가 |
| `신호` | `itc` / `itm` / `lm` | 조합은 하이픈 (`itc-lm`) |
| `방식` | 손실 슬러그 | 소문자 |

변형이 하나뿐이면 `.<L><k>` 생략 가능 (`exp8_lm_distill`).

디렉토리 계층: `tensorboard_output/{target|auxiliary|deprecated}/{base|student}/<이름>/`

| tier | 뜻 |
|---|---|
| `target` | 주력 |
| `auxiliary` | 보조 — 실패했지만 진단 가치가 있는 것 포함 |
| `deprecated` | 폐기 |

### exp10 (ITM) 문자 어휘

| 문자 | 손실 함수 | 슬러그 |
|---|---|---|
| `M` | `itm_target_mix_loss` | `tgtmix` |
| `H` | `itm_hinton_kd_loss` | `hintonkd` |
| `G` | (미구현) gap-MSE | `gapmse` |

---

## 3. 별칭표 — exp10 (ITM)

| 신 이름 | 구 `exp:` 태그 | 구 호칭 | `output/` 디렉토리 | neg | W | T | ckpt | 결과 |
|---|---|---|---|---|---|---|---|---|
| `exp10.M1_itm_tgtmix` | `10.C_itm_teachneg_soft` | **arm C** | `pt_itm_C_teachneg_soft` | teacher | 0.4 | 1.0 | 5 (ep0–4) | 실패, r_mean 59.7 |
| `exp10.M2_itm_tgtmix` | `10.SA_itm_tempered_soft` | **스킴 A** | `pt_itm_schemeA_tempered` | teacher | 1.0 | 2.0 | 20 (완주) | 실패, r_mean 62.3 (rerank +0) |
| `exp10.M3_itm_tgtmix` | `10.SA2_itm_tempered_studentneg` | **SA2** | `pt_itm_schemeA_studentneg` | student | 1.0 | 2.0 | 진행 중 | — |

셋 다 손실은 `itm_target_mix_loss` 하나다. 즉 **같은 손실로 하이퍼만 바꿔 세 번 시도**했다는 뜻.

### 기록만 남기는 것

| 구 호칭 | 상태 | 비고 |
|---|---|---|
| `10.SB_itm_hinton_kd` (**스킴 B**) | 중단, ckpt 0 | 14분 실행 후 정지. 신 ID 미부여 |
| **arm A** (`pretrain_itm_A_studneg_soft.yaml`) | **미실행** | config만 존재 |
| **arm B** (`pretrain_itm_B_teachneg_hard.yaml`) | **미실행** | config만 존재. 하드라벨 조건은 `exp7_student_baseline`이 이미 커버 |

**기준선:** ITM 하드라벨 성능은 별도 ITM run이 아니라 `exp7_student_baseline` (itm 70.9 / itc 63.4).

### tensorboard_output 이동 결과

```
target/student/exp10.C_itm_teachneg_soft  →  auxiliary/student/exp10.M1_itm_tgtmix
target/student/exp10.SA_itm_tempered      →  auxiliary/student/exp10.M2_itm_tgtmix
```

M3는 실행이 끝난 뒤 `auxiliary/student/exp10.M3_itm_tgtmix`로 넣는다.
⚠ 원본 `output/pt_itm_schemeA_studentneg/tensorboard/` 밑에 이벤트 디렉토리가 **2개**다 — `20260730_142211`(헛발), `20260730_142348`(본런). **후자만** 복사할 것.

---

## 4. 구이름을 쓰는 문서

아래 문서들은 "arm C", "scheme A" 등 구 호칭으로 쓰여 있다. 읽을 때 §3 표로 대응시킨다. (내용 수정은 하지 않았다 — 이미 쓰인 서술을 유지하기 위해.)

**조사 문서**
- `critical_bugfix/2026-07-24_itm_distill_sharpness/` — arm C 실패 원인 + 스킴 A/B 설계
- `critical_bugfix/2026-07-27_itm_eval_ranking_and_structure/` — 스킴 A 실패 확정 + ITM 증류의 구조적 한계
- `critical_bugfix/2026-07-30_itm_logit_offset_and_head_geometry/` — 헤드 기하 + 지표 감사

**설계 문서**
- `docs/superpowers/specs/2026-07-24-itm-kd-schemes-design.md`
- `docs/superpowers/plans/2026-07-22-itm-target-mix.md`
- `docs/superpowers/plans/2026-07-24-itm-kd-schemes.md`

**코드/config** — 브랜치 `itm_kd_scheme_ab`에만 존재 (dev에는 없음)
- `configs/pretrain_itm_{A_studneg_soft,B_teachneg_hard,C_teachneg_soft,schemeA_tempered,schemeB_hinton}.yaml`
- `critical_bugfix/2026-07-27_.../probe_schemeA.py`, `schemeA_probe_results.json`

---

## 5. 지표 읽을 때 주의

ITM 관련 숫자가 세 층으로 갈리고, 서로 비교하면 안 된다.

| 지표 | 무엇 | 주의 |
|---|---|---|
| `train_loss_itm` | 학습 손실 | **스킴 간 비교 불가.** M2(W=1.0)는 타깃이 티처 확률이라 손실에 `H(teacher)` floor가 깔린다 (티처 gap 7.8/4.5 기준 대략 0.24). CE run과 곡선 높이를 나란히 놓으면 오해한다 |
| `val_retrieval_itm/r_mean` | **게이트 지표** | R@1/5/10 평균 = 랭킹 적중률. `eval_validation_tool.py:200-230`. loss와 무관 |
| per-pair 이진 정답률 | 진단 전용 | 하드네거에서 degenerate — baseline 52.0 / M2 50.1로 둘 다 동전던지기라 품질 구분 불가 (`2026-07-27` §6b) |

`separation` 원값(match_gap − distractor_gap)은 **모델 간 비교에 쓰면 안 된다** — `‖w_d‖`에 오염돼 있다 (티처 3.34 vs baseline 1.26). 올바른 지표는 within-query `d′` (`2026-07-30` §4).
