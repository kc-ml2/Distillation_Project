# Teacher-target-mixing ITC distillation — design

**날짜:** 2026-07-14
**브랜치(예정):** `exp9_teacher_target_mix` (dev `fe23681`에서 분기, 새 워크트리 — extra_exp가 아닌 정식 실험 exp9)
**실행:** 스페어 서버(현재의 ~2× 느림), config-gated 기능(기본 OFF)
**상위 맥락:** [`critical_bugfix/2026-07-14_kd_grad_balance_and_geometry/README.md`](../../../critical_bugfix/2026-07-14_kd_grad_balance_and_geometry/README.md) §6.2

---

## 1. 동기

Phase 1(별도 KD 손실 + ×τ + λ 튜닝)은 **KD와 ITC가 별개 손실**이라 두 gradient가 방향으로 싸운다
(측정 B: cos(∇L_kd, ∇L_ita) < 0 전 구간). λ로 *약화*할 뿐 구조적 충돌은 남는다.

**본 실험의 가설:** 티처를 별도 손실이 아니라 **ITC의 soft-target 안에 녹이면** 손실이 하나가 되어
**gradient 충돌이 정의상 소멸**한다. 이게 성능을 올리면 "충돌이 병목이었다"가 증명되고, 이 방향이
ITC-distillation의 본선이 된다.

**성공 판정:** ep20 `val_retrieval_itc/r_mean` ≥ 63.4(student solo, 필수), 목표 66+(base). 부차로
Phase 1 최고 arm 초과 여부. (`model/logit_scale`은 참고로 로깅하되 판정 기준에선 제외.)

---

## 2. 핵심 아이디어 — 티처를 ITC 타깃의 soft-slot에 배치

현재 BLIP ITC 타깃 (`models/blip_pretrain.py` forward):
```
target = (1−α)·one-hot + α·softmax(momentum_sim)          # [B, B+queue]
loss_ita = CE(student_logsoftmax[B,B+queue], target)
```
α는 첫 2에폭 램프(0→0.4). momentum을 램프하는 이유는 **momentum 인코더가 초반엔 미숙**(학생 EMA)이라
못 믿어서다. **티처(BLIP-large, frozen)는 step 0부터 신뢰 가능** → momentum이 비워둔 그 자리를 티처가 채운다.

**새 타깃 (soft-slot 하나를 티처↔momentum가 시간에 따라 나눠 가짐):**
```
target = (1−W)·one-hot + W·[ γ(t)·teacher_soft + (1−γ(t))·momentum_soft ]

  W = soft_weight = 0.4  (one-hot = 0.6 상수)
  teacher_soft  = softmax(teacher_sim / τ)      # 티처 분포 (variant별 아래)
  momentum_soft = softmax(momentum_sim)          # 기존 momentum 분포 (safe_scale 적용)
```
- 손실은 여전히 **CE 하나**. 티처는 타깃(no_grad 고정분포)에만 들어가므로 **별도 gradient 없음 → 충돌 소멸**.
- 유효 무게: teacher = 0.4·γ, momentum = 0.4·(1−γ), one-hot = 0.6. 항상 합 1, 모두 ≥0.
- **후반(γ=0) 타깃 = 0.6·one-hot + 0.4·momentum = baseline과 완전 동일** → 6.1식 후반 손해 구조적 불가능.
- 이 실험에서 **기존 α 램프는 미사용**: soft-slot을 0.4 고정으로 두고 γ가 시간 배분을 담당(α의 역할을 티처가 대체).

**중요(양쪽 variant 공통):** 학생 로짓의 후보 열은 momentum 후보군(batch momentum + momentum queue)이다.
soft-target의 momentum·teacher 두 성분 모두 **이 동일 후보군 위**에 정의돼야 CE가 유효하다.

**양방향 대칭:** ITC는 i2t·t2i 두 손실(`sim_i2t_targets`, `sim_t2i_targets`)이다. 티처 항도 **양쪽 모두**에
동일하게 들어간다(기존 momentum이 양방향 대칭인 것과 동형). i2t 타깃엔 `softmax(teacher_img·teacher_txt_all/τ)`,
t2i 타깃엔 `softmax(teacher_txt·teacher_img_all/τ)`. variant D는 in-batch `[B,B]`와 그 transpose, variant C는
각 방향의 teacher_{img,txt}_all(batch+teacher큐)을 사용.

---

## 3. 두 variant (`variant: in_batch | queue`)

teacher_soft를 어느 후보군 위에 만드느냐가 유일한 차이.

### variant D — `in_batch` (최소 변경, 큐 없음)
- 이번 배치 티처 피처 `[B,256]`로 `teacher_sim = teacher_img_feat @ teacher_text_feat.t()` → `[B,B]`.
- `softmax(teacher_sim/τ)` → **queue 열은 0으로 패딩**하여 `[B, B+queue]`로 확장.
- 티처는 in-batch 랭킹만 형성(어차피 티처가 아는 게 그것뿐). 새 인프라 0.

### variant C — `queue` (teacher-feature queue, 리치)
- momentum 큐와 **대칭인 티처 큐** `teacher_image_queue`, `teacher_text_queue` (`[256, 57600]`) 추가.
- 매 스텝 momentum enqueue와 **같은 포인터·같은 순서로 티처 피처도 페어 enqueue** (online_teacher가
  이미 계산한 배치 티처 피처 재사용, 추가 forward 0).
- **정렬 필수:** 큐 열 j는 momentum 큐·teacher 큐에서 동일 item(같은 이미지/텍스트를 momentum vs 티처
  인코더로 본 것)이어야 두 soft 성분의 열이 일치. 어긋나면 티처 신호가 엉뚱한 후보에 붙음.
- `teacher_text_all = [teacher_text_feat | teacher_text_queue]` → `teacher_sim = [B, B+queue]` →
  `softmax(/τ)`. queue 패딩 불필요.
- 티처 frozen이라 큐 피처가 시간에 따라 안 드리프트(momentum 큐보다 정합적). 비용: 버퍼 2개(~118MB fp32, 무시).

**정렬 구현 노트:** 티처 enqueue는 `_dequeue_and_enqueue`와 동일 ptr을 공유해야 한다. 기존
`_dequeue_and_enqueue(image_feat_m, text_feat_m)`를 확장해 티처 피처도 같은 ptr 위치에 쓰거나,
동일 로직의 페어 enqueue를 한 함수에서 처리한다(ptr 두 번 전진 금지).

---

## 4. γ 스케줄 — 2에폭 홀드 + 10에폭 선형 감쇠

`base 모델이 ~450k step(≈ep12)에서 평탄화`한다는 관찰 기반. 초반 불안정 구간엔 티처를 100% 태워
"학생이 티처를 잘 보도록" 학습시키고, 이후 선형으로 넘긴다.

```
hold_end  = 2  epoch (74,692 step)
decay_end = 12 epoch (448,152 step)

step < hold_end                 : γ = 1
hold_end ≤ step < decay_end      : γ = 1 − (step − hold_end)/(decay_end − hold_end)
step ≥ decay_end                : γ = 0
```

| epoch | γ | teacher(0.4γ) | momentum | one-hot |
|---|---|---|---|---|
| 0–2 (홀드) | 1.0 | 0.40 | 0.00 | 0.6 |
| 3 | 0.9 | 0.36 | 0.04 | 0.6 |
| 5 | 0.7 | 0.28 | 0.12 | 0.6 |
| 7 | 0.5 | 0.20 | 0.20 | 0.6 |
| 9 | 0.3 | 0.12 | 0.28 | 0.6 |
| 11 | 0.1 | 0.04 | 0.36 | 0.6 |
| 12–20 | 0.0 | 0.00 | 0.40 | 0.6 |

- 감쇠 구간 에폭당 티처 −0.04 균일.
- γ는 α와 동일하게 **매 스텝** 계산해 forward로 전달(에폭 단위 아님).
- 곡선 모양은 선형 고정(초반 hold가 "초반 불안정 극복" 의도를 담음). 필요시 hold/decay 경계만 config로 조정.

---

## 5. Config 스키마

`distill` 블록에 신규 `itc_target_mix` 추가. 기본 OFF(미설정 시 baseline 동작 완전 동일).
별도 KD(`distill.itc.enabled`)와 **상호배타**(본 실험은 target_mix ON, 별도 KD OFF).

```yaml
distill:
  itc:
    enabled: false          # 별도 KD 손실 OFF (target_mix와 동시 사용 금지)
  itc_target_mix:
    enabled: true
    variant: queue          # 'in_batch'(D) | 'queue'(C)
    temp: 0.05              # τ, 티처 softmax 온도 (타깃 sharpness) — 근거는 아래 주석
    soft_weight: 0.4        # W, soft-slot 무게 (one-hot = 1−W). 기본 α_max와 일치
    hold_epochs: 2          # γ=1 유지 구간
    decay_end_epochs: 12    # γ→0 도달 (base 평탄화 ~450k step)
teacher:
  arch: 'blip_large'
  checkpoint: '.../model_large.pth'
```
- `soft_weight`, `hold_epochs`, `decay_end_epochs`, `temp`, `variant`는 향후 스윕용으로 노출.
- **τ=0.05 근거(mixing 맥락은 Phase 1과 다름):** Phase 1(별도 KD)은 티처가 타깃 전부라 sharp τ0.022가 맞았지만,
  여기선 one-hot(0.6)이 positive를 담당하므로 티처는 **negative 랭킹(dark knowledge) 전달**이 역할 → 더 soft해야 함.
  최종 혼합 타깃(초반 γ=1: `0.6·one-hot + 0.4·teacher`, variant D 기준) 계산: τ0.022→positive질량 0.80·유효neg ~3(dark
  knowledge 거의 없음), **τ0.05→0.69·~18(적정)**, τ0.07→0.66·~27(dark↑ 노이즈꼬리↑), τ0.1→0.64·~33(측정 D의 VG
  노이즈 영역까지). one-hot 앵커라 positive는 어느 τ든 0.64~0.80로 안전 → soft로 갈 여유 있으나, 측정 D(작은 VG 63%
  오답)로 상한 존재. **0.05가 창의 중앙**. τ0.07은 첫 스윕 후보. variant C(큐 57640 후보)는 같은 τ라도 in-batch보다
  자연히 flat → 첫 런은 둘 다 0.05, C가 과도히 평평하면 C만 τ 하향(후속).
- α override(순수 교체 = α=0 특수케이스)는 본 설계에서 soft_weight/γ로 이미 표현되므로 별도 키 불필요.
- 검증(assert): `target_mix.enabled`와 `itc.enabled` 동시 true 금지; `0 ≤ soft_weight ≤ 1`; `hold ≤ decay_end`.

---

## 6. 코드 변경 지점 (구현 계획의 밑그림, 상세는 writing-plans에서)

1. `models/blip_pretrain.py`
   - `__init__`: variant=='queue'면 teacher 큐 버퍼 등록(register_buffer, momentum 큐와 동형).
   - `forward`: `gamma`, target_mix 설정, teacher 피처를 받아 soft-target 계산 분기.
     target_mix OFF면 기존 α 경로(무변경). ON이면 §2 타깃 사용.
   - `_dequeue_and_enqueue`: variant=='queue'면 티처 피처도 동일 ptr에 페어 enqueue.
2. `pretrain.py`
   - train loop: `gamma` 매 스텝 계산(§4), forward에 전달. teacher 피처는 이미 online_teacher로 계산 중.
   - TB: 기존 `train/alpha` 옆에 **`train/beta`(=0.4·γ, 티처 무게)** 추가 (요청사항). `train/gamma`도 병기.
   - config 파싱 + assert(§5).
3. config 2개: `pretrain_ttm_inbatch.yaml`(variant in_batch, exp `9.1.ttm_in_batch`),
   `pretrain_ttm_queue.yaml`(variant queue, exp `9.2.ttm_queue`). output_dir도 각각 분리.
   나머지는 student baseline(exp7)과 동일 + caption-val ON + SPICE CPU 분리(잡A 0-31 / 잡B 32-63).

---

## 7. 실험 계획

- **2런**: D(in_batch) / C(queue), 둘 다 τ0.05·soft_weight0.4·hold2·decay12.
- 스페어 서버(2× 느림): 순차(D 먼저 ~6일) 또는 MPS 동시(~12일). 스케줄은 실행 시 결정.
- 비교 기준: baseline(exp7 solo 63.4), Phase 1 arms(진행 중), 상호(C vs D).
- **진단 우선순위**: (a) ep20 r_mean ≥ 63.4 통과 + Phase 1 최고 arm 초과? (b) C가 D보다 나은가(queue 리치함의
  실익)? (c) train/beta 감쇠(홀드→감쇠 구간)에 따라 r_mean 궤적이 이득을 유지하나 소실하나.
- borderline이면 후속 스윕 변수: `soft_weight`(초반 티처 강도), `decay_end_epochs`(티처 지평).

---

## 8. 범위 밖 (YAGNI)

- teacher-target-mixing과 별도 KD 동시 사용(충돌 재도입이라 무의미).
- β/τ 초기 스윕(우선 2런 후 필요시).
- γ 비선형 곡선(cosine 등) — 선형으로 시작.
- teacher 큐와 momentum 큐 분리 크기/ptr(동일 크기·정렬 유지가 설계 전제).

---

## 9. 테스트

- 단위: γ 스케줄 함수(hold/decay 경계값, 매 스텝 단조), target 무게 합=1·비음수(임의 γ), variant D 패딩
  형상([B,B+queue], queue 열 0), variant C 페어 enqueue 정렬(같은 ptr).
- 통합: target_mix OFF일 때 forward 출력이 baseline과 bit-identical(회귀 방지). 짧은 스모크(수 스텝)로
  두 variant가 크래시 없이 loss 감소 + TB에 train/beta 기록 확인.
