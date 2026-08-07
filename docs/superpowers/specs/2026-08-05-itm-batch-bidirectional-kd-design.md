# ITM 배치 단위 양방향 소프트맥스 증류 — 설계

**작성:** 2026-08-05
**상태:** ⚠️ **폐기** — `2026-08-07-itm-bxb-matrix-kd-design.md`(전체 B×B 매트릭스)로 대체. 이 스펙은 pos/neg_img/neg_txt 3B 블록을 **sample 축**으로 softmax했으나, 올바른 축은 이미지 i vs 전 텍스트의 **retrieval 축(B×B)**이었다. 이력 보존용.
**선행:** `docs/superpowers/specs/2026-07-31-itm-batch-logit-probe-design.md` (측정), `critical_bugfix/2026-07-31_itm_batch_logit_raw/` (실측)
**베이스 브랜치:** `dev`에서 새 분기 (예: `dev/itm_batch_bidir_kd`). 기존 pair-wise arm(arm C / scheme A·B)은 계승하지 않는다.

---

## 1. 왜 하나 — 프로브가 확정한 것

pair-wise 2-class ITM KD는 쌍마다 로짓 2개를 받고 2-class softmax가 `gap = z2 − z1` 만의 함수라 **실질 자유도 1**이다. arm C(확률 target-mix)는 이 신호를 하드타깃에 녹였다가 head separation을 1.15→0.64로 붕괴시켰다(itm<itc). 프로브(`2026-07-31`)는 세 가지를 실측으로 확정했다:

1. **블록 분리 필수 (Axis B 확정).** Δ = mean(gap_pos) − mean(gap_neg) = 4.12(티처)/4.98(학생). 3B 통합 softmax는 확률질량 96~99%를 pos가 독식 → gradient 대부분이 "pos/neg 경계 벌리기"(하드라벨 CE가 이미 하는 일)에 실린다.
2. **티처 m ≈ 상수** (σ(m)=0.006~0.007). row 단위 매칭이 성립하는 동시에, m에는 전달할 dark knowledge가 거의 없다.
3. **티처는 라벨과 자주 불일치**(pos의 36.6%를 no-match, neg의 18.2%를 match)하지만, **블록 내 상대순위로 보면 그 충돌이 사라진다** — 순위는 라벨과 경쟁하지 않는다. 배치 단위 증류의 근거이자 arm C 실패의 직접 원인.

이 스펙은 프로브 §9가 "다음 스펙에서"로 파킹한 **증류 손실 설계**다.

## 2. 무엇을 만드나 — 6개 KL

블록 `b ∈ {pos, neg_img, neg_txt}` (각 B-way), 각 블록에서 z1(no-match)·z2(match) 두 로짓 행을 **각각** softmax → forward KL. 총 6개.

$$\mathcal{L}_{itm\text{-}kd} = \frac{1}{6}\sum_{b}\sum_{k\in\{z_1,z_2\}} \mathrm{KL}\!\Big(\sigma\big(z_k^{T,b}/\tau_{T,b}\big)\,\Big\|\,\sigma\big(z_k^{S,b}/\tau_{S,b}\big)\Big)$$

| 축 | 결정 | 근거 |
|---|---|---|
| Axis B (분포 범위) | 블록 분리, B-way softmax ×3 | 프로브 §8.1, Δ≳2 |
| Axis A (무엇을 매칭) | **양방향** — z1, z2 각각 (블록×2=6 KL) | §3.1 아래 |
| KL 방향 | forward `KL(T‖S)` | 기존 관례(itc/lm과 동일) |
| 온도 | `τ = c·σ(z)_{block,model}`, **c=1** | §3.2 |
| 정규화 | 6개 KL 평균(×1/6) | λ=1을 단일 CE 오더에 맞춤 |
| λ (총손실 결합) | 1 (비율 보고 하향 가능) | 관례 |

### 3.1 왜 양방향 (단측 gap-only 아님)

- 블록 내 softmax는 **shift-invariant** → gap의 절대 오프셋/부호는 무의미, **블록 내 산포 σ만** 분포를 결정. (Δ는 블록 간 평균차라 통합에서만 문제; 블록 분리 후엔 사라진다.)
- m ≈ 상수라 `z2 = m + gap/2`, `z1 = m − gap/2` → `softmax(z2) ≈ softmax(gap/2)`(정방향), `softmax(z1) ≈ softmax(−gap/2)`(역방향). **z1,z2 둘 다 = gap 정/역방향** (온도 2배 차이).
- 정방향은 큰 gap(쉬운 매치)에 질량이 몰려 **하드 샘플(음수 gap)을 굶긴다**. 역방향이 그 하드 샘플에 gradient를 실어준다 — 프로브 §6의 티처-라벨 불일치(하드 네거티브 18.2% / 하드 포지티브 36.6%)가 정확히 이 tail에 산다. 단측이면 이 dark knowledge가 묻힌다.
- 라벨 경계를 건드리지 않고 블록 내 순위로만 들어가므로 arm C의 separation 붕괴를 재현하지 않는다.
- 구현은 z1,z2 원출력 직접 증류(파생량·부호반전 없음, 대칭 자동). gap±와 동치이며 무해한 약한 m-정규화만 추가된다.

### 3.2 온도 τ = c·σ 의 근거

- softmax의 실효 sharpness는 τ가 아니라 **τ/σ** 하나로 결정된다. τ=c·σ로 두면 softmax 입력이 z-score(`ĝ = (g−ḡ)/σ`, 단위분산)가 되어 **스케일 프리 랭킹만** 전달. σ(gap)이 티처/학생/블록마다 다른 것(§8.3: 티처 pos 큼, 학생 neg 큼, 대소 역전)은 nuisance 스케일이므로 이렇게 제거한다.
- **참여비(유효 support)** 유도: `PR = (Σw)²/Σw²`, `w=e^{z/c}`, `z~N(0,1)`, 대B 집중 + 가우시안 MGF → **`PR = B·e^{-1/c²}`**. c=1 → `B/e ≈ 15`. 프로브 실데이터 검증: 배치별(B≈38.5) 실측 PR = 13~21(가우시안 예측 14.1과 정합, 실데이터가 약간 더 퍼짐).
- **c 스윕 범위**: c≤0.5는 argmax 붕괴(PR<1), c≥2는 L2 붕괴(PR→B). 실효 구간 c∈[0.8,1.5], **기본 c=1**, 스윕 {0.8, 1, 1.3}.
- **왜 softmax인가 (선형 정규화 `g/Σg` 기각)**: gap은 부호 있는 값 → 음수 "확률"로 KL의 log 정의 불가, `Σg→0`에서 `1/S²` gradient 폭발, shift-민감(원하는 불변성의 반대)+곱셈 스케일 무통제 제거. softmax는 shift-불변·유계 gradient(`∂KL/∂z = (p^S−p^T)/τ`)·유효 PMF를 동시에 준다. entmax/sparsemax는 escape hatch로만 기록(지금 구현 안 함).

### 3.3 σ 계산과 scale-invariance (arm C 안전장치)

- σ는 **매 배치에서 블록별 std**로 계산(스테이트리스, `itc_distill_loss`처럼 순수함수). B=40 std 추정 노이즈 ~11% → 초기 기본은 per-batch; 노이즈가 문제면 블록별 EMA(momentum≈0.9)로 승급(플래그).
- **학생 σ는 detach하지 않는다.** σ_S를 그래프에 두면 손실이 `z^S → α z^S` 스케일링에 **완전 불변** → KD가 학생 gap의 절대 스케일(=head separation)에 gradient 0. 스케일/라벨은 CE 소유, 순위만 KD 소유 → 둘이 스케일을 두고 싸우지 않음 = **arm C 붕괴를 수식적으로 차단**. (LayerNorm이 std로 나누는 것과 동일한 in-graph 연산, 미분 안정.)
- 티처 로짓은 상수(no_grad)라 σ_T는 자동으로 detached.

## 4. 구현 위치 — 기존 패턴 재사용

세 파일만 손대고, 학습 loop 구조는 `itc`/`lm` KD와 동형.

### 4.1 `distillation/losses.py` — 손실 (순수함수 신설)

```
def itm_bidir_kd_loss(student_itm_logits, teacher_itm_logits, block_size, c=1.0):
    # student_itm_logits, teacher_itm_logits: [3B, 2] fp32, 순서 [pos, neg_img, neg_txt]
    # teacher는 상수(detach), 블록별 z1/z2 softmax(τ=c·σ) forward KL, 6개 평균
```
- 반드시 **fp32**에서 softmax(gap은 로짓 차라 bf16이면 유효숫자 소실 — 프로브 §7). 입력에서 `.float()` 승격.
- 티처 입력 `.detach()`. 학생 σ는 in-graph(§3.3).

### 4.2 `distillation/online_teacher.py` — `OnlineTeacher.itm_logits()`

```
def itm_logits(self, image, caption, neg_idx_t2i, neg_idx_i2t) -> Tensor[3B, 2]:
    # 주어진 neg_idx로 [pos, neg_img, neg_txt] 3B 쌍 조립 후 teacher text_encoder + itm_head
    # 티처는 스스로 네거티브를 뽑지 않는다(neg_idx는 학생 것 재사용).
```
- 조립 로직은 `critical_bugfix/2026-07-31_itm_batch_logit_raw/itm_batch_logit_probe.py`의 3B 조립을 포팅(재구현 금지, 프로브가 프로덕션 대조로 검증된 경로).
- itm_head는 autocast 밖 fp32.

### 4.3 `models/blip_pretrain.py::forward` — 노출만

- itm_kd 켜졌을 때: 샘플된 `neg_idx_t2i, neg_idx_i2t`(라인 456/464)와 학생 `vl_output`([3B,2])를 반환 튜플에 추가. **티처와 직접 결합하지 않음**(디커플).
- 반환 시그니처 확장: `..., loss_itc_kd, loss_lm_kd, student_itm_logits, itm_neg_idx` (itm_kd off면 None).

### 4.4 `pretrain.py` — 배선

- `teacher_keep` 튜플에 `'itm'` 추가. `distill.itm.enabled` config gate.
- 호출 순서(itc/lm과 다른 점): **model forward 후** `teacher_itm = online_teacher.itm_logits(image, caption, *itm_neg_idx)` → `loss_itm_kd = itm_bidir_kd_loss(student_itm_logits, teacher_itm, B, c)` → `loss = loss_ita + loss_itm + loss_lm + λ·loss_itm_kd`.
- TB 로깅 `loss_train/itm_kd`, 초반 `loss_itm_kd/loss_itm` 비율 모니터.

### 4.5 config

```
distill:
  itm:
    enabled: false      # 기본 OFF (게이트)
    c: 1.0              # 온도 상수 τ=c·σ
    lambda: 1.0
    sigma_ema: false    # true면 블록별 σ EMA
```

## 5. 티처가 같은 3B를 본다 (정합성)

- 학생 forward가 자신의 momentum ITC(`weights_t2i/i2t`)로 neg_idx를 샘플 → 그 인덱스를 티처가 **그대로 재사용**. 티처는 momentum 불필요(샘플링 안 함).
- 이미지 전처리는 두 모델 공통(normalize 하나). 캡션 토크나이즈는 각 tokenizer(vocab 동일).
- 블록 순서 `[pos, neg_img, neg_txt]`를 학생/티처 동일하게 조립(프로브 검증 순서).

## 6. 성공 기준

1. **itm retrieval ≥ baseline** (증류 안 한 학생 대비). 최소 비열화, 목표 개선.
2. **head separation 유지** — arm C처럼 붕괴(1.15→0.64) 안 함. §3.3 scale-invariance가 구조적 방어이나 실측으로 확인.
3. **c 어블레이션** {0.8,1,1.3}에서 단조적/해석 가능한 반응.
4. loss_itm_kd가 CE를 삼키지 않음(비율 모니터, 필요시 λ↓).

## 7. 범위 밖 (non-goals)

- entmax/sparsemax (softmax가 측정 가능하게 실패하기 전엔 안 함).
- eval 랭킹 방식(z2 vs gap) 변경 — 별개 질문.
- ITC/LM KD 손실 변경 — itm만 추가.
- arm C / scheme A·B 계승 — 폐기.

## 8. 리스크 / 열린 질문

- **티처 itm forward 비용**: 학생 forward 후 티처 text_encoder cross-attn 1회 추가. itc/lm 티처 forward와 합쳐 스텝당 비용 상승 — 실측 필요(스모크에서 step time).
- **per-batch σ 노이즈**(B=40, ~11%): 학습 불안정하면 EMA 승급.
- **in-graph σ_S 안정성**: LayerNorm류지만 블록 std가 매우 작아지는 초기(head 미분리)에 τ→0 위험 → σ에 하한 clamp(예: 1e-2) 필요할 수 있음. 스모크에서 관찰.
- **neg_idx 왕복 리팩터**: forward가 neg_idx를 반환하도록 하는 변경이 DDP/기존 반환 처리와 충돌하지 않는지 확인(테스트).
