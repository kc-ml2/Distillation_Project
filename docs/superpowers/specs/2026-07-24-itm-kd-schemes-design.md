# ITM 증류 스킴 A/B — 구현 설계 (spec)

**날짜:** 2026-07-24
**브랜치:** `itm_kd_scheme_ab` (off `cc12m_integration`)
**분석 근거:** `critical_bugfix/2026-07-24_itm_distill_sharpness/README.md` (근본원인 + sharpness 실측)
**관련 코드:** `distillation/losses.py`, `distillation/distill_config.py`, `models/blip_pretrain.py`(ITM forward), `pretrain.py`(itm_mix 조립), `configs/`

## 1. 배경 (요약)

arm C(ITM soft target-mix, W=0.4/T=1)에서 `val_retrieval_itm < itc`. 근본원인은 **오보정된 티처 ITM(하드네거를 0.989 match로 오판)을 확률-믹싱으로 증류 → 헤드의 match-vs-hardneg separation 붕괴(1.15→0.64)**. 프로브로 확증(`itm_sharpness_results.jsonl`). 티처 gap: match 7.8 / distractor 4.5 / sep 3.4.

목표: **하드라벨을 타깃에 녹이지 않고**, 티처의 랭킹 지식을 통제된 방식으로 전달해 학생 separation을 올린다. 두 스킴을 실험 config로 추가·검증.

## 2. 설계

### 2.1 공통: `soft_weight`는 "티처 신뢰도" 다이얼 [0,1]
- 0 = 티처 무시(하드라벨만), 1 = 하드라벨 무시(티처만).
- `variant`가 해석을 바꾼다: `target_mix`면 **W**(타깃 믹싱 비중), `hinton_kd`면 **α**(손실 믹싱 비중). 필드 하나 재사용.
- `temp`(T)는 두 스킴 공통 — `OnlineTeacher.itm_soft(temp=T)`가 `softmax(teacher_logits/T)` 반환.

### 2.2 스킴 A — 티처만 temper한 soft-target (config-only, 새 코드 없음)
- 기존 `itm_target_mix_loss`를 **`soft_weight=1.0`(=W, onehot 믹싱 제거) + `temp=2.0`**로 사용.
- 타깃 = `softmax(teacher/T)`, 학생 plain CE → **균형 gap = teacher_gap / T** (통제된 separation).
- 검증됨(`verify_scheme_equilibria.py`): T=2 → match 3.90 / distr 2.25 / **sep 1.65**.
- 신규 config `configs/pretrain_itm_schemeA_tempered.yaml` (arm C 복제, itm 블록 `soft_weight 0.4→1.0`, `temp 1.0→2.0`, output_dir/exp만 변경). `variant` 미지정 → 기본 `target_mix`.

### 2.3 스킴 B — 2텀 Hinton KD (새 손실 + variant 배선)

**손실** `distillation/losses.py`:
```python
def itm_hinton_kd_loss(vl_output, itm_labels, teacher_soft, alpha, temp):
    # loss = (1-alpha)*CE(z, itm_labels) + alpha * T^2 * KL(teacher_soft || softmax(z/T))
    hard = F.cross_entropy(vl_output, itm_labels)
    kd = F.kl_div(F.log_softmax(vl_output / temp, dim=1),
                  teacher_soft.to(vl_output.dtype), reduction='batchmean') * (temp ** 2)
    return (1.0 - alpha) * hard + alpha * kd
```
- `lm_distill_loss`와 동일한 KL·×T² 관례. 하드 CE를 별도 항으로 유지 → 갭 sharpening 안 죽음(균형 gap→teacher gap, gradient는 ×T²로 포화 탈출).
- `teacher_soft`는 반드시 동일 temp로 tempered(itm_soft(temp=T)가 보장).

**config 스키마** — `distill.itm_target_mix`에 `variant` 추가:
```yaml
distill:
  itm_target_mix:
    enabled: true
    neg_source: teacher
    variant: hinton_kd      # 'target_mix'(기본) | 'hinton_kd'
    soft_weight: 0.4        # variant=hinton_kd면 α로 해석
    temp: 2.0               # T
    schedule: constant
```

**배선:**
- `pretrain.py`: `itm_variant = distill_itm.get('variant', 'target_mix')` 읽어 `itm_mix` dict에 `'variant': itm_variant` 추가.
- `models/blip_pretrain.py` ITM forward: `soft_weight>0`일 때 `variant=='hinton_kd'`면 `itm_hinton_kd_loss(vl_output, itm_labels, teacher_soft, soft_weight, temp)`, 아니면 기존 `itm_target_mix_loss`. import에 `itm_hinton_kd_loss` 추가.
- `distillation/distill_config.py` `validate_itm_mix_config`: `variant in ('target_mix','hinton_kd')` assert 추가(기본 target_mix). soft_weight∈[0,1] 그대로.
- 신규 config `configs/pretrain_itm_schemeB_hinton.yaml` (arm C 복제 + 위 itm 블록).

### 2.4 동작 불변 원칙
- `variant` 미지정 시 `target_mix` → arm A/B/C 기존 동작 **바이트 동일**. 스킴 B 손실은 `variant=='hinton_kd'`에서만 진입.

## 3. 인터페이스 / config
신규 키 1개: `distill.itm_target_mix.variant`(default `target_mix`). 나머지 키 재사용. `soft_weight`는 variant에 따라 W/α로 해석(문서화).

## 4. 테스트 계획 (TDD — 구현 전 작성)
1. **`itm_hinton_kd_loss` 단위** (`distillation/test_losses.py`):
   - α=0이면 `F.cross_entropy(vl_output, itm_labels)`와 수치 동일.
   - teacher_soft가 하드라벨과 일치·sharp면 손실이 하드 CE 근처.
   - shape/유한성.
2. **균형 검증** (`verify_scheme_equilibria.py`, 이미 존재): B 경로가 "미구현"→ 실수치. 기대: match gap→teacher 근처, hard-neg는 하드 CE floor로 눌림, sep이 arm C 실측(0.64)·스킴 A(1.65)와 구분됨.
3. **`validate_itm_mix_config`** (`distillation/test_distill_config.py`): variant 유효/무효 통과·거부. 기존 arm A/B/C config 여전히 통과.
4. **forward CPU 스모크** (기존 minilm 스모크 확장): variant='hinton_kd'로 3B forward → 유한 loss, backward OK.

## 5. 검증 순서
단위/스모크 그린 → `verify_scheme_equilibria.py` 재실행(A·B 균형) → (사용자) 학습 몇 에폭 → `itm_sharpness_probe.py`로 **separation + val_retrieval_itm**을 arm B(하드)·arm C(사망) 대비 비교. **게이트: itm − itc > 0 회복 + itc 유지.**

## 6. 범위 밖 (YAGNI)
- α/W의 스케줄·감쇠(constant only).
- logit-MSE 증류 변형(별개 후속).
- 비대칭 temper의 학생-미temper 경로(스킴 A가 이미 그 균형 gap=u/T를 달성하므로 불필요).
- 실제 학습 실행(사용자 담당) 및 arm A/B 스페어 곡선 수집.

## 7. 산출물
- 코드: `itm_hinton_kd_loss` + variant 배선(3파일) + config 2개(A/B).
- 테스트: 위 4종.
- 문서: 본 spec + `critical_bugfix/2026-07-24_itm_distill_sharpness/`(분석·프로브·검증).
