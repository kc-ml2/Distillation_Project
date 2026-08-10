# Refactor: 최소 본선 브랜치 (ITC 불안정성 격리)

**날짜**: 2026-08-10
**브랜치(예정)**: `refactor/minimal_mainline` (dev 분기)
**관련**: teacher-target-mix-exp9, itc-itm-lm-targetmix-merge, itm-distillation-plan(itm_bxb)

## 1. 배경 & 목표

현재 코드는 여러 증류 방식이 config 토글로 얹혀 있지만 **주력으로 실제 학습에 쓰는 경로는 3개뿐**이고, 나머지는 "켤 수 있으나 안 쓰는" 죽은 토글이다. 동시에 **ITC loss 불안정성**(주기적 급락→지표 손상)이 지속 관찰된다:

- holdhalf(γ decay_end=42): ~500k 스텝 결함
- holdteacher(decay_end=1e7, γ≈1 상시): 주기적 2회 급락→회복→추락
- 일반 itc_lm(decay_end=12): ep12(γ→0 시점) 결함
- itc-only: 간헐 드롭
- baseline(외부 증류 안 붙인 모멘텀 분배)에도 text-side에 '작은' 불안정 존재. 단 ttm 불안정이 훨씬 큼.

**이 급락들의 원인은 아직 미상 — 단정 금지.** 순간적·심각한 드롭은 정상 학습 동역학으론 잘 안 나오고 **미묘한 구현 버그**를 강하게 시사한다(현재 가장 유력한 방향). 원인 규명은 task 3이며 이 문서에서 결론짓지 않는다.

**목표**: 죽은 토글을 걷어내 forward/loss 로직을 주력 3-경로만 남긴 **최소 본선 브랜치**를 만든다. 그래야 (a) 코어 ITC를 눈으로 격리해 읽을거리를 줄이고, (b) 이후 모듈화·디버그의 깨끗한 출발점을 얻는다.

**이 문서의 범위**: 최소 브랜치의 KEEP/STRIP/PORT 정의(= task 1). **모듈화(재구획화)는 범위 밖**(task 2, 최소 브랜치 나온 뒤 코드-프리 논의). **불안정성 root-cause도 범위 밖**(task 3).

## 2. 주력 3-메커니즘 (현재 구현 위치)

| 메커니즘 | 방식 | 코드 위치 |
|---|---|---|
| **itc** | ttm(queue): `0.6·onehot + 0.4·(γ·teacher_soft@0.05 + (1−γ)·momentum_soft@safe_scale)`, in-batch:queue 전체 | dev `distillation/target_mix.py` + forward |
| **lm** | traditional KD: 하드 CE + `lm_distill_loss`(KL teacher‖student @T=2 ×T²) | dev `distillation/losses.py` |
| **itm** | traditional KD, k=4 contrastive: 하드 CE + `loss_itm_kd`(gathered top-k=4, bidir) | **`itm_bxb_matrix_kd` 브랜치** (dev엔 없음) |

itc-ttm 타깃 공식은 사용자 의도(하드 + 티처soft + 모멘텀soft 가중 1개 타깃)와 **정확히 일치함**을 코드로 확인했다.

## 3. 리팩터 방침

**베이스 = dev** (itc-ttm + lm-KD 이미 통합). 여기서 죽은 토글을 STRIP하고 itm k=4를 itm_bxb에서 PORT한다.

**방침: 덜어내기는 최대한, 변환은 최소.**
- *덜어내기 최대*: 읽을거리를 줄이는 게 목적. 죽은 토글뿐 아니라 주석처리된 deprecated 블록까지 삭제.
- *변환 최소*: 파일/함수 **재배치·재구조화 없음**. 삭제 + itm 이식만, 코드 이동 없음. (보기좋게 나누는 재구획화는 task 2에서 — 진짜 필요한 모듈만 남긴 뒤에 결정.)

**토글 철학**: config 키(인터페이스)는 유지하되, 주력 아닌 값은 **지원 코드가 없어 동작 안 함**. 미지원 값은 조용한 no-op이 아니라 **assert/에러로 즉시 실패**(디버그 중 오설정 방지).

### 3.1 KEEP (주력)
- **itc-ttm(queue)**: `target_mix.py`의 `ttm_gamma`/`teacher_soft_queue`/`mix_target`/`enqueue_all`; forward의 ttm queue 분기 + 티처 큐(`teacher_image_queue`/`teacher_text_queue`); pretrain의 γ 스케줄 배선.
- **ttm-OFF 폴백**: forward의 비-ttm 경로(`alpha·momentum_soft + (1−alpha)·onehot`, 원조 BLIP 모멘텀 분배). **itc 디버그 baseline 및 A/B 사다리용으로 유지.**
- **lm-KD**: `lm_distill_loss`; `online_teacher.lm_logits`; forward `loss_lm_kd`; pretrain `distill.lm` 배선.
- **online_teacher**: `itc_feats`(ttm가 티처 피처 필요), `lm_logits`, `encode_image`(스텝당 ViT 1회 dedup), 그리고 PORT되는 `itm_matrix`/`itm_matrix_gathered`.
- **arch**: vit `small_reg`(학생)·`base`·`large`(티처) + bert `minilm`(학생)·`base`(티처).

### 3.2 STRIP (죽은 토글)
| 제거 대상 | 위치 | 이유 |
|---|---|---|
| `itc_distill_loss` | `distillation/losses.py` + forward `loss_itc_kd` 블록 + pretrain `distill.itc` 배선 | 별도 B×B KD, ttm과 상호배타·항상 OFF |
| ttm `in_batch` variant | `target_mix.teacher_soft_in_batch` + forward ttm의 in_batch 분기 | 주력은 queue 확정 |
| **itm_target_mix 전부(exp10)** | `losses.itm_target_mix_loss` + `online_teacher.itm_soft` + forward `itm_mix` 경로 + pretrain `distill.itm_target_mix` 배선 | itm-k4로 대체 |
| 관련 테스트 | `distillation/test_losses_itm.py` 등 | 코드 따라 |
| 안쓰는 arch | vit `small`·`small_plus`·`small_plus_reg`; bert `medium` (`__init__`의 elif + `model_specs`) | 미사용 (base는 유지) |
| 주석처리된 deprecated 블록 | `blip_pretrain.py`의 깊은 주석 dead 코드(옛 bert 분기 등) | 읽을거리만 늘림, 삭제-only |

### 3.3 PORT (itm_bxb → dev, k=4 traditional KD)
- **신규 파일**: `distillation/itm_matrix.py` (`itm_bxb_logits`, `itm_pair_logits`)
- `distillation/losses.py` += `itm_matrix_kd_loss`, `itm_gathered_kd_loss`
- `distillation/online_teacher.py` += `itm_matrix`, `itm_matrix_gathered`
- forward: 파라미터 `itm_topk`/`itm_distill_temp`/`itm_distill_direction` 추가, `loss_itm_kd`(gathered top-k 경로) 계산, return에 추가
- pretrain: `distill.itm`(topk/direction/temp/weight) 읽기 + 합산(`loss += itm_kd_weight·loss_itm_kd`) + 로깅
- 테스트: `distillation/test_itm_matrix.py` + `test_forward_kd.py`의 itm 부분 이식
- 구조: `loss_itm`(하드 CE, 그대로) + `itm_kd_weight·loss_itm_kd`(k=4) — lm과 동형

## 4. 결과 인터페이스

**forward 시그니처(후)**:
```python
forward(image, caption, alpha, update_train_state=None,
        teacher_img_feat=None, teacher_text_feat=None,        # ttm용 (itc_distill 제거로 distill_temp는 삭제)
        teacher_lm_logits=None, teacher_lm_input_ids=None, lm_distill_temp=2.0,
        gamma=None, online_teacher=None,
        itm_topk=-1, itm_distill_temp=0.05, itm_distill_direction='bidir')
# return: loss_ita, loss_itm, loss_lm, loss_lm_kd, loss_itm_kd   # loss_itc_kd 제거
```

**config distill 표면(후)**:
```yaml
distill:
  itc_target_mix: {enabled, variant: queue, temp, soft_weight, hold_epochs, decay_end_epochs}
  lm:  {enabled, weight, temp}
  itm: {enabled, weight, temp, direction, topk}   # itm_bxb에서 이식
  # distill.itc (별도 KD) — 제거
  # distill.itm_target_mix — 제거
```
- `itc_target_mix` OFF → forward가 ttm-OFF 폴백(모멘텀 분배 baseline) 사용.
- `variant: in_batch` 지정 시 → assert 에러(지원 제거됨).

**본선 config**: itc-ttm(queue) + lm + itm(topk=4, bidir) 셋 다 ON인 첫 config를 추가(예: `pretrain_mainline.yaml`).

## 5. 테스트
- 이 env는 transformers 버전 비호환으로 모델 스택 임포트 불가(기존 문제, 리팩터 무관) → 순수함수 단위테스트만 로컬 검증: `test_target_mix`, `test_losses`(lm/itm KD), `test_itm_matrix`.
- 모델 스택 테스트(forward/online_teacher)는 실학습 env에서 확인.
- STRIP/PORT 후 각 단위테스트 그린 유지가 수용 기준.

## 6. 범위 밖 (다음 단계)
- **task 2 — 모듈화**: forward 안/밖에 흩어진 증류 로직의 재구획화 설계(코드-프리 논의). 이 최소 브랜치를 출발점으로 삼음.
- **task 3 — 불안정성 root-cause**: **원인 미상, 단정 금지.** 순간적·심각한 드롭은 미묘한 **구현 버그**를 강하게 시사(가장 유력). 접근: 드롭 근처 **체크포인트에서 재시작 + 프로브 왕창** 삽입해 상태 추적. 열어둘 가설(경합, 순위 아님):
  1. ITC/ttm/queue 경로의 **미묘한 구현 버그**
  2. **상태 오염** — 뒤로 가서도 계속 흔들림 → 어딘가 누적 오염(큐/모멘텀/포인터?)
  3. **queue_size=57600 vs in-batch(effective 160/step)** 과대 스테일니스. 큐는 약 360스텝(57600/160) 보관. 티처 큐는 frozen이라 스테일 없음이나, **모멘텀 큐**는 MoCo식 스테일(모멘텀 window ~200스텝 < 360). 57600은 BLIP 기본값이라 '전례상 정당'하나 — **BLIP는 effective batch가 훨씬 커서 큐 깊이가 스텝상 얕음(≪200)이라 안전**했던 것. 그 안전은 큰 배치 전제라, 작은 배치(160/step)로 숫자만 옮기면 360>200으로 깊어져 **BLIP 전례가 전이 안 됨.** 버그인지 정당인지 확인 대상.
  4. **retrieval metric 코드 자체 결함** — 불안정이 측정 아티팩트일 가능성.
  5. 학습가능 `logit_scale`(safe_scale) 자기강화 루프(학생 로짓 스케일 + 모멘텀 타깃 sharpness 양쪽).
  6. γ→0 핸드오프.

  A/B 사다리(순수 onehot → 모멘텀 baseline → ttm)는 '어느 층서 폭증하나'를 좁히는 보조 도구.
