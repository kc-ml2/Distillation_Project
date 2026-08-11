# Forward 메커니즘 구획화 (Approach A) 설계

**날짜**: 2026-08-11
**대상 브랜치**: `refactor/minimal_mainline` (최소 본선; 이 위에서 진행)
**관련**: [[refactor-itc-instability]] (task 2), spec `2026-08-10-refactor-minimal-mainline-design.md`

## 1. 배경 & 목표

리팩터로 `BLIP_Pretrain.forward`(`models/blip_pretrain.py`, ~196줄)는 주력 3경로(itc-ttm·lm-KD·itm-k4)만 남았지만 여전히 **god-function**이다: student 인코드 + 모멘텀/큐 bookkeeping + logit_scale + 3메커니즘(각 base loss + KD)이 한 함수에 뒤섞여 있고, 티처 호출이 **안/밖 혼재**(itc·lm은 pretrain에서 계산해 텐서로 주입, itm은 forward 안에서 호출)한다.

**목표 (사용자 우선순위: 가독성/구조 + 디버그 격리)**:
1. `forward`를 **얇은 오케스트레이터**로, 각 메커니즘을 **명확한 step 메서드**로 구획화.
2. **ITC 코어**(student + momentum + logit_scale + queue + ttm)를 `_itc_step` 하나로 떼어 task 3(불안정성 root-cause)에서 독립적으로 읽고 프로브 삽입 가능하게.
3. 티처 호출 패턴 통일(안/밖 비대칭 해소).

## 2. 제약

- **(제약1) 행동 보존 — 순수 리팩터.** 손실값이 리팩터 전후 **수치적으로 동일**해야 한다(로직 이동만, 계산 변경 없음).
- **(제약2) 체크포인트 키 호환.** stateful `nn.Module`(`visual_encoder(_m)`, `vision_proj(_m)`, `text_encoder(_m)`, `text_proj(_m)`, `text_decoder`, `itm_head`, 큐 버퍼)는 **model 속성 그대로 유지** → `state_dict` 키 불변. **메서드만 추가**, 모듈 트리·서브모듈화 없음. (그래서 Approach B 파일분리는 범위 밖.)
- **검증 env = conda `kd_r4`** (transformers 4.33.3). base env는 모델 import 불가. 모든 모델 테스트는 `conda run -n kd_r4`.

## 3. 설계 (Approach A)

### forward = 얇은 오케스트레이터
```
def forward(image, caption, alpha, update_train_state, *, distill 관련 인자):
    safe_scale = _scale_housekeeping(update_train_state)
    image_embeds, image_atts, image_feat, text, text_feat = _encode_student(image, caption)
    teacher_embeds = (online_teacher.encode_image(image)
                      if (online_teacher is not None and <any teacher mech on>) else None)   # dedup 1회
    loss_ita, sim_i2t, sim_t2i = _itc_step(image, caption, image_feat, text, text_feat,
                                           safe_scale, alpha, gamma, online_teacher, teacher_embeds, update_train_state)
    loss_itm, loss_itm_kd = _itm_step(image, caption, image_embeds, image_atts, text,
                                      sim_i2t, sim_t2i, online_teacher, teacher_embeds,
                                      itm_topk, itm_distill_temp, itm_distill_direction)
    loss_lm, loss_lm_kd = _lm_step(image, caption, image_embeds, image_atts, text,
                                   online_teacher, teacher_embeds, lm_distill_temp)
    return loss_ita, loss_itm, loss_lm, loss_lm_kd, loss_itm_kd
```
반환 튜플 순서 **불변**(`loss_ita, loss_itm, loss_lm, loss_lm_kd, loss_itm_kd`).

### 3 step 메서드 (전부 `BLIP_Pretrain`의 메서드, 파일 이동 없음)

| 메서드 | 책임 (현재 forward의 어느 블록) | 반환 |
|---|---|---|
| `_encode_student` | student 인코드 (현 305–314): image_embeds/atts/feat + text 토크나이즈 + text_feat | `image_embeds, image_atts, image_feat, text, text_feat` |
| **`_itc_step` = ITC 코어** | 모멘텀 갱신·인코드·큐(316–334) + sim_targets + **ttm 타깃**(340–357, 티처 `itc_feats` 내부호출) + student sim(360–361) + itc loss(364–367) + dequeue/enqueue(370–371) | `loss_ita, sim_i2t, sim_t2i` |
| `_itm_step` | neg-mining(379–391, sim 사용) + 3B forward(393–405) + CE(407) + k4 KD(437–462, 티처 `itm_matrix_gathered` 내부호출) | `loss_itm, loss_itm_kd` |
| `_lm_step` | decoder(410–422) + lm CE(422) + lm KD(424–432, 티처 `lm_logits` 내부호출) | `loss_lm, loss_lm_kd` |

`_scale_housekeeping`(289–301)은 forward에 남겨도 되고 작은 메서드로 빼도 됨(선택).

### 티처 호출 통일 (핵심 구조 변화)
- `teacher_image_embeds`는 forward 상단서 **1회** 계산(dedup 유지) 후 세 step에 전달.
- **itc·lm 티처 호출(`itc_feats`/`lm_logits`)을 pretrain → 각 step 내부로 이동.** 결과: 3경로 모두 "step이 자기 티처를 `image_embeds=teacher_embeds`로 내부 호출" = **대칭**.
- forward는 각 메커니즘의 teacher-활성 여부를 알아야 함 → pretrain이 enable 플래그(`lm_kd_enabled`, `itm_kd_enabled`; ttm은 `self.ttm_enabled`)를 forward 인자로 전달(정확한 파라미터화는 플랜에서).

### 불가피한 결합 1개 (누수 아님, 명시적 스레드)
- itm neg-mining이 **itc의 student sim**(`sim_i2t`/`sim_t2i`)에 의존(BLIP 하드네거 본질). → `_itc_step`이 sim을 반환하고 오케스트레이터가 `_itm_step`에 넘긴다. 숨은 전역상태 아니라 인자로 드러냄.

## 4. pretrain.py 변화
- **제거**: 스텝당 티처 pre-compute 블록(`teacher_image_embeds`/`itc_feats`/`lm_logits` 계산, 현 148–167) — forward 안으로 이동.
- **유지**: `gamma = ttm_gamma(global_step, ...)`(step 의존이라 pretrain), 손실 가중합(`loss = ita+itm+lm + lm_kd_w·lm_kd + itm_kd_w·itm_kd`; 가중치는 훈련 config), 로깅. forward엔 `gamma`·enable 플래그·itm/lm 하이퍼만 전달.
- val 경로(`eval_validation_loss.py`)는 `online_teacher=None` → 세 step이 base loss만, KD는 None. 기존과 동일.

## 5. 검증 (제약1 = 수치 동등성)
- **acceptance**: `conda run -n kd_r4 python -m pytest distillation/ tests/ -q` → 현재 **69/69 유지**.
- **수치 동등성 스모크**(핵심): 고정 seed·고정 배치로 리팩터 **전(215dc9a) / 후** 모델을 각각 forward 돌려 5개 손실이 bit-identical(또는 <1e-6)임을 확인하는 일회성 스크립트/테스트. 순수 리팩터이므로 반드시 동일해야 함. (플랜에 태스크로 포함.)
- 각 태스크(step 추출)마다 kd_r4 모델테스트 그린 유지.

## 6. 범위 밖
- **Approach B**(distillation/로 메커니즘 파일분리) — 체크포인트 키 마이그레이션 필요, 별건.
- **task 3**(불안정성 root-cause) — 이 구획화를 발판으로 이후 진행.
- 손실 가중합을 forward로 옮기기 / 반환을 dict화 — 지금은 튜플 유지(호출부 안정).
