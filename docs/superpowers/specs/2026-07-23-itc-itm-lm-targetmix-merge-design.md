# ITC + ITM + LM target-mix 구조적 병합 — 설계 (Phase 1)

날짜: 2026-07-23
브랜치: `itc_itm_lm_targetmix` (dev fe23681에서 분기, dev는 그대로 보존)

## 1. 배경

dev(fe23681)를 공통 조상으로 두 개의 독립 실험 브랜치가 각각 구현·리뷰·테스트를 마쳤다.

- **exp9 `exp9_teacher_target_mix`** (커밋 fe23681..69a8107, origin push됨): ITC teacher-target-mix.
  `target = (1−W)·one-hot + W·[γ·teacher_soft + (1−γ)·momentum_soft]`, W=0.4 상수,
  γ는 2ep 홀드 + 10ep 선형감쇠 스케줄. variant C(queue, 티처 feature 큐 페어 enqueue) /
  variant D(in_batch, B×B softmax 0-패딩). 구현+CPU스모크+리뷰 SHIP READY. 실GPU/DDP
  스모크·본런은 스페어 서버로 이관되어 아직 미실행.
- **exp10 `worktree-exp10_itm_target_mix`** (커밋 fe23681..9bbc546): ITM teacher-target-mix.
  `target = (1−W)·one-hot + W·teacher_soft`(3B ITM 라벨), arm A(neg_source=student,
  soft_weight=0.4) / B(neg_source=teacher, soft_weight=0.0) / C(teacher+0.4). 구현+전체
  CPU테스트(53/53)+실체크포인트 GPU스모크(arm C, 5/5 finite)+최종 리뷰 완료. 실 본런은
  아직 미실행(다음 작업으로 arm C launch 예정, 이번 병합과 별개로 진행).

`git merge-base exp9_teacher_target_mix worktree-exp10_itm_target_mix` = fe23681 —
즉 둘은 dev의 같은 지점에서 갈라진 **형제 브랜치**이며, 서로의 작업을 포함하지 않는다.
`git merge-tree`로 사전 검증한 결과 실제 텍스트 충돌은 `models/blip_pretrain.py`와
`pretrain.py` 두 파일, 5개 지점뿐이며 전부 "같은 삽입 지점에 다른 블록을 나란히
추가"하는 형태로 의미적 모순은 없다(§3에서 상세).

**목표**: 두 실험을 dev는 그대로 둔 채 새 브랜치로 구조적으로 병합하여, 세 증류
메커니즘(ITC target-mix / ITM target-mix / 기존 LM distillation)이 한 코드베이스에서
config로 독립적으로 켜고 끌 수 있게 만든다. 성능 최적화(교사 forward 중복 제거)와
실제 동시-ON 멀티에폭 본런은 이번 범위 밖(§6 참고), 다만 "동시에 켜도 코드가 죽지
않는다"는 최소 스모크 검증까지는 이번 범위에 포함한다.

## 2. 브랜치 & 머지 순서

```
dev(fe23681) ──branch──> itc_itm_lm_targetmix
                              │
                              ├─ git merge exp9_teacher_target_mix   (먼저)
                              │
                              └─ git merge worktree-exp10_itm_target_mix (다음)
```

두 병합 모두 표준 `git merge` 사용(cherry-pick도 수동 재작성도 아님). 논-충돌 파일
(`distillation/losses.py`, `distillation/target_mix.py`, `distillation/online_teacher.py`
의 itm_soft 등, 신규 테스트 파일들)은 git이 자동 병합한다 — `git merge-tree`로 사전
확인 완료.

## 3. 충돌 지점과 해소 (전부 "둘 다 유지", 재작성 없음)

### 3.1 `models/blip_pretrain.py` — `forward()` 시그니처 (1곳)

```python
def forward(self, image, caption, alpha, update_train_state=None,
            teacher_img_feat=None, teacher_text_feat=None, distill_temp=0.05,
            teacher_lm_logits=None, teacher_lm_input_ids=None, lm_distill_temp=2.0,
            gamma=None, online_teacher=None, itm_mix=None):
```

exp9의 `gamma`는 ITC 손실 블록에서만, exp10의 `online_teacher`/`itm_mix`는 ITM 손실
블록에서만 쓰인다. 두 블록은 서로 겹치지 않으므로 시그니처만 합치면 충돌 해소 끝.

### 3.2 `pretrain.py` — distill config 파싱 (상단, 1곳)

ttm 파라미터(`ttm_on`, `ttm_hold_steps`, `ttm_decay_end_steps`, `ttm_soft_weight` 등)와
itm_mix 파라미터(`itm_mix_enabled`, `itm_neg_source`, `itm_soft_weight`,
`itm_teacher_temp`)를 둘 다 계산해 남긴다. `distill.lm`(기존 dev) 파싱은 변경 없음.

### 3.3 `pretrain.py` — `teacher_keep` 산출 (1곳, §4 참고)

두 인라인 버전을 전부 버리고 **통합 `derive_teacher_keep()`** 호출로 교체.

### 3.4 `pretrain.py` — `need_teacher_itc` 게이팅 (루프 내, 1곳)

```python
from distillation.target_mix import ttm_gamma
gamma = ttm_gamma(global_step, ttm_hold_steps, ttm_decay_end_steps) if ttm_enabled else None

need_teacher_itc = (
    itc_kd_enabled
    or (ttm_enabled and gamma is not None and gamma > 0)
    or (itm_mix_enabled and itm_neg_source == 'teacher')
)
if need_teacher_itc and online_teacher is not None:
    teacher_img_feat, teacher_text_feat = online_teacher.itc_feats(image, caption)
else:
    teacher_img_feat = teacher_text_feat = None
```

### 3.5 `pretrain.py` — `model(...)` 호출부(2곳: cuda-autocast 분기 / plain 분기) + TB 로깅

호출부는 `gamma=gamma, online_teacher=itm_online_teacher, itm_mix=itm_mix`를 모두 전달.
TB 로깅은 `if ttm_enabled: ...`(train/gamma, train/beta)와
`if itm_mix_enabled: ...`(train/itm_teacher_weight) 블록을 나란히 유지.

## 4. `derive_teacher_keep()` 통합 (`distillation/distill_config.py`)

exp10에만 존재하는 이 함수를 확장해 exp9의 인라인 로직을 흡수한다 — 이것이 이번
병합에서 유일하게 "재작성"이 필요한 지점이다(단순 병렬 유지가 아님).

```python
def derive_teacher_keep(distill_cfg):
    keep = set()
    if distill_cfg.get('itc', {}).get('enabled', False):
        keep.add('itc')
    if distill_cfg.get('lm', {}).get('enabled', False):
        keep.add('lm')
    if distill_cfg.get('itc_target_mix', {}).get('enabled', False):
        keep.add('itc')
    itm = distill_cfg.get('itm_target_mix', {})
    if itm.get('enabled', False):
        if itm.get('neg_source') == 'teacher':
            keep.add('itc')
        if float(itm.get('soft_weight', 0.0)) > 0.0:
            keep.add('itm')
    return tuple(sorted(keep))
```

`validate_itm_mix_config()`는 변경 없이 그대로 호출. exp9의 `itc` vs `itc_target_mix`
상호배타 assert(`assert not (itc_kd_enabled and ttm_enabled)`)도 그대로 유지.

## 5. Phase 1 산출물

1. §4의 통합 `derive_teacher_keep()` + 단위테스트 확장(4가지 메커니즘 조합 케이스 추가,
   특히 `itc_target_mix` enabled 단독으로도 `'itc'`가 포함되는 케이스).
2. **`configs/pretrain_itc_itm_lm_targetmix_smoke.yaml`**: `pretrain_student.yaml` 복사본에
   아래 `distill` 블록을 채운다(키 형식은 exp9의 `configs/pretrain_ttm_queue.yaml`을 그대로 따름):
   ```yaml
   distill:
     itc:
       enabled: false        # itc_target_mix와 상호배타 (assert)
     lm:
       enabled: true
       weight: 1.0
       temp: 2.0
     itc_target_mix:
       enabled: true
       variant: queue
       temp: 0.05
       soft_weight: 0.4
       hold_epochs: 2
       decay_end_epochs: 12
     itm_target_mix:            # arm C
       enabled: true
       neg_source: teacher
       soft_weight: 0.4
       temp: 1.0
       schedule: constant
   ```
   `teacher.checkpoint`는 실제 `model_large.pth` 경로, `queue_size=57600`(기존 컨벤션).
   결과 `teacher_keep`은 `('itc','itm','lm')` — 이 3-way 조합은 이번이 최초 테스트.
3. **GPU 1대 스모크**: 실제 `-m pretrain --config=./configs/pretrain_itc_itm_lm_targetmix_smoke.yaml`
   엔트리포인트로 몇 스텝만 실행(수 분 내 kill 가능), 예외 없이 뜨는지 + 로그에 찍히는
   loss들(loss_ita, loss_itm, loss_lm, 및 활성화된 kd/target-mix 관련 항목)이 전부
   finite인지 확인. 실제 학습(멀티에폭)은 하지 않음.
4. **`OnlineTeacher` 3-way keep 단위테스트 신규 추가**: `keep=('itc','itm','lm')` 조합
   생존/해제 검증 — 기존 `TestOnlineTeacherKeepBoth`(2-way까지만 커버) 패턴을 확장.
5. **회귀**: exp9/exp10 각자의 기존 CPU 단위테스트 스위트가 병합 후에도 그대로
   통과하는지 확인(브랜치 전체 실행 1회).

## 6. 범위 밖 (Phase 2로 명시적으로 미룸)

- **교사 forward 중복 제거**: arm C에서 `itc_feats`와 `itm_soft`가 각각 독립적으로
  `visual_encoder(image)`를 호출해 같은 이미지에 대해 중복 계산되는 문제(exp10 최종
  리뷰에서 이미 지적됨, spec 2026-07-22 §5). 이번엔 그대로 둔다.
- **실제 멀티에폭 본런**: 세 메커니즘을 동시에 켠 조합으로 실제 학습을 도는 것은
  이번 범위 밖. GPU 자원이 이미 다른 작업으로 포화 상태라 별도로 상의 후 진행.
- **itc_target_mix variant 최종 선택**: queue(C) vs in_batch(D) 중 어느 쪽을 최종
  조합에 쓸지는 실 GPU 검증 결과가 나온 뒤 결정. 스모크 config는 queue로 고정하되,
  이 선택이 최종 결정은 아님.

## 7. 테스트 계획

- 단위: `derive_teacher_keep()` 확장 케이스, `OnlineTeacher` 3-way keep 케이스.
- 회귀: 병합 브랜치에서 `distillation/` + `models/test_blip_pretrain_itm.py` 등 기존
  전체 스위트 재실행, 병합 전과 동일하게 통과해야 함.
- 통합 스모크: §5-3 GPU 1대 실행, finite loss + 예외 없음 확인(자동화된 pass/fail
  스크립트는 아님 — 로그 육안 확인 수준으로 충분, 본격적인 자동 스모크 스크립트는
  Phase 2에서 필요시 추가).
