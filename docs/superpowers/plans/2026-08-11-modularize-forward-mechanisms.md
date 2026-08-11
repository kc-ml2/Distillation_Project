# Forward 메커니즘 구획화 (Approach A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `BLIP_Pretrain.forward`를 얇은 오케스트레이터 + per-mechanism step 메서드(`_encode_student`/`_itc_step`/`_itm_step`/`_lm_step`)로 구획화하고, 티처 호출을 forward 안으로 통일한다. **행동(손실값) 불변**.

**Architecture:** 순수 리팩터. stateful `nn.Module`은 model 속성 그대로(체크포인트 키 불변) — **메서드만 추가**. Phase 1(Task 2–5): 티처를 지금처럼 인자로 받은 채 블록을 메서드로 추출(behavior bit-identical). Phase 2(Task 6): itc·lm 티처 호출을 각 step 안으로 이동 + forward 시그니처 단순화.

**Tech Stack:** PyTorch, HuggingFace BERT(minilm/base), timm DINOv3(small_reg), DDP. 검증 env = conda `kd_r4`.

## Global Constraints

- **행동 보존 = 손실값 불변**: 5개 손실(`loss_ita, loss_itm, loss_lm, loss_lm_kd, loss_itm_kd`)이 각 태스크 전후로 **bit-identical**(고정 seed·고정 배치·stub 티처). Task 1이 golden reference를 박제, 이후 매 태스크가 재확인.
- **체크포인트 키 불변**: encoder(`_m`)·proj(`_m`)·queue 버퍼·`itm_head`·`text_decoder`는 model 속성 그대로. 서브모듈화·파일이동 **금지**. 메서드 추가만.
- **반환 튜플 순서 불변**: `loss_ita, loss_itm, loss_lm, loss_lm_kd, loss_itm_kd`.
- **검증 env = `kd_r4`**: 모델 테스트는 **항상 `conda run -n kd_r4 python -m pytest ...`**. base(miniconda)는 `transformers` 신버전이라 모델 import 불가(순수함수만). 절대 base로 모델테스트 "실패"를 결함으로 오판 말 것.
- 현행 acceptance = `conda run -n kd_r4 python -m pytest distillation/ tests/ -q` → **69 passed** 유지.
- 작업 대상 = `models/blip_pretrain.py`(forward), `pretrain.py`(배선), `data/eval_validation_loss.py`(val 호출). 다른 모델(predict/train_*)·distillation/ 순수모듈은 손대지 않음.

## File Structure

| 파일 | 변경 |
|---|---|
| `models/blip_pretrain.py` | forward 블록 → `_encode_student`/`_itc_step`/`_itm_step`/`_lm_step` 메서드로 추출 + forward=오케스트레이터. Phase2서 시그니처 단순화(티처 텐서 인자 제거→`online_teacher`+flags) |
| `pretrain.py` | Phase2: 스텝당 티처 pre-compute 제거, `online_teacher`+enable플래그+gamma/하이퍼만 전달. gamma 스케줄·손실 가중합·로깅 유지 |
| `data/eval_validation_loss.py` | Phase2: val forward 호출을 새 시그니처로(`online_teacher=None`) |
| `distillation/test_forward_equiv.py` | **NEW**: 수치 동등성 harness (golden reference) |

현재 forward 블록 지도(215dc9a 기준, `models/blip_pretrain.py`): housekeeping 289–301 / encode 305–313 / momentum+ttm타깃 316–357 / itc sim+loss 359–367 / dequeue 369–372 / **itm base** 374–407 / lm 409–422 / lm-KD 424–432 / **itm-KD** 434–462 / return 464. (라인은 태스크마다 이동하니 **심볼/내용으로** 위치 확인.)

---

### Task 0: 브랜치 + 워크트리

**Files:** 없음 (git)

- [ ] **Step 1: 워크트리 생성** (sibling, `.claude/worktrees/` 아님 — [[feedback-worktree-location]])

```bash
git -C /home/minwoo/Distillation_Project_minimal_mainline worktree add -b refactor/modularize_forward \
  /home/minwoo/Distillation_Project_modularize refactor/minimal_mainline
```

- [ ] **Step 2: 확인**

Run: `git -C /home/minwoo/Distillation_Project_modularize log --oneline -1` → `215dc9a`.
Run: `conda run -n kd_r4 python -m pytest distillation/ tests/ -q` (from the worktree) → **69 passed** (baseline sanity).

이후 모든 태스크는 워크트리 `/home/minwoo/Distillation_Project_modularize`에서, 모델테스트는 `conda run -n kd_r4`.

---

### Task 1: 수치 동등성 harness + golden reference

**Files:** Create `distillation/test_forward_equiv.py`

**Interfaces:**
- Produces: a pytest that constructs the student once, runs `forward` on a FIXED input with a deterministic **stub teacher**, and asserts the 5 losses equal stored golden values (`atol=0`, i.e. bit-identical; fall back to `1e-6` only if a nondeterministic op forces it — document which).

- [ ] **Step 1: harness 작성**

`distillation/test_forward_equiv.py`:
- `torch.manual_seed(0)`; construct `blip_pretrain(image_size=224, vit='small_reg', my_bert_size='minilm', queue_size=<batch의 배수, 예: 8>, ttm_enabled=True, ttm_variant='queue', ttm_temp=0.05, ttm_soft_weight=0.4, init_backbone_weights=False)`; `model.eval()`.
- Fixed batch: `image = torch.randn(4,3,224,224, generator=g)`, `caption = ["a cat", "a dog", "two birds", "a red car"]`.
- **Stub teacher**: a tiny object with `encode_image(image)`, `itc_feats(image, caption, image_embeds=None)`, `lm_logits(image, caption, image_embeds=None)`, `itm_matrix(...)`, `itm_matrix_gathered(...)` returning **deterministic seeded tensors of the correct shapes** (image_embeds `[B,N,D]`; itc feats `[B,256]` L2-norm; lm logits `[B,L,V]`; itm matrix/gathered per `distillation/itm_matrix.py` contracts). No real BLIP-large.
- Call `forward` with `update_train_state=False` (no momentum/queue mutation — deterministic), passing the stub's teacher tensors (Phase 1 signature) — for lm: `teacher_lm_logits=stub.lm_logits(...)[0]`; ttm: `teacher_img_feat/teacher_text_feat=stub.itc_feats(...)`; itm: `online_teacher=stub, teacher_image_embeds=stub.encode_image(...)`, `itm_topk=4`.
- Capture the 5 losses → assert against a GOLDEN dict hard-coded in the test (fill in Step 2).

- [ ] **Step 2: golden 값 채점**

Run once on the UNMODIFIED code to print the 5 losses, paste them into the test as the golden dict:
```
conda run -n kd_r4 python -m pytest distillation/test_forward_equiv.py -q -s   # prints losses first run
```
Then hard-code the printed values; re-run → PASS.

- [ ] **Step 3: 확인 + 커밋**

Run: `conda run -n kd_r4 python -m pytest distillation/test_forward_equiv.py -q` → PASS.
```bash
git add distillation/test_forward_equiv.py && git commit -m "test(equiv): forward 5-loss golden reference (fixed seed + stub teacher)"
```

---

### Task 2: `_encode_student` 추출

**Files:** Modify `models/blip_pretrain.py`

**Interfaces:**
- Produces: `_encode_student(self, image, caption) -> (image_embeds, image_atts, image_feat, text, text_feat)`

- [ ] **Step 1: 메서드 추출**

forward의 encode 블록(현 305–313: `image_embeds`/`image_atts`/`image_feat` + `text = self.tokenizer(...)` + `text_output`/`text_feat`)을 `_encode_student`로 이동. forward는 `image_embeds, image_atts, image_feat, text, text_feat = self._encode_student(image, caption)`로 대체. **로직 변경 없음.**

- [ ] **Step 2: 검증**

Run: `conda run -n kd_r4 python -m pytest distillation/test_forward_equiv.py distillation/ tests/ -q` → equiv PASS + **69 passed** 유지.

- [ ] **Step 3: 커밋**

```bash
git add models/blip_pretrain.py && git commit -m "refactor(forward): extract _encode_student (behavior-identical)"
```

---

### Task 3: `_lm_step` 추출

**Files:** Modify `models/blip_pretrain.py`

**Interfaces:**
- Consumes: `image_embeds, image_atts, text` (from `_encode_student`), `teacher_lm_logits`, `teacher_lm_input_ids`, `lm_distill_temp`
- Produces: `_lm_step(self, image_embeds, image_atts, text, teacher_lm_logits, teacher_lm_input_ids, lm_distill_temp) -> (loss_lm, loss_lm_kd)`

- [ ] **Step 1: 메서드 추출**

forward의 lm 블록(현 409–422: decoder_input_ids/targets + `self.text_decoder(...)` + `loss_lm`) + lm-KD 블록(424–432: `loss_lm_kd = lm_distill_loss(...)` with the input-ids assert)을 `_lm_step`으로 이동. **티처는 아직 인자로 받음**(Phase 1). forward는 `loss_lm, loss_lm_kd = self._lm_step(...)`로 대체.

- [ ] **Step 2: 검증**

Run: `conda run -n kd_r4 python -m pytest distillation/test_forward_equiv.py distillation/ tests/ -q` → equiv PASS + 69 passed.

- [ ] **Step 3: 커밋**

```bash
git add models/blip_pretrain.py && git commit -m "refactor(forward): extract _lm_step (behavior-identical)"
```

---

### Task 4: `_itm_step` 추출 (base + KD 통합·재정렬)

**Files:** Modify `models/blip_pretrain.py`

**Interfaces:**
- Consumes: `image, caption, image_embeds, image_atts, text, sim_i2t, sim_t2i, online_teacher, teacher_image_embeds, itm_topk, itm_distill_temp, itm_distill_direction`
- Produces: `_itm_step(...) -> (loss_itm, loss_itm_kd)`

- [ ] **Step 1: 두 블록 모아 추출**

forward의 itm base(현 374–407: encoder_input_ids + neg-mining(`sim_i2t`/`sim_t2i` 사용) + 3B forward + `loss_itm`)과 itm-KD(434–462: gathered top-k + `itm_pair_logits` + `online_teacher.itm_matrix_gathered(..., image_embeds=teacher_image_embeds)` + `itm_gathered_kd_loss`)를 **하나의 `_itm_step`으로 모은다**. lm(409–432)이 사이에 있었지만 itm-KD는 lm에 의존하지 않고 itm base의 산출(`weights_*`, `encoder_input_ids`, `image_embeds`)에만 의존하므로 **재정렬 안전**. forward는 `loss_itm, loss_itm_kd = self._itm_step(...)`로 대체(lm 호출보다 앞/뒤 무관 — 반환 순서만 지키면 됨).

- [ ] **Step 2: 검증**

Run: `conda run -n kd_r4 python -m pytest distillation/test_forward_equiv.py distillation/ tests/ -q` → equiv PASS + 69 passed.
(equiv가 깨지면 재정렬이 뭔가 건드린 것 — 되돌려 원인 확인.)

- [ ] **Step 3: 커밋**

```bash
git add models/blip_pretrain.py && git commit -m "refactor(forward): extract _itm_step (base+kd merged, reordered, behavior-identical)"
```

---

### Task 5: `_itc_step` 추출 (ITC 코어)

**Files:** Modify `models/blip_pretrain.py`

**Interfaces:**
- Consumes: `image, caption, image_feat, text, text_feat, safe_scale, alpha, gamma, teacher_img_feat, teacher_text_feat, update_train_state`
- Produces: `_itc_step(...) -> (loss_ita, sim_i2t, sim_t2i)`

- [ ] **Step 1: 메서드 추출**

forward의 momentum+ttm타깃(현 316–357) + student itc sim(359–361) + itc loss(364–367) + dequeue/enqueue(369–372)를 `_itc_step`으로 이동. **티처 feats는 아직 인자**(Phase 1). `sim_i2t`/`sim_t2i`를 **반환**(itm이 사용). `_momentum_update`/`_dequeue_and_enqueue`는 기존 메서드 그대로 호출. forward는 `loss_ita, sim_i2t, sim_t2i = self._itc_step(...)`로 대체.

- [ ] **Step 2: 검증**

Run: `conda run -n kd_r4 python -m pytest distillation/test_forward_equiv.py distillation/ tests/ -q` → equiv PASS + 69 passed.

- [ ] **Step 3: 커밋** — 이 시점 forward = 오케스트레이터(4 메서드 호출), 행동 불변, ITC 코어 격리 완료(Phase 1 끝).

```bash
git add models/blip_pretrain.py && git commit -m "refactor(forward): extract _itc_step = ITC core (behavior-identical); forward now orchestrator"
```

---

### Task 6: 티처 호출 통일 (Phase 2 — 시그니처 변경)

**Files:** Modify `models/blip_pretrain.py`, `pretrain.py`, `data/eval_validation_loss.py`, `distillation/test_forward_equiv.py`

**Interfaces:**
- Produces: forward 시그니처 = `forward(self, image, caption, alpha, update_train_state=None, *, gamma=None, online_teacher=None, lm_kd_enabled=False, itm_kd_enabled=False, lm_distill_temp=2.0, itm_topk=-1, itm_distill_temp=0.05, itm_distill_direction='bidir')` (pre-computed 티처 텐서 인자 `teacher_img_feat/teacher_text_feat/teacher_lm_logits/teacher_lm_input_ids/teacher_image_embeds` **제거**). 정확한 최종 시그니처는 구현서 확정하되 이 방향.

- [ ] **Step 1: forward가 티처 embeds 1회 계산 + 각 step이 자기 티처 호출**

- forward 상단: `teacher_image_embeds = online_teacher.encode_image(image) if (online_teacher is not None and (self.ttm_enabled or lm_kd_enabled or itm_kd_enabled)) else None`.
- `_itc_step`: ttm 활성(`self.ttm_enabled and gamma is not None and online_teacher is not None`)이면 내부에서 `teacher_img_feat, teacher_text_feat = online_teacher.itc_feats(image, caption, image_embeds=teacher_image_embeds)` 호출(기존 인자 대신). 없으면 alpha 폴백.
- `_lm_step`: `lm_kd_enabled and online_teacher is not None`이면 내부에서 `teacher_lm_logits, teacher_lm_ids = online_teacher.lm_logits(image, caption, image_embeds=teacher_image_embeds)` 호출; input-ids assert 유지.
- `_itm_step`: 이미 online_teacher 내부호출(변화 없음), `teacher_image_embeds`만 forward가 넘김.

- [ ] **Step 2: pretrain.py 배선 정리**

스텝당 티처 pre-compute 블록(`need_teacher_image_embeds`/`encode_image`/`itc_feats`/`lm_logits` 계산 및 관련 지역변수) **제거**. `model(image, caption, alpha=alpha, update_train_state=..., gamma=gamma, online_teacher=online_teacher, lm_kd_enabled=lm_kd_enabled, itm_kd_enabled=itm_kd_enabled, lm_distill_temp=lm_kd_temp, itm_topk=itm_kd_topk, itm_distill_temp=itm_kd_temp, itm_distill_direction=itm_kd_direction)`로 호출. `gamma` 스케줄·손실 가중합·로깅은 그대로. `online_teacher`는 teacher 경로 하나라도 켜지면 넘김(현재는 itm만 조건부였음 → itc_target_mix/lm/itm 중 하나라도 enabled면 넘기게).

- [ ] **Step 3: eval_validation_loss.py val 호출 갱신**

val forward 호출을 새 시그니처로(`online_teacher=None`, KD 플래그 off) — base 손실만. 언팩은 5-tuple 유지.

- [ ] **Step 4: equiv harness를 새 시그니처로 갱신**

`test_forward_equiv.py`가 forward를 `online_teacher=stub, lm_kd_enabled=True, itm_kd_enabled=True, gamma=<고정>`로 호출하도록 수정(stub이 itc_feats/lm_logits/itm_matrix_gathered 제공). **golden 5-loss 값은 그대로여야 함**(같은 티처 출력·같은 수학). 값 안 바꿈.

- [ ] **Step 5: 검증**

Run: `conda run -n kd_r4 python -m pytest distillation/test_forward_equiv.py distillation/ tests/ -q` → equiv PASS(golden 불변) + **69 passed** 유지.
Run: `conda run -n kd_r4 python -c "import ast,sys; ..."` 또는 grep로 pretrain.py에 티처 pre-compute 잔재 없음 확인.

- [ ] **Step 6: 커밋**

```bash
git add -A && git commit -m "refactor(forward): unify teacher calls into steps; simplify forward signature; thin pretrain wiring"
```

---

### Task 7: 최종 검증 스윕

**Files:** 없음(검증) / 필요시 `pretrain.py` 로깅 정리

- [ ] **Step 1: 전체 kd_r4 스위트**

Run: `conda run -n kd_r4 python -m pytest distillation/ tests/ models/ -q` → all green(69 + equiv).

- [ ] **Step 2: 잔재/일관성 grep**

Run: `grep -rnE "teacher_img_feat=|teacher_lm_logits=|need_teacher_image_embeds" --include=*.py pretrain.py data/ models/` → forward 인자로서의 pre-compute 티처 잔재가 pretrain/val에 없어야(있으면 Task6 누락). forward 정의부의 파라미터는 제거됐는지 확인.

- [ ] **Step 3: forward 가독성 눈검사**

`models/blip_pretrain.py` forward가 ~10줄 내외 오케스트레이터(housekeeping → encode → teacher_embeds → 3 step → return)로 읽히는지 확인. `_itc_step`이 ITC 코어(모멘텀·큐·ttm·itc loss)를 독립적으로 담고 있는지.

- [ ] **Step 4: 커밋(있으면)**

```bash
git add -A && git commit -m "chore(forward): final modularization sweep (kd_r4 green, forward reads as orchestrator)"
```

---

## 실행 후 (범위 밖)
- **task 3 불안정성 root-cause**: `_itc_step` = ITC 코어를 발판으로, 체크포인트 재시작 + 프로브 + A/B 사다리(경합 가설 6종).
- Approach B(distillation/로 파일분리) — 체크포인트 키 마이그레이션 별건.
