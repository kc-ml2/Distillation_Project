# 최소 본선 브랜치 리팩터 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** dev를 주력 3-메커니즘(itc-ttm queue + lm-KD + itm k=4 traditional KD)만 남긴 `refactor/minimal_mainline` 브랜치로 만든다 — 죽은 토글·안쓰는 arch 삭제 + itm k=4를 itm_bxb에서 이식.

**Architecture:** base=dev. STRIP(itc_distill_loss / ttm in_batch / itm_target_mix / 안쓰는 arch / 주석 dead) + PORT(itm_bxb의 k=4 matrix·gathered KD). **최소 변환** — 삭제·이식만, 재배치·재구조화 없음(모듈화는 별도 task).

**Tech Stack:** PyTorch, HuggingFace BERT(med.py), timm/DINOv3(small_reg), DDP, online frozen BLIP-large 티처.

## Global Constraints

- **최소 변환**: 삭제 + 이식만. 파일/함수 **재배치·재구조화 금지**(모듈화는 이 플랜 범위 밖). 코드 위치 그대로.
- **env 제약**: 이 서버는 transformers 비호환으로 `models.blip_pretrain` import 불가. **로컬 런타임 검증 = 순수함수 테스트**(`distillation/test_target_mix.py`, `distillation/test_losses.py`, `distillation/test_itm_matrix.py`) **+ `python -m py_compile`**(구문). forward/online_teacher **실행** 검증은 실학습 env로 이연.
- **주력 아키텍처**: 학생 = vit `small_reg` + bert `minilm`; 티처 = vit `large` + bert `base`. arch 유지집합 = vit{small_reg, base, large} + bert{minilm, base}.
- **itc-ttm variant = queue only**. 미지원 값(in_batch, distill.itc, itm_target_mix)은 조용한 no-op 아니라 **assert/에러**.
- **forward return(최종)**: `loss_ita, loss_itm, loss_lm, loss_lm_kd, loss_itm_kd`. 언팩 3곳: `pretrain.py:194`, `pretrain.py:207`, `data/eval_validation_loss.py:297`.
- **커밋**: `refactor/minimal_mainline`은 Claude 작업 브랜치(dev/main 아님) → 자유 커밋 OK. 각 task 끝 커밋.

## File Structure

| 파일 | 역할 | 변경 |
|---|---|---|
| `distillation/itm_matrix.py` | itm B×B / pair 로짓 조립 | **NEW**(port) |
| `distillation/losses.py` | 손실 함수 | +itm_matrix_kd_loss, +itm_gathered_kd_loss; −itm_target_mix_loss, −itc_distill_loss |
| `distillation/online_teacher.py` | 온라인 티처 서빙 | +itm_matrix, +itm_matrix_gathered; −itm_soft |
| `distillation/target_mix.py` | ttm 타깃 조립 | −teacher_soft_in_batch |
| `models/blip_pretrain.py` | 모델 forward | itm 경로 swap, −loss_itc_kd, −ttm in_batch, −죽은 arch, −주석 dead |
| `pretrain.py` | 학습 루프·배선 | −distill.itc, −distill.itm_target_mix, +distill.itm(k4) |
| `data/eval_validation_loss.py` | val forward | 언팩 튜플 갱신 |
| `distillation/test_itm_matrix.py` | itm 로짓 테스트 | **NEW**(port) |
| `distillation/test_losses.py` | 손실 테스트 | +itm-k4, −itm_target_mix |
| `configs/pretrain_mainline.yaml` | 본선 config | **NEW** |

포트 소스 참조: itm_bxb 파일은 `git show itm_bxb_matrix_kd:<path>`로 꺼낸다.

---

### Task 0: 브랜치 + 워크트리 생성

**Files:** 없음 (git 작업)

- [ ] **Step 1: dev 최신 확인**

Run: `git -C /home/minwoo/Distillation_Project fetch origin && git -C /home/minwoo/Distillation_Project log --oneline -1 dev`
Expected: `ef1d982` (spec 커밋) 이거나 그 이후.

- [ ] **Step 2: 워크트리 생성** (sibling 디렉토리, `.claude/worktrees/` 아님)

```bash
git -C /home/minwoo/Distillation_Project worktree add -b refactor/minimal_mainline \
  /home/minwoo/Distillation_Project_minimal_mainline dev
```

- [ ] **Step 3: 확인**

Run: `git -C /home/minwoo/Distillation_Project_minimal_mainline branch --show-current`
Expected: `refactor/minimal_mainline`

이후 모든 task는 워크트리 `/home/minwoo/Distillation_Project_minimal_mainline`에서 작업.

---

### Task 1: ITM k=4 순수함수 이식 (로컬 실행 테스트 O)

**Files:**
- Create: `distillation/itm_matrix.py`, `distillation/test_itm_matrix.py`
- Modify: `distillation/losses.py` (함수 2개 추가)
- Modify: `distillation/test_losses.py` (itm-k4 테스트 추가)

**Interfaces:**
- Produces:
  - `itm_bxb_logits(text_encoder, itm_head, image_embeds, image_atts, text_ids, text_atts) -> [B,B,2]`
  - `itm_pair_logits(...)` (gathered 후보용 pair 로짓)
  - `itm_matrix_kd_loss(student_logits, teacher_logits, direction, temp) -> scalar`
  - `itm_gathered_kd_loss(s_i2t, t_i2t, s_t2i, t_t2i, direction, temp) -> scalar`

- [ ] **Step 1: itm_matrix.py 이식**

```bash
git show itm_bxb_matrix_kd:distillation/itm_matrix.py > distillation/itm_matrix.py
git show itm_bxb_matrix_kd:distillation/test_itm_matrix.py > distillation/test_itm_matrix.py
```

- [ ] **Step 2: losses.py에 itm KD 2함수 추가**

`git show itm_bxb_matrix_kd:distillation/losses.py`의 `itm_matrix_kd_loss`(라인 67~)·`itm_gathered_kd_loss`(라인 97~)를 dev `distillation/losses.py` 끝에 붙인다. (dev losses.py의 `itc_distill_loss`/`lm_distill_loss`는 itm_bxb와 동일하므로 중복 추가 금지 — itm 2개만.)

- [ ] **Step 3: test_losses.py에 itm-k4 테스트 추가**

`git show itm_bxb_matrix_kd:distillation/test_losses.py`에서 `itm_matrix_kd_loss`/`itm_gathered_kd_loss` 테스트(방향 bidir/forward/reverse, backward, sideways assert)를 dev `distillation/test_losses.py`에 병합.

- [ ] **Step 4: 테스트 실행 (FAIL→PASS 확인)**

Run: `python -m pytest distillation/test_itm_matrix.py distillation/test_losses.py -q`
Expected: PASS (이식 코드라 바로 그린).

- [ ] **Step 5: 커밋**

```bash
git add distillation/itm_matrix.py distillation/test_itm_matrix.py distillation/losses.py distillation/test_losses.py
git commit -m "port(itm-kd): itm_matrix + matrix/gathered KD losses from itm_bxb"
```

---

### Task 2: online_teacher ITM 메서드 이식 (구조적)

**Files:**
- Modify: `distillation/online_teacher.py` (+itm_matrix, +itm_matrix_gathered, import)

**Interfaces:**
- Consumes: Task1의 `itm_bxb_logits`, `itm_pair_logits`
- Produces:
  - `OnlineTeacher.itm_matrix(image, caption) -> [B,B,2]` teacher ITM 로짓
  - `OnlineTeacher.itm_matrix_gathered(image, caption, idx_i2t_full, idx_t2i_full) -> (t_i2t, t_t2i)` 각 `[B,k+1,2]`

- [ ] **Step 1: import 추가**

`distillation/online_teacher.py` 상단에:
```python
from distillation.itm_matrix import itm_bxb_logits, itm_pair_logits
```

- [ ] **Step 2: 메서드 2개 이식**

`git show itm_bxb_matrix_kd:distillation/online_teacher.py`의 `itm_matrix`(라인 ~116)·`itm_matrix_gathered`(라인 ~133) 메서드를 dev `online_teacher.py`의 `OnlineTeacher`에 추가. (dev엔 이미 `itm` keep 경로가 있음 — `NEEDS['itm']`/`CRITICAL_PREFIXES['itm']` 확인, itm_head 유지됨.)

- [ ] **Step 3: 구문 확인**

Run: `python -m py_compile distillation/online_teacher.py`
Expected: 에러 없음. (모델 import 실행 안 되므로 transformers 이슈 무관.)

- [ ] **Step 4: 커밋**

```bash
git add distillation/online_teacher.py
git commit -m "port(itm-kd): OnlineTeacher.itm_matrix + itm_matrix_gathered"
```

---

### Task 3: itc_distill_loss(별도 ITC KD) 스트립

**Files:**
- Modify: `distillation/losses.py` (−itc_distill_loss)
- Modify: `models/blip_pretrain.py` (forward −loss_itc_kd, return, 시그니처 −distill_temp)
- Modify: `pretrain.py` (−distill.itc 읽기·배선·상호배타 assert, 언팩)
- Modify: `data/eval_validation_loss.py` (언팩)

**Interfaces:**
- Produces: forward return `loss_ita, loss_itm, loss_lm, loss_lm_kd` (4-tuple, itc_kd 제거)

- [ ] **Step 1: forward에서 loss_itc_kd 블록 제거**

`models/blip_pretrain.py` forward의 loss_itc_kd 계산(현 543~550: `loss_itc_kd = None` + `if teacher_img_feat...: loss_itc_kd = itc_distill_loss(...)`) 삭제. **주의**: `teacher_img_feat/teacher_text_feat`는 ttm이 계속 쓰므로 시그니처에서 **유지**. `distill_temp` 파라미터는 itc_distill_loss 전용이었으니 **제거**.

- [ ] **Step 2: return에서 loss_itc_kd 제거**

`return loss_ita, loss_itm, loss_lm, loss_lm_kd` (현 562 `..., loss_itc_kd, loss_lm_kd` → itc_kd 삭제).

- [ ] **Step 3: import·함수 제거**

`blip_pretrain.py` 상단 `from distillation.losses import itc_distill_loss, lm_distill_loss, itm_target_mix_loss`에서 `itc_distill_loss` 제거(이번 단계). `distillation/losses.py`의 `itc_distill_loss`(라인 5~34) 함수 삭제.

- [ ] **Step 4: pretrain.py 배선 제거 + 언팩 갱신**

`pretrain.py`에서 `distill_itc`/`itc_kd_enabled`/`itc_kd_weight`/`itc_kd_temp` 읽기(85~88), 상호배타 assert(97~98), `distill_temp=itc_kd_temp` 인자, `if itc_kd_enabled...: loss += itc_kd_weight*loss_itc_kd`(202~203, 215~216), 로깅(226~227, 239~240) 삭제. 언팩 2곳(194, 207)을 `loss_ita, loss_itm, loss_lm, loss_lm_kd = model(...)`로.

- [ ] **Step 5: eval_validation_loss.py 언팩 갱신**

`data/eval_validation_loss.py:297`을 `loss_ita, loss_itm, loss_lm, _loss_lm_kd = model(...)`로 (itc_kd 제거).

- [ ] **Step 6: 구문·테스트 확인**

Run: `python -m py_compile models/blip_pretrain.py pretrain.py data/eval_validation_loss.py distillation/losses.py && python -m pytest distillation/test_losses.py distillation/test_target_mix.py -q`
Expected: py_compile 에러 없음; 테스트 PASS. (test_losses에서 itc_distill_loss 테스트가 있으면 함께 삭제.)

- [ ] **Step 7: 커밋**

```bash
git add -A && git commit -m "strip(itc): remove separate itc_distill_loss KD (ttm과 상호배타·항상 off)"
```

---

### Task 4: ITM 경로 swap — itm_target_mix 제거 + itm k=4 배선

**Files:**
- Modify: `models/blip_pretrain.py` (forward itm 경로·시그니처·return)
- Modify: `pretrain.py` (−distill.itm_target_mix, +distill.itm k4, 언팩)
- Modify: `data/eval_validation_loss.py` (언팩)
- Modify: `distillation/online_teacher.py` (−itm_soft)
- Modify: `distillation/losses.py` (−itm_target_mix_loss)
- Delete: `distillation/test_losses_itm.py`, `models/test_blip_pretrain_itm.py`

**Interfaces:**
- Consumes: Task2의 `online_teacher.itm_matrix_gathered`
- Produces: forward return `loss_ita, loss_itm, loss_lm, loss_lm_kd, loss_itm_kd` (5-tuple, 최종형)

- [ ] **Step 1: forward 시그니처 swap**

`itm_mix=None` 파라미터 제거, `itm_topk=-1, itm_distill_temp=0.05, itm_distill_direction='bidir'` 추가 (itm_bxb forward 시그니처 참조: `git show itm_bxb_matrix_kd:models/blip_pretrain.py` 라인 348~351).

- [ ] **Step 2: forward itm 경로 교체**

현 dev forward의 negative-mining(476~500)은 유지. ITM 손실 블록(516~526, `if itm_mix...: loss_itm = itm_target_mix_loss(...) else: F.cross_entropy`)을 **plain CE 고정**으로: `loss_itm = F.cross_entropy(vl_output, itm_labels)`. 그 뒤에 itm k=4 KD 블록 **전체**(학생 gather-idx 산출 + 학생 pair 로짓 계산 + 티처 gather + `itm_gathered_kd_loss` 호출)를 `git show itm_bxb_matrix_kd:models/blip_pretrain.py` 527~552에서 **verbatim 이식** — 학생측 로짓 계산이 미묘하므로 재작성 금지, 그대로 복사. return에 `loss_itm_kd` 추가.

- [ ] **Step 3: online_teacher.itm_soft 제거**

`distillation/online_teacher.py`의 `itm_soft`(133~160) 삭제. (itm_matrix/itm_matrix_gathered는 Task2에서 이미 대체.)

- [ ] **Step 4: itm_target_mix_loss 제거**

`distillation/losses.py`의 `itm_target_mix_loss`(67~77) 삭제. `blip_pretrain.py` import에서 `itm_target_mix_loss` 제거, itm_bxb KD import 추가: `from distillation.losses import lm_distill_loss, itm_matrix_kd_loss, itm_gathered_kd_loss` + `from distillation.itm_matrix import itm_bxb_logits, itm_pair_logits`. (Step 2에서 이식한 forward 블록이 실제 참조하는 심볼에 맞춰 import 최종 정리 — 안 쓰는 건 빼고.)

- [ ] **Step 5: pretrain.py 배선 swap**

`distill_itm`/itm_target_mix 읽기(114~121)와 `itm_mix` dict 구성(185~187), `itm_online_teacher`(190) 제거. itm_bxb식 배선 이식(`git show itm_bxb_matrix_kd:pretrain.py` 95~99, 148~180): `distill.itm`의 enabled/weight/temp/direction/topk 읽기, forward에 `online_teacher=(online_teacher if itm_kd_enabled else None), itm_topk=..., itm_distill_temp=..., itm_distill_direction=...`, 합산 `if itm_kd_enabled and loss_itm_kd is not None: loss += itm_kd_weight*loss_itm_kd`, 로깅 `loss_itm_kd`. 언팩 2곳을 5-tuple(`..., loss_lm_kd, loss_itm_kd`)로.

- [ ] **Step 6: eval_validation_loss.py 언팩 갱신**

`:297` → `loss_ita, loss_itm, loss_lm, _loss_lm_kd, _loss_itm_kd = model(...)`.

- [ ] **Step 7: 죽은 테스트 삭제**

```bash
git rm distillation/test_losses_itm.py models/test_blip_pretrain_itm.py
```
(itm_target_mix·구 itm forward 테스트. itm-k4 forward 테스트는 실행 불가 env라 이식 보류 — 실학습 env에서 itm_bxb `test_forward_kd.py` 참조.) 추가로 `distillation/test_online_teacher.py`에 `itm_soft` 참조가 남아 있으면 제거/갱신(로컬 실행 불가라 구조만 정합).

- [ ] **Step 8: 구문·테스트 확인**

Run: `python -m py_compile models/blip_pretrain.py pretrain.py data/eval_validation_loss.py distillation/online_teacher.py distillation/losses.py && python -m pytest distillation/test_itm_matrix.py distillation/test_losses.py distillation/test_target_mix.py -q`
Expected: py_compile 에러 없음; 순수함수 테스트 PASS.

- [ ] **Step 9: 커밋**

```bash
git add -A && git commit -m "swap(itm): itm_target_mix(exp10) 제거, itm k=4 traditional KD 배선"
```

---

### Task 5: ttm in_batch variant 스트립 (queue만 유지)

**Files:**
- Modify: `distillation/target_mix.py` (−teacher_soft_in_batch)
- Modify: `models/blip_pretrain.py` (forward ttm in_batch 분기 제거)
- Modify: `pretrain.py` (variant assert → queue 강제)
- Modify: `distillation/test_target_mix.py` (teacher_soft_in_batch 테스트 제거)

- [ ] **Step 1: forward ttm 분기 단순화**

`blip_pretrain.py` forward의 ttm 블록(434~451)에서 `if self.ttm_variant == 'queue': ... else: (in_batch) teacher_i2t = teacher_soft_in_batch(...)` 의 **else(in_batch) 가지 삭제**, queue 경로만 남김. `teacher_soft_in_batch` import 제거. `__init__`의 `ttm_variant`는 인터페이스 유지하되 in_batch면 assert(아래 Step 3).

- [ ] **Step 2: target_mix.py 함수 삭제**

`distillation/target_mix.py`의 `teacher_soft_in_batch`(15~22) 삭제.

- [ ] **Step 3: variant 지원값 강제**

`pretrain.py`의 ttm variant assert(108~109)를 `assert distill_ttm.get('variant') == 'queue', "minimal_mainline: itc_target_mix.variant는 'queue'만 지원(in_batch 제거됨)"` 로.

- [ ] **Step 4: 테스트 정리 + 확인**

`distillation/test_target_mix.py`에서 `teacher_soft_in_batch` 관련 테스트 삭제.
Run: `python -m py_compile models/blip_pretrain.py pretrain.py distillation/target_mix.py && python -m pytest distillation/test_target_mix.py -q`
Expected: py_compile 에러 없음; PASS.

- [ ] **Step 5: 커밋**

```bash
git add -A && git commit -m "strip(ttm): remove in_batch variant, queue-only"
```

---

### Task 6: 죽은 arch + 주석 dead 블록 스트립

**Files:**
- Modify: `models/blip_pretrain.py` (`__init__` vit/bert 분기, 주석 블록)

- [ ] **Step 1: 안쓰는 vit 분기 제거**

`__init__`에서 vit `small`(82~92)·`small_plus`(107~116)·`small_plus_reg`(118~128) elif 삭제. `small_reg`(95~105)·`base`(65~72)·`large`(73~80) 유지. 라인 131의 `if vit in ['small','small_plus','small_reg','small_plus_reg']` 리스트를 `['small_reg']`로 축소(유지되는 small 계열은 small_reg뿐). `DINOv3_Wrapper` import는 small_reg가 쓰므로 유지.

- [ ] **Step 2: 안쓰는 bert 제거**

`model_specs` 딕셔너리(147~160)에서 `"medium"` 항목 삭제. `assert my_bert_size in [...]`(180)에서 "medium" 제거.

- [ ] **Step 3: 주석 deprecated 블록 삭제**

`blip_pretrain.py`의 큰 주석 dead 코드 블록: bert 분기 depreciated(182~229), decoder depreciated(314~320) 삭제.

- [ ] **Step 4: 구문 확인**

Run: `python -m py_compile models/blip_pretrain.py`
Expected: 에러 없음.

- [ ] **Step 5: 커밋**

```bash
git add models/blip_pretrain.py
git commit -m "strip(arch): remove unused vit(small/small_plus/small_plus_reg)+bert(medium)+주석 dead"
```

---

### Task 7: 본선 config + 최종 검증

**Files:**
- Create: `configs/pretrain_mainline.yaml`

- [ ] **Step 1: 본선 config 작성**

`configs/pretrain_ttm_queue_lm_holdteacher.yaml`을 베이스로 복사하되 `distill` 블록을 3-메커니즘 전부 ON으로:
```yaml
distill:
  itc_target_mix: {enabled: true, variant: queue, temp: 0.05, soft_weight: 0.4, hold_epochs: 2, decay_end_epochs: 10000000}
  lm:  {enabled: true, weight: 1.0, temp: 2.0}
  itm: {enabled: true, weight: 1.0, temp: 0.05, direction: bidir, topk: 4}
  # distill.itc / distill.itm_target_mix — 제거됨(미지원)
```
`exp: 'refactor.mainline'`, `output_dir`도 갱신. arch(small_reg/minilm), alpha=0.4(soft_weight와 일치) 유지.

- [ ] **Step 2: 전체 순수함수 테스트 스윕**

Run: `python -m pytest distillation/test_target_mix.py distillation/test_losses.py distillation/test_itm_matrix.py -q`
Expected: 전부 PASS.

- [ ] **Step 3: 전체 구문 스윕**

Run: `python -m py_compile models/blip_pretrain.py pretrain.py data/eval_validation_loss.py distillation/*.py`
Expected: 에러 없음.

- [ ] **Step 4: 잔여 참조 확인 (제거 심볼이 안 남았나)**

Run: `grep -rnE "itm_target_mix|itm_soft|itc_distill_loss|teacher_soft_in_batch|loss_itc_kd|itm_mix" --include=*.py . | grep -v test_ | grep -v "\.pyc"`
Expected: 빈 결과(또는 config 키 `itc_target_mix`만 — 이건 별개, 오탐 주의).

- [ ] **Step 5: 커밋**

```bash
git add configs/pretrain_mainline.yaml
git commit -m "config(mainline): itc-ttm(queue)+lm-KD+itm-k4 셋 다 ON 본선 config"
```

---

## 실행 후 남는 것 (범위 밖)
- **실학습 env 검증**: forward/online_teacher 실제 GPU 스모크(itm_matrix_gathered gather 정합, loss finite, 3-경로 공존) — 이 env에선 불가.
- **task 2 모듈화**: 이 최소 브랜치를 출발점으로 forward 안/밖 로직 재구획화 논의(코드-프리).
- **task 3 불안정성 root-cause**: A/B 사다리 + 프로브.
