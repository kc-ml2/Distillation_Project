# ITC + ITM + LM target-mix 구조적 병합 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** exp9(ITC teacher-target-mix)와 exp10(ITM teacher-target-mix)를 dev는 그대로 둔 채 새 브랜치 `itc_itm_lm_targetmix`로 구조적으로 병합해, 세 증류 메커니즘(ITC target-mix / ITM target-mix / 기존 LM distillation)을 config로 독립적·동시적으로 켤 수 있게 만든다.

**Architecture:** `git merge`로 dev(fe23681) → exp9(fast-forward, 무충돌) → exp10(충돌 5곳, 전부 "두 블록 다 유지" 방식으로 해소) 순서 병합. 유일한 실제 재작성은 `distillation/distill_config.py`의 `derive_teacher_keep()` — exp9의 인라인 keep 로직을 흡수해 4가지 메커니즘(itc/lm/itc_target_mix/itm_target_mix) 전부를 커버하도록 확장. 이 문서의 모든 충돌 해소 코드와 신규 코드는 **사전에 디스포저블 클론에서 실제로 병합·실행·테스트하여 검증 완료**된 것이다 — 그대로 옮기면 된다.

**Tech Stack:** Python 3.10, PyTorch, pytest, `/home/minwoo/miniconda3/envs/kd_r4/bin/python`.

## Global Constraints

- 브랜치명: `itc_itm_lm_targetmix`, dev(fe23681)에서 분기. **dev 자체는 건드리지 않는다.**
- 병합 순서: `exp9_teacher_target_mix` 먼저(fast-forward), `worktree-exp10_itm_target_mix` 다음(충돌 해소 필요).
- 충돌 해소는 전부 "두 블록 다 유지"이며 어느 쪽 로직도 삭제하지 않는다. 예외: `pretrain.py`의 `teacher_keep` 산출부는 인라인 두 버전을 버리고 `derive_teacher_keep()` 단일 호출로 교체한다(§Task 2).
- `derive_teacher_keep()` 확장 시 반드시 `itc_target_mix.enabled=True`이면 `'itc'`를 keep 집합에 추가해야 한다 — 이게 빠지면 exp9 config들(`pretrain_ttm_queue.yaml`, `pretrain_ttm_inbatch.yaml`)이 병합 후 조용히 깨진다(teacher_keep이 비어 `online_teacher=None`이 되고, `gamma>0`인데 `teacher_img_feat=None`으로 forward에 전달됨).
- 병합 후에도 exp9/exp10 각자의 기존 상호배타 assert(`assert not (itc_kd_enabled and ttm_enabled)` 등)는 그대로 유지한다.
- 신규 조합 config(`configs/pretrain_itc_itm_lm_targetmix_smoke.yaml`)는 스모크 전용이며 `val_loss_enabled: false`, `val_retrieval_enabled: false`로 부가 검증 루프를 끈다(병합 자체와 무관한 실패 원인 제거).
- Python 실행은 항상 `/home/minwoo/miniconda3/envs/kd_r4/bin/python`을 명시 사용한다(시스템 `python3`가 아님).
- 이번 계획은 Phase 1(구조적 병합)만 다룬다. 교사 forward 중복 제거 최적화, 실제 멀티에폭 본런, itc_target_mix variant(queue vs in_batch) 최종 선택은 범위 밖(스펙 §6) — 어떤 태스크에도 포함하지 않는다.

---

### Task 1: 브랜치 생성 + exp9 fast-forward 병합

**Files:**
- 없음(코드 변경 없음, 순수 git 작업)

**Interfaces:**
- Consumes: 없음
- Produces: 브랜치 `itc_itm_lm_targetmix`가 exp9의 tip(`69a8107` 또는 그 이후 exp9 브랜치의 실제 최신 커밋)을 가리킴. 이후 태스크는 이 브랜치 위에서 작업.

- [ ] **Step 1: 현재 dev의 실제 tip 확인**

```bash
cd /home/minwoo/Distillation_Project
git log -1 --oneline dev
git log -1 --oneline exp9_teacher_target_mix
git merge-base dev exp9_teacher_target_mix
```
Expected: `merge-base`가 `dev`의 tip과 동일(즉 exp9가 dev의 linear descendant). 다르면 STOP하고 사람에게 보고(사전 검증 시점과 dev가 바뀌었다는 뜻).

- [ ] **Step 2: 새 브랜치 생성**

```bash
git checkout -b itc_itm_lm_targetmix dev
```

- [ ] **Step 3: exp9 병합 (fast-forward 예상)**

```bash
git merge --no-edit exp9_teacher_target_mix
```
Expected: `Fast-forward` 메시지, 충돌 없음. Fast-forward가 아니고 충돌이 발생하면 STOP — 이는 사전 검증(디스포저블 클론)과 다른 상황이므로 사람에게 보고.

- [ ] **Step 4: exp9 자체 테스트 회귀 확인**

```bash
/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest distillation/test_target_mix.py -v
```
Expected: 9 passed (exp9 progress ledger 기록과 동일).

- [ ] **Step 5: 커밋 불필요 (fast-forward는 새 커밋을 만들지 않음) — 브랜치 상태만 확인**

```bash
git log --oneline -3
git status --short
```
Expected: working tree clean, HEAD == exp9의 tip.

---

### Task 2: exp10 병합 + 충돌 해소 + `derive_teacher_keep()` 확장

**Files:**
- Modify (병합 충돌 해소): `models/blip_pretrain.py`
- Modify (병합 충돌 해소): `pretrain.py`
- Modify (실제 로직 확장): `distillation/distill_config.py`
- Modify (신규 테스트): `distillation/test_distill_config.py`

**Interfaces:**
- Consumes: `distillation/distill_config.py`의 기존 `derive_teacher_keep(distill_cfg) -> tuple`, `validate_itm_mix_config(distill_cfg) -> None` (exp10에서 옴). `models/blip_pretrain.py`의 `forward()`가 exp9는 `gamma` kwarg만, exp10은 `online_teacher`/`itm_mix` kwarg만 쓴다는 사실.
- Produces: `derive_teacher_keep()`이 `itc_target_mix.enabled=True`일 때도 `'itc'`를 포함하도록 확장됨 — 이후 태스크(Task 4, 5)가 이 동작에 의존.

- [ ] **Step 1: exp10 병합 시도 (충돌 예상)**

```bash
git merge --no-edit worktree-exp10_itm_target_mix
```
Expected: `CONFLICT (content): Merge conflict in models/blip_pretrain.py` 와
`CONFLICT (content): Merge conflict in pretrain.py` 딱 2개 파일만 충돌. 다른 파일이 충돌하면 STOP — 사전 검증과 다른 상황.

- [ ] **Step 2: `models/blip_pretrain.py` 충돌 해소 (forward 시그니처, 1곳)**

`grep -n "<<<<<<<" models/blip_pretrain.py`로 위치 확인 후, 아래처럼 정확히 치환:

```python
# 치환 전 (conflict markers 포함):
                teacher_lm_logits=None, teacher_lm_input_ids=None, lm_distill_temp=2.0,
<<<<<<< HEAD
                gamma=None):
=======
                online_teacher=None, itm_mix=None):
>>>>>>> worktree-exp10_itm_target_mix

# 치환 후:
                teacher_lm_logits=None, teacher_lm_input_ids=None, lm_distill_temp=2.0,
                gamma=None, online_teacher=None, itm_mix=None):
```

- [ ] **Step 3: `pretrain.py` 충돌 해소 (5곳)**

`grep -n "<<<<<<<\|=======\|>>>>>>>" pretrain.py`로 6개 마커 블록(3쌍×2, 총 5개 논리적 위치 — 2곳은 동일 패턴이 2번 반복) 위치 확인. 아래 5개 지점을 순서대로 정확히 치환한다.

**(a) distill config 파싱 블록** (`train()` 함수 상단, `distill_lm` 파싱 직후):

```python
# 치환 전:
<<<<<<< HEAD
    distill_ttm = config.get('distill', {}).get('itc_target_mix', {})
    ttm_enabled = distill_ttm.get('enabled', False)
    assert not (itc_kd_enabled and ttm_enabled), \
        "distill.itc(별도 KD)와 distill.itc_target_mix 동시 사용 금지"
    ttm_soft_weight = float(distill_ttm.get('soft_weight', 0.4))
    ttm_hold_steps = int(distill_ttm.get('hold_epochs', 2) * len(data_loader))
    ttm_decay_end_steps = int(distill_ttm.get('decay_end_epochs', 12) * len(data_loader))

    if ttm_enabled:
        assert 0.0 <= ttm_soft_weight <= 1.0, \
            f"itc_target_mix.soft_weight out of [0,1]: {ttm_soft_weight}"
        assert ttm_hold_steps <= ttm_decay_end_steps, \
            f"itc_target_mix hold_epochs must be <= decay_end_epochs (steps {ttm_hold_steps} > {ttm_decay_end_steps})"
        assert distill_ttm.get('variant', 'in_batch') in ('in_batch', 'queue'), \
            f"itc_target_mix.variant must be 'in_batch' or 'queue': {distill_ttm.get('variant')}"
        assert abs(float(config['alpha']) - ttm_soft_weight) < 1e-9, \
            f"target-mix tail-skip requires config.alpha == soft_weight (got alpha={config['alpha']}, soft_weight={ttm_soft_weight}); " \
            f"else the γ=0 tail target diverges from baseline"
=======
    distill_itm = config.get('distill', {}).get('itm_target_mix', {})
    itm_mix_enabled = distill_itm.get('enabled', False)
    itm_neg_source = distill_itm.get('neg_source', 'student')
    itm_soft_weight = float(distill_itm.get('soft_weight', 0.0))
    itm_teacher_temp = float(distill_itm.get('temp', 1.0))
    itm_sel_scale = online_teacher.teacher_scale if online_teacher is not None else 1.0
>>>>>>> worktree-exp10_itm_target_mix

# 치환 후 (마커 제거, 두 블록 순서대로 이어붙임):
    distill_ttm = config.get('distill', {}).get('itc_target_mix', {})
    ttm_enabled = distill_ttm.get('enabled', False)
    assert not (itc_kd_enabled and ttm_enabled), \
        "distill.itc(별도 KD)와 distill.itc_target_mix 동시 사용 금지"
    ttm_soft_weight = float(distill_ttm.get('soft_weight', 0.4))
    ttm_hold_steps = int(distill_ttm.get('hold_epochs', 2) * len(data_loader))
    ttm_decay_end_steps = int(distill_ttm.get('decay_end_epochs', 12) * len(data_loader))

    if ttm_enabled:
        assert 0.0 <= ttm_soft_weight <= 1.0, \
            f"itc_target_mix.soft_weight out of [0,1]: {ttm_soft_weight}"
        assert ttm_hold_steps <= ttm_decay_end_steps, \
            f"itc_target_mix hold_epochs must be <= decay_end_epochs (steps {ttm_hold_steps} > {ttm_decay_end_steps})"
        assert distill_ttm.get('variant', 'in_batch') in ('in_batch', 'queue'), \
            f"itc_target_mix.variant must be 'in_batch' or 'queue': {distill_ttm.get('variant')}"
        assert abs(float(config['alpha']) - ttm_soft_weight) < 1e-9, \
            f"target-mix tail-skip requires config.alpha == soft_weight (got alpha={config['alpha']}, soft_weight={ttm_soft_weight}); " \
            f"else the γ=0 tail target diverges from baseline"

    distill_itm = config.get('distill', {}).get('itm_target_mix', {})
    itm_mix_enabled = distill_itm.get('enabled', False)
    itm_neg_source = distill_itm.get('neg_source', 'student')
    itm_soft_weight = float(distill_itm.get('soft_weight', 0.0))
    itm_teacher_temp = float(distill_itm.get('temp', 1.0))
    itm_sel_scale = online_teacher.teacher_scale if online_teacher is not None else 1.0
```

**(b) `need_teacher_itc` 게이팅** (학습 루프 내부, `image = image.to(...)` 직후):

```python
# 치환 전:
<<<<<<< HEAD
        from distillation.target_mix import ttm_gamma
        gamma = ttm_gamma(global_step, ttm_hold_steps, ttm_decay_end_steps) if ttm_enabled else None

        # online teacher: same augmented batch -> ITC features (bf16, no_grad). None when distill off.
        # target-mix: teacher only while γ>0 (γ=0 tail is target-identical to baseline → skip teacher forward)
        if (itc_kd_enabled or (ttm_enabled and gamma is not None and gamma > 0)) and online_teacher is not None:
=======
        # online teacher ITC feats: for ITC KD and/or teacher-guided ITM neg selection.
        need_teacher_itc = itc_kd_enabled or (itm_mix_enabled and itm_neg_source == 'teacher')
        if need_teacher_itc and online_teacher is not None:
>>>>>>> worktree-exp10_itm_target_mix

# 치환 후:
        from distillation.target_mix import ttm_gamma
        gamma = ttm_gamma(global_step, ttm_hold_steps, ttm_decay_end_steps) if ttm_enabled else None

        # online teacher: same augmented batch -> ITC features (bf16, no_grad). None when not needed.
        # target-mix: teacher only while γ>0 (γ=0 tail is target-identical to baseline → skip teacher forward)
        need_teacher_itc = (
            itc_kd_enabled
            or (ttm_enabled and gamma is not None and gamma > 0)
            or (itm_mix_enabled and itm_neg_source == 'teacher')
        )
        if need_teacher_itc and online_teacher is not None:
```

**(c) `model(...)` 호출부 — cuda-autocast 분기** (`with torch.amp.autocast(...)` 블록 내부):

```python
# 치환 전:
<<<<<<< HEAD
                    gamma=gamma)
=======
                    online_teacher=itm_online_teacher, itm_mix=itm_mix)
>>>>>>> worktree-exp10_itm_target_mix

# 치환 후:
                    gamma=gamma, online_teacher=itm_online_teacher, itm_mix=itm_mix)
```
(들여쓰기 20칸 — cuda 분기 내부의 `model(...)` 호출부. 아래 (d)의 16칸 들여쓰기 버전과 헷갈리지 말 것.)

**(d) `model(...)` 호출부 — plain(비-cuda) 분기**:

```python
# 치환 전:
<<<<<<< HEAD
                gamma=gamma)
=======
                online_teacher=itm_online_teacher, itm_mix=itm_mix)
>>>>>>> worktree-exp10_itm_target_mix

# 치환 후:
                gamma=gamma, online_teacher=itm_online_teacher, itm_mix=itm_mix)
```
(들여쓰기 16칸.)

**(e) TB 로깅 블록** (`writer.add_scalar("train/alpha", ...)` 직후):

```python
# 치환 전:
<<<<<<< HEAD
                if ttm_enabled:
                    writer.add_scalar("train/gamma", gamma, global_step)
                    writer.add_scalar("train/beta", ttm_soft_weight * gamma, global_step)
=======
                if itm_mix_enabled:
                    writer.add_scalar("train/itm_teacher_weight", itm_soft_weight, global_step)
>>>>>>> worktree-exp10_itm_target_mix

# 치환 후:
                if ttm_enabled:
                    writer.add_scalar("train/gamma", gamma, global_step)
                    writer.add_scalar("train/beta", ttm_soft_weight * gamma, global_step)
                if itm_mix_enabled:
                    writer.add_scalar("train/itm_teacher_weight", itm_soft_weight, global_step)
```

**(f) `teacher_keep` 산출** (`main()` 함수, `#### online teacher (distillation) ...` 블록):

```python
# 치환 전:
<<<<<<< HEAD
    ttm_on = distill_cfg.get('itc_target_mix', {}).get('enabled', False)
    teacher_keep = tuple(k for k in ('itc', 'lm')
                         if distill_cfg.get(k, {}).get('enabled', False)
                         or (k == 'itc' and ttm_on))
=======
    validate_itm_mix_config(distill_cfg)
    teacher_keep = derive_teacher_keep(distill_cfg)
>>>>>>> worktree-exp10_itm_target_mix

# 치환 후 (인라인 두 버전 다 버리고 통합 함수 호출로 교체 — Global Constraints 참고):
    validate_itm_mix_config(distill_cfg)
    teacher_keep = derive_teacher_keep(distill_cfg)
```

- [ ] **Step 4: 마커 잔존 여부 + 문법 확인**

```bash
grep -n "^<<<<<<<\|^=======\|^>>>>>>>" models/blip_pretrain.py pretrain.py
```
Expected: 출력 없음(마커 완전 제거).

```bash
/home/minwoo/miniconda3/envs/kd_r4/bin/python -c "import ast; ast.parse(open('models/blip_pretrain.py').read()); ast.parse(open('pretrain.py').read()); print('OK')"
```
Expected: `OK`.

- [ ] **Step 5: `derive_teacher_keep()` 확장 — 실패하는 테스트부터 작성**

`distillation/test_distill_config.py`의 `TestDeriveTeacherKeep` 클래스에 아래 3개 테스트를 `test_combines_with_itc_lm` 다음에 추가:

```python
    def test_itc_target_mix_alone_needs_itc(self):
        cfg = {'itc_target_mix': {'enabled': True, 'variant': 'queue', 'soft_weight': 0.4}}
        self.assertEqual(derive_teacher_keep(cfg), ('itc',))

    def test_itc_target_mix_combines_with_itm_target_mix(self):
        cfg = {'itc_target_mix': {'enabled': True, 'variant': 'queue', 'soft_weight': 0.4},
               'itm_target_mix': {'enabled': True, 'neg_source': 'teacher', 'soft_weight': 0.4}}
        self.assertEqual(derive_teacher_keep(cfg), ('itc', 'itm'))

    def test_all_three_mechanisms_together(self):
        cfg = {'lm': {'enabled': True},
               'itc_target_mix': {'enabled': True, 'variant': 'queue', 'soft_weight': 0.4},
               'itm_target_mix': {'enabled': True, 'neg_source': 'teacher', 'soft_weight': 0.4}}
        self.assertEqual(derive_teacher_keep(cfg), ('itc', 'itm', 'lm'))
```

- [ ] **Step 6: 테스트 실패 확인**

```bash
/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest distillation/test_distill_config.py -v -k "itc_target_mix or all_three"
```
Expected: 3개 다 FAIL (`derive_teacher_keep`이 아직 `itc_target_mix`를 모름 → `test_itc_target_mix_alone_needs_itc`는 `()`를 반환해 `('itc',)`와 불일치, 나머지 2개도 `'itc'` 누락으로 불일치).

- [ ] **Step 7: `derive_teacher_keep()` 확장 구현**

`distillation/distill_config.py`를 아래로 전체 교체:

```python
"""Teacher keep-set derivation and itm_target_mix config validation."""


def derive_teacher_keep(distill_cfg):
    """Which teacher submodule paths must be kept, given the distill config.
    itc/lm: their own enabled flags. itc_target_mix: 'itc' whenever enabled (teacher
    ITC feats feed the γ-mixed target). itm_target_mix: 'itc' when neg_source=='teacher'
    (selection feats), 'itm' when soft_weight>0 (teacher scoring)."""
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


def validate_itm_mix_config(distill_cfg):
    """Assert itm_target_mix config is well-formed (no-op when disabled/absent)."""
    itm = distill_cfg.get('itm_target_mix', {})
    if not itm.get('enabled', False):
        return
    ns = itm.get('neg_source')
    assert ns in ('teacher', 'student'), \
        f"itm_target_mix.neg_source must be 'teacher'|'student', got {ns!r}"
    w = float(itm.get('soft_weight', 0.0))
    assert 0.0 <= w <= 1.0, f"itm_target_mix.soft_weight must be in [0,1], got {w}"
    sched = itm.get('schedule', 'constant')
    assert sched == 'constant', \
        f"itm_target_mix.schedule only 'constant' implemented, got {sched!r}"
```

- [ ] **Step 8: 테스트 통과 확인**

```bash
/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest distillation/test_distill_config.py -v
```
Expected: 18 passed (기존 15 + 신규 3).

- [ ] **Step 9: 전체 회귀 스위트 실행**

```bash
/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest distillation/ models/test_blip_pretrain_itm.py -v
```
Expected: 65 passed (exp9의 9 + exp10의 53 + 방금 추가한 3 = 65. 사전 검증 시 62/62 확인했고, 이번 태스크에서 3개를 추가로 넣었으므로 65).

- [ ] **Step 10: 커밋**

```bash
git add models/blip_pretrain.py pretrain.py distillation/distill_config.py distillation/test_distill_config.py
git commit -m "merge: resolve exp9/exp10 conflicts + extend derive_teacher_keep for itc_target_mix"
```

---

### Task 3: `OnlineTeacher` 3-way keep (`itc`+`itm`+`lm`) 단위테스트

**Files:**
- Modify: `distillation/test_online_teacher.py`

**Interfaces:**
- Consumes: `OnlineTeacher(checkpoint, image_size, vit, bert, queue_size, keep=(...))`, `itc_feats(image, caption)`, `lm_logits(image, caption)`, `itm_soft(image, enc_input_ids, attention_mask, neg_idx_img, neg_idx_txt, temp)` — 전부 exp10에서 이미 구현·테스트된 기존 인터페이스, 이번 태스크는 새 `keep` 조합만 검증.
- Produces: 없음(이후 태스크가 의존하지 않는 순수 검증).

- [ ] **Step 1: 테스트 클래스 추가**

`distillation/test_online_teacher.py`에서 `class TestOnlineTeacherLargeConstruction(unittest.TestCase):` 바로 앞에 아래 클래스를 삽입:

```python
class TestOnlineTeacherKeepAllThree(unittest.TestCase):
    """keep=('itc','itm','lm'): 3-way 합집합 생존 — 3개 증류 메커니즘이 동시에 켜지는
    병합 config(pretrain_itc_itm_lm_targetmix_smoke.yaml)가 요구하는 조합."""

    @classmethod
    def setUpClass(cls):
        cls.teacher = OnlineTeacher(checkpoint="", image_size=224, vit="base",
                                    bert="base", queue_size=240, keep=("itc", "itm", "lm"))

    def test_union_kept(self):
        m = self.teacher.model
        for attr in ("visual_encoder", "text_encoder", "vision_proj",
                     "text_proj", "text_decoder", "itm_head"):
            self.assertIsNotNone(getattr(m, attr), f"{attr} should be kept")
        for attr in ("visual_encoder_m", "text_encoder_m", "vision_proj_m", "text_proj_m"):
            self.assertIsNone(getattr(m, attr), f"{attr} should be freed")

    def test_all_three_entry_points_work(self):
        image = torch.randn(2, 3, 224, 224)
        caption = ["a green field", "a red car"]
        img_feat, txt_feat = self.teacher.itc_feats(image, caption)
        self.assertEqual(tuple(img_feat.shape), (2, 256))
        logits, dec_ids = self.teacher.lm_logits(image, caption)
        self.assertEqual(tuple(dec_ids.shape), (2, 30))
        enc_ids = txt_feat.new_zeros(2, 5, dtype=torch.long)  # dummy encoder-side ids
        atts = txt_feat.new_ones(2, 5, dtype=torch.long)
        soft = self.teacher.itm_soft(image, enc_ids, atts,
                                     neg_idx_img=[1, 0], neg_idx_txt=[1, 0], temp=1.0)
        self.assertEqual(tuple(soft.shape), (6, 2))
```

- [ ] **Step 2: 실행 확인 (이미 사전 검증에서 통과 확인됨 — 여기선 재확인)**

```bash
/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest distillation/test_online_teacher.py -v -k KeepAllThree
```
Expected: 2 passed.

- [ ] **Step 3: 파일 전체 회귀 (기존 케이스 안 깨졌는지)**

```bash
/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest distillation/test_online_teacher.py -v
```
Expected: 기존 케이스 전부 + 신규 2개 = 전체 pass, 0 fail.

- [ ] **Step 4: 커밋**

```bash
git add distillation/test_online_teacher.py
git commit -m "test(online-teacher): add 3-way keep(itc+itm+lm) coverage"
```

---

### Task 4: 병합 스모크 config 신규 작성

**Files:**
- Create: `configs/pretrain_itc_itm_lm_targetmix_smoke.yaml`

**Interfaces:**
- Consumes: `configs/pretrain_ttm_queue.yaml`(exp9)와 `configs/pretrain_itm_C_teachneg_soft.yaml`(exp10)의 필드 구조. `distill.itc_target_mix`(exp9 스키마), `distill.itm_target_mix`(exp10 스키마), `distill.lm`(dev 기존 스키마).
- Produces: Task 5의 GPU 스모크가 이 파일을 `--config`로 직접 사용.

- [ ] **Step 1: config 파일 작성**

`configs/pretrain_itc_itm_lm_targetmix_smoke.yaml` 신규 생성:

```yaml
train_file: ['/home/minwoo/Distillation_Project/datasets/vision/coco/annotations/coco_karpathy_train.json',
             '/home/minwoo/Distillation_Project/datasets/vision/vg/annotations/vg_train.json'
             ]
laion_path: ''
image_root_coco: '/home/minwoo/Distillation_Project/datasets/vision/coco/images/'
image_root_vg: '/home/minwoo/Distillation_Project/datasets/vision/vg/images/'

cc12m_tar_path: ''
cc12m_ratio: 5

used_dataset: 'coco_vg'

# ===== Phase 1 병합 스모크: itc_target_mix + itm_target_mix(arm C) + lm distill 동시 ON =====
# 설계: docs/superpowers/specs/2026-07-23-itc-itm-lm-targetmix-merge-design.md §5-2
# 목적: 세 증류 메커니즘을 동시에 켰을 때 teacher_keep=('itc','itm','lm')로 뜨고
# 예외 없이 forward/backward가 도는지 GPU 1대·몇 스텝만 확인. 실제 학습(수렴) 목적 아님.
output_dir: '/home/minwoo/Distillation_Project/output/pt_itc_itm_lm_targetmix_smoke'

exp: '11.merge_smoke_itc_itm_lm'
pretrain_train_aug: true

tb_train_log_interval: 50
tb_val_log_interval: 1000

vit: 'small_reg'
vit_grad_ckpt: False
vit_ckpt_layer: 0

my_bert_size: 'minilm'

# ===== 스모크 목적상 검증 루프는 전부 끔 (병합 자체와 무관한 부가 기능) =====
val_loss_enabled: false
val_retrieval_enabled: false

image_size: 224
batch_size: 40

queue_size: 57600
alpha: 0.4

weight_decay: 0.05
init_lr: 1.0e-5

min_lr: 0.0667e-6
warmup_lr: 0.0667e-6
lr_decay_rate: 0.9
max_epoch: 20
warmup_steps: 37000

# ====== knowledge distillation: 3개 메커니즘 동시 ON ======
distill:
  itc:
    enabled: false          # itc_target_mix와 상호배타 (assert)
    weight: 1.0
    temp: 0.05
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
  itm_target_mix:            # arm C: 가장 빡센 조합(teacher neg + soft label)
    enabled: true
    neg_source: teacher
    soft_weight: 0.4
    temp: 1.0
    schedule: constant

teacher:
  arch: 'blip_large'
  checkpoint: '/home/minwoo/Distillation_Project/output/official_pretrain_checkpoint/model_large.pth'
```

- [ ] **Step 2: dry validation (실제 학습 없이 config 자체 정합성만 확인)**

```bash
/home/minwoo/miniconda3/envs/kd_r4/bin/python -c "
import yaml
from distillation.distill_config import derive_teacher_keep, validate_itm_mix_config
cfg = yaml.safe_load(open('configs/pretrain_itc_itm_lm_targetmix_smoke.yaml'))
distill_cfg = cfg['distill']
validate_itm_mix_config(distill_cfg)
keep = derive_teacher_keep(distill_cfg)
print('teacher_keep =', keep)
assert keep == ('itc', 'itm', 'lm'), keep
assert abs(cfg['alpha'] - distill_cfg['itc_target_mix']['soft_weight']) < 1e-9
assert not (distill_cfg['itc']['enabled'] and distill_cfg['itc_target_mix']['enabled'])
import os
assert os.path.isfile(cfg['teacher']['checkpoint']), 'checkpoint missing'
print('ALL DRY CHECKS OK')
"
```
Expected: `teacher_keep = ('itc', 'itm', 'lm')` 및 `ALL DRY CHECKS OK`.

- [ ] **Step 3: 커밋**

```bash
git add configs/pretrain_itc_itm_lm_targetmix_smoke.yaml
git commit -m "feat(merge-smoke): add joint itc_target_mix+itm_target_mix+lm config"
```

---

### Task 5: GPU 1대 통합 스모크

**Files:**
- 없음(코드 변경 없음, 실행 검증만)

**Interfaces:**
- Consumes: Task 4의 `configs/pretrain_itc_itm_lm_targetmix_smoke.yaml`, `pretrain.py`의 `-m pretrain --config=` 엔트리포인트.
- Produces: 없음(Phase 1 종료 검증). 실패 시 Task 2의 병합 해소나 Task 4의 config에 결함이 있다는 뜻 — 해당 태스크로 돌아가 수정.

- [ ] **Step 1: idle GPU 확인**

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
```
가장 여유 있는 GPU 번호를 확인(이 시점 기준 자원 상황에 따라 다를 수 있음 — 이전 세션 기준으로는 GPU 3이 여유 있었으나 재확인 필수).

- [ ] **Step 2: 몇 스텝만 실행 후 수동 중단**

```bash
CUDA_VISIBLE_DEVICES=<여유GPU번호> /home/minwoo/miniconda3/envs/kd_r4/bin/torchrun \
  --nproc_per_node=1 --master_port=29505 \
  -m pretrain --config=./configs/pretrain_itc_itm_lm_targetmix_smoke.yaml
```
로그에서 아래를 확인한 뒤 Ctrl+C로 중단(수 분, 수 스텝이면 충분 — 전체 에폭을 돌 필요 없음):
- `[distill] online teacher loaded (keep=('itc', 'itm', 'lm'))` 로그 라인 출현 확인
- 이후 스텝 로그의 `loss_ita`, `loss_itm`, `loss_lm` 값이 전부 유한(NaN/Inf 아님)
- 크래시/예외 없이 최소 3~5 스텝 진행

- [ ] **Step 3: 스모크 결과 기록**

결과(성공/실패, teacher_keep 로그, 관측 loss 값 범위)를 `.superpowers/sdd/progress.md`(브랜치 워크트리 내)에 한 줄 요약으로 남긴다 — Task 6은 없으므로 이게 Phase 1의 마지막 검증 기록.

---

## Self-Review 메모 (계획 작성자용, 실행 불필요)

- 스펙 §5의 5개 산출물 ↔ 이 계획의 매핑: (1) derive_teacher_keep 통합+테스트 = Task 2, (2) 조합 config = Task 4, (3) GPU 스모크 = Task 5, (4) OnlineTeacher 3-way keep 테스트 = Task 3, (5) 기존 회귀 확인 = Task 2 Step 9(사전에 디스포저블 클론에서 62/62 통과 실측 완료).
- 모든 충돌 해소 코드와 신규 코드(derive_teacher_keep 확장, 3-way keep 테스트, 조합 config)는 이 계획 작성 직전 디스포저블 클론(`/tmp/.../scratchpad/trial_merge`, 이미 삭제됨)에서 실제 `git merge` + `pytest` + dry-run으로 전부 검증했다 — 플레이스홀더 없음.
