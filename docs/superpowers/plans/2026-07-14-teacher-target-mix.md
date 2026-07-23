# Teacher-target-mixing ITC distillation (exp9) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 티처의 유사도 분포를 ITC soft-target에 녹여(별도 KD 손실 대신) gradient 충돌 없이 증류하는 config-gated 기능을 추가한다.

**Architecture:** 순수 함수(스케줄 γ, 티처 soft-target 빌더, 타깃 혼합, 페어 enqueue)를 `distillation/target_mix.py`로 추출해 DDP·모델 없이 단위 테스트한다. `models/blip_pretrain.py`는 forward의 타깃 계산에 `if self.ttm_enabled` 분기를 더해 이 순수 함수를 호출하고(OFF 분기는 기존 코드 그대로 = baseline 불변), variant=='queue'일 때 티처 큐 버퍼를 등록·페어 enqueue한다. `pretrain.py`는 매 스텝 γ를 계산해 forward에 넘기고 `train/beta`·`train/gamma`를 TB에 기록한다.

**Tech Stack:** PyTorch 2.6 (kd_r4 env), unittest(stdlib, pytest 없음), DDP torchrun, TensorBoard.

## Global Constraints

- 기본 OFF: `distill.itc_target_mix.enabled` 미설정 시 baseline 동작과 **bit-identical**. OFF 분기는 기존 타깃 코드를 한 줄도 바꾸지 않는다.
- 별도 KD(`distill.itc.enabled`)와 **상호배타** — 둘 다 true면 assert로 즉시 실패.
- 테스트 실행: `cd /home/minwoo/Distillation_Project && /home/minwoo/miniconda3/envs/kd_r4/bin/python -m unittest distillation.test_target_mix -v`
- 커밋 브랜치: `exp9_teacher_target_mix` (워크트리는 실행 시 `superpowers:using-git-worktrees`로 생성). 커밋 메시지 끝에 `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- 타깃 분포 불변식: 모든 행이 합=1·비음수. one-hot=1−soft_weight 상수, teacher=soft_weight·γ, momentum=soft_weight·(1−γ).
- 스케줄: γ=1 (0 ≤ step < hold), 선형 1→0 (hold ≤ step < decay_end), 0 (step ≥ decay_end). hold=2ep, decay_end=12ep (× len(data_loader)).

---

## File Structure

- **Create** `distillation/target_mix.py` — 순수 함수 5개(`ttm_gamma`, `teacher_soft_in_batch`, `teacher_soft_queue`, `mix_target`, `enqueue_all`). 모델·DDP 의존 없음.
- **Create** `distillation/test_target_mix.py` — 위 함수들의 unittest.
- **Modify** `models/blip_pretrain.py` — `BLIP_Pretrain.__init__`(ttm kwargs + 티처 큐 버퍼), `forward`(gamma 인자 + 타깃 분기), `_dequeue_and_enqueue`(페어 enqueue).
- **Modify** `pretrain.py` — ttm config 파싱, γ 매 스텝 계산, forward 인자, TB 기록, teacher-feat 계산 조건, online_teacher 빌드 조건.
- **Create** `configs/pretrain_ttm_inbatch.yaml`, `configs/pretrain_ttm_queue.yaml` — 두 variant 런 설정.

---

## Task 1: γ 스케줄 순수 함수

**Files:**
- Create: `distillation/target_mix.py`
- Test: `distillation/test_target_mix.py`

**Interfaces:**
- Produces: `ttm_gamma(global_step: int, hold_steps: int, decay_end_steps: int) -> float`

- [ ] **Step 1: 실패하는 테스트 작성** — `distillation/test_target_mix.py`

```python
import unittest
import torch
import torch.nn.functional as F

from distillation.target_mix import ttm_gamma


class TestTtmGamma(unittest.TestCase):
    def test_hold_region_is_one(self):
        self.assertEqual(ttm_gamma(0, 100, 600), 1.0)
        self.assertEqual(ttm_gamma(99, 100, 600), 1.0)
        self.assertEqual(ttm_gamma(100, 100, 600), 1.0)   # 감쇠 시작점 포함

    def test_linear_midpoint(self):
        # hold=100, end=600 → 감쇠구간 500. 중점 step=350 → γ=0.5
        self.assertAlmostEqual(ttm_gamma(350, 100, 600), 0.5, places=6)

    def test_decay_end_and_beyond_is_zero(self):
        self.assertEqual(ttm_gamma(600, 100, 600), 0.0)
        self.assertEqual(ttm_gamma(999, 100, 600), 0.0)

    def test_monotonic_non_increasing(self):
        vals = [ttm_gamma(s, 100, 600) for s in range(0, 700, 10)]
        self.assertTrue(all(a >= b for a, b in zip(vals, vals[1:])))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m unittest distillation.test_target_mix -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'distillation.target_mix'`

- [ ] **Step 3: 최소 구현** — `distillation/target_mix.py`

```python
import torch
import torch.nn.functional as F


def ttm_gamma(global_step, hold_steps, decay_end_steps):
    """teacher↔momentum soft-slot 분배 γ ∈ [0,1].
    γ=1 (홀드), hold~decay_end 선형 1→0, 이후 0. decay_end>hold 전제."""
    if global_step < hold_steps:
        return 1.0
    if global_step >= decay_end_steps:
        return 0.0
    return 1.0 - (global_step - hold_steps) / (decay_end_steps - hold_steps)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m unittest distillation.test_target_mix -v`
Expected: PASS (4 tests)

- [ ] **Step 5: 커밋**

```bash
git add distillation/target_mix.py distillation/test_target_mix.py
git commit -m "feat(ttm): γ 스케줄 함수 (2ep 홀드 + 선형 감쇠)"
```

---

## Task 2: 티처 soft-target 빌더 + 타깃 혼합

**Files:**
- Modify: `distillation/target_mix.py`
- Test: `distillation/test_target_mix.py`

**Interfaces:**
- Consumes: (없음 — 순수 텐서 연산)
- Produces:
  - `teacher_soft_in_batch(row_feat: Tensor[B,D], col_feat: Tensor[B,D], tau: float, n_cols: int) -> Tensor[B, n_cols]` (variant D: in-batch softmax, queue 열 0 패딩)
  - `teacher_soft_queue(row_feat: Tensor[B,D], col_all: Tensor[D, N], tau: float) -> Tensor[B, N]` (variant C)
  - `mix_target(onehot: Tensor[B,N], momentum_soft: Tensor[B,N], teacher_soft: Tensor[B,N], gamma: float, soft_weight: float) -> Tensor[B,N]`

- [ ] **Step 1: 실패하는 테스트 추가** — `distillation/test_target_mix.py`에 클래스 추가

```python
from distillation.target_mix import (
    ttm_gamma, teacher_soft_in_batch, teacher_soft_queue, mix_target,
)


def _norm(x):
    return F.normalize(x, dim=-1)


class TestTeacherSoftAndMix(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B, self.D, self.Q, self.tau = 4, 8, 6, 0.05

    def test_in_batch_shape_and_zero_pad(self):
        row, col = _norm(torch.randn(self.B, self.D)), _norm(torch.randn(self.B, self.D))
        n_cols = self.B + self.Q
        out = teacher_soft_in_batch(row, col, self.tau, n_cols)
        self.assertEqual(tuple(out.shape), (self.B, n_cols))
        self.assertTrue(torch.all(out[:, self.B:] == 0))            # queue 열 0
        self.assertTrue(torch.allclose(out.sum(dim=1), torch.ones(self.B), atol=1e-5))  # 행 합 1

    def test_queue_shape_and_rowsum(self):
        row = _norm(torch.randn(self.B, self.D))
        col_all = _norm(torch.randn(self.B + self.Q, self.D)).t()   # [D, B+Q]
        out = teacher_soft_queue(row, col_all, self.tau)
        self.assertEqual(tuple(out.shape), (self.B, self.B + self.Q))
        self.assertTrue(torch.allclose(out.sum(dim=1), torch.ones(self.B), atol=1e-5))

    def test_mix_target_rowsum_and_endpoints(self):
        N = self.B + self.Q
        onehot = torch.zeros(self.B, N); onehot[:, :self.B].fill_diagonal_(1)
        mom = F.softmax(torch.randn(self.B, N), dim=1)
        tea = F.softmax(torch.randn(self.B, N), dim=1)
        W = 0.4
        for g in (0.0, 0.3, 1.0):
            t = mix_target(onehot, mom, tea, g, W)
            self.assertTrue(torch.allclose(t.sum(dim=1), torch.ones(self.B), atol=1e-5))
            self.assertTrue(torch.all(t >= 0))
        # γ=1 → 티처만, γ=0 → momentum만 (soft 성분)
        t1 = mix_target(onehot, mom, tea, 1.0, W)
        self.assertTrue(torch.allclose(t1, (1 - W) * onehot + W * tea, atol=1e-6))
        t0 = mix_target(onehot, mom, tea, 0.0, W)
        self.assertTrue(torch.allclose(t0, (1 - W) * onehot + W * mom, atol=1e-6))
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m unittest distillation.test_target_mix -v`
Expected: FAIL — `ImportError: cannot import name 'teacher_soft_in_batch'`

- [ ] **Step 3: 구현 추가** — `distillation/target_mix.py`

```python
def teacher_soft_in_batch(row_feat, col_feat, tau, n_cols):
    """variant D: 티처 in-batch softmax [B,B]를 [B, n_cols]로 queue 열 0 패딩."""
    B = row_feat.shape[0]
    sim = (row_feat @ col_feat.t()) / tau            # [B, B]
    soft_bb = F.softmax(sim, dim=1)                   # [B, B]
    out = torch.zeros(B, n_cols, device=soft_bb.device, dtype=soft_bb.dtype)
    out[:, :B] = soft_bb
    return out


def teacher_soft_queue(row_feat, col_all, tau):
    """variant C: 학생 후보군과 정렬된 col_all[D, B+queue] 위 티처 softmax [B, B+queue]."""
    sim = (row_feat @ col_all) / tau                  # [B, B+queue]
    return F.softmax(sim, dim=1)


def mix_target(onehot, momentum_soft, teacher_soft, gamma, soft_weight):
    """(1-W)·onehot + W·(γ·teacher + (1-γ)·momentum). 모든 인자 [B,N] 행분포."""
    soft = gamma * teacher_soft + (1.0 - gamma) * momentum_soft
    return (1.0 - soft_weight) * onehot + soft_weight * soft
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m unittest distillation.test_target_mix -v`
Expected: PASS (7 tests)

- [ ] **Step 5: 커밋**

```bash
git add distillation/target_mix.py distillation/test_target_mix.py
git commit -m "feat(ttm): 티처 soft-target 빌더(in_batch/queue) + 타깃 혼합"
```

---

## Task 3: 페어 enqueue (큐 정렬 보장)

**Files:**
- Modify: `distillation/target_mix.py`
- Test: `distillation/test_target_mix.py`

**Interfaces:**
- Produces: `enqueue_all(pairs: list[tuple[Tensor[D,Q], Tensor[D,bs]]], ptr: int, bs: int, queue_size: int) -> int` — 각 (queue, feats_T)를 **동일 ptr**의 `[:, ptr:ptr+bs]`에 쓰고 ptr을 한 번만 전진. 반환=새 ptr.

- [ ] **Step 1: 실패하는 테스트 추가** — `distillation/test_target_mix.py`

```python
from distillation.target_mix import enqueue_all


class TestEnqueueAll(unittest.TestCase):
    def test_all_queues_written_at_same_columns(self):
        D, Q, bs = 3, 12, 4
        img_q, txt_q = torch.zeros(D, Q), torch.zeros(D, Q)
        t_img_q, t_txt_q = torch.zeros(D, Q), torch.zeros(D, Q)
        img_f = torch.arange(D * bs).float().reshape(D, bs)        # feats_T [D,bs]
        t_img_f = img_f + 100
        ptr = 4
        new_ptr = enqueue_all(
            [(img_q, img_f), (t_img_q, t_img_f)], ptr, bs, Q)
        # 같은 열(4:8)에 각자 값이 쓰였는가
        self.assertTrue(torch.equal(img_q[:, ptr:ptr + bs], img_f))
        self.assertTrue(torch.equal(t_img_q[:, ptr:ptr + bs], t_img_f))
        # 다른 열은 그대로 0
        self.assertTrue(torch.all(img_q[:, :ptr] == 0))
        self.assertEqual(new_ptr, (ptr + bs) % Q)                  # 8

    def test_wraparound(self):
        D, Q, bs = 2, 8, 4
        q = torch.zeros(D, Q)
        f = torch.ones(D, bs)
        new_ptr = enqueue_all([(q, f)], 4, bs, Q)
        self.assertEqual(new_ptr, 0)                               # (4+4)%8
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m unittest distillation.test_target_mix -v`
Expected: FAIL — `ImportError: cannot import name 'enqueue_all'`

- [ ] **Step 3: 구현 추가** — `distillation/target_mix.py`

```python
def enqueue_all(pairs, ptr, bs, queue_size):
    """각 (queue[D,Q], feats_T[D,bs])를 동일 ptr의 열에 써서 큐 간 정렬 보장.
    ptr은 한 번만 전진. momentum·teacher 큐를 이 함수로 함께 넣어야 열이 일치한다."""
    for queue, feats_T in pairs:
        queue[:, ptr:ptr + bs] = feats_T
    return (ptr + bs) % queue_size
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m unittest distillation.test_target_mix -v`
Expected: PASS (9 tests)

- [ ] **Step 5: 커밋**

```bash
git add distillation/target_mix.py distillation/test_target_mix.py
git commit -m "feat(ttm): 페어 enqueue (momentum·teacher 큐 열 정렬)"
```

---

## Task 4: 모델 배선 (blip_pretrain.py)

**Files:**
- Modify: `models/blip_pretrain.py` — `BLIP_Pretrain.__init__`(큐 등록부 근처 line ~266-274, 시그니처), `forward`(시그니처 line 347-349, 타깃 계산 line 414-418, enqueue 호출 line 432), `_dequeue_and_enqueue`(line ~545-560)

**Interfaces:**
- Consumes: `distillation.target_mix.{teacher_soft_in_batch, teacher_soft_queue, mix_target, enqueue_all}`
- Produces: `forward(..., gamma=None)` — ttm_enabled·gamma·teacher feats가 있으면 혼합 타깃 사용. `blip_pretrain(..., ttm_enabled, ttm_variant, ttm_temp, ttm_soft_weight)` 생성자 kwargs.

- [ ] **Step 1: `__init__` 시그니처에 ttm kwargs 추가** — `BLIP_Pretrain.__init__` 파라미터 목록(momentum=0.995 다음)에 추가

```python
                 momentum = 0.995,
                 ttm_enabled = False,
                 ttm_variant = 'in_batch',
                 ttm_temp = 0.05,
                 ttm_soft_weight = 0.4,
```

- [ ] **Step 2: `__init__` 본문에 설정 저장 + 티처 큐 버퍼 등록** — 기존 큐 등록/normalize 직후(line ~271, `self.text_queue = nn.functional.normalize(...)` 다음)에 추가

```python
        #### teacher-target-mixing (exp9) — config-gated, 기본 OFF ####
        self.ttm_enabled = ttm_enabled
        self.ttm_variant = ttm_variant
        self.ttm_temp = ttm_temp
        self.ttm_soft_weight = ttm_soft_weight
        if ttm_enabled and ttm_variant == 'queue':
            # momentum 큐와 동형인 티처 큐 (frozen이라 드리프트 없음)
            self.register_buffer("teacher_image_queue", torch.randn(embed_dim, queue_size))
            self.register_buffer("teacher_text_queue", torch.randn(embed_dim, queue_size))
            self.teacher_image_queue = nn.functional.normalize(self.teacher_image_queue, dim=0)
            self.teacher_text_queue = nn.functional.normalize(self.teacher_text_queue, dim=0)
```

- [ ] **Step 3: `forward` 시그니처에 gamma 추가** — line 347-349

```python
    def forward(self, image, caption, alpha, update_train_state=None,
                teacher_img_feat=None, teacher_text_feat=None, distill_temp=0.05,
                teacher_lm_logits=None, teacher_lm_input_ids=None, lm_distill_temp=2.0,
                gamma=None):
```

- [ ] **Step 4: 타깃 계산에 ttm 분기 추가** — line 414-418(`sim_targets` 생성 ~ `sim_t2i_targets`)을 아래로 교체. **else 분기는 기존 두 줄 그대로**(baseline 불변).

```python
            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)

            if self.ttm_enabled and gamma is not None and teacher_img_feat is not None:
                from distillation.target_mix import (
                    teacher_soft_in_batch, teacher_soft_queue, mix_target)
                n_cols = sim_i2t_m.shape[1]
                mom_i2t = F.softmax(sim_i2t_m, dim=1)
                mom_t2i = F.softmax(sim_t2i_m, dim=1)
                ti = teacher_img_feat.to(image.device).float()
                tt = teacher_text_feat.to(image.device).float()
                if self.ttm_variant == 'queue':
                    t_img_all = torch.cat([ti.t(), self.teacher_image_queue.clone().detach()], dim=1)
                    t_txt_all = torch.cat([tt.t(), self.teacher_text_queue.clone().detach()], dim=1)
                    teacher_i2t = teacher_soft_queue(ti, t_txt_all, self.ttm_temp)
                    teacher_t2i = teacher_soft_queue(tt, t_img_all, self.ttm_temp)
                else:  # in_batch (D)
                    teacher_i2t = teacher_soft_in_batch(ti, tt, self.ttm_temp, n_cols)
                    teacher_t2i = teacher_soft_in_batch(tt, ti, self.ttm_temp, n_cols)
                sim_i2t_targets = mix_target(sim_targets, mom_i2t, teacher_i2t, gamma, self.ttm_soft_weight)
                sim_t2i_targets = mix_target(sim_targets, mom_t2i, teacher_t2i, gamma, self.ttm_soft_weight)
            else:
                sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
```

- [ ] **Step 5: enqueue 호출에 티처 feat 전달** — line 432

```python
        if update_train_state:
            self._dequeue_and_enqueue(image_feat_m, text_feat_m, teacher_img_feat, teacher_text_feat)
```

- [ ] **Step 6: `_dequeue_and_enqueue` 확장** — line ~545-560 전체 교체

```python
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat,
                             teacher_img_feat=None, teacher_text_feat=None):
        from distillation.target_mix import enqueue_all
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        batch_size = image_feats.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        pairs = [(self.image_queue, image_feats.T), (self.text_queue, text_feats.T)]
        if self.ttm_enabled and self.ttm_variant == 'queue' and teacher_img_feat is not None:
            t_img = concat_all_gather(teacher_img_feat.float())
            t_txt = concat_all_gather(teacher_text_feat.float())
            pairs += [(self.teacher_image_queue, t_img.T), (self.teacher_text_queue, t_txt.T)]
        self.queue_ptr[0] = enqueue_all(pairs, ptr, batch_size, self.queue_size)
```

**교체 범위**: line 544 `@torch.no_grad()` 데코레이터 줄부터 메서드 끝(line 560)까지 통째로 위 블록으로 교체(위 블록에 데코레이터 1개 포함 — 중복되지 않음).

- [ ] **Step 7: `blip_pretrain` 팩토리가 kwargs 전달하는지 확인** — line ~523 `def blip_pretrain(**kwargs)`가 `BLIP_Pretrain(**kwargs)`를 호출하면 자동 전달. 별도 수정 불필요. 확인만.

- [ ] **Step 8: import 스모크**

Run: `cd /home/minwoo/Distillation_Project && /home/minwoo/miniconda3/envs/kd_r4/bin/python -c "import models.blip_pretrain; print('import ok')"`
Expected: `import ok` (문법 오류 없음)

- [ ] **Step 9: 커밋**

```bash
git add models/blip_pretrain.py
git commit -m "feat(ttm): blip_pretrain forward 타깃 분기 + 티처 큐 페어 enqueue"
```

---

## Task 5: 학습 루프 배선 (pretrain.py)

**Files:**
- Modify: `pretrain.py` — config 파싱(train() line ~83-88), teacher-feat 계산 조건(line 120-121), γ 계산(line 133 근처), forward 호출(line 138-155, 두 곳: autocast/else), TB(line ~190), online_teacher 빌드 조건(line ~379), 모델 생성(main()의 `blip_pretrain(...)` 호출 line ~330)

**Interfaces:**
- Consumes: `distillation.target_mix.ttm_gamma`, `models.blip_pretrain.blip_pretrain(..., ttm_*)`, `forward(..., gamma=)`
- Produces: (없음 — 최상위 학습 스크립트)

- [ ] **Step 1: train()에서 ttm config 파싱 + 상호배타 assert** — line ~88(itc/lm 파싱 다음)에 추가

```python
    distill_ttm = config.get('distill', {}).get('itc_target_mix', {})
    ttm_enabled = distill_ttm.get('enabled', False)
    assert not (itc_kd_enabled and ttm_enabled), \
        "distill.itc(별도 KD)와 distill.itc_target_mix 동시 사용 금지"
    ttm_soft_weight = float(distill_ttm.get('soft_weight', 0.4))
    ttm_hold_steps = int(distill_ttm.get('hold_epochs', 2) * len(data_loader))
    ttm_decay_end_steps = int(distill_ttm.get('decay_end_epochs', 12) * len(data_loader))
```

- [ ] **Step 2: teacher-feat 계산 조건에 ttm 포함** — line 120

```python
        # online teacher: same augmented batch -> ITC features. itc KD 또는 target_mix면 계산.
        if (itc_kd_enabled or ttm_enabled) and online_teacher is not None:
            teacher_img_feat, teacher_text_feat = online_teacher.itc_feats(image, caption)
        else:
            teacher_img_feat = teacher_text_feat = None
```

- [ ] **Step 3: γ 매 스텝 계산** — line 133(alpha 계산) 다음에 추가. `global_step`은 이미 line 111에 `epoch*len(data_loader)+i`로 정의돼 있으니 그대로 사용.

```python
        from distillation.target_mix import ttm_gamma
        gamma = ttm_gamma(global_step, ttm_hold_steps, ttm_decay_end_steps) if ttm_enabled else None
```

- [ ] **Step 4: 두 forward 호출에 gamma 전달** — line 138-143(autocast 분기)과 line 150-155(else 분기) 각각의 `model(...)` 호출에 `gamma=gamma` 추가. autocast 분기 예:

```python
                loss_ita, loss_itm, loss_lm, loss_itc_kd, loss_lm_kd = model(
                    image, caption, alpha=alpha,
                    teacher_img_feat=teacher_img_feat, teacher_text_feat=teacher_text_feat,
                    distill_temp=itc_kd_temp,
                    teacher_lm_logits=teacher_lm_logits, teacher_lm_input_ids=teacher_lm_ids,
                    lm_distill_temp=lm_kd_temp,
                    gamma=gamma)
```

else(non-cuda) 분기의 `model(...)`에도 동일하게 `gamma=gamma` 추가.

- [ ] **Step 5: TB에 train/beta·train/gamma 기록** — line ~190(`writer.add_scalar("train/alpha", ...)`) 다음에 추가

```python
                if ttm_enabled:
                    writer.add_scalar("train/gamma", gamma, global_step)
                    writer.add_scalar("train/beta", ttm_soft_weight * gamma, global_step)
```

(주의: 이 블록은 기존 alpha 기록과 같은 `if utils.is_main_process()` + TB interval 조건 안에 있어야 함 — alpha 기록 라인과 같은 들여쓰기/블록에 넣을 것.)

- [ ] **Step 6: online_teacher 빌드 조건에 ttm 포함** — main()의 `teacher_keep` 계산부(line ~379)

```python
    distill_cfg = config.get('distill', {})
    ttm_on = distill_cfg.get('itc_target_mix', {}).get('enabled', False)
    teacher_keep = tuple(k for k in ('itc', 'lm')
                         if distill_cfg.get(k, {}).get('enabled', False)
                         or (k == 'itc' and ttm_on))
```

- [ ] **Step 7: 모델 생성에 ttm kwargs 전달** — main()의 `model = blip_pretrain(...)` 호출(line ~330)에 추가

```python
    ttm_cfg = config.get('distill', {}).get('itc_target_mix', {})
    model = blip_pretrain(image_size=config['image_size'],
                          vit=config['vit'],
                          vit_grad_ckpt=config['vit_grad_ckpt'],
                          vit_ckpt_layer=config['vit_ckpt_layer'],
                          queue_size=config['queue_size'],
                          my_bert_size=config['my_bert_size'],
                          ttm_enabled=ttm_cfg.get('enabled', False),
                          ttm_variant=ttm_cfg.get('variant', 'in_batch'),
                          ttm_temp=float(ttm_cfg.get('temp', 0.05)),
                          ttm_soft_weight=float(ttm_cfg.get('soft_weight', 0.4)))
```

(기존 `blip_pretrain(...)` 호출의 인자 목록을 위와 같이 확장. 기존 인자 순서/이름은 현행 유지하고 ttm_* 4개만 추가.)

- [ ] **Step 8: import 스모크**

Run: `cd /home/minwoo/Distillation_Project && /home/minwoo/miniconda3/envs/kd_r4/bin/python -c "import pretrain; print('import ok')"`
Expected: `import ok`

- [ ] **Step 9: 커밋**

```bash
git add pretrain.py
git commit -m "feat(ttm): pretrain 배선 — γ 스케줄, forward gamma, TB beta/gamma, teacher 빌드조건"
```

---

## Task 6: 두 variant config

**Files:**
- Create: `configs/pretrain_ttm_inbatch.yaml`, `configs/pretrain_ttm_queue.yaml`

**Interfaces:**
- Consumes: `pretrain.py`가 읽는 config 키. baseline은 `configs/pretrain_itc_distill_armT.yaml`(caption-val + student recipe 포함) 참고.

- [ ] **Step 1: in_batch config 작성** — `configs/pretrain_itc_distill_armT.yaml`를 복사해 아래만 변경

```yaml
output_dir: '/home/minwoo/Distillation_Project/output/pt_smallreg_minilm_ttm_inbatch'
exp: '9.1.ttm_in_batch'
caption_score_cpu_list: '0-31'          # 잡A
distill:
  itc:
    enabled: false                       # 별도 KD OFF (target_mix와 상호배타)
    weight: 1.0
    temp: 0.05
  itc_target_mix:
    enabled: true
    variant: in_batch
    temp: 0.05
    soft_weight: 0.4
    hold_epochs: 2
    decay_end_epochs: 12
  lm:
    enabled: false
    weight: 1.0
    temp: 2.0
teacher:
  arch: 'blip_large'
  checkpoint: '/home/minwoo/Distillation_Project/output/official_pretrain_checkpoint/model_large.pth'
```

(나머지 키 — train_file/roots/student arch small_reg·minilm/val_loss/val_retrieval/val_caption/batch40/lr/warmup — 는 armT.yaml에서 그대로 유지.)

- [ ] **Step 2: queue config 작성** — Step 1 파일을 복사해 아래만 변경

```yaml
output_dir: '/home/minwoo/Distillation_Project/output/pt_smallreg_minilm_ttm_queue'
exp: '9.2.ttm_queue'
caption_score_cpu_list: '32-63'         # 잡B
distill:
  itc_target_mix:
    variant: queue                       # ← in_batch에서 queue로만 변경, 나머지 동일
```

- [ ] **Step 3: config 파싱 검증**

Run:
```bash
cd /home/minwoo/Distillation_Project && /home/minwoo/miniconda3/envs/kd_r4/bin/python -c "
import yaml
for f in ('inbatch','queue'):
    c=yaml.safe_load(open(f'configs/pretrain_ttm_{f}.yaml'))
    t=c['distill']['itc_target_mix']
    assert t['enabled'] and not c['distill']['itc']['enabled']
    print(f, t['variant'], 'soft', t['soft_weight'], 'hold', t['hold_epochs'], 'end', t['decay_end_epochs'], c['exp'])
"
```
Expected: `inbatch in_batch soft 0.4 hold 2 end 12 9.1.ttm_in_batch` / `queue queue soft 0.4 hold 2 end 12 9.2.ttm_queue`

- [ ] **Step 4: 커밋**

```bash
git add configs/pretrain_ttm_inbatch.yaml configs/pretrain_ttm_queue.yaml
git commit -m "feat(ttm): variant in_batch/queue config 2개 (exp 9.1/9.2)"
```

---

## Task 7: 통합 스모크 (GPU, 수 스텝)

**Files:** (없음 — 실행 검증. `max_epoch`·경로를 건드리지 않고 짧게 돌려 크래시·회귀만 확인)

**Interfaces:**
- Consumes: Task 4·5·6 산출물 전부.

- [ ] **Step 1: OFF-path 회귀 — baseline(exp7) 몇 스텝이 여전히 도는지**

Run (1 GPU, ~30 스텝 후 Ctrl-C):
```bash
cd /home/minwoo/Distillation_Project && CUDA_VISIBLE_DEVICES=0 timeout 300 /home/minwoo/miniconda3/envs/kd_r4/bin/python -m torch.distributed.run --nproc_per_node=1 --master_port=29510 pretrain.py --config configs/pretrain_student.yaml 2>&1 | grep -E "loss_ita|Start training|Error|Traceback" | head
```
Expected: `Start training` + `loss_ita` 값들이 찍히고 크래시 없음(ttm 미설정 config라 OFF 경로 = baseline).

- [ ] **Step 2: variant D(in_batch) 스모크**

Run:
```bash
cd /home/minwoo/Distillation_Project && CUDA_VISIBLE_DEVICES=0 timeout 400 /home/minwoo/miniconda3/envs/kd_r4/bin/python -m torch.distributed.run --nproc_per_node=1 --master_port=29511 pretrain.py --config configs/pretrain_ttm_inbatch.yaml 2>&1 | grep -E "loss_ita|distill.*teacher|Error|Traceback|assert" | head
```
Expected: 티처 로드 후 `loss_ita`가 스텝따라 감소, 크래시·assert 없음.

- [ ] **Step 3: variant C(queue) 스모크 — 페어 enqueue 경로**

Run:
```bash
cd /home/minwoo/Distillation_Project && CUDA_VISIBLE_DEVICES=0 timeout 400 /home/minwoo/miniconda3/envs/kd_r4/bin/python -m torch.distributed.run --nproc_per_node=1 --master_port=29512 pretrain.py --config configs/pretrain_ttm_queue.yaml 2>&1 | grep -E "loss_ita|Error|Traceback|assert|size mismatch" | head
```
Expected: 크래시·shape mismatch 없이 `loss_ita` 감소. (teacher_image_queue 버퍼 등록 + 페어 enqueue 정상.)

- [ ] **Step 4: TB에 train/beta·train/gamma 기록 확인**

Run (Step 2/3 중 하나가 몇 스텝 쓴 뒤):
```bash
cd /home/minwoo/Distillation_Project && /home/minwoo/miniconda3/envs/kd_r4/bin/python -c "
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
d=sorted(glob.glob('output/pt_smallreg_minilm_ttm_inbatch/tensorboard/*'))[-1]
ea=EventAccumulator(d); ea.Reload()
tags=ea.Tags()['scalars']
assert 'train/beta' in tags and 'train/gamma' in tags, tags
print('train/beta','train/gamma OK; gamma[0]=', ea.Scalars('train/gamma')[0].value)
"
```
Expected: `train/beta train/gamma OK; gamma[0]= 1.0` (홀드 구간 초기 γ=1).

- [ ] **Step 5: 스모크 산출물 정리 + 커밋(없으면 생략)**

```bash
rm -rf output/pt_smallreg_minilm_ttm_inbatch output/pt_smallreg_minilm_ttm_queue   # 스모크 출력 제거(본 런은 나중에)
# 코드 변경이 없으면 커밋 불필요. 스모크는 검증 단계.
```

---

## 실행 후 (본 런)

스페어 서버에서 (2× 느림, ~6일/런). armT/armS(Phase 1)와 동일한 MPS·GPU 배치 패턴:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29503 pretrain.py --config configs/pretrain_ttm_inbatch.yaml
CUDA_VISIBLE_DEVICES=3,0,1,2 torchrun --nproc_per_node=4 --master_port=29504 pretrain.py --config configs/pretrain_ttm_queue.yaml
```
성공 게이트·진단은 스펙 §1·§7 참조.
