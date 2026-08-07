# ITM 학습 배치 로짓 원자료 프로브 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** pretrain 학습 배치(3B)를 그대로 재현해 학생·티처의 ITM 로짓 원자료를 블록별 표 6개로 덤프한다.

**Architecture:** 순수 함수(`probe_core.py`)와 모델 I/O(`itm_batch_logit_probe.py`)를 분리한다. 순수 함수는 stub 모델과 합성 배열로 전량 단위 테스트하고, 모델 경로는 인덱스를 식별자로 실어 나르는 stub으로 3B 블록 조립을 검증한다. 실행은 학생 → 캐시 → 티처 순차 로딩으로 메모리 피크를 낮춘다.

**Tech Stack:** PyTorch (conda env `kd_r4`), numpy, unittest, csv

**Spec:** `docs/superpowers/specs/2026-07-31-itm-batch-logit-probe-design.md`

## Global Constraints

- Python 인터프리터는 `/home/minwoo/miniconda3/envs/kd_r4/bin/python`. 모든 명령은 **repo root** (`/home/minwoo/Distillation_Project`)에서 실행한다.
- **학습 코드는 읽기 전용이다.** `models/blip_pretrain.py`, `pretrain.py`, `data/` 를 수정하지 않는다.
- `itm_head`는 **반드시 autocast 밖 fp32**로 적용한다. `gap`이 두 로짓의 차라 bf16이면 유효숫자가 깎인다.
- `torch.set_grad_enabled(False)` 전역. momentum 갱신·queue 갱신 금지.
- 기본 device는 **CPU** (GPU 4장이 학습 잡으로 점유 중). `--device cuda`로 전환 가능하되 bf16 autocast는 GPU 경로에서만 켠다.
- 표본은 **블록당 500행**, `batch_size=40` → 13배치 수집 후 앞 500행.
- 열 이름은 `z1_nomatch`(class 0) / `z2_match`(class 1) / `m` / `gap`으로 고정한다. `gap = z2_match - z1_nomatch`, `m = (z1_nomatch + z2_match) / 2`.
- 분산·표준편차는 **ddof=1**.
- 선행 ITM 조사(`2026-07-24`, `2026-07-27`, `2026-07-30`)의 수치·결론을 근거나 기대값으로 쓰지 않는다.

## File Structure

| 파일 | 책임 |
|---|---|
| `critical_bugfix/2026-07-31_itm_batch_logit_raw/probe_core.py` | 순수 함수: negative 가중치, 블록 분해, 프레임/통계, CSV, 요약 지표. torch·numpy만 의존하고 모델·데이터셋을 모른다 |
| `critical_bugfix/2026-07-31_itm_batch_logit_raw/itm_batch_logit_probe.py` | 모델 로드, ITM forward 조립, CLI, 순차 로딩 오케스트레이션 |
| `tests/test_itm_batch_logit_probe.py` | 위 두 모듈의 단위 테스트 (stub 모델 사용) |

`critical_bugfix/` 하위는 패키지가 아니므로 테스트에서 `sys.path.insert`로 임포트한다 — 기존 `critical_bugfix/2026-07-30_.../itm_logit_dump.py`가 쓰는 관행과 동일하다.

---

### Task 1: negative 샘플링 가중치 + 블록 분해

**Files:**
- Create: `critical_bugfix/2026-07-31_itm_batch_logit_raw/probe_core.py`
- Test: `tests/test_itm_batch_logit_probe.py`

**Interfaces:**
- Produces:
  - `neg_weights(sim: Tensor[B,B]) -> Tensor[B,B]` — `softmax(sim,dim=1) + 1e-4`, 대각 0
  - `sample_neg_idx(weights: Tensor[B,B]) -> LongTensor[B]` — 행마다 multinomial 1개
  - `split_blocks(logits: Tensor[3B,2]) -> dict[str, Tensor[B,2]]` — 키 `"pos"`, `"neg_img"`, `"neg_txt"`

- [ ] **Step 1: Write the failing test**

`tests/test_itm_batch_logit_probe.py` 신규 생성:

```python
import os
import sys
import unittest

import numpy as np
import torch

REPO = "/home/minwoo/Distillation_Project"
sys.path.insert(0, os.path.join(REPO, "critical_bugfix/2026-07-31_itm_batch_logit_raw"))

import probe_core as pc


class NegWeightsTest(unittest.TestCase):
    def test_diagonal_is_zeroed(self):
        sim = torch.randn(5, 5)
        w = pc.neg_weights(sim)
        self.assertTrue(torch.all(torch.diagonal(w) == 0))

    def test_offdiagonal_is_strictly_positive(self):
        # softmax + 1e-4 이므로 비대각은 절대 0이 되지 않는다 (multinomial 이 항상 성공)
        sim = torch.full((4, 4), -50.0)
        w = pc.neg_weights(sim)
        off = w[~torch.eye(4, dtype=torch.bool)]
        self.assertTrue(torch.all(off > 0))

    def test_does_not_mutate_input(self):
        sim = torch.randn(3, 3)
        before = sim.clone()
        pc.neg_weights(sim)
        self.assertTrue(torch.equal(sim, before))


class SampleNegIdxTest(unittest.TestCase):
    def test_never_samples_self(self):
        torch.manual_seed(0)
        sim = torch.randn(6, 6)
        w = pc.neg_weights(sim)
        for _ in range(20):
            idx = pc.sample_neg_idx(w)
            self.assertEqual(idx.shape, (6,))
            self.assertTrue(torch.all(idx != torch.arange(6)))

    def test_seed_reproducible(self):
        sim = torch.randn(8, 8)
        w = pc.neg_weights(sim)
        torch.manual_seed(123)
        a = pc.sample_neg_idx(w)
        torch.manual_seed(123)
        b = pc.sample_neg_idx(w)
        self.assertTrue(torch.equal(a, b))


class SplitBlocksTest(unittest.TestCase):
    def test_splits_into_three_equal_blocks_in_order(self):
        logits = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(12, 2)
        blocks = pc.split_blocks(logits)
        self.assertEqual(set(blocks), {"pos", "neg_img", "neg_txt"})
        self.assertTrue(torch.equal(blocks["pos"], logits[0:4]))
        self.assertTrue(torch.equal(blocks["neg_img"], logits[4:8]))
        self.assertTrue(torch.equal(blocks["neg_txt"], logits[8:12]))

    def test_rejects_non_multiple_of_three(self):
        with self.assertRaises(ValueError):
            pc.split_blocks(torch.zeros(11, 2))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/minwoo/Distillation_Project && /home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_itm_batch_logit_probe.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'probe_core'`

- [ ] **Step 3: Write minimal implementation**

`critical_bugfix/2026-07-31_itm_batch_logit_raw/probe_core.py` 신규 생성:

```python
"""ITM 배치 로짓 프로브의 순수 함수 — 모델·데이터셋을 모른다.

블록 규약 (models/blip_pretrain.py:471-489 와 동일):
    [0  : B ]  pos      (img_i,     txt_i)
    [B  : 2B]  neg_img  (neg_img_i, txt_i)      ← 텍스트가 원본
    [2B : 3B]  neg_txt  (img_i,     neg_txt_i)  ← 이미지가 원본
"""
import csv

import numpy as np
import torch
import torch.nn.functional as F

BLOCK_NAMES = ("pos", "neg_img", "neg_txt")
COLUMNS = ("z1_nomatch", "z2_match", "m", "gap")


def neg_weights(sim):
    """in-batch negative 후보 가중치. blip_pretrain.py:448-451 과 동일.

    sim: [B, B] (이미 logit_scale 이 곱해진 값). 입력을 변형하지 않는다.
    """
    w = F.softmax(sim, dim=1) + 1e-4
    w = w.clone()
    w.fill_diagonal_(0)
    return w


def sample_neg_idx(weights):
    """행마다 multinomial 1개. 전역 RNG 를 쓰므로 torch.manual_seed 로 재현된다."""
    bs = weights.size(0)
    return torch.tensor(
        [int(torch.multinomial(weights[b], 1).item()) for b in range(bs)],
        dtype=torch.long,
    )


def split_blocks(logits):
    """[3B, 2] → {"pos": [B,2], "neg_img": [B,2], "neg_txt": [B,2]}"""
    n = logits.size(0)
    if n % 3 != 0:
        raise ValueError(f"logits rows must be a multiple of 3, got {n}")
    b = n // 3
    return {
        "pos": logits[0:b],
        "neg_img": logits[b : 2 * b],
        "neg_txt": logits[2 * b : 3 * b],
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/minwoo/Distillation_Project && /home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_itm_batch_logit_probe.py -v
```

Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

커밋은 플랜 전체 완료 후 한 번에 한다 (사용자 지시). 이 단계에서는 커밋하지 않고 다음 태스크로 넘어간다.

---

### Task 2: 프레임 조립 · 열 통계 · CSV 출력

**Files:**
- Modify: `critical_bugfix/2026-07-31_itm_batch_logit_raw/probe_core.py`
- Test: `tests/test_itm_batch_logit_probe.py`

**Interfaces:**
- Consumes: `COLUMNS`, `BLOCK_NAMES` (Task 1)
- Produces:
  - `make_frame(batch_ids: list[int], row_ids: list[int], logits: np.ndarray[N,2]) -> dict[str, np.ndarray]` — 키 `"batch"`, `"i"`, 그리고 `COLUMNS` 4개
  - `column_stats(frame: dict) -> dict[str, dict[str, float]]` — `{col: {"mean":…, "var":…, "std":…}}`, ddof=1
  - `write_block_csv(path: str, frame: dict) -> None` — 헤더 + N행 + `mean`/`var`/`std` 3행

- [ ] **Step 1: Write the failing test**

`tests/test_itm_batch_logit_probe.py` 에 추가:

```python
import csv as _csv
import tempfile


class MakeFrameTest(unittest.TestCase):
    def test_derives_m_and_gap(self):
        logits = np.array([[1.0, 5.0], [2.0, -2.0]], dtype=np.float32)
        fr = pc.make_frame([0, 0], [0, 1], logits)
        np.testing.assert_allclose(fr["z1_nomatch"], [1.0, 2.0])
        np.testing.assert_allclose(fr["z2_match"], [5.0, -2.0])
        np.testing.assert_allclose(fr["gap"], [4.0, -4.0])   # z2 - z1
        np.testing.assert_allclose(fr["m"], [3.0, 0.0])      # (z1 + z2) / 2

    def test_keeps_index_columns(self):
        logits = np.zeros((3, 2), dtype=np.float32)
        fr = pc.make_frame([7, 7, 8], [0, 1, 0], logits)
        np.testing.assert_array_equal(fr["batch"], [7, 7, 8])
        np.testing.assert_array_equal(fr["i"], [0, 1, 0])


class ColumnStatsTest(unittest.TestCase):
    def test_uses_sample_variance_ddof1(self):
        logits = np.array([[0.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        fr = pc.make_frame([0, 0], [0, 1], logits)
        st = pc.column_stats(fr)
        # gap = [0, 2] → mean 1, ddof=1 분산 = 2.0
        self.assertAlmostEqual(st["gap"]["mean"], 1.0)
        self.assertAlmostEqual(st["gap"]["var"], 2.0)
        self.assertAlmostEqual(st["gap"]["std"], np.sqrt(2.0), places=6)

    def test_covers_all_four_columns(self):
        fr = pc.make_frame([0], [0], np.zeros((1, 2), dtype=np.float32))
        self.assertEqual(set(pc.column_stats(fr)), set(pc.COLUMNS))


class WriteBlockCsvTest(unittest.TestCase):
    def test_writes_rows_then_three_summary_rows(self):
        logits = np.array([[1.0, 3.0], [2.0, 6.0]], dtype=np.float32)
        fr = pc.make_frame([0, 0], [0, 1], logits)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "b.csv")
            pc.write_block_csv(p, fr)
            with open(p) as f:
                rows = list(_csv.reader(f))
        self.assertEqual(rows[0], ["batch", "i", "z1_nomatch", "z2_match", "m", "gap"])
        self.assertEqual(len(rows), 1 + 2 + 3)          # header + data + mean/var/std
        self.assertEqual(rows[-3][0], "mean")
        self.assertEqual(rows[-2][0], "var")
        self.assertEqual(rows[-1][0], "std")
        self.assertAlmostEqual(float(rows[-3][5]), 3.0)  # gap mean = (2+4)/2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/minwoo/Distillation_Project && /home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_itm_batch_logit_probe.py -v -k "Frame or Stats or Csv"
```

Expected: FAIL — `AttributeError: module 'probe_core' has no attribute 'make_frame'`

- [ ] **Step 3: Write minimal implementation**

`probe_core.py` 에 추가:

```python
def make_frame(batch_ids, row_ids, logits):
    """logits [N,2] → 열 dict. z1=class 0, z2=class 1."""
    arr = np.asarray(logits, dtype=np.float64)
    z1 = arr[:, 0]
    z2 = arr[:, 1]
    return {
        "batch": np.asarray(batch_ids, dtype=np.int64),
        "i": np.asarray(row_ids, dtype=np.int64),
        "z1_nomatch": z1,
        "z2_match": z2,
        "m": 0.5 * (z1 + z2),
        "gap": z2 - z1,
    }


def column_stats(frame):
    out = {}
    for c in COLUMNS:
        a = np.asarray(frame[c], dtype=np.float64)
        out[c] = {
            "mean": float(a.mean()),
            "var": float(a.var(ddof=1)) if a.size > 1 else float("nan"),
            "std": float(a.std(ddof=1)) if a.size > 1 else float("nan"),
        }
    return out


def write_block_csv(path, frame):
    st = column_stats(frame)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch", "i", *COLUMNS])
        for r in range(len(frame["batch"])):
            w.writerow(
                [int(frame["batch"][r]), int(frame["i"][r])]
                + [f"{float(frame[c][r]):.6f}" for c in COLUMNS]
            )
        for key in ("mean", "var", "std"):
            w.writerow([key, ""] + [f"{st[c][key]:.6f}" for c in COLUMNS])
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/minwoo/Distillation_Project && /home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_itm_batch_logit_probe.py -v
```

Expected: PASS (13 tests)

- [ ] **Step 5: 커밋하지 않고 다음 태스크로**

---

### Task 3: 요약 지표 (Δ · P_pos · σ(m) · Var(m)/Var(z2))

**Files:**
- Modify: `critical_bugfix/2026-07-31_itm_batch_logit_raw/probe_core.py`
- Test: `tests/test_itm_batch_logit_probe.py`

**Interfaces:**
- Consumes: `make_frame`, `column_stats` (Task 2)
- Produces: `summarize_blocks(frames: dict[str, dict]) -> dict` — 최상위 키 `"delta"`, `"p_pos_merged_measured"`, `"p_pos_merged_from_delta"`, `"n_per_block"`, 그리고 `BLOCK_NAMES` 각각

**스펙 §8.1 근거:** 통합 softmax에서 pos 블록이 먹는 질량은 근사식 $e^\Delta/(e^\Delta+2)$ 로도 계산하지만, 평균의 exp ≠ exp의 평균이므로 **실측값**(`p_pos_merged_measured`)을 함께 낸다. 판정은 실측값으로 한다.

**무차원 비율 지표는 내지 않는다** (스펙 §3.1). 블록별로는 `sigma_m`·`sigma_gap`(로짓 단위 절대값)과 `stats`만 낸다. `Var(m)/Var(z2)` 같은 비율은 gap 산포가 큰 모델을 자동으로 유리해 보이게 만들고, 계산해 두면 원자료 대신 그 숫자로 판정하게 되므로 제외한다.

**두 지수식 모두 오버플로 안전해야 한다.** `p_pos_merged_measured`는 max 감산으로, `p_pos_merged_from_delta`는 $\Delta \ge 0$ 일 때 $1/(1+2e^{-\Delta})$ 로 갈아타서 처리한다. 테스트가 $\Delta = 900$ 을 넣으므로 순진한 `np.exp(delta)` 는 `RuntimeWarning` 을 낸다.

- [ ] **Step 1: Write the failing test**

```python
class SummarizeBlocksTest(unittest.TestCase):
    def _frames(self, pos_gap, negimg_gap, negtxt_gap):
        """gap 만 지정하고 m=0 이 되도록 z1=-gap/2, z2=+gap/2 로 만든다."""
        fr = {}
        for name, gaps in (("pos", pos_gap), ("neg_img", negimg_gap), ("neg_txt", negtxt_gap)):
            g = np.asarray(gaps, dtype=np.float64)
            logits = np.stack([-g / 2, g / 2], axis=1)
            fr[name] = pc.make_frame([0] * len(g), list(range(len(g))), logits)
        return fr

    def test_delta_is_pos_mean_minus_pooled_neg_mean(self):
        frames = self._frames([4.0, 6.0], [1.0, 1.0], [3.0, 3.0])
        s = pc.summarize_blocks(frames)
        # pos mean = 5.0, pooled neg mean = (1+1+3+3)/4 = 2.0
        self.assertAlmostEqual(s["delta"], 3.0)

    def test_p_pos_from_delta_matches_closed_form(self):
        frames = self._frames([2.0, 2.0], [0.0, 0.0], [0.0, 0.0])
        s = pc.summarize_blocks(frames)
        expected = np.exp(2.0) / (np.exp(2.0) + 2)
        self.assertAlmostEqual(s["p_pos_merged_from_delta"], expected, places=6)

    def test_p_pos_measured_uses_actual_exponentials(self):
        # 모든 gap 이 동일하면 pos 는 정확히 1/3
        frames = self._frames([1.0, 1.0], [1.0, 1.0], [1.0, 1.0])
        s = pc.summarize_blocks(frames)
        self.assertAlmostEqual(s["p_pos_merged_measured"], 1.0 / 3.0, places=6)

    def test_measured_is_overflow_safe(self):
        frames = self._frames([900.0, 900.0], [0.0, 0.0], [0.0, 0.0])
        s = pc.summarize_blocks(frames)
        self.assertTrue(np.isfinite(s["p_pos_merged_measured"]))
        self.assertAlmostEqual(s["p_pos_merged_measured"], 1.0, places=6)

    def test_delta_pools_negatives_rather_than_averaging_block_means(self):
        # 블록 길이를 다르게 해 pooled 와 per-block-average 가 갈리게 한다.
        #   pooled  = (0+0+0+4)/4 = 1.0  → delta = 5.0 - 1.0 = 4.0   ← 올바른 정의
        #   per-blk = (0.0 + 4.0)/2 = 2.0 → delta = 5.0 - 2.0 = 3.0   ← 잘못된 정의
        frames = self._frames([5.0, 5.0], [0.0, 0.0, 0.0], [4.0])
        s = pc.summarize_blocks(frames)
        self.assertAlmostEqual(s["delta"], 4.0)

    def test_per_block_reports_sigma_only(self):
        frames = self._frames([4.0, 6.0], [1.0, 1.0], [3.0, 3.0])
        s = pc.summarize_blocks(frames)
        for name in pc.BLOCK_NAMES:
            self.assertIn("sigma_m", s[name])
            self.assertIn("sigma_gap", s[name])
            self.assertIn("stats", s[name])
            # 무차원 비율 지표는 의도적으로 내지 않는다 (스펙 §3.1)
            self.assertNotIn("var_m_over_var_z2", s[name])
        # 이 합성 데이터는 m ≡ 0 이므로 sigma_m = 0
        self.assertAlmostEqual(s["pos"]["sigma_m"], 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/minwoo/Distillation_Project && /home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_itm_batch_logit_probe.py -v -k Summarize
```

Expected: FAIL — `AttributeError: module 'probe_core' has no attribute 'summarize_blocks'`

- [ ] **Step 3: Write minimal implementation**

`probe_core.py` 에 추가:

```python
def _softmax_mass(subset, whole):
    """sum(exp(subset)) / sum(exp(whole)) — max 를 빼서 오버플로를 막는다."""
    mx = float(np.max(whole))
    return float(np.exp(subset - mx).sum() / np.exp(whole - mx).sum())


def _p_pos_closed_form(delta):
    """e^d / (e^d + 2) 의 오버플로 안전 형태.

    d >= 0 에서는 1 / (1 + 2e^-d) 로 갈아탄다 (e^d 가 터지는 쪽을 피한다).
    d < 0 에서는 원식 그대로가 안전하다 (e^d -> 0).
    """
    d = float(delta)
    if d >= 0.0:
        return float(1.0 / (1.0 + 2.0 * np.exp(-d)))
    e = np.exp(d)
    return float(e / (e + 2.0))


def summarize_blocks(frames):
    """스펙 §8 의 판정 지표. frames 는 BLOCK_NAMES 를 키로 갖는 make_frame 결과."""
    pos_gap = np.asarray(frames["pos"]["gap"], dtype=np.float64)
    neg_gap = np.concatenate(
        [np.asarray(frames["neg_img"]["gap"], dtype=np.float64),
         np.asarray(frames["neg_txt"]["gap"], dtype=np.float64)]
    )
    all_gap = np.concatenate([pos_gap, neg_gap])
    delta = float(pos_gap.mean() - neg_gap.mean())

    out = {
        "delta": delta,
        "p_pos_merged_from_delta": _p_pos_closed_form(delta),
        "p_pos_merged_measured": _softmax_mass(pos_gap, all_gap),
        "n_per_block": {k: int(len(frames[k]["gap"])) for k in BLOCK_NAMES},
    }
    for name in BLOCK_NAMES:
        st = column_stats(frames[name])
        out[name] = {
            "stats": st,
            "sigma_m": st["m"]["std"],
            "sigma_gap": st["gap"]["std"],
        }
    return out
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/minwoo/Distillation_Project && /home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_itm_batch_logit_probe.py -v
```

Expected: PASS (19 tests)

- [ ] **Step 5: 커밋하지 않고 다음 태스크로**

---

### Task 4: ITM forward 조립 — 3B 블록 배치가 올바른지 stub으로 검증

이 태스크가 프로브 전체에서 가장 틀리기 쉬운 부분이다. 블록 ②는 **원본 텍스트 + negative 이미지**, 블록 ③은 **negative 텍스트 + 원본 이미지**인데 `blip_pretrain.py:471-475`에서 두 `cat` 의 순서가 서로 반대다 (`text_ids_all = [원본, neg]` vs `image_embeds_all = [neg, 원본]`). stub 모델이 텍스트/이미지 식별자를 로짓 두 열에 그대로 실어 나르게 해서 조립 결과를 인덱스 수준으로 검증한다.

**Files:**
- Create: `critical_bugfix/2026-07-31_itm_batch_logit_raw/itm_batch_logit_probe.py`
- Test: `tests/test_itm_batch_logit_probe.py`

**Interfaces:**
- Consumes: `probe_core.neg_weights`, `probe_core.sample_neg_idx` (Task 1)
- Produces:
  - `autocast_ctx(device: torch.device, amp: bool)` — GPU+amp일 때만 bf16 autocast, 아니면 nullcontext
  - `itm_logits_from_negs(model, image_embeds, image_atts, enc_ids, text_atts, neg_idx_img, neg_idx_txt) -> Tensor[3B,2]` — fp32 헤드 적용까지 포함
  - `student_itm_logits(model, image, caption, device, amp) -> (Tensor[3B,2], LongTensor[B], LongTensor[B])` — 반환 순서는 `(logits, neg_idx_img, neg_idx_txt)`
  - `teacher_itm_logits(model, image, caption, neg_idx_img, neg_idx_txt, device, amp) -> Tensor[3B,2]`

- [ ] **Step 1: Write the failing test**

`tests/test_itm_batch_logit_probe.py` 에 추가:

```python
sys.path.insert(0, REPO)
import itm_batch_logit_probe as probe  # noqa: E402  (critical_bugfix 경로에서 임포트)


class _StubTokenizerOut(dict):
    """tokenizer(...) 결과 흉내 — .input_ids / .attention_mask / .to(device)"""
    def __init__(self, input_ids, attention_mask):
        super().__init__()
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def to(self, device):
        return self


class _StubTokenizer:
    enc_token_id = 999

    def __call__(self, caption, **kw):
        # caption[i] == "s<i>" → input_ids[i] = [0, i, 0, 0]. 두 번째 토큰이 샘플 식별자.
        ids = torch.tensor([[0, int(c[1:]), 0, 0] for c in caption], dtype=torch.long)
        return _StubTokenizerOut(ids, torch.ones_like(ids))


class _StubTextEncoder:
    """CLS 자리에 [텍스트 식별자, 이미지 식별자] 를 실어 반환.

    mode="text"(=ITC 경로) 호출이 받은 0번 토큰을 기록한다 — 그 자리에 [ENC] 가 오면
    text_feat 이 학습과 달라져 negative 샘플링 분포가 어긋나므로, 테스트가 이를 감시한다."""
    def __init__(self):
        self.text_mode_first_tokens = []

    def __call__(self, input_ids, attention_mask=None, encoder_hidden_states=None,
                 encoder_attention_mask=None, return_dict=True, mode=None):
        n = input_ids.size(0)
        if mode == "text":
            self.text_mode_first_tokens.append(input_ids[:, 0].clone())
            h = torch.zeros(n, 4, 8)
            h[:, 0, 0] = input_ids[:, 1].float()
            return type("O", (), {"last_hidden_state": h})()
        text_id = input_ids[:, 1].float()
        image_id = encoder_hidden_states[:, 0, 0]
        h = torch.zeros(n, 4, 8)
        h[:, 0, 0] = text_id
        h[:, 0, 1] = image_id
        return type("O", (), {"last_hidden_state": h})()


class _StubModel(torch.nn.Module):
    """visual_encoder 는 이미지 식별자를 채널 0 에 그대로 싣는다.
    itm_head 는 [[1,0,...],[0,1,...]] 이므로 z1 = 텍스트 식별자, z2 = 이미지 식별자."""
    def __init__(self, bs):
        super().__init__()
        self.tokenizer = _StubTokenizer()
        self.text_encoder = _StubTextEncoder()
        self.text_encoder_m = _StubTextEncoder()
        self.itm_head = torch.nn.Linear(8, 2, bias=True)
        with torch.no_grad():
            self.itm_head.weight.zero_()
            self.itm_head.weight[0, 0] = 1.0
            self.itm_head.weight[1, 1] = 1.0
            self.itm_head.bias.zero_()
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.0))
        self.vision_proj = torch.nn.Identity()
        self.vision_proj_m = torch.nn.Identity()
        self.text_proj = torch.nn.Identity()
        self.text_proj_m = torch.nn.Identity()
        self._bs = bs

    def _embed(self, image):
        n = image.size(0)
        h = torch.zeros(n, 5, 8)
        h[:, 0, 0] = torch.arange(n, dtype=torch.float32)  # 이미지 식별자 = 배치 위치
        return h

    def visual_encoder(self, image):
        return self._embed(image)

    def visual_encoder_m(self, image):
        return self._embed(image)


class ItmForwardAssemblyTest(unittest.TestCase):
    def setUp(self):
        torch.set_grad_enabled(False)
        self.bs = 4
        self.model = _StubModel(self.bs)
        self.image = torch.zeros(self.bs, 3, 8, 8)
        self.caption = [f"s{i}" for i in range(self.bs)]
        self.device = torch.device("cpu")

    def test_block_composition_matches_blip_pretrain(self):
        torch.manual_seed(0)
        logits, neg_img, neg_txt = probe.student_itm_logits(
            self.model, self.image, self.caption, self.device, amp=False)
        self.assertEqual(tuple(logits.shape), (3 * self.bs, 2))
        z_text, z_image = logits[:, 0], logits[:, 1]
        ar = torch.arange(self.bs, dtype=torch.float32)

        # ① pos: (img_i, txt_i)
        self.assertTrue(torch.equal(z_text[0:self.bs], ar))
        self.assertTrue(torch.equal(z_image[0:self.bs], ar))
        # ② neg_img: 텍스트는 원본, 이미지는 negative
        self.assertTrue(torch.equal(z_text[self.bs:2 * self.bs], ar))
        self.assertTrue(torch.equal(z_image[self.bs:2 * self.bs], neg_img.float()))
        # ③ neg_txt: 이미지는 원본, 텍스트는 negative
        self.assertTrue(torch.equal(z_text[2 * self.bs:3 * self.bs], neg_txt.float()))
        self.assertTrue(torch.equal(z_image[2 * self.bs:3 * self.bs], ar))

    def test_negatives_are_never_self(self):
        torch.manual_seed(1)
        _, neg_img, neg_txt = probe.student_itm_logits(
            self.model, self.image, self.caption, self.device, amp=False)
        ar = torch.arange(self.bs)
        self.assertTrue(torch.all(neg_img != ar))
        self.assertTrue(torch.all(neg_txt != ar))

    def test_teacher_reuses_given_negatives(self):
        neg_img = torch.tensor([1, 2, 3, 0])
        neg_txt = torch.tensor([3, 0, 1, 2])
        logits = probe.teacher_itm_logits(
            self.model, self.image, self.caption, neg_img, neg_txt, self.device, amp=False)
        z_text, z_image = logits[:, 0], logits[:, 1]
        self.assertTrue(torch.equal(z_image[self.bs:2 * self.bs], neg_img.float()))
        self.assertTrue(torch.equal(z_text[2 * self.bs:3 * self.bs], neg_txt.float()))

    def test_head_is_applied_in_fp32(self):
        torch.manual_seed(0)
        logits, _, _ = probe.student_itm_logits(
            self.model, self.image, self.caption, self.device, amp=False)
        self.assertEqual(logits.dtype, torch.float32)

    def test_itc_forward_uses_raw_cls_not_enc_token(self):
        """ITC(mode='text') 는 원본 [CLS] 시퀀스를 받아야 한다 — blip_pretrain.py:388, 404.
        [ENC] 치환은 ITM 경로(:436-437)에서만 일어난다. 여기에 enc_ids 를 넣으면
        text_feat/text_feat_m 이 통째로 달라져 negative 샘플링이 학습과 어긋난다."""
        torch.manual_seed(0)
        probe.student_itm_logits(self.model, self.image, self.caption, self.device, amp=False)
        for enc in (self.model.text_encoder, self.model.text_encoder_m):
            self.assertTrue(enc.text_mode_first_tokens, "mode='text' forward 가 호출되지 않았다")
            for toks in enc.text_mode_first_tokens:
                self.assertTrue(torch.all(toks != _StubTokenizer.enc_token_id))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/minwoo/Distillation_Project && /home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_itm_batch_logit_probe.py -v -k Assembly
```

Expected: FAIL — `ModuleNotFoundError: No module named 'itm_batch_logit_probe'`

- [ ] **Step 3: Write minimal implementation**

`critical_bugfix/2026-07-31_itm_batch_logit_raw/itm_batch_logit_probe.py` 신규 생성 (이 태스크에서는 forward 함수까지만; CLI는 Task 5):

```python
#!/usr/bin/env python
"""ITM 학습 배치 로짓 원자료 덤프 — 블록별 z1·z2·m·gap 표.

blip_pretrain.forward 의 ITM 경로(models/blip_pretrain.py:435-489)를 그대로 재현한다.
retrieval 후보가 아니라 학습 배치 3B(서로 다른 쌍)를 재는 것이 요점이다.

재현에서 놓치기 쉬운 세 가지:
  1) negative 샘플링은 momentum 피처를 쓴다. image_feat_all = cat([image_feat_m.t(), queue])
     이고 sim[:, :bs] 가 그 앞부분만 자르므로, 가중치는 text_feat @ image_feat_m.t() 에서 나온다.
     → 학생은 momentum 인코더 4종을 로드해야 한다(갱신은 하지 않는다).
  2) logit_scale 이 샘플링 분포를 바꾼다. safe_scale = clamp(logit_scale).exp() 를 그대로 쓴다.
  3) itm_head 는 autocast 밖 fp32. gap 이 두 로짓의 차라 bf16 이면 유효숫자가 깎인다.
"""
import contextlib
import os
import sys

import torch
import torch.nn.functional as F

REPO = "/home/minwoo/Distillation_Project"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.blip_pretrain import LOGIT_SCALE_MAX, LOGIT_SCALE_MIN  # noqa: E402

import probe_core as pc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_TEXT_LEN = 30


def autocast_ctx(device, amp):
    """학습(pretrain.py:140)과 동일한 bf16 autocast. CPU 경로에서는 fp32."""
    if amp and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _tokenize(model, caption, device):
    """(raw_ids, enc_ids, attention_mask) 반환.

    raw_ids : 원본 [CLS] 시작. **ITC 경로(mode='text')가 쓴다** — blip_pretrain.py:388, 404
    enc_ids : 0번을 [ENC] 로 치환. **ITM 경로만 쓴다** — blip_pretrain.py:436-437

    [ENC] 는 additional_special_tokens 로 추가된 별도 토큰이라 [CLS] 와 임베딩이 완전히 다르다.
    text_feat = normalize(text_proj(last_hidden_state[:, 0, :])) 가 바로 그 0번 위치라서,
    ITC forward 에 enc_ids 를 넣으면 text_feat/text_feat_m 이 통째로 달라지고
    → sim → negative 샘플링 분포가 학습과 어긋난다. 프로브의 존재 이유가 깨지는 지점이다.
    """
    text = model.tokenizer(caption, padding="max_length", truncation=True,
                           max_length=MAX_TEXT_LEN, return_tensors="pt").to(device)
    raw_ids = text.input_ids
    enc_ids = raw_ids.clone()
    enc_ids[:, 0] = model.tokenizer.enc_token_id
    return raw_ids, enc_ids, text.attention_mask


def itm_logits_from_negs(model, image_embeds, image_atts, enc_ids, text_atts,
                         neg_idx_img, neg_idx_txt):
    """blip_pretrain.py:441-485 와 동일한 3B 조립 + fp32 헤드.

    주의: text_ids_all 은 [원본, neg] 인데 image_embeds_all 은 [neg, 원본] 으로 순서가 반대다.
    그래서 블록 ②(=[B:2B])가 '원본 텍스트 + negative 이미지'가 된다.
    """
    output_pos = model.text_encoder(enc_ids, attention_mask=text_atts,
                                    encoder_hidden_states=image_embeds,
                                    encoder_attention_mask=image_atts, return_dict=True)

    image_embeds_neg = image_embeds[neg_idx_img]
    text_ids_neg = enc_ids[neg_idx_txt]
    text_atts_neg = text_atts[neg_idx_txt]

    text_ids_all = torch.cat([enc_ids, text_ids_neg], dim=0)
    text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0)
    image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
    image_atts_all = torch.cat([image_atts, image_atts], dim=0)

    output_neg = model.text_encoder(text_ids_all, attention_mask=text_atts_all,
                                    encoder_hidden_states=image_embeds_all,
                                    encoder_attention_mask=image_atts_all, return_dict=True)

    vl = torch.cat([output_pos.last_hidden_state[:, 0, :],
                    output_neg.last_hidden_state[:, 0, :]], dim=0)
    # ★ 헤드는 autocast 밖 fp32
    return F.linear(vl.float(), model.itm_head.weight.float(), model.itm_head.bias.float())


def student_itm_logits(model, image, caption, device, amp):
    """학생: negative 를 직접 뽑고 3B 로짓을 낸다. 반환 (logits, neg_idx_img, neg_idx_txt)."""
    raw_ids, enc_ids, text_atts = _tokenize(model, caption, device)
    with autocast_ctx(device, amp):
        image_embeds = model.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
        image_feat = F.normalize(model.vision_proj(image_embeds[:, 0, :]), dim=-1)
        # ★ ITC 경로는 raw_ids([CLS] 시작). enc_ids 를 쓰면 안 된다 — _tokenize docstring 참고
        text_out = model.text_encoder(raw_ids, attention_mask=text_atts,
                                      return_dict=True, mode="text")
        text_feat = F.normalize(model.text_proj(text_out.last_hidden_state[:, 0, :]), dim=-1)

        # negative 샘플링은 momentum 피처를 쓴다 (queue 부분은 [:, :bs] 슬라이싱으로 배제됨)
        image_embeds_m = model.visual_encoder_m(image)
        image_feat_m = F.normalize(model.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
        text_out_m = model.text_encoder_m(raw_ids, attention_mask=text_atts,
                                          return_dict=True, mode="text")
        text_feat_m = F.normalize(model.text_proj_m(text_out_m.last_hidden_state[:, 0, :]), dim=-1)

        safe_scale = model.logit_scale.clamp(LOGIT_SCALE_MIN, LOGIT_SCALE_MAX).exp()
        sim_i2t = (image_feat @ text_feat_m.t()).float() * safe_scale.float()
        sim_t2i = (text_feat @ image_feat_m.t()).float() * safe_scale.float()

        neg_idx_img = pc.sample_neg_idx(pc.neg_weights(sim_t2i)).to(device)
        neg_idx_txt = pc.sample_neg_idx(pc.neg_weights(sim_i2t)).to(device)

        logits = itm_logits_from_negs(model, image_embeds, image_atts, enc_ids, text_atts,
                                      neg_idx_img, neg_idx_txt)
    return logits.float(), neg_idx_img.cpu(), neg_idx_txt.cpu()


def teacher_itm_logits(model, image, caption, neg_idx_img, neg_idx_txt, device, amp):
    """티처: 학생이 뽑은 negative 를 그대로 재사용해 동일한 3B 쌍을 본다.

    ITM forward 만 하므로 raw_ids 는 쓰지 않는다 (ITC 경로가 없다)."""
    _, enc_ids, text_atts = _tokenize(model, caption, device)
    with autocast_ctx(device, amp):
        image_embeds = model.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
        logits = itm_logits_from_negs(model, image_embeds, image_atts, enc_ids, text_atts,
                                      neg_idx_img.to(device), neg_idx_txt.to(device))
    return logits.float()
```

주의: stub 테스트는 `model.tokenizer(...)` 가 `padding`/`truncation`/`max_length`/`return_tensors` 키워드를 받아 무시한다(`**kw`). 실제 모델에서는 `blip_pretrain.forward` 와 동일한 값이 전달된다.

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/minwoo/Distillation_Project && /home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_itm_batch_logit_probe.py -v
```

Expected: PASS (23 tests)

- [ ] **Step 5: 커밋하지 않고 다음 태스크로**

---

### Task 5: CLI 오케스트레이션 — 순차 로딩 · 캐시 · 산출물

**Files:**
- Modify: `critical_bugfix/2026-07-31_itm_batch_logit_raw/itm_batch_logit_probe.py`
- Test: `tests/test_itm_batch_logit_probe.py`

**Interfaces:**
- Consumes: `student_itm_logits`, `teacher_itm_logits` (Task 4), `probe_core.split_blocks`, `make_frame`, `write_block_csv`, `summarize_blocks` (Task 1–3)
- Produces:
  - `collect_frames(logits_list: list[Tensor[3B,2]], batch_ids: list[int], n_samples: int) -> dict[str, dict]` — 블록별로 배치들을 이어붙이고 앞 `n_samples` 행만 남긴다
  - `emit(tag: str, frames: dict, out_dir: str) -> dict` — CSV 3개 + npz 1개를 쓰고 요약 dict 반환
  - `main()` — argparse CLI

- [ ] **Step 1: Write the failing test**

```python
class CollectFramesTest(unittest.TestCase):
    def test_concatenates_per_block_and_truncates(self):
        # 배치 2개 × B=2 → 블록당 4행, n_samples=3 이면 3행
        b = 2
        l0 = torch.arange(3 * b * 2, dtype=torch.float32).reshape(3 * b, 2)
        l1 = l0 + 100.0
        frames = probe.collect_frames([l0, l1], [0, 1], n_samples=3)
        self.assertEqual(set(frames), {"pos", "neg_img", "neg_txt"})
        for name in ("pos", "neg_img", "neg_txt"):
            self.assertEqual(len(frames[name]["gap"]), 3)
        # pos 블록: 배치0의 2행 + 배치1의 첫 행
        np.testing.assert_array_equal(frames["pos"]["batch"], [0, 0, 1])
        np.testing.assert_array_equal(frames["pos"]["i"], [0, 1, 0])
        np.testing.assert_allclose(frames["pos"]["z1_nomatch"], [0.0, 2.0, 100.0])

    def test_keeps_all_rows_when_n_samples_exceeds_available(self):
        b = 2
        l0 = torch.zeros(3 * b, 2)
        frames = probe.collect_frames([l0], [0], n_samples=999)
        self.assertEqual(len(frames["pos"]["gap"]), 2)


class EmitTest(unittest.TestCase):
    def test_writes_three_csvs_and_npz_and_returns_summary(self):
        b = 3
        logits = torch.randn(3 * b, 2)
        frames = probe.collect_frames([logits], [0], n_samples=b)
        with tempfile.TemporaryDirectory() as d:
            summary = probe.emit("student", frames, d)
            for name in ("pos", "neg_img", "neg_txt"):
                self.assertTrue(os.path.exists(os.path.join(d, "csv", f"student_{name}.csv")))
            self.assertTrue(os.path.exists(os.path.join(d, "npz", "student.npz")))
        self.assertIn("delta", summary)
        self.assertIn("pos", summary)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/minwoo/Distillation_Project && /home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_itm_batch_logit_probe.py -v -k "Collect or Emit"
```

Expected: FAIL — `AttributeError: module 'itm_batch_logit_probe' has no attribute 'collect_frames'`

- [ ] **Step 3: Write minimal implementation**

`itm_batch_logit_probe.py` 에 추가 (파일 상단 import 에 `argparse`, `json`, `numpy as np`, `yaml` 추가):

```python
def collect_frames(logits_list, batch_ids, n_samples):
    """배치별 [3B,2] 를 블록별로 이어붙이고 앞 n_samples 행만 남긴다."""
    acc = {name: {"batch": [], "i": [], "logits": []} for name in pc.BLOCK_NAMES}
    for logits, bid in zip(logits_list, batch_ids):
        blocks = pc.split_blocks(logits)
        for name in pc.BLOCK_NAMES:
            blk = blocks[name].detach().cpu().numpy()
            acc[name]["logits"].append(blk)
            acc[name]["batch"].extend([bid] * len(blk))
            acc[name]["i"].extend(range(len(blk)))
    frames = {}
    for name in pc.BLOCK_NAMES:
        arr = np.concatenate(acc[name]["logits"], axis=0)[:n_samples]
        frames[name] = pc.make_frame(acc[name]["batch"][:n_samples],
                                     acc[name]["i"][:n_samples], arr)
    return frames


def emit(tag, frames, out_dir):
    """CSV 3개 + npz 1개를 쓰고 요약을 반환한다."""
    csv_dir = os.path.join(out_dir, "csv")
    npz_dir = os.path.join(out_dir, "npz")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)
    payload = {}
    for name in pc.BLOCK_NAMES:
        pc.write_block_csv(os.path.join(csv_dir, f"{tag}_{name}.csv"), frames[name])
        for col in ("batch", "i", *pc.COLUMNS):
            payload[f"{name}__{col}"] = frames[name][col]
    np.savez_compressed(os.path.join(npz_dir, f"{tag}.npz"), **payload)
    return pc.summarize_blocks(frames)


def _print_summary(tag, summary):
    print(f"\n===== {tag} =====")
    print(f"delta (pos gap mean - pooled neg gap mean) = {summary['delta']:.4f}")
    print(f"P_pos if merged: measured={summary['p_pos_merged_measured']:.4f}  "
          f"from_delta={summary['p_pos_merged_from_delta']:.4f}")
    head = f"{'block':<9}{'n':>5}" + "".join(f"{c:>26}" for c in pc.COLUMNS)
    print(head)
    for name in pc.BLOCK_NAMES:
        st = summary[name]["stats"]
        cells = "".join(f"{st[c]['mean']:>13.4f}{st[c]['var']:>13.4f}" for c in pc.COLUMNS)
        print(f"{name:<9}{summary['n_per_block'][name]:>5}{cells}")
    print("           (각 열은 mean / var 순)")
    for name in pc.BLOCK_NAMES:
        s = summary[name]
        print(f"  {name:<9} sigma_m={s['sigma_m']:.4f}  sigma_gap={s['sigma_gap']:.4f}")


def _build_loader(cfg, num_workers):
    """sampler=None + is_trains=[True] → create_loader 가 shuffle=True, drop_last=True 로 만든다
    (data/__init__.py:102-104). RandomSampler 가 전역 RNG 를 쓰므로, 호출 전에 main 에서
    torch.manual_seed 를 걸어 두면 배치 순서가 재현된다."""
    from data import create_dataset, create_loader
    ds = create_dataset("pretrain", cfg, min_scale=0.2)
    return create_loader([ds], [None], batch_size=[cfg["batch_size"]],
                         num_workers=[num_workers], is_trains=[True], collate_fns=[None])[0]


def _load_student(cfg, ckpt, device):
    import utils
    from models.blip_pretrain import blip_pretrain
    model = blip_pretrain(image_size=cfg["image_size"], vit=cfg["vit"],
                          vit_grad_ckpt=cfg["vit_grad_ckpt"], vit_ckpt_layer=cfg["vit_ckpt_layer"],
                          queue_size=cfg["queue_size"], my_bert_size=cfg["my_bert_size"])
    model = utils.load_model_weights_only(model, ckpt)
    return model.to(device).eval()


def _load_teacher(cfg, ckpt, device):
    import utils
    from models.blip_pretrain import blip_pretrain
    # init_backbone_weights=False: large 의 in21k 초기화 경로가 timm 1.x 에서 깨져 있고,
    # 어차피 체크포인트가 가중치를 전량 덮는다.
    # queue_size 는 반드시 cfg 값(57600)이어야 한다. model_large.pth 의 queue 버퍼가
    # [256, 57600] 이라 크기가 다르면 load_state_dict 가 strict=False 여도 shape mismatch 로 죽는다.
    # (queue 는 buffer 일 뿐이고 ITM 전용 forward 경로는 읽지 않는다.)
    model = blip_pretrain(image_size=cfg["image_size"], vit="large", vit_grad_ckpt=False,
                          vit_ckpt_layer=0, queue_size=cfg["queue_size"], my_bert_size="base",
                          init_backbone_weights=False)
    model = utils.load_model_weights_only(model, ckpt)
    # 티처는 negative 를 뽑지 않으므로 momentum 인코더가 필요 없다 → 메모리 해제
    for attr in ("visual_encoder_m", "text_encoder_m", "vision_proj_m", "text_proj_m"):
        setattr(model, attr, None)
    return model.to(device).eval()


def main():
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO, "output/pt_smallreg_minilm_baseline/config.yaml"))
    ap.add_argument("--student-ckpt", default=os.path.join(REPO, "output/pt_smallreg_minilm_baseline/checkpoint_19.pth"))
    ap.add_argument("--teacher-ckpt", default=os.path.join(REPO, "output/official_pretrain_checkpoint/model_large.pth"))
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--amp", action="store_true", help="GPU 경로에서만 bf16 autocast")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--skip-teacher", action="store_true")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cfg = yaml.safe_load(open(args.config))

    loader = _build_loader(cfg, args.num_workers)
    n_batches = -(-args.n_samples // cfg["batch_size"])   # ceil
    print(f"[plan] batch_size={cfg['batch_size']}  n_batches={n_batches}  "
          f"n_samples={args.n_samples}  device={device}")

    # ---- 1) 학생: forward + 배치 캐시 (티처가 동일한 3B 쌍을 보도록)
    print("[load] student ...", flush=True)
    student = _load_student(cfg, args.student_ckpt, device)
    cache, s_logits, batch_ids = [], [], []
    for bi, (image, caption) in enumerate(loader):
        if bi >= n_batches:
            break
        image = image.to(device)
        logits, neg_img, neg_txt = student_itm_logits(student, image, caption, device, args.amp)
        s_logits.append(logits.cpu())
        batch_ids.append(bi)
        cache.append((image.cpu(), list(caption), neg_img, neg_txt))
        print(f"  [student] batch {bi + 1}/{n_batches}", flush=True)
    del student
    if device.type == "cuda":
        torch.cuda.empty_cache()

    summaries = {}
    frames_s = collect_frames(s_logits, batch_ids, args.n_samples)
    summaries["student"] = emit("student", frames_s, args.out)
    _print_summary("student", summaries["student"])

    # ---- 2) 티처: 캐시된 배치 + 학생이 뽑은 negative 재사용
    if not args.skip_teacher:
        print("[load] teacher ...", flush=True)
        teacher = _load_teacher(cfg, args.teacher_ckpt, device)
        t_logits = []
        for bi, (image, caption, neg_img, neg_txt) in enumerate(cache):
            logits = teacher_itm_logits(teacher, image.to(device), caption,
                                        neg_img, neg_txt, device, args.amp)
            t_logits.append(logits.cpu())
            print(f"  [teacher] batch {bi + 1}/{len(cache)}", flush=True)
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
        frames_t = collect_frames(t_logits, batch_ids, args.n_samples)
        summaries["teacher"] = emit("teacher", frames_t, args.out)
        _print_summary("teacher", summaries["teacher"])

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"\nsaved: {os.path.join(args.out, 'summary.json')}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/minwoo/Distillation_Project && /home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_itm_batch_logit_probe.py -v
```

Expected: PASS (27 tests)

- [ ] **Step 5: CPU 스모크 — 실제 모델 2개로 소량 실행**

```bash
cd /home/minwoo/Distillation_Project && CUDA_VISIBLE_DEVICES="" /home/minwoo/miniconda3/envs/kd_r4/bin/python \
  critical_bugfix/2026-07-31_itm_batch_logit_raw/itm_batch_logit_probe.py \
  --n-samples 8 --device cpu --num-workers 4 \
  --out /tmp/claude-1014/-home-minwoo-Distillation-Project/8157c5cd-7823-4632-96ee-3f1a10f04a97/scratchpad/itm_probe_smoke
```

Expected: 학생·티처 요약 표가 각각 출력되고, `csv/` 6개 · `npz/` 2개 · `summary.json` 생성. 실패하면 아래를 확인:
- `KeyError: 'vit'` → config 키 이름 확인 (`output/pt_smallreg_minilm_baseline/config.yaml`)
- 티처 로드 시 대량의 `missing_keys` → `vit='large'`/`my_bert_size='base'` 조합 확인
- `NaN` 로짓 → `itm_head` fp32 적용 여부 확인

- [ ] **Step 6: 커밋하지 않고 다음 태스크로**

---

### Task 6: 본 실행 (500 샘플) + 결과 기록

**Files:**
- Create: `critical_bugfix/2026-07-31_itm_batch_logit_raw/README.md`
- Output: `critical_bugfix/2026-07-31_itm_batch_logit_raw/{csv,npz}/`, `summary.json`

- [ ] **Step 1: 본 실행**

```bash
cd /home/minwoo/Distillation_Project && CUDA_VISIBLE_DEVICES="" /home/minwoo/miniconda3/envs/kd_r4/bin/python \
  critical_bugfix/2026-07-31_itm_batch_logit_raw/itm_batch_logit_probe.py \
  --n-samples 500 --device cpu --num-workers 16 2>&1 | tee /tmp/claude-1014/-home-minwoo-Distillation-Project/8157c5cd-7823-4632-96ee-3f1a10f04a97/scratchpad/itm_probe_run.log
```

**`CUDA_VISIBLE_DEVICES=""` 는 필수다.** `data/__init__.py:113` 의 `create_loader` 가 `pin_memory=True` 를
하드코딩하는데, GPU 4장이 학습 잡으로 거의 꽉 차 있어(<2GB 여유) `--device cpu` 로 돌려도
pin-memory 스레드의 CUDA 할당이 OOM 난다. 학습 코드는 읽기 전용이므로 호출 측에서 GPU를 숨긴다.

GPU가 비면 `--device cuda --amp` 로 재실행할 수 있지만, 그 전에 파킹된 finding
(`itm_head` 가 autocast 블록 안에서 호출됨 → bf16 gap)을 **반드시 먼저 고쳐야 한다.**
`--amp` 없이 `--device cuda` 만 쓰는 것은 안전하다.

- [ ] **Step 2: 산출물 확인**

```bash
cd /home/minwoo/Distillation_Project && wc -l critical_bugfix/2026-07-31_itm_batch_logit_raw/csv/*.csv && \
  /home/minwoo/miniconda3/envs/kd_r4/bin/python -c "
import json; s=json.load(open('critical_bugfix/2026-07-31_itm_batch_logit_raw/summary.json'))
for k,v in s.items(): print(k, 'delta=%.4f' % v['delta'], 'P_pos=%.4f' % v['p_pos_merged_measured'])"
```

Expected: CSV 6개가 각각 504줄 (헤더 1 + 500 + 요약 3), 학생·티처 delta 출력

- [ ] **Step 3: README.md 작성**

`critical_bugfix/2026-07-31_itm_batch_logit_raw/README.md` 에 다음을 기록한다 (실측값으로 채운다):
- 측정 조건 (체크포인트, 배치 구성, device, seed, 표본 수)
- 표 6개의 mean/var 요약
- **§8.1 판정:** 측정된 Δ와 `p_pos_merged_measured` → 축 B(3B 통합 vs 블록별) 결론
- **§8.2 관찰:** 블록별 σ(m) 학생 vs 티처 대조 → 축 A(행 단위 vs gap 단위) 판단 근거
- **§8.3 관찰:** 블록별 σ(gap) 차이 → 온도 분리 필요성
- 재현 명령

- [ ] **Step 4: 전체 테스트 재실행**

```bash
cd /home/minwoo/Distillation_Project && /home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_itm_batch_logit_probe.py -v
```

Expected: PASS (27 tests)

- [ ] **Step 5: 커밋 (플랜 전체를 한 번에)**

**`critical_bugfix/` 는 `.git/info/exclude:18` 에 등록돼 있어 `git add` 가 조용히 무시한다.**
(기존 `2026-07-24_...` 폴더 4개 파일도 `-f` 로 추가된 것이다 — `git ls-files critical_bugfix/` 로 확인 가능.)
따라서 프로브 디렉토리는 반드시 `-f` 로 추가한다.

```bash
cd /home/minwoo/Distillation_Project && git add \
  docs/superpowers/specs/2026-07-31-itm-batch-logit-probe-design.md \
  docs/superpowers/plans/2026-07-31-itm-batch-logit-probe.md \
  tests/test_itm_batch_logit_probe.py && \
  git add -f critical_bugfix/2026-07-31_itm_batch_logit_raw/ && \
  git status --short && \
  git commit -m "feat(itm-probe): dump ITM train-batch logits per block (student + teacher)"
```

`git status --short` 출력에서 스크립트 2개 · README · CSV 6개 · npz 2개 · summary.json 이 모두
staged 인지 확인한 뒤 커밋한다. 현재 브랜치는 `dev` 이므로 커밋 전 사용자 확인을 받는다.

---

## Self-Review

**Spec coverage**

| 스펙 절 | 태스크 |
|---|---|
| §3 표 6개, 열 4종, 500행 | Task 2 (`make_frame`/`write_block_csv`), Task 5 (`collect_frames`) |
| §3.1 요약 통계 (Δ, P_pos, σ(m), 블록별 σ(gap); 무차원 비율 지표는 제외) | Task 3 (`summarize_blocks`) |
| §4.1 momentum 피처로 negative 샘플링 | Task 4 (`student_itm_logits`) |
| §4.2 logit_scale 적용 | Task 4 (`safe_scale`) |
| §4.3 weights softmax+1e-4, 대각 0, multinomial | Task 1 (`neg_weights`/`sample_neg_idx`) |
| §4.4 itm_head fp32 | Task 4 (`itm_logits_from_negs`), 테스트로 dtype 확인 |
| §4.5 bf16 autocast (GPU만) | Task 4 (`autocast_ctx`) |
| §4.6 no_grad 전역, momentum·queue 갱신 금지 | Task 5 (`main`의 `set_grad_enabled(False)`; `forward`를 호출하지 않으므로 갱신 경로 자체가 없음) |
| §4.1(티처) neg_idx 재사용 | Task 4 (`teacher_itm_logits`), 테스트로 검증 |
| §4.2(순차 로딩) | Task 5 (`main`의 학생→캐시→티처) |
| §5 대상 2개 | Task 5 (`_load_student`/`_load_teacher`) |
| §6 CPU 기본 | Task 5 (`--device` 기본값 `cpu`) |
| §7 산출물 경로 | Task 5 (`emit`), Task 6 |
| §8 판정 기준 | Task 3 (지표 계산), Task 6 Step 3 (README 판정 기록) |

누락 없음.

**Placeholder scan:** "TBD"·"적절히"·"비슷하게" 없음. 모든 코드 단계에 실제 코드가 있다.

**Type consistency:**
- `pc.BLOCK_NAMES` = `("pos","neg_img","neg_txt")` — Task 1 정의, Task 3·5에서 동일 사용
- `pc.COLUMNS` = `("z1_nomatch","z2_match","m","gap")` — Task 1 정의, Task 2·5에서 동일 사용
- `student_itm_logits` 반환 순서 `(logits, neg_idx_img, neg_idx_txt)` — Task 4 정의, Task 5 `main`에서 동일 순서로 언팩
- `neg_idx_img`는 `weights_t2i`에서, `neg_idx_txt`는 `weights_i2t`에서 뽑는다 — `blip_pretrain.py:456`/`:464`와 동일
- `emit(tag, frames, out_dir)` — Task 5 정의, `main`에서 `emit("student", …)`/`emit("teacher", …)`로 호출
