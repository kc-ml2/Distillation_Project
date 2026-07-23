# cc12m 데이터로더 일원화 + 일관성 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** cc12m 사전학습 데이터로더 배선을 단일 팩토리로 일원화하고 base와 일관성을 맞춘다. 학습 동작(에폭 길이·소비 비율·셔플 강도)은 불변으로 유지한다.

**Architecture:** `pretrain.py`에 흩어진 base+cc12m 로더 조립을 `data/pretrain_loader.py`의 `build_pretrain_dataloader()` 하나로 캡슐화한다. cc12m transform을 base와 동일하게 `pretrain_train_aug` 토글에 종속시키고, `shardshuffle`을 명시적 int로, `with_epoch`를 단일/DDP 공통으로 적용한다. 죽은 주석을 제거한다.

**Tech Stack:** PyTorch 2.6 (conda env `kd_v4`), webdataset 0.2.111, torchvision. 테스트는 `python -m pytest`(repo 루트, 설정파일 없음). 이 repo의 와이어링 테스트 관례는 `inspect.getsource` 기반(예: `tests/test_pretrain_teacher_embeds_wiring.py`).

## Global Constraints

- 브랜치: `cc12m_integration` (5fee9ad 분기, randaugment fix 커밋 `77408f7` 포함).
- **동작 불변 원칙**: `cc12m_tar_path: ''`(baseline)에서 반환 경로는 기존과 동일한 base `DataLoader`여야 한다(바이트 동일 경로).
- 반환 객체 계약: `pretrain.py`가 `data_loader.sampler.set_epoch(epoch)`와 `len(data_loader)`에 의존한다 — cc12m OFF(`DataLoader`)/ON(`CombinedLoader`) 둘 다 이 계약을 만족해야 한다.
- ratio 확정값 = 1 (config 기본은 5로 남겨두되 이번 작업은 코드 구조만 변경, 값 변경 없음).
- 테스트 실행: 항상 repo 루트(`/home/minwoo/Distillation_Project`)에서 `python -m pytest`. env는 `kd_v4`.
- spec: `docs/superpowers/specs/2026-07-23-cc12m-dataloader-design.md`.

---

## File Structure

- `data/pretrain_cc12m_webdataset.py` — **수정**: `__init__`에 `shardshuffle_size=100` 파라미터 추가, `shardshuffle=True` → `shardshuffle=shardshuffle_size`.
- `data/__init__.py` — **수정**: (a) `create_dataset`의 cc12m 분기가 `pretrain_train_aug` 토글 반영, (b) `create_loader`의 죽은 cc12m 주석 블록 제거.
- `data/pretrain_loader.py` — **신규**: `build_pretrain_dataloader(config, min_scale=0.2)` 팩토리.
- `pretrain.py` — **수정**: dataset 블록(343–378)을 팩토리 호출로 대체 + 이제 미사용이 되는 임포트(27, 28, 36) 정리.
- `tests/test_cc12m_dataloader.py` — **신규**: 소스-와이어링 테스트 + 실제 샤드 기반 behavioral 회귀(스킵 가드).

---

### Task 1: cc12m shardshuffle 명시적 int

**Files:**
- Modify: `data/pretrain_cc12m_webdataset.py:10`, `data/pretrain_cc12m_webdataset.py:29`
- Test: `tests/test_cc12m_dataloader.py`

**Interfaces:**
- Consumes: 없음
- Produces: `cc12m_webdataset.__init__(self, tar_root, transform, batch_size, shardshuffle_size=100)` — 기본 셔플 강도 100(기존 암묵 동작과 동일).

- [ ] **Step 1: Write the failing test**

`tests/test_cc12m_dataloader.py` 생성:

```python
import inspect

from data.pretrain_cc12m_webdataset import cc12m_webdataset


def test_shardshuffle_size_param_default_100():
    sig = inspect.signature(cc12m_webdataset.__init__)
    assert 'shardshuffle_size' in sig.parameters
    assert sig.parameters['shardshuffle_size'].default == 100


def test_no_literal_shardshuffle_true_in_source():
    src = inspect.getsource(cc12m_webdataset)
    assert 'shardshuffle=True' not in src
    assert 'shardshuffle=shardshuffle_size' in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cc12m_dataloader.py -v`
Expected: FAIL — `shardshuffle_size` 파라미터 없음 / `shardshuffle=True`가 소스에 존재.

- [ ] **Step 3: Write minimal implementation**

`data/pretrain_cc12m_webdataset.py` line 10, 시그니처 변경:

```python
    def __init__(self, tar_root, transform, batch_size, shardshuffle_size=100):
```

`data/pretrain_cc12m_webdataset.py` line 29, `shardshuffle=True` → 파라미터 사용:

```python
            wds.WebDataset(
                self.tar_files,
                shardshuffle=shardshuffle_size,
                nodesplitter=wds.split_by_node,  # 멀티 GPU 분할 가속
                handler=wds.warn_and_continue    # 에러 방어선
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cc12m_dataloader.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add data/pretrain_cc12m_webdataset.py tests/test_cc12m_dataloader.py
git commit -m "refactor(cc12m): make shardshuffle explicit int (default 100)

wds 0.2.111 warns on shardshuffle=True and silently uses 100. Expose it
as shardshuffle_size=100 (behaviorally identical) to remove the warning
and make the shuffle strength explicit."
```

---

### Task 2: cc12m transform이 pretrain_train_aug 토글 반영

**Files:**
- Modify: `data/__init__.py:47-53` (`create_dataset`의 cc12m 분기)
- Test: `tests/test_cc12m_dataloader.py`

**Interfaces:**
- Consumes: 없음
- Produces: `create_dataset('pretrain_cc12m_webdataset', config, ...)`가 `config['pretrain_train_aug']`(기본 True)에 따라 `transform_train`(aug ON) 또는 `transform_test`(결정적) 선택.

- [ ] **Step 1: Write the failing test**

`tests/test_cc12m_dataloader.py`에 추가:

```python
import data as data_pkg


def test_cc12m_transform_respects_pretrain_train_aug():
    src = inspect.getsource(data_pkg.create_dataset)
    # cc12m 분기가 base와 동일하게 토글을 참조해야 한다(하드코딩 transform_train 금지)
    assert "config.get('pretrain_train_aug'" in src
    assert 'cc12m_transform' in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cc12m_dataloader.py::test_cc12m_transform_respects_pretrain_train_aug -v`
Expected: FAIL — 현재 cc12m 분기는 `transform=transform_train` 하드코딩이라 `cc12m_transform`/토글 참조 없음.

- [ ] **Step 3: Write minimal implementation**

`data/__init__.py`의 cc12m 분기(47–53)를 다음으로 교체:

```python
    elif dataset=='pretrain_cc12m_webdataset':
        # base('pretrain')와 동일 규칙: pretrain_train_aug로 aug/no-aug 선택.
        # online teacher distillation은 aug ON 필요(no-aug는 NO-GO). 기본 True.
        use_train_aug = config.get('pretrain_train_aug', True)
        cc12m_transform = transform_train if use_train_aug else transform_test
        dataset = cc12m_webdataset(
            tar_root=config['cc12m_tar_path'],
            transform=cc12m_transform,
            batch_size=config['batch_size'] # 이 코드는 여기서 받아와야함.
        )
        return dataset
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cc12m_dataloader.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add data/__init__.py tests/test_cc12m_dataloader.py
git commit -m "refactor(cc12m): honor pretrain_train_aug toggle for cc12m transform

Previously cc12m hardcoded transform_train while base('pretrain') switched
on pretrain_train_aug. Align cc12m to the same toggle so aug/no-aug control
is consistent across sources."
```

---

### Task 3: build_pretrain_dataloader 팩토리 신규

**Files:**
- Create: `data/pretrain_loader.py`
- Test: `tests/test_cc12m_dataloader.py`

**Interfaces:**
- Consumes: `data.create_dataset`, `data.create_sampler`, `data.create_loader`, `data.combined_loader.CombinedLoader`, `webdataset` (wds), `utils.get_world_size/get_rank`. cc12m transform 토글은 Task 2에서 `create_dataset` 내부에 이미 반영됨.
- Produces: `build_pretrain_dataloader(config, min_scale=0.2) -> data_loader`
  - cc12m OFF: torch `DataLoader` (base)
  - cc12m ON: `CombinedLoader` (loader_map=base, loader_iterable=cc12m WebLoader, ratio)
  - 두 경우 모두 `.sampler`(set_epoch 가능)와 `__len__` 제공.

- [ ] **Step 1: Write the failing test**

`tests/test_cc12m_dataloader.py`에 추가:

```python
def test_factory_returns_base_loader_when_no_cc12m():
    from data.pretrain_loader import build_pretrain_dataloader
    src = inspect.getsource(build_pretrain_dataloader)
    assert "if not config['cc12m_tar_path']:" in src
    assert 'return base_loader' in src


def test_factory_applies_with_epoch_unconditionally():
    from data.pretrain_loader import build_pretrain_dataloader
    src = inspect.getsource(build_pretrain_dataloader)
    # with_epoch는 num_tasks>1 게이트 없이 항상 적용(단일/DDP 일관)
    assert '.with_epoch(' in src
    assert 'num_tasks > 1' not in src


def test_factory_builds_combined_loader():
    from data.pretrain_loader import build_pretrain_dataloader
    src = inspect.getsource(build_pretrain_dataloader)
    assert 'CombinedLoader(' in src


def test_factory_docstring_has_operational_notes():
    from data.pretrain_loader import build_pretrain_dataloader
    doc = build_pretrain_dataloader.__doc__ or ''
    assert 'torchrun' in doc        # queue distill 분산 필수 note
    assert 'warmup_steps' in doc     # 에폭 길이 커플링 note
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cc12m_dataloader.py -v`
Expected: FAIL — `data/pretrain_loader.py` 없음(ModuleNotFoundError/ImportError).

- [ ] **Step 3: Write minimal implementation**

`data/pretrain_loader.py` 생성:

```python
# cc12m 통합 사전학습 데이터로더 팩토리 (일원화)
import webdataset as wds

import utils
from data import create_dataset, create_sampler, create_loader
from data.combined_loader import CombinedLoader


def build_pretrain_dataloader(config, min_scale=0.2):
    """coco+vg base 로더를 만들고, cc12m_tar_path가 있으면 CombinedLoader로 결합해 반환.

    반환 계약(pretrain.py가 의존): 반환 객체는 .sampler(set_epoch 가능)와 len()을
    제공한다. cc12m OFF면 base DataLoader, ON이면 CombinedLoader.

    주의(문서 note): itc_target_mix variant=queue 등 concat_all_gather 경로는
    torch.distributed 초기화가 필요하다 → 반드시 torchrun으로 실행. (assert는 두지 않음)

    주의(에폭 길이 커플링): cc12m ON이면 len(data_loader)가 (1+ratio)배가 된다.
    warmup_steps 등 '절대 step' 하이퍼파라미터는 에폭 길이 변화에 맞춰 재튜닝 필요.
    """
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    base_dataset = create_dataset('pretrain', config, min_scale=min_scale)
    print('number of training samples: %d' % len(base_dataset))
    sampler = create_sampler([base_dataset], [True], num_tasks, global_rank)[0]
    base_loader = create_loader(
        [base_dataset], [sampler],
        batch_size=[config['batch_size']], num_workers=[4],
        is_trains=[True], collate_fns=[None])[0]

    if not config['cc12m_tar_path']:
        return base_loader

    print("Creating cc12m dataset")
    ratio = config['cc12m_ratio']
    cc12m_dataset = create_dataset('pretrain_cc12m_webdataset', config, min_scale=min_scale)
    cc12m_loader = wds.WebLoader(
        cc12m_dataset, batch_size=None, num_workers=4, pin_memory=True)
    # #6 with_epoch 일관 적용(단일 GPU·DDP 공통): rank당 배치수를 len(base)*ratio로 고정.
    #    DDP에서 split_by_node로 rank별 샤드 수가 달라도 collective를 동기화한다.
    cc12m_loader = cc12m_loader.with_epoch(len(base_loader) * ratio)

    return CombinedLoader(
        loader_map=base_loader, loader_iterable=cc12m_loader, ratio=ratio)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cc12m_dataloader.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add data/pretrain_loader.py tests/test_cc12m_dataloader.py
git commit -m "feat(data): add build_pretrain_dataloader factory (cc12m unification)

Encapsulate base(coco+vg)+cc12m loader assembly in one factory. Applies
with_epoch unconditionally (single-GPU/DDP consistent) and documents the
two operational gotchas (queue-distill needs torchrun; epoch length scales
by 1+ratio). Behavior unchanged vs the inline wiring in pretrain.py."
```

---

### Task 4: pretrain.py를 팩토리 호출로 재배선

**Files:**
- Modify: `pretrain.py:343-378` (dataset 블록 교체), `pretrain.py:27-28` (미사용 임포트 제거), `pretrain.py:36` (미사용 임포트 제거), `pretrain.py` 상단(팩토리 임포트 추가)
- Test: `tests/test_cc12m_dataloader.py`

**Interfaces:**
- Consumes: `data.pretrain_loader.build_pretrain_dataloader` (Task 3).
- Produces: `main()`이 `data_loader = build_pretrain_dataloader(config, min_scale=0.2)`로 로더를 얻는다. 인라인 `wds.WebLoader`/`CombinedLoader` 조립 없음.

- [ ] **Step 1: Write the failing test**

`tests/test_cc12m_dataloader.py`에 추가:

```python
import pretrain


def test_pretrain_uses_factory():
    src = inspect.getsource(pretrain.main)
    assert 'build_pretrain_dataloader(' in src


def test_pretrain_has_no_inline_cc12m_wiring():
    src = inspect.getsource(pretrain.main)
    assert 'wds.WebLoader(' not in src
    assert 'CombinedLoader(' not in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cc12m_dataloader.py::test_pretrain_uses_factory tests/test_cc12m_dataloader.py::test_pretrain_has_no_inline_cc12m_wiring -v`
Expected: FAIL — `main()`이 아직 인라인 `wds.WebLoader(`/`CombinedLoader(`를 포함하고 팩토리 미사용.

- [ ] **Step 3: Write minimal implementation**

(a) `pretrain.py`의 dataset 블록(343–378)을 다음으로 교체:

```python
    #### Dataset ####
    print("Creating dataset")
    data_loader = build_pretrain_dataloader(config, min_scale=0.2)
```

(b) `pretrain.py` 상단 임포트 정리 — 27, 28번 줄 삭제:

```python
# 삭제: from data.combined_loader import CombinedLoader
# 삭제: import webdataset as wds # 웹데이터셋용으로 추가
```

(c) `pretrain.py:36` 임포트를 팩토리 임포트로 교체(create_dataset/create_sampler/create_loader는 재배선 후 pretrain.py에서 미사용):

```python
from data.pretrain_loader import build_pretrain_dataloader
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cc12m_dataloader.py -v`
Expected: PASS (9 passed).

추가 임포트 무결성 확인 (미사용 심볼 제거로 import 깨지지 않았는지):

Run: `python -c "import pretrain"`
Expected: 에러 없음(정상 종료).

- [ ] **Step 5: Commit**

```bash
git add pretrain.py tests/test_cc12m_dataloader.py
git commit -m "refactor(pretrain): use build_pretrain_dataloader factory

Replace the scattered inline base+cc12m loader assembly (WebLoader,
with_epoch, CombinedLoader) with a single factory call, and drop the now
unused imports (CombinedLoader, wds, create_dataset/sampler/loader).
Behavior unchanged."
```

---

### Task 5: create_loader 죽은 주석 블록 제거

**Files:**
- Modify: `data/__init__.py:121-139` (주석 처리된 옛 cc12m 분기 삭제)
- Test: `tests/test_cc12m_dataloader.py`

**Interfaces:**
- Consumes: 없음
- Produces: `create_loader`가 loaders를 append한 뒤 바로 `return loaders`로 이어짐(중간 죽은 주석 없음). 기능 불변.

- [ ] **Step 1: Write the failing test**

`tests/test_cc12m_dataloader.py`에 추가:

```python
def test_create_loader_no_dead_cc12m_comment():
    src = inspect.getsource(data_pkg.create_loader)
    assert 'ddp_equalize' not in src
    assert '잘못된 데이터셋' not in src
    assert 'cc12m_loader = wds.WebLoader' not in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cc12m_dataloader.py::test_create_loader_no_dead_cc12m_comment -v`
Expected: FAIL — 죽은 주석에 `ddp_equalize`/`잘못된 데이터셋` 문자열 존재.

- [ ] **Step 3: Write minimal implementation**

`data/__init__.py`의 `create_loader` 안, `loaders.append(loader)` 다음의 주석 블록(121–139)을 삭제해 다음 형태로 만든다:

```python
        loader = DataLoader(
            dataset=dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)

    return loaders
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cc12m_dataloader.py -v`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add data/__init__.py tests/test_cc12m_dataloader.py
git commit -m "chore(data): remove dead cc12m branch comment in create_loader

The commented-out old cc12m WebLoader/ddp_equalize branch is superseded by
data/pretrain_loader.py and only caused confusion. Delete it."
```

---

### Task 6: 실제 샤드 기반 behavioral 회귀 테스트 + 통합 스모크 게이트

**Files:**
- Modify: `tests/test_cc12m_dataloader.py` (스킵 가드 behavioral 테스트 추가)

**Interfaces:**
- Consumes: `data.pretrain_cc12m_webdataset.cc12m_webdataset`, `webdataset`, `torchvision`. 실제 cc12m 샤드(`/home/minwoo/Distillation_Project/datasets/vision/cc12m/cc12m_dataset`).
- Produces: 없음(회귀 안전망).

- [ ] **Step 1: Write the behavioral regression test**

`tests/test_cc12m_dataloader.py`에 추가:

```python
import os
import unittest

CC12M_SHARD_DIR = "/home/minwoo/Distillation_Project/datasets/vision/cc12m/cc12m_dataset"


class CC12MBehavioralRegression(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
            import webdataset  # noqa: F401
            import torchvision  # noqa: F401
        except Exception:
            self.skipTest("torch/webdataset/torchvision not available")
        if not os.path.isdir(CC12M_SHARD_DIR) or not any(
            f.endswith('.tar') for f in os.listdir(CC12M_SHARD_DIR)
        ):
            self.skipTest("cc12m shards not present on this machine")

    def test_batch_format_and_with_epoch_cap(self):
        import torch
        import webdataset as wds
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode
        from data.pretrain_cc12m_webdataset import cc12m_webdataset

        norm = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711))
        tf = transforms.Compose([
            transforms.RandomResizedCrop(
                224, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            norm,
        ])
        ds = cc12m_webdataset(tar_root=CC12M_SHARD_DIR, transform=tf, batch_size=8)
        loader = wds.WebLoader(ds, batch_size=None, num_workers=2).with_epoch(3)

        n = 0
        for img, cap in loader:
            self.assertIsInstance(img, torch.Tensor)
            self.assertEqual(tuple(img.shape), (8, 3, 224, 224))
            self.assertTrue(img.dtype == torch.float32)
            self.assertIsInstance(cap, (list, tuple))
            self.assertEqual(len(cap), 8)
            n += 1
        self.assertEqual(n, 3)  # with_epoch(3) caps the epoch to 3 batches
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/test_cc12m_dataloader.py -v`
Expected: PASS (11 passed) — 이 머신엔 샤드가 있으므로 실행됨. 배치 포맷 parity + with_epoch 캡 확인. (샤드 없는 환경이면 skip.)

- [ ] **Step 3: L2 end-to-end 통합 스모크 게이트 (수동, GPU 필요)**

리팩터 후 cc12m ON 경로가 실제 train 스텝을 통과하는지 재확인. scratchpad의 기존 스모크 config 재사용:

```bash
cd /home/minwoo/Distillation_Project
CUDA_VISIBLE_DEVICES=0 /home/minwoo/miniconda3/envs/kd_v4/bin/torchrun \
  --nproc_per_node=1 --master_port=29531 pretrain.py \
  --config /tmp/claude-1008/-home-minwoo-Distillation-Project/e68da99c-9819-45f3-8418-5c624654d02b/scratchpad/l2_smoke.yaml \
  2>&1 | grep -E "keep=|Start training|Train Epoch: \[0\]|Traceback"
```

Expected:
- `[distill] online teacher loaded (keep=('itc', 'itm', 'lm'))`
- `Train Epoch: [0]  [    0/70842]` (cc12m ON → len = coco×2 = 70842)
- step 50까지 도달, `Traceback` 없음, loss 전부 유한.
- 확인 후 프로세스 종료(Ctrl-C 또는 timeout).

(참고: `l2_smoke.yaml`이 정리됐으면 `configs/pretrain_itc_itm_lm_targetmix_smoke.yaml`을 복제해 `cc12m_tar_path`를 실제 샤드 몇 개 심볼릭 링크 디렉토리로, `cc12m_ratio: 1`, `train_file`를 coco-only로 두고 동일 확인.)

- [ ] **Step 4: Commit**

```bash
git add tests/test_cc12m_dataloader.py
git commit -m "test(cc12m): add shard-based behavioral regression (skip-guarded)

Durable L1 regression: builds cc12m_webdataset over real shards and asserts
batch format parity (FloatTensor[8,3,224,224] + caption list) and that
with_epoch caps the epoch. Skips when shards or deps are absent."
```

---

## Self-Review

**1. Spec coverage** (spec §3 항목 vs 태스크):
- §3.1 `build_pretrain_dataloader` 팩토리 → Task 3. ✓
- §3.1 `with_epoch` 일관 적용(#6) → Task 3(무조건 적용, num_tasks 게이트 없음). ✓
- §3.1 pretrain.py 재배선 → Task 4. ✓
- §3.2 cc12m transform이 pretrain_train_aug 반영(#2) → Task 2. ✓
- §3.3 shardshuffle 명시적 int(#3) → Task 1. ✓
- §3.4 죽은 코드 제거(#4) → Task 5. ✓
- §4 문서 note(queue-torchrun, 에폭길이)(#7) → Task 3 docstring + `test_factory_docstring_has_operational_notes`. ✓
- §5 테스트: L1 회귀 → Task 6 behavioral, baseline 불변 → Task 3 `test_factory_returns_base_loader_when_no_cc12m`(소스) + Task 4 `python -c "import pretrain"`, transform 토글 → Task 2, L2 스모크 → Task 6 Step 3. ✓

**2. Placeholder scan:** 모든 코드 스텝에 실제 코드 포함. "TBD/TODO/적절히 처리" 없음. ✓

**3. Type consistency:** `build_pretrain_dataloader(config, min_scale=0.2)` — Task 3 정의, Task 4 호출 시그니처 일치. `cc12m_webdataset(..., shardshuffle_size=100)` — Task 1 정의, Task 6 호출은 기본값 사용(shardshuffle_size 미지정) — 일치. `CombinedLoader(loader_map=, loader_iterable=, ratio=)` — 기존 시그니처(`data/combined_loader.py`)와 일치. ✓

**Note (baseline 불변 강도):** baseline 경로 불변은 소스-와이어링 + `import pretrain` 무결성으로 검증한다. 완전한 바이트-동일 행위 증명은 현재 진행 중인 baseline 학습 런(`pretrain_student_fixaug.yaml`, cc12m OFF)이 정상 수렴하는지로 사후 확인된다(별도 진행 중, 이 계획의 범위 밖).
