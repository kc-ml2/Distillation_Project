# Pretrain Caption 메트릭 Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** pretrain 학습 루프 안에서 COCO val 캡션을 생성·채점(BLEU/METEOR/ROUGE_L/CIDEr/SPICE)하여 TB에 로깅하고, 터미널 로그에 에폭 실경과 시간을 표시한다.

**Architecture:** 기존 `RetrievalValRunner` 패턴을 대칭 복제한 `CaptionValRunner`(신규 `data/eval_validation_caption.py`)가 DDP 샤딩 생성 → rank0 취합 → `coco_caption_eval` 채점 → TB 로깅을 담당한다. 생성을 위해 `BLIP_Pretrain`에 `generate()`를 추가하고(기존 `BLIP_Decoder.generate` 미러링), SPICE의 코어 독점은 `os.sched_setaffinity` 캡으로 막는다. 실경과는 `MetricLogger.log_every`에 필드 하나 추가.

**Tech Stack:** PyTorch DDP(nccl), transformers `BertLMHeadModel.generate`(beam search), pycocoevalcap(BLEU/METEOR/ROUGE/CIDEr/SPICE, Java), TensorBoard.

## Global Constraints

- **cadence:** 경량 지표(BLEU/METEOR/ROUGE_L/CIDEr)는 `val_caption_mid_interval_steps`(=18675, ITM mid와 동일)와 epoch-end에서 실행(2×/epoch). **SPICE는 epoch-end에서만**(1×/epoch), `val_caption_use_spice`로 제어.
- **CPU 코어 캡:** 채점은 `caption_score_cpu_list`(예 `'0-31'`) 코어셋으로 제한(`null`=캡 없음). MPS 2-잡 시 잡A `'0-31'` / 잡B `'32-63'`로 수동 분리.
- **인라인(블로킹).** async 오프로드 없음. val=full 5000, test split 미채점.
- **기존 패턴 준수:** `RetrievalValRunner`(`data/eval_validation_retrieval.py`)와 대칭. 새 코드는 rank0에서만 채점/로깅.
- **생성 파라미터:** `caption_num_beams=3`, `caption_max_length=20`, `caption_min_length=5`, `caption_prompt=''`.
- **환경:** conda env python = `/home/minwoo/miniconda3/envs/kd_r4/bin/python`. 테스트도 이 인터프리터로 실행. 커밋은 dev 브랜치 정책상 사람 승인 후(에이전트는 각 Task 끝에서 커밋 step을 제안하되 실제 커밋 전 확인).

---

### Task 1: `BLIP_Pretrain.generate()` 추가

**Files:**
- Modify: `models/blip_pretrain.py` (클래스 `BLIP_Pretrain`에 메서드 추가, `forward` 정의 근처 `models/blip_pretrain.py:307` 이후)
- Test: `tests/test_pretrain_generate.py`

**Interfaces:**
- Produces: `BLIP_Pretrain.generate(self, image, sample=False, num_beams=3, max_length=20, min_length=5, top_p=0.9, repetition_penalty=1.0, prompt='') -> list[str]` (길이 = `image.size(0)`).
- Consumes: 기존 인스턴스 속성 `self.visual_encoder`, `self.tokenizer`, `self.text_decoder`(`BertLMHeadModel`).

- [ ] **Step 1: 실패 테스트 작성** — `tests/test_pretrain_generate.py`

```python
import types
import torch
from unittest import mock

from models.blip_pretrain import BLIP_Pretrain
from models.blip import init_tokenizer


def test_generate_returns_one_caption_per_image():
    tokenizer = init_tokenizer()

    def fake_decoder_generate(**kwargs):
        bsz = kwargs["input_ids"].size(0)
        # "a cat" 문장을 batch만큼 반환 (토큰 id 텐서)
        return tokenizer(["a cat"] * bsz, return_tensors="pt").input_ids

    fake = types.SimpleNamespace(
        visual_encoder=lambda img: torch.zeros(img.size(0), 5, 8),
        tokenizer=tokenizer,
        text_decoder=types.SimpleNamespace(generate=fake_decoder_generate),
    )

    image = torch.zeros(3, 3, 224, 224)
    caps = BLIP_Pretrain.generate(fake, image, num_beams=1)

    assert isinstance(caps, list)
    assert len(caps) == 3
    assert all(isinstance(c, str) for c in caps)
```

- [ ] **Step 2: 실패 확인**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_pretrain_generate.py -v`
Expected: FAIL — `AttributeError: type object 'BLIP_Pretrain' has no attribute 'generate'`

- [ ] **Step 3: 최소 구현** — `models/blip_pretrain.py`, `def forward(` 정의 바로 위(또는 아래)에 메서드 추가. `BLIP_Decoder.generate`(`models/blip.py:137`)를 미러링하되 `self.prompt` 대신 `prompt` 인자를 쓴다.

```python
    def generate(self, image, sample=False, num_beams=3, max_length=20, min_length=5,
                 top_p=0.9, repetition_penalty=1.0, prompt=''):
        """캡션 생성 (BLIP_Decoder.generate 미러링). pretrain 모델의
        visual_encoder + text_decoder + tokenizer를 그대로 사용, prompt는 인자."""
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}

        prompts = [prompt] * image.size(0)
        input_ids = self.tokenizer(prompts, return_tensors="pt").input_ids.to(image.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        if sample:
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  do_sample=True,
                                                  top_p=top_p,
                                                  num_return_sequences=1,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,
                                                  repetition_penalty=1.1,
                                                  **model_kwargs)
        else:
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  num_beams=num_beams,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,
                                                  repetition_penalty=repetition_penalty,
                                                  **model_kwargs)

        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(prompt):])
        return captions
```

- [ ] **Step 4: 통과 확인**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_pretrain_generate.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add models/blip_pretrain.py tests/test_pretrain_generate.py
git commit -m "feat(pretrain): add BLIP_Pretrain.generate() mirroring BLIP_Decoder"
```

---

### Task 2: `MetricLogger.log_every`에 에폭 실경과(elapsed) 필드

**Files:**
- Modify: `utils.py:135-179` (`MetricLogger.log_every`)
- Test: `tests/test_metriclogger_elapsed.py`

**Interfaces:**
- Produces: 각 로그 라인에 `elapsed: H:MM:SS` 필드(= `time.time() - start_time`, validation 포함 실경과). ETA(`eta:`)는 유지.

- [ ] **Step 1: 실패 테스트 작성** — `tests/test_metriclogger_elapsed.py`

```python
import io
import contextlib

from utils import MetricLogger


def test_log_every_prints_elapsed():
    ml = MetricLogger()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for _ in ml.log_every([1, 2, 3], print_freq=1, header="H"):
            pass
    out = buf.getvalue()
    assert "elapsed:" in out
```

- [ ] **Step 2: 실패 확인**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_metriclogger_elapsed.py -v`
Expected: FAIL — `assert "elapsed:" in out` (현재 출력에 elapsed 없음)

- [ ] **Step 3: 구현** — `utils.py`의 `log_msg` 리스트(`utils.py:144-151`)에 `'elapsed: {elapsed}'`를 `'eta: {eta}'` 다음에 추가하고, 두 print 분기(`utils.py:163-173`)에 `elapsed`를 넘긴다.

`log_msg` 리스트를 다음으로 교체:
```python
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            'elapsed: {elapsed}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
```

`eta_string` 계산 직후(`utils.py:162` 아래)에 추가:
```python
                elapsed_string = str(datetime.timedelta(seconds=int(time.time() - start_time)))
```

cuda 분기의 `print(log_msg.format(...))`를:
```python
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string, elapsed=elapsed_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
```
non-cuda 분기의 `print(log_msg.format(...))`를:
```python
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string, elapsed=elapsed_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
```

- [ ] **Step 4: 통과 확인**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_metriclogger_elapsed.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add utils.py tests/test_metriclogger_elapsed.py
git commit -m "feat(utils): show epoch-elapsed wall time in MetricLogger.log_every"
```

---

### Task 3: caption 평가 모듈 스캐폴딩 — `parse_cpu_list` + `cpu_affinity`

**Files:**
- Create: `data/eval_validation_caption.py` (헬퍼만 먼저; 러너는 Task 4에서 추가)
- Test: `tests/test_caption_cpu_affinity.py`

**Interfaces:**
- Produces:
  - `parse_cpu_list(spec) -> set[int] | None` (`'0-3,8'`→`{0,1,2,3,8}`, `None`/`''`→`None`)
  - `cpu_affinity(cores)` — 컨텍스트 매니저. 진입 시 `os.sched_setaffinity(0, cores)`, 종료 시 원복. `cores`가 None이거나 플랫폼 미지원이면 no-op.

- [ ] **Step 1: 실패 테스트 작성** — `tests/test_caption_cpu_affinity.py`

```python
import os
import unittest

from data.eval_validation_caption import parse_cpu_list, cpu_affinity


class ParseCpuListTest(unittest.TestCase):
    def test_range_and_singles(self):
        self.assertEqual(parse_cpu_list("0-3"), {0, 1, 2, 3})
        self.assertEqual(parse_cpu_list("0-1,4,6-7"), {0, 1, 4, 6, 7})

    def test_empty_is_none(self):
        self.assertIsNone(parse_cpu_list(None))
        self.assertIsNone(parse_cpu_list(""))


class CpuAffinityTest(unittest.TestCase):
    def test_restores_original(self):
        if not hasattr(os, "sched_getaffinity"):
            self.skipTest("no sched_getaffinity on this platform")
        orig = os.sched_getaffinity(0)
        with cpu_affinity({0}):
            self.assertEqual(os.sched_getaffinity(0), {0})
        self.assertEqual(os.sched_getaffinity(0), orig)

    def test_none_is_noop(self):
        if not hasattr(os, "sched_getaffinity"):
            self.skipTest("no sched_getaffinity on this platform")
        orig = os.sched_getaffinity(0)
        with cpu_affinity(None):
            self.assertEqual(os.sched_getaffinity(0), orig)
        self.assertEqual(os.sched_getaffinity(0), orig)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실패 확인**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_caption_cpu_affinity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data.eval_validation_caption'`

- [ ] **Step 3: 구현** — `data/eval_validation_caption.py` 생성 (헬퍼만)

```python
import os


def parse_cpu_list(spec):
    """'0-31' 또는 '0-3,8,10-12' -> 정렬 불필요한 코어 집합. None/'' -> None."""
    if not spec:
        return None
    cores = set()
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-')
            cores.update(range(int(a), int(b) + 1))
        else:
            cores.add(int(part))
    return cores or None


class cpu_affinity:
    """채점 구간 동안 프로세스(및 SPICE의 java 자식)를 지정 코어에 제한하고
    종료 시 원복. cores=None 또는 플랫폼 미지원이면 no-op."""

    def __init__(self, cores):
        self.cores = cores
        self.orig = None

    def __enter__(self):
        if self.cores and hasattr(os, "sched_setaffinity"):
            self.orig = os.sched_getaffinity(0)
            os.sched_setaffinity(0, self.cores)
        return self

    def __exit__(self, *exc):
        if self.orig is not None:
            os.sched_setaffinity(0, self.orig)
        return False
```

- [ ] **Step 4: 통과 확인**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_caption_cpu_affinity.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: 커밋**

```bash
git add data/eval_validation_caption.py tests/test_caption_cpu_affinity.py
git commit -m "feat(caption-eval): cpu_affinity cap + cpu-list parser"
```

---

### Task 4: `CaptionValRunner` + 로더 빌더 + cadence 테스트

**Files:**
- Modify: `data/eval_validation_caption.py` (Task 3 파일에 러너/로더/팩토리 추가)
- Test: `tests/test_caption_val_runner.py`

**Interfaces:**
- Consumes: `BLIP_Pretrain.generate(...)` (Task 1), `parse_cpu_list`/`cpu_affinity` (Task 3), `data.utils.coco_caption_eval(coco_gt_root, results_file, split, use_spice) -> coco_eval`(`.eval` dict 보유), `data.coco_karpathy_dataset.coco_karpathy_caption_eval`, `data.eval_validation_loss.build_val_loss_transform`.
- Produces:
  - `CaptionValRunner(data_loader, device, config, writer=None)` with
    `val_caption_during_train(model_without_ddp, epoch, iteration, global_step) -> dict`,
    `run_epoch_end(model_without_ddp, epoch, global_step) -> dict`.
  - `build_caption_val_loader(config) -> DataLoader`
  - `build_pretrain_caption_val_runner(config, device, writer=None) -> CaptionValRunner | None`

- [ ] **Step 1: 실패 테스트 작성** — `tests/test_caption_val_runner.py`

```python
import unittest
from unittest import mock

import torch
import torch.nn as nn

import data.eval_validation_caption as evc


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def generate(self, image, **kwargs):
        return ["a caption"] * image.size(0)


def fake_loader():
    # (image, image_id) 배치 하나
    return [(torch.zeros(2, 3, 8, 8), torch.tensor([10, 11]))]


class CaptionValRunnerTest(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.model.train()
        self.config = {
            "val_caption_mid_interval_steps": 50,
            "val_caption_epoch_end": True,
            "val_caption_use_spice": True,
            "val_caption_gt_root": "/tmp/gt",
            "val_caption_split": "val",
            "output_dir": "/tmp",
            "caption_score_cpu_list": None,
        }
        self.runner = evc.CaptionValRunner(
            data_loader=fake_loader(), device="cpu", config=self.config, writer=None)

    def _patch_score(self):
        # coco_caption_eval을 mock: .eval dict를 가진 객체 반환
        m = mock.patch("data.eval_validation_caption.coco_caption_eval")
        mock_eval = m.start()
        self.addCleanup(m.stop)
        obj = mock.Mock()
        obj.eval = {"CIDEr": 1.0, "METEOR": 0.3, "SPICE": 0.2}
        mock_eval.return_value = obj
        return mock_eval

    def test_mid_runs_light_only_at_interval(self):
        mock_eval = self._patch_score()
        # iteration != mid -> 실행 안 함
        self.runner.val_caption_during_train(self.model, epoch=0, iteration=49, global_step=49)
        self.assertEqual(mock_eval.call_count, 0)
        # iteration == mid -> 실행, use_spice=False
        self.runner.val_caption_during_train(self.model, epoch=0, iteration=50, global_step=50)
        self.assertEqual(mock_eval.call_count, 1)
        self.assertFalse(mock_eval.call_args.kwargs["use_spice"])

    def test_epoch_end_uses_spice_and_restores_train(self):
        mock_eval = self._patch_score()
        self.model.train()
        metrics = self.runner.run_epoch_end(self.model, epoch=0, global_step=37346)
        self.assertEqual(mock_eval.call_count, 1)
        self.assertTrue(mock_eval.call_args.kwargs["use_spice"])
        self.assertEqual(metrics["CIDEr"], 1.0)
        self.assertTrue(self.model.training)

    def test_epoch_end_respects_disabled(self):
        mock_eval = self._patch_score()
        self.config["val_caption_epoch_end"] = False
        self.runner.run_epoch_end(self.model, epoch=0, global_step=37346)
        self.assertEqual(mock_eval.call_count, 0)


class BuildCaptionRunnerFactoryTest(unittest.TestCase):
    def test_returns_none_when_disabled(self):
        runner = evc.build_pretrain_caption_val_runner(
            config={"val_caption_enabled": False}, device="cpu", writer=None)
        self.assertIsNone(runner)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실패 확인**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_caption_val_runner.py -v`
Expected: FAIL — `AttributeError: module 'data.eval_validation_caption' has no attribute 'CaptionValRunner'`

- [ ] **Step 3: 구현** — `data/eval_validation_caption.py` 상단 import + 하단에 러너/로더/팩토리 추가.

파일 맨 위 `import os` 아래에 추가:
```python
import json

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import utils
from data.coco_karpathy_dataset import coco_karpathy_caption_eval
from data.eval_validation_loss import build_val_loss_transform
from data.utils import coco_caption_eval
```

파일 하단(헬퍼 뒤)에 추가:
```python
def build_caption_val_loader(config):
    image_root = config["val_caption_image_root"]
    ann_root = config["val_caption_ann_root"]
    split = config.get("val_caption_split", "val")
    image_size = config["image_size"]
    batch_size = config.get("val_caption_batch_size", 32)
    num_workers = config.get("val_caption_num_workers", 4)

    transform = build_val_loss_transform(image_size)
    dataset = coco_karpathy_caption_eval(transform, image_root, ann_root, split)

    sampler = None
    if utils.is_dist_avail_and_initialized() and utils.get_world_size() > 1:
        sampler = DistributedSampler(dataset, num_replicas=utils.get_world_size(),
                                     rank=utils.get_rank(), shuffle=False)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        shuffle=False, num_workers=num_workers,
                        pin_memory=True, drop_last=False)
    return loader


class CaptionValRunner:
    def __init__(self, data_loader, device, config, writer=None):
        self.data_loader = data_loader
        self.device = torch.device(device)
        self.config = config
        self.writer = writer
        self.cpu_cores = parse_cpu_list(config.get("caption_score_cpu_list"))

    def val_caption_during_train(self, model_without_ddp, epoch, iteration, global_step):
        mid = self.config.get("val_caption_mid_interval_steps", 0)
        if mid and iteration == mid:
            return self._run(model_without_ddp, global_step, with_spice=False,
                             header=f"Val Caption Mid: [epoch {epoch} | step {iteration} | global {global_step}]")
        return {}

    def run_epoch_end(self, model_without_ddp, epoch, global_step):
        if not self.config.get("val_caption_epoch_end", True):
            return {}
        with_spice = self.config.get("val_caption_use_spice", True)
        return self._run(model_without_ddp, global_step, with_spice=with_spice,
                         header=f"Val Caption Epoch: [{epoch}]")

    def _run(self, model_without_ddp, global_step, with_spice, header):
        was_training = model_without_ddp.training
        model_without_ddp.eval()
        try:
            preds = self._generate(model_without_ddp)
            all_preds = self._gather(preds)
            if not utils.is_main_process():
                return {}
            metrics = self._score(all_preds, global_step, with_spice)
            msg = f"{header} CIDEr={metrics.get('CIDEr', float('nan')):.3f}"
            if with_spice and "SPICE" in metrics:
                msg += f" SPICE={metrics['SPICE']:.3f}"
            print(msg)
            self._write_tensorboard(metrics, global_step)
            return metrics
        finally:
            if was_training:
                model_without_ddp.train()
            else:
                model_without_ddp.eval()

    @torch.no_grad()
    def _generate(self, model_without_ddp):
        cfg = self.config
        result = []
        for image, image_id in self.data_loader:
            image = image.to(self.device)
            captions = model_without_ddp.generate(
                image, sample=False,
                num_beams=cfg.get("caption_num_beams", 3),
                max_length=cfg.get("caption_max_length", 20),
                min_length=cfg.get("caption_min_length", 5),
                prompt=cfg.get("caption_prompt", ""))
            for cap, iid in zip(captions, image_id):
                result.append({"image_id": int(iid), "caption": cap})
        return result

    def _gather(self, preds):
        if not (utils.is_dist_avail_and_initialized() and utils.get_world_size() > 1):
            return preds
        gathered = [None] * utils.get_world_size()
        dist.all_gather_object(gathered, preds)
        if not utils.is_main_process():
            return None
        merged, seen = [], set()
        for part in gathered:
            for item in part:
                if item["image_id"] not in seen:
                    seen.add(item["image_id"])
                    merged.append(item)
        return merged

    def _score(self, preds, global_step, with_spice):
        out_dir = self.config.get("output_dir", ".")
        res_file = os.path.join(out_dir, f"caption_val_step{global_step}.json")
        with open(res_file, "w") as f:
            json.dump(preds, f)
        with cpu_affinity(self.cpu_cores):
            coco_eval = coco_caption_eval(
                self.config["val_caption_gt_root"], res_file,
                self.config.get("val_caption_split", "val"), use_spice=with_spice)
        return coco_eval.eval

    def _write_tensorboard(self, metrics, global_step):
        if self.writer is None or not utils.is_main_process():
            return
        for key, value in metrics.items():
            self.writer.add_scalar(f"val_caption/{key}", value, global_step)


def build_pretrain_caption_val_runner(config, device, writer=None):
    if not config.get("val_caption_enabled", False):
        return None
    data_loader = build_caption_val_loader(config)
    return CaptionValRunner(data_loader, device, config, writer)
```

- [ ] **Step 4: 통과 확인**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_caption_val_runner.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: 전체 테스트 회귀 확인**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/ -q`
Expected: 기존 테스트 + 신규 모두 PASS (실패 시 import 경로/시그니처 점검)

- [ ] **Step 6: 커밋**

```bash
git add data/eval_validation_caption.py tests/test_caption_val_runner.py
git commit -m "feat(caption-eval): CaptionValRunner - DDP gen/gather + coco_caption_eval + TB"
```

---

### Task 5: `pretrain.py` 배선 + `configs/pretrain.yaml` 키

**Files:**
- Modify: `pretrain.py` (import; `train()` 시그니처 `pretrain.py:77`; train 루프 호출부 `pretrain.py:235` 뒤; main runner 생성 `pretrain.py:323` 뒤; train 호출 `pretrain.py:407`; epoch-end `pretrain.py:432` 뒤)
- Modify: `configs/pretrain.yaml` (caption 키 추가)
- Test: import 스모크(아래) — 배선은 Task 6 e2e에서 실제 검증

**Interfaces:**
- Consumes: `build_pretrain_caption_val_runner`, `CaptionValRunner.val_caption_during_train`, `.run_epoch_end` (Task 4).

- [ ] **Step 1: import 추가** — `pretrain.py:40`(`from data import eval_validation_retrieval`) 아래에:

```python
from data import eval_validation_caption
```

- [ ] **Step 2: `train()` 시그니처에 인자 추가** — `pretrain.py:77` 의 `def train(...)` 끝 `online_teacher=None):` 앞에 `caption_val_runner=None,` 추가:

```python
def train(model, data_loader, optimizer, epoch, device, config, writer=None, val_loss_runner=None, collapse_counter=None, model_without_ddp=None, retrieval_val_runner=None, caption_val_runner=None, online_teacher=None):
```

- [ ] **Step 3: train 루프 호출부 추가** — retrieval `val_retrieval_during_train` 블록(`pretrain.py:227-235`) 바로 뒤에:

```python
        #### 수정부분 시작: train 도중 caption validation optional 실행 (경량 지표만, ITM mid와 동일 시점) ####
        if caption_val_runner is not None:
            caption_val_runner.val_caption_during_train(
                model_without_ddp=model_without_ddp,
                epoch=epoch,
                iteration=i,
                global_step=global_step,
            )
        #### 수정부분 끝 ####
```

- [ ] **Step 4: main()에서 runner 생성** — retrieval runner 생성 블록(`pretrain.py:317-323`) 뒤에:

```python
    #### 수정부분 시작: caption validation runner 생성 (student generate로 COCO val 캡션 채점) ####
    caption_val_runner = eval_validation_caption.build_pretrain_caption_val_runner(
        config=config,
        device=device,
        writer=writer,
    )
    #### 수정부분 끝 ####
```

- [ ] **Step 5: train 호출에 runner 전달** — `pretrain.py:407` 의 `train(...)` 호출에서 `retrieval_val_runner=retrieval_val_runner,` 뒤에 `caption_val_runner=caption_val_runner,` 삽입:

```python
            train_stats = train(model, data_loader, optimizer, epoch, device, config, writer, val_loss_runner=val_loss_runner, collapse_counter=collapse_counter, model_without_ddp=model_without_ddp, retrieval_val_runner=retrieval_val_runner, caption_val_runner=caption_val_runner, online_teacher=online_teacher)
```

- [ ] **Step 6: epoch-end 호출 추가** — retrieval `run_epoch_end` 블록(`pretrain.py:423-432`) 뒤에:

```python
            #### 수정부분 시작: epoch 종료 caption validation 실행 (경량 + SPICE) ####
            if caption_val_runner is not None:
                epoch_end_global_step = (epoch + 1) * len(data_loader)

                caption_val_runner.run_epoch_end(
                    model_without_ddp=model_without_ddp,
                    epoch=epoch,
                    global_step=epoch_end_global_step,
                )
            #### 수정부분 끝 ####
```

- [ ] **Step 7: config 키 추가** — `configs/pretrain.yaml` 의 retrieval 키 블록(`val_retrieval_itm_epoch_end` 아래) 다음에:

```yaml
# ── caption 메트릭 validation ──
val_caption_enabled: true
val_caption_ann_root: '/home/minwoo/Distillation_Project/datasets/vision/coco/annotations/'
val_caption_image_root: '/home/minwoo/Distillation_Project/datasets/vision/coco/images/'
val_caption_gt_root: '/home/minwoo/Distillation_Project/datasets/vision/coco/annotation/coco_gt'
val_caption_split: 'val'
val_caption_batch_size: 32
val_caption_num_workers: 4
val_caption_mid_interval_steps: 18675   # ITM mid와 동일
val_caption_epoch_end: true
val_caption_use_spice: true             # epoch-end에서만 SPICE
caption_score_cpu_list: '0-31'          # 잡A '0-31' / 잡B '32-63' / null=캡 없음
caption_num_beams: 3
caption_max_length: 20
caption_min_length: 5
caption_prompt: ''
```

- [ ] **Step 8: import/파싱 스모크**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -c "import ast, yaml; ast.parse(open('pretrain.py').read()); yaml.safe_load(open('configs/pretrain.yaml')); print('ok')"`
Expected: `ok` (구문/YAML 오류 없음)

- [ ] **Step 9: 커밋**

```bash
git add pretrain.py configs/pretrain.yaml
git commit -m "feat(pretrain): wire CaptionValRunner into train loop + config keys"
```

---

### Task 6: 실기 스모크 검증 (1 GPU, 실제 생성+채점+SPICE+TB)

**Files:**
- Create: `scratchpad/smoke_caption_runner.py` (검증용, 커밋 안 함)

**Interfaces:**
- Consumes: 전체 파이프라인 (Task 1·3·4·5).

- [ ] **Step 1: 스모크 스크립트 작성** — `scratchpad/smoke_caption_runner.py`

```python
import os, yaml, torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from models.blip_pretrain import blip_pretrain
import data.eval_validation_caption as evc

config = yaml.safe_load(open("configs/pretrain.yaml"))
config["output_dir"] = "scratchpad"
device = torch.device("cuda:0")

model = blip_pretrain(image_size=config["image_size"], vit=config["vit"],
                      vit_grad_ckpt=False, vit_ckpt_layer=0,
                      queue_size=config["queue_size"], my_bert_size=config["my_bert_size"],
                      init_backbone_weights=False)
sd = torch.load("output/pt_smallreg_minilm_baseline/checkpoint_19.pth", map_location="cpu", weights_only=False)
model.load_state_dict(sd.get("model", sd), strict=False)
model.to(device).eval()

runner = evc.build_pretrain_caption_val_runner(config, device, writer=None)
metrics = runner.run_epoch_end(model, epoch=0, global_step=999999)
print("METRICS:", {k: round(float(v), 4) for k, v in metrics.items()})
assert "CIDEr" in metrics and "SPICE" in metrics
print("OK")
```

- [ ] **Step 2: 실행** (모든 GPU가 비어있을 때만. `nvidia-smi`로 먼저 확인)

Run: `cd /home/minwoo/Distillation_Project && /home/minwoo/miniconda3/envs/kd_r4/bin/python scratchpad/smoke_caption_runner.py`
Expected: `METRICS: {... 'CIDEr': ~1.06, 'METEOR': ~0.27, 'ROUGE_L': ~0.55, 'SPICE': ~0.20, ...}` 그리고 `OK`. (baseline ep19 zero-shot 기존 값과 일치해야 함)

- [ ] **Step 3: 코어 캡 동작 확인 (선택)** — 실행 중 다른 터미널에서 `top`/`htop`으로 java(SPICE)가 `caption_score_cpu_list` 코어(0–31)에만 붙는지 눈으로 확인.

- [ ] **Step 4: 검증 노트 기록** — 결과(지표 값, 소요시간)를 `docs/superpowers/plans/2026-07-13-caption-metric-eval.md` 하단 "검증 결과"에 한두 줄로 남기고, 문제 없으면 완료.

---

## Self-Review

**Spec coverage:**
- 목표1(생성+채점+TB) → Task 1·4·5. 목표2(인프라 재사용) → Task 4(`coco_caption_eval`/runner 패턴). 목표3(SPICE cadence+코어캡) → Task 3(캡)·Task 4(cadence)·Task 5(config). 목표4(에폭 실경과) → Task 2. ✅
- 컴포넌트1 generate → Task 1. 컴포넌트2 runner → Task 4. 컴포넌트3 배선 → Task 5. 컴포넌트4 elapsed → Task 2. 컴포넌트5 코어캡 → Task 3+Task 4(`_score`에서 `cpu_affinity` 적용). ✅
- config 키 전부 Task 5 Step 7에 포함. TDD 테스트: generate(T1), cadence+SPICE 라우팅(T4, mock으로 실제 SPICE 미실행), 코어캡(T3), elapsed(T2). 실제 SPICE는 Task 6 e2e에서 검증. ✅

**Placeholder scan:** 모든 코드 step에 실제 코드 포함. "적절히 처리" 류 없음. ✅

**Type consistency:** `generate(...)` 시그니처(T1) = runner `_generate`의 호출 인자(T4) 일치. `coco_caption_eval(...use_spice=)` 키워드 = 테스트 assert(`call_args.kwargs["use_spice"]`) 일치. `build_pretrain_caption_val_runner`/`CaptionValRunner`/`build_caption_val_loader` 이름이 T4 정의와 T5 배선에서 동일. `parse_cpu_list`/`cpu_affinity` (T3) = runner에서 사용(T4) 일치. ✅

## 검증 결과

(Task 6 실행 후 기록)
