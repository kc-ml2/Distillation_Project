# ITM B×B 매트릭스 증류 — Phase 1 (full B×B) 구현 플랜

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 학생·티처의 전체 B×B(40×40) cross-attn ITM 매치-로짓 매트릭스를 row(i2t)/col(t2i) softmax forward-KL로 증류하는 손실을 학습 루프에 추가한다(config-gated, 기본 OFF).

**Architecture:** ITC KD의 ITM 미러. 셀당 헤드 raw 로짓 `[z1,z2]`로 `[B,B,2]` 매트릭스를 만들고(행 i=image i vs 전 텍스트), 채널별(z1=reverse/z2=forward)·축별(row/col) softmax forward-KL를 평균. 기존 3B 하드라벨 ITM CE는 100% 불변, 순수 추가 KD 항. Phase 1은 **full B×B만**(서브샘플·union은 스펙에 설계로만, 측정 게이트).

**Tech Stack:** PyTorch, HuggingFace BERT(멀티모달 text_encoder), BLIP pretrain 코드베이스, unittest.

**선행 스펙:** `docs/superpowers/specs/2026-08-07-itm-bxb-matrix-kd-design.md`

## Global Constraints

- **itm_head는 fp32에서만.** bf16이면 gap 오차 8000배(프로브 §7). 매트릭스 헬퍼가 `torch.autocast(enabled=False)` + `.float()`로 강제.
- **KL 방향 = forward `KL(teacher‖student)`**, 정규화 `×τ`(itc_distill_loss와 동일), KL 항 평균. 티처 입력은 `.detach()`.
- **채널 인덱스:** `itm_head` 출력 index 0 = z1(no-match/reverse), 1 = z2(match/forward).
- **기본 OFF:** `distill.itm.enabled=false`면 baseline과 bit-identical(경로 진입 안 함, 반환은 `loss_itm_kd=None`).
- **티처는 no_grad, 학생만 grad.** 티처는 학생과 동일 augmented 배치·동일 토크나이즈 규칙(enc_token_id).
- CPU 스모크 테스트는 `vit="base", bert="base", image_size=224`로 구성(기존 테스트 관례).

---

## 파일 구조

| 파일 | 책임 | 작업 |
|---|---|---|
| `distillation/losses.py` | `itm_matrix_kd_loss` 순수 손실함수 | 수정(추가) |
| `distillation/itm_matrix.py` | `itm_bxb_logits` — B×B 매트릭스 조립(학생·티처 공용) | **신규** |
| `distillation/online_teacher.py` | `OnlineTeacher.itm_matrix` + `NEEDS['itm']` | 수정 |
| `models/blip_pretrain.py` | forward에 학생 매트릭스+손실 배선, 반환 6-튜플 | 수정 |
| `data/eval_validation_loss.py` | forward 반환 언팩 5→6 | 수정 |
| `pretrain.py` | config 게이트·티처 호출·손실합산·TB·teacher_keep | 수정 |
| `configs/pretrain_itm_matrix_kd.yaml` | 실행 config | 신규 |
| `distillation/test_losses.py` | 손실 단위테스트 | 수정 |
| `distillation/test_itm_matrix.py` | 매트릭스 조립 테스트 | 신규 |
| `distillation/test_online_teacher.py` | keep=itm + itm_matrix 계약 | 수정 |
| `distillation/test_forward_kd.py` | 6-튜플 + itm_kd + end-to-end | 수정 |

---

### Task 1: `itm_matrix_kd_loss` 손실함수

**Files:**
- Modify: `distillation/losses.py` (파일 끝에 함수 추가)
- Test: `distillation/test_losses.py` (클래스 추가)

**Interfaces:**
- Produces: `itm_matrix_kd_loss(student_logits, teacher_logits, direction, temp) -> Tensor(scalar)`.
  - `student_logits`, `teacher_logits`: `[B, B, 2]` (index 0=z1, 1=z2). row i=image i, col j=text j.
  - `direction`: `'forward'`(z2) | `'reverse'`(z1) | `'bidir'`(둘).
  - `temp`: float τ.

- [ ] **Step 1: 실패 테스트 작성** — `distillation/test_losses.py` 상단 import에 `itm_matrix_kd_loss` 추가하고 클래스 append:

```python
class TestItmMatrixKdLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B = 5
        self.temp = 0.05

    def test_identical_is_zero(self):
        t = torch.randn(self.B, self.B, 2)
        s = t.clone().requires_grad_(True)
        self.assertLess(itm_matrix_kd_loss(s, t, "bidir", self.temp).item(), 1e-6)

    def test_forward_uses_only_z2_channel(self):
        base = torch.randn(self.B, self.B, 2)
        s, t = base.clone(), base.clone()
        s[:, :, 0] += torch.randn(self.B, self.B)          # z1(reverse)만 교란
        self.assertLess(itm_matrix_kd_loss(s, t, "forward", self.temp).item(), 1e-6)
        self.assertGreater(itm_matrix_kd_loss(s, t, "reverse", self.temp).item(), 1e-6)

    def test_reverse_uses_only_z1_channel(self):
        base = torch.randn(self.B, self.B, 2)
        s, t = base.clone(), base.clone()
        s[:, :, 1] += torch.randn(self.B, self.B)          # z2(forward)만 교란
        self.assertLess(itm_matrix_kd_loss(s, t, "reverse", self.temp).item(), 1e-6)
        self.assertGreater(itm_matrix_kd_loss(s, t, "forward", self.temp).item(), 1e-6)

    def test_gradient_flows_to_student_not_teacher(self):
        s = torch.randn(self.B, self.B, 2, requires_grad=True)
        t = torch.randn(self.B, self.B, 2, requires_grad=True)
        itm_matrix_kd_loss(s, t, "bidir", self.temp).backward()
        self.assertIsNotNone(s.grad)
        self.assertIsNone(t.grad)                          # 내부에서 detach

    def test_invalid_direction_raises(self):
        t = torch.randn(self.B, self.B, 2)
        with self.assertRaises(ValueError):
            itm_matrix_kd_loss(t, t, "sideways", self.temp)
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest distillation/test_losses.py::TestItmMatrixKdLoss -v`
Expected: FAIL (`ImportError` 또는 `NameError: itm_matrix_kd_loss`)

- [ ] **Step 3: 최소 구현** — `distillation/losses.py` 끝에 추가:

```python
def itm_matrix_kd_loss(student_logits, teacher_logits, direction, temp):
    """Full B×B ITM 매치-로짓 매트릭스 관계 KD (ITC KD의 ITM 미러).

    student_logits, teacher_logits: [B, B, 2]. index 0=z1(no-match/reverse),
      1=z2(match/forward). row i = image i vs 전 텍스트, col j = text j vs 전 이미지.
    direction: 'forward'(z2) | 'reverse'(z1) | 'bidir'(둘).
    temp: 증류 온도 τ.
    반환: forward KL(T‖S) 평균 × τ. 채널별(선택)×축(row,col) KL을 평균.
    """
    channels = {"forward": [1], "reverse": [0], "bidir": [0, 1]}.get(direction)
    if channels is None:
        raise ValueError(f"direction must be forward/reverse/bidir, got {direction!r}")

    s = student_logits.float()
    t = teacher_logits.float().detach()

    total = 0.0
    n = 0
    for c in channels:
        ms, mt = s[:, :, c] / temp, t[:, :, c] / temp      # [B, B]
        # i2t: row 분포 (dim=1 = 텍스트 축)
        total = total + F.kl_div(F.log_softmax(ms, dim=1),
                                 F.softmax(mt, dim=1), reduction="batchmean")
        # t2i: col 분포 (transpose 후 dim=1 = 이미지 축)
        total = total + F.kl_div(F.log_softmax(ms.t(), dim=1),
                                 F.softmax(mt.t(), dim=1), reduction="batchmean")
        n += 2
    return (total / n) * temp
```

- [ ] **Step 4: 통과 확인**

Run: `python -m pytest distillation/test_losses.py::TestItmMatrixKdLoss -v`
Expected: 5 PASS

- [ ] **Step 5: 커밋**

```bash
git add distillation/losses.py distillation/test_losses.py
git commit -m "feat(itm-kd): itm_matrix_kd_loss (B×B row/col forward-KL, direction 토글)"
```

---

### Task 2: `itm_bxb_logits` 매트릭스 조립 헬퍼

**Files:**
- Create: `distillation/itm_matrix.py`
- Test: `distillation/test_itm_matrix.py`

**Interfaces:**
- Produces: `itm_bxb_logits(text_encoder, itm_head, image_embeds, image_atts, encoder_input_ids, text_atts) -> Tensor[B, B, 2]` (fp32).
  - `image_embeds`: `[B, L_img, D]`, `image_atts`: `[B, L_img]`, `encoder_input_ids`/`text_atts`: `[B, L_txt]`(enc_token_id는 **호출자가** 이미 설정).
  - 반환 row i = image i vs 배치 전체 텍스트. 학생·티처 공용(grad는 호출자 컨텍스트가 결정).

- [ ] **Step 1: 실패 테스트 작성** — `distillation/test_itm_matrix.py`:

```python
import unittest
from types import SimpleNamespace
import torch
import torch.nn as nn

from distillation.itm_matrix import itm_bxb_logits


class FakeEncoder(nn.Module):
    """CLS[b] = [image-marker, text-marker] — (i,j) 조립 순서를 검증하기 위한 가짜."""
    def forward(self, input_ids, attention_mask, encoder_hidden_states,
                encoder_attention_mask, return_dict):
        B, L = input_ids.shape
        hidden = torch.zeros(B, L, 2)
        hidden[:, 0, 0] = encoder_hidden_states[:, 0, 0]   # image marker (row 내 상수)
        hidden[:, 0, 1] = input_ids[:, 1].float()          # text marker (col 따라 변화)
        return SimpleNamespace(last_hidden_state=hidden)


class TestItmBxbLogits(unittest.TestCase):
    def test_matrix_assembly_row_image_col_text(self):
        B = 4
        image_embeds = torch.zeros(B, 10, 8)
        for i in range(B):
            image_embeds[i, 0, 0] = i                      # image i 표식
        image_atts = torch.ones(B, 10, dtype=torch.long)
        input_ids = torch.zeros(B, 6, dtype=torch.long)
        for j in range(B):
            input_ids[j, 1] = j                            # text j 표식
        text_atts = torch.ones(B, 6, dtype=torch.long)

        logits = itm_bxb_logits(FakeEncoder(), nn.Identity(),
                                image_embeds, image_atts, input_ids, text_atts)

        self.assertEqual(tuple(logits.shape), (B, B, 2))
        self.assertEqual(logits.dtype, torch.float32)
        for i in range(B):
            for j in range(B):
                self.assertAlmostEqual(logits[i, j, 0].item(), i)   # 행 = image i
                self.assertAlmostEqual(logits[i, j, 1].item(), j)   # 열 = text j
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest distillation/test_itm_matrix.py -v`
Expected: FAIL (`ModuleNotFoundError: distillation.itm_matrix`)

- [ ] **Step 3: 최소 구현** — `distillation/itm_matrix.py`:

```python
import torch


def itm_bxb_logits(text_encoder, itm_head, image_embeds, image_atts,
                   encoder_input_ids, text_atts):
    """전체 B×B ITM 매치-로짓 매트릭스. row i = image i vs 배치 전체 텍스트.

    반환 [B, B, 2] fp32. text_encoder는 호출자 autocast(bf16 가능)에서 돌지만
    itm_head는 autocast 밖 fp32로 강제(bf16이면 gap 오차 폭증 — 프로브 §7).
    row별 chunk(image i를 B 텍스트에 repeat)라 메모리는 배치당 B 시퀀스로 유계.
    enc_token_id는 호출자가 encoder_input_ids에 이미 설정한 상태로 받는다.
    """
    B = image_embeds.size(0)
    cls_rows = []
    for i in range(B):
        enc_hidden = image_embeds[i:i + 1].repeat(B, 1, 1)     # [B, L_img, D]
        enc_att = image_atts[i:i + 1].repeat(B, 1)             # [B, L_img]
        out = text_encoder(encoder_input_ids,
                           attention_mask=text_atts,
                           encoder_hidden_states=enc_hidden,
                           encoder_attention_mask=enc_att,
                           return_dict=True)
        cls_rows.append(out.last_hidden_state[:, 0, :])        # [B, D]
    cls = torch.stack(cls_rows, dim=0)                         # [B, B, D]
    with torch.autocast(device_type=image_embeds.device.type, enabled=False):
        logits = itm_head(cls.float())                         # [B, B, 2] fp32
    return logits
```

- [ ] **Step 4: 통과 확인**

Run: `python -m pytest distillation/test_itm_matrix.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add distillation/itm_matrix.py distillation/test_itm_matrix.py
git commit -m "feat(itm-kd): itm_bxb_logits — B×B 매트릭스 조립(row=image/col=text, fp32 head)"
```

---

### Task 3: `OnlineTeacher.itm_matrix` + `keep=('itm',)`

**Files:**
- Modify: `distillation/online_teacher.py`
- Test: `distillation/test_online_teacher.py` (클래스 추가)

**Interfaces:**
- Consumes: `itm_bxb_logits` (Task 2).
- Produces: `OnlineTeacher.itm_matrix(image, caption) -> Tensor[B, B, 2]` (fp32, no_grad). `NEEDS['itm']`, `CRITICAL_PREFIXES['itm']`.

- [ ] **Step 1: 실패 테스트 작성** — `distillation/test_online_teacher.py` 끝에 추가:

```python
class TestOnlineTeacherKeepItm(unittest.TestCase):
    """keep=('itm',): visual_encoder + text_encoder + itm_head만 생존."""

    @classmethod
    def setUpClass(cls):
        cls.teacher = OnlineTeacher(checkpoint="", image_size=224, vit="base",
                                    bert="base", queue_size=240, keep=("itm",))

    def test_kept_and_freed(self):
        m = self.teacher.model
        self.assertIsNotNone(m.visual_encoder)
        self.assertIsNotNone(m.text_encoder)
        self.assertIsNotNone(m.itm_head)
        for attr in ("text_decoder", "vision_proj", "text_proj",
                     "visual_encoder_m", "text_encoder_m", "vision_proj_m", "text_proj_m"):
            self.assertIsNone(getattr(m, attr), f"{attr} should be freed")

    def test_itm_matrix_contract(self):
        image = torch.randn(3, 3, 224, 224)
        caption = ["a green field", "a red car", "a blue sky"]
        M = self.teacher.itm_matrix(image, caption)
        self.assertEqual(tuple(M.shape), (3, 3, 2))
        self.assertEqual(M.dtype, torch.float32)
        self.assertFalse(M.requires_grad)

    def test_itm_matrix_raises_without_itm_keep(self):
        itc_teacher = OnlineTeacher(checkpoint="", image_size=224, vit="base",
                                    bert="base", queue_size=240, keep=("itc",))
        with self.assertRaises(RuntimeError):
            itc_teacher.itm_matrix(torch.randn(1, 3, 224, 224), ["a cat"])
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest distillation/test_online_teacher.py::TestOnlineTeacherKeepItm -v`
Expected: FAIL (`ValueError: unknown keep paths: ['itm']`)

- [ ] **Step 3: 구현** — `distillation/online_teacher.py`:

3a. 상단 import에 추가:
```python
from distillation.itm_matrix import itm_bxb_logits
```

3b. `NEEDS` 딕셔너리에 항목 추가:
```python
    "itm": {"visual_encoder", "text_encoder", "itm_head"},
```

3c. `CRITICAL_PREFIXES`에 항목 추가:
```python
    "itm": ("visual_encoder.", "text_encoder.", "itm_head."),
```

3d. `lm_logits` 메서드 아래에 추가:
```python
    @torch.no_grad()
    def itm_matrix(self, image, caption):
        """전체 B×B ITM 매치-로짓 매트릭스 [B,B,2] fp32. 학생 ITM 조립과 동일 규칙
        (enc_token_id). itm_head는 fp32(itm_bxb_logits 내부 강제)."""
        self._require("itm")
        device = image.device
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            image_embeds = self.model.visual_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
            text = self.tokenizer(caption, padding="max_length", truncation=True,
                                  max_length=30, return_tensors="pt").to(device)
            encoder_input_ids = text.input_ids.clone()
            encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
            logits = itm_bxb_logits(self.model.text_encoder, self.model.itm_head,
                                    image_embeds, image_atts, encoder_input_ids, text.attention_mask)
        return logits
```

- [ ] **Step 4: 통과 확인**

Run: `python -m pytest distillation/test_online_teacher.py::TestOnlineTeacherKeepItm -v`
Expected: 3 PASS (구성 ~1-2분)

- [ ] **Step 5: 커밋**

```bash
git add distillation/online_teacher.py distillation/test_online_teacher.py
git commit -m "feat(itm-kd): OnlineTeacher.itm_matrix + keep=('itm',)"
```

---

### Task 4: `blip_pretrain.forward` 배선 + 반환 5→6 언팩 수정

**Files:**
- Modify: `models/blip_pretrain.py` (import·시그니처·손실·반환)
- Modify: `data/eval_validation_loss.py:297` (언팩 6)
- Test: `distillation/test_forward_kd.py` (6-튜플 + itm_kd + end-to-end)

**Interfaces:**
- Consumes: `itm_bxb_logits`(Task 2), `itm_matrix_kd_loss`(Task 1).
- Produces: `blip_pretrain.forward(...)` 새 인자 `teacher_itm_logits=None, itm_distill_temp=0.05, itm_distill_direction='bidir'`; 반환 `(loss_ita, loss_itm, loss_lm, loss_itc_kd, loss_lm_kd, loss_itm_kd)`(6-튜플).

- [ ] **Step 1: 실패 테스트 작성** — `distillation/test_forward_kd.py`의 `test_five_tuple_with_nones_when_no_teacher`를 아래로 교체 + 신규 2개 추가:

```python
    def test_six_tuple_with_nones_when_no_teacher(self):
        out = self.model(self.image, self.caption, alpha=0.4, update_train_state=False)
        self.assertEqual(len(out), 6)
        loss_ita, loss_itm, loss_lm, loss_itc_kd, loss_lm_kd, loss_itm_kd = out
        for l in (loss_ita, loss_itm, loss_lm):
            self.assertTrue(torch.isfinite(l).all())
        self.assertIsNone(loss_itc_kd)
        self.assertIsNone(loss_lm_kd)
        self.assertIsNone(loss_itm_kd)

    def test_itm_kd_computed_when_teacher_matrix_given(self):
        B = 2
        teacher_itm = torch.randn(B, B, 2)
        out = self.model(self.image, self.caption, alpha=0.4, update_train_state=False,
                         teacher_itm_logits=teacher_itm, itm_distill_temp=0.05,
                         itm_distill_direction="bidir")
        loss_itm_kd = out[5]
        self.assertIsNotNone(loss_itm_kd)
        self.assertTrue(torch.isfinite(loss_itm_kd))

    def test_itm_kd_end_to_end_with_teacher(self):
        from distillation.online_teacher import OnlineTeacher
        teacher = OnlineTeacher(checkpoint="", image_size=224, vit="base",
                                bert="base", queue_size=240, keep=("itm",))
        t_itm = teacher.itm_matrix(self.image, self.caption)
        out = self.model(self.image, self.caption, alpha=0.4, update_train_state=False,
                         teacher_itm_logits=t_itm, itm_distill_direction="bidir")
        loss = out[0] + out[1] + out[2] + 1.0 * out[5]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest distillation/test_forward_kd.py -v`
Expected: FAIL (반환 5-튜플이라 `len(out)==6` 및 언팩 실패)

- [ ] **Step 3: 구현**

3a. `models/blip_pretrain.py:22` import 확장:
```python
from distillation.losses import itc_distill_loss, lm_distill_loss, itm_matrix_kd_loss
from distillation.itm_matrix import itm_bxb_logits
```

3b. forward 시그니처(347-349)에 인자 추가:
```python
    def forward(self, image, caption, alpha, update_train_state=None,
                teacher_img_feat=None, teacher_text_feat=None, distill_temp=0.05,
                teacher_lm_logits=None, teacher_lm_input_ids=None, lm_distill_temp=2.0,
                teacher_itm_logits=None, itm_distill_temp=0.05, itm_distill_direction='bidir'):
```

3c. `loss_lm_kd` 계산 블록 직후·`return` 직전(현재 525 위)에 삽입:
```python
        # external-teacher ITM matrix distillation (full B×B relational KL); None when disabled.
        loss_itm_kd = None
        if teacher_itm_logits is not None:
            student_itm_matrix = itm_bxb_logits(
                self.text_encoder, self.itm_head,
                image_embeds, image_atts, encoder_input_ids, text.attention_mask)
            loss_itm_kd = itm_matrix_kd_loss(
                student_itm_matrix, teacher_itm_logits.to(image.device),
                itm_distill_direction, itm_distill_temp)
```

3d. 반환문(525) 교체:
```python
        return loss_ita, loss_itm, loss_lm, loss_itc_kd, loss_lm_kd, loss_itm_kd
```

3e. `data/eval_validation_loss.py:297` 언팩을 6-튜플로:
```python
                    loss_ita, loss_itm, loss_lm, _loss_itc_kd, _loss_lm_kd, _loss_itm_kd = model(
```

- [ ] **Step 4: 통과 확인**

Run: `python -m pytest distillation/test_forward_kd.py -v`
Expected: 4 PASS (기존 lm_kd 테스트 포함)

- [ ] **Step 5: 커밋**

```bash
git add models/blip_pretrain.py data/eval_validation_loss.py distillation/test_forward_kd.py
git commit -m "feat(itm-kd): forward에 학생 B×B 매트릭스+손실 배선, 반환 6-튜플"
```

---

### Task 5: `pretrain.py` 배선 (config·티처·손실합산·TB·keep)

**Files:**
- Modify: `pretrain.py`

**Interfaces:**
- Consumes: `OnlineTeacher.itm_matrix`(Task 3), forward 6-튜플·새 인자(Task 4).
- 자동 단위테스트 없음(학습 루프). 검증 = 구문 컴파일 + 기존 스위트 무회귀 + Task 6 end-to-end + 실학습 스모크(GPU, 수동).

- [ ] **Step 1: config 파싱 추가** — `train()`의 lm 파싱(89-92) 아래:
```python
    distill_itm = config.get('distill', {}).get('itm', {})
    itm_kd_enabled = distill_itm.get('enabled', False)
    itm_kd_weight = float(distill_itm.get('weight', 1.0))
    itm_kd_temp = float(distill_itm.get('temp', 0.05))
    itm_kd_direction = distill_itm.get('direction', 'bidir')
```

- [ ] **Step 2: 티처 매트릭스 호출 추가** — lm_logits 블록(127-130) 아래:
```python
        # online teacher: same augmented batch -> ITM B×B matrix (bf16, no_grad). None when disabled.
        if itm_kd_enabled and online_teacher is not None:
            teacher_itm_logits = online_teacher.itm_matrix(image, caption)
        else:
            teacher_itm_logits = None
```

- [ ] **Step 3: model() 언팩·손실합산 (cuda·cpu 두 분기 모두)** — 141행 및 153행 블록을 각각 교체. **cuda 분기(139-150):**
```python
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss_ita, loss_itm, loss_lm, loss_itc_kd, loss_lm_kd, loss_itm_kd = model(
                    image, caption, alpha=alpha,
                    teacher_img_feat=teacher_img_feat, teacher_text_feat=teacher_text_feat,
                    distill_temp=itc_kd_temp,
                    teacher_lm_logits=teacher_lm_logits, teacher_lm_input_ids=teacher_lm_ids,
                    lm_distill_temp=lm_kd_temp,
                    teacher_itm_logits=teacher_itm_logits, itm_distill_temp=itm_kd_temp,
                    itm_distill_direction=itm_kd_direction)
                loss = loss_ita + loss_itm + loss_lm
                if itc_kd_enabled and loss_itc_kd is not None:
                    loss = loss + itc_kd_weight * loss_itc_kd
                if lm_kd_enabled and loss_lm_kd is not None:
                    loss = loss + lm_kd_weight * loss_lm_kd
                if itm_kd_enabled and loss_itm_kd is not None:
                    loss = loss + itm_kd_weight * loss_itm_kd
```
**cpu 분기(152-162):** 동일하게 언팩에 `loss_itm_kd` 추가, model() 호출에 세 인자 추가, `itm_kd` 합산 3줄 추가.

- [ ] **Step 4: metric_logger + TB 추가** — metric_logger의 lm_kd(171-172) 아래:
```python
        if loss_itm_kd is not None:
            metric_logger.update(loss_itm_kd=loss_itm_kd.item())
```
TB의 lm_kd(188-189) 아래:
```python
                if loss_itm_kd is not None:
                    writer.add_scalar("loss_train/itm_kd", loss_itm_kd.item(), global_step)
```

- [ ] **Step 5: teacher_keep + run-name 태그** —
5a. `teacher_keep`(400) 확장:
```python
    teacher_keep = tuple(k for k in ('itc', 'lm', 'itm') if distill_cfg.get(k, {}).get('enabled', False))
```
5b. run-name kd 태그(74) `("itc", "lm")` → `("itc", "lm", "itm")`.

- [ ] **Step 6: 구문·무회귀 확인**

Run: `python -m py_compile pretrain.py && python -m pytest distillation/ -q`
Expected: 컴파일 OK, distillation 스위트 전부 PASS

- [ ] **Step 7: 커밋**

```bash
git add pretrain.py
git commit -m "feat(itm-kd): pretrain 배선 — config 게이트·티처 매트릭스·손실합산·TB·keep"
```

---

### Task 6: config 파일 + end-to-end 스모크

**Files:**
- Create: `configs/pretrain_itm_matrix_kd.yaml`

**Interfaces:**
- Consumes: 전 태스크 통합.

- [ ] **Step 1: config 작성** — `configs/pretrain_student.yaml`을 복사해 이름 변경 후, `teacher:` 블록(exact 키는 `configs/pretrain_lm_distill.yaml`의 teacher 블록에서 그대로 복사)과 아래 distill 블록 추가:

```yaml
distill:
  itm:
    enabled: true
    weight: 1.0
    temp: 0.05
    direction: bidir      # {forward, reverse, bidir}
```
(itc/lm 증류를 같이 켜려면 각 블록도 추가. Phase 1 검증은 itm 단독으로 시작.)

- [ ] **Step 2: end-to-end 통합 테스트 재확인** (Task 4 Step 1의 `test_itm_kd_end_to_end_with_teacher`가 이미 티처→학생→손실→backward 전 경로를 커버):

Run: `python -m pytest distillation/test_forward_kd.py::TestForwardLmKd::test_itm_kd_end_to_end_with_teacher -v`
Expected: PASS

- [ ] **Step 3: 전체 스위트 무회귀**

Run: `python -m pytest distillation/ tests/ -q`
Expected: 전부 PASS(기존 포함)

- [ ] **Step 4: 실학습 스모크 (GPU, 수동 — 메모리/step-time 실측)**

Run: 스페어/본 서버에서 `configs/pretrain_itm_matrix_kd.yaml`로 pretrain을 **수 스텝**만 돌려 확인:
- OOM 없이 backward 통과하는가(학생 1600 시퀀스 activation) → OOM이면 **Phase 2(서브샘플)** 착수 신호.
- step time 상승폭(티처+학생 각 1600 forward) 허용 범위인가.
- TB `loss_train/itm_kd`가 유한하고 `itm_kd/itm` 비율이 CE를 삼키지 않는가(삼키면 `weight`↓).

- [ ] **Step 5: 커밋**

```bash
git add configs/pretrain_itm_matrix_kd.yaml
git commit -m "feat(itm-kd): pretrain_itm_matrix_kd config + end-to-end 스모크"
```

---

## Phase 게이트 (이 플랜 범위 밖 — 측정 후 별도 플랜)

- **Phase 2 (서브샘플, fixed 2-gather):** Task 6 Step 4에서 OOM 또는 iterate 불가 시. `topk` config + 방향별 top-k gather(k+1 불변식). 스펙 §4.2-4.3.
- **Phase 3 (union-dedup):** 프로파일에서 Phase 2 forward가 병목일 때만. 스펙 §4.3.

## 검증 요약 (성공 기준 — 스펙 §7, 학습 후)

`itm_sharpness_probe.py`로 separation 유지 + `val_retrieval_itm − val_retrieval_itc > 0` 회복(arm C −1.2 / scheme A ≈0 대비). direction {forward,reverse,bidir}·temp 어블레이션.
