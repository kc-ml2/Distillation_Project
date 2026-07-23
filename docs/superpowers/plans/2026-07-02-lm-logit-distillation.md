# LM Logit Distillation 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** frozen BLIP-large 티처의 teacher-forced 디코더 로짓을 학생(DINOv3 small_reg + MiniLM) 디코더 로짓에 token-level KL로 증류하는 학습 경로를 추가한다.

**Architecture:** 기존 ITC KD 시임을 그대로 확장 — 티처는 train loop에서 rank별 복제(frozen, DDP 미래핑)로 실행하고 로짓 텐서만 `model.forward()`에 전달, KD loss는 forward 안에서 계산해 `total = ita + itm + lm + w·lm_kd`로 가산. `OnlineTeacher`는 keep 인자로 유지할 서브모듈을 선택(LM 런: `visual_encoder + text_decoder`만).

**Tech Stack:** PyTorch(DDP, bf16 autocast), transformers(BertLMHeadModel), unittest.

**스펙:** `/home/minwoo/Distillation_Project/docs/superpowers/specs/2026-07-02-lm-logit-distillation-design.md` (dev 워크트리, gitignored — 절대 경로로 참조)

## Global Constraints

- **작업 위치:** Task 0에서 만드는 새 워크트리 `/home/minwoo/Distillation_Project_lm_distill` (브랜치 `lm_distill`, base=`itc_distill` tip `a382986`). Task 1부터 모든 명령은 이 디렉토리에서 실행.
- **커밋 정책:** `lm_distill`은 사용자 브랜치 — 실행 세션 시작 시 "태스크별 커밋 일괄 허용" 여부를 사용자에게 한 번 확인받고 진행 (프로젝트 정책: dev/main급 브랜치는 커밋 전 확인).
- **테스트 러너:** `python -m unittest` (pytest 아님). CPU 테스트는 캐시된 deit-base/bert-base 가중치를 사용하며 테스트 클래스당 모델 구성에 ~1-2분 소요.
- **회귀 금지:** 기존 ITC KD 동작 불변 — `OnlineTeacher`의 keep 기본값은 `('itc',)`, 기존 `distillation.test_losses`/`test_online_teacher` 테스트 전부 계속 통과해야 함.
- **네이밍(스펙 §5.6):** config 키는 `distill.lm.{enabled, weight, temp}` (기본 weight 1.0, temp 2.0), TB 스칼라는 `loss_train/lm_kd`, forward 반환은 5-tuple `(loss_ita, loss_itm, loss_lm, loss_itc_kd, loss_lm_kd)`.

## 파일 구조 (전체 변경 지도)

| 파일 | 작업 | 책임 |
|---|---|---|
| `configs/pretrain_student.yaml` | 신규 추적 (itc 워크트리에서 복사) | 학생 baseline config (distill OFF) |
| `distillation/losses.py` | 수정 | `lm_distill_loss` 추가 (Task 1) |
| `distillation/test_losses.py` | 수정 | `TestLmDistillLoss` 추가 (Task 1) |
| `distillation/online_teacher.py` | 수정 | keep 인자 + `lm_logits()` (Task 2, 3) |
| `distillation/test_online_teacher.py` | 수정 | keep 조합/lm_logits 테스트 (Task 2, 3) |
| `distillation/test_forward_kd.py` | 신규 | forward 5-tuple CPU 스모크 (Task 4) |
| `models/blip_pretrain.py` | 수정 | forward 확장 + 5-tuple 반환 (Task 4) |
| `data/eval_validation_loss.py` | 수정 | 5-tuple 언팩 (Task 4) |
| `pretrain.py` | 수정 | config·티처 호출·합산·TB·run name (Task 5) |
| `configs/pretrain_lm_distill.yaml` | 신규 | LM-KD 런 config (Task 6) |

---

### Task 0: 워크트리·브랜치 셋업 + 학생 config 추적

**Files:**
- Create: 워크트리 `/home/minwoo/Distillation_Project_lm_distill` (브랜치 `lm_distill`)
- Create: `configs/pretrain_student.yaml` (itc 워크트리의 미추적 파일을 복사해 추적 시작)

**Interfaces:**
- Produces: 이후 모든 태스크의 작업 디렉토리. `pretrain_student.yaml`은 Task 6의 config 원본.

- [ ] **Step 1: 워크트리 + 브랜치 생성**

```bash
cd /home/minwoo/Distillation_Project
git worktree add ../Distillation_Project_lm_distill -b lm_distill itc_distill
```

Expected: `Preparing worktree (new branch 'lm_distill')` … `HEAD is now at a382986 …`

- [ ] **Step 2: 베이스 확인 (itc_distill tip == dev 포함)**

```bash
cd /home/minwoo/Distillation_Project_lm_distill
git log --oneline -1        # a382986 이어야 함
git log --oneline lm_distill..dev | wc -l   # 0 이어야 함 (dev 전체 포함 확인)
```

- [ ] **Step 3: 미추적 학생 config 복사**

```bash
cp /home/minwoo/Distillation_Project_itc_distill/configs/pretrain_student.yaml \
   /home/minwoo/Distillation_Project_lm_distill/configs/pretrain_student.yaml
```

- [ ] **Step 4: 기존 테스트 스위트가 새 워크트리에서 도는지 베이스라인 확인**

```bash
cd /home/minwoo/Distillation_Project_lm_distill
python -m unittest distillation.test_losses -v
```

Expected: 기존 ITC 테스트 5개 전부 `ok`. (`test_online_teacher`는 모델 구성이 느리므로 여기서는 생략 — Task 2에서 실행)

- [ ] **Step 5: Commit**

```bash
git add configs/pretrain_student.yaml
git commit -m "config: track student baseline config (small_reg+minilm, from itc worktree)"
```

---

### Task 1: `lm_distill_loss` (token-level LM logit KD)

**Files:**
- Modify: `distillation/losses.py` (파일 끝에 함수 추가)
- Test: `distillation/test_losses.py` (클래스 추가)

**Interfaces:**
- Produces: `lm_distill_loss(student_logits, teacher_logits, decoder_targets, temp) -> 0-dim Tensor`
  - `student_logits`/`teacher_logits`: `[B, L, V]` (grad 필요/불필요), `decoder_targets`: `[B, L]` long, pad 위치 -100, `temp`: float 상수.
  - 의미: CE와 동일한 shift 프레임(위치 i 로짓 ↔ 토큰 i+1)에서 pad 제외 유효 토큰당 평균 `KL(teacher‖student) × T²`. Task 4의 forward가 호출.

- [ ] **Step 1: 실패하는 테스트 작성** — `distillation/test_losses.py`에 추가

import 줄 수정 (`itc_distill_loss`만 → 둘 다):

```python
from distillation.losses import itc_distill_loss, lm_distill_loss
```

파일 끝(`if __name__ == "__main__":` 위)에 클래스 추가:

```python
class TestLmDistillLoss(unittest.TestCase):
    """token-level LM logit KD. shift 프레임: 위치 i 로짓이 토큰 i+1 예측,
    유효 마스크는 decoder_targets[:, 1:] != -100 (CE와 동일 집합)."""

    def setUp(self):
        torch.manual_seed(0)
        self.B, self.L, self.V, self.temp = 3, 8, 50, 2.0
        # 샘플별 유효 길이 4/6/8, 나머지는 -100 (pad)
        self.targets = torch.full((self.B, self.L), -100, dtype=torch.long)
        for b, n in enumerate((4, 6, 8)):
            self.targets[b, :n] = torch.randint(0, self.V, (n,))

    def test_zero_when_teacher_equals_student(self):
        logits = torch.randn(self.B, self.L, self.V)
        s = logits.clone().requires_grad_(True)
        loss = lm_distill_loss(s, logits, self.targets, temp=1.0)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_positive_when_teacher_differs(self):
        s = torch.randn(self.B, self.L, self.V, requires_grad=True)
        t = torch.randn(self.B, self.L, self.V)
        loss = lm_distill_loss(s, t, self.targets, self.temp)
        self.assertGreater(loss.item(), 0.0)

    def test_matches_manual_reference(self):
        # 독립 레퍼런스: 유효 위치를 루프로 골라 KL을 직접 합산
        torch.manual_seed(1)
        s = torch.randn(self.B, self.L, self.V)
        t = torch.randn(self.B, self.L, self.V)
        T = self.temp
        total, n = 0.0, 0
        for b in range(self.B):
            for j in range(self.L - 1):                      # 위치 j 로짓 → 토큰 j+1 예측
                if self.targets[b, j + 1].item() == -100:    # 다음 토큰이 pad면 제외
                    continue
                p = F.softmax(t[b, j] / T, dim=-1)
                logq = F.log_softmax(s[b, j] / T, dim=-1)
                total += (p * (p.log() - logq)).sum().item()
                n += 1
        expected = total / n * T ** 2
        actual = lm_distill_loss(s, t, self.targets, T).item()
        self.assertAlmostEqual(actual, expected, places=4)

    def test_pad_positions_do_not_affect_loss(self):
        s = torch.randn(self.B, self.L, self.V)
        t = torch.randn(self.B, self.L, self.V)
        base = lm_distill_loss(s, t, self.targets, self.temp)
        # 미사용 로짓 위치 = 마지막 위치(L-1) + "다음 토큰이 pad"인 위치
        unused = torch.cat([self.targets[:, 1:] == -100,
                            torch.ones(self.B, 1, dtype=torch.bool)], dim=1)  # [B, L]
        s2, t2 = s.clone(), t.clone()
        s2[unused] += 100.0
        t2[unused] -= 100.0
        perturbed = lm_distill_loss(s2, t2, self.targets, self.temp)
        self.assertAlmostEqual(base.item(), perturbed.item(), places=5)

    def test_gradient_flows_to_student_not_teacher(self):
        s = torch.randn(self.B, self.L, self.V, requires_grad=True)
        t = torch.randn(self.B, self.L, self.V, requires_grad=True)  # detach가 막아야 함
        loss = lm_distill_loss(s, t, self.targets, self.temp)
        loss.backward()
        self.assertIsNotNone(s.grad)
        self.assertIsNone(t.grad)

    def test_normalized_per_valid_token(self):
        # 모든 위치가 같은 (s_row, t_row) 분포 쌍이면 유효 토큰 수와 무관하게 loss 동일해야 함
        torch.manual_seed(2)
        s_row = torch.randn(self.V)
        t_row = torch.randn(self.V)

        def loss_with(n_valid):
            targets = torch.full((1, self.L), -100, dtype=torch.long)
            targets[0, :n_valid] = 1
            s = s_row.expand(1, self.L, self.V)
            t = t_row.expand(1, self.L, self.V)
            return lm_distill_loss(s, t, targets, self.temp).item()

        self.assertAlmostEqual(loss_with(3), loss_with(7), places=5)

    def test_vocab_mismatch_raises(self):
        s = torch.randn(self.B, self.L, self.V)
        t = torch.randn(self.B, self.L, self.V + 1)
        with self.assertRaises(AssertionError):
            lm_distill_loss(s, t, self.targets, self.temp)

    def test_returns_scalar(self):
        s = torch.randn(self.B, self.L, self.V)
        t = torch.randn(self.B, self.L, self.V)
        self.assertEqual(lm_distill_loss(s, t, self.targets, self.temp).dim(), 0)
```

- [ ] **Step 2: 실패 확인**

```bash
python -m unittest distillation.test_losses -v
```

Expected: FAIL — `ImportError: cannot import name 'lm_distill_loss'`

- [ ] **Step 3: 최소 구현** — `distillation/losses.py` 끝에 추가

```python
def lm_distill_loss(student_logits, teacher_logits, decoder_targets, temp):
    """Token-level LM logit KD (teacher-forced).

    Args:
        student_logits: [B, L, V] student decoder logits, require grad.
        teacher_logits: [B, L, V] teacher decoder logits (constants).
        decoder_targets: [B, L] long, pad 위치는 -100 (BLIP LM CE와 동일 규칙).
        temp: 고정 증류 온도 T (config distill.lm.temp).

    Returns:
        Scalar = 유효 토큰당 평균 KL(teacher ‖ student) × T².
        CE와 동일한 shift 프레임: 위치 i 로짓이 토큰 i+1을 예측하므로
        마지막 위치를 버리고, targets를 한 칸 미뤄 유효 마스크를 만든다.
    """
    assert student_logits.shape[-1] == teacher_logits.shape[-1], (
        f"vocab size mismatch: student {student_logits.shape[-1]} "
        f"vs teacher {teacher_logits.shape[-1]}"
    )
    s = student_logits[:, :-1, :]
    t = teacher_logits[:, :-1, :].detach()
    valid = decoder_targets[:, 1:] != -100      # [B, L-1], CE가 학습하는 위치와 동일 집합
    s = s[valid]                                 # [N_valid, V]
    t = t[valid]
    return F.kl_div(
        F.log_softmax(s / temp, dim=-1),
        F.softmax(t / temp, dim=-1),
        reduction="batchmean",                   # N_valid로 나눔 = 유효 토큰당 평균
    ) * (temp ** 2)
```

- [ ] **Step 4: 통과 확인**

```bash
python -m unittest distillation.test_losses -v
```

Expected: 기존 5개 + 신규 8개 전부 `ok`

- [ ] **Step 5: Commit**

```bash
git add distillation/losses.py distillation/test_losses.py
git commit -m "feat(distill): lm_distill_loss - token-level LM logit KD (shift frame, pad mask, T^2)"
```

---

### Task 2: `OnlineTeacher` keep 리팩토링

**Files:**
- Modify: `distillation/online_teacher.py`
- Test: `distillation/test_online_teacher.py`

**Interfaces:**
- Consumes: 기존 `OnlineTeacher(checkpoint, image_size, vit, bert, queue_size)` + `itc_feats()`.
- Produces: `OnlineTeacher(..., keep=('itc',))` — keep은 `('itc',)`/`('lm',)`/`('itc','lm')` 튜플. keep에 없는 경로의 메서드 호출 시 RuntimeError. **기본값 `('itc',)`로 기존 콜사이트/테스트와 완전 호환.** Task 3이 `'lm'` 경로를, Task 5가 config 유도 keep을 사용.

- [ ] **Step 1: 실패하는 테스트 작성** — `distillation/test_online_teacher.py`에 클래스 2개 추가

기존 `TestOnlineTeacher` 클래스는 **그대로 두고** (기본 keep=('itc',) 회귀 검증 역할), 파일 끝(`if __name__ == "__main__":` 위)에 추가:

```python
class TestOnlineTeacherKeepLm(unittest.TestCase):
    """keep=('lm',): visual_encoder + text_decoder만 생존해야 한다."""

    @classmethod
    def setUpClass(cls):
        cls.teacher = OnlineTeacher(checkpoint="", image_size=224, vit="base",
                                    bert="base", queue_size=240, keep=("lm",))

    def test_kept_and_freed(self):
        m = self.teacher.model
        self.assertIsNotNone(m.visual_encoder)
        self.assertIsNotNone(m.text_decoder)
        for attr in ("text_encoder", "vision_proj", "text_proj",
                     "visual_encoder_m", "text_encoder_m", "vision_proj_m",
                     "text_proj_m", "itm_head"):
            self.assertIsNone(getattr(m, attr), f"{attr} should be freed")
        for buf in ("image_queue", "text_queue", "queue_ptr"):
            self.assertIsNone(getattr(m, buf), f"{buf} should be freed")

    def test_itc_feats_raises_without_itc_keep(self):
        with self.assertRaises(RuntimeError):
            self.teacher.itc_feats(torch.randn(1, 3, 224, 224), ["a cat"])


class TestOnlineTeacherKeepBoth(unittest.TestCase):
    """keep=('itc','lm'): 두 경로의 합집합 생존, 학습 전용 장치는 여전히 해제."""

    @classmethod
    def setUpClass(cls):
        cls.teacher = OnlineTeacher(checkpoint="", image_size=224, vit="base",
                                    bert="base", queue_size=240, keep=("itc", "lm"))

    def test_union_kept(self):
        m = self.teacher.model
        for attr in ("visual_encoder", "text_encoder", "vision_proj",
                     "text_proj", "text_decoder"):
            self.assertIsNotNone(getattr(m, attr), f"{attr} should be kept")
        for attr in ("visual_encoder_m", "text_encoder_m", "vision_proj_m",
                     "text_proj_m", "itm_head"):
            self.assertIsNone(getattr(m, attr), f"{attr} should be freed")

    def test_unknown_keep_raises(self):
        with self.assertRaises(ValueError):
            OnlineTeacher(checkpoint="", image_size=224, vit="base",
                          bert="base", queue_size=240, keep=("itm",))
```

주의: `test_unknown_keep_raises`는 keep 검증이 **모델 구성보다 먼저** 실행되어야 빠르게 실패한다 (구현에서 보장).

- [ ] **Step 2: 실패 확인**

```bash
python -m unittest distillation.test_online_teacher.TestOnlineTeacherKeepLm -v
```

Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'keep'`

- [ ] **Step 3: 구현** — `distillation/online_teacher.py` 전체를 아래로 교체

```python
import torch
import torch.nn.functional as F

from models.blip_pretrain import blip_pretrain

# 경로별 필요 서브모듈. momentum 4개·itm_head·큐는 학습 전용 장치라 어떤 keep에서도 해제.
NEEDS = {
    "itc": {"visual_encoder", "text_encoder", "vision_proj", "text_proj"},
    "lm": {"visual_encoder", "text_decoder"},
}
ALL_SUBMODULES = {
    "visual_encoder", "text_encoder", "vision_proj", "text_proj", "text_decoder",
    "visual_encoder_m", "text_encoder_m", "vision_proj_m", "text_proj_m", "itm_head",
}
CRITICAL_PREFIXES = {
    "itc": ("visual_encoder.", "text_encoder.", "vision_proj.", "text_proj."),
    "lm": ("visual_encoder.", "text_decoder."),
}


class OnlineTeacher:
    """Frozen BLIP teacher served live in the train loop, on the same augmented
    batch the student sees. keep이 지정한 경로의 서브모듈만 유지하고 나머지는
    로드 직후 해제해 티처 메모리를 최소화한다."""

    def __init__(self, checkpoint, image_size, vit="large", bert="base",
                 queue_size=240, keep=("itc",)):
        unknown = set(keep) - set(NEEDS)
        if unknown:
            raise ValueError(f"unknown keep paths: {sorted(unknown)} (choose from {sorted(NEEDS)})")
        self.keep = tuple(keep)

        model = blip_pretrain(image_size=image_size, vit=vit, my_bert_size=bert,
                              queue_size=queue_size)
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            msg = model.load_state_dict(state, strict=False)
            print("teacher load:", msg)
            critical_prefixes = tuple(p for k in self.keep for p in CRITICAL_PREFIXES[k])
            critical_missing = [k for k in msg.missing_keys if k.startswith(critical_prefixes)]
            if critical_missing:
                raise RuntimeError(
                    f"Teacher checkpoint is missing {len(critical_missing)} critical keys "
                    f"for keep={self.keep} (architecture mismatch vs vit='{vit}', bert='{bert}'?). "
                    f"First few: {critical_missing[:5]}"
                )

        # keep 합집합 외 서브모듈/버퍼 해제 (로드 후 → .to(device) 전이므로 GPU엔 안 올라감)
        needed = set().union(*(NEEDS[k] for k in self.keep))
        for attr in sorted(ALL_SUBMODULES - needed):
            setattr(model, attr, None)
        for buf in ("image_queue", "text_queue", "queue_ptr"):
            setattr(model, buf, None)

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        self.model = model
        self.tokenizer = model.tokenizer

    def to(self, device):
        self.model.to(device)
        return self

    def _require(self, path):
        if path not in self.keep:
            raise RuntimeError(
                f"OnlineTeacher was built with keep={self.keep}; '{path}' path is unavailable"
            )

    @torch.no_grad()
    def itc_feats(self, image, caption):
        self._require("itc")
        device = image.device
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            image_embeds = self.model.visual_encoder(image)
            img_feat = F.normalize(self.model.vision_proj(image_embeds[:, 0, :]), dim=-1)
            text = self.tokenizer(caption, padding="max_length", truncation=True,
                                  max_length=30, return_tensors="pt").to(device)
            text_output = self.model.text_encoder(text.input_ids,
                                                  attention_mask=text.attention_mask,
                                                  return_dict=True, mode="text")
            txt_feat = F.normalize(self.model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
        return img_feat, txt_feat
```

(원본 대비 변화: keep 검증/저장, critical prefix를 keep에서 유도, 해제 목록을 집합 연산으로, `_require` 게이트. `itc_feats` 본문은 불변.)

- [ ] **Step 4: 통과 + 회귀 확인** (모델 구성 3회로 수 분 소요)

```bash
python -m unittest distillation.test_online_teacher -v
```

Expected: 기존 `TestOnlineTeacher` 3개 + 신규 4개 전부 `ok`

- [ ] **Step 5: Commit**

```bash
git add distillation/online_teacher.py distillation/test_online_teacher.py
git commit -m "refactor(distill): OnlineTeacher keep arg - per-path submodule retention, itc default"
```

---

### Task 3: `OnlineTeacher.lm_logits()`

**Files:**
- Modify: `distillation/online_teacher.py` (메서드 추가)
- Test: `distillation/test_online_teacher.py` (Task 2의 클래스에 테스트 추가)

**Interfaces:**
- Consumes: Task 2의 keep 게이트 (`_require("lm")`).
- Produces: `lm_logits(image, caption) -> (logits [B,30,V] bf16 no-grad, decoder_input_ids [B,30] long)`. 학생과 동일 토크나이즈 규칙(max_length=30, padding='max_length', truncation, `[:,0]=bos`). Task 5의 train loop가 호출, 반환 텐서는 Task 4의 forward 인자.

- [ ] **Step 1: 실패하는 테스트 작성** — `TestOnlineTeacherKeepLm` 클래스에 메서드 추가

```python
    def test_lm_logits_contract(self):
        image = torch.randn(2, 3, 224, 224)
        caption = ["a green field", "a red car"]
        logits, dec_ids = self.teacher.lm_logits(image, caption)
        vocab = len(self.teacher.tokenizer)          # 30524
        self.assertEqual(tuple(logits.shape), (2, 30, vocab))
        self.assertEqual(tuple(dec_ids.shape), (2, 30))
        self.assertFalse(logits.requires_grad)
        self.assertTrue(torch.isfinite(logits.float()).all())
        # 첫 토큰은 BOS로 치환되어야 함 (학생의 decoder_input_ids 규칙과 동일)
        self.assertTrue((dec_ids[:, 0] == self.teacher.tokenizer.bos_token_id).all())
```

그리고 `TestOnlineTeacher`(기본 keep=('itc',)) 클래스에 게이트 테스트 추가:

```python
    def test_lm_logits_raises_without_lm_keep(self):
        with self.assertRaises(RuntimeError):
            self.teacher.lm_logits(torch.randn(1, 3, 224, 224), ["a cat"])
```

- [ ] **Step 2: 실패 확인**

```bash
python -m unittest distillation.test_online_teacher.TestOnlineTeacherKeepLm.test_lm_logits_contract -v
```

Expected: FAIL — `AttributeError: 'OnlineTeacher' object has no attribute 'lm_logits'`

- [ ] **Step 3: 구현** — `OnlineTeacher`에 메서드 추가 (`itc_feats` 아래)

```python
    @torch.no_grad()
    def lm_logits(self, image, caption):
        """Teacher-forced decoder logits for the same augmented batch.
        학생 forward의 LM 경로와 동일한 토크나이즈/BOS 규칙 — forward 쪽에서
        decoder_input_ids 일치를 assert하므로 규칙이 어긋나면 즉시 검출된다."""
        self._require("lm")
        device = image.device
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            image_embeds = self.model.visual_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
            text = self.tokenizer(caption, padding="max_length", truncation=True,
                                  max_length=30, return_tensors="pt").to(device)
            decoder_input_ids = text.input_ids.clone()
            decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
            out = self.model.text_decoder(decoder_input_ids,
                                          attention_mask=text.attention_mask,
                                          encoder_hidden_states=image_embeds,
                                          encoder_attention_mask=image_atts,
                                          return_dict=True)   # labels 없음 → logits만
        return out.logits, decoder_input_ids
```

- [ ] **Step 4: 통과 확인**

```bash
python -m unittest distillation.test_online_teacher -v
```

Expected: 전부 `ok` (신규 2개 포함)

- [ ] **Step 5: Commit**

```bash
git add distillation/online_teacher.py distillation/test_online_teacher.py
git commit -m "feat(distill): OnlineTeacher.lm_logits - teacher-forced decoder logits (bf16, no_grad)"
```

---

### Task 4: forward 5-tuple 확장 + validation 콜사이트

**Files:**
- Modify: `models/blip_pretrain.py` (import ~L22, forward 시그니처 ~L297, LM 섹션 끝 ~L455-464)
- Modify: `data/eval_validation_loss.py:297` (언팩)
- Test: Create `distillation/test_forward_kd.py`

**Interfaces:**
- Consumes: Task 1의 `lm_distill_loss`.
- Produces: `BLIP_Pretrain.forward(image, caption, alpha, update_train_state=None, teacher_img_feat=None, teacher_text_feat=None, distill_temp=0.05, teacher_lm_logits=None, teacher_lm_input_ids=None, lm_distill_temp=2.0) -> (loss_ita, loss_itm, loss_lm, loss_itc_kd, loss_lm_kd)` — KD 두 항은 해당 티처 인자 미제공 시 None. Task 5의 train loop가 이 시그니처로 호출.

- [ ] **Step 1: 실패하는 테스트 작성** — Create `distillation/test_forward_kd.py`

```python
import unittest
import torch

from models.blip_pretrain import blip_pretrain


class TestForwardLmKd(unittest.TestCase):
    """KD 확장 forward의 CPU 스모크 (small BLIP base/base, 구성 ~1-2분).
    update_train_state=False로 momentum/queue 갱신 경로(분산 필요)를 우회한다."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = blip_pretrain(image_size=224, vit="base", my_bert_size="base",
                                  queue_size=240)
        cls.model.eval()
        cls.image = torch.randn(2, 3, 224, 224)
        cls.caption = ["a green field", "a red car"]

    def _teacher_payload(self):
        # 학생과 동일 규칙으로 만든 가짜 티처 로짓/입력 (forward의 assert를 통과해야 함)
        tok = self.model.tokenizer
        text = tok(self.caption, padding="max_length", truncation=True,
                   max_length=30, return_tensors="pt")
        dec_ids = text.input_ids.clone()
        dec_ids[:, 0] = tok.bos_token_id
        return torch.randn(2, 30, len(tok)), dec_ids

    def test_five_tuple_with_nones_when_no_teacher(self):
        out = self.model(self.image, self.caption, alpha=0.4, update_train_state=False)
        self.assertEqual(len(out), 5)
        loss_ita, loss_itm, loss_lm, loss_itc_kd, loss_lm_kd = out
        for l in (loss_ita, loss_itm, loss_lm):
            self.assertTrue(torch.isfinite(l).all())
        self.assertIsNone(loss_itc_kd)
        self.assertIsNone(loss_lm_kd)

    def test_lm_kd_computed_when_teacher_logits_given(self):
        t_logits, t_ids = self._teacher_payload()
        out = self.model(self.image, self.caption, alpha=0.4, update_train_state=False,
                         teacher_lm_logits=t_logits, teacher_lm_input_ids=t_ids,
                         lm_distill_temp=2.0)
        loss_lm_kd = out[4]
        self.assertIsNotNone(loss_lm_kd)
        self.assertTrue(torch.isfinite(loss_lm_kd))
        self.assertTrue(loss_lm_kd.requires_grad)   # 학생 로짓 경유 grad

    def test_mismatched_teacher_ids_raise(self):
        t_logits, t_ids = self._teacher_payload()
        bad = t_ids.clone()
        bad[0, 1] = (bad[0, 1] + 1) % 30000
        with self.assertRaises(AssertionError):
            self.model(self.image, self.caption, alpha=0.4, update_train_state=False,
                       teacher_lm_logits=t_logits, teacher_lm_input_ids=bad)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실패 확인**

```bash
python -m unittest distillation.test_forward_kd -v
```

Expected: FAIL — `test_five_tuple_with_nones_when_no_teacher`에서 `len(out)` 4 ≠ 5 (`TypeError: forward() got an unexpected keyword argument 'teacher_lm_logits'` 포함)

- [ ] **Step 3: 구현** — `models/blip_pretrain.py` 3곳 수정

(a) import (~L22):

```python
from distillation.losses import itc_distill_loss, lm_distill_loss
```

(b) forward 시그니처 (~L297):

```python
    def forward(self, image, caption, alpha, update_train_state=None,
                teacher_img_feat=None, teacher_text_feat=None, distill_temp=0.05,
                teacher_lm_logits=None, teacher_lm_input_ids=None, lm_distill_temp=2.0):
```

(c) LM 섹션 끝 — 기존 `loss_itc_kd` 블록과 return(~L455-464) 사이에 삽입, return 교체:

```python
        # external-teacher LM logit distillation (token-level, teacher-forced); None when disabled.
        loss_lm_kd = None
        if teacher_lm_logits is not None:
            if teacher_lm_input_ids is not None:
                assert torch.equal(teacher_lm_input_ids, decoder_input_ids), \
                    "teacher/student decoder input mismatch (tokenizer drift?)"
            loss_lm_kd = lm_distill_loss(decoder_output.logits,
                                         teacher_lm_logits.to(image.device),
                                         decoder_targets, lm_distill_temp)

        return loss_ita, loss_itm, loss_lm, loss_itc_kd, loss_lm_kd
```

(d) `data/eval_validation_loss.py:297` 언팩 교체:

```python
                    loss_ita, loss_itm, loss_lm, _loss_itc_kd, _loss_lm_kd = model(
                        image,
                        caption,
                        alpha=alpha,
                        update_train_state=False,
                    )
```

- [ ] **Step 4: 통과 + 전체 회귀 확인**

```bash
python -m unittest distillation.test_forward_kd -v
python -m unittest distillation.test_losses -v
python -c "import data.eval_validation_loss; import pretrain" 2>&1 | tail -1
```

Expected: 테스트 전부 `ok`. 마지막 import 확인은 에러 없이 종료 (pretrain.py의 4-tuple 언팩은 Task 5에서 수정하므로 **이 시점에는 학습 실행 불가** — import만 확인).

주의: Task 4 완료 후 Task 5 완료 전까지는 forward(5-tuple)와 pretrain.py(4-tuple 언팩)가 불일치 상태다. 두 태스크를 연달아 실행할 것.

- [ ] **Step 5: Commit**

```bash
git add models/blip_pretrain.py data/eval_validation_loss.py distillation/test_forward_kd.py
git commit -m "feat(distill): forward returns loss_lm_kd (5-tuple), teacher input-ids assert; val unpack 5"
```

---

### Task 5: `pretrain.py` 배선

**Files:**
- Modify: `pretrain.py` — `make_tb_run_name` (~L58-68), `train()` 상단 config 읽기 (~L74-77), 티처 호출 (~L105-109), model 호출 2곳 (~L118-134), metric/TB 로깅 (~L139-157), `main()` 티처 생성 (~L346-356)

**Interfaces:**
- Consumes: Task 3의 `lm_logits()`, Task 4의 forward 시그니처, Task 2의 keep 인자.
- Produces: config 키 `distill.lm.{enabled, weight, temp}` 소비. TB 스칼라 `loss_train/lm_kd`, run name `kd=` 필드. Task 6의 config가 이 키들을 사용.

- [ ] **Step 1: `train()` 상단 — distill.lm config 읽기** (기존 `distill_itc` 블록 바로 아래에 추가)

```python
    distill_lm = config.get('distill', {}).get('lm', {})
    lm_kd_enabled = distill_lm.get('enabled', False)
    lm_kd_weight = float(distill_lm.get('weight', 1.0))
    lm_kd_temp = float(distill_lm.get('temp', 2.0))
```

- [ ] **Step 2: 티처 LM 호출** (기존 `itc_feats` 호출 블록 바로 아래에 추가)

```python
        # online teacher: same augmented batch -> LM decoder logits (bf16, no_grad). None when distill off.
        if lm_kd_enabled and online_teacher is not None:
            teacher_lm_logits, teacher_lm_ids = online_teacher.lm_logits(image, caption)
        else:
            teacher_lm_logits = teacher_lm_ids = None
```

- [ ] **Step 3: model 호출 2곳(autocast/비-autocast) 언팩·인자·합산 교체** — 두 분기 모두 동일하게:

```python
                loss_ita, loss_itm, loss_lm, loss_itc_kd, loss_lm_kd = model(
                    image, caption, alpha=alpha,
                    teacher_img_feat=teacher_img_feat, teacher_text_feat=teacher_text_feat,
                    distill_temp=itc_kd_temp,
                    teacher_lm_logits=teacher_lm_logits, teacher_lm_input_ids=teacher_lm_ids,
                    lm_distill_temp=lm_kd_temp)
                loss = loss_ita + loss_itm + loss_lm
                if itc_kd_enabled and loss_itc_kd is not None:
                    loss = loss + itc_kd_weight * loss_itc_kd
                if lm_kd_enabled and loss_lm_kd is not None:
                    loss = loss + lm_kd_weight * loss_lm_kd
```

(비-autocast 분기는 들여쓰기 한 단계 얕음 — 내용 동일)

- [ ] **Step 4: 로깅 추가** — metric_logger 블록의 `loss_itc_kd` 줄 아래:

```python
        if loss_lm_kd is not None:
            metric_logger.update(loss_lm_kd=loss_lm_kd.item())
```

TB 블록의 `loss_train/itc_kd` 줄 아래:

```python
                if loss_lm_kd is not None:
                    writer.add_scalar("loss_train/lm_kd", loss_lm_kd.item(), global_step)
```

- [ ] **Step 5: `main()` 티처 생성 교체** — 기존 `online_teacher = None` 블록(~L346-356)을 다음으로 교체:

```python
    #### online teacher (distillation) — replicate per rank, frozen, not DDP-wrapped ####
    online_teacher = None
    distill_cfg = config.get('distill', {})
    teacher_keep = tuple(k for k in ('itc', 'lm') if distill_cfg.get(k, {}).get('enabled', False))
    if teacher_keep:
        ckpt_path = config.get('teacher', {}).get('checkpoint', '')
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"distill enabled (keep={teacher_keep}) but teacher.checkpoint not found: '{ckpt_path}'"
            )
        from distillation.online_teacher import OnlineTeacher
        online_teacher = OnlineTeacher(
            checkpoint=ckpt_path,
            image_size=config['image_size'],
            vit='large', bert='base',
            queue_size=config['queue_size'],
            keep=teacher_keep,
        ).to(device)
        print(f"[distill] online teacher loaded (keep={teacher_keep})")
```

(변경점: `if itc.enabled` → keep 유도, **checkpoint 부재 시 즉시 FileNotFoundError** — 빈 경로면 무작위 티처로 조용히 증류되는 사고 방지. `os`는 이미 import되어 있음.)

- [ ] **Step 6: `make_tb_run_name`에 kd 필드 추가** — `return` 직전에 삽입:

```python
    # distill 활성 시 run name에 kd 태그 (예: kd=lm_w1.0T2.0) — w/T는 고정 하이퍼파라미터라
    # 시계열 로깅 대신 이름+config.yaml 덤프로 기록
    distill_cfg = config.get("distill", {})
    kd_parts = [f"{k}_w{distill_cfg[k].get('weight', 1.0)}T{distill_cfg[k].get('temp')}"
                for k in ("itc", "lm") if distill_cfg.get(k, {}).get("enabled", False)]
    if kd_parts:
        tb_option_dict["kd"] = "+".join(kd_parts)
```

- [ ] **Step 7: 컴파일 + run name 검증**

```bash
python -m py_compile pretrain.py && echo COMPILE_OK
python - <<'EOF'
from pretrain import make_tb_run_name
cfg = {"exp": "8.lm_distill", "vit": "small_reg", "my_bert_size": "minilm",
       "batch_size": 40, "used_dataset": "coco_vg",
       "distill": {"itc": {"enabled": False}, "lm": {"enabled": True, "weight": 1.0, "temp": 2.0}}}
name = make_tb_run_name(cfg)
print(name)
assert "kd=lm_w1.0T2.0" in name
cfg["distill"]["lm"]["enabled"] = False
assert "kd=" not in make_tb_run_name(cfg)   # baseline 이름 불변
print("RUN_NAME_OK")
EOF
```

Expected: `COMPILE_OK`, run name 출력에 `kd=lm_w1.0T2.0`, `RUN_NAME_OK`

- [ ] **Step 8: Commit**

```bash
git add pretrain.py
git commit -m "feat(distill): wire LM logit KD into train loop - teacher lm_logits per step, weighted sum, TB + run-name tag, loud ckpt check"
```

---

### Task 6: LM-KD 런 config + 최종 검증

**Files:**
- Create: `configs/pretrain_lm_distill.yaml`

**Interfaces:**
- Consumes: Task 5의 config 키 (`distill.lm.*`, `teacher.*`), Task 0의 `pretrain_student.yaml` (원본).
- Produces: 본 런 실행 config. `teacher.checkpoint`는 사용자가 다운로드 후 기입하는 **의도된 미결 입력** (빈 값이면 Task 5의 가드가 즉시 실패시킴).

- [ ] **Step 1: config 작성** — Create `configs/pretrain_lm_distill.yaml`

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

# ===== LM logit distillation 런 (student + frozen BLIP-large teacher) =====
output_dir: '/home/minwoo/Distillation_Project/output/pt_smallreg_minilm_lm_distill'

# experiment 태그 (tensorboard run 이름에 사용)
exp: '8.lm_distill'
# 이미지 증강 토글: 온라인 티처 증류는 augmentation ON 필수 (no-aug는 NO-GO였음)
pretrain_train_aug: true

# new) tenserboard log interval
tb_train_log_interval: 50
tb_val_log_interval: 1000

# ===== student architecture (pretrain_student.yaml과 동일 — baseline 대비 변인 통제) =====
vit: 'small_reg'
vit_grad_ckpt: False
vit_ckpt_layer: 0

my_bert_size: 'minilm'


## 벨리데이션 과정
#### COCO Karpathy validation loss ####
val_loss_enabled: true

val_loss_file: '/home/minwoo/Distillation_Project/datasets/vision/coco/annotations/coco_karpathy_val.json'
val_loss_image_root: '/home/minwoo/Distillation_Project/datasets/vision/coco/images/'

val_loss_batch_size: 32
val_loss_num_workers: 4
val_loss_caption_mode: first
val_loss_drop_last: true
val_loss_pin_memory: true

val_loss_print_freq: 50

val_loss_interval_steps: 500
val_loss_max_batches: 30 # 전체가 39더라

val_loss_epoch_end: true
val_loss_epoch_max_batches: 30

val_loss_alpha_mode: current
val_loss_restore_rng: true
val_loss_amp: true

#### COCO Karpathy retrieval validation (momentum 없는 student-only R@1/5/10) ####
val_retrieval_enabled: true
val_retrieval_ann_root: '/home/minwoo/Distillation_Project/datasets/vision/coco/annotations/'
val_retrieval_image_root: '/home/minwoo/Distillation_Project/datasets/vision/coco/images/'
val_retrieval_split: 'val'
val_retrieval_batch_size: 64
val_retrieval_num_workers: 4
val_retrieval_amp: true
k_test: 128

val_retrieval_itc_interval_steps: 1000
val_retrieval_itc_epoch_end: true

val_retrieval_itm_interval_steps: 18675
val_retrieval_itm_epoch_end: true

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

# ====== knowledge distillation ======
# baseline(exp 7)과의 차이는 아래 lm 블록 하나뿐이어야 한다 (실험 통제).
distill:
  itc:
    enabled: false          # 이번 런은 LM-only
    weight: 1.0
    temp: 0.05
  lm:
    enabled: true
    weight: 1.0             # w — 고정 하이퍼파라미터, 첫 런에서 loss_train/lm_kd 스케일 보고 조정
    temp: 2.0               # T — 스윕 후보 1/2/4

teacher:
  arch: 'blip_large'
  # 공식 Salesforce BLIP-large pretrain (ViT-L + BERT-base, 디코더 포함).
  # BLIP 공식 README의 Pre-trained checkpoints 표에서 ViT-L(129M) 링크로 다운로드 후 경로 기입.
  # 빈 값이면 pretrain.py가 FileNotFoundError로 즉시 실패함 (무작위 티처 증류 방지).
  checkpoint: ''
```

- [ ] **Step 2: yaml 파싱 + baseline과의 diff 확인**

```bash
python -c "import yaml; c = yaml.safe_load(open('configs/pretrain_lm_distill.yaml')); print(c['distill'], c['exp'])"
diff <(grep -v -E "output_dir|exp:|distill|teacher|enabled|weight|temp|checkpoint|arch|#|^\s*$" configs/pretrain_student.yaml) \
     <(grep -v -E "output_dir|exp:|distill|teacher|enabled|weight|temp|checkpoint|arch|#|^\s*$" configs/pretrain_lm_distill.yaml) && echo BASE_KEYS_IDENTICAL
```

Expected: distill 딕셔너리 출력 + `8.lm_distill`, 그리고 `BASE_KEYS_IDENTICAL` (변인 통제 확인 — 학습 키는 baseline과 동일)

- [ ] **Step 3: 전체 테스트 스위트 최종 회귀**

```bash
python -m unittest distillation.test_losses distillation.test_forward_kd -v
python -m unittest distillation.test_online_teacher -v   # 느림 (~수 분)
```

Expected: 전부 `ok`

- [ ] **Step 4: Commit**

```bash
git add configs/pretrain_lm_distill.yaml
git commit -m "config: LM logit distillation run (exp 8.lm_distill, lm-only KD w=1.0 T=2.0)"
```

- [ ] **Step 5: GPU 통합 스모크 (수동 — 티처 체크포인트 확보 후)**

사용자가 체크포인트를 다운로드해 `teacher.checkpoint`에 기입한 뒤:

```bash
cd /home/minwoo/Distillation_Project_lm_distill
# 1-GPU 스모크: 기존 4-GPU 본 런과 동일한 런처 사용, nproc만 1
python -m torch.distributed.run --nproc_per_node=1 pretrain.py --config ./configs/pretrain_lm_distill.yaml
# 수 분 관찰 후 Ctrl+C
```

확인 항목:
1. `[distill] online teacher loaded (keep=('lm',))` 출력
2. `teacher load:` 메시지에서 critical missing key 없이 통과
3. 콘솔 metric에 `loss_lm_kd` 등장 + finite (수백 스텝 안정)
4. TB run name에 `kd=lm_w1.0T2.0` 포함, `loss_train/lm_kd` 스칼라 기록
5. GPU 메모리: 티처+학생 동시 탑재 여유 확인 (`nvidia-smi`)
6. `loss_train/lm_kd`와 `loss_train/lm`의 스케일 비교 → w=1.0 유지/조정 판단

본 런(4-GPU)은 스모크 통과 후 사용자 결정으로 시작.

---

## 실행 순서 요약

Task 0 → 1 → 2 → 3 → 4 → **5 (4와 연달아 — 5-tuple 불일치 구간 해소)** → 6. 각 태스크 종료 시 커밋 (실행 시작 시 커밋 일괄 허용 확인). GPU 스모크(Task 6 Step 5)만 티처 체크포인트가 필요하며 그 전까지는 전부 CPU로 검증 가능.

