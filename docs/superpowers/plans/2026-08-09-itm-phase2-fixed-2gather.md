# ITM Phase 2 — fixed 2-gather 서브샘플 구현 플랜

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`).

**Goal:** full B×B(15× 느림)를 `topk=k` 고정 2-gather(`2·B·(k+1)` 셀)로 줄여 학습을 한 자릿수 배수로 낮춘다. 학생이 하드네거를 뽑고 티처가 동일 인덱스로 채점, 방향별 softmax forward-KL.

**Architecture:** 신규 2함수(`itm_pair_logits` 배치 pair-scorer, `itm_gathered_kd_loss`) + 티처 `itm_matrix_gathered`. forward를 **A-통일**로 리팩터: `teacher_itm_logits`(precompute) 제거, `online_teacher` 객체를 forward에 받아 full·gathered 모두 forward 안에서 티처 계산. config `distill.itm.topk`(−1=full/k=gather).

**Tech Stack:** PyTorch, HuggingFace BERT multimodal encoder, BLIP pretrain, unittest.

**선행 스펙:** `docs/superpowers/specs/2026-08-08-itm-phase2-fixed-2gather-design.md`

## Global Constraints
- **fp32 itm_head** (bf16이면 gap 오차 8000배): 모든 pair-scorer가 `torch.autocast(enabled=False)`+`.float()`로 강제.
- **채널 인덱스:** 0=z1(no-match/reverse), 1=z2(match/forward). `direction`={forward:[1], reverse:[0], bidir:[0,1]} — **채널 선택**(Phase 1과 동일). i2t·t2i gather는 **축**이라 direction과 무관하게 항상 둘 다.
- **k+1 고정 shape:** 각 방향 후보 = positive(col 0) + 하드네거 k. `weights_*`는 diagonal-zeroed(`blip_pretrain.py:448-451`)라 `topk(k)`가 순수 네거, positive는 명시 prepend.
- **KL 방향:** forward `KL(teacher‖student)`, `×temp`, 항 평균. 티처 `.detach()`.
- **baseline OFF 불변:** `online_teacher is None`(itm 비활성)이면 미진입, `loss_itm_kd=None`.
- CPU 스모크 테스트: `vit="base", bert="base", image_size=224`.

---

## 파일 구조
| 파일 | 작업 |
|---|---|
| `distillation/itm_matrix.py` | `itm_pair_logits` 신규 추가 |
| `distillation/losses.py` | `itm_gathered_kd_loss` 신규 추가 |
| `distillation/online_teacher.py` | `OnlineTeacher.itm_matrix_gathered` 신규 |
| `models/blip_pretrain.py` | forward A-통일 리팩터(시그니처+itm-kd 블록+import) |
| `pretrain.py` | config `topk` 파싱, 티처 precompute 제거, `online_teacher`+`itm_topk` 전달 |
| `configs/pretrain_itm_matrix_kd.yaml` | `topk: 8` 추가 |
| `distillation/test_itm_matrix.py` | pair-scorer 테스트 |
| `distillation/test_losses.py` | gathered-loss 테스트 |
| `distillation/test_online_teacher.py` | itm_matrix_gathered 계약 |
| `distillation/test_forward_kd.py` | A-통일(online_teacher) + gathered 경로 |

---

### Task 1: `itm_pair_logits` (배치 pair-scorer)

**Files:** Modify `distillation/itm_matrix.py`; Test `distillation/test_itm_matrix.py`

**Interfaces:**
- Produces: `itm_pair_logits(text_encoder, itm_head, image_embeds, image_atts, enc_ids, text_atts, image_idx, text_idx) -> [B, M, 2]` fp32. `image_idx`/`text_idx` `[B,M]` long이 pool(image_embeds/enc_ids)로의 인덱스. row r = `(image_embeds[image_idx[r,m]], enc_ids[text_idx[r,m]])` M개 채점.

- [ ] **Step 1: 실패 테스트** — `test_itm_matrix.py`의 `TestItmBxbLogits` 클래스에 메서드 추가(같은 파일 `FakeEncoder` 재사용). import에 `itm_pair_logits` 추가:

```python
    def test_pair_logits_scores_given_pairs(self):
        B, M = 3, 2
        image_embeds = torch.zeros(4, 5, 8)     # pool=4 이미지 (B와 달라도 됨)
        for i in range(4):
            image_embeds[i, 0, 0] = i
        image_atts = torch.ones(4, 5, dtype=torch.long)
        enc_ids = torch.zeros(4, 6, dtype=torch.long)   # pool=4 텍스트
        for j in range(4):
            enc_ids[j, 1] = j
        text_atts = torch.ones(4, 6, dtype=torch.long)
        image_idx = torch.tensor([[0, 1], [2, 3], [1, 0]])
        text_idx = torch.tensor([[3, 2], [0, 1], [2, 2]])

        logits = itm_pair_logits(FakeEncoder(), nn.Identity(),
                                 image_embeds, image_atts, enc_ids, text_atts,
                                 image_idx, text_idx)
        self.assertEqual(tuple(logits.shape), (B, M, 2))
        self.assertEqual(logits.dtype, torch.float32)
        for r in range(B):
            for m in range(M):
                self.assertAlmostEqual(logits[r, m, 0].item(), image_idx[r, m].item())
                self.assertAlmostEqual(logits[r, m, 1].item(), text_idx[r, m].item())
```
(import 줄을 `from distillation.itm_matrix import itm_bxb_logits, itm_pair_logits`로.)

- [ ] **Step 2: 실패 확인** — `python -m pytest distillation/test_itm_matrix.py::TestItmBxbLogits::test_pair_logits_scores_given_pairs -v` → FAIL (ImportError).

- [ ] **Step 3: 구현** — `distillation/itm_matrix.py` 끝(`itm_bxb_logits` 아래)에 추가:

```python
def itm_pair_logits(text_encoder, itm_head, image_embeds, image_atts,
                    enc_ids, text_atts, image_idx, text_idx):
    """B×M 임의 (image,text) 쌍의 ITM 로짓. image_idx/text_idx [B,M] long이
    pool(image_embeds/enc_ids)로의 인덱스; row r은 (image_embeds[image_idx[r,m]],
    enc_ids[text_idx[r,m]]) M개를 채점 → [B,M,2] fp32. 단일 배치 forward(B*M).
    itm_head는 autocast 밖 fp32(프로브 §7). enc_token_id는 호출자가 enc_ids에 설정."""
    B, M = image_idx.shape
    fi = image_idx.reshape(-1)                                 # [B*M]
    ft = text_idx.reshape(-1)                                  # [B*M]
    out = text_encoder(enc_ids[ft],
                       attention_mask=text_atts[ft],
                       encoder_hidden_states=image_embeds[fi],
                       encoder_attention_mask=image_atts[fi],
                       return_dict=True)
    cls = out.last_hidden_state[:, 0, :]                       # [B*M, D]
    with torch.autocast(device_type=image_embeds.device.type, enabled=False):
        logits = itm_head(cls.float())                         # [B*M, 2]
    return logits.reshape(B, M, 2)
```

- [ ] **Step 4: 통과 확인** — `python -m pytest distillation/test_itm_matrix.py -v` → 4 PASS(기존 3 + 신규).

- [ ] **Step 5: 커밋**
```bash
git add distillation/itm_matrix.py distillation/test_itm_matrix.py
git commit -m "feat(itm-kd): itm_pair_logits — B×M 임의 쌍 배치 pair-scorer"
```

---

### Task 2: `itm_gathered_kd_loss`

**Files:** Modify `distillation/losses.py`; Test `distillation/test_losses.py`

**Interfaces:**
- Produces: `itm_gathered_kd_loss(s_i2t, t_i2t, s_t2i, t_t2i, direction, temp) -> scalar`. 각 `[B,k+1,2]`(0=z1,1=z2; dim1=k+1 후보). gather(i2t,t2i)별 × channel(direction)별 dim=1 softmax forward KL(T‖S) 평균 × temp.

- [ ] **Step 1: 실패 테스트** — `test_losses.py`에 클래스 추가(import에 `itm_gathered_kd_loss`):

```python
class TestItmGatheredKdLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B, self.M, self.temp = 5, 4, 0.05

    def _rand(self):
        return torch.randn(self.B, self.M, 2)

    def test_identical_is_zero(self):
        a, b = self._rand(), self._rand()
        loss = itm_gathered_kd_loss(a.clone().requires_grad_(True), a,
                                    b.clone().requires_grad_(True), b, "bidir", self.temp)
        self.assertLess(loss.item(), 1e-6)

    def test_forward_uses_only_z2(self):
        i, t = self._rand(), self._rand()
        si, st = i.clone(), t.clone()
        si[:, :, 0] += torch.randn(self.B, self.M)      # z1만 교란
        st[:, :, 0] += torch.randn(self.B, self.M)
        self.assertLess(itm_gathered_kd_loss(si, i, st, t, "forward", self.temp).item(), 1e-6)
        self.assertGreater(itm_gathered_kd_loss(si, i, st, t, "reverse", self.temp).item(), 1e-6)

    def test_grad_student_not_teacher(self):
        si = self._rand().requires_grad_(True); ti = self._rand().requires_grad_(True)
        st = self._rand().requires_grad_(True); tt = self._rand().requires_grad_(True)
        itm_gathered_kd_loss(si, ti, st, tt, "bidir", self.temp).backward()
        self.assertIsNotNone(si.grad); self.assertIsNotNone(st.grad)
        self.assertIsNone(ti.grad); self.assertIsNone(tt.grad)

    def test_invalid_direction_raises(self):
        a = self._rand()
        with self.assertRaises(ValueError):
            itm_gathered_kd_loss(a, a, a, a, "sideways", self.temp)
```

- [ ] **Step 2: 실패 확인** — `python -m pytest distillation/test_losses.py::TestItmGatheredKdLoss -v` → FAIL(ImportError).

- [ ] **Step 3: 구현** — `distillation/losses.py` 끝에 추가:

```python
def itm_gathered_kd_loss(s_i2t, t_i2t, s_t2i, t_t2i, direction, temp):
    """Phase 2 서브샘플 ITM KD. 각 [B,k+1,2] (0=z1, 1=z2; dim1=k+1 후보, col0=positive).
    gather(i2t,t2i 둘 다)별 × channel(direction)별 dim=1 softmax forward KL(T‖S) 평균 × temp."""
    channels = {"forward": [1], "reverse": [0], "bidir": [0, 1]}.get(direction)
    if channels is None:
        raise ValueError(f"direction must be forward/reverse/bidir, got {direction!r}")
    total, n = 0.0, 0
    for s, t in ((s_i2t, t_i2t), (s_t2i, t_t2i)):
        s = s.float()
        t = t.float().detach()
        for c in channels:
            total = total + F.kl_div(F.log_softmax(s[:, :, c] / temp, dim=1),
                                     F.softmax(t[:, :, c] / temp, dim=1), reduction="batchmean")
            n += 1
    return (total / n) * temp
```

- [ ] **Step 4: 통과 확인** — `python -m pytest distillation/test_losses.py -q` → 전부 PASS(기존 + 신규 4).

- [ ] **Step 5: 커밋**
```bash
git add distillation/losses.py distillation/test_losses.py
git commit -m "feat(itm-kd): itm_gathered_kd_loss (방향별 k+1 softmax forward-KL)"
```

---

### Task 3: `OnlineTeacher.itm_matrix_gathered`

**Files:** Modify `distillation/online_teacher.py`; Test `distillation/test_online_teacher.py`

**Interfaces:**
- Consumes: `itm_pair_logits`(Task 1).
- Produces: `OnlineTeacher.itm_matrix_gathered(image, caption, idx_i2t_full, idx_t2i_full) -> (t_i2t, t_t2i)`, 각 `[B,k+1,2]` fp32 no_grad. 학생과 동일 인덱스.

- [ ] **Step 1: 실패 테스트** — `test_online_teacher.py`의 `TestOnlineTeacherKeepItm`에 메서드 추가:

```python
    def test_itm_matrix_gathered_contract(self):
        B, k = 3, 2
        image = torch.randn(B, 3, 224, 224)
        caption = ["a green field", "a red car", "a blue sky"]
        ar = torch.arange(B)[:, None]
        idx_i2t = torch.cat([ar, torch.tensor([[1, 2], [0, 2], [0, 1]])], dim=1)  # [B,k+1]
        idx_t2i = torch.cat([ar, torch.tensor([[2, 1], [2, 0], [1, 0]])], dim=1)
        t_i2t, t_t2i = self.teacher.itm_matrix_gathered(image, caption, idx_i2t, idx_t2i)
        for t in (t_i2t, t_t2i):
            self.assertEqual(tuple(t.shape), (B, k + 1, 2))
            self.assertEqual(t.dtype, torch.float32)
            self.assertFalse(t.requires_grad)
```

- [ ] **Step 2: 실패 확인** — `python -m pytest distillation/test_online_teacher.py::TestOnlineTeacherKeepItm::test_itm_matrix_gathered_contract -v` → FAIL(AttributeError). (구성 ~1-2분)

- [ ] **Step 3: 구현** — `online_teacher.py`의 import에 `itm_pair_logits` 추가(`from distillation.itm_matrix import itm_bxb_logits, itm_pair_logits`), `itm_matrix` 메서드 아래에 추가:

```python
    @torch.no_grad()
    def itm_matrix_gathered(self, image, caption, idx_i2t_full, idx_t2i_full):
        """학생이 뽑은 인덱스로 gather된 티처 ITM 로짓 (t_i2t, t_t2i), 각 [B,k+1,2] fp32.
        idx_*_full [B,k+1]: i2t=이미지별 텍스트 인덱스, t2i=텍스트별 이미지 인덱스(col0=positive)."""
        self._require("itm")
        device = image.device
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            image_embeds = self.model.visual_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
            text = self.tokenizer(caption, padding="max_length", truncation=True,
                                  max_length=30, return_tensors="pt").to(device)
            enc_ids = text.input_ids.clone()
            enc_ids[:, 0] = self.tokenizer.enc_token_id
            B, M = idx_i2t_full.shape
            ar = torch.arange(B, device=device)[:, None].expand(B, M)
            t_i2t = itm_pair_logits(self.model.text_encoder, self.model.itm_head, image_embeds,
                                    image_atts, enc_ids, text.attention_mask, ar, idx_i2t_full.to(device))
            t_t2i = itm_pair_logits(self.model.text_encoder, self.model.itm_head, image_embeds,
                                    image_atts, enc_ids, text.attention_mask, idx_t2i_full.to(device), ar)
        return t_i2t, t_t2i
```

- [ ] **Step 4: 통과 확인** — `python -m pytest distillation/test_online_teacher.py::TestOnlineTeacherKeepItm -v` → 4 PASS.

- [ ] **Step 5: 커밋**
```bash
git add distillation/online_teacher.py distillation/test_online_teacher.py
git commit -m "feat(itm-kd): OnlineTeacher.itm_matrix_gathered (동일 인덱스 티처 gather)"
```

---

### Task 4: forward A-통일 리팩터 (full+gathered, 티처 forward-내)

**Files:** Modify `models/blip_pretrain.py`; Test `distillation/test_forward_kd.py`

**Interfaces:**
- Consumes: `itm_pair_logits`(T1), `itm_gathered_kd_loss`(T2), `OnlineTeacher.itm_matrix_gathered`(T3), 기존 `itm_bxb_logits`/`itm_matrix_kd_loss`.
- Produces: `forward(..., online_teacher=None, itm_topk=-1, itm_distill_temp=0.05, itm_distill_direction='bidir')`. `teacher_itm_logits` 인자 **제거**. 반환 6-튜플 유지, `loss_itm_kd` 마지막.

- [ ] **Step 1: 실패 테스트** — `test_forward_kd.py`에서 기존 `test_itm_kd_computed_when_teacher_matrix_given`(raw 로짓 전달)을 **삭제**하고, `test_itm_kd_end_to_end_with_teacher`를 아래로 **교체** + gathered 신규 추가:

```python
    def test_itm_kd_full_with_online_teacher(self):
        from distillation.online_teacher import OnlineTeacher
        teacher = OnlineTeacher(checkpoint="", image_size=224, vit="base",
                                bert="base", queue_size=240, keep=("itm",))
        out = self.model(self.image, self.caption, alpha=0.4, update_train_state=False,
                         online_teacher=teacher, itm_topk=-1, itm_distill_direction="bidir")
        loss_itm_kd = out[5]
        self.assertIsNotNone(loss_itm_kd)
        self.assertTrue(torch.isfinite(loss_itm_kd))
        (out[0] + out[1] + out[2] + out[5]).backward()

    def test_itm_kd_gathered_with_online_teacher(self):
        from distillation.online_teacher import OnlineTeacher
        teacher = OnlineTeacher(checkpoint="", image_size=224, vit="base",
                                bert="base", queue_size=240, keep=("itm",))
        out = self.model(self.image, self.caption, alpha=0.4, update_train_state=False,
                         online_teacher=teacher, itm_topk=1, itm_distill_direction="bidir")
        loss_itm_kd = out[5]
        self.assertIsNotNone(loss_itm_kd)
        self.assertTrue(torch.isfinite(loss_itm_kd))
        (out[0] + out[1] + out[2] + out[5]).backward()
```
(`self.image`는 B=2, `itm_topk=1` → k+1=2 ≤ B=2 OK. 기존 `test_six_tuple_with_nones_when_no_teacher`는 그대로 — online_teacher 미전달이라 None.)

- [ ] **Step 2: 실패 확인** — `python -m pytest distillation/test_forward_kd.py -v` → FAIL(`online_teacher` 인자 없음 → TypeError, 또는 gathered 미구현).

- [ ] **Step 3: 구현**

3a. import(22행 부근) 확장:
```python
from distillation.losses import itc_distill_loss, lm_distill_loss, itm_matrix_kd_loss, itm_gathered_kd_loss
from distillation.itm_matrix import itm_bxb_logits, itm_pair_logits
```

3b. 시그니처(351행) 교체 — `teacher_itm_logits=None` 제거, `online_teacher=None, itm_topk=-1` 추가:
```python
                teacher_lm_logits=None, teacher_lm_input_ids=None, lm_distill_temp=2.0,
                online_teacher=None, itm_topk=-1, itm_distill_temp=0.05, itm_distill_direction='bidir'):
```

3c. itm-kd 블록(527-535) 전체 교체:
```python
        # external-teacher ITM distillation (forward 안에서 티처 계산). None when disabled.
        loss_itm_kd = None
        if online_teacher is not None:
            if itm_topk is not None and itm_topk > 0:
                k = int(itm_topk)
                ar = torch.arange(bs, device=image.device)[:, None]
                idx_i2t_full = torch.cat([ar, weights_i2t.topk(k, dim=1).indices], dim=1)  # [B,k+1]
                idx_t2i_full = torch.cat([ar, weights_t2i.topk(k, dim=1).indices], dim=1)
                arM = ar.expand(bs, k + 1)
                s_i2t = itm_pair_logits(self.text_encoder, self.itm_head, image_embeds, image_atts,
                                        encoder_input_ids, text.attention_mask, arM, idx_i2t_full)
                s_t2i = itm_pair_logits(self.text_encoder, self.itm_head, image_embeds, image_atts,
                                        encoder_input_ids, text.attention_mask, idx_t2i_full, arM)
                t_i2t, t_t2i = online_teacher.itm_matrix_gathered(image, caption, idx_i2t_full, idx_t2i_full)
                loss_itm_kd = itm_gathered_kd_loss(s_i2t, t_i2t.to(image.device),
                                                   s_t2i, t_t2i.to(image.device),
                                                   itm_distill_direction, itm_distill_temp)
            else:
                teacher_itm = online_teacher.itm_matrix(image, caption)
                student_itm_matrix = itm_bxb_logits(
                    self.text_encoder, self.itm_head,
                    image_embeds, image_atts, encoder_input_ids, text.attention_mask,
                    use_checkpoint=True)
                loss_itm_kd = itm_matrix_kd_loss(
                    student_itm_matrix, teacher_itm.to(image.device),
                    itm_distill_direction, itm_distill_temp)
```
(주의: `weights_i2t`/`weights_t2i`는 448-451에서, `bs`는 그 위, `image_embeds`/`image_atts`/`encoder_input_ids`/`text`/`caption`은 모두 이 지점 scope에 있음.)

- [ ] **Step 4: 통과 확인** — `python -m pytest distillation/test_forward_kd.py -v` → PASS(baseline None + full + gathered). 이어서 `python -m pytest distillation/ -q` 무회귀.

- [ ] **Step 5: 커밋**
```bash
git add models/blip_pretrain.py distillation/test_forward_kd.py
git commit -m "feat(itm-kd): forward A-통일 — online_teacher 인자, full+gathered 티처 forward-내"
```

---

### Task 5: `pretrain.py` 재배선 (topk 파싱, precompute 제거, online_teacher 전달)

**Files:** Modify `pretrain.py`

**Interfaces:**
- Consumes: forward 새 시그니처(T4).
- 자동 단위테스트 없음 — `py_compile` + `distillation/` 무회귀 + Task 6 스모크.

- [ ] **Step 1: config topk 파싱 추가** — itm 파싱(98행 `itm_kd_direction` 아래):
```python
    itm_kd_topk = int(distill_itm.get('topk', -1))
```

- [ ] **Step 2: 티처 precompute 제거** — 138-142행(주석 포함 `if itm_kd_enabled ... teacher_itm_logits = online_teacher.itm_matrix(...) ... else: teacher_itm_logits = None`) **삭제**. (online_teacher 객체는 이미 train() 인자로 존재.)

- [ ] **Step 3: model() 호출 교체 (cuda·cpu 두 분기)** — 두 분기 모두 `teacher_itm_logits=teacher_itm_logits, itm_distill_temp=itm_kd_temp, itm_distill_direction=itm_kd_direction`를 아래로 교체:
```python
                    online_teacher=(online_teacher if itm_kd_enabled else None),
                    itm_topk=itm_kd_topk, itm_distill_temp=itm_kd_temp,
                    itm_distill_direction=itm_kd_direction)
```
(cpu 분기도 동일 — 들여쓰기만 맞춤. 손실합산 `if itm_kd_enabled and loss_itm_kd is not None: loss = loss + itm_kd_weight * loss_itm_kd`는 그대로.)

- [ ] **Step 4: 구문·무회귀** — `python -m py_compile pretrain.py && python -m pytest distillation/ -q` → 컴파일 OK + 전부 PASS. 추가로 grep 확인: `grep -n "teacher_itm_logits" pretrain.py` → **0건**(완전 제거).

- [ ] **Step 5: 커밋**
```bash
git add pretrain.py
git commit -m "feat(itm-kd): pretrain A-통일 배선 — topk 파싱, precompute 제거, online_teacher 전달"
```

---

### Task 6: config `topk` + GPU 스모크 timing

**Files:** Modify `configs/pretrain_itm_matrix_kd.yaml`

- [ ] **Step 1: config** — `distill.itm`에 `topk: 8` 추가(direction 아래).

- [ ] **Step 2: 전체 스위트 무회귀** — `python -m pytest distillation/ -q` → 전부 PASS.

- [ ] **Step 3: GPU 스모크 (수동, 1×4090)** — 스크래치 config로 `teacher.checkpoint=/home/minwoo/Distillation_Project/output/official_pretrain_checkpoint/model_large.pth`, `output_dir`=스크래치 채우고:
  - `topk: 8`로 `torchrun --nproc_per_node=1` bs=40 → **OOM 없이(checkpoint 없이도)** 스텝 진입 + 50/100스텝 time 기록.
  - `topk: 4`로 동일.
  - baseline(OFF) 대비 배수 계산 → 목표 **한 자릿수**. `progress.md`(SDD 렛저) + 스펙 §6에 수치.

**게이트:** 배수가 한 자릿수면 Phase 2 성공. `loss_itm_kd` 유한·CE 안 삼킴 확인.

---

## 성공 기준 (스펙 §6)
step-time 한 자릿수 배수, OOM 없음(고정 shape·checkpoint 불필요), 학습 후 itm retrieval ≥ baseline + itm−itc>0.
