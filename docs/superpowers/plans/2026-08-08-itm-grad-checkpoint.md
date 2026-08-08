# ITM 매트릭스 gradient checkpointing — 구현 플랜

> **For agentic workers:** TDD, 체크박스(`- [ ]`) 추적. 소규모 변경(2파일)이라 인라인 실행.

**Goal:** `itm_bxb_logits`의 row별 인코더 forward를 gradient checkpointing으로 감싸 full B×B(bs=40)의 forward peak 메모리를 40row→1row로 낮춰, 24GB에서 OOM 없이 학습되게 한다.

**선행 스펙:** `docs/superpowers/specs/2026-08-08-itm-matrix-oom-gradient-checkpoint-design.md`

## Global Constraints
- checkpoint는 `use_checkpoint and torch.is_grad_enabled()`일 때만(티처 no_grad/eval/테스트 자동 plain).
- `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`.
- `itm_head` fp32 가드(autocast 밖) 불변. checkpoint는 인코더 호출만 감쌈.
- 출력은 checkpoint 유무와 **수치적으로 동일**(투명해야).

---

### Task 1: `itm_bxb_logits`에 `use_checkpoint` 추가

**Files:** Modify `distillation/itm_matrix.py`; Test `distillation/test_itm_matrix.py`

- [ ] **Step 1: 실패 테스트 추가** — `test_itm_matrix.py`에:

```python
    def test_checkpoint_matches_noncheckpoint_and_flows_grad(self):
        torch.manual_seed(0)
        B = 4
        image_embeds = torch.randn(B, 5, 4, requires_grad=True)
        image_atts = torch.ones(B, 5, dtype=torch.long)
        input_ids = torch.zeros(B, 6, dtype=torch.long)
        for j in range(B):
            input_ids[j, 1] = j
        text_atts = torch.ones(B, 6, dtype=torch.long)
        head = nn.Linear(4, 2)

        # 동일 입력에서 checkpoint on/off 출력이 같아야 (투명성)
        out_plain = itm_bxb_logits(FakeEncoder(), head, image_embeds, image_atts,
                                   input_ids, text_atts, use_checkpoint=False)
        out_ckpt = itm_bxb_logits(FakeEncoder(), head, image_embeds, image_atts,
                                  input_ids, text_atts, use_checkpoint=True)
        self.assertTrue(torch.allclose(out_plain, out_ckpt, atol=1e-5))

        # checkpoint 경로로 grad가 이미지 임베딩까지 흐른다
        out_ckpt.sum().backward()
        self.assertIsNotNone(image_embeds.grad)
        self.assertTrue(torch.isfinite(image_embeds.grad).all())
```

(`FakeEncoder`는 `encoder_hidden_states`를 실제로 읽어 grad 경로를 만든다 — 기존 클래스가 `hidden[:,0,0]=encoder_hidden_states[:,0,0]`이라 image_embeds까지 미분 연결됨.)

- [ ] **Step 2: 실패 확인** — `python -m pytest distillation/test_itm_matrix.py::TestItmBxbLogits::test_checkpoint_matches_noncheckpoint_and_flows_grad -v` → FAIL(`use_checkpoint` 인자 없음, TypeError).

- [ ] **Step 3: 구현** — `distillation/itm_matrix.py`:

```python
import torch
from torch.utils.checkpoint import checkpoint


def itm_bxb_logits(text_encoder, itm_head, image_embeds, image_atts,
                   encoder_input_ids, text_atts, use_checkpoint=False):
    """전체 B×B ITM 매치-로짓 매트릭스. row i = image i vs 배치 전체 텍스트.

    반환 [B, B, 2] fp32. text_encoder는 호출자 autocast(bf16 가능)에서 돌지만
    itm_head는 autocast 밖 fp32로 강제(bf16이면 gap 오차 폭증 — 프로브 §7).
    row별 chunk(image i를 B 텍스트에 repeat)라 메모리는 배치당 B 시퀀스로 유계.
    use_checkpoint=True면 row별 인코더 forward를 gradient checkpointing(재계산으로
    activation 절약 → full B×B가 24GB에 맞음). 티처(no_grad)/eval에선 자동 plain.
    enc_token_id는 호출자가 encoder_input_ids에 이미 설정한 상태로 받는다.
    """
    B = image_embeds.size(0)

    def _row(enc_hidden, enc_att):
        out = text_encoder(encoder_input_ids,
                           attention_mask=text_atts,
                           encoder_hidden_states=enc_hidden,
                           encoder_attention_mask=enc_att,
                           return_dict=True)
        return out.last_hidden_state[:, 0, :]              # [B, D]

    cls_rows = []
    for i in range(B):
        enc_hidden = image_embeds[i:i + 1].repeat(B, 1, 1)     # [B, L_img, D]
        enc_att = image_atts[i:i + 1].repeat(B, 1)             # [B, L_img]
        if use_checkpoint and torch.is_grad_enabled():
            cls = checkpoint(_row, enc_hidden, enc_att, use_reentrant=False)
        else:
            cls = _row(enc_hidden, enc_att)
        cls_rows.append(cls)
    cls = torch.stack(cls_rows, dim=0)                         # [B, B, D]
    with torch.autocast(device_type=image_embeds.device.type, enabled=False):
        logits = itm_head(cls.float())                         # [B, B, 2] fp32
    return logits
```

- [ ] **Step 4: 통과 확인** — `python -m pytest distillation/test_itm_matrix.py -v` → 3 PASS(기존 2 + 신규 1).

- [ ] **Step 5: 커밋**
```bash
git add distillation/itm_matrix.py distillation/test_itm_matrix.py
git commit -m "feat(itm-kd): itm_bxb_logits use_checkpoint (row별 인코더 gradient checkpointing)"
```

---

### Task 2: 학생 forward가 checkpoint 사용

**Files:** Modify `models/blip_pretrain.py`

- [ ] **Step 1: 변경** — forward의 학생 매트릭스 호출(`blip_pretrain.py:530` 부근)에 `use_checkpoint=True` 추가:
```python
            student_itm_matrix = itm_bxb_logits(
                self.text_encoder, self.itm_head,
                image_embeds, image_atts, encoder_input_ids, text.attention_mask,
                use_checkpoint=True)
```

- [ ] **Step 2: 무회귀 확인** — `python -m py_compile models/blip_pretrain.py && python -m pytest distillation/ -q` → 전부 PASS(기존 39개 + Task1 신규). (실모델 forward-kd e2e가 checkpoint 경로를 grad-enabled로 실제로 태움.)

- [ ] **Step 3: 커밋**
```bash
git add models/blip_pretrain.py
git commit -m "feat(itm-kd): 학생 B×B forward에 gradient checkpointing 적용"
```

---

### Task 3: GPU 스모크 — OOM 해소 + timing

**자동 테스트 아님(GPU 실행).** 스크래치 스모크 config로 실측:

- [ ] **Step 1:** itm-kd ON(checkpoint) 스모크 → `torchrun --nproc_per_node=1` bs=40. **OOM 없이 스텝이 돌고** 50-step 로그가 찍히는지 확인.
- [ ] **Step 2:** baseline OFF(`distill.itm.enabled=false`) 스모크 → 같은 조건 50-step 시간.
- [ ] **Step 3:** ON/OFF 50-step 간격 시간 비교 → **OFF 대비 배수** 기록. `progress.md`(SDD 렛저) + 스펙 §6에 수치.

**게이트:** ON이 OOM 없이 돌면 checkpointing 성공. step-time 배수가 감당 가능하면 Phase 1(full+checkpoint) 유효; 너무 크면 Phase 2(fixed-alloc) 근거.
