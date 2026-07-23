# 티처 visual_encoder 중복 호출 제거 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `OnlineTeacher.itc_feats/lm_logits/itm_soft`가 각각 독립적으로 `visual_encoder(image)`를 호출하는 중복(세 메커니즘 전부 켜지면 스텝당 3회)을 없애, 스텝당 1회로 줄인다.

**Architecture:** `OnlineTeacher`에 상태 없는 신규 메서드 `encode_image()`를 추가하고, 기존 3개 메서드에 `image_embeds=None` optional 파라미터를 추가한다(기본값 유지 시 기존 동작과 100% 동일 — 하위호환). `pretrain.py`의 학습 루프가 스텝당 한 번만 `encode_image()`를 호출하고 그 결과를 세 메서드 호출부(및 `itm_mix` 딕셔너리를 통해 `blip_pretrain.py`의 `itm_soft` 호출부)에 인자로 전달한다.

**Tech Stack:** Python 3.10, PyTorch, pytest, `/home/minwoo/miniconda3/envs/kd_r4/bin/python`.

## Global Constraints

- 브랜치: `worktree-itc_itm_lm_targetmix` 위에 이어서 커밋(base 5912b65, Phase 1 완료 지점). 새 브랜치 안 만듦.
- `image_embeds=None`이 기본값인 모든 곳에서, `None`이면 반드시 기존과 동일하게 메서드 내부에서 `self.model.visual_encoder(image)`를 계산해야 한다 — 기존 15개+ 테스트가 이 파라미터를 넘기지 않고 부르므로, 이 하위호환이 깨지면 즉시 회귀.
- `encode_image()`는 캐시가 아니다 — 내부에 상태(마지막으로 인코딩한 이미지 등)를 두지 않는다. "한 번만 계산"은 전적으로 호출부(`pretrain.py`)의 책임이다.
- `models/blip_pretrain.py`의 `forward()` 시그니처는 건드리지 않는다(이미 파라미터 13개) — `teacher_image_embeds`는 기존 `itm_mix` 딕셔너리에 새 키로 추가한다.
- `distillation/online_teacher.py`의 `itc_feats`/`lm_logits`/`itm_soft` 세 메서드의 기존 로직(텍스트 인코딩, 네거티브 조립 등)은 `image_embeds` 계산 부분 외에는 한 글자도 바꾸지 않는다.
- Python 실행은 항상 `/home/minwoo/miniconda3/envs/kd_r4/bin/python`을 명시 사용.
- 스펙(§2)에서 검토 후 기각한 대안(내부 identity-캐시, 단일 encode() 통합)은 이번 구현에서 채택하지 않는다.

---

### Task 1: `OnlineTeacher.encode_image()` + 기존 3개 메서드에 `image_embeds` 재사용 지원

**Files:**
- Modify: `distillation/online_teacher.py`
- Test: `distillation/test_online_teacher.py`

**Interfaces:**
- Produces: `OnlineTeacher.encode_image(self, image) -> Tensor` (신규, `@torch.no_grad()`, bf16, shape `[B, patches+1, hidden]`, 상태 없음 — 매 호출 재계산).
- Produces: `itc_feats(self, image, caption, image_embeds=None)`, `lm_logits(self, image, caption, image_embeds=None)`, `itm_soft(self, image, enc_input_ids, attention_mask, neg_idx_img, neg_idx_txt, temp, image_embeds=None)` — 기존 시그니처에 파라미터 1개씩 추가, 그 외 파라미터/반환값 타입 무변경.

- [ ] **Step 1: 현재 회귀 베이스라인 확인**

```bash
cd /home/minwoo/Distillation_Project_itc_itm_lm_targetmix
/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest distillation/ models/test_blip_pretrain_itm.py -q
```
Expected: `67 passed`.

- [ ] **Step 2: 실패하는 테스트부터 작성**

`distillation/test_online_teacher.py`에서 `class TestOnlineTeacherLargeConstruction(unittest.TestCase):` 바로 앞에 아래 클래스를 삽입:

```python
class TestOnlineTeacherImageEmbedsSharing(unittest.TestCase):
    """Phase 2: encode_image() 신규 + itc_feats/lm_logits/itm_soft의 image_embeds 재사용."""

    @classmethod
    def setUpClass(cls):
        cls.teacher = OnlineTeacher(checkpoint="", image_size=224, vit="base",
                                    bert="base", queue_size=240, keep=("itc", "lm", "itm"))

    def test_encode_image_contract(self):
        image = torch.randn(2, 3, 224, 224)
        embeds = self.teacher.encode_image(image)
        self.assertEqual(embeds.shape[0], 2)
        self.assertFalse(embeds.requires_grad)

    def test_itc_feats_reuses_precomputed_embeds(self):
        torch.manual_seed(0)
        image = torch.randn(2, 3, 224, 224)
        caption = ["a green field", "a red car"]
        embeds = self.teacher.encode_image(image)
        img_feat_direct, txt_feat_direct = self.teacher.itc_feats(image, caption)

        call_count = {'n': 0}
        real_forward = self.teacher.model.visual_encoder.forward
        def spy(*args, **kwargs):
            call_count['n'] += 1
            return real_forward(*args, **kwargs)
        with unittest.mock.patch.object(self.teacher.model.visual_encoder, 'forward', side_effect=spy):
            img_feat_reused, txt_feat_reused = self.teacher.itc_feats(
                image, caption, image_embeds=embeds)

        self.assertEqual(call_count['n'], 0)
        self.assertTrue(torch.allclose(img_feat_direct.float(), img_feat_reused.float(), atol=1e-4))
        self.assertTrue(torch.allclose(txt_feat_direct.float(), txt_feat_reused.float(), atol=1e-4))

    def test_lm_logits_reuses_precomputed_embeds(self):
        torch.manual_seed(0)
        image = torch.randn(2, 3, 224, 224)
        caption = ["a green field", "a red car"]
        embeds = self.teacher.encode_image(image)
        logits_direct, ids_direct = self.teacher.lm_logits(image, caption)

        call_count = {'n': 0}
        real_forward = self.teacher.model.visual_encoder.forward
        def spy(*args, **kwargs):
            call_count['n'] += 1
            return real_forward(*args, **kwargs)
        with unittest.mock.patch.object(self.teacher.model.visual_encoder, 'forward', side_effect=spy):
            logits_reused, ids_reused = self.teacher.lm_logits(image, caption, image_embeds=embeds)

        self.assertEqual(call_count['n'], 0)
        self.assertTrue(torch.equal(ids_direct, ids_reused))
        self.assertTrue(torch.allclose(logits_direct.float(), logits_reused.float(), atol=1e-3))

    def test_itm_soft_reuses_precomputed_embeds(self):
        torch.manual_seed(0)
        image = torch.randn(2, 3, 224, 224)
        text = self.teacher.tokenizer(["a green field", "a red car"],
                                      padding="max_length", truncation=True,
                                      max_length=30, return_tensors="pt")
        enc_ids = text.input_ids.clone()
        enc_ids[:, 0] = self.teacher.tokenizer.enc_token_id
        embeds = self.teacher.encode_image(image)
        soft_direct = self.teacher.itm_soft(image, enc_ids, text.attention_mask,
                                            neg_idx_img=[1, 0], neg_idx_txt=[1, 0], temp=1.0)

        call_count = {'n': 0}
        real_forward = self.teacher.model.visual_encoder.forward
        def spy(*args, **kwargs):
            call_count['n'] += 1
            return real_forward(*args, **kwargs)
        with unittest.mock.patch.object(self.teacher.model.visual_encoder, 'forward', side_effect=spy):
            soft_reused = self.teacher.itm_soft(image, enc_ids, text.attention_mask,
                                                neg_idx_img=[1, 0], neg_idx_txt=[1, 0], temp=1.0,
                                                image_embeds=embeds)

        self.assertEqual(call_count['n'], 0)
        self.assertTrue(torch.allclose(soft_direct.float(), soft_reused.float(), atol=1e-3))

    def test_visual_encoder_called_exactly_once_across_all_three(self):
        """헤드라인 테스트: encode_image()를 한 번 계산해서 세 메서드에 재사용하면
        visual_encoder는 총 1번만 호출된다(세 메커니즘이 다 켜진 최악 케이스를 모사)."""
        image = torch.randn(2, 3, 224, 224)
        caption = ["a green field", "a red car"]
        text = self.teacher.tokenizer(caption, padding="max_length", truncation=True,
                                      max_length=30, return_tensors="pt")
        enc_ids = text.input_ids.clone()
        enc_ids[:, 0] = self.teacher.tokenizer.enc_token_id

        call_count = {'n': 0}
        real_forward = self.teacher.model.visual_encoder.forward
        def spy(*args, **kwargs):
            call_count['n'] += 1
            return real_forward(*args, **kwargs)
        with unittest.mock.patch.object(self.teacher.model.visual_encoder, 'forward', side_effect=spy):
            embeds = self.teacher.encode_image(image)
            self.teacher.itc_feats(image, caption, image_embeds=embeds)
            self.teacher.lm_logits(image, caption, image_embeds=embeds)
            self.teacher.itm_soft(image, enc_ids, text.attention_mask,
                                  neg_idx_img=[1, 0], neg_idx_txt=[1, 0], temp=1.0,
                                  image_embeds=embeds)

        self.assertEqual(call_count['n'], 1)
```

- [ ] **Step 3: 테스트 실패 확인**

```bash
/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest distillation/test_online_teacher.py -v -k ImageEmbedsSharing
```
Expected: 5개 다 FAIL (`encode_image` 메서드가 아직 없어서 `AttributeError`, 나머지도 `image_embeds` 키워드 인자를 아직 안 받아서 `TypeError`).

- [ ] **Step 4: `distillation/online_teacher.py` 구현**

`itc_feats`/`lm_logits`/`itm_soft` 세 메서드 전체를 아래로 교체하고, `itm_soft` 바로 앞(또는 뒤)에 `encode_image`를 추가한다:

```python
    @torch.no_grad()
    def encode_image(self, image):
        """Compute teacher visual_encoder(image) fresh — NOT a cache, recomputes every
        call. Callers needing this for more than one of itc_feats/lm_logits/itm_soft in
        the same step should call this ONCE and pass the result via image_embeds= to
        each, to avoid redundant ViT forwards on the identical input."""
        device = image.device
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            return self.model.visual_encoder(image)

    @torch.no_grad()
    def itc_feats(self, image, caption, image_embeds=None):
        self._require("itc")
        device = image.device
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            if image_embeds is None:
                image_embeds = self.model.visual_encoder(image)
            img_feat = F.normalize(self.model.vision_proj(image_embeds[:, 0, :]), dim=-1)
            text = self.tokenizer(caption, padding="max_length", truncation=True,
                                  max_length=30, return_tensors="pt").to(device)
            text_output = self.model.text_encoder(text.input_ids,
                                                  attention_mask=text.attention_mask,
                                                  return_dict=True, mode="text")
            txt_feat = F.normalize(self.model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
        return img_feat, txt_feat

    @torch.no_grad()
    def lm_logits(self, image, caption, image_embeds=None):
        """Teacher-forced decoder logits for the same augmented batch.
        학생 forward의 LM 경로와 동일한 토크나이즈/BOS 규칙 — forward 쪽에서
        decoder_input_ids 일치를 assert하므로 규칙이 어긋나면 즉시 검출된다."""
        self._require("lm")
        device = image.device
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            if image_embeds is None:
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

    @torch.no_grad()
    def itm_soft(self, image, enc_input_ids, attention_mask,
                 neg_idx_img, neg_idx_txt, temp, image_embeds=None):
        """Teacher ITM match distribution over the SAME 3B triplets the student built.
        enc_input_ids/attention_mask come from the student (identical tokenizer, pos-0
        already set to enc_token_id). neg_idx_img/neg_idx_txt: length-B int sequences.
        Returns softmax(teacher_itm_logits / temp) as [3B, 2] (no grad)."""
        self._require("itm")
        device = image.device
        bs = image.size(0)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            if image_embeds is None:
                image_embeds = self.model.visual_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
            img_neg = torch.stack([image_embeds[neg_idx_img[b]] for b in range(bs)])
            txt_neg = torch.stack([enc_input_ids[neg_idx_txt[b]] for b in range(bs)])
            txt_neg_atts = torch.stack([attention_mask[neg_idx_txt[b]] for b in range(bs)])
            text_ids_all = torch.cat([enc_input_ids, enc_input_ids, txt_neg], dim=0)
            text_atts_all = torch.cat([attention_mask, attention_mask, txt_neg_atts], dim=0)
            image_embeds_all = torch.cat([image_embeds, img_neg, image_embeds], dim=0)
            image_atts_all = torch.cat([image_atts, image_atts, image_atts], dim=0)
            out = self.model.text_encoder(text_ids_all,
                                          attention_mask=text_atts_all,
                                          encoder_hidden_states=image_embeds_all,
                                          encoder_attention_mask=image_atts_all,
                                          return_dict=True)
            logits = self.model.itm_head(out.last_hidden_state[:, 0, :]).float()
        return F.softmax(logits / temp, dim=1)
```

- [ ] **Step 5: 테스트 통과 확인**

```bash
/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest distillation/test_online_teacher.py -v
```
Expected: 전체 pass (기존 케이스 + 신규 5개), 0 fail.

- [ ] **Step 6: 전체 회귀 스위트 실행**

```bash
/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest distillation/ models/test_blip_pretrain_itm.py -q
```
Expected: `72 passed` (기존 67 + 신규 5).

- [ ] **Step 7: 커밋**

```bash
git add distillation/online_teacher.py distillation/test_online_teacher.py
git commit -m "perf(teacher): add encode_image() + optional image_embeds reuse on itc_feats/lm_logits/itm_soft"
```

---

### Task 2: `pretrain.py` / `blip_pretrain.py` 배선 — 스텝당 한 번만 계산해서 재사용

**Files:**
- Create: `distillation/distill_config.py` (함수 추가, 파일은 기존 존재)
- Modify: `distillation/distill_config.py`
- Test: `distillation/test_distill_config.py`
- Modify: `pretrain.py:80-165` (train() 상단 + 루프 내 online teacher 호출부), `pretrain.py:173-176` (itm_mix 딕셔너리)
- Modify: `models/blip_pretrain.py:517-520` (itm_soft 호출부)
- Test: `models/test_blip_pretrain_itm.py`
- Create: `tests/test_pretrain_teacher_embeds_wiring.py`

**Interfaces:**
- Consumes: Task 1의 `OnlineTeacher.encode_image()`, `image_embeds=` 파라미터.
- Produces: `distillation.distill_config.need_teacher_image_embeds(need_teacher_itc, lm_kd_enabled, itm_mix_enabled, itm_soft_weight) -> bool` (신규 순수함수).
- Produces: `itm_mix` 딕셔너리에 신규 키 `'teacher_image_embeds'` — 이후 어떤 코드도 이 키를 지운 채 `itm_mix`를 넘기면 안 됨(항상 채워지거나 `None`).

- [ ] **Step 1: 실패하는 테스트부터 작성 — `need_teacher_image_embeds()`**

`distillation/test_distill_config.py` 끝(`if __name__ == "__main__":` 앞)에 아래 클래스를 추가:

```python
class TestNeedTeacherImageEmbeds(unittest.TestCase):
    def test_all_false(self):
        self.assertFalse(need_teacher_image_embeds(False, False, False, 0.0))

    def test_itc_alone(self):
        self.assertTrue(need_teacher_image_embeds(True, False, False, 0.0))

    def test_lm_alone(self):
        self.assertTrue(need_teacher_image_embeds(False, True, False, 0.0))

    def test_itm_mix_enabled_but_soft_weight_zero(self):
        self.assertFalse(need_teacher_image_embeds(False, False, True, 0.0))

    def test_itm_mix_enabled_with_soft_weight(self):
        self.assertTrue(need_teacher_image_embeds(False, False, True, 0.4))

    def test_all_three(self):
        self.assertTrue(need_teacher_image_embeds(True, True, True, 0.4))
```

파일 상단의 import 줄을 아래로 교체:
```python
from distillation.distill_config import (
    derive_teacher_keep, validate_itm_mix_config, need_teacher_image_embeds)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest distillation/test_distill_config.py -v -k NeedTeacherImageEmbeds
```
Expected: `ImportError` (아직 함수 없음) → 전체 6개 FAIL 또는 수집 단계 에러.

- [ ] **Step 3: `distillation/distill_config.py`에 함수 추가**

파일 끝에 추가:

```python


def need_teacher_image_embeds(need_teacher_itc, lm_kd_enabled, itm_mix_enabled, itm_soft_weight):
    """Whether this training step needs the teacher's image_embeds for ANY of
    itc_feats/lm_logits/itm_soft — decides whether OnlineTeacher.encode_image()
    should be called once this step (see pretrain.py train loop)."""
    return bool(need_teacher_itc or lm_kd_enabled or (itm_mix_enabled and itm_soft_weight > 0))
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest distillation/test_distill_config.py -v
```
Expected: 전체 pass (기존 13 + 신규 6 = 19), 0 fail.

- [ ] **Step 5: `pretrain.py` 배선 — import 및 루프 내 호출부 수정**

`pretrain.py` 상단의 `from distillation.distill_config import derive_teacher_keep, validate_itm_mix_config`
줄(414번대, `main()` 함수 안)은 그대로 두고, 대신 파일 최상단 import 블록에 아래를 추가한다(다른 `from distillation...` import가 없다면 새 줄로):

```python
from distillation.distill_config import need_teacher_image_embeds
```

`train()` 함수 안, 루프 내부의 아래 블록(현재 `pretrain.py:149-165`):

```python
        # online teacher: same augmented batch -> ITC features (bf16, no_grad). None when not needed.
        # target-mix: teacher only while γ>0 (γ=0 tail is target-identical to baseline → skip teacher forward)
        need_teacher_itc = (
            itc_kd_enabled
            or (ttm_enabled and gamma is not None and gamma > 0)
            or (itm_mix_enabled and itm_neg_source == 'teacher')
        )
        if need_teacher_itc and online_teacher is not None:
            teacher_img_feat, teacher_text_feat = online_teacher.itc_feats(image, caption)
        else:
            teacher_img_feat = teacher_text_feat = None

        # online teacher: same augmented batch -> LM decoder logits (bf16, no_grad). None when distill off.
        if lm_kd_enabled and online_teacher is not None:
            teacher_lm_logits, teacher_lm_ids = online_teacher.lm_logits(image, caption)
        else:
            teacher_lm_logits = teacher_lm_ids = None
```

를 아래로 교체:

```python
        # online teacher: same augmented batch -> ITC features (bf16, no_grad). None when not needed.
        # target-mix: teacher only while γ>0 (γ=0 tail is target-identical to baseline → skip teacher forward)
        need_teacher_itc = (
            itc_kd_enabled
            or (ttm_enabled and gamma is not None and gamma > 0)
            or (itm_mix_enabled and itm_neg_source == 'teacher')
        )
        # perf: compute the teacher's image_embeds at most ONCE per step and reuse it
        # across itc_feats/lm_logits/itm_soft (each would otherwise re-run visual_encoder).
        need_embeds = need_teacher_image_embeds(need_teacher_itc, lm_kd_enabled,
                                                itm_mix_enabled, itm_soft_weight)
        teacher_image_embeds = (
            online_teacher.encode_image(image)
            if (need_embeds and online_teacher is not None) else None
        )

        if need_teacher_itc and online_teacher is not None:
            teacher_img_feat, teacher_text_feat = online_teacher.itc_feats(
                image, caption, image_embeds=teacher_image_embeds)
        else:
            teacher_img_feat = teacher_text_feat = None

        # online teacher: same augmented batch -> LM decoder logits (bf16, no_grad). None when distill off.
        if lm_kd_enabled and online_teacher is not None:
            teacher_lm_logits, teacher_lm_ids = online_teacher.lm_logits(
                image, caption, image_embeds=teacher_image_embeds)
        else:
            teacher_lm_logits = teacher_lm_ids = None
```

그리고 `itm_mix` 딕셔너리 구성부(현재 `pretrain.py:173-176`):

```python
        itm_mix = None
        if itm_mix_enabled:
            itm_mix = {'neg_source': itm_neg_source, 'soft_weight': itm_soft_weight,
                       'temp': itm_teacher_temp, 'sel_scale': itm_sel_scale}
```

를 아래로 교체:

```python
        itm_mix = None
        if itm_mix_enabled:
            itm_mix = {'neg_source': itm_neg_source, 'soft_weight': itm_soft_weight,
                       'temp': itm_teacher_temp, 'sel_scale': itm_sel_scale,
                       'teacher_image_embeds': teacher_image_embeds}
```

- [ ] **Step 6: 실패하는 테스트 작성 — `blip_pretrain.py`가 `itm_mix['teacher_image_embeds']`를 실제로 전달하는지**

`models/test_blip_pretrain_itm.py`의 `TestForwardItmMix` 클래스에 아래 메서드 추가(기존 `test_teacher_neg_requires_feats` 다음):

```python
    def test_teacher_image_embeds_threaded_into_itm_soft(self):
        teacher = OnlineTeacher(checkpoint="", image_size=224, vit="base",
                                bert="base", queue_size=240, keep=("itm",))
        precomputed_embeds = teacher.encode_image(self.image)
        captured = {}
        real_itm_soft = teacher.itm_soft
        def spy(*args, **kwargs):
            captured['image_embeds'] = kwargs.get('image_embeds')
            return real_itm_soft(*args, **kwargs)
        itm_mix = {'neg_source': 'student', 'soft_weight': 0.4, 'temp': 1.0, 'sel_scale': 14.29,
                   'teacher_image_embeds': precomputed_embeds}
        with unittest.mock.patch.object(teacher, 'itm_soft', side_effect=spy):
            self._forward(online_teacher=teacher, itm_mix=itm_mix)
        self.assertIs(captured['image_embeds'], precomputed_embeds)
```

- [ ] **Step 7: 테스트 실패 확인**

```bash
/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest models/test_blip_pretrain_itm.py -v -k teacher_image_embeds_threaded
```
Expected: FAIL — `blip_pretrain.py`의 `itm_soft` 호출부가 아직 `image_embeds`를 안 넘겨서
`captured['image_embeds']`가 `None`이 되고, `assertIs(None, precomputed_embeds)`가 실패한다.

- [ ] **Step 8: `models/blip_pretrain.py`의 `itm_soft` 호출부 수정**

현재 `models/blip_pretrain.py:517-520`:

```python
        if itm_mix is not None and itm_mix['soft_weight'] > 0 and online_teacher is not None:
            teacher_soft = online_teacher.itm_soft(
                image, encoder_input_ids, text.attention_mask,
                neg_idx_img, neg_idx_txt, itm_mix['temp'])                 # [3B, 2], no grad
```

를 아래로 교체:

```python
        if itm_mix is not None and itm_mix['soft_weight'] > 0 and online_teacher is not None:
            teacher_soft = online_teacher.itm_soft(
                image, encoder_input_ids, text.attention_mask,
                neg_idx_img, neg_idx_txt, itm_mix['temp'],
                image_embeds=itm_mix.get('teacher_image_embeds'))          # [3B, 2], no grad
```

- [ ] **Step 9: 테스트 통과 확인**

```bash
/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest models/test_blip_pretrain_itm.py -v -k teacher_image_embeds_threaded
```
Expected: PASS.

- [ ] **Step 10: 정적 배선 검증 테스트 작성 (`pretrain.train()` 소스가 실제로 encode_image를 스텝당 1곳에서만 호출·전달하는지)**

새 파일 `tests/test_pretrain_teacher_embeds_wiring.py` 생성 (기존 `tests/test_pretrain_retrieval_wiring.py`와 동일한 정적-검사 스타일 — `train()`은 실제 데이터로더 없이는 호출하기 무거워 이 프로젝트에서 이미 이 방식을 씀):

```python
import inspect
import unittest

import pretrain


class PretrainTeacherEmbedsWiringTest(unittest.TestCase):
    def test_encode_image_called_exactly_once_in_source(self):
        source = inspect.getsource(pretrain.train)
        self.assertEqual(source.count("online_teacher.encode_image("), 1)

    def test_encode_image_result_threaded_into_itc_and_lm(self):
        # whitespace/line-break independent: just confirm the same encode_image()
        # result feeds both calls (exactly 2 occurrences: itc_feats + lm_logits) and
        # that both call sites are present.
        source = inspect.getsource(pretrain.train)
        self.assertEqual(source.count("image_embeds=teacher_image_embeds"), 2)
        self.assertIn("online_teacher.itc_feats(", source)
        self.assertIn("online_teacher.lm_logits(", source)

    def test_encode_image_result_threaded_into_itm_mix_dict(self):
        source = inspect.getsource(pretrain.train)
        self.assertIn("'teacher_image_embeds': teacher_image_embeds", source)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 11: 테스트 통과 확인**

```bash
/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_pretrain_teacher_embeds_wiring.py -v
```
Expected: 3개 다 PASS. FAIL하면 Step 5에서 `image_embeds=teacher_image_embeds`가 `itc_feats`/`lm_logits` 호출부 양쪽에 정확히 들어갔는지, `online_teacher.encode_image(image)` 호출이 정확히 한 번인지 확인.

- [ ] **Step 12: 문법 확인 + 전체 회귀 스위트**

`tests/` 디렉토리 전체(`pytest tests/`)는 이번 태스크와 무관한 **사전 존재하는 CUDA 전용
실패 2건**(`test_eval_validation_tool_retrieval.py::EvaluateRetrievalAmpCudaIntegrationTest`,
이 브랜치의 Task 2 작업 시작 전부터 이미 실패 상태 — GPU/CUDA 환경 의존, 이 플랜의 변경과
무관)이 있으므로, `tests/` 전체가 아니라 이번 작업이 실제로 건드린 두 파일만 지정한다:

```bash
/home/minwoo/miniconda3/envs/kd_r4/bin/python -c "import ast; ast.parse(open('pretrain.py').read()); ast.parse(open('models/blip_pretrain.py').read()); print('OK')"
/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest distillation/ models/test_blip_pretrain_itm.py \
  tests/test_pretrain_retrieval_wiring.py tests/test_pretrain_teacher_embeds_wiring.py -q
```
Expected: `OK`, 그리고 `85 passed`(Task 1 이후 baseline 75(= distillation/ + models/test_blip_pretrain_itm.py
72 + tests/test_pretrain_retrieval_wiring.py 기존 3) + 이번 태스크 신규 10(NeedTeacherImageEmbeds 6 +
threaded_into_itm_soft 1 + test_pretrain_teacher_embeds_wiring.py 3) = 85), 0 fail.

- [ ] **Step 13: 커밋**

```bash
git add distillation/distill_config.py distillation/test_distill_config.py \
        pretrain.py models/blip_pretrain.py models/test_blip_pretrain_itm.py \
        tests/test_pretrain_teacher_embeds_wiring.py
git commit -m "perf(teacher): thread single per-step image_embeds through pretrain.py + itm_soft call site"
```

---

### Task 3: 실 GPU 스모크로 최종 확인

**Files:**
- 없음(코드 변경 없음, 실행 검증만 — Phase 1의 `configs/pretrain_itc_itm_lm_targetmix_smoke.yaml` 그대로 재사용)

**Interfaces:**
- Consumes: Task 1+2가 완성한 배선. `configs/pretrain_itc_itm_lm_targetmix_smoke.yaml`(Phase 1에서 이미 커밋됨, 3개 메커니즘 전부 ON).
- Produces: 없음(Phase 2 최종 검증). 실패 시 Task 1/2로 돌아가 원인 파악.

- [ ] **Step 1: 여유 GPU 확인**

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
```
여유 있는 GPU 번호 확인 — 로컬이 계속 포화 상태면 스페어 서버에서 진행(Phase 1 Task 5와 동일 패턴: `git fetch origin && git checkout worktree-itc_itm_lm_targetmix`로 이 브랜치의 최신 커밋을 가져와서 실행).

- [ ] **Step 2: 몇 스텝만 실행**

```bash
CUDA_VISIBLE_DEVICES=<여유GPU번호> /home/minwoo/miniconda3/envs/kd_r4/bin/torchrun \
  --nproc_per_node=1 --master_port=29506 \
  -m pretrain --config=./configs/pretrain_itc_itm_lm_targetmix_smoke.yaml
```

- [ ] **Step 3: Phase 1과 동일한 통과 기준 + 이번 태스크 고유 확인**

Phase 1과 동일 기준(재확인):
- `[distill] online teacher loaded (keep=('itc', 'itm', 'lm'))` 라인 출현
- `loss_ita`/`loss_itm`/`loss_lm`/`loss_itc_kd`/`loss_lm_kd` 전부 finite, 3~5스텝 이상 무크래시

이번 태스크 고유 확인(정성적, 자동화된 벤치마크 아님):
- 스텝 로그의 `time:` 필드(스텝당 평균 시간)를 Phase 1 스모크 로그와 비교 — Phase 1 기록값: step 50~150 구간 `time: 0.7822`~`0.7838`. 이번엔 티처 `visual_encoder` 호출이 3회→1회로 줄었으니 그보다 낮게 나오는 게 기대치(정확한 감소폭은 티처 ViT-L 몫에 달려있어 사전에 숫자로 못박지 않음 — "눈에 띄게 줄었다/안 줄었다" 수준 판단으로 충분).

확인되면 Ctrl+C로 중단 — 전체 에폭 돌릴 필요 없음.

- [ ] **Step 4: 결과 기록**

결과(성공/실패, teacher_keep 로그, step time 비교)를 `.superpowers/sdd/progress.md`에 한 줄 요약으로 남긴다.

---

## Self-Review 메모 (계획 작성자용, 실행 불필요)

- 스펙 §3의 4개 항목(encode_image, 3개 메서드 optional 파라미터, pretrain.py 오케스트레이션, itm_mix 스레딩) ↔ 이 계획: encode_image+3개 메서드=Task 1, pretrain.py 오케스트레이션+itm_mix 스레딩+blip_pretrain.py 호출부=Task 2.
- 스펙 §4 테스트 계획의 "핵심 회귀 테스트"는 Task 1의 `test_visual_encoder_called_exactly_once_across_all_three`로 구현(OnlineTeacher 레벨, 메커니즘 자체의 증명) + Task 2의 정적 배선 테스트(pretrain.py가 실제로 그 메커니즘을 스텝당 1번만 쓰도록 배선했다는 증명) — 두 레벨로 나눠 완전성 확보. `pretrain.train()`을 실제 호출하는 동적 테스트는 이 프로젝트에 기존 선례가 없고(`tests/test_pretrain_retrieval_wiring.py`도 정적 검사만 함) 페이크 DataLoader/DDP 우회 등 복잡도가 과함 — 대신 Task 3의 실 GPU 스모크가 최종 동적 확인을 담당.
- 플레이스홀더 없음, 모든 코드는 현재 worktree(base 5912b65)의 실제 파일 내용을 직접 읽고 정확한 줄 단위로 대조해 작성함.
