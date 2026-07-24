# ITM 증류 스킴 A/B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ITM 헤드 증류의 두 변형(스킴 A=티처만 temper한 soft-target, 스킴 B=2텀 Hinton KD)을 추가해, arm C(확률 target-mix)가 붕괴시킨 match-vs-hardneg separation을 통제된 방식으로 회복 시도한다.

**Architecture:** 스킴 A는 기존 `itm_target_mix_loss`를 `soft_weight=1.0/temp=2.0`으로 쓰는 **config-only**. 스킴 B는 새 손실 `itm_hinton_kd_loss`(`lm_distill_loss`와 동일한 KL·×T² 관례) + `itm_target_mix` config의 `variant` 필드로 opt-in. `variant` 미지정 시 기존 arm A/B/C 경로 **바이트 동일**.

**Tech Stack:** PyTorch, `distillation/losses.py`, `distillation/distill_config.py`, `models/blip_pretrain.py`(ITM forward), `pretrain.py`(itm_mix 조립), unittest. 실행 env: conda `kd_r4`.

## Global Constraints

- `variant` 기본값은 `'target_mix'` — 미지정 config(arm A/B/C)는 기존 손실 경로로 회귀 없이 동작.
- `soft_weight` ∈ [0,1] 는 "티처 신뢰도" 단일 다이얼: `target_mix`면 W, `hinton_kd`면 α로 해석(새 필드 없음).
- `itm_hinton_kd_loss`의 `teacher_soft`는 **반드시 동일 temp로 tempered**(=`OnlineTeacher.itm_soft(temp=T)`가 반환하는 `softmax(teacher_logits/T)`) — ×T² Hinton 스케일 성립 조건.
- 새 실험 config는 arm C 복제 기반, `neg_source: teacher`, `temp: 2.0` 고정.
- 테스트는 기존 `unittest` 컨벤션(seeded, 독립 manual reference) 따름.
- 커밋 메시지는 저장소 관례(`feat(...)`/`test(...)`)를 따르고, 각 Task 끝에서 커밋.

---

### Task 1: 스킴 A config (config-only, 새 코드 없음)

**Files:**
- Create: `configs/pretrain_itm_schemeA_tempered.yaml`
- Test: `distillation/test_scheme_configs.py`

**Interfaces:**
- Consumes: 기존 `validate_itm_mix_config`, `derive_teacher_keep`, `itm_target_mix_loss`(변경 없음).
- Produces: 실험 config 파일 1개. 후속 Task와 코드 의존 없음.

- [ ] **Step 1: Write the failing test**

`distillation/test_scheme_configs.py` (신규):
```python
import unittest
import yaml
from distillation.distill_config import validate_itm_mix_config, derive_teacher_keep


def _distill(path):
    with open(path) as f:
        return yaml.safe_load(f)['distill']


class TestSchemeAConfig(unittest.TestCase):
    PATH = 'configs/pretrain_itm_schemeA_tempered.yaml'

    def test_validates(self):
        validate_itm_mix_config(_distill(self.PATH))            # no raise

    def test_itm_block_values(self):
        itm = _distill(self.PATH)['itm_target_mix']
        self.assertTrue(itm['enabled'])
        self.assertEqual(itm['neg_source'], 'teacher')
        self.assertEqual(float(itm['soft_weight']), 1.0)        # W=1 → 순수 tempered-teacher
        self.assertEqual(float(itm['temp']), 2.0)               # T=2
        self.assertEqual(itm.get('variant', 'target_mix'), 'target_mix')

    def test_teacher_keep(self):
        # teacher neg + soft_weight>0 → itc(선택) + itm(스코어)
        self.assertEqual(derive_teacher_keep(_distill(self.PATH)), ('itc', 'itm'))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n kd_r4 python -m pytest distillation/test_scheme_configs.py -q`
Expected: FAIL — `FileNotFoundError: configs/pretrain_itm_schemeA_tempered.yaml`.

- [ ] **Step 3: Create the config**

`configs/pretrain_itm_schemeA_tempered.yaml` — `configs/pretrain_itm_C_teachneg_soft.yaml`를 복사하고 아래만 변경:
- `output_dir: '/home/minwoo/Distillation_Project/output/pt_itm_schemeA_tempered'`
- `exp: '10.SA_itm_tempered_soft'`
- `distill.itm_target_mix.soft_weight: 1.0`   (기존 0.4)
- `distill.itm_target_mix.temp: 2.0`          (기존 1.0)
- 헤더 주석에 스킴 A 근거(critical_bugfix 문서 참조) 한 줄.
나머지 키(neg_source: teacher, schedule: constant, teacher.checkpoint 등)는 arm C와 동일.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n kd_r4 python -m pytest distillation/test_scheme_configs.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add configs/pretrain_itm_schemeA_tempered.yaml distillation/test_scheme_configs.py
git commit -m "feat(itm-kd): scheme A config (tempered soft-target, W=1/T=2)"
```

---

### Task 2: `itm_hinton_kd_loss` 손실 (스킴 B 핵심, TDD)

**Files:**
- Modify: `distillation/losses.py` (파일 끝에 함수 추가)
- Test: `distillation/test_losses_itm.py` (클래스 추가)

**Interfaces:**
- Consumes: `torch`, `torch.nn.functional as F`(losses.py에 이미 import).
- Produces: `itm_hinton_kd_loss(vl_output, itm_labels, teacher_soft, alpha, temp) -> scalar Tensor`
  - `vl_output`: `[N,2]` student ITM logits. `itm_labels`: `[N]` long {0,1}.
  - `teacher_soft`: `[N,2]` = softmax(teacher_logits/temp), no grad.
  - `alpha`: float [0,1]. `temp`: float(T). 반환: `(1-alpha)*CE + alpha*T²*KL(teacher_soft‖σ(z/T))`.

- [ ] **Step 1: Write the failing tests**

`distillation/test_losses_itm.py`에 클래스 추가 (기존 import 라인에 `itm_hinton_kd_loss` 추가):
```python
from distillation.losses import itm_target_mix_loss, itm_hinton_kd_loss


class TestItmHintonKdLoss(unittest.TestCase):
    def test_alpha0_equals_cross_entropy(self):
        vl = torch.tensor([[2.0, -1.0], [0.5, 0.5], [-1.0, 3.0]])
        labels = torch.tensor([0, 1, 1])
        teacher = torch.softmax(torch.randn(3, 2), dim=1)
        got = itm_hinton_kd_loss(vl, labels, teacher, alpha=0.0, temp=2.0)
        self.assertTrue(torch.allclose(got, F.cross_entropy(vl, labels), atol=1e-6))

    def test_matches_manual_2term(self):
        vl = torch.tensor([[1.5, -0.5], [0.2, 0.9]])
        labels = torch.tensor([0, 1])
        teacher = torch.tensor([[0.3, 0.7], [0.4, 0.6]])
        alpha, T = 0.4, 2.0
        hard = F.cross_entropy(vl, labels)
        logq = F.log_softmax(vl / T, dim=1)
        kl = (teacher * (teacher.log() - logq)).sum(1).mean() * (T ** 2)  # batchmean·T²
        expected = (1 - alpha) * hard + alpha * kl
        got = itm_hinton_kd_loss(vl, labels, teacher, alpha, T)
        self.assertTrue(torch.allclose(got, expected, atol=1e-6))

    def test_gradient_finite_to_logits_only(self):
        vl = torch.randn(4, 2, requires_grad=True)
        teacher = torch.softmax(torch.randn(4, 2), dim=1)   # no grad
        labels = torch.tensor([0, 1, 0, 1])
        itm_hinton_kd_loss(vl, labels, teacher, alpha=0.4, temp=2.0).backward()
        self.assertIsNotNone(vl.grad)
        self.assertTrue(torch.isfinite(vl.grad).all())
        self.assertFalse(teacher.requires_grad)

    def test_returns_scalar(self):
        vl = torch.randn(3, 2)
        teacher = torch.softmax(torch.randn(3, 2), dim=1)
        loss = itm_hinton_kd_loss(vl, torch.tensor([0, 1, 1]), teacher, 0.4, 2.0)
        self.assertEqual(loss.dim(), 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n kd_r4 python -m pytest distillation/test_losses_itm.py -q`
Expected: FAIL — `ImportError: cannot import name 'itm_hinton_kd_loss'`.

- [ ] **Step 3: Implement the loss**

`distillation/losses.py` 끝(`itm_target_mix_loss` 다음)에 추가:
```python
def itm_hinton_kd_loss(vl_output, itm_labels, teacher_soft, alpha, temp):
    """2-term Hinton KD for ITM (스킴 B): 하드 CE 앵커 + tempered 티처 KL.

        loss = (1-alpha)*CE(z, itm_labels) + alpha * T^2 * KL(teacher_soft || softmax(z/T))

    vl_output   : [N, 2] student ITM logits (z).
    itm_labels  : [N]    long {0=no-match, 1=match}.
    teacher_soft: [N, 2] = softmax(teacher_logits / temp), no grad. ×T^2 스케일이 성립하려면
                  반드시 '동일 temp'로 tempered여야 한다 (OnlineTeacher.itm_soft(temp=T)).
    alpha       : soft 항 가중(α). α=0이면 순수 하드 CE.
    temp        : T. KL 항에서 student도 /T로 tempered.

    itm_target_mix(확률 믹싱)와 달리 하드 라벨을 target에 '녹이지' 않고 CE를 별도 항으로
    유지 → 갭 sharpening을 안 죽인다(균형 gap→teacher gap). lm_distill_loss의 KL·×T² 관례를 ITM에 적용.
    """
    hard = F.cross_entropy(vl_output, itm_labels)
    kd = F.kl_div(F.log_softmax(vl_output / temp, dim=1),
                  teacher_soft.to(vl_output.dtype),
                  reduction='batchmean') * (temp ** 2)
    return (1.0 - alpha) * hard + alpha * kd
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n kd_r4 python -m pytest distillation/test_losses_itm.py -q`
Expected: PASS (기존 4 + 신규 4 = 8 tests).

- [ ] **Step 5: Commit**

```bash
git add distillation/losses.py distillation/test_losses_itm.py
git commit -m "feat(itm-kd): itm_hinton_kd_loss (2-term Hinton KD) + unit tests"
```

---

### Task 3: `validate_itm_mix_config`에 variant 검증

**Files:**
- Modify: `distillation/distill_config.py:25-37` (`validate_itm_mix_config`)
- Test: `distillation/test_distill_config.py` (`TestValidateItmMixConfig`에 메서드 추가)

**Interfaces:**
- Consumes: 없음(순수 assert 함수).
- Produces: `validate_itm_mix_config`가 `variant in ('target_mix','hinton_kd')`(기본 target_mix)를 강제. 후속 Task/config가 의존.

- [ ] **Step 1: Write the failing tests**

`distillation/test_distill_config.py`의 `TestValidateItmMixConfig`에 추가:
```python
    def test_variant_target_mix_ok(self):
        validate_itm_mix_config({'itm_target_mix': {'enabled': True, 'neg_source': 'teacher',
                                                    'soft_weight': 0.4, 'variant': 'target_mix'}})

    def test_variant_hinton_kd_ok(self):
        validate_itm_mix_config({'itm_target_mix': {'enabled': True, 'neg_source': 'teacher',
                                                    'soft_weight': 0.4, 'variant': 'hinton_kd', 'temp': 2.0}})

    def test_missing_variant_defaults_ok(self):   # 레거시 arm A/B/C (variant 없음)
        validate_itm_mix_config({'itm_target_mix': {'enabled': True, 'neg_source': 'teacher',
                                                    'soft_weight': 0.4}})

    def test_bad_variant_raises(self):
        with self.assertRaises(AssertionError):
            validate_itm_mix_config({'itm_target_mix': {'enabled': True, 'neg_source': 'teacher',
                                                        'soft_weight': 0.4, 'variant': 'foo'}})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n kd_r4 python -m pytest distillation/test_distill_config.py::TestValidateItmMixConfig -q`
Expected: FAIL — `test_bad_variant_raises`가 실패(아직 variant='foo'를 안 막음).

- [ ] **Step 3: Implement the variant assert**

`distillation/distill_config.py`의 `validate_itm_mix_config`에서 `schedule` assert 앞에 추가:
```python
    variant = itm.get('variant', 'target_mix')
    assert variant in ('target_mix', 'hinton_kd'), \
        f"itm_target_mix.variant must be 'target_mix'|'hinton_kd', got {variant!r}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n kd_r4 python -m pytest distillation/test_distill_config.py -q`
Expected: PASS (기존 + 신규 4 전부).

- [ ] **Step 5: Commit**

```bash
git add distillation/distill_config.py distillation/test_distill_config.py
git commit -m "feat(itm-kd): validate itm_target_mix.variant (target_mix|hinton_kd)"
```

---

### Task 4: forward 분기 + pretrain.py 배선

**Files:**
- Modify: `models/blip_pretrain.py:22` (import), `models/blip_pretrain.py:516-526` (ITM forward 분기)
- Modify: `pretrain.py:112-116` (variant 읽기), `pretrain.py:185-187` (itm_mix dict)
- Test: `distillation/test_forward_kd.py` (클래스 추가)

**Interfaces:**
- Consumes: `itm_hinton_kd_loss`(Task 2), `itm_mix` dict에 `'variant'` 키.
- Produces: forward가 `itm_mix['variant']=='hinton_kd'`일 때 `itm_hinton_kd_loss`로 분기; 그 외/미지정은 `itm_target_mix_loss`(레거시 불변).

- [ ] **Step 1: Write the failing test**

`distillation/test_forward_kd.py`에 클래스 추가:
```python
import torch.nn.functional as F


class TestForwardItmHintonKd(unittest.TestCase):
    """variant='hinton_kd' forward 스모크 (mock 티처, neg_source=student로 티처-feat 배선 우회)."""

    class _MockTeacher:
        def itm_soft(self, image, enc_ids, att, neg_i, neg_t, temp, image_embeds=None):
            bs = image.size(0)
            return F.softmax(torch.randn(3 * bs, 2), dim=1)   # [3B, 2]

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = blip_pretrain(image_size=224, vit="base", my_bert_size="base", queue_size=240)
        cls.model.eval()
        cls.image = torch.randn(2, 3, 224, 224)
        cls.caption = ["a green field", "a red car"]

    def _itm_mix(self, variant):
        m = {'neg_source': 'student', 'soft_weight': 0.4, 'temp': 2.0,
             'sel_scale': 1.0, 'teacher_image_embeds': None}
        if variant is not None:
            m['variant'] = variant
        return m

    def _forward_loss_itm(self, variant, seed):
        # 동일 seed → 동일 neg 샘플(multinomial)·동일 mock teacher_soft(randn) → vl_output/teacher_soft 동일.
        # 그래서 두 forward의 차이는 오직 손실 함수(variant)뿐이다.
        torch.manual_seed(seed)
        out = self.model(self.image, self.caption, alpha=0.4, update_train_state=False,
                         online_teacher=self._MockTeacher(), itm_mix=self._itm_mix(variant))
        return out[1]

    def test_hinton_kd_finite_and_grad(self):
        loss_itm = self._forward_loss_itm('hinton_kd', seed=0)
        self.assertTrue(torch.isfinite(loss_itm).all())
        self.assertTrue(loss_itm.requires_grad)

    def test_hinton_differs_from_target_mix(self):
        # 분기가 실제로 동작해야만 통과 (fail-first): 배선 전엔 둘 다 target_mix라 동일 → 실패.
        l_hinton = self._forward_loss_itm('hinton_kd', seed=1)
        l_tmix = self._forward_loss_itm('target_mix', seed=1)
        self.assertFalse(torch.allclose(l_hinton, l_tmix, atol=1e-4))

    def test_missing_variant_equals_target_mix(self):
        # 레거시 불변: variant 미지정 == 'target_mix' (배선 전후 모두 통과하는 회귀 가드)
        l_none = self._forward_loss_itm(None, seed=2)
        l_tmix = self._forward_loss_itm('target_mix', seed=2)
        self.assertTrue(torch.allclose(l_none, l_tmix, atol=1e-6))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n kd_r4 python -m pytest distillation/test_forward_kd.py::TestForwardItmHintonKd -q`
Expected: FAIL — `test_hinton_differs_from_target_mix`가 실패. 배선 전엔 forward가 `variant` 키를 무시해 두 호출 모두 `itm_target_mix_loss`로 계산 → 동일 seed라 손실이 같아 `assertFalse(allclose)`가 깨진다. (`test_hinton_kd_finite_and_grad`·`test_missing_variant_equals_target_mix`는 이미 통과할 수 있음.)

- [ ] **Step 3: Implement wiring**

(a) `models/blip_pretrain.py:22` import 교체:
```python
from distillation.losses import itc_distill_loss, lm_distill_loss, itm_target_mix_loss, itm_hinton_kd_loss
```
(b) `models/blip_pretrain.py` ITM forward 분기(`itm_target_mix_loss` 호출부)를 교체:
```python
        # ITM loss: variant='hinton_kd'면 2텀 KD, 아니면(기본) target-mix. soft_weight=0이면 plain CE.
        if itm_mix is not None and itm_mix['soft_weight'] > 0 and online_teacher is not None:
            teacher_soft = online_teacher.itm_soft(
                image, encoder_input_ids, text.attention_mask,
                neg_idx_img, neg_idx_txt, itm_mix['temp'],
                image_embeds=itm_mix.get('teacher_image_embeds'))          # [3B, 2], no grad
            if itm_mix.get('variant', 'target_mix') == 'hinton_kd':
                loss_itm = itm_hinton_kd_loss(vl_output, itm_labels,
                                              teacher_soft.to(image.device),
                                              itm_mix['soft_weight'], itm_mix['temp'])
            else:
                loss_itm = itm_target_mix_loss(vl_output, itm_labels,
                                               teacher_soft.to(image.device),
                                               itm_mix['soft_weight'])
        else:
            loss_itm = F.cross_entropy(vl_output, itm_labels)
```
(c) `pretrain.py:112-116` 근처, `itm_teacher_temp` 읽는 줄 아래에 추가:
```python
    itm_variant = distill_itm.get('variant', 'target_mix')
```
(d) `pretrain.py:185-187` itm_mix dict에 variant 추가:
```python
            itm_mix = {'neg_source': itm_neg_source, 'soft_weight': itm_soft_weight,
                       'temp': itm_teacher_temp, 'sel_scale': itm_sel_scale,
                       'teacher_image_embeds': teacher_image_embeds,
                       'variant': itm_variant}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n kd_r4 python -m pytest distillation/test_forward_kd.py distillation/test_losses_itm.py -q`
Expected: PASS (forward 스모크 + itm 손실 회귀 전부). 이어서 diff로 (b)의 `variant=='hinton_kd'` 분기가 실제로 `itm_hinton_kd_loss`를 호출하는지 육안 확인.

- [ ] **Step 5: Commit**

```bash
git add models/blip_pretrain.py pretrain.py distillation/test_forward_kd.py
git commit -m "feat(itm-kd): wire itm_target_mix.variant=hinton_kd into forward + pretrain"
```

---

### Task 5: 스킴 B config + 균형점 실증

**Files:**
- Create: `configs/pretrain_itm_schemeB_hinton.yaml`
- Modify: `distillation/test_scheme_configs.py` (클래스 추가)
- Run: `critical_bugfix/2026-07-24_itm_distill_sharpness/verify_scheme_equilibria.py` (이미 존재; B 경로가 이제 활성)

**Interfaces:**
- Consumes: variant 검증(Task 3), `itm_hinton_kd_loss`(Task 2).
- Produces: 스킴 B 실험 config + verify 스크립트의 B 균형 수치.

- [ ] **Step 1: Write the failing test**

`distillation/test_scheme_configs.py`에 추가:
```python
class TestSchemeBConfig(unittest.TestCase):
    PATH = 'configs/pretrain_itm_schemeB_hinton.yaml'

    def test_validates(self):
        validate_itm_mix_config(_distill(self.PATH))

    def test_itm_block_values(self):
        itm = _distill(self.PATH)['itm_target_mix']
        self.assertTrue(itm['enabled'])
        self.assertEqual(itm['neg_source'], 'teacher')
        self.assertEqual(itm['variant'], 'hinton_kd')
        self.assertEqual(float(itm['temp']), 2.0)
        self.assertGreater(float(itm['soft_weight']), 0.0)      # α>0

    def test_teacher_keep(self):
        self.assertEqual(derive_teacher_keep(_distill(self.PATH)), ('itc', 'itm'))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n kd_r4 python -m pytest distillation/test_scheme_configs.py::TestSchemeBConfig -q`
Expected: FAIL — `FileNotFoundError: configs/pretrain_itm_schemeB_hinton.yaml`.

- [ ] **Step 3: Create the config**

`configs/pretrain_itm_schemeB_hinton.yaml` — `configs/pretrain_itm_C_teachneg_soft.yaml` 복사 후 변경:
- `output_dir: '/home/minwoo/Distillation_Project/output/pt_itm_schemeB_hinton'`
- `exp: '10.SB_itm_hinton_kd'`
- `distill.itm_target_mix` 블록:
  ```yaml
  itm_target_mix:
    enabled: true
    neg_source: teacher
    variant: hinton_kd     # 2텀 KD
    soft_weight: 0.4       # α (soft 항 가중)
    temp: 2.0              # T
    schedule: constant
  ```
- 헤더 주석에 스킴 B 근거(critical_bugfix 문서) 한 줄.

- [ ] **Step 4: Run test + 균형 검증**

Run: `conda run -n kd_r4 python -m pytest distillation/test_scheme_configs.py -q`
Expected: PASS (스킴 A + B 전부).

Run: `conda run -n kd_r4 python critical_bugfix/2026-07-24_itm_distill_sharpness/verify_scheme_equilibria.py`
Expected: `[B]` 줄이 "미구현"이 아니라 실수치 출력. 확인 포인트 — B는 하드 CE 앵커가 있어 하드네거 gap이 스킴 A(sep 1.65)·arm C(target sep 9.1/실측 0.64)와 구분되는 값으로 수렴(match gap이 티처(7.8) 쪽으로 더 큼).

- [ ] **Step 5: Commit**

```bash
git add configs/pretrain_itm_schemeB_hinton.yaml distillation/test_scheme_configs.py
git commit -m "feat(itm-kd): scheme B config (hinton_kd, alpha=0.4/T=2) + equilibrium check"
```

---

## 최종 검증 (전체 스위트)

```bash
conda run -n kd_r4 python -m pytest distillation/ -q
```
Expected: 전부 PASS(기존 arm A/B/C 테스트 포함 회귀 없음).

## 실험 인계 (사용자 담당, 플랜 범위 밖)

`itm_sharpness_probe.py`에 스킴 A/B 체크포인트를 추가해, 학습 몇 에폭 후 **separation + val_retrieval_itm**을 arm B(하드)·arm C(사망) 대비 비교. **게이트: itm − itc > 0 회복 + itc 유지.**
