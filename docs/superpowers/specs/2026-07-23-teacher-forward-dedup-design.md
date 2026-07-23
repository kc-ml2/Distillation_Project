# 티처 visual_encoder 중복 호출 제거 — 설계 (Phase 2)

날짜: 2026-07-23
브랜치: `worktree-itc_itm_lm_targetmix` 위에 이어서 (base 5912b65, Phase 1 완료 지점)

## 1. 배경

Phase 1(ITC target-mix + ITM target-mix + LM distillation 구조적 병합)의 최종 브랜치 리뷰에서
확인된 실측 낭비 지점: `OnlineTeacher`의 `itc_feats()`, `lm_logits()`, `itm_soft()` 세 메서드가
(`distillation/online_teacher.py:88/106/131`) 각각 독립적으로 `self.model.visual_encoder(image)`를
호출한다. 셋 다 완전히 동일한 `image` 텐서를 입력받고, 티처는 frozen·`.eval()`·`@torch.no_grad()`라
결과도 항상 동일하다 — 즉 두 번째, 세 번째 호출은 100% 순수 낭비.

**실측된 중복 정도**:
- exp9(ITC target-mix) + LM distill 조합(병합 이전에도 이미 존재하던 조합): 스텝당 2회.
- exp10(ITM target-mix) arm C(neg_source=teacher, soft_weight>0) 단독: 스텝당 2회.
- 이번 Phase 1 병합 config(`configs/pretrain_itc_itm_lm_targetmix_smoke.yaml`, 세 메커니즘 전부 ON):
  **스텝당 3회** — 이번 병합으로 처음 가능해진 최악의 경우.

ViT-L(티처 vision encoder) forward는 이 세 메서드에서 가장 비싼 연산이므로, 이 중복을 없애면
셋 다 켜진 조합에서 티처 쪽 GPU 시간이 이론상 1/3로 준다.

**범위 밖**: `loss_itc_kd`가 `itc_kd_enabled=False`여도 `teacher_img_feat`만 있으면 매 스텝
계산되는 낭비(`models/blip_pretrain.py:543-549`)는 이번 Phase 2에 포함하지 않는다 — B×B KL
연산이라 ViT-L forward 대비 비용이 작고, 고치려면 `forward()` 시그니처에 플래그를 하나 더
추가해야 해서(이미 파라미터 13개로 늘어난 상태) 비용 대비 이득이 낮다고 판단해 보류.

## 2. 검토한 접근과 선택 근거

- **A. `OnlineTeacher` 내부 스텝 캐시**(텐서 identity 비교로 자동 재사용): 호출부 무변경이 장점이나,
  캐시 히트/미스가 코드에 안 보이는 암묵적 상태(hidden state)가 됨 — 기각.
- **C. API 전체 재구성**(단일 `encode()`가 필요한 걸 한 번에 다 계산): `itm_soft`가 필요로 하는
  `neg_idx_img`/`neg_idx_txt`는 학생 `forward()` 내부, 학생 자신의 유사도 기반 네거티브
  마이닝이 끝난 뒤에야 나온다 — 즉 `itm_soft`는 구조적으로 스텝 초반에 다른 것들과 한 번에
  묶어 계산할 수 없다. "완전 통합"이 애초에 불가능하므로, 기존 15개+ 테스트가 의존하는
  3개 메서드 시그니처를 갈아엎을 이유가 없음 — 기각.
- **B'. (채택) 명시적 신규 프리미티브 + 기존 시그니처에 optional 파라미터 추가**: A의 "숨겨진
  캐시" 문제와 C의 "깨지는 재구성" 문제를 모두 피하면서, C가 노리던 "미래에도 재사용 가능한
  단일 진입점"은 `encode_image()`로 그대로 확보. 기존 호출부·기존 테스트는 새 파라미터의
  기본값(`None`)으로 인해 100% 무변경 통과.

## 3. 설계

### 3.1 `OnlineTeacher.encode_image()` — 신규, 상태 없음

```python
@torch.no_grad()
def encode_image(self, image):
    device = image.device
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        return self.model.visual_encoder(image)
```

캐시가 아니다 — 부를 때마다 매번 새로 계산한다. "중복 계산 방지"는 이 메서드 내부가 아니라
**호출자가 스텝당 몇 번 부르느냐**에서 결정된다(§3.3).

### 3.2 기존 3개 메서드에 `image_embeds=None` 파라미터 추가

```python
@torch.no_grad()
def itc_feats(self, image, caption, image_embeds=None):
    self._require("itc")
    device = image.device
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        if image_embeds is None:
            image_embeds = self.model.visual_encoder(image)
        img_feat = F.normalize(self.model.vision_proj(image_embeds[:, 0, :]), dim=-1)
        # ... 이하 기존과 동일 (텍스트 인코딩 부분 무변경)

@torch.no_grad()
def lm_logits(self, image, caption, image_embeds=None):
    self._require("lm")
    device = image.device
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        if image_embeds is None:
            image_embeds = self.model.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
        # ... 이하 기존과 동일

@torch.no_grad()
def itm_soft(self, image, enc_input_ids, attention_mask,
             neg_idx_img, neg_idx_txt, temp, image_embeds=None):
    self._require("itm")
    device = image.device
    bs = image.size(0)
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        if image_embeds is None:
            image_embeds = self.model.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
        # ... 이하 기존과 동일
```

**하위호환 불변식**: `image_embeds`를 안 넘기면(기본값 `None`) 기존 15개+ 테스트가 부르는
방식과 동작이 100% 동일 — 이 세 메서드의 기존 테스트는 한 줄도 안 고친다.

### 3.3 `pretrain.py` 오케스트레이션 — 스텝당 한 번만 계산

```python
need_teacher_embeds = need_teacher_itc or lm_kd_enabled or (itm_mix_enabled and itm_soft_weight > 0)
teacher_image_embeds = (
    online_teacher.encode_image(image)
    if (need_teacher_embeds and online_teacher is not None) else None
)

if need_teacher_itc and online_teacher is not None:
    teacher_img_feat, teacher_text_feat = online_teacher.itc_feats(
        image, caption, image_embeds=teacher_image_embeds)
else:
    teacher_img_feat = teacher_text_feat = None

if lm_kd_enabled and online_teacher is not None:
    teacher_lm_logits, teacher_lm_ids = online_teacher.lm_logits(
        image, caption, image_embeds=teacher_image_embeds)
else:
    teacher_lm_logits = teacher_lm_ids = None

itm_mix = None
if itm_mix_enabled:
    itm_mix = {'neg_source': itm_neg_source, 'soft_weight': itm_soft_weight,
               'temp': itm_teacher_temp, 'sel_scale': itm_sel_scale,
               'teacher_image_embeds': teacher_image_embeds}
```

`teacher_image_embeds`가 `itm_mix` 딕셔너리에 들어가는 이유: `forward()`는 이미 파라미터가
13개(`image, caption, alpha, update_train_state, teacher_img_feat, teacher_text_feat,
distill_temp, teacher_lm_logits, teacher_lm_input_ids, lm_distill_temp, gamma, online_teacher,
itm_mix`)라 새 단독 파라미터를 또 추가하기보다, 이미 "스텝마다 계산해서 itm_mix에 넣는 값"
선례(`sel_scale`)가 있는 이 딕셔너리에 끼워 넣는 게 일관적이다.

### 3.4 `models/blip_pretrain.py`의 `itm_soft` 호출부 (forward() 내부)

```python
if itm_mix is not None and itm_mix['soft_weight'] > 0 and online_teacher is not None:
    teacher_soft = online_teacher.itm_soft(
        image, encoder_input_ids, text.attention_mask, neg_idx_img, neg_idx_txt,
        itm_mix['temp'], image_embeds=itm_mix.get('teacher_image_embeds'))
    loss_itm = itm_target_mix_loss(vl_output, itm_labels, teacher_soft.to(image.device),
                                   itm_mix['soft_weight'])
```

(기존 호출 형태에 `image_embeds=itm_mix.get('teacher_image_embeds')` 인자만 추가.)

### 3.5 결과

이번 Phase 1 병합 config(세 메커니즘 전부 ON) 기준, 스텝당 티처 `visual_encoder` 호출이
**3회 → 1회**로 감소. `itc_feats`/`lm_logits`/`itm_soft` 각각의 로직·리턴값은 완전히 동일하게
유지된다(동일 입력 → 동일 출력, 순수 계산 재사용일 뿐 로직 변경 없음).

## 4. 테스트 계획

- **`encode_image()` 신규 단위테스트**: shape(`(B, patches+1, hidden)`), `requires_grad=False`,
  bf16 dtype, 기존 `itc_feats`가 내부에서 계산하는 `image_embeds`와 동일한 값을 반환하는지
  (같은 입력 대해 `itc_feats`의 초기 부분과 수치 일치 확인).
- **`itc_feats`/`lm_logits`/`itm_soft` 각각**: `image_embeds` 미지정 시 기존 테스트 전부
  무변경 통과(회귀) + `image_embeds` 지정 시 (a) `self.model.visual_encoder`에 spy를 걸어
  **호출 안 됨**을 확인 (b) 반환값이 `image_embeds` 미지정 호출과 수치적으로 동일함을 확인
  (재사용이 결과를 안 바꾼다는 등가성 보장).
- **핵심 회귀 테스트 (가장 중요)**: `self.model.visual_encoder`에 spy를 걸고, 세 메커니즘
  전부 켜진 상황(itc_target_mix + itm_target_mix arm C + lm distill, Phase 1 스모크와 동일
  조합)을 스텝 단위로 시뮬레이션 — `pretrain.py`의 오케스트레이션 로직을 통해
  `encode_image()`가 스텝당 **정확히 1번**만 호출됐음을 직접 단언(assert call count == 1).
  이 테스트가 "최적화가 실제로 됐다"를 증명하는 유일한 테스트.
- **회귀**: Phase 1의 기존 전체 스위트(67개)가 그대로 통과하는지 확인.
- **(선택) 실측 GPU 스모크**: Phase 1의 `configs/pretrain_itc_itm_lm_targetmix_smoke.yaml`을
  재사용해 몇 스텝 돌리고, 스텝 시간(`time:` 필드)이 Phase 1 스모크 로그 대비 유의미하게
  줄었는지 육안 비교(자동화된 벤치마크는 아님 — 정성적 확인 수준).

## 5. 범위 밖 (Phase 3 혹은 별도로 미룸)

- `loss_itc_kd` 무조건 계산 낭비 (§1에서 언급, 비용 작아 보류).
- 실제 멀티에폭 본런.

**`itc_target_mix` variant는 `queue`로 확정** — 앞으로의 config·실험은 전부 `queue`를 쓴다.
`in_batch`(variant D)는 폐기하진 않지만 "예전에 그런 옵션도 만들어봤다" 정도로만 코드/config에
남겨두고, 이후 별도로 비교·선택할 계획은 없다.
