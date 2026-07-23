# LM Logit Distillation — 설계 (Design)

- **날짜:** 2026-07-02
- **브랜치:** `lm_distill` (신규 — itc_distill tip `a382986`에서 분기)
- **워크트리:** `/home/minwoo/Distillation_Project_lm_distill` (신규)
- **상태:** 사용자 승인된 설계 (구현 전)
- **관련 문서:** `2026-06-29-itc-distillation-online-teacher.md` (온라인 티처 패턴의 원본 — 본 설계는 그 시임을 LM 로짓으로 확장), `2026-06-27-itc-distillation-design.md`

---

## 1. 목표

frozen BLIP-large 티처의 **디코더 출력 로짓**(teacher-forcing)을 학생(DINOv3 small_reg + MiniLM)의 디코더 출력 로짓에 증류한다. GPU 4장 각각에 티처+학생을 동시 탑재(메모리 여유 확인됨)하고, 티처는 학생과 같은 증강 배치를 매 스텝 온라인으로 처리한다. 실험 통제 원칙: **baseline(exp 7.student_baseline)과의 차이는 `+w·loss_lm_kd` 항 단 하나.**

## 2. 확정된 결정 사항 (사용자 Q&A)

| 항목 | 결정 |
|---|---|
| 베이스 브랜치 | itc_distill tip에서 분기 (`itc_distill ⊇ dev` 확인됨 — "dev + ITC 스캐폴딩 체리픽"과 내용 동일, 충돌 없음) |
| KD 결합 | **CE 유지 + 가산**: `total = ita + itm + lm + w·lm_kd` (ITC KD와 동일 철학) |
| 티처 | **공식 Salesforce BLIP-large pretrain** (`model_large.pth`, ViT-L + BERT-base, 디코더 포함). 경로는 사용자가 다운로드 후 기입 |
| 학생/학습 설정 | `pretrain_student.yaml` 그대로 (small_reg + MiniLM, batch 40×4GPU, aug ON, lr 등 동일). 이번 런은 `distill.itc` OFF, `distill.lm`만 ON |
| KD 형태 | token-level teacher-forced KD (autoregressive 생성 없음) |
| w (weight) | **고정 config 하이퍼파라미터** (학습값 아님). TB 시계열 로깅 대신 TB run name에 태그 + config.yaml 덤프로 기록 |

## 3. 실행 가능성 근거 (탐색으로 확인)

- 티처(BERT-base)와 학생(MiniLM) 모두 동일한 `init_tokenizer()` (bert-base-uncased + [DEC]/[ENC], vocab **30524**) 사용 + `resize_token_embeddings` → **디코더 로짓 차원 완전 일치** → 위치별 vocab-KL 직접 가능.
- 학생 LM 경로(`models/blip_pretrain.py` LM 섹션): max_length=30 토크나이즈 → `[:,0]=BOS`, pad→-100 → `text_decoder`가 내부 shift 후 CE(ls=0.1). `decoder_output.logits [B,30,V]` 이미 존재.
- 공식 ckpt의 `temp` 파라미터는 unexpected key로 무시, 우리 모델의 `logit_scale`은 missing이지만 티처 LM 경로와 무관.
- 공식 ckpt queue shape은 `queue_size=config['queue_size']`(57600)로 티처를 생성하므로 로드 시 shape 일치 (기존 배선 그대로).
- no-aug(exp5) NO-GO → 증강 ON 필수 → 온라인 티처 (ITC 피벗과 동일한 논리, LM에도 그대로 적용).

## 4. 아키텍처

한 학습 스텝 (기존 ITC KD 시임과 동일 구조):

```
1. 배치 로드: (image, caption)
2. 티처:  teacher_lm_logits, teacher_lm_ids = online_teacher.lm_logits(image, caption)
          (frozen, rank별 복제, DDP 미래핑, bf16 autocast + no_grad, 학생과 같은 증강 이미지)
3. 학생:  model(image, caption, alpha, ..., teacher_lm_logits=..., teacher_lm_input_ids=...,
                lm_distill_temp=...)
          → forward LM 섹션에서 loss_lm_kd 계산
          → (loss_ita, loss_itm, loss_lm, loss_itc_kd, loss_lm_kd) 반환  [5-tuple]
4. loss = ita + itm + lm + w_lm·lm_kd   (이번 런 itc_kd는 config OFF → None)
```

원칙 유지: 티처는 train loop에서 실행하고 **결과 텐서만** forward에 전달; KD loss는 학생 로짓이 있는 forward 안에서 계산; 티처 가중치는 학생 state_dict/DDP에 절대 섞이지 않음.

## 5. 컴포넌트 변경 상세

### 5.1 `distillation/online_teacher.py` — keep 인자 + `lm_logits()`

- `__init__(checkpoint, image_size, vit='large', bert='base', queue_size, keep=('itc',))`.
- "드랍"의 구현: 전체 모델 생성 → **ckpt 로드** → keep 합집합 외 서브모듈 `setattr(model, attr, None)` → `.to(device)`. (로드가 해제보다 먼저 — 기존 순서 유지)

```python
NEEDS = {
    'itc': {'visual_encoder', 'text_encoder', 'vision_proj', 'text_proj'},
    'lm':  {'visual_encoder', 'text_decoder'},
}
# momentum 4개(visual_encoder_m, text_encoder_m, vision_proj_m, text_proj_m),
# itm_head, 큐 buffer 3개는 어떤 keep에서도 항상 해제 (학습 전용 장치 — 추론 티처에 불필요)
```

- critical-key 검사를 keep별 prefix로 확장: `'itc'` → 기존 4개 prefix, `'lm'` → `visual_encoder.` + `text_decoder.`. 누락 시 RuntimeError.
- 신규 메서드 (기존 `itc_feats`와 대칭):

```python
@torch.no_grad()
def lm_logits(self, image, caption):
    # 'lm' not in keep이면 명시적 RuntimeError
    device = image.device
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        image_embeds = self.model.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
        text = self.tokenizer(caption, padding='max_length', truncation=True,
                              max_length=30, return_tensors='pt').to(device)
        decoder_input_ids = text.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id      # 학생과 동일 규칙
        out = self.model.text_decoder(decoder_input_ids,
                                      attention_mask=text.attention_mask,
                                      encoder_hidden_states=image_embeds,
                                      encoder_attention_mask=image_atts,
                                      return_dict=True)             # labels 없음 → loss 계산 안 함
    return out.logits, decoder_input_ids   # [B, 30, 30524] (bf16), [B, 30]
```

### 5.2 `distillation/losses.py` — `lm_distill_loss` 추가

```python
def lm_distill_loss(student_logits, teacher_logits, decoder_targets, temp):
    """token-level LM logit KD. CE와 동일한 shift 프레임/유효 위치에서
    KL(teacher ‖ student) 을 유효 토큰당 평균 × T²."""
    # vocab 차원 일치 assert (명시적 메시지)
    s = student_logits[:, :-1, :]               # 위치 i 로짓이 토큰 i+1 예측
    t = teacher_logits[:, :-1, :].detach()
    valid = decoder_targets[:, 1:] != -100      # pad 제외 = CE 학습 위치와 동일 집합
    s, t = s[valid], t[valid]                   # [N_valid, V]
    return F.kl_div(F.log_softmax(s / temp, dim=-1),
                    F.softmax(t / temp, dim=-1),
                    reduction='batchmean') * (temp ** 2)   # batchmean = N_valid로 나눔
```

- KL 방향: forward KL(teacher‖student) — Hinton 표준.
- 정밀도: autocast가 softmax/log_softmax를 fp32로 승격 (ITC 때 검증한 메커니즘과 동일).
- **길이 불일치는 발생하지 않음**: 두 모델 모두 자유 생성이 아니라 GT 캡션에 대한 teacher-forcing이므로, 시퀀스(30칸 고정)와 유효 위치 집합은 GT가 결정하고 티처·학생에 동일하다. 각 위치의 KL은 "같은 GT 접두어에서의 다음 토큰 분포" 간 비교 — CE의 one-hot 타깃을 티처 soft 분포로 바꾼 것과 같은 구조.

### 5.3 `models/blip_pretrain.py` — forward 확장

- 시그니처 추가 인자: `teacher_lm_logits=None, teacher_lm_input_ids=None, lm_distill_temp=2.0`.
- LM 섹션 끝:

```python
loss_lm_kd = None
if teacher_lm_logits is not None:
    if teacher_lm_input_ids is not None:
        assert torch.equal(teacher_lm_input_ids, decoder_input_ids), \
            "teacher/student decoder input mismatch"   # 이중 토크나이즈 정합성 보증
    loss_lm_kd = lm_distill_loss(decoder_output.logits, teacher_lm_logits,
                                 decoder_targets, lm_distill_temp)
return loss_ita, loss_itm, loss_lm, loss_itc_kd, loss_lm_kd   # 5-tuple
```

### 5.4 `pretrain.py` — 배선

- config 읽기: `distill.lm.{enabled, weight, temp}` (ITC 블록과 대칭).
- 티처 생성 조건 확장: `distill.itc.enabled or distill.lm.enabled`. `keep`은 enabled 플래그에서 자동 유도: `keep = tuple(k for k in ('itc','lm') if enabled(k))`.
- 스텝마다 (기존 ITC 티처 호출과 같은 자리, 학생 autocast 블록 밖):

```python
teacher_lm_logits = teacher_lm_ids = None
if lm_kd_enabled and online_teacher is not None:
    teacher_lm_logits, teacher_lm_ids = online_teacher.lm_logits(image, caption)
```

- 합산: `loss = ita + itm + lm (+ w_itc·itc_kd if enabled&not None) (+ w_lm·lm_kd if enabled&not None)`.
- 로깅: TB `loss_train/lm_kd` + MetricLogger meter `loss_lm_kd` 추가. `loss_train/total`은 KD 포함값 (기존 동작).
- **TB run name**: `make_tb_run_name()`이 distill 활성 시 `kd=` 필드 추가 (예: `kd=lm_w1.0T2.0`, 결합 시 `kd=itc+lm...`). baseline 런 이름은 불변. w·T의 영구 기록은 `output_dir/config.yaml` 덤프(기존 동작)가 담당.

### 5.5 forward 반환 변경의 콜사이트 (전수 조사 완료, 2026-07-02)

itc_distill 기준 tuple 언팩 지점은 아래 3파일이 전부 (독립 eval 스크립트들은 pretrain forward를 언팩하지 않음 — 영향 없음):

| 파일 | 위치 | 변경 |
|---|---|---|
| `models/blip_pretrain.py` | return문 (~L464) | 5-tuple 반환 |
| `pretrain.py` | autocast/비-autocast 분기 2곳 (~L120, L128) | 5-tuple 언팩 |
| `data/eval_validation_loss.py` | ~L297 | 4→5-tuple 언팩. validation은 티처를 돌리지 않으므로 KD 두 항은 None (ITC와 동일한 취급 — val에서 KD 관찰은 future option) |

### 5.6 config — `configs/pretrain_lm_distill.yaml` (신규, `pretrain_student.yaml` 복사)

```yaml
output_dir: '/home/minwoo/Distillation_Project/output/pt_smallreg_minilm_lm_distill'
exp: '8.lm_distill'            # 6=itc_distill, 7=student_baseline 다음
pretrain_train_aug: true

distill:
  itc: {enabled: false, weight: 1.0, temp: 0.05}   # 이번 런 OFF
  lm:  {enabled: true,  weight: 1.0, temp: 2.0}

teacher:
  arch: 'blip_large'
  checkpoint: '<공식 model_large.pth 경로 — 다운로드 후 기입>'   # 미결 입력
```

- 미추적이던 `pretrain_student.yaml`은 itc_distill 워크트리에서 파일 복사로 가져와 이 브랜치에서 **추적(커밋)** 대상에 포함.

## 6. 하이퍼파라미터 기본값

- `distill.lm.temp: 2.0` (스윕 후보 1 / 2 / 4). vocab-logit KD의 통상 범위; ITC의 τ=0.05는 유사도 로짓용이라 별개 맥락.
- `distill.lm.weight: 1.0` — 첫 런에서 `loss_train/lm_kd` 스케일 확인 후 조정 여지. w는 스텝 0부터 상수 적용 (ramp 없음 — ITC KD와 동일).

## 7. 비용 (정직하게)

- 스텝마다 티처 forward 추가: ViT-L(비용의 대부분, ITC 티처와 동일) + BERT-base 디코더(30 토큰, cross-attn 포함 — ITC의 text encoder 대체분). ITC KD 런과 거의 동급의 오버헤드.
- 티처 상주: ~440M 파라미터(fp32 가중치 ≈ 1.8GB) + bf16 forward activation. 로짓 텐서 [40,30,30524] bf16 ≈ 73MB/step, KL의 fp32 softmax 순간 피크 ≈ 150MB×2. 24GB에 여유 (사용자 확인 전제와 일치).

## 8. 에러 처리 (시끄럽게 죽기)

1. ckpt critical-key 검사 (keep별 prefix) → RuntimeError. `temp`(unexpected)/`logit_scale`(missing)은 무해 — 로드 메시지만.
2. forward에서 `teacher_lm_input_ids == decoder_input_ids` assert (토크나이저 드리프트 즉시 검출).
3. `lm_distill_loss` 진입 시 vocab 차원 assert.
4. keep에 없는 경로 메서드 호출 → 명시적 RuntimeError.
5. enabled인데 checkpoint 부재 → 티처 로드 단계(학습 전) 즉시 실패.
6. 합산은 `enabled && loss is not None` 조건으로만 (조용한 0 가산 금지).

## 9. 테스트 계획

**유닛 (CPU, 기존 파일 확장, TDD로 구현)**
- `lm_distill_loss`: student==teacher & T=1 → ≈0 / pad 위치 로짓 변경에 불변(마스킹) / teacher로 grad 차단(detach) / N_valid 정규화 / T² sanity.
- `OnlineTeacher` keep: `('lm',)` → text_encoder/proj None + visual_encoder/text_decoder 생존 + `lm_logits` shape·no-grad 확인 / `('itc',)` → 기존 테스트 회귀 / `('itc','lm')` → 합집합 / 없는 경로 호출 → RuntimeError.
- forward: 티처 인자 None → `(...,None,None)` + 기존 3 loss 회귀 / 티처 로짓 제공 → loss_lm_kd finite + grad 흐름.

**통합 스모크 (GPU, 실제 ckpt 확보 후)**
- 수십 스텝: `loss_train/lm_kd` finite + TB 기록 + 메모리 실측.
- `enabled=false` 회귀: 사전-KD 동작과 동일.

## 10. 실험·검증 계획

- 비교군: exp `7.student_baseline` — 차이는 `+w·lm_kd` 항 하나.
- 1차 지표: **val loss_lm** (COCO Karpathy val CE, 기존 val loss runner 자동 기록).
- 2차 지표: retrieval R@1/5/10 (LM KD의 ITC/ITM 부작용 감시), `loss_train/lm` 궤적.
- 기존 체크포인트 sweep·TB 비교 툴링 재사용.

## 11. 미결 입력

- `teacher.checkpoint` — 공식 BLIP-large pretrain(`model_large.pth`) 다운로드 경로 (사용자 제공 예정). 이것 없이는 통합 스모크·본 런 불가 (유닛 테스트는 가능).

## 12. 범위 밖 (future)

- ITM distillation.
- ITC+LM 결합 런 (스캐폴딩은 본 설계로 준비됨 — config 두 블록 enabled + `keep=('itc','lm')` 자동 유도).
- validation 중 KD 항 관찰 (val에 티처 투입).
- w 학습화(uncertainty weighting 등) — 실험 통제 원칙과 상충하여 비채택.
