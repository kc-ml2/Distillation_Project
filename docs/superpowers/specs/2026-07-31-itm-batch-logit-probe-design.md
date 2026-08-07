# ITM 학습 배치 로짓 원자료 프로브 — 설계

**작성:** 2026-07-31
**상태:** 설계 확정 대기
**범위:** 측정 도구 1개. 증류 손실의 설계·구현은 이 스펙 밖이다.

---

## 1. 왜 하나

ITM 증류를 **pair-wise 2-class KD**에서 **배치 단위 분포 매칭**으로 바꾸는 방안을 검토 중이다.

현행 ITM KD는 쌍 하나마다 로짓 2개(`z1`=no-match, `z2`=match)를 받고, 2-class softmax가
차분 `gap = z2 - z1` 만의 함수이므로 **실질 자유도가 1**이다. 자유도 1짜리 타깃은
(a) 노이즈에 취약하고, (b) 하드라벨에 잡음을 얹은 것에 가깝고, (c) 티처와 라벨이 반대를
가리키면 "모르겠음"을 학습시킨다.

대안은 **배치를 하나의 분포로 묶어 KL 매칭**하는 것이다. 여기에 **직교하는 선택축이 둘** 있다.

**축 A — 무엇을 매칭하나**

| 변형 | 매칭 대상 | 필요한 전제 |
|---|---|---|
| **행 단위** | `z1` 행과 `z2` 행을 각각 softmax → KL 2개 | **m 상수성** |
| **gap 단위** | `gap` 벡터를 softmax → KL 1개 | 없음 (offset 불변) |

여기서 $m_i = (z_{1,i}+z_{2,i})/2$ 다. 행 단위 변형은 **m이 샘플 간 거의 일정**해야 성립한다.
티처가 `z2` 값만으로 재정렬을 할 수 있는 것도 이 상수성 덕분이라는 것이 이 방안의 출발점이다.

**축 B — 분포를 어디까지 묶나**

| 변형 | 분포 범위 | 결과 |
|---|---|---|
| **3B 통합** | 3B개 전체를 한 softmax | 분포 1개(gap 단위) / 2개(행 단위) |
| **블록별** | ①②③ 블록마다 B-way softmax | 분포 3개(gap 단위) / 6개(행 단위) |

두 축은 독립이므로 **후보는 2×2 = 4가지**다. 축 B가 결정적인 이유는, 통합 softmax에서는
확률 질량이 pos/neg 라벨 차이에 먼저 배분되고 그 차이는 **하드라벨이 이미 주는 정보**이기 때문이다.
질량이 라벨 쪽으로 쏠릴수록 정작 얻으려는 "같은 라벨 안에서의 상대 순위"에 실리는 gradient가 줄어든다.

**이 프로브는 두 축의 전제를 모두 측정한다.** 축 A는 §8.2가, 축 B는 §8.1이 결정한다.
측정 없이는 어느 조합도 고를 수 없다.

## 2. 기존 프로브를 왜 못 쓰나 — 축이 다르다

선행 조사(`critical_bugfix/2026-07-30_itm_logit_offset_and_head_geometry/`)가 같은 이름의 양을
쟀지만 **측정 축이 다르다**:

| | 선행 조사 | 이 프로브 |
|---|---|---|
| 표본 단위 | retrieval 후보 (이미지 1장 고정 × top-128 텍스트) | **학습 배치 3B** (서로 다른 쌍) |
| h의 변동 범위 | 한 쿼리 주변의 좁은 영역 | 배치 전체가 훑는 넓은 영역 |
| 답하는 질문 | eval 랭킹이 오염되나 | **증류 타깃 분포가 오염되나** |

선행 조사는 within-query에서 m의 변동이 작다고 기록했고, 그 이유로 "쿼리 내부에서 h가 움직이는
방향이 $w_s$와 거의 직교"를 들었다. 그 직교성이 **cross-sample 축에서 유지된다는 보장은 없다.**
`itm_head`의 합 성분 $w_s = w_{match}+w_{nomatch}$ 는 ITM CE가 구속하지 않는 방향이고, h가 넓게
움직이면 $m = w_s^\top h + b_s$ 도 따라 움직일 수 있다.

→ 파이프라인 기법(fp32 헤드 분리, `no_grad` 강제)은 참고하되, **측정은 처음부터 다시 한다.**
선행 조사의 수치와 결론은 이 프로브의 근거로 쓰지 않는다.

## 3. 무엇을 재나

`models/blip_pretrain.py:471-489`의 ITM 배치는 세 블록으로 구성된다:

| 블록 | 구성 | 고정된 것 | 라벨 |
|---|---|---|---|
| ① `pos` | (img_i, txt_i) | — | 1 |
| ② `neg_img` | (**neg**\_img_i, txt_i) | txt_i | 0 |
| ③ `neg_txt` | (img_i, **neg**\_txt_i) | img_i | 0 |

**블록마다 표 하나**, 모델 2개(학생/티처) → **표 6개**. 표의 행은 샘플 하나이고 열은:

| 열 | 정의 |
|---|---|
| `batch` | 배치 인덱스 |
| `i` | 배치 내 위치 |
| `z1_nomatch` | `itm_head` 출력의 class 0 |
| `z2_match` | class 1 |
| `m` | $(z_1+z_2)/2$ |
| `gap` | $z_2 - z_1$ |

표 하단에 **mean / var / std** 행을 붙인다 (열별, ddof=1).

표본 수는 **블록당 500행**. `batch_size=40`이므로 13배치를 돌고 앞 500행만 쓴다.

### 3.1 요약 통계 — 파생만 하고 결론은 원자료에서

CSV에는 원자료를 그대로 싣고, 콘솔 요약에는 아래를 계산해 함께 보인다.
어떤 비율 지표도 단독 판정 근거로 쓰지 않는다.

- **Δ** = mean(gap of ①) − mean(gap of ②∪③) — negative 두 블록은 **pooled**(한 배열로 합쳐서) 평균낸다
- **P_pos** — Δ가 통합 softmax에서 ①에 배분하는 확률 질량. 실측값 $\sum e^{g_{pos}} / \sum e^{g_{all}}$ 을 쓴다
- **σ(m)** — 로짓 단위 절대값 (모델 내 비교용)
- 블록별 **σ(gap)**

> 무차원 비율 지표(σ(m)/σ(gap), Var(m)/Var(z2) 등)는 **의도적으로 넣지 않는다.**
> gap 산포가 큰 모델이 자동으로 유리해 보이는 문제가 있고(§8.2), 한 번 계산해 두면
> 원자료 대신 그 숫자로 판정해 버리게 된다. m의 상수성은 표 6개를 직접 대조해 판단한다.

## 4. 재현 사양 — 학습 배치를 그대로 밟는다

`blip_pretrain.forward`의 ITM 경로를 동일하게 재현한다. 확인된 세부:

1. **negative 샘플링은 momentum 피처를 쓴다.**
   `image_feat_all = cat([image_feat_m.t(), image_queue])` 이고 `sim_t2i[:, :bs]`가
   그 앞부분만 자르므로, 후보 가중치는 `text_feat @ image_feat_m.t()`에서 나온다.
   → 학생은 momentum 인코더 4종을 **로드해야 한다**(갱신은 하지 않는다). queue는 슬라이싱으로 배제된다.
2. **logit_scale이 샘플링 분포를 바꾼다.**
   `safe_scale = logit_scale.clamp(log 2, log 1000).exp()` 를 체크포인트 값 그대로 적용한다.
3. `weights = softmax(sim[:, :bs]) + 1e-4`, 대각 0, `multinomial` 1개 — 원 코드와 동일.
4. **`itm_head`는 autocast 밖 fp32.** `gap`이 두 로짓의 차라 bf16이면 유효숫자가 깎인다.
5. 나머지 forward는 학습과 동일하게 **bf16 autocast**(`pretrain.py:140`). CPU 실행 시에는 fp32.
6. `torch.set_grad_enabled(False)` 전역. momentum 갱신·queue 갱신 금지.

### 4.1 티처는 학생과 완전히 동일한 3B 쌍을 본다

티처는 negative를 스스로 뽑지 않는다. **학생이 뽑은 `neg_idx`를 그대로 재사용**한다.
이미지 전처리는 두 모델 공통이다 — `data/__init__.py`의 normalize가 하나뿐이고,
기존 `OnlineTeacher`도 학생이 본 augmented 이미지를 그대로 받는다.

캡션 토크나이즈는 각 모델의 tokenizer로 한다(vocab은 동일).

### 4.2 순차 로딩

GPU 4장이 학습 잡 2개로 점유돼 있다(여유 0.4~1.8GB). 메모리 피크를 낮추기 위해:

1. 학생 로드 → 13배치 forward → `(image, caption, neg_idx_t2i, neg_idx_i2t)`를 CPU RAM에 캐시
2. 학생 언로드 → 티처 로드 → 캐시로 forward

이미지 캐시는 13×40×3×224×224×4B ≈ 313MB로 감당 가능하다.

## 5. 대상

| 역할 | 모델 | 체크포인트 |
|---|---|---|
| 학생 | config 그대로 (DINOv3 ViT-S/16 + MiniLM) | `output/pt_smallreg_minilm_baseline/checkpoint_19.pth` |
| 티처 | `vit='large'`, `my_bert_size='base'`, `init_backbone_weights=False` | `output/official_pretrain_checkpoint/model_large.pth` |

데이터: `create_dataset('pretrain', cfg, min_scale=0.2)`, `pretrain_train_aug: true`(학습과 동일).
config는 `output/pt_smallreg_minilm_baseline/config.yaml`.

## 6. 실행 환경

GPU가 없으므로 **CPU 실행을 기본 경로로 한다**(128코어 / RAM 133GB 여유).
`--device` 로 GPU 전환이 가능하되 기본값은 CPU다. GPU 경로에서만 bf16 autocast를 켠다.

## 7. 산출물

```
critical_bugfix/2026-07-31_itm_batch_logit_raw/
  itm_batch_logit_probe.py       프로브
  csv/{student,teacher}_{pos,neg_img,neg_txt}.csv   표 6개 (500행 + mean/var/std)
  npz/{student,teacher}.npz      재분석용 원자료
  summary.json                   콘솔 요약과 동일한 값
```

## 8. 판정 기준 — 미리 못박는다

측정 후 해석이 흔들리지 않도록 선을 먼저 긋는다.

### 8.1 3B 통합 softmax 가능 여부

$p_i \propto e^{g_i}$ 이므로, ①이 통합 softmax에서 먹는 확률 질량은 **분산과 무관하게** Δ만으로 정해진다:

$$P_{pos} = \frac{Be^{\Delta}}{Be^{\Delta}+2B} = \frac{e^{\Delta}}{e^{\Delta}+2}$$

| Δ | 0 | 1 | 2 | 3 | 5 |
|---|---|---|---|---|---|
| P_pos | 33% | 58% | 79% | 91% | 99% |

- **Δ ≲ 1** → 통합 검토 가능 (pos/neg 경계 calibration까지 한 분포에 담김)
- **Δ ≳ 2** → 블록 분리 필수. 통합하면 gradient 대부분이 하드라벨이 이미 주는 정보에 실린다

### 8.2 m 상수성 (행 단위 변형의 성립 여부)

티처의 σ(m)을 기준선으로 학생을 비교한다. 판정은 **블록별로 따로** 내린다 —
한 블록에서만 깨져도 그 블록의 행 단위 매칭은 오염된다.

여기에는 §8.1과 달리 **수치 임계를 미리 못박지 않는다.** "얼마나 상수여야 충분한가"는
로짓 단위에 의존해 사전에 정할 수 없고, 무차원화하려고 σ(gap)으로 나누면 gap 산포가 큰 모델이
자동으로 유리해진다(m의 상수성이 나아지지 않았는데도). 그래서 이 항목은 **표 6개의 원자료를
직접 대조해** 판단한다 — 프로브가 원자료를 CSV로 내보내는 이유가 이것이다.

### 8.3 블록별 σ(gap)

블록 간 σ(gap)이 크게 다르면 블록마다 온도를 따로 걸 근거가 된다.
같은 블록 안에서 σ(gap)이 과도하게 크면 "쉬운 문제가 softmax 질량을 독식"하는 문제가 실재한다.

## 9. 범위 밖 (non-goals)

- 증류 손실 함수 설계·구현 — 이 프로브의 결과를 입력으로 **다음 스펙**에서 다룬다
- 학습 코드(`blip_pretrain.py`, `pretrain.py`) 수정 — 프로브는 읽기 전용이다
- eval 랭킹 방식(`z2` vs `gap`) 변경 검토 — 별개 질문이다
