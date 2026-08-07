# ITM 배치 B×B 매트릭스 증류 — 설계

**작성:** 2026-08-07
**상태:** 설계 확정 (브레인스토밍 승인 완료) → writing-plans 대기
**선행:** `docs/superpowers/specs/2026-07-31-itm-batch-logit-probe-design.md`(측정), `critical_bugfix/2026-07-31_itm_batch_logit_raw/`(실측), `2026-08-05-itm-batch-bidirectional-kd-design.md`(폐기 — 아래 §1.2)
**베이스 브랜치:** `dev`에서 새 분기 (예: `dev/itm_bxb_matrix_kd`).
**계승 안 함:** arm C(확률 target-mix), scheme A·B, 2026-08-05 3B-블록 스펙. 전부 폐기.

---

## 1. 왜 하나 — 바이너리 벽과 그 해법

### 1.1 문제: pair-wise ITM은 자유도 1

ITM 헤드는 pair 하나당 2-class(`no-match=z1`, `match=z2`)를 뱉는다. 2-class softmax는 `gap = z2 − z1`만의 함수라 **실질 자유도 1**이다. 프로브(2026-07-31)가 실측으로 확정: 중점 `m=(z1+z2)/2`는 사실상 상수(티처 σ(m)=0.007), gap만 움직인다. 즉 pair 단위로 티처에게서 뽑을 dark knowledge가 **라벨(gap 부호) 말고 거의 없다.** pair-wise KD(arm C / scheme A·B)가 전부 하드라벨과 싸우다 head separation만 붕괴시킨 근본 이유다.

### 1.2 해법: 축을 바꾼다 — 전체 B×B 매트릭스

pair 하나가 아니라 **이미지 i를 배치의 B개 텍스트 전부와 cross-attn 시켜 나온 B개 매치 점수의 분포**를 증류한다. 이 row 분포에는 dark knowledge가 가득하다 — "이 이미지에 대해 티처가 어떤 distractor를 더 헷갈려 하는가"라는 **랭킹 정보**. 이건 하드라벨과 경쟁하지 않는다(라벨은 대각선만 건드림). 바이너리 벽이 녹는다.

- **ITC KD의 완전한 미러다.** `itc_distill_loss`가 `image_feat @ text_feat.t()`로 B×B 내적 매트릭스를 만들어 row/col softmax KL을 하듯, 이건 **B×B cross-attn 매치-로짓 매트릭스**로 똑같이 한다. 셀 값이 내적이 아니라 융합 점수라는 것만 다르다.
- **eval과 처음으로 정렬된다.** retrieval ITM rerank(`eval_validation_tool.py:130-196`)가 정확히 이 매트릭스의 한 row를 top-k만 잘라 쓰는 것이다. 학습 타깃 = 평가 신호.
- **2026-08-05(폐기) 대비 개선:** 그 스펙은 pos/neg_img/neg_txt "3B 블록"을 batch/sample 축으로 softmax해서 **retrieval 축이 아니었다.** B×B는 올바른 축(이미지 i vs 전 텍스트)이다. 또한 all-pairs라 neg_idx 왕복/공유 플럼빙이 통째로 사라진다.

### 1.3 왜 ITM은 큐로 못 키우나 (B×B가 천장)

ITC는 큐(57600)로 네거티브를 키운다. **ITM은 구조적으로 불가** — 큐엔 256-d pooled 벡터만 있고, ITM cross-attn은 패치/토큰 hidden state가 필요하다([[project_itm_distillation_plan]], `blip_pretrain.py:448-489`). → **in-batch B×B가 ITM 네거티브 풀의 하드 천장.** "배치 제곱이 유일한 길"의 구조적 근거. (참고: 프로덕션 TTM 큐 경로는 `exp9_teacher_target_mix` 브랜치 `distillation/target_mix.py`; ITM엔 target-mix가 arm C로 죽었고 큐도 못 쓰므로 이 스펙은 **별도-KD-손실 형태**(`itc_distill_loss` 계열)를 따른다.)

---

## 2. 무엇을 만드나 — 셀·방향·손실

### 2.1 셀 = 헤드 raw 채널 z1·z2 (확률 아님)

매트릭스 `S[i,j] = itm_logits(image_i, text_j) ∈ ℝ²` (헤드 raw 출력 [z1, z2]). 셀을 **raw 로짓**으로 둔다. 확률(`σ(gap)`/`softmax[1]`)은 폐기 — 이유:

> KL+softmax는 원래 **batch-softmax 단계에서** 크기를 뭉갠다(shift-invariant, τ만 스케일 남김). 이건 A/B/C 공통. (C)가 죽는 진짜 이유는 [0,1]이 아니라 **이중 비선형**: per-pair 시그모이드 `σ(gap)`이 **batch-softmax 전에** 음수-tail을 미리 포화(gap −2 vs −8 → 0.12 vs 0.0003)시켜 랭킹을 소실시킨다. raw 로짓 → batch-softmax 1회는 그 tail 랭킹(`e^{-2/τ}` vs `e^{-8/τ}` ≈ 400배)을 보존한다.

### 2.2 방향 — {z1, z2} × {row, col} 양방향, 토글

| 축 | 의미 |
|---|---|
| **row (i2t)** | `softmax_j(S[i, :])` — image i의 텍스트 분포 |
| **col (t2i)** | `softmax_i(S[:, j])` — text j의 이미지 분포 |
| **z2 = forward** | `m + gap/2` → `softmax ≈ softmax(gap/2)`: 쉬운 매치에 질량 |
| **z1 = reverse** | `m − gap/2` → `softmax ≈ softmax(−gap/2)`: 하드 네거(음수 gap)에 질량 |

- **bidir:** `{z1,z2} × {row,col}` = **KL 4개**, 평균.
- **forward:** `z2 × {row,col}` = **KL 2개** (쉬운 매치 랭킹).
- **reverse:** `z1 × {row,col}` = **KL 2개** (하드네거 거절 랭킹만 격리 — 네거 tail이 핵심일 때).
- **config 토글** `direction ∈ {forward, reverse, bidir}`. 매트릭스 forward는 동일, KL 항 수만 갈림 → 거의 공짜 플래그.

**왜 reverse(z1)가 필요한가 — ITC엔 없는 이유.** forward KL(T‖S)의 grad = `(p−q)/τ`. 하드 네거에서 티처 `q≈0`, 학생 `p≈0` → **grad≈0** (학생 gap −2, 티처 −8이어도 batch-softmax가 둘 다 ~0으로 뭉갬 → 로짓 6 차이가 학습에 안 실림). forward만 쓰면 **음수 판정 구조를 통째로 못 배운다.** reverse(z1)는 그 하드 네거가 softmax에서 질량을 먹어 grad를 되살린다.
- ITM 헤드는 학습된 바이너리 분류기라 positive를 극단 포화시킨다(프로브 P_pos=0.99) → forward-softmax의 네거 굶주림이 유계 cosine인 ITC보다 훨씬 심하다.
- **ITM gap은 부호 twin(z1)이 있지만 cosine 유사도는 없다.** 그래서 ITM은 reverse가 필요·가능하고, ITC는 필요·가능하지 않았다. (ITC KD도 B×B다 — 큐는 ITC의 *CE*가 쓰지 *KD*는 안 씀.)
- **추가 근거 — 유계 vs 무계 dynamic range.** ITM 로짓은 범위 제한이 없어 헤드가 positive gap을 임의로 크게(→ 결정 포화, 프로브 P_pos=0.99) 밀 수 있다 → row-softmax가 극단적으로 positive에 스파이크 → 네거 굶주림. 반면 ITC cosine은 [−1,+1] **유계**라 모든 정보가 이 안에 갇힌다 — 정답이 확실한 1이어도 오답 몇 개가 애매하게 높은 cosine으로 뜨면 mass를 크게 나눠 갖는다. **ITC가 τ를 매우 낮게(scale 크게) 쓰는 것 자체가 이 유계성의 증거**다: cosine gap이 태생적으로 저-동적범위라 크게 스케일해야 겨우 분리된다. → 무계 포화인 ITM은 reverse 채널로 네거 tail을 되살릴 실익이 크고, 유계 cosine엔 그 twin 채널조차 없다.
- **raw z1·z2 직접 증류 ≡ gap 부호 flip 증류** (m≈상수라 `z1,z2 = m∓gap/2`). 파생량·부호반전 없이 헤드 두 채널을 그대로 softmax → gap±와 동치 + 무해한 약한 m-정규화. 구현이 더 단순한 쪽이 곧 옳은 쪽.

### 2.3 손실 정의

블록 `d ∈ {row, col}`, 채널 `k ∈ {z1(reverse), z2(forward)}` (bidir), 또는 `k ∈ {z2}`(forward):

$$\mathcal{L}_{itm\text{-}kd} = \frac{1}{|K|\cdot 2}\sum_{d\in\{row,col\}}\sum_{k\in K} \mathrm{KL}\!\Big(\mathrm{softmax}\big(z_k^{T,d}/\tau\big)\,\Big\|\,\mathrm{softmax}\big(z_k^{S,d}/\tau\big)\Big)\cdot\tau$$

(`softmax`는 블록 축 위 — row면 텍스트 축, col이면 이미지 축. 셀당 시그모이드 아님 = §2.1의 (C)와 구별.)

| 결정 | 값 | 근거 |
|---|---|---|
| KL 방향 | forward `KL(T‖S)` | itc/lm 관례 동일 |
| 온도 | 고정 τ (config, 기본 itc와 동일 오더) | — |
| 정규화 | `×τ` 보정, KL 항 평균 | `itc_distill_loss` 관례(sharpening regime grad ∝ 1/τ 상쇄, critical_bugfix/2026-07-14) |
| λ (총손실) | 1 (비율 보고 하향) | 관례 |
| dtype | **fp32** | 프로브 §7: bf16이면 gap 오차 8000배 |

---

## 3. 기존 하드라벨 CE와의 관계 — (가) 순수 추가

**기존 3B 하드라벨 ITM CE(`output_pos`+`output_neg`, `blip_pretrain.py:442-501`, 120 forward)를 100% 그대로 둔다.** B×B는 **순수 추가 KD 항.**

- CE = 검증된 separation 앵커(baseline itm 좋음). 건드리면 리스크. 매트릭스로 CE를 재구성하면 네거 선택 로직이 바뀌어 작동하는 걸 흔든다.
- 추가 비용 120/1600 = **7.5%**. 아끼려는 리팩터는 조기최적화.
- **역할 분담:** CE가 대각선을 match로 못박아 separation 유지, KD가 블록 내 랭킹(dark knowledge) 추가. row-softmax는 shift-invariant라 절대 separation과 직접 싸우지 않음.

**VG 라벨 노이즈 = 리스크가 아니라 티처 가치.** forward 방향에서 티처-라벨 불일치(프로브 §6: true-pos의 36.6%를 no-match)는 상당수가 **티처 오답이 아니라 VG 노이즈 캡션(라벨이 틀림)**이고 거기선 티처가 정답이다 — 티처로만 배울 수 있는 신호. retrieval eval은 COCO(깨끗) 도메인이라 VG-교정이 지표를 해치지 않는다. gap≈0 중간지대(경계오염, arm C를 죽인 muddle)는 두 softmax의 어깨라 grad가 거의 안 가 **안 배움 = 티처 오보정 비수입 보호막.**

---

## 4. 매트릭스 계산 — full 기본 + topk knob

### 4.1 default: full B×B (단일 매트릭스)

`S[i,j]`는 pair당 스칼라 벡터 1개. full은 B²개 셀을 **한 번** 계산해 S를 채우고, **row(i2t)·col(t2i) 둘 다 같은 S에서 파생**(softmax는 인코더 forward 아니라 축 연산, 공짜). → 비싼 forward **B²번**(2·B² 아님). B=40 → 1600.

### 4.2 opt-in: `topk` 서브샘플 (진짜 계산 절감)

k<B일 때 방향별 하드네거만 실제로 forward. **불변식 — 각 방향 분포는 항상 `positive + 네거 k` = k+1 고정:**

- 선택은 **기존 diagonal-zeroed weights 재사용**(`blip_pretrain.py:448-451`, `sim_*[:,:bs]`=in-batch B×B, `fill_diagonal_(0)`으로 positive=0). 현재 `multinomial(·,1)`(456·464) → **`.topk(k)` 한 줄 교체.**
  - `weights_i2t.topk(k)` = 이미지별 하드 텍스트(i2t 선택), `weights_t2i.topk(k)` = 텍스트별 하드 이미지(t2i 선택).
- positive는 topk와 **독립적으로 대각선을 명시 추가** → topk에서 밀려 사라지는 일 없음(VG로 positive 점수 낮아도 보장). 네거는 정확히 k개(diagonal이 0이라 topk가 순수 네거).
- `k ∈ [1, B−1]`, **k=B−1이면 각 행이 전체 네거 → full과 동치**(통합 경로).
- 선택 주체 = **학생 sim**. 티처는 **동일 인덱스** 재사용(같은 지지집합에서 비교).

**col(t2i) 지지집합 함정(설계 근거).** per-row(i2t)로만 뽑으면 column j의 지지집합 = "어느 row가 우연히 j를 골랐나"로 정해져 **통제 불능**(자주 positive-only로 붕괴). 그래서 t2i는 **방향별로 따로 뽑아야** 한다(= eval이 i2t·t2i 루프를 나눠 도는 이유). → 서브샘플 bidir = 선택 2벌.

### 4.3 서브샘플 구현 — fixed 2-gather 먼저, union은 측정 후

| 방식 | compute | shape | 코드 |
|---|---|---|---|
| **fixed 2-gather** | `2·B·(k+1)` | **고정** `[B,k+1]`×2 | eval 루프 2벌 그대로 |
| union-dedup | `|U|` (겹침 제거, ≤2·B·k) | **동적** | mask→nonzero→scatter→방향별 gather |

- **union이 반드시 fixed보다 싸다**(겹침 최소 B=대각선, 하드네거 상호성으로 오프대각 겹침도 실전 큼). `U=C_i2t∪C_t2i` dedup 후 한 방 계산 → `store[B,B,2]`에 흩뿌리고 방향별 `idx_*_pos`로 dense `[B,k+1]` gather(**-inf 불필요**: U 밖 셀은 gather 안 함 → `0·(−inf)=NaN` 회피).
- **동적 |U|는 eager 학습에선 대체로 무해**(compile/CUDA-graph 없음, matmul-bound, |U| 유계 → allocator steady-state). 거슬리면 |U|를 8배수 패딩(더미 ≤7). fixed 2-gather는 애초에 고정 shape.
- **ponytail: 서브샘플 = fixed 2-gather로 출발**(단순·고정), union은 프로파일에서 ITM-KD forward가 병목일 때만 승급.

---

## 5. 구현 위치 — 기존 패턴 재사용

### 5.1 `distillation/losses.py` — `itm_matrix_kd_loss(...)` (순수함수 신설)

```
def itm_matrix_kd_loss(student_logits, teacher_logits, direction, temp):
    # logits: full=[B,B,2] / 서브샘플=방향별 [B,k+1,2]. fp32. 티처 detach.
    # z1/z2(bidir) or z2(forward) × row/col softmax → forward KL(T‖S) → ×τ → 평균.
```
- `itc_distill_loss` 미러(입력에서 `.float()`, 티처 `.detach()`, `×temp`).

### 5.2 `distillation/online_teacher.py` — `OnlineTeacher.itm_matrix(...)`

```
@torch.no_grad()
def itm_matrix(self, image, caption, ...) -> Tensor:
    # eval evaluate_retrieval_itm의 per-row 스코어링(158-163) 재사용, top-k 대신 전체 B(또는 학생이 준 인덱스).
    # itm_head는 autocast 밖 fp32.
```
- `NEEDS['itm'] = {visual_encoder, text_encoder, itm_head}`, `CRITICAL_PREFIXES['itm']` 추가(현재 itc/lm만).

### 5.3 `models/blip_pretrain.py::forward` — config-gated 학생 매트릭스 + 손실

- 기존 `image_embeds`/`encoder_input_ids`/`text.attention_mask` **재사용**(재계산 0). full=row별 chunk, 서브샘플=gather. `text_encoder`+`itm_head`(fp32) → 학생 로짓.
- 티처 매트릭스를 **인자로 받아** `itm_matrix_kd_loss` 호출 — **itc_kd가 509에서 forward 안에서 하듯** 동일 위치. 반환 튜플에 `loss_itm_kd` 추가(off면 0/None).
- 서브샘플 시 학생이 뽑은 topk 인덱스를 반환/전달해 티처가 동일 인덱스 사용.

### 5.4 `pretrain.py` — 배선

- config gate `distill.itm.{enabled, weight, temp, direction, topk}`.
- `teacher_keep`에 `'itm'` 추가. model forward **전** `teacher_itm = online_teacher.itm_matrix(image, caption, ...)`(itc_feats처럼) → forward 전달.
- `loss = ... + λ·loss_itm_kd`. TB `loss_train/itm_kd` + `itm_kd/itm` 비율 모니터(CE 삼키면 λ↓).

### 5.5 config

```yaml
distill:
  itm:
    enabled: false      # 기본 OFF (게이트)
    weight: 1.0         # λ
    temp: 0.05          # τ (itc와 동일 오더, 스윕)
    direction: bidir    # {forward, reverse, bidir}
    topk: -1            # -1/≥B-1 = full B×B; k<B-1 = 서브샘플
```
`configs/pretrain_itm_matrix_kd.yaml`(baseline student 기반).

---

## 6. 정합성 — 티처가 같은 매트릭스를 본다

- full: 티처·학생이 **각자 같은 배치로 독립 조립**(all-pairs라 인덱스 공유 불필요).
- 서브샘플: 학생이 topk 인덱스 선택 → 티처 **동일 인덱스** 재사용.
- 이미지 전처리 공통(normalize 하나), 캡션 토크나이즈 각 tokenizer(vocab 동일). 블록/셀 순서 학생·티처 동일.

---

## 7. 성공 기준

1. **itm retrieval ≥ baseline** (증류 안 한 학생 대비). 최소 비열화, 목표 개선.
2. **rerank 이득 회복 — `val_retrieval_itm − val_retrieval_itc > 0`.** arm C(−1.2)·scheme A(≈0)가 죽인 그 이득. baseline은 +5~7.
3. **head separation 유지** — arm C(1.15→0.64)·scheme A(1.67→0.62)처럼 붕괴 안 함. `itm_sharpness_probe.py`로 실측.
4. **direction 어블레이션** {forward, reverse, bidir} 해석 가능한 차이. temp 스윕 단조.
5. `loss_itm_kd`가 CE를 삼키지 않음(비율 모니터).

---

## 8. 범위 밖 (non-goals)

- entmax/sparsemax(softmax가 측정 가능하게 실패하기 전엔 안 함).
- eval 랭킹 방식(z2 vs gap) 변경 — 별개 질문.
- ITC/LM KD 손실 변경 — itm만 추가.
- arm C / scheme A·B / target-mix / ITM 큐 계승 — 폐기·불가.
- Phase 2·3(서브샘플·union) 선제 구현 — 측정 게이트(§9).

---

## 9. 순차 구현 — Phase 게이트 (측정 기반)

셋 다 설계로 박되, **플랜은 Phase 1만 먼저 구현**, 2·3은 조건부:

- **Phase 1 — full B×B.** 개념 검증 + 가장 깨끗한 신호(전 네거). 스모크로 step time·메모리 실측. **baseline 이기면 여기서 끝일 수도.**
- **Phase 2 — 서브샘플(fixed 2-gather).** **Phase 1이 OOM이거나 iterate 불가할 만큼 느릴 때만.** `topk` opt-in. (full이 메모리 못 버티면 이게 첫 배포 config — 스모크가 결정.)
- **Phase 3 — union-dedup.** **프로파일에서 Phase 2 forward가 병목일 때만.** 안 지어질 수도.

---

## 10. 리스크 / 열린 질문 (스모크에서 해소)

- **학생 backward 메모리:** 1600 시퀀스 activation(baseline 3B=120의 13배). plain으로 스모크 → OOM이면 `text_encoder` gradient checkpointing(재계산 ~1.3× compute, 플래그). 학생 작아서(MiniLM, B=40) 그냥 될 수도.
- **티처 itm forward 비용:** BLIP-large 1600 forward(no_grad라 activation 無, 시간 비용). itc/lm 티처와 합쳐 step time 상승 — 실측.
- **동적 |U|**(Phase 3): eager 무해 예상, 실측 후 8배수 패딩 여부 결정.
- **τ 스윕:** itc(0.05) 오더에서 시작, itm 헤드 스케일이 달라 재튜닝 필요할 수 있음.
- **forward KD의 티처 오답 수입:** CE 앵커 + λ로 억제, separation 실측이 가드.
