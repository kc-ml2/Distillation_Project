# ITM teacher-target-mix distillation — design

**날짜:** 2026-07-22
**상위 맥락:** ITC target-mix(exp9, `2026-07-14-teacher-target-mix-design.md`)가 효과적으로 검증됨
(onehot + momentum + teacher_soft 혼합). **다음 단계: ITM 손실의 증류.**
**실행:** config-gated 기능(기본 OFF, OFF면 baseline 동작). 브랜치/서버는 실행 시 결정.

---

## 1. 동기 & 핵심 질문

ITC 증류(exp9)의 교훈: **티처를 별도 손실이 아니라 타깃에 녹이면 gradient 충돌이 정의상 소멸**하고 성능이
오른다. ITC가 잘 증류된다는 건 **학생 모델에 아직 표현력 여유가 있다**는 신호 — 더 공격적으로 정보를 넣어볼 수 있다.

ITM(Image-Text Matching)은 cross-attention **융합** 위의 이진 판별(match/no-match)이다. 티처(BLIP-large,
~100× 크기)의 융합 판별은 학생보다 훨씬 정밀하다. 이를 증류해 학생의 fusion 판별을 끌어올린다.

**핵심 과학 질문:** 티처의 ITM 기여가 **false-negative 제거로 환원되는가**(→ dataloader idx 마스킹으로
충분), 아니면 **환원 불가능한 dark knowledge**(애매 쌍·노이즈 positive에 대한 calibrated 신호)가 있는가?
→ §3의 2×2 실험이 이 질문에 답하도록 설계됨.

**empirical 답(2026-07-22, `scratchpad/quantify_falseneg_reducibility.py`, 1024 negatives):** false-neg
163개(15.9%) 중 **same image_id(idx-마스킹 가능)=0%, diff image_id(의미 near-dup)=100%.** 사전학습 배치는
566k에서 랜덤 추출이라 같은 이미지가 한 배치에 중복될 일이 사실상 없음 → **idx-마스킹은 ~0% 해결.** false-neg의
본질은 "중복"이 아니라 "다른 이미지의 의미 유사"(말밭·인도·야구 등 흔한 장면)이고 **의미를 아는 티처만 잡는다 →
환원 불가능 확정.** A(순수 라벨링)의 강한 근거. graded prob(강한 0.9+ ~ 부분겹침 0.6)이 soft-label의 가치.

**성공 판정:** ITM-rerank `val_retrieval_itm/r_mean`(이미 학습 루프에 연결됨, `evaluate_retrieval_itm`)
≥ baseline(student solo). 부차로 ITC-only r_mean 비퇴행.

---

## 2. 현재 ITM 구조 (`models/blip_pretrain.py:435-489`)

1. **하드 네거티브 마이닝**: 학생 자신의 `sim_t2i`/`sim_i2t`에서 `softmax → 대각 0 → multinomial`로
   **배치 내(큐 제외)** 하드 네거티브를 확률적 샘플링(argmax 아님, 하드일수록 자주 뽑힘).
2. **3B 삼중항**: B(positive, label 1) + B(텍스트당 오답 이미지, 0) + B(이미지당 오답 텍스트, 0).
3. cross-attention `text_encoder` → `[CLS]`(position 0, `enc_token_id`)만 추출 → stack `[3B, D]`
   → `itm_head`(Linear D→2) → `vl_output[3B, 2]`.
4. **one-hot은 암묵적**: `F.cross_entropy(vl_output, itm_labels[3B])`가 정수 클래스 타깃으로 CE 계산.
   target-mix하려면 `F.one_hot(itm_labels, 2)`로 **명시적 [3B,2]**를 만들어 soft-target CE로 전환.
5. **false-negative 무방비**: `forward`는 `idx`/`image_id`를 안 받고 `fill_diagonal_(0)`만 함 →
   같은 이미지의 다른 캡션·generic 캡션·근접 중복이 오답으로 라벨될 수 있음. **학습 내내 상존하는 구조 문제.**

---

## 3. 실험 설계 — 선택 × 라벨링 2×2 factorial

두 독립 축으로 분해된다:
- **네거티브 선택 소스**: 학생 sim `|` 티처 sim
- **ITM 타깃**: 하드 onehot(W=0) `|` `(1−W)onehot + W·teacher_soft`(W>0)

|  | 하드 라벨 | 티처 soft-label |
|---|---|---|
| **neg = 학생** | **baseline** (현행, exp7 solo) | **A** — 순수 *라벨링* 효과 격리 |
| **neg = 티처** | **B** — 순수 *선택* 효과 격리 | **C** — 둘 다(공격적) |

- **A** (학생 선택 + 티처 라벨): 선택은 그대로 두고 **티처 라벨링만**의 이득 측정.
- **B** (티처 선택 + 하드 라벨): **티처 sim은 itc 경로서 이미 계산됨 → 새 forward·새 손실 0** (거의 공짜).
  티처가 100× 크기로 더 나은 하드 네거티브를 고른다는 가설 검증.
- **C** (티처 선택 + 티처 라벨): 가장 리치·가장 비쌈. 티처 ITM forward 필요.

**환원성 판별:** A(라벨링)가 baseline 대비 유의미하면 → 티처엔 false-neg 제거를 넘는 dark knowledge 존재
(dataloader 마스킹으로 대체 불가). B가 주 이득이면 → 선택/false-neg 문제이므로 **dataloader 마스킹이 더 싸고
깨끗한 해**일 수 있음(향후 실험과 비교).

**프로파일링 함의(§6.3, B 리스크):** 티처 선택(scale 63.9)은 **가장 하드한** negs를 뽑는데 그중 **~15%가
false-neg**(티처 기준 실제 match). 즉 **B(티처선택+하드라벨)는 "더 하드한 negs 이득" vs "false-neg 하드-0
손해"가 상충** → baseline 대비 개선 불확실(하락 가능). **C는 그 false-neg를 soft label로 교정**하니 선택·라벨링
시너지. **예측 서열: C ≥ A ≥ baseline, B는 애매.** 이 서열 자체가 §1 환원성 질문의 답을 구성한다.

---

## 4. 조립 아키텍처 — forward-side, `online_teacher.itm_soft` 캡슐화

**결정적 제약:** ITM 손실은 forward 안에서 계산되므로 `teacher_soft`가 **forward 내부 손실 시점에 존재**해야 함.
그러려면 네거티브 인덱스가 필요한데, **A는 인덱스가 학생 sim(=forward 내부)에서만 나온다.** 따라서 pretrain.py에서
미리 만들어 넘기는 방식은 A를 구조적으로 지원 못 함 → **조립·티처채점은 forward 쪽**이어야 세 실험을 한 경로로 덮음.

```
forward(image, caption, ..., online_teacher=None, itm_mix_cfg):
  # sim_i2t, sim_t2i는 ITC 손실·네거티브 마이닝에 이미 계산됨
  if neg_source == 'teacher':
      w = softmax(teacher_sim)   # teacher_img_feat/text_feat로 즉석 계산 (공짜, 이미 인자로 들어옴)
  else:
      w = softmax(student_sim)   # A/baseline
  neg_idx_img, neg_idx_txt = multinomial(w)          # ← 인덱스 확정(한 곳)

  # 학생 3B 조립 → cat 1회 forward → student_vl_output[3B,2]

  if W > 0 and online_teacher is not None:
      teacher_soft = online_teacher.itm_soft(image, enc_ids, attn,
                                             neg_idx_img, neg_idx_txt, temp=T)   # [3B,2]
      onehot = F.one_hot(itm_labels, 2).float()
      target = (1-W)*onehot + W*teacher_soft
      loss_itm = -(target * F.log_softmax(student_vl_output, 1)).sum(1).mean()
  else:
      loss_itm = F.cross_entropy(student_vl_output, itm_labels)   # 기존 경로 그대로
```

**이점:** (1) 인덱스가 한 곳에서 확정 → 학생·티처가 **같은 인덱스**로 각자 3B 조립(정합 보장). (2) 티처 sim
선택은 공짜(feat 이미 있음). (3) 3B 조립 로직 학생·티처 각 한 벌(pretrain.py 중복 없음), 티처 조립은 메서드로
캡슐화 — forward는 인덱스만 주고 `[3B,2]`만 받음(itc/lm과 같은 텐서 계약). (4) `online_teacher None` 또는
`W==0` → **완전한 기존 코드**(회귀 안전). **B는 W=0이라 티처 ITM forward 스킵 → 공짜 유지.**

**3B forward 병합:** 학생·티처 모두 `cat` 후 **1회 3B forward**(cross-attn은 예제별 독립이라 결과 동일).
단 배치 matmul 결합순서로 **bit-identical은 아님**(수치적 동일, << seed 노이즈). exp7 baseline은 유효한 참조로
유지 가능하나, 엄격한 귀속을 원하면 target_mix OFF 신규 baseline 재실행.

---

## 5. `OnlineTeacher` 변경 (`distillation/online_teacher.py`)

- `NEEDS['itm'] = {"visual_encoder", "text_encoder", "itm_head"}` 추가. cross-attn `text_encoder`는
  itc와 **동일 모듈** 재사용(BLIP med.py가 `mode`로 text/multimodal 겸용). `itm_head`만 현재 stripped →
  keep에 itm 있으면 보존.
- `CRITICAL_PREFIXES['itm']` 추가(로드 검증).
- 신규 `itm_soft(image, enc_ids, attn, neg_idx_img, neg_idx_txt, temp)`:
  `visual_encoder(image)`→full embeds → **학생과 동일 인덱스로 3B 조립(cat 1회)** → cross-attn `text_encoder`
  → `itm_head` → `softmax(/temp)` → `[3B,2]`. bf16 autocast + `no_grad`.
- **keep은 실험별로 다름**: **B** = `('itc',)` (티처 sim 선택용 feat만, itm 채점 없음 → 공짜),
  **A** = `('itm',)` (학생 선택이라 티처 sim 불필요, 채점만), **C** = `('itc','itm')` (선택 + 채점).
  즉 `neg_source=teacher`면 itc, `W>0`이면 itm이 keep에 필요. 티처 itc feat은 **선택 전용**이며 ITC KD
  손실은 켜지 않는다(§3 "baseline 동일"). 텍스트 토큰은 학생·티처 동일(LM distill의 `torch.equal` assert가
  토크나이즈 일치 이미 보증).
- **비용 최적화(후속)**: itc 경로가 이미 `visual_encoder(image)` 실행 → image_embeds 1회 계산 후 itc feat·itm
  둘 다에 재사용하면 티처 visual forward 중복 제거. 첫 구현은 단순, 필요 시 캐싱.

---

## 6. 하이퍼파라미터 — 결정과 근거

### 6.1 스케줄 = **상수 W** (exp9 감쇠 미채용)

exp9에서 티처가 감쇠한 이유: (a) momentum이 후반에 신뢰 가능해져 soft-slot 인수, (b) 후반 타깃=baseline이라
**후반 손해 구조적 불가** 안전장치. **ITM엔 momentum이 없어 둘 다 소멸** — 티처를 감쇠시키면 넘겨받을 대상 없이
타깃이 **순수 하드 라벨로 수렴**한다. 그런데 **false-neg는 학습 내내 상존**하고 오히려 **후반(학생 판별력↑)에
하드-0 false-neg가 더 강한 오답 gradient**를 준다 → 감쇠는 *보호가 가장 필요한 시점에 보호를 없애는* 역설.

∴ ITM에서 티처는 warm-up 크러치가 아니라 **영구 정규화기**. **상수 W** 채택. (hedge 필요 시 floor 감쇠
`W→W_min`; exp9-literal `→0`은 비권장.)

### 6.2 W (soft-slot 무게) = **0.4**

`target = (1−W)onehot + W·teacher_soft`. 티처는 **명확한 예제엔 onehot과 거의 동일**한 값을 줌(positive
≈`[0.03,0.97]`, 진짜 하드 네거티브 ≈`[0.92,0.08]`) → 섞어도 거의 무변화. **false-neg(≈`[0.3,0.7]`)에서만
크게 작동**(하드-0 → `[0.72,0.28]`로 완화). 즉 W는 애매 예제에서만 물어서 **키워도 clear 예제엔 안전**.
exp9 parity로 **0.4** 시작. sweep 후보 `{0.3, 0.5}`.

### 6.3 T (티처 ITM softmax 온도) = **1.0** (프로파일링으로 검증 — go/no-go 겸용)

`teacher_soft = softmax(teacher_itm_logits / T)`. 2-way라 단순 Bernoulli sharpness. **T=1**은 티처 native
캘리브레이션 그대로. T↑은 dark knowledge↑지만 **진짜 네거티브까지 match로 새어** 노이즈; T↓은 dark knowledge 소실.

**미지수:** BLIP-large ITM이 false-neg에서 얼마나 confident한가. saturated(`[0.95,0.05]`)면 T=1에선 신호가
거의 없어 A/C 무의미 → T↑ 필요. **→ 본런 전 1-스텝 프로파일링**: 하드 네거티브에서 티처 match-prob 분포를 찍어
애매 구간(`0.2~0.8`) 질량 확인. **이 프로파일링이 (a) A/C의 go/no-go, (b) T 설정을 동시에 해결**. 기본 T=1.0,
sweep `{1.5}`.

**프로파일링 결과(2026-07-22, COCO 384샘플, 티처선택 negs, `scratchpad/profile_teacher_itm.py`, 티처
temp=0.0157→scale 63.9):** 티처 ITM은 **true-neg엔 포화(73.6%가 <0.05), false-neg엔 near-1로 강하게
flag** — 신호가 필요한 곳에만 실린다. NEG(all): 애매[0.2,0.8]=7.9%, **match>0.5(false-neg)=14.7%**,
mean 0.157. POS: mean 0.982지만 5%가 <0.93(min 0.07, 노이즈 캡션). **T=1이 false-neg만 교정하고 true-neg는
near-onehot 유지 → T=1.0 확정**(T↑는 true-neg까지 흐려 노이즈만 추가). **A/C GO(강)**: ~15% false-neg +
~8% 애매 = ~23%가 하드라벨이 틀리거나 뭉개는 신호. (프로파일링 전 "saturated면 신호 없음" 우려는 기우 —
saturated가 true-neg 쪽이라 오히려 이상적.)

### 6.4 적용 범위 = **all-3B** (positive 포함)

티처 가치가 주로 네거티브에 있지만 positive에도 티처는 near-onehot이라 **무해**하고, BLIP 사전학습 데이터의
**노이즈 캡션(나쁜 positive)**을 티처가 softening하는 것도 이득. **프로파일링 확인**: positive 5%가 match<0.93
(min 0.07) — 노이즈 캡션 실재, all-3B가 이 작은 신호도 포착. neg-only 변형은 후속 ablation.

---

## 7. Config 스키마

`distill` 블록에 신규 `itm_target_mix`. 기본 OFF(미설정 시 baseline 완전 동일). exp9의 `itc_target_mix`와 독립.

```yaml
distill:
  itm_target_mix:
    enabled: true
    neg_source: teacher     # 'teacher'(B/C) | 'student'(A/baseline)
    soft_weight: 0.4        # W. clear 예제 무영향, 애매 예제만 작동. W=0 → 하드 라벨(B)
    temp: 1.0               # T. 프로파일링으로 확정
    schedule: constant      # 'constant'(권장) | 'decay_to_floor'(hedge)
    # decay_to_floor 택 시: hold_epochs: 2, decay_end_epochs: 12, w_min: 0.15
teacher:
  arch: 'blip_large'
  checkpoint: '.../model_large.pth'
```

세 실험 config: `10.A`(neg_source=student, W=0.4), `10.B`(neg_source=teacher, W=0),
`10.C`(neg_source=teacher, W=0.4). 나머지는 student baseline(exp7) + caption/retrieval-val ON 동일.

**검증(assert):** `neg_source ∈ {teacher, student}`; `0 ≤ soft_weight ≤ 1`; `W>0`이면 keep에 itm 포함;
`neg_source=teacher`면 keep에 itc 포함; `W>0` 또는 `neg_source=teacher`면 `teacher.checkpoint` 존재;
`schedule ∈ {constant, decay_to_floor}`.

**pretrain.py teacher_keep 파생:** 기존 `('itc','lm')` enabled 플래그 + `itm_target_mix`에서
`neg_source=teacher → 'itc'`, `W>0 → 'itm'`를 합집합으로 산출.

---

## 8. 범위 밖 (YAGNI)

- **dataloader idx-마스킹으로 false-neg 제거** — 측정 결과 false-neg의 **0%만 idx-마스킹 대상**(§1)이라
  이 문제엔 **비대안**으로 판명. 유의미하려면 semantic dedup이 필요한데 그건 티처 재구현에 가까움 → 대조군 가치 낮음, 보류.
- neg-only target-mix(positive 하드 유지) — 후속 ablation.
- W·T 본격 스윕 — 첫 3런(A/B/C) 후 필요 시.
- floor/비선형 감쇠 — 상수로 시작.
- 티처 *ITM* score로 네거티브 선택(ITM-native B) — 티처 후보풀 채점 비용이 A와 동일해 비권장(싼 B는 ITC-guided).

---

## 9. 테스트

- 단위: `itm_soft` 형상(`[3B,2]`, softmax 합=1), 같은 인덱스로 학생·티처 3B 정합, one-hot materialize 정확성
  (positive→`[0,1]`/negative→`[1,0]`), target 무게 합=1·비음수(임의 W), `W=0` → soft-CE == 기존 CE.
- 통합: target_mix OFF → forward 출력이 baseline과 수치적 동일(회귀). 짧은 스모크로 A/B/C 크래시 없이 loss 감소
  + TB에 `train/itm_teacher_weight`(=W) 기록.
- **프로파일링(§6.3)**: 본런 전 하드-neg 티처 match-prob 히스토그램 → T 확정 + A/C go/no-go.
