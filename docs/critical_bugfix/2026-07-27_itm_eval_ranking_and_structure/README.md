# ITM retrieval의 구조적 한계 — scheme A 확정 + base-rate·큐 비대칭·z1 랭킹

**날짜:** 2026-07-27
**대상:** exp10.SA (scheme A, ITM tempered soft-target, W=1.0/T=2, `pt_itm_schemeA_tempered`, ep15/20 진행 중) + ITM 학습/평가 구조 전반
**선행:** `critical_bugfix/2026-07-24_itm_distill_sharpness/` (arm C 사망 + positive-ceiling 가설 + scheme A/B 설계). 이 문서는 그 후속.

**한 줄 결론:** scheme A는 예측대로 **양성 천장(positive ceiling)** 으로 실패했다(arm C와 정반대 축). 더 근본적으로 **티처 ITM 증류는 3중 구조적 어긋남**을 가진다 — (1) per-pair 분류를 랭커로 전용, (2) per-pair 손실은 관계적 랭킹을 못 나르고(티처 강점 유실·오보정만 수입), (3) 큐를 못 써 네거티브가 작고(40) 약함. 셋 다 큰 BLIP에선 마진에 묻혀 살아남지만 작은 학생에서 곱해져 붕괴한다. (별도로 의심했던 "eval이 z1로 랭킹해서 손해"는 §6 실험으로 **기각** — z1≈gap, 현행이 최고.) **랭킹 지식은 ITC 경로(exp9)로, ITM은 하드라벨 baseline로 두는 것**이 데이터·이론이 가리키는 답.

---

## 1. Scheme A 실측 — 양성 천장으로 사망 (arm C와 반대 축)

**config(확정):** `distill.itm_target_mix{enabled, neg_source: teacher, soft_weight(W)=1.0, temp(T)=2.0, schedule: constant}`, `distill.itc.enabled=false` (순수 ITM). `output/pt_itm_schemeA_tempered/config.yaml`.

**retrieval(TB, 수렴):** `val_retrieval_itm/r_mean ≈ 62.3` ≈ `itc 62.8` → **rerank 이득 ≈ 0** (baseline itm 70.9 = itc 63.4 + 7.5 대비 −8.6). arm C(59.7)보다 절대값만 조금 위, rerank 死는 동일.

**sharpness 프로브 (`probe_schemeA.py`, topk=32, limit=500, 같은 런 apples-to-apples):**

| 모델 | match_gap p50 | distractor_gap p50 | **sep(mean)** | match_z1 mean | distractor_z1 mean |
|---|---|---|---|---|---|
| baseline@ep15 | 4.89 | 3.32 | **1.67** | 2.49 | 1.68 |
| **schemeA@ep15** | **3.69** | 3.06 | **0.62** | 1.81 | 1.52 |
| schemeA@ep3 | 3.48 | 2.98 | 0.49 | 1.61 | 1.39 |
| *(2026-07-24) arm C@ep3* | *4.45* | *3.83↑* | *0.64* | | |
| *(2026-07-24) teacher* | *7.8* | *4.5* | *3.40* | | |

**separation 붕괴(1.67→0.62, Δ−1.05) 분해:**
- match_gap: 4.74 → 3.58 = **Δ −1.16** (sep을 깎음)
- distractor_gap: 3.07 → 2.96 = Δ −0.11 (오히려 sep에 +0.11 기여)
- → **붕괴의 ~110%가 match(양성) 하락.** distractor는 문제가 아니었음.

**메커니즘 = 양성 천장.** scheme A 양성 타깃 = σ(teacher_match/T) = σ(7.8/2) = σ(3.9) = 0.980 → 평형 match_gap ≈ 3.9. 실측 3.69(거의 도달, 플래토). 반면 **하드라벨 baseline은 스스로 4.89까지 올림.** 즉 tempered 티처 타깃이 하드라벨보다 **낮게** 양성을 캡했다(W=1.0으로 하드 앵커=무한 push 제거한 대가). **arm C는 반대** — W가 음성 거부를 약화시켜 distractor↑(3.41→3.83)로 사망. 두 실패는 다른 축, 공통 뿌리는 "하드라벨의 무한 push를 희석/제거".

---

## 2. 왜 잘하는 티처가 ITM을 못 가르치나 — relational vs per-pair

- 티처 ITM은 **좋은 랭커**지만 **나쁜 이진 보정기**다: 하드네거도 distractor_gap +4.5 = p=0.989로 "accept"한다. rerank가 되는 건 오직 **진짜 매치를 7.8로 압도적으로 더 올려** 마진(3.3)을 벌리기 때문. 즉 티처의 실력 = **관계적(쌍-간) 랭킹**.
- ITM 손실은 **쌍마다 독립 이진 CE**(cross-pair 비교 항 없음, §4). per-pair 채널로 티처를 증류하면 **관계적 랭킹은 못 실어나르고**, 각 쌍의 절대 match 확률(하드네거 0.989=accept)만 전달 → **티처의 약점(오보정)만 수입**.
- **마진 역설:** tempered 티처 타깃이 요구하는 pos−neg 마진 = σ(3.9) vs σ(2.25), 로짓 1.65. **하드라벨(∞)보다 작다.** 자신감 넘치는 티처가 오히려 하드라벨보다 작은 마진을 요구 → 더 나쁜 reranker. §1의 양성 천장이 이 역설의 실측.

---

## 3. base-rate(클래스 불균형)는 원인이 아니다

- ITM 배치 = 1 pos : 2 neg (positive 비율 1/3). 상수 예측의 CE 최소해는 `c*=1/3`(=prior) → no-match로 기움.
- **그러나 baseline·armC·schemeA가 이 base rate를 동일하게 공유** → 성공(baseline 70.9)과 실패를 가르는 요인일 수 없음. 분기는 base rate가 아니라 **증류 타깃**(§1).
- 게다가 프로브가 재는 하드네거는 g=+3.3(accept)로 prior 평형(−0.69)과 정반대 — **표현 신호가 prior를 압도.** prior는 "이미 풀린 쉬운 네거티브"만 지배, rerank가 걸리는 하드네거엔 불활성.
- 클래스 재조정(reweight/focal)은 **양성 천장을 못 올린다**: 하드라벨 양성 gradient ∝ (1−p)가 g=4.89에서 이미 포화(0.008)라 ×가중 무의미. soft 타깃 천장은 타깃값이 평형을 정하므로 가중치와 무관. → 천장을 올리는 유일 지렛대는 **확률이 아닌 로짓 타깃**(포화 탈출).

---

## 4. ITM vs ITC 네거티브 소싱 — 코드 확인

**ITM = in-batch 마이닝 (큐 아님), per-GPU 40** (`models/blip_pretrain.py:448-489`):
```python
weights_t2i = F.softmax(sim_t2i[:,:bs], dim=1)+1e-4   # [:,:bs] = 배치 내부만 (큐 컬럼 버림)
weights_t2i.fill_diagonal_(0)                          # 자기 정답 제외
neg_idx = torch.multinomial(weights_t2i[b], 1)         # 유사도 가중 확률 추첨(argmax 아님)
```
- 텍스트마다 네거티브 이미지 1 + 이미지마다 네거티브 텍스트 1 = 2·bs, `itm_labels=[1]*bs+[0]*2*bs` → **1:2, 총 3·bs=120** (bs=40).
- 4장(eff160)이어도 **ITM은 로컬 40만** 본다. `image_embeds`는 로컬, cross-GPU all_gather 없음. eff160은 grad all-reduce + 큐 적재 속도일 뿐.

**ITC = 큐 57600, single softmax** (`:266-273, :402-422`):
- `image_queue`/`text_queue` = `[embed_dim, 57600]` — **256-d projected pooled 벡터만** 저장.
- `sim = image_feat @ text_feat_all`, `text_feat_all = cat([momentum(40), queue(57600)])`. 쿼리당 **단일 softmax**.
- gradient: `∂L/∂s_pos=(q_pos−1)/τ`, `∂L/∂s_neg_j=q_j/τ`, `Σ q_j = 1−q_pos` → **총 네거티브 gradient 질량이 bounded, 정답 질량과 balanced.** 네거티브 수 = 불균형이 아니라 난이도. prior 지름길 없음(균등 예측 시 loss 폭발). 하드네거 자동 집중.

**큐를 ITM이 못 쓰는 구조적 이유:** 큐엔 256-d pooled 벡터뿐. ITC는 내적만 하면 되니 충분. **ITM은 텍스트 토큰↔이미지 패치[577×768] cross-attention 전체 forward가 필요**(`:477-485`)한데 큐엔 패치/토큰이 없다. 저장하면 이미지만 ~50GB+ & 샘플당 57600 forward → 불가능. **그래서 ITM은 풀 표현이 메모리에 있는 현재 배치로 강제 국한.**

**train/deploy 난이도 격차:** ITM 학습 네거티브 = 배치 39개 중 확률추첨(약함). 배포 rerank 네거티브 = ITC top-128 of 25000(가장 어려움). 헤드가 그만큼 어려운 네거티브를 학습 때 못 봄 → 일반화 열세(프로브 distractor 3.3의 한 원인).

---

## 5. eval이 z1(match 로짓)으로 랭킹 → loss-unconstrained offset 노이즈  ★신규

`eval_validation_tool.py:129-209`:
```python
score = model.itm_head(...)[:,1]                  # :163  z1 (match 로짓 원값, softmax 없음!)
score_matrix_i2t[start+i, topk_idx] = score + topk_sim   # :164  z1 + cosine
```
- ITM eval = ITC top-128(`k_test`)만 재점수 → `itm_eval`(`:200-209`)이 내림차순 정렬해 정답 rank로 R@1/5/10. **ITM은 top-128 안에서만 뒤섞기**(정답이 top-128 밖이면 못 살림).
- **랭킹을 z1(절대 match 로짓)으로 매긴다 — gap도 확률도 아님.** 그런데 ITM 학습(softmax CE)은 **shift-invariant**라 gap=z1−z0만 구속하고 **z1의 절대 offset은 loss가 방치**한다. 즉 `z1 = gap/2 + offset(h)`, offset(h)는 랭킹 관점 **노이즈**.
- 결과: 확신(gap 큰) 정답이 offset 낮아 밀리고, 애매(gap 작은) distractor가 offset 커서 1등 → 엉뚱한 선택.
- **큰 BLIP:** 마진(gap-sep 3.3)이 offset 노이즈를 압도 → z1/gap 랭킹 일치, 정답 우세 → "그냥 됨". **작은 학생:** z1-sep 0.30(§1) → 신호가 노이즈와 맞먹음 → z1 랭킹 붕괴.

| | gap 분리 | z1 분리 |
|---|---|---|
| baseline@ep15 | 1.67 | 0.82 |
| schemeA@ep15 | 0.62 | **0.30** |

→ eval이 쓰는 z1에서 scheme A는 정답-distractor를 0.30밖에 못 벌림(baseline 0.82의 37%).

> **⚠️ §6 실험으로 이 절의 "offset 노이즈가 랭킹을 망친다"는 실증적으로 기각됨.** z1 랭킹과 gap 랭킹의 R@k가 거의 같았다(현행 z1+cos이 오히려 최고). 이 offset-노이즈는 이론적으로 실재하나 **R@k엔 거의 영향 없어 실패의 원인이 아니다.** 이 절은 "테스트해서 배제한 가설"로 남긴다.

---

## 6. 재정렬 실험 — z1 랭킹 vs gap 랭킹  (`rerank_ranking_variants.py`)

**방법:** 동일한 ITC top-128 위에서 순서만 바꿔 R@1/5/10/MRR 비교. variants: `cos`(ITC만=no-ITM 바닥) / `z1`(offset 노이즈) / `z1+cos`(현행 eval) / `gap`(학습 최적화량=σ(prob)와 동순위) / `gap+cos`. baseline·scheme A, i2t+t2i, 500 queries/방향.

**가설:** gap 랭킹이 z1 랭킹보다 좋으면, "ITM 실패"의 일부는 헤드가 아니라 **z1-랭킹이라는 eval 선택** 탓(증류 무관, baseline까지 공짜 개선 가능).

**결과 (r_mean, top-128, 500 queries/방향):**

| variant | baseline | schemeA |
|---|---|---|
| cos (ITC만=no-ITM 바닥) | 66.36 | 65.87 |
| z1 (순수 로짓) | 71.20 | 63.06 |
| **z1+cos (현행 eval)** | **71.50** | **66.30** |
| gap (=softmax prob) | 71.40 | 63.77 |
| gap+cos | 71.40 | 65.03 |

**해석 — 가설 기각(honest negative):**
1. **z1 ≈ gap.** baseline 71.2 vs 71.4, schemeA 63.06 vs 63.77 — gap 랭킹이 z1보다 유의미하게 낫지 않다. **현행 z1+cos이 오히려 두 모델 다 최고.** → §5의 offset-노이즈는 이론적으로 실재하나 R@k엔 거의 영향 없음(±0.5~0.9). **"z1 때문에 엉뚱하게 정답 체크"는 실증적으로 기각.** eval 방식은 문제가 아니고, 고칠 아티팩트도 없다.
2. **강건한 발견은 그대로.** 모든 variant에서 schemeA(63~66) ≪ baseline(71~71.5). ITM 순수 기여(best − cos): baseline **+5.1**(71.5−66.36) vs schemeA **+0.4**(66.3−65.87). TB의 itm−itc(baseline +7.5 / schemeA ~0)와 정성 일치(절대치 차이는 top-128 내 500-query 서브샘플 탓).
3. **결정적:** schemeA의 **순수 ITM(z1 63.06 / gap 63.77)은 cos(65.87)보다도 낮다** → scheme A의 ITM 헤드는 단독으로 쓰면 ITC 랭킹을 **오히려 악화**시킨다(cosine이 얹혀 z1+cos가 겨우 +0.4). baseline ITM은 단독으로도 +5 기여(z1 71.2 ≫ cos 66.36).

→ **ITM 실패의 원인은 eval 랭킹 방식이 아니라 헤드 자체의 판별력 부족**(§1 양성천장, §2/§4 구조). z1-vs-gap 기각은 오히려 희소식 — 문제가 순수하게 구조적임을 확인.

---

## 6b. per-pair 이진 정답률 — 학습 타깃과 동일 (`binary_accuracy_by_hardness.py`)

**동기:** R@k(랭킹)·z1 말고 **학습 손실(binary CE)과 같은 타깃**으로 보자 — 이미지+정답캡션→match(gap>0) +1 / 이미지+오답캡션→no-match(gap<0) +1. **불균형(정답 1:오답 다수)** 처리: raw accuracy 대신 **balanced acc=(TPR+TNR)/2**, 네거티브를 HARD(top-32)/EASY(random16)로 분리, threshold-0(학습 경계)·OPT·AUC 3종.

**결과 (i2t, 500 imgs):**

| metric | baseline | schemeA |
|---|---|---|
| TPR (정답 match율) | 98.1 | 99.5 |
| TNR_hard (하드 거부율) | 5.9 | **0.7** |
| TNR_easy (쉬움 거부율) | 99.4 | 97.0 |
| **bal_acc_hard @th0 (=유저 정답률)** | **52.0** | **50.1** |
| bal_acc_easy @th0 | 98.7 | 98.2 |
| raw_acc_hard @th0 (불균형) | 18.4 | 14.0 |
| bal_acc_hard @OPT | 66.3 | 61.5 |
| **AUC_hard (마진)** | **70.9** | **63.9** |
| AUC_easy | 99.9 | 99.9 |
| mean_gap true / hard / easy | 4.12 / 2.97 / −6.92 | 3.28 / 2.93 / −3.73 |

**해석:**
1. **학습 타깃(이진 정답률)은 하드네거에서 degenerate.** bal_acc_hard@th0 = baseline **52** / schemeA **50** → 둘 다 ~동전던지기, **구분 불가.** TNR_hard 5.9%/0.7% → **둘 다 하드네거를 match로 찍음(거부 실패).** ITM은(티처·큰 BLIP 포함) 하드네거를 절대값으론 accept.
2. **헤드가 망가진 게 아니다 — EASY는 둘 다 98%+ 거부**(easy gap 크게 음수). **하드네거에서만 이진 결정 붕괴** = 배포형 난이도에서 calibration 실패(§4 train/deploy 불일치의 직접 증거).
3. **품질 차이는 마진에만.** AUC_hard baseline **70.9** > schemeA **63.9**, bal_acc_hard@OPT 66.3 > 61.5. 이진 결정(th0)은 못 잡지만 마진/순위는 잡는다. (마진 true−hard: baseline 1.15 vs schemeA 0.35.)
4. **schemeA는 거부가 전반적으로 약화**: TNR_easy 97.0<99.4, easy gap −3.7 vs −6.9. soft 타깃이 "다 좀 match"를 가르친 잔여 효과.
5. raw_acc(불균형) 14~18% 바닥 — "다 match" 찍어 하드 다수를 틀림 → raw accuracy가 왜 무의미한지 확인.

**결론:** "학습과 동일 타깃 이진 정답률"은 하드네거에서 **degenerate**라 ITM 품질을 못 잡는다. 품질은 **마진(AUC/순위)**에만 있고 거기서 schemeA가 열등(70.9→63.9). ITM이 작동하는 건 "좋은 이진 분류기라서"가 아니라 **"마진이 충분히 크면 순위가 맞아서"** — 큰 모델은 마진이 커서 되고, 작은 학생+증류는 마진이 좁아 무너진다. **per-pair 이진 분류(학습) ↔ 랭킹(배포)의 근본 불일치를 정량 확정.**

---

## 7. 종합 결론 + 방향

티처 ITM 증류는 **3중 구조적 어긋남**이 겹친다:
1. per-pair 이진 분류기를 **랭커로 전용** (§2).
2. per-pair 손실은 **관계적 랭킹을 못 나름** — 티처 강점 유실, 오보정만 수입 (§2).
3. 큐 불가로 **네거티브가 작고(40) 약함**, 배포(top-128)와 난이도 어긋남 (§4).

각각 큰 BLIP에선 마진에 묻히지만 작은 학생에선 곱해져 붕괴. base rate(§3)와 eval의 z1-랭킹(§5)은 **테스트해서 배제** — 전자는 공통 요인, 후자는 R@k에 무영향(§6).

**권고:**
- **랭킹 지식 → ITC 경로.** 구조적으로 정렬(단일 softmax, 큐, balanced grad, deploy 정렬). exp9 teacher-target-mix가 이미 통함.
- **ITM → 하드라벨 baseline 유지** (per-pair 분류엔 하드 앵커가 최적, itm 70.9가 우리 천장).
- (조건부) ITM을 굳이 살리려면: **양성=티처 raw-logit MSE(천장 올리기, 포화 탈출) + 음성=하드라벨(거부 유지)** 비대칭. 단 §4(작은 네거티브 풀)·§2(per-pair) 한계는 그대로라 상한은 낮을 것으로 예상. (eval 랭킹 변경은 §6에서 이득 없음으로 확인 → 불필요.)

---

## 8. 파일
- `probe_schemeA.py` — scheme A sharpness 프로브(§1). 결과 `schemeA_probe_results.json`.
- `rerank_ranking_variants.py` — z1 vs gap 재정렬 실험(§6). 결과 `rerank_ranking_variants_results.json`.
- `binary_accuracy_by_hardness.py` — per-pair 이진 정답률(TPR/TNR/balanced/AUC, hard vs easy)(§6b). 결과 `binary_accuracy_results.json`.
- 선행 조사: `../2026-07-24_itm_distill_sharpness/` (arm C, scheme A/B 설계).
