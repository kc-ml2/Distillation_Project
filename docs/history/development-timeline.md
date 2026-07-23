# Distillation Project — 시간순 개발일지

> **기간:** 2026-05-12 ~ 2026-07-15 (약 9주, 전 브랜치 118커밋)
> **목적:** 개발 흐름을 시간 순서대로 따라가며 "언제, 왜, 무엇을" 했는지 이해하기 위한 문서.
> 기술적 깊이가 필요하면 자매 문서 [technical-report.md](technical-report.md) 참조.
> **원자료:** git 전 브랜치 로그, `docs/superpowers/` 설계문서 7건, `critical_bugfix/` 진단 2건, 실험 메모.

---

## 한눈에 보는 연대기

| 기간 | 국면 | 한 줄 요약 |
|---|---|---|
| 05-12 ~ 05-19 | A. 아키텍처 셋업 | BLIP 베이스 임포트, DINOv3 ViT + MiniLM 학생 모델 골격 |
| 05-27 ~ 06-08 | B. 첫 pretrain 가동 | 소형 모델 CC12M 학습 시작, 수렴 문제 첫 등장 |
| 06-03 | C. 데이터 확장 | Visual Genome 추가 (COCO+VG 체제) |
| 06-15 ~ 06-23 | D. 수렴 위기 해결 | temperature collapse 진단 → 4-실험 ablation → logit_scale 재파라미터화 |
| 06-21 ~ 06-23 | E. Retrieval 평가 인프라 | 학습 중 COCO retrieval validation 체계 구축 |
| 06-27 ~ 06-28 | F. ITC 증류 v1 | offline teacher cache 방식 구현 |
| 06-29 | G. Online teacher 피벗 | 캐시 폐기, frozen BLIP-large 온라인 티처로 전환 |
| 07-02 ~ 07-05 | H. LM logit 증류 | exp8 구현·완주 → **CIDEr +2.5% 확정 성과** |
| 07-03 ~ 07-07 | I. 캡션 zero-shot 평가 | config-driven 캡션 평가 파이프라인, main 머지 |
| 07-06 ~ 07-14 | J. ITC 증류 진단의 계곡 | ×τ² 버그 발견 → τ 실측 선정 → exp6 완주 → grad 압도 진단 |
| 07-13 | K. 캡션 in-loop 평가 | 학습 루프에 BLEU/CIDEr/SPICE 상시 평가 탑재 |
| 07-14 ~ 현재 | L. 두 갈래 재도전 | Phase 1 (armT/armS) 가동 중 + exp9 (teacher-target-mix) 대기 |

---

## Phase A — 학생 모델 아키텍처 셋업 (05-12 ~ 05-19)

**시작점:** Salesforce **BLIP 공식 코드**를 그대로 임포트 (`15fe4f7`, Initial commit).

목표는 BLIP의 ViT-base + BERT-base 구조를 **소형 백본으로 교체 가능한 구조**로 만드는 것.
초기 2주간 세 갈래 브랜치로 병렬 작업했다:

- **`test_vit_output_dim`** — 비전 쪽. DINOv3 wrapper 작성, register token 제어 (`236d79b`),
  `global_pool=''` 등 출력 차원 문제 해결 (`87c70ab`). blip_pretrain에 `small`/`small_plus` ViT 옵션 추가 (`18fcab6`).
- **`dev_text`** — 텍스트 쪽. BERT를 교체 가능하게: `my_bert_size` 옵션, minilm/medium config 추가
  (`c96b2c8`, `e38d6d7`, `0eb4ffa`). Lite-BERT distillation 코드 반입 (`ce77f57`).
- **공통 정비** — config 소문자 통일 (`e6c58f4`), 경로 정비 (`34f44d5`), loss curve 도구 (`337e70c`).

> 이 시기 커밋은 짧은 영어 메모 스타일. 설계 문서 없이 커밋 메시지로만 재구성됨.

## Phase B — 소형 모델 첫 pretrain (05-27 ~ 06-08)

- **minilm + dinov3 조합 완성** (`e280d4c`) — 학생 모델의 최종 형태가 여기서 성립.
  decoder의 `encoder_width=vision_width` 연결 (`bebe8fe`).
- CC12M으로 소형 모델 pretrain 시작 (`d00543c`), 서버 다운 대비 백업 커밋 (`5c35832`).
- 관측 인프라의 시작: TB train 메트릭 (`705133b`), **val loss (ita/itm/lm)** 추가 (`b59aa82`).
- **`d5726b1` (06-08) "temperature scalar added — problem on converge"** —
  이후 2주를 지배할 temperature collapse 문제가 히스토리에 처음 기록된 순간.

## Phase C — Visual Genome 추가 (06-03)

- `vg_captioning` 함수 + 이미지 루트 처리 (`e37b101`), 데이터셋 컨트롤 폴더 (`326803a`).
- 이후 데이터 구성: **VG 5.4M / COCO 566K 캡션** — VG가 mixture의 ~90%를 차지하게 됨.
  이 비대칭이 Phase D의 데이터 버그, Phase J의 티처 노이즈 문제로 두 번 되돌아온다.

## Phase D — Temperature collapse: 진단과 해결 (06-15 ~ 06-23)

**증상:** baseline 런이 epoch 0.5~4는 안정 (logit_scale 38~58 plateau)하다가 epoch ~4.3부터
gradient 폭주 → epoch 6에 클램프 상한(scale ~1000, temp 바닥 0.001)에 고정, `loss_val/ita` 동반 붕괴.

**분석:** weight decay의 temp 기여는 ~1.5%/epoch뿐인데 관측 변동은 30~40% → decay가 아니라
`1/temp` 나눗셈 구조의 **1/temp² gradient 증폭**이 주범이라고 판단. 독립 용의자 둘을 분리:

1. **VG 데이터 버그** — region 캡션이 bbox가 아닌 **전체 이미지**를 쓰고 있었음. `b6a0429`에서
   bbox crop으로 수정 (annotation 재생성 포함).
2. **temp 파라미터화** — 원본 BLIP과 동일한 구조라 "fork가 깨뜨린 것"이 아니라 더 미묘한 문제.

**4-실험 ablation** (전부 dev 커밋, 모두 VG fix 위에서):

| exp | 커밋 | 내용 |
|---|---|---|
| 1.vg_crop | `4cd2924` | VG fix만 (대조군) |
| 2.no_decay_temp | `d059128` | + temp를 weight decay에서 제외 |
| 3 | (skip) | reparam만 — 수학이 이미 답을 가리켜 GPU 절약 차원에서 생략 |
| 4.logit_scale_no_decay | `d0a9b33` | **CLIP식 reparam**: `sim × exp(logit_scale)`, log-space 클램프, decay 제외 |

부속: collapse 자동 정지 트리거 (`2beaed8`, threshold 700 지속 10000스텝), epoch-7 예산 캡 제거 (`30a5209`).

**결말:** exp4 구성이 표준 레시피로 정착. base 모델 r_mean 66.2~66.7 도달.
훗날 exp5-lrup이 logit_scale 110.9로 retrieval 최고(67.97)를 찍으며 **"높은 scale 자체가 병"
가설은 기각** — scale은 증상이지 원인이 아니라는 교훈이 Phase J 진단의 복선이 된다.

## Phase E — Retrieval validation 인프라 (06-21 ~ 06-23)

설계문서: `docs/superpowers/specs/2026-06-21-pretrain-retrieval-validation-design.md` (첫 SDD 문서).

- `RetrievalValRunner` — 학습 중 COCO 5k retrieval 평가 (`13049fe`), ITC-only vs ITM-rerank
  2티어 분리 (`614c98b`), pretrain.py 배선 (`95f7bde`).
- standalone 체크포인트 평가 (`d8b84eb`) + **체크포인트 sweep + TB 비교 툴링** (`be08f23`).
- CUDA device/dtype 버그픽스 + bf16 autocast (`263b021`), ITM은 에폭당 1회로 경량화 (`f270c8d`).
- 케이던스 확정: **ITC 1000스텝마다 + ITM 에폭당 1회**.

> 이때부터 "실험 전에 측정 도구부터"가 프로젝트의 작업 패턴으로 자리잡는다.
> rank 기반 retrieval이 유일하게 scale-불변인 지표라는 사실이 Phase J에서 결정적 역할을 한다.

## Phase F — ITC 증류 v1: offline teacher cache (06-27 ~ 06-28)

설계: `2026-06-27-itc-distillation-design.md`. 브랜치 `dev/itc_distill`.

- `itc_distill_loss` — in-batch **B×B relational KL** (`e9ec273`).
- 티처 feature를 **미리 캐시** (manifest + fp16 memmap, `d3ef234`) + `build_cache` CLI (`cc89780`).
- dataset이 인덱스로 teacher feats 반환 (`e7391f2`), train loop에 weighted itc_kd (`b5b937f`).
- 이미지 aug 끄기 토글 + config-driven exp tag (`b988045`) — no-aug(exp5)가 go/no-go 게이트.

## Phase G — Online teacher 피벗 (06-29, 단 하루)

설계: `2026-06-29-itc-distillation-online-teacher.md`.

offline 캐시는 **aug ON 학습과 캐시(무-aug 뷰)의 불일치** 문제가 있었다. 하루 만에 방향 전환:

- 캐시 전면 제거 + dataset/aug 원복 (`9650845`).
- `OnlineTeacher` — frozen BLIP-large를 매 스텝 같은 증강 배치에 forward (bf16, no_grad) (`bef69b5`, `6fc96df`).
- aug 복원 (`f5dc810`), 티처 ckpt 로드 loud check (`a382986`).

> 매몰비용 없이 이틀치 캐시 구현을 버린 결정. 이후 모든 증류는 online teacher 체제.

## Phase H — LM logit 증류: 첫 확정 성과 (07-02 ~ 07-05)

설계: `2026-07-02-lm-logit-distillation-design.md`. 브랜치 `lm_distill`, exp `8.lm_distill`.

- `lm_distill_loss` — token-level KD (shift frame, pad mask, ×T², T=2.0) (`2bd7817`).
- `OnlineTeacher.lm_logits` — teacher-forced 디코더 로짓 [B,30,30524] (`313a529`).
- train loop 배선 + TB + run-name tag (`05a577f`), config (`9b35c8a`).
- **`8e6f07c`** — timm 1.x에서 vit='large' 티처 생성 크래시 수정 (legacy backbone init 게이트).
  이 커밋이 이후 dev의 새 베이스가 된다 (07-06 fast-forward 머지).

**런:** 07-02 15:44 시작 → 07-05 18:59 **epoch 20 정상 완주** (모니터링 alert 0회).

**결과 (zero-shot 캡션 A/B, checkpoint_19 매칭):**
- small_distill vs small_solo: **CIDEr 1.103 vs 1.076 (+2.5%)**, **SPICE 0.207 vs 0.203 (+2.0%)**
- distill > solo가 **전 에폭(2·5·7·10·13·16·19)에서 성립**.
- solo의 SPICE는 ep10에서 완전 플래토, **distill만 계속 상승** — 증류가 티처로부터
  추가 의미 내용을 계속 추출한다는 가장 강한 증거.
- 부수 발견: 티처 자체가 pretrain ckpt라 val LM CE headroom이 ~0.03 nats뿐
  → "CE로는 증류 효과가 안 보이는 게 정상, downstream 캡션 지표로 봐야 한다"는 평가 원칙 수립.

## Phase I — 캡션 zero-shot 평가 툴링 (07-03 ~ 07-07)

설계: `2026-07-03-caption-minilm-zeroshot-eval-design.md`. 브랜치 `claude/caption-minilm-eval`.

- 캡션 디코더 BERT를 `my_bert_size`로 선택 (`5483617`), DINOv3 pos_embed 가드 (`eefc2a3`).
- `use_spice=False` 경로 (`bbc7b52`), config-driven eval — base/large/minilm (`82a6c74`),
  generate 파라미터 노출 (`125be06`), 계약 테스트 (`f1e0648`), checkpoint arch override (`dfbe304`).
- **`8547ed8` main 머지 (07-07)** — 현재 main tip. 모델 크기 순위 확정:
  large(1.236) > base official(1.150) > base amp(1.120) ≈ base nodecay(1.113) > small distill(1.103) > small solo(1.076).
  **small(73.8M)이 base(247.2M)의 30% 파라미터로 CIDEr 96% 달성.**

병행 실험: VG 전용 `min_scale 0.5` (`cbe9d30`, 브랜치 `extra_exp/vg_transform_min0.5`, base 모델)
→ 07-14 완주, **exp4 대비 무승부** (66.34 vs 66.20, 노이즈 범위). "VG 문제의 지렛대는 crop이 아니라
region 크기/내용 선택"이라는 함의만 남김.

## Phase J — ITC 증류 진단의 계곡 (07-06 ~ 07-14)

이 프로젝트에서 기술적으로 가장 밀도 높은 구간. 기록은 `critical_bugfix/` 두 문서에 있다.

**J-1. ×τ² 버그 발견 (07-07, `critical_bugfix/README.md`)**
exp 6.itc_distill 가동 중 TB에서 `loss_train/itc_kd ≈ 0.0034` — 다른 loss 대비 **~2000× 작음**.
원인: Hinton KD의 ×T² 관례를 무비판적으로 이식. Hinton 보정은 T>1(softening) 전제인데
여기 τ=0.05<1(sharpening)이라 gradient를 오히려 죽였다. KL 자체는 1.36 nats로 정상.

**J-2. 티처 유사도 실측 → τ=0.022 선정 (07-07, `2026-07-07_teacher_sim_and_tau/`)**
"τ를 감이 아니라 실측으로": BLIP-large를 실제 학습 배치에 forward, 4000 pos / 156k neg 측정.
- pos 코사인 0.42 / neg 0.31 — 전체가 좁은 콘에 압축. **pos−hardest_neg 겨우 0.016** (aug ON).
- aug가 positive만 −0.026 훼손 (마진 절반). argmax=diag **58.7%** = 옮길 수 있는 신호의 상한.
- τ 스윕: τ≥0.1은 즉시 균등화("τ 높여 soft KL" 직관 기각). **τ=0.022 선정**
  (pos-prob 0.46, 학생 수렴온도 0.023 앵커).

**J-3. exp 6.1/6.2 완주 → 역설적 결과 (07-14)**
- 6.1 (droptau: τ=0.022, ×τ² 제거, λ=1): **ep1 r_mean 40.2** (baseline 21.9, +18 — 티처 신호의
  가치 증명!) 그러나 ep20 **59.5 vs baseline 63.4 (−3.9)**.
- 6.2 (keeptau2): ×τ²=4.84e-4가 유효 ratio를 0.003으로 → **no-op 대조군** 확인 (62.7 ≈ baseline).

**J-4. Phase 0 진단 (07-14, `2026-07-14_kd_grad_balance_and_geometry/`)**
"왜 val ITC loss는 내리면서 retrieval은 나쁜가?"에 대한 3종 실측:
- **grad probe:** λ=1에서 |g_kd|/|g_ita| = **6.7→2.6** (전 구간 KD가 ITC 압도), cos(g_kd, g_ita) 전 조합 음수.
  초반 순 갱신이 ITC 방향과 **음의 정렬(−0.6)** — 문자 그대로 주 손실을 거슬러 걸었다.
- **학생 기하:** KD가 margin 0.082→0.054로 압축, ITC는 logit_scale 59→88 인플레로 정확히 상쇄
  (**margin×scale 4.84 vs 4.77 — 불변**). 이 줄다리기의 순위 재배열이 retrieval 손상.
- **Hinton crossover 실측:** τ≤0.05는 1/τ regime(|g_kd|·τ 상수), τ≥0.15만 Hinton regime(|g_kd|·τ² 상수)
  → **우리 구간의 올바른 보정은 ×τ** (×τ²도 ×1도 아님).
- 지표 원칙 확립: **val ITC loss는 scale 오염 지표. 런 간 비교는 rank 기반 retrieval로만.**

## Phase K — 캡션 in-loop 평가 (07-13)

설계: `2026-07-13-caption-metric-eval-design.md`. Phase J 진단과 병행 구축.

- `BLIP_Pretrain.generate()` (`ee61bb6`), `CaptionValRunner` — DDP 샤딩 생성 → gather/dedup →
  coco_caption_eval → TB (`0f9cc0a`), train loop 배선 (`889dec5`), rank0 채점 non-fatal 하드닝 (`2237836`).
- SPICE의 CPU 독점 문제 → `cpu_affinity` 코어셋 제한 (`89a5550`). 실측: 32C=62s (64C과 동속).
- 에폭 실경과 시간 로그 (`2326127`). **`de9ee01` dev 머지.**
- 케이던스: 경량 지표 2×/epoch, SPICE 1×/epoch. 다음 런부터 자동 동작.

## Phase L — 두 갈래 재도전 (07-14 ~ 현재)

Phase 0 진단이 가리킨 해법 두 가지를 **동시 추진**:

**갈래 1 — Phase 1: λ 크기 정합 (브랜치 dev, `fe23681`)**
- `itc_distill_loss`에 **×τ 보정** 채택 + 단일 공유 **λ_shared=2.5** (목표 ep0 ratio ~0.4).
- **Arm T** (τ=0.022, sharp, 서포트 4.2) vs **Arm S** (τ=0.05, soft, 서포트 19.4) —
  "같은 압력, 다른 sharpness". config 2개, caption-val ON, SPICE CPU 0-31/32-63 분리.
- **07-14 14:26부터 두 arm 동시 가동 중** (4GPU MPS, eff batch 160).
- 게이트: ep20 r_mean **≥63.4 필수** (student solo 초과), 목표 66+ (base 모델급).

**갈래 2 — exp9: teacher-target-mix (브랜치 `exp9_teacher_target_mix`, 9커밋)**
설계: `2026-07-14-teacher-target-mix-design.md`. 발상의 전환 — 별도 KD loss로 싸우게 두지 말고
**티처를 ITC soft-target에 녹여 손실을 하나로** → grad 충돌이 정의상 소멸.
- `target = (1−W)·one-hot + W·[γ·teacher + (1−γ)·momentum]`, W=0.4.
- γ 스케줄: 2ep 홀드(티처 100%) → ep12까지 선형 감쇠 → 이후 baseline과 동일 (`02f4399`).
- variant 9.1(in_batch) / 9.2(queue — 티처 feature 큐 페어 enqueue) (`01b60e5`~`e140c71`).
- γ=0 구간 티처 forward 스킵 + 불변식 assert (`3c24ec9`, `69a8107`) — 런당 ~1일 절약.
- 상태: 단위테스트 9/9, CPU 스모크, 리뷰 SHIP READY. **스페어 서버 GPU 스모크 → 본런 대기.**

---

## 부록 1 — 브랜치 지도

| 브랜치 | tip | 분기점 | 상태 |
|---|---|---|---|
| `main` | `8547ed8` (07-07) | — | 캡션 eval 머지까지 반영 |
| `dev` | `fe23681` (07-14) | main 앞 | **메인 개발선**, Phase 1 가동 |
| `exp9_teacher_target_mix` | `69a8107` (07-15) | fe23681 | GPU 스모크 대기 |
| `extra_exp/vg_transform_min0.5` | `cbe9d30` (07-06) | 8e6f07c | 완주, 무승부 판정 |
| `claude/caption-minilm-eval` | `dfbe304` (07-07) | — | main에 머지됨 |
| `origin/lm_distill` | `8e6f07c` (07-02) | — | dev에 FF 머지됨 |
| `origin/test_vit_output_dim` | `87c70ab` (05-13) | 초기 | DINOv3 wrapper 작업 |
| `origin/dev_text` | `93b0792` (05-14) | 초기 | BERT 교체 작업 |
| `origin/dev_small_vit` | `95bc146` (06-02) | 초기 | small ViT + CC12M |
| `origin/dev_vg_add` | `fa5723f` (06-03) | 초기 | VG 데이터셋 추가 |

폐기된 브랜치: `extra_exp/itc_distill_only` (07-14 dev로 이관 후 삭제), `claude/caption-metric-eval` (머지 후 삭제).
태그: 없음 — 실험 식별은 config `exp` 태그 + `output/` 런 이름 체계.

## 부록 2 — 실험 런 카탈로그 (output/)

| 런 | 실험 | 핵심 결과 |
|---|---|---|
| `pt_checkpoint_base_amp` | pre-fix baseline | ep4.3 collapse 발생 런 |
| `pt_checkpoint_base_logitscale_nodecay` | exp4 | 해결 확인, r_mean 66.20 (best 66.83) |
| `pt_checkpoint_base_lrup` | exp5-lrup | **retrieval 최고 66.74 (best 67.97)**, scale 110.9 |
| `pt_checkpoint_base_noaug` | exp5 | no-aug 게이트 |
| `pt_checkpoint_base_vg_min0.5` | extra | exp4와 무승부 (66.34) |
| `pt_smallreg_minilm_baseline` | exp7 | 학생 solo — r_mean 63.4, CIDEr 1.076 |
| `pt_smallreg_minilm_lm_distill` | exp8 | **CIDEr 1.103 (+2.5%), SPICE 0.207 (+2.0%)** |
| `pt_smallreg_minilm_itc_distill_tau0.022_droptau` | exp6.1 | ep1 +18 / ep20 −3.9 |
| `pt_smallreg_minilm_itc_distill_tau0.022_keeptau2` | exp6.2 | no-op 대조군 (62.7) |
| `pt_smallreg_minilm_itc_distill_armT_t0.022` | Phase 1 T | **가동 중** |
| `pt_smallreg_minilm_itc_distill_armS_t0.05` | Phase 1 S | **가동 중** |

## 부록 3 — 문서·인프라 자산

- **설계 스펙 + 실행 플랜 각 7건**: `docs/superpowers/{specs,plans}/` — retrieval-val(06-21),
  itc-distill(06-27), online-teacher(06-29), lm-distill(07-02), caption-zeroshot(07-03),
  caption-metric-eval(07-13), teacher-target-mix(07-14). ※ gitignore 로컬 전용.
- **진단 기록 2건**: `critical_bugfix/2026-07-07_teacher_sim_and_tau/`, `critical_bugfix/2026-07-14_kd_grad_balance_and_geometry/` — 측정 스크립트·데이터·재현 명령 포함.
- **모니터링 툴킷**: `claude_skills/monitoring/` — TB collapse 감시 + 테스트, 런별 watch 스크립트,
  ntfy push, cron 기반 3시간 주기 감시.
- **평가 도구**: `eval_pretrain_retrieval{,_sweep}.py`, `eval_official_pretrain_val_loss.py`,
  `train_caption.py --evaluate` (zero-shot), in-loop `CaptionValRunner`.
