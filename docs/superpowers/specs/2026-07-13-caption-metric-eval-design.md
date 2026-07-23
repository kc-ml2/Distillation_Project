# Pretrain 중 Caption 메트릭 기반 Validation 설계

## 배경 / 문제

`pretrain.py`는 현재 학습 루프 안에서 **retrieval 평가만** 주기적으로 돈다
(`data/eval_validation_retrieval.py::RetrievalValRunner` — ITC는 1000 step마다, ITM은
epoch당 2회). Caption 채점 코드(`data/utils.py::coco_caption_eval` / `_score_no_spice`)는
이미 있지만 **학습 루프에 물려있지 않고**, `train_caption.py`의 zero-shot A/B에서만
오프라인으로 쓰인다.

Distillation의 핵심 관심사는 "학생이 티처의 **의미 내용(semantic content)** 을 담는가"인데,
retrieval(ITC/ITM) 지표만으로는 생성 캡션 품질(특히 의미 축)을 추적할 수 없다. 따라서
pretrain 도중 caption 생성 지표(BLEU/METEOR/ROUGE_L/CIDEr/**SPICE**)를 주기적으로 측정해
체크포인트 간 비교 가능한 곡선을 얻는다.

## 목표

1. pretrain 루프 안에서 COCO Karpathy val split 캡션을 생성하고 표준 지표로 채점, TB에 로깅.
2. 기존 채점 인프라(`coco_caption_eval`)와 retrieval runner 패턴을 최대한 재사용.
3. SPICE(의미 축, 비쌈)를 비용 관리하며 포함 — cadence 분리 + CPU 코어 예산 캡.
4. 터미널 로그에 **에폭 시작 후 실경과 시간(validation 포함)** 을 표시.

## 비-목표 (YAGNI)

- **async/백그라운드 채점 오프로드는 하지 않는다.** 인라인(ITM처럼 블로킹). 아래 비용 측정상
  인라인으로 충분하며, 확장이 필요하면 나중에 붙일 수 있게 경계만 깔끔히 둔다.
- val 서브샘플링 안 함 (full 5000 val).
- test split 채점 안 함 (val만).

## 측정 근거 (2026-07-13 실측, COCO val 5000, this server = Xeon Gold 6430, 64C/128T)

| 단계 | 시간 | 자원 |
|---|---|---|
| 생성 (4-GPU 샤딩, 실측) | **13.6s** | GPU×4 |
| PTB tokenize | 0.6s | Java |
| BLEU 1–4 | 0.8s | python |
| METEOR | 9.25s | Java |
| ROUGE_L | 1.0s | python |
| CIDEr | 2.8s | python |
| ↳ **no-SPICE 채점 소계** | **~14.5s** | CPU (GPU 유휴) |
| SPICE (cold, 64C) | 67.3s | Java |
| SPICE (cold, **32C**) | **62.3s** | Java |
| SPICE (cold, 8C) | >120s (timeout) | Java |

핵심 결론:
- SPICE는 채점 비용의 대부분을 차지하는 유일한 무거운 항목이며, **cold-cache가 정상**
  (in-loop은 매 에폭 캡션이 달라져 캐시 미스). 예전 "몇십 분"의 정체 = SPICE, 그리고 그 크기는
  **가용 CPU 코어 수에 강하게 의존**(Stanford 의존구문 파서가 코어에 병렬).
- **SPICE는 32코어에서 이미 포화** — 32C(62s) ≈ 64C(67s). 즉 32코어로 캡해도 속도 손해 0이고,
  나머지 32코어를 **동시 MPS 학습 잡**에 넘길 수 있다.

## Cadence 결정 (baseline 완주 run의 ITM step 실측 기준)

- 1 epoch = **37,346 step** (746,920 / 20 epoch).
- ITM은 epoch당 2회: **#1 = step 18,675 (에폭의 50.006% ≈ 정확히 절반), #2 = epoch-end(37,346)**.
  두 지점은 에폭의 50%·100%로 균등(간격 ~18,673).
- **경량 지표(BLEU/METEOR/ROUGE_L/CIDEr)**: ITM 두 지점 모두에서 → **2×/epoch**.
- **SPICE**: epoch-end에서만 → **1×/epoch** (= 매 2번째 ITM).

per-epoch 비용: mid(경량, ~28s) + end(경량+SPICE, ~90s) ≈ **~118s/epoch**,
20 epoch ≈ **~40분** (한 잡 기준; 다른 MPS 잡과 겹쳐 진행).

## 아키텍처

### 파일 구성

- **신규** `data/eval_validation_caption.py` — `CaptionValRunner` + 로더 빌더
  (`RetrievalValRunner`와 대칭).
- **수정** `models/blip_pretrain.py` — `BLIP_Pretrain.generate()` 추가.
- **수정** `pretrain.py` — caption runner 생성/배선 (retrieval runner와 완전 대칭).
- **수정** `utils.py` — `MetricLogger.log_every`에 에폭 실경과 필드 추가.
- **수정** `configs/pretrain.yaml` — caption val config 키 추가.
- **신규** `tests/` — generate smoke, runner cadence, SPICE 포함 채점 계약.

### 컴포넌트 1 — `BLIP_Pretrain.generate()`

pretrain 모델은 `generate`가 없다(현재 `models/blip.py::BLIP_Decoder.generate`에만 존재).
`BLIP_Pretrain`은 이미 `visual_encoder` + `text_decoder`(LM loss용) + `tokenizer`를
가지고 있으므로, `BLIP_Decoder.generate`를 **미러링**해 동일 로직을 추가한다(가중치 재로딩 없음).

- 시그니처: `generate(self, image, sample=False, num_beams=3, max_length=20, min_length=5, top_p=0.9, repetition_penalty=1.0)`.
- 내부: `visual_encoder(image)` → `image_embeds` → beam search로 `text_decoder.generate`
  (encoder_hidden_states = image_embeds), prompt는 빈 문자열 기준.
- 반환: 캡션 문자열 리스트(길이 = batch).
- 주의: `BLIP_Decoder.generate`의 prompt 슬라이싱/`repetition_penalty`/`sample` 분기를
  그대로 따르되, pretrain 모델의 tokenizer/`text_decoder`를 사용.

### 컴포넌트 2 — `CaptionValRunner`

`RetrievalValRunner` 패턴을 그대로 따른다.

- **로더**: `build_caption_val_loader(config)` — `create_dataset('caption_coco', config)`의
  val 데이터셋을 `DistributedSampler(shuffle=False)`로 4-rank 샤딩, `DataLoader`로 감쌈.
  1회 빌드 후 재사용.
- **트리거 메서드**:
  - `val_caption_during_train(model_without_ddp, epoch, iteration, global_step)`:
    `iteration == config['val_caption_mid_interval_steps']` (= ITM mid step 18675)일 때
    **경량 지표만** 실행.
  - `run_epoch_end(model_without_ddp, epoch, global_step)`:
    `val_caption_epoch_end`가 참이면 실행, `val_caption_use_spice`가 참이면 **SPICE 포함**.
- **실행 로직** (`_run(..., with_spice)`):
  1. `model.eval()` (finally에서 원복 — retrieval `_run_tier` 패턴).
  2. 각 rank가 자기 샤드를 생성 → `[{image_id, caption}, ...]`.
  3. `dist.all_gather_object`로 rank0에 취합, image_id 기준 dedup.
  4. **rank0만**: 예측 JSON을 output_dir에 덤프 → CPU 코어 캡(컴포넌트 5) 적용 →
     `coco_caption_eval(coco_gt_root, results_file, 'val', use_spice=with_spice)` 호출 →
     반환 dict.
  5. rank0가 TB에 `val_caption/CIDEr`, `/METEOR`, `/Bleu_4`, `/ROUGE_L`,
     (with_spice면) `/SPICE` 등을 global_step에 기록.
- **빌더**: `build_pretrain_caption_val_runner(config, device, writer=None)` —
  `config['val_caption_enabled']`가 거짓이면 `None` 반환.

### 컴포넌트 3 — `pretrain.py` 배선

retrieval runner와 완전 대칭:
- `main()`: `caption_val_runner = eval_validation_caption.build_pretrain_caption_val_runner(...)`.
- `train(...)` 시그니처에 `caption_val_runner=None` 추가, 호출부에 전달.
- train 루프 내(retrieval `val_retrieval_during_train` 호출 직후):
  `caption_val_runner.val_caption_during_train(model_without_ddp, epoch, iteration=i, global_step=...)`.
- epoch-end(retrieval `run_epoch_end` 직후):
  `caption_val_runner.run_epoch_end(model_without_ddp, epoch, global_step)`.

### 컴포넌트 4 — 에폭 실경과 로깅 (`utils.py::MetricLogger.log_every`)

`log_every`의 기존 `start_time`(루프 진입 시각)이 곧 **에폭 시작 시각**이고, validation은
`yield obj` 이후 루프 바디에서 돌기 때문에 `time.time() - start_time`이 **validation 포함
실경과**를 자연히 담는다.

- 각 print 시점에 `elapsed_string = str(datetime.timedelta(seconds=int(time.time() - start_time)))`
  계산, `log_msg`에 `'elapsed: {elapsed}'` 필드 추가(두 print 분기 모두).
- ETA(`eta:`)는 기존대로 유지 — 학습 스텝 기반 추정치. 옆에 실경과를 나란히 보여 대비.
- 공유 유틸이라 train/retrieval/caption 생성 로그 전부에 자동 적용.

### 컴포넌트 5 — CPU 코어 예산 캡 (32코어)

MPS로 2개 학습이 동시에 도는 환경. SPICE(및 채점 전체)가 64코어를 다 먹으면 컨텍스트 스위치
폭발 → 채점을 **지정 코어셋에 제한**.

- rank0의 채점 호출 직전 `orig = os.sched_getaffinity(0)` 저장 →
  `os.sched_setaffinity(0, cores)` 로 제한(java 자식이 상속) → 채점 후 finally에서 원복.
- config 키 `caption_score_cpu_list` (예: `'0-31'`) — 문자열을 코어 집합으로 파싱.
  기본값 `null`이면 캡 없음(전체 코어).tt
- **MPS 2-잡 운용 시 잡별 수동 지정 (서로 겹치지 않게):**

  | 잡 | config 파일 | `caption_score_cpu_list` |
  |---|---|---|
  | 잡 A | (예) `configs/pretrain.yaml` | `'0-31'` |
  | 잡 B | (예) `configs/pretrain_jobB.yaml` | `'32-63'` |

  SPICE가 32코어에서 이미 포화(32C 62s ≈ 64C 67s)라 절반씩 나눠도 각자 풀속 + 무경합.
  자동 배정이 아니라 **각 잡의 config에 손으로 다르게** 박는다.
- **단일 잡**일 땐 `'0-31'`/`'0-63'` 아무거나, 또는 `null`(캡 없음). 이 캡은 **채점 버스트
  구간에만** 적용되고 학습 중 데이터로더 코어 사용은 기존대로 전체 공유 — 막으려는 건
  "SPICE가 64코어를 통째로 삼켜 상대 잡을 마비시키는 것"뿐이다.

### config 키 (`configs/pretrain.yaml` 추가)

```yaml
val_caption_enabled: true
val_caption_ann_root: '/home/minwoo/Distillation_Project/datasets/vision/coco/annotations/'
val_caption_image_root: '/home/minwoo/Distillation_Project/datasets/vision/coco/images/'
val_caption_gt_root: '/home/minwoo/Distillation_Project/datasets/vision/coco/annotation/coco_gt'
val_caption_split: 'val'
val_caption_batch_size: 32
val_caption_num_workers: 4
val_caption_mid_interval_steps: 18675   # ITM mid와 동일
val_caption_epoch_end: true
val_caption_use_spice: true             # epoch-end에서만 SPICE
caption_score_cpu_list: '0-31'          # 잡A '0-31' / 잡B '32-63' / null=캡 없음
# 생성 파라미터
caption_num_beams: 3
caption_max_length: 20
caption_min_length: 5
caption_prompt: ''
```

## 테스트 계획 (TDD)

1. **`generate()` smoke** (`tests/test_pretrain_generate.py`): 작은 랜덤/더미 이미지 배치로
   `BLIP_Pretrain.generate`가 batch 길이만큼의 문자열 리스트를 반환. (CPU, 작은 모델 구성)
2. **runner cadence 계약** (`tests/test_caption_val_runner_cadence.py`):
   `val_caption_during_train`이 `iteration == mid_interval_steps`에서만 발화, 그 외엔 no-op;
   `run_epoch_end`가 `val_caption_epoch_end`/`val_caption_use_spice` 플래그를 정확히 반영
   (retrieval runner 테스트 패턴 재사용, 채점은 mock).
3. **SPICE 포함 채점 계약** (기존 `tests/test_caption_score_no_spice.py` 대칭):
   `coco_caption_eval(..., use_spice=True)`가 `SPICE` 키를 포함하고 METEOR/ROUGE_L/CIDEr도
   함께 반환.
4. **코어 캡 파싱**: `caption_score_cpu_list` 문자열('0-31')이 올바른 코어 집합으로 파싱되고,
   캡 컨텍스트가 원래 affinity를 복원.
5. **elapsed 로깅**: `log_every`가 `elapsed:` 필드를 포함해 출력(형식 스모크).

## 리스크 / 유의

- **MPS 하 생성 시간**: 위 13.6s는 GPU 단독. MPS로 2잡 공유 시 생성이 느려질 수 있음(캡션 eval
  고유 문제 아님, MPS 일반 특성). 인라인 블로킹이므로 학습 처짐으로 나타남 — cadence를 2×/epoch로
  제한한 이유.
- **DDP gather 정합**: `all_gather_object`로 모은 예측의 image_id 중복/누락 처리(마지막 배치
  `drop_last=False` + DistributedSampler 패딩) — dedup으로 방어.
- **generate 미러링 정확성**: `BLIP_Decoder.generate`와 토크나이즈/BOS/prompt 규칙이 어긋나면
  캡션이 깨짐. 미러링 시 원본과 1:1 대조.
