# Pretrain 중 COCO Retrieval 기반 Validation 설계

## 배경 / 문제

`pretrain.py`는 현재 `data/eval_validation_loss.py`의 `PretrainValLossRunner`를 통해 주기적으로 COCO Karpathy val split에서 validation loss(`loss_ita`, `loss_itm`, `loss_lm`)를 측정한다.

문제는 `loss_ita`가 `models/blip_pretrain.py`의 `BLIP_Pretrain.forward()` 내부에서 **모멘텀 인코더가 만든 target과 학습 중인 FIFO 큐(`image_queue`/`text_queue`)를 negative pool로 써서** 계산된다는 점이다. 이 큐는 학습이 진행되는 동안 계속 churn되므로, negative pool의 난이도가 모델 품질과 무관하게 시간에 따라 흘러간다. 따라서 epoch N의 `loss_ita`와 epoch M의 `loss_ita`를 직접 비교하기 어렵다 (관련 메모: `project_val_loss_todo.md`).

이를 해결하기 위해, **고정된 held-out negative pool(COCO val set 자체)**을 이용한 retrieval 평가(R@1/R@5/R@10)를 도입한다. 이 평가는 student 인코더만 사용하고 모멘텀 인코더를 전혀 참조하지 않으므로, 체크포인트 간 비교가 가능한 안정적인 지표가 된다.

## 목표

1. 기존 retrieval 평가 코드(`eval_validation_tool.py`, `train_retrieval.py`)를 최대한 재사용한다.
2. 사용자가 프리트레인 체크포인트를 불러와 **단독으로** retrieval 점수를 확인할 수 있는 스크립트를 제공한다.
3. 프리트레인 도중 **주기적으로** retrieval 평가를 돌려 학습 진행 상황을 모니터링할 수 있게 한다.

## 사전 정리 (이미 완료)

- `eval_validation_tool.py`에서 죽은 코드 제거: `evaluate_loss`(→`PretrainValLossRunner`로 대체됨), `run_validation`(→ 존재하지 않는 config 키를 참조하는 미사용 오케스트레이터). `evaluate_caption`, `evaluate_retrieval`, `itm_eval`은 보존.
- `eval_official_pretrain_val_loss.py`의 `dist.barrier()` 무조건 호출 버그 수정 (`if args.distributed:`로 가드) — 분산 환경 없이 단독 실행 시 크래시하던 문제.

## 비용/빈도 트레이드오프

retrieval 평가는 두 단계로 나뉜다:

- **ITC (Image-Text Contrastive) 점수만 사용**: `vision_proj`/`text_proj` 임베딩끼리 코사인 유사도만 계산. 이미지 5000장 + 캡션 25,010개를 forward(backward 없음)만 하면 되므로 가볍다 (추정 10~30초/회, GPU 1장 기준).
- **ITM (Image-Text Matching) rerank**: ITC top-k(`k_test`, 기본 128)개 후보에 대해서만 cross-attention 기반 ITM head로 재채점. 이게 원래 BLIP 논문이 보고하는 R@1/5/10의 정의이지만, cross-attention forward를 이미지당/캡션당 따로 돌려야 해서 훨씬 비싸다 (추정 2~5분/회, 클러스터 전체 기준).

학습 규모: COCO(566,747) + VG(5,408,689) ≈ 597만 샘플, `batch_size=40 × 4GPU=160` 기준 1 epoch ≈ 37,300 step. 실측 1 epoch ≈ 4시간.

결정된 빈도:
- **ITC-only**: `1000 step`마다 (epoch당 ~37번, 추세를 촘촘하게 봄)
- **ITM rerank**: `10000 step`마다 (1시간에 한 번 꼴, epoch당 ~3~4번) — epoch 단위로만 도는 것보다 훨씬 자주 신호를 얻을 수 있음 (전체 학습에서 epoch-end만 쓰면 20번 안팎으로만 찍히는데, 그건 너무 뜸함)

두 티어 모두 step interval과 epoch-end 트리거를 둘 다 가진다 (epoch-end는 체크포인트 저장 시점에 항상 점수가 붙도록 보장하는 역할).

## 아키텍처

### 파일 구성

| 역할 | 파일 | 비고 |
|---|---|---|
| 핵심 계산 ("뭘 계산하나") | `eval_validation_tool.py` (기존, 루트) | `evaluate_retrieval`을 ITC-only / ITM-rerank 두 함수로 분리 |
| 오케스트레이션 ("언제 도나") | `data/eval_validation_retrieval.py` (신규) | `data/eval_validation_loss.py`의 `PretrainValLossRunner` 패턴을 그대로 따라감 |
| pretrain 연결 | `pretrain.py` (수정) | `val_loss_runner`와 나란히 `retrieval_val_runner` 추가 |
| 단독 체크포인트 스크립트 | `eval_pretrain_retrieval.py` (신규, 루트) | `eval_official_pretrain_val_loss.py` 패턴을 그대로 따라감 |
| 공용 헬퍼 | `utils.py` (수정) | `load_model_weights_only`를 여기로 옮겨 두 스크립트에서 재사용 |

### B. 핵심 계산 함수 리팩토링 (`eval_validation_tool.py`)

`evaluate_retrieval`의 "이미지/텍스트 임베딩 추출 + 코사인 유사도 행렬 계산" 부분과 "ITM rerank" 부분을 분리한다.

- `evaluate_retrieval_itc(model, data_loader, device, config)`:
  - 이미지/텍스트 임베딩 추출, `sims_matrix` 계산까지만 수행
  - image patch feature(`image_feats`, GPU→CPU 카피되는 큰 텐서)는 ITM rerank에만 필요하므로 아예 들고 있지 않음 (메모리 절감)
  - 모든 rank가 동일한 전체 sims_matrix를 독립적으로 계산하므로 `dist.all_reduce` 불필요
  - `sims_matrix`, `sims_matrix.t()`를 그대로 score matrix로 반환
- `evaluate_retrieval_itm(model, data_loader, device, config)`:
  - 기존 `evaluate_retrieval` 로직 그대로 (ITC top-k 추출 → ITM head rerank → `dist.all_reduce`로 멀티 GPU 결과 합산)
- `itm_eval(scores_i2t, scores_t2i, txt2img, img2txt)`: 변경 없음. 두 함수 모두 이 함수에 결과를 넘겨 R@1/5/10을 계산.

### C. Runner + config 키 (`data/eval_validation_retrieval.py`, `configs/pretrain.yaml`)

`coco_karpathy_retrieval_eval`(`data/coco_karpathy_dataset.py`, 기존)을 데이터셋으로 재사용. `train_retrieval.py`의 val/test loader와 동일하게 **DistributedSampler 없이** 전체 데이터셋을 모든 rank가 로드하도록 구성 (`evaluate_retrieval_itm` 내부에서 rank별로 일부만 ITM rerank하도록 직접 샤딩하기 때문).

이미지 transform은 `data/eval_validation_loss.py`의 `build_val_loss_transform(image_size)`를 그대로 재사용 (resize+totensor+normalize, augmentation 없음 — retrieval val/test에 동일하게 적용 가능).

`RetrievalValRunner` 클래스 (`PretrainValLossRunner`와 동일한 패턴):
- `__init__(self, data_loader, device, config, writer=None)`
- `val_retrieval_during_train(model_without_ddp, epoch, iteration, global_step, ...)`: 내부에서 ITC interval과 ITM interval을 각각 체크해서 독립적으로 트리거
- `run_epoch_end(model_without_ddp, epoch, global_step, ...)`: `val_retrieval_itc_epoch_end`/`val_retrieval_itm_epoch_end` 플래그를 보고 실행
- eval 진입 시 `model_without_ddp.eval()` → 평가 끝나면 원래 모드로 복구 (was_training 패턴, `PretrainValLossRunner.evaluate`와 동일)
- 결과는 `writer.add_scalar("val_retrieval_itc/...", ...)` / `writer.add_scalar("val_retrieval_itm/...", ...)`로 네임스페이스를 나눠 기록
- 팩토리 함수 `build_pretrain_retrieval_val_runner(config, device, writer=None)`: `config.get("val_retrieval_enabled", False)`가 False면 `None` 반환

`configs/pretrain.yaml`에 추가될 키:
```yaml
val_retrieval_enabled: true
val_retrieval_ann_root: '/home/minwoo/Distillation_Project/datasets/vision/coco/annotations/'
val_retrieval_image_root: '/home/minwoo/Distillation_Project/datasets/vision/coco/images/'
val_retrieval_split: 'val'
val_retrieval_batch_size: 64
val_retrieval_num_workers: 4
k_test: 128

val_retrieval_itc_interval_steps: 1000
val_retrieval_itc_epoch_end: true

val_retrieval_itm_interval_steps: 10000
val_retrieval_itm_epoch_end: true
```

`val_retrieval_ann_root` + `val_retrieval_split`은 `coco_karpathy_retrieval_eval(transform, image_root, ann_root, split)`(`data/coco_karpathy_dataset.py`, 기존)의 생성자 시그니처를 그대로 따른 것 — 클래스를 한 글자도 안 건드리고 재사용하기 위함.

### D. pretrain.py 연결

- `main()`에서 `val_loss_runner`를 만드는 자리 옆에 `retrieval_val_runner = eval_validation_retrieval.build_pretrain_retrieval_val_runner(config=config, device=device, writer=writer)` 추가.
- `train()` 함수가 `model_without_ddp`를 새 인자로 받도록 수정 (retrieval 계산은 `model.tokenizer` 등 raw 속성 접근이 필요해서 DDP wrapper가 아니라 원본 모델이 필요함 — `val_loss_runner`는 `model(...)` forward만 호출하므로 DDP wrapped 모델도 그대로 써도 무방했던 것과 다름).
- `train()` 루프 안, `val_loss_runner.val_loss_during_train(...)` 호출 옆에 `retrieval_val_runner.val_retrieval_during_train(...)` 호출 추가.
- `main()`의 epoch 루프, `val_loss_runner.run_epoch_end(...)` 호출 옆에 `retrieval_val_runner.run_epoch_end(...)` 호출 추가.
- 멀티 GPU 동작은 변경 없음 — `evaluate_retrieval_itm` 내부에 이미 있는 `utils.get_world_size()`/`get_rank()`/`dist.all_reduce` 로직이 그대로 적용됨. DDP wrapper를 안 쓰는 것은 gradient 동기화가 필요 없는 `@torch.no_grad()` 평가이기 때문이며, 멀티 GPU 샤딩 자체는 그대로 동작한다.

### E. 단독 체크포인트 스크립트 (`eval_pretrain_retrieval.py`)

`eval_official_pretrain_val_loss.py`와 동일 패턴:
- `--config`, `--checkpoint`(필수), `--split val|test`(기본 val), `--output_dir`, `--tensorboard` 인자
- `utils.load_model_weights_only`(아래 F 참조)로 체크포인트 로드
- `RetrievalValRunner`를 만들어 ITC + ITM 평가를 모두 1회 실행, 결과를 json으로 저장하고 출력
- `dist.barrier()`는 처음부터 `if args.distributed:`로 가드

### F. 부수 개선

`eval_official_pretrain_val_loss.py`에 있던 `load_model_weights_only(model, checkpoint_path)` 함수를 `utils.py`로 옮긴다 (체크포인트 종류 무관한 범용 로직이라 두 스크립트에서 재사용 가능해짐). `eval_official_pretrain_val_loss.py`는 이 함수를 `utils`에서 import하도록 수정.

## 테스트/검증 계획

- `python -m py_compile`로 신규/수정 파일 문법 검증
- 단독 스크립트(`eval_pretrain_retrieval.py`)를 기존 체크포인트(`output/pt_checkpoint_base_amp/checkpoint_*.pth` 등) 하나로 실제 실행해서 R@1/5/10 출력 및 json 저장 확인 (단, 학습 중인 GPU와 충돌하지 않는 시점에 진행)
- `pretrain.py`는 코드 리뷰 + 짧은 dry-run(소수 step)으로 `val_retrieval_itc_interval_steps`/`val_retrieval_itm_interval_steps` 트리거가 정상 동작하는지 확인 (전체 학습 재시작은 하지 않음)
