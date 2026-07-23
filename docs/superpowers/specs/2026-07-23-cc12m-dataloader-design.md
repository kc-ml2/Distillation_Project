# cc12m 데이터로더 일원화 + 일관성 — 설계

관련 코드: `pretrain.py`(dataset 섹션), `data/__init__.py`, `data/pretrain_cc12m_webdataset.py`, `data/combined_loader.py`
검증 산출물: scratchpad `l1_smoke.py`, `l2_smoke.yaml` (본 문서 §1 근거)

## 1. 배경

cc12m을 coco+vg 사전학습에 섞기 위한 데이터로더 정리 작업. 착수 전 "정합성 버그부터(A)" 잡으려 했으나, **L1(데이터로더 단독)·L2(end-to-end) 스모크로 A 표면이 비어 있음을 확정**했다.

데이터 규모(ratio 근거):
- coco(karpathy_train) 566,747 + vg_train 5,408,689 = **baseline 5,975,436**
- cc12m 실제 tar 성공분 **7,072,146** (1,243 샤드, 다운 성공률 56.9%)
- 자연 비율 cc12m:coco_vg ≈ 1.18 → **ratio=1 확정** (거의 자연 비율)

검증된 사실(실행 근거, 환경 = conda `kd_v4`, torch 2.6.0, webdataset 0.2.111):

| 층 | 검증 | 결과 |
|---|---|---|
| L1 | cc12m 배치 포맷 | base와 동일: image `FloatTensor[B,3,224,224]` 정규화, caption `list[str]` len B |
| L1 | worker split 중복 (4워커·4샤드) | `__key__` 22,659개 **중복 0** (`workersplitter=split_by_worker` 기본 ON) |
| L1 | CombinedLoader | `len = len(base)×(1+ratio)`, ratio 인터리브 패턴 정상 |
| L2 | 3-way distill train 스텝 통과 | step 0~50 **Traceback 0**, loss 전부 유한·안정 |
| L2 | 에폭 길이 | cc12m ON → `len(data_loader)=70842=coco(35421)×2` (ratio=1 정확히 ×2) |

**결론: cc12m 데이터로더는 기능적으로 정확하다.** cc12m 배치는 base와 포맷이 동일해 train 스텝이 둘을 구분조차 못 하며, 가장 무거운 현재 경로(teacher forward + `itc_target_mix(queue)` + `itm_target_mix(teacher-neg)` + `lm`)도 예외 없이 통과한다. 고칠 정합성 버그는 없다.

따라서 이 작업의 실질은 **구조 일원화 + 일관성 정리**다. 동작(학습 결과)은 불변으로 유지한다.

## 2. 검토한 접근과 선택 근거

- **최소 안전 정리** — 동작 안 바꾸는 것만(`shardshuffle` int, 죽은 주석 제거, 문서화). 코드 이동 없음.
- **일원화 + 일관성 (채택)** — 위 + cc12m 로더 생성을 단일 팩토리로 모으고, cc12m transform을 base와 동일하게 `pretrain_train_aug` 토글에 종속시킨다. `pretrain.py`에 흩어진 배선을 한 곳으로 모아 이후 실험에서 손대기 쉽게 한다.
- **결정성까지** — 위 + webdataset 셔플에 seed/detshuffle 도입. 재현성 이득은 크나 셔플 동작이 바뀌어 검증 부담↑. **이번 스코프 제외.**

채택: **일원화 + 일관성**. 학습 의미(에폭 길이·소비 비율·셔플 동작)는 보존하면서 구조만 정리하므로 baseline-우선 진행에 영향이 없다.

## 3. 설계

동작 불변 원칙: `cc12m_tar_path: ''`(baseline)에서는 결과가 기존과 **바이트 동일 경로**여야 한다.

### 3.1 `data/pretrain_loader.py` — 신규 팩토리 (일원화)

`pretrain.py`의 dataset 섹션(현재 base 로더 + cc12m WebLoader + `with_epoch` + `CombinedLoader` 조립)을 한 함수로 캡슐화한다.

```python
# data/pretrain_loader.py (신규)
import webdataset as wds
from data import create_dataset, create_sampler, create_loader
from data.combined_loader import CombinedLoader

def build_pretrain_dataloader(config, num_tasks, global_rank):
    """coco+vg base 로더를 만들고, cc12m_tar_path가 있으면 CombinedLoader로 결합해 반환.

    반환 계약(기존 pretrain.py가 의존): 반환 객체는 .sampler(set_epoch 가능)와
    len()을 제공한다. cc12m OFF면 base DataLoader, ON이면 CombinedLoader.

    주의(문서 note, #7): itc_target_mix variant=queue 등 concat_all_gather 경로는
    torch.distributed 초기화가 필요하다 → 반드시 torchrun으로 실행. (assert는 두지 않음)

    주의(에폭 길이 커플링): cc12m ON이면 len(data_loader)가 (1+ratio)배가 된다.
    warmup_steps 등 '절대 step' 하이퍼파라미터는 에폭 길이 변화에 맞춰 재튜닝 필요.
    """
    base_dataset = create_dataset('pretrain', config, min_scale=0.2)
    print('number of training samples: %d' % len(base_dataset))
    sampler = create_sampler([base_dataset], [True], num_tasks, global_rank)[0]
    base_loader = create_loader(
        [base_dataset], [sampler], batch_size=[config['batch_size']],
        num_workers=[4], is_trains=[True], collate_fns=[None])[0]

    if not config['cc12m_tar_path']:
        return base_loader

    cc12m_dataset = create_dataset('pretrain_cc12m_webdataset', config, min_scale=0.2)
    cc12m_loader = wds.WebLoader(cc12m_dataset, batch_size=None, num_workers=4, pin_memory=True)
    ratio = config['cc12m_ratio']
    # #6: with_epoch 일관 적용 (단일 GPU·DDP 공통).
    #     DDP에서 split_by_node로 rank별 샤드 수가 달라도 rank당 배치수를 고정해 collective 동기.
    #     단일 GPU에선 관측 동작 동일(CombinedLoader가 base 길이에 맞춰 이미 캡)하나 분기를 없애 일관화.
    cc12m_loader = cc12m_loader.with_epoch(len(base_loader) * ratio)
    return CombinedLoader(loader_map=base_loader, loader_iterable=cc12m_loader, ratio=ratio)
```

`pretrain.py`는 현재 dataset 섹션(base_datasets/print/samplers/base_loader/cc12m 블록 ~35줄)을 다음으로 대체:
```python
num_tasks = utils.get_world_size()
global_rank = utils.get_rank()
data_loader = build_pretrain_dataloader(config, num_tasks, global_rank)
```
(`data_loader.dataset.reload_laion`/`data_loader.sampler.set_epoch` 호출 계약은 유지 — laion은 전 config에서 `''`라 미사용, sampler는 CombinedLoader.sampler로 노출됨.)

### 3.2 `data/__init__.py` — cc12m transform 일관성 (#2)

`create_dataset`의 cc12m 분기가 하드코딩 `transform_train` 대신 base와 동일한 토글을 따르게 한다.

```python
elif dataset == 'pretrain_cc12m_webdataset':
    use_train_aug = config.get('pretrain_train_aug', True)   # base와 동일 규칙
    cc12m_transform = transform_train if use_train_aug else transform_test
    dataset = cc12m_webdataset(
        tar_root=config['cc12m_tar_path'],
        transform=cc12m_transform,
        batch_size=config['batch_size'])
    return dataset
```
근거: online teacher distillation은 aug ON을 요구하고, base('pretrain')는 이미 `pretrain_train_aug`로 aug/no-aug를 고른다. cc12m만 항상 aug였던 불일치를 제거.

### 3.3 `data/pretrain_cc12m_webdataset.py` — shardshuffle 명시적 int (#3)

`shardshuffle=True`는 wds 0.2.111에서 경고("set to a positive integer") 후 조용히 100으로 대체된다. 명시적 파라미터로 의도를 드러낸다.
```python
def __init__(self, tar_root, transform, batch_size, shardshuffle_size=100):
    ...
    wds.WebDataset(self.tar_files, shardshuffle=shardshuffle_size,
                   nodesplitter=wds.split_by_node, handler=wds.warn_and_continue)
```
값 100은 기존 암묵 동작과 동일 → 셔플 강도 불변.

### 3.4 `data/__init__.py` — 죽은 코드 제거 (#4)

`create_loader` 내 주석 처리된 옛 cc12m 분기(`elif isinstance(dataset, cc12m_webdataset)` … `ddp_equalize` … `else: raise`) 블록을 삭제. 실제 cc12m 배선은 §3.1 팩토리로 이관되어 이 주석은 오해만 유발.

### 3.5 결과

- cc12m 데이터로더 배선이 `data/pretrain_loader.py` 한 곳에 모임 → 이후 실험에서 진입점 명확.
- cc12m/base transform 규칙 일치.
- `shardshuffle`/`with_epoch` 분기 제거로 경고·특수분기 소거.
- baseline 경로(`cc12m_tar_path: ''`)는 base_loader만 반환 → 동작 불변.

## 4. 인터페이스 / config (변경 없음)

기존 config 키 그대로 사용: `cc12m_tar_path`, `cc12m_ratio`, `pretrain_train_aug`, `batch_size`. 신규 키 없음. `shardshuffle_size`는 코드 기본값(100)으로 두고 config 노출은 하지 않는다(YAGNI).

문서 note로만 남기는 운영 주의(§3.1 docstring에도 명시):
- **queue distill은 torchrun 필수** (`concat_all_gather` → 분산 초기화). — #7, assert 없음.
- **에폭 길이 커플링**: cc12m ON → `len(data_loader)` ×(1+ratio) → `warmup_steps` 등 절대 step 재튜닝.

## 5. 테스트 계획

- **L1 회귀** — `l1_smoke.py`를 `build_pretrain_dataloader` 기반으로 옮겨: cc12m 배치 포맷 parity, worker 중복 0, `len==len(base)*(1+ratio)`, ratio 인터리브 재확인.
- **baseline 불변** — `cc12m_tar_path: ''`로 팩토리 호출 시 `CombinedLoader`가 아니라 base `DataLoader`가 반환되는지(타입·len) 단위 확인.
- **transform 토글** — `pretrain_train_aug: false`면 cc12m가 `transform_test`(결정적)를, `true`면 `transform_train`을 쓰는지 확인.
- **L2 스모크** — `l2_smoke.yaml` 유지, 리팩터 후 torchrun 1proc·cc12m ON·step 50까지 Traceback 0 재확인(회귀 게이트).

## 6. 범위 밖 (YAGNI / 별개 이슈)

- **결정성(seed/detshuffle)** — 재현성 이득은 있으나 셔플 동작 변경. 접근 ② 밖.
- **`transform/randaugment.py` uint8 오버플로** — `autocontrast_func`에서 `offset = -low*scale`의 `low`가 np.uint8이라 음수화가 래핑(5→251) → AutoContrast가 채널을 백색(255)으로 포화. coco/vg/cc12m 공통 사전버그로 cc12m 데이터로더와 무관. 한 줄 수정(`-int(low)*scale`)이나 **별개 이슈**로 분리.
- **ratio 자동화/샘플단위 가중** — 1:1 확정으로 불필요.
- **queue distill assert** — #7은 문서 note로만. 코드 assert는 넣지 않음.
