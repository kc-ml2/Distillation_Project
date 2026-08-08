# ITM 매트릭스 KD — OOM 대응: gradient checkpointing (Phase 1.5)

**작성:** 2026-08-08
**상태:** 설계 확정 (대화 브레인스토밍 산출물) → 구현
**선행:** `2026-08-07-itm-bxb-matrix-kd-design.md`(§10 OOM 리스크), Phase 1 구현 완료(브랜치 `itm_bxb_matrix_kd`), GPU 스모크(아래 §1)
**베이스:** 브랜치 `itm_bxb_matrix_kd` 위에서 계속.

---

## 1. 계기 — GPU 스모크가 OOM을 실측

Phase 1(full B×B) 완성 후 1×RTX4090(24GB)에서 스모크(`torchrun` 단일 프로세스, itm-kd ON, bidir, bs=40):

```
torch.OutOfMemoryError: CUDA out of memory
  blip_pretrain.py:530 (학생 forward) → itm_matrix.py:18 (itm_bxb_logits) → med.py:218 (멀티모달 인코더)
  22.23 GiB allocated, free 4.75 MiB. 스텝 0 backward 도달 전 forward에서 터짐.
```

- **원인:** `itm_bxb_logits`가 row별 `text_encoder`(12-layer cross-attn BERT)를 40번 돌리는데, backward를 위해 **40개 row의 activation을 전부 물고 있어** 22GB 소진. full B×B(1600 시퀀스, grad-retain forward)가 24GB에 안 맞는다.
- **DDP 무관:** per-GPU 배치가 40이라 장수 늘려도 GPU당 메모리 동일 → 안 풀림.
- 이건 스펙 §10이 예고한 **Phase 2 트리거**. 단, 스펙 §10이 지목한 **1번 레버가 gradient checkpointing**이다.
- 부수: 1-GPU라도 `torchrun` 필요(`concat_all_gather`@`blip_pretrain.py:587`가 분산 초기화 전제 — 기존 ITC 큐 경로, itm-kd 무관).

## 2. 결정 — 2단계

1. **(이 스펙) gradient checkpointing**으로 full B×B를 bs=40에 맞춘다 → 실행 가능케 하고 **ON/OFF step-time 실측**.
2. **(다음, 별도) Phase 2 fixed-alloc 서브샘플** — `B·2(k+1)` 고정 할당 2-gather(스펙 §4.2-4.3). 메모리 상수·출렁임 0.

**왜 checkpointing 먼저:** full-B×B 신호(전 네거)를 유지한 채 (a) 실행 가능성과 (b) **실제 오버헤드 배수**를 측정 → Phase 1(full)을 그대로 쓸지 vs Phase 2(서브샘플)로 갈지를 **수치로** 판단. 측정 없이 Phase 2로 직행하면 full-B×B의 실비용을 영영 모른다.

## 3. gradient checkpointing이 하는 일

역전파는 forward의 중간 activation을 backward까지 붙들어야 gradient를 구한다(지금 OOM의 정체). Checkpointing은 체크포인트 구간의 중간 activation을 **forward에서 버리고**(입력·출력만 남김), backward에서 필요할 때 **그 구간 forward를 재실행**해 activation을 재생성한다.

- **트레이드오프:** 메모리 ↓(한 번에 1 row activation) ↔ compute ↑(체크포인트 구간 forward 2회).
- **우리 적용:** row별 `text_encoder` 호출을 `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`로 감싼다 → forward peak을 **40 row → 1 row**로. 대가는 ITM-매트릭스 파트 compute **~+30~50%**(인코더가 그 파트의 대부분).

## 4. 변경 설계 (최소 diff)

### 4.1 `distillation/itm_matrix.py`
- 시그니처에 `use_checkpoint=False` 추가.
- row별 인코더 호출을 헬퍼 클로저 `run(enc_hidden, enc_att)`로 빼고: `if use_checkpoint and torch.is_grad_enabled(): cls = checkpoint(run, enc_hidden, enc_att, use_reentrant=False) else: cls = run(...)`.
- **`torch.is_grad_enabled()` 가드가 핵심:** 티처 `itm_matrix`는 `@torch.no_grad()`라 자동으로 plain 경로(checkpoint는 backward 없으면 무의미+경고). 기존 테스트도 인자 미전달→`False`→무변·무경고.
- `itm_head` fp32 가드(autocast 밖, 루프 밖)는 **그대로**. checkpoint는 인코더만 감싼다.
- `.repeat` 유지(입력 복사 ~수백MB~1.4GB 감내). checkpoint만으로 안 맞으면 `.expand`(뷰) 후속(스펙 Task2 deferred minor).

### 4.2 `models/blip_pretrain.py::forward`
- 학생 매트릭스 호출에 `use_checkpoint=True` 한 줄 추가. (config 토글 없음 — 학생 full-B×B는 항상 checkpoint. Phase 2로 가면 경로 자체가 바뀌므로 토글 불필요.)

## 5. 불변식 보존 (Phase 1 검증분 그대로)
- **fp32 head:** 변경 없음(checkpoint 밖). autocast 상태는 `use_reentrant=False`가 recompute에서 보존.
- **row=image/col=text:** checkpoint는 투명(출력 동일). 조립 테스트 그대로 통과해야.
- **baseline OFF 불변:** 학생 경로에만, grad-enabled일 때만. 티처·eval·비증류 무영향.
- **teacher no_grad:** `is_grad_enabled()=False`로 자동 plain.

## 6. 측정 (구현 후)
- **ON**(itm-kd + checkpoint) vs **OFF**(baseline, `distill.itm.enabled=false`) 50-step 간격 step-time, 1×4090.
- 보고: 스텝당/50스텝당 시간, **OFF 대비 배수**. 이 수치가 Phase 1(full+checkpoint) 채택 vs Phase 2(서브샘플) 결정의 근거.

## 7. 범위 밖
- Phase 2 fixed-alloc 서브샘플(다음 스펙).
- config 토글(`grad_checkpoint` 플래그) — 지금 안 함, 학생 항상 on.
- `.expand` 치환 — checkpoint로 부족할 때만.

## 8. 리스크
- **checkpoint + 입력복사(.repeat):** 40개 row의 `enc_hidden` 입력이 retain됨(~수백MB~1.4GB). 여전히 빡세면 `.expand`.
- **재계산 비용:** step-time이 크게 뛰면 그 자체가 Phase 2로 가라는 신호(측정의 목적).
- **checkpoint 클로저:** `run`이 `encoder_input_ids/text_atts/text_encoder`를 클로저 캡처 — `use_reentrant=False`는 지원. 스모크로 grad 흐름 확인.
