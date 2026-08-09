# ITM 매트릭스 KD — Phase 2: fixed 2-gather 서브샘플

**작성:** 2026-08-08
**상태:** 설계(브레인스토밍 산출물 — 2026-08-07 스펙 §4.2-4.3 확정분) → writing-plans 대기
**선행:** `2026-08-07-itm-bxb-matrix-kd-design.md` §4.2-4.3, Phase 1 완료 + checkpointing(`2026-08-08-itm-matrix-oom-gradient-checkpoint-design.md`)
**베이스:** 브랜치 `itm_bxb_matrix_kd` 위 계속.

---

## 1. 동기 — 실측이 full B×B를 기각

Phase 1(full B×B) + checkpointing 실측(1×4090, bs=40, bidir): OOM은 해소됐지만 **3.18 s/step = baseline(0.21s) 대비 ~15×**. 20에폭 실런 = 4-GPU ~27일(baseline ~1.9일) → **비현실적**. 원인: 학생 1600 fwd + 1600 recompute + 티처 1600 fwd. 서브샘플로 셀 수를 줄인다.

## 2. 무엇 — fixed 2-gather (union 아님)

`topk=k`일 때 방향별로 **정확히 `k+1`개**(positive + 하드네거 k)만 계산. **고정 shape → 메모리 상수·출렁임 0**(union-dedup의 동적 `|U|` 회피, 유저 결정).

- **선택(학생):** `weights_i2t`/`weights_t2i`(forward가 3B ITM CE용으로 이미 계산, diagonal-zeroed, `blip_pretrain.py:448-451`)에서 `.topk(k)` → 순수 네거 k개. positive(대각선) **명시 prepend** → `idx_i2t_full`/`idx_t2i_full` `[B, k+1]`, positive는 항상 index 0.
- **불변식:** 각 방향 분포 = positive + 네거 k = **k+1 고정**(topk와 독립적으로 positive 보장).
- **티처:** 학생이 뽑은 **동일 인덱스** 재사용(같은 지지집합 비교).
- **양방향(bidir):** i2t(이미지별 하드텍스트)·t2i(텍스트별 하드이미지) **각각 gather**(col 지지집합 함정 회피 — 2026-08-07 §4.2). 총 `2·B·(k+1)`. `direction=forward`면 i2t만, `reverse`면 t2i만(1 gather).
- **checkpoint 불필요:** `2·B·(k+1)` ≤ ~720(k=8)이라 24GB에 그냥 맞음. Phase 1의 `use_checkpoint`는 full 경로 전용으로 남김.

## 3. 구현 — 재사용 최대, 신규 최소

### 3.1 신규: 일반 pair-scorer `itm_pair_logits` (`distillation/itm_matrix.py`)
한 함수로 양방향 다 처리(ponytail: 방향별 헬퍼 X). `image_idx`, `text_idx` `[B, M]`가 주는 B·M 쌍을 **한 번의 배치 forward**로 채점:

```
def itm_pair_logits(text_encoder, itm_head, image_embeds, image_atts,
                    enc_ids, text_atts, image_idx, text_idx):
    B, M = image_idx.shape
    fi, ft = image_idx.reshape(-1), text_idx.reshape(-1)          # [B*M]
    out = text_encoder(enc_ids[ft], attention_mask=text_atts[ft],
                       encoder_hidden_states=image_embeds[fi],
                       encoder_attention_mask=image_atts[fi], return_dict=True)
    with torch.autocast(device_type=image_embeds.device.type, enabled=False):
        logits = itm_head(out.last_hidden_state[:, 0, :].float())  # [B*M, 2]
    return logits.reshape(B, M, 2)
```
- i2t: `image_idx = arange(B)[:,None].expand(B,k+1)`, `text_idx = idx_i2t_full`.
- t2i: `image_idx = idx_t2i_full`, `text_idx = arange(B)[:,None].expand(B,k+1)`.
- 루프 없음(단일 forward), fp32 head 가드 동일. 학생·티처 공용.

### 3.2 신규: 손실 `itm_gathered_kd_loss` (`distillation/losses.py`)
방향별 `[B,k+1,2]`를 **dim=1(k+1 후보) softmax** → forward KL. Phase 1 `itm_matrix_kd_loss`의 채널 로직 미러(transpose 없음):

```
def itm_gathered_kd_loss(s_i2t, t_i2t, s_t2i, t_t2i, direction, temp):
    # 각 [B,k+1,2]. channels = {forward:[1],reverse:[0],bidir:[0,1]}.
    # dir별·channel별 softmax(dim=1) forward KL(T‖S) 평균 × temp. 티처 detach.
```
(direction=forward/reverse면 해당 방향 gather 하나만.)

### 3.3 티처: `OnlineTeacher.itm_matrix_gathered(image, caption, idx_i2t, idx_t2i)`
학생 인덱스로 `itm_pair_logits` 2회 → `(t_i2t, t_t2i)`. 기존 `itm_matrix`(full)와 나란히.

### 3.4 배선 — **핵심 결정: 인덱스 왕복**
Phase 2 티처는 **학생 forward 안에서 뽑힌 인덱스**가 필요하다(full엔 없던 의존성). 두 안:

- **(A) 티처를 forward에 주입(추천).** `forward(..., online_teacher=None, itm_topk=-1)`. `itm_topk>0`이면 forward가 선택→학생 gather→`online_teacher.itm_matrix_gathered(idx)`→`itm_gathered_kd_loss`→`loss_itm_kd`. **6-튜플 contract 불변, 전부 한 곳.** Phase 1의 티처 precompute(pretrain)도 forward로 통일(≈2줄 이동). 티처는 frozen/no_grad라 DDP 무영향.
- **(B) 인덱스 반환.** forward가 학생 gather+인덱스 반환, pretrain이 티처+손실. 반환 contract 변경·이질적 튜플.

→ **(A-통일) 확정**(유저). 티처 계산을 **full·gathered 모두 forward 안으로** 일원화:
- forward 시그니처에서 `teacher_itm_logits` **제거**, 대신 `online_teacher=None, itm_topk=-1` 추가.
- pretrain은 티처 precompute를 **없애고** `online_teacher` 객체 + `itm_topk`(+ temp/direction)만 forward에 전달.
- forward 안에서: `itm_topk<=0`이면 `teacher = online_teacher.itm_matrix(image,caption)`+`itm_bxb_logits`(checkpoint)+`itm_matrix_kd_loss`; `itm_topk>0`이면 선택→`itm_pair_logits`×2(학생)+`online_teacher.itm_matrix_gathered(idx)`+`itm_gathered_kd_loss`.
- 티처는 frozen/no_grad·비-DDP라 학생 DDP forward 안에서 호출해도 무영향(params 미등록). autocast 중첩 동dtype 무해.
- 근거: full이 15×라 실런엔 안 쓰이므로 Phase 1 배선 보존 가치 낮음 → 통일이 더 깔끔.

### 3.5 Phase 1(full) 경로
`topk=-1`이면 `itm_bxb_logits`+`itm_matrix_kd_loss`(checkpoint) — 로직은 그대로, **티처 소스만** precompute→forward-내 호출로 이동(§3.4). 레퍼런스로 유지.

## 4. config
```yaml
distill:
  itm:
    enabled: true
    weight: 1.0
    temp: 0.05
    direction: bidir       # {forward, reverse, bidir}
    topk: 8                # -1=full B×B(Phase1); k<B-1=fixed 2-gather(Phase2). 스윕 {4,8,12}
```

## 5. 불변식 보존
- fp32 head(pair-scorer 내 동일 가드), teacher no_grad, baseline OFF 불변(topk 무관, enabled=false면 미진입).
- k+1 고정 shape → 메모리 상수. positive 항상 포함(index 0).

## 6. 성공 기준 + 측정
1. **OOM 없음**(고정 `2·B·(k+1)`, checkpoint 불필요) — 실측.
2. **step-time 실측**: ON(topk=k) vs OFF(baseline), 배수. 목표: full의 15×를 **한 자릿수**로. k∈{4,8} 실측.
3. itm retrieval ≥ baseline + itm−itc>0 회복(학습 후, 별도).
4. 손실 유한·CE 안 삼킴.

## 7. 범위 밖 (ponytail)
- **union-dedup**: 동적 `|U|` 출렁임 회피로 기각(fixed가 목적). softmax 측정 실패 전엔 안 함.
- **gradient checkpointing on 서브샘플**: 720 seq는 그냥 맞아서 불필요.
- Phase 1 full 경로 제거: 레퍼런스로 유지(제거가 더 큰 diff).
- entmax/sparsemax, eval 랭킹 변경 — 이전 스펙과 동일 non-goal.

## 8. 리스크 / 확정
- **배선 = (A-통일) 확정**(§3.4): 티처를 forward 안에서 계산, `teacher_itm_logits`→`online_teacher`. Phase 1 forward-kd 테스트도 이에 맞춰 갱신(티처 객체 전달).
- **k 기본 = 8 확정**(스윕 4/8/12). 신호(네거 수) vs 속도.
- **advanced-indexing 복사**: `image_embeds[fi]`가 이미지 임베딩 B·M 복사(~수백MB) — 유계·무해.
- **실측 배수가 여전히 크면**: k↓ 또는 direction 단측(i2t만).
- **DDP-내 티처 호출**: frozen·no_grad라 무해 예상, 스모크로 확인.
