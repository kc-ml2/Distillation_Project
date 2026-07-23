# ITC Distillation — Online Teacher Pivot (Design)

- **Date:** 2026-06-29
- **Branch:** `itc_distill` (modify forward — do NOT redo from scratch)
- **Status:** draft, pending review
- **Supersedes:** the teacher-delivery half of `2026-06-27-itc-distillation-design.md`. The student-side design there (loss math §5, toggle semantics §8, momentum/queue untouched §2.6) still holds and is reused unchanged.

---

## 1. Why this pivot

The `exp=5.no_aug` control run was a **NO-GO**: training the student with image augmentation OFF degraded quality to nearly unusable. Therefore **augmentation must stay ON**. Offline caching is fundamentally incompatible with fresh per-epoch augmentation (a cached teacher target is one fixed view; matching it to a freshly-augmented student view would require re-running the teacher every epoch = online). So we take the pre-agreed §9 fallback: an **online teacher** that runs live on the *same augmented batch* the student sees, each step. The `TeacherProvider` seam was built for exactly this.

## 2. What stays vs changes (relative to the caching branch)

**Reused unchanged (already implemented + reviewed on `itc_distill`):**
- `distillation/losses.py` — `itc_distill_loss` (in-batch B×B relational KL). Teacher feats are teacher feats regardless of source.
- `models/blip_pretrain.py` — `forward(..., teacher_img_feat, teacher_text_feat, distill_temp)` returning `(loss_ita, loss_itm, loss_lm, loss_itc_kd)`.
- `data/eval_validation_loss.py` — the 4-tuple unpack (forward still returns 4).
- `configs/pretrain.yaml` — the `distill.itc` + `teacher` blocks (drop the now-irrelevant `cache_dir`).

**Removed (offline-cache machinery, now dead):**
- `distillation/teacher_cache.py`, `distillation/test_teacher_cache.py`
- `build_teacher_cache.py` (its ITC feature-extraction logic is salvaged into the online teacher)
- `data/test_pretrain_dataset_cache.py`

**Reverted:**
- `data/pretrain_dataset.py` — drop the `teacher_cache` param and the 4-tuple `__getitem__` return; back to `(image, caption)`.
- `data/__init__.py` — drop the `TeacherCache` import and cache wiring in `create_dataset('pretrain')`. **Keep** the `pretrain_train_aug` toggle.
- `configs/pretrain.yaml` — `pretrain_train_aug: true` (augmentation required).

**New:**
- `distillation/online_teacher.py` — the `OnlineTeacher` provider.
- `pretrain.py` — build the teacher in `main()`, run it each step in `train()`.

## 3. Architecture

The single principle: **the student forward and the KD loss do not change at all** — only the *source* of the teacher features moves from "dataset cache (batch 4-tuple)" to "an `OnlineTeacher` call inside the train loop." Clean realization of the `TeacherProvider` seam.

### 3.1 `OnlineTeacher` (`distillation/online_teacher.py`)
```
class OnlineTeacher:
    __init__(checkpoint, image_size, vit='large', bert='base'):
        build BLIP-large via blip_pretrain(vit=vit, my_bert_size=bert, image_size=image_size, queue_size=...)
        load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'|self], strict=False)
        # free submodules itc_feats never uses, to ~halve teacher memory:
        #   visual_encoder_m, text_encoder_m, vision_proj_m, text_proj_m, text_decoder, itm_head, image_queue, text_queue
        .eval(); for p in params: p.requires_grad_(False)
    @torch.no_grad()
    itc_feats(image, caption) -> (img_feat[B,256], txt_feat[B,256]):   # L2-normalized
        with autocast(bf16):
            image_embeds = visual_encoder(image)
            img_feat = normalize(vision_proj(image_embeds[:,0,:]))
            text = tokenizer(caption, max_length=30, ...).to(image.device)
            text_out = text_encoder(text.input_ids, attention_mask=..., mode='text')
            txt_feat = normalize(text_proj(text_out.last_hidden_state[:,0,:]))
        return img_feat, txt_feat
```
Reuses the exact ITC-extraction logic from the old `build_teacher_cache.py`'s `make_feat_fn`. The teacher only needs its main encoders + projections — **no momentum encoder, no queue** is exercised (those exist on the constructed module but are never called by `itc_feats`).

### 3.2 Deployment — replicated per GPU (option i)
In `pretrain.py main()`, after building the student model + DDP wrap, construct **one `OnlineTeacher` per process** (`.to(device)`), **not** DDP-wrapped (frozen, no grads). Rationale (decided): a frozen teacher in inference (no optimizer state, no backward activations) — and with the unused momentum/decoder/ITM/queue submodules freed per §3.1, only the ITC path resident — is ~2 GB/GPU for BLIP-large (ViT-L + BERT-base); the student (DINOv3-small + MiniLM) is tiny, so replication fits in 24 GB and keeps all 4 GPUs doing student work. A dedicated teacher GPU (option ii) loses a student GPU AND serializes 3 ranks' teacher forwards on one GPU AND needs cross-process IPC — worse on both simplicity and throughput.

### 3.3 Train-loop wiring (`pretrain.py train()`)
Batch returns to a 2-tuple `(image, caption)`. Per step, after `image = image.to(device)`:
```
if itc_kd_enabled:
    with torch.no_grad():
        teacher_img_feat, teacher_text_feat = online_teacher.itc_feats(image, caption)
else:
    teacher_img_feat = teacher_text_feat = None
# unchanged from here: model(image, caption, alpha, teacher_img_feat=..., teacher_text_feat=..., distill_temp=...)
# unchanged: loss = ita+itm+lm + (itc_kd_weight*loss_itc_kd if enabled and not None); TB log of loss_train/itc_kd
```
The teacher sees the **same augmented `image` tensor** the student sees → consistent view, no extra transform.

## 4. Settled parameters

- **Batch fixed at 40** (per-GPU). Deliberate experimental control: the distillation run differs from the baseline ONLY by the KD term, not by batch size — isolates the KD effect. Global batch stays 40×world_size (~160 with 4 GPUs).
- **KD scope = local 40×40, no all-gather.** Each rank computes its own 40×40 teacher/student sim → KL; DDP averages gradients across ranks. No GPU communication for the KD. Consistent with BLIP's own ITC (local batch + queue, not synchronous global all-gather). Note this is a *biased* estimator vs a 160-negative KD (fewer negatives), accepted as the no-communication trade-off; batch-40 is held fixed regardless.
- **Precision = bf16.** Verified (2026-06-29): under `torch.amp.autocast(bf16)` the ITC-logit matmul `image_feat @ text_feat.t()` runs in **bfloat16** while softmax/log_softmax stay fp32. The teacher forward runs under the same bf16 autocast → teacher feats bf16. This is internally consistent with the student's own ITC similarities (also bf16). (Forcing fp32 KD is possible via `autocast(enabled=False)` but not chosen.)
- **Augmentation ON** (`pretrain_train_aug: true`) — the whole reason for this pivot.

## 5. Cost (stated honestly)

Every training step now pays an extra **BLIP-large forward** (the teacher is larger than the small student), in eval + `no_grad` + bf16 (forward-only, no backward). This is the price of online + augmentation that caching was meant to avoid; accepted because augmentation is essential. Mitigations already in the design: bf16, no_grad, eval. (Optional future: `torch.compile` the teacher.)

## 6. Testing

- **`itc_distill_loss`** — already unit-tested (carried over, unchanged).
- **`OnlineTeacher.itc_feats`** — CPU smoke: construct a small `blip_pretrain(vit='base', my_bert_size='base', queue_size=240)` (cached weights), wrap as a teacher, call `itc_feats` on a tiny batch; assert output shape `[B,256]`, L2-normalized (norm≈1), and that outputs carry no grad (`requires_grad is False`). The real BLIP-large path is exercised only in the end-to-end run (needs the checkpoint).
- **Train-loop integration** — short smoke: a few steps with `distill.itc.enabled=true` and the teacher constructed, on a tiny config; confirm `loss_train/itc_kd` is logged and finite and training steps. Regression: with `enabled=false`, batch is a 2-tuple and behavior matches pre-distillation.
- **Removal regression** — after deleting the cache code, the full remaining suite (`distillation.test_losses`) passes and the repo imports cleanly (`import data`, `import distillation`).

## 7. Open / required inputs

- **`teacher.checkpoint`** — path to the trained BLIP-large weights (user supplies). The teacher arch is assumed **ViT-L + BERT-base** (`vit='large'`, `my_bert_size='base'`) per original BLIP-large; if the user's teacher differs, adjust the `OnlineTeacher` construction. A loud check (e.g. fail on too many missing keys in `load_state_dict`) is advisable before a real run.

## 8. Out of scope (future)

- ITM and LM distillation (scaffolding via the same `distill.{itm,lm}` config + per-loss functions).
- Expanding KD negatives via feature all-gather (40×160) or a teacher-feature queue (40×(40+Q)) — both reintroduce communication/machinery and are deferred.

## 9. Branch strategy

Modify `itc_distill` forward with new commits: (a) remove cache files, (b) revert dataset + aug, (c) add `online_teacher.py` + wire `pretrain.py`. ~60% of the branch (loss, forward, config blocks) is reused as-is. History can be squashed at merge time. Branch is kept, not merged, until the online-teacher run is validated.
