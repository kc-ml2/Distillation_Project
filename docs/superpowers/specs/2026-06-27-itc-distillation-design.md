# ITC Knowledge Distillation (offline teacher cache) — Design

- **Date:** 2026-06-27
- **Branch (implementation):** `dev/itc_distill`
- **Status:** draft, pending review
- **Scope:** ITC distillation only. ITM/LM distillation are out of scope here but the
  scaffolding (config toggles, teacher provider, per-loss distill functions) is designed so
  they slot in later without rework.

---

## 1. Context & Goal

The model is a BLIP-style pretraining stack (`models/blip_pretrain.py`):
- ViT image encoder + BERT text encoder + BERT text decoder (decoder weight-tied to encoder),
- three training losses: **ITC** (image-text contrastive), **ITM** (image-text matching),
  **LM** (language modeling). `forward()` returns `(loss_ita, loss_itm, loss_lm)`.

Downstream target is **captioning**, so LM matters most, but the plan is to distill **all three**
losses (reported to give the best student). Each distillation term must be **independently
toggleable** so we can later ablate "which losses to distill."

This spec covers the **first** piece: distilling the **ITC** signal, built on a new branch
`dev/itc_distill`.

### Teacher / Student
- **Teacher:** BLIP-large (ViT-large + BERT-large), frozen. Produces 256-d ITC features.
- **Student:** DINOv3 small ViT + MiniLM BERT (pretrained backbones). Also projects to 256-d.
- Teacher and student are **architecturally unrelated** — different backbones, independently
  trained projection heads. Their 256-d embedding spaces share **no common basis**.

---

## 2. Key Design Decisions (with rationale)

1. **Distill the similarity matrix, not feature vectors (relational / "logit" distillation).**
   In classic KD, the "logit" is the pre-softmax score. For ITC that is the **similarity
   matrix** fed to the contrastive softmax, not the 256-d embeddings. Crucially the similarity
   matrix is **basis-free**: each `sim_ij` is an inner product *inside one model's own space*, so
   it is invariant to how that model rotated/permuted its axes. Feature MSE (matching the raw
   256-d vectors) is **basis-dependent** and would be fragile/meaningless for an unrelated
   teacher↔student pair without a learned aligner. → We match similarity-matrix **distributions**
   via KL. (Both project to 256-d, so dims even match — but equal dim ≠ aligned basis, so feature
   MSE is still out.)

2. **In-batch `B×B` matrix, not the queue-augmented `B×(B+Q)`.**
   Distillation requires teacher and student to compare the **same candidate set** (column
   correspondence). The momentum queue (`Q=57600`) holds *student-specific* past momentum
   features with no aligned teacher counterpart, so we deliberately keep the KD term on the
   clean in-batch `B×B` matrix.

3. **Offline teacher caching** (vs online teacher on GPU).
   The teacher ITC feature is `normalize(proj(CLS))` — **per-sample and batch-independent**.
   So we run the teacher **once** over the dataset, cache per-sample 256-d features to disk
   (~6 GB fp16 for ~6M caption rows), and at train time reconstruct any `B×B` teacher matrix by
   a cheap matmul of the cached features for whatever samples landed in the batch. This frees all
   GPUs for the student and avoids paying an expensive BLIP-large forward every step.

4. **Index-keyed cache.**
   `pretrain_dataset.annotation` is a **fixed-order** concatenation of the COCO and VG JSON lists
   (`data/pretrain_dataset.py`), so **row index `i` is a stable key**. The cache is two arrays
   indexed by `i`; the dataloader returns the cached features by index, so alignment ("same image,
   same caption") is automatic with no filename matching. A manifest guards against stale caches.

5. **Augmentation OFF (deterministic input).**
   Exact teacher↔student view match + caching ⟹ image input must be deterministic (fresh
   per-epoch augmentation would require re-running the teacher each epoch = online). We use
   `pretrain_train_aug=false` so training uses `transform_test` (resize+normalize, no
   RandomResizedCrop/Flip/RandAug). This toggle is already committed (`b988045`). A control run
   (`exp=5.no_aug`) validates that turning aug off does not hurt much — **this is the go/no-go
   gate for the whole caching approach** (see §9).

6. **Student momentum encoder + queue are untouched.**
   They belong to the existing `loss_ita` (BLIP self/momentum distillation) and are unrelated to
   the external teacher. The KD term uses **only** the student's *online* features
   (`image_feat`, `text_feat`) and the cached teacher features. There is nothing to "match"
   between the student's queue and the teacher. With distillation disabled, training is byte-for-
   byte the original.

   > **Note — two distinct "teachers", don't conflate them.** The student's momentum encoder is
   > an EMA copy of the student itself (`_momentum_update`, `blip_pretrain.py:344`). Each step the
   > **momentum (EMA) features of the current batch are enqueued** into the queue
   > (`_dequeue_and_enqueue(image_feat_m, text_feat_m)`, `blip_pretrain.py:379`), so the queue is a
   > rolling buffer of the most recent ~`Q` samples' *momentum* features (not the online ones —
   > the MoCo trick). Inside `loss_ita` this serves two jobs: extra negatives (`[B, B+Q]`,
   > `:368`) and the momentum **self-distillation** soft targets (`:364`). That makes it an
   > *internal self-teacher*, entirely separate from our *external* BLIP-large teacher:
   >
   > | | identity | owning loss | what we do |
   > |---|---|---|---|
   > | momentum self-teacher | student's own EMA | `loss_ita` (existing) | leave untouched |
   > | external teacher | BLIP-large (cached) | `L_itc_kd` (new) | add only this |
   >
   > The KD term deliberately stays on the in-batch `B×B` rather than the queue-augmented matrix
   > because the queue columns are student-time-ordered momentum features with no aligned teacher
   > counterpart; matching them would require running a *parallel teacher-feature queue* (see §10,
   > future). In-batch `B×B` needs no such alignment — the teacher cache already holds the features
   > for exactly those `B` indices.

7. **Additive loss + per-modality toggle.**
   `total = loss_ita + λ_itc·L_itc_kd` (later `+ λ_itm·L_itm_kd + λ_lm·L_lm_kd`). Config block
   `distill.{itc,itm,lm}.{enabled, weight, temp}`. When `itc.enabled=false` the teacher cache is
   not loaded, no teacher features flow, and the KD term is skipped entirely.

---

## 3. Architecture / Components

Designed as small, independently testable units behind clear interfaces.

### 3.1 `TeacherProvider` (interface) — makes GPU-deployment orthogonal
A thin abstraction returning per-batch teacher ITC features. This is what lets us defer/replace
the GPU-deployment choice.

```
TeacherProvider.get_itc_feats(batch_indices) -> (img_feat_t [B,256], txt_feat_t [B,256])
```
- **Now:** `CacheTeacherProvider` — memory-maps the cached arrays, gathers rows by index.
  (In practice the dataset itself reads the cache by index and returns the features inline —
  see §3.3 — so the "provider" for the cache path is effectively the dataset + memmap. The
  interface is kept as the conceptual seam so an online provider can replace it later.)
- **Future (not built now):** `ReplicatedTeacherProvider` / dedicated-GPU provider that runs a
  live teacher forward. Same return contract.

### 3.2 Teacher cache builder (Phase 0)
New standalone script `build_teacher_cache.py` + helper module (`distillation/teacher_cache.py`):
- Loads BLIP-large, frozen/eval.
- Iterates `pretrain_dataset` in index order `0..N-1` with **`transform_test`** (deterministic;
  same VG region crop as training).
- Per row `i`: teacher forward → `img_feat_t[i]`, `txt_feat_t[i]` (256-d, L2-normalized), written
  to disk-backed arrays (`np.memmap`/`.npy`, **fp16**, shape `[N,256]`).
- Writes a **manifest** (JSON) recording: `N`, `dim=256`, ordered `train_file` paths and per-file
  lengths (dataset signature), teacher checkpoint id, and the transform spec. Loaded caches are
  validated against the live dataset signature; mismatch → hard error (never silently use a stale
  cache).
- One-time cost ≈ one forward-only pass with BLIP-large; may be sharded across GPUs by index
  (each rank writes its strided indices), or run single-GPU sequentially.

### 3.3 Dataset change
`data/pretrain_dataset.py`:
- Accepts an optional `teacher_cache` (mmap arrays + manifest).
- When present, `__getitem__(i)` returns `(image, caption, img_feat_t[i], txt_feat_t[i])`.
  Cache is **fp16 on disk**, upcast to **fp32 on read** (the `B×B` matmul + softmax/KL run in
  fp32); default collate stacks the features into `[B,256]`.
- When absent, returns `(image, caption)` exactly as today (backward compatible).

`data/__init__.py`: `create_dataset('pretrain', ...)` wires the cache into `pretrain_dataset`
only when `distill.itc.enabled` is true.

### 3.4 ITC distill loss (pure function)
New `distillation/losses.py`:
```
itc_distill_loss(image_feat_s, text_feat_s, img_feat_t, txt_feat_t, temp) -> scalar
```
Pure, side-effect free, unit-testable (see §6). Teacher features are treated as constants
(`detach`). Math in §5.

### 3.5 Model forward change
`models/blip_pretrain.py`:
- `forward()` gains optional `teacher_img_feat=None, teacher_text_feat=None` (and a distill
  config / temp).
- When provided, compute `loss_itc_kd = itc_distill_loss(image_feat, text_feat,
  teacher_img_feat, teacher_text_feat, temp)` using the **existing online** `image_feat`/
  `text_feat` (already computed at `blip_pretrain.py:331,337`); otherwise `loss_itc_kd = None`.
- Return signature extended to `(loss_ita, loss_itm, loss_lm, loss_itc_kd)`
  (`loss_itc_kd is None` when distillation is off). Momentum/queue path unchanged.

### 3.6 Training loop change
`pretrain.py`:
- Read `distill` config; if `itc.enabled`, load cache (validate manifest) and pass batch teacher
  features into `forward`.
- `total_loss = loss_ita + loss_itm + loss_lm + λ_itc·loss_itc_kd` (λ term added only when
  present).
- Log `loss_train/itc_kd` to TensorBoard alongside the existing losses.

### 3.7 File-change summary
| File | Change |
|---|---|
| `distillation/teacher_cache.py` | **new** — cache builder + memmap loader + manifest |
| `distillation/losses.py` | **new** — `itc_distill_loss` (itm/lm later) |
| `build_teacher_cache.py` | **new** — Phase-0 CLI script |
| `data/pretrain_dataset.py` | optional teacher_cache; return feats by index |
| `data/__init__.py` | wire cache into pretrain dataset when distill enabled |
| `models/blip_pretrain.py` | optional teacher feats → `loss_itc_kd`; extend return |
| `pretrain.py` | load cache, pass feats, add λ·loss, TB logging |
| `configs/pretrain.yaml` | `distill` block |

---

## 4. Data Flow

**Phase 0 — offline cache (once):**
```
BLIP-large (frozen/eval)
pretrain_dataset, transform_test (deterministic), index 0..N-1, no shuffle
  row i -> teacher forward -> img_feat_t[i], txt_feat_t[i]  (256-d, L2norm)
write: teacher_img_feats[N,256] fp16, teacher_txt_feats[N,256] fp16, manifest.json
```

**Phase 1 — student training (all GPUs on student):**
```
__getitem__(i) -> (image, caption, img_feat_t[i], txt_feat_t[i])      # index -> auto-aligned
collate        -> img_feat_t[B,256], txt_feat_t[B,256]

forward(student):
  existing: image_feat, text_feat, loss_ita (queue+momentum+GT)       # unchanged
            loss_itm, loss_lm                                         # unchanged
  new:      loss_itc_kd = itc_distill_loss(image_feat, text_feat,
                                           img_feat_t, txt_feat_t, temp)
total = loss_ita + loss_itm + loss_lm + λ_itc·loss_itc_kd
```

---

## 5. ITC Distillation Loss (math)

Features are L2-normalized, so inner products are cosine similarities. Let `τ = distill.itc.temp`.
Teacher features are detached (constants).

```
S_s = (image_feat_s @ text_feat_s.T) / τ      # [B,B], student, requires grad
S_t = (img_feat_t  @ txt_feat_t.T ) / τ        # [B,B], teacher, detached

# image->text: each row = an image, distribution over the B texts (dim=1)
L_i2t = KL( softmax(S_t, 1) || softmax(S_s, 1) )
      = F.kl_div(log_softmax(S_s,1), softmax(S_t,1), reduction='batchmean')

# text->image: transpose, rows = texts, distribution over the B images
L_t2i = F.kl_div(log_softmax(S_s.T,1), softmax(S_t.T,1), reduction='batchmean')

L_itc_kd = 0.5 * (L_i2t + L_t2i) * (τ**2)
```
- **Same `τ` for teacher and student** so the distributions are comparable.
- `τ²` is the conventional Hinton scaling that keeps gradient magnitude roughly independent of `τ`.
- `τ` is the **distillation temperature**, separate from the model's learnable `logit_scale`
  (which stays with `loss_ita`). Suggested default `τ ≈ 0.05`, tunable in ~`[0.04, 0.1]`; an
  alternative is to set it to the teacher's own effective temperature (cache the teacher scale).
- The diagonal is the true pair; the teacher's off-diagonal soft mass on hard negatives is the
  transferred "dark knowledge." If two batch rows share an image, the teacher naturally assigns
  high off-diagonal similarity and the student learns that soft structure — a benefit of soft
  distillation over hard labels.

---

## 6. Testing Strategy

Unit tests use stdlib `unittest` (no pytest in the `kd_r4` env), matching the existing
`claude_skills/monitoring/test_*.py` convention.

- **`itc_distill_loss`:**
  - teacher == student features ⇒ loss ≈ 0.
  - gradient flows to student features, **not** to teacher features (teacher detached).
  - i2t/t2i symmetry on a transposed input.
  - shape/dtype handling for a small `[B,256]` case.
- **Cache builder:** small synthetic dataset ⇒ cache shape `[N,256]`, dtype fp16; `feat[i]`
  matches a direct teacher forward on sample `i` (index alignment); manifest signature round-trips.
- **Manifest guard:** changed dataset signature ⇒ loader raises (no silent stale-cache use).
- **Toggle regression:** `distill.itc.enabled=false` ⇒ `forward` returns `loss_itc_kd=None` and the
  total loss equals the pre-distillation baseline (guards backward compatibility).
- **Determinism:** with aug off, two `__getitem__(i)` calls return identical image tensors (so the
  cached teacher feature corresponds to the student input).

---

## 7. Config Schema (yaml)

```yaml
distill:
  itc:
    enabled: true
    weight: 1.0          # λ_itc
    temp:   0.05         # τ, distillation temperature (≠ model logit_scale)
    cache_dir: '/home/minwoo/Distillation_Project/output/teacher_cache_blip_large'
  # itm / lm added in later phases, same shape:
  # itm: { enabled: false, weight: 1.0, temp: 1.0 }
  # lm:  { enabled: false, weight: 1.0, temp: 1.0 }

teacher:
  arch: 'blip_large'
  checkpoint: '<REQUIRED USER INPUT: path to the trained BLIP-large checkpoint>'
```
> `teacher.checkpoint` is the one value this design cannot infer — supply the trained BLIP-large
> weights path before running Phase 0.
`pretrain_train_aug: false` (already present) is required whenever distillation with the cache is
enabled.

---

## 8. Toggle Semantics

- `distill.itc.enabled = false` → cache not loaded, dataset returns `(image, caption)`, forward
  returns `loss_itc_kd = None`, loop adds nothing → **exact original training**.
- `weight = 0` with `enabled = true` → KD computed (and logged) but contributes 0 to the gradient;
  useful for measuring the KD value without affecting training.
- ITM/LM distillation reuse this exact pattern under `distill.itm` / `distill.lm`.

---

## 9. Open Dependency & Risk

- **Caching premise depends on the `5.no_aug` control run.** If aug-off degrades quality too much,
  the cache approach is invalid and we revisit: (a) online teacher (replicated or dedicated-GPU
  provider — the `TeacherProvider` seam contains this blast radius to the provider impl + dataset
  cache-loading), (b) seeded per-index deterministic augmentation (fixed view per sample), or
  (c) per-epoch teacher re-cache. **Compare `exp=4.logit_scale_no_decay` (aug on) vs
  `exp=5.no_aug` (aug off)** — same config otherwise.
- **Cache staleness:** any change to `train_file` contents/order invalidates index keys; the
  manifest signature check is the guard.
- **Cache build cost:** one BLIP-large forward pass over ~6M rows; shard by index across GPUs if
  the sequential pass is too slow.

---

## 10. Out of Scope (future phases)
- ITM distillation (likely: distill teacher ITM binary logits on shared hard negatives — connects
  back to the ITC sim used for mining).
- LM distillation (distill teacher decoder token logits / KL on the vocab distribution).
- Online `TeacherProvider` implementations.
- Queue-augmented KD (`B×(B+Q)` with an aligned teacher feature queue).
