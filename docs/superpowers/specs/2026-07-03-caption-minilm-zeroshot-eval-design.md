# Design: Caption-path minilm support + zero-shot captioning eval (M1)

Date: 2026-07-03
Status: Approved (brainstorming) → ready for implementation plan

## Motivation

Pretrain LM cross-entropy (`loss_lm`) turned out to be an insensitive metric: the
BLIP-large teacher's val LM CE (3.732, measured 2026-07-03) is only ~0.03 nats better
than the from-scratch small student (3.760 @ epoch 20), and the LM-distilled student
tracks the solo baseline exactly on val LM CE. Average teacher-forced CE saturates near
the data entropy floor and is dominated by function words, so it cannot distinguish
models that differ in real captioning ability.

The question this spec answers: **do the pretrained models differ on a real
*generation* captioning metric (CIDEr/BLEU/METEOR/ROUGE) that LM CE hides?**

We call this evaluation **M1: zero-shot captioning eval** — generate captions directly
from each pretrain checkpoint (no caption finetuning) and score against COCO references.

## Scope

In scope:
- Enable the caption model (`BLIP_Decoder`) to build the decoder for `my_bert_size` in
  {base, medium, minilm}, so minilm students can be loaded.
- Run M1 zero-shot eval on 4 pretrain checkpoints and produce a text comparison table.
- Add a `use_spice=False` path to `coco_caption_eval` to skip the slow SPICE scorer.

Out of scope (separate later spec):
- Caption **finetuning** of the small models (reuses the same enabling code).
- TensorBoard logging for the caption pipeline (only useful once training curves exist).

## Models compared (M1)

| label         | vit        | my_bert_size | checkpoint |
|---------------|-----------|--------------|------------|
| Large (teacher) | large     | base         | `output/official_pretrain_checkpoint/model_large.pth` |
| Base          | base      | base         | `output/official_pretrain_checkpoint/model_base_14M.pth` |
| Small-solo (exp7) | small_reg | minilm    | `output/pt_smallreg_minilm_baseline/checkpoint_19.pth` (epoch 20 final) |
| Small-distill (exp8) | small_reg | minilm | `output/pt_smallreg_minilm_lm_distill/checkpoint_05.pth` (latest available) |

The scientifically clean A/B is **Small-solo vs Small-distill** (identical arch + pretrain
recipe; only difference = LM distillation). Large/Base are reference ceilings, not
compute-matched.

## Key facts established during exploration

- `create_vit()` (models/blip.py) **already** supports `small_reg` (DINOv3_Wrapper, reg
  tokens 4, `pretrained=False`) — identical construction to `blip_pretrain`. The caption
  model's vision side needs **no change**.
- The only gap is the text decoder: `BLIP_Decoder` hardcodes `med_config.json` (base 768).
- Decoder configs already exist and match checkpoint shapes:
  - base → `configs/med_config.json` (hidden 768)
  - medium → `configs/med_medium_config.json` (hidden 512)
  - minilm → `configs/med_minilm_config.json` (hidden 384, cross-attn true, vocab 30524)
  - minilm vocab 30524 exactly matches the pretrain decoder after
    `resize_token_embeddings(len(tokenizer))` (30522 + [DEC]/[ENC] = 30524).
- `blip_decoder(pretrained='', **kwargs)` passes kwargs straight to `BLIP_Decoder` and
  asserts `len(msg.missing_keys)==0` after `load_checkpoint`. This assert is the main
  risk surface for the new minilm path.
- `train_caption.py` has **no TensorBoard**; results are written to `evaluate.txt`
  (metrics JSON) and `log.txt`. M1 relies on `evaluate.txt`.
- `coco_caption_eval` (data/utils.py) runs `COCOEvalCap.evaluate()`, whose scorer list is
  hardcoded and includes `(Spice(), "SPICE")`. SPICE is separable (not entangled) but has
  no exposed skip flag; it is also the slow scorer (Java scene-graph + CoreNLP download).

## Design

### Component 1 — `BLIP_Decoder.__init__` (models/blip.py)

Add `my_bert_size='base'` parameter. Select the decoder config by size and build the
decoder from it (base path unchanged → no behavior change for existing base callers):

```python
DECODER_CONFIGS = {
    'base':   'configs/med_config.json',
    'medium': 'configs/med_medium_config.json',
    'minilm': 'configs/med_minilm_config.json',
}
decoder_config = BertConfig.from_json_file(DECODER_CONFIGS[my_bert_size])
decoder_config.encoder_width = vision_width          # from create_vit (small_reg = 384)
self.text_decoder = BertLMHeadModel(config=decoder_config)
```

`vision_width` comes from `create_vit(vit, ...)` (768 base / 1024 large / 384 small_reg),
mirroring how `blip_pretrain` sets `decoder_config.encoder_width = vision_width`.

### Component 2 — `blip_decoder()` factory (models/blip.py)

No change. `**kwargs` already forwards `my_bert_size`. The `assert(missing_keys==0)` stays;
the smoke test verifies it holds for the minilm checkpoints.

### Component 3 — `train_caption.py`

Pass the bert size through to the model builder:

```python
model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'],
                     vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'],
                     vit_ckpt_layer=config['vit_ckpt_layer'], prompt=config['prompt'],
                     my_bert_size=config['my_bert_size'])
```

Also thread SPICE control: `coco_caption_eval(..., use_spice=config.get('use_spice', True))`.

### Component 4 — SPICE-off path (data/utils.py `coco_caption_eval`)

Add `use_spice=True` parameter (default preserves current behavior). When `False`, run the
same tokenize + scorer loop as `COCOEvalCap.evaluate()` but with the scorer list reduced to
Bleu/Meteor/Rouge/Cider (drop Spice). Implemented as a small local helper; no monkeypatch.

### Component 5 — Eval configs (configs/)

Four YAML files, explicit per the "settings live in the config/run name" preference.
Common fields: `image_root`, `ann_root`, `coco_gt_root`, `image_size: 224`, `prompt: ''`,
`use_spice: false`, `batch_size`, `max_epoch: 1`, `min_lr`, `weight_decay`, `num_workers`,
per-model `output_dir`. Per-model fields: `vit`, `my_bert_size`, `vit_grad_ckpt`,
`vit_ckpt_layer`, `pretrained`.

Rationale for the common values:
- `image_size: 224` — small_reg DINOv3 is fixed at 224; Large/Base pretrain is native 224.
  The data transform must match, so 224 for all (fair comparison).
- `prompt: ''` — pretrain decoders were trained on raw captions (BOS only), not the
  "a picture of " finetune prompt. Empty prompt is closest to the pretrain distribution and
  is applied identically to all 4 models.

### Component 6 — Run + collect

Run each config with `--evaluate` (single GPU via `CUDA_VISIBLE_DEVICES`; `train_caption.py`
falls back to non-distributed when no dist env vars are set). Each run writes `evaluate.txt`
(val + test metrics; the eval path generates both automatically). Collect the four
`evaluate.txt` files into a comparison table: CIDEr, BLEU@4, METEOR, ROUGE-L.

## Data flow

image → `create_vit` visual_encoder → image_embeds → `text_decoder.generate`
(beam search, prompt='') → caption strings → `save_result` JSON → `coco_caption_eval`
(use_spice=False) → metrics dict → `evaluate.txt` → comparison table.

## Testing

- **Smoke (required, before full runs):** for each of the 4 checkpoints, build via
  `blip_decoder(pretrained=..., vit=..., my_bert_size=..., image_size=224, prompt='')`,
  confirm the `assert(missing_keys==0)` passes, and generate a caption for 1–2 images to
  confirm the output string is sane. The small_reg/minilm students are the new path and the
  primary target of this check.
- **use_spice=False:** run `coco_caption_eval` on a tiny result file and confirm it returns
  Bleu/Meteor/Rouge/Cider and does not invoke SPICE.
- **Full:** run all 4 configs, assemble the table.

## Risks

- `assert(missing_keys==0)` in `blip_decoder` may fail for the minilm students if any key
  fails to match (DINOv3_Wrapper key naming vs the checkpoint, or a decoder shape/vocab
  mismatch). Official base/large use the original BLIP design and are known-good; the new
  risk is confined to the small_reg/minilm path and surfaces immediately in the smoke test.
- Zero-shot generation from a pretrain checkpoint produces rough captions; absolute CIDEr
  will be low for all models. M1 is a **relative** comparison — that is acceptable and the
  point.

## Open items / notes

- Distill checkpoint is `checkpoint_05.pth` (latest at spec time); rerun on a later epoch
  when the run advances if a matched-epoch comparison with the solo baseline is wanted.
- Commit policy: docs under `docs/superpowers/` have been kept untracked/gitignored in this
  repo; do not auto-commit code changes on `dev` without asking.
