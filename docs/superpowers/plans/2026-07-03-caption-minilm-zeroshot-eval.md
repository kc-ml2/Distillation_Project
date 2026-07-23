# Caption-path minilm support + zero-shot captioning eval (M1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the BLIP caption model (`BLIP_Decoder`) load `small_reg`/`minilm` pretrain checkpoints and run zero-shot captioning eval (CIDEr/BLEU/METEOR/ROUGE) on 4 pretrained models, to see whether real captioning metrics reveal differences that pretrain LM cross-entropy hides.

**Architecture:** Three small edits to shared code — (1) select the decoder BERT config by `my_bert_size` in `BLIP_Decoder`, (2) guard the `visual_encoder.pos_embed` interpolation in `load_checkpoint` so DINOv3 encoders don't KeyError, (3) add a `use_spice=False` scoring path to `coco_caption_eval` — then thread `my_bert_size`/`use_spice` through `train_caption.py`, add 4 eval configs, and run `--evaluate` on each.

**Tech Stack:** PyTorch, timm 1.0.27 (DINOv3 ViT), HuggingFace transformers (BertLMHeadModel), pycocoevalcap 9.x (+ Java 11 for tokenizer/METEOR), pytest 9.1.1, conda env `kd_r4`.

## Global Constraints

- Python interpreter: `/home/minwoo/miniconda3/envs/kd_r4/bin/python` (env `kd_r4`). Always use this, not the default `python`.
- All eval configs use `image_size: 224` (small_reg DINOv3 is fixed at 224; base/large pretrain is native 224), `prompt: ''` (pretrain decoders were trained on raw captions, not the finetune prompt), `use_spice: false`.
- Do NOT modify anything under `/home/minwoo/Distillation_Project_lm_distill` — that is the live 4-GPU training worktree. All edits here are in the main repo (`/home/minwoo/Distillation_Project`, branch `dev`) and do not affect the running job.
- Commit policy: this repo asks before committing to `dev`/`main`. Each task below ends with a commit step, but confirm with the user before running it (or create a `dev/*claude*` branch where auto-commit is allowed). Note: the working tree already has unrelated uncommitted changes (`models/blip_pretrain.py`, `eval_official_pretrain_val_loss.py` from teacher-CE tooling) — stage only the files each task names.
- `docs/superpowers/` is gitignored in this repo; the spec/plan are not committed.

**Spec:** `docs/superpowers/specs/2026-07-03-caption-minilm-zeroshot-eval-design.md`

**Models under test:**

| label | vit | my_bert_size | checkpoint |
|---|---|---|---|
| Large (teacher) | large | base | `output/official_pretrain_checkpoint/model_large.pth` |
| Base | base | base | `output/official_pretrain_checkpoint/model_base_14M.pth` |
| Small-solo (exp7) | small_reg | minilm | `output/pt_smallreg_minilm_baseline/checkpoint_19.pth` |
| Small-distill (exp8) | small_reg | minilm | `output/pt_smallreg_minilm_lm_distill/checkpoint_05.pth` (latest) |

---

## File Structure

- Modify `models/blip.py` — add module-level `DECODER_CONFIGS`; `BLIP_Decoder.__init__` gains `my_bert_size`; `load_checkpoint` guards `visual_encoder.pos_embed`.
- Modify `data/utils.py` — `coco_caption_eval` gains `use_spice`; add `_score_no_spice(gts, res)` helper.
- Modify `train_caption.py` — pass `my_bert_size` to `blip_decoder`; pass `use_spice` to `coco_caption_eval`.
- Create `configs/caption_eval_{large,base,small_solo,small_distill}.yaml`.
- Create `tests/test_caption_decoder_bert_size.py`, `tests/test_caption_load_small_checkpoint.py`, `tests/test_caption_score_no_spice.py`.

---

## Task 1: Select decoder BERT config by `my_bert_size` in `BLIP_Decoder`

**Files:**
- Modify: `models/blip.py` (`BLIP_Decoder.__init__`, ~lines 79-101; add module-level constant near top of file)
- Test: `tests/test_caption_decoder_bert_size.py`

**Interfaces:**
- Produces: `BLIP_Decoder(..., my_bert_size='base'|'medium'|'minilm')`; the decoder is built from `configs/med_config.json` (base, 768), `configs/med_medium_config.json` (medium, 512), or `configs/med_minilm_config.json` (minilm, 384). `decoder_config.encoder_width` is set to the vision width from `create_vit` (small_reg → 384).

- [ ] **Step 1: Write the failing test**

Create `tests/test_caption_decoder_bert_size.py`:

```python
from models.blip import BLIP_Decoder


def test_minilm_decoder_hidden_size():
    model = BLIP_Decoder(vit='small_reg', image_size=224, prompt='', my_bert_size='minilm')
    assert model.text_decoder.config.hidden_size == 384
    # encoder_width must be overridden to the small_reg vision width (384), not the json's 768
    assert model.text_decoder.config.encoder_width == 384


def test_base_decoder_hidden_size_unchanged():
    model = BLIP_Decoder(vit='base', image_size=224, prompt='', my_bert_size='base')
    assert model.text_decoder.config.hidden_size == 768
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/minwoo/Distillation_Project && /home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_caption_decoder_bert_size.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'my_bert_size'`.

- [ ] **Step 3: Add the `DECODER_CONFIGS` constant**

In `models/blip.py`, add near the other module-level definitions (after imports, before `class BLIP_Base`):

```python
# BLIP decoder BERT config per language-model size. Mirrors blip_pretrain's model_specs.
# vocab_size 30524 in these already matches the pretrain decoder after
# resize_token_embeddings(len(tokenizer)) (30522 + [DEC]/[ENC]).
DECODER_CONFIGS = {
    'base':   'configs/med_config.json',
    'medium': 'configs/med_medium_config.json',
    'minilm': 'configs/med_minilm_config.json',
}
```

- [ ] **Step 4: Wire `my_bert_size` into `BLIP_Decoder.__init__`**

Change the signature (add `my_bert_size='base'` as the last keyword arg):

```python
    def __init__(self,
                 med_config = 'configs/med_config.json',
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 prompt = 'a picture of ',
                 my_bert_size = 'base',
                 ):
```

Replace the decoder-build block:

```python
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel(config=med_config)
```

with (uses `med_config` as the fallback for unknown sizes, so the param stays meaningful):

```python
        decoder_config = BertConfig.from_json_file(DECODER_CONFIGS.get(my_bert_size, med_config))
        decoder_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel(config=decoder_config)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/minwoo/Distillation_Project && /home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_caption_decoder_bert_size.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit** (confirm per commit policy first)

```bash
git add models/blip.py tests/test_caption_decoder_bert_size.py
git commit -m "feat: select caption decoder BERT config by my_bert_size"
```

---

## Task 2: Guard `visual_encoder.pos_embed` interpolation for DINOv3 encoders

**Files:**
- Modify: `models/blip.py` (`load_checkpoint`)
- Test: `tests/test_caption_load_small_checkpoint.py`

**Interfaces:**
- Consumes: `BLIP_Decoder(..., my_bert_size='minilm')` from Task 1.
- Produces: `blip_decoder(pretrained=<path>, vit='small_reg', my_bert_size='minilm', image_size=224, prompt='')` loads a DINOv3 small_reg/minilm checkpoint without `KeyError` and with `missing_keys == 0`.

Context: `load_checkpoint` currently does `state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(...)` unconditionally. DINOv3 checkpoints have no `visual_encoder.pos_embed` key (they store `visual_encoder.model.*` and handle position internally), so this raises `KeyError`. Loading at native 224 needs no interpolation anyway.

- [ ] **Step 1: Write the failing test**

Create `tests/test_caption_load_small_checkpoint.py`:

```python
import os
import pytest
from models.blip import blip_decoder

CKPT = "output/pt_smallreg_minilm_baseline/checkpoint_19.pth"


@pytest.mark.skipif(not os.path.exists(CKPT), reason="small_reg/minilm checkpoint not present")
def test_load_small_reg_minilm_checkpoint_no_keyerror():
    # blip_decoder asserts missing_keys == 0 internally; must not KeyError on pos_embed either.
    model = blip_decoder(pretrained=CKPT, image_size=224, vit='small_reg',
                         prompt='', my_bert_size='minilm')
    assert model is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/minwoo/Distillation_Project && /home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_caption_load_small_checkpoint.py -v`
Expected: FAIL — `KeyError: 'visual_encoder.pos_embed'`.

- [ ] **Step 3: Add the guard**

In `models/blip.py` `load_checkpoint`, change:

```python
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)
```

to:

```python
    if 'visual_encoder.pos_embed' in state_dict:
        state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/minwoo/Distillation_Project && /home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_caption_load_small_checkpoint.py -v`
Expected: PASS. (If it fails on the internal `assert(len(msg.missing_keys)==0)` instead of KeyError, stop and print `msg.missing_keys` — that indicates a key-name mismatch to investigate, not a pos_embed problem.)

- [ ] **Step 5: Commit** (confirm per commit policy first)

```bash
git add models/blip.py tests/test_caption_load_small_checkpoint.py
git commit -m "fix: guard visual_encoder.pos_embed interpolation for DINOv3 checkpoints"
```

---

## Task 3: Add `use_spice=False` scoring path to `coco_caption_eval`

**Files:**
- Modify: `data/utils.py` (`coco_caption_eval`; add `_score_no_spice`)
- Test: `tests/test_caption_score_no_spice.py`

**Interfaces:**
- Produces: `_score_no_spice(gts, res) -> dict` where `gts`/`res` are `{img_id: [{'caption': str}, ...]}`; returns `{'Bleu_1':..,'Bleu_2':..,'Bleu_3':..,'Bleu_4':..,'METEOR':..,'ROUGE_L':..,'CIDEr':..}` (no SPICE). `coco_caption_eval(coco_gt_root, results_file, split, use_spice=True)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_caption_score_no_spice.py`:

```python
from data.utils import _score_no_spice


def test_score_no_spice_returns_cider_not_spice():
    gts = {1: [{'caption': 'a cat sitting on a mat'}]}
    res = {1: [{'caption': 'a cat sitting on a mat'}]}
    scores = _score_no_spice(gts, res)
    assert 'CIDEr' in scores
    assert 'Bleu_4' in scores
    assert 'SPICE' not in scores
    assert scores['Bleu_1'] > 0.9  # identical hypothesis/reference
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/minwoo/Distillation_Project && /home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_caption_score_no_spice.py -v`
Expected: FAIL — `ImportError: cannot import name '_score_no_spice'`.

- [ ] **Step 3: Implement `_score_no_spice` and thread `use_spice`**

In `data/utils.py`, add the helper (near `coco_caption_eval`):

```python
def _score_no_spice(gts, res):
    """COCOEvalCap scoring without the slow, Java-heavy SPICE scorer.
    gts/res: {img_id: [{'caption': str}, ...]}. Returns {metric: score}."""
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider

    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    out = {}
    for scorer, method in scorers:
        score, _ = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for sc, m in zip(score, method):
                out[m] = sc
        else:
            out[method] = score
    return out
```

Then change `coco_caption_eval`'s signature and body. Signature:

```python
def coco_caption_eval(coco_gt_root, results_file, split, use_spice=True):
```

Replace:

```python
    coco_eval = COCOEvalCap(coco, coco_result)
    ...
    coco_eval.evaluate()
```

with:

```python
    coco_eval = COCOEvalCap(coco, coco_result)

    if use_spice:
        coco_eval.evaluate()
    else:
        imgIds = coco_eval.params['image_id']
        gts = {i: coco_eval.coco.imgToAnns[i] for i in imgIds}
        res = {i: coco_eval.cocoRes.imgToAnns[i] for i in imgIds}
        coco_eval.eval = _score_no_spice(gts, res)
```

(Leave the trailing `for metric, score in coco_eval.eval.items(): print(...)` and `return coco_eval` unchanged — both paths populate `coco_eval.eval`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/minwoo/Distillation_Project && /home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_caption_score_no_spice.py -v`
Expected: PASS. (First run may print CoreNLP/Java tokenizer messages — that is fine.)

- [ ] **Step 5: Commit** (confirm per commit policy first)

```bash
git add data/utils.py tests/test_caption_score_no_spice.py
git commit -m "feat: add use_spice=False path to coco_caption_eval"
```

---

## Task 4: Thread config through `train_caption.py` and add 4 eval configs

**Files:**
- Modify: `train_caption.py` (blip_decoder call ~line 117; the two `coco_caption_eval` calls ~lines 151-152)
- Create: `configs/caption_eval_large.yaml`, `configs/caption_eval_base.yaml`, `configs/caption_eval_small_solo.yaml`, `configs/caption_eval_small_distill.yaml`

**Interfaces:**
- Consumes: `blip_decoder(..., my_bert_size=...)` (Task 1), pos_embed guard (Task 2), `coco_caption_eval(..., use_spice=...)` (Task 3).
- Produces: `train_caption.py --config <cfg> --evaluate` builds the right arch from the config and writes metrics to `<output_dir>/evaluate.txt`.

- [ ] **Step 1: Pass `my_bert_size` to `blip_decoder`**

In `train_caption.py`, change the `blip_decoder(...)` call (line ~117) to add `my_bert_size`:

```python
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'],
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                           prompt=config['prompt'], my_bert_size=config['my_bert_size'])
```

- [ ] **Step 2: Pass `use_spice` to both `coco_caption_eval` calls**

Change the two calls (lines ~151-152):

```python
            coco_val = coco_caption_eval(config['coco_gt_root'],val_result_file,'val', use_spice=config.get('use_spice', True))
            coco_test = coco_caption_eval(config['coco_gt_root'],test_result_file,'test', use_spice=config.get('use_spice', True))
```

- [ ] **Step 3: Create the small-solo config**

Create `configs/caption_eval_small_solo.yaml`:

```yaml
image_root: '/home/minwoo/Distillation_Project/datasets/vision/coco/images/'
ann_root: '/home/minwoo/Distillation_Project/datasets/vision/coco/annotations/'
coco_gt_root: '/home/minwoo/Distillation_Project/datasets/vision/coco/annotation/coco_gt'
pretrained: '/home/minwoo/Distillation_Project/output/pt_smallreg_minilm_baseline/checkpoint_19.pth'
output_dir: 'output/caption_zeroshot/small_solo'
vit: 'small_reg'
vit_grad_ckpt: False
vit_ckpt_layer: 0
my_bert_size: 'minilm'
image_size: 224
batch_size: 32
init_lr: 1.0e-5
weight_decay: 0.05
min_lr: 0
max_epoch: 1
prompt: ''
use_spice: false
```

- [ ] **Step 4: Create the small-distill config**

Create `configs/caption_eval_small_distill.yaml` — identical to Step 3 except:

```yaml
pretrained: '/home/minwoo/Distillation_Project/output/pt_smallreg_minilm_lm_distill/checkpoint_05.pth'
output_dir: 'output/caption_zeroshot/small_distill'
```

(all other fields exactly as `caption_eval_small_solo.yaml`: `vit: small_reg`, `my_bert_size: minilm`, `image_size: 224`, `prompt: ''`, `use_spice: false`, same roots/batch/lr/max_epoch).

- [ ] **Step 5: Create the base config**

Create `configs/caption_eval_base.yaml` — same common fields, with:

```yaml
pretrained: '/home/minwoo/Distillation_Project/output/official_pretrain_checkpoint/model_base_14M.pth'
output_dir: 'output/caption_zeroshot/base'
vit: 'base'
my_bert_size: 'base'
```

(full file: same `image_root`/`ann_root`/`coco_gt_root`, `vit_grad_ckpt: False`, `vit_ckpt_layer: 0`, `image_size: 224`, `batch_size: 32`, `init_lr: 1.0e-5`, `weight_decay: 0.05`, `min_lr: 0`, `max_epoch: 1`, `prompt: ''`, `use_spice: false`.)

- [ ] **Step 6: Create the large config**

Create `configs/caption_eval_large.yaml` — same common fields, with:

```yaml
pretrained: '/home/minwoo/Distillation_Project/output/official_pretrain_checkpoint/model_large.pth'
output_dir: 'output/caption_zeroshot/large'
vit: 'large'
my_bert_size: 'base'
```

(full file: same roots, `vit_grad_ckpt: False`, `vit_ckpt_layer: 0`, `image_size: 224`, `batch_size: 16`, `init_lr: 1.0e-5`, `weight_decay: 0.05`, `min_lr: 0`, `max_epoch: 1`, `prompt: ''`, `use_spice: false`. Note `batch_size: 16` — the large ViT needs more memory; generation is forward-only so 16 is safe alongside the running training job.)

- [ ] **Step 7: Smoke-verify all 4 build + load + generate one caption**

Run this script (single GPU; picks GPU 2 — adjust if busy):

```bash
cd /home/minwoo/Distillation_Project
CUDA_VISIBLE_DEVICES=2 /home/minwoo/miniconda3/envs/kd_r4/bin/python - <<'EOF'
import yaml, torch
from models.blip import blip_decoder
from PIL import Image
from torchvision import transforms
cfgs = ['configs/caption_eval_large.yaml','configs/caption_eval_base.yaml',
        'configs/caption_eval_small_solo.yaml','configs/caption_eval_small_distill.yaml']
tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))])
# any coco val image
import glob, os
img_path = sorted(glob.glob('datasets/vision/coco/images/val2014/*.jpg'))[0]
img = tf(Image.open(img_path).convert('RGB')).unsqueeze(0).cuda()
for c in cfgs:
    cfg = yaml.safe_load(open(c))
    m = blip_decoder(pretrained=cfg['pretrained'], image_size=cfg['image_size'], vit=cfg['vit'],
                     vit_grad_ckpt=False, vit_ckpt_layer=0, prompt=cfg['prompt'],
                     my_bert_size=cfg['my_bert_size']).cuda().eval()
    with torch.no_grad():
        cap = m.generate(img, sample=False, num_beams=3, max_length=30, min_length=5)
    print(f"[OK] {os.path.basename(c)}: {cap}")
    del m; torch.cuda.empty_cache()
EOF
```

Expected: 4 lines `[OK] caption_eval_*.yaml: ['<some caption text>']`, no KeyError / AssertionError. Captions will be rough (zero-shot pretrain) — that is expected; the check is that each model builds, loads, and emits a caption string.

- [ ] **Step 8: Commit** (confirm per commit policy first)

```bash
git add train_caption.py configs/caption_eval_large.yaml configs/caption_eval_base.yaml configs/caption_eval_small_solo.yaml configs/caption_eval_small_distill.yaml
git commit -m "feat: config-driven caption zero-shot eval for base/large/minilm students"
```

---

## Task 5: Run M1 on all 4 models and assemble the comparison table

**Files:**
- Produces: `output/caption_zeroshot/{large,base,small_solo,small_distill}/evaluate.txt`
- No code changes; run + collect only.

**Interfaces:**
- Consumes: Task 4 configs + wiring.

- [ ] **Step 1: Run each config with `--evaluate`**

Run one at a time (each generates on COCO val+test ≈ 5000+5000 images; expect ~10-40 min per model on a shared GPU). Use a GPU with free memory (`nvidia-smi` to pick):

```bash
cd /home/minwoo/Distillation_Project
for cfg in small_solo small_distill base large; do
  echo "=== $cfg ==="
  CUDA_VISIBLE_DEVICES=2 /home/minwoo/miniconda3/envs/kd_r4/bin/python train_caption.py \
    --config configs/caption_eval_${cfg}.yaml --evaluate 2>&1 | tail -8
done
```

Expected: each prints `Bleu_1/.../CIDEr` lines (no SPICE) and writes `output/caption_zeroshot/<cfg>/evaluate.txt`.

- [ ] **Step 2: Assemble the comparison table**

```bash
cd /home/minwoo/Distillation_Project
/home/minwoo/miniconda3/envs/kd_r4/bin/python - <<'EOF'
import json, os
rows = ['large','base','small_solo','small_distill']
keys = ['test_CIDEr','test_Bleu_4','test_METEOR','test_ROUGE_L']
print(f"{'model':16} " + " ".join(f"{k:14}" for k in keys))
for r in rows:
    p = f"output/caption_zeroshot/{r}/evaluate.txt"
    if not os.path.exists(p):
        print(f"{r:16} (missing)"); continue
    d = json.loads(open(p).read().strip().splitlines()[-1])
    print(f"{r:16} " + " ".join(f"{d.get(k,float('nan')):<14.4f}" for k in keys))
EOF
```

Expected: a 4-row table. The headline comparison is **small_solo vs small_distill** CIDEr (the clean A/B); large/base are reference ceilings.

- [ ] **Step 3: Report findings to the user**

Summarize the table in chat: does `small_distill` beat `small_solo` on CIDEr/BLEU@4 (i.e., did LM distillation help a real captioning metric even though val LM CE was flat), and how far below the large/base ceilings the small models sit. No commit (outputs are results, not code).

---

## Self-Review

**Spec coverage:**
- Enable minilm decoder in caption path → Task 1. ✓
- DINOv3 checkpoint load (pos_embed) → Task 2 (this was the runtime KeyError found after the spec was written; added here). ✓
- SPICE off → Task 3. ✓
- Thread through train_caption + 4 configs (image_size 224, prompt '', use_spice false) → Task 4. ✓
- Run + text comparison table → Task 5. ✓
- Smoke (missing_keys==0 + generate) → Task 2 test + Task 4 Step 7. ✓
- Out of scope (finetuning, TB logging) → not present. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; config Steps 4-6 state their full field set explicitly. ✓

**Type consistency:** `my_bert_size` (str) consistent across Task 1/4; `use_spice` (bool) consistent across Task 3/4; `_score_no_spice(gts, res) -> dict` matches its test and its call site in `coco_caption_eval`. ✓

**Note added vs spec:** Task 2 (pos_embed guard) was not in the original spec's risk list — it was discovered during plan exploration and is required for any small_reg load to work. The spec's risk section should be updated to mention it if the spec is revisited.
