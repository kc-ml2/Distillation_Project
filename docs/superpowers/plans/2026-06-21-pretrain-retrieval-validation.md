# Pretrain Retrieval Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the momentum-encoder-dependent `loss_val/ita` signal with a real COCO Karpathy retrieval evaluation (R@1/R@5/R@10), reusing the existing BLIP retrieval code, runnable both as a periodic in-training check and as a standalone checkpoint-eval script.

**Architecture:** Split the existing `evaluate_retrieval` in `eval_validation_tool.py` into a cheap ITC-only tier and an expensive ITM-rerank tier (sharing an embedding-extraction helper). Wrap both in a new `RetrievalValRunner` in `data/eval_validation_retrieval.py`, mirroring the existing `PretrainValLossRunner` pattern in `data/eval_validation_loss.py`. Wire the runner into `pretrain.py` next to the existing `val_loss_runner`, and add a standalone `eval_pretrain_retrieval.py` script mirroring `eval_official_pretrain_val_loss.py`.

**Tech Stack:** PyTorch 2.6 (kd_r4 conda env), `torch.distributed`, tests written as stdlib `unittest.TestCase` classes and run via `pytest` (installed in kd_r4 for nicer output; the test code itself doesn't need pytest-only features).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-21-pretrain-retrieval-validation-design.md` — every requirement in that doc must map to a task below.
- Run all Python commands with `/home/minwoo/miniconda3/envs/kd_r4/bin/python` (this is the env with `torch` and `pytest` installed; the default `python3` on PATH has neither).
- Run all commands from the repo root: `/home/minwoo/Distillation_Project`.
- Write tests as `unittest.TestCase` classes, but run them with `pytest tests/test_x.py -v` (not `python -m unittest`) — pytest runs unittest-style tests natively and gives better output.
- **Commit policy (project standing rule):** never run `git commit` without first showing the diff and getting explicit user confirmation. The "Commit" step in each task below describes *what* to commit — the executor must pause and ask before actually running it, every time.
- GPUs may be occupied by a live training run (`nvidia-smi` to check). Any step that needs a real GPU + real checkpoint + real COCO images is marked "(manual, GPU-dependent)" — do not block the rest of the plan on it; run it later when a GPU is free.
- Do not modify `evaluate_caption` or `itm_eval` in `eval_validation_tool.py` — out of scope, left as-is.

---

### Task 1: Move `load_model_weights_only` into `utils.py`

**Files:**
- Modify: `utils.py` (insert new function after `is_main_process`, around line 247)
- Modify: `eval_official_pretrain_val_loss.py:30-50` (remove local definition), `eval_official_pretrain_val_loss.py:98` (call site)
- Create: `tests/__init__.py` (empty)
- Test: `tests/test_utils_load_model_weights_only.py`

**Interfaces:**
- Produces: `utils.load_model_weights_only(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module` — loads a checkpoint (raw state_dict or `{"model": state_dict, ...}` dict), strips a `"module."` DDP prefix from keys if present, loads with `strict=False`, prints the load message on the main process, returns the model with weights loaded in-place.

- [ ] **Step 1: Write the failing test**

Create `tests/__init__.py`:

```python
```

(empty file, makes `tests` an importable package)

Create `tests/test_utils_load_model_weights_only.py`:

```python
import os
import tempfile
import unittest

import torch
import torch.nn as nn

import utils


class LoadModelWeightsOnlyTest(unittest.TestCase):
    def test_strips_module_prefix_and_loads_weights(self):
        source = nn.Linear(2, 2)
        with torch.no_grad():
            source.weight.fill_(3.0)
            source.bias.fill_(1.5)

        ddp_style_state_dict = {f"module.{k}": v for k, v in source.state_dict().items()}
        checkpoint = {"model": ddp_style_state_dict, "epoch": 7}

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "checkpoint.pth")
            torch.save(checkpoint, ckpt_path)

            target = nn.Linear(2, 2)
            utils.load_model_weights_only(target, ckpt_path)

        self.assertTrue(torch.equal(target.weight, source.weight))
        self.assertTrue(torch.equal(target.bias, source.bias))

    def test_accepts_raw_state_dict_without_model_key(self):
        source = nn.Linear(2, 2)
        with torch.no_grad():
            source.weight.fill_(-2.0)
            source.bias.fill_(0.5)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "checkpoint.pth")
            torch.save(source.state_dict(), ckpt_path)

            target = nn.Linear(2, 2)
            utils.load_model_weights_only(target, ckpt_path)

        self.assertTrue(torch.equal(target.weight, source.weight))
        self.assertTrue(torch.equal(target.bias, source.bias))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_utils_load_model_weights_only.py -v`
Expected: FAIL with `AttributeError: module 'utils' has no attribute 'load_model_weights_only'`

- [ ] **Step 3: Add the function to `utils.py`**

In `utils.py`, find this exact block (currently around lines 245-248):

```python
def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
```

Replace it with:

```python
def is_main_process():
    return get_rank() == 0


def load_model_weights_only(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean_state_dict[k] = v

    msg = model.load_state_dict(clean_state_dict, strict=False)

    if is_main_process():
        print(f"Loaded model weights from: {checkpoint_path}")
        print(msg)

    return model


def save_on_master(*args, **kwargs):
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_utils_load_model_weights_only.py -v`
Expected: `OK` (2 tests pass)

- [ ] **Step 5: Update `eval_official_pretrain_val_loss.py` to use the moved function**

In `eval_official_pretrain_val_loss.py`, remove this entire block (currently lines 30-51):

```python
def load_model_weights_only(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean_state_dict[k] = v

    msg = model.load_state_dict(clean_state_dict, strict=False)

    if utils.is_main_process():
        print(f"Loaded model weights from: {checkpoint_path}")
        print(msg)

    return model


def main(args, config):
```

Replace with just:

```python
def main(args, config):
```

Then find the call site (currently line 98):

```python
    model = load_model_weights_only(model, args.checkpoint)
```

Replace with:

```python
    model = utils.load_model_weights_only(model, args.checkpoint)
```

- [ ] **Step 6: Verify syntax and re-run test**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m py_compile utils.py eval_official_pretrain_val_loss.py && /home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_utils_load_model_weights_only.py -v`
Expected: no syntax errors, `OK` (2 tests pass)

- [ ] **Step 7: Commit (ask user for confirmation first)**

```bash
git add utils.py eval_official_pretrain_val_loss.py tests/__init__.py tests/test_utils_load_model_weights_only.py
git commit -m "$(cat <<'EOF'
refactor: move load_model_weights_only into utils.py for reuse

Pulls the checkpoint-loading helper out of eval_official_pretrain_val_loss.py
so the new standalone retrieval-eval script can reuse it instead of
duplicating it.
EOF
)"
```

---

### Task 2: Split `evaluate_retrieval` into ITC-only and ITM-rerank tiers

**Files:**
- Modify: `eval_validation_tool.py` (replace the single `evaluate_retrieval` function with a shared helper + two public functions)
- Test: `tests/test_eval_validation_tool_retrieval.py`

**Interfaces:**
- Consumes: nothing new from Task 1.
- Produces:
  - `eval_validation_tool.evaluate_retrieval_itc(model, data_loader, device, config) -> (scores_i2t: np.ndarray, scores_t2i: np.ndarray)` — cosine-similarity-only score matrices, no ITM head calls.
  - `eval_validation_tool.evaluate_retrieval_itm(model, data_loader, device, config) -> (scores_i2t: np.ndarray, scores_t2i: np.ndarray)` — same behavior as the old `evaluate_retrieval` (ITC top-k + ITM head rerank), just renamed.
  - `eval_validation_tool.itm_eval(...)` — unchanged, both tiers feed it.
  - Task 3's Runner will call `evaluate_retrieval_itc` / `evaluate_retrieval_itm` by these exact names.

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_validation_tool_retrieval.py`:

```python
import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import eval_validation_tool as evt


class FakeTokenizerOutput:
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def to(self, device):
        return self


class FakeTokenizer:
    enc_token_id = 999

    def __call__(self, texts, padding="max_length", truncation=True, max_length=35, return_tensors="pt"):
        ids = torch.zeros(len(texts), max_length, dtype=torch.long)
        for i, t in enumerate(texts):
            ids[i, 0] = int(t.split("_")[-1])
        mask = torch.ones(len(texts), max_length, dtype=torch.long)
        return FakeTokenizerOutput(ids, mask)


class FakeTextEncoderOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class FakeTextEncoder(nn.Module):
    """Embeds token-id-at-position-0 as a one-hot vector. enc_token_id (999)
    is out of range so rerank calls (which overwrite position 0 with
    enc_token_id) deterministically produce an all-zero hidden state."""

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, input_ids, attention_mask=None, mode=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                return_dict=True):
        batch, seq_len = input_ids.shape
        hidden = torch.zeros(batch, seq_len, self.embed_dim)
        for b in range(batch):
            tok = int(input_ids[b, 0].item())
            if 0 <= tok < self.embed_dim:
                hidden[b, 0, tok] = 1.0
        return FakeTextEncoderOutput(hidden)


class FakeVisualEncoder(nn.Module):
    def forward(self, image):
        return image.unsqueeze(1)  # [B, embed_dim] -> [B, 1, embed_dim]


class CountingLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_count = 0
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        self.call_count += 1
        return super().forward(x)


class FakeRetrievalModel(nn.Module):
    def __init__(self, embed_dim=4):
        super().__init__()
        self.tokenizer = FakeTokenizer()
        self.text_encoder = FakeTextEncoder(embed_dim)
        self.text_proj = nn.Identity()
        self.visual_encoder = FakeVisualEncoder()
        self.vision_proj = nn.Identity()
        self.itm_head = CountingLinear(embed_dim, 2)


class FakeRetrievalDataset(Dataset):
    """n images, n captions, 1:1 matched by index — image i's embedding and
    caption i's embedding are constructed to be the same one-hot vector, so
    a correct implementation always ranks the true match first."""

    def __init__(self, n=4):
        self.text = [f"item_{i}" for i in range(n)]
        self.image = list(range(n))
        self.txt2img = {i: i for i in range(n)}
        self.img2txt = {i: [i] for i in range(n)}
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        vec = torch.zeros(self.n)
        vec[index] = 1.0
        return vec, index


class EvaluateRetrievalSplitTest(unittest.TestCase):
    def setUp(self):
        self.n = 4
        self.model = FakeRetrievalModel(embed_dim=self.n)
        self.dataset = FakeRetrievalDataset(n=self.n)
        self.loader = DataLoader(self.dataset, batch_size=2, shuffle=False)
        self.device = torch.device("cpu")
        self.config = {"k_test": 2}

    def test_itc_only_skips_itm_head_and_gets_perfect_recall(self):
        scores_i2t, scores_t2i = evt.evaluate_retrieval_itc(
            self.model, self.loader, self.device, self.config)

        self.assertEqual(self.model.itm_head.call_count, 0)
        self.assertEqual(scores_i2t.shape, (self.n, self.n))
        self.assertEqual(scores_t2i.shape, (self.n, self.n))

        metrics = evt.itm_eval(scores_i2t, scores_t2i,
                                self.dataset.txt2img, self.dataset.img2txt)
        self.assertEqual(metrics["r_mean"], 100.0)

    def test_itm_rerank_invokes_itm_head_and_gets_perfect_recall(self):
        scores_i2t, scores_t2i = evt.evaluate_retrieval_itm(
            self.model, self.loader, self.device, self.config)

        self.assertGreater(self.model.itm_head.call_count, 0)
        self.assertEqual(scores_i2t.shape, (self.n, self.n))
        self.assertEqual(scores_t2i.shape, (self.n, self.n))

        metrics = evt.itm_eval(scores_i2t, scores_t2i,
                                self.dataset.txt2img, self.dataset.img2txt)
        self.assertEqual(metrics["r_mean"], 100.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_eval_validation_tool_retrieval.py -v`
Expected: FAIL with `AttributeError: module 'eval_validation_tool' has no attribute 'evaluate_retrieval_itc'`

- [ ] **Step 3: Add `torch.distributed` to the module-level imports**

In `eval_validation_tool.py`, find:

```python
import torch
import torch.nn.functional as F
import numpy as np
import time
import datetime
import utils
```

Replace with:

```python
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import time
import datetime
import utils
```

- [ ] **Step 4: Replace `evaluate_retrieval` with the split implementation**

In `eval_validation_tool.py`, find the entire existing function (the whole block from `@torch.no_grad() # 근데 얘내들은...` through the line `return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()` right before the `# itm은 뭐지?` comment):

```python
@torch.no_grad() # 근데 얘내들은 모멘텀 인코더까지 끌어와야하지않나 아닌가 모멘텀 큐가 저장되어있는게 트레이닝이고 벨리데이션에선 그냥 선택만 하면 되나
def evaluate_retrieval(model, data_loader, device, config):
    print('\nComputing features for Retrieval evaluation...')
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Retrieval Evaluation:'    
    start_time = time.time()  

    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []  
    text_atts = []
    
    # Text 특징 추출
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_embed = model.text_proj(text_output.last_hidden_state[:,0,:]) 
        text_embed = F.normalize(text_embed, dim=-1) 
        text_embeds.append(text_embed)   
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    text_ids[:,0] = model.tokenizer.enc_token_id 
    
    image_feats = []
    image_embeds = []
    
    # Image 특징 추출
    for image, img_id in data_loader: 
        image = image.to(device) 
        image_feat = model.visual_encoder(image)   
        image_embed = model.vision_proj(image_feat[:,0,:])            
        image_embed = F.normalize(image_embed, dim=-1)      
        
        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)
     
    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)
    
    # 코사인 유사도 행렬 계산
    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full((len(data_loader.dataset.image), len(texts)), -100.0).to(device)
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0) // num_tasks + 1 
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    # I2T 평가
    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        topk_sim, topk_idx = sims.topk(k=config.get('k_test', 128), dim=0)
        encoder_output = image_feats[start+i].repeat(config.get('k_test', 128), 1, 1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[topk_idx], 
                                    attention_mask = text_atts[topk_idx],
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True)
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_i2t[start+i, topk_idx] = score + topk_sim
        
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)
    
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)    
    
    # T2I 평가
    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        topk_sim, topk_idx = sims.topk(k=config.get('k_test', 128), dim=0)
        encoder_output = image_feats[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[start+i].repeat(config.get('k_test', 128), 1), 
                                    attention_mask = text_atts[start+i].repeat(config.get('k_test', 128), 1),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True)
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2i[start+i, topk_idx] = score + topk_sim

    if utils.is_dist_avail_and_initialized():
        import torch.distributed as dist
        dist.barrier()   
        dist.all_reduce(score_matrix_i2t, op=dist.ReduceOp.SUM) 
        dist.all_reduce(score_matrix_t2i, op=dist.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Retrieval Evaluation time {}'.format(total_time_str)) 

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
```

Replace it with:

```python
def _encode_retrieval_features(model, data_loader, device, need_patch_feats):
    """Shared by both retrieval tiers: extracts text/image embeddings and the
    full cosine-similarity matrix. Image patch features (needed only for ITM
    cross-attention rerank) are skipped entirely when need_patch_feats=False,
    which is what makes the ITC-only tier cheap."""
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []

    # Text 특징 추출
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')
        text_embed = model.text_proj(text_output.last_hidden_state[:,0,:])
        text_embed = F.normalize(text_embed, dim=-1)
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    text_ids[:,0] = model.tokenizer.enc_token_id

    image_feats = [] if need_patch_feats else None
    image_embeds = []

    # Image 특징 추출
    for image, img_id in data_loader:
        image = image.to(device)
        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat[:,0,:])
        image_embed = F.normalize(image_embed, dim=-1)

        if need_patch_feats:
            image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)

    if need_patch_feats:
        image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    # 코사인 유사도 행렬 계산
    sims_matrix = image_embeds @ text_embeds.t()

    return sims_matrix, image_feats, text_ids, text_atts


@torch.no_grad()
def evaluate_retrieval_itc(model, data_loader, device, config):
    """ITC-only tier: cosine-similarity ranking, no ITM head calls. Cheap
    enough to run frequently during pretraining (~10-30s for full COCO val)."""
    print('\nComputing features for ITC-only Retrieval evaluation...')
    start_time = time.time()

    sims_matrix, _, _, _ = _encode_retrieval_features(
        model, data_loader, device, need_patch_feats=False)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('ITC-only Retrieval Evaluation time {}'.format(total_time_str))

    score_matrix_i2t = sims_matrix
    score_matrix_t2i = sims_matrix.t()

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def evaluate_retrieval_itm(model, data_loader, device, config):
    """ITM-rerank tier: ITC top-k candidates re-scored with the cross-attention
    ITM head. This is the BLIP-paper-style R@1/5/10 definition, but much more
    expensive than evaluate_retrieval_itc (~2-5min for full COCO val)."""
    print('\nComputing features for Retrieval evaluation...')
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Retrieval Evaluation:'
    start_time = time.time()

    sims_matrix, image_feats, text_ids, text_atts = _encode_retrieval_features(
        model, data_loader, device, need_patch_feats=True)

    texts = data_loader.dataset.text
    score_matrix_i2t = torch.full((len(data_loader.dataset.image), len(texts)), -100.0).to(device)

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    # I2T 평가
    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config.get('k_test', 128), dim=0)
        encoder_output = image_feats[start+i].repeat(config.get('k_test', 128), 1, 1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[topk_idx],
                                    attention_mask = text_atts[topk_idx],
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,
                                    return_dict = True)
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_i2t[start+i, topk_idx] = score + topk_sim

    sims_matrix_t = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)

    step = sims_matrix_t.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix_t.size(0), start + step)

    # T2I 평가
    for i, sims in enumerate(metric_logger.log_every(sims_matrix_t[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config.get('k_test', 128), dim=0)
        encoder_output = image_feats[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[start+i].repeat(config.get('k_test', 128), 1),
                                    attention_mask = text_atts[start+i].repeat(config.get('k_test', 128), 1),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,
                                    return_dict = True)
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2i[start+i, topk_idx] = score + topk_sim

    if utils.is_dist_avail_and_initialized():
        dist.barrier()
        dist.all_reduce(score_matrix_i2t, op=dist.ReduceOp.SUM)
        dist.all_reduce(score_matrix_t2i, op=dist.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Retrieval Evaluation time {}'.format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_eval_validation_tool_retrieval.py -v`
Expected: `OK` (2 tests pass)

- [ ] **Step 6: Syntax check the whole file**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m py_compile eval_validation_tool.py`
Expected: no output, exit code 0

- [ ] **Step 7: Commit (ask user for confirmation first)**

```bash
git add eval_validation_tool.py tests/test_eval_validation_tool_retrieval.py
git commit -m "$(cat <<'EOF'
refactor: split evaluate_retrieval into ITC-only and ITM-rerank tiers

evaluate_retrieval_itc computes cosine-similarity scores only (no ITM head,
no patch-feature CPU offload), cheap enough to run frequently during
pretraining. evaluate_retrieval_itm keeps the original ITC-top-k + ITM-head
rerank behavior under a new name. Both share _encode_retrieval_features and
feed the unchanged itm_eval.
EOF
)"
```

---

### Task 3: `RetrievalValRunner` orchestration (`data/eval_validation_retrieval.py`)

**Files:**
- Create: `data/eval_validation_retrieval.py`
- Test: `tests/test_eval_validation_retrieval_runner.py`

**Interfaces:**
- Consumes: `eval_validation_tool.evaluate_retrieval_itc`, `eval_validation_tool.evaluate_retrieval_itm`, `eval_validation_tool.itm_eval` (Task 2). `data.coco_karpathy_dataset.coco_karpathy_retrieval_eval` and `data.eval_validation_loss.build_val_loss_transform` (existing, unmodified).
- Produces:
  - `eval_validation_retrieval.build_coco_karpathy_retrieval_val_loader(config) -> torch.utils.data.DataLoader`
  - `eval_validation_retrieval.RetrievalValRunner(data_loader, device, config, writer=None)` with methods:
    - `val_retrieval_during_train(model_without_ddp, epoch, iteration, global_step) -> dict`
    - `run_epoch_end(model_without_ddp, epoch, global_step) -> dict`
  - `eval_validation_retrieval.build_pretrain_retrieval_val_runner(config, device, writer=None) -> RetrievalValRunner | None` — Task 5 (`pretrain.py`) and Task 6 (standalone script) both call this exact factory.

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_validation_retrieval_runner.py`:

```python
import unittest
from unittest import mock

import torch.nn as nn

import data.eval_validation_retrieval as evr


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)


class RetrievalValRunnerTest(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.model.train()

        fake_dataset = mock.Mock()
        fake_dataset.txt2img = {0: 0}
        fake_dataset.img2txt = {0: [0]}

        self.fake_loader = mock.Mock()
        self.fake_loader.dataset = fake_dataset

        self.config = {
            "val_retrieval_itc_interval_steps": 1000,
            "val_retrieval_itm_interval_steps": 10000,
            "val_retrieval_itc_epoch_end": True,
            "val_retrieval_itm_epoch_end": True,
        }

        self.runner = evr.RetrievalValRunner(
            data_loader=self.fake_loader,
            device="cpu",
            config=self.config,
            writer=None,
        )

    @mock.patch("data.eval_validation_retrieval.eval_validation_tool.itm_eval")
    @mock.patch("data.eval_validation_retrieval.eval_validation_tool.evaluate_retrieval_itm")
    @mock.patch("data.eval_validation_retrieval.eval_validation_tool.evaluate_retrieval_itc")
    def test_step_interval_gating(self, mock_itc, mock_itm, mock_eval):
        mock_itc.return_value = (None, None)
        mock_itm.return_value = (None, None)
        mock_eval.return_value = {"r_mean": 0.0}

        self.runner.val_retrieval_during_train(self.model, epoch=0, iteration=999, global_step=1000)
        self.assertEqual(mock_itc.call_count, 1)
        self.assertEqual(mock_itm.call_count, 0)

        self.runner.val_retrieval_during_train(self.model, epoch=0, iteration=9999, global_step=10000)
        self.assertEqual(mock_itc.call_count, 2)
        self.assertEqual(mock_itm.call_count, 1)

        self.runner.val_retrieval_during_train(self.model, epoch=0, iteration=500, global_step=1500)
        self.assertEqual(mock_itc.call_count, 2)
        self.assertEqual(mock_itm.call_count, 1)

        self.runner.val_retrieval_during_train(self.model, epoch=0, iteration=0, global_step=0)
        self.assertEqual(mock_itc.call_count, 2)
        self.assertEqual(mock_itm.call_count, 1)

    @mock.patch("data.eval_validation_retrieval.eval_validation_tool.itm_eval")
    @mock.patch("data.eval_validation_retrieval.eval_validation_tool.evaluate_retrieval_itm")
    @mock.patch("data.eval_validation_retrieval.eval_validation_tool.evaluate_retrieval_itc")
    def test_epoch_end_runs_both_tiers_and_restores_train_mode(self, mock_itc, mock_itm, mock_eval):
        mock_itc.return_value = (None, None)
        mock_itm.return_value = (None, None)
        mock_eval.return_value = {"r_mean": 0.0}

        self.model.train()
        self.runner.run_epoch_end(self.model, epoch=0, global_step=37300)

        self.assertEqual(mock_itc.call_count, 1)
        self.assertEqual(mock_itm.call_count, 1)
        self.assertTrue(self.model.training)

    @mock.patch("data.eval_validation_retrieval.eval_validation_tool.itm_eval")
    @mock.patch("data.eval_validation_retrieval.eval_validation_tool.evaluate_retrieval_itm")
    @mock.patch("data.eval_validation_retrieval.eval_validation_tool.evaluate_retrieval_itc")
    def test_epoch_end_respects_disabled_flags(self, mock_itc, mock_itm, mock_eval):
        mock_itc.return_value = (None, None)
        mock_itm.return_value = (None, None)
        mock_eval.return_value = {"r_mean": 0.0}

        self.config["val_retrieval_itm_epoch_end"] = False
        self.runner.run_epoch_end(self.model, epoch=0, global_step=37300)

        self.assertEqual(mock_itc.call_count, 1)
        self.assertEqual(mock_itm.call_count, 0)


class BuildRunnerFactoryTest(unittest.TestCase):
    def test_returns_none_when_disabled(self):
        runner = evr.build_pretrain_retrieval_val_runner(
            config={"val_retrieval_enabled": False}, device="cpu", writer=None)
        self.assertIsNone(runner)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_eval_validation_retrieval_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.eval_validation_retrieval'`

- [ ] **Step 3: Create `data/eval_validation_retrieval.py`**

```python
import torch
from torch.utils.data import DataLoader

import utils
import eval_validation_tool
from data.coco_karpathy_dataset import coco_karpathy_retrieval_eval
from data.eval_validation_loss import build_val_loss_transform


def build_coco_karpathy_retrieval_val_loader(config):
    ann_root = config["val_retrieval_ann_root"]
    image_root = config.get("val_retrieval_image_root", config["image_root_coco"])
    split = config.get("val_retrieval_split", "val")

    image_size = config["image_size"]
    batch_size = config.get("val_retrieval_batch_size", config["batch_size"])
    num_workers = config.get("val_retrieval_num_workers", 4)

    transform = build_val_loss_transform(image_size)

    dataset = coco_karpathy_retrieval_eval(
        transform=transform,
        image_root=image_root,
        ann_root=ann_root,
        split=split,
    )

    # 모든 rank가 전체 데이터셋을 로드 (DistributedSampler 없음) - evaluate_retrieval_itm이
    # 내부적으로 rank별 샤딩을 직접 수행하기 때문에 train_retrieval.py의 val/test loader와
    # 동일한 패턴을 따른다.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return loader


class RetrievalValRunner:
    def __init__(self, data_loader, device, config, writer=None):
        self.data_loader = data_loader
        self.device = torch.device(device)
        self.config = config
        self.writer = writer

    def val_retrieval_during_train(self, model_without_ddp, epoch, iteration, global_step):
        if global_step <= 0:
            return {}

        results = {}

        itc_interval = self.config.get("val_retrieval_itc_interval_steps", 0)
        if itc_interval and itc_interval > 0 and global_step % itc_interval == 0:
            results["itc"] = self._run_tier(
                model_without_ddp, "itc", global_step,
                header=f"Val Retrieval ITC: [epoch {epoch} | step {iteration} | global {global_step}]",
            )

        itm_interval = self.config.get("val_retrieval_itm_interval_steps", 0)
        if itm_interval and itm_interval > 0 and global_step % itm_interval == 0:
            results["itm"] = self._run_tier(
                model_without_ddp, "itm", global_step,
                header=f"Val Retrieval ITM: [epoch {epoch} | step {iteration} | global {global_step}]",
            )

        return results

    def run_epoch_end(self, model_without_ddp, epoch, global_step):
        results = {}

        if self.config.get("val_retrieval_itc_epoch_end", True):
            results["itc"] = self._run_tier(
                model_without_ddp, "itc", global_step,
                header=f"Val Retrieval ITC Epoch: [{epoch}]",
            )

        if self.config.get("val_retrieval_itm_epoch_end", True):
            results["itm"] = self._run_tier(
                model_without_ddp, "itm", global_step,
                header=f"Val Retrieval ITM Epoch: [{epoch}]",
            )

        return results

    def _run_tier(self, model_without_ddp, tier, global_step, header):
        was_training = model_without_ddp.training
        model_without_ddp.eval()

        try:
            if tier == "itc":
                scores_i2t, scores_t2i = eval_validation_tool.evaluate_retrieval_itc(
                    model_without_ddp, self.data_loader, self.device, self.config,
                )
            elif tier == "itm":
                scores_i2t, scores_t2i = eval_validation_tool.evaluate_retrieval_itm(
                    model_without_ddp, self.data_loader, self.device, self.config,
                )
            else:
                raise ValueError(f"Unknown retrieval tier: {tier}")

            metrics = eval_validation_tool.itm_eval(
                scores_i2t, scores_t2i,
                self.data_loader.dataset.txt2img,
                self.data_loader.dataset.img2txt,
            )

            if utils.is_main_process():
                print(f"{header} r_mean={metrics['r_mean']:.2f}")

            self._write_tensorboard(tier, metrics, global_step)

            return metrics
        finally:
            if was_training:
                model_without_ddp.train()
            else:
                model_without_ddp.eval()

    def _write_tensorboard(self, tier, metrics, global_step):
        if self.writer is None or not utils.is_main_process():
            return

        for key, value in metrics.items():
            self.writer.add_scalar(f"val_retrieval_{tier}/{key}", value, global_step)


def build_pretrain_retrieval_val_runner(config, device, writer=None):
    if not config.get("val_retrieval_enabled", False):
        return None

    data_loader = build_coco_karpathy_retrieval_val_loader(config)

    return RetrievalValRunner(
        data_loader=data_loader,
        device=device,
        config=config,
        writer=writer,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_eval_validation_retrieval_runner.py -v`
Expected: `OK` (4 tests pass)

- [ ] **Step 5: Syntax check**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m py_compile data/eval_validation_retrieval.py`
Expected: no output, exit code 0

- [ ] **Step 6: Commit (ask user for confirmation first)**

```bash
git add data/eval_validation_retrieval.py tests/test_eval_validation_retrieval_runner.py
git commit -m "$(cat <<'EOF'
feat: add RetrievalValRunner for COCO retrieval validation

Mirrors the PretrainValLossRunner pattern: builds a non-distributed-sampler
COCO Karpathy retrieval val/test loader (reusing coco_karpathy_retrieval_eval
and build_val_loss_transform as-is) and gates the ITC-only / ITM-rerank
tiers on independent step intervals plus epoch-end flags.
EOF
)"
```

---

### Task 4: Add `val_retrieval_*` config keys to `configs/pretrain.yaml`

**Files:**
- Modify: `configs/pretrain.yaml`
- Test: `tests/test_pretrain_yaml_retrieval_keys.py`

**Interfaces:**
- Consumes: nothing code-level — this is config data consumed by `eval_validation_retrieval.build_pretrain_retrieval_val_runner` (Task 3) via `config.get(...)`.
- Produces: the keys `val_retrieval_enabled`, `val_retrieval_ann_root`, `val_retrieval_image_root`, `val_retrieval_split`, `val_retrieval_batch_size`, `val_retrieval_num_workers`, `k_test`, `val_retrieval_itc_interval_steps`, `val_retrieval_itc_epoch_end`, `val_retrieval_itm_interval_steps`, `val_retrieval_itm_epoch_end` in the loaded yaml dict — Task 5 (`pretrain.py`) relies on these being present.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pretrain_yaml_retrieval_keys.py`:

```python
import unittest

import yaml


class PretrainYamlRetrievalKeysTest(unittest.TestCase):
    def setUp(self):
        with open("configs/pretrain.yaml", "r") as f:
            self.config = yaml.safe_load(f)

    def test_has_all_val_retrieval_keys(self):
        expected_keys = [
            "val_retrieval_enabled",
            "val_retrieval_ann_root",
            "val_retrieval_image_root",
            "val_retrieval_split",
            "val_retrieval_batch_size",
            "val_retrieval_num_workers",
            "k_test",
            "val_retrieval_itc_interval_steps",
            "val_retrieval_itc_epoch_end",
            "val_retrieval_itm_interval_steps",
            "val_retrieval_itm_epoch_end",
        ]
        for key in expected_keys:
            self.assertIn(key, self.config, f"missing config key: {key}")

    def test_itm_interval_is_coarser_than_itc_interval(self):
        self.assertGreater(
            self.config["val_retrieval_itm_interval_steps"],
            self.config["val_retrieval_itc_interval_steps"],
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_pretrain_yaml_retrieval_keys.py -v`
Expected: FAIL (`missing config key: val_retrieval_enabled`)

- [ ] **Step 3: Add the keys to `configs/pretrain.yaml`**

Find this block (the end of the existing `val_loss_*` section):

```yaml
val_loss_alpha_mode: current
val_loss_restore_rng: true
val_loss_amp: true
#### 수정부분 끝 ####
###
```

Replace with:

```yaml
val_loss_alpha_mode: current
val_loss_restore_rng: true
val_loss_amp: true
#### 수정부분 끝 ####
###

#### 수정부분 시작: COCO Karpathy retrieval validation (momentum 없는 student-only R@1/5/10) ####
# ITC-only: 코사인 유사도만, 가벼움 (~10-30초) -> 자주 체크
# ITM rerank: ITC top-k를 ITM head로 재채점, 무거움 (~2-5분) -> 1시간에 한 번 꼴 (1 epoch ~4시간 실측 기준)
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
#### 수정부분 끝 ####
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_pretrain_yaml_retrieval_keys.py -v`
Expected: `OK` (2 tests pass)

- [ ] **Step 5: Commit (ask user for confirmation first)**

```bash
git add configs/pretrain.yaml tests/test_pretrain_yaml_retrieval_keys.py
git commit -m "$(cat <<'EOF'
config: add val_retrieval_* keys for COCO retrieval validation

ITC-only checks every 1000 steps, ITM-rerank every 10000 steps (~hourly at
the measured ~4h/epoch), both also firing at epoch end.
EOF
)"
```

---

### Task 5: Wire `RetrievalValRunner` into `pretrain.py`

**Files:**
- Modify: `pretrain.py` (import, `train()` signature, two call sites in `train()`/`main()`, runner construction in `main()`)
- Test: `tests/test_pretrain_retrieval_wiring.py`

**Interfaces:**
- Consumes: `eval_validation_retrieval.build_pretrain_retrieval_val_runner` and `RetrievalValRunner.val_retrieval_during_train` / `run_epoch_end` (Task 3).
- Produces: `train(model, data_loader, optimizer, epoch, device, config, writer=None, val_loss_runner=None, collapse_counter=None, model_without_ddp=None, retrieval_val_runner=None)` — the two new trailing kwargs are additive (default `None`), existing callers (none outside this file) are unaffected.

- [ ] **Step 1: Write the failing test**

This test checks the wiring statically (signature + source text) rather than running real training, since `train()`'s body needs a real BLIP model, real COCO/VG data, and (for the distributed branches) multiple GPUs — none of which are appropriate for an automated unit test.

Create `tests/test_pretrain_retrieval_wiring.py`:

```python
import inspect
import unittest

import pretrain


class PretrainRetrievalWiringTest(unittest.TestCase):
    def test_train_accepts_model_without_ddp_and_retrieval_val_runner(self):
        sig = inspect.signature(pretrain.train)
        self.assertIn("model_without_ddp", sig.parameters)
        self.assertIn("retrieval_val_runner", sig.parameters)
        self.assertIsNone(sig.parameters["model_without_ddp"].default)
        self.assertIsNone(sig.parameters["retrieval_val_runner"].default)

    def test_train_body_calls_val_retrieval_during_train(self):
        source = inspect.getsource(pretrain.train)
        self.assertIn("retrieval_val_runner.val_retrieval_during_train(", source)

    def test_main_body_builds_and_calls_runner(self):
        source = inspect.getsource(pretrain.main)
        self.assertIn("eval_validation_retrieval.build_pretrain_retrieval_val_runner(", source)
        self.assertIn("retrieval_val_runner.run_epoch_end(", source)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_pretrain_retrieval_wiring.py -v`
Expected: FAIL (`AssertionError` — `train()` does not yet have `model_without_ddp`/`retrieval_val_runner` params)

Note: `import pretrain` at module load time runs `utils.init_distributed_mode`-free top-level code only (the heavy work is inside `main()`/`if __name__ == "__main__"`), so importing it for introspection is safe and does not require GPUs or launch training.

- [ ] **Step 3: Add the import**

In `pretrain.py`, find:

```python
#### 수정부분 시작: validation loss 모듈 import ####
from data import eval_validation_loss
#### 수정부분 끝 ####
```

Replace with:

```python
#### 수정부분 시작: validation loss 모듈 import ####
from data import eval_validation_loss
#### 수정부분 끝 ####
#### 수정부분 시작: retrieval validation 모듈 import ####
from data import eval_validation_retrieval
#### 수정부분 끝 ####
```

- [ ] **Step 4: Extend `train()`'s signature**

Find:

```python
def train(model, data_loader, optimizer, epoch, device, config, writer=None, val_loss_runner=None, collapse_counter=None): # writer추가 val loss runner 추가, collapse_counter추가
```

Replace with:

```python
def train(model, data_loader, optimizer, epoch, device, config, writer=None, val_loss_runner=None, collapse_counter=None, model_without_ddp=None, retrieval_val_runner=None): # writer추가 val loss runner 추가, collapse_counter추가, retrieval val runner 추가
```

- [ ] **Step 5: Add the in-train call site**

Find:

```python
        #### 수정부분 시작: train 도중 validation loss optional 실행 ####
        if val_loss_runner is not None:
            val_loss_runner.val_loss_during_train(
                model=model,
                epoch=epoch,
                iteration=i,
                global_step=global_step,
                train_loader_len=len(data_loader),
            )
        #### 수정부분 끝 ####
```

Replace with:

```python
        #### 수정부분 시작: train 도중 validation loss optional 실행 ####
        if val_loss_runner is not None:
            val_loss_runner.val_loss_during_train(
                model=model,
                epoch=epoch,
                iteration=i,
                global_step=global_step,
                train_loader_len=len(data_loader),
            )
        #### 수정부분 끝 ####

        #### 수정부분 시작: train 도중 retrieval validation optional 실행 (ITC는 자주, ITM rerank는 드물게) ####
        if retrieval_val_runner is not None:
            retrieval_val_runner.val_retrieval_during_train(
                model_without_ddp=model_without_ddp,
                epoch=epoch,
                iteration=i,
                global_step=global_step,
            )
        #### 수정부분 끝 ####
```

- [ ] **Step 6: Build the runner in `main()`**

Find:

```python
    val_loss_runner = eval_validation_loss.build_pretrain_val_loss_runner( # 러너 생성
        config=config,
        device=device,
        writer=writer,
    )
    #### 수정부분 끝 ####
```

Replace with:

```python
    val_loss_runner = eval_validation_loss.build_pretrain_val_loss_runner( # 러너 생성
        config=config,
        device=device,
        writer=writer,
    )
    #### 수정부분 끝 ####

    #### 수정부분 시작: retrieval validation runner 생성 (momentum 없이 student 인코더만으로 COCO val retrieval 평가) ####
    retrieval_val_runner = eval_validation_retrieval.build_pretrain_retrieval_val_runner(
        config=config,
        device=device,
        writer=writer,
    )
    #### 수정부분 끝 ####
```

- [ ] **Step 7: Pass the new args into the `train()` call site**

Find:

```python
            train_stats = train(model, data_loader, optimizer, epoch, device, config, writer, val_loss_runner=val_loss_runner, collapse_counter=collapse_counter) # writer추가, collapse_counter추가
```

Replace with:

```python
            train_stats = train(model, data_loader, optimizer, epoch, device, config, writer, val_loss_runner=val_loss_runner, collapse_counter=collapse_counter, model_without_ddp=model_without_ddp, retrieval_val_runner=retrieval_val_runner) # writer추가, collapse_counter추가, retrieval val runner 추가
```

- [ ] **Step 8: Add the epoch-end call site**

Find:

```python
            #### 수정부분 시작: epoch 종료 validation loss 실행 ####
            val_stats = {}

            if val_loss_runner is not None:
                epoch_end_global_step = (epoch + 1) * len(data_loader)

                val_stats = val_loss_runner.run_epoch_end(
                    model=model,
                    epoch=epoch,
                    global_step=epoch_end_global_step,
                    train_loader_len=len(data_loader),
                )
            #### 수정부분 끝 ####
```

Replace with:

```python
            #### 수정부분 시작: epoch 종료 validation loss 실행 ####
            val_stats = {}

            if val_loss_runner is not None:
                epoch_end_global_step = (epoch + 1) * len(data_loader)

                val_stats = val_loss_runner.run_epoch_end(
                    model=model,
                    epoch=epoch,
                    global_step=epoch_end_global_step,
                    train_loader_len=len(data_loader),
                )
            #### 수정부분 끝 ####

            #### 수정부분 시작: epoch 종료 retrieval validation 실행 ####
            if retrieval_val_runner is not None:
                epoch_end_global_step = (epoch + 1) * len(data_loader)

                retrieval_val_runner.run_epoch_end(
                    model_without_ddp=model_without_ddp,
                    epoch=epoch,
                    global_step=epoch_end_global_step,
                )
            #### 수정부분 끝 ####
```

- [ ] **Step 9: Run test to verify it passes**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_pretrain_retrieval_wiring.py -v`
Expected: `OK` (3 tests pass)

- [ ] **Step 10: Syntax check**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m py_compile pretrain.py`
Expected: no output, exit code 0

- [ ] **Step 11: (manual, GPU-dependent) Short dry-run once a GPU is free**

This is a real-training smoke test, not part of the automated suite — run it only when `nvidia-smi` shows a free GPU (the repo currently has a live training job using all 4 GPUs; check before running). It is not required to consider this task done; note it as outstanding if no GPU is free yet.

```bash
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
```

If a GPU is free, do a short single-process dry run and confirm the new "Val Retrieval ITC ..." / "Val Retrieval ITM ..." print lines appear and that TensorBoard gets `val_retrieval_itc/*` and `val_retrieval_itm/*` scalars, then kill it (`Ctrl+C`) — a full epoch is not needed, just enough steps to cross `val_retrieval_itc_interval_steps`:

```bash
/home/minwoo/miniconda3/envs/kd_r4/bin/python pretrain.py --config configs/pretrain.yaml --distributed False --device cuda
```

- [ ] **Step 12: Commit (ask user for confirmation first)**

```bash
git add pretrain.py tests/test_pretrain_retrieval_wiring.py
git commit -m "$(cat <<'EOF'
feat: wire RetrievalValRunner into pretrain.py

Builds retrieval_val_runner next to the existing val_loss_runner, threads
model_without_ddp through train() (retrieval eval needs raw model attribute
access, unlike val_loss_runner which only calls model(...) as a forward),
and triggers it both on the step interval and at epoch end.
EOF
)"
```

---

### Task 6: Standalone checkpoint retrieval-eval script (`eval_pretrain_retrieval.py`)

**Files:**
- Create: `eval_pretrain_retrieval.py`
- Test: `tests/test_eval_pretrain_retrieval_script.py`

**Interfaces:**
- Consumes: `utils.load_model_weights_only` (Task 1), `eval_validation_retrieval.build_pretrain_retrieval_val_runner` + `RetrievalValRunner.run_epoch_end` (Task 3), `models.blip_pretrain.blip_pretrain` (existing, unmodified).
- Produces: a CLI script; no other task depends on its internals, only on it existing and parsing args correctly.

- [ ] **Step 1: Write the failing test**

This test only exercises the argument parser (no GPU/model/checkpoint needed) plus a `py_compile`-equivalent import check.

Create `tests/test_eval_pretrain_retrieval_script.py`:

```python
import importlib
import unittest


class EvalPretrainRetrievalScriptTest(unittest.TestCase):
    def test_module_imports_cleanly(self):
        importlib.import_module("eval_pretrain_retrieval")

    def test_parser_requires_checkpoint_and_defaults_split_to_val(self):
        module = importlib.import_module("eval_pretrain_retrieval")
        parser = module.build_arg_parser()

        args = parser.parse_args(["--checkpoint", "/tmp/fake.pth"])
        self.assertEqual(args.checkpoint, "/tmp/fake.pth")
        self.assertEqual(args.split, "val")

        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_parser_accepts_test_split(self):
        module = importlib.import_module("eval_pretrain_retrieval")
        parser = module.build_arg_parser()
        args = parser.parse_args(["--checkpoint", "/tmp/fake.pth", "--split", "test"])
        self.assertEqual(args.split, "test")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_eval_pretrain_retrieval_script.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'eval_pretrain_retrieval'`)

- [ ] **Step 3: Create `eval_pretrain_retrieval.py`**

```python
'''
Evaluate BLIP pretrain retrieval metrics (ITC-only and ITM-rerank) from a
pretrain checkpoint.

This script:
  - loads BLIP pretrain model
  - loads model weights only
  - does not create optimizer
  - does not train
  - runs COCO Karpathy retrieval validation (R@1/R@5/R@10) for both the
    ITC-only and ITM-rerank tiers, on the requested split
'''

import argparse
import os
import json
import datetime
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import yaml

from torch.utils.tensorboard import SummaryWriter

import utils
from models.blip_pretrain import blip_pretrain
from data import eval_validation_retrieval


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/pretrain.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--distributed", default=True, type=bool)
    parser.add_argument("--tensorboard", action="store_true")
    return parser


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)
    cudnn.benchmark = True

    if utils.is_main_process():
        print("Creating pretrain retrieval evaluator")

    writer = None
    if args.tensorboard and utils.is_main_process():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_log_dir = os.path.join(args.output_dir, "tensorboard", f"{timestamp}__pretrain_retrieval")
        writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"[TensorBoard] log_dir: {tb_log_dir}")

    # 단독 체크포인트 평가에서는 항상 두 티어 다 돌리고, split은 CLI 인자를 우선한다.
    config = dict(config)
    config["val_retrieval_enabled"] = True
    config["val_retrieval_split"] = args.split
    config["val_retrieval_itc_epoch_end"] = True
    config["val_retrieval_itm_epoch_end"] = True

    retrieval_val_runner = eval_validation_retrieval.build_pretrain_retrieval_val_runner(
        config=config,
        device=device,
        writer=writer,
    )

    if retrieval_val_runner is None:
        raise RuntimeError("retrieval_val_runner is None. Set val_retrieval_enabled: true in config.")

    if utils.is_main_process():
        print("Creating model")

    model = blip_pretrain(
        image_size=config["image_size"],
        vit=config["vit"],
        vit_grad_ckpt=config["vit_grad_ckpt"],
        vit_ckpt_layer=config["vit_ckpt_layer"],
        queue_size=config["queue_size"],
        my_bert_size=config["my_bert_size"],
    )

    model = model.to(device)

    if not args.checkpoint:
        raise ValueError("--checkpoint is required")

    model = utils.load_model_weights_only(model, args.checkpoint)

    if utils.is_main_process():
        print(f"Start checkpoint retrieval evaluation (split={args.split})")

    global_step = int(config.get("official_eval_global_step", 0))

    val_stats = retrieval_val_runner.run_epoch_end(
        model_without_ddp=model,
        epoch=0,
        global_step=global_step,
    )

    if utils.is_main_process():
        print("Checkpoint retrieval evaluation stats:")
        print(json.dumps(val_stats, indent=2))

        out_path = os.path.join(args.output_dir, f"pretrain_retrieval_{args.split}.json")
        with open(out_path, "w") as f:
            json.dump(val_stats, f, indent=2)
        print(f"Saved stats to: {out_path}")

    if args.distributed:
        dist.barrier()

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = config.get("output_dir", "./output_pretrain_retrieval")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args, config)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m pytest tests/test_eval_pretrain_retrieval_script.py -v`
Expected: `OK` (3 tests pass)

- [ ] **Step 5: Syntax check**

Run: `/home/minwoo/miniconda3/envs/kd_r4/bin/python -m py_compile eval_pretrain_retrieval.py`
Expected: no output, exit code 0

- [ ] **Step 6: (manual, GPU-dependent) Real checkpoint run once a GPU is free**

Not required for this task to be considered done if no GPU is free yet — note as outstanding. An existing checkpoint to use: `output/pt_checkpoint_base_amp/checkpoint_11.pth`.

```bash
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
```

If a GPU is free:

```bash
/home/minwoo/miniconda3/envs/kd_r4/bin/python eval_pretrain_retrieval.py \
  --config configs/pretrain.yaml \
  --checkpoint output/pt_checkpoint_base_amp/checkpoint_11.pth \
  --split val \
  --distributed False \
  --device cuda
```

Expected: prints `r_mean=` lines for both `itc` and `itm` tiers, and writes `output/pt_checkpoint_base_amp/pretrain_retrieval_val.json` (or wherever `output_dir` resolves — check the printed "Saved stats to:" line) containing both tiers' R@1/5/10.

- [ ] **Step 7: Commit (ask user for confirmation first)**

```bash
git add eval_pretrain_retrieval.py tests/test_eval_pretrain_retrieval_script.py
git commit -m "$(cat <<'EOF'
feat: add standalone checkpoint retrieval-eval script

Mirrors eval_official_pretrain_val_loss.py's load-checkpoint -> evaluate ->
save-json pattern (dist.barrier() already guarded by args.distributed from
the start), but for COCO retrieval R@1/5/10 (both ITC-only and ITM-rerank
tiers) instead of validation loss. --split val|test selects the Karpathy
split.
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** Goal 1 (reuse existing retrieval code) → Task 2 keeps `evaluate_retrieval_itm`'s logic byte-for-byte (renamed only) and reuses `itm_eval` unchanged; Task 3 reuses `coco_karpathy_retrieval_eval` and `build_val_loss_transform` unmodified. Goal 2 (standalone checkpoint check) → Task 6. Goal 3 (periodic in-training check) → Task 5. Dual ITC/ITM cadence → Task 3 + Task 4's config values (1000 / 10000 steps, both with epoch-end). Pre-cleanup (dead code removal, `dist.barrier()` fix) was already done earlier in this session, predating this plan.
- **Placeholder scan:** no TBD/TODO; the two GPU-dependent steps (Task 5 Step 11, Task 6 Step 6) are explicitly marked manual/non-blocking rather than faked, per the design doc's own verification plan.
- **Type consistency:** `RetrievalValRunner.val_retrieval_during_train`/`run_epoch_end` signatures introduced in Task 3 are used identically in Task 5 (`pretrain.py`) and Task 6 (standalone script) — `model_without_ddp`, `epoch`, `iteration`, `global_step` keyword names match across all call sites.
