# Research README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the inherited BLIP README with concise English and Korean research READMEs that lead with the compact unified model's performance and center the loss-specific ITC, ITM, and LM analysis.

**Architecture:** `README.md` is the canonical English document and `README.ko.md` mirrors it in Korean with identical section order, tables, numbers, links, and final-result replacement tokens. Both documents derive claims from the approved design and local evaluation artifacts; no scripts, generated assets, dependencies, or additional license files are introduced.

**Tech Stack:** GitHub-flavored Markdown, LaTeX math, local TensorBoard/evaluation artifacts, primary paper/repository links.

**Spec:** `docs/superpowers/specs/2026-08-24-research-readme-design.md`

## Global Constraints

- Audience: vision-language and knowledge-distillation researchers; optimize for first-screen comprehension.
- Language: English is canonical; Korean mirrors every claim and quantitative table.
- Navigation: both files begin with reciprocal `English | 한국어` links.
- Naming: use `Base-capacity model`, `ITC r_mean`, `ITM r_mean`, and `ROUGE-L` consistently.
- Scope: analysis contribution only; omit systems/infrastructure contribution claims and the inherited BLIP usage catalog.
- Evidence: distinguish observations from hypotheses; the vocabulary dark-knowledge explanation remains explicitly labeled **Hypothesis**.
- Comparison: use the internally trained `pt_checkpoint_base_logitscale_nodecay` run, not official BLIP-base/large results, in performance tables.
- Metrics: do not mix COCO validation and test values in one table; final-checkpoint values are primary.
- Math: include exactly one integrated objective equation and no per-loss derivations.
- Final run: retain literal `{{FINAL_*}}` replacement tokens until the unified 20-epoch run and matched evaluation finish.
- Licensing: identify BLIP BSD-3-Clause, MiniLM MIT, DINOv3 License, and dataset terms separately; do not imply that DINOv3 is BSD-licensed.
- Changes: modify `README.md` and create `README.ko.md` only; do not create images, helper scripts, or a new license file.
- Git: do not commit; the user explicitly requested uncommitted implementation.

---

### Task 1: Write the canonical English research README

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: `docs/superpowers/specs/2026-08-24-research-readme-design.md`, local result artifacts listed below, and primary-source URLs in Step 5.
- Produces: the complete canonical English narrative and the exact section/table contract that `README.ko.md` mirrors.

- [ ] **Step 1: Replace the inherited first screen with the approved hero**

Start `README.md` with:

```markdown
[English](README.md) | [한국어](README.ko.md)

# Diagnosing Knowledge Distillation for Compact Vision-Language Pre-training

A 73.8M-parameter BLIP student reaches strong retrieval and captioning performance after loss-specific analysis and redesign of ITC, ITM, and LM distillation.

| Model | Params | ITC r_mean | ITM r_mean | CIDEr | SPICE |
|---|---:|---:|---:|---:|---:|
| Small student baseline | 73.8M | 63.41 | 70.93 | 1.063 | 0.203 |
| **Unified distillation** | **73.8M** | **`{{FINAL_ITC}}`** | **`{{FINAL_ITM}}`** | **`{{FINAL_CIDER}}`** | **`{{FINAL_SPICE}}`** |
| Base-capacity model | 247.2M | 66.20 | 73.22 | 1.095 | 0.208 |
```

Immediately state that rows use matched 20-epoch COCO validation runs and the same caption generation/scoring protocol. Follow with exactly three finding bullets: candidate-structure mismatch broke naive ITC KD; binary ITM soft targets weakened hard-negative discrimination; the loss-specific redesign improved retrieval and all eight caption metrics.

- [ ] **Step 2: Add the loss-by-loss analysis and supporting ablations**

Add `## Why naive distillation failed` with this five-column summary:

| Loss | Observation | Diagnosis or hypothesis | Final formulation | Outcome |
|---|---|---|---|---|
| ITC | In-batch teacher targets failed in the queue-based student objective | Shape alignment does not align candidate set, queue state, or probability structure | Aligned teacher queue + probability-level mixing into the native target | Retrieval improved |
| ITM | Per-pair soft targets severely damaged retrieval | A binary head carries limited relational dark knowledge; soft targets weaken discrimination | Hard-label CE + contrastive KD over each positive and its top-k hard negatives | Retrieval improved |
| LM | Token KD improved captioning | **Hypothesis:** a large vocabulary distribution carries richer dark knowledge than a binary head | Shift-aligned, padding-masked Hinton KD with `T^2` scaling | All eight caption metrics improved |

Then add three short subsections with these exact evidence tables and conclusions:

```markdown
### ITC: matching shape is not enough

| Teacher target | Candidate structure | ITC r_mean |
|---|---|---:|
| Student baseline | Native momentum queue | 63.41 |
| In-batch teacher + zero padding | Shape only aligned | 57.64 |
| Teacher queue | Candidate space fully aligned | 64.49 |
```

State that zero padding matched width but not candidate semantics. Conclude: “Matching the tensor shape was insufficient. Effective distillation required the teacher and student to share the same candidate set, queue state, and probability structure.”

```markdown
### ITM: distill contrastive structure, not binary targets

| ITM supervision | ITC r_mean | ITM r_mean |
|---|---:|---:|
| Hard-label baseline | 63.41 | 70.93 |
| Per-pair soft target | 64.37 | 63.51 |
| Hard CE + top-k contrastive KD | 64.87 | 71.60 |
```

Use the sentence: “We retained hard-label ITM classification and added a contrastive distillation loss over each positive and its top-k hard negatives.” Do not introduce bidirectional ranking terminology, a derived `ITM - ITC` column, or the invalid cross-model raw-separation comparison.

```markdown
### LM: a large output space is a better distillation target

| Metric | Student baseline | LM distillation | Absolute delta |
|---|---:|---:|---:|
| BLEU-1 | 0.7354 | 0.7437 | +0.0083 |
| BLEU-2 | 0.5720 | 0.5820 | +0.0100 |
| BLEU-3 | 0.4334 | 0.4436 | +0.0102 |
| BLEU-4 | 0.3263 | 0.3360 | +0.0097 |
| METEOR | 0.2707 | 0.2733 | +0.0026 |
| ROUGE-L | 0.5515 | 0.5556 | +0.0041 |
| CIDEr | 1.0630 | 1.0934 | +0.0304 |
| SPICE | 0.203 | 0.207 | +0.004 |
```

Label the richer-vocabulary-dark-knowledge explanation as a hypothesis. State the implementation facts only: approximately 30K vocabulary logits, shifted prediction alignment, padding mask, and standard `T^2` scaling.

Use these traceability sources while writing; do not expose absolute machine paths in the README:

- ITC/ITM values and experiment interpretation: `docs/history/technical-report.md`, `docs/history/development-timeline.md`, and `scratchpad/tb_results.json`.
- Caption baseline/distillation values: `output/caption_zeroshot/small_solo_ep19/evaluate.txt`, `output/caption_zeroshot/small_distill_ep19/evaluate.txt`, and `docs/history/technical-report.md` for SPICE.
- Base-capacity caption values: `output/caption_zeroshot/base_nodecay_ep19/val_metrics_with_spice.json`.
- Parameter and backbone labels: `configs/model_size.md` and `docs/model_specs/README.md`.

- [ ] **Step 3: Add the single unified objective and limitations**

Add exactly this equation:

```markdown
$$
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{ITC}}^{\mathrm{mixed\ target}}
+
\mathcal{L}_{\mathrm{ITM}}^{\mathrm{hard\ CE}}
+
\lambda_{\mathrm{ITM}}\mathcal{L}_{\mathrm{ITM}}^{\mathrm{contrastive\ KD}}
+
\mathcal{L}_{\mathrm{LM}}^{\mathrm{CE}}
+
\lambda_{\mathrm{LM}}\mathcal{L}_{\mathrm{LM}}^{\mathrm{token\ KD}}.
$$
```

Define the three added mechanisms in one bullet each. Do not add component equations.

Add `## Limitation` with both points:

- ITM candidate scaling is memory-bound because every image-text pair requires cross-attention over patch and token states; these states cannot be queued compactly like ITC features.
- Full `B x B` evaluation is roughly 13 times the native `3B` pair forwards at `B=40`. Top-k bounds but does not remove linear candidate cost. Hard-label CE remains the stable native separation anchor; extending CE itself to top-k would change the controlled objective and further increase memory.

- [ ] **Step 4: Close with the full matched comparison and four conclusions**

Add a final table containing all retrieval and caption metrics:

| Model | Params | ITC r_mean | ITM r_mean | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Small student baseline | 73.8M | 63.41 | 70.93 | 0.7354 | 0.5720 | 0.4334 | 0.3263 | 0.2707 | 0.5515 | 1.0630 | 0.2030 |
| **Unified distillation** | **73.8M** | **`{{FINAL_ITC}}`** | **`{{FINAL_ITM}}`** | **`{{FINAL_BLEU_1}}`** | **`{{FINAL_BLEU_2}}`** | **`{{FINAL_BLEU_3}}`** | **`{{FINAL_BLEU_4}}`** | **`{{FINAL_METEOR}}`** | **`{{FINAL_ROUGE_L}}`** | **`{{FINAL_CIDER}}`** | **`{{FINAL_SPICE}}`** |
| Base-capacity model | 247.2M | 66.20 | 73.22 | 0.7314 | 0.5675 | 0.4328 | 0.3302 | 0.2783 | 0.5534 | 1.0945 | 0.2082 |

Close the scientific narrative with exactly these claims:

1. Match the teacher signal to the student's native objective structure, not merely its output shape.
2. Large output spaces such as vocabularies appear to provide richer dark knowledge than binary heads.
3. Preserve strong hard supervision when soft targets weaken discrimination.
4. The unified model improves the compact student and approaches the base-capacity model across retrieval and captioning.

- [ ] **Step 5: Add concise sources, artifact links, and license boundaries**

Use primary sources only:

- BLIP paper: `https://arxiv.org/abs/2201.12086`; repository: `https://github.com/salesforce/BLIP`; BSD-3-Clause text: `https://github.com/salesforce/BLIP/blob/main/LICENSE.txt`.
- Hinton KD paper: `https://arxiv.org/abs/1503.02531`.
- DINOv3 paper: `https://arxiv.org/abs/2508.10104`; repository: `https://github.com/facebookresearch/dinov3`; license: `https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md`.
- MiniLM paper: `https://arxiv.org/abs/2002.10957`; repository: `https://github.com/microsoft/unilm/tree/master/minilm`; MIT license: `https://github.com/microsoft/unilm/blob/master/LICENSE`.
- COCO Captions paper: `https://arxiv.org/abs/1504.00325`; dataset terms: `https://cocodataset.org/#termsofuse`.
- Visual Genome paper: `https://arxiv.org/abs/1602.07332`; dataset site/license: `https://visualgenome.org/about`.

State that this repository is derived from Salesforce BLIP; upstream BLIP code remains under BSD-3-Clause. State separately that DINOv3 materials, MiniLM materials, COCO, and Visual Genome remain subject to their respective licenses/terms. Do not claim ownership of datasets or relicense third-party weights.

Link the compact local evidence index with relative links: `docs/history/technical-report.md`, `docs/history/development-timeline.md`, `docs/model_specs/README.md`, and `docs/experiments/README.md`.

- [ ] **Step 6: Verify the English README contract**

Run:

```bash
rg -n "^#|^\| Model|\{\{FINAL_|bidirectional|ITM - ITC|reranking|ROUGE_L|capacity target|official BLIP" README.md
```

Expected:

- One H1 and the approved section order.
- Literal final-result tokens appear only in the hero and final comparison.
- No occurrences of `bidirectional`, `ITM - ITC`, `reranking`, `ROUGE_L`, `capacity target`, or official BLIP performance rows.

Run:

```bash
test "$(rg -c '^\$\$$' README.md)" -eq 2
```

Expected: exit 0, proving exactly one display equation block.

---

### Task 2: Create the Korean companion README

**Files:**
- Create: `README.ko.md`

**Interfaces:**
- Consumes: the completed `README.md` section structure, tables, links, metric tokens, and equation.
- Produces: a faithful Korean rendering with no numerical or structural divergence.

- [ ] **Step 1: Translate prose while preserving the full structural contract**

Create `README.ko.md` with this first line and translated title:

```markdown
[English](README.md) | [한국어](README.ko.md)

# 경량 비전-언어 사전학습을 위한 지식 증류 진단
```

Translate prose for a Korean research audience. Keep section order, heading levels, table row order, numeric precision, equation, URLs, inline code, and all `{{FINAL_*}}` tokens byte-for-byte identical to `README.md`.

Use these fixed terminology mappings:

| English | Korean |
|---|---|
| Small student baseline | 소형 학생 기준선 |
| Unified distillation | 통합 증류 |
| Base-capacity model | 베이스 용량 모델 |
| candidate set/space | 후보 집합/공간 |
| momentum queue | 모멘텀 큐 |
| hard negative | 하드 네거티브 |
| hard-label CE | 하드 라벨 CE |
| contrastive KD | 대조 증류 |
| dark knowledge | 다크 놀리지 |
| limitation | 한계 |

Retain metric names (`ITC r_mean`, `ITM r_mean`, BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr, SPICE) in English.

- [ ] **Step 2: Verify reciprocal navigation and token parity**

Run:

```bash
head -1 README.md
head -1 README.ko.md
```

Expected from both: `[English](README.md) | [한국어](README.ko.md)`.

Run:

```bash
diff <(rg -o '\{\{FINAL_[A-Z0-9_]+\}\}' README.md | sort -u) <(rg -o '\{\{FINAL_[A-Z0-9_]+\}\}' README.ko.md | sort -u)
```

Expected: no output and exit 0.

Run:

```bash
test "$(rg -c '^\$\$$' README.ko.md)" -eq 2
```

Expected: exit 0.

---

### Task 3: Audit numerical, structural, and licensing integrity

**Files:**
- Verify: `README.md`
- Verify: `README.ko.md`

**Interfaces:**
- Consumes: both completed READMEs and their cited local/primary sources.
- Produces: a review-ready, uncommitted two-file README change with deliberate final-run tokens as the only unresolved values.

- [ ] **Step 1: Check forbidden placeholders and stale inherited copy**

Run:

```bash
rg -n "TBD|TODO|baseline value|DEPRECATED|Inference demo|Pre-trained checkpoints|Finetuned checkpoints|VQA|NLVR2" README.md README.ko.md
```

Expected: no output. The literal `{{FINAL_*}}` tokens are allowed because they are controlled result-replacement points.

- [ ] **Step 2: Check key value parity across languages**

Run:

```bash
for value in 63.41 70.93 57.64 64.49 64.87 71.60 1.063 0.203 66.20 73.22 1.095 0.208; do
  test "$(rg -o "$value" README.md | wc -l)" -eq "$(rg -o "$value" README.ko.md | wc -l)" || exit 1
done
```

Expected: exit 0.

- [ ] **Step 3: Check Markdown links and relative artifact targets**

Run:

```bash
for path in docs/history/technical-report.md docs/history/development-timeline.md docs/model_specs/README.md docs/experiments/README.md; do
  test -f "$path" || exit 1
  rg -q "$path" README.md || exit 1
  rg -q "$path" README.ko.md || exit 1
done
```

Expected: exit 0.

Manually open each external primary-source link listed in Task 1 Step 5 and confirm it resolves to the named paper, repository, dataset terms, or license. Confirm that the README never groups DINOv3 under BLIP's BSD-3-Clause terms.

- [ ] **Step 4: Review the final diff without committing**

Run:

```bash
git diff --check -- README.md
git diff --no-index --check /dev/null README.ko.md
git diff -- README.md
git diff --no-index /dev/null README.ko.md
git status --short README.md README.ko.md
```

Expected:

- Both diff checks emit no whitespace errors.
- The diff changes only the inherited `README.md` and adds `README.ko.md`.
- Both files remain uncommitted for user review.
