# Research README Redesign

**Date:** 2026-08-24  
**Audience:** Vision-language and knowledge-distillation researchers  
**Primary language:** English (`README.md`) with a Korean companion (`README.ko.md`)

## 1. Goal

Replace the inherited BLIP README with a concise research-facing account of this project's main contribution:

> Naive knowledge distillation behaves differently across BLIP's ITC, ITM, and LM objectives. Matching the teacher signal to each student's native objective structure enables a compact unified model to approach a base-capacity model across retrieval and captioning.

The README should lead and close with the unified model's performance, while the main body emphasizes the analysis that led from failed distillation objectives to the final loss-specific design.

## 2. Scope and editorial rules

- Optimize for fast review by researchers rather than chronological project history.
- Present experimentally supported observations separately from mechanistic hypotheses.
- Use matched internal runs as the primary comparison set.
- Omit system/infrastructure contributions except where required for reproducibility.
- Avoid listing every abandoned experiment; retain only the ablations needed to support a conclusion.
- Use one integrated objective equation. Link detailed derivations and diagnostic records instead of reproducing them.
- Draft final-run values as explicit replacement tokens until the 20-epoch unified run is complete.

## 3. Files and language navigation

- `README.md`: canonical English document.
- `README.ko.md`: Korean document with the same headings, tables, figure numbering, and metric tokens.
- Both documents begin with reciprocal `English | 한국어` links.
- The existing upstream BLIP usage information is not repeated wholesale. Required attribution, source links, and license notices remain in the final sections.

## 4. README structure

### 4.1 Hero: result first

Title:

> Diagnosing Knowledge Distillation for Compact Vision-Language Pre-training

Opening claim:

> A 73.8M-parameter BLIP student reaches strong retrieval and captioning performance after loss-specific analysis and redesign of ITC, ITM, and LM distillation.

The first table contains only the most decision-relevant metrics:

| Model | Params | ITC r_mean | ITM r_mean | CIDEr | SPICE |
|---|---:|---:|---:|---:|---:|
| Small student baseline | 73.8M | 63.41 | 70.93 | 1.063 | 0.203 |
| Unified distillation | 73.8M | `{{FINAL_ITC}}` | `{{FINAL_ITM}}` | `{{FINAL_CIDER}}` | `{{FINAL_SPICE}}` |
| Base-capacity model | 247.2M | 66.20 | 73.22 | 1.095 | 0.208 |

The table uses COCO validation metrics from matched 20-epoch runs. Caption metrics must use the same generation and scoring protocol across rows.

Three bullets summarize the main findings:

- Naive ITC distillation failed when teacher and student candidate structures differed.
- Binary ITM soft targets weakened hard-negative discrimination.
- Loss-specific redesign improved the compact model across retrieval and all eight captioning metrics.

### 4.2 Analysis overview

Use one compact table:

| Loss | Observation | Diagnosis or hypothesis | Final formulation | Outcome |
|---|---|---|---|---|
| ITC | In-batch teacher targets failed to transfer to the queue-based student objective | Matching tensor shape alone does not align the candidate set, queue state, or probability structure | Maintain an aligned teacher queue and mix teacher probabilities into the native student target | Retrieval improved |
| ITM | Per-pair soft targets severely damaged retrieval | A binary head carries limited relational dark knowledge, and soft targets weaken hard-negative discrimination | Hard-label CE plus contrastive KD over each positive and its top-k hard negatives | Retrieval improved |
| LM | Token KD consistently improved captioning | **Hypothesis:** a large vocabulary distribution contains richer dark knowledge than a binary head | Shift-aligned, padding-masked Hinton KD with `T^2` scaling | BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr, and SPICE improved |

### 4.3 Evidence behind the redesign

#### ITC: matching shape is not enough

- The native student target is normalized over the current batch plus a momentum queue.
- The controlled in-batch variant zero-padded teacher probabilities to the queue width, matching tensor shape without matching candidate semantics.
- The zero-padded variant ended at ITC r_mean 57.64; the aligned teacher-queue variant reached 64.49.
- Conclusion: effective ITC distillation requires the teacher and student to share the candidate set, queue state, and probability structure.

Key sentence:

> Matching the tensor shape was insufficient. Effective distillation required the teacher and student to share the same candidate set, queue state, and probability structure.

#### ITM: distill contrastive structure, not binary targets

- Per-pair binary soft targets damaged retrieval.
- Preserve the original hard-label ITM CE as a match/non-match separation anchor.
- Add contrastive KD over a positive and its top-k hard negatives.
- Show only within-query diagnostics and retrieval metrics; do not reuse the retracted raw separation comparison that is contaminated by head-weight norm.

#### LM: large output spaces carry richer dark knowledge

- Distill the approximately 30K-dimensional vocabulary distribution.
- Align the KD mask and shifted prediction frame with the original LM CE.
- Apply standard Hinton `T^2` scaling.
- Label vocabulary dark knowledge as a hypothesis, not a proven causal mechanism.
- Report improvement across all eight captioning metrics.

### 4.4 Unified objective

Use a single equation:

$$
\mathcal{L}_{total}
=
\mathcal{L}_{ITC}^{mixed\ target}
+
\mathcal{L}_{ITM}^{hard\ CE}
+
\lambda_{ITM}\mathcal{L}_{ITM}^{contrastive\ KD}
+
\mathcal{L}_{LM}^{CE}
+
\lambda_{LM}\mathcal{L}_{LM}^{token\ KD}.
$$

Immediately define the three mechanisms in plain language:

- ITC embeds teacher knowledge into the native queue-based target.
- ITM preserves hard-label classification and adds contrastive KD over positive and top-k hard negatives.
- LM applies token-level Hinton KD over the vocabulary distribution.

### 4.5 Ablations

#### ITC candidate alignment

| Teacher target | Candidate structure | ITC r_mean |
|---|---|---:|
| Student baseline | Native momentum queue | 63.41 |
| In-batch teacher + zero padding | Shape only aligned | 57.64 |
| Teacher queue | Candidate space fully aligned | 64.49 |

#### ITM objective

| ITM supervision | ITC r_mean | ITM r_mean |
|---|---:|---:|
| Hard-label baseline | 63.41 | 70.93 |
| Per-pair soft target | 64.37 | 63.51 |
| Hard CE + top-k contrastive KD | 64.87 | 71.60 |

Do not include a derived `ITM - ITC` column in the README; it is a diagnostic reranking quantity and can be mistaken for overall distillation improvement.

#### LM captioning

Compare the student baseline and LM-distilled student on COCO Karpathy test using BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr, and SPICE. Include absolute deltas. Keep the detailed caption table out of the hero to preserve first-screen readability.

### 4.6 Final unified result

Repeat the three-row comparison—small student baseline, unified distillation, and base-capacity model—with full retrieval and caption metrics. This is the closing quantitative result. Do not add official BLIP-base or BLIP-large rows; they are not matched training baselines.

### 4.7 Limitations

State the principal scalability limit:

> ITM candidate scaling is memory-bound because every image-text pair requires cross-attention over patch and token states. Unlike ITC, these states cannot be stored in a compact feature queue.

Full `B x B` ITM evaluation requires roughly 13 times as many pair forwards as the native `3B` ITM path. Top-k sampling bounds the cost but remains linear in the number of candidates.

Explain why hard-label CE was not replaced by a top-k objective:

> We preserved the original hard-label CE as a stable separation anchor and added top-k contrastive KD as an isolated auxiliary objective. Extending CE itself to all top-k negatives would change the native BLIP objective and further increase cross-attention memory.

### 4.8 Conclusion

Close with four findings:

1. Distillation works best when the teacher signal matches the student's native objective structure, not merely its output shape.
2. Large output spaces such as vocabularies appear to provide richer dark knowledge than binary heads.
3. Strong hard supervision should be preserved when soft targets weaken discrimination.
4. The unified model improves the compact student and approaches the base-capacity model across retrieval and captioning.

The conclusion links back to the final result table so the README begins and ends with the unified model's performance.

### 4.9 Sources, citation, and licenses

Keep this section concise but complete:

- Cite the original BLIP paper and repository.
- Cite Hinton et al. for temperature-scaled knowledge distillation.
- Attribute the vision and language backbones used by the compact student and teacher.
- Attribute COCO and Visual Genome and state that dataset licenses/terms remain with their respective owners.
- Preserve the repository's BSD-3-Clause license notice and distinguish upstream BLIP code from project modifications.
- Link the detailed experiment, diagnostic, and implementation documents used to support the README claims.

## 5. Result integrity

- `{{FINAL_*}}` tokens are deliberate replacement points, not publishable values.
- Replace them only after the unified 20-epoch run and matched COCO evaluation complete.
- Report final-checkpoint values as the primary result; best-validation values may appear only when explicitly labeled with their checkpoint.
- Do not mix COCO validation and test metrics within one comparison table.
- Do not describe hypotheses as experimentally established mechanisms.

## 6. Acceptance criteria

- A researcher can identify the central result and contribution from the first screen.
- The main analysis can be read without following the chronological development history.
- Every numerical claim is traceable to an event file, evaluation result, or diagnostic artifact.
- English and Korean documents contain the same claims, tables, and metric values.
- The README contains one integrated objective equation and no redundant derivations.
- Sources and licenses clearly distinguish upstream work, third-party components, datasets, and project modifications.
