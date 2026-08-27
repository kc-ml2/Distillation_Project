[English](README.md) | [한국어](README.ko.md)

# Diagnosing Knowledge Distillation for Compact Vision-Language Pre-training

A compact BLIP student with 69.4M unique online parameters reaches strong retrieval and captioning performance after loss-specific analysis and redesign of ITC, ITM, and LM distillation.

## Quick start

Use Python 3.10, install the dependencies, and place COCO, Visual Genome, and the
[BLIP-large checkpoint](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth)
at the relative paths declared in `configs/pretrain_mainline.yaml`.

```bash
pip install -r requirements.txt
python -m torch.distributed.run --nproc_per_node=4 pretrain.py \
  --config configs/pretrain_mainline.yaml
```

The default configuration writes checkpoints and logs under `output/pt_refactor_mainline`.
Change the process count and per-GPU batch size to match the available GPUs.

The following table reports final-checkpoint COCO Karpathy validation metrics; all caption rows use the same saved-generation and COCOEvalCap scoring protocol.

| Model | Online unique params | ITC r_mean | ITM r_mean | CIDEr | SPICE |
|---|---:|---:|---:|---:|---:|
| Small student baseline | 69.4M | 63.41 | 70.93 | 1.063 | 0.201 |
| **Unified distillation** | 69.4M | 🔴 **66.52** | 73.16 | 🔴 **1.104** | 0.207 |
| Base-capacity model | 252.4M | 66.20 | 73.22 | 1.095 | 0.208 |

Parameter counts use unique parameters in the online pre-training graph, with tied weights deduplicated and momentum copies and queue buffers excluded.

- Competing ITC objectives over different candidate spaces broke naive KD.
- Binary ITM soft targets weakened hard-negative discrimination.
- The loss-specific redesign improved retrieval and all eight caption metrics.

## Why naive distillation failed

| Loss | Observation | Diagnosis or hypothesis | Final formulation | Outcome |
|---|---|---|---|---|
| ITC | Adding separate in-batch KD to native queue-based ITC degraded retrieval | The two losses normalized over different candidate spaces and targets, producing competing gradients for the same features | Mix the teacher's batch-and-queue probabilities into the native target, leaving one queue-based objective | Retrieval improved |
| ITM | Per-pair soft targets severely damaged retrieval | A binary head carries limited relational dark knowledge; soft targets weaken discrimination | Hard-label CE + contrastive KD over each positive and its top-k hard negatives | Retrieval improved |
| LM | Token KD improved captioning | **Hypothesis:** a large vocabulary distribution carries richer dark knowledge than a binary head | Shift-aligned, padding-masked Hinton KD with `T^2` scaling | All eight caption metrics improved |

### ITC: one candidate geometry, one objective

| Setup | Optimization path | ITC r_mean |
|---|---|---:|
| Small student baseline | One native batch-and-queue objective | 63.41 |
| Separate in-batch KD (zero-padded) | Native batch-and-queue ITC + separate in-batch KD gradient | 57.64 |
| Queue-aligned target mixing | Teacher probabilities absorbed into the native batch-and-queue target | 64.49 |
| Base-capacity model | Native batch-and-queue objective | 66.20 |

Native ITC learns a distribution over the current batch and momentum queue. Naive KD imposed a second distribution over a different in-batch candidate set, so the same features received gradients optimized against different probability spaces and targets. Zero padding merely matched tensor width and reduced ITC r_mean further. The final design evaluates the teacher over the same batch-and-queue candidates and mixes its probabilities into the native target, replacing two competing gradients with one queue-based contrastive objective.

### ITM: distill contrastive structure, not binary targets

| ITM supervision | ITC r_mean | ITM r_mean |
|---|---:|---:|
| Hard-label baseline | 63.41 | 70.93 |
| Per-pair soft target | 64.37 | 63.51 |
| Hard CE + top-k contrastive KD | 64.87 | 71.60 |
| Base-capacity model | 66.20 | 73.22 |

Within-query diagnostics showed that per-pair soft targets compressed the ranking margin and damaged retrieval. We retained hard-label ITM classification and added a contrastive distillation loss over each positive and its top-k hard negatives.

### LM: a large output space is a better distillation target

The following ablation uses final-checkpoint COCO Karpathy test generations scored with the same COCOEvalCap protocol, including SPICE.

| Metric | Student baseline | LM distillation | Base-capacity model | Delta vs. student |
|---|---:|---:|---:|---:|
| BLEU-1 | 0.7350 | 0.7444 | 0.7326 | +0.0094 |
| BLEU-2 | 0.5705 | 0.5811 | 0.5690 | +0.0105 |
| BLEU-3 | 0.4334 | 0.4425 | 0.4360 | +0.0090 |
| BLEU-4 | 0.3267 | 0.3343 | 0.3342 | +0.0076 |
| METEOR | 0.2719 | 0.2738 | 0.2801 | +0.0019 |
| ROUGE-L | 0.5538 | 0.5577 | 0.5564 | +0.0039 |
| CIDEr | 1.0764 | 1.1028 | 1.1131 | +0.0263 |
| SPICE | 0.2029 | 0.2067 | 0.2105 | +0.0038 |

The LM objective distills approximately 30K vocabulary logits with shifted prediction alignment, a padding mask, and standard `T^2` scaling. **Hypothesis:** this large vocabulary distribution carries richer dark knowledge than a binary head, explaining why token KD improved all eight caption metrics.

## Unified objective

```math
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
```

- ITC mixes aligned teacher probabilities into the student's native queue-based target.
- ITM preserves hard-label CE and adds contrastive KD over each positive and its top-k hard negatives.
- LM applies shift-aligned, padding-masked token KD over the vocabulary distribution with `T^2` scaling.

## Limitation

- ITM candidate scaling is memory-bound because every image-text pair requires cross-attention over patch and token states; these states cannot be queued compactly like ITC features.
- Full `B x B` evaluation is roughly 13 times the native `3B` pair forwards at `B=40`. Top-k bounds but does not remove linear candidate cost. Hard-label CE remains the stable native separation anchor; extending CE itself to top-k would change the controlled objective and further increase memory.

## Matched comparison

The following final-checkpoint comparison reports COCO Karpathy validation metrics; all caption rows use the same saved-generation and COCOEvalCap scoring protocol.

| Model | Online unique params | ITC r_mean | ITM r_mean | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Small student baseline | 69.4M | 63.41 | 70.93 | 0.7354 | 0.5720 | 0.4334 | 0.3263 | 0.2707 | 0.5515 | 1.0630 | 0.2012 |
| **Unified distillation** | 69.4M | 🔴 **66.52** | 73.16 | 🔴 **0.7481** | 🔴 **0.5866** | 🔴 **0.4480** | 🔴 **0.3402** | 0.2749 | 🔴 **0.5578** | 🔴 **1.1044** | 0.2072 |
| Base-capacity model | 252.4M | 66.20 | 73.22 | 0.7314 | 0.5675 | 0.4328 | 0.3302 | 0.2783 | 0.5534 | 1.0945 | 0.2082 |

## Conclusions

1. Distill ITC inside the student's native candidate probability space; separate objectives over different candidate sets produce competing gradients.
2. Large output spaces such as vocabularies appear to provide richer dark knowledge than binary heads.
3. Preserve strong hard supervision when soft targets weaken discrimination.
4. The unified model improves the compact student and approaches the base-capacity model across retrieval and captioning.

## Sources, artifacts, and licenses

- The project is derived from [Salesforce BLIP](https://github.com/salesforce/BLIP) and its [paper](https://arxiv.org/abs/2201.12086). Upstream BLIP code remains under the [BSD-3-Clause license](https://github.com/salesforce/BLIP/blob/main/LICENSE.txt).
- The distillation formulation builds on [Hinton et al.](https://arxiv.org/abs/1503.02531).
- The compact student uses materials from [DINOv3](https://arxiv.org/abs/2508.10104) ([repository](https://github.com/facebookresearch/dinov3), [license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)) and [MiniLM](https://arxiv.org/abs/2002.10957) ([repository](https://github.com/microsoft/unilm/tree/master/minilm), [MIT license](https://github.com/microsoft/unilm/blob/master/LICENSE)). Those materials remain subject to their respective licenses.
- Experiments use [COCO Captions](https://arxiv.org/abs/1504.00325) ([dataset terms](https://cocodataset.org/#termsofuse)) and [Visual Genome](https://arxiv.org/abs/1602.07332) ([dataset site and license](https://visualgenome.org/about)). Each dataset remains subject to its respective terms; this project does not claim ownership of the datasets or relicense third-party weights.
- Local evidence: [technical report](docs/history/technical-report.md) and [development timeline](docs/history/development-timeline.md).
