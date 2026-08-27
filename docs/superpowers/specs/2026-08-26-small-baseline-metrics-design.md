# Small Baseline Metrics Design

Add the normal Small Baseline to the existing 20-epoch experiment export.

- Use CPU only; do not invoke CUDA or GPU tools.
- Reuse the seven existing Small caption result files and generate only the 13 missing epochs.
- Score all 20 Small epochs with Bleu 1-4, METEOR, ROUGE-L, CIDEr, and SPICE.
- Keep Base Capacity, LM Distillation, and Distill All, adding Small Baseline as the fourth plotted series.
- Add an Excel sheet named `Final Base Relative (%)` using checkpoint epoch 19 and `(model / Base Capacity) * 100` for Small Baseline and Distill All across all 26 metrics.
- Preserve the large presentation legends and save outputs under `docs/experiments`.
