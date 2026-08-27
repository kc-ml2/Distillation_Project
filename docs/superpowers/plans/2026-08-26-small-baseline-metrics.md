# Small Baseline Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add all 20 Small Baseline epochs to the experiment workbook and plots, plus epoch-19 performance percentages against Base Capacity, using CPU only.

**Architecture:** Extend the existing model registries and workbook exporter rather than adding new scripts. Reuse existing caption outputs, generate only missing files, then run the existing CPU scorer and plotter.

**Tech Stack:** Python, PyTorch CPU, openpyxl, matplotlib, unittest

**Spec:** `docs/superpowers/specs/2026-08-26-small-baseline-metrics-design.md`

## Global Constraints

- Do not invoke CUDA or GPU tools.
- Save final artifacts under `docs/experiments`.
- Use checkpoint epoch 19 and `(model / Base Capacity) * 100` for relative performance.

---

### Task 1: Small model support and relative-performance export

**Files:**
- Modify: `scratchpad/generate_epoch_captions.py`
- Modify: `scratchpad/score_epoch_captions.py`
- Modify: `docs/experiments/plot_epoch_metrics.py`
- Test: `tests/test_generate_epoch_captions.py`
- Test: `tests/test_score_epoch_captions.py`
- Test: `tests/test_plot_epoch_metrics.py`

**Interfaces:**
- Consumes: existing Small checkpoints and `small_solo_epXX` caption directories
- Produces: `small` CLI model choice, fourth plot series, and `Final Base Relative (%)` workbook sheet

- [ ] **Step 1: Write failing tests**

Add assertions that both caption scripts accept the Small registry entry and that `write_excel` writes hand-calculated epoch-19 ratios for Small Baseline and Distill All.

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m unittest tests.test_generate_epoch_captions tests.test_score_epoch_captions tests.test_plot_epoch_metrics -v`

Expected: FAIL because `small` and `Final Base Relative (%)` do not exist.

- [ ] **Step 3: Implement the minimum changes**

Add the existing Small checkpoint/config/output paths to both registries, add Small Baseline to the plot registry, and write one percentage worksheet from epoch-19 values.

- [ ] **Step 4: Run tests to verify GREEN**

Run the same unittest command and expect zero failures.

### Task 2: CPU-only data generation and final verification

**Files:**
- Generate: `output/caption_zeroshot/small_solo_epXX/result/val_epoch0.json`
- Generate: `output/caption_zeroshot/small_solo_epXX/val_metrics_with_spice.json`
- Generate: `docs/experiments/per_epoch_all_metrics.xlsx`
- Generate: `docs/experiments/per_epoch_all_metrics.png`
- Generate: `docs/experiments/per_metric_plots/*.png`

**Interfaces:**
- Consumes: Task 1 CLI/model registry support
- Produces: complete Small metric data and refreshed presentation artifacts

- [ ] **Step 1: Generate missing Small captions on CPU**

Run: `python -m scratchpad.generate_epoch_captions small --device cpu`

- [ ] **Step 2: Score all Small epochs on CPU**

Run: `python -m scratchpad.score_epoch_captions small --cpu-list 0-31`

- [ ] **Step 3: Regenerate workbook and plots**

Run: `python -m docs.experiments.plot_epoch_metrics`

- [ ] **Step 4: Verify outputs and tests**

Confirm 20 caption files, 20 complete metric files, four 80-row metric sheets, one 26-row ratio sheet, 26 per-metric PNGs, valid images, and a clean full unittest run.
