# TensorBoard Output Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate `tensorboard_output/` with the approved full and meaningful stopped TensorBoard runs and replace one incomplete copied event.

**Architecture:** Perform a one-time, explicit source-to-destination copy. Capture a source manifest first, copy each event through a temporary file followed by `os.replace`, then verify source immutability, SHA-256 equality, TensorBoard readability, and expected maximum steps.

**Tech Stack:** Python 3.10 standard library, TensorBoard `EventAccumulator`

**Spec:** `docs/superpowers/specs/2026-08-27-tensorboard-output-sync-design.md`

## Global Constraints

- `output/` is read-only.
- Only TensorBoard event files are copied.
- No reusable synchronization script or backup copy is created.
- Existing `target/auxiliary/deprecated` semantics remain unchanged.
- Event files shorter than one epoch are excluded.

---

### Task 1: Synchronize and verify approved event files

**Files:**
- Create: eight approved run directories under `tensorboard_output/`
- Replace: `tensorboard_output/auxiliary/student/exp10.M2_itm_tgtmix/events.out.tfevents.1784905684.R4G4.4180408.0`
- Do not modify: `output/**`

**Interfaces:**
- Consumes: the exact mapping and expected steps in the design spec
- Produces: destination event files byte-identical to their selected sources

- [ ] **Step 1: Capture source metadata**

Use Python to record each `output/**/events.out.tfevents.*` path, size, mtime in nanoseconds, and SHA-256 digest in memory before copying.

- [ ] **Step 2: Copy the approved files atomically**

For each approved mapping, create the destination directory, copy to `<event>.tmp`, verify the temporary file digest equals the source digest, and call `os.replace(tmp, destination)`. Select only these event files:

```text
pt_distill_all/...1787193971... -> target/student/exp12_distill_all/
pt_smallreg_minilm_itm_matrix_kd_configfix/...1786327330... -> target/student/exp10.1_itm_matrix_kd_configfix/
pt_ttm_queue_lm_holdteacher/...1785390167... -> auxiliary/student/exp9.4_ttm_queue_lm_holdteacher/
pt_ttm_queue_lm_holdhalf/...1785727307... -> auxiliary/student/exp9.5_ttm_queue_lm_holdhalf/
pt_ttm_queue_lm_holdhalf_tctmp00258/...1786869626... -> auxiliary/student/exp9.5.1_ttm_queue_lm_tctmp0.0258/
pt_ttm_queue_lm_holdhalf_tctmp00157/...1787017321... -> auxiliary/student/exp9.5.2_ttm_queue_lm_tctmp0.0157/
pt_itm_schemeA_studentneg/...1785389028... -> auxiliary/student/exp10.SA2_itm_tempered_studentneg__stopped_ep08/
pt_smallreg_minilm_itm_matrix_kd/...1786253274... -> deprecated/student/exp10_itm_matrix_kd_t0.05__stopped_ep03/
pt_itm_schemeA_tempered/...1784905684... -> replace auxiliary/student/exp10.M2_itm_tgtmix/
```

- [ ] **Step 3: Verify digests and TensorBoard steps**

Load every copied destination with `EventAccumulator(size_guidance={'scalars': 0})`. Assert source and destination SHA-256 equality and these maximum scalar steps:

```text
full runs: 746920
exp10.SA2 stopped ep08: 329200
exp10 matrix KD stopped ep03: 145350
exp10.M2 replacement: same max step as its source
```

- [ ] **Step 4: Verify exclusions and source immutability**

Assert each new destination directory contains exactly one event file. Recompute the complete source manifest and assert it equals the Step 1 manifest. Confirm no `.tmp` files remain.

- [ ] **Step 5: Review repository status**

Run `git status --short` and report all created/replaced paths. Do not commit without separate user approval.
