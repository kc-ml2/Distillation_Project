# TensorBoard Output Sync Design

## Goal

Synchronize selected training event files from `output/` into the existing research-oriented `tensorboard_output/{target,auxiliary,deprecated}/{base,student}` hierarchy.

## Constraints

- Treat `output/` as read-only.
- Copy only `events.out.tfevents.*` files; do not copy checkpoints, logs, or configs.
- Keep the existing `target`, `auxiliary`, and `deprecated` classifications.
- Exclude starts that ran for less than one epoch.
- Include incomplete runs that completed at least one epoch, with `__stopped_epXX` in the destination name.
- Put compatible restart segments in one experiment directory only when their step ranges continue; exclude tiny failed starts.
- This is a one-time cleanup. Do not add a reusable synchronization script.

## Approved Mapping

| Source run | Destination |
|---|---|
| `pt_distill_all` full run | `target/student/exp12_distill_all/` |
| `pt_smallreg_minilm_itm_matrix_kd_configfix` full run | `target/student/exp10.1_itm_matrix_kd_configfix/` |
| `pt_ttm_queue_lm_holdteacher` | `auxiliary/student/exp9.4_ttm_queue_lm_holdteacher/` |
| `pt_ttm_queue_lm_holdhalf` full run | `auxiliary/student/exp9.5_ttm_queue_lm_holdhalf/` |
| `pt_ttm_queue_lm_holdhalf_tctmp00258` | `auxiliary/student/exp9.5.1_ttm_queue_lm_tctmp0.0258/` |
| `pt_ttm_queue_lm_holdhalf_tctmp00157` | `auxiliary/student/exp9.5.2_ttm_queue_lm_tctmp0.0157/` |
| `pt_itm_schemeA_studentneg` 8-epoch run | `auxiliary/student/exp10.SA2_itm_tempered_studentneg__stopped_ep08/` |
| `pt_smallreg_minilm_itm_matrix_kd` 3-epoch run | `deprecated/student/exp10_itm_matrix_kd_t0.05__stopped_ep03/` |

Replace the incomplete copy in `auxiliary/student/exp10.M2_itm_tgtmix/` with the same-named 9,150,269-byte source event. No backup is required because `output/` retains the source.

## Verification

- Every copied destination has the same SHA-256 digest as its source.
- Every destination event loads through TensorBoard's `EventAccumulator`.
- Full runs reach step `746920`.
- Stopped runs reach steps `329200` (ep08) and `145350` (ep03).
- Excluded short-start event files are absent from the new destination directories.
- A before/after manifest confirms no file under `output/` changed.
