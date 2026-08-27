import glob, os, sys
from tensorboard.backend.event_processing import event_accumulator as EA

# candidate run roots (dir containing tensorboard/ subdir OR direct tfevents)
runs = {
 "baseline(exp7)": "output/pt_smallreg_minilm_baseline",
 "baseline_autocontrast(target)": "tensorboard_output/target/student/exp7_student_baseline_autocontrast_fix",
 "ITC_ttm_queue(9.2)": "output/pt_smallreg_minilm_ttm_queue",
 "ITC_armS(6.4)": "tensorboard_output/target/student/exp6.4_itc_armS_t0.05",
 "LM(8)": "output/pt_smallreg_minilm_lm_distill",
 "ITC+LM_ttm_queue_lm(9.3)": "output/pt_smallreg_minilm_ttm_queue_lm",
 "ITC+LM_holdteacher(9.4)": "output/pt_ttm_queue_lm_holdteacher",
 "ITC+LM_holdhalf(9.5)": "output/pt_ttm_queue_lm_holdhalf",
 "ITC+LM_holdhalf_t00258(9.5.1)": "output/pt_ttm_queue_lm_holdhalf_tctmp00258",
 "ITC+LM_holdhalf_t00157(9.5.2)": "output/pt_ttm_queue_lm_holdhalf_tctmp00157",
 "ITM_matrix(10)": "output/pt_smallreg_minilm_itm_matrix_kd",
 "ITM_matrix_configfix(10.1)": "output/pt_smallreg_minilm_itm_matrix_kd_configfix",
 "ALL_distill_all(12)": "output/pt_distill_all",
}

def find_events(root):
    if os.path.isfile(root): return [root]
    ev = glob.glob(os.path.join(root, "**", "events.out.tfevents*"), recursive=True)
    return sorted(ev)

for name, root in runs.items():
    evs = find_events(root)
    print(f"\n===== {name}  ({root})  [{len(evs)} event file(s)]")
    if not evs:
        print("  (no tfevents)"); continue
    # merge tags across event files, track step ranges per tag
    tag_steps = {}
    for ev in evs:
        try:
            acc = EA.EventAccumulator(ev, size_guidance={EA.SCALARS:0})
            acc.Reload()
        except Exception as e:
            print("  ERR", ev, e); continue
        for t in acc.Tags().get('scalars', []):
            s = acc.Scalars(t)
            steps = [x.step for x in s]
            if t not in tag_steps: tag_steps[t] = [min(steps), max(steps), len(steps)]
            else:
                tag_steps[t][0] = min(tag_steps[t][0], min(steps))
                tag_steps[t][1] = max(tag_steps[t][1], max(steps))
                tag_steps[t][2] += len(steps)
    # only show retrieval/caption-ish + a couple loss tags
    keys = sorted(tag_steps)
    interesting = [k for k in keys if any(w in k.lower() for w in
                   ['r1','r5','r10','recall','retr','r_mean','rmean','cider','spice','bleu','meteor','rouge','itm_acc','val'])]
    print("  interesting tags (min_step, max_step, n):")
    for k in interesting:
        print(f"    {k:45s} {tag_steps[k]}")
    print("  ALL tags:", keys)
