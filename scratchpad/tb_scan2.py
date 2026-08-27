import glob, os
from tensorboard.backend.event_processing import event_accumulator as EA
runs = {
 "baseline(7)": "output/pt_smallreg_minilm_baseline",
 "baseline_autoc(7fix)": "tensorboard_output/target/student/exp7_student_baseline_autocontrast_fix",
 "ITC_ttm_queue(9.2)": "output/pt_smallreg_minilm_ttm_queue",
 "ITC_armS(6.4)": "tensorboard_output/target/student/exp6.4_itc_armS_t0.05",
 "LM(8)": "output/pt_smallreg_minilm_lm_distill",
 "ITC+LM(9.3)": "output/pt_smallreg_minilm_ttm_queue_lm",
 "ITC+LM_holdteacher(9.4)": "output/pt_ttm_queue_lm_holdteacher",
 "ITC+LM_holdhalf(9.5)": "output/pt_ttm_queue_lm_holdhalf",
 "ITC+LM_holdhalf_t00258(9.5.1)": "output/pt_ttm_queue_lm_holdhalf_tctmp00258",
 "ITC+LM_holdhalf_t00157(9.5.2)": "output/pt_ttm_queue_lm_holdhalf_tctmp00157",
 "ITM_matrix(10)": "output/pt_smallreg_minilm_itm_matrix_kd",
 "ITM_matrix_fix(10.1)": "output/pt_smallreg_minilm_itm_matrix_kd_configfix",
 "ALL(12)": "output/pt_distill_all",
}
def evs(root):
    return [root] if os.path.isfile(root) else sorted(glob.glob(os.path.join(root,"**","events.out.tfevents*"),recursive=True))
def load(root):
    d={}
    for ev in evs(root):
        try:
            a=EA.EventAccumulator(ev,size_guidance={EA.SCALARS:0}); a.Reload()
        except Exception: continue
        for t in a.Tags().get('scalars',[]):
            s=a.Scalars(t)
            for x in s: d.setdefault(t,[]).append((x.step,x.value))
    for t in d: d[t]=sorted(set(d[t]))
    return d
print(f"{'run':32s}{'maxstep':>9}{'itc_n':>7}{'CIDEr_n':>8}{'CIDEr_max':>10}{'itmR_n':>7}")
for name,root in runs.items():
    d=load(root)
    def info(t):
        if t not in d: return (0,0)
        st=[s for s,_ in d[t]]; return (len(st),max(st))
    itc_n,itc_max=info('val_retrieval_itc/r_mean')
    cid_n,cid_max=info('val_caption/CIDEr')
    itm_n,itm_max=info('val_retrieval_itm/r_mean')
    print(f"{name:32s}{itc_max:>9}{itc_n:>7}{cid_n:>8}{cid_max:>10}{itm_n:>7}")
