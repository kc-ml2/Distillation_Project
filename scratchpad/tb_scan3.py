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
 "ITC+LM_hh_t00258(9.5.1)": "output/pt_ttm_queue_lm_holdhalf_tctmp00258",
 "ITC+LM_hh_t00157(9.5.2)": "output/pt_ttm_queue_lm_holdhalf_tctmp00157",
 "ITM_fix(10.1)": "output/pt_smallreg_minilm_itm_matrix_kd_configfix",
 "ALL(12)*진행중": "output/pt_distill_all",
}
def evs(root):
    return [root] if os.path.isfile(root) else sorted(glob.glob(os.path.join(root,"**","events.out.tfevents*"),recursive=True))
def load(root):
    d={}
    for ev in evs(root):
        try: a=EA.EventAccumulator(ev,size_guidance={EA.SCALARS:0}); a.Reload()
        except Exception: continue
        for t in a.Tags().get('scalars',[]):
            for x in a.Scalars(t): d.setdefault(t,[]).append((x.step,x.value))
    for t in d: d[t]=sorted(set(d[t]))
    return d
def series(d,t): return d.get(t,[])
def tail_mean(s,win=15000):
    if not s: return None
    mx=max(st for st,_ in s); return sum(v for st,v in s if st>=mx-win)/max(1,sum(1 for st,_ in s if st>=mx-win))
def peak(s): return max((v for _,v in s), default=None)
def last(s): return s[-1][1] if s else None
print(f"{'run':26s}| ITC r_mean: tail15k / peak / drop  |  ITM r_mean last | CIDEr last")
for name,root in runs.items():
    d=load(root); itc=series(d,'val_retrieval_itc/r_mean')
    tm=tail_mean(itc); pk=peak(itc); drop=(pk-tm) if (tm is not None and pk is not None) else None
    itm=last(series(d,'val_retrieval_itm/r_mean')); cid=last(series(d,'val_caption/CIDEr'))
    f=lambda x:'  -  ' if x is None else f'{x:5.2f}'
    print(f"{name:26s}|   {f(tm)} / {f(pk)} / {f(drop)}   |     {f(itm)}      |  {f(cid)}")
