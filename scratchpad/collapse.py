import glob,os
from tensorboard.backend.event_processing import event_accumulator as EA
runs={
 "9.3 plain ttm_queue_lm":"output/pt_smallreg_minilm_ttm_queue_lm",
 "9.5 holdhalf":"output/pt_ttm_queue_lm_holdhalf",
 "9.4 holdteacher":"output/pt_ttm_queue_lm_holdteacher",
}
def load(root,tag):
    d=[]
    for ev in sorted(glob.glob(os.path.join(root,"**","events.out.tfevents*"),recursive=True)):
        try:a=EA.EventAccumulator(ev,size_guidance={EA.SCALARS:0});a.Reload()
        except Exception:continue
        if tag in a.Tags().get('scalars',[]):
            for x in a.Scalars(tag): d.append((x.step,x.value))
    return sorted(set(d))
for name,root in runs.items():
    itc=load(root,'val_retrieval_itc/r_mean')
    if not itc: print(name,"no data");continue
    # window 450k-747k
    w=[(s,v) for s,v in itc if s>=450000]
    peak_all=max(v for _,v in itc); peak_step=[s for s,v in itc if v==peak_all][0]
    mn=min(w,key=lambda x:x[1]); last=itc[-1]
    print(f"\n### {name}")
    print(f"  peak(all)={peak_all:.2f}@{peak_step}  |  min(after450k)={mn[1]:.2f}@{mn[0]}  |  last={last[1]:.2f}@{last[0]}")
    print("  r_mean @ 480k~747k every ~27k:")
    import bisect
    for target in range(480000,760000,27000):
        # nearest point
        near=min(itc,key=lambda x:abs(x[0]-target))
        print(f"    step~{target:>7}: {near[1]:.2f} (@{near[0]})")

print("\n### 9.5.2 tctmp00157 (fix?) 480~600k")
s=load("output/pt_ttm_queue_lm_holdhalf_tctmp00157",'val_retrieval_itc/r_mean')
w=[(a,b) for a,b in s if 480000<=a<=620000]
print("  min in 480-620k:", min(w,key=lambda x:x[1]) if w else None)
for target in range(490000,620000,20000):
    near=min(s,key=lambda x:abs(x[0]-target)); print(f"    step~{target}: {near[1]:.2f}")
