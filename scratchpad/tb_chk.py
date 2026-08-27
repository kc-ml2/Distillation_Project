import glob, os
from tensorboard.backend.event_processing import event_accumulator as EA
runs = {
 "9.5 holdhalf": "output/pt_ttm_queue_lm_holdhalf",
 "9.5.2 t00157": "output/pt_ttm_queue_lm_holdhalf_tctmp00157",
 "BASE exp4 logitscale_nodecay": "output/pt_checkpoint_base_logitscale_nodecay",
}
def evs(root): return sorted(glob.glob(os.path.join(root,"**","events.out.tfevents*"),recursive=True))
def load(root):
    d={}
    for ev in evs(root):
        try: a=EA.EventAccumulator(ev,size_guidance={EA.SCALARS:0}); a.Reload()
        except Exception: continue
        for t in a.Tags().get('scalars',[]):
            for x in a.Scalars(t): d.setdefault(t,[]).append((x.step,x.value))
    for t in d: d[t]=sorted(set(d[t]))
    return d
for name,root in runs.items():
    d=load(root); itc=d.get('val_retrieval_itc/r_mean',[])
    steps=[s for s,_ in itc]
    print(f"\n{name}: itc r_mean n={len(steps)} min={min(steps) if steps else '-'} max={max(steps) if steps else '-'}")
    print("  has CIDEr:", 'val_caption/CIDEr' in d, "| itm r_mean n:", len(d.get('val_retrieval_itm/r_mean',[])))
    if steps: 
        mx=max(steps); tail=[v for s,v in itc if s>=mx-15000]; print(f"  tail15k mean={sum(tail)/len(tail):.2f} (n={len(tail)})")
