import glob,os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator as EA
runs={
 "ITC+LM plain (9.3)":("output/pt_smallreg_minilm_ttm_queue_lm","#d62728"),
 "ITC+LM holdhalf (9.5)":("output/pt_ttm_queue_lm_holdhalf","#ff7f0e"),
 "ITC only (9.2, stable ref)":("output/pt_smallreg_minilm_ttm_queue","#2ca02c"),
 "LM only (8, stable ref)":("output/pt_smallreg_minilm_lm_distill","#1f77b4"),
}
def load(root,tag):
    d=[]
    for ev in sorted(glob.glob(os.path.join(root,"**","events.out.tfevents*"),recursive=True)):
        try:a=EA.EventAccumulator(ev,size_guidance={EA.SCALARS:0});a.Reload()
        except Exception:continue
        if tag in a.Tags().get('scalars',[]):
            for x in a.Scalars(tag): d.append((x.step,x.value))
    return sorted(set(d))
plt.figure(figsize=(12,6.5))
for name,(root,c) in runs.items():
    s=load(root,'val_retrieval_itc/r_mean')
    s=[(st,v) for st,v in s if st>=400000]
    xs=[a for a,_ in s]; ys=[b for _,b in s]
    lw=2.2 if "plain" in name or "holdhalf" in name else 1.2
    al=1.0 if "plain" in name or "holdhalf" in name else 0.55
    plt.plot(xs,ys,label=name,color=c,linewidth=lw,alpha=al)
plt.axvline(500000,color='gray',ls=':',alpha=0.5)
plt.title("ITC+LM late-training oscillation (raw r_mean, no smoothing)",fontsize=15)
plt.xlabel("step");plt.ylabel("r_mean (%)")
plt.legend(fontsize=10);plt.grid(alpha=0.3);plt.tight_layout()
plt.savefig("발표자료/assets/curves/collapse_zoom_raw.png",dpi=200)
print("saved 발표자료/assets/curves/collapse_zoom_raw.png")
