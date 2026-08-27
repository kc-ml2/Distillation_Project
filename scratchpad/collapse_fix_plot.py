import glob,os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator as EA
def load(root,tag):
    d=[]
    for ev in sorted(glob.glob(os.path.join(root,"**","events.out.tfevents*"),recursive=True)):
        try:a=EA.EventAccumulator(ev,size_guidance={EA.SCALARS:0});a.Reload()
        except Exception:continue
        if tag in a.Tags().get('scalars',[]):
            for x in a.Scalars(tag): d.append((x.step,x.value))
    return sorted(set(d))
runs=[("ITC+LM teacher-mix (collapse)","output/pt_ttm_queue_lm_holdhalf","#d62728"),
      ("+ teacher-temp fix tctmp0.0157 (stable)","output/pt_ttm_queue_lm_holdhalf_tctmp00157","#2ca02c")]
plt.figure(figsize=(12,6))
for name,root,c in runs:
    s=[(a,b) for a,b in load(root,'val_retrieval_itc/r_mean') if a>=400000]
    plt.plot([a for a,_ in s],[b for _,b in s],label=name,color=c,linewidth=2.0)
plt.axvspan(505000,600000,color='red',alpha=0.06)
plt.axvline(505000,color='gray',ls=':',alpha=0.6)
plt.annotate("collapse @~505k",xy=(505000,58.5),xytext=(540000,59.2),fontsize=11,color='#d62728')
plt.title("ITC+LM late collapse and teacher-temp fix (raw ITC r_mean)",fontsize=14)
plt.xlabel("step");plt.ylabel("r_mean (%)");plt.legend(fontsize=11,loc='lower right')
plt.grid(alpha=0.3);plt.tight_layout()
plt.savefig("발표자료/assets/curves/collapse_vs_fix.png",dpi=200)
print("saved")
