from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
run="pt_smallreg_minilm_ttm_queue_lm"
ev=sorted(glob.glob(f"output/{run}/tensorboard/**/events*",recursive=True))
ea=EventAccumulator(ev[-1],size_guidance={'scalars':0}); ea.Reload()
for t in ['val_caption/Bleu_1','val_caption/Bleu_2','val_caption/Bleu_3','val_caption/Bleu_4','val_caption/METEOR','val_caption/ROUGE_L','val_caption/CIDEr','val_caption/SPICE']:
    s=ea.Scalars(t)
    print(f"{t:24s} n={len(s):2d}  laststep={s[-1].step}  last={s[-1].value:.4f}  max={max(x.value for x in s):.4f}")
# also print step->epoch mapping: total steps and per-epoch
c=ea.Scalars('val_caption/CIDEr')
print("\nCIDEr per eval point (step,value):")
for x in c: print(f"  {x.step}\t{x.value:.4f}")
