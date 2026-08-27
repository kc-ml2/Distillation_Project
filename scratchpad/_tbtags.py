from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
for run in ["pt_smallreg_minilm_ttm_queue_lm","pt_ttm_queue_lm_holdhalf"]:
    ev=sorted(glob.glob(f"output/{run}/tensorboard/**/events*",recursive=True))
    print("==",run,"== files:",len(ev))
    if not ev: continue
    ea=EventAccumulator(ev[-1],size_guidance={'scalars':0}); ea.Reload()
    print("ALL scalar tags:",ea.Tags()['scalars'])
