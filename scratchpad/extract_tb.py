import glob, json, sys
from tensorboard.backend.event_processing import event_accumulator

RUNS = {
    "base_baseline": "output/pt_checkpoint_base_logitscale_nodecay",
    "small_baseline": "output/pt_smallreg_minilm_baseline",
    "ITC_only": "output/pt_smallreg_minilm_ttm_queue",
    "LM_only": "output/pt_smallreg_minilm_lm_distill",
    "ITC+LM": "output/pt_ttm_queue_lm_holdhalf_tctmp00157",
    "ITM": "output/pt_smallreg_minilm_itm_matrix_kd_configfix",
    "ITC+ITM+LM": "output/pt_distill_all",
}

TAGS = []
for grp in ["val_retrieval_itc", "val_retrieval_itm"]:
    for m in ["txt_r1","txt_r5","txt_r10","img_r1","img_r5","img_r10","r_mean"]:
        TAGS.append(f"{grp}/{m}")
for m in ["CIDEr","SPICE","Bleu_4","METEOR","ROUGE_L"]:
    TAGS.append(f"val_caption/{m}")

def load_all_scalars(run_dir):
    files = sorted(glob.glob(f"{run_dir}/tensorboard/**/events.out.tfevents*", recursive=True))
    data = {}  # tag -> list of (step, value)
    for f in files:
        ea = event_accumulator.EventAccumulator(f, size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            for ev in ea.Scalars(tag):
                data.setdefault(tag, []).append((ev.step, ev.value))
    # dedup by step (keep last occurrence in file order = later file wins if overlapping)
    for tag in data:
        d = {}
        for step, val in data[tag]:
            d[step] = val
        data[tag] = sorted(d.items())
    return data

def summarize(run_dir):
    data = load_all_scalars(run_dir)
    if not data:
        return None
    max_step = max(s for tag in data.values() for s,_ in tag)
    out = {"max_step": max_step, "metrics": {}, "peak": {}}
    for tag in TAGS:
        series = data.get(tag)
        if not series:
            out["metrics"][tag] = None
            continue
        if tag.startswith("val_retrieval_itc"):
            tail = [v for s,v in series if s >= max_step - 15000]
            out["metrics"][tag] = sum(tail)/len(tail) if tail else None
        else:
            # last-epoch value = point at max step (or closest below)
            last = series[-1][1]
            out["metrics"][tag] = last
        # peak (for anomaly check on r_mean-like/CIDEr metrics)
        vals = [v for s,v in series]
        out["peak"][tag] = max(vals)
    return out

results = {}
for label, run_dir in RUNS.items():
    results[label] = summarize(run_dir)

print(json.dumps(results, indent=2))
