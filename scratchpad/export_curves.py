import glob, os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/minwoo/Distillation_Project/output"
OUT = "/home/minwoo/Distillation_Project/발표자료/assets/curves"
STEPS_PER_EPOCH = 37346

RUN_DIRS = {
    "base_baseline": "pt_checkpoint_base_logitscale_nodecay",
    "small_baseline": "pt_smallreg_minilm_baseline",
    "ITC_only": "pt_smallreg_minilm_ttm_queue",
    "LM_only": "pt_smallreg_minilm_lm_distill",
    "ITM": "pt_smallreg_minilm_itm_matrix_kd_configfix",
    "ITC+LM_part1": "pt_ttm_queue_lm_holdhalf",
    "ITC+LM_part2": "pt_ttm_queue_lm_holdhalf_tctmp00157",
}

def load_scalar(run_dir, tag):
    """Load (step, value) pairs for tag across all tfevents files under run_dir, merged+sorted."""
    pattern = os.path.join(ROOT, run_dir, "tensorboard", "**", "events.out.tfevents*")
    files = sorted(glob.glob(pattern, recursive=True))
    pairs = {}
    for f in files:
        ea = event_accumulator.EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            continue
        for ev in ea.Scalars(tag):
            pairs[ev.step] = ev.value  # later file overwrites dup step
    return sorted(pairs.items())

def splice_itc_lm(tag):
    p1 = load_scalar(RUN_DIRS["ITC+LM_part1"], tag)
    p2 = load_scalar(RUN_DIRS["ITC+LM_part2"], tag)
    merged = {s: v for s, v in p1 if s < 374000}
    merged.update({s: v for s, v in p2 if s >= 374000})
    return sorted(merged.items())

COLORS = {
    "base_baseline": "#7f7f7f",
    "small_baseline": "#1f77b4",
    "ITC_only": "#2ca02c",
    "LM_only": "#ff7f0e",
    "ITM": "#9467bd",
    "ITC+LM": "#d62728",
}
LABELS = {
    "base_baseline": "base baseline",
    "small_baseline": "small baseline",
    "ITC_only": "ITC",
    "LM_only": "LM",
    "ITM": "ITM",
    "ITC+LM": "ITC+LM",
}

def report(name, series):
    if not series:
        print(f"  {name}: NO DATA")
        return
    print(f"  {name}: n={len(series)} first=({series[0][0]},{series[0][1]:.3f}) last=({series[-1][0]},{series[-1][1]:.3f})")

# ---------- Figure 1: ITC r_mean vs step, all 6 runs ----------
print("=== Figure 1: itc_r_mean_overlay ===")
itc_series = {}
for key in ["base_baseline", "small_baseline", "ITC_only", "LM_only", "ITM"]:
    s = load_scalar(RUN_DIRS[key], "val_retrieval_itc/r_mean")
    itc_series[key] = s
    report(key, s)
itc_lm = splice_itc_lm("val_retrieval_itc/r_mean")
itc_series["ITC+LM"] = itc_lm
report("ITC+LM (spliced)", itc_lm)

# check splice continuity around 374000
near = [(s, v) for s, v in itc_lm if 370000 <= s <= 378000]
print("  splice region (370000-378000):", near)

fig, ax = plt.subplots(figsize=(11, 6.5))
for key in ["base_baseline", "small_baseline", "ITC_only", "LM_only", "ITM", "ITC+LM"]:
    s = itc_series[key]
    if not s:
        continue
    xs = [p[0] for p in s]
    ys = [p[1] * 100 if max(p[1] for p in s) <= 1.5 else p[1] for p in s]
    ax.plot(xs, ys, label=LABELS[key], color=COLORS[key], linewidth=1.8)

base_final = itc_series["base_baseline"][-1][1]
base_final_pct = base_final * 100 if base_final <= 1.5 else base_final
ax.axhline(base_final_pct, color=COLORS["base_baseline"], linestyle="--", alpha=0.4, linewidth=1)

ax.set_title("ITC Retrieval r_mean", fontsize=16)
ax.set_xlabel("step", fontsize=12)
ax.set_ylabel("r_mean (%)", fontsize=12)
ax.legend(fontsize=11, loc="lower right")
ax.grid(alpha=0.25)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "itc_r_mean_overlay.png"), dpi=200)
plt.close(fig)

# ---------- Figure 2: caption CIDEr per epoch ----------
print("=== Figure 2: caption_cider_epoch ===")
cider_series = {}
for key in ["ITC_only"]:
    s = load_scalar(RUN_DIRS[key], "val_caption/CIDEr")
    cider_series[key] = s
    report(key, s)
cider_lm = splice_itc_lm("val_caption/CIDEr")
cider_series["ITC+LM"] = cider_lm
report("ITC+LM (spliced)", cider_lm)
# also check others in case they have the tag
for key in ["base_baseline", "small_baseline", "LM_only", "ITM"]:
    s = load_scalar(RUN_DIRS[key], "val_caption/CIDEr")
    report(key + " (check)", s)
    if s:
        cider_series[key] = s

fig, ax = plt.subplots(figsize=(9, 6))
for key, s in cider_series.items():
    if not s:
        continue
    xs = [p[0] / STEPS_PER_EPOCH for p in s]
    ys = [p[1] for p in s]
    ax.plot(xs, ys, marker="o", label=LABELS.get(key, key), color=COLORS.get(key, None), linewidth=1.8)
ax.set_title("Caption CIDEr (per epoch)", fontsize=16)
ax.set_xlabel("epoch", fontsize=12)
ax.set_ylabel("CIDEr", fontsize=12)
ax.legend(fontsize=11)
ax.grid(alpha=0.25)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "caption_cider_epoch.png"), dpi=200)
plt.close(fig)

# ---------- Figure 3: ITM r_mean per epoch ----------
print("=== Figure 3: itm_r_mean_epoch ===")
itm_series = {}
for key in ["base_baseline", "small_baseline", "ITM"]:
    s = load_scalar(RUN_DIRS[key], "val_retrieval_itm/r_mean")
    itm_series[key] = s
    report(key, s)
itm_lm = splice_itc_lm("val_retrieval_itm/r_mean")
itm_series["ITC+LM"] = itm_lm
report("ITC+LM (spliced)", itm_lm)

fig, ax = plt.subplots(figsize=(9, 6))
for key, s in itm_series.items():
    if not s:
        continue
    xs = [p[0] / STEPS_PER_EPOCH for p in s]
    ys = [(v * 100 if max(vv for _, vv in s) <= 1.5 else v) for _, v in s]
    ax.plot(xs, ys, marker="o", label=LABELS.get(key, key), color=COLORS.get(key, None), linewidth=1.8)
ax.set_title("ITM Retrieval r_mean (per epoch)", fontsize=16)
ax.set_xlabel("epoch", fontsize=12)
ax.set_ylabel("r_mean (%)", fontsize=12)
ax.legend(fontsize=11)
ax.grid(alpha=0.25)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "itm_r_mean_epoch.png"), dpi=200)
plt.close(fig)


# ---------- Figure 1b: ITC r_mean zoom (250k-746920, EMA smoothed) ----------
print("=== Figure 1b: itc_r_mean_overlay_zoom ===")

def ema(ys, span=18):
    alpha = 2 / (span + 1)
    out = []
    m = ys[0]
    for y in ys:
        m = alpha * y + (1 - alpha) * m
        out.append(m)
    return out

LW = {"base_baseline": 2.2, "small_baseline": 2.2, "ITC_only": 1.6, "LM_only": 1.6, "ITM": 1.6, "ITC+LM": 2.6}

fig, ax = plt.subplots(figsize=(11, 6.5))
for key in ["base_baseline", "small_baseline", "ITC_only", "LM_only", "ITM", "ITC+LM"]:
    s = [(x, y) for x, y in itc_series[key] if x >= 250000]
    if not s:
        continue
    xs = [p[0] for p in s]
    ys = [p[1] for p in s]  # already in percent scale (checked in fig1)
    ax.plot(xs, ys, color=COLORS[key], linewidth=1, alpha=0.15)
    ax.plot(xs, ema(ys), color=COLORS[key], linewidth=LW[key], label=LABELS[key])

ax.axhline(base_final_pct, color=COLORS["base_baseline"], linestyle="--", alpha=0.4, linewidth=1)
ax.set_xlim(250000, 746920)
all_zoom_ys = [y for key in itc_series for x, y in itc_series[key] if x >= 250000]
ax.set_ylim(min(all_zoom_ys) - 1, max(all_zoom_ys) + 1)
ax.set_title("ITC Retrieval r_mean (zoom, EMA-smoothed)", fontsize=16)
ax.set_xlabel("step", fontsize=12)
ax.set_ylabel("r_mean (%)", fontsize=12)
ax.legend(fontsize=11, loc="lower right")
ax.grid(alpha=0.25)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "itc_r_mean_overlay_zoom.png"), dpi=200)
plt.close(fig)

print("DONE")
