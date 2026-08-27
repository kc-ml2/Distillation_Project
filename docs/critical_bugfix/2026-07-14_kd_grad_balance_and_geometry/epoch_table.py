"""Per-epoch table: logit_scale, ITC r_mean, txt_r1, img_r1, loss_val/ita, itc_kd."""
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE = '/home/minwoo/Distillation_Project/output'
RUNS = {
    'base': f'{BASE}/pt_smallreg_minilm_baseline/tensorboard/*',
    'keep': f'{BASE}/pt_smallreg_minilm_itc_distill_tau0.022_keeptau2/tensorboard/*',
    'drop': f'{BASE}/pt_smallreg_minilm_itc_distill_tau0.022_droptau/tensorboard/*',
}
EPOCH = 37346
TAGS = ['model/logit_scale', 'val_retrieval_itc/r_mean', 'val_retrieval_itc/txt_r_mean',
        'val_retrieval_itc/img_r_mean', 'loss_val/ita', 'loss_train/itc_kd']

def series(run_glob, tag):
    pts = []
    for d in sorted(glob.glob(run_glob)):
        ea = EventAccumulator(d, size_guidance={'scalars': 0})
        ea.Reload()
        if tag in ea.Tags().get('scalars', []):
            pts += [(e.step, e.value) for e in ea.Scalars(tag)]
    return sorted(set(pts))

def at(pts, step):
    best = None
    for s, v in pts:
        if best is None or abs(s - step) < abs(best[0] - step):
            best = (s, v)
    return best[1] if best and abs(best[0] - step) <= EPOCH // 2 else float('nan')

cache = {(r, t): series(g, t) for r, g in RUNS.items() for t in TAGS}
eps = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
for tag in TAGS:
    print(f"\n### {tag}")
    print("ep   " + "".join(f"{e:>9}" for e in eps))
    for r in RUNS:
        vals = [at(cache[(r, tag)], e * EPOCH) for e in eps]
        print(f"{r:<5}" + "".join(f"{v:>9.3f}" for v in vals))
