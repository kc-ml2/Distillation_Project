import json, os
eps=[2,5,7,10,13,16,19]
def cider(model,ep):
    p=f"output/caption_zeroshot/{model}_ep{ep:02d}/evaluate.txt"
    if not os.path.exists(p): return None
    d=json.loads(open(p).readline()); return d.get("test_CIDEr")
print(f"{'ep':>4} | {'solo':>7} | {'distill':>8} | {'Δ(dist-solo)':>12}")
for ep in eps:
    s=cider("small_solo",ep); d=cider("small_distill",ep)
    if s is None or d is None: print(f"{ep:>4} |  missing"); continue
    print(f"{ep:>4} | {s:7.4f} | {d:8.4f} | {d-s:+12.4f}")
