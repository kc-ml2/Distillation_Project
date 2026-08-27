import json
METS=["Bleu_4","METEOR","ROUGE_L","CIDEr"]
runs=[("solo","small_solo_ep19"),("LM","small_distill_ep19"),("ITC+LM","itc_lm_ep19"),
      ("base_nd","base_nodecay_ep19"),("teacher","large")]
d={}
for n,dir_ in runs:
    j=json.loads(open(f"output/caption_zeroshot/{dir_}/evaluate.txt").read().strip().splitlines()[-1])
    d[n]={m:j[f"test_{m}"] for m in METS}
print(f"{'':8s}"+ "".join(f"{m:>9s}" for m in METS))
for n,_ in runs: print(f"{n:8s}"+"".join(f"{d[n][m]:9.4f}" for m in METS))
print("\n증류 효과 (solo→ITC+LM), 상대%:")
for m in METS:
    a,b=d['solo'][m],d['ITC+LM'][m]; print(f"  {m:8s} +{b-a:.4f}  (+{(b-a)/a*100:4.1f}%)")
print("\n티처 대비 회복률 (student/teacher):")
for n in ['solo','ITC+LM','base_nd']:
    print(f"  {n:8s} "+"  ".join(f"{m}={d[n][m]/d['teacher'][m]*100:4.1f}%" for m in METS))
