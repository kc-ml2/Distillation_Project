import json
METS=["Bleu_1","Bleu_2","Bleu_3","Bleu_4","METEOR","ROUGE_L","CIDEr"]
runs=[("solo","small_solo_ep19"),("ITC+LM","itc_lm_ep19"),("base_nd","base_nodecay_ep19"),("teacher","large")]
d={}
for n,dir_ in runs:
    j=json.loads(open(f"output/caption_zeroshot/{dir_}/evaluate.txt").read().strip().splitlines()[-1])
    d[n]={m:j[f"test_{m}"] for m in METS}
print("티처 대비 회복률 (student/teacher):")
print(f"{'':9s}"+"".join(f"{m.replace('Bleu_','B'):>8s}" for m in METS))
for n in ['solo','ITC+LM','base_nd']:
    print(f"{n:9s}"+"".join(f"{d[n][m]/d['teacher'][m]*100:7.1f}%" for m in METS))
print("\nBLEU 차수별 티처와의 절대격차 (teacher−ITC+LM):")
for m in ["Bleu_1","Bleu_2","Bleu_3","Bleu_4"]:
    print(f"  {m}: {d['teacher'][m]-d['ITC+LM'][m]:.4f}")
