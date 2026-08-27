import json, openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.label import DataLabelList

METS = ["Bleu_1","Bleu_2","Bleu_3","Bleu_4","METEOR","ROUGE_L","CIDEr"]  # SPICE 제외 = 7개

# label, zeroshot dir (test 지표 사용)
ROWS = [
    ("small solo (no distill)", "small_solo_ep19"),
    ("small + LM distill",     "small_distill_ep19"),
    ("small + ITC+LM distill", "itc_lm_ep19"),
    ("base_nodecay (ours)",    "base_nodecay_ep19"),
    ("BLIP-large (teacher)",   "large"),
]

def read_test(d):
    line = open(f"output/caption_zeroshot/{d}/evaluate.txt").read().strip().splitlines()[-1]
    j = json.loads(line)
    return {m: j[f"test_{m}"] for m in METS}

data = [(label, read_test(d)) for label, d in ROWS]

wb = openpyxl.Workbook(); ws = wb.active; ws.title = "test ep19"
hdr = PatternFill("solid", fgColor="2A78D6"); white = Font(color="FFFFFF", bold=True)
thin = Side(style="thin", color="DEDCD6"); border = Border(thin,thin,thin,thin)
center = Alignment(horizontal="center")

ws.cell(1,1,"Model (ep19, test)")
for j,m in enumerate(METS): ws.cell(1,2+j,m)
for c in range(1,2+len(METS)):
    cell=ws.cell(1,c); cell.fill=hdr; cell.font=white; cell.alignment=center; cell.border=border
for i,(label,vals) in enumerate(data):
    r=2+i; ws.cell(r,1,label).border=border
    for j,m in enumerate(METS):
        cell=ws.cell(r,2+j, round(vals[m],4)); cell.alignment=center; cell.border=border
        if m=="CIDEr": cell.font=Font(bold=True)
ws.column_dimensions['A'].width=24
for col in "BCDEFGH": ws.column_dimensions[col].width=9

# 지표별 막대그래프 7개: 각 그래프 = 한 지표, x축=모델 4개
nrows=len(data); anchor_row=2
for j,m in enumerate(METS):
    ch=BarChart(); ch.type="col"; ch.title=m; ch.height=8; ch.width=13
    datacol=2+j
    vals=Reference(ws, min_col=datacol, min_row=1, max_row=1+nrows)  # include header as series title
    cats=Reference(ws, min_col=1, min_row=2, max_row=1+nrows)
    ch.add_data(vals, titles_from_data=True); ch.set_categories(cats)
    # 축 강제 표시 (openpyxl 기본 delete=True면 라벨이 안 뜸)
    ch.x_axis.delete=False; ch.y_axis.delete=False
    ch.x_axis.title="Model"; ch.y_axis.title=m
    ch.x_axis.tickLblPos="low"
    ch.legend=None                 # 단일 시리즈 → x축 모델명으로 구분
    ch.gapWidth=30                 # 막대 간격 좁게 (기본 150)
    ch.dataLabels=DataLabelList(); ch.dataLabels.showVal=True; ch.dataLabels.numFmt="0.000"
    ch.y_axis.majorGridlines=None
    col_letter = "J" if j%2==0 else "T"
    ws.add_chart(ch, f"{col_letter}{anchor_row + (j//2)*17}")

out="docs/experiments/caption_scores_ep19_test.xlsx"
wb.save(out); print("saved", out)
for label,vals in data: print(f"  {label:24s} CIDEr={vals['CIDEr']:.4f}")
