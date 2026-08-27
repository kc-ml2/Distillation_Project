from pptx import Presentation
p=Presentation("발표자료/파워포인트_템플릿.pptx")
# check which layouts have non-placeholder decorative shapes, esp on the right half (L>5in)
for i,lay in enumerate(p.slide_layouts):
    deco=[]
    for sh in lay.shapes:
        if sh.is_placeholder: continue
        try:
            L=sh.left/914400; T=sh.top/914400; W=sh.width/914400; H=sh.height/914400
        except: 
            L=T=W=H=0
        side = "RIGHT" if L>=5 else ("left" if L+W<=5 else "mid")
        deco.append(f"{sh.shape_type}|{side}|L{L:.1f}T{T:.1f}W{W:.1f}H{H:.1f}|{sh.name}")
    if deco:
        print(f"[{i}] {lay.name}: "+ " ; ".join(deco))
