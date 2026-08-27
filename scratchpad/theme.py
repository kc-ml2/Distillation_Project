from pptx import Presentation
from pptx.util import Emu
p=Presentation("발표자료/파워포인트_템플릿.pptx")
# theme via first slide master xml
import re
m=p.slide_masters[0]
el=m.element
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
# fonts
for tag in ['majorFont','minorFont']:
    f=el.find('.//a:'+tag+'/a:latin',ns)
    print(tag, f.get('typeface') if f is not None else None)
# theme colors
clrs=el.findall('.//a:clrScheme/*',ns)
print("colors:")
for c in clrs:
    name=c.tag.split('}')[1]
    val=c.find('a:srgbClr',ns) or c.find('a:sysClr',ns)
    v=val.get('val') if val is not None else '?'
    print("   ",name,v)
print("\n=== TITLE layout(0) placeholder geometry ===")
for lay_i in [0,6,7,8]:
    lay=p.slide_layouts[lay_i]
    print(f"[layout {lay_i} {lay.name}]")
    for ph in lay.placeholders:
        try:
            print(f"   idx={ph.placeholder_format.idx} {ph.name!r} L={round(ph.left/914400,2)} T={round(ph.top/914400,2)} W={round(ph.width/914400,2)} H={round(ph.height/914400,2)}")
        except Exception as e:
            print("   idx",ph.placeholder_format.idx,"(no geom)")
