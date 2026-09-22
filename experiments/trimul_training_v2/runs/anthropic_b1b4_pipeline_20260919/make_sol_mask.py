"""Decode cached dropout bits directly into float scales, preserving y*ds order."""
from pathlib import Path
r=Path(__file__).resolve().parent
base=(r/'dual_balanced.cu').read_text()
start=base.index('  if(mask.cached){uint32_t* dd=')
end=base.index('  uint32_t *yy=',start)
s=base[:start]+'''  if(!mask.cached)ds=ldg128(p.ds+jr*128+cb*64+c);
'''+base[end:]
old='float ya=bf16lo(yy[q])*bf16lo(dd[q]),yb=bf16hi(yy[q])*bf16hi(dd[q]),ga=bf16lo(gg[q]),gb=bf16hi(gg[q]);'
new='''int bit=8*(i/256)+2*q;
   float scale=__uint_as_float(mask.scale<<16);
   float da=mask.cached?(((bits>>bit)&1)?scale:0.f):bf16lo(dd[q]);
   float db=mask.cached?(((bits>>(bit+1))&1)?scale:0.f):bf16hi(dd[q]);
   float ya=bf16lo(yy[q])*da,yb=bf16hi(yy[q])*db,ga=bf16lo(gg[q]),gb=bf16hi(gg[q]);'''
assert old in s
s=s.replace(old,new)
(r/'dual_mask_float.cu').write_text('// Experiment: no BF16 repack for cached mask.\n'+s)
(r/'dual_mask_float.py').write_text('from dual_experiment import Experiment\nclass Plan(Experiment):\n    def __init__(self,d,dy,saved,count=132,part=2):\n        super().__init__(d,dy,saved,count,part,"dual_mask_float")\n')
