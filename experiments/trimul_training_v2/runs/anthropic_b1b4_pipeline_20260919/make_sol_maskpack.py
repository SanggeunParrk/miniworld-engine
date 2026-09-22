"""Pack 8 positive BF16 dropout scales with byte predicates, without a q loop.

Contract: ds is zero or BF16(1/(1-p)), p in[0,1). A nonzero ds therefore
has a nonzero high byte. Scale is still derived from the actual input.
"""
from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'dual_ln_prefetch.cu').read_text()
old='''   uint4 ds=ldg128(p.ds+jr*128+cb*64+c);uint32_t* dd=reinterpret_cast<uint32_t*>(&ds);
#pragma unroll
   for(int q=0;q<4;++q){uint32_t lo=dd[q]&65535u,hi=dd[q]>>16;int bit=8*(i/256)+2*q;
    bits|=(uint32_t(lo!=0)<<bit)|(uint32_t(hi!=0)<<(bit+1));v.scale|=lo|hi;}
'''
new='''   uint4 ds=ldg128(p.ds+jr*128+cb*64+c);
   uint32_t ab=__byte_perm(ds.x,ds.y,0x7531),cd=__byte_perm(ds.z,ds.w,0x7531);
   uint32_t ba=(__vsetne4(ab,0)*0x01020408u)>>24;
   uint32_t bc=(__vsetne4(cd,0)*0x01020408u)>>24;
   bits|=(ba|(bc<<4))<<(8*(i/256));
   uint32_t scale=ds.x|ds.y|ds.z|ds.w;v.scale|=(scale|(scale>>16))&65535u;
'''
assert old in s
s=s.replace(old,new)
(r/'dual_mask_pack.cu').write_text('// Experiment: vector mask bit packing in the CTA prologue.\n'+s)
(r/'dual_mask_pack.py').write_text('from dual_experiment import Experiment\nclass Plan(Experiment):\n    def __init__(self,d,dy,saved,count=132,part=2):\n        super().__init__(d,dy,saved,count,part,"dual_mask_pack")\n')
