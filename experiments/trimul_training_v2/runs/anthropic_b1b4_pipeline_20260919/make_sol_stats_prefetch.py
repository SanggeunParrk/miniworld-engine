"""Load saved LN statistics before B1; its existing CTA barrier publishes them."""
from pathlib import Path
r=Path(__file__).resolve().parent
base=(r/'dual_balanced.cu').read_text()
s=base
start=s.index(' float *mus=stats+256')
end=s.index(' int ra=w*16+lane/4',start)
s=s[:start]+''' // 73728..74752: double-buffered saved mean/rstd. Gamma is resident
 // at74752..75776. B1's existing CTA barrier publishes the row statistics.
 float* mus=reinterpret_cast<float*>(sm+73728+slot*512);
 float* rss=mus+64;float* gam=reinterpret_cast<float*>(sm+74752);
'''+s[end:]
helper='''TMN_DEVI void prefetch_stats(const Params& p,uint8_t* sm,int tile,int slot){
 if(threadIdx.x>=16)return;
 int r=tile*64+threadIdx.x*4;
 uint4 mu=ldg128(p.mean+r),rs=ldg128(p.rs+r);
 uint32_t dst=smem_u32(sm+73728+slot*512+threadIdx.x*16);
 asm volatile("st.shared.v4.b32 [%0],{%1,%2,%3,%4};"::"r"(dst),"r"(mu.x),"r"(mu.y),"r"(mu.z),"r"(mu.w):"memory");
 asm volatile("st.shared.v4.b32 [%0],{%1,%2,%3,%4};"::"r"(dst+256),"r"(rs.x),"r"(rs.y),"r"(rs.z),"r"(rs.w):"memory");
}
'''
s=s.replace('// DX: two64KiB',helper+'// DX: two64KiB')
old='  gate_backward<false>(p,sm+slot*98304,it*64,mask,mi);'
assert old in s
s=s.replace(old,'  prefetch_stats(p,sm,it,slot);\n'+old)
s=s.replace(' // Publish initialized slot barriers',
            ' if(!dw)reinterpret_cast<float*>(sm+74752)[threadIdx.x]=p.gamma[threadIdx.x];\n // Publish initialized slot barriers')

early=s.replace('  prefetch_stats(p,sm,it,slot);\n','')
early=early.replace('  if(next)issue_slot<false>(p,sm,full,slot,it+2*DXCOUNT,false);',
'''  if(next){issue_slot<false>(p,sm,full,slot,it+2*DXCOUNT,false);prefetch_stats(p,sm,it+2*DXCOUNT,slot);}''')
early=early.replace(' if(!dw)reinterpret_cast<float*>(sm+74752)[threadIdx.x]=p.gamma[threadIdx.x];',
''' if(!dw){
  reinterpret_cast<float*>(sm+74752)[threadIdx.x]=p.gamma[threadIdx.x];
  if(split<p.tiles)prefetch_stats(p,sm,split,0);
  if(split+DXCOUNT<p.tiles)prefetch_stats(p,sm,split+DXCOUNT,1);
 }''')
for name,source in [('dual_ln_prefetch',s),('dual_ln_prefetch2',early)]:
 (r/(name+'.cu')).write_text('// Experiment: '+name+'\n'+source)
 (r/(name+'.py')).write_text('from dual_experiment import Experiment\nclass Plan(Experiment):\n    def __init__(self,d,dy,saved,count=132,part=2):\n        super().__init__(d,dy,saved,count,part,'+repr(name)+')\n')
