"""Keep asynchronous accumulator state in named PTX registers, avoiding CUDA
SSA loop copies while WGMMA groups are in flight. Copy out only after wait0.
"""
from pathlib import Path
p=Path(__file__).resolve().parent
regs=','.join('pipeline_acc%d'%i for i in range(32))
helper='''TMN_DEVI void mma_pipeline(uint64_t a,uint64_t b,int accumulate){asm volatile("{.reg .pred p;setp.ne.b32 p,%%2,0;wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%s},%%0,%%1,p,1,1,1,0;}"::"l"(a),"l"(b),"r"(accumulate):"memory");}
'''%regs
for base in ['front_ring96_cache3','front_prefetch_lnpair_storepipe','front_ring_paircta_u8']:
 for unroll in [False,True]:
  src=base+'_mmapipeline'+('_unroll' if unroll else '')
  s=(p/(src+'.cu')).read_text();pos=s.index('TMN_DEVI void input_role');s=s[:pos]+helper+s[pos:]
  a=s.index('TMN_DEVI void input_role');b=s.index('TMN_DEVI void reduce_at',a);v=s[a:b]
  v=v.replace('{\n int split','{\n asm volatile(".reg .f32 pipeline_acc<32>;");\n int split',1)
  assert '.reg .f32 pipeline_acc<32>' in v
  v=v.replace('float acc[32]={};','float acc[32];').replace('if(h==0){fence_regs(acc);wgmma_fence();}','if(h==0)wgmma_fence();')
  v=v.replace('mma_input64(acc,','mma_pipeline(').replace('wgmma_wait<0>();fence_regs(acc);','wgmma_wait<0>();')
  marker='  // B10 outputs BF16 dx_n.';assert marker in v
  copies='\n'.join(' asm volatile("mov.f32 %%0,pipeline_acc%d;":"=f"(acc[%d])::"memory");'%(i,i) for i in range(32))+'\n'
  v=v.replace(marker,copies+marker)
  s=s[:a]+v+s[b:]
  name=base+'_namedpipe'+('_unroll' if unroll else '');(p/(name+'.cu')).write_text(s);(p/(name+'.launch.json')).write_text((p/(src+'.launch.json')).read_text())
