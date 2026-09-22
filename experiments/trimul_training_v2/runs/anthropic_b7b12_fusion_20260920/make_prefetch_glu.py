from pathlib import Path
p=Path(__file__).resolve().parent
for base in ['front_prefetch_lnpair','front_prefetch_lnpairall','front_prefetch_lnregpair']:
 for u in [2,4,8]:
  for cache in ([False,True] if base=='front_prefetch_lnpair' else [False]):
   if u==8 and not cache:continue
   s=(p/(base+'.cu')).read_text();a=s.index('TMN_DEVI void glu_small');b=s.index('TMN_DEVI void load_dw',a);f=s[a:b].replace('glu_small(', 'glu_input(').replace('#pragma unroll 8','#pragma unroll %d'%u)
   if cache:f=f.replace('int row){','int row,uint32_t mask){').replace('unsigned tid=threadIdx.x;uint32_t mask=*reinterpret_cast<const uint32_t*>(p.mask+row+(tid%32)*2);','unsigned tid=threadIdx.x;')
   s=s[:b]+f+s[b:];a=s.index('TMN_DEVI void input_role');b=s.index('TMN_DEVI void reduce_at',a);v=s[a:b].replace('glu_small(', 'glu_input(')
   if cache:
    v=v.replace('int row=tile*64;','int row=tile*64;uint32_t tilemask=*reinterpret_cast<const uint32_t*>(p.mask+row+(threadIdx.x%32)*2);')
    v=v.replace('sm+81920+h*8192,row);','sm+81920+h*8192,row,tilemask);')
   s=s[:a]+v+s[b:];name=base+'_dxu%d'%u+('_mask' if cache else '');(p/(name+'.cu')).write_text(s);(p/(name+'.launch.json')).write_text((p/'front_kindprefetch.launch.json').read_text())
