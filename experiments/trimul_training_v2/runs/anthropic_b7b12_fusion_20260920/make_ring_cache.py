from pathlib import Path
p=Path(__file__).resolve().parent;src=(p/'front_ring96_pipe_earlyfree.cu').read_text()
helper=''
for name,pol in [('last','evict_last'),('first','evict_first')]:
 helper+='''TMN_DEVI void tma_%s(void* dst,const CUtensorMap* map,uint64_t* bar,int c0,int c1){asm volatile("{.reg .b64 policy;createpolicy.fractional.L2::%s.b64 policy,1.0;cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%%0],[%%1,{%%3,%%4}],[%%2],policy;}"::"r"(smem_u32(dst)),"l"(map),"r"(smem_u32(bar)),"r"(c0),"r"(c1):"memory");}\n'''%(name,pol)
helper+='''TMN_DEVI void ring_store_last(const CUtensorMap* map,const void* s,int c,int r){asm volatile("{.reg .b64 policy;createpolicy.fractional.L2::evict_last.b64 policy,1.0;cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.L2::cache_hint [%0,{%2,%3}],[%1],policy;}"::"l"(map),"r"(smem_u32(s)),"r"(c),"r"(r):"memory");}\n'''
a=src.index('TMN_DEVI void mma_weight64');src=src[:a]+helper+src[a:]
for mode in [1,2,3,4]:
 s=src.replace('store2d(&p.ring,','ring_store_last(&p.ring,')
 if mode>=2:s=s.replace('tma_load_2d(s+16384,&p.ring','tma_last(s+16384,&p.ring').replace('tma_load_2d(sm+81920+h*8192,&p.ring','tma_last(sm+81920+h*8192,&p.ring')
 if mode>=3:
  s=s.replace('tma_load_2d(s,&p.pre','tma_first(s,&p.pre').replace('tma_load_2d(s+16384,side?&p.dr:&p.dl','tma_first(s+16384,side?&p.dr:&p.dl')
 if mode>=4:
  s=s.replace('tma_load_2d(sm+k*8192,&p.dg','tma_first(sm+k*8192,&p.dg').replace('tma_load_2d(sm+c*8192,&p.x','tma_first(sm+c*8192,&p.x').replace('tma_load_2d(sm+16384+c*8192,&p.res','tma_first(sm+16384+c*8192,&p.res')
 name='front_ring96_cache%d'%mode;(p/(name+'.cu')).write_text(s);(p/(name+'.launch.json')).write_text((p/'front_ring96_pipe_earlyfree.launch.json').read_text())
