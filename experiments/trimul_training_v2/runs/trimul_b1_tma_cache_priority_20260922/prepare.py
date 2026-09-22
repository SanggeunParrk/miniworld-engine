from pathlib import Path
R=Path(__file__).resolve().parent;S=R.parent/'trimul_b1_weight_store_late_20260922'
for p in [S/'replace_plan.py',*S.glob('*.cu'),*S.glob('*.cuh'),*S.glob('*.inc')]:
    (R/p.name).write_text(p.read_text())
helper='''// NVIDIA PTX cp.async.bulk.tensor cache-policy hints affect replacement
// priority only; no correctness assumption depends on cache residency.
template<int PRIORITY> TMN_DEVI uint64_t cache_policy(){
 uint64_t v;
 if constexpr(PRIORITY>0)asm volatile("createpolicy.fractional.L2::evict_last.L2::evict_unchanged.b64 %0, 1.0;":"=l"(v));
 else asm volatile("createpolicy.fractional.L2::evict_first.L2::evict_unchanged.b64 %0, 1.0;":"=l"(v));
 return v;
}
template<int PRIORITY> TMN_DEVI void cached_load(void* dst,const CUtensorMap* map,uint64_t* bar,int c,int r){
 if constexpr(PRIORITY==0)tma_load_2d(dst,map,bar,c,r);
 else asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%3,%4}],[%2],%5;"::"r"(smem_u32(dst)),"l"(map),"r"(smem_u32(bar)),"r"(c),"r"(r),"l"(cache_policy<PRIORITY>()):"memory");
}
TMN_DEVI void cached_dtri_store(const CUtensorMap* map,const void* src,int a,int b,int c){
#if B1_TMA_PRIORITY & 8
 asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group.L2::cache_hint [%0,{%2,%3,%4}],[%1],%5;"::"l"(map),"r"(smem_u32(src)),"r"(a),"r"(b),"r"(c),"l"(cache_policy<-1>()):"memory");
#else
 tma_store_3d(map,src,a,b,c);
#endif
}
'''
p=R/'b1_fused.cu';s=p.read_text().replace('#include <cuda_fp16.h>','#include <cuda_fp16.h>\n'+helper)
a=s.index('TMN_DEVI void load_raw(const Params& p');b=s.index('#ifndef GATE_PHASE',a)
part=s[a:b]
part=part.replace('tma_load_2d(dst,&p.x,','cached_load<(B1_TMA_PRIORITY&1)?1:0>(dst,&p.x,')
part=part.replace('tma_load_2d(dst+8192,&p.x,','cached_load<(B1_TMA_PRIORITY&1)?1:0>(dst+8192,&p.x,')
part=part.replace('tma_load_2d(dst+16384,&p.xhat,','cached_load<(B1_TMA_PRIORITY&4)?-1:0>(dst+16384,&p.xhat,')
part=part.replace('tma_load_2d(sm,&p.dy,','cached_load<(B1_TMA_PRIORITY&4)?-1:0>(sm,&p.dy,')
part=part.replace('tma_load_2d(sm+8192,&p.dy,','cached_load<(B1_TMA_PRIORITY&4)?-1:0>(sm+8192,&p.dy,')
s=s[:a]+part+s[b:]
a=s.index('TMN_DEVI void dg_store(');b=s.index('\nstruct FragmentMask',a)
old=s[a:b]
new='''TMN_DEVI void dg_store(const CUtensorMap* map,const void* src,int c,int r){
 asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.L2::cache_hint [%0,{%2,%3}],[%1],%4;"::"l"(map),"r"(smem_u32(src)),"r"(c),"r"(r),"l"(cache_policy<1>()):"memory");
}
'''
s=s[:a]+'#if B1_TMA_PRIORITY & 2\n'+new+'#else\n'+old+'\n#endif\n'+s[b:];p.write_text(s)
p=R/'lowreg_stats.inc';p.write_text(p.read_text().replace('tma_store_3d(','cached_dtri_store('))
t=(S/'tune.py').read_text().replace('B1_LATE_WEIGHT','B1_TMA_PRIORITY').replace("default='0,1,2,3'","default='0,1,2,3,4,7,8,11,15'")
t=t.replace("cfg['defines']['B1_TMA_PRIORITY']=level","cfg['defines']['B1_TMA_PRIORITY']=level;cfg['defines']['B1_LATE_WEIGHT']=3")
(R/'tune.py').write_text(t)
(R/'tune.sbatch').write_text((S/'tune.sbatch').read_text().replace(S.name,R.name).replace('b1-weight-late','b1-tma-priority'))
