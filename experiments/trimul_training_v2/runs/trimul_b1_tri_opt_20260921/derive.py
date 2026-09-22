from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent/'trimul_ln_policy_v4_20260921'
for p in [*P.glob('*.cuh'),*P.glob('*.inc'),P/'b1_fused.cu',P/'replace_plan.py']:(R/p.name).write_bytes(p.read_bytes())
p=R/'b1_fused.cu';s=p.read_text();s=s.replace('#include "lowreg_stats.inc"','TMN_DEVI void load_raw(const Params&,uint8_t*,uint64_t*,int,int);\n#include "lowreg_stats.inc"')
s=s.replace('#endif\n fence_proxy_async();allsync();\n recompute_gemm<256', '#endif\n#if !B1_NO_REDUNDANT_SYNC\n fence_proxy_async();allsync();\n#endif\n recompute_gemm<256')
s=s.replace('if(blockIdx.x<p.tiles)load_raw(p,sm,bars,0,blockIdx.x*64);','if(blockIdx.x<p.tiles)load_raw(p,sm,bars,0,blockIdx.x*64);\n#if B1_PREFETCH_DY\n if(blockIdx.x<p.tiles)load_dy(p,sm,bars+2,blockIdx.x*64);\n#endif')
s=s.replace('  load_dy(p,sm,bars+2,tile*64);','  #if !B1_PREFETCH_DY\n  load_dy(p,sm,bars+2,tile*64);\n  #endif')
s=s.replace('  for(int j=threadIdx.x;j<4096;j+=256)reinterpret_cast<uint32_t*>(sm)[j]=reinterpret_cast<uint32_t*>(norm+32768)[j];\n  fence_proxy_async();allsync();', '''#if !B1_DIRECT_DP
  for(int j=threadIdx.x;j<4096;j+=256)reinterpret_cast<uint32_t*>(sm)[j]=reinterpret_cast<uint32_t*>(norm+32768)[j];
  fence_proxy_async();allsync();
#endif
#if B1_PREFETCH_DY
  if(tile+UCOUNT<p.tiles)load_dy(p,sm,bars+2,(tile+UCOUNT)*64);
#endif''')
s=s.replace('  lowreg_dgrad(p,sm,tile*64,slot);allsync();\n  if(tile+UCOUNT<p.tiles){','  lowreg_dgrad(p,sm,bars,tile*64,slot);allsync();\n  if(!B1_EARLY_RAW && tile+UCOUNT<p.tiles){')
s=s.replace('#include <cuda_fp16.h>', '''#include <cuda_fp16.h>
#ifndef B1_NO_REDUNDANT_SYNC
#define B1_NO_REDUNDANT_SYNC 0
#endif
#ifndef B1_DIRECT_DP
#define B1_DIRECT_DP 0
#endif
#ifndef B1_PREFETCH_DY
#define B1_PREFETCH_DY 0
#endif
#ifndef B1_EARLY_RAW
#define B1_EARLY_RAW 0
#endif
static_assert(!B1_PREFETCH_DY || B1_DIRECT_DP, "dy prefetch requires direct dProj operand");''')
p.write_text(s)
p=R/'lowreg_stats.inc';s=p.read_text().replace('uint8_t* sm,int m0,int slot','uint8_t* sm,uint64_t* bars,int m0,int slot')
s=s.replace(' float* stats=', ' uint8_t* dp=B1_DIRECT_DP?sn+32768:sm;\n float* stats=',1)
s=s.replace('smem_u32(sm+(k/4)*8192+(k%4)*32)','smem_u32(dp+(k/4)*8192+(k%4)*32)')
s=s.replace(' });allsync();\n int c=threadIdx.x;', ''' });allsync();
#if B1_EARLY_RAW
 // dNorm/dProj consumers have finished. The opposite slot can receive
 // the next tri/x_n/stats while the current dTri TMA store completes.
 if(m0/64+UCOUNT<p.tiles)load_raw(p,sm,bars,1-slot,m0+UCOUNT*64);
#endif
 int c=threadIdx.x;''')
p.write_text(s)
print('Created independent B1 optimization candidate')
