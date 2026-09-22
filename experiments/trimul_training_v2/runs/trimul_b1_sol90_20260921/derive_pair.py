from pathlib import Path
R=Path(__file__).resolve().parent/'pair';P=R.parent/'store';R.mkdir(exist_ok=True)
for p in [*P.glob('*.cuh'),*P.glob('*.inc'),P/'b1_fused.cu',P/'replace_plan.py']:(R/p.name).write_bytes(p.read_bytes())
p=R/'b1_fused.cu';p.write_text(p.read_text().replace('#include <cuda_fp16.h>','#include <cuda_fp16.h>\n#ifndef B1_DN_PAIR\n#define B1_DN_PAIR 0\n#endif'))
p=R/'lowreg_stats.inc';s=p.read_text();start=s.index(' static_for<2>');end=s.index('  static_for<4>',start)
old=s[start:end]
new=''' #if B1_DN_PAIR
 float pair_acc[2][32]={};fence_regs(pair_acc[0]);fence_regs(pair_acc[1]);wgmma_fence();
 static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;
  static_for<2>([&](auto ni){constexpr int nl=decltype(ni)::value;int n=wi*2+nl;uint8_t* sw=sm+147456+n*16384;
   mma_dgrad(pair_acc[nl],smem_desc(smem_u32(dp+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sw+(k/4)*8192+(k%4)*32),16,1024,1),k>0);
  });
 });wgmma_commit();wgmma_wait<0>();fence_regs(pair_acc[0]);fence_regs(pair_acc[1]);
 static_for<2>([&](auto ni){constexpr int nl=decltype(ni)::value;int n=wi*2+nl;auto& acc=pair_acc[nl];
 #else
'''+old+''' #endif
'''
s=s[:start]+new+s[end:];p.write_text(s)
