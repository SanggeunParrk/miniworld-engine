"""Reduce WGMMA fences/commits and interleave independent accumulator chains."""
from pathlib import Path
r=Path(__file__).resolve().parent
base=(r/'dual_balanced.cu').read_text()

def dw(s):
 a=s.index('  static_for<3>([&](auto ni){constexpr int n=',s.index('TMN_DEVI void weight_role'))
 b=s.index('  bool next=',a)
 return s[:a]+'''  // All three independent accumulator sets are ready before the one fence.
  fence_regs(acc[0]);fence_regs(acc[1]);fence_regs(acc[2]);wgmma_fence();
  static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;
   static_for<3>([&](auto ni){constexpr int n=decltype(ni)::value;int t=wi*3+n;
    uint8_t* sa=t<2?s+49152+t*8192:s+((t-2)/2)*8192;
    uint8_t* sb=t<2?s+16384:s+65536+((t-2)%2)*16384;
    mma_ss128(acc[n],smem_desc(smem_u32(sa+k*2048),16,1024,1),smem_desc(smem_u32(sb+k*2048),8192,1024,1),it>split||k>0);
   });
  });
  wgmma_commit();wgmma_wait<0>();
  fence_regs(acc[0]);fence_regs(acc[1]);fence_regs(acc[2]);
'''+s[b:]

def dx(s):
 a=s.index(' static_for<2>([&](auto ni)',s.index('TMN_DEVI void dual_dgrad'))
 b=s.index('  static_for<4>([&](auto qi)',a)
 return s[:a]+''' float both[2][32]={};
 fence_regs(both[0]);fence_regs(both[1]);wgmma_fence();
 static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;
  static_for<2>([&](auto ni){constexpr int nl=decltype(ni)::value;int n=wi*2+nl;
   uint8_t* sw=sm+163840+n*16384;
   mma_dgrad(both[nl],smem_desc(smem_u32(sm+slot*98304+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sw+(k/4)*8192+(k%4)*32),16,1024,1),k>0);
  });
 });wgmma_commit();wgmma_wait<0>();fence_regs(both[0]);fence_regs(both[1]);
 static_for<2>([&](auto ni){constexpr int nlocal=decltype(ni)::value;int n=wi*2+nlocal;
  auto& acc=both[nlocal];
'''+s[b:]

for name,s in [('dual_mma_dw',dw(base)),('dual_mma_dx',dx(base)),('dual_mma_both',dx(dw(base)))]:
 (r/(name+'.cu')).write_text('// Experiment: '+name+'\n'+s)
 (r/(name+'.py')).write_text('from dual_experiment import Experiment\nclass Plan(Experiment):\n    def __init__(self,d,dy,saved,count=132,part=2):\n        super().__init__(d,dy,saved,count,part,'+repr(name)+')\n')
