from pathlib import Path
r=Path(__file__).resolve().parent;s=(r/'dual_roles_frag.cu').read_text()
s=s.replace('static_assert(UCOUNT%2==0,"two CTAs per cluster");\nconstexpr int PAIRS=UCOUNT/2;', 'constexpr int DWCOUNT=UCOUNT*DW_RATIO/(DW_RATIO+DX_RATIO);\nconstexpr int DXCOUNT=UCOUNT-DWCOUNT;\nstatic_assert(DWCOUNT>0&&DXCOUNT>0,"both CTA roles required");')
s=s.replace('TMN_DEVI MaskCycle mask_cycle(const Params& p,int first)', 'template<int STRIDE> TMN_DEVI MaskCycle mask_cycle(const Params& p,int first)')
a=s.index('template<int STRIDE>');b=s.index('template<bool DW> TMN_DEVI void cluster_b1',a)
s=s[:a]+s[a:b].replace('PAIRS','STRIDE')+s[b:]
a=s.index('TMN_DEVI void cluster_dw');b=s.index('// DX: two',a)
x=s[a:b].replace('int split=blockIdx.x/2','int split=blockIdx.x').replace('mask_cycle(p,split)','mask_cycle<DWCOUNT>(p,split)').replace('PAIRS','DWCOUNT')
s=s[:a]+x+s[b:]
a=s.index('TMN_DEVI void cluster_dx');b=s.index('extern "C" __global__',a)
x=s[a:b].replace('int split=blockIdx.x/2','int split=blockIdx.x-DWCOUNT').replace('mask_cycle(p,split)','mask_cycle<DXCOUNT>(p,split)').replace('PAIRS','DXCOUNT')
s=s[:a]+x+s[b:]
s=s.replace('const bool dw=(blockIdx.x&1)==0;int split=blockIdx.x/2;', 'const bool dw=blockIdx.x<DWCOUNT;int split=dw?blockIdx.x:blockIdx.x-DWCOUNT;int stride=dw?DWCOUNT:DXCOUNT;')
a=s.index('extern "C" __global__');b=s.index('#if PART_ONLY == 2',a)
s=s[:a]+s[a:b].replace('PAIRS','stride')+s[b:]
s=s.replace('for(int b=0;b<PAIRS;++b)v+=reinterpret_cast<volatile float*>(p.partw)[(tile*PAIRS+b)', 'for(int b=0;b<DWCOUNT;++b)v+=reinterpret_cast<volatile float*>(p.partw)[(tile*DWCOUNT+b)')
s=s.replace('for(int b=0;b<PAIRS;++b)v+=reinterpret_cast<volatile float*>(p.partln)', 'for(int b=0;b<DXCOUNT;++b)v+=reinterpret_cast<volatile float*>(p.partln)')
s=s.replace('for(int b=0;b<PAIRS;++b)v+=p.partw[(tile*PAIRS+b)', 'for(int b=0;b<DWCOUNT;++b)v+=p.partw[(tile*DWCOUNT+b)')
s=s.replace('for(int b=0;b<PAIRS;++b)v+=p.partln', 'for(int b=0;b<DXCOUNT;++b)v+=p.partln')
assert 'PAIRS' not in s
# These independent roles do not use a CUDA cluster or multicast. Remove the
# unused wrapper and stale cluster calls so source describes actual execution.
a=s.index('// Upstream has no multicast wrapper.');b=s.index('template<bool DW> TMN_DEVI void cluster_issue',a)
s=s[:a]+'// Independent CTA roles; each barrier is armed before its own TMA loads.\n'+s[b:]
s=s.replace('auto cluster=cg::this_cluster();','').replace('#include <cooperative_groups.h>\nnamespace cg=cooperative_groups;\n','')
s=s.replace('// Both consumers finished and armed barriers before multicast reuse.','// CTA consumers finished before refilling this independent slot.')
s=s.replace('// DX store completion protects tri reuse; join DW before multicast.','// DX store completion protects tri reuse within this independent CTA.')
s=s.replace('// Both destinations install byte counts before any multicast completion.','// Publish initialized slot barriers and LN sums before issuing local TMA.')
for dw,dx in [(1,1),(1,2),(1,3),(2,1)]:
 name=f'dual_ratio{dw}{dx}';src=f'// Independent CTA roles, DW:DX={dw}:{dx}. B1-B4 remains one cooperative launch.\n#define DW_RATIO {dw}\n#define DX_RATIO {dx}\n'+s[s.index('#include "dual_primitives.cuh"'):]
 (r/(name+'.cu')).write_text(src)
 (r/(name+'.py')).write_text('from dual_experiment import Experiment\nclass Plan(Experiment):\n    def __init__(self,*args,**kwargs):\n        super().__init__(*args,source="'+name+'",**kwargs)\n')
