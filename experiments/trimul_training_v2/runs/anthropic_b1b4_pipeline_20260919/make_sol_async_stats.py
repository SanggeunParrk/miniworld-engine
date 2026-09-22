from pathlib import Path
r=Path(__file__).resolve().parent
base=(r/'dual_ln_prefetch.cu').read_text()
before=base.replace('  mbar_wait(full+slot,(round/2)&1);\n  prefetch_stats(p,sm,it,slot);',
                    '  prefetch_stats(p,sm,it,slot);\n  mbar_wait(full+slot,(round/2)&1);')
assert before!=base
a=before.index('TMN_DEVI void prefetch_stats')
b=before.index('// DX: two64KiB',a)
asyn=before[:a]+'''TMN_DEVI void prefetch_stats(const Params& p,uint8_t* sm,int tile,int slot){
 if(threadIdx.x<16){
  int r=tile*64+threadIdx.x*4;
  uint32_t dst=smem_u32(sm+73728+slot*512+threadIdx.x*16);
  asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(dst),"l"(p.mean+r):"memory");
  asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(dst+256),"l"(p.rs+r):"memory");
 }
 asm volatile("cp.async.commit_group;":::"memory");
}
'''+before[b:]
old=' fence_proxy_async();allsync();'
assert asyn.count(old)==1
asyn=asyn.replace(old,''' // Only DX has issued cp.async statistics copies. Every issuer waits for
 // its own group, then B1's existing CTA barrier publishes the completed data.
 if constexpr(!DW)asm volatile("cp.async.wait_group 0;":::"memory");
 fence_proxy_async();allsync();''')
for name,s in [('dual_ln_early',before),('dual_ln_async',asyn)]:
 (r/(name+'.cu')).write_text('// Experiment: '+name+'\n'+s)
 (r/(name+'.py')).write_text('from dual_experiment import Experiment\nclass Plan(Experiment):\n    def __init__(self,d,dy,saved,count=132,part=2):\n        super().__init__(d,dy,saved,count,part,'+repr(name)+')\n')
