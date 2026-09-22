"""TMA multicast experiment: adjacent CTAs share invariant weights/saved x_n.

Every participating CTA sets transaction bytes before a cluster barrier; only
rank0 issues multicast. Each consumer completes its MMA before the next shared
stage reuse. Training bucket tile counts and partitions are even, so paired
CTAs take exactly the same number of rounds. No dynamic input queue is allowed.
"""
from pathlib import Path
import json
p=Path(__file__).resolve().parent
helper='''TMN_DEVI void cluster_sync_pair(){asm volatile("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;":::"memory");}
TMN_DEVI void tma_pair(void* dst,const CUtensorMap* map,uint64_t* bar,int c0,int c1){
 if(blockIdx.x&1)return;
 asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0],[%1,{%3,%4}],[%2],%5;"::"r"(smem_u32(dst)),"l"(map),"r"(smem_u32(bar)),"r"(c0),"r"(c1),"h"((unsigned short)3):"memory");
}
'''
for base in ['front_ring96_cache3','front_prefetch_lnpair_storepipe']:
 for mode in ['weights','xn','both']:
  s=(p/(base+'.cu')).read_text();pos=s.index('TMN_DEVI void mma_weight64');s=s[:pos]+helper+s[pos:]
  names=['load_g','load_p','issue_gate'] if mode=='weights' else ['load_dw'] if mode=='xn' else ['load_dw','load_g','load_p','issue_gate']
  for name in names:
   a=s.index('TMN_DEVI void '+name+'(');b=s.index('\n}',a)+2;f=s[a:b]
   assert 'if(threadIdx.x)return;' in f
   f=f.replace('if(threadIdx.x)return;','',1)
   import re
   pattern=r'mbar_arrive_expect_tx\(([^;]+)\);'
   f,n=re.subn(pattern,r'if(threadIdx.x==0)mbar_arrive_expect_tx(\1);cluster_sync_pair();if(threadIdx.x)return;',f,count=1)
   assert n==1
   if name=='load_dw':f=f.replace('tma_load_2d(s+24576+n*8192,&p.xn','tma_pair(s+24576+n*8192,&p.xn')
   if name=='load_g':f=f.replace('tma_load_2d(s+24576+n*8192,side?&p.wrg:&p.wlg','tma_pair(s+24576+n*8192,side?&p.wrg:&p.wlg')
   if name=='load_p':f=f.replace('tma_load_2d(s+n*8192,side?&p.wr:&p.wl','tma_pair(s+n*8192,side?&p.wr:&p.wl')
   if name=='issue_gate':f=f.replace('tma_load_2d(sm+16384+n*16384+k*8192,&p.wgate','tma_pair(sm+16384+n*16384+k*8192,&p.wgate')
   s=s[:a]+f+s[b:]
  s=s.replace('extern "C" __global__ __launch_bounds__(256,2)','extern "C" __global__ __cluster_dims__(2,1,1) __launch_bounds__(256,2)')
  old='fence_barrier_init();}allsync();';assert old in s;s=s.replace(old,old+'cluster_sync_pair();')
  old='#if PART_ONLY == 2\n';assert old in s;s=s.replace(old,old+' cluster_sync_pair();\n')
  name=base+'_mcast_'+mode;(p/(name+'.cu')).write_text(s)
  cfg=json.loads((p/(base+'.launch.json')).read_text());cfg['cluster_size']=2
  (p/(name+'.launch.json')).write_text(json.dumps(cfg))
