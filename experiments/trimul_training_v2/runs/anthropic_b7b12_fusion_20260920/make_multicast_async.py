from pathlib import Path
import json
p=Path(__file__).resolve().parent
helper='''TMN_DEVI void pair_ready(uint64_t* bar,int phase){
 uint32_t remote;asm("mapa.shared::cluster.u32 %0,%1,0;":"=r"(remote):"r"(smem_u32(bar)));
 asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _,[%0];"::"r"(remote):"memory");
 if(!(blockIdx.x&1))mbar_wait(bar,phase);
}
'''
for base in ['front_ring96_cache3','front_prefetch_lnpair_storepipe']:
 for mode in ['weights','xn','both']:
  src=base+'_mcast_'+mode;s=(p/(src+'.cu')).read_text();pos=s.index('TMN_DEVI void mma_weight64');s=s[:pos]+helper+s[pos:]
  names=['load_g','load_p','issue_gate'] if mode=='weights' else ['load_dw'] if mode=='xn' else ['load_dw','load_g','load_p','issue_gate']
  for name in names:
   a=s.index('TMN_DEVI void '+name+'(');b=s.index('\n}',a)+2;f=s[a:b]
   phase='(row/(64*DW_SPLITS)/2)&1' if name=='load_dw' else '(row/64/DXCOUNT)&1' if name=='issue_gate' else 'h/2'
   bar='bar+4' if name=='issue_gate' else 'b+slot+4'
   f=f.replace('{\n','{\n if(threadIdx.x)return;',1).replace('if(threadIdx.x==0)mbar_arrive_expect_tx','mbar_arrive_expect_tx').replace('cluster_sync_pair();if(threadIdx.x)return;',f'pair_ready({bar},{phase});')
   s=s[:a]+f+s[b:]
  s=s.replace('__shared__ uint64_t bar[4]','__shared__ uint64_t bar[8]').replace('for(int i=0;i<4;++i)mbar_init(bar+i,1);','for(int i=0;i<4;++i){mbar_init(bar+i,1);mbar_init(bar+i+4,2);}')
  name=base+'_mcastasync_'+mode;(p/(name+'.cu')).write_text(s);(p/(name+'.launch.json')).write_text((p/(src+'.launch.json')).read_text())
