from pathlib import Path
import shutil
R=Path(__file__).resolve().parent;P=R.parent/'trimul_recompute_next_20260921'
for name in ['common_recompute.cuh','rs_recompute.cuh','ln_recompute.cuh','b7_packed_dx.inc']:
 shutil.copy2(P/name,R/name)
s=(P/'b7_warp_specialized.cu').read_text()
s=s.replace('#ifndef PIPE_HIDDEN64','''#ifndef SPLIT_ROLE
#define SPLIT_ROLE 0
#endif
#ifndef PIPE_CONSUMERS
#define PIPE_CONSUMERS 2
#endif
#ifndef DX_CTAS
#define DX_CTAS 132
#endif
#ifndef DW_PROD_REGS
#define DW_PROD_REGS 40
#endif
#ifndef DW_CONS_REGS
#define DW_CONS_REGS 232
#endif
constexpr int DW_THREADS=128*(1+PIPE_CONSUMERS);
constexpr int THREADS=SPLIT_ROLE==2?256:DW_THREADS;
constexpr int PIPE_SG_BASE=PIPE_CONSUMERS*40960,PIPE_W_BASE=PIPE_SG_BASE+PIPE_CONSUMERS*8192;
#ifndef PIPE_HIDDEN64''',1)
s=s.replace('DXCOUNT=UCOUNT-DWCOUNT','DXCOUNT=SPLIT_ROLE?DX_CTAS:UCOUNT-DWCOUNT')
s=s.replace('int split=blockIdx.x-DWCOUNT,','int split=blockIdx.x-(SPLIT_ROLE==2?0:DWCOUNT),',1)
s=s.replace('tma_load_2d(sm+81920+c*8192,&p.x,bar+4,c*64,row);','tma_load_2d(sm+81920+c*8192,USE_SAVED_XN?&p.xn:&p.x,bar+4,c*64,row);')
s=s.replace('  ln_recompute_fragment<8,B7_LN_SERIAL>(xn_frag,gamma,beta,lane,1e-5f);','''#if !USE_SAVED_XN
  ln_recompute_fragment<8,B7_LN_SERIAL>(xn_frag,gamma,beta,lane,1e-5f);
#endif''')
s=s.replace('DW_SPLITS*2*WGRAD_SLICES','DW_SPLITS*PIPE_CONSUMERS*WGRAD_SLICES').replace('group*DW_SPLITS*2*WGRAD_SLICES','group*DW_SPLITS*PIPE_CONSUMERS*WGRAD_SLICES')
a=s.index('extern "C" __global__ __launch_bounds__(384,1)')
s=s[:a]+r'''
// Independent launches specialize resource budgets and grid-wide reductions.
extern "C" __global__ __launch_bounds__(THREADS,(PIPE_CONSUMERS==1 && SPLIT_ROLE==1)?2:1)
void front_b7b12(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t bar[10];__shared__ float gamma[128],beta[128];
 if(threadIdx.x<128){gamma[threadIdx.x]=p.gamma[threadIdx.x];beta[threadIdx.x]=p.beta[threadIdx.x];}
 if(threadIdx.x==0){for(int i=0;i<10;++i)mbar_init(bar+i,1);fence_barrier_init();}named_bar_sync(0,THREADS);
#if SPLIT_ROLE != 2
 if(SPLIT_ROLE==1 || blockIdx.x<DWCOUNT){
  if(threadIdx.x<128){setmaxnreg_dec<DW_PROD_REGS>();
   if(threadIdx.x==0){mbar_arrive_expect_tx(bar+4*PIPE_CONSUMERS,16384);for(int k=0;k<2;++k)tma_load_2d(sm+PIPE_W_BASE+k*8192,&p.pre,bar+4*PIPE_CONSUMERS,k*64,(blockIdx.x%PIPE_GROUPS)*64);}
   pipe_producer(p,sm,bar);
  }else{setmaxnreg_inc<DW_CONS_REGS>();pipe_consumer(p,sm,bar,gamma,beta);}
 }
#endif
#if SPLIT_ROLE != 1
 if(SPLIT_ROLE==2 || blockIdx.x>=DWCOUNT){
#if SPLIT_ROLE == 0
  if(threadIdx.x>=256)setmaxnreg_dec<32>();
  else{setmaxnreg_inc<232>();input_role(p,sm,bar,gamma,beta);}
#else
  input_role(p,sm,bar,gamma,beta);
#endif
 }
#endif
 named_bar_sync(15,THREADS);
 __threadfence();named_bar_sync(15,THREADS);
 if(threadIdx.x==0){atomicAdd(p.counts,1u);while(atomicAdd(p.counts,0u)!=UCOUNT)__nanosleep(32);}named_bar_sync(15,THREADS);
#if SPLIT_ROLE == 1
 for(int i=blockIdx.x*THREADS+threadIdx.x;i<131072;i+=UCOUNT*THREADS)reduce_at(p,i);
#elif SPLIT_ROLE == 2
 for(int i=blockIdx.x*THREADS+threadIdx.x;i<256;i+=UCOUNT*THREADS)reduce_at(p,131072+i);
#else
 for(int i=blockIdx.x*THREADS+threadIdx.x;i<131328;i+=UCOUNT*THREADS)reduce_at(p,i);
#endif
 __threadfence();named_bar_sync(15,THREADS);
 if(threadIdx.x==0&&atomicAdd(p.counts+1,1u)==UCOUNT-1){atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);}
}
'''
(R/'b7_roles.cu').write_text(s)
s=(P/'b7_pipe_dw.inc').read_text()
s=s.replace('2*split','PIPE_CONSUMERS*split').replace('2*DW_SPLITS','PIPE_CONSUMERS*DW_SPLITS').replace('c<2;','c<PIPE_CONSUMERS;').replace('bar+4+','bar+2*PIPE_CONSUMERS+').replace('split*2+cc','split*PIPE_CONSUMERS+cc').replace('sm+81920+','sm+PIPE_SG_BASE+').replace('bar+8','bar+4*PIPE_CONSUMERS').replace('sm+98304+','sm+PIPE_W_BASE+').replace(')*2+cc',')*PIPE_CONSUMERS+cc')
s=s.replace('&p.x,bar+id','USE_SAVED_XN?&p.xn:&p.x,bar+id')
s=s.replace('ln_recompute_fragment<8,B7_LN_SERIAL>(xn,gamma,beta,lane,1e-5f);','''
#if !USE_SAVED_XN
  ln_recompute_fragment<8,B7_LN_SERIAL>(xn,gamma,beta,lane,1e-5f);
#endif''')
(R/'b7_pipe_dw.inc').write_text(s)
