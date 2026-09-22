"""Three WGs / CTA: two original C64 MMA consumers, one TMA producer.

Initial quota80*384=30720 registers; dW104+104+32 and dX32+104+104
both fit. Restore only after all384 threads rendezvous.
"""
from pathlib import Path
import json
R=Path(__file__).resolve().parent
base=(R/'front_ring96_cache3.cu').read_text()
prefix=base[:base.index('TMN_DEVI void load_g(')]
prefix=prefix.replace('#pragma unroll 8','#pragma unroll 2')
# dW has256 active threads; the third WG is idle.
prefix=prefix.replace('allsync();','named_bar_sync(1,256);')
old='mma_weight128(acc,smem_desc(smem_u32(s+40960+wi*8192+k*32),16,1024,1),smem_desc(smem_u32(s+24576+k*2048),8192,1024,1),r%segmentRounds>0||k>0);'
assert old in prefix
prefix=prefix.replace(old,'''static_for<2>([&](auto ni){constexpr int n=decltype(ni)::value;
   mma_weight64(*reinterpret_cast<float(*)[32]>(acc+n*32),
    smem_desc(smem_u32(s+40960+wi*8192+k*32),16,1024,1),
    smem_desc(smem_u32(s+24576+n*8192+k*2048),16,1024,1),r%segmentRounds>0||k>0);
   });''')
ws=(R/'ring_ws_input.cuh').read_text()
producer=ws[:ws.index('TMN_DEVI void ws_consumer')]
consumer=r'''
TMN_DEVI void ws_consumer(const Params& p,uint8_t* sm,WsBarriers* b,const float* gamma){
 int tid=threadIdx.x-128,split=blockIdx.x-DWCOUNT,wi=tid/128,lane=tid%32,w=(tid/32)%4;
 int ra=w*16+lane/4,rb=ra+8,round=0;float running=0;
 for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){
  int row=tile*64;uint32_t gate_packed[16];float acc[32]={};
  mbar_wait(b->tx+4,round&1);
  {float gate[32]={};uint8_t* s=sm+WS_GATE;
   fence_regs(gate);wgmma_fence();
   static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;
    mma_dgrad(gate,smem_desc(smem_u32(s+(k/4)*8192+(k%4)*32),16,1024,1),
      smem_desc(smem_u32(s+16384+(k/4)*16384+wi*8192+(k%4)*32),16,1024,1),k>0);
   });wgmma_commit();wgmma_wait<0>();fence_regs(gate);
   static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;
    gate_packed[q]=pack_bf16(gate[q*2],gate[q*2+1]);});
  }
  if(tid%128==0)mbar_arrive(b->empty+4);
  for(int step=0;step<16;++step){
   int slot=step%WS_STAGES,epoch=step/WS_STAGES;uint8_t* s=sm+slot*WS_SLOT;
   mbar_wait(b->tx+slot,epoch&1);fence_regs(acc);wgmma_fence();
   static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;
    mma_input64(acc,smem_desc(smem_u32(s+k*2048),16,1024,1),
      smem_desc(smem_u32(s+8192+wi*8192+k*32),16,1024,1),step>0||k>0);
   });wgmma_commit();wgmma_wait<0>();fence_regs(acc);
   if(tid%128==0)mbar_arrive(b->empty+slot);
  }
'''
a=base.index('  // B10 outputs BF16 dx_n.')
b=base.index('TMN_DEVI void reduce_at(',a)
ln=base[a:b]
ln=ln.replace('uint8_t* lnsm=sm+65536;mbar_wait(bar+3,round&1);','uint8_t* lnsm=sm;mbar_wait(b->tx+5,round&1);')
ln=ln.replace('allsync();','named_bar_sync(1,256);')
ln=ln.replace('tmp[threadIdx.x]','tmp[tid]').replace('tmp[256+threadIdx.x]','tmp[256+tid]').replace('tmp[512+threadIdx.x]','tmp[512+tid]').replace('tmp[768+threadIdx.x]','tmp[768+tid]')
ln=ln.replace('if(threadIdx.x==0){store2d','if(tid==0){store2d')
ln=ln.replace('p.partln[split*256+threadIdx.x]','p.partln[split*256+tid]')
reduce=base[b:base.index('extern "C" __global__ __launch_bounds__',b)]
main=r'''
extern "C" __global__ __launch_bounds__(384,2)
void front_b7b12(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];
 __shared__ WsBarriers bars;__shared__ float gamma[128];
 if(threadIdx.x<128)gamma[threadIdx.x]=p.gamma[threadIdx.x];
 if(threadIdx.x==0){
  for(int i=0;i<6;++i)mbar_init(bars.tx+i,1);
  for(int i=0;i<5;++i)mbar_init(bars.empty+i,2);
  fence_barrier_init();
 }
 __syncthreads();int wg=__shfl_sync(0xffffffff,threadIdx.x/128,0);
 if(blockIdx.x<DWCOUNT){
  if(wg<2){setmaxnreg_inc<104>();weight_role(p,sm,bars.tx);}
  else setmaxnreg_dec<32>();
  __syncthreads();
  if(wg<2)setmaxnreg_dec<80>();else setmaxnreg_inc<80>();
 }else{
  if(wg==0){setmaxnreg_dec<32>();ws_producer(p,sm,&bars);}
  else{setmaxnreg_inc<104>();ws_consumer(p,sm,&bars,gamma);}
  __syncthreads();
  if(wg==0)setmaxnreg_inc<80>();else setmaxnreg_dec<80>();
 }
#if PART_ONLY == 2
 __threadfence();__syncthreads();
 if(threadIdx.x==0){atomicAdd(p.counts,1u);while(atomicAdd(p.counts,0u)!=UCOUNT)__nanosleep(32);}
 __syncthreads();
 for(int i=blockIdx.x*384+threadIdx.x;i<9*RING_TILES;i+=UCOUNT*384)p.counts[2+i]=0;
 for(int i=blockIdx.x*384+threadIdx.x;i<131328;i+=UCOUNT*384)reduce_at(p,i);
 __threadfence();__syncthreads();
 if(threadIdx.x==0&&atomicAdd(p.counts+1,1u)==UCOUNT-1){atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);}
#endif
}
extern "C" __global__ void front_reduce(__grid_constant__ const Params p){reduce_at(p,blockIdx.x*256+threadIdx.x);}
'''
name='front_ring_threewg'
(R/(name+'.cu')).write_text(prefix+producer+consumer+ln+reduce+main)
cfg=json.loads((R/'front_ring96_cache3.launch.json').read_text())
cfg.update(threads=384,weight_tma_rows=128,gate_tma_rows=128)
(R/(name+'.launch.json')).write_text(json.dumps(cfg))
print(name)
