from pathlib import Path
p=Path(__file__).resolve().parent
old=(p/'front_roundonly.cu').read_text();small=(p/'front_twocta_maskrcp.cu').read_text()
prefix=old[:old.index('// Each pair')].replace('constexpr int DWCOUNT=4*DW_SPLITS,DXCOUNT=UCOUNT-DWCOUNT;','constexpr int DWCOUNT=0,DXCOUNT=UCOUNT;')
ss=small[small.index('TMN_DEVI void mma_weight64'):small.index('TMN_DEVI void store2d')]
up=p.parent/'trimul_sm90_parity_20260917/engine/third_party/anthropic/upstream/common/opt_core/opt_core/kernels/trimul/native/pkg/v5/csrc/tmn_ptx.cuh'
t=up.read_text();a=t.index('template <int OFF_BYTES>\nTMN_DEVI void wgmma_m64n64k16_rs_off');b=t.index('// A full K chain',a);rs=t[a:b].replace('wgmma_m64n64k16_rs_off','mma_projection_rs').replace('dsc, p, 1, 1, 0;','dsc, p, 1, 1, 1;')
body=r'''
// dX-only feasibility kernel. No dW output is written by this experiment.
struct Bars{uint64_t tx[5],ready[2],empty[4];};
TMN_DEVI void fullsync(){named_bar_sync(0,512);}
TMN_DEVI void psync(){named_bar_sync(1,256);}
TMN_DEVI void csync(){named_bar_sync(2,256);}
TMN_DEVI int cid(){return int(threadIdx.x)-256;}
TMN_DEVI void release(Bars* b,int slot){fence_proxy_async();csync();if(cid()==0)mbar_arrive(b->empty+slot);}
TMN_DEVI void publish(Bars* b,int slot){fence_proxy_async();psync();if(threadIdx.x==0)mbar_arrive(b->ready+slot);}
TMN_DEVI void load_projection(const Params& p,uint8_t* sm,Bars* b,int side){
 if(threadIdx.x)return;mbar_arrive_expect_tx(b->tx+2,65536);
 for(int c=0;c<2;++c)for(int h=0;h<4;++h)tma_load_2d(sm+c*32768+h*8192,side?&p.wr:&p.wl,b->tx+2,h*64,c*64);
}
TMN_DEVI void load_resident(const Params& p,uint8_t* sm,Bars* b){
 if(threadIdx.x)return;mbar_arrive_expect_tx(b->tx+4,131072);
 for(int side=0;side<2;++side)for(int c=0;c<2;++c)for(int h=0;h<4;++h)tma_load_2d(sm+98304+side*65536+c*32768+h*8192,side?&p.wrg:&p.wlg,b->tx+4,h*64,c*64);
}
TMN_DEVI void load_gate(const Params& p,uint8_t* sm,Bars* b,int row){
 if(threadIdx.x)return;uint8_t* s=sm+49152;mbar_arrive_expect_tx(b->tx+2,49152);
 for(int k=0;k<2;++k){tma_load_2d(s+k*8192,&p.dg,b->tx+2,k*64,row);for(int c=0;c<2;++c)tma_load_2d(s+16384+c*16384+k*8192,&p.wgate,b->tx+2,k*64,c*64);}
}
TMN_DEVI void load_front(const Params& p,uint8_t* sm,Bars* b,int group,int row){
 if(threadIdx.x)return;int slot=group&1;uint8_t* s=sm+slot*49152;mbar_arrive_expect_tx(b->tx+slot,49152);
 tma_load_2d(s,&p.pre,b->tx+slot,row,(group/2)*512+(group%2)*256);tma_load_2d(s+32768,group>=2?&p.dr:&p.dl,b->tx+slot,row,(group%2)*128);
}
TMN_DEVI void load_ln(const Params& p,uint8_t* sm,Bars* b,int row){
 if(threadIdx.x)return;mbar_arrive_expect_tx(b->tx+3,32768);for(int c=0;c<2;++c){tma_load_2d(sm+16384+c*8192,&p.x,b->tx+3,c*64,row);tma_load_2d(sm+32768+c*8192,&p.res,b->tx+3,c*64,row);}
}
TMN_DEVI void glu_cached(const Params& p,uint8_t* sm,int group,int row){
 unsigned tid=threadIdx.x;uint8_t* s=sm+(group&1)*49152;uint32_t mask=*reinterpret_cast<const uint32_t*>(p.mask+row+(tid%32)*2);
 static_for<2>([&](auto hi){constexpr int half=decltype(hi)::value;uint32_t pp[8];
  static_for<8>([&](auto qi){constexpr int q=decltype(qi)::value;unsigned i=tid+(q+half*8)*256,c=i/32,r=(i%32)*2;
   uint32_t dy=pair_get(s+32768,c,r),gl=pair_get(s,c*2,r),pr=pair_get(s,c*2+1,r),masked;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(dy),"r"(mask));
   float ga=math::sigmoid(bf16lo(gl)),gb=math::sigmoid(bf16hi(gl));uint32_t g=pack_bf16(((bf16lo(masked)*bf16lo(pr))*ga)*(1.f-ga),((bf16hi(masked)*bf16hi(pr))*gb)*(1.f-gb));pp[q]=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
   *reinterpret_cast<uint32_t*>(s+32768+swz128(c,r*2))=g;
  });psync();
  static_for<8>([&](auto qi){constexpr int q=decltype(qi)::value;unsigned i=tid+(q+half*8)*256,c=i/32,r=(i%32)*2;*reinterpret_cast<uint32_t*>(s+swz128(c,r*2))=pp[q];});
 });
}
TMN_DEVI void producer(const Params& p,uint8_t* sm,Bars* b){
 load_resident(p,sm,b);load_projection(p,sm,b,0);if(threadIdx.x==0)mbar_wait(b->empty+2,0);load_projection(p,sm,b,1);if(threadIdx.x==0)mbar_wait(b->empty+2,1);psync();
 int round=0;for(int tile=blockIdx.x;tile<p.tiles;tile+=UCOUNT,++round){int row=tile*64,ph=round&1;
  if(round==0)load_gate(p,sm,b,row);
  if(threadIdx.x==0&&round)mbar_wait(b->empty+3,ph^1);load_front(p,sm,b,0,row);mbar_wait(b->tx,0);glu_cached(p,sm,0,row);publish(b,0);
  if(threadIdx.x==0)mbar_wait(b->empty+2,ph);load_front(p,sm,b,1,row);mbar_wait(b->tx+1,0);glu_cached(p,sm,1,row);publish(b,1);
  if(threadIdx.x==0)mbar_wait(b->empty,0);load_front(p,sm,b,2,row);mbar_wait(b->tx,1);glu_cached(p,sm,2,row);publish(b,0);
  if(threadIdx.x==0)mbar_wait(b->empty+1,0);load_front(p,sm,b,3,row);mbar_wait(b->tx+1,1);glu_cached(p,sm,3,row);publish(b,1);
  if(threadIdx.x==0){mbar_wait(b->empty,1);load_ln(p,sm,b,row);mbar_wait(b->empty+1,1);if(tile+UCOUNT<p.tiles)load_gate(p,sm,b,(tile+UCOUNT)*64);}
 }
}
TMN_DEVI void consumer(const Params& p,uint8_t* sm,Bars* b,const float* gamma,float* mus,float* rss){
 int tid=cid(),wi=tid/128,lane=tid%32,w=(tid/32)%4,cr=w*16+lane/4;uint32_t pw[2][16][4];
 static_for<2>([&](auto si){constexpr int side=decltype(si)::value;mbar_wait(b->tx+2,side);load_frag_bf16<16,8192>(pw[side],smem_u32(sm+wi*32768),w*16,lane);release(b,2);});mbar_wait(b->tx+4,0);
 float running=0;int round=0;
 for(int tile=blockIdx.x;tile<p.tiles;tile+=UCOUNT,++round){int row=tile*64,ph=round&1;float acc[32]={};uint32_t gatepacked[16];mbar_wait(b->tx+2,ph);
  {float gate[32]={};uint8_t* s=sm+49152;fence_regs(gate);wgmma_fence();static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;
   mma_dgrad(gate,smem_desc(smem_u32(s+16384+wi*16384+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(s+(k/4)*8192+(k%4)*32),16,1024,1),k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(gate);static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;gatepacked[q]=pack_bf16(gate[q*2],gate[q*2+1]);});}release(b,2);
  static_for<2>([&](auto si){constexpr int side=decltype(si)::value;
   static_for<2>([&](auto hi){constexpr int half=decltype(hi)::value;uint8_t* s=sm+half*49152;mbar_wait(b->ready+half,side);fence_regs(acc);wgmma_fence();
    static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;mma_weight64(acc,smem_desc(smem_u32(sm+98304+side*65536+wi*32768+(half*2+k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(s+32768+k*2048),16,1024,1),side>0||half>0||k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);
   });
   static_for<2>([&](auto hi){constexpr int half=decltype(hi)::value;uint64_t bd=smem_desc(smem_u32(sm+half*49152),16,1024,1);fence_regs(acc);wgmma_fence();
    static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;mma_projection_rs<k*2048>(acc,pw[side][half*8+k],uint32_t(bd),uint32_t(bd>>32),1);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);release(b,half);
   });
  });
  static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;acc[q*2]=__bfloat162float(__float2bfloat16_rn(acc[q*2]+bf16lo(gatepacked[q])));acc[q*2+1]=__bfloat162float(__float2bfloat16_rn(acc[q*2+1]+bf16hi(gatepacked[q])));});
  // Transpose dXn to row-major shared memory; no global dXn materialization.
  static_for<8>([&](auto qi){constexpr int q=decltype(qi)::value;int rr=q*8+2*(lane%4);uint8_t* out=sm+wi*8192;
   put(out,rr,cr,acc[q*4]);put(out,rr+1,cr,acc[q*4+1]);put(out,rr,cr+8,acc[q*4+2]);put(out,rr+1,cr+8,acc[q*4+3]);
  });csync();mbar_wait(b->tx+3,ph);
  if(tid<64){mus[tid]=p.mean[row+tid];rss[tid]=p.rs[row+tid];}csync();
  int cc=tid%128,cb=(cc/64)*8192,lc=cc%64;float dg=0,db=0;
  #pragma unroll 1
  for(int r=tid/128*32;r<(tid/128+1)*32;++r){float dy=get(sm+cb,r,lc),xh=__fmul_rn(__fsub_rn(get(sm+16384+cb,r,lc),mus[r]),rss[r]);dg+=dy*xh;db+=dy;}
  csync();
  #pragma unroll 1
  for(int half=0;half<2;++half){int r=half*32+tid/8,col=tid%8;float mu=mus[r],rs=rss[r],s1=0,s2=0;
   #pragma unroll 1
   for(int c=col;c<128;c+=8){int off=(c/64)*8192;float xh=__fmul_rn(__fsub_rn(get(sm+16384+off,r,c%64),mu),rs),wdy=get(sm+off,r,c%64)*gamma[c];s1+=xh*wdy;s2+=wdy;}
   for(int sh=1;sh<8;sh*=2){s1+=__shfl_xor_sync(0xffffffff,s1,sh);s2+=__shfl_xor_sync(0xffffffff,s2,sh);}s1*=1.f/128;s2*=1.f/128;
   #pragma unroll 1
   for(int c=col;c<128;c+=8){int off=(c/64)*8192;float xh=__fmul_rn(__fsub_rn(get(sm+16384+off,r,c%64),mu),rs),dy=get(sm+off,r,c%64);float ln=__bfloat162float(__float2bfloat16_rn((__fmul_rn(dy,gamma[c])-fmaf(xh,s1,s2))*rs));put(sm+16384+off,r,c%64,ln+get(sm+32768+off,r,c%64));}
  }csync();
  float* tmp=reinterpret_cast<float*>(sm);tmp[tid]=dg;tmp[256+tid]=db;csync();int pc=tid%128;running+=(tid<128?(tmp[pc]+tmp[128+pc]):(tmp[256+pc]+tmp[384+pc]));
  fence_proxy_async();csync();if(tid==0){store2d(&p.dx,sm+16384,0,row);store2d(&p.dx,sm+24576,64,row);tma_store_commit();tma_store_wait_all();}release(b,3);
 }
 p.partln[blockIdx.x*256+tid]=running;
 static_for<2>([&](auto si){constexpr int side=decltype(si)::value;static_for<16>([&](auto ki){constexpr int k=decltype(ki)::value;fence_regs(pw[side][k]);});});
}
extern "C" __global__ __launch_bounds__(512,1)
void front_b7b12(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ Bars b;__shared__ float gamma[128],mus[64],rss[64];
 if(threadIdx.x<128)gamma[threadIdx.x]=p.gamma[threadIdx.x];if(threadIdx.x==0){for(int i=0;i<5;++i){mbar_init(b.tx+i,1);if(i<4)mbar_init(b.empty+i,1);if(i<2)mbar_init(b.ready+i,1);}fence_barrier_init();}fullsync();
 int wg=__shfl_sync(0xffffffff,threadIdx.x/128,0);if(wg<2){setmaxnreg_dec<48>();producer(p,sm,&b);}else{setmaxnreg_inc<208>();consumer(p,sm,&b,gamma,mus,rss);}fullsync();
 __threadfence();fullsync();if(threadIdx.x==0){atomicAdd(p.counts,1u);while(atomicAdd(p.counts,0u)!=UCOUNT)__nanosleep(32);}fullsync();
 for(int c=blockIdx.x*512+threadIdx.x;c<256;c+=UCOUNT*512){float v=0;for(int b=0;b<UCOUNT;++b)v+=p.partln[b*256+c];(c<128?p.dgam:p.dbeta)[c%128]=v;}
 __threadfence();fullsync();if(threadIdx.x==0&&atomicAdd(p.counts+1,1u)==UCOUNT-1){atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);}
}
extern "C" __global__ void front_reduce(__grid_constant__ const Params p){}
'''
(p/'front_cached_dx.cu').write_text(prefix+ss+rs+body);(p/'front_cached_dx.launch.json').write_text('{"direct_weights":true}\n')
