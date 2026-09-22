from pathlib import Path
p=Path(__file__).resolve().parent
base=(p/'front_twocta_kindwg.cu').read_text();rows=(p/'front_rows128_p48c224.cu').read_text()
a=base.index('TMN_DEVI void load_g');prefix=base[:a];end=base[base.index('TMN_DEVI void reduce_at'):base.index('extern "C" __global__ __launch_bounds__')]
body=r'''
struct Bars{uint64_t tx[4],ready[2],empty[4];};
TMN_DEVI void psync(){named_bar_sync(1,128);}
TMN_DEVI void csync(){named_bar_sync(2,128);}
TMN_DEVI int cid(){return int(threadIdx.x)-128;}
TMN_DEVI void publish(Bars* b,int slot){fence_proxy_async();psync();if(threadIdx.x==0)mbar_arrive(b->ready+slot);}
TMN_DEVI void release(Bars* b,int slot){fence_proxy_async();csync();if(cid()==0)mbar_arrive(b->empty+slot);}
TMN_DEVI void load_front(const Params& p,uint8_t* sm,Bars* b,int row,int side,int h){
 if(threadIdx.x)return;int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_arrive_expect_tx(b->tx+slot,40960);
 tma_load_2d(s,&p.pre,b->tx+slot,row,side*512+h*128);tma_load_2d(s+16384,side?&p.dr:&p.dl,b->tx+slot,row,h*64);
 for(int n=0;n<2;++n)tma_load_2d(s+24576+n*8192,side?&p.wrg:&p.wlg,b->tx+slot,h*64,n*64);
}
TMN_DEVI void load_pw(const Params& p,uint8_t* sm,Bars* b,int side,int h){
 if(threadIdx.x)return;int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_arrive_expect_tx(b->tx+slot,16384);
 for(int n=0;n<2;++n)tma_load_2d(s+n*8192,side?&p.wr:&p.wl,b->tx+slot,h*64,n*64);
}
TMN_DEVI void load_gate(const Params& p,uint8_t* sm,Bars* b,int row){
 if(threadIdx.x)return;mbar_arrive_expect_tx(b->tx+2,49152);
 for(int k=0;k<2;++k){tma_load_2d(sm+40960+k*8192,&p.dg,b->tx+2,k*64,row);for(int n=0;n<2;++n)tma_load_2d(sm+81920+k*16384+n*8192,&p.wgate,b->tx+2,k*64,n*64);}
}
TMN_DEVI void load_ln(const Params& p,uint8_t* sm,Bars* b,int row){
 if(threadIdx.x)return;mbar_arrive_expect_tx(b->tx+3,32768);
 for(int c=0;c<2;++c){tma_load_2d(sm+c*8192,&p.x,b->tx+3,c*64,row);tma_load_2d(sm+16384+c*8192,&p.res,b->tx+3,c*64,row);}
}
TMN_DEVI void glu_prod(const Params& p,uint8_t* sm,int h,int row){
 unsigned tid=threadIdx.x;uint8_t* s=sm+(h&1)*DX_SLOT;uint32_t mask=*reinterpret_cast<const uint32_t*>(p.mask+row+(tid%32)*2);
 #pragma unroll 2
 for(unsigned q=0;q<16;++q){unsigned i=tid+q*128,c=i/32,r=(i%32)*2;uint32_t dy=pair_get(s+16384,c,r),gl=pair_get(s,c*2,r),pr=pair_get(s,c*2+1,r),masked;
  asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(dy),"r"(mask));float ga=math::sigmoid(bf16lo(gl)),gb=math::sigmoid(bf16hi(gl));
  uint32_t g=pack_bf16(((bf16lo(masked)*bf16lo(pr))*ga)*(1.f-ga),((bf16hi(masked)*bf16hi(pr))*gb)*(1.f-gb)),pp=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
  *reinterpret_cast<uint32_t*>(s+16384+swz128(c,r*2))=g;*reinterpret_cast<uint32_t*>(sm+81920+h*8192+swz128(c,r*2))=pp;
 }
}
TMN_DEVI void dx_producer(const Params& p,uint8_t* sm,Bars* b){
 int split=blockIdx.x-DWCOUNT,round=0;
 for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){int row=tile*64,ph=round&1;
  if(round==0)load_gate(p,sm,b,row);
  if(threadIdx.x==0){if(round)mbar_wait(b->empty+3,ph^1);load_front(p,sm,b,row,0,0);mbar_wait(b->empty+2,ph);load_front(p,sm,b,row,0,1);}psync();
  for(int side=0;side<2;++side){
   for(int h=0;h<4;++h){int slot=h&1;mbar_wait(b->tx+slot,h/2);glu_prod(p,sm,h,row);publish(b,slot);
    if(threadIdx.x==0){mbar_wait(b->empty+slot,h/2);if(h<2)load_front(p,sm,b,row,side,h+2);}
   }
   if(threadIdx.x==0){load_pw(p,sm,b,side,0);load_pw(p,sm,b,side,1);
    mbar_wait(b->empty,0);load_pw(p,sm,b,side,2);mbar_wait(b->empty+1,0);load_pw(p,sm,b,side,3);
    mbar_wait(b->empty,1);if(side==0)load_front(p,sm,b,row,1,0);else load_ln(p,sm,b,row);
    mbar_wait(b->empty+1,1);if(side==0)load_front(p,sm,b,row,1,1);else if(tile+DXCOUNT<p.tiles)load_gate(p,sm,b,(tile+DXCOUNT)*64);
   }psync();
  }
 }
}
TMN_DEVI void dx_consumer(const Params& p,uint8_t* sm,Bars* b,const float* gamma){
 int split=blockIdx.x-DWCOUNT,tid=cid(),lane=tid%32,w=tid/32,ra=w*16+lane/4,rb=ra+8,round=0;float running=0,runningb=0;
 for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){int row=tile*64,ph=round&1;uint32_t packed[32];float acc[64]={};mbar_wait(b->tx+2,ph);
  {float gate[64]={};fence_regs(gate);wgmma_fence();static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;
   mma_gate128(gate,smem_desc(smem_u32(sm+40960+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sm+81920+(k/4)*16384+(k%4)*32),16,1024,1),k>0);
  });wgmma_commit();wgmma_wait<0>();fence_regs(gate);static_for<32>([&](auto qi){constexpr int q=decltype(qi)::value;packed[q]=pack_bf16(gate[q*2],gate[q*2+1]);});}release(b,2);
  for(int side=0;side<2;++side){
   for(int h=0;h<4;++h){int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_wait(b->ready+slot,h/2);fence_regs(acc);wgmma_fence();
    static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_front128(acc,smem_desc(smem_u32(s+16384+k*2048),16,1024,1),smem_desc(smem_u32(s+24576+k*32),16,1024,1),side>0||h>0||k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);release(b,slot);
   }
   for(int h=0;h<4;++h){int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_wait(b->tx+slot,h/2);fence_regs(acc);wgmma_fence();
    static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_front128(acc,smem_desc(smem_u32(sm+81920+h*8192+k*2048),16,1024,1),smem_desc(smem_u32(s+k*32),16,1024,1),1);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);release(b,slot);
   }
  }
'''
start=rows.index('  static_for<32>',rows.index('   for(int h=0;h<4;++h){int slot=h&1;mbar_wait(b->tx+slot+2'))
stop=rows.index('TMN_DEVI void reduce_at',start)
ln=rows[start:stop].replace('mbar_wait(b->tx,ph);uint8_t* ln=sm+wi*16384;','mbar_wait(b->tx+3,ph);uint8_t* ln=sm;').replace('row+wi*64+','row+').replace('(wi*4+w)*256','w*256').replace('sm+PGRAD','sm+65536').replace('ln+32768','ln+16384').replace('dxsync();','csync();')
a=ln.index('  csync();float sum=0;');ln=ln[:a]+r'''
  csync();float sum=0,sumb=0;for(int w=0;w<4;++w){sum+=tmp[w*256+tid];sumb+=tmp[w*256+128+tid];}running+=sum;runningb+=sumb;
  fence_proxy_async();csync();if(tid==0){store2d(&p.dx,sm,0,row);store2d(&p.dx,sm+8192,64,row);tma_store_commit();tma_store_wait_all();}release(b,3);
 }
 p.partln[split*256+tid]=running;p.partln[split*256+128+tid]=runningb;
}
'''
main=r'''
extern "C" __global__ __launch_bounds__(256,2)
void front_b7b12(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ Bars b;__shared__ float gamma[128];
 if(threadIdx.x<128)gamma[threadIdx.x]=p.gamma[threadIdx.x];
 if(threadIdx.x==0){for(int i=0;i<4;++i){mbar_init(b.tx+i,1);mbar_init(b.empty+i,1);if(i<2)mbar_init(b.ready+i,1);}fence_barrier_init();}allsync();
 if(blockIdx.x<DWCOUNT)weight_role(p,sm,b.tx);else{
  int wg=__shfl_sync(0xffffffff,threadIdx.x/128,0);
  if(wg==0){setmaxnreg_dec<32>();dx_producer(p,sm,&b);}else{setmaxnreg_inc<224>();dx_consumer(p,sm,&b,gamma);}
 }
 allsync();
#if PART_ONLY == 2
 __threadfence();allsync();if(threadIdx.x==0){atomicAdd(p.counts,1u);while(atomicAdd(p.counts,0u)!=UCOUNT)__nanosleep(32);}allsync();
 for(int i=blockIdx.x*256+threadIdx.x;i<131328;i+=UCOUNT*256)reduce_at(p,i);
 __threadfence();allsync();if(threadIdx.x==0&&atomicAdd(p.counts+1,1u)==UCOUNT-1){atomicExch(p.counts,0u);atomicExch(p.counts+1,0u);}
#endif
}
extern "C" __global__ void front_reduce(__grid_constant__ const Params p){reduce_at(p,blockIdx.x*256+threadIdx.x);}
'''
(p/'front_twows.cu').write_text(prefix+body+ln+end+main);(p/'front_twows.launch.json').write_text((p/'front_twocta.launch.json').read_text())
