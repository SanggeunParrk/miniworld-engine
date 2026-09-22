// Dedicated producer warpgroup: TMA + B1; consumer warpgroup: SS WGMMA.
TMN_DEVI void wgrad_ws(const Params& p,uint8_t* sm,uint64_t* bars,int* last){
 const int tid=threadIdx.x%128,lane=tid%32,w=tid/32;
 int b=blockIdx.x-p.tiles,tile=b/SPLITS,split=b%SPLITS;
 bool wg=tile<4;int t=wg?tile:tile-4,ncols=wg?2:4,mch=(t/ncols)*64,nch=(t%ncols)*64;
 int chunks=(p.M+63)/64,first=chunks*split/SPLITS,end=chunks*(split+1)/SPLITS;
 uint64_t* raw=bars;uint64_t* ready=bars+WSTAGES;uint64_t* empty=bars+2*WSTAGES;
 if(threadIdx.x==0){for(int i=0;i<WSTAGES;++i){mbar_init(raw+i,1);mbar_init(ready+i,1);mbar_init(empty+i,1);}fence_barrier_init();}__syncthreads();
 if(threadIdx.x<128){
  for(int it=first;it<end;++it){int stage=(it-first)%WSTAGES,phase=((it-first)/WSTAGES)&1,row=it*64,cc=wg?nch:mch;
   uint8_t* buf=sm+stage*32768;
   if(it-first>=WSTAGES)mbar_wait(empty+stage,phase^1);
   if(tid==0){mbar_arrive_expect_tx(raw+stage,wg?32768:24576);tma_load_2d(buf,&p.dy,raw+stage,cc,row);tma_load_2d(buf+8192,&p.gate,raw+stage,cc,row);if(wg)tma_load_2d(buf+16384,&p.proj,raw+stage,cc,row);tma_load_2d(buf+24576,wg?&p.xn:&p.norm,raw+stage,wg?mch:nch,row);}
   sync_group();mbar_wait(raw+stage,phase);
   int j0=row%p.L;
   for(int i=tid;i<2048;i+=128){int r=i/32,c=2*(i%32),jr=j0+r;if(jr>=p.L)jr-=p.L;
    uint32_t y=pair_get(buf,r,c),g=pair_get(buf+8192,r,c),v=wg?pair_get(buf+16384,r,c):0,ds=ldg32(p.ds+jr*128+cc+c);float ya=bf16lo(y)*bf16lo(ds),yb=bf16hi(y)*bf16hi(ds),ga=bf16lo(g),gb=bf16hi(g);
    pair_put(buf,r,c,pack_bf16(wg?((ya*bf16lo(v))*ga)*(1.f-ga):ya*ga,wg?((yb*bf16hi(v))*gb)*(1.f-gb):yb*gb));
   }
   fence_proxy_async();sync_group();if(tid==0)mbar_arrive(ready+stage);
  }
 }else{
  float acc[32]={};
  for(int it=first;it<end;++it){int stage=(it-first)%WSTAGES,phase=((it-first)/WSTAGES)&1;uint8_t* buf=sm+stage*32768;
   mbar_wait(ready+stage,phase);
   uint8_t* sa=wg?buf+24576:buf;uint8_t* sb=wg?buf:buf+24576;
   fence_regs(acc);wgmma_fence();
   static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_ss(acc,smem_desc(smem_u32(sa+k*2048),16,1024,1),smem_desc(smem_u32(sb+k*2048),16,1024,1),it>first||k>0);});
   wgmma_commit();wgmma_wait<0>();fence_regs(acc);sync_group();if(tid==0)mbar_arrive(empty+stage);
  }
 float* part=p.partw+(tile*SPLITS+split)*4096;
#pragma unroll
 for(int q=0;q<8;++q){int r=w*16+lane/4,c=q*8+2*(lane%4);part[r*64+c]=acc[4*q];part[r*64+c+1]=acc[4*q+1];part[(r+8)*64+c]=acc[4*q+2];part[(r+8)*64+c+1]=acc[4*q+3];}
 if(ticket(p.counts+tile,SPLITS,last)){
  for(int i=tid;i<4096;i+=128){float v=0;for(int s=0;s<SPLITS;++s)v+=reinterpret_cast<volatile float*>(p.partw)[(tile*SPLITS+s)*4096+i];int r=mch+i/64,c=nch+i%64;(wg?p.dwg:p.dwp)[r*(wg?128:256)+c]=__float2bfloat16_rn(v);}
  sync_group();if(tid==0)atomicExch(p.counts+tile,0u);
 }
}

}
