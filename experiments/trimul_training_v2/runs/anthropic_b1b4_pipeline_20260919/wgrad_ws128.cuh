// NVIDIA WGMMA SS N128 operand contract; raw TMA buffers stay MN-major.
TMN_DEVI void mma_ss128(float (&d)[64],uint64_t a,uint64_t b,int accumulate){
 asm volatile("{ .reg .pred p; setp.ne.b32 p, %66, 0; wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1, 1, 1; }"
 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),"+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63]) : "l"(a),"l"(b),"r"(accumulate));
}
// Dedicated producer warpgroup: TMA + B1; consumer warpgroup: SS WGMMA.
TMN_DEVI void wgrad_ws128(const Params& p,uint8_t* sm,uint64_t* bars,int* last){
 const int tid=threadIdx.x%128,lane=tid%32,w=tid/32;
 int b=blockIdx.x-p.tiles,tile=b/SPLITS,split=b%SPLITS;
 bool wg=tile<2;int t=wg?tile:tile-2,mch=wg?t*64:(t/2)*64,nch=wg?0:(t%2)*128;
 int chunks=(p.M+63)/64,first=chunks*split/SPLITS,end=chunks*(split+1)/SPLITS;
 uint64_t* raw=bars;uint64_t* ready=bars+WSTAGES;uint64_t* empty=bars+2*WSTAGES;
 if(threadIdx.x==0){for(int i=0;i<WSTAGES;++i){mbar_init(raw+i,1);mbar_init(ready+i,1);mbar_init(empty+i,1);}fence_barrier_init();}__syncthreads();
 if(threadIdx.x<128){
  for(int it=first;it<end;++it){int stage=(it-first)%WSTAGES,phase=((it-first)/WSTAGES)&1,row=it*64,cc=wg?nch:mch;
   uint8_t* buf=sm+stage*57344;
   if(it-first>=WSTAGES)mbar_wait(empty+stage,phase^1);
   if(tid==0){mbar_arrive_expect_tx(raw+stage,wg?57344:32768);
    if(wg){for(int n=0;n<2;++n){tma_load_2d(buf+n*8192,&p.dy,raw+stage,n*64,row);tma_load_2d(buf+16384+n*8192,&p.gate,raw+stage,n*64,row);tma_load_2d(buf+32768+n*8192,&p.proj,raw+stage,n*64,row);}tma_load_2d(buf+49152,&p.xn,raw+stage,mch,row);}
    else{tma_load_2d(buf,&p.dy,raw+stage,mch,row);tma_load_2d(buf+16384,&p.gate,raw+stage,mch,row);for(int n=0;n<2;++n)tma_load_2d(buf+32768+n*8192,&p.norm,raw+stage,nch+n*64,row);}
   }
   sync_group();mbar_wait(raw+stage,phase);
   int j0=row%p.L;
   for(int i=tid;i<(wg?4096:2048);i+=128){int n=i/2048,r=(i%2048)/32,c=2*(i%32),jr=j0+r;if(jr>=p.L)jr-=p.L;
    uint32_t y=pair_get(buf+n*8192,r,c),g=pair_get(buf+16384+n*8192,r,c),v=wg?pair_get(buf+32768+n*8192,r,c):0,ds=ldg32(p.ds+jr*128+(wg?n*64:mch)+c);float ya=bf16lo(y)*bf16lo(ds),yb=bf16hi(y)*bf16hi(ds),ga=bf16lo(g),gb=bf16hi(g);
    pair_put(buf+n*8192,r,c,pack_bf16(wg?((ya*bf16lo(v))*ga)*(1.f-ga):ya*ga,wg?((yb*bf16hi(v))*gb)*(1.f-gb):yb*gb));
   }
   fence_proxy_async();sync_group();if(tid==0)mbar_arrive(ready+stage);
  }
 }else{
  float acc[64]={};
  for(int it=first;it<end;++it){int stage=(it-first)%WSTAGES,phase=((it-first)/WSTAGES)&1;uint8_t* buf=sm+stage*57344;
   mbar_wait(ready+stage,phase);
   uint8_t* sa=wg?buf+49152:buf;uint8_t* sb=wg?buf:buf+32768;
   fence_regs(acc);wgmma_fence();
   static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_ss128(acc,smem_desc(smem_u32(sa+k*2048),16,1024,1),smem_desc(smem_u32(sb+k*2048),8192,1024,1),it>first||k>0);});
   wgmma_commit();wgmma_wait<0>();fence_regs(acc);sync_group();if(tid==0)mbar_arrive(empty+stage);
  }
 float* part=p.partw+(tile*SPLITS+split)*8192;
#pragma unroll
 for(int q=0;q<16;++q){int r=w*16+lane/4,c=q*8+2*(lane%4);part[r*128+c]=acc[4*q];part[r*128+c+1]=acc[4*q+1];part[(r+8)*128+c]=acc[4*q+2];part[(r+8)*128+c+1]=acc[4*q+3];}
 if(ticket(p.counts+tile,SPLITS,last)){
  for(int i=tid;i<8192;i+=128){float v=0;for(int s=0;s<SPLITS;++s)v+=reinterpret_cast<volatile float*>(p.partw)[(tile*SPLITS+s)*8192+i];int r=mch+i/128,c=nch+i%128;(wg?p.dwg:p.dwp)[r*(wg?128:256)+c]=__float2bfloat16_rn(v);}
  sync_group();if(tid==0)atomicExch(p.counts+tile,0u);
 }
 }
}
