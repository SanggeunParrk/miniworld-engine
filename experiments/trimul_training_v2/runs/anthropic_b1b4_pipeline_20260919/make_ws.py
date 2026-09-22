from pathlib import Path
r=Path(__file__).resolve().parent;s=(r/'wgrad_ss.cuh').read_text();tail=s[s.index(' float* part='):];tail=tail.replace('sync_group();if(tid==0)','sync_group();if(tid==0)')
code='''// Dedicated producer warpgroup: TMA + B1; consumer warpgroup: SS WGMMA.
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
'''+tail+'\n}\n'
(r/'wgrad_ws.cuh').write_text(code)
p=r/'fused.cu';s=p.read_text().replace('#ifndef WSS','#ifndef WSTAGES\n#define WSTAGES 2\n#endif\n#ifndef WSS',1).replace('#include "wgrad_ss.cuh"','#include "wgrad_ss.cuh"\n#include "wgrad_ws.cuh"').replace('bar[WGROUPS*2]','bar[16]').replace('if(WSS)wgrad_ss','if(WSS==3)wgrad_ws(p,sm,bar,&last[1]);else if(WSS)wgrad_ss');p.write_text(s)
p=r/'core.py';s=p.read_text().replace("+(R/'wgrad_ss.cuh').read_bytes()","+(R/'wgrad_ss.cuh').read_bytes()+(R/'wgrad_ws.cuh').read_bytes()").replace('(65536 if wss>=2 else 40960)*wgroups','(65536 if wss==3 else (65536 if wss>=2 else 40960)*wgroups)').replace('splits//wgroups','splits//(1 if wss==3 else wgroups)');p.write_text(s)
