// SPDX-License-Identifier: Apache-2.0
// Anthropic v5 training extension. One low-register TMA WG and one N128 MMA WG.
// Four 24-KiB stages; gate prefetch at64KiB overlaps the LN epilogue at0KiB.
constexpr int WS_STAGES=4,WS_SLOT=24576,WS_GATE=65536;
struct WsBarriers { uint64_t tx[6],empty[5]; };
TMN_DEVI void ws_gate(const Params& p,uint8_t* sm,WsBarriers* b,int row){
 mbar_arrive_expect_tx(b->tx+4,49152);
 for(int k=0;k<2;++k){
  tma_load_2d(sm+WS_GATE+k*8192,&p.dg,b->tx+4,k*64,row);
  tma_load_2d(sm+WS_GATE+16384+k*16384,&p.wgate,b->tx+4,k*64,0);
 }
}
TMN_DEVI void ws_producer(const Params& p,uint8_t* sm,WsBarriers* b){
 if(threadIdx.x)return;
 int split=blockIdx.x-DWCOUNT,round=0;
 if(split<p.tiles)ws_gate(p,sm,b,split*64);
 for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){
  int row=tile*64;
  for(int group=0;group<8;++group)ring_wait(p.counts+2+(tile%RING_TILES)*8+group,tile+1);
  // Gate MMA retires all operands in the upper shared-memory region.
  mbar_wait(b->empty+4,round&1);
  for(int step=0;step<16;++step){
   int slot=step%WS_STAGES,epoch=step/WS_STAGES;
   if(round||step>=WS_STAGES)mbar_wait(b->empty+slot,(epoch+1)&1);
   int side=step/8,kind=(step/4)%2,h=step%4;
   const CUtensorMap* weight=side?(kind?&p.wr:&p.wrg):(kind?&p.wl:&p.wlg);
   uint8_t* s=sm+slot*WS_SLOT;
   mbar_arrive_expect_tx(b->tx+slot,24576);
   tma_last(s,&p.ring,b->tx+slot,row%(RING_TILES*64),side*512+kind*256+h*64);
   tma_load_2d(s+8192,weight,b->tx+slot,h*64,0);
  }
  // P is loaded on demand. Publish the ring slot only after the final P read.
  mbar_wait(b->tx+3,1);
  ring_publish(p.counts+2+8*RING_TILES+tile%RING_TILES,tile+1);
  mbar_wait(b->empty,1);mbar_wait(b->empty+1,1);
  mbar_arrive_expect_tx(b->tx+5,32768);
  for(int c=0;c<2;++c){
   tma_load_2d(sm+c*8192,&p.x,b->tx+5,c*64,row);
   tma_load_2d(sm+16384+c*8192,&p.res,b->tx+5,c*64,row);
  }
  if(tile+DXCOUNT<p.tiles){
   mbar_wait(b->empty+2,1);mbar_wait(b->empty+3,1);
   ws_gate(p,sm,b,(tile+DXCOUNT)*64);
  }
 }
}
TMN_DEVI void ws_consumer(const Params& p,uint8_t* sm,WsBarriers* b,const float* gamma){
 int split=blockIdx.x-DWCOUNT,tid=threadIdx.x-128,lane=tid%32,w=tid/32;
 int ra=w*16+lane/4,rb=ra+8,round=0;float running_g=0,running_b=0;
 for(int tile=split;tile<p.tiles;tile+=DXCOUNT,++round){
  int row=tile*64;uint32_t packed[32];float acc[64]={};
  mbar_wait(b->tx+4,round&1);
  {float gate[64]={};uint8_t* s=sm+WS_GATE;
   fence_regs(gate);wgmma_fence();
   static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;
    mma_gate128(gate,smem_desc(smem_u32(s+(k/4)*8192+(k%4)*32),16,1024,1),
      smem_desc(smem_u32(s+16384+(k/4)*16384+(k%4)*32),16,1024,1),k>0);
   });wgmma_commit();wgmma_wait<0>();fence_regs(gate);
   static_for<32>([&](auto qi){constexpr int q=decltype(qi)::value;packed[q]=pack_bf16(gate[q*2],gate[q*2+1]);});
  }
  if(tid==0)mbar_arrive(b->empty+4);
  for(int step=0;step<16;++step){
   int slot=step%WS_STAGES,epoch=step/WS_STAGES;uint8_t* s=sm+slot*WS_SLOT;
   mbar_wait(b->tx+slot,epoch&1);fence_regs(acc);wgmma_fence();
   static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;
    mma_front128(acc,smem_desc(smem_u32(s+k*2048),16,1024,1),
      smem_desc(smem_u32(s+8192+k*32),16,1024,1),step>0||k>0);
   });wgmma_commit();wgmma_wait<0>();fence_regs(acc);
   if(tid==0)mbar_arrive(b->empty+slot);
  }
  static_for<32>([&](auto qi){constexpr int q=decltype(qi)::value;
   acc[q*2]=__bfloat162float(__float2bfloat16_rn(acc[q*2]+bf16lo(packed[q])));
   acc[q*2+1]=__bfloat162float(__float2bfloat16_rn(acc[q*2+1]+bf16hi(packed[q])));
  });
  mbar_wait(b->tx+5,round&1);
  float mu[2]={p.mean[row+ra],p.mean[row+rb]},rs[2]={p.rs[row+ra],p.rs[row+rb]};
  float s1[2][2]={},s2[2][2]={};
  // Match the original two C64 halves and their final addition exactly.
  static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;constexpr int half=q/8;
   int c=q*8+2*(lane%4),base=(c/64)*8192,cc=c%64;
   static_for<2>([&](auto ri){constexpr int r=decltype(ri)::value;int rr=r?rb:ra;
    float xa=__fmul_rn(__fsub_rn(get(sm+base,rr,cc),mu[r]),rs[r]);
    float xb=__fmul_rn(__fsub_rn(get(sm+base,rr,cc+1),mu[r]),rs[r]);
    float ha=acc[q*4+r*2]*gamma[c],hb=acc[q*4+r*2+1]*gamma[c+1];
    s1[half][r]+=ha*xa+hb*xb;s2[half][r]+=ha+hb;
   });
  });
  float c1[2]={quad_sum(s1[0][0])/128.f+quad_sum(s1[1][0])/128.f,
               quad_sum(s1[0][1])/128.f+quad_sum(s1[1][1])/128.f};
  float c2[2]={quad_sum(s2[0][0])/128.f+quad_sum(s2[1][0])/128.f,
               quad_sum(s2[0][1])/128.f+quad_sum(s2[1][1])/128.f};
  float* tmp=reinterpret_cast<float*>(sm+32768);
  static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;
   int c=q*8+2*(lane%4),base=(c/64)*8192,cc=c%64;
   float xaa=__fmul_rn(__fsub_rn(get(sm+base,ra,cc),mu[0]),rs[0]);
   float xab=__fmul_rn(__fsub_rn(get(sm+base,ra,cc+1),mu[0]),rs[0]);
   float xba=__fmul_rn(__fsub_rn(get(sm+base,rb,cc),mu[1]),rs[1]);
   float xbb=__fmul_rn(__fsub_rn(get(sm+base,rb,cc+1),mu[1]),rs[1]);
   float da=acc[q*4],db=acc[q*4+1],dc=acc[q*4+2],dd=acc[q*4+3],ga=gamma[c],gb=gamma[c+1];
   uint32_t oa=pack_bf16((__fmul_rn(da,ga)-fmaf(xaa,c1[0],c2[0]))*rs[0],(__fmul_rn(db,gb)-fmaf(xab,c1[0],c2[0]))*rs[0]);
   uint32_t ob=pack_bf16((__fmul_rn(dc,ga)-fmaf(xba,c1[1],c2[1]))*rs[1],(__fmul_rn(dd,gb)-fmaf(xbb,c1[1],c2[1]))*rs[1]);
   uint32_t resa=pair_get(sm+16384+base,ra,cc),resb=pair_get(sm+16384+base,rb,cc);
   *reinterpret_cast<uint32_t*>(sm+base+swz128(ra,cc*2))=pack_bf16(bf16lo(oa)+bf16lo(resa),bf16hi(oa)+bf16hi(resa));
   *reinterpret_cast<uint32_t*>(sm+base+swz128(rb,cc*2))=pack_bf16(bf16lo(ob)+bf16lo(resb),bf16hi(ob)+bf16hi(resb));
   float dga=da*xaa+dc*xba,dgb=db*xab+dd*xbb,dba=da+dc,dbb=db+dd;
   for(int sh=4;sh<32;sh*=2){dga+=__shfl_xor_sync(0xffffffff,dga,sh);dgb+=__shfl_xor_sync(0xffffffff,dgb,sh);dba+=__shfl_xor_sync(0xffffffff,dba,sh);dbb+=__shfl_xor_sync(0xffffffff,dbb,sh);}
   if(lane<4){tmp[w*256+c]=dga;tmp[w*256+c+1]=dgb;tmp[w*256+128+c]=dba;tmp[w*256+128+c+1]=dbb;}
  });
  named_bar_sync(1,128);
  running_g+=(tmp[tid]+tmp[256+tid])+(tmp[512+tid]+tmp[768+tid]);
  running_b+=(tmp[128+tid]+tmp[384+tid])+(tmp[640+tid]+tmp[896+tid]);
  fence_proxy_async();named_bar_sync(1,128);
  if(tid==0){store2d(&p.dx,sm,0,row);store2d(&p.dx,sm+8192,64,row);tma_store_commit();tma_store_wait_all();}
  named_bar_sync(1,128);
 }
 p.partln[split*256+tid]=running_g;p.partln[split*256+128+tid]=running_b;
}
