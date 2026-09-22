// Two row tiles per CTA, one N128 warpgroup per row tile.
// B9 and rounded dx_n share32KiB. Two32KiB TMA slots share projection weights.
constexpr int PAIR_SLOT=32768,PAIR_VALUE=81920;
TMN_DEVI void pair_gate_load(const Params& p,uint8_t* sm,uint64_t* b,int row){
 if(threadIdx.x)return;mbar_arrive_expect_tx(b,65536);
 for(int w=0;w<2;++w)for(int k=0;k<2;++k)
  tma_load_2d(sm+w*16384+k*8192,&p.dg,b,k*64,row+w*64);
 for(int k=0;k<2;++k)tma_load_2d(sm+32768+k*16384,&p.wgate,b,k*64,0);
}
TMN_DEVI void pair_front_load(const Params& p,uint8_t* sm,uint64_t* b,int tile,int step){
 if(threadIdx.x)return;int slot=step&1,side=step/8,kind=(step/4)%2,h=step%4;
 uint8_t* s=sm+slot*PAIR_SLOT;const CUtensorMap* weight=side?(kind?&p.wr:&p.wrg):(kind?&p.wl:&p.wlg);
 mbar_arrive_expect_tx(b+slot,32768);
 for(int w=0;w<2;++w)tma_last(s+w*8192,&p.ring,b+slot,((tile+w)%RING_TILES)*64,side*512+kind*256+h*64);
 tma_load_2d(s+16384,weight,b+slot,h*64,0);
}
TMN_DEVI void pair_ln_load(const Params& p,uint8_t* sm,uint64_t* b,int row){
 if(threadIdx.x)return;mbar_arrive_expect_tx(b,65536);
 for(int w=0;w<2;++w)for(int c=0;c<2;++c){
  tma_load_2d(sm+w*32768+c*8192,&p.x,b,c*64,row+w*64);
  tma_load_2d(sm+w*32768+16384+c*8192,&p.res,b,c*64,row+w*64);
 }
}
TMN_DEVI void pair_ln(const Params& p,uint8_t* sm,const float* gamma,int row,float& run_g,float& run_b){
 int tid=threadIdx.x%128,wi=threadIdx.x/128,lane=tid%32,w=tid/32,ra=w*16+lane/4,rb=ra+8;
 uint8_t* ln=sm+wi*32768;uint8_t* value=sm+PAIR_VALUE+wi*16384;
 float mu[2]={p.mean[row+wi*64+ra],p.mean[row+wi*64+rb]},rs[2]={p.rs[row+wi*64+ra],p.rs[row+wi*64+rb]};
 float s1[2][2]={},s2[2][2]={};
 // Each C64 partial retains the selected kernel's scalar addition order.
 #pragma unroll 2
 for(int half=0;half<2;++half){
  #pragma unroll 1
  for(int q=0;q<8;++q){int c=q*8+2*(lane%4),gc=half*64+c;
   #pragma unroll 2
   for(int r=0;r<2;++r){int rr=r?rb:ra;
    float xa=__fmul_rn(__fsub_rn(get(ln+half*8192,rr,c),mu[r]),rs[r]);
    float xb=__fmul_rn(__fsub_rn(get(ln+half*8192,rr,c+1),mu[r]),rs[r]);
    float ha=get(value+half*8192,rr,c)*gamma[gc],hb=get(value+half*8192,rr,c+1)*gamma[gc+1];
    s1[half][r]+=ha*xa+hb*xb;s2[half][r]+=ha+hb;
   }
  }
 }
 float c1[2]={quad_sum(s1[0][0])/128.f+quad_sum(s1[1][0])/128.f,quad_sum(s1[0][1])/128.f+quad_sum(s1[1][1])/128.f};
 float c2[2]={quad_sum(s2[0][0])/128.f+quad_sum(s2[1][0])/128.f,quad_sum(s2[0][1])/128.f+quad_sum(s2[1][1])/128.f};
 float* tmp=reinterpret_cast<float*>(sm+65536+wi*4096);
 #pragma unroll 1
 for(int q=0;q<16;++q){int c=q*8+2*(lane%4),base=(c/64)*8192,cc=c%64;
  float xaa=__fmul_rn(__fsub_rn(get(ln+base,ra,cc),mu[0]),rs[0]);
  float xab=__fmul_rn(__fsub_rn(get(ln+base,ra,cc+1),mu[0]),rs[0]);
  float xba=__fmul_rn(__fsub_rn(get(ln+base,rb,cc),mu[1]),rs[1]);
  float xbb=__fmul_rn(__fsub_rn(get(ln+base,rb,cc+1),mu[1]),rs[1]);
  float da=get(value+base,ra,cc),db=get(value+base,ra,cc+1),dc=get(value+base,rb,cc),dd=get(value+base,rb,cc+1);
  uint32_t oa=pack_bf16((__fmul_rn(da,gamma[c])-fmaf(xaa,c1[0],c2[0]))*rs[0],(__fmul_rn(db,gamma[c+1])-fmaf(xab,c1[0],c2[0]))*rs[0]);
  uint32_t ob=pack_bf16((__fmul_rn(dc,gamma[c])-fmaf(xba,c1[1],c2[1]))*rs[1],(__fmul_rn(dd,gamma[c+1])-fmaf(xbb,c1[1],c2[1]))*rs[1]);
  uint32_t resa=pair_get(ln+16384+base,ra,cc),resb=pair_get(ln+16384+base,rb,cc);
  *reinterpret_cast<uint32_t*>(ln+base+swz128(ra,cc*2))=pack_bf16(bf16lo(oa)+bf16lo(resa),bf16hi(oa)+bf16hi(resa));
  *reinterpret_cast<uint32_t*>(ln+base+swz128(rb,cc*2))=pack_bf16(bf16lo(ob)+bf16lo(resb),bf16hi(ob)+bf16hi(resb));
  float dga=da*xaa+dc*xba,dgb=db*xab+dd*xbb,dba=da+dc,dbb=db+dd;
  for(int sh=4;sh<32;sh*=2){dga+=__shfl_xor_sync(0xffffffff,dga,sh);dgb+=__shfl_xor_sync(0xffffffff,dgb,sh);dba+=__shfl_xor_sync(0xffffffff,dba,sh);dbb+=__shfl_xor_sync(0xffffffff,dbb,sh);}
  if(lane<4){tmp[w*256+c]=dga;tmp[w*256+c+1]=dgb;tmp[w*256+128+c]=dba;tmp[w*256+128+c+1]=dbb;}
 }
 allsync();
 run_g+=(tmp[tid]+tmp[256+tid])+(tmp[512+tid]+tmp[768+tid]);
 run_b+=(tmp[128+tid]+tmp[384+tid])+(tmp[640+tid]+tmp[896+tid]);
 fence_proxy_async();allsync();
 if(threadIdx.x==0){for(int w=0;w<2;++w)for(int c=0;c<2;++c)store2d(&p.dx,sm+w*32768+c*8192,c*64,row+w*64);tma_store_commit();tma_store_wait_all();}
 allsync();
}
TMN_DEVI void input_role(const Params& p,uint8_t* sm,uint64_t* b,const float* gamma){
 int split=blockIdx.x-DWCOUNT,wi=threadIdx.x/128,tid=threadIdx.x%128,lane=tid%32,w=tid/32;
 int ra=w*16+lane/4,rb=ra+8,round=0;float run_g=0,run_b=0;
 for(int tile=split*2;tile<p.tiles;tile+=DXCOUNT*2,++round){int row=tile*64;
  pair_gate_load(p,sm,b+2,row);mbar_wait(b+2,round&1);
  {float gate[64]={};fence_regs(gate);wgmma_fence();
   static_for<8>([&](auto ki){constexpr int k=decltype(ki)::value;
    mma_gate128(gate,smem_desc(smem_u32(sm+wi*16384+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sm+32768+(k/4)*16384+(k%4)*32),16,1024,1),k>0);
   });wgmma_commit();wgmma_wait<0>();fence_regs(gate);
   static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int c=q*8+2*(lane%4),base=(c/64)*8192,cc=c%64;
    *reinterpret_cast<uint32_t*>(sm+PAIR_VALUE+wi*16384+base+swz128(ra,cc*2))=pack_bf16(gate[q*4],gate[q*4+1]);
    *reinterpret_cast<uint32_t*>(sm+PAIR_VALUE+wi*16384+base+swz128(rb,cc*2))=pack_bf16(gate[q*4+2],gate[q*4+3]);
   });
  }
  allsync();
  if(threadIdx.x<16){int t=tile+threadIdx.x/8;ring_wait(p.counts+2+(t%RING_TILES)*8+threadIdx.x%8,t+1);}
  allsync();pair_front_load(p,sm,b,tile,0);pair_front_load(p,sm,b,tile,1);
  {float acc[64]={};
   for(int step=0;step<16;++step){int slot=step&1;uint8_t* s=sm+slot*PAIR_SLOT;mbar_wait(b+slot,(step/2)&1);
    if(step==15&&threadIdx.x==0)for(int w=0;w<2;++w)ring_publish(p.counts+2+8*RING_TILES+(tile+w)%RING_TILES,tile+w+1);
    fence_regs(acc);wgmma_fence();
    static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;
     mma_front128(acc,smem_desc(smem_u32(s+wi*8192+k*2048),16,1024,1),smem_desc(smem_u32(s+16384+k*32),16,1024,1),step>0||k>0);
    });wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();
    if(step<14)pair_front_load(p,sm,b,tile,step+2);
   }
   pair_ln_load(p,sm,b+3,row);
   static_for<16>([&](auto qi){constexpr int q=decltype(qi)::value;int c=q*8+2*(lane%4),base=(c/64)*8192,cc=c%64;uint8_t* v=sm+PAIR_VALUE+wi*16384+base;
    uint32_t ga=pair_get(v,ra,cc),gb=pair_get(v,rb,cc);
    *reinterpret_cast<uint32_t*>(v+swz128(ra,cc*2))=pack_bf16(acc[q*4]+bf16lo(ga),acc[q*4+1]+bf16hi(ga));
    *reinterpret_cast<uint32_t*>(v+swz128(rb,cc*2))=pack_bf16(acc[q*4+2]+bf16lo(gb),acc[q*4+3]+bf16hi(gb));
   });
  }
  mbar_wait(b+3,round&1);allsync();pair_ln(p,sm,gamma,row,run_g,run_b);
 }
 float* tmp=reinterpret_cast<float*>(sm);tmp[wi*256+tid]=run_g;tmp[wi*256+128+tid]=run_b;
 allsync();p.partln[split*256+threadIdx.x]=tmp[threadIdx.x]+tmp[256+threadIdx.x];
}
