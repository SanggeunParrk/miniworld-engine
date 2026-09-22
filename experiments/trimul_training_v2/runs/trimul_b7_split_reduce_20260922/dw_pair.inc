// SPDX-License-Identifier: Apache-2.0
// Two consecutive row tiles: GP1 overlaps GLU0, then one K128 dW group.
TMN_DEVI void pipe_producer(const Params& p,uint8_t* sm,uint64_t* bar){
 int group=blockIdx.x%16,split=blockIdx.x/16;
 for(int r=0,ti=split;ti<p.tiles;++r,ti+=DW_SPLITS){int slot=r&3;
  if(r>=4)mbar_wait(bar+4+slot,((r/4)-1)&1);
  if(threadIdx.x==0){uint8_t* s=sm+slot*20480;mbar_arrive_expect_tx(bar+slot,20480);
   tma_load_2d(s,&p.xn,bar+slot,0,ti*64);tma_load_2d(s+8192,&p.xn,bar+slot,64,ti*64);
   tma_load_2d(s+16384,group<8?&p.dl:&p.dr,bar+slot,ti*64,(group%8)*32);
  }
 }
}
TMN_DEVI void pair_gp(float (&a)[32],uint32_t (&xn)[8][4],uint64_t desc){
 uint32_t lo=desc,hi=desc>>32;fence_regs(a);wgmma_fence();
 static_for<8>([&](auto kk){constexpr int k=decltype(kk)::value;wgmma_m64n64k16_rs_off<(k/4)*8192+(k%4)*32>(a,xn[k],lo,hi,k>0);});wgmma_commit();
}
TMN_DEVI void pair_glu(float (&a)[32],uint8_t* s,uint8_t* sg,uint32_t ma,uint32_t mb){
 int lane=threadIdx.x%32,w=(threadIdx.x/32)%4,mat=lane/8;
 #pragma unroll
 for(int q=0;q<2;++q){uint32_t dy[4],dg[4],dp[4];ldsm_x4_t(dy,smem_u32(s+16384)+swz128(q*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2));
  #pragma unroll
  for(int j=0;j<4;++j){uint32_t masked,m=(j&1)?mb:ma;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(dy[j]),"r"(m));
   float ga=math::sigmoid(math::round_bf16(a[q*8+j*2])),gb=math::sigmoid(math::round_bf16(a[q*8+j*2+1])),pa=math::round_bf16(a[(q+2)*8+j*2]),pb=math::round_bf16(a[(q+2)*8+j*2+1]);
   dg[j]=pack_bf16(((bf16lo(masked)*pa)*ga)*(1.f-ga),((bf16hi(masked)*pb)*gb)*(1.f-gb));dp[j]=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
  }
  uint32_t addr=swz128(q*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2);stsm_x4_t(smem_u32(sg)+addr,dg[0],dg[1],dg[2],dg[3]);stsm_x4_t(smem_u32(sg+4096)+addr,dp[0],dp[1],dp[2],dp[3]);
 }
}
TMN_DEVI void pipe_consumer(const Params& p,uint8_t* sm,uint64_t* bar,const float*,const float*){
 int group=blockIdx.x%16,split=blockIdx.x/16,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 int rounds=(p.tiles-split+DW_SPLITS-1)/DW_SPLITS,seg=(rounds+WGRAD_SLICES-1)/WGRAD_SLICES;
 float dw[64]={};mbar_wait(bar+8,0);uint64_t desc=smem_desc(smem_u32(sm+PIPE_W_BASE),16,1024,1);
 for(int r=0;r<rounds;r+=2){int slot=r&3,ti0=split+r*DW_SPLITS,ti1=ti0+DW_SPLITS;
  uint8_t* s0=sm+slot*20480;uint8_t* s1=s0+20480;uint8_t* sg0=sm+PIPE_SG_BASE;uint8_t* sg1=sg0+8192;
  mbar_wait(bar+slot,(r/4)&1);uint32_t xn[8][4];load_frag_bf16<8,8192>(xn,smem_u32(s0),w*16,lane);
  uint16_t x0,y0,x1,y1;int ra=w*16+lane/4;
  asm volatile("ld.global.b16 %0,[%1];":"=h"(x0):"l"(p.mask+ti0*64+ra):"memory");asm volatile("ld.global.b16 %0,[%1];":"=h"(y0):"l"(p.mask+ti0*64+ra+8):"memory");
  float a0[32]={},a1[32]={};pair_gp(a0,xn,desc);wgmma_wait<0>();fence_regs(a0);
  mbar_wait(bar+slot+1,(r/4)&1);load_frag_bf16<8,8192>(xn,smem_u32(s1),w*16,lane);
  asm volatile("ld.global.b16 %0,[%1];":"=h"(x1):"l"(p.mask+ti1*64+ra):"memory");asm volatile("ld.global.b16 %0,[%1];":"=h"(y1):"l"(p.mask+ti1*64+ra+8):"memory");
  pair_gp(a1,xn,desc);pair_glu(a0,s0,sg0,uint32_t(x0)*0x10001u,uint32_t(y0)*0x10001u);
  wgmma_wait<0>();fence_regs(a1);pair_glu(a1,s1,sg1,uint32_t(x1)*0x10001u,uint32_t(y1)*0x10001u);
  fence_proxy_async();named_bar_sync(1,128);fence_regs(dw);wgmma_fence();
  static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;mma_weight128(dw,smem_desc(smem_u32(sg0+k*32),16,1024,1),smem_desc(smem_u32(s0+k*2048),8192,1024,1),r%seg>0||k>0);});
  static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;mma_weight128(dw,smem_desc(smem_u32(sg1+k*32),16,1024,1),smem_desc(smem_u32(s1+k*2048),8192,1024,1),1);});
  wgmma_commit();wgmma_wait<0>();fence_regs(dw);named_bar_sync(1,128);if(threadIdx.x%128==0){mbar_arrive(bar+4+slot);mbar_arrive(bar+4+slot+1);}
  if((r+2)%seg==0||r+2>=rounds){float* out=p.partw+((group*DW_SPLITS+split)*WGRAD_SLICES+r/seg)*8192;
   static_for<16>([&](auto qq){constexpr int q=decltype(qq)::value;int rr=w*16+lane/4,c=q*8+2*(lane%4);stg64f(out+rr*128+c,dw[q*4],dw[q*4+1]);stg64f(out+(rr+8)*128+c,dw[q*4+2],dw[q*4+3]);});
  }
 }
}
