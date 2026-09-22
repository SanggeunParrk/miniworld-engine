// SPDX-License-Identifier: Apache-2.0
// Native K1's packed gate32/projection32 WGMMA result is consumed directly.
TMN_DEVI void load_dw_packed(const Params& p,uint8_t* sm,uint64_t* b,int slot,int row,int group){
 if(threadIdx.x)return;uint8_t* s=sm+slot*24576;mbar_arrive_expect_tx(b+slot,24576);
 tma_load_2d(s,&p.x,b+slot,0,row);tma_load_2d(s+8192,&p.x,b+slot,64,row);
 tma_load_2d(s+16384,group<4?&p.dl:&p.dr,b+slot,row,(group%4)*64);
}
TMN_DEVI void weight_role(const Params& p,uint8_t* sm,uint64_t* b,const float* gamma,const float* beta){
 int group=blockIdx.x%8,split=blockIdx.x/8,wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4,mat=lane/8;
 int rounds=(p.tiles-split+DW_SPLITS-1)/DW_SPLITS,segmentRounds=(rounds+WGRAD_SLICES-1)/WGRAD_SLICES,r=0;float dw[64]={};
 if(threadIdx.x==0){mbar_arrive_expect_tx(b+4,32768);for(int wg=0;wg<2;++wg)for(int k=0;k<2;++k)tma_load_2d(sm+49152+wg*16384+k*8192,&p.pre,b+4,k*64,group*128+wg*64);}
 if(split<p.tiles)load_dw_packed(p,sm,b,0,split*64,group);if(split+DW_SPLITS<p.tiles)load_dw_packed(p,sm,b,1,(split+DW_SPLITS)*64,group);
 mbar_wait(b+4,0);
 for(int tile=split;tile<p.tiles;tile+=DW_SPLITS,++r){int slot=r&1;uint8_t* s=sm+slot*24576;mbar_wait(b+slot,(r/2)&1);
  uint32_t xn[8][4];load_frag_bf16<8,8192>(xn,smem_u32(s),w*16,lane);allsync();
  ln_recompute_fragment<8,B7_LN_SERIAL>(xn,gamma,beta,lane,1e-5f);
  if(wi==0){static_for<8>([&](auto kk){constexpr int k=decltype(kk)::value;
   stsm_x4(smem_u32(s+(k/4)*8192)+swz128(w*16+lane%8+8*(mat&1),(2*(k%4)+(mat>>1))*16),xn[k][0],xn[k][1],xn[k][2],xn[k][3]);});}
  float acc[32]={};uint64_t desc=smem_desc(smem_u32(sm+49152+wi*16384),16,1024,1);uint32_t lo=uint32_t(desc),hi=uint32_t(desc>>32);
  fence_regs(acc);wgmma_fence();
  static_for<8>([&](auto kk){constexpr int k=decltype(kk)::value;wgmma_m64n64k16_rs_off<(k/4)*8192+(k%4)*32>(acc,xn[k],lo,hi,k>0);});
  wgmma_commit();wgmma_wait<0>();fence_regs(acc);
  uint8_t* sg=sm+81920+wi*8192;int ra=w*16+lane/4;uint32_t maskA=__bfloat16_as_ushort(p.mask[tile*64+ra]),maskB=__bfloat16_as_ushort(p.mask[tile*64+ra+8]);maskA|=maskA<<16;maskB|=maskB<<16;
  #pragma unroll
  for(int q=0;q<2;++q){uint32_t dy[4],dg[4],dp[4];
   ldsm_x4_t(dy,smem_u32(s+16384)+swz128(wi*32+q*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2));
   #pragma unroll
   for(int j=0;j<4;++j){uint32_t masked,m=(j&1)?maskB:maskA;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(dy[j]),"r"(m));
    float ga=math::sigmoid(math::round_bf16(acc[q*8+j*2])),gb=math::sigmoid(math::round_bf16(acc[q*8+j*2+1]));
    float pa=math::round_bf16(acc[(q+2)*8+j*2]),pb=math::round_bf16(acc[(q+2)*8+j*2+1]);
    dg[j]=pack_bf16(((bf16lo(masked)*pa)*ga)*(1.f-ga),((bf16hi(masked)*pb)*gb)*(1.f-gb));dp[j]=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
   }
   uint32_t addr=swz128(q*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2);
   stsm_x4_t(smem_u32(sg)+addr,dg[0],dg[1],dg[2],dg[3]);stsm_x4_t(smem_u32(sg+4096)+addr,dp[0],dp[1],dp[2],dp[3]);
  }
  fence_proxy_async();allsync();fence_regs(dw);wgmma_fence();
  static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;mma_weight128(dw,smem_desc(smem_u32(sg+k*32),16,1024,1),smem_desc(smem_u32(s+k*2048),8192,1024,1),r%segmentRounds>0||k>0);});
  wgmma_commit();wgmma_wait<0>();fence_regs(dw);allsync();
  if(tile+2*DW_SPLITS<p.tiles)load_dw_packed(p,sm,b,slot,(tile+2*DW_SPLITS)*64,group);
  if((r+1)%segmentRounds==0||tile+DW_SPLITS>=p.tiles){float* out=p.partw+((group*DW_SPLITS+split)*WGRAD_SLICES+r/segmentRounds)*16384;
   static_for<16>([&](auto qq){constexpr int q=decltype(qq)::value;int rr=w*16+lane/4,c=q*8+2*(lane%4),kind=rr/32,h=wi*32+rr%32;
    stg64f(out+kind*8192+h*128+c,dw[q*4],dw[q*4+1]);stg64f(out+kind*8192+(h+8)*128+c,dw[q*4+2],dw[q*4+3]);});}
 }
}
