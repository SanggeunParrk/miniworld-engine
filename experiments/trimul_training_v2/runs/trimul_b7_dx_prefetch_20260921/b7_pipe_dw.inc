// SPDX-License-Identifier: Apache-2.0
// Anthropic-style producer / two independent row consumers for fused dW.
TMN_DEVI void pipe_producer(const Params& p,uint8_t* sm,uint64_t* bar){
 int group=blockIdx.x%PIPE_GROUPS,split=blockIdx.x/PIPE_GROUPS;
 for(int r=0;(PIPE_HIDDEN64?split:PIPE_CONSUMERS*split)+r*(PIPE_HIDDEN64?DW_SPLITS:PIPE_CONSUMERS*DW_SPLITS)<p.tiles;++r){int slot=r&1;
  for(int c=0;c<PIPE_CONSUMERS;++c){int hg=PIPE_HIDDEN64?group*2+c:group,tile=(PIPE_HIDDEN64?split:PIPE_CONSUMERS*split+c)+r*(PIPE_HIDDEN64?DW_SPLITS:PIPE_CONSUMERS*DW_SPLITS),id=c*2+slot;if(tile>=p.tiles)continue;
   if(r>=2)mbar_wait(bar+2*PIPE_CONSUMERS+id,(r/2-1)&1);
   if(threadIdx.x==0){uint8_t* s=sm+id*20480;mbar_arrive_expect_tx(bar+id,20480);
    tma_load_2d(s,USE_SAVED_XN?&p.xn:&p.x,bar+id,0,tile*64);tma_load_2d(s+8192,USE_SAVED_XN?&p.xn:&p.x,bar+id,64,tile*64);
    tma_load_2d(s+16384,hg<8?&p.dl:&p.dr,bar+id,tile*64,(hg%8)*32);
   }
  }
 }
}
TMN_DEVI void pipe_consumer(const Params& p,uint8_t* sm,uint64_t* bar,const float* gamma,const float* beta){
 int group=blockIdx.x%PIPE_GROUPS,split=blockIdx.x/PIPE_GROUPS,cc=threadIdx.x/128-1,lane=threadIdx.x%32,w=(threadIdx.x/32)%4,mat=lane/8;
 int stride=PIPE_HIDDEN64?DW_SPLITS:PIPE_CONSUMERS*DW_SPLITS,first=PIPE_HIDDEN64?split:split*PIPE_CONSUMERS+cc,rounds=(p.tiles-first+stride-1)/stride,seg=(rounds+WGRAD_SLICES-1)/WGRAD_SLICES,r=0;
 float dw[64]={};uint8_t* sg=sm+PIPE_SG_BASE+cc*8192;
 mbar_wait(bar+4*PIPE_CONSUMERS,0);
 for(int tile=first;tile<p.tiles;tile+=stride,++r){int slot=r&1,id=cc*2+slot;uint8_t* s=sm+id*20480;mbar_wait(bar+id,(r/2)&1);
  uint32_t xn[8][4];load_frag_bf16<8,8192>(xn,smem_u32(s),w*16,lane);
#if !USE_SAVED_XN
  ln_recompute_fragment<8,B7_LN_SERIAL>(xn,gamma,beta,lane,1e-5f);
#endif
#if !B7_SKIP_XN_RESTORE || !USE_SAVED_XN
  static_for<8>([&](auto kk){constexpr int k=decltype(kk)::value;stsm_x4(smem_u32(s+(k/4)*8192)+swz128(w*16+lane%8+8*(mat&1),(2*(k%4)+(mat>>1))*16),xn[k][0],xn[k][1],xn[k][2],xn[k][3]);});
#endif
  float acc[32]={};uint64_t desc=smem_desc(smem_u32(sm+PIPE_W_BASE+(PIPE_HIDDEN64?cc*16384:0)),16,1024,1);uint32_t lo=uint32_t(desc),hi=uint32_t(desc>>32);
  fence_regs(acc);wgmma_fence();static_for<8>([&](auto kk){constexpr int k=decltype(kk)::value;wgmma_m64n64k16_rs_off<(k/4)*8192+(k%4)*32>(acc,xn[k],lo,hi,k>0);});wgmma_commit();wgmma_wait<0>();fence_regs(acc);
#if DW_OVERLAP
  // GP wait retires the preceding dW group. Delay releasing its XN tile
  // until here so next-tile LN and GP computation overlap that dW.
  if(r>0 && r%seg!=0){fence_regs(dw);named_bar_sync(1+cc,128);if(threadIdx.x%128==0)mbar_arrive(bar+2*PIPE_CONSUMERS+cc*2+((r-1)&1));}
#endif
  int ra=w*16+lane/4;uint32_t ma=__bfloat16_as_ushort(p.mask[tile*64+ra]),mb=__bfloat16_as_ushort(p.mask[tile*64+ra+8]);ma|=ma<<16;mb|=mb<<16;
  #pragma unroll
  for(int q=0;q<2;++q){uint32_t dy[4],dg[4],dp[4];ldsm_x4_t(dy,smem_u32(s+16384)+swz128(q*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2));
   #pragma unroll
   for(int j=0;j<4;++j){uint32_t masked,m=(j&1)?mb:ma;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(dy[j]),"r"(m));float ga=math::sigmoid(math::round_bf16(acc[q*8+j*2])),gb=math::sigmoid(math::round_bf16(acc[q*8+j*2+1])),pa=math::round_bf16(acc[(q+2)*8+j*2]),pb=math::round_bf16(acc[(q+2)*8+j*2+1]);
    dg[j]=pack_bf16(((bf16lo(masked)*pa)*ga)*(1.f-ga),((bf16hi(masked)*pb)*gb)*(1.f-gb));dp[j]=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
   }
   uint32_t addr=swz128(q*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2);stsm_x4_t(smem_u32(sg)+addr,dg[0],dg[1],dg[2],dg[3]);stsm_x4_t(smem_u32(sg+4096)+addr,dp[0],dp[1],dp[2],dp[3]);
  }
  fence_proxy_async();named_bar_sync(1+cc,128);fence_regs(dw);wgmma_fence();
  static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;mma_weight128(dw,smem_desc(smem_u32(sg+k*32),16,1024,1),smem_desc(smem_u32(s+k*2048),8192,1024,1),r%seg>0||k>0);});
  wgmma_commit();
#if DW_OVERLAP
  if((r+1)%seg==0||tile+stride>=p.tiles)
#endif
  {wgmma_wait<0>();fence_regs(dw);named_bar_sync(1+cc,128);if(threadIdx.x%128==0)mbar_arrive(bar+2*PIPE_CONSUMERS+id);}
  if((r+1)%seg==0||tile+stride>=p.tiles){float* out=p.partw+(((group*DW_SPLITS+split)*PIPE_CONSUMERS+cc)*WGRAD_SLICES+r/seg)*8192;
   static_for<16>([&](auto qq){constexpr int q=decltype(qq)::value;int rr=w*16+lane/4,c=q*8+2*(lane%4);stg64f(out+rr*128+c,dw[q*4],dw[q*4+1]);stg64f(out+(rr+8)*128+c,dw[q*4+2],dw[q*4+3]);});}
 }
}
