// SPDX-License-Identifier: Apache-2.0
TMN_DEVI void mma_dxpacked(float (&d)[32],uint64_t a,uint64_t b,int accumulate){
 asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 1, 1; }"
 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]) : "l"(a),"l"(b),"r"(accumulate));
}
TMN_DEVI void packed_derivatives(const Params& p,uint32_t (&xn)[8][4],uint8_t* weights,uint8_t* dy_s,uint8_t* sg,uint8_t* sp,int row){
 int wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4,mat=lane/8;
  float acc[32]={};uint64_t desc=smem_desc(smem_u32(weights+wi*16384),16,1024,1);uint32_t lo=uint32_t(desc),hi=uint32_t(desc>>32);
  fence_regs(acc);wgmma_fence();
  static_for<8>([&](auto kk){constexpr int k=decltype(kk)::value;wgmma_m64n64k16_rs_off<(k/4)*8192+(k%4)*32>(acc,xn[k],lo,hi,k>0);});
  wgmma_commit();wgmma_wait<0>();fence_regs(acc);
  int ra=w*16+lane/4;uint32_t maskA=__bfloat16_as_ushort(p.mask[row+ra]),maskB=__bfloat16_as_ushort(p.mask[row+ra+8]);maskA|=maskA<<16;maskB|=maskB<<16;
  #pragma unroll
  for(int q=0;q<2;++q){uint32_t dy[4],dg[4],dp[4];
   ldsm_x4_t(dy,smem_u32(dy_s)+swz128(wi*32+q*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2));
   #pragma unroll
   for(int j=0;j<4;++j){uint32_t masked,m=(j&1)?maskB:maskA;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(masked):"r"(dy[j]),"r"(m));
    float ga=math::sigmoid(math::round_bf16(acc[q*8+j*2])),gb=math::sigmoid(math::round_bf16(acc[q*8+j*2+1]));
    float pa=math::round_bf16(acc[(q+2)*8+j*2]),pb=math::round_bf16(acc[(q+2)*8+j*2+1]);
    dg[j]=pack_bf16(((bf16lo(masked)*pa)*ga)*(1.f-ga),((bf16hi(masked)*pb)*gb)*(1.f-gb));dp[j]=pack_bf16(bf16lo(masked)*ga,bf16hi(masked)*gb);
   }
   uint32_t addr=swz128(wi*32+q*16+lane%8+8*(mat>>1),(w*16+8*(mat&1))*2);
   stsm_x4_t(smem_u32(sg)+addr,dg[0],dg[1],dg[2],dg[3]);stsm_x4_t(smem_u32(sp)+addr,dp[0],dp[1],dp[2],dp[3]);
  }
 fence_proxy_async();allsync();
}
