// SPDX-License-Identifier: Apache-2.0
// Anthropic v5 TMA/WGMMA and LN fragments; Miniworld on-chip recomputation.
#pragma once
#include "warp_primitives.cuh"
#include "ln_recompute.cuh"
TMN_DEVI void sync_group(){named_bar_sync(1+threadIdx.x/128,128);}
TMN_DEVI void mma_recompute(float (&d)[32],uint64_t a,uint64_t b,int accumulate){
 asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 0, 1; }"
 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]) : "l"(a),"l"(b),"r"(accumulate));
}
template<int KS,bool TRANSPOSE=false,bool SERIAL=false,bool STORE=true>
TMN_DEVI LnStats normalize_tile(uint8_t* src,uint8_t* dst,const float* gamma,const float* beta){
 int lane=threadIdx.x%32,w=(threadIdx.x/32)%4,mat=lane/8,r8=lane%8;
 uint32_t f[KS][4];
 if constexpr(TRANSPOSE){
  static_for<KS>([&](auto kk){constexpr int k=decltype(kk)::value;
   ldsm_x4_t(f[k],smem_u32(src)+swz128(k*16+r8+8*(mat>>1),(w*16+8*(mat&1))*2));});
 }else load_frag_bf16<KS,8192>(f,smem_u32(src),w*16,lane);
 if constexpr(TRANSPOSE && STORE){if(src==dst)sync_group();}
 LnStats st=ln_recompute_fragment<KS,(SERIAL && STORE)>(f,gamma,beta,lane,1e-5f);
 if constexpr(STORE){
  int lrow=w*16+r8+8*(mat&1);
  static_for<KS>([&](auto kk){constexpr int k=decltype(kk)::value;uint8_t* s=dst+(k/4)*8192;
   stsm_x4(smem_u32(s)+swz128(lrow,(2*(k%4)+(mat>>1))*16),f[k][0],f[k][1],f[k][2],f[k][3]);});
 }
 return st;
}
template<int K,int BPITCH>
TMN_DEVI void recompute_gemm(float (&acc)[32],uint8_t* a,uint8_t* b){
#pragma unroll
 for(int j=0;j<32;++j)acc[j]=0.f;
 fence_regs(acc);wgmma_fence();
 static_for<K/16>([&](auto kk){constexpr int k=decltype(kk)::value;
  mma_recompute(acc,smem_desc(smem_u32(a+(k/4)*8192+(k%4)*32),16,1024,1),
    smem_desc(smem_u32(b+(k/4)*BPITCH+(k%4)*2048),8192,1024,1),k>0);});
 wgmma_commit();wgmma_wait<0>();fence_regs(acc);
}
template<bool SIGMOID=false>
TMN_DEVI void store_recomputed(float (&acc)[32],uint8_t* dst){
 int lane=threadIdx.x%32,w=(threadIdx.x/32)%4,mat=lane/8,lrow=w*16+(lane%8)+8*(mat&1);
#pragma unroll
 for(int q=0;q<4;++q){uint32_t f[4];
#pragma unroll
  for(int j=0;j<4;++j){float a=acc[q*8+j*2],b=acc[q*8+j*2+1];
   if constexpr(SIGMOID){a=math::sigmoid(math::round_bf16(a));b=math::sigmoid(math::round_bf16(b));}f[j]=pack_bf16(a,b);}
  stsm_x4(smem_u32(dst)+swz128(lrow,(2*q+(mat>>1))*16),f[0],f[1],f[2],f[3]);
 }
}
