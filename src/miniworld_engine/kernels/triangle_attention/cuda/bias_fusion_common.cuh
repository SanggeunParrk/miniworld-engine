// Helpers shared by the two grouped dK/dV + bias-gradient sources: the prebuilt H=4 bias_fusion.cu and the
// runtime-head h100_sources/bias_fusion.cu (B200-style route). Included after `using Element` / `using namespace cute`.
#pragma once
__device__ __forceinline__ float exp2_approx(float x) {
  float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}


__device__ __forceinline__ float sm_load_bf16(uint32_t p) {
  uint16_t v; asm volatile("ld.shared.b16 %0, [%1];" : "=h"(v) : "r"(p) : "memory");
  return float(Element::bitcast(v));
}
__device__ __forceinline__ void sm_store_bf16(uint32_t p,Element v) {
  asm volatile("st.shared.b16 [%0], %1;" :: "r"(p),"h"(v.raw()) : "memory");
}
__device__ __forceinline__ float sm_load_f32(uint32_t p) {
  float v; asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(p) : "memory"); return v;
}
__device__ __forceinline__ void sm_store_f32(uint32_t p,float v) {
  asm volatile("st.shared.f32 [%0], %1;" :: "r"(p),"f"(v) : "memory");
}

// PTX cvt packs its first input high, second input low (CUTLASS convention).
__device__ __forceinline__ void sm_store_pair(uint32_t p,float lo,float hi) {
  uint32_t packed;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(packed) : "f"(hi),"f"(lo));
  asm volatile("st.shared.b32 [%0], %1;" :: "r"(p),"r"(packed) : "memory");
}


// Two independent MMAs share one WGMMA fence/commit/wait group.
template<bool Zero, class MMA, class A0, class B0, class C0, class A1, class B1, class C1>
__device__ __forceinline__ void gemm_pair(MMA& mma,A0& a0,B0& b0,C0& c0,A1& a1,B1& b1,C1& c1){
  constexpr bool RS=!cute::is_base_of<cute::GMMA::DescriptorIterator,typename MMA::FrgTypeA>::value;
  if constexpr(RS){warpgroup_fence_operand(a0);warpgroup_fence_operand(a1);}
  warpgroup_fence_operand(c0);warpgroup_fence_operand(c1);warpgroup_arrive();
  mma.accumulate_=Zero?GMMA::ScaleOut::Zero:GMMA::ScaleOut::One;
  #pragma unroll
  for(int k=0;k<size<2>(a0);++k){cute::gemm(mma,a0(_,_,k),b0(_,_,k),c0);mma.accumulate_=GMMA::ScaleOut::One;}
  mma.accumulate_=Zero?GMMA::ScaleOut::Zero:GMMA::ScaleOut::One;
  #pragma unroll
  for(int k=0;k<size<2>(a1);++k){cute::gemm(mma,a1(_,_,k),b1(_,_,k),c1);mma.accumulate_=GMMA::ScaleOut::One;}
  warpgroup_commit_batch();warpgroup_wait<0>();
  warpgroup_fence_operand(c0);warpgroup_fence_operand(c1);
  if constexpr(RS){warpgroup_fence_operand(a0);warpgroup_fence_operand(a1);}
}

template<class TMA,class GT,class ST>
__device__ __forceinline__ void tma_copy(TMA const& tma,GT const& src,ST const& dst,cutlass::arch::ClusterTransactionBarrier& full) {
  auto c=tma.get_slice(_0{});
  copy(tma.with(reinterpret_cast<uint64_t&>(full)),c.partition_S(src),c.partition_D(dst));
}

__global__ void reduce_bias(float const* part,Element* out,int L,int groups) {
  int h=blockIdx.z,nt=L/32,nkt=L/64;
  float acc[8]={};
  for(int g=0;g<groups;++g) {
    int64_t base=(((int64_t(h)*groups+g)*nt+blockIdx.y)*nkt+blockIdx.x)*512+threadIdx.x;
    #pragma unroll
    for(int j=0;j<2;++j) {
      float4 x=reinterpret_cast<float4 const*>(part)[base+j*256];
      acc[j*4+0]+=x.x;acc[j*4+1]+=x.y;acc[j*4+2]+=x.z;acc[j*4+3]+=x.w;
    }
  }
  #pragma unroll
  for(int j=0;j<2;++j) {
    #pragma unroll
    for(int c=0;c<4;++c) {
      int idx=(threadIdx.x+j*256)*4+c,lane=(idx/4)%128,x=(idx/512)*4+((idx%4)/2)*2+(idx%2);
      int k=blockIdx.x*64+(lane/32)*16+(lane%32)/4+((x%4)/2)*8;
      int q=blockIdx.y*32+(lane%4)*2+(x%2)+(x/4)*8;
      out[(h*L+q)*L+k]=Element(acc[j*4+c]);
    }
  }
}

// Small lengths need more independent output CTAs to fill the GPU.
__global__ void reduce_bias_small(float const* part,Element* out,int L,int groups) {
  int h=blockIdx.z,nt=L/32,nkt=L/64,jt=blockIdx.y/4,chunk=blockIdx.y%4;
  float a=0.f,b=0.f;
  for(int g=0;g<groups;++g) {
    int64_t base=(((int64_t(h)*groups+g)*nt+jt)*nkt+blockIdx.x)*1024+chunk*256+threadIdx.x;
    float2 x=reinterpret_cast<float2 const*>(part)[base];a+=x.x;b+=x.y;
  }
  #pragma unroll
  for(int c=0;c<2;++c) {
    int idx=chunk*512+threadIdx.x*2+c,lane=(idx/4)%128,x=(idx/512)*4+((idx%4)/2)*2+(idx%2);
    int k=blockIdx.x*64+(lane/32)*16+(lane%32)/4+((x%4)/2)*8;
    int q=jt*32+(lane%4)*2+(x%2)+(x/4)*8;
    out[(h*L+q)*L+k]=Element(c?b:a);
  }
}
