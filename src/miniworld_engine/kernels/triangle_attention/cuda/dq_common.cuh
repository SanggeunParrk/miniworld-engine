// Helpers shared by the two query-owned dQ sources: the prebuilt H=4 dq.cu and the runtime-head h100_sources/dq.cu
// (B200-style route). Included after `using Element` / `using namespace cute`.
#pragma once
__device__ __forceinline__ float ex2(float x){float y;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
__device__ __forceinline__ float sload(uint32_t p){uint16_t v;asm volatile("ld.shared.b16 %0,[%1];":"=h"(v):"r"(p):"memory");return float(Element::bitcast(v));}
__device__ __forceinline__ float sloadf(uint32_t p){float v;asm volatile("ld.shared.f32 %0,[%1];":"=f"(v):"r"(p):"memory");return v;}
__device__ __forceinline__ void spair(uint32_t p,float lo,float hi){uint32_t x;asm("cvt.rn.bf16x2.f32 %0,%1,%2;":"=r"(x):"f"(hi),"f"(lo));asm volatile("st.shared.b32 [%0],%1;"::"r"(p),"r"(x):"memory");}
template<class TMA,class G,class S> __device__ __forceinline__ void copy_tile(TMA const& tma,G const& src,S const& dst,cutlass::arch::ClusterTransactionBarrier& ready){auto c=tma.get_slice(_0{});copy(tma.with(reinterpret_cast<uint64_t&>(ready)),c.partition_S(src),c.partition_D(dst));}
