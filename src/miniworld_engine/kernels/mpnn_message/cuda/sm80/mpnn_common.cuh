// mpnn_common.cuh -- the PTX helpers and the GELU of the A100 (sm_80) MPNN message-side kernels (mpnn_message, mpnn_node_message, mpnn_relative_position):
// cp.async, ldmatrix, mma.sync m16n8k16 bf16 -> fp32, bf16 packing, and the exact-GELU / GELU-derivative statements.
//
// GELU.  gelu(x) = x Phi(x) = relu(x) - u tail(u), u = |x|, tail(u) = 1 - Phi(u) = 0.5 erfc(u / sqrt 2); gelu'(x) = Phi(x) + x phi(x) = (x >= 0) ? 1 - w : w with
// w = tail(u) - u phi(u).  tail(u) is a Hastings-type form  e R(t),  e = exp(-u^2 / 2) = 2^(-v^2) (v = sqrt(log2(e) / 2) u: ONE ex2), t = 1 / (1 + c u) (ONE rcp), R(t) = t (r_0 + r_1 t + ...)
// a polynomial of degree D in t, fitted (probes/fit_gelu2.py) to minimise max_u e(u) |dR| max(1, u): the absolute error of gelu and of gelu' together.  D = 3: 1.35e-5 (both), 9 FMA-pipe
// instructions + 2 MUFU per gelu against ~30 + 1 of libdevice's erff; D = 4: 1.1e-6, D = 5: 9.8e-8.  Both functions are exact in the sense that matters: the result is rounded to bf16
// right away (half an ulp at 0.5 is 2e-3).  gelu / gelu' are bit-identical wherever they are recomputed (forward replay in a backward pass).
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__

namespace mp80 {

typedef __nv_bfloat16 bf;

DEVI uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// ---- cp.async (16 B, L2-only .cg); src_bytes 0 zero-fills the destination (the operand must be a register)
DEVI void cp_async16(uint32_t dst, const void* src, uint32_t src_bytes = 16) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
DEVI void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> DEVI void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

// ---- ldmatrix.  Not volatile, with a memory clobber: ordered against the stores / barriers / cp.async waits that make the tile valid, free to move against the arithmetic -- the compiler can hoist the
// next step's fragment loads above the mma that consume this step's (volatile asm statements keep their source order among themselves, which serialised ldmatrix -> mma chains).
#ifndef MP_ASM_RELAXED
#define MP_ASM_RELAXED 1
#endif
#if MP_ASM_RELAXED
#define MP_ASM_FRAG asm
#else
#define MP_ASM_FRAG asm volatile
#endif
DEVI void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  MP_ASM_FRAG("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
              : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
DEVI void ldsm_x4_t(uint32_t (&r)[4], uint32_t addr) {
  MP_ASM_FRAG("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
              : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}

// ---- mma.sync m16n8k16 bf16 x bf16 -> fp32 (accumulate in place): a pure register operation
DEVI void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  MP_ASM_FRAG("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
              : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
              : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// ---- shared / global vector access
DEVI uint2 lds64(uint32_t a) { uint2 v; asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "r"(a)); return v; }
DEVI uint4 lds128(uint32_t a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v;
}
DEVI uint32_t lds32(uint32_t a) { uint32_t v; asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(a)); return v; }
DEVI void sts32(uint32_t a, uint32_t v) { asm volatile("st.shared.u32 [%0], %1;\n" ::"r"(a), "r"(v) : "memory"); }
DEVI void sts64(uint32_t a, uint2 v) { asm volatile("st.shared.v2.u32 [%0], {%1,%2};\n" ::"r"(a), "r"(v.x), "r"(v.y) : "memory"); }
DEVI void sts128(uint32_t a, uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
DEVI uint4 ldg128(const void* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }
// a read-only global load that stays where it is written (volatile asm): issued EARLY by the kernels that use the value much later (the compiler would sink a plain __ldg to its first use and stall there)
DEVI float ldg_f32(const float* p) { float v; asm volatile("ld.global.nc.f32 %0, [%1];\n" : "=f"(v) : "l"(p) : "memory"); return v; }
DEVI void stg128(void* p, uint4 v) {
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
DEVI void stg32(void* p, uint32_t v) { asm volatile("st.global.u32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory"); }

// ---- bf16 packing: lo is the lower half (the lower channel of a pair)
DEVI uint32_t pack_bf16(float lo, float hi) {
  uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
DEVI float round_bf16f(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
DEVI float bf16lo(uint32_t v) { return __uint_as_float(v << 16); }
DEVI float bf16hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

// bf16(a + b) per half of two bf16 pairs, ONE rounding of the exact sum (fma.rn.bf16x2 a * 1 + b; sm_80 has no add.bf16x2): one instruction for what is 7 with fp32 adds (2 unpacks of each operand, 2 FADD, 1 pack).
// The fp32 route of the reference rounds twice (the exact sum, then to bf16); the two differ only when the fp32 sum itself is inexact (exponents more than 16 apart) AND lands on a bf16 tie.
DEVI uint32_t add_bf16x2(uint32_t a, uint32_t b) {
  uint32_t r; asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(0x3F803F80u), "r"(b)); return r;
}

#ifndef MP_NODE_HADD2
#define MP_NODE_HADD2 1          // 1: the bf16 additions of the node message epilogue are packed bf16x2 instructions
#endif
// pre = bf16(bf16(pr + q) + nb) on bf16 pairs (pr = the rounded edge projection, q = the query projection, nb = the gathered neighbour projection): the node message's two bf16 additions
DEVI uint32_t node_pre(uint32_t pr, uint32_t q, uint32_t nb) {
#if MP_NODE_HADD2
  return add_bf16x2(add_bf16x2(pr, q), nb);
#else
  const uint32_t sq = pack_bf16(bf16lo(pr) + bf16lo(q), bf16hi(pr) + bf16hi(q));
  return pack_bf16(bf16lo(sq) + bf16lo(nb), bf16hi(sq) + bf16hi(nb));
#endif
}

// ---- warp helpers
DEVI float quad_sum(float v) {          // the four lanes of a quad (lane & 3)
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v;
}
DEVI float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// 16-byte granule XOR swizzle of a 256-B row (a row of 128 bf16): granule 0..15 of row r is stored at granule (g ^ (r & 7)); the eight rows of an 8x8 ldmatrix matrix then hit eight distinct 16-B bank groups
DEVI uint32_t sw256(uint32_t row, uint32_t granule) { return row * 256u + ((granule ^ (row & 7u)) << 4); }

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ GELU
DEVI float rcp_approx(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }
DEVI float ex2_approx(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }

constexpr float kKc = 0.84932180028801907f;          // sqrt(log2(e) / 2): v = kKc u, exp(-u^2 / 2) = 2^(-v^2)
constexpr float kInvSqrt2Pi = 0.39894228040143268f;

// minimax constants (probes/fit_gelu2.py): c and the r_u of tail(u) = e(u) R(t), t = 1 / (1 + c u), R(t) = t (r_0 + r_1 t + ...); the v-scaled forms are c_v = c / kKc, r_v = r_u / kKc
template <int D> struct GeluK;
template <> struct GeluK<3> {
  static constexpr float c = 0.33570095608458006f, cv = 0.39525767026201175f;
  static DEVI float polyu(float t) { float q = fmaf(0.3594824876934086f, t, -0.029304118095285766f); q = fmaf(q, t, 0.1698351342152147f); return q * t; }
  static DEVI float polyv(float t) { float q = fmaf(0.4232582839290151f, t, -0.034502962346366545f); q = fmaf(q, t, 0.19996558920025462f); return q * t; }
};
template <> struct GeluK<4> {
  static constexpr float c = 0.2718752016288661f, cv = 0.3201085872712424f;
  static DEVI float polyu(float t) {
    float q = fmaf(0.40617876294637273f, t, -0.26696117204089403f); q = fmaf(q, t, 0.282929418149925f); q = fmaf(q, t, 0.07785408901814653f); return q * t;
  }
  static DEVI float polyv(float t) {
    float q = fmaf(0.47823894642599635f, t, -0.31432275958342654f); q = fmaf(q, t, 0.3331239325941933f); q = fmaf(q, t, 0.09166618470377767f); return q * t;
  }
};
template <> struct GeluK<5> {
  static constexpr float c = 0.2328617183436297f, cv = 0.27417372103796517f;
  static DEVI float polyu(float t) {
    float q = fmaf(0.5016138636073363f, t, -0.6558108610322126f); q = fmaf(q, t, 0.6425999236852433f); q = fmaf(q, t, -0.1109846894769802f); q = fmaf(q, t, 0.12258186161097141f); return q * t;
  }
  static DEVI float polyv(float t) {
    float q = fmaf(0.5906051904439881f, t, -0.7721582806538303f); q = fmaf(q, t, 0.7566035906146846f); q = fmaf(q, t, -0.1306744857359642f); q = fmaf(q, t, 0.14432911243936264f); return q * t;
  }
};

#ifndef MP_GELU_D
#define MP_GELU_D 3
#endif

// gelu(x), D = degree of R: 1 FMUL (v), 1 FFMA, rcp, D - 1 FFMA + 1 FMUL (R), 1 FMUL (v^2), ex2, 1 FMUL, 1 FFMA, 1 FMNMX
template <int D = MP_GELU_D>
DEVI float gelu_f(float x) {
  using K = GeluK<D>;
  const float v = fabsf(x) * kKc;
  const float t = rcp_approx(fmaf(K::cv, v, 1.f));
  const float q = K::polyv(t);
  const float e = ex2_approx(-(v * v));
  return fmaf(-v, e * q, fmaxf(x, 0.f));
}

// gelu'(x)
template <int D = MP_GELU_D>
DEVI float gelu_grad_f(float x) {
  using K = GeluK<D>;
  const float u = fabsf(x);
  const float t = rcp_approx(fmaf(K::c, u, 1.f));
  const float q = K::polyu(t);
  const float w0 = fmaf(-kInvSqrt2Pi, u, q);
  const float v = u * kKc;
  const float w = ex2_approx(-(v * v)) * w0;
  return 0.5f + copysignf(0.5f - w, x);                // x >= 0: 1 - w;  x < 0: w
}

}  // namespace mp80
