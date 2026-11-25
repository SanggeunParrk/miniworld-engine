// esm_pack.cu — ESMFold2's TRAINING weight packs in ONE launch (AtomDiTESMFold2Pair on the tcgen05 local-DiT block, round 3 part 7,
// 2026-11-17; sm_100a, no tensor cores). A captured training step repeats its packs (the weights change between replays); the torch packs
// were ~30 launches of casts, cats and transposes (bf16) or ~12 (fp32). The layouts are the caller's (``local_dit_esm_b200``): a table of
// up to 24 [128][128] units, each copied from its source (fp32 or bf16, a row pitch) to one or two places of the flat forward pack BF and,
// optionally, transposed into the flat pack BT; a null source writes zeros. F32 = 0: bf16 outputs (the bf16 / master block); F32 = 1: fp32
// outputs on the TF32 grid (cvt.rna, the fp32 block's kind::tf32 MMAs). The last CTA copies up to 4 [128] vectors into VEC (fp32, unrounded;
// a null source: zeros).
// One CTA per 16-column strip of a unit (8 per unit, as var_pack.cu). The unit table is a by-value parameter read with CONSTANT indices (a
// compare chain): indexed at run time the compiler copies it into every thread's local memory (col_jobs.cu).
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <cuda_bf16.h>

#ifndef F32
#define F32 0
#endif

struct Unit {
  const void* src;                 // [128][128] at src (row pitch sp elements); null: zeros
  int32_t sp, sbf;                 // source row pitch, source is bf16
  int32_t d0, p0;                  // BF: first element, row pitch
  int32_t d1, p1;                  // BF second copy (d1 < 0: none)
  int32_t dt, pt;                  // BT transposed (dt < 0: none)
  int32_t pad_;
};
struct Units {
  Unit u[24];
  const void* vsrc[4];
  int32_t vbf[4];
};

static __device__ __forceinline__ float rna_tf32(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return __uint_as_float(r); }

extern "C" __global__ void __launch_bounds__(256)
atom_esm_pack(const Units U, int nunit, int nvec, void* __restrict__ BF, void* __restrict__ BT, float* __restrict__ VEC) {
  const int tid = threadIdx.x;
  if ((int)blockIdx.x >= 8 * nunit) {                                      // the vectors
    for (int t = tid; t < nvec * 128; t += 256) {
      const int s = t >> 7, c = t & 127;
      const void* vs = nullptr;
      int vb = 0;
#pragma unroll
      for (int k = 0; k < 4; ++k)
        if (s == k) { vs = U.vsrc[k]; vb = U.vbf[k]; }
      float x = 0.f;
      if (vs != nullptr) x = vb ? __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(vs)[c]) : reinterpret_cast<const float*>(vs)[c];
      VEC[t] = x;
    }
    return;
  }
  const int u = (int)blockIdx.x >> 3, c16 = 16 * ((int)blockIdx.x & 7);
  const void* src = nullptr;
  int sp = 0, sbf = 0, d0 = 0, p0 = 0, d1 = -1, p1 = 0, dt = -1, pt = 0;
#pragma unroll
  for (int k = 0; k < 24; ++k)
    if (u == k) { src = U.u[k].src; sp = U.u[k].sp; sbf = U.u[k].sbf; d0 = U.u[k].d0; p0 = U.u[k].p0; d1 = U.u[k].d1; p1 = U.u[k].p1; dt = U.u[k].dt; pt = U.u[k].pt; }
  __shared__ float tile[128][17];
  const int r = tid & 127, h8 = 8 * (tid >> 7);                            // row r, columns c16 + h8 .. + 7
  float x[8];
  if (src == nullptr) {
#pragma unroll
    for (int e = 0; e < 8; ++e) x[e] = 0.f;
  } else if (sbf) {
    const uint4 h = __ldg(reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(src) + (size_t)r * sp + c16 + h8));
    const uint32_t w4[4] = {h.x, h.y, h.z, h.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) { x[2 * e] = __uint_as_float(w4[e] << 16); x[2 * e + 1] = __uint_as_float(w4[e] & 0xffff0000u); }
  } else {
    const float* p = reinterpret_cast<const float*>(src) + (size_t)r * sp + c16 + h8;
    const float4 a = __ldg(reinterpret_cast<const float4*>(p)), b = __ldg(reinterpret_cast<const float4*>(p + 4));
    x[0] = a.x; x[1] = a.y; x[2] = a.z; x[3] = a.w; x[4] = b.x; x[5] = b.y; x[6] = b.z; x[7] = b.w;
  }
#if F32
#pragma unroll
  for (int e = 0; e < 8; ++e) x[e] = rna_tf32(x[e]);
  {
    float* o = reinterpret_cast<float*>(BF);
    const float4 a = make_float4(x[0], x[1], x[2], x[3]), b = make_float4(x[4], x[5], x[6], x[7]);
    *reinterpret_cast<float4*>(o + d0 + r * p0 + c16 + h8) = a; *reinterpret_cast<float4*>(o + d0 + r * p0 + c16 + h8 + 4) = b;
    if (d1 >= 0) { *reinterpret_cast<float4*>(o + d1 + r * p1 + c16 + h8) = a; *reinterpret_cast<float4*>(o + d1 + r * p1 + c16 + h8 + 4) = b; }
  }
#else
  {
    uint32_t w[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const __nv_bfloat16 lo = __float2bfloat16_rn(x[2 * e]), hi = __float2bfloat16_rn(x[2 * e + 1]);
      w[e] = (uint32_t)(*reinterpret_cast<const uint16_t*>(&lo)) | ((uint32_t)(*reinterpret_cast<const uint16_t*>(&hi)) << 16);
      x[2 * e] = __bfloat162float(lo); x[2 * e + 1] = __bfloat162float(hi);  // (the transpose takes the rounded values)
    }
    uint16_t* o = reinterpret_cast<uint16_t*>(BF);
    *reinterpret_cast<uint4*>(o + d0 + r * p0 + c16 + h8) = make_uint4(w[0], w[1], w[2], w[3]);
    if (d1 >= 0) *reinterpret_cast<uint4*>(o + d1 + r * p1 + c16 + h8) = make_uint4(w[0], w[1], w[2], w[3]);
  }
#endif
  if (dt < 0) return;
#pragma unroll
  for (int e = 0; e < 8; ++e) tile[r][h8 + e] = x[e];
  __syncthreads();
  const int j = tid >> 4, r8 = 8 * (tid & 15);                             // BT row c16 + j (the strip's column j), columns r8 .. + 7
#if F32
  float* o = reinterpret_cast<float*>(BT) + dt + (c16 + j) * pt + r8;
  *reinterpret_cast<float4*>(o) = make_float4(tile[r8][j], tile[r8 + 1][j], tile[r8 + 2][j], tile[r8 + 3][j]);
  *reinterpret_cast<float4*>(o + 4) = make_float4(tile[r8 + 4][j], tile[r8 + 5][j], tile[r8 + 6][j], tile[r8 + 7][j]);
#else
  uint32_t w[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const __nv_bfloat16 lo = __float2bfloat16_rn(tile[r8 + 2 * e][j]), hi = __float2bfloat16_rn(tile[r8 + 2 * e + 1][j]);
    w[e] = (uint32_t)(*reinterpret_cast<const uint16_t*>(&lo)) | ((uint32_t)(*reinterpret_cast<const uint16_t*>(&hi)) << 16);
  }
  *reinterpret_cast<uint4*>(reinterpret_cast<uint16_t*>(BT) + dt + (c16 + j) * pt + r8) = make_uint4(w[0], w[1], w[2], w[3]);
#endif
}
