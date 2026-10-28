// Bias-only token DiT, attention core with a pair bias of its own per sample (bf16, mma.sync m16n8k16): the forward and backward
// that replace the shared-P cores (pvx / dpb2) when P differs per (head, sample).
//
// With P per sample nothing is shared across the samples (the shared-P cores read P once for 5 samples, and sum dO v^T over the
// samples), so the attention side is a pure stream: P [H, A, L, L] is read / written a fixed small number of times and the rest is
// K = 48 products. Both kernels are HBM-bound; mma.sync is enough (no wgmma / TMA).
//
//   psb_fwd   bias -> P = softmax(bias + key mask) -> og = sigmoid(g) (P v)       one pass: the bias tile is staged in shared
//             memory by cp.async, softmaxed in place (P written to HBM for the backward and kept as the MMA's A operand), v of the
//             (head, sample) staged beside it. Reads bias, writes P and og: 2 L^2 bytes per (head, sample) of HBM traffic.
//   psb_bwd   dV = P^T dO and dbias = P o (dO v^T - D), one pass over P: a CTA owns 128 keys of one (head, sample) and walks the
//             query rows in blocks of 64 (P tile and dO tile double-buffered by cp.async): dP = dO v^T on the tensor cores, the
//             bias gradient staged in shared memory and stored as 16-byte rows, dV accumulated in registers (P^T from the same
//             shared tile through ldmatrix.trans). Reads P, writes dbias: 2 L^2 bytes per (head, sample).
//
// Layouts: bias, P, dbias head-major [H, A, L, L]; v | g = vg [A L, 2 D_A] (columns h * DH + d), og / do [A L, D_A], dd [A, H, L].
// No atomics: every output element has one writer, so the results are bitwise repeatable.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <algorithm>
#include "bo_ptx.cuh"

namespace {
using bf = __nv_bfloat16;

template <typename T> T* P_(at::Tensor t) { return reinterpret_cast<T*>(t.data_ptr()); }

using namespace bop;

__device__ __forceinline__ void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// four 8 x 8 b16 matrices; lane l gives the row address of row l % 8 of matrix l / 8
__device__ __forceinline__ void ldsm4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm4t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm2t(uint32_t addr, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(addr));
}
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}
__device__ __forceinline__ float2 unpack2(uint32_t u) {
  const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&u);
  return make_float2(__bfloat162float(v.x), __bfloat162float(v.y));
}
__device__ __forceinline__ float sg(float v) { return __fdividef(1.f, 1.f + __expf(-v)); }
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// ------------------------------------------------------------------------------------------------------------------ forward
// CTA = (head, sample, a slice of the query-row blocks). The V of its (head, sample) and the key mask (as an additive 0 / -3e38 row)
// are staged once; the bias tiles [BM][L + 8] are double-buffered by cp.async (the next tile streams in while this one is softmaxed
// in place and multiplied), P goes to HBM as 16-byte rows from registers. Softmax: a half-warp (16 lanes x L / 128 chunks of 8
// elements) per row, two rows per warp, so no lane idles and every reduction stays inside 16 lanes. The product og = P v runs on
// BM / 16 row groups x 2 column halves of warps.
constexpr float L2E = 1.4426950408889634f;
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float hmax16(float v) {
#pragma unroll
  for (int o = 8; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float hsum16(float v) {
#pragma unroll
  for (int o = 8; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// softmax of one row (NCH chunks of 8 bf16 per lane of a half-warp) in shared memory, in place; the normalised bf16 row also goes to HBM
template <int NCH, bool MASKED, bool STORE_P>
__device__ __forceinline__ void soft_row(bf* __restrict__ row, bf* __restrict__ grow, const float* __restrict__ Ma, int hl, float2* __restrict__ st) {
  float v[NCH][8];
  float mx = -3.4e38f;
#pragma unroll
  for (int k = 0; k < NCH; ++k) {
    const int ch = hl + 16 * k;
    const uint4 u = *reinterpret_cast<const uint4*>(row + ch * 8);
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      v[k][2 * e] = __uint_as_float(w[e] << 16);
      v[k][2 * e + 1] = __uint_as_float(w[e] & 0xffff0000u);
    }
    if constexpr (MASKED) {
      const float4 m0 = *reinterpret_cast<const float4*>(Ma + ch * 8), m1 = *reinterpret_cast<const float4*>(Ma + ch * 8 + 4);
      v[k][0] += m0.x; v[k][1] += m0.y; v[k][2] += m0.z; v[k][3] += m0.w;
      v[k][4] += m1.x; v[k][5] += m1.y; v[k][6] += m1.z; v[k][7] += m1.w;
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) mx = fmaxf(mx, v[k][e]);
  }
  mx = hmax16(mx);
  const float mc = mx * L2E;
  float sum = 0.f;
#pragma unroll
  for (int k = 0; k < NCH; ++k)
#pragma unroll
    for (int e = 0; e < 8; ++e) { v[k][e] = ex2(fmaf(v[k][e], L2E, -mc)); sum += v[k][e]; }
  const float inv = 1.f / hsum16(sum);
  if (!STORE_P && hl == 0) *st = make_float2(mc, inv);       // the backward rebuilds P = ex2(x log2 e - mc) inv from these two
#pragma unroll
  for (int k = 0; k < NCH; ++k) {
    const int ch = hl + 16 * k;
    uint4 o;
    o.x = pack2(v[k][0] * inv, v[k][1] * inv);
    o.y = pack2(v[k][2] * inv, v[k][3] * inv);
    o.z = pack2(v[k][4] * inv, v[k][5] * inv);
    o.w = pack2(v[k][6] * inv, v[k][7] * inv);
    *reinterpret_cast<uint4*>(row + ch * 8) = o;
    if (STORE_P) *reinterpret_cast<uint4*>(grow + ch * 8) = o;
  }
}

// two rows (one per half-warp, two row pairs of a warp) softmaxed together: their load / reduction / exponential chains interleave
template <int NCH, bool MASKED, bool STORE_P>
__device__ __forceinline__ void soft_rows2(bf* __restrict__ row0, bf* __restrict__ row1, bf* __restrict__ g0, bf* __restrict__ g1,
                                           const float* __restrict__ Ma, int hl, float2* __restrict__ st0, float2* __restrict__ st1) {
  float v[2][NCH][8];
  float mx0 = -3.4e38f, mx1 = -3.4e38f;
#pragma unroll
  for (int k = 0; k < NCH; ++k) {
    const int ch = hl + 16 * k;
    const uint4 u0 = *reinterpret_cast<const uint4*>(row0 + ch * 8), u1 = *reinterpret_cast<const uint4*>(row1 + ch * 8);
    const uint32_t w0[4] = {u0.x, u0.y, u0.z, u0.w}, w1[4] = {u1.x, u1.y, u1.z, u1.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      v[0][k][2 * e] = __uint_as_float(w0[e] << 16);
      v[0][k][2 * e + 1] = __uint_as_float(w0[e] & 0xffff0000u);
      v[1][k][2 * e] = __uint_as_float(w1[e] << 16);
      v[1][k][2 * e + 1] = __uint_as_float(w1[e] & 0xffff0000u);
    }
    if constexpr (MASKED) {
      const float4 m0 = *reinterpret_cast<const float4*>(Ma + ch * 8), m1 = *reinterpret_cast<const float4*>(Ma + ch * 8 + 4);
      const float m[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
#pragma unroll
      for (int e = 0; e < 8; ++e) { v[0][k][e] += m[e]; v[1][k][e] += m[e]; }
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) { mx0 = fmaxf(mx0, v[0][k][e]); mx1 = fmaxf(mx1, v[1][k][e]); }
  }
#pragma unroll
  for (int o = 8; o; o >>= 1) {
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
  }
  const float mc0 = mx0 * L2E, mc1 = mx1 * L2E;
  float s0 = 0.f, s1 = 0.f;
#pragma unroll
  for (int k = 0; k < NCH; ++k)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      v[0][k][e] = ex2(fmaf(v[0][k][e], L2E, -mc0)); s0 += v[0][k][e];
      v[1][k][e] = ex2(fmaf(v[1][k][e], L2E, -mc1)); s1 += v[1][k][e];
    }
#pragma unroll
  for (int o = 8; o; o >>= 1) {
    s0 += __shfl_xor_sync(0xffffffffu, s0, o);
    s1 += __shfl_xor_sync(0xffffffffu, s1, o);
  }
  const float inv0 = 1.f / s0, inv1 = 1.f / s1;
  if (!STORE_P && hl == 0) { *st0 = make_float2(mc0, inv0); *st1 = make_float2(mc1, inv1); }
#pragma unroll
  for (int k = 0; k < NCH; ++k) {
    const int ch = hl + 16 * k;
    uint4 o0, o1;
    o0.x = pack2(v[0][k][0] * inv0, v[0][k][1] * inv0); o0.y = pack2(v[0][k][2] * inv0, v[0][k][3] * inv0);
    o0.z = pack2(v[0][k][4] * inv0, v[0][k][5] * inv0); o0.w = pack2(v[0][k][6] * inv0, v[0][k][7] * inv0);
    o1.x = pack2(v[1][k][0] * inv1, v[1][k][1] * inv1); o1.y = pack2(v[1][k][2] * inv1, v[1][k][3] * inv1);
    o1.z = pack2(v[1][k][4] * inv1, v[1][k][5] * inv1); o1.w = pack2(v[1][k][6] * inv1, v[1][k][7] * inv1);
    *reinterpret_cast<uint4*>(row0 + ch * 8) = o0;
    *reinterpret_cast<uint4*>(row1 + ch * 8) = o1;
    if (STORE_P) { *reinterpret_cast<uint4*>(g0 + ch * 8) = o0; *reinterpret_cast<uint4*>(g1 + ch * 8) = o1; }
  }
}


// BM = 32: the 8 warps are 2 row groups x 2 column halves x 2 key halves (partial sums added through shared memory); BM = 64: 4 x 2 x 1.
template <int DH, int BM, bool MASKED, bool STORE_P>
__global__ void __launch_bounds__(256, 2) psb_fwd_k(const bf* __restrict__ BIAS, bf* __restrict__ P, float2* __restrict__ STATS, const bool* __restrict__ MASK_ALL,
                                                 long mask_stride, const bf* __restrict__ VG, bf* __restrict__ OG, int L, int A, int H,
                                                 int nrb_per_cta) {
  constexpr int NT = DH / 16, VP = DH + 8, VCH = DH / 8, KSPLIT = 64 / BM;
  extern __shared__ __align__(16) uint8_t sm_raw[];
  const int pitch = L + 8;
  bf* Ps = reinterpret_cast<bf*>(sm_raw);                 // 2 x [BM][pitch]
  bf* Vs = Ps + 2 * BM * pitch;                           // [L][VP]
  float* Ma = reinterpret_cast<float*>(Vs + (size_t)L * VP);   // [L] additive key mask (MASKED)
  float* Red = Ma + L;                                    // [BM / 16 * 2][NT][4][32] partial sums of the key-half warps (KSPLIT > 1)
  const int a = blockIdx.x, h = blockIdx.y;
  const int nrb = L / BM, rb0 = blockIdx.z * nrb_per_cta, rb1 = min(nrb, rb0 + nrb_per_cta);
  const long ha = (long)h * A + a;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const long DA = (long)H * DH, ldv = 2 * DA;
  const int g = lane >> 2, q = lane & 3, hh = lane >> 4, hl = lane & 15, nch = L / 128;

  auto load_tile = [&](int buf, int rb) {                  // half-warp per row, 16-byte chunks hl + 16 k
    const bf* src = BIAS + (ha * L + (long)rb * BM) * L + hl * 8;
    const uint32_t dst = smem_u32(Ps + buf * BM * pitch) + hl * 16;
#pragma unroll
    for (int j = 0; j < BM / 16; ++j) {
      const int r = 2 * warp + hh + 16 * j;
#pragma unroll
      for (int k = 0; k < 6; ++k)
        if (k < nch) cp_async16(dst + (r * pitch + k * 128) * 2, src + (long)r * L + k * 128);
    }
  };
  load_tile(0, rb0);
  cp_commit();                                             // group 0: tile 0
  const bf* vbase = VG + (long)a * L * ldv + h * DH;
  for (int idx = tid; idx < L * VCH; idx += 256) {
    const int r = idx / VCH, c = idx - r * VCH;
    cp_async16(Vs + r * VP + c * 8, vbase + (long)r * ldv + c * 8);
  }
  cp_commit();                                             // group 1: v
  if constexpr (MASKED) {
    const bool* mk = MASK_ALL + (long)a * mask_stride;
    for (int j = tid; j < L; j += 256) Ma[j] = mk[j] ? 0.f : -3.0e38f;
  }

  const int wm = warp % (BM / 16), wn = (warp / (BM / 16)) % 2, kq = warp / (BM / 8);
  const uint32_t va = smem_u32(Vs);

  for (int rb = rb0; rb < rb1; ++rb) {
    cp_wait<0>();
    __syncthreads();                                       // tile rb (v, mask) landed; the other buffer's last reader (rb - 1) is done
    if (rb + 1 < rb1) { load_tile((rb + 1 - rb0) & 1, rb + 1); cp_commit(); }
    bf* tile = Ps + ((rb - rb0) & 1) * BM * pitch;
    // softmax of the staged rows, in place; P goes to HBM (16-byte stores) and stays in shared memory as the MMA's A operand
    for (int rp = warp; rp < BM / 2; rp += 16) {
      const int r0 = rp * 2 + hh, r1 = (rp + 8) * 2 + hh;
      bf* row0 = tile + r0 * pitch;
      bf* row1 = tile + r1 * pitch;
      bf* g0 = STORE_P ? P + (ha * L + (long)rb * BM + r0) * L : nullptr;
      bf* g1 = STORE_P ? P + (ha * L + (long)rb * BM + r1) * L : nullptr;
      float2* s0 = STORE_P ? nullptr : STATS + ha * L + (long)rb * BM + r0;
      float2* s1 = STORE_P ? nullptr : STATS + ha * L + (long)rb * BM + r1;
      switch (nch) {
        case 1: soft_rows2<1, MASKED, STORE_P>(row0, row1, g0, g1, Ma, hl, s0, s1); break;
        case 2: soft_rows2<2, MASKED, STORE_P>(row0, row1, g0, g1, Ma, hl, s0, s1); break;
        case 3: soft_rows2<3, MASKED, STORE_P>(row0, row1, g0, g1, Ma, hl, s0, s1); break;
        case 4: soft_rows2<4, MASKED, STORE_P>(row0, row1, g0, g1, Ma, hl, s0, s1); break;
        case 5:   // the two-row form would need 80+ registers for the rows alone: one row at a time
          soft_row<5, MASKED, STORE_P>(row0, g0, Ma, hl, s0);
          soft_row<5, MASKED, STORE_P>(row1, g1, Ma, hl, s1);
          break;
        default:
          soft_row<6, MASKED, STORE_P>(row0, g0, Ma, hl, s0);
          soft_row<6, MASKED, STORE_P>(row1, g1, Ma, hl, s1);
          break;
      }
    }
    __syncthreads();

    // og tile [BM, DH] = P [BM, L] v [L, DH]; this warp: rows 16 wm, columns wn DH / 2 .., keys kq L / KSPLIT ..
    float acc[NT][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
    const uint32_t pa = smem_u32(tile);
    const int kbeg = kq * (L / KSPLIT), kend = kbeg + L / KSPLIT;
#pragma unroll 4
    for (int k0 = kbeg; k0 < kend; k0 += 16) {
      uint32_t a0, a1, a2, a3;
      ldsm4(pa + ((wm * 16 + (lane & 15)) * pitch + k0 + (lane >> 4) * 8) * 2, a0, a1, a2, a3);
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        uint32_t b0, b1;
        ldsm2t(va + ((k0 + (lane & 15)) * VP + wn * (DH / 2) + nt * 8) * 2, b0, b1);
        mma16816(acc[nt], a0, a1, a2, a3, b0, b1);
      }
    }
    if constexpr (KSPLIT > 1) {
      float* red = Red + ((wm * 2 + wn) * NT * 4) * 32;
      if (kq == 1) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e) red[(nt * 4 + e) * 32 + lane] = acc[nt][e];
      }
      __syncthreads();
      if (kq == 0) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[nt][e] += red[(nt * 4 + e) * 32 + lane];
      }
    }
    if (kq == 0) {
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const long row = (long)a * L + (long)rb * BM + wm * 16 + g + half * 8;
          const int col = h * DH + wn * (DH / 2) + nt * 8 + 2 * q;
          const float2 gg = unpack2(*reinterpret_cast<const uint32_t*>(VG + row * ldv + DA + col));
          *reinterpret_cast<uint32_t*>(OG + row * DA + col) = pack2(acc[nt][2 * half] * sg(gg.x), acc[nt][2 * half + 1] * sg(gg.y));
        }
    }
  }
}

// ----------------------------------------------------------------------------------------------------------------- backward
constexpr int BBM = 64, BBN = 128;   // query rows per step, keys per CTA

// CTA = (128 keys, sample, head); walks the L query rows in blocks of 64. Warp w: dP rows 16 (w & 3), keys 64 (w >> 2); dV keys 16 w.
// RC: the first operand is the bias, and P is rebuilt in place in the shared tile with the forward's own arithmetic from the stored row
// statistics (mc, inv): ex2(x log2 e - mc) inv -- the same bits as the forward's P, with no P in HBM.
template <int DH, bool RC, bool MASKED>
__global__ void __launch_bounds__(256, DH == 64 ? 2 : 3) psb_bwd_k(const bf* __restrict__ P, const float2* __restrict__ STATS, const bool* __restrict__ MASK_ALL,
                                                 long mask_stride, const bf* __restrict__ DO, const bf* __restrict__ VG,
                                                 const float* __restrict__ DD, bf* __restrict__ DBIAS, bf* __restrict__ DVG, int L, int A, int H) {
  constexpr int VP = DH + 8, PP = BBN + 8, NV = DH / 8, KS = DH / 16, VCH = DH / 8;
  extern __shared__ __align__(16) uint8_t sm_raw[];
  bf* Vs = reinterpret_cast<bf*>(sm_raw);                 // [BBN][VP] keys j0..j0+127
  bf* Ps = Vs + BBN * VP;                                 // 2 x [BBM][PP]
  bf* Os = Ps + 2 * BBM * PP;                             // 2 x [BBM][VP]
  float* Ds = reinterpret_cast<float*>(Os + 2 * BBM * VP); // 2 x [BBM]
  float2* Ss = reinterpret_cast<float2*>(Ds + 2 * BBM);   // 2 x [BBM] (RC)
  float* Mb = reinterpret_cast<float*>(Ss + 2 * BBM);     // [BBN] additive key mask (RC, MASKED)
  const int j0 = blockIdx.x * BBN, a = blockIdx.y, h = blockIdx.z;
  const long ha = (long)h * A + a;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const long DA = (long)H * DH, ldv = 2 * DA;
  const int nib = L / BBM;

  const bf* vbase = VG + ((long)a * L + j0) * ldv + h * DH;
  for (int idx = tid; idx < BBN * VCH; idx += 256) {
    const int r = idx / VCH, c = idx - r * VCH;
    cp_async16(Vs + r * VP + c * 8, vbase + (long)r * ldv + c * 8);
  }
  auto load_stage = [&](int s, int ib) {
    const int i0 = ib * BBM;
    const bf* pg = P + (ha * L + i0) * L + j0;
    for (int idx = tid; idx < BBM * (BBN / 8); idx += 256) {
      const int r = idx / (BBN / 8), c = idx % (BBN / 8);
      cp_async16(Ps + s * BBM * PP + r * PP + c * 8, pg + (long)r * L + c * 8);
    }
    const bf* og = DO + ((long)a * L + i0) * DA + h * DH;
    for (int idx = tid; idx < BBM * VCH; idx += 256) {
      const int r = idx / VCH, c = idx - r * VCH;
      cp_async16(Os + s * BBM * VP + r * VP + c * 8, og + (long)r * DA + c * 8);
    }
    if (tid < BBM / 4) cp_async16(Ds + s * BBM + tid * 4, DD + ((long)a * H + h) * L + i0 + tid * 4);
    if (RC && tid >= 32 && tid < 64) cp_async16(Ss + s * BBM + (tid - 32) * 2, STATS + ha * L + i0 + (tid - 32) * 2);
  };
  load_stage(0, 0);
  cp_commit();
  if constexpr (RC && MASKED) {
    const bool* mk = MASK_ALL + (long)a * mask_stride + j0;
    if (tid < BBN) Mb[tid] = mk[tid] ? 0.f : -3.0e38f;
  }

  const int wm = warp & 3, wn = warp >> 2, g = lane >> 2, q = lane & 3;
  float dv[NV][4];
#pragma unroll
  for (int n = 0; n < NV; ++n) dv[n][0] = dv[n][1] = dv[n][2] = dv[n][3] = 0.f;
  const uint32_t va = smem_u32(Vs);

  for (int ib = 0; ib < nib; ++ib) {
    const int s = ib & 1;
    cp_wait<0>();
    __syncthreads();                                        // stage s landed; the other stage and Xs are free
    if (ib + 1 < nib) { load_stage(s ^ 1, ib + 1); cp_commit(); }
    if constexpr (RC) {                                     // bias tile -> P tile, in place (16 bytes = 8 keys per step)
      bf* tile = Ps + s * BBM * PP;
      const float2* st = Ss + s * BBM;
      for (int idx = tid; idx < BBM * (BBN / 8); idx += 256) {
        const int r = idx / (BBN / 8), c = idx % (BBN / 8);
        const float2 ms = st[r];
        const uint4 u = *reinterpret_cast<const uint4*>(tile + r * PP + c * 8);
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
        float x[8];
#pragma unroll
        for (int e = 0; e < 4; ++e) { x[2 * e] = __uint_as_float(w[e] << 16); x[2 * e + 1] = __uint_as_float(w[e] & 0xffff0000u); }
        if constexpr (MASKED) {
          const float4 m0 = *reinterpret_cast<const float4*>(Mb + c * 8), m1 = *reinterpret_cast<const float4*>(Mb + c * 8 + 4);
          x[0] += m0.x; x[1] += m0.y; x[2] += m0.z; x[3] += m0.w; x[4] += m1.x; x[5] += m1.y; x[6] += m1.z; x[7] += m1.w;
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) x[e] = ex2(fmaf(x[e], L2E, -ms.x));
        uint4 o;
        o.x = pack2(x[0] * ms.y, x[1] * ms.y);
        o.y = pack2(x[2] * ms.y, x[3] * ms.y);
        o.z = pack2(x[4] * ms.y, x[5] * ms.y);
        o.w = pack2(x[6] * ms.y, x[7] * ms.y);
        *reinterpret_cast<uint4*>(tile + r * PP + c * 8) = o;
      }
      __syncthreads();
    }
    const uint32_t oa = smem_u32(Os + s * BBM * VP), pa = smem_u32(Ps + s * BBM * PP);
    const bf* Pt = Ps + s * BBM * PP;
    const float* dd = Ds + s * BBM;

    // dP [16, 64] per warp = dO [16, DH] v^T
    float dp[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) dp[n][0] = dp[n][1] = dp[n][2] = dp[n][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      uint32_t a0, a1, a2, a3;
      ldsm4(oa + ((wm * 16 + (lane & 15)) * VP + ks * 16 + (lane >> 4) * 8) * 2, a0, a1, a2, a3);
#pragma unroll
      for (int np = 0; np < 4; ++np) {                      // two n8 tiles of keys per ldmatrix.x4
        uint32_t b0, b1, b2, b3;
        const int m = lane >> 3;
        ldsm4(va + ((wn * 64 + np * 16 + (lane & 7) + (m >> 1) * 8) * VP + ks * 16 + (m & 1) * 8) * 2, b0, b1, b2, b3);
        mma16816(dp[2 * np], a0, a1, a2, a3, b0, b1);
        mma16816(dp[2 * np + 1], a0, a1, a2, a3, b2, b3);
      }
    }
    // dbias = P (dP - D): P read in the accumulator layout; the packed result waits in registers until dV has read the P tile
    uint32_t xb[8][2];
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int r = wm * 16 + g + half * 8, c = wn * 64 + n * 8 + 2 * q;
        const float2 p = unpack2(*reinterpret_cast<const uint32_t*>(Pt + r * PP + c));
        const float d = dd[r];
        xb[n][half] = pack2(p.x * (dp[n][2 * half] - d), p.y * (dp[n][2 * half + 1] - d));
      }
    // dV [128, DH] += P^T [128, 64] dO [64, DH]; warp w owns keys 16 w .. 16 w + 15
#pragma unroll
    for (int k0 = 0; k0 < BBM; k0 += 16) {
      uint32_t a0, a1, a2, a3;
      {
        const int m = lane >> 3;
        ldsm4t(pa + ((k0 + (m >> 1) * 8 + (lane & 7)) * PP + warp * 16 + (m & 1) * 8) * 2, a0, a1, a2, a3);
      }
#pragma unroll
      for (int np = 0; np < NV / 2; ++np) {
        uint32_t b0, b1, b2, b3;
        const int m = lane >> 3;
        ldsm4t(oa + ((k0 + (m & 1) * 8 + (lane & 7)) * VP + np * 16 + (m >> 1) * 8) * 2, b0, b1, b2, b3);
        mma16816(dv[2 * np], a0, a1, a2, a3, b0, b1);
        mma16816(dv[2 * np + 1], a0, a1, a2, a3, b2, b3);
      }
    }
    __syncthreads();                                        // every warp has read the P tile: the gradient replaces it in place
    bf* Xs = Ps + s * BBM * PP;
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int half = 0; half < 2; ++half)
        *reinterpret_cast<uint32_t*>(Xs + (wm * 16 + g + half * 8) * PP + wn * 64 + n * 8 + 2 * q) = xb[n][half];
    __syncthreads();
    bf* og = DBIAS + (ha * L + ib * BBM) * L + j0;
    for (int idx = tid; idx < BBM * (BBN / 8); idx += 256) {
      const int r = idx / (BBN / 8), c = idx % (BBN / 8);
      *reinterpret_cast<uint4*>(og + (long)r * L + c * 8) = *reinterpret_cast<const uint4*>(Xs + r * PP + c * 8);
    }
  }
  // dV: keys j0 + 16 w + g (+ 8), columns h DH + 8 n + 2 q
#pragma unroll
  for (int n = 0; n < NV; ++n)
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const long row = (long)a * L + j0 + warp * 16 + g + half * 8;
      *reinterpret_cast<uint32_t*>(DVG + row * ldv + h * DH + n * 8 + 2 * q) = pack2(dv[n][2 * half], dv[n][2 * half + 1]);
    }
}

size_t fwd_smem(int L, int DH, int bm) {
  const size_t red = bm == 32 ? (size_t)(bm / 16) * 2 * (DH / 16) * 4 * 32 * 4 : 0;     // the key-half partial sums (BM 32)
  return (size_t)2 * bm * (L + 8) * 2 + (size_t)L * (DH + 8) * 2 + (size_t)L * 4 + red;
}
constexpr size_t SMEM_MAX = 232448;
size_t bwd_smem(int DH) {
  const size_t VP = DH + 8, PP = BBN + 8;
  return (BBN * VP + 2 * BBM * PP + 2 * BBM * VP) * 2 + 2 * BBM * 4 + 2 * BBM * 8 + BBN * 4;
}

template <int DH, int BM, bool MASKED, bool STORE_P> void launch_fwd_bmm(at::Tensor bias, c10::optional<at::Tensor> p, c10::optional<at::Tensor> stats, const bool* mk, long mstride, at::Tensor vg, at::Tensor og, int L, int A, int H, int nsm) {
  auto k = psb_fwd_k<DH, BM, MASKED, STORE_P>;
  const size_t smem = fwd_smem(L, DH, BM);
  static bool attr = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_MAX) == cudaSuccess;
  TORCH_CHECK(attr && smem <= SMEM_MAX, "psb_fwd: shared memory");
  int per = 1;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k, 256, smem);
  const int slots = std::max(per, 1) * nsm, nrb = L / BM, pairs = A * H;
  // row-block slices per (head, sample): the fewest waves x blocks-per-CTA over the resident slots
  int best = 1;
  long best_cost = -1;
  for (int sl = 1; sl <= nrb; ++sl) {
    const int per_cta = (nrb + sl - 1) / sl, ctas = pairs * ((nrb + per_cta - 1) / per_cta);
    const long cost = (long)((ctas + slots - 1) / slots) * (per_cta + 1);      // +1: the v / first-tile prologue
    if (best_cost < 0 || cost < best_cost) { best_cost = cost; best = sl; }
  }
  const int per_cta = (nrb + best - 1) / best;
  k<<<dim3(A, H, (nrb + per_cta - 1) / per_cta), 256, smem, at::cuda::getCurrentCUDAStream()>>>(
      P_<const bf>(bias), STORE_P ? P_<bf>(*p) : nullptr, STORE_P ? nullptr : P_<float2>(*stats), mk, mstride, P_<const bf>(vg), P_<bf>(og), L, A, H, per_cta);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <int DH, int BM> void launch_fwd_bm(at::Tensor bias, c10::optional<at::Tensor> p, c10::optional<at::Tensor> stats, const bool* mk, long mstride, at::Tensor vg, at::Tensor og, int L, int A, int H, int nsm) {
  if (p.has_value()) {
    if (mk) launch_fwd_bmm<DH, BM, true, true>(bias, p, stats, mk, mstride, vg, og, L, A, H, nsm);
    else launch_fwd_bmm<DH, BM, false, true>(bias, p, stats, mk, mstride, vg, og, L, A, H, nsm);
  } else {
    if (mk) launch_fwd_bmm<DH, BM, true, false>(bias, p, stats, mk, mstride, vg, og, L, A, H, nsm);
    else launch_fwd_bmm<DH, BM, false, false>(bias, p, stats, mk, mstride, vg, og, L, A, H, nsm);
  }
}
template <int DH> void launch_fwd(at::Tensor bias, c10::optional<at::Tensor> p, c10::optional<at::Tensor> stats, const bool* mk, long mstride, at::Tensor vg, at::Tensor og, int L, int A, int H) {
  const int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  launch_fwd_bm<DH, 32>(bias, p, stats, mk, mstride, vg, og, L, A, H, nsm);   // 32 rows per tile (BM 64 measured slower: 279 vs 237 us at A 48, L 384)
}
template <int DH, bool RC, bool MASKED> void launch_bwd_t(at::Tensor p, c10::optional<at::Tensor> stats, const bool* mk, long mstride, at::Tensor dout, at::Tensor vg,
                                                           at::Tensor dd, at::Tensor dbias, at::Tensor dvg, int L, int A, int H) {
  auto k = psb_bwd_k<DH, RC, MASKED>;
  const size_t smem = bwd_smem(DH);
  static bool attr = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448) == cudaSuccess;
  TORCH_CHECK(attr, "psb_bwd: shared memory");
  k<<<dim3(L / BBN, A, H), 256, smem, at::cuda::getCurrentCUDAStream()>>>(P_<const bf>(p), RC ? P_<const float2>(*stats) : nullptr, mk, mstride, P_<const bf>(dout),
                                                                         P_<const bf>(vg), P_<const float>(dd), P_<bf>(dbias), P_<bf>(dvg), L, A, H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <int DH> void launch_bwd(at::Tensor p, c10::optional<at::Tensor> stats, const bool* mk, long mstride, at::Tensor dout, at::Tensor vg, at::Tensor dd,
                                  at::Tensor dbias, at::Tensor dvg, int L, int A, int H) {
  if (stats.has_value()) {
    if (mk) launch_bwd_t<DH, true, true>(p, stats, mk, mstride, dout, vg, dd, dbias, dvg, L, A, H);
    else launch_bwd_t<DH, true, false>(p, stats, mk, mstride, dout, vg, dd, dbias, dvg, L, A, H);
  } else launch_bwd_t<DH, false, false>(p, stats, nullptr, 0, dout, vg, dd, dbias, dvg, L, A, H);
}

void check_bf(const at::Tensor& t, const char* n) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.stride(-1) == 1, n, ": bf16 CUDA, unit column stride");
}
}  // namespace

// og = sigmoid(g) (P v) with P = softmax(bias + key mask), P written. bias, p [H A L, L]; vg [A L, 2 H dh] (v | g); og [A L, H dh]
// (row stride H dh); mask bool [L] or [A, L]. L a multiple of 128 up to 768.
void psb_fwd(at::Tensor bias, c10::optional<at::Tensor> p, c10::optional<at::Tensor> stats, c10::optional<at::Tensor> mask, at::Tensor vg,
             at::Tensor og, int64_t samples, int64_t heads, int64_t dh) {
  check_bf(bias, "bias"); check_bf(vg, "vg"); check_bf(og, "og");
  const int64_t L = bias.size(1), A = samples, H = heads;
  TORCH_CHECK(bias.is_contiguous() && bias.size(0) == H * A * L, "psb_fwd: bias [H A L, L]");
  TORCH_CHECK(p.has_value() != stats.has_value(), "psb_fwd: either P is stored (p) or the row statistics are (stats)");
  if (p.has_value()) { check_bf(*p, "p"); TORCH_CHECK(p->is_contiguous() && p->sizes() == bias.sizes(), "psb_fwd: p [H A L, L]"); }
  else TORCH_CHECK(stats->scalar_type() == at::kFloat && stats->is_contiguous() && stats->numel() == 2 * H * A * L, "psb_fwd: stats [H A L, 2] fp32");
  TORCH_CHECK(L % 128 == 0 && L <= 768, "psb_fwd: L a multiple of 128, at most 768");
  TORCH_CHECK(vg.size(0) == A * L && vg.size(1) == 2 * H * dh && vg.stride(0) == 2 * H * dh && og.size(0) == A * L && og.size(1) == H * dh &&
              og.stride(0) == H * dh, "psb_fwd: vg [A L, 2 H dh], og [A L, H dh]");
  const bool* mk = nullptr;
  long mstride = 0;
  if (mask.has_value()) {
    TORCH_CHECK(mask->scalar_type() == at::kBool && mask->is_contiguous() && (mask->numel() == L || mask->numel() == A * L), "psb_fwd: bool mask [L] or [A, L]");
    mk = mask->data_ptr<bool>();
    mstride = mask->numel() == L ? 0 : L;
  }
  const at::cuda::CUDAGuard g(bias.device());
  if (dh == 48) launch_fwd<48>(bias, p, stats, mk, mstride, vg, og, (int)L, (int)A, (int)H);
  else if (dh == 32) launch_fwd<32>(bias, p, stats, mk, mstride, vg, og, (int)L, (int)A, (int)H);
  else if (dh == 64) launch_fwd<64>(bias, p, stats, mk, mstride, vg, og, (int)L, (int)A, (int)H);
  else TORCH_CHECK(false, "psb_fwd: head width 32, 48 or 64");
}

// dV = P^T dO (into dvg's v columns) and dbias = P o (dO v^T - D). Either p is P [H A L, L] (stats absent), or p is the BIAS and stats the
// forward's row statistics [H A L, 2] fp32 (P is rebuilt, with the key mask [L] or [A, L] when given). dout [A L, H dh]; vg [A L, 2 H dh]
// (v read); dd [A, H, L] fp32; dbias [H A L, L]; dvg [A L, 2 H dh] (the v columns written).
void psb_bwd(at::Tensor p, c10::optional<at::Tensor> stats, c10::optional<at::Tensor> mask, at::Tensor dout, at::Tensor vg, at::Tensor dd,
             at::Tensor dbias, at::Tensor dvg, int64_t samples, int64_t heads, int64_t dh) {
  check_bf(p, "p"); check_bf(dout, "do"); check_bf(vg, "vg"); check_bf(dbias, "dbias"); check_bf(dvg, "dvg");
  const int64_t L = p.size(1), A = samples, H = heads;
  TORCH_CHECK(p.is_contiguous() && dbias.is_contiguous() && p.size(0) == H * A * L && dbias.sizes() == p.sizes(), "psb_bwd: p, dbias [H A L, L]");
  TORCH_CHECK(L % 128 == 0 && L <= 768, "psb_bwd: L a multiple of 128, at most 768");
  TORCH_CHECK(dout.is_contiguous() && dout.size(0) == A * L && dout.size(1) == H * dh, "psb_bwd: do [A L, H dh]");
  TORCH_CHECK(vg.size(0) == A * L && vg.stride(0) == 2 * H * dh && dvg.size(0) == A * L && dvg.stride(0) == 2 * H * dh && dvg.size(1) == 2 * H * dh,
              "psb_bwd: vg, dvg [A L, 2 H dh]");
  TORCH_CHECK(dd.scalar_type() == at::kFloat && dd.is_contiguous() && dd.numel() == A * H * L, "psb_bwd: dd [A, H, L] fp32");
  if (stats.has_value()) TORCH_CHECK(stats->scalar_type() == at::kFloat && stats->is_contiguous() && stats->numel() == 2 * H * A * L, "psb_bwd: stats [H A L, 2] fp32");
  const bool* mk = nullptr;
  long mstride = 0;
  if (mask.has_value()) {
    TORCH_CHECK(stats.has_value(), "psb_bwd: a key mask goes with the statistics (P stored already has it)");
    TORCH_CHECK(mask->scalar_type() == at::kBool && mask->is_contiguous() && (mask->numel() == L || mask->numel() == A * L), "psb_bwd: bool mask [L] or [A, L]");
    mk = mask->data_ptr<bool>();
    mstride = mask->numel() == L ? 0 : L;
  }
  const at::cuda::CUDAGuard g(p.device());
  if (dh == 48) launch_bwd<48>(p, stats, mk, mstride, dout, vg, dd, dbias, dvg, (int)L, (int)A, (int)H);
  else if (dh == 32) launch_bwd<32>(p, stats, mk, mstride, dout, vg, dd, dbias, dvg, (int)L, (int)A, (int)H);
  else if (dh == 64) launch_bwd<64>(p, stats, mk, mstride, dout, vg, dd, dbias, dvg, (int)L, (int)A, (int)H);
  else TORCH_CHECK(false, "psb_bwd: head width 32, 48 or 64");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("psb_fwd", &psb_fwd);
  m.def("psb_bwd", &psb_bwd);
}
