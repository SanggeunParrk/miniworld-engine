// bo_pack_sm100.cu -- the bias-only token DiT inference runner's weight pack on sm_100a in ONE launch per block (B200 only;
// runner.FusedBiasOnlyDiT._pack_b200). The torch pack it replaces is ~40 small kernels (casts, products, cats, the pair packs'
// slices), which a CUDA-graph replay re-runs every call (the pack is scoped to the capture): ~100 us of a ~200 us step at L384.
//
// A launch walks up to MAXSEG segments, each a contiguous destination of n elements (a multiple of 4), the blocks of a segment
// consecutive (b0 = its first block). Per destination element:
//   mode 0  dst[e] = src[e] (x scale[e % period] when period > 0)      the casts, cats (one segment per part) and folded LN weights
//   mode 1  dst[e] = cs (0: the AdaLN shift biases; AttentionPairBias: the gate bias)
//   mode 3  head-padded rows: dst row h dhp + j = src row h dh + j for j < dh, else 0 (rows of ``cols``; AttentionPairBias fp32, 16 x 24 as 16 x 32)
//   mode 4  head-padded columns: dst[r, h dhp + j] = src[r, h dh + j] for j < dh, else 0 (dst rows of ``cols`` = H dhp)
//   (every mode but 1 then multiplies by cs)
//   mode 2  the pair pack of a [768, K] weight for the tail kernels (bo_tail_bf16.cu / bo_tail_tf32.cu): cluster cl, NC = 768 / cl
//           output rows per CTA, k-blocks of bw columns (64 bf16, 32 fp32), two per box:
//           dst[((c (K / 2bw) + p) 2 + h) NC + n, k] = W[NC c + n, bw o_c(2 p + h) + k], o_c the natural order or, with zo > 0, CTA c's
//           z order (its own zo k-blocks first: j < zo -> zo c + j; j < zo + zo c -> j - zo; else j)
// Arithmetic: the value in fp32 (src and scale read as stored), one product, then one rounding to the destination: bf16 (rn), fp32,
// or fp32 holding the nearest TF32 (``_tf32.tf32_grid``, which ``tf32.round_tf32`` applies). The same values as the torch pack, bit for bit.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <cuda_bf16.h>

// UNR vector groups per thread, every load before any store (2026-11-17: one load in flight per thread left a 7 M-element pack at
// ~17 us, latency-bound): a block covers PER_BLK consecutive elements of one segment, group u of thread t at (u TPB + t) VEC
constexpr int MAXSEG = 48, VEC = 4, TPB = 256, UNR = 4, PER_BLK = TPB * VEC * UNR;

struct Seg {
  const void* src;
  void* dst;
  const void* scale;
  long long n;
  int b0, sdt, ddt, scdt;   // sdt / scdt: 0 bf16, 1 fp32; ddt: 0 bf16, 1 fp32, 2 fp32 rounded to the nearest tf32
  int mode, period, K, cl;  // mode 0 copy, 1 zeros, 2 pair pack; period: the scale's (0: none); K: the pair pack's source columns
  int bw, zo;               // pair pack: k-block width, z-order k-blocks per CTA (0: natural)
  float cs;                 // a constant factor (modes 0, 2, 3, 4; 1: none) or the fill value (mode 1)
  int dh, dhp, cols;        // modes 3 / 4: real and padded head width, the row length (mode 3: of both; mode 4: of dst)
  int pad_;
};
struct Segs {
  Seg s[MAXSEG];
  int n;
};

// four consecutive elements (16-byte fp32 / 8-byte bf16 accesses: every segment, slice and pair-pack run is a multiple of 4 elements
// and 4-element aligned)
__device__ __forceinline__ float4 ld4(const void* p, int dt, int i) {
  if (dt) return *reinterpret_cast<const float4*>(static_cast<const float*>(p) + i);
  const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(p) + i);
  const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
  const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
  return make_float4(a.x, a.y, b.x, b.y);
}

__device__ __forceinline__ float tf32r(float v) { return __uint_as_float((__float_as_uint(v) + 0x1000u) & ~0x1fffu); }   // _tf32.tf32_grid

__device__ __forceinline__ void st4(void* p, int dt, int i, float4 v) {
  if (dt == 0) {
    const __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
    uint2 u;
    u.x = *reinterpret_cast<const uint32_t*>(&a);
    u.y = *reinterpret_cast<const uint32_t*>(&b);
    *reinterpret_cast<uint2*>(static_cast<__nv_bfloat16*>(p) + i) = u;
    return;
  }
  if (dt == 2) v = make_float4(tf32r(v.x), tf32r(v.y), tf32r(v.z), tf32r(v.w));
  *reinterpret_cast<float4*>(static_cast<float*>(p) + i) = v;
}

__device__ __forceinline__ float4 pack_load(const Seg& g, int e) {
  float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
  if (g.mode == 0) {
    v = ld4(g.src, g.sdt, e);
    if (g.period) {
      const float4 s = ld4(g.scale, g.scdt, e % g.period);
      v = make_float4(v.x * s.x, v.y * s.y, v.z * s.z, v.w * s.w);
    }
  } else if (g.mode == 1) {
    v = make_float4(g.cs, g.cs, g.cs, g.cs);
  } else if (g.mode == 3 || g.mode == 4) {
    const int r = e / g.cols, c = e % g.cols;
    const int q = g.mode == 3 ? r : c, j = q % g.dhp, h = q / g.dhp;
    if (j < g.dh) {
      const int H = g.cols / g.dhp;
      v = ld4(g.src, g.sdt, g.mode == 3 ? (h * g.dh + j) * g.cols + c : r * (H * g.dh) + h * g.dh + j);
    }
  } else if (g.mode == 2) {
    const int nc = 768 / g.cl, nkp = g.K / (2 * g.bw);
    const int kk = e % g.bw, row = e / g.bw;
    const int n = row % nc, t = row / nc;
    const int h = t & 1, p = (t >> 1) % nkp, c = (t >> 1) / nkp;
    int q = 2 * p + h;
    if (g.zo) q = q < g.zo ? g.zo * c + q : (q < g.zo + g.zo * c ? q - g.zo : q);
    v = ld4(g.src, g.sdt, (nc * c + n) * g.K + g.bw * q + kk);
  }
  if (g.mode != 1 && g.cs != 1.f) v = make_float4(v.x * g.cs, v.y * g.cs, v.z * g.cs, v.w * g.cs);
  return v;
}

extern "C" __global__ void __launch_bounds__(TPB) bo_pack_sm100(const __grid_constant__ Segs S) {
  const int b = blockIdx.x;
  int k = 0;
  while (k + 1 < S.n && S.s[k + 1].b0 <= b) ++k;
  const Seg& g = S.s[k];
  const int e0 = (b - g.b0) * PER_BLK + (int)threadIdx.x * VEC;   // segments < 2^31 elements (pack_sm100.py)
  float4 v[UNR];
#pragma unroll
  for (int u = 0; u < UNR; ++u) {
    const int e = e0 + u * TPB * VEC;
    if (e < (int)g.n) v[u] = pack_load(g, e);
  }
#pragma unroll
  for (int u = 0; u < UNR; ++u) {
    const int e = e0 + u * TPB * VEC;
    if (e < (int)g.n) st4(g.dst, g.ddt, e, v[u]);
  }
}
