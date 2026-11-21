// gattn_tf32.cu — the glue of the fp32 SWA atom block's GLOBAL attention on sm_100a (the TF32 row stages around the bf16 global attention,
// FlashAttention-4): what were ~10 (forward) / ~50 (backward) small torch launches per block, each in ONE launch.
//   swa_gcast_tf32_sm100   NC fp32 arrays -> bf16 (round to nearest even, torch's cast) into one packed buffer [NC + NZ, n], the NZ arrays
//                          after them zero-filled (FA4 leaves the rows past seqused unwritten):
//                            forward   Qh | Kh | Vh -> bf16, O16 = 0                       (NC 3, NZ 1)
//                            backward  Qh | Kh | Vh | O | dO -> bf16, dQh = dKh = dVh = 0  (NC 5, NZ 3)
//   swa_gtail_tf32_sm100   the head-wise tail of the qkvg backward for q and k (head = rope(RMSNorm_head(p)), affine-free, fp32 eps; the rotary
//                          on the channel pairs (c, c + 16)) and the v columns: dQh, dKh, dVh [N, 4, S, 32] bf16, pq, pk [N S, 128] fp32,
//                          cos, sin [N S, 16] fp32 -> dP [N S, 4 C] columns 0 .. 3C = dpq | dpk | dpv, dpq / dpk rounded to TF32 (cvt.rna)
//                          as the window kernels' epilogues write them (dpv: bf16 values, TF32-exact). One thread per (row, head).
// No programmatic dependent launch (loaded without it): each starts after the kernel before it has finished.
// SPDX-License-Identifier: Apache-2.0
#include <stdint.h>

constexpr int C = 128, H = 4, D = 32, R = 16;

__device__ __forceinline__ float rna_tf32(float x) {
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return __uint_as_float(r & 0xffffe000u);
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ float bf16lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

// n8 = elements per array / 8; thread i of an array handles its elements 8 i .. 8 i + 7
template <bool EARLY>
__device__ __forceinline__ void gcast_body(const float* __restrict__ s0, const float* __restrict__ s1, const float* __restrict__ s2, const float* __restrict__ s3,
                     const float* __restrict__ s4, uint16_t* __restrict__ dst, long long n8, int nc, int nz) {
  asm volatile("griddepcontrol.wait;" ::: "memory");  // (a no-op unless launched with the PDL attribute)
  if (EARLY) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const float* const src[5] = {s0, s1, s2, s3, s4};
  const long long total = n8 * (nc + nz);
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += (long long)gridDim.x * blockDim.x) {
    const int a = (int)(t / n8);
    const long long i = t - (long long)a * n8;
    uint4* o = reinterpret_cast<uint4*>(dst) + t;
    if (a >= nc) {
      *o = make_uint4(0u, 0u, 0u, 0u);
      continue;
    }
    const float4* p = reinterpret_cast<const float4*>(src[a]) + 2 * i;
    const float4 x = p[0], y = p[1];
    *o = make_uint4(pack_bf16(x.x, x.y), pack_bf16(x.z, x.w), pack_bf16(y.x, y.y), pack_bf16(y.z, y.w));
  }
}

__device__ __forceinline__ void load_head(const uint16_t* p, float* g) {
  const uint4* q = reinterpret_cast<const uint4*>(p);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint4 v = q[i];
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      g[8 * i + 2 * e] = bf16lo(w[e]);
      g[8 * i + 2 * e + 1] = bf16hi(w[e]);
    }
  }
}

// d p (one head of one row) from d head: the rotary's transpose, then the affine-free RMSNorm's backward
__device__ __forceinline__ void tail(const uint16_t* dh, const float* p, const float* cs, const float* sn, float eps, float* out) {
  float g[D], x[D], dy[D];
  load_head(dh, g);
  float ss = 0.f, dot = 0.f;
#pragma unroll
  for (int j = 0; j < D; j += 4) {
    const float4 v = *reinterpret_cast<const float4*>(p + j);
    x[j] = v.x, x[j + 1] = v.y, x[j + 2] = v.z, x[j + 3] = v.w;
  }
#pragma unroll
  for (int j = 0; j < R; ++j) {
    dy[j] = g[j] * cs[j] + g[j + R] * sn[j];
    dy[j + R] = g[j + R] * cs[j] - g[j] * sn[j];
  }
#pragma unroll
  for (int j = 0; j < D; ++j) {
    ss += x[j] * x[j];
    dot += dy[j] * x[j];
  }
  const float rs = rsqrtf(ss * (1.f / D) + eps), k = rs * rs * (dot * (1.f / D));
#pragma unroll
  for (int j = 0; j < D; j += 4)
    *reinterpret_cast<float4*>(out + j) = make_float4(rna_tf32(rs * (dy[j] - x[j] * k)), rna_tf32(rs * (dy[j + 1] - x[j + 1] * k)),
                                                      rna_tf32(rs * (dy[j + 2] - x[j + 2] * k)), rna_tf32(rs * (dy[j + 3] - x[j + 3] * k)));
}

extern "C" __global__ void __launch_bounds__(256)
swa_gtail_tf32_sm100(const uint16_t* __restrict__ DQH, const uint16_t* __restrict__ DKH, const uint16_t* __restrict__ DVH,
                     const float* __restrict__ PQ, const float* __restrict__ PK, const float* __restrict__ COS, const float* __restrict__ SIN,
                     float* __restrict__ DP, int N, int S, float eps) {
  const long long total = (long long)N * S * H;
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += (long long)gridDim.x * blockDim.x) {
    const long long m = t >> 2;                                           // row n S + s
    const int h = (int)(t & 3), n = (int)(m / S), s = (int)(m - (long long)n * S);
    const size_t ho = (((size_t)n * H + h) * S + s) * D;                  // head-major [N, H, S, D]
    float cs[R], sn[R];
#pragma unroll
    for (int j = 0; j < R; j += 4) {
      const float4 a = *reinterpret_cast<const float4*>(COS + m * R + j), b = *reinterpret_cast<const float4*>(SIN + m * R + j);
      cs[j] = a.x, cs[j + 1] = a.y, cs[j + 2] = a.z, cs[j + 3] = a.w;
      sn[j] = b.x, sn[j + 1] = b.y, sn[j + 2] = b.z, sn[j + 3] = b.w;
    }
    float* dp = DP + (size_t)m * (4 * C) + h * D;
    tail(DQH + ho, PQ + (size_t)m * C + h * D, cs, sn, eps, dp);
    tail(DKH + ho, PK + (size_t)m * C + h * D, cs, sn, eps, dp + C);
    float v[D];
    load_head(DVH + ho, v);
#pragma unroll
    for (int j = 0; j < D; j += 4) *reinterpret_cast<float4*>(dp + 2 * C + j) = make_float4(v[j], v[j + 1], v[j + 2], v[j + 3]);
  }
}

// swa_gdelta_tf32_sm100 (round 3, part 2: the native global backward, gattn_bwd.cu, on the fp32 block): dO [N S, C] fp32 -> dO16 bf16 (the
// backward's operand, rounded as the cast) and D [N, H, S] = sum_{d in head h} bf16(dO) o with o the saved fp32 O (bf16 values) -- the D of
// the same rounded dO the backward's dP = dO16 V^T uses (the TF32 out-projection's Dv is of the unrounded dO). One thread per (row, head).
extern "C" __global__ void __launch_bounds__(256)
swa_gdelta_tf32_sm100(const float* __restrict__ O, const float* __restrict__ DO, uint16_t* __restrict__ DO16, float* __restrict__ DD, int N, int S) {
  const long long total = (long long)N * S * H;
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += (long long)gridDim.x * blockDim.x) {
    const long long m = t >> 2;
    const int h = (int)(t & 3), n = (int)(m / S), s = (int)(m - (long long)n * S);
    const float4* po = reinterpret_cast<const float4*>(O + (size_t)m * C + h * D);
    const float4* pd = reinterpret_cast<const float4*>(DO + (size_t)m * C + h * D);
    uint4* out = reinterpret_cast<uint4*>(DO16 + (size_t)m * C + h * D);
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float4 a = pd[2 * i], b = pd[2 * i + 1], oa = po[2 * i], ob = po[2 * i + 1];
      const uint4 u = make_uint4(pack_bf16(a.x, a.y), pack_bf16(a.z, a.w), pack_bf16(b.x, b.y), pack_bf16(b.z, b.w));
      out[i] = u;
      acc = fmaf(bf16lo(u.x), oa.x, acc); acc = fmaf(bf16hi(u.x), oa.y, acc); acc = fmaf(bf16lo(u.y), oa.z, acc); acc = fmaf(bf16hi(u.y), oa.w, acc);
      acc = fmaf(bf16lo(u.z), ob.x, acc); acc = fmaf(bf16hi(u.z), ob.y, acc); acc = fmaf(bf16lo(u.w), ob.z, acc); acc = fmaf(bf16hi(u.w), ob.w, acc);
    }
    DD[((size_t)n * H + h) * S + s] = acc;
  }
}

// B200 round 3 part 16: the tail and the delta with EIGHT threads per (row, head) (lane k of the group: channels 4 k .. 4 k + 3), the
// head sums by shuffles in the group. One thread per (row, head) launched N S H threads -- at one sample of 2048 atoms 8192 threads, 32
// blocks on 32 of 148 SMs (~8.7 / 4.2 us a block backward, ~0.7 TB/s). The same equations; the head sums' fp32 order differs (fixed:
// deterministic).
__device__ __forceinline__ float4 bf4(const uint16_t* p) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  return make_float4(bf16lo(u.x), bf16hi(u.x), bf16lo(u.y), bf16hi(u.y));
}
__device__ __forceinline__ float gsum8(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  return v;
}
// lane k of the group: d p channels 4 k .. 4 k + 3 of one head (rotary pairs (j, j + 16) = lanes (k, k ^ 4))
__device__ __forceinline__ float4 tail8(float4 g, float4 x, float4 cs, float4 sn, bool lo, float eps) {
  const float4 gp = make_float4(__shfl_xor_sync(0xffffffffu, g.x, 4), __shfl_xor_sync(0xffffffffu, g.y, 4),
                                __shfl_xor_sync(0xffffffffu, g.z, 4), __shfl_xor_sync(0xffffffffu, g.w, 4));
  float4 dy;
  if (lo) dy = make_float4(g.x * cs.x + gp.x * sn.x, g.y * cs.y + gp.y * sn.y, g.z * cs.z + gp.z * sn.z, g.w * cs.w + gp.w * sn.w);
  else dy = make_float4(g.x * cs.x - gp.x * sn.x, g.y * cs.y - gp.y * sn.y, g.z * cs.z - gp.z * sn.z, g.w * cs.w - gp.w * sn.w);
  const float ss = gsum8(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w);
  const float dot = gsum8(dy.x * x.x + dy.y * x.y + dy.z * x.z + dy.w * x.w);
  const float rs = rsqrtf(ss * (1.f / D) + eps), k = rs * rs * (dot * (1.f / D));
  return make_float4(rna_tf32(rs * (dy.x - x.x * k)), rna_tf32(rs * (dy.y - x.y * k)), rna_tf32(rs * (dy.z - x.z * k)),
                     rna_tf32(rs * (dy.w - x.w * k)));
}

extern "C" __global__ void __launch_bounds__(256)
swa_gtail8_tf32_sm100(const uint16_t* __restrict__ DQH, const uint16_t* __restrict__ DKH, const uint16_t* __restrict__ DVH,
                      const float* __restrict__ PQ, const float* __restrict__ PK, const float* __restrict__ COS, const float* __restrict__ SIN,
                      float* __restrict__ DP, int N, int S, float eps) {
  asm volatile("griddepcontrol.wait;" ::: "memory");  // (N1 round 4: a no-op unless launched with the PDL attribute)
  const long long total = (long long)N * S * H * 8;                       // a multiple of 32: whole warps take the loop together
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += (long long)gridDim.x * blockDim.x) {
    const long long grp = t >> 3, m = grp >> 2;
    const int k = (int)(t & 7), h = (int)(grp & 3), n = (int)(m / S), s = (int)(m - (long long)n * S), c0 = 4 * k;
    const size_t ho = (((size_t)n * H + h) * S + s) * D + c0;
    const float4 cs = *reinterpret_cast<const float4*>(COS + m * R + (c0 & 15)), sn = *reinterpret_cast<const float4*>(SIN + m * R + (c0 & 15));
    const float4 gq = bf4(DQH + ho), gk = bf4(DKH + ho), gv = bf4(DVH + ho);
    const float4 xq = *reinterpret_cast<const float4*>(PQ + (size_t)m * C + h * D + c0);
    const float4 xk = *reinterpret_cast<const float4*>(PK + (size_t)m * C + h * D + c0);
    float* dp = DP + (size_t)m * (4 * C) + h * D + c0;
    *reinterpret_cast<float4*>(dp) = tail8(gq, xq, cs, sn, k < 4, eps);
    *reinterpret_cast<float4*>(dp + C) = tail8(gk, xk, cs, sn, k < 4, eps);
    *reinterpret_cast<float4*>(dp + 2 * C) = gv;
  }
}

extern "C" __global__ void __launch_bounds__(256)
swa_gdelta8_tf32_sm100(const float* __restrict__ O, const float* __restrict__ DO, uint16_t* __restrict__ DO16, float* __restrict__ DD, int N, int S) {
  const long long total = (long long)N * S * H * 8;
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += (long long)gridDim.x * blockDim.x) {
    const long long grp = t >> 3, m = grp >> 2;
    const int k = (int)(t & 7), h = (int)(grp & 3), n = (int)(m / S), s = (int)(m - (long long)n * S);
    const size_t off = (size_t)m * C + h * D + 4 * k;
    const float4 a = *reinterpret_cast<const float4*>(DO + off), o = *reinterpret_cast<const float4*>(O + off);
    const uint2 u = make_uint2(pack_bf16(a.x, a.y), pack_bf16(a.z, a.w));
    *reinterpret_cast<uint2*>(DO16 + off) = u;
    float acc = bf16lo(u.x) * o.x;
    acc = fmaf(bf16hi(u.x), o.y, acc); acc = fmaf(bf16lo(u.y), o.z, acc); acc = fmaf(bf16hi(u.y), o.w, acc);
    acc = gsum8(acc);
    if (k == 0) DD[((size_t)n * H + h) * S + s] = acc;
  }
}

// B200 N1 round 4: the native global backward's three head casts (swa_gcast, nc 3 / nz 0) and the rounded dO with its D (swa_gdelta8) in ONE
// launch -- the two are independent (disjoint inputs and outputs), the same per-element arithmetic, so bitwise the same results. The first
// 3 n8 threads cast, the next 8 N S H threads take the delta (warp-uniform: 3 n8 is a multiple of 32, the delta's shuffles need whole warps).
template <bool EARLY>
__device__ __forceinline__ void gprep_body(const float* __restrict__ s0, const float* __restrict__ s1, const float* __restrict__ s2, uint16_t* __restrict__ dst,
                     const float* __restrict__ O, const float* __restrict__ DO, uint16_t* __restrict__ DO16, float* __restrict__ DD, int n8, int N, int S) {
  asm volatile("griddepcontrol.wait;" ::: "memory");  // (a no-op unless launched with the PDL attribute)
  if (EARLY) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // the dependents (gattn_bwd: waits on griddepcontrol.wait) start their prologue
  const float* const src[3] = {s0, s1, s2};
  const long long total1 = 3ll * n8, total = total1 + (long long)N * S * H * 8;
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += (long long)gridDim.x * blockDim.x) {
    if (t < total1) {
      const int a = (int)(t / n8);
      const long long i = t - (long long)a * n8;
      const float4* p = reinterpret_cast<const float4*>(src[a]) + 2 * i;
      const float4 x = p[0], y = p[1];
      reinterpret_cast<uint4*>(dst)[t] = make_uint4(pack_bf16(x.x, x.y), pack_bf16(x.z, x.w), pack_bf16(y.x, y.y), pack_bf16(y.z, y.w));
    } else {
      const long long u = t - total1;
      const long long grp = u >> 3, m = grp >> 2;
      const int k = (int)(u & 7), h = (int)(grp & 3), n = (int)(m / S), s = (int)(m - (long long)n * S);
      const size_t off = (size_t)m * C + h * D + 4 * k;
      const float4 a = *reinterpret_cast<const float4*>(DO + off), o = *reinterpret_cast<const float4*>(O + off);
      const uint2 w = make_uint2(pack_bf16(a.x, a.y), pack_bf16(a.z, a.w));
      *reinterpret_cast<uint2*>(DO16 + off) = w;
      float acc = bf16lo(w.x) * o.x;
      acc = fmaf(bf16hi(w.x), o.y, acc); acc = fmaf(bf16lo(w.y), o.z, acc); acc = fmaf(bf16hi(w.y), o.w, acc);
      acc = gsum8(acc);
      if (k == 0) DD[((size_t)n * H + h) * S + s] = acc;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256)
swa_gprep_tf32_sm100(const float* __restrict__ s0, const float* __restrict__ s1, const float* __restrict__ s2, uint16_t* __restrict__ dst,
                     const float* __restrict__ O, const float* __restrict__ DO, uint16_t* __restrict__ DO16, float* __restrict__ DD, int n8, int N, int S) {
  gprep_body<false>(s0, s1, s2, dst, O, DO, DO16, DD, n8, N, S);
}
// the same, triggering the dependent launch first (programmatic dependent launch; launched with the PDL attribute)
extern "C" __global__ void __launch_bounds__(256)
swa_gprep_pdl_tf32_sm100(const float* __restrict__ s0, const float* __restrict__ s1, const float* __restrict__ s2, uint16_t* __restrict__ dst,
                         const float* __restrict__ O, const float* __restrict__ DO, uint16_t* __restrict__ DO16, float* __restrict__ DD, int n8, int N, int S) {
  gprep_body<true>(s0, s1, s2, dst, O, DO, DO16, DD, n8, N, S);
}

extern "C" __global__ void __launch_bounds__(256)
swa_gcast_tf32_sm100(const float* __restrict__ s0, const float* __restrict__ s1, const float* __restrict__ s2, const float* __restrict__ s3,
                     const float* __restrict__ s4, uint16_t* __restrict__ dst, long long n8, int nc, int nz) {
  gcast_body<false>(s0, s1, s2, s3, s4, dst, n8, nc, nz);
}
// N1 round 4: the same, triggering the dependent launch (the global forward attention) first
extern "C" __global__ void __launch_bounds__(256)
swa_gcast_pdl_tf32_sm100(const float* __restrict__ s0, const float* __restrict__ s1, const float* __restrict__ s2, const float* __restrict__ s3,
                         const float* __restrict__ s4, uint16_t* __restrict__ dst, long long n8, int nc, int nz) {
  gcast_body<true>(s0, s1, s2, s3, s4, dst, n8, nc, nz);
}
