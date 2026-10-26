// Pair -> bias projection with its LayerNorm folded in, for any d_pair in {64, 128, 256} and NH heads, forward and backward.
//
//   pair_rows_fwd   bias[h, r] = scale * sum_c Wf[h, c] * xhat[r, c]       r = i * L + j, xhat = LayerNorm(pair[r]) without affine
//                   (its weight is folded into Wf by the caller); a masked key j gets ``neg``. bf16 head-major.
//   pair_rows_bwd   dpair[r, c] = rstd (g - mean g - xhat mean(g xhat)),  g[c] = sum_h Wf[h, c] dbias[h, r]
//                   dWf[h, c]   = sum_r dbias[h, r] xhat[r, c]            (block partial rows; the caller adds them in order)
//
// A warp owns a row at a time (lane l: channels DP / 32 l ..), the pair tensor is read once with 8 / 16-byte loads, the heads' dot
// products are warp butterfly sums; 32 rows' results are collected one per lane so the bias is written coalesced along j (64 B per
// head and 32 rows). CUDA cores only: the step is bound by the pair read (one pass of L * L * DP * 2 bytes), the arithmetic
// (NH * DP FMA a row) is a fifth of the memory time. The unfused route reads / writes the pair tensor four times (LayerNorm, GEMM,
// permute, scale).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAGuard.h>

#include "../../conditioned_transition/cuda/token_dit_common.cuh"  // tdr::warp_sum, tdr::transpose_k: shared with the token DiT row kernels

namespace {
using bf = __nv_bfloat16;
constexpr int BT = 256;
constexpr int WPB = BT / 32;
constexpr int PF = 4;  // rows whose loads are in flight together


template <int DPL>
__device__ __forceinline__ void load_row(const bf* __restrict__ z, float (&x)[DPL]) {
  static_assert(DPL == 2 || DPL == 4 || DPL == 8, "channels per lane");
  if constexpr (DPL == 8) {
    const uint4 a = __ldg(reinterpret_cast<const uint4*>(z));
    const uint32_t w[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      x[2 * i] = __uint_as_float(w[i] << 16);
      x[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
  } else if constexpr (DPL == 4) {
    const uint2 a = __ldg(reinterpret_cast<const uint2*>(z));
    const uint32_t w[2] = {a.x, a.y};
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      x[2 * i] = __uint_as_float(w[i] << 16);
      x[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
  } else {
    const uint32_t w = __ldg(reinterpret_cast<const unsigned*>(z));
    x[0] = __uint_as_float(w << 16);
    x[1] = __uint_as_float(w & 0xffff0000u);
  }
}
template <int DPL>
__device__ __forceinline__ void store_row(bf* __restrict__ z, const float (&x)[DPL]) {
  uint32_t o[DPL / 2];
#pragma unroll
  for (int i = 0; i < DPL / 2; ++i) {
    const __nv_bfloat162 p = __floats2bfloat162_rn(x[2 * i], x[2 * i + 1]);
    o[i] = *reinterpret_cast<const uint32_t*>(&p);
  }
  if constexpr (DPL == 8) *reinterpret_cast<uint4*>(z) = make_uint4(o[0], o[1], o[2], o[3]);
  else if constexpr (DPL == 4) *reinterpret_cast<uint2*>(z) = make_uint2(o[0], o[1]);
  else *reinterpret_cast<unsigned*>(z) = o[0];
}
template <int DPL>
__device__ __forceinline__ void load_row(const float* __restrict__ z, float (&x)[DPL]) {
  static_assert(DPL % 4 == 0, "fp32 rows: 4 channels per load");
#pragma unroll
  for (int i = 0; i < DPL / 4; ++i) {
    const float4 a = __ldg(reinterpret_cast<const float4*>(z) + i);
    x[4 * i] = a.x; x[4 * i + 1] = a.y; x[4 * i + 2] = a.z; x[4 * i + 3] = a.w;
  }
}
template <int DPL>
__device__ __forceinline__ void store_row(float* __restrict__ z, const float (&x)[DPL]) {
#pragma unroll
  for (int i = 0; i < DPL / 4; ++i) reinterpret_cast<float4*>(z)[i] = make_float4(x[4 * i], x[4 * i + 1], x[4 * i + 2], x[4 * i + 3]);
}
template <int DP, int DPL>
__device__ __forceinline__ void normalize(float (&x)[DPL], float eps, float& rstd) {
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < DPL; ++i) s += x[i];
  const float mean = tdr::warp_sum(s) * (1.f / DP);
  float v = 0.f;
#pragma unroll
  for (int i = 0; i < DPL; ++i) {
    x[i] -= mean;
    v += x[i] * x[i];
  }
  rstd = rsqrtf(tdr::warp_sum(v) * (1.f / DP) + eps);
#pragma unroll
  for (int i = 0; i < DPL; ++i) x[i] *= rstd;
}

// 16 values per lane summed over the warp and scattered: value i ends (summed) in lanes 2i, 2i + 1 -- 16 shuffles for what 16
// butterfly sums do in 80
__device__ __forceinline__ float reduce_scatter16(const float (&v)[16], int lane) {
  float a[8], b[4], c[2];
  {
    const bool hi = lane & 16;
#pragma unroll
    for (int i = 0; i < 8; ++i) a[i] = (hi ? v[i + 8] : v[i]) + __shfl_xor_sync(0xffffffffu, hi ? v[i] : v[i + 8], 16);
  }
  {
    const bool hi = lane & 8;
#pragma unroll
    for (int i = 0; i < 4; ++i) b[i] = (hi ? a[i + 4] : a[i]) + __shfl_xor_sync(0xffffffffu, hi ? a[i] : a[i + 4], 8);
  }
  {
    const bool hi = lane & 4;
#pragma unroll
    for (int i = 0; i < 2; ++i) c[i] = (hi ? b[i + 2] : b[i]) + __shfl_xor_sync(0xffffffffu, hi ? b[i] : b[i + 2], 4);
  }
  const bool hi = lane & 2;
  float d = (hi ? c[1] : c[0]) + __shfl_xor_sync(0xffffffffu, hi ? c[0] : c[1], 2);
  return d + __shfl_xor_sync(0xffffffffu, d, 1);
}

// NH = 8 heads: slots 0, 1 = sum x, sum x^2; 2 + h = sum_c Wf[h, c] x[c] (raw x: LN folded as rstd (dot - mean sum_c Wf[h, c]))
template <int DP>
__global__ void __launch_bounds__(BT) fwd_k(const bf* __restrict__ Z, const float* __restrict__ W, const bool* __restrict__ MASK,
                                            bf* __restrict__ OUT, float2* __restrict__ ST, long R, int L, float eps, float scale, float neg) {
  constexpr int DPL = DP / 32, NH = 8;
  __shared__ float stage[WPB][32][NH + 1];
  __shared__ float sumw[NH];
  const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  float w[NH][DPL];
#pragma unroll
  for (int h = 0; h < NH; ++h) {
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < DPL; ++i) {
      w[h][i] = W[h * DP + lane * DPL + i] * scale;
      s += w[h][i];
    }
    s = tdr::warp_sum(s);
    if (threadIdx.x == 0) sumw[h] = s;
  }
  __syncthreads();
  const long warp = ((long)blockIdx.x * BT + threadIdx.x) >> 5, nwarps = (long)gridDim.x * WPB;
  for (long base = warp * 32; base < R; base += nwarps * 32) {
    const int n = (int)min(32L, R - base);
    for (int rb = 0; rb < n; rb += PF) {
      float xb[PF][DPL];
#pragma unroll
      for (int q = 0; q < PF; ++q) load_row<DPL>(Z + min(base + rb + q, R - 1) * DP + lane * DPL, xb[q]);
#pragma unroll
      for (int q = 0; q < PF; ++q) {
      const int r = rb + q;
      if (r >= n) break;
      float x[DPL];
#pragma unroll
      for (int i = 0; i < DPL; ++i) x[i] = xb[q][i];
      float v[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) v[i] = 0.f;
      float s1 = 0.f, s2 = 0.f;
#pragma unroll
      for (int i = 0; i < DPL; ++i) {
        s1 += x[i];
        s2 += x[i] * x[i];
      }
      v[0] = s1;
      v[1] = s2;
#pragma unroll
      for (int h = 0; h < NH; ++h) {
        float d = 0.f;
#pragma unroll
        for (int i = 0; i < DPL; ++i) d += w[h][i] * x[i];
        v[2 + h] = d;
      }
      const float u = reduce_scatter16(v, lane);
      const float t1 = __shfl_sync(0xffffffffu, u, 0), t2 = __shfl_sync(0xffffffffu, u, 2);
      const float mean = t1 * (1.f / DP), rstd = rsqrtf(fmaxf(t2 * (1.f / DP) - mean * mean, 0.f) + eps);
      const int idx = lane >> 1;
      if ((lane & 1) == 0 && idx >= 2 && idx < 2 + NH) stage[wp][r][idx - 2] = rstd * (u - mean * sumw[idx - 2]);
      if (lane == 0) ST[base + r] = make_float2(mean, rstd);
      }
    }
    __syncwarp();
    if (lane < n) {
      const long row = base + lane;
      const bool drop = MASK != nullptr && !MASK[row % L];
#pragma unroll
      for (int h = 0; h < NH; ++h) OUT[(long)h * R + row] = __float2bfloat16_rn(drop ? neg : stage[wp][lane][h]);
    }
    __syncwarp();
  }
}

template <int DP, int NH, typename T, bool HAVE_ST>
__global__ void __launch_bounds__(BT) bwd_k(const T* __restrict__ Z, const float* __restrict__ DB, const float* __restrict__ W,
                                            const float2* __restrict__ ST, T* __restrict__ DZ, float* __restrict__ ACC, long R, float eps) {
  constexpr int DPL = DP / 32;
  __shared__ float red[WPB][DP];
  const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  constexpr bool WSM = NH > 8;  // 16 heads: the weights live in shared memory (64 fewer registers: occupancy)
  __shared__ float wsm[WSM ? NH * DP : 1];
  float w[WSM ? 1 : NH][DPL], part[NH][DPL];
  if constexpr (WSM) {
    for (int c = threadIdx.x; c < NH * DP; c += BT) wsm[c] = W[c];
    __syncthreads();
  }
#pragma unroll
  for (int h = 0; h < NH; ++h)
#pragma unroll
    for (int i = 0; i < DPL; ++i) {
      if constexpr (!WSM) w[h][i] = W[h * DP + lane * DPL + i];
      part[h][i] = 0.f;
    }
  const long warp = ((long)blockIdx.x * BT + threadIdx.x) >> 5, nwarps = (long)gridDim.x * WPB;
  for (long base = warp * 32; base < R; base += nwarps * 32) {
    float dbr[NH];
    const long mine = base + lane;
    float2 stm = make_float2(0.f, 1.f);
    if constexpr (HAVE_ST) stm = mine < R ? ST[mine] : stm;
#pragma unroll
    for (int h = 0; h < NH; ++h) dbr[h] = mine < R ? __ldg(DB + (long)h * R + mine) : 0.f;
    const int n = (int)min(32L, R - base);
    for (int rb = 0; rb < n; rb += PF) {
      float xb[PF][DPL];
#pragma unroll
      for (int q = 0; q < PF; ++q) load_row<DPL>(Z + min(base + rb + q, R - 1) * DP + lane * DPL, xb[q]);
#pragma unroll
      for (int q = 0; q < PF; ++q) {
      const int r = rb + q;
      if (r >= n) break;
      float x[DPL], d[NH], g[DPL];
#pragma unroll
      for (int i = 0; i < DPL; ++i) x[i] = xb[q][i];
      float rstd;
      if constexpr (HAVE_ST) {
        const float mean = __shfl_sync(0xffffffffu, stm.x, r);
        rstd = __shfl_sync(0xffffffffu, stm.y, r);
#pragma unroll
        for (int i = 0; i < DPL; ++i) x[i] = (x[i] - mean) * rstd;
      } else {
        normalize<DP, DPL>(x, eps, rstd);
      }
#pragma unroll
      for (int h = 0; h < NH; ++h) d[h] = __shfl_sync(0xffffffffu, dbr[h], r);
      float mg = 0.f, mgx = 0.f;
#pragma unroll
      for (int i = 0; i < DPL; ++i) {
        float s = 0.f;
#pragma unroll
        for (int h = 0; h < NH; ++h) {
          s += (WSM ? wsm[h * DP + lane * DPL + i] : w[WSM ? 0 : h][i]) * d[h];
          part[h][i] += d[h] * x[i];
        }
        g[i] = s;
        mg += s;
        mgx += s * x[i];
      }
      mg = tdr::warp_sum(mg) * (1.f / DP);
      mgx = tdr::warp_sum(mgx) * (1.f / DP);
#pragma unroll
      for (int i = 0; i < DPL; ++i) g[i] = rstd * (g[i] - mg - x[i] * mgx);
      store_row<DPL>(DZ + (base + r) * DP + lane * DPL, g);
      }
    }
  }
  for (int h = 0; h < NH; ++h) {
#pragma unroll
    for (int i = 0; i < DPL; ++i) red[wp][lane * DPL + i] = part[h][i];
    __syncthreads();
    for (int c = threadIdx.x; c < DP; c += BT) {
      float v = 0.f;
#pragma unroll
      for (int q = 0; q < WPB; ++q) v += red[q][c];
      ACC[((long)blockIdx.x * NH + h) * DP + c] = v;
    }
    __syncthreads();
  }
}


// ------------------------------------------------------------------------------------------------ fp32 TF32-core glue
// key_perm (conditioned_transition.cuda): position p of each group of 8 holds key perm(p); inv is its inverse
__device__ __forceinline__ int kperm(int p) { const int r = p & 7; return p - r + 4 * (r & 1) + (r >> 1); }
__device__ __forceinline__ int kinv(int k) { const int s = k & 7; return k - s + 2 * (s & 3) + (s >> 2); }

// TF32 round to nearest (ties away from zero) of a finite float: what cuBLAS does to an fp32 operand and what the tensor core would otherwise do by
// truncation (1.2 - 3x the rounding error end to end, measured)
__device__ __forceinline__ float rna_tf32(float x) { return __uint_as_float((__float_as_uint(x) + 0x1000u) & 0xffffe000u); }

// q * scale, k, v rounded to the TF32 grid: QS, KR, VR [M, N] and VT [N, M] = VR^T, one pass over q, k, v
__global__ void __launch_bounds__(256) prep_k(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V, float* __restrict__ QS,
                                              float* __restrict__ KR, float* __restrict__ VR, float* __restrict__ VT, long M, long N, float scale) {
  __shared__ float tile[32][33];
  const long r0 = (long)blockIdx.y * 32, c0 = (long)blockIdx.x * 32;
  const int tx = threadIdx.x % 32, ty = threadIdx.x / 32;
#pragma unroll
  for (int i = 0; i < 32; i += 8) {
    const long idx = (r0 + ty + i) * N + c0 + tx;
    QS[idx] = rna_tf32(Q[idx] * scale);
    KR[idx] = rna_tf32(K[idx]);
    const float v = rna_tf32(V[idx]);
    VR[idx] = v;
    tile[ty + i][tx] = v;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 32; i += 8) VT[(c0 + ty + i) * M + r0 + tx] = tile[tx][ty + i];
}

// D[a, h, l] = sum_d dO[a L + l, 48 h + d] O[a L + l, 48 h + d], plus its query-permuted copy and the permuted copy of the LSE
__global__ void __launch_bounds__(256) dd_k(const float* __restrict__ DO, const float* __restrict__ O, const float* __restrict__ LSE, float* __restrict__ DD,
                                            float* __restrict__ DDP, float* __restrict__ LSEP, float* __restrict__ DOR, long rows, int L) {
  // a block: 16 consecutive rows (one sample: L % 16 == 0) of 192 float4: dO is read, rounded and written, O read, all coalesced; the 12 float4 of a
  // head are summed in a fixed order from shared memory
  __shared__ float s[16 * 192];  // the 192 float4 partial dot products of each row
  const long row0 = (long)blockIdx.x * 16;
  const float4* a = reinterpret_cast<const float4*>(DO + row0 * 768);
  const float4* b = reinterpret_cast<const float4*>(O + row0 * 768);
  float4* r = reinterpret_cast<float4*>(DOR + row0 * 768);
#pragma unroll
  for (int i = threadIdx.x; i < 16 * 192; i += 256) {
    const float4 x = __ldg(a + i), y = __ldg(b + i);
    r[i] = make_float4(rna_tf32(x.x), rna_tf32(x.y), rna_tf32(x.z), rna_tf32(x.w));
    s[i] = x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w;
  }
  __syncthreads();
  const int h2 = threadIdx.x >> 4, r2 = threadIdx.x & 15;
  const long smp = row0 / L;
  const int l = (int)(row0 % L) + r2;
  const long o = (smp * 16 + h2) * L;
  float v = 0.f;  // fixed order: bitwise repeatable
#pragma unroll
  for (int j = 0; j < 12; ++j) v += s[r2 * 192 + h2 * 12 + j];
  DD[o + l] = v;
  DDP[o + kinv(l)] = v;
  LSEP[o + kinv(l)] = LSE[o + l];
}

// Z[h, k, q] = bias_p[h, perm(q), inv(k)]: the backward's transposed bias (key rows, permuted query columns)
__global__ void __launch_bounds__(256) bias_t_k(const float* __restrict__ B, float* __restrict__ Z, long total, int L) {
  const long idx = (long)blockIdx.x * 256 + threadIdx.x;
  if (idx >= total) return;
  const int q = (int)(idx % L);
  const long hk = idx / L;
  const int k = (int)(hk % L);
  const long h = hk / L;
  Z[idx] = B[(h * L + kperm(q)) * L + kinv(k)];
}
// ---- the dense atom stack: d_pair 16, a thread per pair row (16 channels are 32 bytes: two 16-byte loads)
constexpr int C = 16;
constexpr int ABT = 256;

__device__ __forceinline__ void atom_load_row(const bf* __restrict__ z, float (&x)[C]) {
  const uint4 a = __ldg(reinterpret_cast<const uint4*>(z)), b = __ldg(reinterpret_cast<const uint4*>(z) + 1);
  const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    x[2 * i] = __uint_as_float(w[i] << 16);
    x[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}

__device__ __forceinline__ void atom_normalize(float (&x)[C], float eps, float& rstd) {
  float s = 0.f;
#pragma unroll
  for (int c = 0; c < C; ++c) s += x[c];
  const float mean = s * (1.f / C);
  float v = 0.f;
#pragma unroll
  for (int c = 0; c < C; ++c) {
    x[c] -= mean;
    v += x[c] * x[c];
  }
  rstd = rsqrtf(v * (1.f / C) + eps);
#pragma unroll
  for (int c = 0; c < C; ++c) x[c] *= rstd;
}

template <int NH>
__global__ void __launch_bounds__(ABT) atom_fwd_k(const bf* __restrict__ Z, const float* __restrict__ W, const bool* __restrict__ MASK,
                                            bf* __restrict__ OUT, long R, int L, float eps, float scale, float neg) {
  __shared__ float w[NH * C];
  if (threadIdx.x < NH * C) w[threadIdx.x] = W[threadIdx.x] * scale;
  __syncthreads();
  for (long r = (long)blockIdx.x * ABT + threadIdx.x; r < R; r += (long)gridDim.x * ABT) {
    float x[C], rstd;
    atom_load_row(Z + r * C, x);
    atom_normalize(x, eps, rstd);
    const bool drop = MASK != nullptr && !MASK[r % L];
#pragma unroll
    for (int h = 0; h < NH; ++h) {
      float d = 0.f;
#pragma unroll
      for (int c = 0; c < C; ++c) d += w[h * C + c] * x[c];
      OUT[(long)h * R + r] = __float2bfloat16_rn(drop ? neg : d);
    }
  }
}

template <int NH>
__global__ void __launch_bounds__(ABT) atom_bwd_k(const bf* __restrict__ Z, const float* __restrict__ DB, const float* __restrict__ W,
                                            bf* __restrict__ DZ, float* __restrict__ ACC, long R, float eps) {
  __shared__ float w[NH * C];
  __shared__ float red[ABT / 32][NH * C];
  if (threadIdx.x < NH * C) w[threadIdx.x] = W[threadIdx.x];
  __syncthreads();
  float part[NH * C];
#pragma unroll
  for (int i = 0; i < NH * C; ++i) part[i] = 0.f;
  for (long r = (long)blockIdx.x * ABT + threadIdx.x; r < R; r += (long)gridDim.x * ABT) {
    float x[C], rstd, db[NH], g[C];
    atom_load_row(Z + r * C, x);
    atom_normalize(x, eps, rstd);
#pragma unroll
    for (int h = 0; h < NH; ++h) db[h] = __ldg(DB + (long)h * R + r);
    float mg = 0.f, mgx = 0.f;
#pragma unroll
    for (int c = 0; c < C; ++c) {
      float s = 0.f;
#pragma unroll
      for (int h = 0; h < NH; ++h) {
        s += w[h * C + c] * db[h];
        part[h * C + c] += db[h] * x[c];
      }
      g[c] = s;
      mg += s;
      mgx += s * x[c];
    }
    mg *= 1.f / C;
    mgx *= 1.f / C;
    uint32_t o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float a = rstd * (g[2 * i] - mg - x[2 * i] * mgx), b = rstd * (g[2 * i + 1] - mg - x[2 * i + 1] * mgx);
      const __nv_bfloat162 p = __floats2bfloat162_rn(a, b);
      o[i] = *reinterpret_cast<const uint32_t*>(&p);
    }
    uint4* dz = reinterpret_cast<uint4*>(DZ + r * C);
    dz[0] = make_uint4(o[0], o[1], o[2], o[3]);
    dz[1] = make_uint4(o[4], o[5], o[6], o[7]);
  }
  // block partial of dWf: warp shuffle, then the warps in order
#pragma unroll
  for (int i = 0; i < NH * C; ++i) {
    float v = part[i];
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5][i] = v;
  }
  __syncthreads();
  if (threadIdx.x < NH * C) {
    float v = 0.f;
#pragma unroll
    for (int wp = 0; wp < ABT / 32; ++wp) v += red[wp][threadIdx.x];
    ACC[(long)blockIdx.x * NH * C + threadIdx.x] = v;
  }
}

int nsm_of(const at::Tensor& t) {
  int n = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, t.get_device()));
  return n;
}
void chk(const at::Tensor& t, at::ScalarType dt, const char* n) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == dt && t.is_contiguous(), n, ": contiguous CUDA tensor of the expected dtype");
}

#define PAIR_SWITCH(DPV, NHV, ...)                                                      \
  do {                                                                                  \
    TORCH_CHECK((NHV) == 8, "pair_rows: 8 heads, got ", (long)(NHV));                   \
    if ((DPV) == 128) { constexpr int DPK = 128; __VA_ARGS__; }                         \
    else if ((DPV) == 256) { constexpr int DPK = 256; __VA_ARGS__; }                    \
    else if ((DPV) == 64) { constexpr int DPK = 64; __VA_ARGS__; }                      \
    else TORCH_CHECK(false, "pair_rows: d_pair 64 / 128 / 256, got ", (long)(DPV));     \
  } while (0)
}  // namespace

at::Tensor atom_pair_fwd(at::Tensor z, at::Tensor wf, c10::optional<at::Tensor> mask, int64_t L, double eps, double scale, double neg) {
  chk(z, at::kBFloat16, "pair");
  chk(wf, at::kFloat, "wf");
  const int64_t nh = wf.size(0), R = L * L;
  TORCH_CHECK(wf.size(1) == C && z.numel() == R * C, "atom_pair_fwd shapes");
  if (mask) {
    chk(*mask, at::kBool, "mask");
    TORCH_CHECK(mask->numel() == L, "mask [L]");
  }
  const at::cuda::CUDAGuard gd(z.device());
  auto out = at::empty({nh, L, L}, z.options());
  const unsigned grid = (unsigned)std::min<int64_t>((R + ABT - 1) / ABT, 8L * nsm_of(z));
  const bool* mp = mask ? mask->data_ptr<bool>() : nullptr;
  auto s = at::cuda::getCurrentCUDAStream();
  const bf* zp = reinterpret_cast<const bf*>(z.data_ptr());
  bf* op = reinterpret_cast<bf*>(out.data_ptr());
  if (nh == 4) atom_fwd_k<4><<<grid, ABT, 0, s>>>(zp, wf.data_ptr<float>(), mp, op, R, (int)L, (float)eps, (float)scale, (float)neg);
  else if (nh == 8) atom_fwd_k<8><<<grid, ABT, 0, s>>>(zp, wf.data_ptr<float>(), mp, op, R, (int)L, (float)eps, (float)scale, (float)neg);
  else TORCH_CHECK(false, "atom_pair: 4 or 8 heads");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<at::Tensor> atom_pair_bwd(at::Tensor z, at::Tensor db, at::Tensor wf, int64_t L, double eps) {
  chk(z, at::kBFloat16, "pair");
  chk(db, at::kFloat, "dbias");
  chk(wf, at::kFloat, "wf");
  const int64_t nh = wf.size(0), R = L * L;
  TORCH_CHECK(wf.size(1) == C && z.numel() == R * C && db.numel() == nh * R, "atom_pair_bwd shapes");
  const at::cuda::CUDAGuard gd(z.device());
  auto dz = at::empty_like(z);
  const unsigned grid = (unsigned)std::min<int64_t>((R + ABT - 1) / ABT, 4L * nsm_of(z));
  auto acc = at::empty({(int64_t)grid, nh * C}, z.options().dtype(at::kFloat));
  auto s = at::cuda::getCurrentCUDAStream();
  const bf* zp = reinterpret_cast<const bf*>(z.data_ptr());
  bf* dp = reinterpret_cast<bf*>(dz.data_ptr());
  if (nh == 4) atom_bwd_k<4><<<grid, ABT, 0, s>>>(zp, db.data_ptr<float>(), wf.data_ptr<float>(), dp, acc.data_ptr<float>(), R, (float)eps);
  else if (nh == 8) atom_bwd_k<8><<<grid, ABT, 0, s>>>(zp, db.data_ptr<float>(), wf.data_ptr<float>(), dp, acc.data_ptr<float>(), R, (float)eps);
  else TORCH_CHECK(false, "atom_pair: 4 or 8 heads");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dz, acc};
}

std::vector<at::Tensor> pair_rows_fwd(at::Tensor z, at::Tensor wf, c10::optional<at::Tensor> mask, int64_t L, double eps, double scale, double neg) {
  chk(z, at::kBFloat16, "pair");
  chk(wf, at::kFloat, "wf");
  const int64_t nh = wf.size(0), dp = wf.size(1), R = L * L;
  TORCH_CHECK(z.numel() == R * dp, "pair_rows_fwd shapes");
  if (mask) {
    chk(*mask, at::kBool, "mask");
    TORCH_CHECK(mask->numel() == L, "mask [L]");
  }
  const at::cuda::CUDAGuard gd(z.device());
  auto out = at::empty({nh, L, L}, z.options());
  auto st = at::empty({R, 2}, z.options().dtype(at::kFloat));
  const unsigned grid = (unsigned)std::min<int64_t>((R / 32 + WPB - 1) / WPB, 4L * nsm_of(z));
  const bool* mp = mask ? mask->data_ptr<bool>() : nullptr;
  auto s = at::cuda::getCurrentCUDAStream();
  const bf* zp = reinterpret_cast<const bf*>(z.data_ptr());
  bf* op = reinterpret_cast<bf*>(out.data_ptr());
  PAIR_SWITCH(dp, nh, fwd_k<DPK><<<grid, BT, 0, s>>>(zp, wf.data_ptr<float>(), mp, op, reinterpret_cast<float2*>(st.data_ptr()), R, (int)L, (float)eps, (float)scale, (float)neg));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, st};
}

std::vector<at::Tensor> pair_rows_bwd(at::Tensor z, at::Tensor db, at::Tensor wf, at::Tensor st, int64_t L, double eps) {
  chk(z, at::kBFloat16, "pair");
  chk(db, at::kFloat, "dbias");
  chk(wf, at::kFloat, "wf");
  const int64_t nh = wf.size(0), dp = wf.size(1), R = L * L;
  TORCH_CHECK(z.numel() == R * dp && db.numel() == nh * R, "pair_rows_bwd shapes");
  const at::cuda::CUDAGuard gd(z.device());
  auto dz = at::empty_like(z);
  const unsigned grid = (unsigned)std::min<int64_t>((R / 32 + WPB - 1) / WPB, 2L * nsm_of(z));
  auto acc = at::empty({(int64_t)grid, nh * dp}, z.options().dtype(at::kFloat));
  auto s = at::cuda::getCurrentCUDAStream();
  const bf* zp = reinterpret_cast<const bf*>(z.data_ptr());
  bf* dp_ = reinterpret_cast<bf*>(dz.data_ptr());
  PAIR_SWITCH(dp, nh, bwd_k<DPK, 8, bf, true><<<grid, BT, 0, s>>>(zp, db.data_ptr<float>(), wf.data_ptr<float>(), reinterpret_cast<const float2*>(st.data_ptr()), dp_, acc.data_ptr<float>(), R, (float)eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dz, acc};
}

std::vector<at::Tensor> prep_fwd32(at::Tensor q, at::Tensor k, at::Tensor v, double scale) {
  chk(q, at::kFloat, "q");
  chk(k, at::kFloat, "k");
  chk(v, at::kFloat, "v");
  const int64_t M = q.size(0), N = q.size(1);
  TORCH_CHECK(M % 32 == 0 && N % 32 == 0 && k.sizes() == q.sizes() && v.sizes() == q.sizes(), "prep_fwd32 shapes");
  const at::cuda::CUDAGuard gd(q.device());
  auto qs = at::empty_like(q), kr = at::empty_like(q), vr = at::empty_like(q), vt = at::empty({N, M}, q.options());
  prep_k<<<dim3((unsigned)(N / 32), (unsigned)(M / 32)), 256, 0, at::cuda::getCurrentCUDAStream()>>>(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
      qs.data_ptr<float>(), kr.data_ptr<float>(), vr.data_ptr<float>(), vt.data_ptr<float>(), M, N, (float)scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {qs, kr, vr, vt};
}

at::Tensor transpose32(at::Tensor src) {
  chk(src, at::kFloat, "src");
  const int64_t M = src.size(0), N = src.size(1);
  TORCH_CHECK(M % 32 == 0 && N % 32 == 0, "transpose32: multiples of 32");
  const at::cuda::CUDAGuard gd(src.device());
  auto dst = at::empty({N, M}, src.options());
  tdr::transpose_k<float><<<dim3((unsigned)(N / 32), (unsigned)(M / 32)), 256, 0, at::cuda::getCurrentCUDAStream()>>>(src.data_ptr<float>(), dst.data_ptr<float>(), M, N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dst;
}

// dO, O [A L, 768] fp32, lse [A, 16, L] -> (D [A, 16, L], D permuted, lse permuted, dO rounded to the TF32 grid)
std::vector<at::Tensor> dd_perm(at::Tensor dO, at::Tensor O, at::Tensor lse, int64_t A, int64_t L) {
  chk(dO, at::kFloat, "dO");
  chk(O, at::kFloat, "O");
  chk(lse, at::kFloat, "lse");
  TORCH_CHECK(dO.numel() == A * L * 768 && L % 16 == 0, "dd_perm shapes");
  const at::cuda::CUDAGuard gd(dO.device());
  auto dd = at::empty({A, 16, L}, dO.options()), ddp = at::empty_like(dd), lsep = at::empty_like(dd), dor = at::empty_like(dO);
  dd_k<<<(unsigned)(A * L / 16), 256, 0, at::cuda::getCurrentCUDAStream()>>>(dO.data_ptr<float>(), O.data_ptr<float>(), lse.data_ptr<float>(),
      dd.data_ptr<float>(), ddp.data_ptr<float>(), lsep.data_ptr<float>(), dor.data_ptr<float>(), A * L, (int)L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dd, ddp, lsep, dor};
}

at::Tensor bias_transposed(at::Tensor bias_p, int64_t L) {
  chk(bias_p, at::kFloat, "bias");
  TORCH_CHECK(bias_p.numel() == 16 * L * L && L % 8 == 0, "bias_transposed shapes");
  const at::cuda::CUDAGuard gd(bias_p.device());
  auto z = at::empty_like(bias_p);
  const long n = bias_p.numel();
  bias_t_k<<<(unsigned)((n + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(bias_p.data_ptr<float>(), z.data_ptr<float>(), n, (int)L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return z;
}

// fp32 pair [L L, 128], 16 heads (the fp32 token stack): dpair and the dWf block partials; the LayerNorm statistics are recomputed
std::vector<at::Tensor> pair_rows_bwd32(at::Tensor z, at::Tensor db, at::Tensor wf, int64_t L, double eps) {
  chk(z, at::kFloat, "pair");
  chk(db, at::kFloat, "dbias");
  chk(wf, at::kFloat, "wf");
  const int64_t R = L * L;
  TORCH_CHECK(wf.size(0) == 16 && wf.size(1) == 128 && z.numel() == R * 128 && db.numel() == 16 * R, "pair_rows_bwd32 shapes");
  const at::cuda::CUDAGuard gd(z.device());
  auto dz = at::empty_like(z);
  const unsigned grid = (unsigned)std::min<int64_t>((R / 32 + WPB - 1) / WPB, 2L * nsm_of(z));
  auto acc = at::empty({(int64_t)grid, 16 * 128}, z.options());
  bwd_k<128, 16, float, false><<<grid, BT, 0, at::cuda::getCurrentCUDAStream()>>>(
      z.data_ptr<float>(), db.data_ptr<float>(), wf.data_ptr<float>(), nullptr, dz.data_ptr<float>(), acc.data_ptr<float>(), R, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dz, acc};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pair_rows_bwd32", &pair_rows_bwd32);
  m.def("transpose32", &transpose32);
  m.def("prep_fwd32", &prep_fwd32);
  m.def("dd_perm", &dd_perm);
  m.def("bias_transposed", &bias_transposed);
  m.def("atom_pair_fwd", &atom_pair_fwd);
  m.def("atom_pair_bwd", &atom_pair_bwd);
  m.def("pair_rows_fwd", &pair_rows_fwd);
  m.def("pair_rows_bwd", &pair_rows_bwd);
}
