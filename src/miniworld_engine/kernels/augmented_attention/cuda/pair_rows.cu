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

// ------------------------------------------------------------------------------------------------ fp32 TF32-core glue
// key_perm (conditioned_transition.cuda): position p of each group of 8 holds key perm(p); inv is its inverse
__device__ __forceinline__ int kperm(int p) { const int r = p & 7; return p - r + 4 * (r & 1) + (r >> 1); }
__device__ __forceinline__ int kinv(int k) { const int s = k & 7; return k - s + 2 * (s & 3) + (s >> 2); }
// position of pair row r in a bias/dbias buffer whose key columns are kept in the TF32 core's permuted order (PL = L), or r itself (PL = 0)
__device__ __forceinline__ long dbpos(long r, int PL) { if (PL == 0) return r; const int j = (int)(r % PL); return r - j + kinv(j); }


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

template <int DP, int NH, typename T, bool HAVE_ST, int CH = 32>   // CH rows per warp-chunk (8 when few rows must still fill the GPU)
__global__ void __launch_bounds__(BT) bwd_k(const T* __restrict__ Z, const float* __restrict__ DB, const float* __restrict__ W,
                                            const float2* __restrict__ ST, T* __restrict__ DZ, float* __restrict__ ACC, long R, float eps, int PL) {
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
  for (long base = warp * CH; base < R; base += nwarps * CH) {
    float dbr[NH];
    const long mine = base + lane;
    float2 stm = make_float2(0.f, 1.f);
    if constexpr (HAVE_ST) stm = mine < R ? ST[mine] : stm;
#pragma unroll
    const long dpos = mine < R ? dbpos(mine, PL) : 0;
    for (int h = 0; h < NH; ++h) dbr[h] = mine < R ? __ldg(DB + (long)h * R + dpos) : 0.f;
    const int n = (int)min((long)CH, R - base);
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


// TF32 round to nearest (ties away from zero) of a finite float: what cuBLAS does to an fp32 operand and what the tensor core would otherwise do by
// truncation (1.2 - 3x the rounding error end to end, measured)
__device__ __forceinline__ float rna_tf32(float x) { return __uint_as_float((__float_as_uint(x) + 0x1000u) & 0xffffe000u); }

// q * scale, k, v rounded to the TF32 grid: QS, KR, VR [M, N] and VT [N, M] = VR^T, one pass over q, k, v; with WT (a training forward) also QT = QS^T and
// KT = KR^T [N, M], which the backward's transposed operands would otherwise cost two more passes for
template <bool WT>
__global__ void __launch_bounds__(256) prep_k(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V, float* __restrict__ QS,
                                              float* __restrict__ KR, float* __restrict__ VR, float* __restrict__ VT, float* __restrict__ QT,
                                              float* __restrict__ KT, long M, long N, float scale, long ldq, long ldk, long ldv) {
  __shared__ float tile[32][33];
  __shared__ float tq[WT ? 32 : 1][WT ? 33 : 1], tk[WT ? 32 : 1][WT ? 33 : 1];
  const long r0 = (long)blockIdx.y * 32, c0 = (long)blockIdx.x * 32;
  const int tx = threadIdx.x % 32, ty = threadIdx.x / 32;
#pragma unroll
  for (int i = 0; i < 32; i += 8) {
    const long idx = (r0 + ty + i) * N + c0 + tx;
    const long rr = r0 + ty + i;
    const float q = rna_tf32(Q[rr * ldq + c0 + tx] * scale), k = rna_tf32(K[rr * ldk + c0 + tx]);
    QS[idx] = q;
    KR[idx] = k;
    const float v = rna_tf32(V[rr * ldv + c0 + tx]);
    VR[idx] = v;
    tile[ty + i][tx] = v;
    if constexpr (WT) { tq[ty + i][tx] = q; tk[ty + i][tx] = k; }
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 32; i += 8) {
    VT[(c0 + ty + i) * M + r0 + tx] = tile[tx][ty + i];
    if constexpr (WT) { QT[(c0 + ty + i) * M + r0 + tx] = tq[tx][ty + i]; KT[(c0 + ty + i) * M + r0 + tx] = tk[tx][ty + i]; }
  }
}


// D[a, h, l] = sum_d dO[a L + l, DH h + d] O[a L + l, DH h + d], plus its query-permuted copy and the permuted copy of the LSE (NH heads of width DH)
template <int NH, int DH>
__global__ void __launch_bounds__(256) dd_k(const float* __restrict__ DO, const float* __restrict__ O, const float* __restrict__ LSE, float* __restrict__ DD,
                                            float* __restrict__ DDP, float* __restrict__ LSEP, float* __restrict__ DOR, float* __restrict__ DOT, long rows, int L) {
  // a block: 16 consecutive rows (one sample: L % 16 == 0) of NH DH / 4 float4: dO is read, rounded and written, O read, all coalesced; the DH / 4
  // float4 of a head are summed in a fixed order from shared memory
  constexpr int F4 = NH * DH / 4, HF = DH / 4;
  __shared__ float s[16 * F4];  // the float4 partial dot products of each row
  extern __shared__ float dsm[];   // [16][NH DH + 1]: the rounded dO of the block, only with DOT (its transpose is written from here)
  constexpr int N = NH * DH;
  const long row0 = (long)blockIdx.x * 16;
  const float4* a = reinterpret_cast<const float4*>(DO + row0 * NH * DH);
  const float4* b = reinterpret_cast<const float4*>(O + row0 * NH * DH);
  float4* r = reinterpret_cast<float4*>(DOR + row0 * NH * DH);
#pragma unroll
  for (int i = threadIdx.x; i < 16 * F4; i += 256) {
    const float4 x = __ldg(a + i), y = __ldg(b + i);
    const float4 rr = make_float4(rna_tf32(x.x), rna_tf32(x.y), rna_tf32(x.z), rna_tf32(x.w));
    r[i] = rr;
    if (DOT != nullptr) {
      float* p = dsm + (i / F4) * (N + 1) + (i % F4) * 4;
      p[0] = rr.x; p[1] = rr.y; p[2] = rr.z; p[3] = rr.w;
    }
    s[i] = x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w;
  }
  __syncthreads();
  if (DOT != nullptr)   // dO^T [N, rows]: 16 consecutive rows of a column are 64 contiguous bytes
    for (int idx = threadIdx.x; idx < 16 * N; idx += 256) DOT[(long)(idx / 16) * rows + row0 + (idx % 16)] = dsm[(idx % 16) * (N + 1) + idx / 16];
  const int h2 = threadIdx.x >> 4, r2 = threadIdx.x & 15;
  if (h2 >= NH) return;
  const long smp = row0 / L;
  const int l = (int)(row0 % L) + r2;
  const long o = (smp * NH + h2) * L;
  float v = 0.f;  // fixed order: bitwise repeatable
#pragma unroll
  for (int j = 0; j < HF; ++j) v += s[r2 * F4 + h2 * HF + j];
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

// fp32 pair [R, 256], 16 heads, backward (dpair and the dWf block partials): TWO warps share a row (channels half * 128 + 4 lane .. + 3), so a lane keeps 4
// channels and 64 dWf partial registers instead of 128 (which one warp per row spilled: 40x slower). The warps of a pair exchange the LayerNorm
// statistics and the two means of g through shared memory under a named barrier.
__global__ void __launch_bounds__(BT) bwd256_k(const float* __restrict__ Z, const float* __restrict__ DB, const float* __restrict__ W,
                                               float* __restrict__ DZ, float* __restrict__ ACC, long R, float eps, int PL) {
  constexpr int DP = 256, NH = 16, HALF = 128;
  __shared__ __align__(16) float wsm[NH * DP];
  __shared__ float red[WPB][HALF];
  __shared__ float2 xch[WPB / 2][2][2];   // [pair][round][half]
  for (int c = threadIdx.x; c < NH * DP / 4; c += BT) reinterpret_cast<float4*>(wsm)[c] = reinterpret_cast<const float4*>(W)[c];
  __syncthreads();
  const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5, pr = wp >> 1, half = wp & 1;
  const long gp = (long)blockIdx.x * (WPB / 2) + pr, npairs = (long)gridDim.x * (WPB / 2);
  float part[NH][4];
#pragma unroll
  for (int h = 0; h < NH; ++h)
#pragma unroll
    for (int i = 0; i < 4; ++i) part[h][i] = 0.f;
  for (long base = gp * 32; base < R; base += npairs * 32) {
    float dbr[NH];
#pragma unroll
    const long dpos = dbpos(base + lane, PL);
    for (int h = 0; h < NH; ++h) dbr[h] = __ldg(DB + (long)h * R + dpos);
    for (int rb = 0; rb < 32; rb += PF) {
      float4 xb[PF];
#pragma unroll
      for (int q = 0; q < PF; ++q) xb[q] = __ldg(reinterpret_cast<const float4*>(Z + (base + rb + q) * DP + half * HALF + lane * 4));
#pragma unroll
      for (int q = 0; q < PF; ++q) {
        const int r = rb + q;
        float x[4] = {xb[q].x, xb[q].y, xb[q].z, xb[q].w};
        // LayerNorm statistics of the 256 channels: (mean, M2) of each half, merged
        float m = tdr::warp_sum(x[0] + x[1] + x[2] + x[3]) * (1.f / HALF), m2 = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) m2 += (x[i] - m) * (x[i] - m);
        m2 = tdr::warp_sum(m2);
        if (lane == 0) xch[pr][0][half] = make_float2(m, m2);
        asm volatile("bar.sync %0, 64;" ::"r"(1 + pr) : "memory");
        const float2 a = xch[pr][0][0], b = xch[pr][0][1];
        const float mean = 0.5f * (a.x + b.x);
        const float var = (a.y + b.y + HALF * ((a.x - mean) * (a.x - mean) + (b.x - mean) * (b.x - mean))) * (1.f / DP);
        const float rstd = rsqrtf(var + eps);
#pragma unroll
        for (int i = 0; i < 4; ++i) x[i] = (x[i] - mean) * rstd;
        float d[NH];
#pragma unroll
        for (int h = 0; h < NH; ++h) d[h] = __shfl_sync(0xffffffffu, dbr[h], r);
        float g[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int h = 0; h < NH; ++h) {
          const float4 w = *reinterpret_cast<const float4*>(wsm + h * DP + half * HALF + lane * 4);
          g[0] += w.x * d[h]; g[1] += w.y * d[h]; g[2] += w.z * d[h]; g[3] += w.w * d[h];
#pragma unroll
          for (int i = 0; i < 4; ++i) part[h][i] += d[h] * x[i];
        }
        float mg = tdr::warp_sum(g[0] + g[1] + g[2] + g[3]);
        float mgx = tdr::warp_sum(g[0] * x[0] + g[1] * x[1] + g[2] * x[2] + g[3] * x[3]);
        if (lane == 0) xch[pr][1][half] = make_float2(mg, mgx);
        asm volatile("bar.sync %0, 64;" ::"r"(1 + pr) : "memory");
        const float2 c0 = xch[pr][1][0], c1 = xch[pr][1][1];
        mg = (c0.x + c1.x) * (1.f / DP);
        mgx = (c0.y + c1.y) * (1.f / DP);
        reinterpret_cast<float4*>(DZ + (base + r) * DP + half * HALF + lane * 4)[0] =
            make_float4(rstd * (g[0] - mg - x[0] * mgx), rstd * (g[1] - mg - x[1] * mgx), rstd * (g[2] - mg - x[2] * mgx), rstd * (g[3] - mg - x[3] * mgx));
      }
    }
  }
#pragma unroll
  for (int h = 0; h < NH; ++h) {
#pragma unroll
    for (int i = 0; i < 4; ++i) red[wp][lane * 4 + i] = part[h][i];
    __syncthreads();
    for (int c = threadIdx.x; c < DP; c += BT) {
      const int hf = c / HALF, col = c % HALF;
      float v = 0.f;
#pragma unroll
      for (int p = 0; p < WPB / 2; ++p) v += red[2 * p + hf][col];
      ACC[((long)blockIdx.x * NH + h) * DP + c] = v;
    }
    __syncthreads();
  }
}

// ---- the dense atom stack on the TF32 core (fp32 pair, d_pair 16, a thread per pair row): the forward writes the bias key-permuted
__device__ __forceinline__ void atom_load_row32(const float* __restrict__ z, float (&x)[C]) {
#pragma unroll
  for (int i = 0; i < C / 4; ++i) {
    const float4 a = __ldg(reinterpret_cast<const float4*>(z) + i);
    x[4 * i] = a.x; x[4 * i + 1] = a.y; x[4 * i + 2] = a.z; x[4 * i + 3] = a.w;
  }
}

template <int NH>
__global__ void __launch_bounds__(ABT) atom32_fwd_k(const float* __restrict__ Z, const float* __restrict__ W, const bool* __restrict__ MASK,
                                                     float* __restrict__ OUT, long R, int L, float eps, float neg) {
  __shared__ float w[NH * C];
  if (threadIdx.x < NH * C) w[threadIdx.x] = W[threadIdx.x];
  __syncthreads();
  for (long r = (long)blockIdx.x * ABT + threadIdx.x; r < R; r += (long)gridDim.x * ABT) {
    float x[C], rstd;
    atom_load_row32(Z + r * C, x);
    atom_normalize(x, eps, rstd);
    const int j = (int)(r % L);
    const bool drop = MASK != nullptr && !MASK[j];
    const long o = r - j + kinv(j);   // position kinv(j) of the row holds key j (position p holds key kperm(p))
#pragma unroll
    for (int h = 0; h < NH; ++h) {
      float d = 0.f;
#pragma unroll
      for (int c = 0; c < C; ++c) d += w[h * C + c] * x[c];
      OUT[(long)h * R + o] = drop ? neg : d;
    }
  }
}

template <int NH>
__global__ void __launch_bounds__(ABT) atom32_bwd_k(const float* __restrict__ Z, const float* __restrict__ DB, const float* __restrict__ W,
                                                     float* __restrict__ DZ, float* __restrict__ ACC, long R, float eps, int PL) {
  __shared__ float w[NH * C];
  __shared__ float red[ABT / 32][NH * C];
  if (threadIdx.x < NH * C) w[threadIdx.x] = W[threadIdx.x];
  __syncthreads();
  float part[NH * C];
#pragma unroll
  for (int i = 0; i < NH * C; ++i) part[i] = 0.f;
  for (long r = (long)blockIdx.x * ABT + threadIdx.x; r < R; r += (long)gridDim.x * ABT) {
    float x[C], rstd, db[NH], g[C];
    atom_load_row32(Z + r * C, x);
    atom_normalize(x, eps, rstd);
#pragma unroll
    for (int h = 0; h < NH; ++h) db[h] = __ldg(DB + (long)h * R + dbpos(r, PL));
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
    float4* dz = reinterpret_cast<float4*>(DZ + r * C);
#pragma unroll
    for (int i = 0; i < C / 4; ++i) {
      float a[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) a[e] = rstd * (g[4 * i + e] - mg - x[4 * i + e] * mgx);
      dz[i] = make_float4(a[0], a[1], a[2], a[3]);
    }
  }
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
  PAIR_SWITCH(dp, nh, bwd_k<DPK, 8, bf, true><<<grid, BT, 0, s>>>(zp, db.data_ptr<float>(), wf.data_ptr<float>(), reinterpret_cast<const float2*>(st.data_ptr()), dp_, acc.data_ptr<float>(), R, (float)eps, 0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dz, acc};
}

std::vector<at::Tensor> prep_fwd32(at::Tensor q, at::Tensor k, at::Tensor v, double scale, bool want_t) {
  for (const at::Tensor* t : {&q, &k, &v}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kFloat && t->dim() == 2 && t->stride(1) == 1 && t->stride(0) % 4 == 0, "prep_fwd32: fp32 rows (a column slice of a wider matrix is fine)");
  const int64_t M = q.size(0), N = q.size(1);
  TORCH_CHECK(M % 32 == 0 && N % 32 == 0 && k.sizes() == q.sizes() && v.sizes() == q.sizes(), "prep_fwd32 shapes");
  const at::cuda::CUDAGuard gd(q.device());
  auto qs = at::empty({M, N}, q.options()), kr = at::empty({M, N}, q.options()), vr = at::empty({M, N}, q.options()), vt = at::empty({N, M}, q.options());
  auto qt = want_t ? at::empty({N, M}, q.options()) : at::empty({1}, q.options()), kt = want_t ? at::empty({N, M}, q.options()) : at::empty({1}, q.options());
  const dim3 grid((unsigned)(N / 32), (unsigned)(M / 32));
  auto s = at::cuda::getCurrentCUDAStream();
  if (want_t)
    prep_k<true><<<grid, 256, 0, s>>>(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), qs.data_ptr<float>(), kr.data_ptr<float>(), vr.data_ptr<float>(),
        vt.data_ptr<float>(), qt.data_ptr<float>(), kt.data_ptr<float>(), M, N, (float)scale, q.stride(0), k.stride(0), v.stride(0));
  else
    prep_k<false><<<grid, 256, 0, s>>>(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), qs.data_ptr<float>(), kr.data_ptr<float>(), vr.data_ptr<float>(),
        vt.data_ptr<float>(), nullptr, nullptr, M, N, (float)scale, q.stride(0), k.stride(0), v.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {qs, kr, vr, vt, qt, kt};
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
  const int64_t nh = lse.size(1), dh = dO.size(1) / nh;
  TORCH_CHECK(dO.dim() == 2 && dO.numel() == A * L * nh * dh && L % 16 == 0 && ((nh == 16 && dh == 48) || (nh == 4 && dh == 32)), "dd_perm shapes");
  const at::cuda::CUDAGuard gd(dO.device());
  auto dd = at::empty({A, nh, L}, dO.options()), ddp = at::empty_like(dd), lsep = at::empty_like(dd), dor = at::empty_like(dO), dot = at::empty({nh * dh, A * L}, dO.options());
  auto s = at::cuda::getCurrentCUDAStream();
  auto go = [&](auto kern, int64_t n) {
    const size_t smem = 16 * (n + 1) * sizeof(float);
    C10_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(kern), cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    kern<<<(unsigned)(A * L / 16), 256, smem, s>>>(dO.data_ptr<float>(), O.data_ptr<float>(), lse.data_ptr<float>(), dd.data_ptr<float>(), ddp.data_ptr<float>(),
        lsep.data_ptr<float>(), dor.data_ptr<float>(), dot.data_ptr<float>(), A * L, (int)L);
  };
  if (nh == 16) go(dd_k<16, 48>, 768);
  else go(dd_k<4, 32>, 128);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dd, ddp, lsep, dor, dot};
}

at::Tensor bias_transposed(at::Tensor bias_p, int64_t L) {
  chk(bias_p, at::kFloat, "bias");
  TORCH_CHECK(bias_p.numel() % (L * L) == 0 && L % 8 == 0, "bias_transposed shapes");
  const at::cuda::CUDAGuard gd(bias_p.device());
  auto z = at::empty_like(bias_p);
  const long n = bias_p.numel();
  bias_t_k<<<(unsigned)((n + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(bias_p.data_ptr<float>(), z.data_ptr<float>(), n, (int)L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return z;
}

// fp32 pair [L L, 128 or 256], 16 heads (the fp32 token stack): dpair and the dWf block partials; the LayerNorm statistics are recomputed. ``perm``: db keeps the TF32 core's key-permuted columns
std::vector<at::Tensor> pair_rows_bwd32(at::Tensor z, at::Tensor db, at::Tensor wf, int64_t L, double eps, bool perm) {
  chk(z, at::kFloat, "pair");
  chk(db, at::kFloat, "dbias");
  chk(wf, at::kFloat, "wf");
  const int64_t R = L * L;
  const int64_t dp = wf.size(1);
  const int64_t nh = wf.size(0);
  TORCH_CHECK(((nh == 16 && (dp == 128 || dp == 256)) || (nh == 4 && dp == 16)) && z.numel() == R * dp && db.numel() == nh * R, "pair_rows_bwd32 shapes");
  const at::cuda::CUDAGuard gd(z.device());
  auto dz = at::empty_like(z);
  TORCH_CHECK(R % 32 == 0, "pair_rows_bwd32: rows a multiple of 32");
  const int64_t per_block = dp == 256 ? WPB / 2 : WPB;   // rows (in 32s) a block handles at a time: two warps share a row at d_pair 256
  // few rows (L 128: 16 384): 8-row chunks per warp so that the grid still fills the GPU
  const bool small = dp == 128 && (R / 32 + WPB - 1) / WPB < 2L * nsm_of(z);
  const int64_t ch = small ? 8 : 32;
  const unsigned grid = (unsigned)std::min<int64_t>((R / ch + per_block - 1) / per_block, 2L * nsm_of(z));
  auto acc = at::empty({(int64_t)(dp == 16 ? std::min<int64_t>((R + ABT - 1) / ABT, 4L * nsm_of(z)) : grid), nh * dp}, z.options());
  auto s = at::cuda::getCurrentCUDAStream();
  if (dp == 16)
    atom32_bwd_k<4><<<(unsigned)acc.size(0), ABT, 0, s>>>(
        z.data_ptr<float>(), db.data_ptr<float>(), wf.data_ptr<float>(), dz.data_ptr<float>(), acc.data_ptr<float>(), R, (float)eps, perm ? (int)L : 0);
  else if (dp == 128) {
    if (small)
      bwd_k<128, 16, float, false, 8><<<grid, BT, 0, s>>>(
          z.data_ptr<float>(), db.data_ptr<float>(), wf.data_ptr<float>(), nullptr, dz.data_ptr<float>(), acc.data_ptr<float>(), R, (float)eps, perm ? (int)L : 0);
    else
      bwd_k<128, 16, float, false, 32><<<grid, BT, 0, s>>>(
          z.data_ptr<float>(), db.data_ptr<float>(), wf.data_ptr<float>(), nullptr, dz.data_ptr<float>(), acc.data_ptr<float>(), R, (float)eps, perm ? (int)L : 0);
  }
  else
    bwd256_k<<<grid, BT, 0, s>>>(
        z.data_ptr<float>(), db.data_ptr<float>(), wf.data_ptr<float>(), dz.data_ptr<float>(), acc.data_ptr<float>(), R, (float)eps, perm ? (int)L : 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dz, acc};
}


// fp32 pair [L L, DP] (DP = 128 or 256), 16 heads: bias[h, i, p] (key-permuted for the TF32 core: position p of each group of 8 holds key kperm(p)) =
// sum_c Wf[h, c] LN(pair[i L + key])[c], a masked key = neg. Wf (carries the LayerNorm weight, the head projection and log2 e) lives in shared memory;
// a warp owns a row at a time (channels q * 128 + 4 lane ..), 32 rows' results are staged so the bias is written coalesced along the key.
template <int DP, int NW>
__global__ void __launch_bounds__(NW * 32) bias32_k(const float* __restrict__ Z, const float* __restrict__ W, const bool* __restrict__ MASK,
                                                float* __restrict__ OUT, long R, int L, float eps, float neg) {
  constexpr int DPL = DP / 32, NQ = DP / 128;
  __shared__ __align__(16) float wsm[16 * DP];
  __shared__ float stage[NW][32][17];
  for (int c = threadIdx.x; c < 16 * DP / 4; c += NW * 32) reinterpret_cast<float4*>(wsm)[c] = reinterpret_cast<const float4*>(W)[c];
  __syncthreads();
  const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  const long warp = ((long)blockIdx.x * NW * 32 + threadIdx.x) >> 5, nwarps = (long)gridDim.x * NW;
  for (long base = warp * 32; base < R; base += nwarps * 32) {
    for (int rb = 0; rb < 32; rb += PF) {
      float xb[PF][DPL];
#pragma unroll
      for (int q = 0; q < PF; ++q)
#pragma unroll
        for (int c = 0; c < NQ; ++c) {
          const float4 a = __ldg(reinterpret_cast<const float4*>(Z + (base + rb + q) * DP + c * 128 + lane * 4));
          xb[q][4 * c] = a.x; xb[q][4 * c + 1] = a.y; xb[q][4 * c + 2] = a.z; xb[q][4 * c + 3] = a.w;
        }
#pragma unroll
      for (int q = 0; q < PF; ++q) {
        float x[DPL], v[16], rstd;
#pragma unroll
        for (int i = 0; i < DPL; ++i) x[i] = xb[q][i];
        normalize<DP, DPL>(x, eps, rstd);
#pragma unroll
        for (int h = 0; h < 16; ++h) {
          float d = 0.f;
#pragma unroll
          for (int c = 0; c < NQ; ++c) {
            const float4 w = *reinterpret_cast<const float4*>(wsm + h * DP + c * 128 + lane * 4);
            d += w.x * x[4 * c] + w.y * x[4 * c + 1] + w.z * x[4 * c + 2] + w.w * x[4 * c + 3];
          }
          v[h] = d;
        }
        const float u = reduce_scatter16(v, lane);
        if ((lane & 1) == 0) stage[wp][rb + q][lane >> 1] = u;
      }
    }
    __syncwarp();
    const int j0 = (int)(base % L), kk = kperm(lane);
    const bool drop = MASK != nullptr && !MASK[j0 + kk];
#pragma unroll
    for (int h = 0; h < 16; ++h) OUT[(long)h * R + base + lane] = drop ? neg : stage[wp][kk][h];
    __syncwarp();
  }
}

void pair_bias_rows32(at::Tensor z, at::Tensor wf, at::Tensor out, c10::optional<at::Tensor> mask, int64_t L, double eps, double neg) {
  chk(z, at::kFloat, "pair");
  chk(wf, at::kFloat, "wf");
  chk(out, at::kFloat, "bias");
  const int64_t R = L * L, dp = wf.size(1);
  const int64_t nh = wf.size(0);
  TORCH_CHECK(((nh == 16 && (dp == 128 || dp == 256)) || (nh == 4 && dp == 16)) && z.numel() == R * dp && out.numel() == nh * R && L % 32 == 0, "pair_bias_rows32 shapes");
  if (mask) {
    chk(*mask, at::kBool, "mask");
    TORCH_CHECK(mask->numel() == L, "mask [L]");
  }
  const at::cuda::CUDAGuard gd(z.device());
  const unsigned grid = (unsigned)std::min<int64_t>((R / 32 + WPB - 1) / WPB, 4L * nsm_of(z));
  const bool* mp = mask ? mask->data_ptr<bool>() : nullptr;
  auto s = at::cuda::getCurrentCUDAStream();
  if (dp == 16) {
    const unsigned g = (unsigned)std::min<int64_t>((R + ABT - 1) / ABT, 8L * nsm_of(z));
    atom32_fwd_k<4><<<g, ABT, 0, s>>>(z.data_ptr<float>(), wf.data_ptr<float>(), mp, out.data_ptr<float>(), R, (int)L, (float)eps, (float)neg);
  } else {
    // few rows (L 128): small blocks so that every SM has work
    const bool small = (R / 32 + 7) / 8 < 2L * nsm_of(z);
    const unsigned g = (unsigned)std::min<int64_t>((R / 32 + (small ? 2 : 8) - 1) / (small ? 2 : 8), 4L * nsm_of(z));
    const float* zp = z.data_ptr<float>(); const float* wp = wf.data_ptr<float>(); float* op = out.data_ptr<float>();
    if (dp == 128) { if (small) bias32_k<128, 2><<<g, 64, 0, s>>>(zp, wp, mp, op, R, (int)L, (float)eps, (float)neg); else bias32_k<128, 8><<<g, 256, 0, s>>>(zp, wp, mp, op, R, (int)L, (float)eps, (float)neg); }
    else { if (small) bias32_k<256, 2><<<g, 64, 0, s>>>(zp, wp, mp, op, R, (int)L, (float)eps, (float)neg); else bias32_k<256, 8><<<g, 256, 0, s>>>(zp, wp, mp, op, R, (int)L, (float)eps, (float)neg); }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- out = sigmoid(g) * o (fp32), forward and backward in one pass each; g may be a column slice of a wider matrix (row stride ldg)
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }

__global__ void __launch_bounds__(256) gate_fwd_k(const float* __restrict__ G, const float* __restrict__ O, float* __restrict__ OUT, long n4, int n, long ldg) {
  const int c4 = n / 4;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < n4; i += (long)gridDim.x * 256) {
    const long row = i / c4, c = (i % c4) * 4;
    const float4 g = __ldg(reinterpret_cast<const float4*>(G + row * ldg + c)), o = __ldg(reinterpret_cast<const float4*>(O) + i);
    reinterpret_cast<float4*>(OUT)[i] = make_float4(sigm(g.x) * o.x, sigm(g.y) * o.y, sigm(g.z) * o.z, sigm(g.w) * o.w);
  }
}

__global__ void __launch_bounds__(256) gate_bwd_k(const float* __restrict__ G, const float* __restrict__ O, const float* __restrict__ DY, float* __restrict__ DG, long lddg,
                                                  float* __restrict__ DO, long n4, int n, long ldg) {
  const int c4 = n / 4;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < n4; i += (long)gridDim.x * 256) {
    const long row = i / c4, c = (i % c4) * 4;
    const float4 g = __ldg(reinterpret_cast<const float4*>(G + row * ldg + c)), o = __ldg(reinterpret_cast<const float4*>(O) + i), dy = __ldg(reinterpret_cast<const float4*>(DY) + i);
    const float gv[4] = {g.x, g.y, g.z, g.w}, ov[4] = {o.x, o.y, o.z, o.w}, dv[4] = {dy.x, dy.y, dy.z, dy.w};
    float dg[4], dd[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float s = sigm(gv[e]);
      dd[e] = dv[e] * s;
      dg[e] = dv[e] * ov[e] * s * (1.f - s);
    }
    *reinterpret_cast<float4*>(DG + row * lddg + c) = make_float4(dg[0], dg[1], dg[2], dg[3]);
    reinterpret_cast<float4*>(DO)[i] = make_float4(dd[0], dd[1], dd[2], dd[3]);
  }
}

at::Tensor gate_mul_fwd(at::Tensor g, at::Tensor o) {
  chk(o, at::kFloat, "o");
  TORCH_CHECK(g.is_cuda() && g.scalar_type() == at::kFloat && g.dim() == 2 && g.stride(1) == 1 && g.stride(0) % 4 == 0 && g.sizes() == o.sizes() && o.size(1) % 4 == 0, "gate_mul_fwd shapes");
  const at::cuda::CUDAGuard gd(o.device());
  auto out = at::empty_like(o);
  const long n4 = o.numel() / 4;
  gate_fwd_k<<<(unsigned)std::min<long>((n4 + 255) / 256, 4L * nsm_of(o)), 256, 0, at::cuda::getCurrentCUDAStream()>>>(g.data_ptr<float>(), o.data_ptr<float>(), out.data_ptr<float>(), n4, (int)o.size(1), g.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<at::Tensor> gate_mul_bwd(at::Tensor g, at::Tensor o, at::Tensor dy) {
  chk(o, at::kFloat, "o");
  chk(dy, at::kFloat, "dy");
  TORCH_CHECK(g.is_cuda() && g.scalar_type() == at::kFloat && g.dim() == 2 && g.stride(1) == 1 && g.stride(0) % 4 == 0 && g.sizes() == o.sizes() && o.sizes() == dy.sizes() && o.size(1) % 4 == 0, "gate_mul_bwd shapes");
  const at::cuda::CUDAGuard gd(o.device());
  auto dg = at::empty_like(o), dd = at::empty_like(o);
  const long n4 = o.numel() / 4;
  gate_bwd_k<<<(unsigned)std::min<long>((n4 + 255) / 256, 4L * nsm_of(o)), 256, 0, at::cuda::getCurrentCUDAStream()>>>(g.data_ptr<float>(), o.data_ptr<float>(), dy.data_ptr<float>(),
      dg.data_ptr<float>(), (long)dg.stride(0), dd.data_ptr<float>(), n4, (int)o.size(1), g.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dg, dd};
}

// the backward with d gate written into a column slice of a wider matrix (the packed gradient of the projections); returns d o
at::Tensor gate_mul_bwd_into(at::Tensor g, at::Tensor o, at::Tensor dy, at::Tensor dg) {
  chk(o, at::kFloat, "o");
  chk(dy, at::kFloat, "dy");
  TORCH_CHECK(g.is_cuda() && g.scalar_type() == at::kFloat && g.dim() == 2 && g.stride(1) == 1 && g.stride(0) % 4 == 0 && g.sizes() == o.sizes() && o.sizes() == dy.sizes() && o.size(1) % 4 == 0, "gate_mul_bwd_into shapes");
  TORCH_CHECK(dg.is_cuda() && dg.scalar_type() == at::kFloat && dg.sizes() == o.sizes() && dg.stride(1) == 1 && dg.stride(0) % 4 == 0, "gate_mul_bwd_into: d gate rows");
  const at::cuda::CUDAGuard gd(o.device());
  auto dd = at::empty_like(o);
  const long n4 = o.numel() / 4;
  gate_bwd_k<<<(unsigned)std::min<long>((n4 + 255) / 256, 4L * nsm_of(o)), 256, 0, at::cuda::getCurrentCUDAStream()>>>(g.data_ptr<float>(), o.data_ptr<float>(), dy.data_ptr<float>(),
      dg.data_ptr<float>(), (long)dg.stride(0), dd.data_ptr<float>(), n4, (int)o.size(1), g.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dd;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pair_rows_bwd32", &pair_rows_bwd32);
  m.def("gate_mul_fwd", &gate_mul_fwd);
  m.def("gate_mul_bwd", &gate_mul_bwd);
  m.def("gate_mul_bwd_into", &gate_mul_bwd_into);
  m.def("pair_bias_rows32", &pair_bias_rows32);
  m.def("transpose32", &transpose32);
  m.def("prep_fwd32", &prep_fwd32);
  m.def("dd_perm", &dd_perm);
  m.def("bias_transposed", &bias_transposed);
  m.def("atom_pair_fwd", &atom_pair_fwd);
  m.def("atom_pair_bwd", &atom_pair_bwd);
  m.def("pair_rows_fwd", &pair_rows_fwd);
  m.def("pair_rows_bwd", &pair_rows_bwd);
}
