// ct_rows.cu -- the elementwise passes between the GEMMs of the ConditionedTransition tail's training step (bf16, or fp32 on TF32 GEMMs), sm_90.
//
//   swiglu_fwd   h = silu(a) * b                         ab [M, 2 ND] = [a | b]  ->  h [M, ND]
//   swiglu_bwd   da = dh b silu'(a), db = dh silu(a)     dh [M, ND], ab  ->  dab [M, 2 ND] = [da | db]
//   sigmul_fwd   y = res + sigmoid(sp) * out             (the gated update rounds to the stream dtype before the add, as the unfused ops do)
//   sigmul_bwd   dout = dy g, dscale = dy out g (1 - g)  g = sigmoid(sp)
//
// Every pass is a vectorised grid-stride loop over 16-byte chunks (8 bf16 / 4 fp32): they are bandwidth-bound and read each operand once.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include "tpose.cuh"
#include "row_math.cuh"

namespace {
using bf = __nv_bfloat16;
template <bool F32> struct Vec;
template <> struct Vec<false> {
  static constexpr int N = 8;
  using T = bf;
  __device__ static void ld(const bf* p, float (&v)[8]) {
    const uint4 q = *reinterpret_cast<const uint4*>(p);
    const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) { v[2 * j] = __uint_as_float(w[j] << 16); v[2 * j + 1] = __uint_as_float(w[j] & 0xffff0000u); }
  }
  __device__ static void st(bf* p, const float (&v)[8]) {
    uint32_t w[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      __nv_bfloat162 t = __floats2bfloat162_rn(v[2 * j], v[2 * j + 1]);
      w[j] = *reinterpret_cast<uint32_t*>(&t);
    }
    *reinterpret_cast<uint4*>(p) = make_uint4(w[0], w[1], w[2], w[3]);
  }
  __device__ static float rnd(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
};
template <> struct Vec<true> {
  static constexpr int N = 4;
  using T = float;
  __device__ static void ld(const float* p, float (&v)[4]) {
    const float4 q = *reinterpret_cast<const float4*>(p);
    v[0] = q.x; v[1] = q.y; v[2] = q.z; v[3] = q.w;
  }
  __device__ static void st(float* p, const float (&v)[4]) { *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]); }
  __device__ static float rnd(float x) { return x; }
};
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }
// the fp32 forward passes' sigmoid: __fdividef (2 ulp) instead of the IEEE division -- the value feeds a TF32 GEMM or the fp32 output at ~1e-7
__device__ __forceinline__ float sigm_fast(float x) { return __fdividef(1.f, 1.f + __expf(-x)); }
// (also the fp32 instantiations of the training passes swiglu_bwd, sigmul_fwd, sigmul_bwd(_x): bf16 keeps the IEEE division)
template <bool F32> __device__ __forceinline__ float sgf(float x) { return F32 ? sigm_fast(x) : sigm(x); }
__device__ __forceinline__ float tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return __uint_as_float(r); }

template <bool F32>
__global__ void swiglu_fwd_k(const typename Vec<F32>::T* __restrict__ ab, typename Vec<F32>::T* __restrict__ h, long chunks, int ndc) {
  constexpr int V = Vec<F32>::N;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");   // launched as a programmatic dependent (bf16): the producer's results are visible after this
#endif
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < chunks; i += (long)gridDim.x * blockDim.x) {
    const unsigned q = (unsigned)i, mu = q / (unsigned)ndc;   // row, chunk of the row (32-bit: chunks < 2^32; a 64-bit division per chunk cost ~10 %)
    const long m = mu, j = q - mu * (unsigned)ndc;
    float a[V], b[V], o[V];
    Vec<F32>::ld(ab + m * 2 * ndc * V + j * V, a);
    Vec<F32>::ld(ab + m * 2 * ndc * V + ndc * V + j * V, b);
#pragma unroll
    for (int k = 0; k < V; ++k) o[k] = F32 ? tf32r(a[k] * sigm_fast(a[k]) * b[k]) : a[k] * sigm(a[k]) * b[k];   // fp32 h feeds only TF32 GEMMs
    Vec<F32>::st(h + i * V, o);
  }
}
template <bool F32>
__global__ void swiglu_bwd_k(const typename Vec<F32>::T* __restrict__ dh, const typename Vec<F32>::T* __restrict__ ab, typename Vec<F32>::T* __restrict__ dab,
                             long chunks, int ndc) {
  constexpr int V = Vec<F32>::N;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < chunks; i += (long)gridDim.x * blockDim.x) {
    const unsigned q = (unsigned)i, mu = q / (unsigned)ndc;
    const long m = mu, j = q - mu * (unsigned)ndc;
    float a[V], b[V], g[V], da[V], db[V];
    Vec<F32>::ld(ab + m * 2 * ndc * V + j * V, a);
    Vec<F32>::ld(ab + m * 2 * ndc * V + ndc * V + j * V, b);
    Vec<F32>::ld(dh + i * V, g);
#pragma unroll
    for (int k = 0; k < V; ++k) {
      const float s = (F32 ? sigm_fast(a[k]) : sigm(a[k]));
      da[k] = g[k] * b[k] * s * (1.f + a[k] * (1.f - s));
      db[k] = g[k] * a[k] * s;
    }
    Vec<F32>::st(dab + m * 2 * ndc * V + j * V, da);
    Vec<F32>::st(dab + m * 2 * ndc * V + ndc * V + j * V, db);
  }
}
template <bool F32, bool RES>
__global__ void sigmul_fwd_k(const typename Vec<F32>::T* __restrict__ out, const typename Vec<F32>::T* __restrict__ sp,
                             const typename Vec<F32>::T* __restrict__ res, typename Vec<F32>::T* __restrict__ y, long chunks) {
  constexpr int V = Vec<F32>::N;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");   // launched as a programmatic dependent (bf16): the producer's results are visible after this
#endif
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < chunks; i += (long)gridDim.x * blockDim.x) {
    float o[V], s[V], r[V], v[V];
    Vec<F32>::ld(out + i * V, o);
    Vec<F32>::ld(sp + i * V, s);
    if (RES) Vec<F32>::ld(res + i * V, r);
#pragma unroll
    for (int k = 0; k < V; ++k) v[k] = RES ? Vec<F32>::rnd(sgf<F32>(s[k]) * o[k]) + r[k] : sgf<F32>(s[k]) * o[k];
    Vec<F32>::st(y + i * V, v);
  }
}
// dout = dy g, dscale = dy out g (1 - g); also the block partials of the column sums of dscale (the gate-bias gradient): the block is a whole number
// of rows of chunks, so a thread's column chunk never changes and it accumulates in registers; the row slots are summed in order through shared memory.
template <bool F32>
__global__ void sigmul_bwd_k(const typename Vec<F32>::T* __restrict__ dy, const typename Vec<F32>::T* __restrict__ out, const typename Vec<F32>::T* __restrict__ sp,
                             typename Vec<F32>::T* __restrict__ dout, typename Vec<F32>::T* __restrict__ dsc, float* __restrict__ part, long chunks, int ndc) {
  constexpr int V = Vec<F32>::N;
  extern __shared__ float red[];
  float acc[V];
#pragma unroll
  for (int k = 0; k < V; ++k) acc[k] = 0.f;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < chunks; i += (long)gridDim.x * blockDim.x) {
    float d[V], o[V], s[V], a[V], b[V];
    Vec<F32>::ld(dy + i * V, d);
    Vec<F32>::ld(out + i * V, o);
    Vec<F32>::ld(sp + i * V, s);
#pragma unroll
    for (int k = 0; k < V; ++k) {
      const float g = (F32 ? sigm_fast(s[k]) : sigm(s[k]));
      a[k] = F32 ? tf32r(d[k] * g) : d[k] * g;   // fp32: dout feeds only TF32 GEMMs; on the grid for the tcgen05 dh GEMM (ct_tc GAB), which truncates
      b[k] = Vec<F32>::rnd(d[k] * o[k] * g * (1.f - g));
      acc[k] += b[k];
    }
    Vec<F32>::st(dout + i * V, a);
    Vec<F32>::st(dsc + i * V, b);
  }
  const int slot = threadIdx.x / ndc, col = threadIdx.x % ndc, slots = blockDim.x / ndc;
#pragma unroll
  for (int k = 0; k < V; ++k) red[slot * ndc * V + col * V + k] = acc[k];
  __syncthreads();
  for (int c = threadIdx.x; c < ndc * V; c += blockDim.x) {
    float t = 0.f;
    for (int q = 0; q < slots; ++q) t += red[q * ndc * V + c];
    part[(long)blockIdx.x * ndc * V + c] = t;
  }
}


// ---- fp32 feature-major gradient passes (tpose.cuh): the TF32 weight gradients contract over the rows, so they want the gradient with the rows contiguous.
// Tiles of 32 rows x 128 features: loaded row-major (a warp reads 512 B of a row), transposed through shared memory, written as 128 B rows of [features, ldd].
// swiglu_bwd_t:  dab^T [2 ND, ldd]  (rows f: da, rows ND + f: db)
// sigmul_bwd_t:  dout^T and dscale^T [D, ldd] and the block partials of sum(dscale) over the rows (gate-bias gradient): per-lane accumulators over the
//                tiles a block takes, the warps summed in order through shared memory.
constexpr int TW = 8;   // warps a block
__global__ void __launch_bounds__(32 * TW) swiglu_bwd_t_k(const float* __restrict__ dh, const float* __restrict__ ab, float* __restrict__ dabT, long ldd, long R, int nd) {
  __shared__ float Ta[128 * tpose::LD], Tb[128 * tpose::LD];
  const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  for (long m0 = (long)blockIdx.x * tpose::TR; m0 < R; m0 += (long)gridDim.x * tpose::TR) {
    const int rows = (int)min((long)tpose::TR, R - m0);
    for (int f0 = 0; f0 < nd; f0 += 128) {
      for (int q = 0; q < tpose::TR / TW; ++q) {
        const int rl = q * TW + wp;
        if (rl < rows) {
          const long row = m0 + rl;
          float a[4], b[4], g[4];
          Vec<true>::ld(ab + row * 2 * nd + f0 + 4 * lane, a);
          Vec<true>::ld(ab + row * 2 * nd + nd + f0 + 4 * lane, b);
          Vec<true>::ld(dh + row * nd + f0 + 4 * lane, g);
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const float s = sigm(a[k]);
            tpose::put(Ta, 4 * lane + k, rl, g[k] * b[k] * s * (1.f + a[k] * (1.f - s)));
            tpose::put(Tb, 4 * lane + k, rl, g[k] * a[k] * s);
          }
        }
      }
      __syncthreads();
      tpose::flush(Ta, dabT, ldd, f0, 128, (int)m0, rows, wp, TW, lane);
      tpose::flush(Tb, dabT, ldd, nd + f0, 128, (int)m0, rows, wp, TW, lane);
      __syncthreads();
    }
  }
}
template <int NCH>
__global__ void __launch_bounds__(32 * TW) sigmul_bwd_t_k(const float* __restrict__ dy, const float* __restrict__ out, const float* __restrict__ sp, float* __restrict__ doutT,
                                                         float* __restrict__ dscT, long ldd, float* __restrict__ part, long R) {
  constexpr int D = NCH * 128;
  extern __shared__ float sm[];
  float* Ta = sm;
  float* Tb = sm + 128 * tpose::LD;
  const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  float acc[NCH][4];
#pragma unroll
  for (int c = 0; c < NCH; ++c)
#pragma unroll
    for (int k = 0; k < 4; ++k) acc[c][k] = 0.f;
  for (long m0 = (long)blockIdx.x * tpose::TR; m0 < R; m0 += (long)gridDim.x * tpose::TR) {
    const int rows = (int)min((long)tpose::TR, R - m0);
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      for (int q = 0; q < tpose::TR / TW; ++q) {
        const int rl = q * TW + wp;
        if (rl < rows) {
          const long row = m0 + rl;
          float d[4], o[4], s[4];
          Vec<true>::ld(dy + row * D + 128 * c + 4 * lane, d);
          Vec<true>::ld(out + row * D + 128 * c + 4 * lane, o);
          Vec<true>::ld(sp + row * D + 128 * c + 4 * lane, s);
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const float g = sigm(s[k]);
            const float b = d[k] * o[k] * g * (1.f - g);
            acc[c][k] += b;
            tpose::put(Ta, 4 * lane + k, rl, d[k] * g);
            tpose::put(Tb, 4 * lane + k, rl, b);
          }
        }
      }
      __syncthreads();
      tpose::flush(Ta, doutT, ldd, 128 * c, 128, (int)m0, rows, wp, TW, lane);
      tpose::flush(Tb, dscT, ldd, 128 * c, 128, (int)m0, rows, wp, TW, lane);
      __syncthreads();
    }
  }
  float (*red)[D] = reinterpret_cast<float (*)[D]>(sm);
#pragma unroll
  for (int c = 0; c < NCH; ++c)
#pragma unroll
    for (int k = 0; k < 4; ++k) red[wp][128 * c + 4 * lane + k] = acc[c][k];
  __syncthreads();
  for (int c = threadIdx.x; c < D; c += 32 * TW) {
    float t = 0.f;
#pragma unroll
    for (int w = 0; w < TW; ++w) t += red[w][c];
    part[(long)blockIdx.x * D + c] = t;
  }
}

// sigmul_bwd_x: sigmul_bwd (row-major dout, dscale, the block partials) plus the K-major copy of dout, [D, ldd]: the left operand of dWs = dout^T h that cuBLAS runs at
// ~290 TFLOP/s (with dout read MN-major it is ~185). One more write of dout (56 MB at 18432 x 768) instead of a separate transposing kernel (read + write).
// Grid (row-tile blocks, D / 128): block (x, c) takes the 128 features of chunk c of the 32-row tiles x, x + gridDim.x, ... and writes columns 128 c .. of partial
// row x. (Was one block per row-tile walking every chunk in turn: at a few hundred tiles (6144 rows) a block's serial chain of 32-row tiles bounded the pass,
// 35 us where its traffic is ~17.)
__global__ void __launch_bounds__(32 * TW) sigmul_bwd_x_k(const float* __restrict__ dy, const float* __restrict__ out, const float* __restrict__ sp, float* __restrict__ dout,
                                                         float* __restrict__ dsc, float* __restrict__ doutT, long ldd, float* __restrict__ part, long R) {
  const int D = (int)gridDim.y * 128, c = (int)blockIdx.y;
  __shared__ float Ta[128 * tpose::LD];
  const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (long m0 = (long)blockIdx.x * tpose::TR; m0 < R; m0 += (long)gridDim.x * tpose::TR) {
    const int rows = (int)min((long)tpose::TR, R - m0);
#pragma unroll
    for (int q = 0; q < tpose::TR / TW; ++q) {
      const int rl = q * TW + wp;
      if (rl < rows) {
        const long row = m0 + rl;
        float d[4], o[4], s[4], a[4], b[4];
        Vec<true>::ld(dy + row * D + 128 * c + 4 * lane, d);
        Vec<true>::ld(out + row * D + 128 * c + 4 * lane, o);
        Vec<true>::ld(sp + row * D + 128 * c + 4 * lane, s);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float g = sigm_fast(s[k]);
          a[k] = tf32r(d[k] * g);   // dout feeds only TF32 GEMMs (dWs, dh): on the grid, for the tcgen05 dh GEMM that truncates (ct_tc GAB)
          b[k] = d[k] * o[k] * g * (1.f - g);
          acc[k] += b[k];
          tpose::put(Ta, 4 * lane + k, rl, a[k]);
        }
        Vec<true>::st(dout + row * D + 128 * c + 4 * lane, a);
        Vec<true>::st(dsc + row * D + 128 * c + 4 * lane, b);
      }
    }
    __syncthreads();
    tpose::flush(Ta, doutT, ldd, 128 * c, 128, (int)m0, rows, wp, TW, lane);
    __syncthreads();
  }
  float (*red)[128] = reinterpret_cast<float (*)[128]>(Ta);
#pragma unroll
  for (int k = 0; k < 4; ++k) red[wp][4 * lane + k] = acc[k];
  __syncthreads();
  if (threadIdx.x < 128) {
    float t = 0.f;
#pragma unroll
    for (int w = 0; w < TW; ++w) t += red[w][threadIdx.x];
    part[(long)blockIdx.x * D + 128 * c + threadIdx.x] = t;
  }
}


// ---- fp32 inference with the AdaLN's [scale | bias] projection and the tail's gate projection as ONE GEMM (B200, ct_sm100.ct_inference_full_f32):
//   cond_center:    cc = cond - mean_c (and the row's mean_c, rstd_c)                    LN(cond) wc W^T = rstd_c (cc (W wc)^T)
//   raw = cc [Ws wc; Wb wc; Wsc]^T  (cuBLAS TF32)                                         cond Wsc^T = cc Wsc^T + mean_c rowsum(Wsc)
//   adaln_fold_fwd: y = sigmoid(rstd_c raw_s + bs) * LN(x) + rstd_c raw_b
//   sigmul_fwd_ld:  y = res + sigmoid(raw_g + mean_c rowsum(Wsc) + bsc) * out            (raw_g: columns 2 D .. 3 D of raw)
// The conditioning is centred BEFORE the GEMM (the TF32 operand is cond - mean, as LN(cond) is): folding the mean in after the GEMM instead
// cancels against the TF32 rounding of cond itself (3.4x the error of the PyTorch TF32 statements at a row mean of 10 std, 1.4x at 3).
// One warp a row (D, DC multiples of 128), as adaln_rows.cu's row kernels.
template <int N>
__device__ __forceinline__ void ldrow(const float* __restrict__ p, int lane, float (&v)[N / 32]) {
#pragma unroll
  for (int k = 0; k < N / 128; ++k) {
    const float4 t = *reinterpret_cast<const float4*>(p + 4 * (lane + 32 * k));
    v[4 * k] = t.x; v[4 * k + 1] = t.y; v[4 * k + 2] = t.z; v[4 * k + 3] = t.w;
  }
}
template <int N>
__device__ __forceinline__ void strow(float* __restrict__ p, int lane, const float (&v)[N / 32]) {
#pragma unroll
  for (int k = 0; k < N / 128; ++k) *reinterpret_cast<float4*>(p + 4 * (lane + 32 * k)) = make_float4(v[4 * k], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3]);
}
template <int N>
__device__ __forceinline__ void rowstats(const float (&v)[N / 32], float eps, float& mean, float& rstd) {
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < N / 32; ++i) s += v[i];
  mean = rowmath::warp_sum(s) * (1.f / N);
  float q = 0.f;
#pragma unroll
  for (int i = 0; i < N / 32; ++i) q += (v[i] - mean) * (v[i] - mean);
  rstd = rsqrtf(rowmath::warp_sum(q) * (1.f / N) + eps);
}
template <int DC>
__global__ void __launch_bounds__(256) cond_center_k(const float* __restrict__ C, float* __restrict__ CC, float2* __restrict__ ST, long R, float eps_c) {
  const int lane = threadIdx.x & 31;
  const long row = (long)blockIdx.x * 8 + (threadIdx.x >> 5);
  if (row >= R) return;
  float c[DC / 32], mc, rc;
  ldrow<DC>(C + row * DC, lane, c);
  rowstats<DC>(c, eps_c, mc, rc);
#pragma unroll
  for (int i = 0; i < DC / 32; ++i) c[i] -= mc;
  strow<DC>(CC + row * DC, lane, c);
  if (lane == 0) ST[row] = make_float2(mc, rc);
}
template <int D>
__global__ void __launch_bounds__(256) adaln_fold_fwd_k(const float* __restrict__ X, const float* __restrict__ RAW, long ldraw, const float2* __restrict__ ST,
                                                        const float* __restrict__ BS, float* __restrict__ Y, long R, float eps_x) {
  const int lane = threadIdx.x & 31;
  const long row = (long)blockIdx.x * 8 + (threadIdx.x >> 5);
  if (row >= R) return;
  float x[D / 32], mx, rx, sr[D / 32], br[D / 32], bs[D / 32];
  ldrow<D>(X + row * D, lane, x);                  // every load issued before the statistics' shuffles
  ldrow<D>(RAW + row * ldraw, lane, sr);
  ldrow<D>(RAW + row * ldraw + D, lane, br);
  ldrow<D>(BS, lane, bs);
  const float rc = ST[row].y;
  rowstats<D>(x, eps_x, mx, rx);
#pragma unroll
  for (int i = 0; i < D / 32; ++i) x[i] = tf32r(rowmath::modulate(x[i], mx, rx, sigm_fast(rc * sr[i] + bs[i]), rc * br[i]));   // a TF32 GEMM operand
  strow<D>(Y + row * D, lane, x);
}
__global__ void sigmul_fwd_ld_k(const float* __restrict__ out, const float* __restrict__ raw, long ldraw, long goff, const float2* __restrict__ ST,
                                const float* __restrict__ csg, const float* __restrict__ bsc, const float* __restrict__ res, float* __restrict__ y, long chunks,
                                int ndc) {
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < chunks; i += (long)gridDim.x * blockDim.x) {
    const unsigned q = (unsigned)i, mu = q / (unsigned)ndc, j = q - mu * (unsigned)ndc;
    float o[4], g[4], b[4], cs[4], r[4], v[4];
    Vec<true>::ld(out + i * 4, o);
    Vec<true>::ld(raw + (long)mu * ldraw + goff + 4 * j, g);
    Vec<true>::ld(bsc + 4 * j, b);
    Vec<true>::ld(csg + 4 * j, cs);
    if (res) Vec<true>::ld(res + i * 4, r);
    const float mc = ST[mu].x;
#pragma unroll
    for (int k = 0; k < 4; ++k) v[k] = sigm_fast(g[k] + mc * cs[k] + b[k]) * o[k] + (res ? r[k] : 0.f);
    Vec<true>::st(y + i * 4, v);
  }
}

}  // namespace
namespace {
cudaStream_t S() { return at::cuda::getCurrentCUDAStream(); }
// one chunk a thread (the grid-stride loops then take one trip): a capped grid left the passes at a few thousand rows a second, partial trip
// (swiglu_fwd at 3840 x 1536: 10.0 us, the portable pass 8.5)
unsigned grid_of(long chunks) { return (unsigned)std::max<long>(1, std::min<long>((chunks + 255) / 256, 1L << 30)); }
// a launch with programmatic stream serialization (PDL): the grid may be scheduled before the previous kernel on the stream has finished
// (its own griddepcontrol.wait holds it until that kernel's results are visible), so the launch latency overlaps that kernel's tail. Used
// for the bf16 row passes that follow a cuBLAS GEMM.
template <typename... KArgs, typename... Args>
void launch_pdl(void (*kernel)(KArgs...), unsigned grid, unsigned block, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(block);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...));
}
bool is_f32(const at::Tensor& t) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && (t.scalar_type() == at::kBFloat16 || t.scalar_type() == at::kFloat), "contiguous bf16 / fp32 CUDA tensor");
  return t.scalar_type() == at::kFloat;
}
template <typename T> const T* P(const at::Tensor& t) { return reinterpret_cast<const T*>(t.data_ptr()); }
template <typename T> T* PM(at::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }
}  // namespace

at::Tensor swiglu_fwd(at::Tensor ab) {
  const bool f = is_f32(ab);
  const int V = f ? 4 : 8;
  const long M = ab.size(0), nd = ab.size(1) / 2;
  TORCH_CHECK(nd % V == 0, "ND % 8");
  const at::cuda::CUDAGuard g(ab.device());
  auto h = at::empty({M, nd}, ab.options());
  const long chunks = M * nd / V;
  TORCH_CHECK(chunks < (1L << 32), "swiglu: rows x width");
  if (f) swiglu_fwd_k<true><<<grid_of(chunks), 256, 0, S()>>>(P<float>(ab), PM<float>(h), chunks, (int)(nd / V));
  else launch_pdl(swiglu_fwd_k<false>, grid_of(chunks), 256, P<bf>(ab), PM<bf>(h), chunks, (int)(nd / V));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return h;
}
at::Tensor swiglu_bwd(at::Tensor dh, at::Tensor ab) {
  const bool f = is_f32(ab);
  is_f32(dh);
  const int V = f ? 4 : 8;
  const long M = ab.size(0), nd = ab.size(1) / 2;
  TORCH_CHECK(dh.size(0) == M && dh.size(1) == nd && nd % V == 0 && dh.scalar_type() == ab.scalar_type(), "shapes");
  const at::cuda::CUDAGuard g(ab.device());
  auto dab = at::empty_like(ab);
  const long chunks = M * nd / V;
  TORCH_CHECK(chunks < (1L << 32), "swiglu: rows x width");
  if (f) swiglu_bwd_k<true><<<grid_of(chunks), 256, 0, S()>>>(P<float>(dh), P<float>(ab), PM<float>(dab), chunks, (int)(nd / V));
  else swiglu_bwd_k<false><<<grid_of(chunks), 256, 0, S()>>>(P<bf>(dh), P<bf>(ab), PM<bf>(dab), chunks, (int)(nd / V));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dab;
}
at::Tensor sigmul_fwd(at::Tensor out, at::Tensor sp, c10::optional<at::Tensor> res) {
  const bool f = is_f32(out);
  is_f32(sp);
  const int V = f ? 4 : 8;
  TORCH_CHECK(out.numel() % V == 0 && sp.sizes() == out.sizes() && sp.scalar_type() == out.scalar_type(), "shapes");
  if (res) TORCH_CHECK(res->sizes() == out.sizes() && res->scalar_type() == out.scalar_type() && res->is_contiguous(), "residual");
  const at::cuda::CUDAGuard g(out.device());
  auto y = at::empty_like(out);
  const long chunks = out.numel() / V;
  const bool r = res.has_value();
#define SF(F, T, R) sigmul_fwd_k<F, R><<<grid_of(chunks), 256, 0, S()>>>(P<T>(out), P<T>(sp), r ? P<T>(*res) : nullptr, PM<T>(y), chunks)
  if (f) { if (r) SF(true, float, true); else SF(true, float, false); }
  else if (r) launch_pdl(sigmul_fwd_k<false, true>, grid_of(chunks), 256, P<bf>(out), P<bf>(sp), P<bf>(*res), PM<bf>(y), chunks);
  else launch_pdl(sigmul_fwd_k<false, false>, grid_of(chunks), 256, P<bf>(out), P<bf>(sp), (const bf*)nullptr, PM<bf>(y), chunks);
#undef SF
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
std::vector<at::Tensor> sigmul_bwd(at::Tensor dy, at::Tensor out, at::Tensor sp, int64_t grid_blocks) {
  const bool f = is_f32(dy);
  is_f32(out);
  is_f32(sp);
  const int V = f ? 4 : 8;
  TORCH_CHECK(dy.numel() % V == 0 && out.sizes() == dy.sizes() && sp.sizes() == dy.sizes(), "shapes");
  const at::cuda::CUDAGuard g(dy.device());
  auto dout = at::empty_like(dy), dsc = at::empty_like(dy);
  const long chunks = dy.numel() / V;
  const int ndc = (int)(dy.size(-1) / V);
  // bf16: ~512 threads a block (the grid is capped at a few blocks per SM, and one chunk per thread in flight left the pass latency-bound:
  // 32 -> ~24 us at [18432, 768]); fp32 keeps 128
  const int threads = ndc * std::max(1, (f ? 128 : 512) / ndc);
  TORCH_CHECK(threads <= 1024, "width");
  const unsigned grid = (unsigned)grid_blocks;
  auto part = at::empty({(long)grid, dy.size(-1)}, dy.options().dtype(at::kFloat));
  const size_t shm = (size_t)threads / ndc * ndc * V * sizeof(float);
  if (f) sigmul_bwd_k<true><<<grid, threads, shm, S()>>>(P<float>(dy), P<float>(out), P<float>(sp), PM<float>(dout), PM<float>(dsc), PM<float>(part), chunks, ndc);
  else sigmul_bwd_k<false><<<grid, threads, shm, S()>>>(P<bf>(dy), P<bf>(out), P<bf>(sp), PM<bf>(dout), PM<bf>(dsc), PM<float>(part), chunks, ndc);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dout, dsc, part};
}

// fp32 feature-major backward passes: [features, ldd] outputs, ldd = rows rounded up to 32 (a GEMM operand is its [:, :rows] view)
at::Tensor swiglu_bwd_t(at::Tensor dh, at::Tensor ab) {
  TORCH_CHECK(is_f32(ab) && is_f32(dh), "fp32 only");
  const long M = ab.size(0), nd = ab.size(1) / 2;
  TORCH_CHECK(dh.size(0) == M && dh.size(1) == nd && nd % 128 == 0, "shapes: ND % 128");
  const at::cuda::CUDAGuard g(ab.device());
  const long ldd = (M + 31) / 32 * 32;
  auto dabT = at::empty({2 * nd, ldd}, ab.options());
  const unsigned grid = (unsigned)std::min<long>((M + 31) / 32, 132L * 6);
  swiglu_bwd_t_k<<<grid, 32 * TW, 0, S()>>>(P<float>(dh), P<float>(ab), PM<float>(dabT), ldd, M, (int)nd);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dabT;
}
std::vector<at::Tensor> sigmul_bwd_t(at::Tensor dy, at::Tensor out, at::Tensor sp, int64_t grid_blocks) {
  TORCH_CHECK(is_f32(dy) && is_f32(out) && is_f32(sp), "fp32 only");
  const long D = dy.size(-1), M = dy.numel() / D;
  TORCH_CHECK(D % 128 == 0 && D <= 1024 && out.sizes() == dy.sizes() && sp.sizes() == dy.sizes(), "shapes: D % 128, <= 1024");
  const at::cuda::CUDAGuard g(dy.device());
  const long ldd = (M + 31) / 32 * 32;
  auto doutT = at::empty({D, ldd}, dy.options()), dscT = at::empty({D, ldd}, dy.options());
  const unsigned grid = (unsigned)std::min<long>(grid_blocks, (M + 31) / 32);
  auto part = at::empty({(long)grid, D}, dy.options());
  const size_t shm = std::max<size_t>(2 * 128 * tpose::LD * sizeof(float), (size_t)TW * D * sizeof(float));
#define SBT(NCH) sigmul_bwd_t_k<NCH><<<grid, 32 * TW, shm, S()>>>(P<float>(dy), P<float>(out), P<float>(sp), PM<float>(doutT), PM<float>(dscT), ldd, PM<float>(part), M)
  switch (D / 128) {
    case 1: SBT(1); break; case 2: SBT(2); break; case 3: SBT(3); break; case 4: SBT(4); break;
    case 5: SBT(5); break; case 6: SBT(6); break; case 7: SBT(7); break; default: SBT(8); break;
  }
#undef SBT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {doutT, dscT, part};
}

// (dout, dsc, part, doutT [D, ldd]); `grid_blocks` as sigmul_bwd_t (the partials have one row per block)
std::vector<at::Tensor> sigmul_bwd_x(at::Tensor dy, at::Tensor out, at::Tensor sp, int64_t grid_blocks) {
  TORCH_CHECK(is_f32(dy) && is_f32(out) && is_f32(sp), "fp32 only");
  const long D = dy.size(-1), M = dy.numel() / D;
  TORCH_CHECK(D % 128 == 0 && D <= 1024 && out.sizes() == dy.sizes() && sp.sizes() == dy.sizes(), "shapes: D % 128, <= 1024");
  const at::cuda::CUDAGuard g(dy.device());
  const long ldd = (M + 31) / 32 * 32;
  auto dout = at::empty_like(dy), dsc = at::empty_like(dy), doutT = at::empty({D, ldd}, dy.options());
  const unsigned grid = (unsigned)std::min<long>(grid_blocks, (M + 31) / 32);
  auto part = at::empty({(long)grid, D}, dy.options());
  sigmul_bwd_x_k<<<dim3(grid, (unsigned)(D / 128)), 32 * TW, 0, S()>>>(P<float>(dy), P<float>(out), P<float>(sp), PM<float>(dout), PM<float>(dsc), PM<float>(doutT), ldd,
                                                             PM<float>(part), M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dout, dsc, part, doutT};
}

std::vector<at::Tensor> cond_center(at::Tensor cond, double eps_c) {
  TORCH_CHECK(is_f32(cond), "fp32");
  const long R = cond.size(0), DC = cond.size(1);
  const at::cuda::CUDAGuard g(cond.device());
  auto cc = at::empty_like(cond);
  auto st = at::empty({R, 2}, cond.options());
  const unsigned grid = (unsigned)((R + 7) / 8);
#define CC_(N) cond_center_k<N><<<grid, 256, 0, S()>>>(P<float>(cond), PM<float>(cc), reinterpret_cast<float2*>(st.data_ptr()), R, (float)eps_c)
  switch (DC) { case 256: CC_(256); break; case 384: CC_(384); break; case 512: CC_(512); break; case 768: CC_(768); break;
                default: TORCH_CHECK(false, "cond_center: width"); }
#undef CC_
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {cc, st};
}
at::Tensor adaln_fold_fwd(at::Tensor x, at::Tensor raw, at::Tensor st, at::Tensor bs, double eps_x) {
  TORCH_CHECK(is_f32(x) && is_f32(st) && is_f32(bs) && raw.scalar_type() == at::kFloat && raw.stride(1) == 1, "fp32");
  const long R = x.size(0), D = x.size(1);
  TORCH_CHECK(raw.size(0) == R && raw.size(1) >= 2 * D && bs.numel() == D && st.size(0) == R, "shapes");
  const at::cuda::CUDAGuard g(x.device());
  auto y = at::empty_like(x);
  const unsigned grid = (unsigned)((R + 7) / 8);
#define AF(N) adaln_fold_fwd_k<N><<<grid, 256, 0, S()>>>(P<float>(x), P<float>(raw), raw.stride(0), reinterpret_cast<const float2*>(st.data_ptr()), P<float>(bs), PM<float>(y), R, (float)eps_x)
  switch (D) { case 256: AF(256); break; case 384: AF(384); break; case 512: AF(512); break; case 768: AF(768); break;
               default: TORCH_CHECK(false, "adaln_fold_fwd: width"); }
#undef AF
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
at::Tensor sigmul_fwd_ld(at::Tensor out, at::Tensor raw, int64_t goff, at::Tensor st, at::Tensor csg, at::Tensor bsc, c10::optional<at::Tensor> res) {
  TORCH_CHECK(is_f32(out) && is_f32(bsc) && is_f32(csg) && is_f32(st) && raw.scalar_type() == at::kFloat && raw.stride(1) == 1, "fp32");
  const long D = out.size(1);
  TORCH_CHECK(D % 4 == 0 && bsc.numel() == D && csg.numel() == D && raw.size(0) == out.size(0) && goff + D <= raw.size(1), "shapes");
  if (res) TORCH_CHECK(is_f32(*res) && res->sizes() == out.sizes(), "res");
  const at::cuda::CUDAGuard g(out.device());
  auto y = at::empty_like(out);
  const long chunks = out.numel() / 4;
  sigmul_fwd_ld_k<<<grid_of(chunks), 256, 0, S()>>>(P<float>(out), P<float>(raw), raw.stride(0), goff, reinterpret_cast<const float2*>(st.data_ptr()),
                                                   P<float>(csg), P<float>(bsc), res ? P<float>(*res) : nullptr, PM<float>(y), chunks, (int)(D / 4));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
// ---- t^T of a bf16 matrix [R, C] (32 x 32 tiles through shared memory): Ws^T for the B200 bf16 backward's fused dh GEMM (ct_tg gate7), whose
// B operand is K-major
__global__ void __launch_bounds__(256) transpose_bf16_k(const bf* __restrict__ src, bf* __restrict__ dst, int R, int C) {
  __shared__ bf tile[32][34];
  const int c0 = blockIdx.x * 32, r0 = blockIdx.y * 32, tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
#pragma unroll
  for (int k = 0; k < 32; k += 8) {
    const int r = r0 + ty + k, c = c0 + tx;
    if (r < R && c < C) tile[ty + k][tx] = src[(long)r * C + c];
  }
  __syncthreads();
#pragma unroll
  for (int k = 0; k < 32; k += 8) {
    const int c = c0 + ty + k, r = r0 + tx;
    if (r < R && c < C) dst[(long)c * R + r] = tile[tx][ty + k];
  }
}
at::Tensor transpose_bf16(at::Tensor t) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.dim() == 2 && t.is_contiguous(), "transpose_bf16: bf16 [R, C] contiguous");
  const int R = (int)t.size(0), C = (int)t.size(1);
  const at::cuda::CUDAGuard g(t.device());
  auto o = at::empty({C, R}, t.options());
  transpose_bf16_k<<<dim3((unsigned)((C + 31) / 32), (unsigned)((R + 31) / 32)), 256, 0, S()>>>(P<bf>(t), PM<bf>(o), R, C);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return o;
}

// ---- the row statistics of the B200 bf16 inference head (ct_head_sm100.cu): ST[r] = (rstd_x, mean_x rstd_x, rstd_c, mean_c rstd_c) of x [R, NX] and
// cond [R, NC] (two passes over each row held in registers, fp32), one warp per row. It lets its dependent launch at once
// (griddepcontrol.launch_dependents first): the head kernel runs as its programmatic dependent and waits for these only in its epilogue.
__device__ __forceinline__ void row_stats(const bf* __restrict__ p, int N, int lane, float eps, float& rstd, float& mean) {
  float v[4][8];
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int c = lane + 32 * j;
    if (8 * c < N) {
      const uint4 q = *reinterpret_cast<const uint4*>(p + 8 * c);
      const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) { v[j][2 * e] = __uint_as_float(w[e] << 16); v[j][2 * e + 1] = __uint_as_float(w[e] & 0xffff0000u); }
#pragma unroll
      for (int e = 0; e < 8; ++e) s += v[j][e];
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  mean = s * (1.f / N);
  float q2 = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    if (8 * (lane + 32 * j) < N) {
#pragma unroll
      for (int e = 0; e < 8; ++e) q2 += (v[j][e] - mean) * (v[j][e] - mean);
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) q2 += __shfl_xor_sync(0xffffffffu, q2, o);
  rstd = rsqrtf(q2 * (1.f / N) + eps);
}
__global__ void __launch_bounds__(256) head_stats_k(const bf* __restrict__ X, const bf* __restrict__ C, float4* __restrict__ ST, long R, int NX, int NC,
                                                    float eps_x, float eps_c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
  const int lane = threadIdx.x & 31;
  const long row = (long)blockIdx.x * 8 + (threadIdx.x >> 5);
  if (row >= R) return;
  float rx, mx, rc, mc;
  row_stats(X + row * NX, NX, lane, eps_x, rx, mx);
  row_stats(C + row * NC, NC, lane, eps_c, rc, mc);
  if (lane == 0) ST[row] = make_float4(rx, mx * rx, rc, mc * rc);
}
at::Tensor head_stats(at::Tensor x, at::Tensor cond, double eps_x, double eps_c) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && cond.scalar_type() == at::kBFloat16 && x.dim() == 2 && cond.dim() == 2 &&
              x.is_contiguous() && cond.is_contiguous() && x.size(0) == cond.size(0), "head_stats: bf16 [R, NX], [R, NC] contiguous");
  const long R = x.size(0);
  const int NX = (int)x.size(1), NC = (int)cond.size(1);
  TORCH_CHECK(NX % 8 == 0 && NX <= 1024 && NC % 8 == 0 && NC <= 1024, "head_stats: widths multiples of 8 up to 1024");
  const at::cuda::CUDAGuard g(x.device());
  auto st = at::empty({R, 4}, x.options().dtype(at::kFloat));
  if (R > 0)
    head_stats_k<<<(unsigned)((R + 7) / 8), 256, 0, S()>>>(P<bf>(x), P<bf>(cond), reinterpret_cast<float4*>(st.data_ptr()), R, NX, NC, (float)eps_x,
                                                           (float)eps_c);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return st;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cond_center", &cond_center);
  m.def("adaln_fold_fwd", &adaln_fold_fwd);
  m.def("sigmul_fwd_ld", &sigmul_fwd_ld);
  m.def("swiglu_fwd", &swiglu_fwd);
  m.def("swiglu_bwd", &swiglu_bwd);
  m.def("sigmul_fwd", &sigmul_fwd);
  m.def("sigmul_bwd", &sigmul_bwd);
  m.def("swiglu_bwd_t", &swiglu_bwd_t);
  m.def("sigmul_bwd_t", &sigmul_bwd_t);
  m.def("sigmul_bwd_x", &sigmul_bwd_x);
  m.def("transpose_bf16", &transpose_bf16);
  m.def("head_stats", &head_stats);
}
