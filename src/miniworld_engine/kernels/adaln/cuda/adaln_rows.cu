// Row kernels of the adaptive LayerNorm  y = sigmoid(scale) * LN(x) + bias,  [scale | bias] = LN_c(cond) [Ws | Wb]^T + [bs | 0],
// bf16 activations, fp32 statistics, one warp per row (rows are 128 .. 1024 wide; two bf16 per lane per load).
//
//   ln_cond_fwd   cn = LN(cond) * wc  (bf16) and the row statistics             -> the GEMM input of the [scale | bias] projection
//   adaln_fwd     y = sigmoid(sb[:, :N]) * LN(x) + sb[:, N:]  and x's statistics
//   adaln_bwd     dy -> dsb = [dy * LN(x) * s (1 - s) | dy] (bf16, the GEMM input of both weight gradients and of dcn), dx (LayerNorm
//                 backward of dy * s) and the block partials of the scale-bias gradient (persistent grid: no atomics, fixed order)
//   ln_cond_bwd   dcn -> dcond (LayerNorm backward through wc) and the block partials of dwc
//   pack          Wcat [2N, Nc] = [Ws | Wb] and bcat = [bs | 0] in bf16 (one launch; fp32 or bf16 weights)
//
// The PyTorch leg of the same computation runs the LayerNorms in fp32 (casts, two statistics passes, an fp32 elementwise chain and a cast
// back): ten kernels and ~4x the traffic of the three row kernels here.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include "row_math.cuh"
#include "tpose.cuh"

namespace {
using bf = __nv_bfloat16;
using bf2 = __nv_bfloat162;
constexpr int WPB = 8;  // warps per block

__device__ __forceinline__ float tf32_rn(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return __uint_as_float(r); }
__device__ __forceinline__ float wsum(float v) { return rowmath::warp_sum(v); }
__device__ __forceinline__ float sigm(float x) { return rowmath::sigmoid(x); }

// a row of N elements held as N / 32 floats per lane in chunks of four consecutive elements (8 B bf16 / 16 B fp32 per lane and load, a warp reads 256 B / 512 B):
// element 4 (lane + 32 k) + j is v[4 k + j]
__device__ __forceinline__ float4 ld4(const bf* p) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  const float2 a = __bfloat1622float2(*reinterpret_cast<const bf2*>(&u.x)), b = __bfloat1622float2(*reinterpret_cast<const bf2*>(&u.y));
  return make_float4(a.x, a.y, b.x, b.y);
}
__device__ __forceinline__ float4 ld4(const float* p) { return *reinterpret_cast<const float4*>(p); }
__device__ __forceinline__ void st4(bf* p, float4 v) {
  const bf2 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
  uint2 u;
  u.x = *reinterpret_cast<const unsigned*>(&a);
  u.y = *reinterpret_cast<const unsigned*>(&b);
  *reinterpret_cast<uint2*>(p) = u;
}
__device__ __forceinline__ void st4(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }
template <int N, typename T>
__device__ __forceinline__ void ld_row(const T* __restrict__ p, int lane, float (&v)[N / 32]) {
#pragma unroll
  for (int k = 0; k < N / 128; ++k) {
    const float4 t = ld4(p + 4 * (lane + 32 * k));
    v[4 * k] = t.x;
    v[4 * k + 1] = t.y;
    v[4 * k + 2] = t.z;
    v[4 * k + 3] = t.w;
  }
}
template <int N, typename T>
__device__ __forceinline__ void st_row(T* __restrict__ p, int lane, const float (&v)[N / 32]) {
#pragma unroll
  for (int k = 0; k < N / 128; ++k) st4(p + 4 * (lane + 32 * k), make_float4(v[4 * k], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3]));
}
template <int N>
__device__ __forceinline__ void stats(const float (&v)[N / 32], float eps, float& mean, float& rstd) {
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < N / 32; ++i) s += v[i];
  mean = wsum(s) * (1.f / N);
  float q = 0.f;
#pragma unroll
  for (int i = 0; i < N / 32; ++i) q += (v[i] - mean) * (v[i] - mean);
  rstd = rsqrtf(wsum(q) * (1.f / N) + eps);
}

template <int N, typename T>
__global__ void __launch_bounds__(32 * WPB) ln_cond_fwd_k(const T* __restrict__ C, const float* __restrict__ W, T* __restrict__ CN,
                                                          float2* __restrict__ ST, long R, float eps) {
  const int lane = threadIdx.x & 31;
  const long row = (long)blockIdx.x * WPB + (threadIdx.x >> 5);
  if (row >= R) return;
  float v[N / 32], mean, rstd;
  ld_row<N>(C + row * N, lane, v);
  stats<N>(v, eps, mean, rstd);
#pragma unroll
  for (int k = 0; k < N / 128; ++k) {
    const float4 w = *reinterpret_cast<const float4*>(W + 4 * (lane + 32 * k));
    v[4 * k] = (v[4 * k] - mean) * rstd * w.x;
    v[4 * k + 1] = (v[4 * k + 1] - mean) * rstd * w.y;
    v[4 * k + 2] = (v[4 * k + 2] - mean) * rstd * w.z;
    v[4 * k + 3] = (v[4 * k + 3] - mean) * rstd * w.w;
  }
  st_row<N>(CN + row * N, lane, v);
  if (lane == 0) ST[row] = make_float2(mean, rstd);
}

// ln_cond_fwd with the K-major copy of its output: CNT [N, ldd] (row f: feature f of LN(cond) * wc, one column per activation row), written through shared-memory tiles of
// 32 rows (tpose.cuh). The fp32 weight gradient dW = dsb^T cn then runs as G^T (cnT)^T, the one TF32 layout cuBLAS runs at ~300 TFLOP/s for a left operand it must read MN-major;
// the copy costs one more write of cn (a separate transposing kernel read it again as well). Same arithmetic as ln_cond_fwd_k.
template <int N>
__global__ void __launch_bounds__(32 * WPB) ln_cond_fwd_t_k(const float* __restrict__ C, const float* __restrict__ W, float* __restrict__ CN, float2* __restrict__ ST,
                                                            float* __restrict__ CNT, long ldd, long R, float eps) {
  extern __shared__ float Ts[];
  const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  const long m0 = (long)blockIdx.x * tpose::TR;
  const int rows = (int)min((long)tpose::TR, R - m0);
#pragma unroll
  for (int q = 0; q < tpose::TR / WPB; ++q) {
    const int rl = q * WPB + wp;
    if (rl < rows) {
      const long row = m0 + rl;
      float v[N / 32], mean, rstd;
      ld_row<N>(C + row * N, lane, v);
      stats<N>(v, eps, mean, rstd);
#pragma unroll
      for (int k = 0; k < N / 128; ++k) {
        const float4 w = *reinterpret_cast<const float4*>(W + 4 * (lane + 32 * k));
        v[4 * k] = (v[4 * k] - mean) * rstd * w.x;
        v[4 * k + 1] = (v[4 * k + 1] - mean) * rstd * w.y;
        v[4 * k + 2] = (v[4 * k + 2] - mean) * rstd * w.z;
        v[4 * k + 3] = (v[4 * k + 3] - mean) * rstd * w.w;
      }
      st_row<N>(CN + row * N, lane, v);
      if (lane == 0) ST[row] = make_float2(mean, rstd);
#pragma unroll
      for (int k = 0; k < N / 128; ++k)
#pragma unroll
        for (int j = 0; j < 4; ++j) tpose::put(Ts, 4 * (lane + 32 * k) + j, rl, v[4 * k + j]);
    }
  }
  __syncthreads();
  tpose::flush(Ts, CNT, ldd, 0, N, (int)m0, rows, wp, WPB, lane);
}

template <int N, typename T>
__global__ void __launch_bounds__(32 * WPB) adaln_fwd_k(const T* __restrict__ X, const T* __restrict__ SB, T* __restrict__ Y,
                                                        float2* __restrict__ ST, long R, float eps, bool rna) {
  const int lane = threadIdx.x & 31;
  const long row = (long)blockIdx.x * WPB + (threadIdx.x >> 5);
  if (row >= R) return;
  float x[N / 32], sp[N / 32], bb[N / 32], mean, rstd;
  ld_row<N>(X + row * N, lane, x);
  ld_row<N>(SB + row * 2 * N, lane, sp);
  ld_row<N>(SB + row * 2 * N + N, lane, bb);
  stats<N>(x, eps, mean, rstd);
#pragma unroll
  for (int i = 0; i < N / 32; ++i) x[i] = rowmath::modulate(x[i], mean, rstd, sigm(sp[i]), bb[i]);
  if (rna) {   // fp32 output whose consumer is a TF32 MMA that truncates its operands (the B200 ConditionedTransition tail): on the TF32 grid
#pragma unroll
    for (int i = 0; i < N / 32; ++i) x[i] = tf32_rn(x[i]);
  }
  st_row<N>(Y + row * N, lane, x);
  if (lane == 0) ST[row] = make_float2(mean, rstd);
}

template <int N, typename T>
__global__ void __launch_bounds__(32 * WPB, 2) adaln_bwd_k(const T* __restrict__ DY, const T* __restrict__ X, const T* __restrict__ SB,
                                                        const float2* __restrict__ ST, T* __restrict__ DSB, T* __restrict__ DX,
                                                        float* __restrict__ PB, long R, const T* __restrict__ ADD, bool RNA) {
  __shared__ float red[WPB][N];
  const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  float part[N / 32];
#pragma unroll
  for (int i = 0; i < N / 32; ++i) part[i] = 0.f;
  for (long row = (long)blockIdx.x * WPB + wp; row < R; row += (long)gridDim.x * WPB) {
    float dy[N / 32], x[N / 32], sp[N / 32];
    ld_row<N>(DY + row * N, lane, dy);
    ld_row<N>(X + row * N, lane, x);
    ld_row<N>(SB + row * 2 * N, lane, sp);
    const float2 s2 = ST[row];
    float g[N / 32], dsp[N / 32], mg = 0.f, mgx = 0.f;
#pragma unroll
    for (int i = 0; i < N / 32; ++i) {
      const float xh = (x[i] - s2.x) * s2.y, s = sigm(sp[i]);
      g[i] = dy[i] * s;
      dsp[i] = dy[i] * xh * s * (1.f - s);
      x[i] = xh;
      mg += g[i];
      mgx += g[i] * xh;
      part[i] += dsp[i];
    }
    mg = wsum(mg) * (1.f / N);
    mgx = wsum(mgx) * (1.f / N);
    float dx[N / 32];
#pragma unroll
    for (int i = 0; i < N / 32; ++i) dx[i] = s2.y * (g[i] - mg - x[i] * mgx);
    if (ADD) {   // the gradient of the stream the caller adds x to: joins dx in fp32, rounded once
      float ad[N / 32];
      ld_row<N>(ADD + row * N, lane, ad);
#pragma unroll
      for (int i = 0; i < N / 32; ++i) dx[i] += ad[i];
    }
    st_row<N>(DX + row * N, lane, dx);
    if (RNA && sizeof(T) == 4) {   // the next consumer is a TF32 tensor-core GEMM: hand it the nearest-rounded operand (a TMA-fed fp32 operand is truncated)
#pragma unroll
      for (int i = 0; i < N / 32; ++i) { dsp[i] = tf32_rn(dsp[i]); dy[i] = tf32_rn(dy[i]); }
    }
    st_row<N>(DSB + row * 2 * N, lane, dsp);
    st_row<N>(DSB + row * 2 * N + N, lane, dy);
  }
#pragma unroll
  for (int k = 0; k < N / 128; ++k)
#pragma unroll
    for (int j = 0; j < 4; ++j) red[wp][4 * (lane + 32 * k) + j] = part[4 * k + j];
  __syncthreads();
  for (int c = threadIdx.x; c < N; c += 32 * WPB) {
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < WPB; ++w) v += red[w][c];
    PB[(long)blockIdx.x * N + c] = v;
  }
}

// fp32 feature-major variant of adaln_bwd_k (the TF32 weight-gradient GEMM wants K = rows contiguous, see tpose.cuh): the gradient D = [dscale ; dy]^T is
// written as [2 N, ldd] (row f: feature f of dscale, row N + f: feature f of dy, one column per activation row), tiles of 32 rows through shared memory.
// Same arithmetic and the same fixed-order partials as adaln_bwd_k; dsb [M, 2N] is never materialised. 256 threads; dynamic shared memory N * 33 floats.
template <int N>
__global__ void __launch_bounds__(32 * WPB, 2) adaln_bwd_t_k(const float* __restrict__ DY, const float* __restrict__ X, const float* __restrict__ SB,
                                                           const float2* __restrict__ ST, float* __restrict__ DX, float* __restrict__ D, long ldd,
                                                           float* __restrict__ PB, long R, const float* __restrict__ ADD) {
  extern __shared__ float Ts[];
  const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  float part[N / 32];
#pragma unroll
  for (int i = 0; i < N / 32; ++i) part[i] = 0.f;
  for (long m0 = (long)blockIdx.x * tpose::TR; m0 < R; m0 += (long)gridDim.x * tpose::TR) {
    const int rows = (int)min((long)tpose::TR, R - m0);
    for (int q = 0; q < tpose::TR / WPB; ++q) {
      const int rl = q * WPB + wp;
      if (rl < rows) {
        const long row = m0 + rl;
        float dy[N / 32], x[N / 32], sp[N / 32];
        ld_row<N>(DY + row * N, lane, dy);
        ld_row<N>(X + row * N, lane, x);
        ld_row<N>(SB + row * 2 * N, lane, sp);
        const float2 s2 = ST[row];
        float g[N / 32], dsp[N / 32], mg = 0.f, mgx = 0.f;
#pragma unroll
        for (int i = 0; i < N / 32; ++i) {
          const float xh = (x[i] - s2.x) * s2.y, s = sigm(sp[i]);
          g[i] = dy[i] * s;
          dsp[i] = dy[i] * xh * s * (1.f - s);
          x[i] = xh;
          mg += g[i];
          mgx += g[i] * xh;
          part[i] += dsp[i];
        }
        mg = wsum(mg) * (1.f / N);
        mgx = wsum(mgx) * (1.f / N);
        float dx[N / 32];
#pragma unroll
        for (int i = 0; i < N / 32; ++i) dx[i] = s2.y * (g[i] - mg - x[i] * mgx);
        if (ADD) {
          float ad[N / 32];
          ld_row<N>(ADD + row * N, lane, ad);
#pragma unroll
          for (int i = 0; i < N / 32; ++i) dx[i] += ad[i];
        }
        st_row<N>(DX + row * N, lane, dx);
#pragma unroll
        for (int k = 0; k < N / 128; ++k)
#pragma unroll
          for (int j = 0; j < 4; ++j) tpose::put(Ts, 4 * (lane + 32 * k) + j, rl, dsp[4 * k + j]);
      }
    }
    __syncthreads();
    tpose::flush(Ts, D, ldd, 0, N, (int)m0, rows, wp, WPB, lane);
    __syncthreads();
    for (int q = 0; q < tpose::TR / WPB; ++q) {   // the second half of D: dy itself (re-read: it was read a moment ago, an L2 hit)
      const int rl = q * WPB + wp;
      if (rl < rows) {
        float dy[N / 32];
        ld_row<N>(DY + (m0 + rl) * N, lane, dy);
#pragma unroll
        for (int k = 0; k < N / 128; ++k)
#pragma unroll
          for (int j = 0; j < 4; ++j) tpose::put(Ts, 4 * (lane + 32 * k) + j, rl, dy[4 * k + j]);
      }
    }
    __syncthreads();
    tpose::flush(Ts, D, ldd, N, N, (int)m0, rows, wp, WPB, lane);
    __syncthreads();
  }
  float (*red)[N] = reinterpret_cast<float (*)[N]>(Ts);
#pragma unroll
  for (int k = 0; k < N / 128; ++k)
#pragma unroll
    for (int j = 0; j < 4; ++j) red[wp][4 * (lane + 32 * k) + j] = part[4 * k + j];
  __syncthreads();
  for (int c = threadIdx.x; c < N; c += 32 * WPB) {
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < WPB; ++w) v += red[w][c];
    PB[(long)blockIdx.x * N + c] = v;
  }
}

template <int N, typename T>
__global__ void __launch_bounds__(32 * WPB) ln_cond_bwd_k(const T* __restrict__ DCN, const T* __restrict__ C, const float2* __restrict__ ST,
                                                          const float* __restrict__ W, T* __restrict__ DC, float* __restrict__ PB, long R, const T* __restrict__ ADD) {
  __shared__ float red[WPB][N];
  const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  float part[N / 32], w[N / 32];
#pragma unroll
  for (int k = 0; k < N / 128; ++k) {
    const float4 t = *reinterpret_cast<const float4*>(W + 4 * (lane + 32 * k));
    w[4 * k] = t.x;
    w[4 * k + 1] = t.y;
    w[4 * k + 2] = t.z;
    w[4 * k + 3] = t.w;
  }
#pragma unroll
  for (int i = 0; i < N / 32; ++i) part[i] = 0.f;
  for (long row = (long)blockIdx.x * WPB + wp; row < R; row += (long)gridDim.x * WPB) {
    float d[N / 32], c[N / 32];
    ld_row<N>(DCN + row * N, lane, d);
    ld_row<N>(C + row * N, lane, c);
    const float2 s2 = ST[row];
    float g[N / 32], mg = 0.f, mgx = 0.f;
#pragma unroll
    for (int i = 0; i < N / 32; ++i) {
      const float xh = (c[i] - s2.x) * s2.y;
      part[i] += d[i] * xh;
      g[i] = d[i] * w[i];
      c[i] = xh;
      mg += g[i];
      mgx += g[i] * xh;
    }
    mg = wsum(mg) * (1.f / N);
    mgx = wsum(mgx) * (1.f / N);
#pragma unroll
    for (int i = 0; i < N / 32; ++i) d[i] = s2.y * (g[i] - mg - c[i] * mgx);
    if (ADD) {   // the gradient the caller sends to cond by another path (the tail's gate projection): joins in fp32, rounded once
      float ad[N / 32];
      ld_row<N>(ADD + row * N, lane, ad);
#pragma unroll
      for (int i = 0; i < N / 32; ++i) d[i] += ad[i];
    }
    st_row<N>(DC + row * N, lane, d);
  }
#pragma unroll
  for (int k = 0; k < N / 128; ++k)
#pragma unroll
    for (int j = 0; j < 4; ++j) red[wp][4 * (lane + 32 * k) + j] = part[4 * k + j];
  __syncthreads();
  for (int c = threadIdx.x; c < N; c += 32 * WPB) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q < WPB; ++q) v += red[q][c];
    PB[(long)blockIdx.x * N + c] = v;
  }
}

template <typename T>
__global__ void pack_k(const T* __restrict__ WS, const T* __restrict__ WB, const T* __restrict__ BS, bf* __restrict__ WC, bf* __restrict__ BC,
                       long nw, long nb) {
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < 2 * nw + 2 * nb; i += (long)gridDim.x * blockDim.x) {
    if (i < nw) WC[i] = __float2bfloat16_rn((float)WS[i]);
    else if (i < 2 * nw) WC[i] = __float2bfloat16_rn((float)WB[i - nw]);
    else if (i < 2 * nw + nb) BC[i - 2 * nw] = __float2bfloat16_rn(BS ? (float)BS[i - 2 * nw] : 0.f);
    else BC[i - 2 * nw] = __float2bfloat16_rn(0.f);
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
void chk_act(const at::Tensor& t, const char* n) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && (t.scalar_type() == at::kBFloat16 || t.scalar_type() == at::kFloat), n,
              ": contiguous bf16 or fp32 CUDA tensor");
}
template <typename T> const T* P(const at::Tensor& t) { return reinterpret_cast<const T*>(t.data_ptr()); }
template <typename T> T* PM(at::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }
cudaStream_t S() { return at::cuda::getCurrentCUDAStream(); }
unsigned rows_grid(long R) { return (unsigned)((R + WPB - 1) / WPB); }

// the row kernels' width (compile-time) and activation dtype (bf16 or fp32, all activations of one call share it)
#define ACT_SWITCH(N, REF, ...)                                                                  \
  do {                                                                                           \
    const bool bf16_ = (REF).scalar_type() == at::kBFloat16;                                     \
    TORCH_CHECK(bf16_ || (REF).scalar_type() == at::kFloat, "adaln rows: bf16 or fp32");         \
    if (bf16_) { using T = bf; WIDTH_SWITCH(N, __VA_ARGS__); }                                   \
    else { using T = float; WIDTH_SWITCH(N, __VA_ARGS__); }                                      \
  } while (0)
#define WIDTH_SWITCH(N, ...)                                                                     \
  do {                                                                                           \
    if ((N) == 128) { constexpr int NN = 128; __VA_ARGS__; }                                     \
    else if ((N) == 256) { constexpr int NN = 256; __VA_ARGS__; }                                \
    else if ((N) == 384) { constexpr int NN = 384; __VA_ARGS__; }                                \
    else if ((N) == 512) { constexpr int NN = 512; __VA_ARGS__; }                                \
    else if ((N) == 768) { constexpr int NN = 768; __VA_ARGS__; }                                \
    else if ((N) == 1024) { constexpr int NN = 1024; __VA_ARGS__; }                              \
    else TORCH_CHECK(false, "adaln rows: width 128 / 256 / 384 / 512 / 768 / 1024, got ", (long)(N)); \
  } while (0)
}  // namespace

std::vector<at::Tensor> ln_cond_fwd(at::Tensor cond, at::Tensor w, double eps) {
  chk_act(cond, "cond");
  chk(w, at::kFloat, "wc");
  const long N = w.numel(), R = cond.numel() / N;
  const at::cuda::CUDAGuard gd(cond.device());
  auto cn = at::empty_like(cond);
  auto st = at::empty({R, 2}, cond.options().dtype(at::kFloat));
  ACT_SWITCH(N, cond, ln_cond_fwd_k<NN, T><<<rows_grid(R), 32 * WPB, 0, S()>>>(P<T>(cond), w.data_ptr<float>(), PM<T>(cn),
                                                                               reinterpret_cast<float2*>(st.data_ptr()), R, (float)eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {cn, st};
}

// fp32: (cn, stats, cnT [N, ldd] with ldd = R rounded up to 32; use cnT[:, :R])
std::vector<at::Tensor> ln_cond_fwd_t(at::Tensor cond, at::Tensor w, double eps) {
  chk(cond, at::kFloat, "cond");
  chk(w, at::kFloat, "wc");
  const long N = w.numel(), R = cond.numel() / N;
  TORCH_CHECK(N == 256 || N == 384 || N == 512 || N == 768, "ln_cond_fwd_t: width 256 / 384 / 512 / 768");
  const at::cuda::CUDAGuard gd(cond.device());
  const long ldd = (R + 31) / 32 * 32;
  auto cn = at::empty_like(cond);
  auto st = at::empty({R, 2}, cond.options());
  auto cnt = at::empty({N, ldd}, cond.options());
  const size_t shm = (size_t)N * tpose::LD * sizeof(float);
  const unsigned grid = (unsigned)((R + 31) / 32);
#define LT(NN)                                                                                                                                         \
  do {                                                                                                                                                 \
    auto k = ln_cond_fwd_t_k<NN>;                                                                                                                       \
    C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));                                                    \
    k<<<grid, 32 * WPB, shm, S()>>>(cond.data_ptr<float>(), w.data_ptr<float>(), cn.data_ptr<float>(), reinterpret_cast<float2*>(st.data_ptr()),         \
                                    cnt.data_ptr<float>(), ldd, R, (float)eps);                                                                          \
  } while (0)
  if (N == 256) LT(256); else if (N == 384) LT(384); else if (N == 512) LT(512); else LT(768);
#undef LT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {cn, st, cnt};
}

std::vector<at::Tensor> adaln_fwd(at::Tensor x, at::Tensor sb, double eps, bool rna) {
  chk_act(x, "x");
  TORCH_CHECK(sb.scalar_type() == x.scalar_type() && sb.is_contiguous(), "sb dtype");
  const long N = sb.size(1) / 2, R = sb.size(0);
  TORCH_CHECK(x.numel() == R * N, "adaln_fwd shapes");
  const at::cuda::CUDAGuard gd(x.device());
  auto y = at::empty_like(x);
  auto st = at::empty({R, 2}, x.options().dtype(at::kFloat));
  ACT_SWITCH(N, x, adaln_fwd_k<NN, T><<<rows_grid(R), 32 * WPB, 0, S()>>>(P<T>(x), P<T>(sb), PM<T>(y), reinterpret_cast<float2*>(st.data_ptr()), R,
                                                                          (float)eps, rna && x.scalar_type() == at::kFloat));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, st};
}

std::vector<at::Tensor> adaln_bwd(at::Tensor dy, at::Tensor x, at::Tensor sb, at::Tensor st, c10::optional<at::Tensor> add, bool rna = false) {
  if (add) { chk_act(*add, "add"); TORCH_CHECK(add->scalar_type() == dy.scalar_type() && add->numel() == dy.numel(), "add shape"); }
  chk_act(dy, "dy");
  TORCH_CHECK(x.scalar_type() == dy.scalar_type() && sb.scalar_type() == dy.scalar_type() && x.is_contiguous() && sb.is_contiguous(), "dtypes");
  chk(st, at::kFloat, "st");
  const long N = sb.size(1) / 2, R = sb.size(0);
  TORCH_CHECK(x.numel() == R * N && dy.numel() == R * N && st.numel() == 2 * R, "adaln_bwd shapes");
  const at::cuda::CUDAGuard gd(x.device());
  auto dsb = at::empty_like(sb);
  auto dx = at::empty_like(x);
  const unsigned grid = std::min<unsigned>(rows_grid(R), 4u * nsm_of(x));
  auto pb = at::empty({(long)grid, N}, x.options().dtype(at::kFloat));
  ACT_SWITCH(N, x, adaln_bwd_k<NN, T><<<grid, 32 * WPB, 0, S()>>>(P<T>(dy), P<T>(x), P<T>(sb), reinterpret_cast<const float2*>(st.data_ptr()), PM<T>(dsb),
                                                                 PM<T>(dx), pb.data_ptr<float>(), R, add ? P<T>(*add) : nullptr, rna));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dsb, dx, pb};
}

// fp32 feature-major backward row pass: ([2 N, ldd] D = [dscale ; dy]^T with ldd = R rounded up to 32, dx, block partials of the scale-bias gradient)
std::vector<at::Tensor> adaln_bwd_t(at::Tensor dy, at::Tensor x, at::Tensor sb, at::Tensor st, c10::optional<at::Tensor> add, int64_t gmul = 2) {
  if (add) { chk_act(*add, "add"); TORCH_CHECK(add->scalar_type() == at::kFloat && add->numel() == dy.numel(), "add shape"); }
  chk(dy, at::kFloat, "dy");
  chk(x, at::kFloat, "x");
  chk(sb, at::kFloat, "sb");
  chk(st, at::kFloat, "st");
  const long N = sb.size(1) / 2, R = sb.size(0);
  TORCH_CHECK(x.numel() == R * N && dy.numel() == R * N && st.numel() == 2 * R, "adaln_bwd_t shapes");
  TORCH_CHECK(N == 256 || N == 384 || N == 512 || N == 768, "adaln_bwd_t: width 256 / 384 / 512 / 768");
  const at::cuda::CUDAGuard gd(x.device());
  const long ldd = (R + 31) / 32 * 32;
  auto D = at::empty({2 * N, ldd}, x.options());
  auto dx = at::empty_like(x);
  const unsigned grid = std::min<unsigned>((unsigned)((R + 31) / 32), (unsigned)gmul * nsm_of(x));
  auto pb = at::empty({(long)grid, N}, x.options());
  const size_t shm = (size_t)N * tpose::LD * sizeof(float);
  const float* addp = add ? add->data_ptr<float>() : nullptr;
#define BT(NN)                                                                                                                                         \
  do {                                                                                                                                                 \
    auto k = adaln_bwd_t_k<NN>;                                                                                                                         \
    C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));                                                    \
    k<<<grid, 32 * WPB, shm, S()>>>(dy.data_ptr<float>(), x.data_ptr<float>(), sb.data_ptr<float>(), reinterpret_cast<const float2*>(st.data_ptr()),    \
                                    dx.data_ptr<float>(), D.data_ptr<float>(), ldd, pb.data_ptr<float>(), R, addp);                                      \
  } while (0)
  if (N == 256) BT(256); else if (N == 384) BT(384); else if (N == 512) BT(512); else BT(768);
#undef BT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {D, dx, pb};
}

std::vector<at::Tensor> ln_cond_bwd(at::Tensor dcn, at::Tensor cond, at::Tensor st, at::Tensor w, c10::optional<at::Tensor> add) {
  if (add) { chk_act(*add, "add"); TORCH_CHECK(add->scalar_type() == dcn.scalar_type() && add->numel() == dcn.numel(), "add shape"); }
  chk_act(dcn, "dcn");
  TORCH_CHECK(cond.scalar_type() == dcn.scalar_type() && cond.is_contiguous(), "dtypes");
  chk(st, at::kFloat, "st");
  chk(w, at::kFloat, "wc");
  const long N = w.numel(), R = cond.numel() / N;
  TORCH_CHECK(dcn.numel() == R * N && st.numel() == 2 * R, "ln_cond_bwd shapes");
  const at::cuda::CUDAGuard gd(cond.device());
  auto dc = at::empty_like(cond);
  const unsigned grid = std::min<unsigned>(rows_grid(R), 4u * nsm_of(cond));
  auto pb = at::empty({(long)grid, N}, cond.options().dtype(at::kFloat));
  ACT_SWITCH(N, cond, ln_cond_bwd_k<NN, T><<<grid, 32 * WPB, 0, S()>>>(P<T>(dcn), P<T>(cond), reinterpret_cast<const float2*>(st.data_ptr()),
                                                                       w.data_ptr<float>(), PM<T>(dc), pb.data_ptr<float>(), R, add ? P<T>(*add) : nullptr));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dc, pb};
}

// [Ws | Wb] -> Wcat [2 N, Nc] bf16 and [bs | 0] -> bcat [2 N] bf16
void pack(at::Tensor ws, at::Tensor wb, c10::optional<at::Tensor> bs, at::Tensor wcat, at::Tensor bcat) {
  TORCH_CHECK(ws.is_cuda() && ws.is_contiguous() && wb.is_contiguous() && ws.sizes() == wb.sizes() && ws.scalar_type() == wb.scalar_type(), "pack weights");
  chk(wcat, at::kBFloat16, "wcat");
  chk(bcat, at::kBFloat16, "bcat");
  const long nw = ws.numel(), nb = ws.size(0);
  TORCH_CHECK(wcat.numel() == 2 * nw && bcat.numel() == 2 * nb, "pack shapes");
  const at::cuda::CUDAGuard gd(ws.device());
  const unsigned grid = (unsigned)std::min<long>((2 * nw + 2 * nb + 255) / 256, 8L * nsm_of(ws));
  if (ws.scalar_type() == at::kFloat) {
    pack_k<float><<<grid, 256, 0, S()>>>(ws.data_ptr<float>(), wb.data_ptr<float>(), bs ? bs->data_ptr<float>() : nullptr, PM<bf>(wcat), PM<bf>(bcat), nw, nb);
  } else {
    TORCH_CHECK(ws.scalar_type() == at::kBFloat16, "pack: fp32 or bf16 weights");
    pack_k<bf><<<grid, 256, 0, S()>>>(P<bf>(ws), P<bf>(wb), bs ? P<bf>(*bs) : nullptr, PM<bf>(wcat), PM<bf>(bcat), nw, nb);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ======================================================================================================================== fp32 weight packing, one launch
// A training step's fp32 weight operands: concatenations ([Ws; Wb]) and K-major (transposed) copies for the TF32 backward GEMMs (the "nt" cuBLAS kernel), written
// by ONE launch instead of a PyTorch cat + fill + transposing copy per tensor (5 - 20 us each, 40 - 60 us a step). Job j copies the 2-D fp32 tensor src[j] [R, C] into
// dst[j] (row stride ldd[j]) at (row0, col0): mode 0 as it is (dst[row0 + r, col0 + c]), mode 1 transposed (dst[row0 + c, col0 + r]), mode 2 zero fill of an R x C block.
struct PackJobs {
  const float* src[8];
  float* dst[8];
  int rows[8], cols[8], ldd[8], row0[8], col0[8], mode[8], rna[8];
  int tile_end[8];   // prefix sums of the 32 x 32 tile counts
  int n;
};
__global__ void __launch_bounds__(256) pack_f32_k(PackJobs jb) {
  __shared__ float tile[32][33];
  int t = blockIdx.x, j = 0;
  while (t >= jb.tile_end[j]) ++j;
  if (j) t -= jb.tile_end[j - 1];
  const int R = jb.rows[j], C = jb.cols[j], tc = (C + 31) / 32;
  const int r0 = (t / tc) * 32, c0 = (t % tc) * 32;
  const int tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
  float* dst = jb.dst[j];
  const long ldd = jb.ldd[j];
  const int mode = jb.mode[j];
  const bool rna = jb.rna[j] != 0;
  if (mode == 2) {
#pragma unroll
    for (int i = 0; i < 4; ++i) { const int r = r0 + ty + 8 * i, c = c0 + tx; if (r < R && c < C) dst[(long)(jb.row0[j] + r) * ldd + jb.col0[j] + c] = 0.f; }
    return;
  }
  const float* src = jb.src[j];
#pragma unroll
  for (int i = 0; i < 4; ++i) { const int r = r0 + ty + 8 * i, c = c0 + tx; tile[ty + 8 * i][tx] = (r < R && c < C) ? src[(long)r * C + c] : 0.f; }
  __syncthreads();
  if (mode == 0) {
#pragma unroll
    for (int i = 0; i < 4; ++i) { const int r = r0 + ty + 8 * i, c = c0 + tx; if (r < R && c < C) dst[(long)(jb.row0[j] + r) * ldd + jb.col0[j] + c] = rna ? tf32_rn(tile[ty + 8 * i][tx]) : tile[ty + 8 * i][tx]; }
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) { const int c = c0 + ty + 8 * i, r = r0 + tx; if (r < R && c < C) dst[(long)(jb.row0[j] + c) * ldd + jb.col0[j] + r] = rna ? tf32_rn(tile[tx][ty + 8 * i]) : tile[tx][ty + 8 * i]; }
  }
}
// srcs[k]: fp32 [R, C] (ignored for mode 2: pass any tensor with the block's shape); dsts[k]: fp32 2-D; spec[k] = (mode, row0, col0)
void pack_f32(std::vector<at::Tensor> srcs, std::vector<at::Tensor> dsts, std::vector<std::vector<int64_t>> spec) {
  TORCH_CHECK(!srcs.empty() && srcs.size() <= 8 && srcs.size() == dsts.size() && srcs.size() == spec.size(), "pack_f32: 1 .. 8 jobs");
  PackJobs jb{};
  int tiles = 0;
  for (size_t k = 0; k < srcs.size(); ++k) {
    const auto& s = srcs[k];
    const auto& d = dsts[k];
    TORCH_CHECK(s.is_cuda() && d.is_cuda() && s.scalar_type() == at::kFloat && d.scalar_type() == at::kFloat && s.dim() == 2 && d.dim() == 2 && s.is_contiguous() &&
                d.stride(1) == 1, "pack_f32: fp32 2-D, contiguous source, row-contiguous destination");
    const int mode = (int)spec[k][0], row0 = (int)spec[k][1], col0 = (int)spec[k][2];
    const int R = (int)s.size(0), C = (int)s.size(1);
    const int rr = mode == 1 ? C : R, cc = mode == 1 ? R : C;   // extent in the destination
    TORCH_CHECK(row0 + rr <= d.size(0) && col0 + cc <= d.size(1), "pack_f32: destination too small");
    jb.src[k] = s.data_ptr<float>();
    jb.dst[k] = d.data_ptr<float>();
    jb.rows[k] = R; jb.cols[k] = C; jb.ldd[k] = (int)d.stride(0); jb.row0[k] = row0; jb.col0[k] = col0; jb.mode[k] = mode; jb.rna[k] = spec[k].size() > 3 ? (int)spec[k][3] : 0;
    tiles += ((R + 31) / 32) * ((C + 31) / 32);
    jb.tile_end[k] = tiles;
  }
  jb.n = (int)srcs.size();
  const at::cuda::CUDAGuard gd(srcs[0].device());
  pack_f32_k<<<tiles, 256, 0, S()>>>(jb);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ======================================================================================================================== column sums of block partials
// out[n] = sum_p parts[p, n] in a fixed order (32 columns x 8 row slices a block, the slices summed in order). The sm_90a extension
// (adaln_sm90.cu) has the same kernels; these are the ones B200 (sm_100a) runs, where that extension does not build.
__global__ void __launch_bounds__(256) colsum_rows_k(const float* __restrict__ pa, float* __restrict__ oa, int Pa, int Na,
                                                     const float* __restrict__ pb, float* __restrict__ ob, int Pb, int Nb) {
  __shared__ float red[8][33];
  const int ba = (Na + 31) / 32;
  const bool second = (int)blockIdx.x >= ba;
  const float* parts = second ? pb : pa;
  float* out = second ? ob : oa;
  const int P = second ? Pb : Pa, N = second ? Nb : Na;
  const int c = ((int)blockIdx.x - (second ? ba : 0)) * 32 + (threadIdx.x & 31), s = threadIdx.x >> 5;
  float v = 0.f;
  if (c < N) for (int p = s; p < P; p += 8) v += parts[(int64_t)p * N + c];
  red[s][threadIdx.x & 31] = v;
  __syncthreads();
  if (s == 0 && c < N) {
    float t = 0.f;
#pragma unroll
    for (int q = 0; q < 8; ++q) t += red[q][threadIdx.x & 31];
    out[c] = t;
  }
}
std::vector<at::Tensor> colsum2(at::Tensor a, at::Tensor b) {
  chk(a, at::kFloat, "parts a");
  chk(b, at::kFloat, "parts b");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "parts [P, N] fp32");
  const at::cuda::CUDAGuard gd(a.device());
  auto oa = at::empty({a.size(1)}, a.options()), ob = at::empty({b.size(1)}, b.options());
  colsum_rows_k<<<(unsigned)((a.size(1) + 31) / 32 + (b.size(1) + 31) / 32), 256, 0, S()>>>(
      P<float>(a), PM<float>(oa), (int)a.size(0), (int)a.size(1), P<float>(b), PM<float>(ob), (int)b.size(0), (int)b.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {oa, ob};
}
at::Tensor colsum(at::Tensor parts) {
  chk(parts, at::kFloat, "parts");
  TORCH_CHECK(parts.dim() == 2, "parts [P, N] fp32");
  const at::cuda::CUDAGuard gd(parts.device());
  auto out = at::empty({parts.size(1)}, parts.options());
  colsum_rows_k<<<(unsigned)((parts.size(1) + 31) / 32), 256, 0, S()>>>(P<float>(parts), PM<float>(out), (int)parts.size(0),
                                                                          (int)parts.size(1), nullptr, nullptr, 0, 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("colsum", &colsum);
  m.def("colsum2", &colsum2);
  m.def("ln_cond_fwd", &ln_cond_fwd);
  m.def("ln_cond_fwd_t", &ln_cond_fwd_t);
  m.def("adaln_fwd", &adaln_fwd, py::arg("x"), py::arg("sb"), py::arg("eps"), py::arg("rna") = false);
  m.def("adaln_bwd", &adaln_bwd, py::arg("dy"), py::arg("x"), py::arg("sb"), py::arg("st"), py::arg("add") = py::none(), py::arg("rna") = false);
  m.def("adaln_bwd_t", &adaln_bwd_t, py::arg("dy"), py::arg("x"), py::arg("sb"), py::arg("st"), py::arg("add") = py::none(), py::arg("gmul") = 2);
  m.def("ln_cond_bwd", &ln_cond_bwd);
  m.def("pack", &pack);
  m.def("pack_f32", &pack_f32);
}
