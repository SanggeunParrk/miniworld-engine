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

namespace {
using bf = __nv_bfloat16;
using bf2 = __nv_bfloat162;
constexpr int WPB = 8;  // warps per block

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

template <int N, typename T>
__global__ void __launch_bounds__(32 * WPB) adaln_fwd_k(const T* __restrict__ X, const T* __restrict__ SB, T* __restrict__ Y,
                                                        float2* __restrict__ ST, long R, float eps) {
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
  st_row<N>(Y + row * N, lane, x);
  if (lane == 0) ST[row] = make_float2(mean, rstd);
}

template <int N, typename T>
__global__ void __launch_bounds__(32 * WPB, 2) adaln_bwd_k(const T* __restrict__ DY, const T* __restrict__ X, const T* __restrict__ SB,
                                                        const float2* __restrict__ ST, T* __restrict__ DSB, T* __restrict__ DX,
                                                        float* __restrict__ PB, long R, const T* __restrict__ ADD) {
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

std::vector<at::Tensor> adaln_fwd(at::Tensor x, at::Tensor sb, double eps) {
  chk_act(x, "x");
  TORCH_CHECK(sb.scalar_type() == x.scalar_type() && sb.is_contiguous(), "sb dtype");
  const long N = sb.size(1) / 2, R = sb.size(0);
  TORCH_CHECK(x.numel() == R * N, "adaln_fwd shapes");
  const at::cuda::CUDAGuard gd(x.device());
  auto y = at::empty_like(x);
  auto st = at::empty({R, 2}, x.options().dtype(at::kFloat));
  ACT_SWITCH(N, x, adaln_fwd_k<NN, T><<<rows_grid(R), 32 * WPB, 0, S()>>>(P<T>(x), P<T>(sb), PM<T>(y), reinterpret_cast<float2*>(st.data_ptr()), R,
                                                                          (float)eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, st};
}

std::vector<at::Tensor> adaln_bwd(at::Tensor dy, at::Tensor x, at::Tensor sb, at::Tensor st, c10::optional<at::Tensor> add) {
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
                                                                 PM<T>(dx), pb.data_ptr<float>(), R, add ? P<T>(*add) : nullptr));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dsb, dx, pb};
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ln_cond_fwd", &ln_cond_fwd);
  m.def("adaln_fwd", &adaln_fwd);
  m.def("adaln_bwd", &adaln_bwd);
  m.def("ln_cond_bwd", &ln_cond_bwd);
  m.def("pack", &pack);
}
