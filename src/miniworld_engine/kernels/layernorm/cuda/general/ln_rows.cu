// General LayerNorm / RMSNorm, the B200 form for rows a (sub-)warp holds without idle lanes (TPR <= 32 threads a row, CHUNKS
// 16-byte vectors a thread, shuffle reductions only, no shared-memory round trip): small CTAs (WARPS warps) on a resident grid of
// ``ctas`` CTAs a SM, RPT rows a thread loaded before any reduction. The forward reads x through the non-coherent path without L1
// allocation and writes y with streaming stores (both touched once). Measured on B200 against the Triton kernels (``config_sm100``).
//
//   ln_rows_fwd  y = (x - mean) rstd w + b (RMS: no mean, no bias); mean / rstd per row when asked
//   ln_rows_bwd  dx, and per-CTA partial rows of dw / db: per thread over its rows, the row slots of a warp folded by shuffles, the
//                warps of the CTA in order -- fixed order, then ln_bwd.cu's column pass (one launch): a replay is bit-identical.
#include "ln_common.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace mwln {
namespace rows {

// the streaming hints are taken for fp32 rows only: measured on B200 they took an fp32 [147456, 128] forward 23.3 -> 21.1 us and
// cost the bf16 ones (a [147456, 128] bf16 forward 9.7 -> 12.5 us)
template <typename T, int VEC>
__device__ __forceinline__ Pack<T, VEC> ld_stream(const T* p) {
  if constexpr (sizeof(T) == 4 && sizeof(Pack<T, VEC>) == 16) {
    uint4 u;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(u.x), "=r"(u.y), "=r"(u.z), "=r"(u.w)
                 : "l"(p));
    return *reinterpret_cast<Pack<T, VEC>*>(&u);
  } else {
    return load_pack<T, VEC>(p);
  }
}

template <typename T, int VEC>
__device__ __forceinline__ void st_stream(T* p, const float* o) {
  Pack<T, VEC> pk;
#pragma unroll
  for (int i = 0; i < VEC; ++i) pk.v[i] = from_f<T>(o[i]);
  if constexpr (sizeof(T) == 4 && sizeof(Pack<T, VEC>) == 16) __stcs(reinterpret_cast<uint4*>(p), *reinterpret_cast<uint4*>(&pk));
  else *reinterpret_cast<Pack<T, VEC>*>(p) = pk;
}

template <typename T, int TPR, int CH, int VEC, bool RMS, int RPT, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) fwd(const T* __restrict__ x, const void* __restrict__ w, const void* __restrict__ b, int wbf,
                                                  int bbf, T* __restrict__ y, float* __restrict__ mo, float* __restrict__ ro, int64_t M,
                                                  int N, float eps) {
  static_assert(TPR <= 32, "sub-warp rows");
  constexpr int RPC = WARPS * 32 / TPR, RT = RPC * RPT, E = CH * VEC;
  const int r_in = threadIdx.x / TPR, lane = threadIdx.x % TPR;
  const float inv_n = 1.f / float(N);
  float wr[E], br[E];
  load_affine<TPR, CH, VEC>(w, wbf, 1.f, lane, N, wr);
  if constexpr (!RMS) load_affine<TPR, CH, VEC>(b, bbf, 0.f, lane, N, br);
  for (int64_t t = blockIdx.x; t * RT < M; t += gridDim.x) {
    Pack<T, VEC> raw[RPT][CH];
#pragma unroll
    for (int k = 0; k < RPT; ++k) {
      const int64_t row = t * RT + k * RPC + r_in;
#pragma unroll
      for (int c = 0; c < CH; ++c)
        raw[k][c] = row < M ? ld_stream<T, VEC>(x + row * N + (c * TPR + lane) * VEC) : zero_pack<T, VEC>();
    }
#pragma unroll
    for (int k = 0; k < RPT; ++k) {
      const int64_t row = t * RT + k * RPC + r_in;
      float mean = 0.f;
      if constexpr (!RMS) {
        float s = 0.f;
#pragma unroll
        for (int c = 0; c < CH; ++c)
#pragma unroll
          for (int i = 0; i < VEC; ++i) s += to_f<T>(raw[k][c].v[i]);
        mean = row_sum<TPR>(s, nullptr, 0, lane) * inv_n;
      }
      float q = 0.f;
#pragma unroll
      for (int c = 0; c < CH; ++c)
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          const float d = to_f<T>(raw[k][c].v[i]) - mean;
          q += d * d;
        }
      const float rs = rsqrtf(row_sum<TPR>(q, nullptr, 0, lane) * inv_n + eps);
      if (row < M) {
#pragma unroll
        for (int c = 0; c < CH; ++c) {
          float o[VEC];
#pragma unroll
          for (int i = 0; i < VEC; ++i) {
            const float xn = (to_f<T>(raw[k][c].v[i]) - mean) * rs;
            o[i] = RMS ? xn * wr[c * VEC + i] : xn * wr[c * VEC + i] + br[c * VEC + i];
          }
          st_stream<T, VEC>(y + row * N + (c * TPR + lane) * VEC, o);
        }
        if (lane == 0) {
          if (mo) mo[row] = mean;
          if (ro) ro[row] = rs;
        }
      }
    }
  }
}

template <typename T, int TPR, int CH, int VEC, bool RMS, int RPT, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) bwd(const T* __restrict__ dy, const T* __restrict__ x, const void* __restrict__ w, int wbf,
                                                  const float* __restrict__ mean, const float* __restrict__ rstd, T* __restrict__ dx,
                                                  float* __restrict__ part, int pstride, int64_t M, int N) {
  static_assert(TPR <= 32, "sub-warp rows");
  constexpr int RPC = WARPS * 32 / TPR, RT = RPC * RPT, E = CH * VEC;
  const int r_in = threadIdx.x / TPR, lane = threadIdx.x % TPR;
  const float inv_n = 1.f / float(N);
  const bool wgrad = part != nullptr;
  float wr[E], aw[E], ab[E];
  load_affine<TPR, CH, VEC>(w, wbf, 1.f, lane, N, wr);
#pragma unroll
  for (int e = 0; e < E; ++e) aw[e] = ab[e] = 0.f;
  for (int64_t t = blockIdx.x; t * RT < M; t += gridDim.x) {
    Pack<T, VEC> xr[RPT][CH], dr[RPT][CH];
    float mu[RPT], rs[RPT];
#pragma unroll
    for (int k = 0; k < RPT; ++k) {
      const int64_t row = t * RT + k * RPC + r_in;
      const bool v = row < M;
      mu[k] = v && !RMS ? mean[row] : 0.f;
      rs[k] = v ? rstd[row] : 0.f;
#pragma unroll
      for (int c = 0; c < CH; ++c) {
        xr[k][c] = v ? load_pack<T, VEC>(x + row * N + (c * TPR + lane) * VEC) : zero_pack<T, VEC>();
        dr[k][c] = v ? load_pack<T, VEC>(dy + row * N + (c * TPR + lane) * VEC) : zero_pack<T, VEC>();
      }
    }
#pragma unroll
    for (int k = 0; k < RPT; ++k) {
      const int64_t row = t * RT + k * RPC + r_in;
      float s1 = 0.f, s2 = 0.f;
#pragma unroll
      for (int c = 0; c < CH; ++c)
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          const int e = c * VEC + i;
          const float dv = to_f<T>(dr[k][c].v[i]);
          const float xh = (to_f<T>(xr[k][c].v[i]) - mu[k]) * rs[k];
          const float g = dv * wr[e];
          s1 += g;
          s2 += g * xh;
          if (wgrad) {
            aw[e] += dv * xh;
            if constexpr (!RMS) ab[e] += dv;
          }
        }
      if constexpr (RMS) s1 = 0.f;
      else s1 = row_sum<TPR>(s1, nullptr, 0, lane) * inv_n;
      s2 = row_sum<TPR>(s2, nullptr, 0, lane) * inv_n;
      if (row < M) {
#pragma unroll
        for (int c = 0; c < CH; ++c) {
          float o[VEC];
#pragma unroll
          for (int i = 0; i < VEC; ++i) {
            const int e = c * VEC + i;
            const float xh = (to_f<T>(xr[k][c].v[i]) - mu[k]) * rs[k];
            o[i] = rs[k] * (to_f<T>(dr[k][c].v[i]) * wr[e] - s1 - xh * s2);
          }
          store_vec<T, VEC>(dx + row * N + (c * TPR + lane) * VEC, o);
        }
      }
    }
  }
  if (!wgrad) return;
  // the row slots of a warp hold the same columns: fold them (fixed shuffle tree), then the CTA's warps in order
  __shared__ float red[WARPS][2][E * TPR];
#pragma unroll
  for (int e = 0; e < E; ++e) {
#pragma unroll
    for (int o = 16; o >= TPR; o >>= 1) {
      aw[e] += __shfl_xor_sync(0xffffffffu, aw[e], o);
      if constexpr (!RMS) ab[e] += __shfl_xor_sync(0xffffffffu, ab[e], o);
    }
  }
  const int warp = threadIdx.x >> 5;
  if ((threadIdx.x & 31) < TPR) {
#pragma unroll
    for (int e = 0; e < E; ++e) {
      red[warp][0][e * TPR + lane] = aw[e];
      if constexpr (!RMS) red[warp][1][e * TPR + lane] = ab[e];
    }
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < E * TPR; idx += WARPS * 32) {
    const int e = idx / TPR, l = idx % TPR;
    const int col = ((e / VEC) * TPR + l) * VEC + e % VEC;
    float s = 0.f, sb = 0.f;
#pragma unroll
    for (int wi = 0; wi < WARPS; ++wi) {
      s += red[wi][0][idx];
      if constexpr (!RMS) sb += red[wi][1][idx];
    }
    part[int64_t(blockIdx.x) * N + col] = s;
    if constexpr (!RMS) part[(int64_t(pstride) + blockIdx.x) * N + col] = sb;
  }
}

// the builds: (TPR, CH) layouts x RPT {1, 2} x WARPS {1, 2}
#define MWLN_ROWS_LAYOUTS(X, ...)                                                                                            \
  X(4, 1, __VA_ARGS__) X(4, 2, __VA_ARGS__) X(4, 4, __VA_ARGS__) X(8, 1, __VA_ARGS__) X(8, 2, __VA_ARGS__) X(8, 4, __VA_ARGS__) \
  X(16, 1, __VA_ARGS__) X(16, 2, __VA_ARGS__) X(16, 4, __VA_ARGS__) X(32, 1, __VA_ARGS__) X(32, 2, __VA_ARGS__) X(32, 3, __VA_ARGS__) \
  X(32, 4, __VA_ARGS__)

template <typename T, int VEC, bool RMS>
bool dispatch(bool is_fwd, int tpr, int ch, int rpt, int warps, int grid, cudaStream_t s, const T* x, const T* dy, const void* w,
              const void* b, int wbf, int bbf, T* out, float* mean, float* rstd, float* part, int pstride, int64_t M, int N, float eps) {
#define MWLN_RL(R, W, TPR_, CH_)                                                                             \
  if (rpt == R && warps == W) {                                                                              \
    if (is_fwd) fwd<T, TPR_, CH_, VEC, RMS, R, W><<<grid, W * 32, 0, s>>>(x, w, b, wbf, bbf, out, mean, rstd, M, N, eps); \
    else bwd<T, TPR_, CH_, VEC, RMS, R, W><<<grid, W * 32, 0, s>>>(dy, x, w, wbf, mean, rstd, out, part, pstride, M, N);  \
    return true;                                                                                             \
  }
#define MWLN_RX(TPR_, CH_, unused)                                                                                       \
  if (tpr == TPR_ && ch == CH_) {                                                                                         \
    MWLN_RL(1, 1, TPR_, CH_) MWLN_RL(2, 1, TPR_, CH_) MWLN_RL(1, 2, TPR_, CH_) MWLN_RL(2, 2, TPR_, CH_)                     \
  }
  MWLN_ROWS_LAYOUTS(MWLN_RX, 0)
#undef MWLN_RX
#undef MWLN_RL
  return false;
}

}  // namespace rows
}  // namespace mwln

void ln_col_reduce_launch(const float* part, int P, int N, void* out, void* out2, int obf, cudaStream_t s);

// x [M, N] bf16 / fp32 contiguous, rows 16-byte aligned; w / b fp32 or bf16 [N] or empty. (y, mean, rstd) like ln_fwd.
std::vector<torch::Tensor> ln_rows_fwd(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps, bool save, bool rms, int64_t tpr,
                                       int64_t ch, int64_t rpt, int64_t warps, int64_t ctas) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "x");
  TORCH_CHECK(!rms || b.numel() == 0, "RMSNorm has no bias");
  const int64_t M = x.size(0);
  const int N = int(x.size(1));
  auto y = torch::empty_like(x);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto mean = save && !rms ? torch::empty({M}, f32) : torch::empty({0}, f32);
  auto rstd = save ? torch::empty({M}, f32) : torch::empty({0}, f32);
  if (M == 0) return {y, mean, rstd};
  for (auto* t : {&w, &b})
    if (t->numel() && (reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 || !t->is_contiguous())) *t = t->contiguous().clone();
  const int wbf = w.numel() && w.scalar_type() == torch::kBFloat16, bbf = b.numel() && b.scalar_type() == torch::kBFloat16;
  const int64_t rt = int64_t(warps) * 32 / tpr * rpt;
  const int64_t tiles = (M + rt - 1) / rt, cap = int64_t(mwln::num_sms()) * ctas;
  const int grid = int(tiles < cap ? tiles : cap);
  auto s = at::cuda::getCurrentCUDAStream();
  bool ok = false;
  const void* wp = w.numel() ? w.data_ptr() : nullptr;
  const void* bp = b.numel() ? b.data_ptr() : nullptr;
  float* mp = mean.numel() ? mean.data_ptr<float>() : nullptr;
  float* rp = rstd.numel() ? rstd.data_ptr<float>() : nullptr;
#define MWLN_GO(CT, V)                                                                                                         \
  ok = (rms ? mwln::rows::dispatch<CT, V, true> : mwln::rows::dispatch<CT, V, false>)(                                         \
      true, int(tpr), int(ch), int(rpt), int(warps), grid, s, reinterpret_cast<const CT*>(x.data_ptr()), nullptr, wp, bp, wbf, bbf, \
      reinterpret_cast<CT*>(y.data_ptr()), mp, rp, nullptr, 0, M, N, float(eps));
  if (x.scalar_type() == torch::kBFloat16) { MWLN_GO(__nv_bfloat16, 8) }
  else if (x.scalar_type() == torch::kFloat32) { MWLN_GO(float, 4) }
#undef MWLN_GO
  TORCH_CHECK(ok, "B200 LayerNorm rows forward: no build for tpr=", tpr, " ch=", ch, " rpt=", rpt, " warps=", warps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, mean, rstd};
}

// (dx, dw, db) like ln_bwd; `ctas` CTAs a SM = the partial rows' count / SMs.
std::vector<torch::Tensor> ln_rows_bwd(torch::Tensor dy, torch::Tensor x, torch::Tensor w, torch::Tensor mean, torch::Tensor rstd, bool wgrad,
                                       bool rms, int64_t tpr, int64_t ch, int64_t rpt, int64_t warps, int64_t ctas, bool gbf16) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && dy.is_contiguous() && dy.sizes() == x.sizes() &&
                  dy.scalar_type() == x.scalar_type() && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(dy.data_ptr()) % 16 == 0,
              "dy, x");
  const int64_t M = x.size(0);
  const int N = int(x.size(1));
  auto dx = torch::empty_like(x);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto gopt = x.options().dtype(gbf16 ? torch::kBFloat16 : torch::kFloat32);
  if (w.numel() && (reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 || !w.is_contiguous())) w = w.contiguous().clone();
  const int wbf = w.numel() && w.scalar_type() == torch::kBFloat16;
  const int64_t rt = int64_t(warps) * 32 / tpr * rpt;
  const int64_t tiles = (M + rt - 1) / rt, cap = int64_t(mwln::num_sms()) * ctas;
  const int grid = int(std::max<int64_t>(1, tiles < cap ? tiles : cap));
  auto part = wgrad ? torch::empty({rms ? 1 : 2, grid, N}, f32) : torch::empty({0}, f32);
  auto dw = wgrad ? (M == 0 ? torch::zeros({N}, gopt) : torch::empty({N}, gopt)) : torch::empty({0}, f32);
  auto db = wgrad && !rms ? (M == 0 ? torch::zeros({N}, gopt) : torch::empty({N}, gopt)) : torch::empty({0}, f32);
  if (M == 0) return {dx, dw, db};
  auto s = at::cuda::getCurrentCUDAStream();
  bool ok = false;
  float* pp = wgrad ? part.data_ptr<float>() : nullptr;
#define MWLN_GO(CT, V)                                                                                                             \
  ok = (rms ? mwln::rows::dispatch<CT, V, true> : mwln::rows::dispatch<CT, V, false>)(                                             \
      false, int(tpr), int(ch), int(rpt), int(warps), grid, s, reinterpret_cast<const CT*>(x.data_ptr()),                            \
      reinterpret_cast<const CT*>(dy.data_ptr()), w.numel() ? w.data_ptr() : nullptr, nullptr, wbf, 0, reinterpret_cast<CT*>(dx.data_ptr()), \
      rms ? nullptr : mean.data_ptr<float>(), rstd.data_ptr<float>(), pp, grid, M, N, 0.f);
  if (x.scalar_type() == torch::kBFloat16) { MWLN_GO(__nv_bfloat16, 8) }
  else if (x.scalar_type() == torch::kFloat32) { MWLN_GO(float, 4) }
#undef MWLN_GO
  TORCH_CHECK(ok, "B200 LayerNorm rows backward: no build for tpr=", tpr, " ch=", ch, " rpt=", rpt, " warps=", warps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (wgrad) {
    ln_col_reduce_launch(pp, grid, N, dw.data_ptr(), rms ? nullptr : db.data_ptr(), int(gbf16), s);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {dx, dw, db};
}
