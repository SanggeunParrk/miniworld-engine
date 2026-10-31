// General LayerNorm / RMSNorm, the B200 (sm_90+ bulk-copy) form for rows a (sub-)warp holds without idle lanes: a CTA walks tiles
// of RS contiguous rows; each tile reaches shared memory by one cp.async.bulk (TMA bulk copy, mbarrier completion) S stages ahead,
// the rows are normalized in place by TPR-thread row groups (shuffle reductions only), and the tile leaves by one bulk store.
// The register-path kernels (ln_fwd.cu / ln_bwd.cu) issue a 16-byte load per thread and chunk and wait on it; here a single thread
// keeps S tiles (tens of KB) in flight per CTA, which is what reaches the B200's HBM rate on these short rows.
//
//   ln_fwd_bulk  y = (x - mean) rstd w + b (RMS: no mean, no bias); mean / rstd stored per row when asked
//   ln_bwd_bulk  dx (written in place over the dy tile, then one bulk store), and the affine gradients' per-CTA partial rows (fixed
//                order: per thread over its tiles, then the CTA's row groups in order) -- the column pass is ln_bwd.cu's ln_col_reduce.
#include "ln_common.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace mwln {
namespace bulk {

__device__ __forceinline__ uint32_t sa(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ void bar_init(uint64_t* b) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(sa(b))); }
__device__ __forceinline__ void bar_expect(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sa(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bar_wait(uint64_t* b, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n W%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W%=;\n}\n" ::"r"(sa(b)),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(sa(dst)), "l"(src),
               "r"(bytes), "r"(sa(b))
               : "memory");
}
__device__ __forceinline__ void s2g(void* dst, const void* src, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst), "r"(sa(src)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
__device__ __forceinline__ void wait_read_all() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }
__device__ __forceinline__ void wait_all() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
#define MWLN_BULK 1
#else
#define MWLN_BULK 0
#endif

}  // namespace bulk

// smem: S stage buffers of RS rows (bwd: x and dy per stage), then S mbarriers
template <typename T, int TPR, int CHUNKS, int VEC, bool RMS>
__global__ void __launch_bounds__(256) ln_fwd_bulk_kernel(const T* __restrict__ x, const void* __restrict__ w, const void* __restrict__ b,
                                                         int wbf, int bbf, T* __restrict__ y, float* __restrict__ mean_out,
                                                         float* __restrict__ rstd_out, int64_t M, int N, float eps, int RS, int S) {
#if MWLN_BULK
  static_assert(TPR <= 32, "sub-warp rows");
  constexpr int RPC = 256 / TPR;
  constexpr int E = CHUNKS * VEC;
  extern __shared__ __align__(128) unsigned char smem[];
  const int64_t rowb = int64_t(N) * sizeof(T);
  const int64_t tileb = RS * rowb;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + S * tileb);
  const int r_in = threadIdx.x / TPR, lane = threadIdx.x % TPR;
  const float inv_n = 1.f / float(N);
  float wr[E], br[E];
  load_affine<TPR, CHUNKS, VEC>(w, wbf, 1.f, lane, N, wr);
  if constexpr (!RMS) load_affine<TPR, CHUNKS, VEC>(b, bbf, 0.f, lane, N, br);
  const int64_t tiles = (M + RS - 1) / RS;
  const int64_t mine = tiles > blockIdx.x ? (tiles - 1 - blockIdx.x) / gridDim.x + 1 : 0;
  auto tile_of = [&](int64_t i) { return int64_t(blockIdx.x) + i * gridDim.x; };
  auto rows_of = [&](int64_t t) { return int(min(int64_t(RS), M - t * RS)); };
  if (threadIdx.x == 0) {
    for (int s = 0; s < S; ++s) bulk::bar_init(bars + s);
    bulk::fence_init();
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int64_t i = 0; i < mine && i < S; ++i) {
      const int64_t t = tile_of(i);
      const uint32_t bytes = uint32_t(rows_of(t) * rowb);
      bulk::bar_expect(bars + i, bytes);
      bulk::g2s(smem + i * tileb, x + t * RS * int64_t(N), bytes, bars + i);
    }
  }
  for (int64_t i = 0; i < mine; ++i) {
    const int s = int(i % S);
    const int64_t t = tile_of(i);
    const int nr = rows_of(t);
    T* buf = reinterpret_cast<T*>(smem + s * tileb);
    bulk::bar_wait(bars + s, uint32_t((i / S) & 1));
    for (int r0 = 0; r0 < nr; r0 += RPC) {   // uniform over the CTA: every row group runs the shuffles of a row
      const int r = r0 + r_in;
      const bool live = r < nr;
      T* xr = buf + int64_t(live ? r : 0) * N;
      Pack<T, VEC> raw[CHUNKS];
#pragma unroll
      for (int c = 0; c < CHUNKS; ++c) raw[c] = live ? load_pack<T, VEC>(xr + (c * TPR + lane) * VEC) : zero_pack<T, VEC>();
      float mean = 0.f;
      if constexpr (!RMS) {
        float sum = 0.f;
#pragma unroll
        for (int c = 0; c < CHUNKS; ++c)
#pragma unroll
          for (int k = 0; k < VEC; ++k) sum += to_f<T>(raw[c].v[k]);
        mean = row_sum<TPR>(sum, nullptr, 0, lane) * inv_n;
      }
      float q = 0.f;
#pragma unroll
      for (int c = 0; c < CHUNKS; ++c)
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
          const float d = to_f<T>(raw[c].v[k]) - mean;
          q += d * d;
        }
      const float rstd = rsqrtf(row_sum<TPR>(q, nullptr, 0, lane) * inv_n + eps);
      if (live) {
#pragma unroll
        for (int c = 0; c < CHUNKS; ++c) {
          float o[VEC];
#pragma unroll
          for (int k = 0; k < VEC; ++k) {
            const float xn = (to_f<T>(raw[c].v[k]) - mean) * rstd;
            o[k] = RMS ? xn * wr[c * VEC + k] : xn * wr[c * VEC + k] + br[c * VEC + k];
          }
          store_vec<T, VEC>(xr + (c * TPR + lane) * VEC, o);
        }
        if (lane == 0) {
          const int64_t row = t * RS + r;
          if (mean_out) mean_out[row] = mean;
          if (rstd_out) rstd_out[row] = rstd;
        }
      }
    }
    bulk::fence_async();
    __syncthreads();
    if (threadIdx.x == 0) {
      bulk::s2g(y + t * RS * int64_t(N), buf, uint32_t(nr * rowb));
      bulk::commit();
      if (i + S < mine) {   // the stage's next tile, once the store has read it out
        const int64_t tn = tile_of(i + S);
        const uint32_t bytes = uint32_t(rows_of(tn) * rowb);
        bulk::wait_read_all();
        bulk::bar_expect(bars + s, bytes);
        bulk::g2s(smem + s * tileb, x + tn * RS * int64_t(N), bytes, bars + s);
      }
    }
  }
  if (threadIdx.x == 0) bulk::wait_all();
#endif
}

template <typename T, int TPR, int CHUNKS, int VEC, bool RMS>
__global__ void __launch_bounds__(256) ln_bwd_bulk_kernel(const T* __restrict__ dy, const T* __restrict__ x, const void* __restrict__ w,
                                                         int wbf, const float* __restrict__ mean, const float* __restrict__ rstd,
                                                         T* __restrict__ dx, float* __restrict__ dw_part, float* __restrict__ db_part,
                                                         int64_t M, int N, int RS, int S) {
#if MWLN_BULK
  static_assert(TPR <= 32, "sub-warp rows");
  constexpr int RPC = 256 / TPR;
  constexpr int E = CHUNKS * VEC;
  extern __shared__ __align__(128) unsigned char smem[];
  const int64_t rowb = int64_t(N) * sizeof(T);
  const int64_t tileb = RS * rowb;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + 2 * S * tileb);
  const int r_in = threadIdx.x / TPR, lane = threadIdx.x % TPR;
  const float inv_n = 1.f / float(N);
  const bool wgrad = dw_part != nullptr;
  float aw[E], ab[E], wr[E];
  load_affine<TPR, CHUNKS, VEC>(w, wbf, 1.f, lane, N, wr);
#pragma unroll
  for (int e = 0; e < E; ++e) aw[e] = ab[e] = 0.f;
  const int64_t tiles = (M + RS - 1) / RS;
  const int64_t mine = tiles > blockIdx.x ? (tiles - 1 - blockIdx.x) / gridDim.x + 1 : 0;
  auto tile_of = [&](int64_t i) { return int64_t(blockIdx.x) + i * gridDim.x; };
  auto rows_of = [&](int64_t t) { return int(min(int64_t(RS), M - t * RS)); };
  auto xbuf = [&](int s) { return smem + (2 * s) * tileb; };
  auto dbuf = [&](int s) { return smem + (2 * s + 1) * tileb; };
  if (threadIdx.x == 0) {
    for (int s = 0; s < S; ++s) bulk::bar_init(bars + s);
    bulk::fence_init();
  }
  __syncthreads();
  auto issue = [&](int s, int64_t t) {
    const uint32_t bytes = uint32_t(rows_of(t) * rowb);
    bulk::bar_expect(bars + s, 2 * bytes);
    bulk::g2s(xbuf(s), x + t * RS * int64_t(N), bytes, bars + s);
    bulk::g2s(dbuf(s), dy + t * RS * int64_t(N), bytes, bars + s);
  };
  if (threadIdx.x == 0)
    for (int64_t i = 0; i < mine && i < S; ++i) issue(int(i), tile_of(i));
  for (int64_t i = 0; i < mine; ++i) {
    const int s = int(i % S);
    const int64_t t = tile_of(i);
    const int nr = rows_of(t);
    const T* xb = reinterpret_cast<const T*>(xbuf(s));
    T* db = reinterpret_cast<T*>(dbuf(s));
    float mu_r[1] = {0.f}, rs_r[1] = {0.f};
    bulk::bar_wait(bars + s, uint32_t((i / S) & 1));
    for (int r0 = 0; r0 < nr; r0 += RPC) {
      const int r = r0 + r_in;
      const bool live = r < nr;
      const int64_t row = t * RS + (live ? r : 0);
      mu_r[0] = live && !RMS ? __ldg(mean + row) : 0.f;
      rs_r[0] = live ? __ldg(rstd + row) : 0.f;
      Pack<T, VEC> xp[CHUNKS], dp[CHUNKS];
#pragma unroll
      for (int c = 0; c < CHUNKS; ++c) {
        const int col = (c * TPR + lane) * VEC;
        xp[c] = live ? load_pack<T, VEC>(xb + int64_t(r) * N + col) : zero_pack<T, VEC>();
        dp[c] = live ? load_pack<T, VEC>(db + int64_t(r) * N + col) : zero_pack<T, VEC>();
      }
      float xh[E], g[E];
      float s1 = 0.f, s2 = 0.f;
#pragma unroll
      for (int c = 0; c < CHUNKS; ++c)
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
          const int e = c * VEC + k;
          const float dv = to_f<T>(dp[c].v[k]);
          xh[e] = (to_f<T>(xp[c].v[k]) - mu_r[0]) * rs_r[0];
          g[e] = dv * wr[e];
          s1 += g[e];
          s2 += g[e] * xh[e];
          if (wgrad) {
            aw[e] += dv * xh[e];
            if constexpr (!RMS) ab[e] += dv;
          }
        }
      if constexpr (RMS) s1 = 0.f;
      else s1 = row_sum<TPR>(s1, nullptr, 0, lane) * inv_n;
      s2 = row_sum<TPR>(s2, nullptr, 0, lane) * inv_n;
      if (live) {
#pragma unroll
        for (int c = 0; c < CHUNKS; ++c) {
          float o[VEC];
#pragma unroll
          for (int k = 0; k < VEC; ++k) {
            const int e = c * VEC + k;
            o[k] = rs_r[0] * (g[e] - s1 - xh[e] * s2);
          }
          store_vec<T, VEC>(db + int64_t(r) * N + (c * TPR + lane) * VEC, o);
        }
      }
    }
    bulk::fence_async();
    __syncthreads();
    if (threadIdx.x == 0) {
      bulk::s2g(dx + t * RS * int64_t(N), db, uint32_t(nr * rowb));
      bulk::commit();
      if (i + S < mine) {
        bulk::wait_read_all();
        issue(s, tile_of(i + S));
      }
    }
  }
  if (threadIdx.x == 0) bulk::wait_all();
  if (!wgrad) return;
  __syncthreads();   // every stage consumed and stored: the stage memory holds the partial sums now
  float* part_smem = reinterpret_cast<float*>(smem);
  float* out_w = dw_part + int64_t(blockIdx.x) * N;
  float* out_b = RMS ? nullptr : db_part + int64_t(blockIdx.x) * N;
#pragma unroll
  for (int pass = 0; pass < (RMS ? 1 : 2); ++pass) {
    const float* acc = pass == 0 ? aw : ab;
    float* out = pass == 0 ? out_w : out_b;
#pragma unroll
    for (int e = 0; e < E; ++e) part_smem[(r_in * TPR + lane) * E + e] = acc[e];
    __syncthreads();
    for (int idx = threadIdx.x; idx < TPR * E; idx += 256) {
      float tt = 0.f;
      for (int r = 0; r < RPC; ++r) tt += part_smem[r * TPR * E + idx];
      const int l = idx / E, e = idx % E;
      out[((e / VEC) * TPR + l) * VEC + e % VEC] = tt;
    }
    __syncthreads();
  }
#endif
}

template <typename T, int VEC, bool RMS>
bool dispatch_bulk(bool fwd, int tpr, int chunks, dim3 grid, size_t smem, cudaStream_t st, const T* x, const T* dy, const void* w,
                   const void* b, int wbf, int bbf, T* out, float* mean, float* rstd, float* dwp, float* dbp, int64_t M, int N, float eps,
                   int RS, int S) {
#define MWLN_BK(TPR_, CH)                                                                                                       \
  if (tpr == TPR_ && chunks == CH) {                                                                                            \
    if (fwd) {                                                                                                                  \
      auto k = ln_fwd_bulk_kernel<T, TPR_, CH, VEC, RMS>;                                                                       \
      cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));                                          \
      k<<<grid, 256, smem, st>>>(x, w, b, wbf, bbf, out, mean, rstd, M, N, eps, RS, S);                                         \
    } else {                                                                                                                    \
      auto k = ln_bwd_bulk_kernel<T, TPR_, CH, VEC, RMS>;                                                                       \
      cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));                                          \
      k<<<grid, 256, smem, st>>>(dy, x, w, wbf, mean, rstd, out, dwp, dbp, M, N, RS, S);                                        \
    }                                                                                                                           \
    return true;                                                                                                                \
  }
#define MWLN_BK_T(TPR_) MWLN_BK(TPR_, 1) MWLN_BK(TPR_, 2) MWLN_BK(TPR_, 3) MWLN_BK(TPR_, 4)
  MWLN_BK_T(4) MWLN_BK_T(8) MWLN_BK_T(16) MWLN_BK_T(32)
#undef MWLN_BK_T
#undef MWLN_BK
  return false;
}

}  // namespace mwln

// The bulk forms (B200; host checks: 16-byte aligned rows, a sub-warp layout). `rs` rows a tile, `stages` tiles in flight, `ctas` CTAs
// a SM. fwd: (y, mean, rstd) like ln_fwd. bwd: (dx, dw, db) like ln_bwd (partials -> ln_col_reduce in `gbf16` / fp32).
void ln_col_reduce_launch(const float* part, int P, int N, void* out, void* out2, int obf, cudaStream_t s);

static void bulk_grid(int64_t M, int rs, int ctas, dim3& grid) {
  const int64_t tiles = (M + rs - 1) / rs;
  const int64_t cap = int64_t(mwln::num_sms()) * ctas;
  grid = dim3(unsigned(tiles < cap ? tiles : cap));
}

std::vector<torch::Tensor> ln_fwd_bulk(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps, bool save, bool rms, int64_t tpr,
                                       int64_t chunks, int64_t rs, int64_t stages, int64_t ctas) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "x");
  const int64_t M = x.size(0);
  const int N = int(x.size(1));
  TORCH_CHECK((N * x.element_size()) % 16 == 0, "bulk rows must be 16-byte multiples");
  auto y = torch::empty_like(x);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto mean = save && !rms ? torch::empty({M}, f32) : torch::empty({0}, f32);
  auto rstd = save ? torch::empty({M}, f32) : torch::empty({0}, f32);
  if (M == 0) return {y, mean, rstd};
  for (auto* t : {&w, &b})
    if (t->numel() && (reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 || !t->is_contiguous())) *t = t->contiguous().clone();
  const int wbf = w.numel() && w.scalar_type() == torch::kBFloat16, bbf = b.numel() && b.scalar_type() == torch::kBFloat16;
  dim3 grid;
  bulk_grid(M, int(rs), int(ctas), grid);
  const size_t smem = size_t(stages) * rs * N * x.element_size() + 8 * stages;
  auto s = at::cuda::getCurrentCUDAStream();
  bool ok = false;
#define MWLN_GO(CT, V)                                                                                                             \
  ok = rms ? mwln::dispatch_bulk<CT, V, true>(true, int(tpr), int(chunks), grid, smem, s, reinterpret_cast<const CT*>(x.data_ptr()), \
                                              nullptr, w.numel() ? w.data_ptr() : nullptr, nullptr, wbf, 0,                         \
                                              reinterpret_cast<CT*>(y.data_ptr()), nullptr, save ? rstd.data_ptr<float>() : nullptr, \
                                              nullptr, nullptr, M, N, float(eps), int(rs), int(stages))                             \
           : mwln::dispatch_bulk<CT, V, false>(true, int(tpr), int(chunks), grid, smem, s, reinterpret_cast<const CT*>(x.data_ptr()), \
                                               nullptr, w.numel() ? w.data_ptr() : nullptr, b.numel() ? b.data_ptr() : nullptr, wbf, bbf, \
                                               reinterpret_cast<CT*>(y.data_ptr()), save ? mean.data_ptr<float>() : nullptr,          \
                                               save ? rstd.data_ptr<float>() : nullptr, nullptr, nullptr, M, N, float(eps), int(rs),  \
                                               int(stages));
  if (x.scalar_type() == torch::kBFloat16) { MWLN_GO(__nv_bfloat16, 8) }
  else if (x.scalar_type() == torch::kHalf) { MWLN_GO(__half, 8) }
  else if (x.scalar_type() == torch::kFloat32) { MWLN_GO(float, 4) }
#undef MWLN_GO
  TORCH_CHECK(ok, "general LayerNorm bulk forward: no build for tpr=", tpr, " chunks=", chunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, mean, rstd};
}

std::vector<torch::Tensor> ln_bwd_bulk(torch::Tensor dy, torch::Tensor x, torch::Tensor w, torch::Tensor mean, torch::Tensor rstd,
                                       bool wgrad, bool rms, int64_t tpr, int64_t chunks, int64_t rs, int64_t stages, int64_t ctas,
                                       bool gbf16) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && dy.is_contiguous() && dy.sizes() == x.sizes() &&
                  dy.scalar_type() == x.scalar_type() && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(dy.data_ptr()) % 16 == 0,
              "dy, x");
  const int64_t M = x.size(0);
  const int N = int(x.size(1));
  TORCH_CHECK((N * x.element_size()) % 16 == 0, "bulk rows must be 16-byte multiples");
  auto dx = torch::empty_like(x);
  auto f32 = x.options().dtype(torch::kFloat32);
  auto gopt = x.options().dtype(gbf16 ? torch::kBFloat16 : torch::kFloat32);
  if (w.numel() && (reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 || !w.is_contiguous())) w = w.contiguous().clone();
  const int wbf = w.numel() && w.scalar_type() == torch::kBFloat16;
  dim3 grid;
  bulk_grid(M, int(rs), int(ctas), grid);
  const int P = int(grid.x);
  auto part = wgrad ? torch::empty({rms ? 1 : 2, P, N}, f32) : torch::empty({0}, f32);
  auto dw = wgrad ? (M == 0 ? torch::zeros({N}, gopt) : torch::empty({N}, gopt)) : torch::empty({0}, f32);
  auto db = wgrad && !rms ? (M == 0 ? torch::zeros({N}, gopt) : torch::empty({N}, gopt)) : torch::empty({0}, f32);
  if (M == 0) return {dx, dw, db};
  // the stage memory also holds the partial sums at the end: 256 threads x E floats
  const size_t stage = size_t(stages) * 2 * rs * N * x.element_size();
  const size_t need_part = size_t(256) * size_t(chunks) * (16 / x.element_size()) * sizeof(float);
  const size_t smem = std::max(stage, need_part) + 8 * stages;
  float* dwp = wgrad ? part.data_ptr<float>() : nullptr;
  float* dbp = wgrad && !rms ? part.data_ptr<float>() + int64_t(P) * N : nullptr;
  auto s = at::cuda::getCurrentCUDAStream();
  bool ok = false;
#define MWLN_GO(CT, V)                                                                                                               \
  ok = (rms ? mwln::dispatch_bulk<CT, V, true> : mwln::dispatch_bulk<CT, V, false>)(                                                 \
      false, int(tpr), int(chunks), grid, smem, s, reinterpret_cast<const CT*>(x.data_ptr()), reinterpret_cast<const CT*>(dy.data_ptr()), \
      w.numel() ? w.data_ptr() : nullptr, nullptr, wbf, 0, reinterpret_cast<CT*>(dx.data_ptr()), rms ? nullptr : mean.data_ptr<float>(), \
      rstd.data_ptr<float>(), dwp, dbp, M, N, 0.f, int(rs), int(stages));
  if (x.scalar_type() == torch::kBFloat16) { MWLN_GO(__nv_bfloat16, 8) }
  else if (x.scalar_type() == torch::kHalf) { MWLN_GO(__half, 8) }
  else if (x.scalar_type() == torch::kFloat32) { MWLN_GO(float, 4) }
#undef MWLN_GO
  TORCH_CHECK(ok, "general LayerNorm bulk backward: no build for tpr=", tpr, " chunks=", chunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (wgrad) {
    ln_col_reduce_launch(dwp, P, N, dw.data_ptr(), rms ? nullptr : db.data_ptr(), int(gbf16), s);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {dx, dw, db};
}
