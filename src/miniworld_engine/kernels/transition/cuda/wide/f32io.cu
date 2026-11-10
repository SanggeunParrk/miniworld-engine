// Torch binding of the fp32 Transition forward at one channel width F32_D and hidden multiple F32_N: kernels/transition_fwd_f32io.cu (D 64 /
// 128, xn in the A fragments) and kernels/transition_fwd_f32xs.cu (D 256, xn in shared memory); inference (fwd) and the training forward that
// saves xn / rstd / c1 (fwd_train, D 128 / 256). The builds of a kernel are template instances; ``variant`` picks one.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <array>
#include <map>
#include <mutex>
#include <stdexcept>

#include "kernels/transition_fwd_f32io.cu"
#include "kernels/transition_fwd_f32xs.cu"
#include "kernels/transition_lnsg_f32.cu"
#include "kernels/transition_bwd_f32dx.cu"
#include "kernels/transition_bwd_f32dw.cu"
#include "launch_util.cuh"

#ifndef F32_D
#error "F32_D (64 / 128 / 256) must be defined"
#endif
#ifndef F32_N
#define F32_N 4
#endif

namespace {

using EncodeTiled = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                 const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                 CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

EncodeTiled encoder() {
  static EncodeTiled fn = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
    cudaError_t e = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    if (e != cudaSuccess || p == nullptr) throw std::runtime_error("cuTensorMapEncodeTiled unavailable: the driver predates TMA");
    return reinterpret_cast<EncodeTiled>(p);
  }();
  return fn;
}

// 2-D row-major contiguous [rows][cols] descriptor (fp32 or fp16), 128-byte swizzle, box innermost-first; cached on (pointer, shape, box)
const CUtensorMap& tile_map(const torch::Tensor& t, uint32_t box_inner, uint32_t box_outer) {
  using Key = std::array<uint64_t, 6>;
  static std::map<Key, CUtensorMap> cache;
  static std::mutex lock;
  const bool f32 = t.scalar_type() == torch::kFloat32;
  const uint64_t inner = (uint64_t)t.size(1), outer = (uint64_t)t.size(0);
  const Key key{reinterpret_cast<uint64_t>(t.data_ptr()), inner, outer, box_inner, box_outer, (uint64_t)f32};
  std::lock_guard<std::mutex> guard(lock);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  CUtensorMap map{};
  const cuuint64_t dims[2] = {inner, outer};
  const cuuint64_t strides[1] = {inner * (f32 ? 4 : 2)};
  const cuuint32_t box[2] = {box_inner, box_outer};
  const cuuint32_t elem[2] = {1, 1};
  if (cudaSetDevice(t.get_device()) != cudaSuccess) throw std::runtime_error("cudaSetDevice failed before cuTensorMapEncodeTiled");
  CUresult r = encoder()(&map, f32 ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 : CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, t.data_ptr(), dims, strides, box,
                         elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: result=", int(r), " inner=", inner, " outer=", outer,
              " box=", box_inner, "x", box_outer, " pointer_mod_128=", reinterpret_cast<uintptr_t>(t.data_ptr()) % 128);
  return cache.emplace(key, map).first->second;
}

void expect(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("transition f32io sm90a: ") + what);
}
void check_2d(const torch::Tensor& t, torch::ScalarType dt, int64_t rows, int64_t cols, const char* what) {
  expect(t.is_cuda() && t.is_contiguous() && t.scalar_type() == dt && t.dim() == 2 && t.size(0) == rows && t.size(1) == cols, what);
}

constexpr int64_t D = F32_D, H = F32_N * F32_D;

// the saves of the training forward (xn fp32 [M, D], rstd / c1 fp32 [M]); null pointers for inference
struct Saves { float* xn = nullptr; float* rstd = nullptr; float* c1 = nullptr; };

template <int HS, int NWG, int NBUF, int CPS, bool SAVE = false>
void launch(const torch::Tensor& x, const torch::Tensor& gamma, const torch::Tensor& beta, const torch::Tensor& wa, const torch::Tensor& wb,
            const torch::Tensor& wst, const torch::Tensor& sc, const torch::Tensor& out, float eps, int sms, Saves sv = {}) {
  using C = F32Cfg<F32_D, F32_N, HS, NWG, NBUF, CPS>;
  auto fn = transition_fwd_f32io<F32_D, F32_N, HS, NWG, NBUF, CPS, SAVE>;
  static const bool ready = (wide_smem_optin(reinterpret_cast<const void*>(fn), C::SMEM, "f32io fwd"), true);
  (void)ready;
  const int64_t M = x.size(0);
  const int tiles = (int)((M + C::ROWS - 1) / C::ROWS), grid = tiles < CPS * sms ? tiles : CPS * sms;
  fn<<<grid, 128 * NWG, C::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      tile_map(x, 32, 64), tile_map(wa, 64, HS), tile_map(wb, 64, HS), tile_map(wst, 64, HS), tile_map(out, 32, 64),
      gamma.data_ptr<float>(), beta.data_ptr<float>(), sc.data_ptr<float>(), tiles, eps, sv.xn, sv.rstd, sv.c1, (int)M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// D = 256: xn in shared memory (transition_fwd_f32xs.cu), x read and y written in the fragment layout; 128-row tiles, one CTA per SM
template <int HS, int NSLOT, bool SAVE = false>
void launch_xs(const torch::Tensor& x, const torch::Tensor& gamma, const torch::Tensor& beta, const torch::Tensor& wa, const torch::Tensor& wb,
               const torch::Tensor& wst, const torch::Tensor& sc, torch::Tensor& out, float eps, int sms, Saves sv = {}) {
  using C = F32XsCfg<F32_D, F32_N, HS, NSLOT>;
  auto fn = transition_fwd_f32xs<F32_D, F32_N, HS, NSLOT, SAVE>;
  static const bool ready = (wide_smem_optin(reinterpret_cast<const void*>(fn), C::SMEM, "f32xs fwd"), true);
  (void)ready;
  const int64_t M = x.size(0);
  const int tiles = (int)((M + C::ROWS - 1) / C::ROWS), grid = tiles < sms ? tiles : sms;
  fn<<<grid, 256, C::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      tile_map(wa, 64, HS), tile_map(wb, 64, HS), tile_map(wst, 64, HS), x.data_ptr<float>(), out.data_ptr<float>(),
      gamma.data_ptr<float>(), beta.data_ptr<float>(), sc.data_ptr<float>(), sv.xn, sv.rstd, sv.c1, (int)M, tiles, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_args(const torch::Tensor& x, const torch::Tensor& gamma, const torch::Tensor& beta, const torch::Tensor& wa, const torch::Tensor& wb,
                const torch::Tensor& wst, const torch::Tensor& sc) {
  const int64_t M = x.size(0);
  check_2d(x, torch::kFloat32, M, D, "x must be contiguous fp32 [M, D]");
  expect(M > 0, "empty input");
  expect(gamma.is_cuda() && gamma.is_contiguous() && gamma.scalar_type() == torch::kFloat32 && gamma.numel() == D, "gamma must be fp32 [D]");
  expect(beta.is_cuda() && beta.is_contiguous() && beta.scalar_type() == torch::kFloat32 && beta.numel() == D, "beta must be fp32 [D]");
  check_2d(wa, torch::kHalf, H, D, "wa must be fp16 [H, D]");
  check_2d(wb, torch::kHalf, H, D, "wb must be fp16 [H, D]");
  check_2d(wst, torch::kHalf, H, D, "wst must be fp16 Ws^T [H, D]");
  expect(sc.is_cuda() && sc.scalar_type() == torch::kFloat32 && sc.numel() == 4 && sc.is_contiguous(), "sc must be fp32 [4]");
}

template <bool SAVE>
void run(const torch::Tensor& x, const torch::Tensor& gamma, const torch::Tensor& beta, const torch::Tensor& wa, const torch::Tensor& wb,
         const torch::Tensor& wst, const torch::Tensor& sc, torch::Tensor& out, float e, int64_t variant, int s, Saves sv) {
#if F32_D == 64
  expect(!SAVE, "the D = 64 build is inference only");
  switch (variant) {
    case 0: launch<64, 1, 2, 2>(x, gamma, beta, wa, wb, wst, sc, out, e, s); break;
    case 1: launch<64, 1, 1, 3>(x, gamma, beta, wa, wb, wst, sc, out, e, s); break;
    case 2: launch<64, 2, 2, 1>(x, gamma, beta, wa, wb, wst, sc, out, e, s); break;
    default: expect(false, "variant must be 0 .. 2 at D = 64");
  }
#elif F32_D == 128
  if constexpr (SAVE) {
    expect(variant == 1, "the D = 128 training forward is variant 1");
    launch<64, 2, 2, 1, true>(x, gamma, beta, wa, wb, wst, sc, out, e, s, sv);
  } else {
    switch (variant) {
      case 0: launch<32, 1, 1, 2>(x, gamma, beta, wa, wb, wst, sc, out, e, s); break;
      case 1: launch<64, 2, 2, 1>(x, gamma, beta, wa, wb, wst, sc, out, e, s); break;
      case 2: launch<64, 2, 1, 1>(x, gamma, beta, wa, wb, wst, sc, out, e, s); break;
      default: expect(false, "variant must be 0 .. 2 at D = 128");
    }
  }
#elif F32_D == 256
  switch (variant) {
    case 1: launch_xs<32, 3, SAVE>(x, gamma, beta, wa, wb, wst, sc, out, e, s, sv); break;
    case 2: launch_xs<32, 2, SAVE>(x, gamma, beta, wa, wb, wst, sc, out, e, s, sv); break;
    default: expect(false, "variant must be 1 .. 2 at D = 256");
  }
#elif F32_D == 512
  expect(false, "D = 512 runs the two-kernel forward: expand (here) + the generic squeeze_res_h16");
#else
#error "F32_D must be 64, 128, 256 or 512"
#endif
}

}  // namespace

// x fp32 [M, D]; gamma / beta fp32 [D]; wa16, wb16 fp16 [H, D] (scaled); wst16 = (Ws / s_ws)^T fp16 [H, D]; sc fp32 [4]. Returns x + transition(x).
torch::Tensor fwd(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, torch::Tensor wa, torch::Tensor wb, torch::Tensor wst,
                  torch::Tensor sc, double eps, int64_t variant, int64_t sms) {
  check_args(x, gamma, beta, wa, wb, wst, sc);
  auto out = torch::empty_like(x);
  run<false>(x, gamma, beta, wa, wb, wst, sc, out, (float)eps, variant, (int)sms, Saves{});
  return out;
}

// The training forward: (out, xn, rstd, c1), xn the TF32 value of the fp16 operand the products read (see the kernels' headers).
std::vector<torch::Tensor> fwd_train(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, torch::Tensor wa, torch::Tensor wb,
                                     torch::Tensor wst, torch::Tensor sc, double eps, int64_t variant, int64_t sms) {
  check_args(x, gamma, beta, wa, wb, wst, sc);
  const int64_t M = x.size(0);
  auto out = torch::empty_like(x), xn = torch::empty_like(x);
  auto rstd = torch::empty({M}, x.options()), c1 = torch::empty({M}, x.options());
  Saves sv{xn.data_ptr<float>(), rstd.data_ptr<float>(), c1.data_ptr<float>()};
  run<true>(x, gamma, beta, wa, wb, wst, sc, out, (float)eps, variant, (int)sms, sv);
  return {out, xn, rstd, c1};
}

// ------------------------------------------------------------------------------------------------ the fp16 weight forms and scales (two launches)
// stats (zeroed first): [0] max|gamma|, [1] max|beta|, [2] max(|Wa|, |Wb|), [3] max row L1 of Wa, Wb, [4] max|Ws| -- nonnegative floats, so
// atomicMax on their bit patterns is the float max (order-free: deterministic).
__global__ void __launch_bounds__(256) f32_forms_stats(const float* __restrict__ gamma, const float* __restrict__ beta, const float* __restrict__ wa,
                                                      const float* __restrict__ wb, const float* __restrict__ ws, float* __restrict__ st) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31, nwarps = (gridDim.x * blockDim.x) >> 5;
  float mg = 0.f, mb = 0.f, mw = 0.f, ml = 0.f, ms = 0.f;
  if (warp == 0)
    for (int c = lane; c < D; c += 32) { mg = fmaxf(mg, fabsf(gamma[c])); mb = fmaxf(mb, fabsf(beta[c])); }
  for (int r = warp; r < 2 * H; r += nwarps) {           // rows of Wa then Wb: absmax and L1
    const float* p = (r < H ? wa + (size_t)r * D : wb + (size_t)(r - H) * D);
    float l1 = 0.f;
    for (int c = lane; c < D; c += 32) { const float v = fabsf(p[c]); l1 += v; mw = fmaxf(mw, v); }
#pragma unroll
    for (int o = 16; o; o >>= 1) l1 += __shfl_xor_sync(0xffffffffu, l1, o);
    ml = fmaxf(ml, l1);
  }
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < D * H; i += (long long)gridDim.x * blockDim.x) ms = fmaxf(ms, fabsf(ws[i]));
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    mg = fmaxf(mg, __shfl_xor_sync(0xffffffffu, mg, o)); mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, o));
    mw = fmaxf(mw, __shfl_xor_sync(0xffffffffu, mw, o)); ms = fmaxf(ms, __shfl_xor_sync(0xffffffffu, ms, o));
  }
  if (lane == 0) {
    int* si = reinterpret_cast<int*>(st);
    atomicMax(si, __float_as_int(mg)); atomicMax(si + 1, __float_as_int(mb)); atomicMax(si + 2, __float_as_int(mw));
    atomicMax(si + 3, __float_as_int(ml)); atomicMax(si + 4, __float_as_int(ms));
  }
}

// sc = [1 / s_x, s_x s_w, 1 / s_h, s_h s_ws] (the kernels' scales, see transition_fwd_f32io.cu); Wa16 = Wa / s_w, Wb16 = Wb / s_w,
// Wst16 = (Ws / s_ws)^T, all fp16; each scale is max(bound / 30000, 1) (fused_wide_sm90a._F16_BOUND).
__global__ void __launch_bounds__(256) f32_forms_apply(const float* __restrict__ wa, const float* __restrict__ wb, const float* __restrict__ ws,
                                                      const float* __restrict__ st, __half* __restrict__ wa16, __half* __restrict__ wb16,
                                                      __half* __restrict__ wst16, float* __restrict__ sc) {
  constexpr float B = 30000.f;
  const float xn_max = sqrtf((float)D) * st[0] + st[1];
  const float s_x = fmaxf(xn_max / B, 1.f), s_w = fmaxf(st[2] / B, 1.f);
  const float a_max = xn_max * st[3];
  const float s_h = fmaxf(a_max * a_max / B, 1.f), s_ws = fmaxf(st[4] / B, 1.f);
  const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x, step = (long long)gridDim.x * blockDim.x;
  if (tid == 0) { sc[0] = 1.f / s_x; sc[1] = s_x * s_w; sc[2] = 1.f / s_h; sc[3] = s_h * s_ws; }
  for (long long i = tid; i < D * H; i += step) {
    wa16[i] = __float2half_rn(wa[i] / s_w);
    wb16[i] = __float2half_rn(wb[i] / s_w);
    const long long h = i / D, d = i % D;                 // Wst16[h][d] = Ws[d][h] / s_ws
    wst16[i] = __float2half_rn(ws[d * H + h] / s_ws);
  }
}

// (wa16, wb16, wst16, sc) from fp32 gamma / beta [D], Wa / Wb [H, D], Ws [D, H]: device arithmetic only (graph-capturable, no host sync).
std::vector<torch::Tensor> forms(torch::Tensor gamma, torch::Tensor beta, torch::Tensor wa, torch::Tensor wb, torch::Tensor ws) {
  check_2d(wa, torch::kFloat32, H, D, "wa must be fp32 [H, D]");
  check_2d(wb, torch::kFloat32, H, D, "wb must be fp32 [H, D]");
  check_2d(ws, torch::kFloat32, D, H, "ws must be fp32 [D, H]");
  expect(gamma.numel() == D && beta.numel() == D && gamma.is_contiguous() && beta.is_contiguous() && gamma.scalar_type() == torch::kFloat32 &&
         beta.scalar_type() == torch::kFloat32, "gamma / beta fp32 [D]");
  auto f32 = wa.options(), f16 = wa.options().dtype(torch::kHalf);
  auto st = torch::empty({8}, f32), sc = torch::empty({4}, f32);
  auto wa16 = torch::empty({H, D}, f16), wb16 = torch::empty({H, D}, f16), wst16 = torch::empty({H, D}, f16);
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaMemsetAsync(st.data_ptr(), 0, 8 * sizeof(float), stream);
  f32_forms_stats<<<64, 256, 0, stream>>>(gamma.data_ptr<float>(), beta.data_ptr<float>(), wa.data_ptr<float>(), wb.data_ptr<float>(),
                                           ws.data_ptr<float>(), st.data_ptr<float>());
  f32_forms_apply<<<256, 256, 0, stream>>>(wa.data_ptr<float>(), wb.data_ptr<float>(), ws.data_ptr<float>(), st.data_ptr<float>(),
                                           reinterpret_cast<__half*>(wa16.data_ptr()), reinterpret_cast<__half*>(wb16.data_ptr()),
                                           reinterpret_cast<__half*>(wst16.data_ptr()), sc.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {wa16, wb16, wst16, sc};
}

#if F32_D == 512
// D = 512 forward, first half (kernels/transition_lnsg_f32.cu): h16 = fp16(silu(a) b / s_h) [M, H] with the LayerNorm inside the expand GEMM;
// returns [h16, xn, rstd, c1] (the last three 1-element placeholders unless ``save``). The second half is the generic squeeze_res_h16.
std::vector<torch::Tensor> expand(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, torch::Tensor wa, torch::Tensor wb, torch::Tensor sc,
                                  double eps, bool save, int64_t sms) {
  const int64_t M = x.size(0);
  check_args(x, gamma, beta, wa, wb, wa, sc);
  auto h = torch::empty({M, H}, x.options().dtype(torch::kHalf));
  const int64_t ms = save ? M : 1;
  auto xn = torch::empty({ms, save ? D : 1}, x.options()), rstd = torch::empty({ms}, x.options()), c1 = torch::empty({ms}, x.options());
  constexpr int NSTAGE = 4;
  using C = LnsgF32Cfg<F32_D, NSTAGE>;
  const int blocks = (int)((M + C::ROWS - 1) / C::ROWS), grid = blocks < sms ? blocks : (int)sms;
  auto go = [&](auto fn) {
    wide_smem_optin(reinterpret_cast<const void*>(fn), C::SMEM, "lnsg f32");
    fn<<<grid, 256, C::SMEM, at::cuda::getCurrentCUDAStream()>>>(tile_map(wa, 64, 64), tile_map(wb, 64, 64), tile_map(h, 64, 64), x.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(), sc.data_ptr<float>(), save ? xn.data_ptr<float>() : nullptr,
        save ? rstd.data_ptr<float>() : nullptr, save ? c1.data_ptr<float>() : nullptr, (int)M, (int)H, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };
  if (save) go(transition_lnsg_f32<F32_D, NSTAGE, true>);
  else go(transition_lnsg_f32<F32_D, NSTAGE, false>);
  return {h, xn, rstd, c1};
}
#endif

#if F32_D == 128
// D = 128 backward dgrad (kernels/transition_bwd_f32dx.cu): returns [dx, h, dab, dgamma, dbeta]. xn, x, dy fp32 [M, D]; wst_p = Ws^T [H, D] and
// wabt_p = [Wa; Wb]^T [D, 2 H] with their K columns permuted per 8 (0, 2, 4, 6, 1, 3, 5, 7), wa / wb [H, D]; all weights TF32-rounded.
std::vector<torch::Tensor> bwd_dx(torch::Tensor dy, torch::Tensor xn, torch::Tensor x, torch::Tensor rstd, torch::Tensor c1, torch::Tensor gamma,
                                  torch::Tensor wst_p, torch::Tensor wa, torch::Tensor wb, torch::Tensor wabt_p, int64_t sms, bool store) {
  const int64_t M = x.size(0);
  for (auto* t : {&dy, &xn, &x}) check_2d(*t, torch::kFloat32, M, D, "dy / xn / x must be contiguous fp32 [M, D]");
  check_2d(wst_p, torch::kFloat32, H, D, "wst_p fp32 [H, D]");
  check_2d(wa, torch::kFloat32, H, D, "wa fp32 [H, D]");
  check_2d(wb, torch::kFloat32, H, D, "wb fp32 [H, D]");
  check_2d(wabt_p, torch::kFloat32, D, 2 * H, "wabt_p fp32 [D, 2 H]");
  expect(rstd.numel() == M && c1.numel() == M && gamma.numel() == D && rstd.is_contiguous() && c1.is_contiguous() && gamma.is_contiguous(), "stats");
  expect(M > 0, "empty input");
  using C = F32DxCfg<F32_D, F32_N>;
  auto fn = store ? transition_bwd_f32dx<F32_D, F32_N, true> : transition_bwd_f32dx<F32_D, F32_N, false>;
  wide_smem_optin(reinterpret_cast<const void*>(fn), C::SMEM, "f32 bwd dx");
  const int tiles = (int)((M + C::ROWS - 1) / C::ROWS), grid = tiles < sms ? tiles : (int)sms;
  auto f32 = x.options();
  auto dx = torch::empty_like(x), h = torch::empty({store ? M : 1, H}, f32), dab = torch::empty({store ? M : 1, 2 * H}, f32);
  auto part = torch::empty({(int64_t)grid * 8, 2, D}, f32), dg = torch::empty({D}, f32), db = torch::empty({D}, f32);
  auto stream = at::cuda::getCurrentCUDAStream();
  fn<<<grid, 384, C::SMEM, stream>>>(tile_map(xn, 32, 64), tile_map(wst_p, 32, C::HS), tile_map(wa, 32, C::HS), tile_map(wb, 32, C::HS),
                                     tile_map(wabt_p, 32, D), dy.data_ptr<float>(), x.data_ptr<float>(), rstd.data_ptr<float>(),
                                     c1.data_ptr<float>(), gamma.data_ptr<float>(), dx.data_ptr<float>(), h.data_ptr<float>(),
                                     dab.data_ptr<float>(), part.data_ptr<float>(), (int)M, tiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  f32_part_reduce<<<(int)((2 * D + 31) / 32), 1024, 0, stream>>>(part.data_ptr<float>(), dg.data_ptr<float>(), db.data_ptr<float>(), grid * 8, (int)D);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dx, h, dab, dg, db};
}

// D = 128 weight gradients (kernels/transition_bwd_f32dw.cu): returns [dWa, dWb, dWs]. dy, xn fp32 [M, D]; wst = Ws^T [H, D], wa / wb [H, D]
// (TF32-rounded); `repl` replicas of each 32-unit hidden slice (grid = H / 32 * repl CTAs).
std::vector<torch::Tensor> bwd_dw(torch::Tensor dy, torch::Tensor xn, torch::Tensor wst, torch::Tensor wa, torch::Tensor wb, int64_t repl) {
  const int64_t M = xn.size(0);
  check_2d(dy, torch::kFloat32, M, D, "dy fp32 [M, D]");
  check_2d(xn, torch::kFloat32, M, D, "xn fp32 [M, D]");
  check_2d(wst, torch::kFloat32, H, D, "wst fp32 [H, D]");
  check_2d(wa, torch::kFloat32, H, D, "wa fp32 [H, D]");
  check_2d(wb, torch::kFloat32, H, D, "wb fp32 [H, D]");
  expect(M > 0 && repl > 0, "empty input / replicas");
  using C = F32DwCfg<F32_D, F32_N>;
  auto fn = transition_bwd_f32dw<F32_D, F32_N>;
  static const bool ready = (wide_smem_optin(reinterpret_cast<const void*>(fn), C::SMEM, "f32 bwd dw"), true);
  (void)ready;
  const int grid = (int)(C::NSL * repl);
  auto f32 = xn.options();
  auto part = torch::empty({(int64_t)grid, D, 3 * C::HS}, f32);
  auto dwa = torch::empty({H, D}, f32), dwb = torch::empty({H, D}, f32), dws = torch::empty({D, H}, f32);
  auto stream = at::cuda::getCurrentCUDAStream();
  fn<<<grid, 384, C::SMEM, stream>>>(tile_map(xn, 32, 64), tile_map(dy, 32, 64), tile_map(wst, 32, C::HS), tile_map(wa, 32, C::HS),
                                     tile_map(wb, 32, C::HS), part.data_ptr<float>(), (int)M, (int)repl);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  f32_dw_reduce<<<128, 256, 0, stream>>>(part.data_ptr<float>(), dwa.data_ptr<float>(), dwb.data_ptr<float>(), dws.data_ptr<float>(), (int)D,
                                          (int)H, (int)repl);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dwa, dwb, dws};
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#if F32_D == 128
  m.def("bwd_dx", &bwd_dx, "fp32 Transition backward dgrad at D 128 (TF32): (dx, h, dab, dgamma, dbeta)");
  m.def("bwd_dw", &bwd_dw, "fp32 Transition weight gradients at D 128 (TF32): (dWa, dWb, dWs)");
#endif
  m.def("forms", &forms, "fp16 weight forms and device scales of the fp32 forward: (wa16, wb16, wst16, sc)", py::arg("gamma"), py::arg("beta"),
        py::arg("wa"), py::arg("wb"), py::arg("ws"));
#if F32_D == 512
  m.def("expand", &expand, "fp32 Transition forward at D 512: LayerNorm + expand + SwiGLU -> fp16 h (scaled)", py::arg("x"), py::arg("gamma"),
        py::arg("beta"), py::arg("wa"), py::arg("wb"), py::arg("sc"), py::arg("eps"), py::arg("save"), py::arg("sms"));
#endif
  m.def("fwd", &fwd, "fp32 inference Transition forward (+ residual), fp16 tensor cores at TF32 rounding", py::arg("x"), py::arg("gamma"),
        py::arg("beta"), py::arg("wa"), py::arg("wb"), py::arg("wst"), py::arg("sc"), py::arg("eps"), py::arg("variant"), py::arg("sms"));
  m.def("fwd_train", &fwd_train, "fp32 training Transition forward: (out, xn, rstd, c1)", py::arg("x"), py::arg("gamma"),
        py::arg("beta"), py::arg("wa"), py::arg("wb"), py::arg("wst"), py::arg("sc"), py::arg("eps"), py::arg("variant"), py::arg("sms"));
  m.attr("width") = F32_D;
  m.attr("n") = F32_N;
}
