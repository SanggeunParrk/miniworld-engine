// transition_wide_sm80.cu -- torch bindings of the wide-width A100 (sm_80) Transition kernels (tw_kernels_sm80.cuh), on the current CUDA stream.
//   ln_fwd       x [M, D] bf16 -> (xn bf16, stats [M, 2] f32 | [0])
//   dual_swiglu  xn [M, K], wa, wb [H, K] -> h [M, H]
//   gate_bwd     xn, wa, wb, dh [M, H] -> (dab [M, 2H] = dA | dB, h [M, H])
//   ln_bwd       d_xn [M, D] f32, x, stats, gamma, dy -> (dx, dgamma, dbeta)
// The squeeze GEMM (+ residual), dh, the weight gradients and d_xn are cuBLAS (the Python side).
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>
#include <vector>

#include "tw_kernels_sm80.cuh"

namespace {
int num_sms() {
  static int n = 0;
  if (n == 0) cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, at::cuda::current_device());
  return n;
}
template <class K>
void set_smem(K kernel, int bytes) {
  static std::vector<const void*> done;
  const void* k = reinterpret_cast<const void*>(kernel);
  if (std::find(done.begin(), done.end(), k) == done.end()) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    done.push_back(k);
  }
}
template <class T> const T* cptr(const torch::Tensor& t) { return reinterpret_cast<const T*>(t.data_ptr()); }
template <class T> T* mptr(torch::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }

// the tile configurations of the dual-B kernels:  <BM, HN (hidden units per CTA tile), WM, WN, stages, BK>
using Cfg0 = a100::DualCfg<128, 64, 2, 2, 4>;       // 4 warps of 64 x (32 a + 32 b), two CTAs per SM
using Cfg1 = a100::DualCfg<128, 128, 2, 4, 3>;      // 8 warps of 64 x (32 a + 32 b), one CTA per SM
using Cfg2 = a100::DualCfg<64, 64, 2, 2, 4>;        // 4 warps of 32 x (32 a + 32 b), small M
using Cfg3 = a100::DualCfg<128, 64, 2, 2, 3, 64>;   // BK = 64
using Cfg4 = a100::DualCfg<128, 32, 4, 1, 4>;       // 4 warps of 32 x (16 a + 16 b)
using Cfg5 = a100::DualCfg<256, 64, 4, 2, 3>;       // 8 warps of 64 x (32 a + 32 b)
using Cfg6 = a100::DualCfg<128, 128, 2, 4, 4>;      // Cfg1 with a 4-slice ring
using Cfg7 = a100::DualCfg<128, 128, 2, 4, 3, 64>;  // Cfg1 with BK = 64
using Cfg8 = a100::DualCfg<256, 64, 4, 2, 4>;       // Cfg5 with a 4-slice ring
using Cfg9 = a100::DualCfg<128, 64, 2, 2, 5>;       // Cfg0 with a 5-slice ring (80 KB: still two CTAs per SM)
using Cfg10 = a100::DualCfg<64, 64, 2, 2, 5>;       // Cfg2 with a 5-slice ring
using Cfg11 = a100::DualCfg<128, 32, 4, 1, 5>;      // Cfg4 with a 5-slice ring

template <class C>
void launch_dual(const a100::DualParams& p, bool gate, int epi, cudaStream_t st) {
  const int hn = C::HPW * C::WN;
  TORCH_CHECK((gate ? C::GATE_SMEM : C::SMEM) <= 163840, "this tile config needs ", (gate ? C::GATE_SMEM : C::SMEM), " B of shared memory for ", gate ? "the gate backward" : "the forward",
              " (an A100 block has 163840)");
  const dim3 grid((unsigned)(p.H / hn), (unsigned)((p.M + C::BM - 1) / C::BM));
#define LAUNCH_EPI(KERNEL, SMEMB)                                                                                                                 \
  if (epi == 0) {                                                                                                                                      \
    set_smem(a100::KERNEL<C, 0>, SMEMB);                                                                                                               \
    a100::KERNEL<C, 0><<<grid, C::NTHR, SMEMB, st>>>(p);                                                                                               \
  } else {                                                                                                                                             \
    set_smem(a100::KERNEL<C, 1>, SMEMB);                                                                                                               \
    a100::KERNEL<C, 1><<<grid, C::NTHR, SMEMB, st>>>(p);                                                                                               \
  }
  if (gate) { LAUNCH_EPI(gate_bwd_kernel, C::GATE_SMEM) }
  else { LAUNCH_EPI(dual_swiglu_kernel, C::SMEM) }
#undef LAUNCH_EPI
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// epi 0: SwiGLU (silu(a) b); 1: a sigmoid(b) (the triangle-multiplication gate)
void dispatch_dual(int cfg, const a100::DualParams& p, bool gate, int epi, cudaStream_t st) {
  switch (cfg) {
    case 0: launch_dual<Cfg0>(p, gate, epi, st); break;
    case 1: launch_dual<Cfg1>(p, gate, epi, st); break;
    case 2: launch_dual<Cfg2>(p, gate, epi, st); break;
    case 3: launch_dual<Cfg3>(p, gate, epi, st); break;
    case 4: launch_dual<Cfg4>(p, gate, epi, st); break;
    case 5: launch_dual<Cfg5>(p, gate, epi, st); break;
    case 6: launch_dual<Cfg6>(p, gate, epi, st); break;
    case 7: launch_dual<Cfg7>(p, gate, epi, st); break;
    case 8: launch_dual<Cfg8>(p, gate, epi, st); break;
    case 9: launch_dual<Cfg9>(p, gate, epi, st); break;
    case 10: launch_dual<Cfg10>(p, gate, epi, st); break;
    case 11: launch_dual<Cfg11>(p, gate, epi, st); break;
    default: TORCH_CHECK(false, "dual tile config ", cfg);
  }
}
int hidden_per_cta(int cfg) {
  switch (cfg) {
    case 0: return Cfg0::HPW * Cfg0::WN;
    case 1: return Cfg1::HPW * Cfg1::WN;
    case 2: return Cfg2::HPW * Cfg2::WN;
    case 3: return Cfg3::HPW * Cfg3::WN;
    case 4: return Cfg4::HPW * Cfg4::WN;
    case 5: return Cfg5::HPW * Cfg5::WN;
    case 6: return Cfg6::HPW * Cfg6::WN;
    case 7: return Cfg7::HPW * Cfg7::WN;
    case 8: return Cfg8::HPW * Cfg8::WN;
    case 9: return Cfg9::HPW * Cfg9::WN;
    case 10: return Cfg10::HPW * Cfg10::WN;
    case 11: return Cfg11::HPW * Cfg11::WN;
  }
  return 64;
}
int bk_of(int cfg) { return cfg == 3 || cfg == 7 ? 64 : 32; }

// the tile configurations of the GEMM + residual kernel: <BM, BN, BK, WM, WN, stages>
using RCfg0 = a100::GCfg<128, 128, 32, 2, 2, 4>;    // 4 warps of 64 x 64, two CTAs per SM
using RCfg1 = a100::GCfg<128, 256, 32, 2, 4, 3>;    // 8 warps of 64 x 64
using RCfg2 = a100::GCfg<64, 128, 32, 2, 2, 4>;     // 4 warps of 32 x 64
using RCfg3 = a100::GCfg<128, 64, 32, 4, 1, 4>;     // 4 warps of 32 x 64, N = 64
using RCfg4 = a100::GCfg<256, 128, 32, 4, 2, 3>;    // 8 warps of 64 x 64
using RCfg5 = a100::GCfg<128, 128, 32, 2, 2, 5>;    // RCfg0 with a 5-slice ring
using RCfg6 = a100::GCfg<128, 256, 32, 2, 4, 4>;    // RCfg1 with a 4-slice ring
using RCfg7 = a100::GCfg<256, 128, 32, 4, 2, 4>;    // RCfg4 with a 4-slice ring
using RCfg8 = a100::GCfg<128, 64, 32, 4, 1, 5>;     // RCfg3 with a 5-slice ring
using RCfg9 = a100::GCfg<128, 256, 64, 2, 4, 3>;    // the shape of cuBLAS's 128x256_64x3 kernel (147 KB, one CTA per SM)
using RCfg10 = a100::GCfg<256, 128, 64, 4, 2, 3>;

template <class C>
void launch_res(const a100::ResParams& p, cudaStream_t st) {
  TORCH_CHECK(p.N % C::BN == 0 && p.K % C::BK == 0, "res tiling: N % BN, K % BK");
  const dim3 grid((unsigned)(p.N / C::BN), (unsigned)((p.M + C::BM - 1) / C::BM));
  set_smem(a100::gemm_res_kernel<C>, C::SMEM);
  a100::gemm_res_kernel<C><<<grid, C::NTHR, C::SMEM, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
int bn_of_res(int cfg) {
  switch (cfg) {
    case 0: return RCfg0::BN; case 1: return RCfg1::BN; case 2: return RCfg2::BN; case 3: return RCfg3::BN; case 4: return RCfg4::BN;
    case 5: return RCfg5::BN; case 6: return RCfg6::BN; case 7: return RCfg7::BN; case 8: return RCfg8::BN; case 9: return RCfg9::BN; case 10: return RCfg10::BN;
  }
  return 128;
}
}  // namespace

std::vector<torch::Tensor> ln_fwd(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps, bool save) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 && x.scalar_type() == at::kBFloat16, "x: [M, D] bf16");
  TORCH_CHECK(gamma.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat && gamma.is_contiguous() && beta.is_contiguous(), "f32 affine");
  const int64_t M = x.size(0);
  const int D = (int)x.size(1);
  TORCH_CHECK(gamma.numel() == D && beta.numel() == D, "affine width");
  auto xn = torch::empty_like(x);
  auto stats = torch::empty({save ? M : 0, 2}, x.options().dtype(at::kFloat));
  if (M == 0) return {xn, stats};
  float2* sp = save ? mptr<float2>(stats) : nullptr;
  const int tot = (int)M;
  auto st = at::cuda::getCurrentCUDAStream();
#define LN_FWD_CASE(DV)                                                                                                                                \
  case DV: {                                                                                                                                           \
    constexpr int rpw = 32 / (DV / 8 < 32 ? DV / 8 : 32);                                                                                              \
    const int blocks = std::max(1, std::min((tot + 8 * rpw - 1) / (8 * rpw), 8 * num_sms()));                                                          \
    a100::ln_fwd_kernel<DV><<<blocks, 256, 0, st>>>(cptr<__nv_bfloat16>(x), cptr<float>(gamma), cptr<float>(beta), mptr<__nv_bfloat16>(xn), sp, tot,     \
                                                     (float)eps);                                                                                      \
    break;                                                                                                                                             \
  }
  switch (D) {
    LN_FWD_CASE(64) LN_FWD_CASE(128) LN_FWD_CASE(256) LN_FWD_CASE(384) LN_FWD_CASE(512) LN_FWD_CASE(768)
    default: TORCH_CHECK(false, "ln_fwd: unsupported width ", D);
  }
#undef LN_FWD_CASE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {xn, stats};
}

torch::Tensor dual_swiglu(torch::Tensor a, torch::Tensor wa, torch::Tensor wb, int64_t cfg, int64_t epi = 0) {
  TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.dim() == 2 && a.scalar_type() == at::kBFloat16, "a: [M, K] bf16");
  TORCH_CHECK(wa.is_contiguous() && wb.is_contiguous() && wa.sizes() == wb.sizes() && wa.size(1) == a.size(1), "wa / wb: [H, K]");
  const int64_t M = a.size(0), K = a.size(1), H = wa.size(0);
  TORCH_CHECK(K % bk_of((int)cfg) == 0 && H % hidden_per_cta((int)cfg) == 0, "dual tiling: K % BK, H % hidden per tile");
  auto h = torch::empty({M, H}, a.options());
  if (M == 0) return h;
  a100::DualParams p{};
  p.a = cptr<__nv_bfloat16>(a); p.wa = cptr<__nv_bfloat16>(wa); p.wb = cptr<__nv_bfloat16>(wb); p.h = mptr<__nv_bfloat16>(h);
  p.M = (int)M; p.K = (int)K; p.H = (int)H;
  dispatch_dual((int)cfg, p, false, (int)epi, at::cuda::getCurrentCUDAStream());
  return h;
}

std::vector<torch::Tensor> gate_bwd(torch::Tensor a, torch::Tensor wa, torch::Tensor wb, torch::Tensor dh, int64_t cfg, int64_t epi = 0) {
  TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.dim() == 2 && a.scalar_type() == at::kBFloat16, "a: [M, K] bf16");
  TORCH_CHECK(wa.is_contiguous() && wb.is_contiguous() && wa.sizes() == wb.sizes() && wa.size(1) == a.size(1), "wa / wb: [H, K]");
  const int64_t M = a.size(0), K = a.size(1), H = wa.size(0);
  TORCH_CHECK(dh.is_contiguous() && dh.size(0) == M && dh.size(1) == H && dh.scalar_type() == at::kBFloat16, "dh: [M, H] bf16");
  TORCH_CHECK(K % bk_of((int)cfg) == 0 && H % hidden_per_cta((int)cfg) == 0, "dual tiling: K % BK, H % hidden per tile");
  auto dab = torch::empty({M, 2 * H}, a.options());
  auto h = torch::empty({M, H}, a.options());
  if (M == 0) return {dab, h};
  a100::DualParams p{};
  p.a = cptr<__nv_bfloat16>(a); p.wa = cptr<__nv_bfloat16>(wa); p.wb = cptr<__nv_bfloat16>(wb); p.dh = cptr<__nv_bfloat16>(dh);
  p.h = mptr<__nv_bfloat16>(h); p.dab = mptr<__nv_bfloat16>(dab);
  p.M = (int)M; p.K = (int)K; p.H = (int)H;
  dispatch_dual((int)cfg, p, true, (int)epi, at::cuda::getCurrentCUDAStream());
  return {dab, h};
}

// dxn [M, D] f32 -> (dx [M, D] bf16, dgamma, dbeta in adt)
std::vector<torch::Tensor> ln_bwd(torch::Tensor dxn, torch::Tensor x, torch::Tensor stats, torch::Tensor gamma, torch::Tensor dy, at::ScalarType adt) {
  TORCH_CHECK(dxn.is_cuda() && dxn.is_contiguous() && dxn.scalar_type() == at::kFloat && dxn.dim() == 2, "dxn: [M, D] f32");
  const bool has_dy = dy.numel() > 0;                                       // an empty dy: no residual branch (the bare LayerNorm + Linear backward)
  TORCH_CHECK(x.is_contiguous() && dy.is_contiguous() && x.sizes() == dxn.sizes() && (!has_dy || dy.sizes() == dxn.sizes()), "x / dy: [M, D]");
  TORCH_CHECK(stats.is_contiguous() && stats.size(0) == dxn.size(0) && stats.scalar_type() == at::kFloat && gamma.scalar_type() == at::kFloat, "stats / gamma");
  TORCH_CHECK(adt == at::kFloat || adt == at::kBFloat16, "f32 / bf16 affine gradients");
  const int64_t M = dxn.size(0);
  const int D = (int)dxn.size(1);
  auto dx = torch::empty_like(x);
  auto dgam = torch::zeros({D}, x.options().dtype(adt)), dbeta = torch::zeros({D}, x.options().dtype(adt));
  if (M == 0) return {dx, dgam, dbeta};
  auto st = at::cuda::getCurrentCUDAStream();
  const int tot = (int)M;
  int nb = 1;
  torch::Tensor part;
#define LN_BWD_CASE(DV)                                                                                                                                \
  case DV: {                                                                                                                                           \
    constexpr int rpw = 32 / (DV / 8 < 32 ? DV / 8 : 32);                                                                                              \
    nb = std::max(1, std::min((tot + 4 * rpw - 1) / (4 * rpw), 2 * num_sms()));                                                                        \
    part = torch::empty({nb, 2, DV}, x.options().dtype(at::kFloat));                                                                                   \
    if (has_dy)                                                                                                                                        \
      a100::ln_bwd_kernel<DV, true><<<nb, 128, 0, st>>>(cptr<float>(dxn), cptr<__nv_bfloat16>(x), cptr<float2>(stats), cptr<float>(gamma),           \
                                                         cptr<__nv_bfloat16>(dy), mptr<__nv_bfloat16>(dx), mptr<float>(part), tot);                     \
    else                                                                                                                                               \
      a100::ln_bwd_kernel<DV, false><<<nb, 128, 0, st>>>(cptr<float>(dxn), cptr<__nv_bfloat16>(x), cptr<float2>(stats), cptr<float>(gamma),          \
                                                          cptr<__nv_bfloat16>(x), mptr<__nv_bfloat16>(dx), mptr<float>(part), tot);                     \
    break;                                                                                                                                             \
  }
  switch (D) {
    LN_BWD_CASE(64) LN_BWD_CASE(128) LN_BWD_CASE(256) LN_BWD_CASE(384) LN_BWD_CASE(512) LN_BWD_CASE(768)
    default: TORCH_CHECK(false, "ln_bwd: unsupported width ", D);
  }
#undef LN_BWD_CASE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int fin_blocks = (2 * D + 31) / 32;
  if (adt == at::kFloat)
    a100::ln_finalize_kernel<float><<<fin_blocks, 256, 0, st>>>(cptr<float>(part), nb, D, mptr<float>(dgam), mptr<float>(dbeta));
  else
    a100::ln_finalize_kernel<__nv_bfloat16><<<fin_blocks, 256, 0, st>>>(cptr<float>(part), nb, D, mptr<__nv_bfloat16>(dgam), mptr<__nv_bfloat16>(dbeta));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dx, dgam, dbeta};
}

// out = rn(rn(a w^T) + res); res may be an empty tensor (no residual)
torch::Tensor gemm_res(torch::Tensor a, torch::Tensor w, torch::Tensor res, int64_t cfg) {
  TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.dim() == 2 && a.scalar_type() == at::kBFloat16, "a: [M, K] bf16");
  TORCH_CHECK(w.is_contiguous() && w.dim() == 2 && w.size(1) == a.size(1) && w.scalar_type() == at::kBFloat16, "w: [N, K] bf16");
  const int64_t M = a.size(0), K = a.size(1), N = w.size(0);
  const bool has_res = res.numel() > 0;
  TORCH_CHECK(!has_res || (res.is_contiguous() && res.size(0) == M && res.size(1) == N && res.scalar_type() == at::kBFloat16), "res: [M, N] bf16");
  TORCH_CHECK(N % bn_of_res((int)cfg) == 0, "N % BN");
  auto out = torch::empty({M, N}, a.options());
  if (M == 0) return out;
  a100::ResParams p{};
  p.a = cptr<__nv_bfloat16>(a); p.w = cptr<__nv_bfloat16>(w); p.res = has_res ? cptr<__nv_bfloat16>(res) : nullptr; p.out = mptr<__nv_bfloat16>(out);
  p.M = (int)M; p.N = (int)N; p.K = (int)K;
  auto st = at::cuda::getCurrentCUDAStream();
  switch ((int)cfg) {
    case 0: launch_res<RCfg0>(p, st); break;
    case 1: launch_res<RCfg1>(p, st); break;
    case 2: launch_res<RCfg2>(p, st); break;
    case 3: launch_res<RCfg3>(p, st); break;
    case 4: launch_res<RCfg4>(p, st); break;
    case 5: launch_res<RCfg5>(p, st); break;
    case 6: launch_res<RCfg6>(p, st); break;
    case 7: launch_res<RCfg7>(p, st); break;
    case 8: launch_res<RCfg8>(p, st); break;
    case 9: launch_res<RCfg9>(p, st); break;
    case 10: launch_res<RCfg10>(p, st); break;
    default: TORCH_CHECK(false, "res tile config ", cfg);
  }
  return out;
}

// y = rn(y + res) in place (y, res [M, N] bf16 contiguous, N % 8 == 0)
void add_res(torch::Tensor y, torch::Tensor res) {
  TORCH_CHECK(y.is_cuda() && y.is_contiguous() && res.is_contiguous() && y.sizes() == res.sizes() && y.scalar_type() == at::kBFloat16 && res.scalar_type() == at::kBFloat16,
              "y / res: matching contiguous bf16");
  TORCH_CHECK(y.numel() % 8 == 0, "numel % 8");
  const int64_t nvec = y.numel() / 8;
  if (nvec == 0) return;
  const int blocks = (int)std::min<int64_t>((nvec + 255) / 256, 8 * (int64_t)num_sms());
  a100::add_res_kernel<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(mptr<__nv_bfloat16>(y), cptr<__nv_bfloat16>(res), nvec);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_res", &add_res);
  m.def("gemm_res", &gemm_res);
  m.def("ln_fwd", &ln_fwd);
  m.def("dual_swiglu", &dual_swiglu, pybind11::arg("a"), pybind11::arg("wa"), pybind11::arg("wb"), pybind11::arg("cfg"), pybind11::arg("epi") = 0);
  m.def("gate_bwd", &gate_bwd, pybind11::arg("a"), pybind11::arg("wa"), pybind11::arg("wb"), pybind11::arg("dh"), pybind11::arg("cfg"), pybind11::arg("epi") = 0);
  m.def("ln_bwd", &ln_bwd);
}
