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
  static std::vector<std::pair<const void*, int>> done;   // (kernel, largest dynamic shared memory requested so far)
  const void* k = reinterpret_cast<const void*>(kernel);
  auto it = std::find_if(done.begin(), done.end(), [&](const auto& e) { return e.first == k; });
  if (it == done.end() || it->second < bytes) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    if (it == done.end()) done.emplace_back(k, bytes); else it->second = bytes;
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
  else { const int fwd_smem = p.ab != nullptr && 3 * C::OUT_TILE > C::SMEM ? 3 * C::OUT_TILE : C::SMEM;
         TORCH_CHECK(fwd_smem <= 163840, "forward with saved a | b needs ", fwd_smem, " B of shared memory");
         LAUNCH_EPI(dual_swiglu_kernel, fwd_smem) }
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

// the tile configurations of gate_dh (dh GEMM + SwiGLU backward from the saved a | b):  <BM, HN (hidden units per CTA tile), WM, WN, stages, BK>
using DCfg0 = a100::DhCfg<128, 64, 2, 2, 4>;        // 4 warps of 64 x 32, two CTAs per SM (80 KB)
using DCfg1 = a100::DhCfg<128, 128, 2, 2, 4>;       // 4 warps of 64 x 64, one CTA per SM (128 KB)
using DCfg2 = a100::DhCfg<128, 128, 2, 4, 3>;       // 8 warps of 64 x 32, one CTA per SM
using DCfg3 = a100::DhCfg<64, 64, 2, 2, 4>;         // 4 warps of 32 x 32, small M
using DCfg4 = a100::DhCfg<128, 64, 2, 2, 5>;        // DCfg0 with a 5-slice ring (92 KB: one CTA per SM)
using DCfg5 = a100::DhCfg<256, 64, 4, 2, 3>;        // 8 warps of 64 x 32
using DCfg6 = a100::DhCfg<64, 64, 2, 2, 3>;         // 4 warps of 32 x 32, 40 KB: up to four CTAs per SM
using DCfg7 = a100::DhCfg<128, 64, 4, 2, 3>;        // 8 warps of 32 x 32, 68 KB: two CTAs per SM
using DCfg8 = a100::DhCfg<64, 128, 2, 4, 3>;        // 8 warps of 32 x 32, 68 KB

template <class C>
void launch_dh(const a100::DualParams& p, cudaStream_t st) {
  TORCH_CHECK(C::TOTAL_SMEM <= 163840, "gate_dh tile config needs ", C::TOTAL_SMEM, " B of shared memory");
  TORCH_CHECK(p.H % C::BN == 0 && p.K % C::BK == 0, "gate_dh tiling: H % hidden per tile, K % BK");
  const dim3 grid((unsigned)(p.H / C::BN), (unsigned)((p.M + C::BM - 1) / C::BM));
  set_smem(a100::gate_dh_kernel<C>, C::TOTAL_SMEM);
  a100::gate_dh_kernel<C><<<grid, C::NTHR, C::TOTAL_SMEM, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void dispatch_dh(int cfg, const a100::DualParams& p, cudaStream_t st) {
  switch (cfg) {
    case 0: launch_dh<DCfg0>(p, st); break;
    case 1: launch_dh<DCfg1>(p, st); break;
    case 2: launch_dh<DCfg2>(p, st); break;
    case 3: launch_dh<DCfg3>(p, st); break;
    case 4: launch_dh<DCfg4>(p, st); break;
    case 5: launch_dh<DCfg5>(p, st); break;
    case 6: launch_dh<DCfg6>(p, st); break;
    case 7: launch_dh<DCfg7>(p, st); break;
    case 8: launch_dh<DCfg8>(p, st); break;
    default: TORCH_CHECK(false, "gate_dh tile config ", cfg);
  }
}

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

// training forward: (h [M, H], ab [M, 2H] = a | b rounded to bf16); ab is saved for gate_dh
std::vector<torch::Tensor> dual_swiglu_save(torch::Tensor a, torch::Tensor wa, torch::Tensor wb, int64_t cfg) {
  TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.dim() == 2 && a.scalar_type() == at::kBFloat16, "a: [M, K] bf16");
  TORCH_CHECK(wa.is_contiguous() && wb.is_contiguous() && wa.sizes() == wb.sizes() && wa.size(1) == a.size(1), "wa / wb: [H, K]");
  const int64_t M = a.size(0), K = a.size(1), H = wa.size(0);
  TORCH_CHECK(K % bk_of((int)cfg) == 0 && H % hidden_per_cta((int)cfg) == 0, "dual tiling: K % BK, H % hidden per tile");
  auto h = torch::empty({M, H}, a.options());
  auto ab = torch::empty({M, 2 * H}, a.options());
  if (M == 0) return {h, ab};
  a100::DualParams p{};
  p.a = cptr<__nv_bfloat16>(a); p.wa = cptr<__nv_bfloat16>(wa); p.wb = cptr<__nv_bfloat16>(wb); p.h = mptr<__nv_bfloat16>(h); p.ab = mptr<__nv_bfloat16>(ab);
  p.M = (int)M; p.K = (int)K; p.H = (int)H;
  dispatch_dual((int)cfg, p, false, 0, at::cuda::getCurrentCUDAStream());
  return {h, ab};
}

// dab [M, 2H] = (dA | dB) from dy [M, D], wsT = Ws^T [H, D] and the saved ab [M, 2H]
torch::Tensor gate_dh(torch::Tensor dy, torch::Tensor wsT, torch::Tensor ab, int64_t cfg) {
  TORCH_CHECK(dy.is_cuda() && dy.is_contiguous() && dy.dim() == 2 && dy.scalar_type() == at::kBFloat16, "dy: [M, D] bf16");
  TORCH_CHECK(wsT.is_contiguous() && wsT.dim() == 2 && wsT.size(1) == dy.size(1) && wsT.scalar_type() == at::kBFloat16, "wsT: [H, D] bf16");
  const int64_t M = dy.size(0), K = dy.size(1), H = wsT.size(0);
  TORCH_CHECK(ab.is_contiguous() && ab.dim() == 2 && ab.size(0) == M && ab.size(1) == 2 * H && ab.scalar_type() == at::kBFloat16, "ab: [M, 2H] bf16");
  auto dab = torch::empty({M, 2 * H}, dy.options());
  if (M == 0) return dab;
  a100::DualParams p{};
  p.a = cptr<__nv_bfloat16>(dy); p.wsT = cptr<__nv_bfloat16>(wsT); p.ab = const_cast<__nv_bfloat16*>(cptr<__nv_bfloat16>(ab)); p.dab = mptr<__nv_bfloat16>(dab);
  p.M = (int)M; p.K = (int)K; p.H = (int)H;
  dispatch_dh((int)cfg, p, at::cuda::getCurrentCUDAStream());
  return dab;
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

// dxn [M, D] f32 -> (dx [M, D] like x: bf16, or fp32 for the fp32 Transition (x / dy fp32), dgamma, dbeta in adt)
std::vector<torch::Tensor> ln_bwd_impl(torch::Tensor dxn, torch::Tensor x, torch::Tensor stats, torch::Tensor gamma, torch::Tensor dy, at::ScalarType adt,
                                       const float* dscale, const a100::Scale2* sj = nullptr) {
  TORCH_CHECK(dxn.is_cuda() && dxn.is_contiguous() && dxn.scalar_type() == at::kFloat && dxn.dim() == 2, "dxn: [M, D] f32");
  const bool has_dy = dy.numel() > 0;                                       // an empty dy: no residual branch (the bare LayerNorm + Linear backward)
  TORCH_CHECK(x.is_contiguous() && dy.is_contiguous() && x.sizes() == dxn.sizes() && (!has_dy || dy.sizes() == dxn.sizes()), "x / dy: [M, D]");
  TORCH_CHECK(stats.is_contiguous() && stats.size(0) == dxn.size(0) && stats.scalar_type() == at::kFloat && gamma.scalar_type() == at::kFloat, "stats / gamma");
  TORCH_CHECK(adt == at::kFloat || adt == at::kBFloat16, "f32 / bf16 affine gradients");
  const bool f32 = x.scalar_type() == at::kFloat;
  TORCH_CHECK((f32 || x.scalar_type() == at::kBFloat16) && (!has_dy || dy.scalar_type() == x.scalar_type()), "x / dy: bf16 or fp32, one dtype");
  const int64_t M = dxn.size(0);
  const int D = (int)dxn.size(1);
  auto dx = torch::empty_like(x);
  auto dgam = torch::empty({D}, x.options().dtype(adt)), dbeta = torch::empty({D}, x.options().dtype(adt));   // ln_finalize_kernel writes every element
  if (M == 0) { dgam.zero_(); dbeta.zero_(); return {dx, dgam, dbeta}; }
  auto st = at::cuda::getCurrentCUDAStream();
  const int tot = (int)M;
  int nb = 1;
  torch::Tensor part;
#define LN_BWD_CASE(DV)                                                                                                                                \
  case DV: {                                                                                                                                           \
    constexpr int rpw = 32 / (DV / 8 < 32 ? DV / 8 : 32);                                                                                              \
    nb = std::max(1, std::min((tot + 4 * rpw - 1) / (4 * rpw), 2 * num_sms()));                                                                        \
    part = torch::empty({nb, 2, DV}, x.options().dtype(at::kFloat));                                                                                   \
    if (f32 && has_dy)                                                                                                                                 \
      a100::ln_bwd_kernel<DV, true, true><<<nb, 128, 0, st>>>(cptr<float>(dxn), x.data_ptr(), cptr<float2>(stats), cptr<float>(gamma), dy.data_ptr(),   \
                                                               dx.data_ptr(), mptr<float>(part), tot, dscale);                                                \
    else if (f32)                                                                                                                                      \
      a100::ln_bwd_kernel<DV, false, true><<<nb, 128, 0, st>>>(cptr<float>(dxn), x.data_ptr(), cptr<float2>(stats), cptr<float>(gamma), x.data_ptr(),  \
                                                                dx.data_ptr(), mptr<float>(part), tot, dscale);                                               \
    else if (has_dy)                                                                                                                                   \
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
  a100::Scale2 job = sj != nullptr ? *sj : a100::Scale2();
  int scale_blocks = 0;
  if (sj != nullptr) {
    job.fin_blocks = fin_blocks;
    scale_blocks = (int)std::max<long long>(1, std::min<long long>((job.na4 + job.nb4 + 255) / 256, 4 * (long long)num_sms()));
  }
  TORCH_CHECK(sj == nullptr || adt == at::kFloat, "weight-gradient scaling: fp32 affine gradients");
  if (adt == at::kFloat)
    a100::ln_finalize_kernel<float><<<fin_blocks + scale_blocks, 256, 0, st>>>(cptr<float>(part), nb, D, mptr<float>(dgam), mptr<float>(dbeta), job);
  else
    a100::ln_finalize_kernel<__nv_bfloat16><<<fin_blocks, 256, 0, st>>>(cptr<float>(part), nb, D, mptr<__nv_bfloat16>(dgam), mptr<__nv_bfloat16>(dbeta));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dx, dgam, dbeta};
}
std::vector<torch::Tensor> ln_bwd(torch::Tensor dxn, torch::Tensor x, torch::Tensor stats, torch::Tensor gamma, torch::Tensor dy, at::ScalarType adt) {
  return ln_bwd_impl(dxn, x, stats, gamma, dy, adt, nullptr);
}
// the fp32 Transition: d_xn scaled by *dscale (fp32 [1]: the fp16 d_xn GEMM's output scale) inside the kernel, and the finalize launch also scales the
// weight gradients in place (a *= sa, b *= sb: the fp16 weight-gradient GEMMs' output scales)
std::vector<torch::Tensor> ln_bwd_scaled(torch::Tensor dxn, torch::Tensor x, torch::Tensor stats, torch::Tensor gamma, torch::Tensor dy, at::ScalarType adt,
                                         torch::Tensor dscale, torch::Tensor a, torch::Tensor sa, torch::Tensor b, torch::Tensor sb) {
  TORCH_CHECK(x.scalar_type() == at::kFloat && dscale.is_cuda() && dscale.scalar_type() == at::kFloat && dscale.numel() >= 1, "ln_bwd_scaled: fp32 rows, fp32 scale");
  for (auto* t : {&a, &sa, &b, &sb}) TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->scalar_type() == at::kFloat, "ln_bwd_scaled: contiguous fp32 a / b / scales");
  TORCH_CHECK(a.numel() % 4 == 0 && b.numel() % 4 == 0 && sa.numel() >= 1 && sb.numel() >= 1, "ln_bwd_scaled: numel % 4");
  a100::Scale2 job;
  job.a = mptr<float4>(a); job.na4 = a.numel() / 4; job.sa = cptr<float>(sa);
  job.b = mptr<float4>(b); job.nb4 = b.numel() / 4; job.sb = cptr<float>(sb);
  return ln_bwd_impl(dxn, x, stats, gamma, dy, adt, dscale.data_ptr<float>(), &job);
}

// out = rn(rn(a w^T) + res); res may be an empty tensor (no residual). out_buf: a caller-owned contiguous [M, N] bf16 buffer to write into (a row block of
// a larger output: transition_sm80 splits a call past one launch's rows and joins the blocks without a copy), else a fresh tensor
torch::Tensor gemm_res(torch::Tensor a, torch::Tensor w, torch::Tensor res, int64_t cfg, c10::optional<torch::Tensor> out_buf = c10::nullopt) {
  TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.dim() == 2 && a.scalar_type() == at::kBFloat16, "a: [M, K] bf16");
  TORCH_CHECK(w.is_contiguous() && w.dim() == 2 && w.size(1) == a.size(1) && w.scalar_type() == at::kBFloat16, "w: [N, K] bf16");
  const int64_t M = a.size(0), K = a.size(1), N = w.size(0);
  const bool has_res = res.numel() > 0;
  TORCH_CHECK(!has_res || (res.is_contiguous() && res.size(0) == M && res.size(1) == N && res.scalar_type() == at::kBFloat16), "res: [M, N] bf16");
  TORCH_CHECK(N % bn_of_res((int)cfg) == 0, "N % BN");
  if (out_buf.has_value())
    TORCH_CHECK(out_buf->is_cuda() && out_buf->is_contiguous() && out_buf->dim() == 2 && out_buf->size(0) == M && out_buf->size(1) == N &&
                    out_buf->scalar_type() == at::kBFloat16 && reinterpret_cast<uintptr_t>(out_buf->data_ptr()) % 16 == 0,
                "out: contiguous [M, N] bf16, 16-byte aligned");
  auto out = out_buf.has_value() ? *out_buf : torch::empty({M, N}, a.options());
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

namespace {
// Finalize both FP32 weight-gradient accumulators in one launch, rounding once.
__global__ void transition_grad_cast_kernel(const float* ab, const float* s,
                                            __nv_bfloat16* out_ab, __nv_bfloat16* out_s,
                                            int64_t ab_vectors, int64_t total_vectors) {
  for (int64_t vector = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       vector < total_vectors; vector += (int64_t)blockDim.x * gridDim.x) {
    const bool is_ab = vector < ab_vectors;
    const int64_t offset = is_ab ? vector : vector - ab_vectors;
    const float* source = (is_ab ? ab : s) + offset * 8;
    const float4 lo = *reinterpret_cast<const float4*>(source);
    const float4 hi = *reinterpret_cast<const float4*>(source + 4);
    const uint4 packed = make_uint4(a100::pack_bf16(lo.x, lo.y), a100::pack_bf16(lo.z, lo.w),
                                    a100::pack_bf16(hi.x, hi.y), a100::pack_bf16(hi.z, hi.w));
    *reinterpret_cast<uint4*>((is_ab ? out_ab : out_s) + offset * 8) = packed;
  }
}
}  // namespace

std::vector<torch::Tensor> grad_cast(torch::Tensor ab, torch::Tensor s) {
  TORCH_CHECK(ab.is_cuda() && s.is_cuda() && ab.device() == s.device() &&
              ab.is_contiguous() && s.is_contiguous() &&
              ab.scalar_type() == at::kFloat && s.scalar_type() == at::kFloat,
              "weight accumulators: matching-device contiguous FP32");
  TORCH_CHECK(ab.numel() % 8 == 0 && s.numel() % 8 == 0, "weight gradient elements % 8");
  auto out_ab = torch::empty_like(ab, ab.options().dtype(at::kBFloat16));
  auto out_s = torch::empty_like(s, s.options().dtype(at::kBFloat16));
  const int64_t ab_vectors = ab.numel() / 8, total_vectors = ab_vectors + s.numel() / 8;
  if (total_vectors) {
    const int blocks = (int)std::min<int64_t>((total_vectors + 255) / 256, 4 * (int64_t)num_sms());
    transition_grad_cast_kernel<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        cptr<float>(ab), cptr<float>(s), mptr<__nv_bfloat16>(out_ab), mptr<__nv_bfloat16>(out_s),
        ab_vectors, total_vectors);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {out_ab, out_s};
}

// ===================================================================================================================== the fp32 Transition (fp16 operands at TF32's rounding)
// fp32 activations: the bf16 structure (ln_fwd -> dual_swiglu -> gemm_res) on fp16 tensor cores with fp32 accumulation. fp16's 10-bit mantissa is TF32's, so each
// operand is rounded as the TF32 path rounds it; the device scales keep every fp16 operand inside fp16's range (the sm_90a f32io scheme for the forward).
//   f32_forms        gamma, beta, Wa, Wb [H, D], Ws [D, H] fp32 -> (Wab16 [2H, D], Ws16 [D, H], Ws16^T [H, D] | [0], sc [8])
//   ln_fwd_f32       x [M, D] fp32 -> (xn16 = fp16 xn / s_x, stats | [0])
//   dual_swiglu_f16  xn16, Wa16, Wb16, sc -> h16 = fp16 silu(a) b / s_h
//   gemm_res_f32     h16, Ws16, x, sc -> x + h Ws^T fp32 (split K over ``splits`` CTAs per tile and a reduction pass when splits > 1)
//   dy_f16           dy fp32 -> (dy16 = fp16 dy / s_dy, bs [8])
//   gate_f16         xn16, Wa16, Wb16, dy16, Ws16^T, sc, bs -> (dab16 = fp16 [dA | dB] / s_dab, h16)   (the GEMMs after it cuBLAS fp16 -> fp32)
// ln_bwd takes fp32 x / dy (dx fp32); ln_bwd_scaled multiplies d_xn by a device scale and scales the weight gradients in its finalize launch.
// Tile configs of the fp16 dual GEMMs: 0 / 2 as the bf16 dual_swiglu (Cfg0 / Cfg2), 12: 64 rows x 32 hidden units (few rows: twice the CTAs of Cfg2).
namespace {
using Cfg12 = a100::DualCfg<64, 32, 2, 2, 4>;        // 4 warps of 32 x (16 a + 16 b)
int f16_hidden_per_cta(int cfg) { return cfg == 12 ? Cfg12::HPW * Cfg12::WN : hidden_per_cta(cfg); }
template <class C>
void launch_dual_f16(const a100::DualParams& p, cudaStream_t st) {
  using F = a100::F16Of<C>;
  const dim3 grid((unsigned)(p.H / (F::HPW * F::WN)), (unsigned)((p.M + F::BM - 1) / F::BM));
  set_smem(a100::dual_swiglu_kernel<F, 0>, F::SMEM);
  a100::dual_swiglu_kernel<F, 0><<<grid, F::NTHR, F::SMEM, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <class C>
void launch_gate_f16(const a100::DualParams& p, cudaStream_t st) {
  using F = a100::F16Of<C>;
  using Dh = a100::F16Of<a100::GCfg<F::BM, F::HPW * F::WN, F::BK, F::WM, F::WN, F::ST, false>>;
  constexpr int smem = F::SMEM > Dh::SMEM ? F::SMEM : Dh::SMEM;
  const dim3 grid((unsigned)(p.H / (F::HPW * F::WN)), (unsigned)((p.M + F::BM - 1) / F::BM));
  set_smem(a100::gate_f16_kernel<F>, smem);
  a100::gate_f16_kernel<F><<<grid, F::NTHR, smem, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
using RCfg11 = a100::GCfg<64, 64, 32, 2, 2, 4>;     // 4 warps of 32 x 32: few rows
template <class C>
void launch_res_f16(const a100::ResParams& p, int splits, cudaStream_t st) {
  using F = a100::F16Of<C>;
  TORCH_CHECK(p.N % F::BN == 0 && p.K % (F::BK * splits) == 0, "res tiling: N % BN, K % (BK splits)");
  const dim3 grid((unsigned)(p.N / F::BN), (unsigned)((p.M + F::BM - 1) / F::BM), (unsigned)splits);
  set_smem(a100::gemm_res_kernel<F>, F::SMEM);
  a100::gemm_res_kernel<F><<<grid, F::NTHR, F::SMEM, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void check_f16(const torch::Tensor& t, int64_t rows, int64_t cols, const char* what) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.dim() == 2 && t.scalar_type() == at::kHalf && t.size(0) == rows && t.size(1) == cols, what);
}
void check_sc(const torch::Tensor& sc) {
  TORCH_CHECK(sc.is_cuda() && sc.is_contiguous() && sc.scalar_type() == at::kFloat && sc.numel() == 8, "sc: fp32 [8]");
}
}  // namespace

std::vector<torch::Tensor> f32_forms(torch::Tensor gamma, torch::Tensor beta, torch::Tensor wa, torch::Tensor wb, torch::Tensor ws, bool transposed) {
  const int64_t H = wa.size(0), D = wa.size(1);
  for (auto* t : {&gamma, &beta, &wa, &wb, &ws}) TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->scalar_type() == at::kFloat, "f32_forms: contiguous fp32");
  TORCH_CHECK(wb.sizes() == wa.sizes() && ws.size(0) == D && ws.size(1) == H && gamma.numel() == D && beta.numel() == D, "f32_forms: shapes");
  TORCH_CHECK(D % 32 == 0 && H % 32 == 0, "f32_forms: D, H multiples of 32");
  auto f16 = wa.options().dtype(at::kHalf);
  auto st = torch::empty({8}, wa.options()), sc = torch::empty({8}, wa.options());
  auto wab16 = torch::empty({2 * H, D}, f16), ws16 = torch::empty({D, H}, f16), wst16 = torch::empty({transposed ? H : 0, D}, f16);
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(st.data_ptr(), 0, 8 * sizeof(float), stream));
  const int sb = std::max(1, std::min((int)((2 * H + 7) / 8), 4 * num_sms()));   // a warp per weight row (8 per block), at most 4 waves of blocks
  a100::f32_forms_stats_kernel<<<sb, 256, 0, stream>>>(cptr<float>(gamma), cptr<float>(beta), cptr<float>(wa), cptr<float>(wb), cptr<float>(ws), mptr<float>(st),
                                                       (int)D, (int)H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int nflat = std::max(1, std::min((int)((D * H / 4 + 255) / 256), 4 * num_sms()));
  const int ntr = transposed ? (int)(D / 32 * (H / 32)) : 0;
  a100::f32_forms_apply_kernel<<<nflat + ntr, 256, 0, stream>>>(cptr<float>(wa), cptr<float>(wb), cptr<float>(ws), cptr<float>(st), mptr<uint16_t>(wab16),
                                                                mptr<uint16_t>(ws16), transposed ? mptr<uint16_t>(wst16) : nullptr, mptr<float>(sc), (int)D, (int)H,
                                                                nflat);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {wab16, ws16, wst16, sc};
}

std::vector<torch::Tensor> dy_f16(torch::Tensor dy, torch::Tensor sc) {
  TORCH_CHECK(dy.is_cuda() && dy.is_contiguous() && dy.scalar_type() == at::kFloat && dy.numel() % 4 == 0, "dy: contiguous fp32");
  check_sc(sc);
  auto dy16 = torch::empty_like(dy, dy.options().dtype(at::kHalf));
  auto bs = torch::empty({8}, dy.options());
  auto mx = torch::empty({1}, dy.options().dtype(at::kInt));
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(mx.data_ptr(), 0, sizeof(int), stream));
  const long long n4 = dy.numel() / 4;
  const int blocks = (int)std::max<long long>(1, std::min<long long>((n4 + 255) / 256, 4 * (long long)num_sms()));
  a100::dy_absmax_kernel<<<blocks, 256, 0, stream>>>(cptr<float4>(dy), n4, mptr<int>(mx));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  a100::dy_convert_kernel<<<blocks, 256, 0, stream>>>(cptr<float4>(dy), n4, cptr<int>(mx), cptr<float>(sc), mptr<uint2>(dy16), mptr<float>(bs));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dy16, bs};
}

std::vector<torch::Tensor> ln_fwd_f32(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, torch::Tensor sc, double eps, bool save) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 && x.scalar_type() == at::kFloat, "x: [M, D] fp32");
  TORCH_CHECK(gamma.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat && gamma.is_contiguous() && beta.is_contiguous(), "f32 affine");
  check_sc(sc);
  const int64_t M = x.size(0);
  const int D = (int)x.size(1);
  TORCH_CHECK(gamma.numel() == D && beta.numel() == D, "affine width");
  auto xn = torch::empty({M, D}, x.options().dtype(at::kHalf));
  auto stats = torch::empty({save ? M : 0, 2}, x.options());
  if (M == 0) return {xn, stats};
  float2* sp = save ? mptr<float2>(stats) : nullptr;
  const int tot = (int)M;
  auto st = at::cuda::getCurrentCUDAStream();
#define LN_FWD_F32_CASE(DV)                                                                                                                            \
  case DV: {                                                                                                                                           \
    constexpr int rpw = 32 / (DV / 8 < 32 ? DV / 8 : 32);                                                                                              \
    const int blocks = std::max(1, std::min((tot + 8 * rpw - 1) / (8 * rpw), 8 * num_sms()));                                                          \
    a100::ln_fwd_kernel<DV, true><<<blocks, 256, 0, st>>>(cptr<float>(x), cptr<float>(gamma), cptr<float>(beta), mptr<__nv_bfloat16>(xn), sp, tot,     \
                                                           (float)eps, cptr<float>(sc));                                                               \
    break;                                                                                                                                             \
  }
  switch (D) {
    LN_FWD_F32_CASE(64) LN_FWD_F32_CASE(128) LN_FWD_F32_CASE(256) LN_FWD_F32_CASE(384) LN_FWD_F32_CASE(512) LN_FWD_F32_CASE(768)
    default: TORCH_CHECK(false, "ln_fwd_f32: unsupported width ", D);
  }
#undef LN_FWD_F32_CASE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {xn, stats};
}

torch::Tensor dual_swiglu_f16(torch::Tensor a, torch::Tensor wa, torch::Tensor wb, torch::Tensor sc, int64_t cfg) {
  const int64_t M = a.size(0), K = a.size(1), H = wa.size(0);
  check_f16(a, M, K, "a: [M, K] fp16");
  check_f16(wa, H, K, "wa: [H, K] fp16");
  check_f16(wb, H, K, "wb: [H, K] fp16");
  check_sc(sc);
  TORCH_CHECK(K % 32 == 0 && H % f16_hidden_per_cta((int)cfg) == 0, "dual tiling: K % 32, H % hidden per tile");
  auto h = torch::empty({M, H}, a.options());
  if (M == 0) return h;
  a100::DualParams p{};
  p.a = cptr<__nv_bfloat16>(a); p.wa = cptr<__nv_bfloat16>(wa); p.wb = cptr<__nv_bfloat16>(wb); p.h = mptr<__nv_bfloat16>(h); p.sc = cptr<float>(sc);
  p.M = (int)M; p.K = (int)K; p.H = (int)H;
  auto st = at::cuda::getCurrentCUDAStream();
  switch ((int)cfg) {
    case 0: launch_dual_f16<Cfg0>(p, st); break;
    case 2: launch_dual_f16<Cfg2>(p, st); break;
    case 12: launch_dual_f16<Cfg12>(p, st); break;
    default: TORCH_CHECK(false, "fp32 dual tile config ", cfg, " (0 / 2 / 12)");
  }
  return h;
}

std::vector<torch::Tensor> gate_f16(torch::Tensor a, torch::Tensor wa, torch::Tensor wb, torch::Tensor dy16, torch::Tensor wst16, torch::Tensor sc, torch::Tensor bs,
                                    int64_t cfg) {
  const int64_t M = a.size(0), K = a.size(1), H = wa.size(0);
  check_f16(a, M, K, "a: [M, K] fp16");
  check_f16(wa, H, K, "wa: [H, K] fp16");
  check_f16(wb, H, K, "wb: [H, K] fp16");
  check_f16(dy16, M, K, "dy16: [M, K] fp16");
  check_f16(wst16, H, K, "wst16: [H, K] fp16");
  check_sc(sc);
  TORCH_CHECK(bs.is_cuda() && bs.scalar_type() == at::kFloat && bs.numel() == 8, "bs: fp32 [8]");
  TORCH_CHECK(K % 32 == 0 && H % f16_hidden_per_cta((int)cfg) == 0, "gate tiling: K % 32, H % hidden per tile");
  auto dab = torch::empty({M, 2 * H}, a.options());
  auto h = torch::empty({M, H}, a.options());
  if (M == 0) return {dab, h};
  a100::DualParams p{};
  p.a = cptr<__nv_bfloat16>(a); p.wa = cptr<__nv_bfloat16>(wa); p.wb = cptr<__nv_bfloat16>(wb); p.a2 = cptr<__nv_bfloat16>(dy16);
  p.wsT = cptr<__nv_bfloat16>(wst16); p.sc = cptr<float>(sc); p.bs = cptr<float>(bs); p.h = mptr<__nv_bfloat16>(h); p.dab = mptr<__nv_bfloat16>(dab);
  p.M = (int)M; p.K = (int)K; p.H = (int)H;
  auto st = at::cuda::getCurrentCUDAStream();
  switch ((int)cfg) {   // 64-row tiles only: the dh and a | b accumulators are both live
    case 2: launch_gate_f16<Cfg2>(p, st); break;
    case 12: launch_gate_f16<Cfg12>(p, st); break;
    default: TORCH_CHECK(false, "gate_f16 tile config ", cfg, " (2 / 12)");
  }
  return {dab, h};
}

// cfg: 0 / 2 / 3 as gemm_res; 11: 64 x 64 tiles (few rows). splits > 1: split K, the partials summed in order by split_res_kernel (with the residual and the scale)
torch::Tensor gemm_res_f32(torch::Tensor a, torch::Tensor w, torch::Tensor res, torch::Tensor sc, int64_t cfg, int64_t splits) {
  const int64_t M = a.size(0), K = a.size(1), N = w.size(0);
  check_f16(a, M, K, "a: [M, K] fp16");
  check_f16(w, N, K, "w: [N, K] fp16");
  check_sc(sc);
  TORCH_CHECK(res.is_cuda() && res.is_contiguous() && res.scalar_type() == at::kFloat && res.size(0) == M && res.size(1) == N, "res: [M, N] fp32");
  TORCH_CHECK(splits >= 1 && K % (32 * splits) == 0, "splits must divide the K tiles");
  auto out = torch::empty({M, N}, res.options());
  if (M == 0) return out;
  auto part = splits > 1 ? torch::empty({splits, M, N}, res.options()) : out;
  a100::ResParams p{};
  p.a = cptr<__nv_bfloat16>(a); p.w = cptr<__nv_bfloat16>(w); p.sc = cptr<float>(sc); p.res32 = cptr<float>(res); p.out32 = mptr<float>(part);
  p.M = (int)M; p.N = (int)N; p.K = (int)K; p.kt = splits > 1 ? (int)(K / 32 / splits) : 0;
  auto st = at::cuda::getCurrentCUDAStream();
  const int s = (int)splits;
  switch ((int)cfg) {
    case 0: launch_res_f16<RCfg0>(p, s, st); break;
    case 2: launch_res_f16<RCfg2>(p, s, st); break;
    case 3: launch_res_f16<RCfg3>(p, s, st); break;
    case 11: launch_res_f16<RCfg11>(p, s, st); break;
    default: TORCH_CHECK(false, "fp32 res tile config ", cfg, " (0 / 2 / 3 / 11)");
  }
  if (splits > 1) {
    const long long n4 = M * N / 4;
    const int blocks = (int)std::max<long long>(1, std::min<long long>((n4 + 255) / 256, 4 * (long long)num_sms()));
    a100::split_res_kernel<<<blocks, 256, 0, st>>>(cptr<float4>(part), s, n4, cptr<float4>(res), cptr<float>(sc), mptr<float4>(out));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("f32_forms", &f32_forms);
  m.def("ln_fwd_f32", &ln_fwd_f32);
  m.def("dual_swiglu_f16", &dual_swiglu_f16);
  m.def("gemm_res_f32", &gemm_res_f32);
  m.def("dy_f16", &dy_f16);
  m.def("gate_f16", &gate_f16);
  m.def("ln_bwd_scaled", &ln_bwd_scaled);
  m.def("grad_cast", &grad_cast);
  m.def("add_res", &add_res);
  m.def("gemm_res", &gemm_res, pybind11::arg("a"), pybind11::arg("w"), pybind11::arg("res"), pybind11::arg("cfg"), pybind11::arg("out") = pybind11::none());
  m.def("ln_fwd", &ln_fwd);
  m.def("dual_swiglu", &dual_swiglu, pybind11::arg("a"), pybind11::arg("wa"), pybind11::arg("wb"), pybind11::arg("cfg"), pybind11::arg("epi") = 0);
  m.def("dual_swiglu_save", &dual_swiglu_save);
  m.def("gate_dh", &gate_dh);
  m.def("gate_bwd", &gate_bwd, pybind11::arg("a"), pybind11::arg("wa"), pybind11::arg("wb"), pybind11::arg("dh"), pybind11::arg("cfg"), pybind11::arg("epi") = 0);
  m.def("ln_bwd", &ln_bwd);
}
