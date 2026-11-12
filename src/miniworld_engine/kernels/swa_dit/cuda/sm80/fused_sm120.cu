// fused_sm120.cu -- the RTX 5090 (sm_120a) extension of the SWA atom DiT block's fused bf16 inference stages: qkvg (qkvg_fwd_sm120.cuh, the weights split over
// CTA pairs) and the out-projection + FFN (ffn_fwd_sm120.cuh, 32-unit weight chunks), each within the card's 99 KB of shared memory.  The window attention between
// them is the sm_80 extension's (``swa_dit_sm80``, its sm_120a build).  Built for sm_120a only; the sm_80 sources and their builds are untouched.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>

#include "qkvg_fwd_sm120.cuh"
#include "ffn_fwd_sm120.cuh"
#include "fp32_fwd_sm120.cuh"

#define CHECK_CUDA_BF16(x) TORCH_CHECK((x).is_cuda() && (x).scalar_type() == torch::kBFloat16, #x " must be a CUDA bf16 tensor")
#define CHECK_CUDA_F32(x) TORCH_CHECK((x).is_cuda() && (x).scalar_type() == torch::kFloat32, #x " must be a CUDA fp32 tensor")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

namespace {
using bf = __nv_bfloat16;
inline const bf* bptr(const torch::Tensor& t) { return reinterpret_cast<const bf*>(t.data_ptr()); }
inline bf* bptr_w(torch::Tensor& t) { return reinterpret_cast<bf*>(t.data_ptr()); }
inline int num_sms() { return at::cuda::getCurrentDeviceProperties()->multiProcessorCount; }

template <class G>
void launch_qkvg_half(const sw80::QkvgFwdParams& p, int64_t max_pairs) {
  static bool attr[64] = {};
  const int dev = at::cuda::current_device() & 63;
  if (!attr[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(sw80::qkvg_fwd_half_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    attr[dev] = true;
  }
  const int nitem = (p.M + G::ROWS - 1) / G::ROWS * p.nsplit;   // work items of each half: (tile, block group)
  const int pairs = std::max(1, std::min({num_sms() / 2, nitem, max_pairs > 0 ? (int)max_pairs : nitem}));
  sw80::qkvg_fwd_half_kernel<G><<<2 * pairs, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class G>
void launch_ffn_u32(const sw80::FfnFwdParams& p) {
  static bool attr[64] = {};
  const int dev = at::cuda::current_device() & 63;
  if (!attr[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(sw80::ffn_fwd_u32_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    attr[dev] = true;
  }
  const int ntile = ((p.S + 15) / 16) * (p.M / p.S);
  const unsigned ctas = (unsigned)std::max(1, std::min(num_sms(), (ntile + G::NW - 1) / G::NW));
  sw80::ffn_fwd_u32_kernel<G><<<ctas, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

// qkvg stage (inference): x [M, C] bf16, mod [B S, 6C] fp32, cos / sin [B S, 16] fp32, Wqkv [3C, C], Wg [C, C] -> Qh, Kh, Vh [N, H, S, 32] (tm: token-major [M, H, 32]), G [M, C]
void swa_qkvg_fwd_half(torch::Tensor x, torch::Tensor mod, torch::Tensor cos, torch::Tensor sin, torch::Tensor wqkv, torch::Tensor wg, torch::Tensor qh, torch::Tensor kh,
                       torch::Tensor vh, torch::Tensor g, int64_t S, int64_t B, double eps, double qk_eps, int64_t cfg, bool tm, int64_t nsplit, int64_t max_pairs) {
  TORCH_CHECK(nsplit == 1 || nsplit == 2 || nsplit == 4, "nsplit must be 1, 2 or 4 (of a half's 4 output blocks)");
  CHECK_CUDA_BF16(x); CHECK_CUDA_BF16(wqkv); CHECK_CUDA_BF16(wg); CHECK_CUDA_BF16(qh); CHECK_CUDA_BF16(kh); CHECK_CUDA_BF16(vh); CHECK_CUDA_BF16(g);
  CHECK_CUDA_F32(mod); CHECK_CUDA_F32(cos); CHECK_CUDA_F32(sin);
  for (const auto* t : {&x, &mod, &cos, &sin, &wqkv, &wg, &qh, &kh, &vh, &g}) CHECK_CONTIG(*t);
  const int64_t M = x.size(0);
  TORCH_CHECK(x.size(1) == 128 && M % S == 0 && mod.numel() == B * S * 768 && cos.numel() == B * S * 16 && sin.numel() == B * S * 16, "bad operand sizes");
  TORCH_CHECK(wqkv.size(0) == 384 && wqkv.size(1) == 128 && wg.size(0) == 128 && wg.size(1) == 128, "bad weight sizes");
  TORCH_CHECK(qh.numel() == M * 128 && kh.numel() == M * 128 && vh.numel() == M * 128 && g.numel() == M * 128, "bad output sizes");
  if (M == 0) return;
  c10::cuda::CUDAGuard guard(x.device());
  sw80::QkvgFwdParams p{bptr(x), mod.data_ptr<float>(), cos.data_ptr<float>(), sin.data_ptr<float>(), bptr(wqkv), bptr(wg), bptr_w(qh), bptr_w(kh), bptr_w(vh), bptr_w(g),
                        nullptr, nullptr, nullptr, (int)M, (int)S, (int)B, (float)eps, (float)qk_eps, tm ? 1 : 0, (int)nsplit};
  switch (cfg) {
    case 0: launch_qkvg_half<sw80::QkvgHalfCfg<8, 2>>(p, max_pairs); break;
    case 1: launch_qkvg_half<sw80::QkvgHalfCfg<16, 1>>(p, max_pairs); break;
    case 2: launch_qkvg_half<sw80::QkvgHalfCfg<8, 1>>(p, max_pairs); break;
    case 3: launch_qkvg_half<sw80::QkvgHalfCfg<4, 1>>(p, max_pairs); break;
    default: TORCH_CHECK(false, "unknown qkvg configuration");
  }
}

// out-projection + gated residual + FFN stage (inference): Qin, G, O [M, C] bf16, mod [B S, 6C] fp32, Wo [C, C], Wu [2 hidden, C], Wd [C, hidden] -> out [M, C]
void swa_ffn_fwd_u32(torch::Tensor qi, torch::Tensor g, torch::Tensor o, torch::Tensor mod, torch::Tensor wo, torch::Tensor wu, torch::Tensor wd, torch::Tensor out, int64_t S,
                     int64_t B, double eps, int64_t cfg) {
  CHECK_CUDA_BF16(qi); CHECK_CUDA_BF16(g); CHECK_CUDA_BF16(o); CHECK_CUDA_BF16(wo); CHECK_CUDA_BF16(wu); CHECK_CUDA_BF16(wd); CHECK_CUDA_BF16(out); CHECK_CUDA_F32(mod);
  for (const auto* t : {&qi, &g, &o, &mod, &wo, &wu, &wd, &out}) CHECK_CONTIG(*t);
  const int64_t M = qi.size(0);
  TORCH_CHECK(qi.size(1) == 128 && M % S == 0 && mod.numel() == B * S * 768 && g.numel() == M * 128 && o.numel() == M * 128 && out.numel() == M * 128, "bad operand sizes");
  TORCH_CHECK(wo.size(0) == 128 && wo.size(1) == 128 && wu.size(0) == 512 && wu.size(1) == 128 && wd.size(0) == 128 && wd.size(1) == 256, "bad weight sizes");
  if (M == 0) return;
  c10::cuda::CUDAGuard guard(qi.device());
  sw80::FfnFwdParams p{bptr(qi), bptr(g), bptr(o), mod.data_ptr<float>(), bptr(wo), bptr(wu), bptr(wd), bptr_w(out), nullptr, nullptr, nullptr, nullptr,
                       (int)M, (int)S, (int)B, (float)eps, nullptr};
  switch (cfg) {
    case 0: launch_ffn_u32<sw80::FfnHalfCfg<8>>(p); break;
    case 1: launch_ffn_u32<sw80::FfnHalfCfg<4>>(p); break;
    case 2: launch_ffn_u32<sw80::FfnHalfCfg<2>>(p); break;
    case 3: launch_ffn_u32<sw80::FfnHalfCfg<1>>(p); break;
    default: TORCH_CHECK(false, "unknown ffn configuration");
  }
}

template <int MT>
void launch_qkvg_f32(const sw120f::QkvgParams& p) {
  static bool attr[64] = {};
  const int dev = at::cuda::current_device() & 63;
  if (!attr[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(sw120f::qkvg_f32_kernel<MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, sw120f::qkvg_smem<MT>()));
    attr[dev] = true;
  }
  const int ntile = (p.M + 16 * MT - 1) / (16 * MT);
  const unsigned ctas = (unsigned)std::min(num_sms(), ntile);
  sw120f::qkvg_f32_kernel<MT><<<ctas, sw120f::NTHR, sw120f::qkvg_smem<MT>(), at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// the fp32 qkvg stage on TF32 (fp32_fwd_sm120.cuh; ops.cu swa_qkvg_f32's contract): q [M, C] fp32, the packed [Wq; Wk; Wv; Wg] -> Qh, Kh, Vh bf16, G fp32; mt 1, 2 or 4
void swa_qkvg_f32(torch::Tensor q, torch::Tensor mod, torch::Tensor cos, torch::Tensor sin, torch::Tensor w, torch::Tensor qh, torch::Tensor kh, torch::Tensor vh,
                  torch::Tensor g, int64_t S, int64_t B, double eps, double qk_eps, bool hm, int64_t mt) {
  CHECK_CUDA_F32(q); CHECK_CUDA_F32(mod); CHECK_CUDA_F32(cos); CHECK_CUDA_F32(sin); CHECK_CUDA_F32(w); CHECK_CUDA_F32(g);
  CHECK_CUDA_BF16(qh); CHECK_CUDA_BF16(kh); CHECK_CUDA_BF16(vh);
  for (const auto* t : {&q, &mod, &cos, &sin, &w, &qh, &kh, &vh, &g}) CHECK_CONTIG(*t);
  const int64_t M = q.numel() / 128;
  TORCH_CHECK(q.numel() == M * 128 && M % S == 0 && mod.numel() == B * S * 768 && cos.numel() == B * S * 16 && sin.numel() == B * S * 16 && w.numel() == 512 * 128,
              "bad operand sizes");
  TORCH_CHECK(qh.numel() == M * 128 && kh.numel() == M * 128 && vh.numel() == M * 128 && g.numel() == M * 128, "bad output sizes");
  if (M == 0) return;
  c10::cuda::CUDAGuard guard(q.device());
  sw120f::QkvgParams p{q.data_ptr<float>(), mod.data_ptr<float>(), cos.data_ptr<float>(), sin.data_ptr<float>(), w.data_ptr<float>(), bptr_w(qh), bptr_w(kh), bptr_w(vh),
                       g.data_ptr<float>(), (int)M, (int)S, (int)B, hm ? 1 : 0, (float)eps, (float)qk_eps};
  switch (mt) {
    case 1: launch_qkvg_f32<1>(p); break;
    case 2: launch_qkvg_f32<2>(p); break;
    case 4: launch_qkvg_f32<4>(p); break;
    default: TORCH_CHECK(false, "mt must be 1, 2 or 4");
  }
}

// the fp32 out-projection + FFN stage on TF32 (ops.cu swa_ffn_f32's contract), one shape: 16-row tiles (MT 1)
void swa_ffn_f32(torch::Tensor q, torch::Tensor g, torch::Tensor o, torch::Tensor mod, torch::Tensor wo, torch::Tensor wu, torch::Tensor wd, torch::Tensor out, int64_t S,
                 int64_t B, double eps) {
  CHECK_CUDA_F32(q); CHECK_CUDA_F32(g); CHECK_CUDA_F32(mod); CHECK_CUDA_F32(wo); CHECK_CUDA_F32(wu); CHECK_CUDA_F32(wd); CHECK_CUDA_F32(out);
  TORCH_CHECK(o.is_cuda() && (o.scalar_type() == torch::kFloat32 || o.scalar_type() == torch::kBFloat16), "o must be a CUDA fp32 or bf16 tensor");
  for (const auto* t : {&q, &g, &o, &mod, &wo, &wu, &wd, &out}) CHECK_CONTIG(*t);
  const int64_t M = q.numel() / 128;
  TORCH_CHECK(q.numel() == M * 128 && g.numel() == M * 128 && o.numel() == M * 128 && out.numel() == M * 128 && M % S == 0 && mod.numel() == B * S * 768, "bad operand sizes");
  TORCH_CHECK(wo.numel() == 128 * 128 && wu.numel() == 512 * 128 && wd.numel() == 128 * 256, "bad weight sizes");
  if (M == 0) return;
  c10::cuda::CUDAGuard guard(q.device());
  sw120f::FfnParams p{q.data_ptr<float>(), g.data_ptr<float>(), o.data_ptr(), mod.data_ptr<float>(), wo.data_ptr<float>(), wu.data_ptr<float>(), wd.data_ptr<float>(),
                      out.data_ptr<float>(), (int)M, (int)S, (int)B, o.scalar_type() == torch::kBFloat16 ? 1 : 0, (float)eps};
  static bool attr[64] = {};
  const int dev = at::cuda::current_device() & 63;
  if (!attr[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(sw120f::ffn_f32_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, sw120f::ffn_smem<1>()));
    attr[dev] = true;
  }
  const unsigned ctas = (unsigned)std::min<int64_t>(num_sms(), (M + 15) / 16);
  sw120f::ffn_f32_kernel<1><<<ctas, sw120f::NTHR, sw120f::ffn_smem<1>(), at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("swa_qkvg_f32", &swa_qkvg_f32);
  m.def("swa_ffn_f32", &swa_ffn_f32);
  m.def("swa_qkvg_fwd_half", &swa_qkvg_fwd_half);
  m.def("swa_ffn_fwd_u32", &swa_ffn_fwd_u32);
}
