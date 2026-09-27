// ops.cu -- torch bindings of the A100 Transition backward kernels (P, X, W), on the current CUDA stream.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>
#include <vector>

#include "tr_bwd_px_sm80.cuh"
#include "tr_bwd_fused_sm80.cuh"
#include "tr_bwd_ring_sm80.cuh"
#include "tr_bwd_w_sm80.cuh"
#include "tr_bwd_pw_sm80.cuh"
#ifdef DW_V3
#include "archive/tr_bwd_dw_v3.cuh"
#else
#include "tr_bwd_dw_sm80.cuh"
#endif

namespace {
int num_sms() {
  static int n = 0;
  if (n == 0) cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, at::cuda::current_device());
  return n;
}
template <class K>
void set_smem(K kernel, int bytes) {   // once per kernel (P and X share a function type, so the flag is keyed on the pointer)
  static std::vector<const void*> done;
  const void* k = reinterpret_cast<const void*>(kernel);
  if (std::find(done.begin(), done.end(), k) == done.end()) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    done.push_back(k);
  }
}
template <class T> const T* cptr(const torch::Tensor& t) { return reinterpret_cast<const T*>(t.data_ptr()); }
template <class T> T* mptr(torch::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }
}  // namespace

a100::BwdParams params(torch::Tensor x, torch::Tensor dy, torch::Tensor wp, torch::Tensor wx, torch::Tensor gb, torch::Tensor gamma,
                       torch::Tensor xn, torch::Tensor stats, torch::Tensor ab, torch::Tensor dx,
                       torch::Tensor dgb, double eps) {
  TORCH_CHECK(x.is_contiguous() && x.size(1) == 128 && x.scalar_type() == at::kBFloat16, "x: [T,128] bf16");
  TORCH_CHECK(x.size(0) % 256 == 0, "T must be a multiple of 256");
  a100::BwdParams p;
  p.x = cptr<__nv_bfloat16>(x); p.dy = cptr<__nv_bfloat16>(dy); p.wp = cptr<__nv_bfloat16>(wp); p.wx = cptr<__nv_bfloat16>(wx);
  p.gb = cptr<float4>(gb); p.gamma = cptr<float>(gamma);
  p.xn = mptr<__nv_bfloat16>(xn); p.stats = mptr<float2>(stats);
  p.ab = mptr<uint4>(ab);
  p.dx = mptr<__nv_bfloat16>(dx); p.dgb = mptr<float>(dgb);
  p.T = (int)x.size(0); p.num_tiles = p.T / 256; p.eps = (float)eps;
  return p;
}

void bwd_p(torch::Tensor x, torch::Tensor dy, torch::Tensor wp, torch::Tensor wx, torch::Tensor gb, torch::Tensor gamma, torch::Tensor xn,
           torch::Tensor stats, torch::Tensor ab, torch::Tensor dx, torch::Tensor dgb, double eps) {
  auto p = params(x, dy, wp, wx, gb, gamma, xn, stats, ab, dx, dgb, eps);
  const int g = std::min(p.num_tiles, num_sms());
  set_smem(a100::tr_bwd_p_kernel, a100::CfgP::SMEM);
  a100::tr_bwd_p_kernel<<<g, 256, a100::CfgP::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void bwd_x(torch::Tensor x, torch::Tensor dy, torch::Tensor wp, torch::Tensor wx, torch::Tensor gb, torch::Tensor gamma, torch::Tensor xn,
           torch::Tensor stats, torch::Tensor ab, torch::Tensor dx, torch::Tensor dgb, double eps) {
  auto p = params(x, dy, wp, wx, gb, gamma, xn, stats, ab, dx, dgb, eps);
  const int g = std::min(p.num_tiles, num_sms());
  TORCH_CHECK(dgb.numel() >= g * 256, "dgb: [grid, 256]");
  set_smem(a100::tr_bwd_x_kernel, a100::CfgX::SMEM);
  a100::tr_bwd_x_kernel<<<g, 256, a100::CfgX::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void bwd_px(torch::Tensor x, torch::Tensor dy, torch::Tensor wp, torch::Tensor wx, torch::Tensor gb, torch::Tensor gamma, torch::Tensor xn,
            torch::Tensor stats, torch::Tensor ab, torch::Tensor dx, torch::Tensor dgb, double eps) {
  auto p = params(x, dy, wp, wx, gb, gamma, xn, stats, ab, dx, dgb, eps);     // wp: the PX weights [16 chunks][W1 | W3 | WX]
  const int g = std::min(p.T / a100::CfgPX::BM, num_sms());
  TORCH_CHECK(dgb.numel() >= g * 256, "dgb: [grid, 256]");
  set_smem(a100::tr_bwd_px_kernel, a100::CfgPX::SMEM);
  a100::tr_bwd_px_kernel<<<g, 256, a100::CfgPX::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void bwd_w(torch::Tensor x, torch::Tensor dy, torch::Tensor stats, torch::Tensor gamma, torch::Tensor beta, torch::Tensor ab,
           torch::Tensor part) {
  a100::WParams p;
  p.x = cptr<__nv_bfloat16>(x); p.dy = cptr<__nv_bfloat16>(dy); p.stats = cptr<float2>(stats);
  p.gamma = cptr<float>(gamma); p.beta = cptr<float>(beta);
  p.ab = cptr<uint4>(ab); p.rsc = nullptr;
  p.part = mptr<float>(part); p.T = (int)x.size(0);
  TORCH_CHECK(p.T % a100::CfgW::RS == 0, "T must be a multiple of 64");
  const int nrep = (int)part.size(0);
  set_smem(a100::tr_bwd_w_kernel<false>, a100::CfgW::SMEM);
  a100::tr_bwd_w_kernel<false><<<4 * nrep, a100::CfgW::NTHR, a100::CfgW::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void bwd_dw(torch::Tensor xn, torch::Tensor dy, torch::Tensor wdw, torch::Tensor gamma, torch::Tensor beta, torch::Tensor part, double eps) {
  a100::DWParams p;
  p.xn = cptr<__nv_bfloat16>(xn); p.dy = cptr<__nv_bfloat16>(dy); p.wdw = cptr<__nv_bfloat16>(wdw);
  p.gamma = cptr<float>(gamma); p.beta = cptr<float>(beta); p.part = mptr<float>(part);
  p.T = (int)xn.size(0); p.eps = (float)eps;
  TORCH_CHECK(p.T % a100::CfgDW::RS == 0, "T must be a multiple of 64");
  const int nrep = (int)part.size(0);
  set_smem(a100::tr_bwd_dw_kernel, a100::CfgDW::SMEM);
  a100::tr_bwd_dw_kernel<<<8 * nrep, 256, a100::CfgDW::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void bwd_dx(torch::Tensor x, torch::Tensor xn, torch::Tensor dy, torch::Tensor stats, torch::Tensor w, torch::Tensor gamma, torch::Tensor dx,
            torch::Tensor dgb) {
  a100::DXParams p;
  p.x = cptr<__nv_bfloat16>(x); p.xn = cptr<__nv_bfloat16>(xn); p.dy = cptr<__nv_bfloat16>(dy); p.stats = cptr<float2>(stats);
  p.w = cptr<__nv_bfloat16>(w); p.gamma = cptr<float>(gamma); p.dx = mptr<__nv_bfloat16>(dx); p.dgb = mptr<float>(dgb);
  p.T = (int)x.size(0);
  TORCH_CHECK(p.T % a100::CfgDX::BM == 0, "T must be a multiple of 256");
  const int g = std::min(p.T / a100::CfgDX::BM, num_sms());
  TORCH_CHECK(dgb.numel() >= g * 256, "dgb");
  set_smem(a100::tr_bwd_dx_kernel, a100::CfgDX::SMEM);
  a100::tr_bwd_dx_kernel<<<g, 256, a100::CfgDX::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// one launch: x, dy -> dx, dgb [ndx][256], part [nrep][3][512][128] (dWa part = 2 dA... see the DW role), via xn / stats scratch
void bwd_fused(torch::Tensor x, torch::Tensor dy, torch::Tensor wdx, torch::Tensor wdw, torch::Tensor gamma, torch::Tensor beta,
               torch::Tensor xn, torch::Tensor stats, torch::Tensor barrier, torch::Tensor dx, torch::Tensor dgb, torch::Tensor part,
               int64_t ndx, double eps) {
  a100::FusedParams p;
  p.dx.x = cptr<__nv_bfloat16>(x); p.dx.xn = cptr<__nv_bfloat16>(xn); p.dx.dy = cptr<__nv_bfloat16>(dy); p.dx.stats = cptr<float2>(stats);
  p.dx.w = cptr<__nv_bfloat16>(wdx); p.dx.gamma = cptr<float>(gamma); p.dx.dx = mptr<__nv_bfloat16>(dx); p.dx.dgb = mptr<float>(dgb);
  p.dx.T = (int)x.size(0);
  p.dw.xn = cptr<__nv_bfloat16>(xn); p.dw.dy = cptr<__nv_bfloat16>(dy); p.dw.wdw = cptr<__nv_bfloat16>(wdw);
  p.dw.gamma = cptr<float>(gamma); p.dw.beta = cptr<float>(beta); p.dw.part = mptr<float>(part); p.dw.T = (int)x.size(0); p.dw.eps = (float)eps;
  p.beta = cptr<float>(beta); p.xn = mptr<__nv_bfloat16>(xn); p.stats = mptr<float2>(stats);
  p.barrier = reinterpret_cast<unsigned int*>(barrier.data_ptr()); p.ndx = (int)ndx; p.eps = (float)eps;
  const int grid = num_sms();
  TORCH_CHECK((grid - p.ndx) % 8 == 0 && (grid - p.ndx) / 8 == part.size(0), "DW CTAs: 8 slices x part.size(0) replicas");
  TORCH_CHECK(p.dx.T % 256 == 0, "T % 256");
  C10_CUDA_CHECK(cudaMemsetAsync(p.barrier, 0, 4, at::cuda::getCurrentCUDAStream()));
  set_smem(a100::tr_bwd_fused_kernel, a100::FUSED_SMEM);
  a100::tr_bwd_fused_kernel<<<grid, 256, a100::FUSED_SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// one launch, no recomputation: DXP CTAs hand h | 2 dA | dB to W CTAs through an L2-resident ring (tr_bwd_ring_sm80.cuh)
void bwd_ring(torch::Tensor x, torch::Tensor dy, torch::Tensor wdx, torch::Tensor gb, torch::Tensor gamma, torch::Tensor beta,
              torch::Tensor stats, torch::Tensor rsc, torch::Tensor dx, torch::Tensor dgb, torch::Tensor part, torch::Tensor ring, torch::Tensor flags,
              int64_t ndxp, double eps, torch::Tensor prof) {
  a100::RingParams p;
  p.prof = prof.numel() ? reinterpret_cast<unsigned long long*>(prof.data_ptr()) : nullptr;
  p.x = cptr<__nv_bfloat16>(x); p.dy = cptr<__nv_bfloat16>(dy); p.wdx = cptr<__nv_bfloat16>(wdx); p.gb = cptr<float4>(gb);
  p.gamma = cptr<float>(gamma); p.beta = cptr<float>(beta); p.stats = mptr<float2>(stats); p.rsc = mptr<float>(rsc); p.dx = mptr<__nv_bfloat16>(dx);
  p.dgb = mptr<float>(dgb); p.part = mptr<float>(part); p.ring = mptr<uint4>(ring);
  p.ready = reinterpret_cast<unsigned int*>(flags.data_ptr()); p.consumed = p.ready + ndxp * RING_K;
  p.T = (int)x.size(0); p.ndxp = (int)ndxp; p.nrep = (int)part.size(0); p.eps = (float)eps;
  const int grid = num_sms();
  TORCH_CHECK(p.ndxp % 4 == 0 && grid - p.ndxp == 4 * p.nrep, "grid = ndxp (multiple of 4) + 4 slices x part.size(0) replicas");
  TORCH_CHECK(p.T % 256 == 0 && ring.numel() * ring.element_size() >= (int64_t)ndxp * RING_K * a100::CHUNK_U4 * 16, "T / ring size");
  TORCH_CHECK(flags.numel() >= 2 * ndxp * RING_K, "flags");
  C10_CUDA_CHECK(cudaMemsetAsync(flags.data_ptr(), 0, 2 * ndxp * RING_K * 4, at::cuda::getCurrentCUDAStream()));
  set_smem(a100::tr_bwd_ring_kernel, a100::RING_SMEM);
  a100::tr_bwd_ring_kernel<<<grid, 256, a100::RING_SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// two kernels: PX (x, dy -> dA | dB | h blocks, dx, dgb, stats, row scales) then W with the row scales
void bwd_2k(torch::Tensor x, torch::Tensor dy, torch::Tensor wdx, torch::Tensor gb, torch::Tensor gamma, torch::Tensor beta,
            torch::Tensor stats, torch::Tensor rsc, torch::Tensor ab, torch::Tensor dx, torch::Tensor dgb, torch::Tensor part, int64_t gpx, double eps) {
  a100::RingParams p{};
  p.x = cptr<__nv_bfloat16>(x); p.dy = cptr<__nv_bfloat16>(dy); p.wdx = cptr<__nv_bfloat16>(wdx); p.gb = cptr<float4>(gb);
  p.gamma = cptr<float>(gamma); p.beta = cptr<float>(beta); p.stats = mptr<float2>(stats); p.rsc = mptr<float>(rsc); p.dx = mptr<__nv_bfloat16>(dx);
  p.dgb = mptr<float>(dgb); p.ab = mptr<uint4>(ab);
  p.T = (int)x.size(0); p.eps = (float)eps;
  TORCH_CHECK(p.T % 256 == 0, "T % 256");
  const int g = std::min<int>(p.T / 256, gpx > 0 ? (int)gpx : num_sms());
  p.ndxp = g;
  TORCH_CHECK(dgb.numel() >= g * 256, "dgb");
  auto st = at::cuda::getCurrentCUDAStream();
  set_smem(a100::tr_bwd_pxg_kernel, a100::PXG_SMEM);
  a100::tr_bwd_pxg_kernel<<<g, 256, a100::PXG_SMEM, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  a100::WParams w;
  w.x = p.x; w.dy = p.dy; w.stats = p.stats; w.rsc = p.rsc; w.gamma = p.gamma; w.beta = p.beta; w.ab = p.ab;
  w.part = mptr<float>(part); w.T = p.T;
  const int nrep = (int)part.size(0);
  set_smem(a100::tr_bwd_w_kernel<true>, a100::CfgW::SMEM);
  a100::tr_bwd_w_kernel<true><<<4 * nrep, a100::CfgW::NTHR, a100::CfgW::SMEM, st>>>(w);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// one launch, the two roles in turn per CTA (tr_bwd_seq_kernel); done: [T / 256] int32 scratch (zeroed here)
void bwd_seq(torch::Tensor x, torch::Tensor dy, torch::Tensor wdx, torch::Tensor gb, torch::Tensor gamma, torch::Tensor beta,
             torch::Tensor stats, torch::Tensor rsc, torch::Tensor ab, torch::Tensor dx, torch::Tensor dgb, torch::Tensor part, torch::Tensor done,
             int64_t wr, double eps, torch::Tensor prof) {
  a100::SeqParams q{};
  a100::RingParams& p = q.r;
  p.x = cptr<__nv_bfloat16>(x); p.dy = cptr<__nv_bfloat16>(dy); p.wdx = cptr<__nv_bfloat16>(wdx); p.gb = cptr<float4>(gb);
  p.gamma = cptr<float>(gamma); p.beta = cptr<float>(beta); p.stats = mptr<float2>(stats); p.rsc = mptr<float>(rsc); p.dx = mptr<__nv_bfloat16>(dx);
  p.dgb = mptr<float>(dgb); p.ab = mptr<uint4>(ab); p.done = reinterpret_cast<unsigned*>(done.data_ptr());
  p.T = (int)x.size(0); p.eps = (float)eps;
  const int g = num_sms();
  p.ndxp = g;
  TORCH_CHECK(p.T % 256 == 0 && g % 4 == 0 && part.size(0) == g / 4 && dgb.numel() >= g * 256 && done.numel() >= p.T / 256, "seq shapes");
  a100::WParams& w = q.w;
  w.x = p.x; w.dy = p.dy; w.stats = p.stats; w.rsc = p.rsc; w.gamma = p.gamma; w.beta = p.beta; w.ab = p.ab;
  w.part = mptr<float>(part); w.T = p.T;
  q.wr = (int)wr; q.prof = prof.numel() ? reinterpret_cast<unsigned long long*>(prof.data_ptr()) : nullptr;
  auto st = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(p.done, 0, (p.T / 256) * 4, st));
  set_smem(a100::tr_bwd_seq_kernel, a100::SEQ_SMEM);
  a100::tr_bwd_seq_kernel<<<g, 256, a100::SEQ_SMEM, st>>>(q);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// two kernels, variant PW + X: kernel 1 = tr_bwd_pw_kernel (8 slices x part.size(0) replicas), kernel 2 = bwd_x on its blocks
void bwd_pw(torch::Tensor x, torch::Tensor dy, torch::Tensor wdw, torch::Tensor gamma, torch::Tensor beta, torch::Tensor stats, torch::Tensor ab,
            torch::Tensor part, double eps) {
  a100::PWParams p;
  p.x = cptr<__nv_bfloat16>(x); p.dy = cptr<__nv_bfloat16>(dy); p.wdw = cptr<__nv_bfloat16>(wdw);
  p.gamma = cptr<float>(gamma); p.beta = cptr<float>(beta); p.stats = mptr<float2>(stats); p.ab = mptr<uint4>(ab); p.part = mptr<float>(part);
  p.T = (int)x.size(0); p.eps = (float)eps;
  TORCH_CHECK(p.T % 256 == 0, "T % 256");
  const int nrep = (int)part.size(0);
  set_smem(a100::tr_bwd_pw_kernel, a100::CfgDW::SMEM);
  a100::tr_bwd_pw_kernel<<<8 * nrep, 256, a100::CfgDW::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bwd_p", &bwd_p);
  m.def("bwd_x", &bwd_x);
  m.def("bwd_w", &bwd_w);
  m.def("bwd_px", &bwd_px);
  m.def("bwd_dw", &bwd_dw);
  m.def("bwd_dx", &bwd_dx);
  m.def("bwd_fused", &bwd_fused);
  m.def("bwd_ring", &bwd_ring);
  m.def("bwd_2k", &bwd_2k);
  m.def("bwd_seq", &bwd_seq);
  m.def("bwd_pw", &bwd_pw);
}
