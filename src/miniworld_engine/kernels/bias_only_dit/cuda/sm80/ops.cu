// ops.cu -- the torch extension of the A100 (sm_80) bias-only token DiT cores: out = sigmoid(g) * (P v) per head and sample (see pv_gate_sm80.cuh) and the bias gradient
// dbias = P (sum_a do v^T - D) (see dpb_sm80.cuh).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "dpb_sm80.cuh"
#include "pv_gate_sm80.cuh"

namespace {
using bf = __nv_bfloat16;
inline const bf* bptr(const torch::Tensor& t) { return reinterpret_cast<const bf*>(t.data_ptr()); }
inline bf* bptr_w(torch::Tensor& t) { return reinterpret_cast<bf*>(t.data_ptr()); }

void check_view(const torch::Tensor& t, const char* what) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kBFloat16 && t.dim() == 2 && t.stride(1) == 1, what, " must be a CUDA bf16 matrix with unit column stride");
  TORCH_CHECK(t.size(0) <= 1 || t.stride(0) % 8 == 0, what, " must have a row stride that is a multiple of 8 elements");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(t.data_ptr()) & 15) == 0, what, " must be 16-byte aligned");
}

template <int DH, int NST, int SG>
void launch(const bo80::PvParams& p) {
  using G = bo80::PvCfg<DH, NST, SG>;
  static bool attr = false;
  if (!attr) {
    TORCH_CHECK(cudaFuncSetAttribute(bo80::pv_gate_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM) == cudaSuccess, "the core needs ", G::SMEM, " bytes of shared memory");
    attr = true;
  }
  const dim3 grid((unsigned)(p.L / G::QT), (unsigned)p.nh, (unsigned)((p.S + SG - 1) / SG));
  bo80::pv_gate_kernel<G><<<grid, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int DH, int NST>
void launch_sg(const bo80::PvParams& p, int sg) {
  switch (sg) {
    case 1: launch<DH, NST, 1>(p); break;
    case 2: launch<DH, NST, 2>(p); break;
    case 3: launch<DH, NST, 3>(p); break;
    default: launch<DH, NST, 4>(p); break;
  }
}

template <int DH>
void launch_stages(const bo80::PvParams& p, int stages, int sg) {
  if (stages == 2) launch_sg<DH, 2>(p, sg);
  else launch_sg<DH, 3>(p, sg);
}
}  // namespace

// out [S L, W] = sigmoid(g) * (P v) per head (g may be an empty tensor: the product alone): P [nh L, L] bf16 contiguous, v / g / out bf16 views [S L, W = nh DH] (any row stride), S samples;
// ``stages`` (2 or 3) is the depth of the cp.async ring, ``sg`` (1 .. 4) the samples a CTA takes
void pv_gate(torch::Tensor v, torch::Tensor P, torch::Tensor out, torch::Tensor g, int64_t S, int64_t nh, int64_t dh, int64_t stages, int64_t sg) {
  check_view(v, "v"); check_view(out, "out");
  const bool gate = g.numel() > 0;
  if (gate) check_view(g, "g");
  TORCH_CHECK(P.is_cuda() && P.scalar_type() == torch::kBFloat16 && P.is_contiguous() && P.dim() == 2, "P must be a contiguous CUDA bf16 [nh L, L]");
  const int64_t L = P.size(1);
  TORCH_CHECK(P.size(0) == nh * L && L % 128 == 0, "P must be [nh L, L] with L a multiple of 128");
  TORCH_CHECK(v.size(0) == S * L && out.size(0) == S * L && v.size(1) >= nh * dh && out.size(1) >= nh * dh && (!gate || (g.size(0) == S * L && g.size(1) >= nh * dh)), "bad operand sizes");
  TORCH_CHECK(stages >= 2 && stages <= 3, "the ring has 2 or 3 stages");
  TORCH_CHECK(sg >= 1 && sg <= 4, "a CTA takes 1 to 4 samples");
  c10::cuda::CUDAGuard guard(v.device());
  bo80::PvParams p{bptr(P), bptr(v), gate ? bptr(g) : nullptr, bptr_w(out), (int)L, (int)nh, (int)S, v.stride(0), gate ? g.stride(0) : 0, out.stride(0), dh, dh, dh};
  switch (dh) {
    case 32: launch_stages<32>(p, (int)stages, (int)sg); break;
    case 48: launch_stages<48>(p, (int)stages, (int)sg); break;
    case 64: launch_stages<64>(p, (int)stages, (int)sg); break;
    default: TORCH_CHECK(false, "head width must be 32, 48 or 64");
  }
}

namespace {
template <int DH, int NST, int SG, int MINB>
void launch_dpb(const bo80::DpbParams& p) {
  using G = bo80::DpbCfg<DH, NST, SG, MINB>;
  TORCH_CHECK(G::SMEM <= 163 * 1024, "the bias-gradient core needs ", G::SMEM, " bytes of shared memory (head width ", DH, ", ", NST, " stages x ", SG, " samples)");
  static bool attr = false;
  if (!attr) {
    TORCH_CHECK(cudaFuncSetAttribute(bo80::dpb_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM) == cudaSuccess, "the bias-gradient core needs ", G::SMEM, " bytes of shared memory");
    attr = true;
  }
  const dim3 grid((unsigned)(p.L / G::BN), (unsigned)(p.L / G::BM), (unsigned)p.nh);
  bo80::dpb_kernel<G><<<grid, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int DH>
void launch_dpb_cfg(const bo80::DpbParams& p, int cfg) {
  switch (cfg) {
    case 0: launch_dpb<DH, 2, 1, 2>(p); break;
    case 1: launch_dpb<DH, 3, 1, 1>(p); break;
    case 2: launch_dpb<DH, 2, 2, 1>(p); break;
    default: launch_dpb<DH, 3, 2, 1>(p); break;
  }
}
}  // namespace

// dbias [nh L, L] = P (sum_a do v^T - D) per head, D[h, i] = sum_a dd[a, h, i]: do / v bf16 views [A L, W = nh DH] (any row stride), P / dbias bf16 [nh L, L] contiguous, dd fp32 [A, nh, L];
// ``cfg`` picks the schedule (0: 2 stages x 1 sample, two CTAs per SM; 1: 3 x 1; 2: 2 x 2; 3: 3 x 2)
void dpb(torch::Tensor dout, torch::Tensor v, torch::Tensor P, torch::Tensor dd, torch::Tensor dbias, int64_t A, int64_t nh, int64_t dh, int64_t cfg) {
  check_view(dout, "do"); check_view(v, "v");
  TORCH_CHECK(P.is_cuda() && P.scalar_type() == torch::kBFloat16 && P.is_contiguous() && P.dim() == 2 && dbias.is_cuda() && dbias.scalar_type() == torch::kBFloat16 && dbias.is_contiguous()
              && dbias.sizes() == P.sizes(), "P and dbias must be contiguous CUDA bf16 [nh L, L]");
  const int64_t L = P.size(1);
  TORCH_CHECK(P.size(0) == nh * L && L % 128 == 0, "P must be [nh L, L] with L a multiple of 128");
  TORCH_CHECK(dout.size(0) == A * L && v.size(0) == A * L && dout.size(1) >= nh * dh && v.size(1) >= nh * dh, "bad operand sizes");
  TORCH_CHECK(dd.is_cuda() && dd.scalar_type() == torch::kFloat && dd.is_contiguous() && dd.numel() == A * nh * L, "dd must be a contiguous fp32 [A, nh, L]");
  TORCH_CHECK(cfg >= 0 && cfg <= 3, "cfg 0 .. 3");
  c10::cuda::CUDAGuard guard(dout.device());
  bo80::DpbParams p{bptr(dout), bptr(v), bptr(P), dd.data_ptr<float>(), bptr_w(dbias), (int)L, (int)nh, (int)A, dout.stride(0), v.stride(0), dh, dh};
  switch (dh) {
    case 32: launch_dpb_cfg<32>(p, (int)cfg); break;
    case 48: launch_dpb_cfg<48>(p, (int)cfg); break;
    case 64: launch_dpb_cfg<64>(p, (int)cfg); break;
    default: TORCH_CHECK(false, "head width must be 32, 48 or 64");
  }
}


// ---- head PLANES (the bias-only attention family: v / out [B H, L (t), L (n), D] -- head plane h is [S L][D], S = L samples, one plane after the other; P [nh L, L] has the plane's P at rows h L)
void check_planes(const torch::Tensor& t, const char* what, int64_t rows, int64_t dh) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kBFloat16 && t.is_contiguous() && t.dim() == 2 && t.size(0) == rows && t.size(1) == dh, what, " must be a contiguous CUDA bf16 [planes S L, DH]");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(t.data_ptr()) & 15) == 0, what, " must be 16-byte aligned");
}

// out [nh S L, DH] = P v per head plane (the product alone): P [nh L, L] bf16, v / out contiguous bf16 [nh S L, DH]; S samples (the t axis); ``stages`` / ``sg`` as ``pv_gate``
void pv_planes(torch::Tensor v, torch::Tensor P, torch::Tensor out, int64_t S, int64_t nh, int64_t dh, int64_t stages, int64_t sg) {
  TORCH_CHECK(P.is_cuda() && P.scalar_type() == torch::kBFloat16 && P.is_contiguous() && P.dim() == 2, "P must be a contiguous CUDA bf16 [nh L, L]");
  const int64_t L = P.size(1);
  TORCH_CHECK(P.size(0) == nh * L && L % 128 == 0, "P must be [nh L, L] with L a multiple of 128");
  check_planes(v, "v", nh * S * L, dh); check_planes(out, "out", nh * S * L, dh);
  TORCH_CHECK(stages >= 2 && stages <= 3 && sg >= 1 && sg <= 4, "stages 2 or 3, 1 to 4 samples a CTA");
  c10::cuda::CUDAGuard guard(v.device());
  bo80::PvParams p{bptr(P), bptr(v), nullptr, bptr_w(out), (int)L, (int)nh, (int)S, dh, 0, dh, S * L * dh, 0, S * L * dh};
  switch (dh) {
    case 32: launch_stages<32>(p, (int)stages, (int)sg); break;
    case 48: launch_stages<48>(p, (int)stages, (int)sg); break;
    case 64: launch_stages<64>(p, (int)stages, (int)sg); break;
    default: TORCH_CHECK(false, "head width must be 32, 48 or 64");
  }
}

// dbias [nh L, L] = P (sum_a do v^T - D) per head plane, D[h, i] = sum_a dd[a, h, i]: do / v contiguous bf16 [nh A L, DH] (A samples), dd fp32 [A, nh, L]
void dpb_planes(torch::Tensor dout, torch::Tensor v, torch::Tensor P, torch::Tensor dd, torch::Tensor dbias, int64_t A, int64_t nh, int64_t dh, int64_t cfg) {
  TORCH_CHECK(P.is_cuda() && P.scalar_type() == torch::kBFloat16 && P.is_contiguous() && P.dim() == 2 && dbias.is_cuda() && dbias.scalar_type() == torch::kBFloat16 && dbias.is_contiguous()
              && dbias.sizes() == P.sizes(), "P and dbias must be contiguous CUDA bf16 [nh L, L]");
  const int64_t L = P.size(1);
  TORCH_CHECK(P.size(0) == nh * L && L % 128 == 0, "P must be [nh L, L] with L a multiple of 128");
  check_planes(dout, "do", nh * A * L, dh); check_planes(v, "v", nh * A * L, dh);
  TORCH_CHECK(dd.is_cuda() && dd.scalar_type() == torch::kFloat && dd.is_contiguous() && dd.numel() == A * nh * L, "dd must be a contiguous fp32 [A, nh, L]");
  TORCH_CHECK(cfg >= 0 && cfg <= 3, "cfg 0 .. 3");
  c10::cuda::CUDAGuard guard(dout.device());
  bo80::DpbParams p{bptr(dout), bptr(v), bptr(P), dd.data_ptr<float>(), bptr_w(dbias), (int)L, (int)nh, (int)A, dh, dh, A * L * dh, A * L * dh};
  switch (dh) {
    case 32: launch_dpb_cfg<32>(p, (int)cfg); break;
    case 48: launch_dpb_cfg<48>(p, (int)cfg); break;
    case 64: launch_dpb_cfg<64>(p, (int)cfg); break;
    default: TORCH_CHECK(false, "head width must be 32, 48 or 64");
  }
}

// dd [A, nh, L] fp32 = sum_d dO o per (sample, plane, query) from dO / O contiguous bf16 [nh A L, DH]
void delta_planes(torch::Tensor dO, torch::Tensor O, torch::Tensor dd, int64_t A, int64_t L, int64_t nh, int64_t dh) {
  check_planes(dO, "dO", nh * A * L, dh); check_planes(O, "O", nh * A * L, dh);
  TORCH_CHECK(dd.is_cuda() && dd.scalar_type() == torch::kFloat && dd.is_contiguous() && dd.numel() == A * nh * L, "dd must be a contiguous fp32 [A, nh, L]");
  c10::cuda::CUDAGuard guard(dO.device());
  const dim3 grid((unsigned)((A * L + 127) / 128), (unsigned)nh);
  auto st = at::cuda::getCurrentCUDAStream();
  const long long hs = A * L * dh;
  switch (dh) {
    case 32: bo80::delta_planes_kernel<32><<<grid, 128, 0, st>>>(bptr(dO), bptr(O), dd.data_ptr<float>(), (int)A, (int)L, (int)nh, hs); break;
    case 48: bo80::delta_planes_kernel<48><<<grid, 128, 0, st>>>(bptr(dO), bptr(O), dd.data_ptr<float>(), (int)A, (int)L, (int)nh, hs); break;
    case 64: bo80::delta_planes_kernel<64><<<grid, 128, 0, st>>>(bptr(dO), bptr(O), dd.data_ptr<float>(), (int)A, (int)L, (int)nh, hs); break;
    default: TORCH_CHECK(false, "head width must be 32, 48 or 64");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pv_gate", &pv_gate);
  m.def("dpb", &dpb);
  m.def("pv_planes", &pv_planes);
  m.def("dpb_planes", &dpb_planes);
  m.def("delta_planes", &delta_planes);
}
