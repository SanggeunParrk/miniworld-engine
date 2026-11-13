// ops.cu -- the torch extension of the A100 (sm_80) bias-only token DiT cores: out = sigmoid(g) * (P v) per head and sample (see pv_gate_sm80.cuh) and the bias gradient
// dbias = P (sum_a do v^T - D) (see dpb_sm80.cuh).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "dpb_sm80.cuh"
#include "pv_gate_sm80.cuh"
#include "pv_gate_tf32_sm80.cuh"
#include "dpb_tf32_sm80.cuh"

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
void check_view32(const torch::Tensor& t, const char* what) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 && t.dim() == 2 && t.stride(1) == 1, what, " must be a CUDA fp32 matrix with unit column stride");
  TORCH_CHECK(t.size(0) <= 1 || t.stride(0) % 4 == 0, what, " must have a row stride that is a multiple of 4 elements");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(t.data_ptr()) & 15) == 0, what, " must be 16-byte aligned");
}
template <int DH, int NST, int SG, int MT>
void launch32(const bo80::PvTParams& p) {
  using G = bo80::PvTCfg<DH, NST, SG, MT>;
  static bool attr = false;
  if (!attr) {
    TORCH_CHECK(cudaFuncSetAttribute(bo80::pv_gate_tf32_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM) == cudaSuccess, "the TF32 core needs ", G::SMEM, " bytes of shared memory");
    attr = true;
  }
  const dim3 grid((unsigned)(p.L / G::QT), (unsigned)p.nh, (unsigned)((p.S + SG - 1) / SG));
  bo80::pv_gate_tf32_kernel<G><<<grid, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <int DH, int NST, int MT>
void launch32_sg(const bo80::PvTParams& p, int sg) {
  switch (sg) {
    case 1: launch32<DH, NST, 1, MT>(p); break;
    case 2: launch32<DH, NST, 2, MT>(p); break;
    case 3: launch32<DH, NST, 3, MT>(p); break;
    default: launch32<DH, NST, 4, MT>(p); break;
  }
}
template <int DH>
void launch32_stages(const bo80::PvTParams& p, int stages, int sg, int mt) {
  if (stages == 2) { if (mt == 2) launch32_sg<DH, 2, 2>(p, sg); else launch32_sg<DH, 2, 1>(p, sg); }
  else { if (mt == 2) launch32_sg<DH, 3, 2>(p, sg); else launch32_sg<DH, 3, 1>(p, sg); }
}
}  // namespace

// the fp32 / TF32 twin of pv_gate: P [nh L, L], v / g / out [S L, W] fp32
void pv_gate32(torch::Tensor v, torch::Tensor P, torch::Tensor out, torch::Tensor g, int64_t S, int64_t nh, int64_t dh, int64_t stages, int64_t sg, int64_t mt) {
  check_view32(v, "v"); check_view32(out, "out");
  const bool gate = g.numel() > 0;
  if (gate) check_view32(g, "g");
  TORCH_CHECK(P.is_cuda() && P.scalar_type() == torch::kFloat32 && P.is_contiguous() && P.dim() == 2, "P must be a contiguous CUDA fp32 [nh L, L]");
  const int64_t L = P.size(1);
  TORCH_CHECK(P.size(0) == nh * L && L % 128 == 0, "P must be [nh L, L] with L a multiple of 128");
  TORCH_CHECK(v.size(0) == S * L && out.size(0) == S * L && v.size(1) >= nh * dh && out.size(1) >= nh * dh && (!gate || (g.size(0) == S * L && g.size(1) >= nh * dh)), "bad operand sizes");
  TORCH_CHECK(stages >= 2 && stages <= 3, "the ring has 2 or 3 stages");
  TORCH_CHECK(sg >= 1 && sg <= 4 && (mt == 1 || mt == 2), "a CTA takes 1 to 4 samples, a warp 1 or 2 m16 tiles");
  c10::cuda::CUDAGuard guard(v.device());
  bo80::PvTParams p{P.data_ptr<float>(), v.data_ptr<float>(), gate ? g.data_ptr<float>() : nullptr, out.data_ptr<float>(), (int)L, (int)nh, (int)S, v.stride(0), gate ? g.stride(0) : 0, out.stride(0)};
  switch (dh) {
    case 32: launch32_stages<32>(p, (int)stages, (int)sg, (int)mt); break;
    case 48: launch32_stages<48>(p, (int)stages, (int)sg, (int)mt); break;
    case 64: launch32_stages<64>(p, (int)stages, (int)sg, (int)mt); break;
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


namespace {
template <int DH, int NST, int MINB>
void launch_dpb32(const bo80::DpbTParams& p) {
  using G = bo80::DpbTCfg<DH, NST, MINB>;
  static bool attr = false;
  if (!attr) {
    TORCH_CHECK(cudaFuncSetAttribute(bo80::dpb_tf32_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM) == cudaSuccess, "the TF32 bias-gradient core needs ", G::SMEM, " bytes of shared memory");
    attr = true;
  }
  const dim3 grid((unsigned)(p.L / G::BN), (unsigned)(p.L / G::BM), (unsigned)p.nh);
  bo80::dpb_tf32_kernel<G><<<grid, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

// the fp32 / TF32 twin of dpb: do / v fp32 views [A L, W], P / dbias fp32 [nh L, L], dd fp32 [A, nh, L]
void dpb32(torch::Tensor dout, torch::Tensor v, torch::Tensor P, torch::Tensor dd, torch::Tensor dbias, int64_t A, int64_t nh, int64_t dh) {
  check_view32(dout, "do"); check_view32(v, "v");
  TORCH_CHECK(P.is_cuda() && P.scalar_type() == torch::kFloat32 && P.is_contiguous() && P.dim() == 2 && dbias.is_cuda() && dbias.scalar_type() == torch::kFloat32 && dbias.is_contiguous()
              && dbias.sizes() == P.sizes(), "P and dbias must be contiguous CUDA fp32 [nh L, L]");
  const int64_t L = P.size(1);
  TORCH_CHECK(P.size(0) == nh * L && L % 128 == 0, "P must be [nh L, L] with L a multiple of 128");
  TORCH_CHECK(dout.size(0) == A * L && v.size(0) == A * L && dout.size(1) >= nh * dh && v.size(1) >= nh * dh, "bad operand sizes");
  TORCH_CHECK(dd.is_cuda() && dd.scalar_type() == torch::kFloat && dd.is_contiguous() && dd.numel() == A * nh * L, "dd must be a contiguous fp32 [A, nh, L]");
  c10::cuda::CUDAGuard guard(dout.device());
  bo80::DpbTParams p{dout.data_ptr<float>(), v.data_ptr<float>(), P.data_ptr<float>(), dd.data_ptr<float>(), dbias.data_ptr<float>(), (int)L, (int)nh, (int)A, dout.stride(0), v.stride(0)};
  switch (dh) {
    case 32: launch_dpb32<32, 2, 2>(p); break;
    case 48: launch_dpb32<48, 2, 1>(p); break;
#ifndef MWE_SM120  // head width 64: 131 KB even unpadded; the RTX 5090 path refuses it (bias_only_dit_sm80.refusal)
    case 64: launch_dpb32<64, 2, 1>(p); break;
#endif
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

#ifdef MWE_SM120
#include "gemm_sm120.cuh"
#include "rows_f32_sm120.cuh"
#include "lt_f32_sm120.cuh"
namespace {
template <typename T, int EPI, int BM, int BN, int WM, int WN, int ST, int RB, int MINB>
void run_gemm120(const g120::Params<T>& p, cudaStream_t st) {
  constexpr int smem = g120::smem_bytes<BM, BN, ST, RB>();
  static_assert(smem <= 101376 && MINB * smem <= 101376, "the RTX 5090's 99 KB of shared memory an SM");
  constexpr bool PAIRED = EPI == g120::EPI_ADALN || EPI == g120::EPI_SWIGLU;
  TORCH_CHECK(p.M % BM == 0 && p.N % (PAIRED ? 64 : BN) == 0 && p.K % (RB / (int)sizeof(T)) == 0, "gemm120: M, N, K multiples of the tile (", BM, ", ",
              PAIRED ? 64 : BN, ", ", RB / (int)sizeof(T), ")");
  auto k = g120::gemm<T, EPI, BM, BN, WM, WN, ST, RB, MINB>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  k<<<dim3(p.N / (PAIRED ? 64 : BN), p.M / BM), WM * WN * 32, smem, st>>>(p);
}
// cfg: the CTA tile (BM x BN, K-block row bytes, stages, CTAs an SM). 128-byte K-blocks: 0: 128 x 128 (3 stages), 1: 64 x 128, 2: 64 x 64,
// 3: 128 x 64, 4: 32 x 128 (4 stages); 64-byte K-blocks: 5: 128 x 128 (3 stages, 2 CTAs an SM), 6: 64 x 128 (4, 2), 7: 128 x 128 (4, 1),
// 8: 64 x 64 (4, 3), 9: 32 x 64 (4, 4), 10: 32 x 128 (4, 2). The paired epilogues (64 output columns = BN 128) take 0, 1, 4, 5, 6, 7, 10.
template <typename T, int EPI>
void dispatch_gemm120(const g120::Params<T>& p, int cfg, cudaStream_t st) {
  constexpr bool PAIRED = EPI == g120::EPI_ADALN || EPI == g120::EPI_SWIGLU;
  switch (cfg) {
    case 0: run_gemm120<T, EPI, 128, 128, 2, 4, 3, 128, 1>(p, st); break;
    case 1: run_gemm120<T, EPI, 64, 128, 2, 4, 4, 128, 1>(p, st); break;
    case 4: run_gemm120<T, EPI, 32, 128, 1, 4, 4, 128, 1>(p, st); break;
    case 5: run_gemm120<T, EPI, 128, 128, 2, 4, 3, 64, 2>(p, st); break;
    case 6: run_gemm120<T, EPI, 64, 128, 2, 4, 4, 64, 2>(p, st); break;
    case 7: run_gemm120<T, EPI, 128, 128, 2, 4, 4, 64, 1>(p, st); break;
    case 10: run_gemm120<T, EPI, 32, 128, 1, 4, 4, 64, 2>(p, st); break;
    case 11: if constexpr (!PAIRED) { run_gemm120<T, EPI, 64, 64, 2, 2, 6, 64, 2>(p, st); break; } [[fallthrough]];
    case 12: if constexpr (!PAIRED) { run_gemm120<T, EPI, 128, 64, 4, 2, 6, 64, 1>(p, st); break; } [[fallthrough]];
    case 9: if constexpr (!PAIRED) { run_gemm120<T, EPI, 32, 64, 1, 2, 4, 64, 4>(p, st); break; } [[fallthrough]];
    case 2: if constexpr (!PAIRED) { run_gemm120<T, EPI, 64, 64, 2, 2, 4, 128, 1>(p, st); break; } [[fallthrough]];
    case 3: if constexpr (!PAIRED) { run_gemm120<T, EPI, 128, 64, 4, 2, 4, 128, 1>(p, st); break; } [[fallthrough]];
    case 8: if constexpr (!PAIRED) { run_gemm120<T, EPI, 64, 64, 2, 2, 4, 64, 3>(p, st); break; } [[fallthrough]];
    default: TORCH_CHECK(false, "gemm120: config 0 .. 8 (paired epilogues: 0, 1, 4, 5, 6, 7), got ", cfg);
  }
}
template <typename T>
const T* opt(const c10::optional<torch::Tensor>& t) {
  return t.has_value() ? reinterpret_cast<const T*>(t->data_ptr()) : nullptr;
}
template <typename T>
void gemm120_t(int64_t epi, const torch::Tensor& a, const torch::Tensor& w1, const c10::optional<torch::Tensor>& w2, torch::Tensor& out,
               const c10::optional<torch::Tensor>& bias, const c10::optional<torch::Tensor>& r, const c10::optional<torch::Tensor>& g,
               const c10::optional<torch::Tensor>& x, const c10::optional<torch::Tensor>& xst, const c10::optional<torch::Tensor>& sb, int cfg,
               cudaStream_t st) {
  g120::Params<T> p{};
  const bool paired = epi == g120::EPI_ADALN || epi == g120::EPI_SWIGLU;
  p.A = reinterpret_cast<const T*>(a.data_ptr());
  p.W1 = reinterpret_cast<const T*>(w1.data_ptr());
  p.W2 = opt<T>(w2);
  p.N1 = (int)w1.size(0);
  p.M = (int)a.size(0);
  p.K = (int)a.size(1);
  p.N = paired ? (int)w1.size(0) : (int)(w1.size(0) + (w2.has_value() ? w2->size(0) : 0));
  p.C = reinterpret_cast<T*>(out.data_ptr());
  p.ldc = out.stride(0);
  p.bias = opt<T>(bias);
  p.R = opt<T>(r);
  p.G = opt<T>(g);
  p.P = g.has_value() ? g->size(0) : 1;
  p.X = opt<T>(x);
  p.xst = xst.has_value() ? reinterpret_cast<const float2*>(xst->data_ptr<float>()) : nullptr;
  p.sb = opt<T>(sb);
  TORCH_CHECK(out.dim() == 2 && out.size(0) == p.M && out.size(1) == p.N && out.stride(1) == 1, "gemm120: out a row-major [M, N] view");
  TORCH_CHECK(!paired || (w2.has_value() && w2->size(0) == w1.size(0)), "gemm120: the paired epilogues take W1, W2 of one shape");
  TORCH_CHECK(epi != g120::EPI_GATERES || (g.has_value() && g->size(1) == p.N && (!r.has_value() || r->size(1) == p.N)), "gemm120: G [P, N], R [M, N]");
  TORCH_CHECK(epi != g120::EPI_ADALN || (x.has_value() && xst.has_value() && sb.has_value() && x->size(1) == p.N && p.ldc == p.N),
              "gemm120: ADALN takes X [M, N], its row statistics, sb [N]");
  switch (epi) {
    case g120::EPI_PLAIN: dispatch_gemm120<T, g120::EPI_PLAIN>(p, cfg, st); break;
    case g120::EPI_GATERES: dispatch_gemm120<T, g120::EPI_GATERES>(p, cfg, st); break;
    case g120::EPI_ADALN: dispatch_gemm120<T, g120::EPI_ADALN>(p, cfg, st); break;
    case g120::EPI_SWIGLU: dispatch_gemm120<T, g120::EPI_SWIGLU>(p, cfg, st); break;
    default: TORCH_CHECK(false, "gemm120: epilogue 0 .. 3");
  }
}
}  // namespace

// RTX 5090: the block's inference GEMMs with their epilogues (gemm_sm120.cuh): out = epi(a [M, K] [w1; w2]^T); every tensor contiguous
// (out: a row-major view, any leading dimension), one dtype (bf16, or fp32 on TF32), xst fp32 [M, 2]
void gemm120(int64_t epi, torch::Tensor a, torch::Tensor w1, c10::optional<torch::Tensor> w2, torch::Tensor out, c10::optional<torch::Tensor> bias,
             c10::optional<torch::Tensor> r, c10::optional<torch::Tensor> g, c10::optional<torch::Tensor> x, c10::optional<torch::Tensor> xst,
             c10::optional<torch::Tensor> sb, int64_t cfg) {
  const at::cuda::CUDAGuard guard(a.device());
  TORCH_CHECK(a.is_contiguous() && w1.is_contiguous() && w1.size(1) == a.size(1) && a.scalar_type() == w1.scalar_type() && out.scalar_type() == a.scalar_type(),
              "gemm120: contiguous a [M, K], w [N, K] of one dtype, out of that dtype");
  auto st = at::cuda::getCurrentCUDAStream();
  if (a.scalar_type() == at::kBFloat16) gemm120_t<__nv_bfloat16>(epi, a, w1, w2, out, bias, r, g, x, xst, sb, (int)cfg, st);
  else {
    TORCH_CHECK(a.scalar_type() == at::kFloat, "gemm120: bf16 or fp32");
    gemm120_t<float>(epi, a, w1, w2, out, bias, r, g, x, xst, sb, (int)cfg, st);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// RTX 5090 training: dh = a w^T (a = dz [M, K], w = Ws^T [N, K]) with the SwiGLU backward in the epilogue: dab [M, 2 N] = [da | db], h [M, N]
// from ab [M, 2 N] = [a | b] (bf16, contiguous; dh never leaves the registers)
namespace {
template <typename B>
void swiglu_bwd120_t(torch::Tensor& a, torch::Tensor& w, torch::Tensor& ab, torch::Tensor& dab, torch::Tensor& h, int64_t cfg) {
  const int64_t M = a.size(0), K = a.size(1), N = w.size(0);
  g120::Params<B> p{};
  p.A = reinterpret_cast<const B*>(a.data_ptr());
  p.W1 = reinterpret_cast<const B*>(w.data_ptr());
  p.N1 = (int)N;
  p.M = (int)M;
  p.K = (int)K;
  p.N = (int)N;
  p.C = reinterpret_cast<B*>(dab.data_ptr());
  p.ldc = 2 * N;
  p.AB = reinterpret_cast<const B*>(ab.data_ptr());
  p.H = reinterpret_cast<B*>(h.data_ptr());
  dispatch_gemm120<B, g120::EPI_SWIGLU_BWD>(p, (int)cfg, at::cuda::getCurrentCUDAStream());
}
}  // namespace
void gemm120_swiglu_bwd(torch::Tensor a, torch::Tensor w, torch::Tensor ab, torch::Tensor dab, torch::Tensor h, int64_t cfg) {
  const at::cuda::CUDAGuard guard(a.device());
  const int64_t M = a.size(0), K = a.size(1), N = w.size(0);
  for (auto* t : {&a, &w, &ab, &dab, &h})
    TORCH_CHECK(t->is_contiguous() && t->scalar_type() == a.scalar_type() && t->dim() == 2, "gemm120_swiglu_bwd: contiguous 2-D of one dtype (bf16, or fp32 on TF32)");
  TORCH_CHECK(w.size(1) == K && ab.size(0) == M && ab.size(1) == 2 * N && dab.sizes() == ab.sizes() && h.size(0) == M && h.size(1) == N, "gemm120_swiglu_bwd: shapes");
  if (a.scalar_type() == at::kBFloat16) swiglu_bwd120_t<__nv_bfloat16>(a, w, ab, dab, h, cfg);
  else {
    TORCH_CHECK(a.scalar_type() == at::kFloat, "gemm120_swiglu_bwd: bf16 or fp32");
    swiglu_bwd120_t<float>(a, w, ab, dab, h, cfg);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// the (mean, rstd) [M, 2] fp32 of every row of x [M, 768] (bf16 / fp32): the AdaLN epilogue's LayerNorm statistics
void row_stats120(torch::Tensor x, torch::Tensor st, double eps) {
  const at::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.is_contiguous() && x.size(1) == 768 && st.scalar_type() == at::kFloat, "row_stats120: x [M, 768] contiguous, st fp32");
  const int M = x.size(0);
  auto s = at::cuda::getCurrentCUDAStream();
  auto* sp = reinterpret_cast<float2*>(st.data_ptr<float>());
  if (x.scalar_type() == at::kBFloat16)
    g120::row_stats<__nv_bfloat16, 768><<<(M + 7) / 8, 256, 0, s>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), sp, M, (float)eps);
  else g120::row_stats<float, 768><<<(M + 7) / 8, 256, 0, s>>>(x.data_ptr<float>(), sp, M, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---- the fp32 inference block's row passes (rows_f32_sm120.cuh)
namespace {
void f32_rows(const torch::Tensor& t, int64_t cols, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kFloat && t.dim() == 2 && t.size(1) == cols && t.stride(1) == 1 && t.stride(0) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
              name, ": an fp32 [rows, ", cols, "] view, unit column stride, 16-byte aligned rows");
}
}  // namespace

// c3 [3, M, dc] = (LN(c) w1, LN(c) w2, c); c [M, dc] fp32 rows, dc 384 / 768
void cond3_f32(torch::Tensor c, torch::Tensor w1, torch::Tensor w2, torch::Tensor c3, double eps) {
  const at::cuda::CUDAGuard guard(c.device());
  const int64_t M = c.size(0), dc = c.size(1);
  f32_rows(c, dc, "cond3_f32 c");
  TORCH_CHECK(c3.is_contiguous() && c3.scalar_type() == at::kFloat && (c3.numel() == 3 * M * dc || c3.numel() == 2 * M * dc), "cond3_f32: c3 [3 or 2, M, dc] fp32 contiguous");
  const bool copy = c3.numel() == 3 * M * dc;
  TORCH_CHECK(w1.is_contiguous() && w2.is_contiguous() && w1.numel() == dc && w2.numel() == dc && w1.scalar_type() == at::kFloat && w2.scalar_type() == at::kFloat,
              "cond3_f32: w1, w2 [dc] fp32");
  auto s = at::cuda::getCurrentCUDAStream();
  const unsigned grid = (unsigned)((M + r120::RW - 1) / r120::RW);
  if (dc == 384) r120::cond3_kernel<384><<<grid, r120::RW * 32, 0, s>>>(c.data_ptr<float>(), c.stride(0), w1.data_ptr<float>(), w2.data_ptr<float>(), c3.data_ptr<float>(), M, (float)eps, copy);
  else if (dc == 768) r120::cond3_kernel<768><<<grid, r120::RW * 32, 0, s>>>(c.data_ptr<float>(), c.stride(0), w1.data_ptr<float>(), w2.data_ptr<float>(), c3.data_ptr<float>(), M, (float)eps, copy);
  else TORCH_CHECK(false, "cond3_f32: dc 384 or 768");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out = x + sigmoid(g + bs) y, xa = sigmoid(S + sb) LN(out) + B; y / x / out / xa [M, 768] contiguous, g [M, 768] view, sbt [M, 1536] view
void resgate_adaln_f32(torch::Tensor y, torch::Tensor g, torch::Tensor bs, torch::Tensor x, torch::Tensor sbt, torch::Tensor sb, torch::Tensor out, torch::Tensor xa,
                       double eps) {
  const at::cuda::CUDAGuard guard(y.device());
  const int64_t M = y.size(0);
  for (auto* t : {&y, &x, &out, &xa}) TORCH_CHECK(t->is_contiguous() && t->size(0) == M, "resgate_adaln_f32: contiguous [M, 768] rows");
  f32_rows(y, 768, "resgate_adaln_f32 y");
  f32_rows(x, 768, "resgate_adaln_f32 x");
  f32_rows(out, 768, "resgate_adaln_f32 out");
  f32_rows(xa, 768, "resgate_adaln_f32 xa");
  f32_rows(g, 768, "resgate_adaln_f32 g");
  f32_rows(sbt, 1536, "resgate_adaln_f32 sbt");
  TORCH_CHECK(g.size(0) == M && sbt.size(0) == M, "resgate_adaln_f32: g, sbt with M rows");
  TORCH_CHECK(bs.is_contiguous() && sb.is_contiguous() && bs.numel() == 768 && sb.numel() == 768 && bs.scalar_type() == at::kFloat && sb.scalar_type() == at::kFloat,
              "resgate_adaln_f32: bs, sb [768] fp32");
  r120::resgate_adaln_kernel<768><<<(unsigned)((M + r120::RW - 1) / r120::RW), r120::RW * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      y.data_ptr<float>(), g.data_ptr<float>(), g.stride(0), bs.data_ptr<float>(), x.data_ptr<float>(), sbt.data_ptr<float>(), sbt.stride(0), sb.data_ptr<float>(),
      out.data_ptr<float>(), xa.data_ptr<float>(), M, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// P [16, L, L] fp32 = softmax over the keys of the pair bias of z [L, L, C] fp32 (C 128 / 256) with W' = wcm [C, 16] fp32, kv a bool [L] key mask or empty;
// L a multiple of 128, at most 1024 (the row's biases in shared memory)
void pair_softmax_f32(torch::Tensor z, torch::Tensor wcm, torch::Tensor kv, torch::Tensor p, int64_t L, double eps, int64_t cfg) {
  const at::cuda::CUDAGuard guard(z.device());
  TORCH_CHECK(z.is_cuda() && z.scalar_type() == at::kFloat && z.is_contiguous() && wcm.is_contiguous() && wcm.scalar_type() == at::kFloat && wcm.dim() == 2 && wcm.size(1) == r120::PS_H,
              "pair_softmax_f32: z fp32 contiguous [L, L, C], wcm fp32 [C, 16]");
  const int64_t C = wcm.size(0);
  TORCH_CHECK(L % 128 == 0 && L <= 1024 && z.numel() == L * L * C && (C == 128 || C == 256), "pair_softmax_f32: L a multiple of 128 <= 1024, C 128 or 256");
  TORCH_CHECK(p.is_contiguous() && p.scalar_type() == at::kFloat && p.numel() == r120::PS_H * L * L, "pair_softmax_f32: p fp32 [16, L, L]");
  const bool* mk = nullptr;
  if (kv.numel() > 0) {
    TORCH_CHECK(kv.scalar_type() == at::kBool && kv.is_contiguous() && kv.numel() == L, "pair_softmax_f32: kv bool [L]");
    mk = kv.data_ptr<bool>();
  }
  auto s = at::cuda::getCurrentCUDAStream();
  auto run = [&](auto kern, int smem, int nt) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kern<<<(unsigned)L, nt, smem, s>>>(z.data_ptr<float>(), wcm.data_ptr<float>(), mk, p.data_ptr<float>(), (int)L, (float)eps);
  };
  // cfg: 0 256 threads, 4 elements a warp; 1 512 threads, 2 elements; 2 512 threads, 4 elements
  if (C == 128) {
    if (cfg == 1) run(r120::pair_softmax_kernel<128, 2, 512>, r120::ps_smem<128>((int)L), 512);
    else if (cfg == 2) run(r120::pair_softmax_kernel<128, 4, 512>, r120::ps_smem<128>((int)L), 512);
    else run(r120::pair_softmax_kernel<128, 4>, r120::ps_smem<128>((int)L), 256);
  } else {
    if (cfg == 1) run(r120::pair_softmax_kernel<256, 2, 512>, r120::ps_smem<256>((int)L), 512);
    else if (cfg == 2) run(r120::pair_softmax_kernel<256, 4, 512>, r120::ps_smem<256>((int)L), 512);
    else run(r120::pair_softmax_kernel<256, 4>, r120::ps_smem<256>((int)L), 256);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// y = x + sigmoid(g + b) z; x / z / y [M, 768] contiguous, g [M, 768] view
void gate_res_b_f32(torch::Tensor x, torch::Tensor z, torch::Tensor g, torch::Tensor b, torch::Tensor y) {
  const at::cuda::CUDAGuard guard(x.device());
  const int64_t M = x.size(0);
  for (auto* t : {&x, &z, &y}) TORCH_CHECK(t->is_contiguous() && t->size(0) == M, "gate_res_b_f32: contiguous [M, 768] rows");
  f32_rows(x, 768, "gate_res_b_f32 x");
  f32_rows(z, 768, "gate_res_b_f32 z");
  f32_rows(y, 768, "gate_res_b_f32 y");
  f32_rows(g, 768, "gate_res_b_f32 g");
  TORCH_CHECK(g.size(0) == M && b.is_contiguous() && b.numel() == 768 && b.scalar_type() == at::kFloat, "gate_res_b_f32: g [M, 768], b [768] fp32");
  const long n = M * 192;
  const unsigned grid = (unsigned)std::min<long>((n + 255) / 256, 170L * 8);
  r120::gate_res_b_kernel<768><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(), z.data_ptr<float>(), g.data_ptr<float>(), g.stride(0),
                                                                                    b.data_ptr<float>(), y.data_ptr<float>(), M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef MWE_SM120
  m.def("gemm120", &gemm120);
  m.def("row_stats120", &row_stats120);
  m.def("gemm120_swiglu_bwd", &gemm120_swiglu_bwd);
  m.def("cond3_f32", &cond3_f32);
  m.def("resgate_adaln_f32", &resgate_adaln_f32);
  m.def("gate_res_b_f32", &gate_res_b_f32);
  m.def("pair_softmax_f32", &pair_softmax_f32);
  m.def("lt_mm32", &lt120::mm);
  m.def("lt_algos32", &lt120::algos);
  m.def("lt_mm", &lt120::mmg);
  m.def("lt_algos", &lt120::algosg);
#endif
  m.def("pv_gate", &pv_gate);
  m.def("pv_gate32", &pv_gate32);
  m.def("dpb", &dpb);
  m.def("dpb32", &dpb32);
  m.def("pv_planes", &pv_planes);
  m.def("dpb_planes", &dpb_planes);
  m.def("delta_planes", &delta_planes);
}
