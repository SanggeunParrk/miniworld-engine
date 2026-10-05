// ops.cu -- the torch extension of the A100 (sm_80) SWA atom DiT kernels: the stages of the fused block (kernels/swa_dit), bf16, C = 128, 4 heads x 32.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "attn_fwd_sm80.cuh"
#include "qkvg_fwd_sm80.cuh"
#include "ffn_fwd_sm80.cuh"
#include "mod_sm80.cuh"
#include "oproj_bwd_sm80.cuh"
#include "ffn_bwd_sm80.cuh"
#include "qkvg_bwd_sm80.cuh"
#include "attn_bwd_sm80.cuh"

#define CHECK_CUDA_BF16(x) TORCH_CHECK((x).is_cuda() && (x).scalar_type() == torch::kBFloat16, #x " must be a CUDA bf16 tensor")
#define CHECK_CUDA_F32(x) TORCH_CHECK((x).is_cuda() && (x).scalar_type() == torch::kFloat32, #x " must be a CUDA fp32 tensor")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

namespace {
using bf = __nv_bfloat16;
inline const bf* bptr(const torch::Tensor& t) { return reinterpret_cast<const bf*>(t.data_ptr()); }
inline bf* bptr_w(torch::Tensor& t) { return reinterpret_cast<bf*>(t.data_ptr()); }
inline int num_sms() { return at::cuda::getCurrentDeviceProperties()->multiProcessorCount; }
}  // namespace

template <class G>
void launch_qkvg(const sw80::QkvgFwdParams& p) {
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(sw80::qkvg_fwd_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM);
    attr = true;
  }
  const int ntile = (p.M + G::ROWS - 1) / G::ROWS;
  const unsigned ctas = (unsigned)std::min(num_sms(), ntile);
  sw80::qkvg_fwd_kernel<G><<<ctas, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// the element strides (sample, row, head) of a bf16 [N, S, 4, 32] view: any strides with contiguous channels and 16-byte aligned rows (a dimension of size 1 may carry any stride)
sw80::Str3 str3(const torch::Tensor& t, const char* what) {
  CHECK_CUDA_BF16(t);
  TORCH_CHECK(t.dim() == 4 && t.size(2) == sw80::AF_H && t.size(3) == sw80::AF_HD && t.stride(3) == 1, what, " must be a [N, S, 4, 32] view with unit channel stride");
  for (int d = 0; d < 3; ++d) TORCH_CHECK(t.size(d) == 1 || t.stride(d) % 8 == 0, what, " must have strides that are multiples of 8 elements");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(t.data_ptr()) & 15) == 0, what, " must be 16-byte aligned");
  return {t.stride(0), t.stride(1), t.stride(2)};
}

// window attention forward: Q, K, V viewed as [N, S, 4, 32] bf16 (any strides, see str3), seqused [N] int32 -> O [N S, C] bf16, lse [N, H, S] fp32 (preallocated outputs)
void swa_attn_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor seqused, torch::Tensor o, torch::Tensor lse) {
  CHECK_CUDA_BF16(o); CHECK_CONTIG(o); CHECK_CONTIG(lse);
  const sw80::Str3 qs = str3(q, "q"), ks = str3(k, "k"), vs = str3(v, "v");
  TORCH_CHECK(seqused.scalar_type() == torch::kInt32 && seqused.is_cuda() && seqused.numel() == q.size(0), "seqused must be int32 [N]");
  const int64_t N = q.size(0), S = q.size(1);
  TORCH_CHECK(k.size(0) == N && k.size(1) == S && v.size(0) == N && v.size(1) == S, "q, k, v must have the same [N, S]");
  TORCH_CHECK(o.numel() == N * S * sw80::AF_C && lse.numel() == N * sw80::AF_H * S && lse.scalar_type() == torch::kFloat32, "bad output sizes");
  c10::cuda::CUDAGuard guard(q.device());
  sw80::AttnFwdParams p{bptr(q), bptr(k), bptr(v), seqused.data_ptr<int>(), bptr_w(o), lse.data_ptr<float>(), (int)S,
                        0.17677669529663687f, 0.17677669529663687f * 1.4426950408889634f, qs, ks, vs};
  const dim3 grid((unsigned)((S + sw80::AF_QT - 1) / sw80::AF_QT), (unsigned)(N * sw80::AF_H));
  const bool hm = qs.s == sw80::AF_HD && ks.s == sw80::AF_HD && vs.s == sw80::AF_HD;       // the fused block's head-major planes: the row strides are compile-time
  if (hm) sw80::attn_fwd_kernel<true><<<grid, sw80::AF_NW * 32, sw80::AF_SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  else sw80::attn_fwd_kernel<false><<<grid, sw80::AF_NW * 32, sw80::AF_SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// qkvg stage: x [M, C] bf16 (M = N S), mod [B S, 6C] fp32, cos / sin [B S, 16] fp32, Wqkv [3C, C], Wg [C, C] -> Qh, Kh, Vh [N, H, S, 32], G [M, C]; with save also x, p_q, p_k [M, C]
void swa_qkvg_fwd(torch::Tensor x, torch::Tensor mod, torch::Tensor cos, torch::Tensor sin, torch::Tensor wqkv, torch::Tensor wg, torch::Tensor qh,
                  torch::Tensor kh, torch::Tensor vh, torch::Tensor g, torch::Tensor xs, torch::Tensor pqs, torch::Tensor pks, int64_t S, int64_t B,
                  double eps, double qk_eps, bool save, int64_t cfg) {
  CHECK_CUDA_BF16(x); CHECK_CUDA_BF16(wqkv); CHECK_CUDA_BF16(wg); CHECK_CUDA_BF16(qh); CHECK_CUDA_BF16(kh); CHECK_CUDA_BF16(vh); CHECK_CUDA_BF16(g);
  CHECK_CUDA_F32(mod); CHECK_CUDA_F32(cos); CHECK_CUDA_F32(sin);
  CHECK_CONTIG(x); CHECK_CONTIG(mod); CHECK_CONTIG(cos); CHECK_CONTIG(sin); CHECK_CONTIG(wqkv); CHECK_CONTIG(wg);
  CHECK_CONTIG(qh); CHECK_CONTIG(kh); CHECK_CONTIG(vh); CHECK_CONTIG(g);
  const int64_t M = x.size(0);
  TORCH_CHECK(x.size(1) == 128 && M % S == 0 && mod.numel() == B * S * 768 && cos.numel() == B * S * 16 && sin.numel() == B * S * 16, "bad operand sizes");
  TORCH_CHECK(wqkv.size(0) == 384 && wqkv.size(1) == 128 && wg.size(0) == 128 && wg.size(1) == 128, "bad weight sizes");
  if (save) { CHECK_CUDA_BF16(xs); CHECK_CUDA_BF16(pqs); CHECK_CUDA_BF16(pks); CHECK_CONTIG(xs); CHECK_CONTIG(pqs); CHECK_CONTIG(pks); }
  c10::cuda::CUDAGuard guard(x.device());
  sw80::QkvgFwdParams p{bptr(x), mod.data_ptr<float>(), cos.data_ptr<float>(), sin.data_ptr<float>(), bptr(wqkv), bptr(wg), bptr_w(qh), bptr_w(kh), bptr_w(vh), bptr_w(g),
                        save ? bptr_w(xs) : nullptr, save ? bptr_w(pqs) : nullptr, save ? bptr_w(pks) : nullptr, (int)M, (int)S, (int)B, (float)eps, (float)qk_eps};
  switch (cfg) {
    case 0: launch_qkvg<sw80::QkvgCfg<8, 2>>(p); break;
    case 1: launch_qkvg<sw80::QkvgCfg<16, 1>>(p); break;
    case 2: launch_qkvg<sw80::QkvgCfg<8, 1>>(p); break;
    default: TORCH_CHECK(false, "unknown qkvg configuration");
  }
}

template <class G>
void launch_ffn(const sw80::FfnFwdParams& p) {
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(sw80::ffn_fwd_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM);
    attr = true;
  }
  const int ntile = ((p.S + 15) / 16) * (p.M / p.S);
  const unsigned ctas = (unsigned)std::min(num_sms(), (ntile + G::NW - 1) / G::NW);
  sw80::ffn_fwd_kernel<G><<<ctas, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out-projection + gated residual + FFN stage: Qin, G, O [M, C] bf16, mod [B S, 6C] fp32, Wo [C, C], Wu [2 hidden, C], Wd [C, hidden] -> out [M, C]; with save also q1, att, y, ffn [M, C]
void swa_ffn_fwd(torch::Tensor qi, torch::Tensor g, torch::Tensor o, torch::Tensor mod, torch::Tensor wo, torch::Tensor wu, torch::Tensor wd, torch::Tensor out,
                 torch::Tensor q1s, torch::Tensor atts, torch::Tensor ys, torch::Tensor ffs, int64_t S, int64_t B, double eps, bool save, int64_t cfg,
                 torch::Tensor prof) {
  CHECK_CUDA_BF16(qi); CHECK_CUDA_BF16(g); CHECK_CUDA_BF16(o); CHECK_CUDA_BF16(wo); CHECK_CUDA_BF16(wu); CHECK_CUDA_BF16(wd); CHECK_CUDA_BF16(out); CHECK_CUDA_F32(mod);
  CHECK_CONTIG(qi); CHECK_CONTIG(g); CHECK_CONTIG(o); CHECK_CONTIG(mod); CHECK_CONTIG(wo); CHECK_CONTIG(wu); CHECK_CONTIG(wd); CHECK_CONTIG(out);
  const int64_t M = qi.size(0);
  TORCH_CHECK(qi.size(1) == 128 && M % S == 0 && mod.numel() == B * S * 768, "bad operand sizes");
  TORCH_CHECK(wo.size(0) == 128 && wo.size(1) == 128 && wu.size(0) == 512 && wu.size(1) == 128 && wd.size(0) == 128 && wd.size(1) == 256, "bad weight sizes");
  if (save) { CHECK_CUDA_BF16(q1s); CHECK_CUDA_BF16(atts); CHECK_CUDA_BF16(ys); CHECK_CUDA_BF16(ffs); CHECK_CONTIG(q1s); CHECK_CONTIG(atts); CHECK_CONTIG(ys); CHECK_CONTIG(ffs); }
  c10::cuda::CUDAGuard guard(qi.device());
  sw80::FfnFwdParams p{bptr(qi), bptr(g), bptr(o), mod.data_ptr<float>(), bptr(wo), bptr(wu), bptr(wd), bptr_w(out), save ? bptr_w(q1s) : nullptr,
                       save ? bptr_w(atts) : nullptr, save ? bptr_w(ys) : nullptr, save ? bptr_w(ffs) : nullptr, (int)M, (int)S, (int)B, (float)eps,
                       prof.numel() ? reinterpret_cast<unsigned long long*>(prof.data_ptr()) : nullptr};
  switch (cfg) {
    case 0: launch_ffn<sw80::FfnCfg<8>>(p); break;
    default: TORCH_CHECK(false, "unknown ffn configuration");
  }
}

template <class G>
void launch_mod_fwd(const sw80::ModFwdParams& p) {
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(sw80::mod_fwd_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM);
    attr = true;
  }
  const int ntile = (p.R + G::BM - 1) / G::BM;
  const unsigned ctas = (unsigned)std::min(num_sms() * G::MINB, ntile);
  sw80::mod_fwd_kernel<G><<<ctas, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class G>
void launch_mod_dc(const sw80::ModDcParams& p) {
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(sw80::mod_dc_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM);
    attr = true;
  }
  const int ntile = (p.R + G::BM - 1) / G::BM;
  const unsigned ctas = (unsigned)std::min(num_sms() * G::MINB, ntile);
  sw80::mod_dc_kernel<G><<<ctas, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int TERMS>
void launch_mod_dw(const sw80::ModDwParams& p, int64_t parts) {
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(sw80::mod_dw_kernel<TERMS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sw80::DW_SMEM);
    attr = true;
  }
  sw80::mod_dw_kernel<TERMS><<<dim3((unsigned)parts, 6), sw80::DW_THR, sw80::DW_SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// adaLN modulation forward: c [R, 128] bf16, Wmod [768, 128] bf16 -> out [R, 768] fp32 = rn(silu(c)) Wmod^T; with save also a = rn(silu(c)) [R, 128] bf16 (the backward's operand);
// cfg 0: row-stationary kernel with tiles of 128 rows, 1: of 64; 2: weight-stationary kernel
void swa_mod_fwd(torch::Tensor c, torch::Tensor w, torch::Tensor out, torch::Tensor a, bool save, int64_t cfg) {
  CHECK_CUDA_BF16(c); CHECK_CUDA_BF16(w); CHECK_CUDA_F32(out);
  CHECK_CONTIG(c); CHECK_CONTIG(w); CHECK_CONTIG(out);
  const int64_t R = c.size(0);
  TORCH_CHECK(c.dim() == 2 && c.size(1) == 128 && w.dim() == 2 && w.size(0) == 768 && w.size(1) == 128 && out.dim() == 2 && out.size(0) == R && out.size(1) == 768,
              "bad modulation operand sizes");
  if (save) { CHECK_CUDA_BF16(a); CHECK_CONTIG(a); TORCH_CHECK(a.numel() == R * 128, "bad saved-activation size"); }
  if (R == 0) return;
  c10::cuda::CUDAGuard guard(c.device());
  sw80::ModFwdParams p{bptr(c), bptr(w), out.data_ptr<float>(), save ? bptr_w(a) : nullptr, (int)R};
  switch (cfg) {
    case 0: launch_mod_fwd<sw80::ModFwdCfg<4>>(p); break;
    case 1: launch_mod_fwd<sw80::ModFwdCfg<2>>(p); break;
    case 2: {
      const int ntile = (int)((R + sw80::MW_TR - 1) / sw80::MW_TR);
      const unsigned parts = (unsigned)std::max(1, std::min(ntile, 3 * num_sms() / sw80::MW_CHUNKS));
      sw80::mod_fwd_ws_kernel<<<dim3(parts, sw80::MW_CHUNKS), sw80::MW_THR, sw80::MW_SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } break;
    default: TORCH_CHECK(false, "unknown modulation forward configuration");
  }
}

// adaLN modulation backward: g = d mod [R, 768] fp32, c and a = rn(silu(c)) [R, 128] bf16, Wmod [768, 128] bf16 -> dc [R, 128] bf16, dWmod [768, 128] bf16; part: fp32 [P, 768, 128]
// scratch (P partial sums of dWmod).  dc_cfg 0: tiles of 128 rows, 2 bf16 terms of g; 1: 64 rows, 2 terms; 2 / 3: the same with 1 term; 4 .. 7: 0 .. 3 with chunks of 64 channels.  dw_terms 1 or 2.
void swa_mod_bwd(torch::Tensor g, torch::Tensor c, torch::Tensor a, torch::Tensor w, torch::Tensor dc, torch::Tensor part, torch::Tensor dw, int64_t dc_cfg, int64_t dw_terms) {
  CHECK_CUDA_F32(g); CHECK_CUDA_BF16(c); CHECK_CUDA_BF16(a); CHECK_CUDA_BF16(w); CHECK_CUDA_BF16(dc); CHECK_CUDA_BF16(dw); CHECK_CUDA_F32(part);
  CHECK_CONTIG(g); CHECK_CONTIG(c); CHECK_CONTIG(a); CHECK_CONTIG(w); CHECK_CONTIG(dc); CHECK_CONTIG(dw); CHECK_CONTIG(part);
  const int64_t R = c.size(0), P = part.size(0);
  TORCH_CHECK(c.dim() == 2 && c.size(1) == 128 && g.dim() == 2 && g.size(0) == R && g.size(1) == 768 && a.numel() == R * 128 && w.dim() == 2 && w.size(0) == 768 && w.size(1) == 128 &&
              dc.numel() == R * 128 && dw.numel() == 768 * 128 && part.dim() == 3 && part.size(1) == 768 && part.size(2) == 128 && P >= 1,
              "bad modulation backward operand sizes");
  if (R == 0) { dw.zero_(); return; }
  c10::cuda::CUDAGuard guard(c.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  sw80::ModDcParams pd{g.data_ptr<float>(), bptr(c), bptr(w), bptr_w(dc), (int)R};
  switch (dc_cfg) {
    case 0: launch_mod_dc<sw80::ModDcCfg<8, 2>>(pd); break;
    case 1: launch_mod_dc<sw80::ModDcCfg<4, 2>>(pd); break;
    case 2: launch_mod_dc<sw80::ModDcCfg<8, 1>>(pd); break;
    case 3: launch_mod_dc<sw80::ModDcCfg<4, 1>>(pd); break;
    case 4: launch_mod_dc<sw80::ModDcCfg<4, 2, 64, 2>>(pd); break;
    case 5: launch_mod_dc<sw80::ModDcCfg<4, 1, 64, 2>>(pd); break;
    case 6: launch_mod_dc<sw80::ModDcCfg<8, 2, 64, 2>>(pd); break;
    case 7: launch_mod_dc<sw80::ModDcCfg<8, 1, 64, 2>>(pd); break;
    default: TORCH_CHECK(false, "unknown modulation backward configuration");
  }
  const int nchunk = (int)((R + sw80::DW_RC - 1) / sw80::DW_RC), cpc = (int)((nchunk + P - 1) / P);
  sw80::ModDwParams pw{g.data_ptr<float>(), bptr(a), part.data_ptr<float>(), (int)R, cpc};
  if (dw_terms == 1) launch_mod_dw<1>(pw, P);
  else if (dw_terms == 2) launch_mod_dw<2>(pw, P);
  else TORCH_CHECK(false, "dw_terms must be 1 or 2");
  sw80::mod_dw_reduce_kernel<<<(768 * 128 / 4 + 255) / 256, 256, 0, stream>>>(part.data_ptr<float>(), bptr_w(dw), (int)P);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ================================================================== the row-tiled backward stages: two modes (``bwd_rows_sm80.cuh``): 0 every row has its own modulation row (the modulation gradient is
// stored into dmod [B S, 768]), 1 the conditioning is shared by the A = N / B samples of a batch element, A a multiple of 16 (the modulation gradient goes into the A / 16 partial buffers
// dmod [A / 16, B S, 768], which the caller adds)

// the number of 16-row tiles of a call
inline int bwd_ntile_host(int64_t mode, int64_t M, int64_t S, int64_t B) {
  if (mode == sw80::MODE_SINGLE) return (int)(((S + 15) / 16) * (M / S));
  return (int)(S * B * (M / S / B / 16));
}

template <class F>
void with_mode(int64_t mode, F&& f) {
  if (mode == sw80::MODE_SINGLE) f(std::integral_constant<int, sw80::MODE_SINGLE>{});
  else if (mode == sw80::MODE_HOIST) f(std::integral_constant<int, sw80::MODE_HOIST>{});
  else TORCH_CHECK(false, "unknown backward mode");
}

// the shape rules of the two modes: the modulation gradient buffer holds B S rows (mode 0) or A / 16 times that (mode 1)
inline void check_bwd_mode(int64_t mode, int64_t M, int64_t S, int64_t B, int64_t dmod_numel) {
  TORCH_CHECK(M % S == 0 && (M / S) % B == 0, "bad sequence shape");
  if (mode == sw80::MODE_SINGLE) {
    TORCH_CHECK(M / S == B, "mode 0 needs one modulation row per sample (N == B)");
    TORCH_CHECK(dmod_numel == B * S * 768, "bad dmod size");
  } else {
    const int64_t A = M / S / B;
    TORCH_CHECK(A % 16 == 0 && A >= 16, "mode 1 needs A = N / B a multiple of 16");
    TORCH_CHECK(dmod_numel == (A / 16) * B * S * 768, "bad dmod partial size");
  }
}

template <class G, int MODE>
void launch_oproj_bwd(const sw80::OprojBwdParams& p) {
  const int ntile = bwd_ntile_host(MODE, p.M, p.S, p.B);
  const unsigned ctas = (unsigned)std::min(num_sms(), (ntile + G::NW - 1) / G::NW);
  sw80::oproj_bwd_kernel<G, MODE><<<ctas, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out-projection backward: dq1, O, G, Att [M, C] bf16, mod [B S, 6C] fp32, Wo^T [C, C] bf16 -> dO, dG, datt, gated [M, C] bf16, dv [N, 4, S] fp32 and d gate_a of dmod (columns 2C .. 3C)
void swa_oproj_bwd(torch::Tensor dq1, torch::Tensor o, torch::Tensor g, torch::Tensor att, torch::Tensor mod, torch::Tensor wot, torch::Tensor dO, torch::Tensor dG, torch::Tensor datt,
                   torch::Tensor gated, torch::Tensor dv, torch::Tensor dmod, int64_t S, int64_t B, int64_t mode, int64_t cfg) {
  CHECK_CUDA_BF16(dq1); CHECK_CUDA_BF16(o); CHECK_CUDA_BF16(g); CHECK_CUDA_BF16(att); CHECK_CUDA_BF16(wot); CHECK_CUDA_BF16(dO); CHECK_CUDA_BF16(dG); CHECK_CUDA_BF16(datt); CHECK_CUDA_BF16(gated);
  CHECK_CUDA_F32(mod); CHECK_CUDA_F32(dv); CHECK_CUDA_F32(dmod);
  CHECK_CONTIG(dq1); CHECK_CONTIG(o); CHECK_CONTIG(g); CHECK_CONTIG(att); CHECK_CONTIG(mod); CHECK_CONTIG(wot); CHECK_CONTIG(dO); CHECK_CONTIG(dG); CHECK_CONTIG(datt); CHECK_CONTIG(gated);
  CHECK_CONTIG(dv); CHECK_CONTIG(dmod);
  const int64_t M = dq1.size(0);
  TORCH_CHECK(dq1.size(1) == 128 && mod.numel() == B * S * 768 && dv.numel() == (M / S) * 4 * S && wot.size(0) == 128 && wot.size(1) == 128, "bad operand sizes");
  check_bwd_mode(mode, M, S, B, dmod.numel());
  c10::cuda::CUDAGuard guard(dq1.device());
  sw80::OprojBwdParams p{bptr(dq1), bptr(o), bptr(g), bptr(att), mod.data_ptr<float>(), bptr(wot), bptr_w(dO), bptr_w(dG), bptr_w(datt), bptr_w(gated), dv.data_ptr<float>(), dmod.data_ptr<float>(),
                         (int)M, (int)S, (int)B};
  with_mode(mode, [&](auto mc) {
    constexpr int MODE = decltype(mc)::value;
    switch (cfg) {
      case 0: launch_oproj_bwd<sw80::OprojBwdCfg<8>, MODE>(p); break;
      case 1: launch_oproj_bwd<sw80::OprojBwdCfg<4>, MODE>(p); break;
      default: TORCH_CHECK(false, "unknown out-projection backward configuration");
    }
  });
}

template <class G, int MODE>
void launch_ffn_bwd_gate(const sw80::FfnBwdGateParams& p) {
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(sw80::ffn_bwd_gate_kernel<G, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM);
    attr = true;
  }
  const int ntile = bwd_ntile_host(MODE, p.M, p.S, p.B), ngrp = (ntile + G::NW - 1) / G::NW;
  const unsigned ctas = (unsigned)std::min(num_sms() * G::MINB, ngrp);
  sw80::ffn_bwd_gate_kernel<G, MODE><<<ctas, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class G, int MODE>
void launch_ffn_bwd_dy(const sw80::FfnBwdDyParams& p) {
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(sw80::ffn_bwd_dy_kernel<G, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM);
    attr = true;
  }
  const int ntile = bwd_ntile_host(MODE, p.M, p.S, p.B);
  const unsigned ctas = (unsigned)std::min(num_sms(), (ntile + G::NW - 1) / G::NW);
  sw80::ffn_bwd_dy_kernel<G, MODE><<<ctas, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// FFN backward (two kernels): dy (= dq2), q1, y, ffn [M, C] bf16, mod [B S, 6C] fp32, Wu [2 hidden, C], Wd^T [hidden, C], Wu^T [C, 2 hidden] -> dq1, dffn [M, C], hh [M, hidden], dab [M, 2 hidden] bf16
// and d shift_f | d scale_f | d gate_f of dmod (columns 3C .. 6C).  gate_cfg 0: 8 warps x 1 CTA, 1: 4 x 2, 2: 6 x 2; dy_cfg 0: 16 warps, 1: 8
void swa_ffn_bwd(torch::Tensor dy, torch::Tensor q1, torch::Tensor y, torch::Tensor ffn, torch::Tensor mod, torch::Tensor wu, torch::Tensor wdt, torch::Tensor wut, torch::Tensor dq1, torch::Tensor dffn,
                 torch::Tensor hh, torch::Tensor dab, torch::Tensor dmod, int64_t S, int64_t B, double eps, int64_t mode, int64_t gate_cfg, int64_t dy_cfg) {
  CHECK_CUDA_BF16(dy); CHECK_CUDA_BF16(q1); CHECK_CUDA_BF16(y); CHECK_CUDA_BF16(ffn); CHECK_CUDA_BF16(wu); CHECK_CUDA_BF16(wdt); CHECK_CUDA_BF16(wut);
  CHECK_CUDA_BF16(dq1); CHECK_CUDA_BF16(dffn); CHECK_CUDA_BF16(hh); CHECK_CUDA_BF16(dab); CHECK_CUDA_F32(mod); CHECK_CUDA_F32(dmod);
  CHECK_CONTIG(dy); CHECK_CONTIG(q1); CHECK_CONTIG(y); CHECK_CONTIG(ffn); CHECK_CONTIG(mod); CHECK_CONTIG(wu); CHECK_CONTIG(wdt); CHECK_CONTIG(wut);
  CHECK_CONTIG(dq1); CHECK_CONTIG(dffn); CHECK_CONTIG(hh); CHECK_CONTIG(dab); CHECK_CONTIG(dmod);
  const int64_t M = dy.size(0);
  TORCH_CHECK(dy.size(1) == 128 && mod.numel() == B * S * 768 && wu.size(0) == 512 && wu.size(1) == 128 && wdt.size(0) == 256 && wdt.size(1) == 128 &&
              wut.size(0) == 128 && wut.size(1) == 512 && hh.numel() == M * 256 && dab.numel() == M * 512, "bad operand sizes");
  check_bwd_mode(mode, M, S, B, dmod.numel());
  c10::cuda::CUDAGuard guard(dy.device());
  sw80::FfnBwdGateParams pg{bptr(dy), bptr(y), bptr(ffn), mod.data_ptr<float>(), bptr(wu), bptr(wdt), bptr_w(dffn), bptr_w(hh), bptr_w(dab), dmod.data_ptr<float>(), (int)M, (int)S, (int)B};
  sw80::FfnBwdDyParams pd{bptr(dab), bptr(q1), bptr(dy), mod.data_ptr<float>(), bptr(wut), bptr_w(dq1), dmod.data_ptr<float>(), (int)M, (int)S, (int)B, (float)eps};
  with_mode(mode, [&](auto mc) {
    constexpr int MODE = decltype(mc)::value;
    switch (gate_cfg) {
      case 0: launch_ffn_bwd_gate<sw80::FfnBwdGateCfg<8, 1>, MODE>(pg); break;
      case 1: launch_ffn_bwd_gate<sw80::FfnBwdGateCfg<4, 2>, MODE>(pg); break;
      case 2: launch_ffn_bwd_gate<sw80::FfnBwdGateCfg<6, 2>, MODE>(pg); break;
      default: TORCH_CHECK(false, "unknown FFN backward (gate) configuration");
    }
    switch (dy_cfg) {
      case 0: launch_ffn_bwd_dy<sw80::FfnBwdDyCfg<16>, MODE>(pd); break;
      case 1: launch_ffn_bwd_dy<sw80::FfnBwdDyCfg<8>, MODE>(pd); break;
      default: TORCH_CHECK(false, "unknown FFN backward (dy) configuration");
    }
  });
}

template <class G, int MODE>
void launch_qkvg_bwd(const sw80::QkvgBwdParams& p) {
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(sw80::qkvg_bwd_kernel<G, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM);
    attr = true;
  }
  const int ntile = bwd_ntile_host(MODE, p.M, p.S, p.B);
  const unsigned ctas = (unsigned)std::min(num_sms(), (ntile + G::NW - 1) / G::NW);
  sw80::qkvg_bwd_kernel<G, MODE><<<ctas, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// qkvg backward: qi, pq, pk, dG, dq1 [M, C] bf16, dQ / dK / dV head-major [N, 4, S, 32] bf16, mod [B S, 6C] fp32, cos / sin [B S, 16] fp32, (Wqkv | Wg)^T [C, 4 C] bf16
// -> dq [M, C], dP [M, 4 C] bf16 and d shift_a | d scale_a of dmod (columns 0 .. 2C)
void swa_qkvg_bwd(torch::Tensor qi, torch::Tensor pq, torch::Tensor pk, torch::Tensor dqh, torch::Tensor dkh, torch::Tensor dvh, torch::Tensor dg, torch::Tensor dq1, torch::Tensor mod, torch::Tensor cos,
                  torch::Tensor sin, torch::Tensor wt, torch::Tensor dq, torch::Tensor dp, torch::Tensor dmod, int64_t S, int64_t B, double eps, double qk_eps, int64_t mode, int64_t cfg) {
  CHECK_CUDA_BF16(qi); CHECK_CUDA_BF16(pq); CHECK_CUDA_BF16(pk); CHECK_CUDA_BF16(dqh); CHECK_CUDA_BF16(dkh); CHECK_CUDA_BF16(dvh); CHECK_CUDA_BF16(dg); CHECK_CUDA_BF16(dq1);
  CHECK_CUDA_BF16(wt); CHECK_CUDA_BF16(dq); CHECK_CUDA_BF16(dp); CHECK_CUDA_F32(mod); CHECK_CUDA_F32(cos); CHECK_CUDA_F32(sin); CHECK_CUDA_F32(dmod);
  CHECK_CONTIG(qi); CHECK_CONTIG(pq); CHECK_CONTIG(pk); CHECK_CONTIG(dqh); CHECK_CONTIG(dkh); CHECK_CONTIG(dvh); CHECK_CONTIG(dg); CHECK_CONTIG(dq1); CHECK_CONTIG(mod); CHECK_CONTIG(cos);
  CHECK_CONTIG(sin); CHECK_CONTIG(wt); CHECK_CONTIG(dq); CHECK_CONTIG(dp); CHECK_CONTIG(dmod);
  const int64_t M = qi.size(0);
  TORCH_CHECK(qi.size(1) == 128 && mod.numel() == B * S * 768 && cos.numel() == B * S * 16 && sin.numel() == B * S * 16 && wt.size(0) == 128 && wt.size(1) == 512 && dp.numel() == M * 512 &&
              dqh.numel() == M * 128 && dkh.numel() == M * 128 && dvh.numel() == M * 128, "bad operand sizes");
  check_bwd_mode(mode, M, S, B, dmod.numel());
  c10::cuda::CUDAGuard guard(qi.device());
  sw80::QkvgBwdParams p{bptr(qi), bptr(pq), bptr(pk), bptr(dqh), bptr(dkh), bptr(dvh), bptr(dg), bptr(dq1), mod.data_ptr<float>(), cos.data_ptr<float>(), sin.data_ptr<float>(), bptr(wt),
                        bptr_w(dq), bptr_w(dp), dmod.data_ptr<float>(), (int)M, (int)S, (int)B, (float)eps, (float)qk_eps};
  with_mode(mode, [&](auto mc) {
    constexpr int MODE = decltype(mc)::value;
    switch (cfg) {
      case 0: launch_qkvg_bwd<sw80::QkvgBwdCfg<8>, MODE>(p); break;
      case 1: launch_qkvg_bwd<sw80::QkvgBwdCfg<4>, MODE>(p); break;
      default: TORCH_CHECK(false, "unknown qkvg backward configuration");
    }
  });
}

// window attention backward: Q, K, V and the outputs dQ, dK, dV viewed as [N, S, 4, 32] bf16 (any strides, see str3), dO [N S, C] bf16, lse and D [N, H, S] fp32, seqused [N] int32 (preallocated outputs)
void swa_attn_bwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor d_o, torch::Tensor lse, torch::Tensor dvv, torch::Tensor seqused, torch::Tensor dq, torch::Tensor dk, torch::Tensor dvo) {
  CHECK_CUDA_BF16(d_o); CHECK_CUDA_F32(lse); CHECK_CUDA_F32(dvv);
  CHECK_CONTIG(d_o); CHECK_CONTIG(lse); CHECK_CONTIG(dvv);
  const sw80::Str3 qs = str3(q, "q"), ks = str3(k, "k"), vs = str3(v, "v"), dqs = str3(dq, "dq"), dks = str3(dk, "dk"), dvs = str3(dvo, "dv");
  TORCH_CHECK(seqused.scalar_type() == torch::kInt32 && seqused.is_cuda() && seqused.numel() == q.size(0), "seqused must be int32 [N]");
  const int64_t N = q.size(0), S = q.size(1);
  TORCH_CHECK(d_o.numel() == N * S * sw80::AF_C && lse.numel() == N * sw80::AF_H * S && dvv.numel() == lse.numel() && dq.sizes() == q.sizes() && dk.sizes() == q.sizes() && dvo.sizes() == q.sizes() &&
              k.sizes() == q.sizes() && v.sizes() == q.sizes(), "bad operand sizes");
  c10::cuda::CUDAGuard guard(q.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  sw80::AttnBwdParams p{bptr(q), bptr(k), bptr(v), bptr(d_o), lse.data_ptr<float>(), dvv.data_ptr<float>(), seqused.data_ptr<int>(), bptr_w(dq), bptr_w(dk), bptr_w(dvo), (int)S,
                        0.17677669529663687f, 0.17677669529663687f * 1.4426950408889634f, qs, ks, vs, dqs, dks, dvs};
  const dim3 grid((unsigned)((S + sw80::AF_QT - 1) / sw80::AF_QT), (unsigned)(N * sw80::AF_H));
  const bool hm = qs.s == sw80::AF_HD && ks.s == sw80::AF_HD && vs.s == sw80::AF_HD && dqs.s == sw80::AF_HD && dks.s == sw80::AF_HD && dvs.s == sw80::AF_HD;
  if (hm) sw80::attn_bwd_dq_kernel<true><<<grid, sw80::AF_NW * 32, sw80::AF_SMEM, stream>>>(p);
  else sw80::attn_bwd_dq_kernel<false><<<grid, sw80::AF_NW * 32, sw80::AF_SMEM, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (hm) sw80::attn_bwd_dkv_kernel<true><<<grid, sw80::AF_NW * 32, sw80::AB_SMEM, stream>>>(p);
  else sw80::attn_bwd_dkv_kernel<false><<<grid, sw80::AF_NW * 32, sw80::AB_SMEM, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// D = rowsum(dO o) per row and head: dO and o [M, C] bf16 -> dvv [N, H, S] fp32 (M = N S; preallocated)
void swa_attn_delta(torch::Tensor d_o, torch::Tensor o, torch::Tensor dvv, int64_t S) {
  CHECK_CUDA_BF16(d_o); CHECK_CUDA_BF16(o); CHECK_CUDA_F32(dvv);
  CHECK_CONTIG(d_o); CHECK_CONTIG(o); CHECK_CONTIG(dvv);
  const int64_t M = d_o.numel() / sw80::AF_C;
  TORCH_CHECK(d_o.numel() == M * sw80::AF_C && o.numel() == d_o.numel() && M % S == 0 && dvv.numel() == M * sw80::AF_H, "bad operand sizes");
  c10::cuda::CUDAGuard guard(d_o.device());
  sw80::AttnDeltaParams p{bptr(d_o), bptr(o), dvv.data_ptr<float>(), (int)M, (int)S};
  sw80::attn_delta_kernel<<<(unsigned)((M * sw80::AF_H + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("swa_attn_fwd", &swa_attn_fwd);
  m.def("swa_attn_bwd", &swa_attn_bwd);
  m.def("swa_attn_delta", &swa_attn_delta);
  m.def("swa_qkvg_fwd", &swa_qkvg_fwd);
  m.def("swa_ffn_fwd", &swa_ffn_fwd);
  m.def("swa_mod_fwd", &swa_mod_fwd);
  m.def("swa_mod_bwd", &swa_mod_bwd);
  m.def("swa_oproj_bwd", &swa_oproj_bwd);
  m.def("swa_ffn_bwd", &swa_ffn_bwd);
  m.def("swa_qkvg_bwd", &swa_qkvg_bwd);
}
