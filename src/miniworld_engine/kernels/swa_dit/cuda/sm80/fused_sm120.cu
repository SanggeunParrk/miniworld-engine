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
#include "attn_global_sm120.cuh"
#include "ffn_bwd_sm80.cuh"
#include "bwd_ks_sm120.cuh"
#include "../../../_sm_count.cuh"

#define CHECK_CUDA_BF16(x) TORCH_CHECK((x).is_cuda() && (x).scalar_type() == torch::kBFloat16, #x " must be a CUDA bf16 tensor")
#define CHECK_CUDA_F32(x) TORCH_CHECK((x).is_cuda() && (x).scalar_type() == torch::kFloat32, #x " must be a CUDA fp32 tensor")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

namespace {
using bf = __nv_bfloat16;
inline const bf* bptr(const torch::Tensor& t) { return reinterpret_cast<const bf*>(t.data_ptr()); }
inline bf* bptr_w(torch::Tensor& t) { return reinterpret_cast<bf*>(t.data_ptr()); }
inline int num_sms() { return mw::sm_count(); }

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

template <int NW>
void launch_ks(const sw80::FfnBwdDyParams* pd, const sw80::QkvgBwdParams* pq, int ngrp) {
  using G = sw80::KsCfg<NW>;
  static bool attr[2][64] = {};
  const int dev = at::cuda::current_device() & 63, w = pd ? 0 : 1;
  if (pd) {
    if (!attr[w][dev]) { C10_CUDA_CHECK(cudaFuncSetAttribute(sw80::ffn_bwd_dy_ks_kernel<G, sw80::MODE_SINGLE>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM)); attr[w][dev] = true; }
    sw80::ffn_bwd_dy_ks_kernel<G, sw80::MODE_SINGLE><<<ngrp, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(*pd);
  } else {
    if (!attr[w][dev]) { C10_CUDA_CHECK(cudaFuncSetAttribute(sw80::qkvg_bwd_ks_kernel<G, sw80::MODE_SINGLE>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM)); attr[w][dev] = true; }
    sw80::qkvg_bwd_ks_kernel<G, sw80::MODE_SINGLE><<<ngrp, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(*pq);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// the K-split kernels (bwd_ks_sm120.cuh), NW tiles per CTA = the fewest that give one group per SM (at most 4)
void ks_dispatch(const sw80::FfnBwdDyParams* pd, const sw80::QkvgBwdParams* pq) {
  const int M = pd ? pd->M : pq->M, S = pd ? pd->S : pq->S;
  const int ntile = ((S + 15) / 16) * (M / S), sms = num_sms();
  for (int nw = 1; nw <= 4; ++nw) {
    if ((ntile + nw - 1) / nw > sms && nw < 4) continue;
    TORCH_CHECK((ntile + nw - 1) / nw <= sms, "K-split backward: too many row tiles for one group per SM");
    const int ngrp = (ntile + nw - 1) / nw;
    if (nw == 1) launch_ks<1>(pd, pq, ngrp); else if (nw == 2) launch_ks<2>(pd, pq, ngrp); else if (nw == 3) launch_ks<3>(pd, pq, ngrp); else launch_ks<4>(pd, pq, ngrp);
    return;
  }
}

}  // namespace

// qkvg stage (inference): x [M, C] bf16, mod [B S, 6C] fp32, cos / sin [B S, 16] fp32, Wqkv [3C, C], Wg [C, C] -> Qh, Kh, Vh [N, H, S, 32] (tm: token-major [M, H, 32]), G [M, C]
void swa_qkvg_fwd_half(torch::Tensor x, torch::Tensor mod, torch::Tensor cos, torch::Tensor sin, torch::Tensor wqkv, torch::Tensor wg, torch::Tensor qh, torch::Tensor kh,
                       torch::Tensor vh, torch::Tensor g, int64_t S, int64_t B, double eps, double qk_eps, int64_t cfg, bool tm, int64_t nsplit, int64_t max_pairs,
                       c10::optional<torch::Tensor> xs, c10::optional<torch::Tensor> pqs, c10::optional<torch::Tensor> pks) {
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
  if (xs.has_value()) {   // training: the stage keeps x, p_q, p_k (bf16 [M, C]) for the backward
    for (const torch::Tensor* t : {&*xs, &*pqs, &*pks}) { CHECK_CUDA_BF16(*t); CHECK_CONTIG(*t); TORCH_CHECK(t->numel() == M * 128, "bad save sizes"); }
    p.xs = bptr_w(*xs); p.pqs = bptr_w(*pqs); p.pks = bptr_w(*pks);
  }
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
                     int64_t B, double eps, int64_t cfg, c10::optional<torch::Tensor> q1s, c10::optional<torch::Tensor> atts, c10::optional<torch::Tensor> ys,
                     c10::optional<torch::Tensor> ffs) {
  CHECK_CUDA_BF16(qi); CHECK_CUDA_BF16(g); CHECK_CUDA_BF16(o); CHECK_CUDA_BF16(wo); CHECK_CUDA_BF16(wu); CHECK_CUDA_BF16(wd); CHECK_CUDA_BF16(out); CHECK_CUDA_F32(mod);
  for (const auto* t : {&qi, &g, &o, &mod, &wo, &wu, &wd, &out}) CHECK_CONTIG(*t);
  const int64_t M = qi.size(0);
  TORCH_CHECK(qi.size(1) == 128 && M % S == 0 && mod.numel() == B * S * 768 && g.numel() == M * 128 && o.numel() == M * 128 && out.numel() == M * 128, "bad operand sizes");
  TORCH_CHECK(wo.size(0) == 128 && wo.size(1) == 128 && wu.size(0) == 512 && wu.size(1) == 128 && wd.size(0) == 128 && wd.size(1) == 256, "bad weight sizes");
  if (M == 0) return;
  c10::cuda::CUDAGuard guard(qi.device());
  sw80::FfnFwdParams p{bptr(qi), bptr(g), bptr(o), mod.data_ptr<float>(), bptr(wo), bptr(wu), bptr(wd), bptr_w(out), nullptr, nullptr, nullptr, nullptr,
                       (int)M, (int)S, (int)B, (float)eps, nullptr};
  if (q1s.has_value()) {   // training: q1, att, y, ffn (bf16 [M, C]) for the backward
    for (const torch::Tensor* t : {&*q1s, &*atts, &*ys, &*ffs}) { CHECK_CUDA_BF16(*t); CHECK_CONTIG(*t); TORCH_CHECK(t->numel() == M * 128, "bad save sizes"); }
    p.q1s = bptr_w(*q1s); p.atts = bptr_w(*atts); p.ys = bptr_w(*ys); p.ffs = bptr_w(*ffs);
  }
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

template <class G>
void launch_ffn_bwd_gate_sm120(const sw80::FfnBwdGateParams& p) {
  static bool attr[64] = {};
  const int dev = at::cuda::current_device() & 63;
  if (!attr[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(sw80::ffn_bwd_gate_kernel<G, sw80::MODE_SINGLE>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    attr[dev] = true;
  }
  const int ntile = ((p.S + 15) / 16) * (p.M / p.S), ngrp = (ntile + G::NW - 1) / G::NW;
  const unsigned ctas = (unsigned)std::max(1, std::min(num_sms() * G::MINB, ngrp * p.hsplit));
  sw80::ffn_bwd_gate_kernel<G, sw80::MODE_SINGLE><<<ctas, G::NTHR, G::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// the FFN backward's gate kernel alone (ffn_bwd_sm80.cuh; its 72 KB weight ring fits the 99 KB; the dy kernel keeps 128 KB of Wu^T and does not): dy [M, 128] bf16 (dq2), y, ffn, mod [M, 768] fp32
// (one modulation row per sequence row: MODE_SINGLE), Wu [512, 128], Wd^T [256, 128] -> dffn [M, 128], hh [M, 256], dab [M, 512] bf16, d gate_f in dmod columns 640 .. 767
void swa_ffn_bwd_gate(torch::Tensor dy, torch::Tensor y, torch::Tensor ffn, torch::Tensor mod, torch::Tensor wu, torch::Tensor wdt, torch::Tensor dffn, torch::Tensor hh,
                      torch::Tensor dab, torch::Tensor dmod, int64_t S, int64_t B, int64_t cfg, int64_t hsplit) {
  TORCH_CHECK(hsplit == 1 || hsplit == 2 || hsplit == 4 || hsplit == 8, "hsplit must be 1, 2, 4 or 8");
  for (const auto* t : {&dy, &y, &ffn, &wu, &wdt, &dffn, &hh, &dab}) { CHECK_CUDA_BF16(*t); CHECK_CONTIG(*t); }
  CHECK_CUDA_F32(mod); CHECK_CUDA_F32(dmod); CHECK_CONTIG(mod); CHECK_CONTIG(dmod);
  const int64_t M = dy.size(0);
  TORCH_CHECK(dy.size(1) == 128 && M % S == 0 && M / S == B && mod.numel() == B * S * 768 && dmod.numel() == B * S * 768 && wu.size(0) == 512 && wu.size(1) == 128 && wdt.size(0) == 256 &&
              wdt.size(1) == 128 && hh.numel() == M * 256 && dab.numel() == M * 512 && dffn.numel() == M * 128, "bad operand sizes");
  if (M == 0) return;
  c10::cuda::CUDAGuard guard(dy.device());
  sw80::FfnBwdGateParams p{bptr(dy), bptr(y), bptr(ffn), mod.data_ptr<float>(), bptr(wu), bptr(wdt), bptr_w(dffn), bptr_w(hh), bptr_w(dab), dmod.data_ptr<float>(), (int)M, (int)S, (int)B, (int)hsplit};
  switch (cfg) {
    case 0: launch_ffn_bwd_gate_sm120<sw80::FfnBwdGateCfg<8, 1>>(p); break;
    case 1: launch_ffn_bwd_gate_sm120<sw80::FfnBwdGateCfg<4, 2>>(p); break;
    case 2: launch_ffn_bwd_gate_sm120<sw80::FfnBwdGateCfg<6, 2>>(p); break;
    default: TORCH_CHECK(false, "unknown FFN backward (gate) configuration");
  }
}

// global-attention forward (attn_global_sm120.cuh): q, k, v head-major [N, 4, S, 32] bf16, seqused [N] int32 -> o [N S, C] bf16 (no lse)
void swa_attn_global_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor seqused, torch::Tensor o) {
  CHECK_CUDA_BF16(q); CHECK_CUDA_BF16(k); CHECK_CUDA_BF16(v); CHECK_CUDA_BF16(o);
  for (const auto* t : {&q, &k, &v, &o}) CHECK_CONTIG(*t);
  TORCH_CHECK(q.dim() == 4 && q.size(1) == 4 && q.size(3) == 32 && k.sizes() == q.sizes() && v.sizes() == q.sizes(), "q, k, v must be [N, 4, S, 32]");
  TORCH_CHECK(seqused.scalar_type() == torch::kInt32 && seqused.is_cuda() && seqused.numel() == q.size(0), "seqused must be int32 [N]");
  const int64_t N = q.size(0), S = q.size(2);
  TORCH_CHECK(o.numel() == N * S * 128, "bad output size");
  if (N * S == 0) return;
  c10::cuda::CUDAGuard guard(q.device());
  sw80::AttnGParams p{bptr(q), bptr(k), bptr(v), seqused.data_ptr<int>(), bptr_w(o), (int)S, 0.17677669529663687f * 1.4426950408889634f};
  const dim3 grid((unsigned)((S + sw80::AG_BQ - 1) / sw80::AG_BQ), (unsigned)(N * 4));
  sw80::attn_global_fwd_kernel<<<grid, sw80::AG_NW * 32, sw80::AG_SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// the FFN backward's dy kernel streaming its weight (bwd_ks_sm120.cuh): dab [M, 512] da | db, q1, dy (= dq2) [M, 128] bf16, mod [M, 768] fp32, Wu^T [128, 512] -> dq1 [M, 128] bf16;
// d shift_f | d scale_f go into dmod columns 384 .. 639
void swa_ffn_bwd_dy_stream(torch::Tensor dab, torch::Tensor q1, torch::Tensor dy, torch::Tensor mod, torch::Tensor wut, torch::Tensor dq1, torch::Tensor dmod, int64_t S, int64_t B, double eps) {
  for (const auto* t : {&dab, &q1, &dy, &wut, &dq1}) { CHECK_CUDA_BF16(*t); CHECK_CONTIG(*t); }
  CHECK_CUDA_F32(mod); CHECK_CUDA_F32(dmod); CHECK_CONTIG(mod); CHECK_CONTIG(dmod);
  const int64_t M = dy.size(0);
  TORCH_CHECK(dy.size(1) == 128 && M % S == 0 && M / S == B && mod.numel() == B * S * 768 && dmod.numel() == B * S * 768 && wut.size(0) == 128 && wut.size(1) == 512 && dab.numel() == M * 512 &&
              q1.numel() == M * 128 && dq1.numel() == M * 128, "bad operand sizes");
  if (M == 0) return;
  c10::cuda::CUDAGuard guard(dy.device());
  sw80::FfnBwdDyParams p{bptr(dab), bptr(q1), bptr(dy), mod.data_ptr<float>(), bptr(wut), bptr_w(dq1), dmod.data_ptr<float>(), (int)M, (int)S, (int)B, (float)eps};
  ks_dispatch(&p, nullptr);
}

// the qkvg backward streaming its weight: operands of swa_qkvg_bwd (ops.cu), MODE_SINGLE
void swa_qkvg_bwd_stream(torch::Tensor qi, torch::Tensor pq, torch::Tensor pk, torch::Tensor dqh, torch::Tensor dkh, torch::Tensor dvh, torch::Tensor dg, torch::Tensor dq1, torch::Tensor mod,
                         torch::Tensor cos, torch::Tensor sin, torch::Tensor wt, torch::Tensor dq, torch::Tensor dp, torch::Tensor dmod, int64_t S, int64_t B, double eps, double qk_eps) {
  for (const auto* t : {&qi, &pq, &pk, &dqh, &dkh, &dvh, &dg, &dq1, &wt, &dq, &dp}) { CHECK_CUDA_BF16(*t); CHECK_CONTIG(*t); }
  for (const auto* t : {&mod, &cos, &sin, &dmod}) { CHECK_CUDA_F32(*t); CHECK_CONTIG(*t); }
  const int64_t M = qi.size(0);
  TORCH_CHECK(qi.size(1) == 128 && M % S == 0 && M / S == B && mod.numel() == B * S * 768 && dmod.numel() == B * S * 768 && cos.numel() == B * S * 16 && sin.numel() == B * S * 16 &&
              wt.size(0) == 128 && wt.size(1) == 512 && dp.numel() == M * 512 && dqh.numel() == M * 128 && dkh.numel() == M * 128 && dvh.numel() == M * 128, "bad operand sizes");
  if (M == 0) return;
  c10::cuda::CUDAGuard guard(qi.device());
  sw80::QkvgBwdParams p{bptr(qi), bptr(pq), bptr(pk), bptr(dqh), bptr(dkh), bptr(dvh), bptr(dg), bptr(dq1), mod.data_ptr<float>(), cos.data_ptr<float>(), sin.data_ptr<float>(), bptr(wt),
                        bptr_w(dq), bptr_w(dp), dmod.data_ptr<float>(), (int)M, (int)S, (int)B, (float)eps, (float)qk_eps};
  ks_dispatch(nullptr, &p);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("swa_attn_global_fwd", &swa_attn_global_fwd);
  m.def("swa_ffn_bwd_gate", &swa_ffn_bwd_gate);
  m.def("swa_ffn_bwd_dy_stream", &swa_ffn_bwd_dy_stream);
  m.def("swa_qkvg_bwd_stream", &swa_qkvg_bwd_stream);
  m.def("swa_qkvg_f32", &swa_qkvg_f32);
  m.def("swa_ffn_f32", &swa_ffn_f32);
  m.def("swa_qkvg_fwd_half", &swa_qkvg_fwd_half);
  m.def("swa_ffn_fwd_u32", &swa_ffn_fwd_u32);
}
