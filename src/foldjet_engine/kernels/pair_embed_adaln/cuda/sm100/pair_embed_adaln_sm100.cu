// pair_embed_adaln_sm100.cu -- the B200 (sm_100a) build of the La-Proteina denoiser's pair builder (pair embedding + adaptive LayerNorm).
//
// The A100 source (../sm80/pair_embed_adaln_sm80.cu: sm90/pair_embed_adaln.cu compiled unchanged with the mma.sync forms of its wgmma
// helpers) is included as it is (forward, backward with its impls 0 / 1: legal sm_100a code). Its module definition is renamed into a
// function this file's module calls, which adds `backward_tc`: the same backward with the table gradient dT = onehot^T dx on the B200
// tensor cores (pair_embed_geom/cuda/sm100/pe_tbl_sm100.cuh: tcgen05 / TMEM / TMA; fp32 partials per fixed pair slice, summed in order:
// bitwise deterministic, no atomics). The rows kernel (the pair recomputed, dx in bf16, the parameter partials) is the H100 one.
#include <torch/extension.h>
#pragma push_macro("PYBIND11_MODULE")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, variable) [[maybe_unused]] static void pair_embed_adaln_sm80_bindings(pybind11::module_& variable)
#include "../sm80/pair_embed_adaln_sm80.cu"
#undef PYBIND11_MODULE
#pragma pop_macro("PYBIND11_MODULE")
#include "../../../pair_embed_geom/cuda/sm100/pe_tbl_sm100.cuh"
#include "pe_adaln_sm100.cuh"

namespace {
// the (at most three) table rows of a pair: sequence separation | nS + CA-distance bin | nS + nX + self-conditioning bin; NT <= 224
struct AdalnRows {
  static constexpr int NPADMAX = 224, TCOLS = 256, MAXR = 3, CB = 4;
  using Code = uint32_t;
  static constexpr uint32_t DEAD = MASKED;
  int nS, nX;
  __device__ __forceinline__ int rows(const uint32_t& w, int (&r)[MAXR], uint32_t (&v)[MAXR]) const {
    v[0] = v[1] = v[2] = 0x3F80u;
    if (w & MASKED) return 0;
    r[0] = (int)(w & 255);
    r[1] = nS + (int)((w >> 8) & 255);
    if (!(w & HAS_SC)) return 2;
    r[2] = nS + nX + (int)((w >> 16) & 255);
    return 3;
  }
};
}  // namespace

// as `backward` (the H100 host function), the table gradient on tcgen05: tpart [nchunk][pe100::NBLK][NT][256] fp32
void backward_tc(torch::Tensor tbl, torch::Tensor lnw, torch::Tensor lnb, torch::Tensor gamma, torch::Tensor bins, torch::Tensor dout,
                 torch::Tensor dx, torch::Tensor part, torch::Tensor tpart, torch::Tensor dtbl, torch::Tensor dgamma, torch::Tensor dbeta,
                 torch::Tensor dlw, torch::Tensor dlb, int64_t chunk_b, int64_t nS, int64_t nX, double e1, double e2) {
  const at::cuda::CUDAGuard guard(tbl.device());
  const int B = bins.size(0), n = bins.size(1), NT = tbl.size(0);
  TORCH_CHECK(NT <= AdalnRows::NPADMAX, "pair_embed_adaln sm100: at most ", AdalnRows::NPADMAX, " table rows");
  const int nrb = (n + NW - 1) / NW;
  const int nchunk = (B + chunk_b - 1) / chunk_b;
  TORCH_CHECK(tpart.numel() >= (long long)nchunk * pe100::NBLK * NT * C, "pair_embed_adaln backward: tpart too small");
  TORCH_CHECK((long long)chunk_b * n * n < (1ll << 31), "pair_embed_adaln backward: at most 2^31 pairs per chunk");
  auto st = at::cuda::getCurrentCUDAStream();
  auto* dxp = reinterpret_cast<__nv_bfloat16*>(dx.data_ptr<at::BFloat16>());
  auto* dop = reinterpret_cast<const __nv_bfloat16*>(dout.data_ptr<at::BFloat16>());
  const int ncw = (NT + 15) / 16 * 16;
  for (int ch = 0; ch < nchunk; ch++) {
    const int b0 = ch * chunk_b, Bc = std::min<int>(chunk_b, B - b0);
    pe_bwd_rows<<<Bc * nrb, NW * 32, 0, st>>>(tbl.data_ptr<float>(), lnw.data_ptr<float>(), lnb.data_ptr<float>(), gamma.data_ptr<float>(),
                                              bins.data_ptr<int>(), dop, dxp, part.data_ptr<float>() + (size_t)b0 * nrb * 4 * C, b0, n, nrb,
                                              (int)nS, (int)nX, (float)e1, (float)e2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // a zero bin word past the end of the bins would read as a live pair: the kernel kills every code past the chunk's last pair
    pe100::launch(dxp, (long long)Bc * n * n, bins.data_ptr<int>(), (long long)B * n * n, (long long)b0 * n * n, AdalnRows{(int)nS, (int)nX}, NT,
                  1, ncw, tpart.data_ptr<float>() + (size_t)ch * pe100::NBLK * NT * C, st);
  }
  pe_reduce_rows<<<B + 1, C, 0, st>>>(part.data_ptr<float>(), dgamma.data_ptr<float>(), dbeta.data_ptr<float>(), dlw.data_ptr<float>(),
                                      dlb.data_ptr<float>(), B, nrb);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  pe100::reduce_parts<float><<<(NT * C / 4 + 255) / 256, 256, 0, st>>>(tpart.data_ptr<float>(), dtbl.data_ptr<float>(), nchunk * pe100::NBLK,
                                                                      (long)NT * C);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// ---- B200 forward / backward (pe_adaln_sm100.cuh, 2026-11-17) -----------------------------------------------------------------------------
namespace {
template <typename T> const T* cptr(const torch::Tensor& t) { return static_cast<const T*>(t.data_ptr()); }
void check_gb(const torch::Tensor& gamma, const torch::Tensor& beta) {
  TORCH_CHECK(gamma.scalar_type() == beta.scalar_type() && (gamma.scalar_type() == at::kFloat || gamma.scalar_type() == at::kBFloat16) &&
                  gamma.is_contiguous() && beta.is_contiguous(),
              "pair_embed_adaln sm100: gamma / beta contiguous, both fp32 or both bf16");
}
void check_tbl(const torch::Tensor& tbl, const torch::Tensor& lnw, const torch::Tensor& lnb) {
  TORCH_CHECK(tbl.scalar_type() == at::kFloat && lnw.scalar_type() == at::kFloat && lnb.scalar_type() == at::kFloat && tbl.is_contiguous() &&
                  lnw.is_contiguous() && lnb.is_contiguous() && tbl.dim() == 2 && tbl.size(1) == C,
              "pair_embed_adaln sm100: tbl [NT, 256], ln_weight, ln_bias contiguous fp32");
}
}  // namespace

// out [B, n, n, 256] bf16; bins [B, n, n] int32, or empty (inference: no codes written). g: lanes a pair (8 / 16 / 32); cps: CTAs (of 128
// threads) per SM at most. idx int32 / int64, mask bytes, gamma / beta fp32 / bf16: read as they are.
void forward_b200(torch::Tensor tbl, torch::Tensor lnw, torch::Tensor lnb, torch::Tensor gamma, torch::Tensor beta, torch::Tensor xt, torch::Tensor sc,
                  int64_t has_sc, torch::Tensor idx, torch::Tensor mask, torch::Tensor xl, torch::Tensor cl, torch::Tensor out, torch::Tensor bins,
                  int64_t nS, int64_t nX, double e1, double e2, int64_t g, int64_t cps, int64_t ch) {
  const at::cuda::CUDAGuard guard(tbl.device());
  check_tbl(tbl, lnw, lnb);
  check_gb(gamma, beta);
  TORCH_CHECK(xt.scalar_type() == at::kFloat && sc.scalar_type() == at::kFloat && xt.is_contiguous() && sc.is_contiguous() && idx.is_contiguous() &&
                  mask.is_contiguous() && mask.element_size() == 1 && xl.numel() <= 255 && cl.numel() <= 255,
              "pair_embed_adaln sm100: xt / sc contiguous fp32, idx contiguous, mask contiguous bytes, at most 255 bin limits");
  const long long B = idx.size(0), n = idx.size(1), PL = B * n * n;
  TORCH_CHECK(PL < (1ll << 31), "pair_embed_adaln sm100: at most 2^31 pairs");
  if (PL == 0) return;
  const bool train = bins.numel() > 0;
  TORCH_CHECK(!train || bins.numel() == PL, "pair_embed_adaln sm100: bins [B, n, n]");
  const unsigned P = (unsigned)PL;
  auto st = at::cuda::getCurrentCUDAStream();
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  TORCH_CHECK(ch == 8 || ch == 16 || ch == 32, "pair_embed_adaln sm100: chunks of 8, 16 or 32 pairs");
  const long long chunks = (PL + ch - 1) / ch;
  const long long ctas = std::max<long long>(1, std::min<long long>((long long)sms * std::max<int64_t>(1, cps), (chunks + pe100ad::WARPS - 1) / pe100ad::WARPS));
  const dim3 grid((unsigned)ctas), block(pe100ad::WARPS * 32);
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  int* bp = train ? bins.data_ptr<int>() : nullptr;
  auto go = [&](auto ip, auto gp, auto btp) {
    using I = std::remove_cv_t<std::remove_pointer_t<decltype(ip)>>;
    using GT = std::remove_cv_t<std::remove_pointer_t<decltype(gp)>>;
#define PE_FWD(G, TR)                                                                                                                           \
  pe100ad::fwd<I, GT, G, TR><<<grid, block, 0, st>>>(tbl.data_ptr<float>(), lnw.data_ptr<float>(), lnb.data_ptr<float>(), gp, btp,              \
                                                     xt.data_ptr<float>(), sc.data_ptr<float>(), (int)has_sc, ip, cptr<uint8_t>(mask),          \
                                                     xl.data_ptr<float>(), (int)xl.numel(), cl.data_ptr<float>(), (int)cl.numel(), op, bp,      \
                                                     (unsigned)n, P, (int)nS, (int)nX, (float)e1, (float)e2, (int)ch)
    if (g == 8) { if (train) PE_FWD(8, true); else PE_FWD(8, false); }
    else if (g == 16) { if (train) PE_FWD(16, true); else PE_FWD(16, false); }
    else { if (train) PE_FWD(32, true); else PE_FWD(32, false); }
#undef PE_FWD
  };
  auto gi = [&](auto ip) {
    if (gamma.scalar_type() == at::kFloat) go(ip, cptr<float>(gamma), cptr<float>(beta));
    else go(ip, cptr<__nv_bfloat16>(gamma), cptr<__nv_bfloat16>(beta));
  };
  if (idx.scalar_type() == at::kLong) gi(cptr<long long>(idx));
  else {
    TORCH_CHECK(idx.scalar_type() == at::kInt, "pair_embed_adaln sm100: idx int32 or int64");
    gi(cptr<int>(idx));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dw [256, NT] fp32 (the weight's layout), dlw / dlb [256] fp32, dgamma / dbeta [B, 256] in gamma's dtype, from d pair = dout [B, n, n, 256] bf16.
// part [B cps][4][256], tpart [nchunk][parts][NT][256] fp32 scratch (parts: pe100::NBLK with tc, else NBLK), dx [chunk_b, n, n, 256] bf16.
// tc: the table gradient on tcgen05 (pe_tbl_sm100.cuh), else the warpgroup kernel as mma.sync (pe_bwd_tbl4, NT <= 192).
void backward_b200(torch::Tensor tbl, torch::Tensor lnw, torch::Tensor lnb, torch::Tensor gamma, torch::Tensor bins, torch::Tensor dout,
                   torch::Tensor dx, torch::Tensor part, torch::Tensor tpart, torch::Tensor dw, torch::Tensor dgamma, torch::Tensor dbeta,
                   torch::Tensor dlw, torch::Tensor dlb, int64_t chunk_b, int64_t nS, int64_t nX, double e1, double e2, int64_t g, int64_t cps, int64_t ch,
                   int64_t tc) {
  const at::cuda::CUDAGuard guard(tbl.device());
  check_tbl(tbl, lnw, lnb);
  check_gb(gamma, dgamma);
  check_gb(gamma, dbeta);
  const int B = bins.size(0), n = bins.size(1), NT = tbl.size(0);
  const long long nn = (long long)n * n;
  TORCH_CHECK(NT <= AdalnRows::NPADMAX && (tc || NT <= NR4), "pair_embed_adaln sm100: too many table rows");
  TORCH_CHECK(ch == 8 || ch == 16 || ch == 32, "pair_embed_adaln sm100: chunks of 8, 16 or 32 pairs");
  TORCH_CHECK(nn < (1ll << 31) && (long long)chunk_b * nn < (1ll << 31) && cps >= 1, "pair_embed_adaln backward: at most 2^31 pairs per chunk");
  const int nchunk = (B + chunk_b - 1) / chunk_b, parts = tc ? pe100::NBLK : NBLK;
  TORCH_CHECK(tpart.numel() >= (long long)nchunk * parts * NT * C && part.numel() >= (long long)B * cps * 4 * C && dw.numel() == (long long)NT * C &&
                  dx.numel() >= std::min<long long>(chunk_b, B) * nn * C,
              "pair_embed_adaln backward: scratch too small");
  auto st = at::cuda::getCurrentCUDAStream();
  auto* dxp = reinterpret_cast<__nv_bfloat16*>(dx.data_ptr<at::BFloat16>());
  auto* dop = reinterpret_cast<const __nv_bfloat16*>(dout.data_ptr<at::BFloat16>());
  const int ncw = (NT + 15) / 16 * 16;
  const bool gf = gamma.scalar_type() == at::kFloat;
  const dim3 block(pe100ad::WARPS * 32);
  for (int ci = 0; ci < nchunk; ci++) {
    const int b0 = ci * chunk_b, Bc = std::min<int>(chunk_b, B - b0);
    const dim3 grid((unsigned)(Bc * cps));
    const int* cp = bins.data_ptr<int>() + (size_t)b0 * nn;
    const __nv_bfloat16* gp = dop + (size_t)b0 * nn * C;
    float* pp = part.data_ptr<float>() + (size_t)b0 * cps * 4 * C;
#define PE_BWD(GT, G)                                                                                                                           \
  pe100ad::bwd_rows<GT, G><<<grid, block, 0, st>>>(tbl.data_ptr<float>(), lnw.data_ptr<float>(), lnb.data_ptr<float>(),                         \
                                                   cptr<GT>(gamma) + (size_t)b0 * C, cp, gp, dxp, pp, (unsigned)nn, (int)cps, (int)nS, (int)nX, \
                                                   (float)e1, (float)e2, (int)ch)
    if (g == 16) { if (gf) PE_BWD(float, 16); else PE_BWD(__nv_bfloat16, 16); }
    else { if (gf) PE_BWD(float, 32); else PE_BWD(__nv_bfloat16, 32); }
#undef PE_BWD
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (tc) {
      pe100::launch(dxp, (long long)Bc * nn, bins.data_ptr<int>(), (long long)B * nn, (long long)b0 * nn, AdalnRows{(int)nS, (int)nX}, NT, 1, ncw,
                    tpart.data_ptr<float>() + (size_t)ci * pe100::NBLK * NT * C, st);
    } else {
      const size_t sm4 = (size_t)ST4 * TPB * 128 + (size_t)2 * NR4 * 128 + 1024;
      cudaFuncSetAttribute(pe_bwd_tbl4, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm4);
      pe_bwd_tbl4<<<dim3(4, NBLK), NTH4, sm4, st>>>(dxp, cp, (long long)Bc * nn, (int)nS, (int)nX, NT, tpart.data_ptr<float>() + (size_t)ci * NBLK * NT * C);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
  if (gf) pe100ad::red_rows<float><<<B + 1, 1024, 0, st>>>(part.data_ptr<float>(), dgamma.data_ptr<float>(), dbeta.data_ptr<float>(), dlw.data_ptr<float>(), dlb.data_ptr<float>(), B, (int)cps);
  else pe100ad::red_rows<__nv_bfloat16><<<B + 1, 1024, 0, st>>>(part.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(dgamma.data_ptr<at::BFloat16>()), reinterpret_cast<__nv_bfloat16*>(dbeta.data_ptr<at::BFloat16>()), dlw.data_ptr<float>(), dlb.data_ptr<float>(), B, (int)cps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  pe100::reduce_parts_t<false><<<(NT * C / 4 + 255) / 256, 256, 0, st>>>(tpart.data_ptr<float>(), dw.data_ptr<float>(), nchunk * parts, NT);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The inference tables of a weight [256, NT] fp32 and the LayerNorm affine [256] fp32 (pe_adaln_sm100.cuh, "statistics from tables"):
// U [NT, 256], GG [NT, NT, 2], V [NT], BT [256], BB [1] fp32; tc [NT, 256] is scratch. Two launches.
void stats_b200(torch::Tensor w, torch::Tensor lnw, torch::Tensor lnb, torch::Tensor u, torch::Tensor tc, torch::Tensor gg, torch::Tensor v,
                torch::Tensor bt, torch::Tensor bb) {
  const at::cuda::CUDAGuard guard(w.device());
  TORCH_CHECK(w.dim() == 2 && w.size(0) == C && w.scalar_type() == at::kFloat && w.is_contiguous() && lnw.scalar_type() == at::kFloat &&
                  lnb.scalar_type() == at::kFloat && lnw.is_contiguous() && lnb.is_contiguous() && lnw.numel() == C && lnb.numel() == C,
              "pair_embed_adaln stats: weight [256, NT], ln_weight, ln_bias [256] contiguous fp32");
  const int NT = w.size(1);
  TORCH_CHECK(NT <= 256 && u.numel() == (long long)NT * C && tc.numel() == (long long)NT * C && gg.numel() == (long long)NT * NT * 2 &&
                  v.numel() == NT && bt.numel() == C && bb.numel() == 1,
              "pair_embed_adaln stats: table sizes");
  auto st = at::cuda::getCurrentCUDAStream();
  pe100ad::stats_rows<<<NT + 1, C, 0, st>>>(w.data_ptr<float>(), lnw.data_ptr<float>(), lnb.data_ptr<float>(), u.data_ptr<float>(), tc.data_ptr<float>(),
                                           v.data_ptr<float>(), bt.data_ptr<float>(), bb.data_ptr<float>(), NT);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  pe100ad::stats_gram<<<NT, C, 0, st>>>(tc.data_ptr<float>(), u.data_ptr<float>(), reinterpret_cast<float2*>(gg.data_ptr<float>()), NT);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Inference: out [B, n, n, 256] bf16 from the tables of stats_b200; gamma / beta fp32 / bf16, idx int32 / int64, mask bytes: read as they are.
void forward_s_b200(torch::Tensor u, torch::Tensor gg, torch::Tensor v, torch::Tensor bt, torch::Tensor bb, torch::Tensor gamma, torch::Tensor beta,
                    torch::Tensor xt, torch::Tensor sc, int64_t has_sc, torch::Tensor idx, torch::Tensor mask, torch::Tensor xl, torch::Tensor cl,
                    torch::Tensor out, int64_t nS, int64_t nX, double e1, double e2, int64_t g, int64_t cps, int64_t ch) {
  const at::cuda::CUDAGuard guard(u.device());
  check_gb(gamma, beta);
  TORCH_CHECK(xt.scalar_type() == at::kFloat && sc.scalar_type() == at::kFloat && xt.is_contiguous() && sc.is_contiguous() && idx.is_contiguous() &&
                  mask.is_contiguous() && mask.element_size() == 1 && xl.numel() <= 255 && cl.numel() <= 255 && u.scalar_type() == at::kFloat &&
                  u.dim() == 2 && u.size(1) == C && gg.numel() == u.size(0) * u.size(0) * 2 && v.numel() == u.size(0) && bt.numel() == C,
              "pair_embed_adaln sm100: xt / sc contiguous fp32, idx contiguous, mask contiguous bytes, at most 255 bin limits, the stats tables");
  TORCH_CHECK(ch == 8 || ch == 16 || ch == 32, "pair_embed_adaln sm100: chunks of 8, 16 or 32 pairs");
  const long long B = idx.size(0), n = idx.size(1), PL = B * n * n;
  TORCH_CHECK(PL < (1ll << 31), "pair_embed_adaln sm100: at most 2^31 pairs");
  if (PL == 0) return;
  const int NT = u.size(0);
  auto st = at::cuda::getCurrentCUDAStream();
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const long long chunks = (PL + ch - 1) / ch;
  const long long ctas = std::max<long long>(1, std::min<long long>((long long)sms * std::max<int64_t>(1, cps), (chunks + pe100ad::WARPS - 1) / pe100ad::WARPS));
  const dim3 grid((unsigned)ctas), block(pe100ad::WARPS * 32);
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  auto go = [&](auto ip, auto gp, auto btp) {
    using I = std::remove_cv_t<std::remove_pointer_t<decltype(ip)>>;
    using GT = std::remove_cv_t<std::remove_pointer_t<decltype(gp)>>;
#define PE_FWS(G)                                                                                                                               \
  pe100ad::fwd_s<I, GT, G><<<grid, block, 0, st>>>(u.data_ptr<float>(), reinterpret_cast<const float2*>(gg.data_ptr<float>()), v.data_ptr<float>(), \
                                                   bt.data_ptr<float>(), bb.data_ptr<float>(), gp, btp, xt.data_ptr<float>(), sc.data_ptr<float>(), \
                                                   (int)has_sc, ip, cptr<uint8_t>(mask), xl.data_ptr<float>(), (int)xl.numel(), cl.data_ptr<float>(), \
                                                   (int)cl.numel(), op, (unsigned)n, (unsigned)PL, (int)nS, (int)nX, NT, (float)e1, (float)e2, (int)ch)
    if (g == 8) PE_FWS(8);
    else PE_FWS(16);
#undef PE_FWS
  };
  auto gi = [&](auto ip) {
    if (gamma.scalar_type() == at::kFloat) go(ip, cptr<float>(gamma), cptr<float>(beta));
    else go(ip, cptr<__nv_bfloat16>(gamma), cptr<__nv_bfloat16>(beta));
  };
  if (idx.scalar_type() == at::kLong) gi(cptr<long long>(idx));
  else {
    TORCH_CHECK(idx.scalar_type() == at::kInt, "pair_embed_adaln sm100: idx int32 or int64");
    gi(cptr<int>(idx));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pair_embed_adaln_sm80_bindings(m);
  m.def("backward_tc", &backward_tc);
  m.def("forward_b200", &forward_b200);
  m.def("backward_b200", &backward_b200);
  m.def("stats_b200", &stats_b200);
  m.def("forward_s_b200", &forward_s_b200);
}
