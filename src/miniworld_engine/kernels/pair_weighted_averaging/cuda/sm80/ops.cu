// ops.cu -- torch bindings of the A100 (sm_80) MSA pair-weighted-averaging kernels.
//
//   pair_fwd   pwa_pair_fwd_sm80.cuh   LN(pair) . Wb, key mask, softmax over the keys -> w [8, L, L]
//   ln_v       pwa_ln_v_sm80.cuh       LN(msa) . Wv -> v [8, L, S C] (head-major), LayerNorm statistics
//   gate_out   pwa_gate_out_sm80.cuh   sigmoid gate, output projection, dropout, residual -> out [S, L, D]
// Every function takes its output tensors (allocated by the caller) and launches on the current stream.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <set>
#include <type_traits>
#include <utility>

#include "pwa_dgv_bwd_sm80.cuh"
#include "pwa_gate_out_sm80.cuh"
#include "pwa_glue_sm80.cuh"
#include "pwa_ln_v_sm80.cuh"
#include "pwa_pair_bwd_sm80.cuh"
#include "pwa_pair_fwd_sm80.cuh"

namespace {

void check_bf16(const torch::Tensor& t, const char* name) { TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.is_contiguous(), name, ": a contiguous bf16 CUDA tensor"); }
void check_f32(const torch::Tensor& t, const char* name) { TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kFloat && t.is_contiguous(), name, ": a contiguous fp32 CUDA tensor"); }
void check_mask(const torch::Tensor& t, int64_t n, const char* name) {
  TORCH_CHECK(t.numel() == 0 || (t.is_cuda() && t.scalar_type() == at::kByte && t.is_contiguous() && t.numel() == n), name, ": uint8 [", n, "] (or empty)");
}

// the dynamic shared-memory limit of a kernel, raised once per (kernel, device) -- keyed by the address: kernels of one signature share a pointer type
template <class K> void raise_smem(K kernel, int bytes) {
  static std::set<std::pair<const void*, int>> done;
  const std::pair<const void*, int> key{reinterpret_cast<const void*>(kernel), (int)at::cuda::current_device()};
  if (done.insert(key).second) C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

template <typename T> const T* ptr(const torch::Tensor& t) { return t.numel() ? reinterpret_cast<const T*>(t.data_ptr()) : nullptr; }
// the head-major layout of v / o / dO / dgp / dv: [8 ns, L, kp C] with S = ns kp (the MSA rows in ns chunks, see hm_row in sm80_common.cuh)
struct HeadMajor { int ns, kp; };
HeadMajor head_major(const torch::Tensor& t, int S, int L, int C, const char* name) {
  TORCH_CHECK(t.dim() == 3 && t.size(0) % 8 == 0 && t.size(1) == L && t.size(2) % C == 0, name, ": [8 ns, L, kp C]");
  const int ns = (int)(t.size(0) / 8), kp = (int)(t.size(2) / C);
  TORCH_CHECK((int64_t)ns * kp == S && (ns == 1 || kp % 128 == 0), name, ": S = ns kp, with kp a multiple of 128 when ns > 1");
  return {ns, kp};
}


int num_sms() { return at::cuda::getCurrentDeviceProperties()->multiProcessorCount; }

template <int DZ> void launch_pair_fwd(const pwa80::PairFwdParams& p, cudaStream_t st) {
  using C = pwa80::PairFwdCfg<DZ>;
  const int smem = C::FIXED + 8 * (p.L + 4) * 4;
  TORCH_CHECK(smem <= 166912, "pair_fwd: L too long for the shared-memory logits");
  raise_smem(pwa80::pwa_pair_fwd_kernel<DZ>, 166912);                 // the attribute is set once: the largest dynamic size any L may ask for
  pwa80::pwa_pair_fwd_kernel<DZ><<<p.L, C::NTHR, smem, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int D, int C, bool ST> void launch_ln_v(const pwa80::LnVParams& p, cudaStream_t st) {
  using G = pwa80::LnVCfg<D, C>;
  raise_smem(pwa80::pwa_ln_v_kernel<D, C, ST>, G::SMEM);
  const int grid = std::min(p.ntile, num_sms() * (G::SMEM <= 80 * 1024 ? 2 : 1));
  pwa80::pwa_ln_v_kernel<D, C, ST><<<grid, G::NTHR, G::SMEM, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int D, int C> void launch_gate_out(const pwa80::GateOutParams& p, cudaStream_t st) {
  using G = pwa80::GoCfg<D, C>;
  raise_smem(pwa80::pwa_gate_out_kernel<D, C>, G::SMEM);
  const int grid = std::min(p.ntile, num_sms());
  pwa80::pwa_gate_out_kernel<D, C><<<grid, G::NTHR, G::SMEM, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// persistent grids: the glue's CTAs per channel group, the dgv kernel's CTAs (also what the partial buffers are sized by)
template <int D, int C> int glue_ctas(int ntile) {
  using G = pwa80::GlueCfg<D, C>;
  return std::max(1, std::min(ntile, num_sms() * 2 / G::NG));
}
template <int D, int C> int dgv_ctas(int ntile) { return std::min(ntile, num_sms()); }

template <int D, int C> void launch_glue(const pwa80::GlueParams& p, int nbx, cudaStream_t st) {
  using G = pwa80::GlueCfg<D, C>;
  raise_smem(pwa80::pwa_glue_kernel<D, C>, G::SMEM);
  pwa80::pwa_glue_kernel<D, C><<<dim3(nbx, G::NG), G::NTHR, G::SMEM, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int D, int C> void launch_dgv(const pwa80::DgvParams& p, int nb, cudaStream_t st) {
  using G = pwa80::DgvCfg<D, C>;
  raise_smem(pwa80::pwa_dgv_bwd_kernel<D, C>, G::SMEM);
  pwa80::pwa_dgv_bwd_kernel<D, C><<<nb, G::NTHR, G::SMEM, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int DZ, int NMINB> void launch_pair_bwd(const pwa80::PairBwdParams& p, cudaStream_t st) {
  using C = pwa80::PairBwdCfg<DZ>;
  const int smem = C::TILES + C::PREP + 16 * p.L;
  TORCH_CHECK(smem <= 166912, "pair_bwd: L too long for shared memory");
  raise_smem(pwa80::pwa_pair_bwd_kernel<DZ, NMINB>, 166912);
  pwa80::pwa_pair_bwd_kernel<DZ, NMINB><<<p.L, C::NTHR, smem, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// block = 32 column quads x 8 row groups: thread (x, y) sums the rows y, y + 8, ... of its quad (loads in flight: R / 8 per thread), then the eight partials are summed in order
// through shared memory: a fixed association, so the result does not depend on scheduling
template <typename TOut> __global__ void __launch_bounds__(256) reduce_rows_kernel(const float* __restrict__ in, TOut* __restrict__ out, int R, long W) {
  __shared__ float4 red[8][32];
  const long c = ((long)blockIdx.x * 32 + threadIdx.x) * 4;
  float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
  if (c < W) {
#pragma unroll 4
    for (int r = threadIdx.y; r < R; r += 8) {
      const float4 v = *reinterpret_cast<const float4*>(in + (long)r * W + c);
      s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
  }
  red[threadIdx.y][threadIdx.x] = s;
  __syncthreads();
  if (threadIdx.y == 0 && c < W) {
    float4 t = red[0][threadIdx.x];
#pragma unroll
    for (int k = 1; k < 8; ++k) { t.x += red[k][threadIdx.x].x; t.y += red[k][threadIdx.x].y; t.z += red[k][threadIdx.x].z; t.w += red[k][threadIdx.x].w; }
    if constexpr (std::is_same<TOut, float>::value) {
      *reinterpret_cast<float4*>(out + c) = t;
    } else {
      *reinterpret_cast<uint2*>(out + c) = make_uint2(pwa80::pack_bf16(t.x, t.y), pwa80::pack_bf16(t.z, t.w));
    }
  }
}

}  // namespace

// z [L, L, DZ] bf16, key mask uint8 [L] (empty: all valid), lnw / lnb fp32 [DZ], wb [8, DZ] bf16 -> w [8, L, L] bf16
void pair_fwd(torch::Tensor z, torch::Tensor mask, torch::Tensor lnw, torch::Tensor lnb, torch::Tensor wb, torch::Tensor w, double eps) {
  check_bf16(z, "z"); check_bf16(wb, "wb"); check_bf16(w, "w"); check_f32(lnw, "lnw"); check_f32(lnb, "lnb");
  const int L = (int)z.size(0), DZ = (int)z.size(2);
  TORCH_CHECK(z.dim() == 3 && z.size(1) == L && L % 16 == 0 && wb.sizes() == torch::IntArrayRef({8, DZ}) && w.sizes() == torch::IntArrayRef({8, L, L}), "pair_fwd shapes");
  check_mask(mask, L, "mask");
  const at::cuda::CUDAGuard guard(z.device());
  pwa80::PairFwdParams p;
  p.z = ptr<__nv_bfloat16>(z); p.mask = mask.numel() ? reinterpret_cast<const uint8_t*>(mask.data_ptr()) : nullptr; p.lnw = lnw.data_ptr<float>(); p.lnb = lnb.data_ptr<float>();
  p.wb = ptr<__nv_bfloat16>(wb); p.w = reinterpret_cast<__nv_bfloat16*>(w.data_ptr()); p.L = L; p.eps = (float)eps;
  auto st = at::cuda::getCurrentCUDAStream();
  if (DZ == 128) launch_pair_fwd<128>(p, st);
  else if (DZ == 256) launch_pair_fwd<256>(p, st);
  else if (DZ == 384) launch_pair_fwd<384>(p, st);
  else TORCH_CHECK(false, "pair_fwd: d_pair must be 128, 256 or 384");
}

// m [S, L, D] bf16, lnw / lnb fp32 [D], wv [8 C, D] bf16 -> v [8, L, S C] bf16 and stats fp32 [S, L, 2] (empty: not saved)
void ln_v(torch::Tensor m, torch::Tensor lnw, torch::Tensor lnb, torch::Tensor wv, torch::Tensor v, torch::Tensor stats, double eps) {   // v: [8 ns, L, kp C]
  check_bf16(m, "m"); check_bf16(wv, "wv"); check_bf16(v, "v"); check_f32(lnw, "lnw"); check_f32(lnb, "lnb");
  const int S = (int)m.size(0), L = (int)m.size(1), D = (int)m.size(2), HC = (int)wv.size(0), C = HC / 8;
  TORCH_CHECK(m.dim() == 3 && wv.dim() == 2 && wv.size(1) == D && HC == 8 * C, "ln_v shapes");
  const HeadMajor hm = head_major(v, S, L, C, "v");
  TORCH_CHECK(stats.numel() == 0 || (stats.is_cuda() && stats.scalar_type() == at::kFloat && stats.is_contiguous() && stats.numel() == (int64_t)S * L * 2), "stats: fp32 [S, L, 2]");
  const at::cuda::CUDAGuard guard(m.device());
  pwa80::LnVParams p;
  p.m = ptr<__nv_bfloat16>(m); p.lnw = lnw.data_ptr<float>(); p.lnb = lnb.data_ptr<float>(); p.wv = ptr<__nv_bfloat16>(wv); p.v = reinterpret_cast<__nv_bfloat16*>(v.data_ptr());
  p.stats = stats.numel() ? reinterpret_cast<float2*>(stats.data_ptr()) : nullptr;
  p.S = S; p.L = L; p.ntile_s = (S + 127) / 128; p.ntile = L * p.ntile_s; p.kp = hm.kp; p.ns = hm.ns; p.eps = (float)eps;
  auto st = at::cuda::getCurrentCUDAStream();
  const bool save = stats.numel() > 0;
#define LNV(D_, C_) if (D == D_ && C == C_) { if (save) launch_ln_v<D_, C_, true>(p, st); else launch_ln_v<D_, C_, false>(p, st); return; }
  LNV(64, 8) LNV(64, 16) LNV(64, 32) LNV(128, 8) LNV(128, 16)
#undef LNV
  TORCH_CHECK(false, "ln_v: unsupported (d_msa, d_hidden)");
}

// m [S, L, D], o [8, L, S C] (the contraction w v), wg [8 C, D], wo [D, 8 C], keep [L, D] bf16 (empty: no dropout) -> out [S, L, D] = m + dropout(Wo (sigmoid(y Wg^T) o))
void gate_out(torch::Tensor m, torch::Tensor o, torch::Tensor lnw, torch::Tensor lnb, torch::Tensor wg, torch::Tensor wo, torch::Tensor keep, torch::Tensor out, double eps,
              double dscale) {
  check_bf16(m, "m"); check_bf16(o, "o"); check_bf16(wg, "wg"); check_bf16(wo, "wo"); check_bf16(out, "out"); check_f32(lnw, "lnw"); check_f32(lnb, "lnb");
  const int S = (int)m.size(0), L = (int)m.size(1), D = (int)m.size(2), HC = (int)wg.size(0), C = HC / 8;
  TORCH_CHECK(m.dim() == 3 && wg.sizes() == torch::IntArrayRef({HC, D}) && wo.sizes() == torch::IntArrayRef({D, HC}) && out.sizes() == m.sizes(), "gate_out shapes");
  const HeadMajor hm = head_major(o, S, L, C, "o");
  TORCH_CHECK(keep.numel() == 0 || (keep.is_cuda() && keep.scalar_type() == at::kBFloat16 && keep.is_contiguous() && keep.numel() == (int64_t)L * D), "keep: bf16 [L, D]");
  const at::cuda::CUDAGuard guard(m.device());
  pwa80::GateOutParams p;
  p.m = ptr<__nv_bfloat16>(m); p.o = ptr<__nv_bfloat16>(o); p.lnw = lnw.data_ptr<float>(); p.lnb = lnb.data_ptr<float>(); p.wg = ptr<__nv_bfloat16>(wg); p.wo = ptr<__nv_bfloat16>(wo);
  p.keep = keep.numel() ? ptr<__nv_bfloat16>(keep) : nullptr; p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.S = S; p.L = L; p.ntile_s = (S + 127) / 128; p.ntile = L * p.ntile_s; p.kp = hm.kp; p.ns = hm.ns; p.eps = (float)eps; p.dscale = (float)dscale;
  auto st = at::cuda::getCurrentCUDAStream();
#define GO(D_, C_) if (D == D_ && C == C_) { launch_gate_out<D_, C_>(p, st); return; }
  GO(64, 8) GO(64, 16) GO(64, 32) GO(128, 8) GO(128, 16)
#undef GO
  TORCH_CHECK(false, "gate_out: unsupported (d_msa, d_hidden)");
}

// ---------------------------------------------------------------------------------------------------------------- backward
// CTAs per channel group of the glue / CTAs of the dgv kernel for a problem (the caller sizes the partial buffers with them)
std::vector<int64_t> bwd_grids(int64_t S, int64_t L, int64_t D, int64_t C) {
  const int ntok = D == 64 ? pwa80::GlueCfg<64, 8>::NTOK : pwa80::GlueCfg<128, 8>::NTOK;      // rows per glue tile: 16 per warp
  const int ntile_g = (int)(L * ((S + ntok - 1) / ntok));
  const int ntile_d = (int)(L * ((S + 127) / 128));
#define GR(D_, C_) if (D == D_ && C == C_) return {glue_ctas<D_, C_>(ntile_g), dgv_ctas<D_, C_>(ntile_d)};
  GR(64, 8) GR(64, 16) GR(64, 32) GR(128, 8) GR(128, 16)
#undef GR
  TORCH_CHECK(false, "bwd_grids: unsupported (d_msa, d_hidden)");
  return {};
}

// m, dres [S, L, D], o [8, L, S C], wg [8 C, D], wo [D, 8 C], keep [L, D] (or empty) -> dO, dgp [8, L, S C] bf16 and part fp32 [nbx, NG, 2 D GCH] (dWo slice | dWg slice per group)
void glue(torch::Tensor m, torch::Tensor dres, torch::Tensor o, torch::Tensor lnw, torch::Tensor lnb, torch::Tensor wg, torch::Tensor wo, torch::Tensor keep, torch::Tensor dO,
          torch::Tensor dgp, torch::Tensor part, double eps, double dscale) {
  check_bf16(m, "m"); check_bf16(dres, "dres"); check_bf16(o, "o"); check_bf16(wg, "wg"); check_bf16(wo, "wo"); check_bf16(dO, "dO"); check_bf16(dgp, "dgp");
  check_f32(lnw, "lnw"); check_f32(lnb, "lnb"); check_f32(part, "part");
  const int S = (int)m.size(0), L = (int)m.size(1), D = (int)m.size(2), HC = (int)wg.size(0), C = HC / 8;
  const HeadMajor hm = head_major(o, S, L, C, "o");
  TORCH_CHECK(m.dim() == 3 && dres.sizes() == m.sizes() && dO.sizes() == o.sizes() && dgp.sizes() == o.sizes() &&
              wg.sizes() == torch::IntArrayRef({HC, D}) && wo.sizes() == torch::IntArrayRef({D, HC}), "glue shapes");
  TORCH_CHECK(keep.numel() == 0 || (keep.is_cuda() && keep.scalar_type() == at::kBFloat16 && keep.is_contiguous() && keep.numel() == (int64_t)L * D), "keep: bf16 [L, D]");
  const at::cuda::CUDAGuard guard(m.device());
  pwa80::GlueParams p;
  p.m = ptr<__nv_bfloat16>(m); p.dres = ptr<__nv_bfloat16>(dres); p.o = ptr<__nv_bfloat16>(o); p.lnw = lnw.data_ptr<float>(); p.lnb = lnb.data_ptr<float>();
  p.wg = ptr<__nv_bfloat16>(wg); p.wo = ptr<__nv_bfloat16>(wo); p.keep = keep.numel() ? ptr<__nv_bfloat16>(keep) : nullptr;
  p.dO = reinterpret_cast<__nv_bfloat16*>(dO.data_ptr()); p.dgp = reinterpret_cast<__nv_bfloat16*>(dgp.data_ptr()); p.part = part.data_ptr<float>();
  p.S = S; p.L = L; p.kp = hm.kp; p.ns = hm.ns; p.eps = (float)eps; p.dscale = (float)dscale;
  auto st = at::cuda::getCurrentCUDAStream();
#define GL(D_, C_) if (D == D_ && C == C_) { \
    using G = pwa80::GlueCfg<D_, C_>; \
    p.ntile_s = (S + G::NTOK - 1) / G::NTOK; p.ntile = L * p.ntile_s; const int nbx = glue_ctas<D_, C_>(p.ntile); \
    TORCH_CHECK(part.numel() == (int64_t)nbx * G::NG * 2 * D_ * G::GCH, "glue: part [nbx, NG, 2 D GCH]"); launch_glue<D_, C_>(p, nbx, st); return; }
  GL(64, 8) GL(64, 16) GL(64, 32) GL(128, 8) GL(128, 16)
#undef GL
  TORCH_CHECK(false, "glue: unsupported (d_msa, d_hidden)");
}

// m, dres [S, L, D], dgp, dv [8, L, S C], wg, wv [8 C, D] -> dm [S, L, D] bf16 and part fp32 [nb, 8 C D + 2 D] (dWv, dgamma, dbeta)
void dgv_bwd(torch::Tensor m, torch::Tensor dres, torch::Tensor dgp, torch::Tensor dv, torch::Tensor lnw, torch::Tensor lnb, torch::Tensor wg, torch::Tensor wv, torch::Tensor dm,
             torch::Tensor part, double eps) {
  check_bf16(m, "m"); check_bf16(dres, "dres"); check_bf16(dgp, "dgp"); check_bf16(dv, "dv"); check_bf16(wg, "wg"); check_bf16(wv, "wv"); check_bf16(dm, "dm");
  check_f32(lnw, "lnw"); check_f32(lnb, "lnb"); check_f32(part, "part");
  const int S = (int)m.size(0), L = (int)m.size(1), D = (int)m.size(2), HC = (int)wg.size(0), C = HC / 8;
  const HeadMajor hm = head_major(dgp, S, L, C, "dgp");
  TORCH_CHECK(m.dim() == 3 && dres.sizes() == m.sizes() && dm.sizes() == m.sizes() && dv.sizes() == dgp.sizes() &&
              wg.sizes() == torch::IntArrayRef({HC, D}) && wv.sizes() == wg.sizes(), "dgv_bwd shapes");
  const at::cuda::CUDAGuard guard(m.device());
  pwa80::DgvParams p;
  p.m = ptr<__nv_bfloat16>(m); p.dres = ptr<__nv_bfloat16>(dres); p.dgp = ptr<__nv_bfloat16>(dgp); p.dv = ptr<__nv_bfloat16>(dv); p.lnw = lnw.data_ptr<float>(); p.lnb = lnb.data_ptr<float>();
  p.wg = ptr<__nv_bfloat16>(wg); p.wv = ptr<__nv_bfloat16>(wv); p.dm = reinterpret_cast<__nv_bfloat16*>(dm.data_ptr()); p.part = part.data_ptr<float>();
  p.S = S; p.L = L; p.ntile_s = (S + 127) / 128; p.ntile = L * p.ntile_s; p.kp = hm.kp; p.ns = hm.ns; p.eps = (float)eps;
  auto st = at::cuda::getCurrentCUDAStream();
#define DG(D_, C_) if (D == D_ && C == C_) { const int nb = dgv_ctas<D_, C_>(p.ntile); \
    TORCH_CHECK(part.numel() == (int64_t)nb * (8 * C_ * D_ + 2 * D_), "dgv_bwd: part [nb, 8 C D + 2 D]"); launch_dgv<D_, C_>(p, nb, st); return; }
  DG(64, 8) DG(64, 16) DG(64, 32) DG(128, 8) DG(128, 16)
#undef DG
  TORCH_CHECK(false, "dgv_bwd: unsupported (d_msa, d_hidden)");
}

// z [L, L, DZ], w bf16 [8, L, L], dw fp32 [8, L, L], key mask uint8 [L] (or empty), lnw / lnb fp32 [DZ], wb [8, DZ] bf16 -> dz [L, L, DZ] bf16, pM fp32 [L, 8, DZ], pS fp32 [L, 8]
void pair_bwd(torch::Tensor z, torch::Tensor w, torch::Tensor dw, torch::Tensor mask, torch::Tensor lnw, torch::Tensor lnb, torch::Tensor wb, torch::Tensor dz, torch::Tensor pM,
              torch::Tensor pS, double eps, int64_t variant) {
  check_bf16(z, "z"); check_bf16(w, "w"); check_bf16(wb, "wb"); check_bf16(dz, "dz"); check_f32(dw, "dw"); check_f32(lnw, "lnw"); check_f32(lnb, "lnb"); check_f32(pM, "pM"); check_f32(pS, "pS");
  const int L = (int)z.size(0), DZ = (int)z.size(2);
  TORCH_CHECK(z.dim() == 3 && z.size(1) == L && L % 16 == 0 && w.sizes() == torch::IntArrayRef({8, L, L}) && dw.dim() == 3 && dw.size(0) % 8 == 0 && dw.size(0) >= 8 &&
              dw.size(1) == L && dw.size(2) == L && dz.sizes() == z.sizes() &&
              pM.numel() == (int64_t)L * 8 * DZ && pS.numel() == (int64_t)L * 8 && wb.sizes() == torch::IntArrayRef({8, DZ}), "pair_bwd shapes");
  check_mask(mask, L, "mask");
  const at::cuda::CUDAGuard guard(z.device());
  pwa80::PairBwdParams p;
  p.z = ptr<__nv_bfloat16>(z); p.w = ptr<__nv_bfloat16>(w); p.dw = dw.data_ptr<float>(); p.mask = mask.numel() ? reinterpret_cast<const uint8_t*>(mask.data_ptr()) : nullptr;
  p.lnw = lnw.data_ptr<float>(); p.lnb = lnb.data_ptr<float>(); p.wb = ptr<__nv_bfloat16>(wb); p.dz = reinterpret_cast<__nv_bfloat16*>(dz.data_ptr()); p.pM = pM.data_ptr<float>();
  p.pS = pS.data_ptr<float>(); p.L = L; p.ns = (int)(dw.size(0) / 8); p.eps = (float)eps;
  auto st = at::cuda::getCurrentCUDAStream();
  if (DZ == 128) { if (variant == 1) launch_pair_bwd<128, 1>(p, st); else launch_pair_bwd<128, 2>(p, st); }   // two CTAs per SM (<= 128 registers) is the shipped schedule; variant 1: one
  else if (DZ == 256) launch_pair_bwd<256, 1>(p, st);
  else if (DZ == 384) launch_pair_bwd<384, 1>(p, st);
  else TORCH_CHECK(false, "pair_bwd: d_pair must be 128, 256 or 384");
}

// in fp32 [R, W] -> out [W] (fp32 or bf16): the rows summed in order. W a multiple of 4.
void reduce_rows(torch::Tensor in, torch::Tensor out) {
  check_f32(in, "in");
  TORCH_CHECK(in.dim() == 2 && in.size(1) % 4 == 0 && out.is_cuda() && out.is_contiguous() && out.numel() == in.size(1), "reduce_rows: [R, W] -> [W], W a multiple of 4");
  const int R = (int)in.size(0);
  const long W = in.size(1);
  const at::cuda::CUDAGuard guard(in.device());
  const long blocks = (W / 4 + 31) / 32;
  const dim3 thr(32, 8);
  auto st = at::cuda::getCurrentCUDAStream();
  if (out.scalar_type() == at::kFloat) reduce_rows_kernel<float><<<blocks, thr, 0, st>>>(in.data_ptr<float>(), out.data_ptr<float>(), R, W);
  else if (out.scalar_type() == at::kBFloat16) reduce_rows_kernel<__nv_bfloat16><<<blocks, thr, 0, st>>>(in.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), R, W);
  else TORCH_CHECK(false, "reduce_rows: out must be fp32 or bf16");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("pair_fwd", &pair_fwd);
  mod.def("ln_v", &ln_v);
  mod.def("gate_out", &gate_out);
  mod.def("bwd_grids", &bwd_grids);
  mod.def("glue", &glue);
  mod.def("dgv_bwd", &dgv_bwd);
  mod.def("pair_bwd", &pair_bwd);
  mod.def("reduce_rows", &reduce_rows);
}
