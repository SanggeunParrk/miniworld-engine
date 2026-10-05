// ops.cu -- torch bindings of the A100 (sm_80) OuterProductMean kernels.
//
//   prologue      opm_prologue_sm80.cuh      LayerNorm + left / right projections + mask -> A, B [S, (L 32)] (the s-major operands of the grouped outer product)
//   epilogue      opm_epilogue_sm80.cuh      O [(L 32), (L 32)] -> / n, d_hidden^2 -> d_pair projection, bias, residual
//   dgrad         opm_dgrad_sm80.cuh         dz -> dzn, dO (grouped layout), dbo partials
//   dwo           opm_dwo_sm80.cuh           dWo partials (split over rows of i)
//   prologue_bwd  opm_prologue_bwd_sm80.cuh  dA | dB -> dm, the dW / dgamma / dbeta partials
//   reduce_rows   fixed-order sum of fp32 partial rows
// Every function takes its output tensors (allocated by the caller) and launches on the current stream.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <set>
#include <type_traits>
#include <utility>

#include "opm_dgrad_sm80.cuh"
#include "opm_dwo_sm80.cuh"
#include "opm_epilogue_sm80.cuh"
#include "opm_prologue_bwd_sm80.cuh"
#include "opm_prologue_sm80.cuh"

namespace {

void check_bf16(const torch::Tensor& t, const char* name) { TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.is_contiguous(), name, ": a contiguous bf16 CUDA tensor"); }
void check_f32(const torch::Tensor& t, const char* name) { TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kFloat && t.is_contiguous(), name, ": a contiguous fp32 CUDA tensor"); }

// the dynamic shared-memory limit of a kernel, raised once per (kernel, device): kernels of one signature share a function-pointer type, so the
// flag is keyed by the address
template <class K> void raise_smem(K kernel, int bytes) {
  static std::set<std::pair<const void*, int>> done;
  const std::pair<const void*, int> key{reinterpret_cast<const void*>(kernel), (int)at::cuda::current_device()};
  if (done.insert(key).second) C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

template <typename T> const T* ptr(const torch::Tensor& t) { return t.numel() ? reinterpret_cast<const T*>(t.data_ptr()) : nullptr; }

template <int CM, int NB, bool ST> void launch_prologue(opm80::PrologueParams p, int L, cudaStream_t st) {
  using C = opm80::PrologueCfg<CM, NB>;
  constexpr int NMINB = C::SMEM <= 80 * 1024 ? 2 : 1;                 // CTAs per SM
  raise_smem(opm80::opm_prologue_kernel<CM, NB, ST, NMINB>, C::SMEM);
  p.nti = (L + C::NTOK - 1) / C::NTOK;
  p.ntile = p.S * p.nti;
  const int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int grid = std::min(p.ntile, nsm * NMINB);
  opm80::opm_prologue_kernel<CM, NB, ST, NMINB><<<grid, C::NTHR, C::SMEM, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// variant 0: the shipped prefetch depth (3 tile buffers); 1: 2 buffers; 2: 4 buffers
template <int CM, bool ST> void launch_prologue_v(const opm80::PrologueParams& p, int L, int variant, cudaStream_t st) {
  if (variant == 1) launch_prologue<CM, 2, ST>(p, L, st);
  else if (variant == 2) launch_prologue<CM, 4, ST>(p, L, st);
  else launch_prologue<CM, 3, ST>(p, L, st);
}

template <int CZ, int NS, int NB, bool WIDE = false> void launch_epilogue(const opm80::EpilogueParams& p, cudaStream_t st) {
  using C = opm80::EpCfg<CZ, NS, NB, WIDE>;
  raise_smem(opm80::opm_epilogue_kernel<CZ, NS, NB, WIDE>, C::SMEM);
  dim3 grid(((p.L + C::BJ - 1) / C::BJ) * p.ntile, (p.L + C::BI - 1) / C::BI);
  opm80::opm_epilogue_kernel<CZ, NS, NB, WIDE><<<grid, C::NTHR, C::SMEM, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int CZ, int NS, int NB> void launch_dgrad(const opm80::DgradParams& p, cudaStream_t st) {
  using C = opm80::DgCfg<CZ, NS, NB>;
  raise_smem(opm80::opm_dgrad_kernel<CZ, NS, NB>, C::SMEM);
  dim3 grid((p.L + C::BJ - 1) / C::BJ, (p.L + C::BI - 1) / C::BI);
  opm80::opm_dgrad_kernel<CZ, NS, NB><<<grid, C::NTHR, C::SMEM, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int CM, int NB, int NMINB> void launch_pb(const opm80::PbParams& p, int nblk, cudaStream_t st) {
  using C = opm80::PbCfg<CM, NB>;
  raise_smem(opm80::opm_prologue_bwd_kernel<CM, NB, NMINB>, C::SMEM);
  opm80::opm_prologue_bwd_kernel<CM, NB, NMINB><<<nblk, C::NTHR, C::SMEM, st>>>(p);
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
      *reinterpret_cast<uint2*>(out + c) = make_uint2(opm80::pack_bf16(t.x, t.y), opm80::pack_bf16(t.z, t.w));
    }
  }
}

}  // namespace

// m [S, L, CM] bf16, mask uint8 [S, L] (empty: all valid), lnw / lnb fp32 [CM], wl / wr [32, CM] bf16 -> a / b [S, L 32] bf16 (the s-major operands of the outer product),
// stats fp32 [S, L, 2] (empty: not saved)
void prologue(torch::Tensor m, torch::Tensor mask, torch::Tensor lnw, torch::Tensor lnb, torch::Tensor wl, torch::Tensor wr, torch::Tensor a, torch::Tensor b,
              torch::Tensor stats, double eps, int64_t variant) {
  check_bf16(m, "m"); check_bf16(wl, "wl"); check_bf16(wr, "wr"); check_bf16(a, "a"); check_bf16(b, "b");
  check_f32(lnw, "lnw"); check_f32(lnb, "lnb");
  TORCH_CHECK(m.dim() == 3, "m: [S, L, CM]");
  const int S = (int)m.size(0), L = (int)m.size(1), CM = (int)m.size(2);
  TORCH_CHECK(a.dim() == 2 && a.size(0) == S && a.size(1) == (int64_t)L * 32 && b.sizes() == a.sizes(), "a / b: [S, L * 32]");
  TORCH_CHECK(wl.sizes() == torch::IntArrayRef({32, CM}) && wr.sizes() == wl.sizes() && lnw.numel() == CM && lnb.numel() == CM, "weights");
  TORCH_CHECK(mask.numel() == 0 || (mask.is_cuda() && mask.scalar_type() == at::kByte && mask.is_contiguous() && mask.numel() == (int64_t)S * L), "mask: uint8 [S, L]");
  TORCH_CHECK(stats.numel() == 0 || (stats.is_cuda() && stats.scalar_type() == at::kFloat && stats.is_contiguous() && stats.numel() == (int64_t)S * L * 2), "stats: fp32 [S, L, 2]");
  const at::cuda::CUDAGuard guard(m.device());
  opm80::PrologueParams p;
  p.m = ptr<__nv_bfloat16>(m); p.mask = mask.numel() ? reinterpret_cast<const uint8_t*>(mask.data_ptr()) : nullptr;
  p.lnw = lnw.data_ptr<float>(); p.lnb = lnb.data_ptr<float>(); p.wl = ptr<__nv_bfloat16>(wl); p.wr = ptr<__nv_bfloat16>(wr);
  p.a = reinterpret_cast<__nv_bfloat16*>(a.data_ptr()); p.b = reinterpret_cast<__nv_bfloat16*>(b.data_ptr());
  p.stats = stats.numel() ? reinterpret_cast<float2*>(stats.data_ptr()) : nullptr;
  p.S = S; p.L = L; p.eps = (float)eps;
  auto st = at::cuda::getCurrentCUDAStream();
  const bool save = stats.numel() > 0;
  if (CM == 64) { if (save) launch_prologue_v<64, true>(p, L, (int)variant, st); else launch_prologue_v<64, false>(p, L, (int)variant, st); }
  else if (CM == 128) { if (save) launch_prologue_v<128, true>(p, L, (int)variant, st); else launch_prologue_v<128, false>(p, L, (int)variant, st); }
  else TORCH_CHECK(false, "prologue: d_msa must be 64 or 128");
}

// O [L 32, L 32] bf16, norm fp32 [L, L] (empty: the constant norm_const), wo [CZ, 1024] bf16, bias fp32 [CZ], residual [L, L, CZ] bf16 (empty: none) -> out [L, L, CZ] bf16
// variant: the schedule (0: the shipped one; 1..: ring depth / CTAs per SM experiments, see the dispatch below)
void epilogue(torch::Tensor O, torch::Tensor norm, double norm_const, torch::Tensor wo, torch::Tensor bias, torch::Tensor residual, torch::Tensor out, int64_t norm_first,
              int64_t variant) {
  check_bf16(O, "O"); check_bf16(wo, "wo"); check_bf16(out, "out"); check_f32(bias, "bias");
  const int L = (int)out.size(0), CZ = (int)out.size(2);
  TORCH_CHECK(out.dim() == 3 && out.size(1) == L && O.dim() == 2 && O.size(0) == (int64_t)L * 32 && O.size(1) == (int64_t)L * 32, "O / out shapes");
  TORCH_CHECK(wo.sizes() == torch::IntArrayRef({CZ, 1024}) && bias.numel() == CZ, "wo / bias");
  TORCH_CHECK(norm.numel() == 0 || (norm.is_cuda() && norm.scalar_type() == at::kFloat && norm.is_contiguous() && norm.numel() == (int64_t)L * L), "norm: fp32 [L, L]");
  TORCH_CHECK(residual.numel() == 0 || (residual.is_cuda() && residual.scalar_type() == at::kBFloat16 && residual.is_contiguous() && residual.numel() == out.numel()), "residual");
  const at::cuda::CUDAGuard guard(O.device());
  opm80::EpilogueParams p;
  p.O = ptr<__nv_bfloat16>(O); p.norm = norm.numel() ? norm.data_ptr<float>() : nullptr; p.wo = ptr<__nv_bfloat16>(wo); p.bias = bias.data_ptr<float>();
  p.residual = residual.numel() ? ptr<__nv_bfloat16>(residual) : nullptr; p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.ldo = O.stride(0); p.L = L; p.norm_const = (float)norm_const; p.norm_first = (int)norm_first; p.cz = CZ; p.ntile = 1;
  auto st = at::cuda::getCurrentCUDAStream();
  if (CZ == 128) {
    if (variant == 1) launch_epilogue<128, 3, 1>(p, st);        // one CTA per SM, 3 stages
    else if (variant == 2) launch_epilogue<128, 4, 1>(p, st);   // one CTA, 4 stages
    else launch_epilogue<128, 2, 2>(p, st);                     // two CTAs per SM, 2 stages: the shipped schedule
  } else if (CZ == 256 || CZ == 384) {
    if (variant == 1) {                                                          // wide: 16 warps, 128 pairs and all d_pair channels, one CTA per SM
      if (CZ == 256) launch_epilogue<256, 2, 1, true>(p, st); else launch_epilogue<384, 2, 1, true>(p, st);
    } else if (variant == 2) {                                                   // the first version: 8 warps, 64 pairs, all channels (two CTAs per SM at 256, one at 384)
      if (CZ == 256) launch_epilogue<256, 2, 2>(p, st); else launch_epilogue<384, 2, 1>(p, st);
    } else {                                                                     // column tiles of 128 channels: the d_pair 128 CTA, two per SM (the shipped schedule)
      p.ntile = CZ / 128;
      launch_epilogue<128, 2, 2>(p, st);
    }
  } else {
    TORCH_CHECK(false, "epilogue: d_pair must be 128, 256 or 384");
  }
}

// dz [L, L, CZ] bf16, norm (as above), wo [CZ, 1024] -> dO [L 32, L 32] bf16, dzn [L, L, CZ] bf16, dbo_part fp32 [ceil(L / 32) * ceil(L / 4), CZ]
void dgrad(torch::Tensor dz, torch::Tensor norm, double norm_const, torch::Tensor wo, torch::Tensor dO, torch::Tensor dzn, torch::Tensor dbo_part, int64_t norm_first,
           int64_t variant) {
  check_bf16(dz, "dz"); check_bf16(wo, "wo"); check_bf16(dO, "dO"); check_bf16(dzn, "dzn"); check_f32(dbo_part, "dbo_part");
  const int L = (int)dz.size(0), CZ = (int)dz.size(2);
  TORCH_CHECK(dz.dim() == 3 && dz.size(1) == L && dO.size(0) == (int64_t)L * 32 && dO.size(1) == (int64_t)L * 32 && dzn.sizes() == dz.sizes(), "dz / dO / dzn shapes");
  const int gx = (L + 31) / 32, gy = (L + 3) / 4;
  TORCH_CHECK(dbo_part.numel() == (int64_t)gx * gy * CZ, "dbo_part: [gx * gy, CZ]");
  TORCH_CHECK(norm.numel() == 0 || (norm.is_cuda() && norm.scalar_type() == at::kFloat && norm.is_contiguous() && norm.numel() == (int64_t)L * L), "norm: fp32 [L, L]");
  const at::cuda::CUDAGuard guard(dz.device());
  opm80::DgradParams p;
  p.dz = ptr<__nv_bfloat16>(dz); p.norm = norm.numel() ? norm.data_ptr<float>() : nullptr; p.wo = ptr<__nv_bfloat16>(wo);
  p.dO = reinterpret_cast<__nv_bfloat16*>(dO.data_ptr()); p.dzn = reinterpret_cast<__nv_bfloat16*>(dzn.data_ptr()); p.dbo_part = dbo_part.data_ptr<float>();
  p.ldo = dO.stride(0); p.L = L; p.norm_const = (float)norm_const; p.norm_first = (int)norm_first;
  auto st = at::cuda::getCurrentCUDAStream();
  if (CZ == 128) {
    if (variant == 1) launch_dgrad<128, 3, 1>(p, st);           // one CTA per SM, 3 stages
    else launch_dgrad<128, 2, 2>(p, st);                        // two CTAs per SM, 2 stages: the shipped schedule
  } else if (CZ == 256) {
    launch_dgrad<256, 3, 1>(p, st);
  } else if (CZ == 384) {
    launch_dgrad<384, 3, 1>(p, st);
  } else {
    TORCH_CHECK(false, "dgrad: d_pair must be 128, 256 or 384");
  }
}

// dzn [L, L, CZ] bf16, O [L 32, L 32] bf16 -> part fp32 [splits, CZ, 1024]; each split covers i_per rows of i
void dwo(torch::Tensor dzn, torch::Tensor O, torch::Tensor part, int64_t i_per) {
  check_bf16(dzn, "dzn"); check_bf16(O, "O"); check_f32(part, "part");
  const int L = (int)dzn.size(0), CZ = (int)dzn.size(2);
  TORCH_CHECK(CZ % 128 == 0 && O.size(0) == (int64_t)L * 32, "dwo: d_pair a multiple of 128");
  const int splits = (int)((L + i_per - 1) / i_per);
  TORCH_CHECK(part.numel() == (int64_t)splits * CZ * 1024, "part: [splits, CZ, 1024]");
  const at::cuda::CUDAGuard guard(dzn.device());
  opm80::DwoParams p;
  p.dzn = ptr<__nv_bfloat16>(dzn); p.O = ptr<__nv_bfloat16>(O); p.part = part.data_ptr<float>(); p.ldo = O.stride(0); p.L = L; p.CZ = CZ; p.i_per = (int)i_per;
  raise_smem(opm80::opm_dwo_kernel, opm80::DwoCfg::SMEM);
  dim3 grid(8, CZ / 128, splits);
  opm80::opm_dwo_kernel<<<grid, opm80::DwoCfg::NTHR, opm80::DwoCfg::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dA, dB [rows >= S, L 32] bf16, m [S, L, CM], stats fp32 [S, L, 2], mask uint8 [S, L] (or empty), lnw / lnb fp32 [CM], wl / wr [32, CM] -> dm [S, L, CM] bf16 and
// part fp32 [nblk, 64 CM + 2 CM] (rows: dWl | dWr [64][CM], dgamma, dbeta)
void prologue_bwd(torch::Tensor dA, torch::Tensor dB, torch::Tensor m, torch::Tensor stats, torch::Tensor mask, torch::Tensor lnw, torch::Tensor lnb, torch::Tensor wl,
                  torch::Tensor wr, torch::Tensor dm, torch::Tensor part, int64_t variant) {
  check_bf16(dA, "dA"); check_bf16(dB, "dB"); check_bf16(m, "m"); check_bf16(dm, "dm"); check_bf16(wl, "wl"); check_bf16(wr, "wr");
  check_f32(stats, "stats"); check_f32(lnw, "lnw"); check_f32(lnb, "lnb"); check_f32(part, "part");
  const int S = (int)m.size(0), L = (int)m.size(1), CM = (int)m.size(2);
  TORCH_CHECK(dA.dim() == 2 && dA.size(0) >= S && dA.size(1) == (int64_t)L * 32 && dB.sizes() == dA.sizes() && dm.sizes() == m.sizes(), "dA / dB / dm shapes");
  TORCH_CHECK(mask.numel() == 0 || (mask.is_cuda() && mask.scalar_type() == at::kByte && mask.is_contiguous() && mask.numel() == (int64_t)S * L), "mask: uint8 [S, L]");
  const int ntile_s = (S + 63) / 64, ntile = ntile_s * ((L + 1) / 2);
  const int nblk = (int)part.size(0);
  TORCH_CHECK(part.dim() == 2 && part.size(1) == 66 * CM && nblk >= 1 && nblk <= ntile, "part: [nblk <= tiles, 66 CM]");
  const at::cuda::CUDAGuard guard(m.device());
  opm80::PbParams p;
  p.dA = ptr<__nv_bfloat16>(dA); p.dB = ptr<__nv_bfloat16>(dB); p.m = ptr<__nv_bfloat16>(m); p.stats = reinterpret_cast<const float2*>(stats.data_ptr());
  p.mask = mask.numel() ? reinterpret_cast<const uint8_t*>(mask.data_ptr()) : nullptr; p.lnw = lnw.data_ptr<float>(); p.lnb = lnb.data_ptr<float>();
  p.wl = ptr<__nv_bfloat16>(wl); p.wr = ptr<__nv_bfloat16>(wr); p.dm = reinterpret_cast<__nv_bfloat16*>(dm.data_ptr()); p.part = part.data_ptr<float>();
  p.ldd = dA.stride(0); p.S = S; p.L = L; p.ntile_s = ntile_s; p.ntile = ntile;
  auto st = at::cuda::getCurrentCUDAStream();
  // variant 0 (shipped): one tile buffer, two CTAs per SM (d_msa 64); variant 1: double-buffered tiles, one CTA per SM
  if (CM == 64) { if (variant == 1) launch_pb<64, 2, 1>(p, nblk, st); else launch_pb<64, 1, 2>(p, nblk, st); }
  else if (CM == 128) launch_pb<128, 2, 1>(p, nblk, st);
  else TORCH_CHECK(false, "prologue_bwd: d_msa must be 64 or 128");
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
  mod.def("prologue", &prologue);
  mod.def("epilogue", &epilogue);
  mod.def("dgrad", &dgrad);
  mod.def("dwo", &dwo);
  mod.def("prologue_bwd", &prologue_bwd);
  mod.def("reduce_rows", &reduce_rows);
}
