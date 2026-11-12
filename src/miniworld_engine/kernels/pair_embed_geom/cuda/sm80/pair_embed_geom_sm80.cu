// pair_embed_geom_sm80.cu -- the A100 (sm_80) build of sm90/pair_embed_geom.cu: the same kernels and host entry points, the source compiled
// unchanged (the H100 build's file stays byte-identical); pe_sm80.cuh supplies the sm_80 forms of its five sm_90-only helpers (the table
// gradient's warpgroup product as mma.sync on the same swizzled shared-memory operands).
//
// One addition, the table gradient's impl 3 (this card's choice for the decoder table, sm80/__init__.py): the H100 grid of the mma.sync
// kernel (pg_bwd_mma, 4 x NBLK = 132 CTAs of 5 warps: sized for the H100's 132 SMs) leaves the A100's 108 SMs a few warps each, so the
// pairs are split in KS contiguous chunks whose grids run concurrently on two streams (fork / join by events: capturable), each writing
// its own NBLK partials; the KS x NBLK partials are then summed in a fixed order (bitwise deterministic, as the H100 kernels).
// `backward` below is the module's entry point; the H100 host function (renamed backward_h100 here) serves every other impl.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "pe_sm80.cuh"

void backward(torch::Tensor g, torch::Tensor codes, torch::Tensor tpart, torch::Tensor dtbl, int64_t mode, int64_t impl);
#define backward(...) backward_h100(__VA_ARGS__)   // the H100 definition only: the pybind entry `&backward` names the one below
#ifdef MWE_SM120
// RTX 5090 (sm_120a, 99 KB of shared memory a block): the LayerNorm backward of the encoder pair (mode 0) cannot keep its 317-row table
// (162 KB) in shared memory; `ln_backward` below reads it through the L1 there (mode 1, 80 KB, keeps the H100 kernel).
void ln_backward(torch::Tensor tbl, torch::Tensor codes, torch::Tensor dh, torch::Tensor dx, int64_t mode, double eps);
#define ln_backward(...) ln_backward_h100(__VA_ARGS__)
#endif
#include "../sm90/pair_embed_geom.cu"
#undef backward
#ifdef MWE_SM120
#undef ln_backward
#endif

namespace {
constexpr int KS = 2;   // concurrent pair chunks of impl 3

__global__ void pg80_reduce_tbl(const float* __restrict__ tpart, __nv_bfloat16* __restrict__ dtbl, int NTAB, int nparts) {
  const size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= (size_t)NTAB * C) return;
  float s = 0.f;
  for (int r = 0; r < nparts; r++) s += tpart[(size_t)r * NTAB * C + k];
  dtbl[k] = __float2bfloat16_rn(s);
}

struct Fork { cudaEvent_t start, done; };
Fork& fork_events() {
  static thread_local Fork f = [] {
    Fork e;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&e.start, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&e.done, cudaEventDisableTiming));
    return e;
  }();
  return f;
}
}  // namespace

// impl 3: tpart fp32 scratch [KS][NBLK][NTAB][256]; else the H100 entry point (tpart [NBLK][NTAB][256])
void backward(torch::Tensor g, torch::Tensor codes, torch::Tensor tpart, torch::Tensor dtbl, int64_t mode, int64_t impl) {
  if (impl != 3) { backward_h100(g, codes, tpart, dtbl, mode, impl); return; }
  const at::cuda::CUDAGuard guard(g.device());
  const int B = g.size(0), n = g.size(1), ntab = mode == 0 ? NT : NT_CA;
  const long long P = (long long)B * n * n, per = (P + KS - 1) / KS;
  TORCH_CHECK(tpart.numel() == (int64_t)KS * NBLK * ntab * C, "backward impl 3: tpart [", KS, ", NBLK, NTAB, 256] fp32");
  auto* gp = reinterpret_cast<const __nv_bfloat16*>(g.data_ptr<at::BFloat16>());
  const uint4* cp = reinterpret_cast<const uint4*>(codes.data_ptr<int>());
  float* tp = tpart.data_ptr<float>();
  auto main = at::cuda::getCurrentCUDAStream();
  auto side = at::cuda::getStreamFromPool(false, g.get_device());
  Fork& ev = fork_events();
  C10_CUDA_CHECK(cudaEventRecord(ev.start, main));
  C10_CUDA_CHECK(cudaStreamWaitEvent(side, ev.start, 0));
  for (int k = 0; k < KS; k++) {
    const long long p0 = k * per, np = std::max(0LL, std::min(P, p0 + per) - p0);
    cudaStream_t s = k == 0 ? main.stream() : side.stream();
    float* part = tp + (size_t)k * NBLK * ntab * C;
    if (mode == 0) pg_bwd_mma<0><<<dim3(4, NBLK), 13 * 32, 0, s>>>(gp + (size_t)p0 * C, cp + p0, np, part);
    else pg_bwd_mma<1><<<dim3(4, NBLK), 5 * 32, 0, s>>>(gp + (size_t)p0 * C, cp + p0, np, part);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  C10_CUDA_CHECK(cudaEventRecord(ev.done, side));
  C10_CUDA_CHECK(cudaStreamWaitEvent(main, ev.done, 0));
  pg80_reduce_tbl<<<(ntab * C + 255) / 256, 256, 0, main>>>(tp, reinterpret_cast<__nv_bfloat16*>(dtbl.data_ptr<at::BFloat16>()), ntab, KS * NBLK);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#ifdef MWE_SM120
namespace {
// pg_ln_bwd with the table read from global memory (through the L1) instead of a shared-memory copy: the same warp-per-pair schedule, the
// same two-deep load pipeline and the same arithmetic (ln_bwd_group), so the result is bitwise the shared-memory kernel's.
template <int MODE, int UL>
__global__ void __launch_bounds__(16 * 32, 1)
pg120_ln_bwd_gt(const __nv_bfloat16* __restrict__ tbl, const uint4* codes, const __nv_bfloat16* dh, __nv_bfloat16* dx, long long P, float eps) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, c0 = lane * 8;
  const long long stride = (long long)gridDim.x * 16;
  const long long p0 = (long long)blockIdx.x * 16 + warp;
  auto ldc = [&](long long p) { return p < P ? codes[p] : make_uint4(0u, 0u, 0u, 0u); };
  auto ldg = [&](long long p, const uint4& cd) { return (p < P && (cd.x & LIVE)) ? *reinterpret_cast<const uint4*>(dh + (size_t)p * C + c0) : make_uint4(0u, 0u, 0u, 0u); };
  uint4 cA[UL], gA[UL], cB[UL], gB[UL], cC[UL];
#pragma unroll
  for (int u = 0; u < UL; u++) { cA[u] = ldc(p0 + u * stride); cB[u] = ldc(p0 + (UL + u) * stride); }
#pragma unroll
  for (int u = 0; u < UL; u++) gA[u] = ldg(p0 + u * stride, cA[u]);
  for (long long p = p0; p < P; p += UL * stride) {
#pragma unroll
    for (int u = 0; u < UL; u++) gB[u] = ldg(p + (UL + u) * stride, cB[u]);
#pragma unroll
    for (int u = 0; u < UL; u++) cC[u] = ldc(p + (2 * UL + u) * stride);
    {
      uint4 o[UL];
      bool any = false;
#pragma unroll
      for (int u = 0; u < UL; u++) any |= (cA[u].x & LIVE) != 0;
      if (any) ln_bwd_group<MODE, UL>(tbl, cA, gA, c0, eps, o);
      else {
#pragma unroll
        for (int u = 0; u < UL; u++) o[u] = make_uint4(0u, 0u, 0u, 0u);
      }
#pragma unroll
      for (int u = 0; u < UL; u++) {
        const long long pu = p + u * stride;
        if (pu < P) *reinterpret_cast<uint4*>(dx + (size_t)pu * C + c0) = o[u];
      }
    }
#pragma unroll
    for (int u = 0; u < UL; u++) { cA[u] = cB[u]; gA[u] = gB[u]; cB[u] = cC[u]; }
  }
}
}  // namespace

void ln_backward(torch::Tensor tbl, torch::Tensor codes, torch::Tensor dh, torch::Tensor dx, int64_t mode, double eps) {
  static_assert((size_t)NT_CA * C * sizeof(__nv_bfloat16) <= 101376, "the decoder table must fit the 5090's 99 KB of shared memory");
  if (mode != 0) { ln_backward_h100(tbl, codes, dh, dx, mode, eps); return; }
  const at::cuda::CUDAGuard guard(tbl.device());
  const long long P = (long long)dh.size(0) * dh.size(1) * dh.size(2);
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  pg120_ln_bwd_gt<0, UL_G><<<sms, 16 * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(tbl.data_ptr<at::BFloat16>()), reinterpret_cast<const uint4*>(codes.data_ptr<int>()),
      reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr<at::BFloat16>()), reinterpret_cast<__nv_bfloat16*>(dx.data_ptr<at::BFloat16>()), P, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#endif
