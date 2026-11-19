// The OuterProductMean's grouped outer product on A100 with the dead tokens skipped: O[(i, c), (j, e)] = sum_s a[s, i, c] b[s, j, e], i.e. O = A^T B with
// A, B [S, 32 L] (the prologue's s-major operands), on a CUTLASS 2.x sm_80 GEMM (128 x 128 x 32 tiles, 64 x 64 warps, 4-stage cp.async pipeline: 0.96 of
// cuBLAS on the bare product).  A token whose mask column is all zero (padding: masked in every MSA row) has a = b = 0, so every O block of a pair with
// such a token is exactly 0: a threadblock whose 4 row tokens (or 4 column tokens) are all dead writes its zero tile instead of running the K loop.  The
// liveness comes from the device ([L] uint8), so the launch is shape-static (CUDA graphs) and the result is the dense product's, bit for bit.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"

namespace {
using bf = cutlass::bfloat16_t;
constexpr int TM = 128, TN = 128, TK = 32;
using Gemm = cutlass::gemm::device::Gemm<bf, cutlass::layout::ColumnMajor, bf, cutlass::layout::RowMajor, bf, cutlass::layout::RowMajor, float,
                                         cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<TM, TN, TK>,
                                         cutlass::gemm::GemmShape<64, 64, TK>, cutlass::gemm::GemmShape<16, 8, 16>,
                                         cutlass::epilogue::thread::LinearCombination<bf, 8, float, float>,
                                         cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 4>;
using GK = Gemm::GemmKernel;

__global__ void __launch_bounds__(GK::kThreadCount) ogemm_skip_kernel(typename GK::Params params, const uint8_t* __restrict__ live, bf* __restrict__ out,
                                                                     int ldo, int ch) {   // ch: channels per token (32: the OPM, 8: a Chai-1 group)
  extern __shared__ __align__(16) unsigned char smem[];
  typename GK::ThreadblockSwizzle sw;
  const cutlass::gemm::GemmCoord tile = sw.get_tile_offset(params.swizzle_log_tile);
  if (params.grid_tiled_shape.m() <= tile.m() || params.grid_tiled_shape.n() <= tile.n()) return;
  const int tpt = TM / ch, t0 = tile.m() * tpt, u0 = tile.n() * tpt;      // tokens per tile side (4 or 16), 4-byte words of the liveness bytes
  uint32_t lm = 0u, ln = 0u;
  for (int w = 0; w < tpt / 4; ++w) {
    lm |= *reinterpret_cast<const uint32_t*>(live + t0 + 4 * w);
    ln |= *reinterpret_cast<const uint32_t*>(live + u0 + 4 * w);
  }
  if (lm != 0u && ln != 0u) {
    GK op;
    op(params, *reinterpret_cast<typename GK::SharedStorage*>(smem));
    return;
  }
  // a dead tile: zeros (128 rows x 256 B)
  const int r0 = tile.m() * TM, c0 = tile.n() * TN;
  for (int v = threadIdx.x; v < TM * (TN / 8); v += GK::kThreadCount) {
    const int r = v / (TN / 8), c = v % (TN / 8);
    *reinterpret_cast<uint4*>(out + (long)(r0 + r) * ldo + c0 + c * 8) = make_uint4(0, 0, 0, 0);
  }
}
}  // namespace

#ifndef MWE_SM120
// The batched form (the Chai-1 groups: G products of one launch share the liveness): one launch of G x tiles instead of G launches of 3 waves with a
// 0.67-full last one (A100, L 384, 8 groups: 8 x 90 us = 721 us -> one launch of 21 waves; cuBLAS's bmm does the same).
namespace batched {
using GemmB = cutlass::gemm::device::GemmBatched<bf, cutlass::layout::ColumnMajor, bf, cutlass::layout::RowMajor, bf, cutlass::layout::RowMajor, float,
                                                 cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<TM, TN, TK>,
                                                 cutlass::gemm::GemmShape<64, 64, TK>, cutlass::gemm::GemmShape<16, 8, 16>,
                                                 cutlass::epilogue::thread::LinearCombination<bf, 8, float, float>,
                                                 cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 4>;
using GKB = GemmB::GemmKernel;

__global__ void __launch_bounds__(GKB::kThreadCount) kernel(typename GKB::Params params, const uint8_t* __restrict__ live, bf* __restrict__ out, int ldo,
                                                          int64_t ostride, int ch) {
  extern __shared__ __align__(16) unsigned char smem[];
  typename GKB::ThreadblockSwizzle sw;
  const cutlass::gemm::GemmCoord tile = sw.get_tile_offset(params.swizzle_log_tile);
  if (params.grid_tiled_shape.m() <= tile.m() || params.grid_tiled_shape.n() <= tile.n()) return;
  const int tpt = TM / ch, t0 = tile.m() * tpt, u0 = tile.n() * tpt;
  uint32_t lm = 0u, ln = 0u;
  for (int w = 0; w < tpt / 4; ++w) {
    lm |= *reinterpret_cast<const uint32_t*>(live + t0 + 4 * w);
    ln |= *reinterpret_cast<const uint32_t*>(live + u0 + 4 * w);
  }
  if (lm != 0u && ln != 0u) {
    GKB op;
    op(params, *reinterpret_cast<typename GKB::SharedStorage*>(smem));
    return;
  }
  bf* o = out + (int64_t)sw.get_batch_idx() * ostride;
  const int r0 = tile.m() * TM, c0 = tile.n() * TN;
  for (int v = threadIdx.x; v < TM * (TN / 8); v += GKB::kThreadCount) {
    const int r = v / (TN / 8), c = v % (TN / 8);
    *reinterpret_cast<uint4*>(o + (long)(r0 + r) * ldo + c0 + c * 8) = make_uint4(0, 0, 0, 0);
  }
}
}  // namespace batched
#endif

#ifdef MWE_SM120
// The RTX 5090 (sm_120a, 99 KB of shared memory): from 16384 product rows (L 512 at 32 channels) a 256 x 128 tile, 3 stages (72 KB; 3-5 % ahead
// of 128 x 128 there, behind it on the shorter products, which keep the kernel above).
namespace sm120 {
constexpr int TM = 256, TN = 128, TK = 32;
using Gemm = cutlass::gemm::device::Gemm<bf, cutlass::layout::ColumnMajor, bf, cutlass::layout::RowMajor, bf, cutlass::layout::RowMajor, float,
                                         cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<TM, TN, TK>,
                                         cutlass::gemm::GemmShape<64, 64, TK>, cutlass::gemm::GemmShape<16, 8, 16>,
                                         cutlass::epilogue::thread::LinearCombination<bf, 8, float, float>,
                                         cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 3>;
using GK = Gemm::GemmKernel;
static_assert(sizeof(GK::SharedStorage) <= 101376, "the RTX 5090's 99 KB of shared memory per block");

__global__ void __launch_bounds__(GK::kThreadCount) ogemm_skip_kernel(typename GK::Params params, const uint8_t* __restrict__ live, bf* __restrict__ out,
                                                                     int ldo, int ch) {
  extern __shared__ __align__(16) unsigned char smem[];
  typename GK::ThreadblockSwizzle sw;
  const cutlass::gemm::GemmCoord tile = sw.get_tile_offset(params.swizzle_log_tile);
  if (params.grid_tiled_shape.m() <= tile.m() || params.grid_tiled_shape.n() <= tile.n()) return;
  const int tpm = TM / ch, tpn = TN / ch, t0 = tile.m() * tpm, u0 = tile.n() * tpn;  // tokens per tile side
  uint32_t lm = 0u, ln = 0u;
  for (int w = 0; w < tpm / 4; ++w) lm |= *reinterpret_cast<const uint32_t*>(live + t0 + 4 * w);
  for (int w = 0; w < tpn / 4; ++w) ln |= *reinterpret_cast<const uint32_t*>(live + u0 + 4 * w);
  if (lm != 0u && ln != 0u) {
    GK op;
    op(params, *reinterpret_cast<typename GK::SharedStorage*>(smem));
    return;
  }
  const int r0 = tile.m() * TM, c0 = tile.n() * TN;
  for (int v = threadIdx.x; v < TM * (TN / 8); v += GK::kThreadCount) {
    const int r = v / (TN / 8), c = v % (TN / 8);
    *reinterpret_cast<uint4*>(out + (long)(r0 + r) * ldo + c0 + c * 8) = make_uint4(0, 0, 0, 0);
  }
}

void launch(bf* A, bf* B, bf* C, const uint8_t* live, int M, int K, int ch) {
  const cutlass::gemm::GemmCoord problem(M, M, K);
  typename GK::ThreadblockSwizzle sw;
  const cutlass::gemm::GemmCoord grid_shape = sw.get_tiled_shape(problem, {TM, TN, TK}, 1);
  typename GK::Params params(problem, grid_shape, {A, M}, {B, M}, {C, M}, {C, M}, {1.f, 0.f});
  const int smem = (int)sizeof(typename GK::SharedStorage);
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(ogemm_skip_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    configured = true;
  }
  ogemm_skip_kernel<<<sw.get_grid_shape(grid_shape), GK::kThreadCount, smem, at::cuda::getCurrentCUDAStream()>>>(params, live, C, M, ch);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace sm120
#endif

// a, b [S, ch L] bf16 (row-major), live [L] uint8 (0 = a token masked in every MSA row), out [ch L, ch L] bf16 = a^T b; ch = 32 or 8 (128 / ch tokens a tile side)
void ogemm_skip(torch::Tensor a, torch::Tensor b, torch::Tensor live, torch::Tensor out, int64_t ch) {
  TORCH_CHECK(ch == 32 || ch == 8, "ogemm_skip: 32 or 8 channels a token");
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == at::kBFloat16 && a.is_contiguous() && b.sizes() == a.sizes() && b.is_contiguous() && b.scalar_type() == at::kBFloat16,
              "ogemm_skip: a / b contiguous bf16 [S, 32 L]");
  const int K = (int)a.size(0), M = (int)a.size(1);
  TORCH_CHECK(M % TM == 0 && K % 8 == 0, "ogemm_skip: ch L a multiple of 128, S % 8");
  TORCH_CHECK(live.is_cuda() && live.scalar_type() == at::kByte && live.is_contiguous() && live.numel() == M / ch && reinterpret_cast<uintptr_t>(live.data_ptr()) % 4 == 0,
              "ogemm_skip: live [L] uint8, 4-byte aligned");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.size(0) == M && out.size(1) == M, "ogemm_skip: out [ch L, ch L] bf16");
  const at::cuda::CUDAGuard guard(a.device());
  bf* A = reinterpret_cast<bf*>(a.data_ptr());
  bf* B = reinterpret_cast<bf*>(b.data_ptr());
  bf* C = reinterpret_cast<bf*>(out.data_ptr());
#ifdef MWE_SM120
  if (M >= 16384 && M % sm120::TM == 0) {
    sm120::launch(A, B, C, reinterpret_cast<const uint8_t*>(live.data_ptr()), M, K, (int)ch);
    return;
  }
#endif
  const cutlass::gemm::GemmCoord problem(M, M, K);
  typename GK::ThreadblockSwizzle sw;
  const cutlass::gemm::GemmCoord grid_shape = sw.get_tiled_shape(problem, {TM, TN, TK}, 1);
  typename GK::Params params(problem, grid_shape, {A, M}, {B, M}, {C, M}, {C, M}, {1.f, 0.f});
  const dim3 grid = sw.get_grid_shape(grid_shape);
  const int smem = (int)sizeof(typename GK::SharedStorage);
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(ogemm_skip_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    configured = true;
  }
  ogemm_skip_kernel<<<grid, GK::kThreadCount, smem, at::cuda::getCurrentCUDAStream()>>>(params, reinterpret_cast<const uint8_t*>(live.data_ptr()), C, M, (int)ch);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// live[l] = max over the MSA rows s of mask[s, l] (one launch; torch's max reduction over dim 0 costs 3-4x as much at S <= 1024)
__global__ void live_tokens_k(const uint8_t* __restrict__ m, uint8_t* __restrict__ live, int S, int L) {
  __shared__ uint8_t red[32][33];
  const int l = blockIdx.x * 32 + threadIdx.x;
  uint8_t v = 0;
  if (l < L) {
#pragma unroll 8
    for (int s = threadIdx.y; s < S; s += 32) v = max(v, m[(size_t)s * L + l]);
  }
  red[threadIdx.y][threadIdx.x] = v;
  __syncthreads();
  if (threadIdx.y == 0 && l < L) {
    for (int k = 0; k < 32; ++k) v = max(v, red[k][threadIdx.x]);
    live[l] = v;
  }
}

#ifndef MWE_SM120
// a, b [G, S, ch L] bf16, live [L] uint8, out [G, ch L, ch L] bf16: out[g] = a[g]^T b[g] with the dead tokens' blocks written as zeros (ch 8: a Chai-1 group)
void ogemm_skip_batched(torch::Tensor a, torch::Tensor b, torch::Tensor live, torch::Tensor out, int64_t ch) {
  TORCH_CHECK(ch == 32 || ch == 8, "ogemm_skip_batched: 32 or 8 channels a token");
  TORCH_CHECK(a.is_cuda() && a.dim() == 3 && a.scalar_type() == at::kBFloat16 && a.is_contiguous() && b.sizes() == a.sizes() && b.is_contiguous() && b.scalar_type() == at::kBFloat16,
              "ogemm_skip_batched: a / b contiguous bf16 [G, S, ch L]");
  const int G = (int)a.size(0), K = (int)a.size(1), M = (int)a.size(2);
  TORCH_CHECK(M % TM == 0 && K % 8 == 0, "ogemm_skip_batched: ch L a multiple of 128, S % 8");
  TORCH_CHECK(live.is_cuda() && live.scalar_type() == at::kByte && live.is_contiguous() && live.numel() == M / ch && reinterpret_cast<uintptr_t>(live.data_ptr()) % 4 == 0,
              "ogemm_skip_batched: live [L] uint8, 4-byte aligned");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.dim() == 3 && out.size(0) == G && out.size(1) == M && out.size(2) == M,
              "ogemm_skip_batched: out [G, ch L, ch L] bf16");
  const at::cuda::CUDAGuard guard(a.device());
  bf* A = reinterpret_cast<bf*>(a.data_ptr());
  bf* B = reinterpret_cast<bf*>(b.data_ptr());
  bf* C = reinterpret_cast<bf*>(out.data_ptr());
  using namespace batched;
  const cutlass::gemm::GemmCoord problem(M, M, K);
  typename GKB::ThreadblockSwizzle sw;
  const cutlass::gemm::GemmCoord grid_shape = sw.get_tiled_shape(problem, {TM, TN, TK}, G);
  typename GKB::Params params(problem, grid_shape, {A, M}, (int64_t)K * M, {B, M}, (int64_t)K * M, {C, M}, (int64_t)M * M, {C, M}, (int64_t)M * M, {1.f, 0.f}, G);
  const int smem = (int)sizeof(typename GKB::SharedStorage);
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    configured = true;
  }
  kernel<<<sw.get_grid_shape(grid_shape), GKB::kThreadCount, smem, at::cuda::getCurrentCUDAStream()>>>(params, reinterpret_cast<const uint8_t*>(live.data_ptr()), C, M, (int64_t)M * M, (int)ch);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#endif

// L a multiple of 16: 16 columns a thread (one 16-byte load), 128 rows in parallel a block, a tree max over the rows
__global__ void __launch_bounds__(1024) live_tokens_vec_k(const uint4* __restrict__ m, uint4* __restrict__ live, int S, int L16) {
  __shared__ uint4 red[128][8];
  const int g = blockIdx.x * 8 + threadIdx.x;
  uint4 v = make_uint4(0, 0, 0, 0);
  if (g < L16) {
#pragma unroll 4
    for (int s = threadIdx.y; s < S; s += 128) {
      const uint4 w = m[(size_t)s * L16 + g];
      v.x = __vmaxu4(v.x, w.x); v.y = __vmaxu4(v.y, w.y); v.z = __vmaxu4(v.z, w.z); v.w = __vmaxu4(v.w, w.w);
    }
  }
  red[threadIdx.y][threadIdx.x] = v;
  __syncthreads();
  for (int st = 64; st > 0; st >>= 1) {
    if (threadIdx.y < st) {
      const uint4 w = red[threadIdx.y + st][threadIdx.x];
      uint4 u = red[threadIdx.y][threadIdx.x];
      u.x = __vmaxu4(u.x, w.x); u.y = __vmaxu4(u.y, w.y); u.z = __vmaxu4(u.z, w.z); u.w = __vmaxu4(u.w, w.w);
      red[threadIdx.y][threadIdx.x] = u;
    }
    __syncthreads();
  }
  if (threadIdx.y == 0 && g < L16) live[g] = red[0][threadIdx.x];
}

torch::Tensor live_tokens(torch::Tensor mask) {
  TORCH_CHECK(mask.is_cuda() && mask.scalar_type() == at::kByte && mask.is_contiguous() && mask.dim() == 2, "live_tokens: a contiguous uint8 [S, L] CUDA mask");
  const int S = (int)mask.size(0), L = (int)mask.size(1);
  const at::cuda::CUDAGuard guard(mask.device());
  torch::Tensor live = torch::empty({L}, mask.options());
  if (L % 16 == 0 && reinterpret_cast<uintptr_t>(mask.data_ptr()) % 16 == 0) {
    live_tokens_vec_k<<<(L / 16 + 7) / 8, dim3(8, 128), 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const uint4*>(mask.data_ptr()), reinterpret_cast<uint4*>(live.data_ptr()), S, L / 16);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return live;
  }
  live_tokens_k<<<(L + 31) / 32, dim3(32, 32), 0, at::cuda::getCurrentCUDAStream()>>>(mask.data_ptr<uint8_t>(), live.data_ptr<uint8_t>(), S, L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return live;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ogemm_skip", &ogemm_skip);
#ifndef MWE_SM120
  m.def("ogemm_skip_batched", &ogemm_skip_batched);
#endif
  m.def("live_tokens", &live_tokens);
}
