// adaln_launch.cuh -- launch helper of the persistent atom-width kernels (host side; included by the extensions' ops.cu).
#pragma once
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>

namespace adl {

// a persistent kernel over tiles of 16 rows: one CTA of G::NW warps per G::MINB per SM at most, tiles dealt round-robin to the warps (the kernel loops over them)
template <class G, typename K, typename Pm>
void launch_persistent(K kernel, const Pm& p, int64_t ntile, cudaStream_t st, int minb = -1) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int per_sm = minb > 0 ? minb : G::MINB;
  const unsigned grid = (unsigned)std::max<int64_t>(1, std::min<int64_t>((ntile + G::NW - 1) / G::NW, (int64_t)sms * per_sm));
  kernel<<<grid, G::NTHR, G::SMEM, st>>>(p);
}

}  // namespace adl
