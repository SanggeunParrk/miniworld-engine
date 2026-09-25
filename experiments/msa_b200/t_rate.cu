// tcgen05 issue-rate probe: one CTA per SM issues `iters` x 4 MMAs (K=16 each) from fixed smem tiles; cycles per MMA.
#include <torch/extension.h>
#include "sm100.cuh"
using namespace sm100;
__global__ void __launch_bounds__(128) rate(int iters, int mode, int N, int nacc, unsigned long long* out) {
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* A = reinterpret_cast<__nv_bfloat16*>(smb);            // 128 x 64 (16 KiB)
  __nv_bfloat16* B = A + 128 * 64;                                      // up to 256 x 64 (32 KiB)
  __shared__ uint64_t bar; __shared__ uint32_t slot;
  for (int v = threadIdx.x; v < (128 + 256) * 64 / 8; v += 128) reinterpret_cast<uint4*>(A)[v] = make_uint4(0x3f803f80u, 0, 0x3f803f80u, 0);
  if (threadIdx.x == 0) { bar_init(&bar, 1); bar_init_fence(); }
  if (threadIdx.x < 32) tmem_alloc(&slot, 512);
  fence_proxy_async(); tc_fence_before(); __syncthreads(); tc_fence_after();
  const uint32_t tm = slot;
  if (threadIdx.x < 32) {
    // warp-wide issue (the MMA is issued by one elected lane; the whole warp walks the loop, as CUTLASS's MMA warp does),
    // descriptors precomputed, nacc independent accumulators unrolled -- nothing but UTCHMMA in the steady state
    const int M = (mode == 4) ? 64 : 128;
    const uint32_t id = idesc_bf16(M, N, 0, (mode == 1 || mode == 2) ? 1 : 0);
    uint64_t da[4], db[4];
    for (int ks = 0; ks < 4; ++ks) {
      da[ks] = desc_k128(A + ks * 16);
      db[ks] = mode == 1 ? desc_mn128(B + ks * 16 * 64, 64 * 64 * 2) : mode == 2 ? sdesc(sa(B + ks * 16 * 32), 64 * 32 * 2, 512, 4) : desc_k128(B + ks * 16);
    }
    const uint32_t stride = N > 128 ? 256 : 128;
    long long c0 = clock64();
    for (int it = 0; it < iters; ++it) {
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        if (elect_one()) {
          if (nacc == 1) mma_ss(tm, da[ks], db[ks], id, 1u);
          else if (nacc == 11) { mma_ss(tm, da[ks], db[ks], id, 1u); mma_ss(tm, da[ks], db[ks], id, 1u); mma_ss(tm, da[ks], db[ks], id, 1u); mma_ss(tm, da[ks], db[ks], id, 1u); }
          else if (nacc == 2) { mma_ss(tm, da[ks], db[ks], id, 1u); mma_ss(tm + stride, da[ks], db[ks], id, 1u); }
          else { mma_ss(tm, da[ks], db[ks], id, 1u); mma_ss(tm + stride, da[ks], db[ks], id, 1u);
                 mma_ss(tm + 2 * stride, da[ks], db[ks], id, 1u); mma_ss(tm + 3 * stride, da[ks], db[ks], id, 1u); }
        }
        __syncwarp();
      }
    }
    if (elect_one()) mma_commit(&bar);
    __syncwarp();
    wait(&bar, 0);
    if (threadIdx.x == 0) out[blockIdx.x] = (clock64() - c0);
  }
  tc_fence_before(); __syncthreads();
  if (threadIdx.x < 32) tmem_dealloc(tm, 512);
}
double run(int64_t iters, int64_t mode, int64_t N, int64_t ctas, int64_t nacc) {
  auto o = torch::zeros({ctas}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64));
  const int smem = 1024 + (128 + 256) * 64 * 2;
  cudaFuncSetAttribute(rate, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  rate<<<ctas, 128, smem, at::cuda::getCurrentCUDAStream()>>>((int)iters, (int)mode, (int)N, (int)nacc, reinterpret_cast<unsigned long long*>(o.data_ptr<int64_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return o.to(torch::kFloat64).mean().item<double>() / (iters * 4 * (nacc == 11 ? 4 : nacc));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
