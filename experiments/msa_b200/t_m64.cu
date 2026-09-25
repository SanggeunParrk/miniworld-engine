// where does an M = 64 (cta_group::1) accumulator land in TMEM?  D[r][n] = r * 1000 + n via A = one-hot rows scaled... simpler:
// A[r][k] = (k == 0) ? r : 0, B[n][k] = (k == 0) ? 1 : 0  ->  D[r][n] = r for every n.  Read all 128 lanes x 8 columns.
#include <torch/extension.h>
#include "sm100.cuh"
using namespace sm100;
__global__ void k(float* out, int n_off) {
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* A = reinterpret_cast<__nv_bfloat16*>(smb);   // [64][64] K-major SW128
  __nv_bfloat16* B = A + 64 * 64;                              // [64 n][64 k]
  __shared__ uint64_t bar; __shared__ uint32_t slot;
  const int t = threadIdx.x;
  for (int v = t; v < 128 * 64; v += 128) {
    const int r = (v / 64) % 64, kk = v % 64; const bool isA = v < 64 * 64;
    const float val = kk == 0 ? (isA ? (float)(r + 1) : 1.f) : 0.f;
    (isA ? A : B)[sw128(r, kk)] = __float2bfloat16(val);
  }
  if (t == 0) { bar_init(&bar, 1); bar_init_fence(); }
  if (t < 32) tmem_alloc(&slot, 128);
  fence_proxy_async(); tc_fence_before(); __syncthreads(); tc_fence_after();
  const uint32_t tm = slot;
  if (t < 32) {
    // zero-fill all lanes first with an M=128 MMA of zeros? instead: store zeros via tcgen05.st
  }
  { uint32_t z[8] = {0,0,0,0,0,0,0,0}; tmem_st8(tmem_at(tm, (t >> 5) * 32, 0), z); tmem_wait_st(); }
  tc_fence_before(); __syncthreads(); tc_fence_after();
  if (t < 32) { if (elect_one()) { mma_ss(tm + ((uint32_t)n_off << 16), desc_k128(A), desc_k128(B), idesc_bf16(64, 64), 0u); mma_commit(&bar); } __syncwarp(); }
  wait(&bar, 0); tc_fence_after();
  float v[8]; tmem_ld8(tmem_at(tm, (t >> 5) * 32, 0), v); tmem_wait_ld();
  for (int q = 0; q < 8; ++q) out[t * 8 + q] = v[q];
  tc_fence_before(); __syncthreads();
  if (t < 32) tmem_dealloc(tm, 128);
}
torch::Tensor run(int64_t n_off) {
  auto o = torch::zeros({128, 8}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 40000);
  k<<<1, 128, 40000, at::cuda::getCurrentCUDAStream()>>>(o.data_ptr<float>(), (int)n_off);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return o;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
