// tcgen05.mma per-instruction cost: B layout (K-major SW128 / SW64, MN-major SW64 like a V tile), A in smem or TMEM, with or
// without 4 warps hammering tcgen05.ld/st on other TMEM columns.
#include <torch/extension.h>
#include "sm100.cuh"
using namespace sm100;
__global__ void mb(int n, int N, int ts, int lay, int noise, long long* out) {
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* A = reinterpret_cast<__nv_bfloat16*>(smb);        // 16 KiB
  __nv_bfloat16* B = A + 128 * 64;                                  // 32 KiB
  uint64_t* bar = reinterpret_cast<uint64_t*>(B + 256 * 64);
  uint32_t* slot = reinterpret_cast<uint32_t*>(bar + 1);
  volatile int* stop = reinterpret_cast<volatile int*>(slot + 1);
  for (int i = threadIdx.x; i < (128 + 256) * 64; i += blockDim.x) A[i] = __float2bfloat16(0.f);
  if (threadIdx.x == 0) { bar_init(bar, 1); bar_init_fence(); *stop = 0; }
  fence_proxy_async();
  if (threadIdx.x < 32) tmem_alloc(slot, 512);
  tc_fence_before(); __syncthreads(); tc_fence_after();
  const uint32_t tmem = *slot;
  const int warp = threadIdx.x >> 5;
#if MB_TM
  if (warp < 4 && noise >= 0) {                                   // zero all of TMEM (garbage operands may run at a different speed)
    uint32_t z[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int c = 0; c < 512; c += 8) tmem_st8(tmem_at(tmem, warp * 32, c), z);
    tmem_wait_st();
  }
#endif
  tc_fence_before(); __syncthreads(); tc_fence_after();
  if (warp == 0) {
    const uint32_t id = idesc_bf16(128, N, 0, lay == 2 || lay >= 4);
    for (int rep = 0; rep < 2; ++rep) {
      long long t0 = clock64();
      if (elect_one()) {
        for (int i = 0; i < n; ++i) {
          uint64_t bd;
          if (lay == 0) bd = desc_k128(B + (i & 1) * 16);
          else if (lay == 1) bd = desc_k64(B + (i & 1) * 16);
          else if (lay == 2) bd = sdesc(sa(B + (i & 3) * 16 * 32), 4096, 512, 4);        // MN-major SW64 [k][32]
          else if (lay == 3) bd = desc_k128(B + (i & 3) * 16);                          // K-major SW128 [n][64 k], 4 k-steps
          else if (lay == 4) bd = desc_mn128(B + (i & 3) * 16 * 64, 8192);              // MN-major SW128 [k][64]
          else bd = sdesc(sa(B + (i & 3) * 16 * 32), 4096, 512, 6);                     // MN-major SW32 (garbage layout, cost only)
          if (ts) mma_ts(tmem, tmem + 256 + (i & 3) * 8, bd, id, 1u);
          else mma_ss(tmem, lay == 0 ? desc_k128(A + (i & 1) * 16) : desc_k64(A + (i & 1) * 16), bd, id, 1u);
        }
        mma_commit(bar);
      }
      __syncwarp();
      wait(bar, rep & 1);
      long long t1 = clock64();
      if (threadIdx.x == 0 && rep == 1) out[blockIdx.x] = t1 - t0;
    }
    if (threadIdx.x == 0) *stop = 1;
  }
#if MB_TM
  else if (warp >= 4 && noise > 0) {                // warps 4-7: tcgen05.ld 64 cols + st 32 cols in a loop on columns 384..511
    const uint32_t tb = tmem_at(tmem + 384, (warp & 3) * 32, 0);
    float v[32]; uint32_t w[16];
    for (int i = 0; i < 16; ++i) w[i] = i;
    while (!*stop) {
      tmem_ld32(tb, v); tmem_ld32(tb + 32, v); tmem_wait_ld();
      for (int i = 0; i < 16; ++i) w[i] += __float_as_uint(v[i]);
      tmem_st8(tb + 64, w); tmem_st8(tb + 72, w + 8); tmem_wait_st();
    }
    if (w[0] == 12345) out[0] = 0;
  }
#endif
  tc_fence_before(); __syncthreads();
  if (threadIdx.x < 32) tmem_dealloc(tmem, 512);
}
torch::Tensor run(int n, int N, int ts, int lay, int noise, int ctas, int threads) {
  auto out = torch::zeros({ctas}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  const int sm = 1024 + (128 + 256) * 64 * 2 + 64;
  cudaFuncSetAttribute(mb, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
  mb<<<ctas, threads, sm>>>(n, N, ts, lay, noise, reinterpret_cast<long long*>(out.data_ptr<int64_t>()));
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
