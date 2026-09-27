// The KV backward's per-stage MMA stream in isolation: S^T, dP^T (ts, N = 64, B = [64][32] K-major SW64) then dV / dK (ts, N = 32,
// B = [64][32] as MN-major SW64 or a transposed [32][64] K-major SW128 tile), repeated; clk per stage.
#include <torch/extension.h>
#include "sm100.cuh"
using namespace sm100;
template <int BLAY, int ONLY, int COMP, int CMT, int RND, int FEN = 0>   // BLAY: 0 MN-SW64 (the kernel), 1 K-major SW128 [32 d][64 q]; ONLY: 0 all, 1 S/dP only, 2 grad only
__global__ void __launch_bounds__(384, 1) mb(int stages, long long* out, float* sink) {
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* Q = reinterpret_cast<__nv_bfloat16*>(smb);   // [2 slots][Q | dO] 4 KiB each
  uint64_t* bar = reinterpret_cast<uint64_t*>(Q + 4 * 2048);   // [0] final, [1..4] per-stage dummies
  uint32_t* slot = reinterpret_cast<uint32_t*>(bar + 5);   // [0] tmem, [2] stop
  for (int i = threadIdx.x; i < 4 * 2048; i += blockDim.x) {   // RND: random-looking operands instead of zeros
    const uint32_t h = (i * 2654435761u) ^ (blockIdx.x * 97u);
    Q[i] = __float2bfloat16(RND ? ((int)(h >> 8) % 2001 - 1000) * 1e-3f : 0.f);
  }
  if (threadIdx.x == 0) { for (int i = 0; i < 5; ++i) bar_init(bar + i, 1); bar_init_fence(); }
  fence_proxy_async();
  if (threadIdx.x < 32) tmem_alloc(slot, 512);
  tc_fence_before(); __syncthreads(); tc_fence_after();
  const uint32_t tmem = *slot;
  {
    uint32_t z[8];
    for (int i = 0; i < 8; ++i) z[i] = RND ? (0x3c003c00u ^ ((threadIdx.x * 131u + i * 7u) & 0x00ff00ffu)) : 0u;
    if (threadIdx.x < 128) for (int c = 0; c < 512; c += 8) tmem_st8(tmem_at(tmem, (threadIdx.x >> 5) * 32, c), z);
    tmem_wait_st();
  }
  tc_fence_before(); __syncthreads(); tc_fence_after();
  volatile int* stop = reinterpret_cast<volatile int*>(slot + 2);
  if (threadIdx.x == 0) *stop = 0;
  __syncthreads();
  const int warp = threadIdx.x >> 5;
  if (warp > 0 && (warp & 3) == 0 && COMP) {     // warps 4, 8: MUFU + FMA streams on the MMA warp's SMSP (COMP 2: all SMSPs busy too)
    float x0 = threadIdx.x * 1e-3f, x1 = x0 + 1.f, x2 = x0 + 2.f, x3 = x0 + 3.f;
    while (!*stop) {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        float y0, y1, y2, y3;
        asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y0) : "f"(x0)); asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y1) : "f"(x1));
        asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y2) : "f"(x2)); asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y3) : "f"(x3));
        x0 = fmaf(y0, 0.5f, x1); x1 = fmaf(y1, 0.5f, x2); x2 = fmaf(y2, 0.5f, x3); x3 = fmaf(y3, 0.5f, x0);
      }
    }
    if (x0 == 1.2345f) sink[0] = x1;
  }
  if (threadIdx.x < 32) {
    constexpr uint32_t ID_ST = idesc_bf16(128, 64, 0, 0), ID_G = idesc_bf16(128, 32, 0, BLAY == 0);
    for (int rep = 0; rep < 2; ++rep) {
      long long t0 = clock64();
      if (elect_one()) {
        for (int y = 0; y < stages; ++y) {
          const int b = y & 1, r = (y >> 1) & 1;
          const __nv_bfloat16* q = Q + b * 4096;
          if (FEN) tc_fence_after();
          if (ONLY != 2) {
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) mma_ts(tmem + b * 128, tmem + 384 + r * 16 + ks * 8, desc_k64(q + ks * 16), ID_ST, ks);
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) mma_ts(tmem + b * 128 + 64, tmem + 416 + r * 16 + ks * 8, desc_k64(q + 2048 + ks * 16), ID_ST, ks);
          }
          if (CMT) mma_commit(bar + 1 + b);
          if (FEN) tc_fence_after();
          if (ONLY != 1) {
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
              const uint32_t a = tmem + b * 128 + (ks >> 1) * 32 + (ks & 1) * 8;
              if (BLAY == 0) {
                mma_ts(tmem + 256 + r * 64, a, sdesc(sa(q + 2048 + ks * 16 * 32), 4096, 512, 4), ID_G, 1u);
                mma_ts(tmem + 256 + r * 64 + 32, a + 64, sdesc(sa(q + ks * 16 * 32), 4096, 512, 4), ID_G, 1u);
              } else {
                mma_ts(tmem + 256 + r * 64, a, desc_k128(q + 2048 + ks * 16), ID_G, 1u);
                mma_ts(tmem + 256 + r * 64 + 32, a + 64, desc_k128(q + ks * 16), ID_G, 1u);
              }
            }
            if (CMT) mma_commit(bar + 3 + b);
          }
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
  tc_fence_before(); __syncthreads();
  if (threadIdx.x < 32) tmem_dealloc(tmem, 512);
}
torch::Tensor run(int comp, int only, int stages) {
  auto out = torch::zeros({148}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto sink = torch::zeros({1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  const int sm = 1024 + 4 * 2048 * 2 + 64;
  auto k = comp == 6 ? mb<0, 0, 0, 1, 1, 1> : comp == 3 ? mb<0, 0, 0, 1, 1> : comp == 4 ? mb<0, 2, 0, 1, 1> : comp == 5 ? mb<0, 1, 0, 1, 1> : comp == 2 ? mb<0, 0, 0, 1, 0> : mb<0, 0, 0, 0, 0>;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
  k<<<148, 384, sm>>>(stages, reinterpret_cast<long long*>(out.data_ptr<int64_t>()), sink.data_ptr<float>());
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
