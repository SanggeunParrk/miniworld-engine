// Unit test of the sm100 toolkit: C[128, N] = A[128, K] * B^T, B either K-major [N][K] or MN-major [K][N],
// optionally with A routed through TMEM (tcgen05.mma A-from-TMEM form).  One CTA, 128 threads.
#include <torch/extension.h>
#include "sm100.cuh"
using namespace sm100;

template <int N, int BMN, int ATM>
__global__ void __launch_bounds__(128) gemm_test(int K, const __grid_constant__ CUtensorMap amap, const __grid_constant__ CUtensorMap bmap, float* C) {
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sA = reinterpret_cast<__nv_bfloat16*>(smb);            // [128][64] SW128
  __nv_bfloat16* sB = sA + 128 * 64;                                    // K-major: [N][64]; MN-major: [N/64][64 k][64 n]
  __shared__ uint64_t bar_ld, bar_mma;
  __shared__ uint32_t tslot;
  const int tid = threadIdx.x, warp = tid >> 5;
  if (tid == 0) { bar_init(&bar_ld, 1); bar_init(&bar_mma, 1); bar_init_fence(); }
  if (warp == 0) tmem_alloc(&tslot, ATM ? 512 : (N < 32 ? 32 : N));
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tm = tslot;
  const uint32_t tA = tm + 256;                                         // A copy in TMEM (ATM): 64 bf16 = 32 columns
  constexpr uint32_t ID = idesc_bf16(128, N, 0, BMN);
  for (int k0 = 0, it = 0; k0 < K; k0 += 64, ++it) {
    if (tid == 0) {
      expect_tx(&bar_ld, (128 * 64 + N * 64) * 2);
      load_2d(&amap, sA, &bar_ld, k0, 0);
      if (BMN) { for (int nb = 0; nb < N / 64; ++nb) load_2d(&bmap, sB + nb * 64 * 64, &bar_ld, nb * 64, k0); }
      else load_2d(&bmap, sB, &bar_ld, k0, 0);
    }
    wait(&bar_ld, it & 1);
    if (ATM) {                                                          // A tile -> TMEM: thread = row, 32 columns of packed bf16 pairs
      uint32_t r[32];
      for (int c = 0; c < 32; ++c) {
        const int col = 2 * c;
        r[c] = *reinterpret_cast<const uint32_t*>(sA + sw128(tid, col));
      }
      for (int q = 0; q < 4; ++q) tmem_st8(tmem_at(tA, warp * 32, q * 8), r + q * 8);
      tmem_wait_st();
      tc_fence_before();
      __syncthreads();
      tc_fence_after();
    }
    if (tid == 0) {
      tc_fence_after();
      for (int ks = 0; ks < 4; ++ks) {
        const uint64_t db = BMN ? desc_mn128(sB + ks * 16 * 64, 64 * 64 * 2) : desc_k128(sB + ks * 16);
        if (ATM) mma_ts(tm, tA + ks * 8, db, ID, (it | ks) ? 1u : 0u);
        else mma_ss(tm, desc_k128(sA + ks * 16), db, ID, (it | ks) ? 1u : 0u);
      }
      mma_commit(&bar_mma);
    }
    wait(&bar_mma, it & 1);
    tc_fence_after();
  }
  // TMEM -> C
  for (int n0 = 0; n0 < N; n0 += 16) {
    float v[16];
    tmem_ld16(tmem_at(tm, warp * 32, n0), v);
    tmem_wait_ld();
    for (int q = 0; q < 16; ++q) C[(size_t)tid * N + n0 + q] = v[q];
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 0) tmem_dealloc(tm, ATM ? 512 : (N < 32 ? 32 : N));
}

torch::Tensor run(torch::Tensor A, torch::Tensor B, bool b_mn, bool a_tmem) {
  const int K = (int)A.size(1);
  const int N = b_mn ? (int)B.size(1) : (int)B.size(0);
  auto C = torch::empty({128, N}, A.options().dtype(torch::kFloat32));
  CUtensorMap am = make_map<2>(A.data_ptr(), {(uint64_t)K, 128}, {(uint64_t)K}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B, "A");
  CUtensorMap bm = b_mn ? make_map<2>(B.data_ptr(), {(uint64_t)N, (uint64_t)K}, {(uint64_t)N}, {64, 64}, CU_TENSOR_MAP_SWIZZLE_128B, "Bmn")
                        : make_map<2>(B.data_ptr(), {(uint64_t)K, (uint64_t)N}, {(uint64_t)K}, {64, (uint32_t)N}, CU_TENSOR_MAP_SWIZZLE_128B, "Bk");
  const int smem = 1024 + (128 * 64 + N * 64) * 2;
  auto st = at::cuda::getCurrentCUDAStream();
#define L(NN, BM, AT)                                                                                   \
  if (N == NN && b_mn == BM && a_tmem == AT) {                                                         \
    cudaFuncSetAttribute(gemm_test<NN, BM, AT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);    \
    gemm_test<NN, BM, AT><<<1, 128, smem, st>>>(K, am, bm, C.data_ptr<float>());                        \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                     \
    return C;                                                                                           \
  }
  L(64, 0, 0) L(128, 0, 0) L(256, 0, 0) L(64, 1, 0) L(128, 1, 0) L(128, 0, 1) L(64, 1, 1) L(32, 0, 0) L(32, 0, 1)
  TORCH_CHECK(false, "unsupported test case");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
