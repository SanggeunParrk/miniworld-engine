// Milestone 0: minimal hand-written Hopper SM90 TF32 WGMMA GEMM in cute.
//   C(M,N) = A(M,K) @ B(N,K)^T   (TN: both operands K-major/contiguous, = x @ W^T form)
// Single warpgroup (128 threads), one CTA tile = (BLOCK_M, N, K) all loaded at once
// (K=128, N=128 tiny), cp.async gmem->smem, cute::gemm WGMMA, no pipeline.
// fp32 in/out, tfloat32 operands (smem typed tfloat32_t), fp32 accumulate.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cutlass/numeric_types.h>

using namespace cute;

template <int BLOCK_M, int N_, int K_>
__global__ static __launch_bounds__(128)
void m0_kernel(const float* __restrict__ A, const float* __restrict__ B,
               float* __restrict__ C, int M) {
  using TA = tfloat32_t; using TB = tfloat32_t; using TC = float;
  constexpr int N = N_, K = K_;

  // gmem tensors (row-major). A:(M,K) stride(K,1). B:(N,K) stride(K,1). C:(M,N) stride(N,1).
  Tensor mA = make_tensor(make_gmem_ptr(reinterpret_cast<const TA*>(A)),
                          make_shape(M, Int<K>{}), make_stride(Int<K>{}, Int<1>{}));
  Tensor mB = make_tensor(make_gmem_ptr(reinterpret_cast<const TB*>(B)),
                          make_shape(Int<N>{}, Int<K>{}), make_stride(Int<K>{}, Int<1>{}));
  Tensor mC = make_tensor(make_gmem_ptr(C),
                          make_shape(M, Int<N>{}), make_stride(Int<N>{}, Int<1>{}));

  // CTA tiles
  Tensor gA = local_tile(mA, make_shape(Int<BLOCK_M>{}, Int<K>{}), make_coord(blockIdx.x, 0)); // (BM,K)
  Tensor gB = local_tile(mB, make_shape(Int<N>{}, Int<K>{}), make_coord(0, 0));                // (N,K)
  Tensor gC = local_tile(mC, make_shape(Int<BLOCK_M>{}, Int<N>{}), make_coord(blockIdx.x, 0)); // (BM,N)

  // smem layouts: K-major swizzled atoms (operands are K-contiguous = TN).
  auto sA_layout = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(Int<BLOCK_M>{}, Int<K>{}));
  auto sB_layout = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(Int<N>{}, Int<K>{}));
  extern __shared__ char smem_raw[];                          // dynamic smem (>48KB needs this)
  TA* smemA = reinterpret_cast<TA*>(smem_raw);
  TB* smemB = reinterpret_cast<TB*>(smem_raw + cosize_v<decltype(sA_layout)> * sizeof(TA));
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

  // cp.async gmem->smem: simple thread-linear copy (convert fp32->tf32 on store).
  auto copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint32_t>, TA>{},
                               Layout<Shape<_16,_8>, Stride<_8,_1>>{},   // 128 thr
                               Layout<Shape<_1,_1>>{});
  auto copyB = copyA;
  auto thrA = copyA.get_slice(threadIdx.x);
  auto thrB = copyB.get_slice(threadIdx.x);
  // partition (note: cp.async needs matching gmem/smem partition; gA/gB are TA-typed views)
  copy(thrA.partition_S(gA), thrA.partition_D(sA));
  copy(thrB.partition_S(gB), thrB.partition_D(sB));
  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  // TiledMMA: one warpgroup (128 thr), single SM90 TF32 SS atom (M=64, N=128, K=8).
  TiledMMA mma = make_tiled_mma(SM90_64x128x8_F32TF32TF32_SS_TN<>{});
  auto thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  warpgroup_fence_operand(tCrC);
  warpgroup_arrive();
  cute::gemm(mma, tCrA, tCrB, tCrC);   // SS: fragments are smem descriptors
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(tCrC);

  copy(tCrC, tCgC);
}

torch::Tensor m0_gemm(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.dtype() == torch::kFloat32, "fp32 cuda");
  A = A.contiguous(); B = B.contiguous();
  int M = A.size(0), K = A.size(1), N = B.size(0);
  TORCH_CHECK(K == 128 && N == 128, "M0 fixed N=K=128");
  auto C = torch::empty({M, N}, A.options());
  constexpr int BM = 64;
  // dynamic smem: sA(BM,128) + sB(128,128) tfloat32 = (64+128)*128*4 = 98304 B
  int smem_bytes = (BM * 128 + 128 * 128) * (int)sizeof(tfloat32_t);
  auto kern = m0_kernel<BM, 128, 128>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  dim3 grid((M + BM - 1) / BM);
  kern<<<grid, 128, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M);
  C10_CUDA_CHECK(cudaGetLastError());
  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("m0_gemm", &m0_gemm, "C = A@B^T sm90 tf32 wgmma (M0)");
}
