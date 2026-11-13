// The RTX 5090's (sm_120a) batched GEMMs of the MSA pair-weighted averaging: the key contraction o = w v, its transpose dv = w^T do and the
// weight-gradient product dw = do v^T, on CUTLASS 2.x sm_80 tensor-op GEMMs (mma.sync + cp.async multistage, every tile under the 99 KB of
// shared memory).  cuBLAS's heuristic on this card picks a 32 x 32 WMMA kernel for the batched shapes with M = L in 128..640 (80 TFLOP/s, half
// of what it reaches from L 768 on), so these replace it there and everywhere else on the 5090 (one GEMM per product, the same at every L).
//
// The head-major tensors are [8 ns, L, K] (the MSA rows of a head in ns chunks); w [8, L, L] is shared by the ns chunks of its head: batch b
// reads the A matrix at (b / a_div) * stride_A (a_div = ns for w, 1 for a per-batch A).  fp32 accumulation, no split-K: bit-reproducible.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_batched.h"

namespace {
using bf = cutlass::bfloat16_t;
using RowM = cutlass::layout::RowMajor;
using ColM = cutlass::layout::ColumnMajor;

template <class LA, class LB, class EC, int TM, int TN, int TK, int WM, int WN, int ST>
struct Cfg {
  static constexpr int kAlignC = 128 / cutlass::sizeof_bits<EC>::value;
  using Device = cutlass::gemm::device::GemmBatched<
      bf, LA, bf, LB, EC, RowM, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<TM, TN, TK>,
      cutlass::gemm::GemmShape<WM, WN, TK>, cutlass::gemm::GemmShape<16, 8, 16>, cutlass::epilogue::thread::LinearCombination<EC, kAlignC, float, float>,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, ST, 8, 8>;
  using GK = typename Device::GemmKernel;
  static_assert(sizeof(typename GK::SharedStorage) <= 101376, "the RTX 5090's 99 KB of shared memory per block");
};

// GemmBatched's body (cutlass/gemm/kernel/gemm_batched.h) with the A matrix of batch b at (b / a_div) * stride_A: w shared by a head's chunks
template <class GK>
__global__ void __launch_bounds__(GK::kThreadCount) bgemm_kernel(typename GK::Params params, int a_div) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto& shared = *reinterpret_cast<typename GK::SharedStorage*>(smem_raw);
  using Mma = typename GK::Mma;
  using Epilogue = typename GK::Epilogue;
  typename GK::ThreadblockSwizzle sw;
  cutlass::gemm::GemmCoord tile = sw.get_tile_offset(params.swizzle_log_tile);
  if (params.grid_tiled_shape.m() <= tile.m() || params.grid_tiled_shape.n() <= tile.n()) return;
  const int batch = blockIdx.z;
  const int thread_idx = threadIdx.x;
  typename Mma::IteratorA it_a(params.params_A, params.ref_A.data(), params.problem_size.mk(), thread_idx, {tile.m() * Mma::Shape::kM, 0});
  it_a.add_pointer_offset(params.stride_A * (batch / a_div));
  typename Mma::IteratorB it_b(params.params_B, params.ref_B.data(), params.problem_size.kn(), thread_idx, {0, tile.n() * Mma::Shape::kN});
  it_b.add_pointer_offset(params.stride_B * batch);
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int lane_idx = threadIdx.x % 32;
  Mma mma(shared.main_loop, thread_idx, warp_idx, lane_idx);
  typename Mma::FragmentC acc;
  acc.clear();
  mma(params.gemm_k_iterations, acc, it_a, it_b, acc);
  typename GK::OutputOp out_op(params.epilogue);
  const cutlass::MatrixCoord off(tile.m() * Mma::Shape::kM, tile.n() * Mma::Shape::kN);
  typename Epilogue::OutputTileIterator it_c(params.params_C, params.ref_C.data(), params.problem_size.mn(), thread_idx, off);
  it_c.add_pointer_offset(params.stride_C * batch);
  typename Epilogue::OutputTileIterator it_d(params.params_D, params.ref_D.data(), params.problem_size.mn(), thread_idx, off);
  it_d.add_pointer_offset(params.stride_D * batch);
  Epilogue epi(shared.epilogue, thread_idx, warp_idx, lane_idx);
  epi(out_op, it_d, acc, it_c);
}

template <class C>
void run(const void* a, int64_t lda, int64_t stride_a, int a_div, const void* b, int64_t ldb, int64_t stride_b, void* c, int64_t ldc, int64_t stride_c,
         int M, int N, int K, int batches, cudaStream_t stream) {
  using GK = typename C::GK;
  using EC = typename GK::Epilogue::OutputTileIterator::Element;
  using LA = typename GK::Mma::IteratorA::Layout;
  using LB = typename GK::Mma::IteratorB::Layout;
  const cutlass::gemm::GemmCoord problem(M, N, K);
  typename GK::ThreadblockSwizzle sw;
  const auto tiled = sw.get_tiled_shape(problem, {GK::Mma::Shape::kM, GK::Mma::Shape::kN, GK::Mma::Shape::kK}, batches);
  typename GK::Params p(problem, tiled, {reinterpret_cast<bf*>(const_cast<void*>(a)), LA(lda)}, stride_a,
                        {reinterpret_cast<bf*>(const_cast<void*>(b)), LB(ldb)}, stride_b, {reinterpret_cast<EC*>(c), RowM(ldc)}, stride_c,
                        {reinterpret_cast<EC*>(c), RowM(ldc)}, stride_c, {1.f, 0.f}, batches);
  const int smem = (int)sizeof(typename GK::SharedStorage);
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(bgemm_kernel<GK>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    configured = true;
  }
  bgemm_kernel<GK><<<sw.get_grid_shape(tiled), GK::kThreadCount, smem, stream>>>(p, a_div);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// tile configurations (TM, TN, TK, warp M, warp N, stages)
#define MWE_CFGS(X)                   \
  X(0, 128, 128, 32, 64, 64, 4)       \
  X(1, 128, 256, 32, 64, 64, 3)       \
  X(2, 256, 128, 32, 64, 64, 3)       \
  X(3, 128, 128, 64, 64, 64, 3)       \
  X(4, 64, 128, 64, 32, 64, 4)        \
  X(5, 64, 64, 64, 32, 32, 6)

template <class LA, class LB, class EC>
void dispatch(int cfg, const void* a, int64_t lda, int64_t sa, int ad, const void* b, int64_t ldb, int64_t sb, void* c, int64_t ldc, int64_t sc, int M, int N,
              int K, int batches, cudaStream_t s) {
  switch (cfg) {
#define X(id, tm, tn, tk, wm, wn, st) \
  case id: return run<Cfg<LA, LB, EC, tm, tn, tk, wm, wn, st>>(a, lda, sa, ad, b, ldb, sb, c, ldc, sc, M, N, K, batches, s);
    MWE_CFGS(X)
#undef X
  }
  TORCH_CHECK(false, "pwa gemm_sm120: unknown tile configuration ", cfg);
}

void check_bf(const torch::Tensor& t, const char* what) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16 && t.is_contiguous() && t.dim() == 3, "pwa gemm_sm120: ", what, " contiguous bf16 [B, R, C]");
}
}  // namespace

// o[b] = w[b / ns] v[b]: w [H, L, L], v [H ns, L, K] -> out [H ns, L, K] bf16 (transpose: w[b / ns]^T)
void contract(torch::Tensor w, torch::Tensor v, torch::Tensor out, bool transpose, int64_t cfg) {
  check_bf(w, "w");
  check_bf(v, "v");
  check_bf(out, "out");
  const int64_t H = w.size(0), L = w.size(1), B = v.size(0), K = v.size(2);
  TORCH_CHECK(w.size(2) == L && v.size(1) == L && B % H == 0 && out.sizes() == v.sizes() && L % 8 == 0 && K % 8 == 0, "pwa gemm_sm120: contract shapes");
  const at::cuda::CUDAGuard guard(w.device());
  auto s = at::cuda::getCurrentCUDAStream();
  const int ns = (int)(B / H);
  if (transpose)
    dispatch<ColM, RowM, bf>((int)cfg, w.data_ptr(), L, L * L, ns, v.data_ptr(), K, L * K, out.data_ptr(), K, L * K, (int)L, (int)K, (int)L, (int)B, s);
  else
    dispatch<RowM, RowM, bf>((int)cfg, w.data_ptr(), L, L * L, ns, v.data_ptr(), K, L * K, out.data_ptr(), K, L * K, (int)L, (int)K, (int)L, (int)B, s);
}

// dw[b] = d[b] v[b]^T: d, v [B, L, K] bf16 -> out [B, L, L] fp32
void weight_grad(torch::Tensor d, torch::Tensor v, torch::Tensor out, int64_t cfg) {
  check_bf(d, "d");
  check_bf(v, "v");
  TORCH_CHECK(d.sizes() == v.sizes(), "pwa gemm_sm120: d / v shapes");
  const int64_t B = d.size(0), L = d.size(1), K = d.size(2);
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kFloat && out.is_contiguous() && out.dim() == 3 && out.size(0) == B && out.size(1) == L && out.size(2) == L &&
                  L % 8 == 0 && K % 8 == 0,
              "pwa gemm_sm120: out fp32 [B, L, L]");
  const at::cuda::CUDAGuard guard(d.device());
  dispatch<RowM, ColM, float>((int)cfg, d.data_ptr(), K, L * K, 1, v.data_ptr(), K, L * K, out.data_ptr(), L, L * L, (int)L, (int)L, (int)K, (int)B,
                              at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("contract", &contract);
  m.def("weight_grad", &weight_grad);
}
