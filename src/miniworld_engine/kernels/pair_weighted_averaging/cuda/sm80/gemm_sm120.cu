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
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/permute.h"

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
constexpr int GROUP = 8;

template <class GK>
__global__ void __launch_bounds__(GK::kThreadCount) bgemm_kernel(typename GK::Params params, int a_div) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto& shared = *reinterpret_cast<typename GK::SharedStorage*>(smem_raw);
  using Mma = typename GK::Mma;
  using Epilogue = typename GK::Epilogue;
  // grouped raster: the CTAs of a wave walk GROUP rows of tiles column by column (the A rows and B columns they share stay in L2)
  const int tm = params.grid_tiled_shape.m(), tn = params.grid_tiled_shape.n();
  const int id = blockIdx.x + blockIdx.y * gridDim.x;
  const int per = GROUP * tn, first = (id / per) * GROUP, gsz = min(tm - first, GROUP);
  const cutlass::gemm::GemmCoord tile(first + (id % per) % gsz, (id % per) / gsz, 0);
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
void run(const void* a, int64_t lda, int64_t stride_a, int a_div, const void* b, int64_t ldb, int64_t stride_b, void* c, int64_t ldc, int64_t stride_c, const void* cin,
         int M, int N, int K, int batches, cudaStream_t stream) {
  using GK = typename C::GK;
  using EC = typename GK::Epilogue::OutputTileIterator::Element;
  using LA = typename GK::Mma::IteratorA::Layout;
  using LB = typename GK::Mma::IteratorB::Layout;
  const cutlass::gemm::GemmCoord problem(M, N, K);
  typename GK::ThreadblockSwizzle sw;
  const auto tiled = sw.get_tiled_shape(problem, {GK::Mma::Shape::kM, GK::Mma::Shape::kN, GK::Mma::Shape::kK}, batches);
  typename GK::Params p(problem, tiled, {reinterpret_cast<bf*>(const_cast<void*>(a)), LA(lda)}, stride_a,
                        {reinterpret_cast<bf*>(const_cast<void*>(b)), LB(ldb)}, stride_b, {reinterpret_cast<EC*>(cin ? const_cast<void*>(cin) : c), RowM(ldc)}, stride_c,
                        {reinterpret_cast<EC*>(c), RowM(ldc)}, stride_c, {1.f, cin ? 1.f : 0.f}, batches);
  const int smem = (int)sizeof(typename GK::SharedStorage);
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(bgemm_kernel<GK>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    configured = true;
  }
  bgemm_kernel<GK><<<dim3(tiled.m(), tiled.n(), batches), GK::kThreadCount, smem, stream>>>(p, a_div);
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
void dispatch(int cfg, const void* a, int64_t lda, int64_t sa, int ad, const void* b, int64_t ldb, int64_t sb, void* c, int64_t ldc, int64_t sc, const void* cin, int M, int N,
              int K, int batches, cudaStream_t s) {
  switch (cfg) {
#define X(id, tm, tn, tk, wm, wn, st) \
  case id: return run<Cfg<LA, LB, EC, tm, tn, tk, wm, wn, st>>(a, lda, sa, ad, b, ldb, sb, c, ldc, sc, cin, M, N, K, batches, s);
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
    dispatch<ColM, RowM, bf>((int)cfg, w.data_ptr(), L, L * L, ns, v.data_ptr(), K, L * K, out.data_ptr(), K, L * K, nullptr, (int)L, (int)K, (int)L, (int)B, s);
  else
    dispatch<RowM, RowM, bf>((int)cfg, w.data_ptr(), L, L * L, ns, v.data_ptr(), K, L * K, out.data_ptr(), K, L * K, nullptr, (int)L, (int)K, (int)L, (int)B, s);
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
  dispatch<RowM, ColM, float>((int)cfg, d.data_ptr(), K, L * K, 1, v.data_ptr(), K, L * K, out.data_ptr(), L, L * L, nullptr, (int)L, (int)L, (int)K, (int)B,
                              at::cuda::getCurrentCUDAStream());
}

// out = op(a) op(b) (+ c) (one GEMM, row-major bf16): a [M, K] (trans_a: [K, M]), b [K, N] (trans_b: [N, K]), c [M, N] or empty -> out [M, N] bf16
void mm(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor out, bool trans_a, bool trans_b, int64_t cfg) {
  for (auto* t : {&a, &b, &out})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kBFloat16 && t->is_contiguous() && t->dim() == 2, "pwa gemm_sm120 mm: contiguous bf16 2-D");
  const int64_t M = trans_a ? a.size(1) : a.size(0), K = trans_a ? a.size(0) : a.size(1), N = trans_b ? b.size(0) : b.size(1);
  TORCH_CHECK((trans_b ? b.size(1) : b.size(0)) == K && out.size(0) == M && out.size(1) == N && M % 8 == 0 && N % 8 == 0 && K % 8 == 0, "pwa gemm_sm120 mm: shapes");
  const at::cuda::CUDAGuard guard(a.device());
  auto s = at::cuda::getCurrentCUDAStream();
  const int64_t lda = a.size(1), ldb = b.size(1);
  TORCH_CHECK(c.numel() == 0 || (c.scalar_type() == at::kBFloat16 && c.is_contiguous() && c.sizes() == out.sizes()), "pwa gemm_sm120 mm: c bf16 [M, N] or empty");
  const void* cin = c.numel() ? c.data_ptr() : nullptr;
  if (trans_a && trans_b)
    dispatch<ColM, ColM, bf>((int)cfg, a.data_ptr(), lda, 0, 1, b.data_ptr(), ldb, 0, out.data_ptr(), N, 0, cin, (int)M, (int)N, (int)K, 1, s);
  else if (trans_a)
    dispatch<ColM, RowM, bf>((int)cfg, a.data_ptr(), lda, 0, 1, b.data_ptr(), ldb, 0, out.data_ptr(), N, 0, cin, (int)M, (int)N, (int)K, 1, s);
  else if (trans_b)
    dispatch<RowM, ColM, bf>((int)cfg, a.data_ptr(), lda, 0, 1, b.data_ptr(), ldb, 0, out.data_ptr(), N, 0, cin, (int)M, (int)N, (int)K, 1, s);
  else
    dispatch<RowM, RowM, bf>((int)cfg, a.data_ptr(), lda, 0, 1, b.data_ptr(), ldb, 0, out.data_ptr(), N, 0, cin, (int)M, (int)N, (int)K, 1, s);
}

// ---- the BIG width's value | gate projection with the head-major store (pwa_wide_sm80): [v | g] = y [Wv; Wg]^T, y [S L, 256] (rows (s, j)),
// v written head-major [8, L, S 32] (the contraction's operand: no separate to_hm pass) and g natural [S L, 256] right after it in one buffer.
// The epilogue's permuted store (CUTLASS PermuteDLayout) maps (row, column) -> offset; L rides in the store's stride argument (the source C is
// never read: beta 0, null C).
namespace {
class HeadMajorVG : public cutlass::layout::PermuteBase {
  using Index = int32_t;
  using LongIndex = int64_t;
  Index L_, S_;

 public:
  CUTLASS_HOST_DEVICE HeadMajorVG(cutlass::MatrixCoord extent, Index stride) : L_(stride), S_(extent.row() / stride) {}
  CUTLASS_HOST_DEVICE HeadMajorVG(cutlass::layout::PitchLinearCoord extent, Index stride)
      : HeadMajorVG(cutlass::MatrixCoord(extent.strided(), extent.contiguous()), stride) {}
  CUTLASS_HOST_DEVICE LongIndex operator()(cutlass::MatrixCoord coord) const {
    const Index r = coord.row(), c = coord.column();
    if (c >= 256) return (LongIndex)S_ * L_ * 256 + (LongIndex)r * 256 + (c - 256);
    const Index s = r / L_, j = r - s * L_;
    return (((LongIndex)(c >> 5) * L_ + j) * S_ + s) * 32 + (c & 31);
  }
  CUTLASS_HOST_DEVICE LongIndex operator()(cutlass::layout::PitchLinearCoord coord) const {
    return operator()(cutlass::MatrixCoord(coord.strided(), coord.contiguous()));
  }
};
using VgGemm = cutlass::gemm::device::Gemm<bf, RowM, bf, ColM, bf, RowM, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                                           cutlass::gemm::GemmShape<256, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
                                           cutlass::epilogue::thread::LinearCombination<bf, 8, float, float>,
                                           cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 3, 8, 8, false,
                                           cutlass::arch::OpMultiplyAdd, false, false, false, HeadMajorVG>;
using VgK = VgGemm::GemmKernel;
static_assert(sizeof(VgK::SharedStorage) <= 101376, "the RTX 5090's 99 KB of shared memory per block");
}  // namespace

// y [S L, 256], wvg [512, 256] bf16 -> out [2 S L 256] bf16: v head-major [8, L, S 32] then g [S L, 256]
void vg_head_major(torch::Tensor y, torch::Tensor wvg, torch::Tensor out, int64_t S, int64_t L) {
  TORCH_CHECK(y.is_cuda() && y.scalar_type() == at::kBFloat16 && y.is_contiguous() && y.dim() == 2 && y.size(0) == S * L && y.size(1) == 256, "vg_head_major: y bf16 [S L, 256]");
  TORCH_CHECK(wvg.scalar_type() == at::kBFloat16 && wvg.is_contiguous() && wvg.size(0) == 512 && wvg.size(1) == 256, "vg_head_major: wvg bf16 [512, 256]");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() == 2 * S * L * 256, "vg_head_major: out bf16 [2 S L 256]");
  const at::cuda::CUDAGuard guard(y.device());
  const int M = (int)(S * L);
  const cutlass::gemm::GemmCoord problem(M, 512, 256);
  typename VgK::ThreadblockSwizzle sw;
  const cutlass::gemm::GemmCoord grid_shape = sw.get_tiled_shape(problem, {256, 128, 32}, 1);
  bf* D = reinterpret_cast<bf*>(out.data_ptr());
  typename VgK::Params params(problem, grid_shape, {reinterpret_cast<bf*>(y.data_ptr()), RowM(256)}, {reinterpret_cast<bf*>(wvg.data_ptr()), ColM(256)},
                              {nullptr, RowM((int)L)}, {D, RowM((int)L)}, {1.f, 0.f});
  const int smem = (int)sizeof(typename VgK::SharedStorage);
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(cutlass::Kernel<VgK>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    configured = true;
  }
  cutlass::Kernel<VgK><<<sw.get_grid_shape(grid_shape), VgK::kThreadCount, smem, at::cuda::getCurrentCUDAStream()>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("contract", &contract);
  m.def("weight_grad", &weight_grad);
  m.def("mm", &mm);
  m.def("vg_head_major", &vg_head_major);
}
