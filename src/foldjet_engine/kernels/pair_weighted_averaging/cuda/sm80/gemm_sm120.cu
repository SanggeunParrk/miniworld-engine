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
#include <cuda_bf16.h>
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
__global__ void __launch_bounds__(GK::kThreadCount) bgemm_kernel(typename GK::Params params, int a_div, const uint8_t* __restrict__ live, int ch,
                                                                 typename GK::Epilogue::OutputTileIterator::Element* __restrict__ dptr, long ldd, long sdd) {
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
  if (live != nullptr) {  // rows / columns of ch channels per token: a tile whose row tokens or column tokens are all dead (masked in every MSA row: a = b = 0) is 0
    const int M = params.problem_size.m(), N = params.problem_size.n(), r0 = tile.m() * Mma::Shape::kM, c0 = tile.n() * Mma::Shape::kN;
    const int r1 = min(M, r0 + Mma::Shape::kM), c1 = min(N, c0 + Mma::Shape::kN);
    bool lr = false, lc = false;
    for (int t = r0 / ch; t < (r1 + ch - 1) / ch; ++t) lr |= live[t] != 0;
    for (int t = c0 / ch; t < (c1 + ch - 1) / ch; ++t) lc |= live[t] != 0;
    if (!(lr && lc)) {
      using EC = typename GK::Epilogue::OutputTileIterator::Element;
      constexpr int V = 16 / sizeof(EC), W = Mma::Shape::kN / V;
      EC* d = dptr + (long)batch * sdd;
      for (int v = threadIdx.x; v < Mma::Shape::kM * W; v += GK::kThreadCount) {
        const int r = r0 + v / W, c = c0 + (v % W) * V;
        if (r < r1 && c < c1) *reinterpret_cast<uint4*>(d + (long)r * ldd + c) = make_uint4(0, 0, 0, 0);
      }
      return;
    }
  }
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
         int M, int N, int K, int batches, cudaStream_t stream, const uint8_t* live = nullptr, int ch = 1) {
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
  bgemm_kernel<GK><<<dim3(tiled.m(), tiled.n(), batches), GK::kThreadCount, smem, stream>>>(p, a_div, live, ch, reinterpret_cast<EC*>(c), ldc, stride_c);
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
              int K, int batches, cudaStream_t s, const uint8_t* live = nullptr, int ch = 1) {
  switch (cfg) {
#define X(id, tm, tn, tk, wm, wn, st) \
  case id: return run<Cfg<LA, LB, EC, tm, tn, tk, wm, wn, st>>(a, lda, sa, ad, b, ldb, sb, c, ldc, sc, cin, M, N, K, batches, s, live, ch);
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

// out = op(a) op(b) (+ c), row-major bf16, one GEMM or a batch: a [(B,) M, K] (trans_a: [(B,) K, M]), b [(B,) K, N] (trans_b: [(B,) N, K]),
// c [(B,) M, N] or empty -> out [(B,) M, N] bf16 (a 2-D operand is shared by the batch)
void mm(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor out, bool trans_a, bool trans_b, int64_t cfg, torch::Tensor live, int64_t ch) {
  for (auto* t : {&a, &b, &out})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kBFloat16 && t->is_contiguous() && (t->dim() == 2 || t->dim() == 3), "pwa gemm_sm120 mm: contiguous bf16 2-D / 3-D");
  const int64_t B = out.dim() == 3 ? out.size(0) : 1;
  TORCH_CHECK((a.dim() == 2 || a.size(0) == B) && (b.dim() == 2 || b.size(0) == B), "pwa gemm_sm120 mm: batch");
  const int64_t ar = a.size(a.dim() - 2), ac = a.size(a.dim() - 1), br = b.size(b.dim() - 2), bc = b.size(b.dim() - 1);
  const int64_t M = trans_a ? ac : ar, K = trans_a ? ar : ac, N = trans_b ? br : bc;
  TORCH_CHECK((trans_b ? bc : br) == K && out.size(out.dim() - 2) == M && out.size(out.dim() - 1) == N && M % 8 == 0 && N % 8 == 0 && K % 8 == 0, "pwa gemm_sm120 mm: shapes");
  const at::cuda::CUDAGuard guard(a.device());
  auto s = at::cuda::getCurrentCUDAStream();
  const int64_t sa = a.dim() == 3 ? ar * ac : 0, sb = b.dim() == 3 ? br * bc : 0, so = M * N;
  TORCH_CHECK(c.numel() == 0 || (c.scalar_type() == at::kBFloat16 && c.is_contiguous() && c.sizes() == out.sizes()), "pwa gemm_sm120 mm: c bf16 like out or empty");
  const void* cin = c.numel() ? c.data_ptr() : nullptr;
  // live (optional, [M / ch] uint8 tokens; requires M == N, no c): output tiles whose row or column tokens are all dead are written as zeros
  TORCH_CHECK(live.numel() == 0 || (live.is_cuda() && live.scalar_type() == at::kByte && live.is_contiguous() && M == N && M % ch == 0 && live.numel() == M / ch && !cin),
              "pwa gemm_sm120 mm: live [M / ch] uint8, square, no c");
  const uint8_t* lp = live.numel() ? live.data_ptr<uint8_t>() : nullptr;
  if (trans_a && trans_b)
    dispatch<ColM, ColM, bf>((int)cfg, a.data_ptr(), ac, sa, 1, b.data_ptr(), bc, sb, out.data_ptr(), N, so, cin, (int)M, (int)N, (int)K, (int)B, s, lp, (int)ch);
  else if (trans_a)
    dispatch<ColM, RowM, bf>((int)cfg, a.data_ptr(), ac, sa, 1, b.data_ptr(), bc, sb, out.data_ptr(), N, so, cin, (int)M, (int)N, (int)K, (int)B, s, lp, (int)ch);
  else if (trans_b)
    dispatch<RowM, ColM, bf>((int)cfg, a.data_ptr(), ac, sa, 1, b.data_ptr(), bc, sb, out.data_ptr(), N, so, cin, (int)M, (int)N, (int)K, (int)B, s, lp, (int)ch);
  else
    dispatch<RowM, RowM, bf>((int)cfg, a.data_ptr(), ac, sa, 1, b.data_ptr(), bc, sb, out.data_ptr(), N, so, cin, (int)M, (int)N, (int)K, (int)B, s, lp, (int)ch);
}

// split-K weight gradient: a [K, M], b [K, N] bf16 -> out fp32 [P, M, N], out[p] = a[p Kc:(p+1) Kc]^T b[p Kc:(p+1) Kc] (Kc = K / P; the caller sums
// the P partials in order): the few-tile, long-K products a^T b (M, N <= 512, K = the pairs or the tokens) that cuBLAS runs on a WMMA kernel on the 5090
void wgrad_splitk(torch::Tensor a, torch::Tensor b, torch::Tensor out, int64_t cfg) {
  for (auto* t : {&a, &b})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kBFloat16 && t->is_contiguous() && t->dim() == 2, "pwa gemm_sm120 wgrad_splitk: contiguous bf16 2-D");
  const int64_t K = a.size(0), M = a.size(1), N = b.size(1), P = out.size(0);
  TORCH_CHECK(b.size(0) == K && out.is_cuda() && out.scalar_type() == at::kFloat && out.is_contiguous() && out.dim() == 3 && out.size(1) == M && out.size(2) == N &&
                  K % P == 0 && (K / P) % 8 == 0 && M % 8 == 0 && N % 8 == 0,
              "pwa gemm_sm120 wgrad_splitk: shapes");
  const at::cuda::CUDAGuard guard(a.device());
  const int64_t Kc = K / P;
  dispatch<ColM, RowM, float>((int)cfg, a.data_ptr(), M, Kc * M, 1, b.data_ptr(), N, Kc * N, out.data_ptr(), N, M * N, nullptr, (int)M, (int)N, (int)Kc, (int)P,
                              at::cuda::getCurrentCUDAStream());
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

// ---- the Chai-1 OuterProductMean's out-projection gradients in one pass (outer_product_mean/chai.py on the RTX 5090; W' = W gamma, b' = b + W beta):
//   dWo[o, p] = dW'[o, p] gamma[p] + dbo[o] beta[p],  dgamma[p] = sum_o W[o, p] dW'[o, p],  dbeta[p] = sum_o W[o, p] dbo[o]
// dwp fp32 [CZ, P], dbo fp32 [CZ], w [CZ, P] (any float dtype, read as fp32), gamma / beta fp32 [P] -> dwo fp32 [CZ, P], dgamma / dbeta fp32 [P].
// 32 columns x 8 row groups a CTA (fp32 sums of the reference's (W * dW').sum(0) / W^T dbo).
namespace {
__device__ __forceinline__ float to_f32(float x) { return x; }
__device__ __forceinline__ float to_f32(__nv_bfloat16 x) { return __bfloat162float(x); }
}  // namespace
namespace {
template <typename TW>
__global__ void __launch_bounds__(256) chai_wo_grads_k(const float* __restrict__ dwp, const float* __restrict__ dbo, const TW* __restrict__ w,
                                                      const float* __restrict__ gamma, const float* __restrict__ beta, float* __restrict__ dwo,
                                                      float* __restrict__ dgam, float* __restrict__ dbet, int CZ, int P) {
  // a CTA: 32 columns x 8 row groups (rows o = group, group + 8, ..), the 8 partial sums added in group order
  __shared__ float red[2][8][32];
  const int lane = threadIdx.x & 31, grp = threadIdx.x >> 5, p = blockIdx.x * 32 + lane;
  float sg = 0.f, sb = 0.f;
  if (p < P) {
    const float g = gamma[p], b = beta[p];
    for (int o = grp; o < CZ; o += 8) {
      const long i = (long)o * P + p;
      const float d = dwp[i], wv = to_f32(w[i]), bo = dbo[o];
      dwo[i] = d * g + bo * b;
      sg = fmaf(wv, d, sg);
      sb = fmaf(wv, bo, sb);
    }
  }
  red[0][grp][lane] = sg;
  red[1][grp][lane] = sb;
  __syncthreads();
  if (grp < 2 && p < P) {
    float t = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) t += red[grp][k][lane];
    (grp == 0 ? dgam : dbet)[p] = t;
  }
}
}  // namespace

void chai_wo_grads(torch::Tensor dwp, torch::Tensor dbo, torch::Tensor w, torch::Tensor gamma, torch::Tensor beta, torch::Tensor dwo, torch::Tensor dgam,
                   torch::Tensor dbet) {
  const int CZ = (int)dwp.size(0), P = (int)dwp.size(1);
  for (auto* t : {&dwp, &dbo, &gamma, &beta, &dwo, &dgam, &dbet})
    TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kFloat && t->is_contiguous(), "chai_wo_grads: fp32 contiguous");
  TORCH_CHECK(w.is_contiguous() && w.size(0) == CZ && w.size(1) == P && dbo.numel() == CZ && gamma.numel() == P && beta.numel() == P && dwo.sizes() == dwp.sizes() &&
                  dgam.numel() == P && dbet.numel() == P,
              "chai_wo_grads: shapes");
  const at::cuda::CUDAGuard guard(dwp.device());
  auto s = at::cuda::getCurrentCUDAStream();
  const dim3 grid((P + 31) / 32);
  if (w.scalar_type() == at::kFloat)
    chai_wo_grads_k<float><<<grid, 256, 0, s>>>(dwp.data_ptr<float>(), dbo.data_ptr<float>(), w.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
                                                dwo.data_ptr<float>(), dgam.data_ptr<float>(), dbet.data_ptr<float>(), CZ, P);
  else if (w.scalar_type() == at::kBFloat16)
    chai_wo_grads_k<__nv_bfloat16><<<grid, 256, 0, s>>>(dwp.data_ptr<float>(), dbo.data_ptr<float>(), reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
                                                        gamma.data_ptr<float>(), beta.data_ptr<float>(), dwo.data_ptr<float>(), dgam.data_ptr<float>(),
                                                        dbet.data_ptr<float>(), CZ, P);
  else
    TORCH_CHECK(false, "chai_wo_grads: w fp32 or bf16");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// live[t] = any_s mask[s, t] (uint8 0 / 1) for mask [S, N] bytes: 32 tokens x 8 row groups a CTA (the torch column reduction runs few CTAs here)
namespace {
__global__ void __launch_bounds__(256) live_tokens_k(const uint8_t* __restrict__ mask, uint8_t* __restrict__ live, int S, int N) {
  __shared__ int red[8][32];
  const int lane = threadIdx.x & 31, grp = threadIdx.x >> 5, t = blockIdx.x * 32 + lane;
  int any = 0;
  if (t < N) {
#pragma unroll 8
    for (int s = grp; s < S; s += 8) any |= mask[(long)s * N + t];
  }
  red[grp][lane] = any;
  __syncthreads();
  if (grp == 0 && t < N) {
    int v = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) v |= red[k][lane];
    live[t] = v != 0;
  }
}
}  // namespace

void live_tokens(torch::Tensor mask, torch::Tensor live) {
  TORCH_CHECK(mask.is_cuda() && mask.element_size() == 1 && mask.is_contiguous() && mask.dim() == 2 && live.scalar_type() == at::kByte && live.is_contiguous() &&
                  live.numel() == mask.size(1),
              "live_tokens: mask [S, N] bytes, live [N] uint8");
  const at::cuda::CUDAGuard guard(mask.device());
  const int S = (int)mask.size(0), N = (int)mask.size(1);
  live_tokens_k<<<(N + 31) / 32, 256, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const uint8_t*>(mask.data_ptr()), live.data_ptr<uint8_t>(), S, N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// column sums of fp32 partials [R, C] into two outputs (columns < split -> out_a, the rest -> out_b): the Chai-1 backward's per-block dgamma | dbeta
// rows and its split-K dW partials in one launch each, without the slice copies (32 columns x 8 row groups a CTA, the groups added in order)
namespace {
__global__ void __launch_bounds__(1024) colsum2_k(const float* __restrict__ in, float* __restrict__ out_a, float* __restrict__ out_b, int R, int C, int split) {
  __shared__ float red[32][32];
  const int ng = blockDim.x >> 5;                          // row groups (8 or 32)
  const int lane = threadIdx.x & 31, grp = threadIdx.x >> 5, c = blockIdx.x * 32 + lane;
  float acc = 0.f;
  if (c < C) {
#pragma unroll 4
    for (int r = grp; r < R; r += ng) acc += in[(long)r * C + c];
  }
  red[grp][lane] = acc;
  __syncthreads();
  if (grp == 0 && c < C) {
    float v = 0.f;
#pragma unroll 8
    for (int k = 0; k < ng; ++k) v += red[k][lane];
    if (c < split) out_a[c] = v; else out_b[c - split] = v;
  }
}
}  // namespace

void colsum2(torch::Tensor in, torch::Tensor out_a, torch::Tensor out_b) {
  TORCH_CHECK(in.is_cuda() && in.scalar_type() == at::kFloat && in.is_contiguous() && in.dim() == 2, "colsum2: in fp32 [R, C]");
  const int R = (int)in.size(0), C = (int)in.size(1), split = (int)out_a.numel();
  TORCH_CHECK(out_a.scalar_type() == at::kFloat && out_b.scalar_type() == at::kFloat && out_a.is_contiguous() && out_b.is_contiguous() && split + out_b.numel() == C,
              "colsum2: out_a / out_b fp32, sizes adding to C");
  const at::cuda::CUDAGuard guard(in.device());
  const int threads = (C + 31) / 32 >= 64 || R < 512 ? 256 : 1024;   // few columns over many rows: 32 row groups a CTA
  colsum2_k<<<(C + 31) / 32, threads, 0, at::cuda::getCurrentCUDAStream()>>>(in.data_ptr<float>(), out_a.data_ptr<float>(), out_b.data_ptr<float>(), R, C, split);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("contract", &contract);
  m.def("weight_grad", &weight_grad);
  m.def("mm", &mm);
  m.def("vg_head_major", &vg_head_major);
  m.def("wgrad_splitk", &wgrad_splitk);
  m.def("chai_wo_grads", &chai_wo_grads);
  m.def("live_tokens", &live_tokens);
  m.def("colsum2", &colsum2);
}
