// The TriMul contraction on A100 bounded by the live extents of the pair mask, read on the device: a CUTLASS 2.x sm_80 batched GEMM (one
// plane per batch) wrapped so that each threadblock reads the live extents (rows, columns, k) of its plane's sample and
//   * writes its zero tile without the product when its rows or columns are all past them (their operand rows are zero: K1 masks a / b
//     by m[i] & m[k]), and
//   * otherwise runs the K loop only up to the live k rounded up to the K tile (the k tiles past it hold zero products).
// The launch is shape-static (CUDA graphs: the extents are read by every replay), and the output is the full-K product of the same kernel
// bit for bit whenever L is a multiple of the K tile (no K residue moves).
//   mode 0 (outgoing): t[i, j] = sum_k a[i, k] b[j, k];  mode 1 (incoming): t[i, j] = sum_k a[k, i] b[k, j]
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_batched.h"

namespace {
using bf = cutlass::bfloat16_t;
using RowM = cutlass::layout::RowMajor;
using ColM = cutlass::layout::ColumnMajor;

template <int MODE, int TM, int TN, int TK, int WM, int WN, int STAGES>
struct Cfg {
  using LA = std::conditional_t<MODE == 0, RowM, ColM>;
  using LB = std::conditional_t<MODE == 0, ColM, RowM>;
  using Gemm = cutlass::gemm::device::GemmBatched<bf, LA, bf, LB, bf, RowM, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                                                  cutlass::gemm::GemmShape<TM, TN, TK>, cutlass::gemm::GemmShape<WM, WN, TK>,
                                                  cutlass::gemm::GemmShape<16, 8, 16>, cutlass::epilogue::thread::LinearCombination<bf, 8, float, float>,
                                                  cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, STAGES, 8, 8>;
  using GK = typename Gemm::GemmKernel;
};

template <class GK, int TM, int TN, int TK>
__global__ void __launch_bounds__(GK::kThreadCount) bounded_kernel(typename GK::Params params, const int* __restrict__ bounds, int nsamples,
                                                                   bf* __restrict__ out, int n) {
  extern __shared__ __align__(16) unsigned char smem[];
  typename GK::ThreadblockSwizzle sw;
  const cutlass::gemm::GemmCoord tile = sw.get_tile_offset(params.swizzle_log_tile);
  if (params.grid_tiled_shape.m() <= tile.m() || params.grid_tiled_shape.n() <= tile.n()) return;
  const int batch = sw.get_batch_idx();
  if (bounds == nullptr) {
    GK op;
    op(params, *reinterpret_cast<typename GK::SharedStorage*>(smem));
    return;
  }
  // one sample per launch on this card (a batch of planes all belongs to it) unless nsamples > 1: plane l is sample l % nsamples, and a
  // threadblock walks planes batch, batch + gridDim.z, ... -- the per-plane extents differ then, so it takes one plane per pass below
  const int r0 = tile.m() * TM, c0 = tile.n() * TN;
  for (int l = batch; l < params.batch_count; l += gridDim.z) {
    const int* b = bounds + 3 * (l % nsamples);
    const int live_m = __ldg(b), live_n = __ldg(b + 1), live_k = __ldg(b + 2);
    if (r0 >= live_m || c0 >= live_n) {
      bf* o = out + (int64_t)l * n * n;
      for (int v = threadIdx.x; v < TM * (TN / 8); v += GK::kThreadCount) {
        const int r = r0 + v / (TN / 8), c = c0 + (v % (TN / 8)) * 8;
        if (r < n && c < n) *reinterpret_cast<uint4*>(o + (int64_t)r * n + c) = make_uint4(0, 0, 0, 0);
      }
      continue;
    }
    typename GK::Params p = params;
    int k = min(n, ((max(live_k, 1) + TK - 1) / TK) * TK);
    p.problem_size = cutlass::gemm::GemmCoord(params.problem_size.m(), params.problem_size.n(), k);
    p.gemm_k_iterations = (k + TK - 1) / TK;
    // this plane only (the kernel's own batch loop starts at get_batch_idx() and strides by gridDim.z: offset the operands to plane l and
    // run one batch)
    p.ref_A.reset(params.ref_A.data() + params.stride_A * l);
    p.ref_B.reset(params.ref_B.data() + params.stride_B * l);
    p.ref_C.reset(params.ref_C.data() + params.stride_C * l);
    p.ref_D.reset(params.ref_D.data() + params.stride_D * l);
    p.batch_count = batch + 1;  // the loop body runs once, at batch_idx = batch: shift the pointers back by that
    p.ref_A.reset(p.ref_A.data() - params.stride_A * batch);
    p.ref_B.reset(p.ref_B.data() - params.stride_B * batch);
    p.ref_C.reset(p.ref_C.data() - params.stride_C * batch);
    p.ref_D.reset(p.ref_D.data() - params.stride_D * batch);
    GK op;
    op(p, *reinterpret_cast<typename GK::SharedStorage*>(smem));
    __syncthreads();
  }
}

template <int MODE, int TM, int TN, int TK, int WM, int WN, int STAGES>
void launch(torch::Tensor a, torch::Tensor b, torch::Tensor out, c10::optional<torch::Tensor> bounds, int64_t nsamples) {
  using C = Cfg<MODE, TM, TN, TK, WM, WN, STAGES>;
  using GK = typename C::GK;
  const int planes = (int)a.size(0), n = (int)a.size(1);
  const int64_t ps = (int64_t)n * n;
  const cutlass::gemm::GemmCoord problem(n, n, n);
  typename GK::ThreadblockSwizzle sw;
  const cutlass::gemm::GemmCoord grid_shape = sw.get_tiled_shape(problem, {TM, TN, TK}, planes);
  bf* A = reinterpret_cast<bf*>(a.data_ptr());
  bf* B = reinterpret_cast<bf*>(b.data_ptr());
  bf* D = reinterpret_cast<bf*>(out.data_ptr());
  // a / b are the planes as stored ([P, n, n], row k of b / row i of a contiguous); the layouts above read them as the mode wants
  typename C::LA la(n);
  typename C::LB lb(n);
  typename GK::Params params(problem, grid_shape, {A, la}, ps, {B, lb}, ps, {D, RowM(n)}, ps, {D, RowM(n)}, ps, {1.f, 0.f}, planes);
  dim3 grid = sw.get_grid_shape(grid_shape);
  const int smem = (int)sizeof(typename GK::SharedStorage);
  auto kernel = bounded_kernel<GK, TM, TN, TK>;
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    configured = true;
  }
  const int* bd = bounds.has_value() ? bounds->data_ptr<int>() : nullptr;
  kernel<<<grid, GK::kThreadCount, smem, at::cuda::getCurrentCUDAStream()>>>(params, bd, (int)nsamples, D, n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int MODE>
void dispatch(int64_t cfg, torch::Tensor a, torch::Tensor b, torch::Tensor out, c10::optional<torch::Tensor> bounds, int64_t ns) {
  switch (cfg) {
    case 0: launch<MODE, 128, 256, 32, 64, 64, 3>(a, b, out, bounds, ns); break;
    case 1: launch<MODE, 256, 128, 32, 64, 64, 3>(a, b, out, bounds, ns); break;
    case 2: launch<MODE, 128, 128, 32, 64, 64, 4>(a, b, out, bounds, ns); break;
    case 3: launch<MODE, 128, 128, 64, 64, 64, 3>(a, b, out, bounds, ns); break;
    case 4: launch<MODE, 128, 256, 64, 64, 64, 3>(a, b, out, bounds, ns); break;
    case 5: launch<MODE, 128, 128, 32, 64, 64, 5>(a, b, out, bounds, ns); break;
    default: TORCH_CHECK(false, "contract_bounded: config ", cfg);
  }
}
}  // namespace

// a, b: contiguous bf16 planes [P, n, n] (n % 8 == 0); out [P, n, n]; bounds: int32 [nsamples, 3] (live rows, columns, k) on the device, or
// None for the plain product; plane l belongs to sample l % nsamples
void contract(torch::Tensor a, torch::Tensor b, torch::Tensor out, int64_t mode, int64_t cfg, c10::optional<torch::Tensor> bounds, int64_t nsamples) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == at::kBFloat16 && a.is_contiguous() && a.dim() == 3 && a.size(1) == a.size(2),
              "contract_bounded: a contiguous bf16 [P, n, n]");
  TORCH_CHECK(b.sizes() == a.sizes() && b.is_contiguous() && b.scalar_type() == at::kBFloat16, "contract_bounded: b like a");
  TORCH_CHECK(out.sizes() == a.sizes() && out.is_contiguous() && out.scalar_type() == at::kBFloat16, "contract_bounded: out like a");
  TORCH_CHECK(a.size(1) % 8 == 0, "contract_bounded: n % 8 == 0");
  if (bounds.has_value())
    TORCH_CHECK(bounds->is_cuda() && bounds->scalar_type() == at::kInt && bounds->is_contiguous() && bounds->numel() == 3 * nsamples && nsamples >= 1,
                "contract_bounded: bounds int32 [nsamples, 3]");
  const at::cuda::CUDAGuard guard(a.device());
  if (mode == 0)
    dispatch<0>(cfg, a, b, out, bounds, nsamples);
  else
    dispatch<1>(cfg, a, b, out, bounds, nsamples);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("contract", &contract); }
