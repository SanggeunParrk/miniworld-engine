// The TriMul contraction on A100 bounded by the live extents of the pair mask, read on the device: CUTLASS 2.x sm_80 threadblock
// mainloop and epilogue (DefaultGemm's Mma / Epilogue) under an own batched kernel (one plane per blockIdx.z, a swizzled m / n raster) in
// which each threadblock reads the live extents (rows, columns, k) of its plane's sample and
//   * writes its zero tile without the product when its rows or columns are all past them (their operand rows are zero: K1 masks a / b
//     by m[i] & m[k]), and
//   * otherwise runs the K loop only up to the live k rounded up to the K tile (the k tiles past it hold zero products).
// The launch is shape-static (CUDA graphs: the extents are read by every replay), and the output is the full-K product of the same kernel
// bit for bit whenever n is a multiple of the K tile (no K residue moves).
//   mode 0 (outgoing): t[i, j] = sum_k a[i, k] b[j, k];  mode 1 (incoming): t[i, j] = sum_k a[k, i] b[k, j]
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/kernel/default_gemm.h"

namespace {
using bf = cutlass::bfloat16_t;
using RowM = cutlass::layout::RowMajor;
using ColM = cutlass::layout::ColumnMajor;

template <int MODE, int TM, int TN, int TK, int WM, int WN, int STAGES>
struct Cfg {
  using LA = std::conditional_t<MODE == 0, RowM, ColM>;
  using LB = std::conditional_t<MODE == 0, ColM, RowM>;
  using GK = typename cutlass::gemm::kernel::DefaultGemm<
      bf, LA, 8, bf, LB, 8, bf, RowM, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<TM, TN, TK>,
      cutlass::gemm::GemmShape<WM, WN, TK>, cutlass::gemm::GemmShape<16, 8, 16>, cutlass::epilogue::thread::LinearCombination<bf, 8, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, STAGES, false, cutlass::arch::OpMultiplyAdd>::GemmKernel;
  using Mma = typename GK::Mma;
  using Epilogue = typename GK::Epilogue;
  struct Args {
    typename Mma::IteratorA::Params pa;
    typename Mma::IteratorB::Params pb;
    typename Epilogue::OutputTileIterator::Params pd;
    const bf* a;
    const bf* b;
    bf* d;
    const int* bounds;  // int32 [nsamples, 3] or nullptr
    int nsamples, n, tiles_m, tiles_n;
  };
};

template <class C, int TM, int TN, int TK, int LOG>
__global__ void __launch_bounds__(C::GK::kThreadCount) bounded_kernel(typename C::Args args) {
  using Mma = typename C::Mma;
  using Epilogue = typename C::Epilogue;
  extern __shared__ __align__(16) unsigned char smem[];
  auto& ss = *reinterpret_cast<typename C::GK::SharedStorage*>(smem);
  const int tm = blockIdx.x >> LOG, tn = (blockIdx.y << LOG) + (blockIdx.x & ((1 << LOG) - 1));
  if (tm >= args.tiles_m || tn >= args.tiles_n) return;
  const int n = args.n, l = blockIdx.z;
  const int64_t ps = (int64_t)n * n;
  const int r0 = tm * TM, c0 = tn * TN;
  int k = n;
  if (args.bounds != nullptr) {
    const int* b = args.bounds + 3 * (l % args.nsamples);
    const int live_m = __ldg(b), live_n = __ldg(b + 1), live_k = __ldg(b + 2);
    if (r0 >= live_m || c0 >= live_n) {  // a dead tile: zeros
      bf* o = args.d + l * ps;
      for (int v = threadIdx.x; v < TM * (TN / 8); v += C::GK::kThreadCount) {
        const int r = r0 + v / (TN / 8), c = c0 + (v % (TN / 8)) * 8;
        if (r < n && c < n) *reinterpret_cast<uint4*>(o + (int64_t)r * n + c) = make_uint4(0, 0, 0, 0);
      }
      return;
    }
    k = min(n, ((max(live_k, 1) + TK - 1) / TK) * TK);
  }
  const int tid = threadIdx.x;
  typename Mma::IteratorA it_a(args.pa, const_cast<bf*>(args.a) + l * ps, {n, k}, tid, {r0, 0});
  typename Mma::IteratorB it_b(args.pb, const_cast<bf*>(args.b) + l * ps, {k, n}, tid, {0, c0});
  const int warp = cutlass::canonical_warp_idx_sync(), lane = tid % 32;
  Mma mma(ss.main_loop, tid, warp, lane);
  typename Mma::FragmentC acc;
  acc.clear();
  mma((k + TK - 1) / TK, acc, it_a, it_b, acc);
  typename Epilogue::OutputOp op({1.f, 0.f});
  typename Epilogue::OutputTileIterator it_d(args.pd, args.d + l * ps, {n, n}, tid, {r0, c0});
  Epilogue epilogue(ss.epilogue, tid, warp, lane);
  epilogue(op, it_d, acc, it_d);
}

template <class C, int TM, int TN, int TK, int LOG>
void launch_log(typename C::Args const& args, int planes) {
  auto kernel = bounded_kernel<C, TM, TN, TK, LOG>;
  const int smem = (int)sizeof(typename C::GK::SharedStorage);
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    configured = true;
  }
  const dim3 grid(args.tiles_m << LOG, (args.tiles_n + (1 << LOG) - 1) >> LOG, planes);
  kernel<<<grid, C::GK::kThreadCount, smem, at::cuda::getCurrentCUDAStream()>>>(args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int MODE, int TM, int TN, int TK, int WM, int WN, int STAGES>
void launch(torch::Tensor a, torch::Tensor b, torch::Tensor out, const int* bounds, int64_t nsamples, int64_t log_tile) {
  using C = Cfg<MODE, TM, TN, TK, WM, WN, STAGES>;
  const int planes = (int)a.size(0), n = (int)a.size(1);
  typename C::Args args{typename C::Mma::IteratorA::Params(typename C::LA(n)),
                        typename C::Mma::IteratorB::Params(typename C::LB(n)),
                        typename C::Epilogue::OutputTileIterator::Params(RowM(n)),
                        reinterpret_cast<const bf*>(a.data_ptr()),
                        reinterpret_cast<const bf*>(b.data_ptr()),
                        reinterpret_cast<bf*>(out.data_ptr()),
                        bounds,
                        (int)nsamples,
                        n,
                        (n + TM - 1) / TM,
                        (n + TN - 1) / TN};
  switch (log_tile) {
    case 0: launch_log<C, TM, TN, TK, 0>(args, planes); break;
    case 1: launch_log<C, TM, TN, TK, 1>(args, planes); break;
    case 2: launch_log<C, TM, TN, TK, 2>(args, planes); break;
    default: launch_log<C, TM, TN, TK, 3>(args, planes); break;
  }
}

template <int MODE>
void dispatch(int64_t cfg, torch::Tensor a, torch::Tensor b, torch::Tensor out, const int* bounds, int64_t ns, int64_t lg) {
  switch (cfg) {
    case 0: launch<MODE, 128, 256, 32, 64, 64, 3>(a, b, out, bounds, ns, lg); break;
    case 1: launch<MODE, 256, 128, 32, 64, 64, 3>(a, b, out, bounds, ns, lg); break;
    case 2: launch<MODE, 128, 128, 32, 64, 64, 4>(a, b, out, bounds, ns, lg); break;
    case 3: launch<MODE, 128, 128, 64, 64, 64, 3>(a, b, out, bounds, ns, lg); break;
    case 4: launch<MODE, 128, 128, 32, 64, 64, 5>(a, b, out, bounds, ns, lg); break;
    case 5: launch<MODE, 128, 256, 64, 64, 64, 3>(a, b, out, bounds, ns, lg); break;
    case 6: launch<MODE, 128, 128, 32, 64, 32, 5>(a, b, out, bounds, ns, lg); break;
    case 7: launch<MODE, 128, 128, 32, 32, 64, 5>(a, b, out, bounds, ns, lg); break;
    case 8: launch<MODE, 256, 128, 64, 64, 64, 3>(a, b, out, bounds, ns, lg); break;
    case 9: launch<MODE, 256, 128, 32, 64, 64, 4>(a, b, out, bounds, ns, lg); break;
    case 10: launch<MODE, 128, 256, 32, 64, 64, 4>(a, b, out, bounds, ns, lg); break;
    case 11: launch<MODE, 128, 128, 64, 64, 64, 4>(a, b, out, bounds, ns, lg); break;
    default: TORCH_CHECK(false, "contract_bounded: config ", cfg);
  }
}
}  // namespace

// a, b: contiguous bf16 planes [P, n, n] (n % 8 == 0); out [P, n, n]; bounds: int32 [nsamples, 3] (live rows, columns, k) on the device, or
// None for the plain product; plane l belongs to sample l % nsamples. cfg: the tile set; log_tile: the m / n raster swizzle (0..3).
void contract(torch::Tensor a, torch::Tensor b, torch::Tensor out, int64_t mode, int64_t cfg, c10::optional<torch::Tensor> bounds, int64_t nsamples,
              int64_t log_tile) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == at::kBFloat16 && a.is_contiguous() && a.dim() == 3 && a.size(1) == a.size(2),
              "contract_bounded: a contiguous bf16 [P, n, n]");
  TORCH_CHECK(b.sizes() == a.sizes() && b.is_contiguous() && b.scalar_type() == at::kBFloat16, "contract_bounded: b like a");
  TORCH_CHECK(out.sizes() == a.sizes() && out.is_contiguous() && out.scalar_type() == at::kBFloat16, "contract_bounded: out like a");
  TORCH_CHECK(a.size(1) % 8 == 0, "contract_bounded: n % 8 == 0");
  TORCH_CHECK(a.size(0) <= 65535, "contract_bounded: at most 65535 planes");
  const int* bd = nullptr;
  if (bounds.has_value()) {
    TORCH_CHECK(bounds->is_cuda() && bounds->scalar_type() == at::kInt && bounds->is_contiguous() && bounds->numel() == 3 * nsamples && nsamples >= 1,
                "contract_bounded: bounds int32 [nsamples, 3]");
    bd = bounds->data_ptr<int>();
  }
  const at::cuda::CUDAGuard guard(a.device());
  if (mode == 0)
    dispatch<0>(cfg, a, b, out, bd, nsamples, log_tile);
  else
    dispatch<1>(cfg, a, b, out, bd, nsamples, log_tile);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("contract", &contract); }
