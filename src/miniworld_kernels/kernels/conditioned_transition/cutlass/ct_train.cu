// Fused-epilogue training (fwd+bwd) for the post-AdaLN ConditionedTransition tail.
// Hopper TF32 WGMMA (sm_90a), CUTLASS 3.x CollectiveBuilder + EVT epilogue fusion.
//
// FORWARD (saves a,b,h,out,scale for bwd):
//   G1  b   = x @ Wb^T                                  (plain GEMM)
//   G2  h   = silu(a) * b   (+ store a)                 (a=x@Wa^T acc; EVT: silu(acc)*aux(b))
//   G3  out = h @ Ws^T                                  (plain GEMM)
//   G4  y   = sigmoid(scale)*out  (+ store scale)       (scale=cond@Wsc^T acc; EVT: rowbias(b_sc),
//                                                         AuxStore(scale), sigmoid, *aux(out))
//
// BACKWARD: tuned plain CUTLASS GEMMs (which match/beat cuBLAS) + fused-elementwise CUDA kernels
//   for the gate-bwd (dout,dscale) and swiglu-bwd (dab=[da|db]); wgrads via plain GEMMs.
//   (Phase-B input-fusion of dh->dab into the dh-GEMM epilogue is layered on separately.)
#include "ct_common.cuh"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/thread/activation.h"

namespace ct {
namespace fusion = cutlass::epilogue::fusion;
using cutlass::FloatRoundStyle;
constexpr auto RN = FloatRoundStyle::round_to_nearest;

using StrideAux = cutlass::detail::TagToStrideC_t<LayoutC>;  // row-major (M,N) aux
using CtaTile = TileShape;                                   // for broadcast tile shape

// ---- builders parameterized by a custom fusion op -----------------------------------
template <class FusionOp>
struct FwdEpi {
  using Op = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OpClass, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute, ElementC, LayoutC, Align, ElementC, LayoutC, Align,
      cutlass::epilogue::TmaWarpSpecializedCooperative, FusionOp>::CollectiveOp;
};
template <class Epi>
struct FwdMain {
  using Op = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OpClass, ElementA, LayoutA, Align, ElementB, LayoutB, Align, ElementAcc,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epi::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
};
template <class FusionOp>
struct FwdGemm {
  using Epi  = typename FwdEpi<FusionOp>::Op;
  using Main = typename FwdMain<Epi>::Op;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

// ============ G2 fusion: h = silu(acc) * aux(b) =======================================
using EVT_h =
  fusion::Sm90EVT<fusion::Sm90Compute<cutlass::multiplies, ElementC, ElementCompute, RN>,
    fusion::Sm90EVT<fusion::Sm90Compute<cutlass::epilogue::thread::SiLu, ElementCompute, ElementCompute, RN>,
      fusion::Sm90AccFetch>,
    fusion::Sm90AuxLoad<0, void, ElementC, LayoutC, void, void>>;
using GemmH = FwdGemm<EVT_h>;

// ============ G4 fusion: y = sigmoid( store(acc + rowbias(b_sc)) ) * aux(out) =========
// b_sc is per-N-column (D) -> RowBroadcast with stride <_0,_1,L>.
using RowBiasStride = cute::Stride<_0, _1, int>;
using ScaleSubtree =
  fusion::Sm90EVT<fusion::Sm90Compute<cutlass::plus, ElementCompute, ElementCompute, RN>,
    fusion::Sm90AccFetch,
    fusion::Sm90RowBroadcast<0, CtaTile, ElementC, ElementCompute, RowBiasStride>>;
// AuxStore(scale) passes scale through while persisting it; then sigmoid; then * out.
#ifndef CT_NO_FUSED_Y
using EVT_y =
  fusion::Sm90EVT<fusion::Sm90Compute<cutlass::multiplies, ElementC, ElementCompute, RN>,
    fusion::Sm90EVT<fusion::Sm90Compute<cutlass::epilogue::thread::Sigmoid, ElementCompute, ElementCompute, RN>,
      fusion::Sm90EVT<fusion::Sm90AuxStore<0, void, ElementC, RN, LayoutC, void, void>,
        ScaleSubtree>>,
    fusion::Sm90AuxLoad<0, void, ElementC, LayoutC, void, void>>;
using GemmY = FwdGemm<EVT_y>;
#endif

// ---- helpers to fill the arg structs for the fused GEMMs ----------------------------
template <class G>
static void fwd_strides(int M, int N, int K,
    typename G::GemmKernel::StrideA& dA, typename G::GemmKernel::StrideB& dB,
    typename G::GemmKernel::StrideC& dC, typename G::GemmKernel::StrideD& dD) {
  dA = cutlass::make_cute_packed_stride(typename G::GemmKernel::StrideA{}, cute::make_shape(M, K, 1));
  dB = cutlass::make_cute_packed_stride(typename G::GemmKernel::StrideB{}, cute::make_shape(N, K, 1));
  dC = cutlass::make_cute_packed_stride(typename G::GemmKernel::StrideC{}, cute::make_shape(M, N, 1));
  dD = cutlass::make_cute_packed_stride(typename G::GemmKernel::StrideD{}, cute::make_shape(M, N, 1));
}

template <class G>
static void launch(typename G::Arguments& args, at::Device dev) {
  G gemm;
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "can_implement");
  size_t ws = G::get_workspace_size(args);
  auto wbuf = torch::empty({(int64_t)ws}, torch::dtype(torch::kUInt8).device(dev));
  TORCH_CHECK(gemm.initialize(args, wbuf.data_ptr()) == cutlass::Status::kSuccess, "init");
  TORCH_CHECK(gemm.run(at::cuda::getCurrentCUDAStream()) == cutlass::Status::kSuccess, "run");
}

// G2: h = silu(x@Wa^T) * b ; b is (M,ND) aux ; D output = h (M,ND).  (a recovered as x@Wa^T -> we
// also need 'a' saved; we store a by a separate trivial path: see python, which recomputes a via G1b.
// To keep one kernel, we additionally output a by... not directly. So caller also runs G_a (plain) for a.)
static torch::Tensor fused_h(torch::Tensor x, torch::Tensor Wa, torch::Tensor b) {
  using G = GemmH::Gemm;
  int M = x.size(0), K = x.size(1), N = Wa.size(0);
  auto h = torch::empty({M, N}, x.options());
  typename G::GemmKernel::StrideA dA; typename G::GemmKernel::StrideB dB;
  typename G::GemmKernel::StrideC dC; typename G::GemmKernel::StrideD dD;
  fwd_strides<G>(M, N, K, dA, dB, dC, dD);
  // EVT_h = Sm90EVT<multiplies, SiluSub, AuxLoadB>.  Arguments tuple order is
  //   { SiluSub_args, AuxLoadB_args, multiplies_args }   (NodeOp last).
  // SiluSub = Sm90EVT<SiLu, AccFetch> -> { AccFetch_args{}, SiLu_args{} }.
  // AuxLoadB_args = { ptr_aux, null_default, dAux }.
  typename GemmH::Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
    {x.data_ptr<float>(), dA, Wa.data_ptr<float>(), dB},
    { { /*SiluSub*/ { /*AccFetch*/ {}, /*SiLu*/ {} },
        /*AuxLoadB*/ { b.data_ptr<float>(), ElementC(0), dD },
        /*multiplies*/ {} },
      nullptr, dC, h.data_ptr<float>(), dD } };
  launch<G>(args, x.device());
  return h;
}

// G4: y = sigmoid(scale)*out ; scale = cond@Wsc^T + b_sc (rowbias) ; AuxStore scale ; aux out.
#ifndef CT_NO_FUSED_Y
static std::vector<torch::Tensor> fused_y(torch::Tensor cond, torch::Tensor Wsc,
                                          torch::Tensor b_sc, torch::Tensor out) {
  using G = GemmY::Gemm;
  int M = cond.size(0), K = cond.size(1), N = Wsc.size(0);  // N = D
  auto y = torch::empty({M, N}, cond.options());
  auto scale = torch::empty({M, N}, cond.options());
  typename G::GemmKernel::StrideA dA; typename G::GemmKernel::StrideB dB;
  typename G::GemmKernel::StrideC dC; typename G::GemmKernel::StrideD dD;
  fwd_strides<G>(M, N, K, dA, dB, dC, dD);
  // EVT_y = Sm90EVT<multiplies, SigmoidSub, AuxLoadOut>
  //   tuple { SigmoidSub_args, AuxLoadOut_args, mult{} }
  // SigmoidSub = Sm90EVT<Sigmoid, AuxStoreSub> -> { AuxStoreSub_args, sigmoid{} }
  // AuxStoreSub = Sm90EVT<AuxStore, ScaleSubtree> -> { ScaleSubtree_args, AuxStore_args{ptr,dAux} }
  // ScaleSubtree = Sm90EVT<plus, AccFetch, RowBroadcast> -> { AccFetch{}, RowBroadcast{ptr,null,stride}, plus{} }
  RowBiasStride bsc_stride{_0{}, _1{}, 0};
  typename GemmY::Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
    {cond.data_ptr<float>(), dA, Wsc.data_ptr<float>(), dB},
    { /*epilogue*/
      { /*thread tuple = {SigmoidSub, AuxLoadOut, multiplies}*/
        { /*SigmoidSub = {AuxStoreSub, sigmoid}*/
          { /*AuxStoreSub = {ScaleSubtree, AuxStore_args}*/
            { /*ScaleSubtree (plus) = {AccFetch, RowBroadcast, plus}*/
              {},                                                     // AccFetch
              { b_sc.data_ptr<float>(), ElementC(0), bsc_stride },    // RowBroadcast(b_sc)
              {} },                                                   // plus
            { scale.data_ptr<float>(), dD } },                        // AuxStore(scale)
          {} },                                                       // sigmoid
        { out.data_ptr<float>(), ElementC(0), dD },                   // AuxLoad(out)
        {} },                                                         // multiplies
      nullptr, dC, y.data_ptr<float>(), dD } };
  launch<G>(args, cond.device());
  return {y, scale};
}
#endif  // CT_NO_FUSED_Y

// pybind: expose the two fused-epilogue forward GEMMs; plain GEMMs + bwd elementwise are
// in ct_plain.cu / python. (split TUs keep compile time sane.)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_h", &fused_h, "h = silu(x@Wa^T) * b  (fused epilogue)");
#ifndef CT_NO_FUSED_Y
  m.def("fused_y", &fused_y, "y = sigmoid(cond@Wsc^T + b_sc) * out ; returns {y, scale}");
#endif
}

} // namespace ct
