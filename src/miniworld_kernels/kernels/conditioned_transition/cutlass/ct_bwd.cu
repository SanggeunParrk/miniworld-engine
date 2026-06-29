// Backward INPUT-FUSION GEMMs for the ConditionedTransition tail (Hopper TF32 WGMMA).
//
// The prize: the swiglu-bwd operands da, db are produced DIRECTLY by the dh-GEMM epilogue,
// recomputing the elementwise transform from saved a,b in the epilogue — dh and dab NEVER
// materialize in HBM (vs the cuBLAS+triton path which writes dh:(M,ND) and dab:(M,2ND)).
//
//   da = (dout @ Ws) * b * silu'(a)      [acc=dh ; aux-load a,b ; custom SiLuBwd(a)]
//   db = (dout @ Ws) * silu(a)           [acc=dh ; aux-load a   ; builtin SiLu(a)]
//
// Two single-output EVT GEMMs (no DAG / no mid-tree AuxStore) sharing the recomputed dh=dout@Ws.
// dh is recomputed twice (cheap: (M,d)x(d,ND)) to avoid materializing dh + dab + a swiglu-bwd
// elementwise launch.  dout itself = sigmoid(scale)*dy is produced by a tiny elementwise kernel
// (gate-bwd) since it is the A operand (mainloop side, not epilogue-fusable here).
#include "ct_common.cuh"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/thread/activation.h"

namespace ct {
namespace fusion = cutlass::epilogue::fusion;
using cutlass::FloatRoundStyle;
constexpr auto RN = FloatRoundStyle::round_to_nearest;

// ---- custom epilogue compute: silu'(x) = s*(1 + x*(1-s)), s=sigmoid(x) ---------------
template <typename T>
struct CTSiLuBwd {
  CUTLASS_HOST_DEVICE T operator()(T const& x) const {
    cutlass::epilogue::thread::Sigmoid<T> sig;
    T s = sig(x);
    return s * (T(1) + x * (T(1) - s));
  }
};
template <typename T, int N>
struct CTSiLuBwd<cutlass::Array<T, N>> {
  CUTLASS_HOST_DEVICE cutlass::Array<T, N>
  operator()(cutlass::Array<T, N> const& x) const {
    cutlass::Array<T, N> out;
    CTSiLuBwd<T> op;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) out[i] = op(x[i]);
    return out;
  }
};

// ---- builders (explicit cooperative schedule, required for custom EVT) ---------------
template <class FusionOp>
struct BwdGemm {
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OpClass, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute, ElementC, LayoutC, Align, ElementC, LayoutC, Align,
      cutlass::epilogue::TmaWarpSpecializedCooperative, FusionOp>::CollectiveOp;
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OpClass, ElementA, LayoutA, Align, ElementB, LayoutB, Align, ElementAcc,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epi::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

// da = multiplies( multiplies(acc, AuxLoad(b)), SiLuBwd(AuxLoad(a)) )
using EVT_da =
  fusion::Sm90EVT<fusion::Sm90Compute<cutlass::multiplies, ElementC, ElementCompute, RN>,
    fusion::Sm90EVT<fusion::Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, RN>,
      fusion::Sm90AccFetch,
      fusion::Sm90AuxLoad<0, void, ElementC, LayoutC, void, void>>,            // b
    fusion::Sm90EVT<fusion::Sm90Compute<CTSiLuBwd, ElementCompute, ElementCompute, RN>,
      fusion::Sm90AuxLoad<0, void, ElementC, LayoutC, void, void>>>;           // silu'(a)
using GemmDa = BwdGemm<EVT_da>;

// db = multiplies( acc, SiLu(AuxLoad(a)) )
using EVT_db =
  fusion::Sm90EVT<fusion::Sm90Compute<cutlass::multiplies, ElementC, ElementCompute, RN>,
    fusion::Sm90AccFetch,
    fusion::Sm90EVT<fusion::Sm90Compute<cutlass::epilogue::thread::SiLu, ElementCompute, ElementCompute, RN>,
      fusion::Sm90AuxLoad<0, void, ElementC, LayoutC, void, void>>>;           // silu(a)
using GemmDb = BwdGemm<EVT_db>;

template <class G>
static void strides(int M, int N, int K,
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
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "bwd can_implement");
  size_t ws = G::get_workspace_size(args);
  auto wbuf = torch::empty({(int64_t)ws}, torch::dtype(torch::kUInt8).device(dev));
  TORCH_CHECK(gemm.initialize(args, wbuf.data_ptr()) == cutlass::Status::kSuccess, "bwd init");
  TORCH_CHECK(gemm.run(at::cuda::getCurrentCUDAStream()) == cutlass::Status::kSuccess, "bwd run");
}

// da = (dout@Ws^T... no: dh = dout@Ws) ; here A=dout (M,D), B=Ws (ND,D) read as Ws^T? -----
// dh = dout @ Ws.  Ws is (D, ND).  dh:(M,ND) = dout:(M,D) @ Ws:(D,ND).
// Our GEMM computes A@B^T with B stored (N,K) row-major.  We want dh = dout @ Ws, i.e.
// A=dout (M,K=D), Bop=Ws (K=D, N=ND).  Ws is (D,ND) row-major -> to get Bop=(D,ND) we need
// B stored (ND, D) row-major (so B^T = (D,ND)).  So pass Ws^T (ND,D) as B.  Caller passes wsT.
static torch::Tensor fused_da(torch::Tensor dout, torch::Tensor wsT,
                              torch::Tensor a, torch::Tensor b) {
  using G = GemmDa::Gemm;
  int M = dout.size(0), K = dout.size(1), N = wsT.size(0);  // N = ND
  auto da = torch::empty({M, N}, dout.options());
  typename G::GemmKernel::StrideA dA; typename G::GemmKernel::StrideB dB;
  typename G::GemmKernel::StrideC dC; typename G::GemmKernel::StrideD dD;
  strides<G>(M, N, K, dA, dB, dC, dD);
  typename G::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
    {dout.data_ptr<float>(), dA, wsT.data_ptr<float>(), dB},
    { { /*EVT_da*/
        { /*mul(acc,b)*/ {}, { b.data_ptr<float>(), ElementC(0), dD }, {} },
        { /*SiLuBwd(a)*/ { a.data_ptr<float>(), ElementC(0), dD }, {} },
        {} },
      nullptr, dC, da.data_ptr<float>(), dD } };
  launch<G>(args, dout.device());
  return da;
}

static torch::Tensor fused_db(torch::Tensor dout, torch::Tensor wsT, torch::Tensor a) {
  using G = GemmDb::Gemm;
  int M = dout.size(0), K = dout.size(1), N = wsT.size(0);
  auto db = torch::empty({M, N}, dout.options());
  typename G::GemmKernel::StrideA dA; typename G::GemmKernel::StrideB dB;
  typename G::GemmKernel::StrideC dC; typename G::GemmKernel::StrideD dD;
  strides<G>(M, N, K, dA, dB, dC, dD);
  typename G::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
    {dout.data_ptr<float>(), dA, wsT.data_ptr<float>(), dB},
    { { /*EVT_db*/
        {},                                                 // acc
        { { a.data_ptr<float>(), ElementC(0), dD }, {} },   // SiLu(a)
        {} },                                               // multiplies
      nullptr, dC, db.data_ptr<float>(), dD } };
  launch<G>(args, dout.device());
  return db;
}

// ============ DUAL-OUTPUT collapse: one dh-GEMM emits da (D out) + db (AuxStore) =========
// dh computed ONCE. Topological DAG:
//   0 AccFetch(dh) ; 1 AuxLoad(a) ; 2 AuxLoad(b)
//   3 SiLuBwd(a)=silu'(a)      <- {1}
//   4 mul(dh,b)                <- {0,2}
//   5 SiLu(a)=silu(a)          <- {1}
//   6 mul(dh,silu(a))=db       <- {0,5}
//   7 AuxStore(db)             <- {6}   (stores db, passes through)
//   8 mul(node4,node3)=da      <- {4,3} (LAST = D output)
using EVT_dab =
  fusion::Sm90TopologicalVisitor<ElementCompute,
    cute::tuple<
      cute::seq<>, cute::seq<>, cute::seq<>,
      cute::seq<1>,
      cute::seq<0,2>,
      cute::seq<1>,
      cute::seq<0,5>,
      cute::seq<6>,
      cute::seq<4,3>>,
    fusion::Sm90AccFetch,                                               // 0
    fusion::Sm90AuxLoad<0, void, ElementC, LayoutC, void, void>,        // 1 a
    fusion::Sm90AuxLoad<0, void, ElementC, LayoutC, void, void>,        // 2 b
    fusion::Sm90Compute<CTSiLuBwd, ElementCompute, ElementCompute, RN>, // 3
    fusion::Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, RN>, // 4
    fusion::Sm90Compute<cutlass::epilogue::thread::SiLu, ElementCompute, ElementCompute, RN>, // 5
    fusion::Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, RN>, // 6
    fusion::Sm90AuxStore<0, void, ElementC, RN, LayoutC, void, void>,   // 7 store db
    fusion::Sm90Compute<cutlass::multiplies, ElementC, ElementCompute, RN>>;      // 8 da (output)
using GemmDab = BwdGemm<EVT_dab>;

// returns da (D out); db written into the provided db tensor via AuxStore.
static torch::Tensor fused_dab(torch::Tensor dout, torch::Tensor wsT,
                               torch::Tensor a, torch::Tensor b, torch::Tensor db) {
  using G = GemmDab::Gemm;
  int M = dout.size(0), K = dout.size(1), N = wsT.size(0);  // N = ND
  auto da = torch::empty({M, N}, dout.options());
  typename G::GemmKernel::StrideA dA; typename G::GemmKernel::StrideB dB;
  typename G::GemmKernel::StrideC dC; typename G::GemmKernel::StrideD dD;
  strides<G>(M, N, K, dA, dB, dC, dD);
  typename G::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
    {dout.data_ptr<float>(), dA, wsT.data_ptr<float>(), dB},
    { { /*EVT_dab topological args, in op order 0..8*/
        {},                                            // 0 AccFetch
        { a.data_ptr<float>(), ElementC(0), dD },      // 1 AuxLoad a
        { b.data_ptr<float>(), ElementC(0), dD },      // 2 AuxLoad b
        {},                                            // 3 SiLuBwd
        {},                                            // 4 mul
        {},                                            // 5 SiLu
        {},                                            // 6 mul
        { db.data_ptr<float>(), dD },                  // 7 AuxStore db
        {} },                                          // 8 mul -> da
      nullptr, dC, da.data_ptr<float>(), dD } };
  launch<G>(args, dout.device());
  return da;
}

// PACKED dual-output: write da -> dab[:, :ND] and db -> dab[:, ND:] directly (row pitch 2ND),
// eliminating the torch.cat. a,b may be STRIDED views (e.g. halves of a packed ab tensor) — we
// read their actual stride(0) row pitch, so no .contiguous() copy is needed.
static void fused_dab_packed(torch::Tensor dout, torch::Tensor wsT,
                             torch::Tensor a, torch::Tensor b, torch::Tensor dab) {
  using G = GemmDab::Gemm;
  int M = dout.size(0), K = dout.size(1), N = wsT.size(0);  // N = ND
  int ND2 = dab.size(1);                                    // 2*ND
  typename G::GemmKernel::StrideA dA; typename G::GemmKernel::StrideB dB;
  typename G::GemmKernel::StrideC dC; typename G::GemmKernel::StrideD dD;
  strides<G>(M, N, K, dA, dB, dC, dD);
  using SAux = cutlass::detail::TagToStrideC_t<LayoutC>;
  SAux dA_a = SAux{(int)a.stride(0), cute::_1{}, 0};  // a row pitch (ND if contiguous, 2ND if view)
  SAux dA_b = SAux{(int)b.stride(0), cute::_1{}, 0};
  SAux dPack = SAux{ND2, cute::_1{}, 0};              // da/db row pitch = 2ND
  float* da_ptr = dab.data_ptr<float>();             // dab[:, :ND]
  float* db_ptr = dab.data_ptr<float>() + N;         // dab[:, ND:]
  typename G::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
    {dout.data_ptr<float>(), dA, wsT.data_ptr<float>(), dB},
    { { {},                                            // 0 AccFetch
        { a.data_ptr<float>(), ElementC(0), dA_a },    // 1 AuxLoad a
        { b.data_ptr<float>(), ElementC(0), dA_b },    // 2 AuxLoad b
        {}, {}, {}, {},                                // 3-6
        { db_ptr, dPack },                             // 7 AuxStore db -> dab[:,ND:]
        {} },                                          // 8 mul -> da
      nullptr, dC, da_ptr, dPack } };                  // D output da -> dab[:,:ND]
  launch<G>(args, dout.device());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_da", &fused_da, "da = (dout@Ws) * b * silu'(a)  [input-fused]");
  m.def("fused_db", &fused_db, "db = (dout@Ws) * silu(a)       [input-fused]");
  m.def("fused_dab", &fused_dab, "da=out, db via AuxStore; dh computed once [dual-output]");
  m.def("fused_dab_packed", &fused_dab_packed, "write da,db directly into packed dab (no cat)");
}

} // namespace ct
