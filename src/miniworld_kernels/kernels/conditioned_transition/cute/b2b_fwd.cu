// TRUE b2b fused forward for the ConditionedTransition tail (atom: d=128), ONE launch.
//   a=x@Wa^T, b=x@Wb^T ; h=silu(a)*b ; out=h@Ws^T ; scale=cond@Wsc^T+bsc ; y=sigmoid(scale)*out
// h and out_acc never leave smem/registers. Hand-written cute SM90 TF32 WGMMA, single warpgroup,
// cp.async loads, dynamic smem. Proven M0 mechanics (cos=1.0) extended to the full chain.
//
// Per CTA: owns BLOCK_M rows, ALL of ND/D/DC. Loops ND in BLOCK_N chunks building gated h and
// accumulating out_acc[BM,D] in registers; then the cond-gate; one HBM write of y.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/arch/barrier.h>   // fence_view_async_shared

using namespace cute;
using TF = tfloat32_t;

// cp.async load a (ROWS, COLS) K-major (COLS contiguous) gmem tile -> swizzled smem.
// 128-bit vectorized: each thread copies 4 contiguous fp32 (tf32) along the K(col) dim.
template <class GTensor, class STensor>
__device__ inline void load_tile(GTensor const& g, STensor s, int tid) {  // s by value (view)
  // 128 threads (1 warpgroup), 128-bit vectorized: 32 rows x 4 col-groups x (1 x 4 vals).
  auto cp = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, TF>{},
                            Layout<Shape<_32,_4>, Stride<_4,_1>>{},   // 128 threads
                            Layout<Shape<_1,_4>>{});                  // 4 elems/thread (K contiguous)
  auto thr = cp.get_slice(tid);
  copy(thr.partition_S(g), thr.partition_D(s));
}

template <int BM, int K_, int ND_, int D_, int DC_, int BN_, int BDC_>
__global__ static __launch_bounds__(128)
void b2b_kernel(const float* __restrict__ X, const float* __restrict__ Wa,
                const float* __restrict__ Wb, const float* __restrict__ Ws,
                const float* __restrict__ Cond, const float* __restrict__ Wsc,
                const float* __restrict__ Bsc, float* __restrict__ Y, int M) {
  constexpr int K = K_, ND = ND_, D = D_, DC = DC_, BN = BN_, BDC = BDC_;
  int tid = threadIdx.x;

  // ---- gmem tensors (all row-major) ----
  Tensor mX  = make_tensor(make_gmem_ptr(reinterpret_cast<const TF*>(X)),  make_shape(M, Int<K>{}),  make_stride(Int<K>{}, _1{}));
  Tensor mWa = make_tensor(make_gmem_ptr(reinterpret_cast<const TF*>(Wa)), make_shape(Int<ND>{}, Int<K>{}), make_stride(Int<K>{}, _1{}));
  Tensor mWb = make_tensor(make_gmem_ptr(reinterpret_cast<const TF*>(Wb)), make_shape(Int<ND>{}, Int<K>{}), make_stride(Int<K>{}, _1{}));
  Tensor mWs = make_tensor(make_gmem_ptr(reinterpret_cast<const TF*>(Ws)), make_shape(Int<D>{}, Int<ND>{}), make_stride(Int<ND>{}, _1{}));
  Tensor mC  = make_tensor(make_gmem_ptr(reinterpret_cast<const TF*>(Cond)),make_shape(M, Int<DC>{}), make_stride(Int<DC>{}, _1{}));
  Tensor mWsc= make_tensor(make_gmem_ptr(reinterpret_cast<const TF*>(Wsc)),make_shape(Int<D>{}, Int<DC>{}), make_stride(Int<DC>{}, _1{}));
  Tensor mY  = make_tensor(make_gmem_ptr(Y), make_shape(M, Int<D>{}), make_stride(Int<D>{}, _1{}));

  Tensor gX = local_tile(mX, make_shape(Int<BM>{}, Int<K>{}), make_coord(blockIdx.x, 0));   // (BM,K)
  Tensor gY = local_tile(mY, make_shape(Int<BM>{}, Int<D>{}), make_coord(blockIdx.x, 0));   // (BM,D)
  Tensor gCond = local_tile(mC, make_shape(Int<BM>{}, Int<DC>{}), make_coord(blockIdx.x, 0)); // (BM,DC)

  // ---- smem layouts (K-major swizzled) ----
  auto lX  = tile_to_shape(GMMA::Layout_K_SW128_Atom<TF>{}, make_shape(Int<BM>{}, Int<K>{}));
  auto lW  = tile_to_shape(GMMA::Layout_K_SW128_Atom<TF>{}, make_shape(Int<BN>{}, Int<K>{}));   // Wa/Wb chunk (BN,K)
  auto lH  = tile_to_shape(GMMA::Layout_K_SW128_Atom<TF>{}, make_shape(Int<BM>{}, Int<BN>{}));  // h chunk (BM,BN), BN-major
  auto lWs = tile_to_shape(GMMA::Layout_K_SW128_Atom<TF>{}, make_shape(Int<D>{}, Int<BN>{}));   // Ws chunk (D,BN), BN-major
  auto lC  = tile_to_shape(GMMA::Layout_K_SW128_Atom<TF>{}, make_shape(Int<BM>{}, Int<BDC>{})); // cond chunk (BM,BDC)
  auto lWsc= tile_to_shape(GMMA::Layout_K_SW128_Atom<TF>{}, make_shape(Int<D>{}, Int<BDC>{}));  // Wsc chunk (D,BDC)

  // BEST config (44us): single-stage smem, serial per-chunk loop, 128-bit cp.async. The WGMMA
  // software-pipeline (overlap squeeze(c) with expand(c+1)) and 2-warpgroup variants both
  // REGRESSED (66us / 56us) — register pressure from ring rmem + per-chunk smem syncs outweigh
  // the small overlap window at 4 ND chunks. Beating triton-b2b's 25us needs the full TMA
  // warp-specialized producer/consumer collective (separate load/MMA warpgroups), not this.
  extern __shared__ char smem_raw[];
  int off = 0;
  auto take = [&](int n){ TF* p = reinterpret_cast<TF*>(smem_raw + off); off += n * (int)sizeof(TF); return p; };
  TF* pX  = take(cosize_v<decltype(lX)>);
  TF* pWa = take(cosize_v<decltype(lW)>);
  TF* pWb = take(cosize_v<decltype(lW)>);
  TF* pWs = take(cosize_v<decltype(lWs)>);
  TF* pH  = take(cosize_v<decltype(lH)>);
  TF* pWsc = pX;                              // alias: sWsc onto sX — x dead after ND loop
  TF* pC   = pH;                              // alias: sC onto sH — h dead after ND loop
  Tensor sX  = make_tensor(make_smem_ptr(pX), lX);
  Tensor sWa = make_tensor(make_smem_ptr(pWa), lW);
  Tensor sWb = make_tensor(make_smem_ptr(pWb), lW);
  Tensor sWs = make_tensor(make_smem_ptr(pWs), lWs);
  Tensor sH  = make_tensor(make_smem_ptr(pH), lH);
  Tensor sC  = make_tensor(make_smem_ptr(pC), lC);
  Tensor sWsc= make_tensor(make_smem_ptr(pWsc), lWsc);

  // ---- MMAs: single warpgroup (BM=64, 128 threads) ----
  TiledMMA mmaE = make_tiled_mma(SM90_64x64x8_F32TF32TF32_SS_TN<>{});
  TiledMMA mmaS = make_tiled_mma(SM90_64x128x8_F32TF32TF32_SS_TN<>{});
  auto teE = mmaE.get_slice(tid);
  auto teS = mmaS.get_slice(tid);

  // out_acc[BM,D] register-resident
  Tensor tSgY = teS.partition_C(gY);
  Tensor out_acc = teS.make_fragment_C(tSgY);
  clear(out_acc);

  cutlass::epilogue::thread::SiLu<float> silu;
  constexpr int NCHUNK = ND / BN;

  // load x once
  load_tile(gX, sX, tid);
  cp_async_fence(); cp_async_wait<0>(); __syncthreads();

  // ---- ND loop: serial expand -> swiglu -> squeeze (best measured config, 44-46us) ----
  CUTE_NO_UNROLL
  for (int c = 0; c < NCHUNK; ++c) {
    Tensor gWa = local_tile(mWa, make_shape(Int<BN>{}, Int<K>{}), make_coord(c, 0));
    Tensor gWb = local_tile(mWb, make_shape(Int<BN>{}, Int<K>{}), make_coord(c, 0));
    Tensor gWs = local_tile(mWs, make_shape(Int<D>{},  Int<BN>{}), make_coord(0, c));
    load_tile(gWa, sWa, tid); load_tile(gWb, sWb, tid); load_tile(gWs, sWs, tid);
    cp_async_fence(); cp_async_wait<0>(); __syncthreads();

    Tensor rA = partition_fragment_C(mmaE, make_shape(Int<BM>{}, Int<BN>{}));
    Tensor rB = make_fragment_like(rA);
    clear(rA); clear(rB);
    Tensor tX  = teE.make_fragment_A(teE.partition_A(sX));
    Tensor tWa = teE.make_fragment_B(teE.partition_B(sWa));
    Tensor tWb = teE.make_fragment_B(teE.partition_B(sWb));
    warpgroup_fence_operand(rA);
    warpgroup_arrive();
    cute::gemm(mmaE, tX, tWa, rA);
    cute::gemm(mmaE, tX, tWb, rB);
    warpgroup_commit_batch(); warpgroup_wait<0>();
    warpgroup_fence_operand(rA); warpgroup_fence_operand(rB);

    CUTE_UNROLL
    for (int i = 0; i < size(rA); ++i) rA(i) = silu(rA(i)) * rB(i);
    copy(rA, teE.partition_C(sH));
    __syncthreads();

    Tensor tH  = teS.make_fragment_A(teS.partition_A(sH));
    Tensor tWs = teS.make_fragment_B(teS.partition_B(sWs));
    warpgroup_fence_operand(out_acc);
    warpgroup_arrive();
    cute::gemm(mmaS, tH, tWs, out_acc);
    warpgroup_commit_batch(); warpgroup_wait<0>();
    warpgroup_fence_operand(out_acc);
    __syncthreads();
  }

  // ---- scale = cond @ Wsc^T + bsc  (tiled DC) ; y = sigmoid(scale)*out ----
  Tensor scale_acc = make_fragment_like(out_acc);
  clear(scale_acc);
  CUTE_NO_UNROLL
  for (int c0 = 0; c0 < DC; c0 += BDC) {
    Tensor gC2  = local_tile(gCond, make_shape(Int<BM>{}, Int<BDC>{}), make_coord(0, c0 / BDC)); // (BM,BDC)
    Tensor gWsc = local_tile(mWsc, make_shape(Int<D>{}, Int<BDC>{}), make_coord(0, c0 / BDC));   // (D,BDC)
    load_tile(gC2, sC, tid); load_tile(gWsc, sWsc, tid);
    cp_async_fence(); cp_async_wait<0>(); __syncthreads();
    Tensor tScC   = teS.make_fragment_A(teS.partition_A(sC));
    Tensor tScWsc = teS.make_fragment_B(teS.partition_B(sWsc));
    warpgroup_fence_operand(scale_acc);
    warpgroup_arrive();
    cute::gemm(mmaS, tScC, tScWsc, scale_acc);
    warpgroup_commit_batch(); warpgroup_wait<0>();
    warpgroup_fence_operand(scale_acc);
    __syncthreads();
  }

  // y = sigmoid(scale + bsc) * out_acc.  bsc is per-D-column; map fragment coord -> D index.
  Tensor cD = teS.partition_C(make_identity_tensor(make_shape(Int<BM>{}, Int<D>{})));  // coords
  cutlass::epilogue::thread::Sigmoid<float> sig;
  CUTE_UNROLL
  for (int i = 0; i < size(out_acc); ++i) {
    int d = get<1>(cD(i));                 // column (D) index
    float sc = scale_acc(i) + (float)Bsc[d];
    out_acc(i) = sig(sc) * out_acc(i);
  }
  copy(out_acc, tSgY);
}

torch::Tensor b2b_forward(torch::Tensor x, torch::Tensor cond, torch::Tensor wa,
                          torch::Tensor wb, torch::Tensor ws, torch::Tensor wsc, torch::Tensor bsc) {
  for (auto& t : {x, cond, wa, wb, ws, wsc, bsc})
    TORCH_CHECK(t.is_cuda() && t.dtype() == torch::kFloat32, "fp32 cuda");
  x = x.contiguous(); cond = cond.contiguous(); wa = wa.contiguous(); wb = wb.contiguous();
  ws = ws.contiguous(); wsc = wsc.contiguous(); bsc = bsc.contiguous();
  int M = x.size(0), K = x.size(1), ND = wa.size(0), D = ws.size(0), DC = cond.size(1);
  TORCH_CHECK(K == 128 && ND == 256 && D == 128 && DC == 384, "atom dims fixed");
  auto y = torch::empty({M, D}, x.options());
  // BEST config: BM=64 single warpgroup, single-stage smem, serial loop = ~44-46us (0.57x triton-b2b).
  // (R1 2-stage Wa/Wb, R2 2-warpgroup, and the WGMMA software-pipeline all measured no better/worse.)
  constexpr int BM = 64, BN = 64, BDC = 64;
  // smem: sX + sWa + sWb + sWs + sH ; sC/sWsc alias sX/sH (cond runs after ND loop).
  int smem = ((BM*128) + (BN*128)*2 + (D*BN) + (BM*BN)) * (int)sizeof(tfloat32_t);
  auto kern = b2b_kernel<BM,128,256,128,384,BN,BDC>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  dim3 grid((M + BM - 1) / BM);
  kern<<<grid, 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), wa.data_ptr<float>(), wb.data_ptr<float>(), ws.data_ptr<float>(),
      cond.data_ptr<float>(), wsc.data_ptr<float>(), bsc.data_ptr<float>(), y.data_ptr<float>(), M);
  C10_CUDA_CHECK(cudaGetLastError());
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("b2b_forward", &b2b_forward, "fused b2b forward (atom)");
}
