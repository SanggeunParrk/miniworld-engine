// Fused transition-forward b2b kernel for NVIDIA B200 (sm_100), raw CUDA C++/CuTe.
// Ported from the Hopper hand-CUDA b2b (transition/cuda/transition_b2b_kernel.cu).
//
// Math (pre-normalized xn):
//   a = xn@wa^T ; b = xn@wb^T ; h = silu(a)*b = a*sigmoid(a)*b (fp32->bf16)
//   out = h@ws^T -> (M,D) bf16
// xn:(M,K) bf16, wa/wb:(ND,K) bf16, ws:(D,ND) bf16, out:(M,D) bf16.
//
// Blackwell has NO register-source MMA: the squeeze A operand (h) round-trips
// TMEM(acc) -> regs (tcgen05.ld) -> silu -> smem (sH) -> squeeze MMA reads sH (SS).
// This is the correctness-first (single-stage, serialized) port; perf comes later.
//
// TRANSITION_B2B_STAGE == 1 : expand+silu only, emits h[M,ND]  (de-risks tcgen05 path)
// TRANSITION_B2B_STAGE == 2 : full fused b2b, emits out[M,D]

#ifndef TRANSITION_B2B_STAGE
#define TRANSITION_B2B_STAGE 2
#endif

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>

namespace b2bsm100 {

using namespace cute;
using BF = cutlass::bfloat16_t;

inline void check_cuda(cudaError_t e, const char* msg) {
    TORCH_CHECK(e == cudaSuccess, msg, ": ", cudaGetErrorString(e));
}

__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + __expf(-x)); }

// ------------------------------------------------------------------------------------------------
// Stage 1: expand + SwiGLU. h = silu(xn@wa^T) * (xn@wb^T) -> h[M,ND].
// ------------------------------------------------------------------------------------------------
#if TRANSITION_B2B_STAGE == 1

template <int CTA_M, int K, int ND, int D, int BN,
          class MmaTilerE, class TiledMmaE,
          class SXnLayout, class SWLayout,
          class TmaXn, class TmaWa, class TmaWb>
__global__ static __launch_bounds__(128) void expand_silu_kernel(
    CUTE_GRID_CONSTANT TmaXn const tma_xn,
    CUTE_GRID_CONSTANT TmaWa const tma_wa,
    CUTE_GRID_CONSTANT TmaWb const tma_wb,
    __nv_bfloat16* __restrict__ h_raw, int M,
    MmaTilerE mma_tiler_e, TiledMmaE mma_e,
    SXnLayout sXn_layout, SWLayout sW_layout) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    constexpr int NCHUNK = ND / BN;
    const int m_tile = blockIdx.x;

    extern __shared__ char smem_raw[];
    char* p = smem_raw;
    BF* pXn = reinterpret_cast<BF*>(p);           p += cosize_v<SXnLayout> * sizeof(BF);
    BF* pWa = reinterpret_cast<BF*>(p);           p += cosize_v<SWLayout>  * sizeof(BF);
    BF* pWb = reinterpret_cast<BF*>(p);           p += cosize_v<SWLayout>  * sizeof(BF);
    // align barriers to 16
    uintptr_t pb = (reinterpret_cast<uintptr_t>(p) + 15) & ~uintptr_t(15);
    uint64_t* xn_bar = reinterpret_cast<uint64_t*>(pb); pb += 16;
    uint64_t* tma_bar = reinterpret_cast<uint64_t*>(pb); pb += 16;
    uint64_t* mma_bar = reinterpret_cast<uint64_t*>(pb); pb += 16;
    uint32_t* tmem_base = reinterpret_cast<uint32_t*>(pb); pb += 16;

    Tensor sXn = make_tensor(make_smem_ptr(pXn), sXn_layout)(_, _, _, Int<0>{});  // drop PIPE
    Tensor sWa = make_tensor(make_smem_ptr(pWa), sW_layout)(_, _, _, Int<0>{});
    Tensor sWb = make_tensor(make_smem_ptr(pWb), sW_layout)(_, _, _, Int<0>{});

    auto mXn = tma_xn.get_tma_tensor(make_shape(M, Int<K>{}));
    auto mWa = tma_wa.get_tma_tensor(make_shape(Int<ND>{}, Int<K>{}));
    auto mWb = tma_wb.get_tma_tensor(make_shape(Int<ND>{}, Int<K>{}));
    auto mH  = make_tensor(make_gmem_ptr(reinterpret_cast<BF*>(h_raw)),
                           make_shape(M, Int<ND>{}), make_stride(Int<ND>{}, Int<1>{}));

    auto cta_mma = mma_e.get_slice(0);

    Tensor gXn = local_tile(mXn, mma_tiler_e, make_coord(m_tile, 0, 0), Step<_1, X, _1>{}); // (CTA_M,K)
    Tensor tCgXn = cta_mma.partition_A(gXn);
    Tensor tCrXn = cta_mma.make_fragment_A(sXn);
    Tensor tCrWa = cta_mma.make_fragment_B(sWa);
    Tensor tCrWb = cta_mma.make_fragment_B(sWb);

    // TMEM accumulators
    Tensor cAB = make_identity_tensor(make_shape(Int<CTA_M>{}, Int<BN>{}));
    Tensor tCcAB = cta_mma.partition_C(cAB);
    Tensor a_acc = cta_mma.make_fragment_C(tCcAB);
    Tensor b_acc = cta_mma.make_fragment_C(tCcAB);

    uint32_t elect_thr = cute::elect_one_sync();
    uint32_t elect_warp = (threadIdx.x / 32 == 0);
    using TmemAlloc = cute::TMEM::Allocator1Sm;
    TmemAlloc tmem_allocator{};
    if (elect_warp) tmem_allocator.allocate(TmemAlloc::Sm100TmemCapacityColumns, tmem_base);
    if (elect_warp && elect_thr) {
        cute::initialize_barrier(*xn_bar, 1);
        cute::initialize_barrier(*tma_bar, 1);
        cute::initialize_barrier(*mma_bar, 1);
    }
    __syncthreads();
    uint32_t tmem_ptr = *tmem_base;
    a_acc.data() = tmem_ptr;
    b_acc.data() = tmem_ptr + BN;

    // TMA partitions (smem side fixed)
    auto [tXgXn, tXsXn] = tma_partition(tma_xn, Int<0>{}, Layout<_1>{},
                                        group_modes<0, 3>(sXn), group_modes<0, 3>(tCgXn));
    int xn_bytes = sizeof(make_tensor_like(tXsXn));

    // Load xn once
    if (elect_warp && elect_thr) {
        cute::set_barrier_transaction_bytes(*xn_bar, xn_bytes);
        copy(tma_xn.with(*xn_bar), tXgXn, tXsXn);
    }
    cute::wait_barrier(*xn_bar, 0);

    int tma_phase = 0, mma_phase = 0;

    // t2r copy for the expand accumulators
    TiledCopy t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, a_acc);
    ThrCopy thr_t2r = t2r.get_slice(threadIdx.x);
    Tensor tDtA = thr_t2r.partition_S(a_acc);
    Tensor tDtB = thr_t2r.partition_S(b_acc);

    for (int c = 0; c < NCHUNK; ++c) {
        Tensor gWa = local_tile(mWa, mma_tiler_e, make_coord(m_tile, c, 0), Step<X, _1, _1>{}); // (BN,K)
        Tensor gWb = local_tile(mWb, mma_tiler_e, make_coord(m_tile, c, 0), Step<X, _1, _1>{});
        Tensor tCgWa = cta_mma.partition_B(gWa);
        Tensor tCgWb = cta_mma.partition_B(gWb);
        auto [tWagWa, tWasWa] = tma_partition(tma_wa, Int<0>{}, Layout<_1>{},
                                              group_modes<0, 3>(sWa), group_modes<0, 3>(tCgWa));
        auto [tWbgWb, tWbsWb] = tma_partition(tma_wb, Int<0>{}, Layout<_1>{},
                                              group_modes<0, 3>(sWb), group_modes<0, 3>(tCgWb));
        int w_bytes = sizeof(make_tensor_like(tWasWa)) + sizeof(make_tensor_like(tWbsWb));

        if (elect_warp && elect_thr) {
            cute::set_barrier_transaction_bytes(*tma_bar, w_bytes);
            copy(tma_wa.with(*tma_bar), tWagWa, tWasWa);
            copy(tma_wb.with(*tma_bar), tWbgWb, tWbsWb);
        }
        cute::wait_barrier(*tma_bar, tma_phase); tma_phase ^= 1;

        if (elect_warp) {
            mma_e.accumulate_ = UMMA::ScaleOut::Zero;
            CUTE_UNROLL
            for (int kb = 0; kb < size<2>(tCrXn); ++kb) {
                gemm(mma_e, tCrXn(_, _, kb), tCrWa(_, _, kb), a_acc);
                mma_e.accumulate_ = UMMA::ScaleOut::One;
            }
            mma_e.accumulate_ = UMMA::ScaleOut::Zero;
            CUTE_UNROLL
            for (int kb = 0; kb < size<2>(tCrXn); ++kb) {
                gemm(mma_e, tCrXn(_, _, kb), tCrWb(_, _, kb), b_acc);
                mma_e.accumulate_ = UMMA::ScaleOut::One;
            }
            cutlass::arch::umma_arrive(mma_bar);
        }
        cute::wait_barrier(*mma_bar, mma_phase); mma_phase ^= 1;

        // TMEM -> RMEM, silu, store to gH
        Tensor gH = local_tile(mH, make_shape(Int<CTA_M>{}, Int<BN>{}), make_coord(m_tile, c)); // (CTA_M,BN)
        Tensor tCgH = cta_mma.partition_C(gH);
        Tensor tDgH = thr_t2r.partition_D(tCgH);
        Tensor rA = make_tensor<float>(shape(tDgH));
        Tensor rB = make_tensor<float>(shape(tDgH));
        copy(t2r, tDtA, rA);
        copy(t2r, tDtB, rB);
        Tensor rH = make_tensor<BF>(shape(tDgH));
        CUTE_UNROLL
        for (int i = 0; i < size(rH); ++i) {
            float a = rA(i), b = rB(i);
            rH(i) = static_cast<BF>(a * sigmoidf(a) * b);
        }
        copy(rH, tDgH);
        __syncthreads();  // ensure acc read done before next chunk overwrites
    }

    if (elect_warp) {
        tmem_allocator.release_allocation_lock();
        tmem_allocator.free(tmem_ptr, TmemAlloc::Sm100TmemCapacityColumns);
    }
#else
    (void)tma_xn; (void)tma_wa; (void)tma_wb; (void)h_raw; (void)M;
    (void)mma_tiler_e; (void)mma_e; (void)sXn_layout; (void)sW_layout;
#endif
}

template <int CTA_M, int K, int ND, int D, int BN>
void launch_expand_silu(const torch::Tensor& xn, const torch::Tensor& wa,
                        const torch::Tensor& wb, torch::Tensor& h, int M, cudaStream_t stream) {
    auto mma_e = make_tiled_mma(
        SM100_MMA_F16BF16_SS<BF, BF, float, CTA_M, BN, UMMA::Major::K, UMMA::Major::K>{});
    auto mma_tiler_e = make_shape(Int<CTA_M>{}, Int<BN>{}, Int<K>{});

    auto mma_shape_A = partition_shape_A(mma_e, make_shape(Int<CTA_M>{}, Int<K>{}));
    auto mma_shape_B = partition_shape_B(mma_e, make_shape(Int<BN>{}, Int<K>{}));
    // append a trivial PIPE=1 mode so Step<_1,_2,_3> tiles K-first -> canonical UMMA-K layout
    auto sXn_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<BF>{},
                                              append(mma_shape_A, Int<1>{}), Step<_1, _2, _3>{});
    auto sW_layout  = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<BF>{},
                                              append(mma_shape_B, Int<1>{}), Step<_1, _2, _3>{});

    auto mXn = make_tensor(make_gmem_ptr(reinterpret_cast<const BF*>(xn.data_ptr<at::BFloat16>())),
                           make_shape(M, Int<K>{}), make_stride(Int<K>{}, Int<1>{}));
    auto mWa = make_tensor(make_gmem_ptr(reinterpret_cast<const BF*>(wa.data_ptr<at::BFloat16>())),
                           make_shape(Int<ND>{}, Int<K>{}), make_stride(Int<K>{}, Int<1>{}));
    auto mWb = make_tensor(make_gmem_ptr(reinterpret_cast<const BF*>(wb.data_ptr<at::BFloat16>())),
                           make_shape(Int<ND>{}, Int<K>{}), make_stride(Int<K>{}, Int<1>{}));

    auto tma_xn = make_tma_atom(SM90_TMA_LOAD{}, mXn, sXn_layout, select<0, 2>(mma_tiler_e));
    auto tma_wa = make_tma_atom(SM90_TMA_LOAD{}, mWa, sW_layout, select<1, 2>(mma_tiler_e));
    auto tma_wb = make_tma_atom(SM90_TMA_LOAD{}, mWb, sW_layout, select<1, 2>(mma_tiler_e));

    int smem = int(cosize_v<decltype(sXn_layout)> + 2 * cosize_v<decltype(sW_layout)>) * int(sizeof(BF)) + 256;

    auto* kern = &expand_silu_kernel<CTA_M, K, ND, D, BN,
                                     decltype(mma_tiler_e), decltype(mma_e),
                                     decltype(sXn_layout), decltype(sW_layout),
                                     decltype(tma_xn), decltype(tma_wa), decltype(tma_wb)>;
    check_cuda(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem),
               "expand_silu smem attr");
    dim3 grid(M / CTA_M);
    kern<<<grid, 128, smem, stream>>>(tma_xn, tma_wa, tma_wb,
                                      reinterpret_cast<__nv_bfloat16*>(h.data_ptr<at::BFloat16>()), M,
                                      mma_tiler_e, mma_e, sXn_layout, sW_layout);
    check_cuda(cudaGetLastError(), "expand_silu launch");
}

torch::Tensor expand_silu_fwd(const torch::Tensor& xn, const torch::Tensor& wa, const torch::Tensor& wb) {
    c10::cuda::CUDAGuard g(xn.device());
    int64_t M = xn.size(0), K = xn.size(1), ND = wa.size(0);
    auto h = torch::empty({M, ND}, xn.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (K == 128 && ND == 512) {
        launch_expand_silu<128, 128, 512, 128, 128>(xn, wa, wb, h, M, stream);
    } else if (K == 256 && ND == 1024) {
        launch_expand_silu<128, 256, 1024, 256, 64>(xn, wa, wb, h, M, stream);
    } else {
        TORCH_CHECK(false, "unsupported shape");
    }
    return h;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("expand_silu_fwd", &expand_silu_fwd, "expand+silu -> h (sm100 stage1)");
}

#endif  // STAGE 1

// ------------------------------------------------------------------------------------------------
// Stage 2: full fused b2b. out = (silu(xn@wa^T)*(xn@wb^T)) @ ws^T -> out[M,D].
// h round-trips TMEM(acc)->regs->silu->smem(sH); squeeze MMA reads sH (SS).
// ------------------------------------------------------------------------------------------------
#if TRANSITION_B2B_STAGE == 2

template <int CTA_M, int K, int ND, int D, int BN,
          class MmaTilerE, class TiledMmaE, class MmaTilerS, class TiledMmaS,
          class SXnLayout, class SWLayout, class SWsLayout, class SHmmaLayout, class SHmnLayout,
          class TmaXn, class TmaWa, class TmaWb, class TmaWs>
__global__ static __launch_bounds__(128) void b2b_fused_kernel(
    CUTE_GRID_CONSTANT TmaXn const tma_xn,
    CUTE_GRID_CONSTANT TmaWa const tma_wa,
    CUTE_GRID_CONSTANT TmaWb const tma_wb,
    CUTE_GRID_CONSTANT TmaWs const tma_ws,
    __nv_bfloat16* __restrict__ out_raw, int M,
    MmaTilerE mma_tiler_e, TiledMmaE mma_e, MmaTilerS mma_tiler_s, TiledMmaS mma_s,
    SXnLayout sXn_layout, SWLayout sW_layout, SWsLayout sWs_layout,
    SHmmaLayout sH_mma_layout, SHmnLayout sH_mn_layout) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    constexpr int NCHUNK = ND / BN;
    const int m_tile = blockIdx.x;

    extern __shared__ char smem_raw[];
    char* p = smem_raw;
    BF* pXn = reinterpret_cast<BF*>(p); p += cosize_v<SXnLayout> * sizeof(BF);
    BF* pWa = reinterpret_cast<BF*>(p); p += cosize_v<SWLayout>  * sizeof(BF);
    BF* pWb = reinterpret_cast<BF*>(p); p += cosize_v<SWLayout>  * sizeof(BF);
    BF* pWs = reinterpret_cast<BF*>(p); p += cosize_v<SWsLayout> * sizeof(BF);
    BF* pH  = reinterpret_cast<BF*>(p); p += cosize_v<SHmmaLayout> * sizeof(BF);
    uintptr_t pb = (reinterpret_cast<uintptr_t>(p) + 15) & ~uintptr_t(15);
    uint64_t* xn_bar   = reinterpret_cast<uint64_t*>(pb); pb += 16;
    uint64_t* tma_bar  = reinterpret_cast<uint64_t*>(pb); pb += 16;
    uint64_t* mma_bar_e = reinterpret_cast<uint64_t*>(pb); pb += 16;
    uint64_t* mma_bar_s = reinterpret_cast<uint64_t*>(pb); pb += 16;
    uint32_t* tmem_base = reinterpret_cast<uint32_t*>(pb); pb += 16;

    Tensor sXn = make_tensor(make_smem_ptr(pXn), sXn_layout)(_, _, _, Int<0>{});
    Tensor sWa = make_tensor(make_smem_ptr(pWa), sW_layout)(_, _, _, Int<0>{});
    Tensor sWb = make_tensor(make_smem_ptr(pWb), sW_layout)(_, _, _, Int<0>{});
    Tensor sWs = make_tensor(make_smem_ptr(pWs), sWs_layout)(_, _, _, Int<0>{});
    Tensor sH  = make_tensor(make_smem_ptr(pH), sH_mma_layout)(_, _, _, Int<0>{});
    Tensor sH_mn = make_tensor(make_smem_ptr(pH), sH_mn_layout);  // (CTA_M,BN) view for scatter

    auto mXn = tma_xn.get_tma_tensor(make_shape(M, Int<K>{}));
    auto mWa = tma_wa.get_tma_tensor(make_shape(Int<ND>{}, Int<K>{}));
    auto mWb = tma_wb.get_tma_tensor(make_shape(Int<ND>{}, Int<K>{}));
    auto mWs = tma_ws.get_tma_tensor(make_shape(Int<D>{}, Int<ND>{}));
    auto mOut = make_tensor(make_gmem_ptr(reinterpret_cast<BF*>(out_raw)),
                            make_shape(M, Int<D>{}), make_stride(Int<D>{}, Int<1>{}));

    auto cta_e = mma_e.get_slice(0);
    auto cta_s = mma_s.get_slice(0);

    Tensor gXn = local_tile(mXn, mma_tiler_e, make_coord(m_tile, 0, 0), Step<_1, X, _1>{});
    Tensor tCgXn = cta_e.partition_A(gXn);
    Tensor tCrXn = cta_e.make_fragment_A(sXn);
    Tensor tCrWa = cta_e.make_fragment_B(sWa);
    Tensor tCrWb = cta_e.make_fragment_B(sWb);
    Tensor tCrH  = cta_s.make_fragment_A(sH);
    Tensor tCrWs = cta_s.make_fragment_B(sWs);

    // TMEM accumulators: a@0, b@BN, out@2*BN
    Tensor cAB = make_identity_tensor(make_shape(Int<CTA_M>{}, Int<BN>{}));
    Tensor tCcAB = cta_e.partition_C(cAB);
    Tensor a_acc = cta_e.make_fragment_C(tCcAB);
    Tensor b_acc = cta_e.make_fragment_C(tCcAB);
    Tensor cO = make_identity_tensor(make_shape(Int<CTA_M>{}, Int<D>{}));
    Tensor tCcO = cta_s.partition_C(cO);
    Tensor out_acc = cta_s.make_fragment_C(tCcO);

    uint32_t elect_thr = cute::elect_one_sync();
    uint32_t elect_warp = (threadIdx.x / 32 == 0);
    using TmemAlloc = cute::TMEM::Allocator1Sm;
    TmemAlloc tmem_allocator{};
    if (elect_warp) tmem_allocator.allocate(TmemAlloc::Sm100TmemCapacityColumns, tmem_base);
    if (elect_warp && elect_thr) {
        cute::initialize_barrier(*xn_bar, 1);
        cute::initialize_barrier(*tma_bar, 1);
        cute::initialize_barrier(*mma_bar_e, 1);
        cute::initialize_barrier(*mma_bar_s, 1);
    }
    __syncthreads();
    uint32_t tmem_ptr = *tmem_base;
    a_acc.data() = tmem_ptr;
    b_acc.data() = tmem_ptr + BN;
    out_acc.data() = tmem_ptr + 2 * BN;

    auto [tXgXn, tXsXn] = tma_partition(tma_xn, Int<0>{}, Layout<_1>{},
                                        group_modes<0, 3>(sXn), group_modes<0, 3>(tCgXn));
    int xn_bytes = sizeof(make_tensor_like(tXsXn));
    if (elect_warp && elect_thr) {
        cute::set_barrier_transaction_bytes(*xn_bar, xn_bytes);
        copy(tma_xn.with(*xn_bar), tXgXn, tXsXn);
    }
    cute::wait_barrier(*xn_bar, 0);

    int tma_phase = 0, e_phase = 0, s_phase = 0;

    TiledCopy t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, a_acc);
    ThrCopy thr_t2r = t2r.get_slice(threadIdx.x);
    Tensor tDtA = thr_t2r.partition_S(a_acc);
    Tensor tDtB = thr_t2r.partition_S(b_acc);
    Tensor tDcH = thr_t2r.partition_D(tCcAB);  // per-reg (m,n) coords in (CTA_M,BN)
    Tensor rA = make_tensor<float>(shape(tDcH));
    Tensor rB = make_tensor<float>(shape(tDcH));

    for (int c = 0; c < NCHUNK; ++c) {
        Tensor gWa = local_tile(mWa, mma_tiler_e, make_coord(m_tile, c, 0), Step<X, _1, _1>{});
        Tensor gWb = local_tile(mWb, mma_tiler_e, make_coord(m_tile, c, 0), Step<X, _1, _1>{});
        Tensor gWs = local_tile(mWs, mma_tiler_s, make_coord(m_tile, 0, c), Step<X, _1, _1>{}); // (D,BN)
        Tensor tCgWa = cta_e.partition_B(gWa);
        Tensor tCgWb = cta_e.partition_B(gWb);
        Tensor tCgWs = cta_s.partition_B(gWs);
        auto [tWagWa, tWasWa] = tma_partition(tma_wa, Int<0>{}, Layout<_1>{},
                                              group_modes<0, 3>(sWa), group_modes<0, 3>(tCgWa));
        auto [tWbgWb, tWbsWb] = tma_partition(tma_wb, Int<0>{}, Layout<_1>{},
                                              group_modes<0, 3>(sWb), group_modes<0, 3>(tCgWb));
        auto [tWsgWs, tWssWs] = tma_partition(tma_ws, Int<0>{}, Layout<_1>{},
                                              group_modes<0, 3>(sWs), group_modes<0, 3>(tCgWs));
        int w_bytes = sizeof(make_tensor_like(tWasWa)) + sizeof(make_tensor_like(tWbsWb)) +
                      sizeof(make_tensor_like(tWssWs));

        if (elect_warp && elect_thr) {
            cute::set_barrier_transaction_bytes(*tma_bar, w_bytes);
            copy(tma_wa.with(*tma_bar), tWagWa, tWasWa);
            copy(tma_wb.with(*tma_bar), tWbgWb, tWbsWb);
            copy(tma_ws.with(*tma_bar), tWsgWs, tWssWs);
        }
        cute::wait_barrier(*tma_bar, tma_phase); tma_phase ^= 1;

        if (elect_warp) {
            mma_e.accumulate_ = UMMA::ScaleOut::Zero;
            CUTE_UNROLL
            for (int kb = 0; kb < size<2>(tCrXn); ++kb) {
                gemm(mma_e, tCrXn(_, _, kb), tCrWa(_, _, kb), a_acc);
                mma_e.accumulate_ = UMMA::ScaleOut::One;
            }
            mma_e.accumulate_ = UMMA::ScaleOut::Zero;
            CUTE_UNROLL
            for (int kb = 0; kb < size<2>(tCrXn); ++kb) {
                gemm(mma_e, tCrXn(_, _, kb), tCrWb(_, _, kb), b_acc);
                mma_e.accumulate_ = UMMA::ScaleOut::One;
            }
            cutlass::arch::umma_arrive(mma_bar_e);
        }
        cute::wait_barrier(*mma_bar_e, e_phase); e_phase ^= 1;

        // TMEM -> RMEM, silu -> h, scatter to sH_mn
        copy(t2r, tDtA, rA);
        copy(t2r, tDtB, rB);
        CUTE_UNROLL
        for (int i = 0; i < size(rA); ++i) {
            float a = rA(i), b = rB(i);
            auto coord = tDcH(i);
            sH_mn(get<0>(coord), get<1>(coord)) = static_cast<BF>(a * sigmoidf(a) * b);
        }
        cutlass::arch::fence_view_async_shared();
        __syncthreads();

        if (elect_warp) {
            mma_s.accumulate_ = (c == 0) ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
            CUTE_UNROLL
            for (int kb = 0; kb < size<2>(tCrH); ++kb) {
                gemm(mma_s, tCrH(_, _, kb), tCrWs(_, _, kb), out_acc);
                mma_s.accumulate_ = UMMA::ScaleOut::One;
            }
            cutlass::arch::umma_arrive(mma_bar_s);
        }
        cute::wait_barrier(*mma_bar_s, s_phase); s_phase ^= 1;
        __syncthreads();  // sH/sWs free for next chunk
    }

    // Epilogue: out_acc TMEM -> RMEM -> global
    TiledCopy t2r_o = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, out_acc);
    ThrCopy thr_o = t2r_o.get_slice(threadIdx.x);
    Tensor gO = local_tile(mOut, make_shape(Int<CTA_M>{}, Int<D>{}), make_coord(m_tile, 0));
    Tensor tCgO = cta_s.partition_C(gO);
    Tensor tDgO = thr_o.partition_D(tCgO);
    Tensor tDtO = thr_o.partition_S(out_acc);
    Tensor rO = make_tensor<float>(shape(tDgO));
    copy(t2r_o, tDtO, rO);
    Tensor rOb = make_tensor<BF>(shape(tDgO));
    CUTE_UNROLL
    for (int i = 0; i < size(rO); ++i) rOb(i) = static_cast<BF>(rO(i));
    copy(rOb, tDgO);
    __syncthreads();

    if (elect_warp) {
        tmem_allocator.release_allocation_lock();
        tmem_allocator.free(tmem_ptr, TmemAlloc::Sm100TmemCapacityColumns);
    }
#else
    (void)tma_xn; (void)tma_wa; (void)tma_wb; (void)tma_ws; (void)out_raw; (void)M;
    (void)mma_tiler_e; (void)mma_e; (void)mma_tiler_s; (void)mma_s;
    (void)sXn_layout; (void)sW_layout; (void)sWs_layout; (void)sH_mma_layout; (void)sH_mn_layout;
#endif
}

template <int CTA_M, int K, int ND, int D, int BN>
void launch_b2b_fused(const torch::Tensor& xn, const torch::Tensor& wa, const torch::Tensor& wb,
                      const torch::Tensor& ws, torch::Tensor& out, int M, cudaStream_t stream) {
    auto mma_e = make_tiled_mma(
        SM100_MMA_F16BF16_SS<BF, BF, float, CTA_M, BN, UMMA::Major::K, UMMA::Major::K>{});
    auto mma_s = make_tiled_mma(
        SM100_MMA_F16BF16_SS<BF, BF, float, CTA_M, D, UMMA::Major::K, UMMA::Major::K>{});
    auto mma_tiler_e = make_shape(Int<CTA_M>{}, Int<BN>{}, Int<K>{});
    auto mma_tiler_s = make_shape(Int<CTA_M>{}, Int<D>{}, Int<BN>{});

    auto sXn_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<BF>{},
        append(partition_shape_A(mma_e, make_shape(Int<CTA_M>{}, Int<K>{})), Int<1>{}), Step<_1, _2, _3>{});
    auto sW_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<BF>{},
        append(partition_shape_B(mma_e, make_shape(Int<BN>{}, Int<K>{})), Int<1>{}), Step<_1, _2, _3>{});
    auto sWs_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<BF>{},
        append(partition_shape_B(mma_s, make_shape(Int<D>{}, Int<BN>{})), Int<1>{}), Step<_1, _2, _3>{});
    auto sH_mma_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<BF>{},
        append(partition_shape_A(mma_s, make_shape(Int<CTA_M>{}, Int<BN>{})), Int<1>{}), Step<_1, _2, _3>{});
    auto sH_mn_layout = tile_to_shape(UMMA::Layout_K_SW128_Atom<BF>{}, make_shape(Int<CTA_M>{}, Int<BN>{}));

    auto mXn = make_tensor(make_gmem_ptr(reinterpret_cast<const BF*>(xn.data_ptr<at::BFloat16>())),
                           make_shape(M, Int<K>{}), make_stride(Int<K>{}, Int<1>{}));
    auto mWa = make_tensor(make_gmem_ptr(reinterpret_cast<const BF*>(wa.data_ptr<at::BFloat16>())),
                           make_shape(Int<ND>{}, Int<K>{}), make_stride(Int<K>{}, Int<1>{}));
    auto mWb = make_tensor(make_gmem_ptr(reinterpret_cast<const BF*>(wb.data_ptr<at::BFloat16>())),
                           make_shape(Int<ND>{}, Int<K>{}), make_stride(Int<K>{}, Int<1>{}));
    auto mWs = make_tensor(make_gmem_ptr(reinterpret_cast<const BF*>(ws.data_ptr<at::BFloat16>())),
                           make_shape(Int<D>{}, Int<ND>{}), make_stride(Int<ND>{}, Int<1>{}));

    auto tma_xn = make_tma_atom(SM90_TMA_LOAD{}, mXn, sXn_layout, select<0, 2>(mma_tiler_e));
    auto tma_wa = make_tma_atom(SM90_TMA_LOAD{}, mWa, sW_layout, select<1, 2>(mma_tiler_e));
    auto tma_wb = make_tma_atom(SM90_TMA_LOAD{}, mWb, sW_layout, select<1, 2>(mma_tiler_e));
    auto tma_ws = make_tma_atom(SM90_TMA_LOAD{}, mWs, sWs_layout, select<1, 2>(mma_tiler_s));

    int smem = int(cosize_v<decltype(sXn_layout)> + 2 * cosize_v<decltype(sW_layout)> +
                   cosize_v<decltype(sWs_layout)> + cosize_v<decltype(sH_mma_layout)>) * int(sizeof(BF)) + 256;

    auto* kern = &b2b_fused_kernel<CTA_M, K, ND, D, BN,
        decltype(mma_tiler_e), decltype(mma_e), decltype(mma_tiler_s), decltype(mma_s),
        decltype(sXn_layout), decltype(sW_layout), decltype(sWs_layout),
        decltype(sH_mma_layout), decltype(sH_mn_layout),
        decltype(tma_xn), decltype(tma_wa), decltype(tma_wb), decltype(tma_ws)>;
    check_cuda(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem),
               "b2b_fused smem attr");
    dim3 grid(M / CTA_M);
    kern<<<grid, 128, smem, stream>>>(tma_xn, tma_wa, tma_wb, tma_ws,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), M,
        mma_tiler_e, mma_e, mma_tiler_s, mma_s,
        sXn_layout, sW_layout, sWs_layout, sH_mma_layout, sH_mn_layout);
    check_cuda(cudaGetLastError(), "b2b_fused launch");
}

torch::Tensor transition_b2b_fwd(const torch::Tensor& xn, const torch::Tensor& wa,
                                 const torch::Tensor& wb, const torch::Tensor& ws) {
    c10::cuda::CUDAGuard g(xn.device());
    int64_t M = xn.size(0), K = xn.size(1), ND = wa.size(0), D = ws.size(0);
    auto out = torch::empty({M, D}, xn.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (K == 128 && ND == 512 && D == 128) {
        launch_b2b_fused<128, 128, 512, 128, 64>(xn, wa, wb, ws, out, M, stream);
    } else if (K == 256 && ND == 1024 && D == 256) {
        launch_b2b_fused<128, 256, 1024, 256, 64>(xn, wa, wb, ws, out, M, stream);
    } else {
        TORCH_CHECK(false, "unsupported shape");
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transition_b2b_fwd", &transition_b2b_fwd, "fused transition b2b forward (sm100)");
}

#endif  // STAGE 2

}  // namespace b2bsm100
