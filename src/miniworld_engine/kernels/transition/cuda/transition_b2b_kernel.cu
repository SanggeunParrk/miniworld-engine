#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/BFloat16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/algorithm/prefetch.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>

namespace b2b_detail {

using namespace cute;

using BF = cutlass::bfloat16_t;

constexpr int kWarpgroupM = 64;
constexpr int kWarpgroups = 2;
constexpr int kBlockM = kWarpgroups * kWarpgroupM;
constexpr int kBn = 128;
constexpr int kDn = 128;
constexpr int kWarpgroupThreads = 128;
constexpr int kThreads = kWarpgroups * kWarpgroupThreads;
constexpr int kPipelineStages = 2;
constexpr int kPipelineStagesD256 = 1;

constexpr int align_up(int value, int alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

template <int CTA_M, int K, int D, int BN, int STAGES>
struct B2BSmem {
    using WeightPipeline = cutlass::PipelineTmaAsync<STAGES>;
    static constexpr int kXnElems = CTA_M * K;
    static constexpr int kExpandWeightElemsPerStage = K * BN + K * BN;
    static constexpr int kWsWeightElemsPerStage = D * BN;
    static constexpr int kTmaWeightTransactionBytes =
        (kExpandWeightElemsPerStage + kWsWeightElemsPerStage) * static_cast<int>(sizeof(BF));
    static constexpr int kNormParamSmemBytes = (2 * CTA_M) * static_cast<int>(sizeof(float));
    static constexpr int kTensorSmemBytes =
        static_cast<int>((kXnElems + STAGES * kExpandWeightElemsPerStage +
                         STAGES * kWsWeightElemsPerStage) * sizeof(BF)) +
        kNormParamSmemBytes;
    static constexpr int kPipelineStorageOffsetBytes = align_up(kTensorSmemBytes, 16);
    static constexpr int kPipelineStorageSmemBytes =
        align_up(static_cast<int>(sizeof(typename WeightPipeline::SharedStorage)), 16);
    static constexpr int kDynamicSmemBytes =
        kPipelineStorageOffsetBytes + kPipelineStorageSmemBytes;
};

static_assert(kWarpgroups == 2, "cooperative CTA expects exactly two consumer warpgroups");

inline void check_cuda(cudaError_t error, const char* message) {
    TORCH_CHECK(error == cudaSuccess, message, ": ", cudaGetErrorString(error));
}

__device__ __forceinline__ float bf16_to_float(const __nv_bfloat16 v) {
    return __bfloat162float(v);
}

[[maybe_unused]] __device__ __forceinline__ __nv_bfloat16 float_to_bf16(const float v) {
    return __float2bfloat16(v);
}

__device__ __forceinline__ float sigmoidf_fast(const float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ uint32_t smem_addr_u32(const void* ptr) {
    uint32_t addr;
    asm volatile(
        "{ .reg .u64 addr64; cvta.to.shared.u64 addr64, %1; cvt.u32.u64 %0, addr64; }\n"
        : "=r"(addr)
        : "l"(ptr)
    );
    return addr;
}

__device__ __forceinline__ void cp_async_16(void* dst, const void* src, bool pred) {
    const int bytes = pred ? 16 : 0;
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :
        : "r"(smem_addr_u32(dst)), "l"(src), "r"(bytes)
    );
}

__device__ __forceinline__ uint4 smem_load_u128(const void* ptr) {
    uint4 value;
    asm volatile(
        "ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "r"(smem_addr_u32(ptr))
        : "memory"
    );
    return value;
}

__device__ __forceinline__ void smem_store_u128(void* ptr, uint4 value) {
    asm volatile(
        "st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n"
        :
        : "r"(smem_addr_u32(ptr)), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w)
        : "memory"
    );
}

__device__ __forceinline__ void smem_store_u32(void* ptr, uint32_t value) {
    asm volatile(
        "st.shared.u32 [%0], %1;\n"
        :
        : "r"(smem_addr_u32(ptr)), "r"(value)
        : "memory"
    );
}

__device__ __forceinline__ void global_store_u128(void* ptr, uint4 value, bool pred) {
    asm volatile(
        "{ .reg .pred p;\n"
        "  setp.ne.b32 p, %5, 0;\n"
        "  @p st.global.v4.b32 [%0], {%1, %2, %3, %4};\n"
        "}\n"
        :
        : "l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w),
          "r"(static_cast<int>(pred))
        : "memory"
    );
}

template <int DN>
__device__ __forceinline__ int output_shuffle_idx(int m, int n) {
    constexpr int kOutShuffleSwizzleGranularity = 8;
    constexpr int kOutShuffleSwizzleMask = (DN / kOutShuffleSwizzleGranularity) - 1;
    const int row_swizzle = (m & kOutShuffleSwizzleMask) * kOutShuffleSwizzleGranularity;
    return m * DN + (n ^ row_swizzle);
}

template <class MMA, class Layout0>
CUTE_HOST_DEVICE auto convert_layout_acc_Aregs(Layout0 acc_layout) {
    using Traits = MMA_Traits<MMA>;

    static_assert(decltype(rank<0>(acc_layout))::value == 3, "expected SM90 GMMA accumulator layout");
    static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
    static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
    static_assert(decltype(rank(acc_layout))::value == 3);
    static_assert(decltype(rank(get<0>(acc_layout)))::value == 3);
    static_assert(sizeof(typename Traits::ValTypeA) == 2, "this transform is for FP16/BF16 RS A");

    auto l = logical_divide(get<0, 2>(acc_layout), Tile<_2>{});  // ((2, N / 16))
    return make_layout(
        make_layout(get<0, 0>(acc_layout), get<0, 1>(acc_layout), get<0, 0>(l)),
        get<1>(acc_layout),
        coalesce(make_layout(get<0, 1>(l), get<2>(acc_layout)))
    );
}

template <class GTensor, class STensor>
__device__ __forceinline__ void cp_async_bf16_tile(GTensor const& g, STensor const& s, int tid) {
    auto tiled_copy =
        make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>, BF>{},
                        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                        Layout<Shape<_1, _8>>{});
    auto thr_copy = tiled_copy.get_slice(tid);
    copy(thr_copy.partition_S(g), thr_copy.partition_D(s));
}

template <int BLOCK_M>
__global__ __launch_bounds__(128, 4) void transition_b2b_scalar_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ rstd,
    const float* __restrict__ c1,
    const __nv_bfloat16* __restrict__ g,
    const __nv_bfloat16* __restrict__ beta,
    const __nv_bfloat16* __restrict__ wa,
    const __nv_bfloat16* __restrict__ wb,
    const __nv_bfloat16* __restrict__ ws,
    __nv_bfloat16* __restrict__ out,
    const int64_t row_start,
    const int64_t M
) {
#if 0
    const int tid = threadIdx.x;
    const int64_t row0 = row_start + static_cast<int64_t>(blockIdx.x) * BLOCK_M;
    __shared__ __nv_bfloat16 xn_s[BLOCK_M][kK];
    __shared__ float partial_a_s[BLOCK_M][kK];
    __shared__ float partial_b_s[BLOCK_M][kK];
    float out_acc[BLOCK_M];
#pragma unroll
    for (int bm = 0; bm < BLOCK_M; ++bm) {
        out_acc[bm] = 0.0f;
    }
    const float gamma = bf16_to_float(g[tid]);
    const float bias = bf16_to_float(beta[tid]);
#pragma unroll
    for (int bm = 0; bm < BLOCK_M; ++bm) {
        const int64_t row = row0 + bm;
        float xn = 0.0f;
        if (row < M) {
            xn = (bf16_to_float(x[row * kK + tid]) * rstd[row] - c1[row]) * gamma + bias;
        }
        xn_s[bm][tid] = float_to_bf16(xn);
    }
    __syncthreads();
    for (int nd = 0; nd < kND; ++nd) {
        const float wa_v = bf16_to_float(wa[nd * kK + tid]);
        const float wb_v = bf16_to_float(wb[nd * kK + tid]);
#pragma unroll
        for (int bm = 0; bm < BLOCK_M; ++bm) {
            const float xn = bf16_to_float(xn_s[bm][tid]);
            partial_a_s[bm][tid] = xn * wa_v;
            partial_b_s[bm][tid] = xn * wb_v;
        }
        __syncthreads();
        for (int stride = kK / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
#pragma unroll
                for (int bm = 0; bm < BLOCK_M; ++bm) {
                    partial_a_s[bm][tid] += partial_a_s[bm][tid + stride];
                    partial_b_s[bm][tid] += partial_b_s[bm][tid + stride];
                }
            }
            __syncthreads();
        }
        const float ws_v = bf16_to_float(ws[tid * kND + nd]);
#pragma unroll
        for (int bm = 0; bm < BLOCK_M; ++bm) {
            const float a = partial_a_s[bm][0];
            const float b = partial_b_s[bm][0];
            out_acc[bm] += bf16_to_float(float_to_bf16(a * sigmoidf_fast(a) * b)) * ws_v;
        }
        __syncthreads();
    }
#pragma unroll
    for (int bm = 0; bm < BLOCK_M; ++bm) {
        const int64_t row = row0 + bm;
        if (row < M) {
            out[row * kD + tid] = float_to_bf16(out_acc[bm]);
        }
    }
#else
    (void)x;
    (void)rstd;
    (void)c1;
    (void)g;
    (void)beta;
    (void)wa;
    (void)wb;
    (void)ws;
    (void)out;
    (void)row_start;
    (void)M;
#endif
}

template <
    int CTA_M,
    int WG_M,
    int K,
    int ND,
    int D,
    int BN,
    int DN,
    int STAGES,
    class TmaWa,
    class TmaWb,
    class TmaWs>
__global__ __launch_bounds__(256, 1) void transition_b2b_rs_wgmma_kernel(
    const __nv_bfloat16* __restrict__ x_raw,
    const float* __restrict__ rstd,
    const float* __restrict__ c1,
    const __nv_bfloat16* __restrict__ g_raw,
    const __nv_bfloat16* __restrict__ beta_raw,
    const __nv_bfloat16* __restrict__ wa_raw,
    const __nv_bfloat16* __restrict__ wb_raw,
    const __nv_bfloat16* __restrict__ ws_raw,
    __nv_bfloat16* __restrict__ out_raw,
    const int64_t M,
    const bool add_residual,
    __grid_constant__ TmaWa const tma_wa,
    __grid_constant__ TmaWb const tma_wb,
    __grid_constant__ TmaWs const tma_ws
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using Smem = B2BSmem<CTA_M, K, D, BN, STAGES>;
    using KernelWeightPipeline = typename Smem::WeightPipeline;
    static_assert(ND % BN == 0, "ND must be divisible by BLOCK_N");
    static_assert(D % DN == 0, "D must be divisible by output tile width DN");
    static_assert(K % 8 == 0, "K must be divisible by 8 for vectorized normalization");
    static_assert(DN % 8 == 0, "DN must be divisible by 8 for vectorized output");
    static_assert(DN == 128, "SM90_64x128x16 squeeze atom expects a 128-column output tile");
    static_assert(D / DN == 1 || D / DN == 2, "transition_b2b expects one or two output D tiles");
    static_assert(Smem::kDynamicSmemBytes <= 227 * 1024,
                  "dynamic smem should fit H100 opt-in shared memory");
    constexpr int kNumDTiles = D / DN;

    const int tid = threadIdx.x;
    const int wg_id = tid / kWarpgroupThreads;
    const int wg_tid = tid - wg_id * kWarpgroupThreads;
    const int wg_barrier_id = wg_id + 1;
    const int row0 = blockIdx.x * CTA_M;
    const int wg_row0 = row0 + wg_id * WG_M;

    (void)wa_raw;
    (void)wb_raw;
    (void)ws_raw;
    auto mOut = make_tensor(make_gmem_ptr(reinterpret_cast<BF*>(out_raw)),
                            make_shape(M, Int<D>{}),
                            make_stride(Int<D>{}, Int<1>{}));

    auto lXn = tile_to_shape(GMMA::Layout_K_SW128_Atom<BF>{},
                             make_shape(Int<WG_M>{}, Int<K>{}));
    auto lW = tile_to_shape(GMMA::Layout_K_SW128_Atom<BF>{},
                            make_shape(Int<BN>{}, Int<K>{}));
    auto lWs = tile_to_shape(GMMA::Layout_K_SW128_Atom<BF>{},
                             make_shape(Int<D>{}, Int<BN>{}));

    extern __shared__ __align__(16) unsigned char smem_raw[];
    BF* s_base = reinterpret_cast<BF*>(smem_raw);
    BF* pXn = s_base;
    BF* pWa = pXn + kWarpgroups * cosize_v<decltype(lXn)>;
    BF* pWb = pWa + STAGES * cosize_v<decltype(lW)>;
    BF* pWs = pWb + STAGES * cosize_v<decltype(lW)>;
    float* pRstd = reinterpret_cast<float*>(pWs + STAGES * cosize_v<decltype(lWs)>);
    float* pC1 = pRstd + CTA_M;
    BF* pXnWg = pXn + wg_id * cosize_v<decltype(lXn)>;

    auto sXn = make_tensor(make_smem_ptr(pXnWg), lXn);
    auto sWa = [&](int stage) {
        return make_tensor(make_smem_ptr(pWa + stage * cosize_v<decltype(lW)>), lW);
    };
    auto sWb = [&](int stage) {
        return make_tensor(make_smem_ptr(pWb + stage * cosize_v<decltype(lW)>), lW);
    };
    auto sWs = [&](int stage) {
        return make_tensor(make_smem_ptr(pWs + stage * cosize_v<decltype(lWs)>), lWs);
    };

    auto* weight_pipe_storage = reinterpret_cast<typename KernelWeightPipeline::SharedStorage*>(
        smem_raw + Smem::kPipelineStorageOffsetBytes
    );
    typename KernelWeightPipeline::Params weight_pipe_params;
    weight_pipe_params.transaction_bytes = Smem::kTmaWeightTransactionBytes;
    weight_pipe_params.role = KernelWeightPipeline::ThreadCategory::ProducerConsumer;
    weight_pipe_params.is_leader = tid == 0;
    weight_pipe_params.num_consumers = kThreads;
    KernelWeightPipeline weight_pipe(*weight_pipe_storage, weight_pipe_params, Shape<_1, _1, _1>{});

    __syncthreads();

    auto tmaWaTensor = tma_wa.get_tma_tensor(make_shape(Int<ND>{}, Int<K>{}));
    auto tmaWbTensor = tma_wb.get_tma_tensor(make_shape(Int<ND>{}, Int<K>{}));
    auto tmaWsTensor = tma_ws.get_tma_tensor(make_shape(Int<D>{}, Int<ND>{}));
    auto tmaWaSlice = tma_wa.get_slice(Int<0>{});
    auto tmaWbSlice = tma_wb.get_slice(Int<0>{});
    auto tmaWsSlice = tma_ws.get_slice(Int<0>{});

    auto issue_tma_weight_stage = [&](auto& producer_state, int stage, int nd_base) {
        if (tid == 0) {
            weight_pipe.producer_acquire(producer_state);
            using BarrierType = typename KernelWeightPipeline::ProducerBarrierType;
            BarrierType* tma_barrier = weight_pipe.producer_get_barrier(producer_state);

            auto gWa = local_tile(tmaWaTensor, make_shape(Int<BN>{}, Int<K>{}), make_coord(nd_base / BN, 0));
            auto gWb = local_tile(tmaWbTensor, make_shape(Int<BN>{}, Int<K>{}), make_coord(nd_base / BN, 0));
            auto gWs = local_tile(tmaWsTensor, make_shape(Int<D>{}, Int<BN>{}), make_coord(0, nd_base / BN));

            copy(tma_wa.with(*tma_barrier), tmaWaSlice.partition_S(gWa), tmaWaSlice.partition_D(sWa(stage)));
            copy(tma_wb.with(*tma_barrier), tmaWbSlice.partition_S(gWb), tmaWbSlice.partition_D(sWb(stage)));
            copy(tma_ws.with(*tma_barrier), tmaWsSlice.partition_S(gWs), tmaWsSlice.partition_D(sWs(stage)));
            // PipelineTmaAsync: producer_acquire set expect_tx; the TMA copies auto-arrive on the
            // barrier with their transaction bytes -> no explicit producer_commit needed.
            ++producer_state;
        }
    };

    if (tid == 0) {
        cute::prefetch_tma_descriptor(tma_wa.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_wb.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_ws.get_tma_descriptor());
    }

    static constexpr int kXVecElems = 8;
    static_assert(K % kXVecElems == 0, "x vectorization requires K to be divisible by 8");
    static_assert((kXVecElems * sizeof(__nv_bfloat16)) == sizeof(uint4), "x vector must be 128-bit");
    static_assert((kXVecElems * sizeof(BF)) == sizeof(uint4), "BF x vector must be 128-bit");
    static_assert((kWarpgroupThreads * kXVecElems) % K == 0,
                  "each normalization thread must keep a fixed k-slice");
    union XnVec {
        uint4 vec;
        BF bf[kXVecElems];
    };

    auto stage_x_into_sxn = [&]() {
        for (int vec = wg_tid; vec < (WG_M * K) / kXVecElems; vec += kWarpgroupThreads) {
            const int idx = vec * kXVecElems;
            const int m = idx / K;
            const int k0 = idx - m * K;
            const int64_t row = static_cast<int64_t>(wg_row0 + m);
            const bool valid = row < M;
            const __nv_bfloat16* src = valid ? x_raw + row * K + k0 : x_raw;
            cp_async_16(&sXn(m, k0), src, valid);
        }
    };

    typename KernelWeightPipeline::PipelineState first_d_tma_producer_state =
        cutlass::make_producer_start_state<KernelWeightPipeline>();
    CUTE_UNROLL
    for (int stage = 0; stage < STAGES - 1; ++stage) {
        issue_tma_weight_stage(first_d_tma_producer_state, stage, stage * BN);
    }

    stage_x_into_sxn();
    cp_async_fence();

    if (wg_tid < WG_M) {
        const int64_t row = static_cast<int64_t>(wg_row0 + wg_tid);
        const int cta_m = wg_id * WG_M + wg_tid;
        pRstd[cta_m] = row < M ? rstd[row] : 0.0f;
        pC1[cta_m] = row < M ? c1[row] : 0.0f;
    }
    cutlass::arch::NamedBarrier::arrive_and_wait(kWarpgroupThreads, wg_barrier_id);

    cp_async_wait<0>();
    const BF* g_bf = reinterpret_cast<const BF*>(g_raw);
    const BF* beta_bf = reinterpret_cast<const BF*>(beta_raw);
    const int param_k0 = (wg_tid % (K / kXVecElems)) * kXVecElems;
    BF gamma_reg[kXVecElems];
    BF beta_reg[kXVecElems];
    CUTE_UNROLL
    for (int i = 0; i < kXVecElems; ++i) {
        gamma_reg[i] = g_bf[param_k0 + i];
        beta_reg[i] = beta_bf[param_k0 + i];
    }
    for (int vec = wg_tid; vec < (WG_M * K) / kXVecElems; vec += kWarpgroupThreads) {
        const int idx = vec * kXVecElems;
        const int m = idx / K;
        const int k0 = idx - m * K;
        const int64_t row = static_cast<int64_t>(wg_row0 + m);
        const bool valid = row < M;
        const int cta_m = wg_id * WG_M + m;
        const float row_rstd = pRstd[cta_m];
        const float row_c1 = pC1[cta_m];
        XnVec x_vec;
        XnVec xn_vec;
        x_vec.vec = smem_load_u128(&sXn(m, k0));
        CUTE_UNROLL
        for (int i = 0; i < kXVecElems; ++i) {
            float xn = 0.0f;
            if (valid) {
                const float xv = static_cast<float>(x_vec.bf[i]);
                const float gamma = static_cast<float>(gamma_reg[i]);
                const float bias = static_cast<float>(beta_reg[i]);
                xn = (xv * row_rstd - row_c1) * gamma + bias;
            }
            xn_vec.bf[i] = static_cast<BF>(xn);
        }
        smem_store_u128(&sXn(m, k0), xn_vec.vec);
    }
    cutlass::arch::NamedBarrier::arrive_and_wait(kWarpgroupThreads, wg_barrier_id);

    using MmaExpandOp = std::conditional_t<
        BN == 64,
        SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>,
        SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>>;
    using MmaSqueezeOp = SM90_64x128x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>;
    TiledMMA mmaE = make_tiled_mma(MmaExpandOp{});
    TiledMMA mmaS = make_tiled_mma(MmaSqueezeOp{});
    auto thrE = mmaE.get_slice(wg_tid);
    auto thrS = mmaS.get_slice(wg_tid);

    auto cOut = make_identity_tensor(make_shape(Int<WG_M>{}, Int<DN>{}));
    auto tCOut = thrS.partition_C(cOut);
    auto gOut0 = local_tile(mOut, make_shape(Int<WG_M>{}, Int<DN>{}),
                            make_coord(blockIdx.x * kWarpgroups + wg_id, 0));
    auto tGOut0 = thrS.partition_C(gOut0);
    auto out_acc0 = thrS.make_fragment_C(tGOut0);
    clear(out_acc0);
    warpgroup_fence_operand(out_acc0);

    auto store_output_tile = [&](auto& out_acc, int d_tile) {
        BF* pOutShuffle = pWa + wg_id * (WG_M * DN);
        static_assert(kWarpgroups * WG_M * DN <= cosize_v<decltype(lW)>,
                      "output shuffle tiles must fit in one pWa stage after the ND loop");
        static_assert((DN % kXVecElems) == 0, "output vectorization requires DN divisible by 8");
        static_assert((kXVecElems * sizeof(BF)) == sizeof(uint4), "output vector must be 128-bit");
        static_assert((DN / kXVecElems) >= 1, "output swizzle requires at least one vector per row");
        union OutPair {
            uint32_t u32;
            BF bf[2];
        };

        CUTE_UNROLL
        for (int i = 0; i < size(out_acc); i += 2) {
            const int m = get<0>(tCOut(i));
            const int n = get<1>(tCOut(i));
            const int m_next = get<0>(tCOut(i + 1));
            const int n_next = get<1>(tCOut(i + 1));
            if (m_next == m && n_next == n + 1 && (n % 2) == 0) {
                OutPair pair;
                pair.bf[0] = static_cast<BF>(out_acc(i));
                pair.bf[1] = static_cast<BF>(out_acc(i + 1));
                smem_store_u32(pOutShuffle + output_shuffle_idx<DN>(m, n), pair.u32);
            } else {
                pOutShuffle[output_shuffle_idx<DN>(m, n)] = static_cast<BF>(out_acc(i));
                pOutShuffle[output_shuffle_idx<DN>(m_next, n_next)] = static_cast<BF>(out_acc(i + 1));
            }
        }
        cutlass::arch::NamedBarrier::arrive_and_wait(kWarpgroupThreads, wg_barrier_id);

        constexpr int kOutVecsPerRow = DN / kXVecElems;
        CUTE_UNROLL
        for (int vec = wg_tid; vec < WG_M * kOutVecsPerRow; vec += kWarpgroupThreads) {
            const int m = vec / kOutVecsPerRow;
            const int n = (vec - m * kOutVecsPerRow) * kXVecElems;
            const int64_t row = static_cast<int64_t>(wg_row0 + m);
            const bool valid = row < M;
            uint4 out_vec = smem_load_u128(pOutShuffle + output_shuffle_idx<DN>(m, n));
            // Fuse the post-transition residual add: y = transition(x) + x. The residual is
            // the kernel's OWN pre-LN input x_raw (module never mutates it before
            // `pair + transition(pair)`), D == K so it shares out_raw's row-major layout —
            // one coalesced 128-bit load at the same (row, col) + a bf16x2 add before store.
            if (add_residual && valid) {
                const uint4 res_vec =
                    *reinterpret_cast<const uint4*>(x_raw + row * D + d_tile * DN + n);
                __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(&out_vec);
                const __nv_bfloat162* r = reinterpret_cast<const __nv_bfloat162*>(&res_vec);
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    o[t] = __hadd2(o[t], r[t]);
                }
            }
            __nv_bfloat16* dst = valid ? out_raw + row * D + d_tile * DN + n : out_raw;
            global_store_u128(dst, out_vec, valid);
        }
        cutlass::arch::NamedBarrier::arrive_and_wait(kWarpgroupThreads, wg_barrier_id);
    };

    auto run_nd_loop = [&](auto& out_acc_first, auto& out_acc_second) {
        typename KernelWeightPipeline::PipelineState tma_producer_state = first_d_tma_producer_state;
        typename KernelWeightPipeline::PipelineState tma_consumer_state;

        CUTE_NO_UNROLL
        for (int nd_base = 0, chunk = 0; nd_base < ND; nd_base += BN, ++chunk) {
            const int read_stage = chunk % STAGES;
            const int write_stage = (chunk + STAGES - 1) % STAGES;
            const int prefetch_nd = nd_base + (STAGES - 1) * BN;

            if (prefetch_nd < ND) {
                issue_tma_weight_stage(tma_producer_state, write_stage, prefetch_nd);
            }
            weight_pipe.consumer_wait(tma_consumer_state);

            auto a_acc = partition_fragment_C(mmaE, make_shape(Int<WG_M>{}, Int<BN>{}));
            auto b_acc = partition_fragment_C(mmaE, make_shape(Int<WG_M>{}, Int<BN>{}));
            clear(a_acc);
            clear(b_acc);

            auto tXnA = thrE.make_fragment_A(thrE.partition_A(sXn));
            auto tWaB = thrE.make_fragment_B(thrE.partition_B(sWa(read_stage)));
            auto tWbB = thrE.make_fragment_B(thrE.partition_B(sWb(read_stage)));

            warpgroup_fence_operand(a_acc);
            warpgroup_arrive();
            cute::gemm(mmaE, tXnA, tWaB, a_acc);
            warpgroup_commit_batch();

            warpgroup_fence_operand(b_acc);
            warpgroup_arrive();
            cute::gemm(mmaE, tXnA, tWbB, b_acc);
            warpgroup_commit_batch();
            warpgroup_wait<0>();
            warpgroup_fence_operand(a_acc);
            warpgroup_fence_operand(b_acc);

            auto h_acc = make_fragment_like(a_acc);

            CUTE_UNROLL
            for (int i = 0; i < size(a_acc); ++i) {
                const float a = a_acc(i);
                const float b = b_acc(i);
                h_acc(i) = a * sigmoidf_fast(a) * b;
            }

            auto h_bf = make_fragment_like<BF>(h_acc);
            CUTE_UNROLL
            for (int i = 0; i < size(h_acc); ++i) {
                h_bf(i) = static_cast<BF>(h_acc(i));
            }

            auto h_frag = make_tensor(h_bf.data(), convert_layout_acc_Aregs<MmaSqueezeOp>(a_acc.layout()));
            auto sWs0 = local_tile(sWs(read_stage), make_shape(Int<DN>{}, Int<BN>{}), make_coord(0, 0));
            auto tWsB0 = thrS.make_fragment_B(thrS.partition_B(sWs0));
            CUTE_STATIC_ASSERT_V(size<2>(h_frag) == size<2>(tWsB0));
            auto h_frag_regs = recast<uint32_t>(h_frag(_, _, Int<0>{}));
            static constexpr int kSqueezeRegNumA = extent<typename MmaSqueezeOp::ARegisters>::value;
            CUTE_STATIC_ASSERT_V(size(h_frag_regs) == Int<kSqueezeRegNumA>{});

            warpgroup_arrive();
            cute::gemm(mmaS, h_frag, tWsB0, out_acc_first);
            warpgroup_commit_batch();

            if constexpr (kNumDTiles == 2) {
                auto sWs1 = local_tile(sWs(read_stage), make_shape(Int<DN>{}, Int<BN>{}), make_coord(1, 0));
                auto tWsB1 = thrS.make_fragment_B(thrS.partition_B(sWs1));
                CUTE_STATIC_ASSERT_V(size<2>(h_frag) == size<2>(tWsB1));
                warpgroup_arrive();
                cute::gemm(mmaS, h_frag, tWsB1, out_acc_second);
                warpgroup_commit_batch();
            }

            weight_pipe.consumer_release(tma_consumer_state);
            ++tma_consumer_state;
        }
    };

    if constexpr (kNumDTiles == 2) {
        auto gOut1 = local_tile(mOut, make_shape(Int<WG_M>{}, Int<DN>{}),
                                make_coord(blockIdx.x * kWarpgroups + wg_id, 1));
        auto tGOut1 = thrS.partition_C(gOut1);
        auto out_acc1 = thrS.make_fragment_C(tGOut1);
        clear(out_acc1);
        warpgroup_fence_operand(out_acc1);

        run_nd_loop(out_acc0, out_acc1);
        warpgroup_wait<0>();
        warpgroup_fence_operand(out_acc0);
        warpgroup_fence_operand(out_acc1);
        store_output_tile(out_acc0, 0);
        store_output_tile(out_acc1, 1);
    } else {
        run_nd_loop(out_acc0, out_acc0);
        warpgroup_wait<0>();
        warpgroup_fence_operand(out_acc0);
        store_output_tile(out_acc0, 0);
    }
#else
    (void)x_raw;
    (void)rstd;
    (void)c1;
    (void)g_raw;
    (void)beta_raw;
    (void)wa_raw;
    (void)wb_raw;
    (void)ws_raw;
    (void)out_raw;
    (void)M;
    (void)add_residual;
    (void)tma_wa;
    (void)tma_wb;
    (void)tma_ws;
#endif
}

void check_transition_b2b_inputs(
    const torch::Tensor& x,
    const torch::Tensor& rstd,
    const torch::Tensor& c1,
    const torch::Tensor& g,
    const torch::Tensor& beta,
    const torch::Tensor& wa,
    const torch::Tensor& wb,
    const torch::Tensor& ws
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(rstd.is_cuda(), "rstd must be a CUDA tensor");
    TORCH_CHECK(c1.is_cuda(), "c1 must be a CUDA tensor");
    TORCH_CHECK(g.is_cuda(), "g must be a CUDA tensor");
    TORCH_CHECK(beta.is_cuda(), "beta must be a CUDA tensor");
    TORCH_CHECK(wa.is_cuda(), "wa must be a CUDA tensor");
    TORCH_CHECK(wb.is_cuda(), "wb must be a CUDA tensor");
    TORCH_CHECK(ws.is_cuda(), "ws must be a CUDA tensor");

    const auto device = x.get_device();
    TORCH_CHECK(
        rstd.get_device() == device && c1.get_device() == device &&
            g.get_device() == device && beta.get_device() == device &&
            wa.get_device() == device && wb.get_device() == device &&
            ws.get_device() == device,
        "all tensors must be on the same CUDA device"
    );

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(rstd.is_contiguous(), "rstd must be contiguous");
    TORCH_CHECK(c1.is_contiguous(), "c1 must be contiguous");
    TORCH_CHECK(g.is_contiguous(), "g must be contiguous");
    TORCH_CHECK(beta.is_contiguous(), "beta must be contiguous");
    TORCH_CHECK(wa.is_contiguous(), "wa must be contiguous");
    TORCH_CHECK(wb.is_contiguous(), "wb must be contiguous");
    TORCH_CHECK(ws.is_contiguous(), "ws must be contiguous");

    TORCH_CHECK(x.scalar_type() == torch::ScalarType::BFloat16, "x must be bf16");
    TORCH_CHECK(g.scalar_type() == torch::ScalarType::BFloat16, "g must be bf16");
    TORCH_CHECK(beta.scalar_type() == torch::ScalarType::BFloat16, "beta must be bf16");
    TORCH_CHECK(wa.scalar_type() == torch::ScalarType::BFloat16, "wa must be bf16");
    TORCH_CHECK(wb.scalar_type() == torch::ScalarType::BFloat16, "wb must be bf16");
    TORCH_CHECK(ws.scalar_type() == torch::ScalarType::BFloat16, "ws must be bf16");
    TORCH_CHECK(rstd.scalar_type() == torch::ScalarType::Float, "rstd must be fp32");
    TORCH_CHECK(c1.scalar_type() == torch::ScalarType::Float, "c1 must be fp32");

    TORCH_CHECK(x.dim() == 2, "x must have shape (M, K)");
    const int64_t K = x.size(1);
    TORCH_CHECK(K == 128 || K == 256, "transition_b2b CUDA path supports K=128 or K=256");
    TORCH_CHECK(rstd.dim() == 1 && rstd.size(0) == x.size(0), "rstd must have shape (M,)");
    TORCH_CHECK(c1.dim() == 1 && c1.size(0) == x.size(0), "c1 must have shape (M,)");
    TORCH_CHECK(g.dim() == 1 && g.size(0) == K, "g must have shape (K,)");
    TORCH_CHECK(beta.dim() == 1 && beta.size(0) == K, "beta must have shape (K,)");
    TORCH_CHECK(wa.dim() == 2 && wa.size(1) == K, "wa must have shape (ND, K)");
    TORCH_CHECK(wb.sizes() == wa.sizes(), "wb must have shape (ND, K)");
    TORCH_CHECK(ws.dim() == 2, "ws must have shape (D, ND)");
    const int64_t ND = wa.size(0);
    const int64_t D = ws.size(0);
    TORCH_CHECK(ws.size(1) == ND, "ws must have shape (D, ND)");
    TORCH_CHECK(D == K, "transition_b2b CUDA path requires D == K");
    TORCH_CHECK(ND == 4 * K, "transition_b2b CUDA path requires n=4, i.e. ND == 4*K");
    TORCH_CHECK(
        (K == 128 && ND == 512 && D == 128) || (K == 256 && ND == 1024 && D == 256),
        "transition_b2b CUDA path supports (K,ND,D) = (128,512,128) or (256,1024,256)"
    );
}

}  // namespace b2b_detail

using namespace b2b_detail;

template <int K, int ND, int D, int CTA_M, int WG_M, int BN, int DN, int STAGES>
void launch_transition_b2b_kernel(
    const torch::Tensor& x,
    const torch::Tensor& rstd,
    const torch::Tensor& c1,
    const torch::Tensor& g,
    const torch::Tensor& beta,
    const torch::Tensor& wa,
    const torch::Tensor& wb,
    const torch::Tensor& ws,
    torch::Tensor& out,
    int64_t M,
    bool add_residual,
    cudaStream_t stream
) {
    using Smem = B2BSmem<CTA_M, K, D, BN, STAGES>;
    static_assert(Smem::kDynamicSmemBytes <= 227 * 1024,
                  "dynamic smem should fit H100 opt-in shared memory");

    auto lW = tile_to_shape(GMMA::Layout_K_SW128_Atom<BF>{},
                            make_shape(Int<BN>{}, Int<K>{}));
    auto lWs = tile_to_shape(GMMA::Layout_K_SW128_Atom<BF>{},
                             make_shape(Int<D>{}, Int<BN>{}));
    auto mWaTma = make_tensor(make_gmem_ptr(reinterpret_cast<const BF*>(wa.data_ptr<at::BFloat16>())),
                              make_shape(Int<ND>{}, Int<K>{}),
                              make_stride(Int<K>{}, Int<1>{}));
    auto mWbTma = make_tensor(make_gmem_ptr(reinterpret_cast<const BF*>(wb.data_ptr<at::BFloat16>())),
                              make_shape(Int<ND>{}, Int<K>{}),
                              make_stride(Int<K>{}, Int<1>{}));
    auto mWsTma = make_tensor(make_gmem_ptr(reinterpret_cast<const BF*>(ws.data_ptr<at::BFloat16>())),
                              make_shape(Int<D>{}, Int<ND>{}),
                              make_stride(Int<ND>{}, Int<1>{}));
    auto tma_wa = make_tma_copy(SM90_TMA_LOAD{}, mWaTma, lW,
                                make_shape(Int<BN>{}, Int<K>{}), Int<1>{});
    auto tma_wb = make_tma_copy(SM90_TMA_LOAD{}, mWbTma, lW,
                                make_shape(Int<BN>{}, Int<K>{}), Int<1>{});
    auto tma_ws = make_tma_copy(SM90_TMA_LOAD{}, mWsTma, lWs,
                                make_shape(Int<D>{}, Int<BN>{}), Int<1>{});

    auto* kernel = transition_b2b_rs_wgmma_kernel<
        CTA_M, WG_M, K, ND, D, BN, DN, STAGES, decltype(tma_wa), decltype(tma_wb), decltype(tma_ws)>;
    check_cuda(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Smem::kDynamicSmemBytes),
        "transition_b2b_rs_wgmma_kernel dynamic shared-memory attribute failed"
    );
    check_cuda(
        cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100),
        "transition_b2b_rs_wgmma_kernel shared-memory carveout attribute failed"
    );

    const dim3 grid(static_cast<unsigned int>((M + CTA_M - 1) / CTA_M));
    const dim3 block(kThreads);
    kernel<<<grid, block, Smem::kDynamicSmemBytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        rstd.data_ptr<float>(),
        c1.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(g.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(wa.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(wb.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(ws.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        M,
        add_residual,
        tma_wa,
        tma_wb,
        tma_ws
    );
    check_cuda(cudaGetLastError(), "transition_b2b_rs_wgmma_kernel launch failed");
}

torch::Tensor transition_b2b_fwd(
    const torch::Tensor& x,
    const torch::Tensor& rstd,
    const torch::Tensor& c1,
    const torch::Tensor& g,
    const torch::Tensor& beta,
    const torch::Tensor& wa,
    const torch::Tensor& wb,
    const torch::Tensor& ws,
    bool add_residual
) {
    check_transition_b2b_inputs(x, rstd, c1, g, beta, wa, wb, ws);
    if (add_residual) {
        TORCH_CHECK(x.is_contiguous(), "transition_b2b residual fuse requires contiguous x");
    }

    c10::cuda::CUDAGuard device_guard(x.device());
    const int64_t K = x.size(1);
    const int64_t ND = wa.size(0);
    const int64_t D = ws.size(0);
    auto out = torch::empty({x.size(0), D}, x.options());
    const int64_t M = x.size(0);
    if (M == 0) {
        return out;
    }
    TORCH_CHECK(M % kBlockM == 0, "transition_b2b CUDA path requires M to be divisible by 128");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (K == 128 && ND == 512 && D == 128) {
        launch_transition_b2b_kernel<
            128, 512, 128, kBlockM, kWarpgroupM, kBn, kDn, kPipelineStages>(
            x, rstd, c1, g, beta, wa, wb, ws, out, M, add_residual, stream
        );
    } else if (K == 256 && ND == 1024 && D == 256) {
        launch_transition_b2b_kernel<
            256, 1024, 256, kBlockM, kWarpgroupM, 64, kDn, kPipelineStagesD256>(
            x, rstd, c1, g, beta, wa, wb, ws, out, M, add_residual, stream
        );
    } else {
        TORCH_CHECK(false, "unsupported transition_b2b CUDA shape");
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transition_b2b_fwd", &transition_b2b_fwd, "Fused transition b2b forward (CUDA)",
          pybind11::arg("x"), pybind11::arg("rstd"), pybind11::arg("c1"), pybind11::arg("g"),
          pybind11::arg("beta"), pybind11::arg("wa"), pybind11::arg("wb"), pybind11::arg("ws"),
          pybind11::arg("add_residual") = false);
}
