// Hopper (sm_90a) WGMMA fused expand + SwiGLU-gate BACKWARD.
//
// The sm90 counterpart of the sm100 tcgen05 `transition_expand_gatebwd_sm100`: it recomputes
// the expand a=xn@Wa^T, b=xn@Wb^T ONCE (LN folded, WGMMA) and emits, per (m,n):
//     h  = silu(a) * b                         (for dWs = go^T @ h)
//     dA = ge * b * silu'(a)                   (ge = grad_expand)
//     dB = ge * silu(a)
// plus xn (M,K) for the wgrad GEMMs. Replaces the Triton `_transition_expand_gatebwd_stacked`
// (which is ~53% of the transition training backward and ~2x off roofline on H100).
//
// Structure mirrors `transition_expand_gate_kernel.cu` (LN -> xn in smem -> ND-tiled dual-WGMMA
// a,b). The ONLY delta is the epilogue: instead of storing just h, we stage grad_expand for the
// tile, compute h/dA/dB in fp32, and store all three coalesced via the smem-shuffle trick.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/BFloat16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/algorithm/prefetch.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>

namespace gatebwd_detail {

using namespace cute;

using BF = cutlass::bfloat16_t;

constexpr int kWarpgroupM = 64;
// ONE warpgroup / CTA (128 threads). This is the key occupancy lever for this backward:
// with 128-thread blocks the register limit on blocks/SM is floor(65536/(128*R)) — 2x more
// forgiving than 256-thread blocks (floor(256/R), which pinned >128-reg kernels to 1 block/SM
// and 12.5% occupancy). Plus pXn halves (only WG_M rows resident). Together -> ~3 blocks/SM.
constexpr int kWarpgroups = 2;
constexpr int kBlockM = kWarpgroups * kWarpgroupM;
constexpr int kWarpgroupThreads = 128;
constexpr int kThreads = kWarpgroups * kWarpgroupThreads;

constexpr int align_up(int value, int alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

template <int CTA_M, int WG_M, int K, int BN, int STAGES, int KT>
struct GateBwdSmem {
    using WeightPipeline = cutlass::PipelineTmaAsync<STAGES>;
    static constexpr int kXnElems = CTA_M * K;
    static constexpr int kExpandWeightElemsPerStage = KT * BN + KT * BN;  // Wa + Wb tile
    static constexpr int kNormParamSmemBytes = (2 * CTA_M) * static_cast<int>(sizeof(float));
    static constexpr int kTensorSmemBytes =
        static_cast<int>((kXnElems + STAGES * kExpandWeightElemsPerStage) * sizeof(BF)) +
        kNormParamSmemBytes;
    static constexpr int kPipelineStorageOffsetBytes = align_up(kTensorSmemBytes, 16);
    static constexpr int kPipelineStorageSmemBytes =
        align_up(static_cast<int>(sizeof(typename WeightPipeline::SharedStorage)), 16);
    static constexpr int kDynamicSmemBytes =
        kPipelineStorageOffsetBytes + kPipelineStorageSmemBytes;
    static constexpr int kTmaWeightTransactionBytes =
        kExpandWeightElemsPerStage * static_cast<int>(sizeof(BF));
};

inline void check_cuda(cudaError_t error, const char* message) {
    TORCH_CHECK(error == cudaSuccess, message, ": ", cudaGetErrorString(error));
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

__device__ __forceinline__ uint4 global_load_u128(const void* ptr, bool pred) {
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (pred) {
        v = *reinterpret_cast<const uint4*>(ptr);
    }
    return v;
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

template <int N>
__device__ __forceinline__ int output_shuffle_idx(int m, int n) {
    constexpr int kOutShuffleSwizzleGranularity = 8;
    constexpr int kOutShuffleSwizzleMask = (N / kOutShuffleSwizzleGranularity) - 1;
    const int row_swizzle = (m & kOutShuffleSwizzleMask) * kOutShuffleSwizzleGranularity;
    return m * N + (n ^ row_swizzle);
}

template <
    int CTA_M,
    int WG_M,
    int K,
    int ND,
    int BN,
    int STAGES,
    int KT,
    class TmaWa,
    class TmaWb>
__global__ __launch_bounds__(256, 1) void transition_gatebwd_rs_wgmma_kernel(
    const __nv_bfloat16* __restrict__ x_raw,
    const float* __restrict__ rstd,
    const float* __restrict__ c1,
    const __nv_bfloat16* __restrict__ g_raw,
    const __nv_bfloat16* __restrict__ beta_raw,
    const __nv_bfloat16* __restrict__ ge_raw,   // grad_expand (M, ND)
    __nv_bfloat16* __restrict__ h_raw,          // (M, ND)
    __nv_bfloat16* __restrict__ dab_raw,        // (M, 2*ND) = [dA | dB]
    __nv_bfloat16* __restrict__ xn_raw,         // (M, K)
    const int64_t M,
    __grid_constant__ TmaWa const tma_wa,
    __grid_constant__ TmaWb const tma_wb
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using Smem = GateBwdSmem<CTA_M, WG_M, K, BN, STAGES, KT>;
    using KernelWeightPipeline = typename Smem::WeightPipeline;
    static_assert(CTA_M == kBlockM, "CTA_M must match the two-warpgroups CTA shape");
    static_assert(WG_M == kWarpgroupM, "WG_M must match one warpgroup tile");
    static_assert(ND % BN == 0, "ND must be divisible by BN");
    static_assert(K % KT == 0, "K must be divisible by KT");
    static_assert(K % 8 == 0, "K must be divisible by 8 for vectorized normalization");
    static_assert(BN % 8 == 0, "BN must be divisible by 8 for vectorized stores");
    static_assert(BN == 64 || BN == 128, "expand WGMMA atom supports BN=64 or BN=128");
    static_assert(STAGES >= 1, "weight pipeline must have at least one stage");
    static_assert(Smem::kDynamicSmemBytes <= 227 * 1024,
                  "dynamic smem should fit H100 opt-in shared memory");

    const int tid = threadIdx.x;
    const int wg_id = tid / kWarpgroupThreads;
    const int wg_tid = tid - wg_id * kWarpgroupThreads;
    const int wg_barrier_id = wg_id + 1;
    const int row0 = blockIdx.x * CTA_M;
    const int wg_row0 = row0 + wg_id * WG_M;

    auto lXn = tile_to_shape(GMMA::Layout_K_SW128_Atom<BF>{},
                             make_shape(Int<WG_M>{}, Int<K>{}));
    auto lW = tile_to_shape(GMMA::Layout_K_SW128_Atom<BF>{},
                            make_shape(Int<BN>{}, Int<KT>{}));

    extern __shared__ __align__(16) unsigned char smem_raw[];
    BF* s_base = reinterpret_cast<BF*>(smem_raw);
    BF* pXn = s_base;
    BF* pWa = pXn + kWarpgroups * cosize_v<decltype(lXn)>;
    BF* pWb = pWa + STAGES * cosize_v<decltype(lW)>;
    float* pRstd = reinterpret_cast<float*>(pWb + STAGES * cosize_v<decltype(lW)>);
    float* pC1 = pRstd + CTA_M;
    BF* pXnWg = pXn + wg_id * cosize_v<decltype(lXn)>;

    auto sXn = make_tensor(make_smem_ptr(pXnWg), lXn);
    auto sWa = [&](int stage) {
        return make_tensor(make_smem_ptr(pWa + stage * cosize_v<decltype(lW)>), lW);
    };
    auto sWb = [&](int stage) {
        return make_tensor(make_smem_ptr(pWb + stage * cosize_v<decltype(lW)>), lW);
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
    auto tmaWaSlice = tma_wa.get_slice(Int<0>{});
    auto tmaWbSlice = tma_wb.get_slice(Int<0>{});

    auto issue_tma_weight_stage = [&](auto& producer_state, int stage, int nd_base, int kt_base) {
        if (tid == 0) {
            weight_pipe.producer_acquire(producer_state);
            using BarrierType = typename KernelWeightPipeline::ProducerBarrierType;
            BarrierType* tma_barrier = weight_pipe.producer_get_barrier(producer_state);

            auto gWa = local_tile(tmaWaTensor, make_shape(Int<BN>{}, Int<KT>{}),
                                  make_coord(nd_base / BN, kt_base / KT));
            auto gWb = local_tile(tmaWbTensor, make_shape(Int<BN>{}, Int<KT>{}),
                                  make_coord(nd_base / BN, kt_base / KT));
            copy(tma_wa.with(*tma_barrier), tmaWaSlice.partition_S(gWa), tmaWaSlice.partition_D(sWa(stage)));
            copy(tma_wb.with(*tma_barrier), tmaWbSlice.partition_S(gWb), tmaWbSlice.partition_D(sWb(stage)));
            ++producer_state;
        }
    };

    if (tid == 0) {
        cute::prefetch_tma_descriptor(tma_wa.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_wb.get_tma_descriptor());
    }

    static constexpr int kVecElems = 8;
    static_assert((kVecElems * sizeof(__nv_bfloat16)) == sizeof(uint4), "bf16x8 must be 128-bit");
    static_assert((kWarpgroupThreads * kVecElems) % K == 0,
                  "each normalization thread must keep a fixed k-slice");
    union Bf16Vec {
        uint4 vec;
        BF bf[kVecElems];
    };

    for (int vec = wg_tid; vec < (WG_M * K) / kVecElems; vec += kWarpgroupThreads) {
        const int idx = vec * kVecElems;
        const int m = idx / K;
        const int k0 = idx - m * K;
        const int64_t row = static_cast<int64_t>(wg_row0 + m);
        const bool valid = row < M;
        const __nv_bfloat16* src = valid ? x_raw + row * K + k0 : x_raw;
        cp_async_16(&sXn(m, k0), src, valid);
    }
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
    const int param_k0 = (wg_tid % (K / kVecElems)) * kVecElems;
    BF gamma_reg[kVecElems];
    BF beta_reg[kVecElems];
    CUTE_UNROLL
    for (int i = 0; i < kVecElems; ++i) {
        gamma_reg[i] = g_bf[param_k0 + i];
        beta_reg[i] = beta_bf[param_k0 + i];
    }
    // Normalize x -> xn (in smem for WGMMA), AND emit xn to global (Version A: the wgrad GEMMs
    // consume it). Same bf16-rounded value goes to both, matching the Triton reference.
    for (int vec = wg_tid; vec < (WG_M * K) / kVecElems; vec += kWarpgroupThreads) {
        const int idx = vec * kVecElems;
        const int m = idx / K;
        const int k0 = idx - m * K;
        const int64_t row = static_cast<int64_t>(wg_row0 + m);
        const bool valid = row < M;
        const int cta_m = wg_id * WG_M + m;
        const float row_rstd = pRstd[cta_m];
        const float row_c1 = pC1[cta_m];
        Bf16Vec x_vec;
        Bf16Vec xn_vec;
        x_vec.vec = smem_load_u128(&sXn(m, k0));
        CUTE_UNROLL
        for (int i = 0; i < kVecElems; ++i) {
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
        global_store_u128(xn_raw + row * K + k0, xn_vec.vec, valid);
    }
    cutlass::arch::NamedBarrier::arrive_and_wait(kWarpgroupThreads, wg_barrier_id);

    using MmaExpandOp = std::conditional_t<
        BN == 64,
        SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>,
        SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>>;
    TiledMMA mmaE = make_tiled_mma(MmaExpandOp{});
    auto thrE = mmaE.get_slice(wg_tid);
    auto cH = make_identity_tensor(make_shape(Int<WG_M>{}, Int<BN>{}));
    auto tCH = thrE.partition_C(cH);

    // Stage grad_expand for the current ND-tile into the (now-consumed) Wa weight smem, coalesced,
    // via the SAME output_shuffle_idx mapping the stores use — so a fragment element at (m,n) reads
    // its ge back from pGe[output_shuffle_idx(m,n)].
    auto stage_ge = [&](int nd_base) {
        BF* pGe = pWa + wg_id * (WG_M * BN);
        constexpr int kVecsPerRow = BN / kVecElems;
        CUTE_UNROLL
        for (int vec = wg_tid; vec < WG_M * kVecsPerRow; vec += kWarpgroupThreads) {
            const int m = vec / kVecsPerRow;
            const int n = (vec - m * kVecsPerRow) * kVecElems;
            const int64_t row = static_cast<int64_t>(wg_row0 + m);
            const bool valid = row < M;
            const __nv_bfloat16* src = valid ? ge_raw + row * ND + nd_base + n : ge_raw;
            uint4 v = global_load_u128(src, valid);
            smem_store_u128(pGe + output_shuffle_idx<BN>(m, n), v);
        }
        cutlass::arch::NamedBarrier::arrive_and_wait(kWarpgroupThreads, wg_barrier_id);
    };

    // Coalesced write of one (WG_M, BN) tile -> a bf16 (M, *) tensor tile at column base. Takes a
    // per-fragment-element COMPUTE lambda (returns float) instead of a materialized fragment, so the
    // caller needs no persistent output fragment -> lower register pressure (the lever that lets 2
    // blocks/SM fit at 256 threads; see __launch_bounds__).
    auto store_tile = [&](auto compute, __nv_bfloat16* dst_raw, int dst_row_stride, int dst_col_base) {
        BF* pShuf = pWa + wg_id * (WG_M * BN);
        static_assert(kWarpgroups * WG_M * BN <= STAGES * cosize_v<decltype(lW)>,
                      "shuffle tiles must fit in the Wa smem staging area");
        union OutPair {
            uint32_t u32;
            BF bf[2];
        };
        CUTE_UNROLL
        for (int i = 0; i < size(tCH); i += 2) {
            const int m = get<0>(tCH(i));
            const int n = get<1>(tCH(i));
            const int m_next = get<0>(tCH(i + 1));
            const int n_next = get<1>(tCH(i + 1));
            const float v0 = compute(i);
            const float v1 = compute(i + 1);
            if (m_next == m && n_next == n + 1 && (n % 2) == 0) {
                OutPair pair;
                pair.bf[0] = static_cast<BF>(v0);
                pair.bf[1] = static_cast<BF>(v1);
                smem_store_u32(pShuf + output_shuffle_idx<BN>(m, n), pair.u32);
            } else {
                pShuf[output_shuffle_idx<BN>(m, n)] = static_cast<BF>(v0);
                pShuf[output_shuffle_idx<BN>(m_next, n_next)] = static_cast<BF>(v1);
            }
        }
        cutlass::arch::NamedBarrier::arrive_and_wait(kWarpgroupThreads, wg_barrier_id);

        constexpr int kOutVecsPerRow = BN / kVecElems;
        CUTE_UNROLL
        for (int vec = wg_tid; vec < WG_M * kOutVecsPerRow; vec += kWarpgroupThreads) {
            const int m = vec / kOutVecsPerRow;
            const int n = (vec - m * kOutVecsPerRow) * kVecElems;
            const int64_t row = static_cast<int64_t>(wg_row0 + m);
            const bool valid = row < M;
            uint4 out_vec = smem_load_u128(pShuf + output_shuffle_idx<BN>(m, n));
            __nv_bfloat16* dst = valid ? dst_raw + row * dst_row_stride + dst_col_base + n : dst_raw;
            global_store_u128(dst, out_vec, valid);
        }
        cutlass::arch::NamedBarrier::arrive_and_wait(kWarpgroupThreads, wg_barrier_id);
    };

    static constexpr int kKtChunks = K / KT;
    typename KernelWeightPipeline::PipelineState tma_producer_state =
        cutlass::make_producer_start_state<KernelWeightPipeline>();
    typename KernelWeightPipeline::PipelineState tma_consumer_state;

    CUTE_NO_UNROLL
    for (int nd_base = 0, nd_chunk = 0; nd_base < ND; nd_base += BN, ++nd_chunk) {
        const int nd_linear_base = nd_chunk * kKtChunks;

        CUTE_UNROLL
        for (int prefetch = 0; prefetch < STAGES - 1; ++prefetch) {
            if (prefetch < kKtChunks) {
                const int linear_tile = nd_linear_base + prefetch;
                const int kt_prefetch = prefetch * KT;
                issue_tma_weight_stage(tma_producer_state, linear_tile % STAGES, nd_base, kt_prefetch);
            }
        }

        auto a_acc = partition_fragment_C(mmaE, make_shape(Int<WG_M>{}, Int<BN>{}));
        auto b_acc = partition_fragment_C(mmaE, make_shape(Int<WG_M>{}, Int<BN>{}));
        clear(a_acc);
        clear(b_acc);

        CUTE_NO_UNROLL
        for (int kt_base = 0, kt_chunk = 0; kt_base < K; kt_base += KT, ++kt_chunk) {
            const int linear_tile = nd_linear_base + kt_chunk;
            const int read_stage = linear_tile % STAGES;
            const int prefetch_kt = kt_base + (STAGES - 1) * KT;
            if (prefetch_kt < K) {
                const int prefetch_linear_tile = linear_tile + STAGES - 1;
                issue_tma_weight_stage(
                    tma_producer_state, prefetch_linear_tile % STAGES, nd_base, prefetch_kt);
            }
            weight_pipe.consumer_wait(tma_consumer_state);

            auto sXnKt = local_tile(sXn, make_shape(Int<WG_M>{}, Int<KT>{}), make_coord(0, kt_base / KT));
            auto tXnA = thrE.make_fragment_A(thrE.partition_A(sXnKt));
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

            weight_pipe.consumer_release(tma_consumer_state);
            ++tma_consumer_state;
        }

        // Epilogue: stage ge, read it into a bf16 register fragment, then compute + store h, dA, dB
        // ON THE FLY (no fp32 output fragment -> a_acc, b_acc, ge_frag are the only live fragments).
        stage_ge(nd_base);
        auto ge_frag = make_tensor<BF>(a_acc.layout());
        BF* pGe = pWa + wg_id * (WG_M * BN);
        CUTE_UNROLL
        for (int i = 0; i < size(a_acc); ++i) {
            ge_frag(i) = pGe[output_shuffle_idx<BN>(get<0>(tCH(i)), get<1>(tCH(i)))];
        }
        // All ge reads done before store_tile overwrites pGe (== the shuffle scratch).
        cutlass::arch::NamedBarrier::arrive_and_wait(kWarpgroupThreads, wg_barrier_id);

        // h = silu(a) * b
        store_tile([&](int i) {
            const float a = a_acc(i);
            return a * sigmoidf_fast(a) * b_acc(i);
        }, h_raw, ND, nd_base);
        // dA = ge * b * silu'(a),  silu'(a) = sig + silu*(1-sig)
        store_tile([&](int i) {
            const float a = a_acc(i);
            const float sig = sigmoidf_fast(a);
            const float silu = a * sig;
            return static_cast<float>(ge_frag(i)) * b_acc(i) * (sig + silu * (1.0f - sig));
        }, dab_raw, 2 * ND, nd_base);
        // dB = ge * silu(a)
        store_tile([&](int i) {
            const float a = a_acc(i);
            return static_cast<float>(ge_frag(i)) * (a * sigmoidf_fast(a));
        }, dab_raw, 2 * ND, ND + nd_base);

        __syncthreads();  // all pWa scratch use done before the next ND-tile's TMA prefetch
    }
#else
    (void)x_raw; (void)rstd; (void)c1; (void)g_raw; (void)beta_raw; (void)ge_raw;
    (void)h_raw; (void)dab_raw; (void)xn_raw; (void)M; (void)tma_wa; (void)tma_wb;
#endif
}

void check_gatebwd_inputs(
    const torch::Tensor& x, const torch::Tensor& rstd, const torch::Tensor& c1,
    const torch::Tensor& g, const torch::Tensor& beta, const torch::Tensor& wa,
    const torch::Tensor& wb, const torch::Tensor& ge
) {
    TORCH_CHECK(x.is_cuda() && rstd.is_cuda() && c1.is_cuda() && g.is_cuda() && beta.is_cuda() &&
                wa.is_cuda() && wb.is_cuda() && ge.is_cuda(), "all tensors must be CUDA");
    const auto device = x.get_device();
    TORCH_CHECK(rstd.get_device() == device && c1.get_device() == device &&
                g.get_device() == device && beta.get_device() == device &&
                wa.get_device() == device && wb.get_device() == device &&
                ge.get_device() == device, "all tensors must be on the same device");
    TORCH_CHECK(x.is_contiguous() && rstd.is_contiguous() && c1.is_contiguous() &&
                g.is_contiguous() && beta.is_contiguous() && wa.is_contiguous() &&
                wb.is_contiguous() && ge.is_contiguous(), "all tensors must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::ScalarType::BFloat16, "x must be bf16");
    TORCH_CHECK(g.scalar_type() == torch::ScalarType::BFloat16, "g must be bf16");
    TORCH_CHECK(beta.scalar_type() == torch::ScalarType::BFloat16, "beta must be bf16");
    TORCH_CHECK(wa.scalar_type() == torch::ScalarType::BFloat16, "wa must be bf16");
    TORCH_CHECK(wb.scalar_type() == torch::ScalarType::BFloat16, "wb must be bf16");
    TORCH_CHECK(ge.scalar_type() == torch::ScalarType::BFloat16, "grad_expand must be bf16");
    TORCH_CHECK(rstd.scalar_type() == torch::ScalarType::Float, "rstd must be fp32");
    TORCH_CHECK(c1.scalar_type() == torch::ScalarType::Float, "c1 must be fp32");
    TORCH_CHECK(x.dim() == 2, "x must be (M, K)");
    const int64_t K = x.size(1);
    TORCH_CHECK(K == 128 || K == 256 || K == 512, "gatebwd supports K in {128,256,512}");
    TORCH_CHECK(rstd.dim() == 1 && rstd.size(0) == x.size(0), "rstd must be (M,)");
    TORCH_CHECK(c1.dim() == 1 && c1.size(0) == x.size(0), "c1 must be (M,)");
    TORCH_CHECK(g.dim() == 1 && g.size(0) == K, "g must be (K,)");
    TORCH_CHECK(beta.dim() == 1 && beta.size(0) == K, "beta must be (K,)");
    TORCH_CHECK(wa.dim() == 2 && wa.size(1) == K, "wa must be (ND, K)");
    TORCH_CHECK(wb.sizes() == wa.sizes(), "wb must match wa");
    const int64_t ND = wa.size(0);
    TORCH_CHECK(ND == 4 * K, "gatebwd requires ND == 4*K (n=4)");
    TORCH_CHECK(ge.dim() == 2 && ge.size(0) == x.size(0) && ge.size(1) == ND,
                "grad_expand must be (M, ND)");
}

template <int K, int ND, int CTA_M, int WG_M, int BN, int STAGES, int KT>
void launch_gatebwd_kernel(
    const torch::Tensor& x, const torch::Tensor& rstd, const torch::Tensor& c1,
    const torch::Tensor& g, const torch::Tensor& beta, const torch::Tensor& wa,
    const torch::Tensor& wb, const torch::Tensor& ge,
    torch::Tensor& h, torch::Tensor& dab, torch::Tensor& xn,
    int64_t M, cudaStream_t stream
) {
    using Smem = GateBwdSmem<CTA_M, WG_M, K, BN, STAGES, KT>;
    static_assert(Smem::kDynamicSmemBytes <= 227 * 1024, "smem must fit H100");

    auto lW = tile_to_shape(GMMA::Layout_K_SW128_Atom<BF>{}, make_shape(Int<BN>{}, Int<KT>{}));
    auto mWaTma = make_tensor(make_gmem_ptr(reinterpret_cast<const BF*>(wa.data_ptr<at::BFloat16>())),
                              make_shape(Int<ND>{}, Int<K>{}), make_stride(Int<K>{}, Int<1>{}));
    auto mWbTma = make_tensor(make_gmem_ptr(reinterpret_cast<const BF*>(wb.data_ptr<at::BFloat16>())),
                              make_shape(Int<ND>{}, Int<K>{}), make_stride(Int<K>{}, Int<1>{}));
    auto tma_wa = make_tma_copy(SM90_TMA_LOAD{}, mWaTma, lW, make_shape(Int<BN>{}, Int<KT>{}), Int<1>{});
    auto tma_wb = make_tma_copy(SM90_TMA_LOAD{}, mWbTma, lW, make_shape(Int<BN>{}, Int<KT>{}), Int<1>{});

    auto* kernel = transition_gatebwd_rs_wgmma_kernel<
        CTA_M, WG_M, K, ND, BN, STAGES, KT, decltype(tma_wa), decltype(tma_wb)>;
    check_cuda(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Smem::kDynamicSmemBytes),
        "gatebwd dynamic shared-memory attribute failed");
    check_cuda(
        cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100),
        "gatebwd shared-memory carveout attribute failed");

    const dim3 grid(static_cast<unsigned int>((M + CTA_M - 1) / CTA_M));
    const dim3 block(kThreads);
    kernel<<<grid, block, Smem::kDynamicSmemBytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        rstd.data_ptr<float>(), c1.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(g.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(ge.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(h.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(dab.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(xn.data_ptr<at::BFloat16>()),
        M, tma_wa, tma_wb);
    check_cuda(cudaGetLastError(), "gatebwd kernel launch failed");
}

}  // namespace gatebwd_detail

using namespace gatebwd_detail;

std::vector<torch::Tensor> transition_expand_gatebwd_wgmma(
    const torch::Tensor& x, const torch::Tensor& rstd, const torch::Tensor& c1,
    const torch::Tensor& g, const torch::Tensor& beta, const torch::Tensor& wa,
    const torch::Tensor& wb, const torch::Tensor& grad_expand
) {
    check_gatebwd_inputs(x, rstd, c1, g, beta, wa, wb, grad_expand);
    c10::cuda::CUDAGuard device_guard(x.device());
    const int64_t K = x.size(1);
    const int64_t ND = wa.size(0);
    const int64_t M = x.size(0);
    auto h = torch::empty({M, ND}, x.options());
    auto dab = torch::empty({M, 2 * ND}, x.options());
    auto xn = torch::empty({M, K}, x.options());
    if (M == 0) {
        return {h, dab, xn};
    }
    TORCH_CHECK(M % kBlockM == 0, "gatebwd requires M divisible by kBlockM");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (K == 128 && ND == 512) {
        launch_gatebwd_kernel<128, 512, kBlockM, kWarpgroupM, 64, 3, 128>(
            x, rstd, c1, g, beta, wa, wb, grad_expand, h, dab, xn, M, stream);
    } else if (K == 256 && ND == 1024) {
        launch_gatebwd_kernel<256, 1024, kBlockM, kWarpgroupM, 64, 2, 128>(
            x, rstd, c1, g, beta, wa, wb, grad_expand, h, dab, xn, M, stream);
    } else if (K == 512 && ND == 2048) {
        launch_gatebwd_kernel<512, 2048, kBlockM, kWarpgroupM, 64, 2, 64>(
            x, rstd, c1, g, beta, wa, wb, grad_expand, h, dab, xn, M, stream);
    } else {
        TORCH_CHECK(false, "unsupported gatebwd shape");
    }
    return {h, dab, xn};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transition_expand_gatebwd_wgmma", &transition_expand_gatebwd_wgmma,
          "Transition WGMMA fused expand + SwiGLU gate backward (CUDA sm90) -> {h, dAB, xn}");
}
