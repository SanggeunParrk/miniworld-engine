// vendored from team-gm psk/benchmark@e085d6d : src/team_gm/modules/kernels/cuda/layernorm/layer_norm_cuda_kernel.cu
/*
 * layer_norm_cuda_kernel.cu
 *
 * Forward LayerNorm in plain CUDA.
 * Matches the Triton layer_norm_fwd_fused algorithm:
 *   mean  = sum(x) / N
 *   var   = sum((x - mean)^2) / N
 *   rstd  = 1 / sqrt(var + eps)
 *   y     = (x - mean) * rstd * w + b
 *
 * One CUDA block per row.
 * Supports float32 / float16 / bfloat16 via template.
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <cstdlib>
#include <type_traits>

// ─────────────────────────────────────────────
// 1.  Warp-level reduction helpers
// ─────────────────────────────────────────────

__inline__ __device__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Full block reduction.
// smem must have at least (blockDim.x / 32) floats.
// Returns the reduced value in thread 0; other threads get garbage.
__inline__ __device__ float block_reduce_sum(float val, float* smem) {
    const int lane  = threadIdx.x & 31;          // lane within warp
    const int wid   = threadIdx.x >> 5;          // warp index

    val = warp_reduce_sum(val);                   // reduce within warp

    if (lane == 0) smem[wid] = val;              // each warp's partial
    __syncthreads();

    const int n_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < n_warps) ? smem[threadIdx.x] : 0.f;
    val = warp_reduce_sum(val);                   // reduce across warps
    return val;
}

// ─────────────────────────────────────────────
// 2.  Forward kernel  (one block = one row)
// ─────────────────────────────────────────────

template <typename scalar_t>
__global__ void layer_norm_fwd_kernel(
    const scalar_t* __restrict__ X,   // [M, N]
    scalar_t*       __restrict__ Y,   // [M, N]
    const scalar_t* __restrict__ W,   // [N]
    const scalar_t* __restrict__ B,   // [N]
    float*          __restrict__ Mean, // [M]
    float*          __restrict__ Rstd, // [M]
    const int N,
    const float eps)
{
    // Shared memory: enough for one float per warp
    extern __shared__ float smem[];   // size = ceil(blockDim.x/32) floats

    const int row = blockIdx.x;
    const scalar_t* __restrict__ x = X + row * N;
    scalar_t*       __restrict__ y = Y + row * N;

    // ── Pass 1: mean ─────────────────────────
    float sum = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        sum += (float)x[i];

    sum = block_reduce_sum(sum, smem);

    float mean;
    if (threadIdx.x == 0) {
        mean    = sum / (float)N;
        smem[0] = mean;              // broadcast via smem
    }
    __syncthreads();
    mean = smem[0];

    // ── Pass 2: variance ─────────────────────
    float var = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float d = (float)x[i] - mean;
        var += d * d;
    }

    var = block_reduce_sum(var, smem);

    float rstd;
    if (threadIdx.x == 0) {
        rstd    = rsqrtf(var / (float)N + eps);
        smem[0] = rstd;              // broadcast via smem
        Mean[row] = mean;
        Rstd[row] = rstd;
    }
    __syncthreads();
    rstd = smem[0];

    // ── Pass 3: normalize + affine ────────────
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float x_hat = ((float)x[i] - mean) * rstd;
        y[i] = (scalar_t)(x_hat * (float)W[i] + (float)B[i]);
    }
}

// ─────────────────────────────────────────────
// 3.  Host-side launcher
// ─────────────────────────────────────────────

// Next power of two, clamped to [32, 1024]
static int choose_block_size(int N) {
    int t = 32;
    while (t < N && t < 1024) t <<= 1;
    return t;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
layer_norm_cuda_fwd(
    torch::Tensor x,        // [..., N]  – any leading dims
    torch::Tensor weight,   // [N]
    torch::Tensor bias,     // [N]
    float eps)
{
    TORCH_CHECK(x.is_cuda(),      "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(),   "bias must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    const int N = x.size(-1);
    const int M = x.numel() / N;

    auto y    = torch::empty_like(x);
    auto mean = torch::empty({M}, x.options().dtype(torch::kFloat32));
    auto rstd = torch::empty({M}, x.options().dtype(torch::kFloat32));

    const int block = choose_block_size(N);
    const int grid  = M;
    // smem: one float per warp
    const int smem_bytes = ((block + 31) / 32) * sizeof(float);
    const auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "layer_norm_fwd_cuda",
        [&]() {
            layer_norm_fwd_kernel<scalar_t><<<grid, block, smem_bytes, stream>>>(
                x.data_ptr<scalar_t>(),
                y.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                mean.data_ptr<float>(),
                rstd.data_ptr<float>(),
                N, eps);
        });

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "layer_norm_fwd_cuda kernel failed");
    return {y, mean, rstd};
}

// ─────────────────────────────────────────────
// 4.  Backward kernels  (warp-per-row + register column-partials, persistent grid)
// ─────────────────────────────────────────────

__inline__ __device__ float warp_reduce_sum_xor(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset, 32);
    }
    return val;
}

constexpr int LN_BWD_WARPS_PER_BLOCK = 4;
constexpr int LN_BWD_BLOCK_THREADS = LN_BWD_WARPS_PER_BLOCK * 32;

static int choose_bwd_grid(int N) {
    static int sm_count = []() {
        int device = 0;
        int count = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device);
        return count;
    }();
    static int env_waves = []() {
        const char* e = getenv("LNBWD_WAVES");
        return e ? std::max(1, atoi(e)) : 0;
    }();
    // d<=128 has tiny rows (4 cols/lane) -> needs more waves than the D512 sweet spot (4).
    const int waves = env_waves > 0 ? env_waves : ((N <= 128) ? 8 : 4);
    return std::max(1, sm_count * waves);
}

// Vectorized backward main kernel.
//   Layout: contiguous-per-thread. Vector transaction v, lane l owns the EPT columns
//   [(v*32 + l)*EPT .. +EPT). A warp's transaction therefore covers 32*EPT consecutive
//   columns as ONE coalesced wide load (uint2 = 256B, uint4 = 512B) instead of the old
//   strided scalar load (64B / 2 sectors per instruction). This raises bytes-in-flight
//   per instruction (memory-level parallelism) without spending registers.
//   TX_BYTES = width of one vector transaction (8 -> uint2, 16 -> uint4);
//   EPT = elements per transaction = TX_BYTES / sizeof(scalar_t).
//   The register-column-partial design is unchanged: each lane still privately owns its
//   K = N/32 columns in acc_dw[K]/acc_db[K] (no atomics / no shared / no spill).
template <typename scalar_t, int MAX_K, int TX_BYTES>
__launch_bounds__(LN_BWD_BLOCK_THREADS, 2)
__global__ void layer_norm_bwd_main_kernel(
    const scalar_t* __restrict__ DY,       // [M, N]
    const scalar_t* __restrict__ X,        // [M, N]
    const scalar_t* __restrict__ W,        // [N]
    const float*    __restrict__ Mean,     // [M]
    const float*    __restrict__ Rstd,     // [M]
    const float*    __restrict__ RS,       // [M] per-row scale (mask fold), or nullptr
    scalar_t*       __restrict__ DX,       // [M, N]
    float*          __restrict__ PartDW,   // [gridDim.x * 4, N]
    float*          __restrict__ PartDB,   // [gridDim.x * 4, N]
    const int M,
    const int N)
{
    static_assert(MAX_K == 4 || MAX_K == 8 || MAX_K == 16 || MAX_K == 32,
                  "MAX_K must cover D<=128/256/512/1024");
    static_assert(TX_BYTES == 8 || TX_BYTES == 16, "TX_BYTES must be 8 (uint2) or 16 (uint4)");

    using VecT = typename std::conditional<TX_BYTES == 16, uint4, uint2>::type;
    constexpr int EPT = TX_BYTES / (int)sizeof(scalar_t);   // elements per vector transaction
    constexpr int NV_MAX = (MAX_K + EPT - 1) / EPT;         // vector transactions per lane
    union Pack { VecT vec; scalar_t s[EPT]; };

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int global_warp = blockIdx.x * LN_BWD_WARPS_PER_BLOCK + warp;
    const int total_warps = gridDim.x * LN_BWD_WARPS_PER_BLOCK;
    const int K = (N + 31) >> 5;

    float acc_dw[MAX_K];
    float acc_db[MAX_K];
#pragma unroll
    for (int k = 0; k < MAX_K; ++k) {
        acc_dw[k] = 0.f;
        acc_db[k] = 0.f;
    }

    for (int row = global_warp; row < M; row += total_warps) {
        const scalar_t* __restrict__ x = X + row * N;
        const scalar_t* __restrict__ dy = DY + row * N;
        scalar_t* __restrict__ dx = DX + row * N;
        const float mean = Mean[row];
        const float rstd = Rstd[row];
        // Backward of y = LN(x)*rs: scale the incoming grad by the per-row rs (uniform across
        // the warp, no divergence). dx/dw/db all follow from the scaled dy, matching triton.
        const float rs = (RS != nullptr) ? RS[row] : 1.0f;

        scalar_t x_vals[MAX_K];
        scalar_t dy_vals[MAX_K];
        // Cache w*dy per column so the dx pass reuses it instead of reloading W and
        // recomputing the product (this bwd is SM-issue-bound on sm100, not DRAM-bound).
        float wdy_vals[MAX_K];

        // Coalesced vector loads. N % EPT == 0 for every launched (N, TX_BYTES) combo, so a
        // transaction whose base column is in range never overshoots N.
#pragma unroll
        for (int v = 0; v < NV_MAX; ++v) {
            const int col0 = (v * 32 + lane) * EPT;
            if (col0 < N) {
                Pack xp, dyp;
                xp.vec = *reinterpret_cast<const VecT*>(x + col0);
                dyp.vec = *reinterpret_cast<const VecT*>(dy + col0);
#pragma unroll
                for (int e = 0; e < EPT; ++e) {
                    x_vals[v * EPT + e] = xp.s[e];
                    dy_vals[v * EPT + e] = dyp.s[e];
                }
            }
        }

        float sum_wdy = 0.f;
        float sum_wdy_xhat = 0.f;
#pragma unroll
        for (int v = 0; v < NV_MAX; ++v) {
            const int col0 = (v * 32 + lane) * EPT;
            if (col0 < N) {
                Pack wp;
                wp.vec = *reinterpret_cast<const VecT*>(W + col0);
#pragma unroll
                for (int e = 0; e < EPT; ++e) {
                    const int k = v * EPT + e;
                    const float x_v = (float)x_vals[k];
                    const float dy_v = (float)dy_vals[k] * rs;
                    const float xhat = (x_v - mean) * rstd;
                    const float wdy = (float)wp.s[e] * dy_v;
                    wdy_vals[k] = wdy;
                    sum_wdy += wdy;
                    sum_wdy_xhat += wdy * xhat;
                }
            }
        }

        const float s2 = warp_reduce_sum_xor(sum_wdy);
        const float s1 = warp_reduce_sum_xor(sum_wdy_xhat);
        const float c1 = s1 / (float)N;
        const float c2 = s2 / (float)N;

#pragma unroll
        for (int v = 0; v < NV_MAX; ++v) {
            const int col0 = (v * 32 + lane) * EPT;
            if (col0 < N) {
                Pack dxp;
#pragma unroll
                for (int e = 0; e < EPT; ++e) {
                    const int k = v * EPT + e;
                    const float x_v = (float)x_vals[k];
                    const float dy_v = (float)dy_vals[k] * rs;
                    const float xhat = (x_v - mean) * rstd;
                    const float wdy = wdy_vals[k];
                    const float dx_v = (wdy - (xhat * c1 + c2)) * rstd;
                    dxp.s[e] = (scalar_t)dx_v;
                    acc_dw[k] += dy_v * xhat;
                    acc_db[k] += dy_v;
                }
                *reinterpret_cast<VecT*>(dx + col0) = dxp.vec;
            }
        }
    }

    // Partial write: acc for column c lands at PartDW[global_warp*N + c]; the reduce kernel is
    // column-major agnostic to which lane produced c. Written once per persistent warp (tiny).
#pragma unroll
    for (int v = 0; v < NV_MAX; ++v) {
#pragma unroll
        for (int e = 0; e < EPT; ++e) {
            const int k = v * EPT + e;
            const int col = (v * 32 + lane) * EPT + e;
            if (k < K && col < N) {
                PartDW[global_warp * N + col] = acc_dw[k];
                PartDB[global_warp * N + col] = acc_db[k];
            }
        }
    }
}

template <typename scalar_t>
__global__ void layer_norm_bwd_reduce_kernel(
    const float* __restrict__ PartDW,   // [total_warps, N]
    const float* __restrict__ PartDB,
    scalar_t*    __restrict__ DW,        // [N]
    scalar_t*    __restrict__ DB,        // [N]
    const int total_warps,
    const int N)
{
    extern __shared__ float sdata[];
    float* smem_dw = sdata;
    float* smem_db = sdata + blockDim.x;
    const int col = blockIdx.x;          // one block reduces one column over all warps
    if (col >= N) return;

    float acc_dw = 0.f;
    float acc_db = 0.f;
    for (int r = threadIdx.x; r < total_warps; r += blockDim.x) {
        acc_dw += PartDW[r * N + col];
        acc_db += PartDB[r * N + col];
    }
    smem_dw[threadIdx.x] = acc_dw;
    smem_db[threadIdx.x] = acc_db;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem_dw[threadIdx.x] += smem_dw[threadIdx.x + stride];
            smem_db[threadIdx.x] += smem_db[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        DW[col] = (scalar_t)smem_dw[0];
        DB[col] = (scalar_t)smem_db[0];
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
layer_norm_cuda_bwd(
    torch::Tensor dy,       // [..., N]
    torch::Tensor x,        // [..., N]
    torch::Tensor weight,   // [N]
    torch::Tensor mean,     // [M], fp32
    torch::Tensor rstd,     // [M], fp32
    c10::optional<torch::Tensor> rowscale)  // [M], fp32, or None (per-row mask fold)
{
    TORCH_CHECK(dy.is_cuda(),     "dy must be a CUDA tensor");
    TORCH_CHECK(x.is_cuda(),      "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(mean.is_cuda(),   "mean must be a CUDA tensor");
    TORCH_CHECK(rstd.is_cuda(),   "rstd must be a CUDA tensor");
    TORCH_CHECK(dy.scalar_type() == x.scalar_type(), "dy and x must have the same dtype");
    TORCH_CHECK(weight.scalar_type() == x.scalar_type(), "weight and x must have the same dtype");
    TORCH_CHECK(mean.scalar_type() == at::ScalarType::Float, "mean must be fp32");
    TORCH_CHECK(rstd.scalar_type() == at::ScalarType::Float, "rstd must be fp32");
    TORCH_CHECK(x.size(-1) == dy.size(-1), "x and dy must have the same last dimension");
    TORCH_CHECK(weight.numel() == x.size(-1), "weight must have shape [N]");
    TORCH_CHECK(x.numel() == dy.numel(), "x and dy must have the same number of elements");

    const auto out_sizes = dy.sizes().vec();
    auto x_contig = x.contiguous();
    auto dy_contig = dy.contiguous();
    auto w_contig = weight.contiguous();
    auto mean_contig = mean.contiguous();
    auto rstd_contig = rstd.contiguous();

    const int N = x_contig.size(-1);
    const int M = x_contig.numel() / N;
    TORCH_CHECK(N > 0 && N <= 1024, "LayerNorm CUDA backward supports 1 <= N <= 1024");
    TORCH_CHECK(mean_contig.numel() == M, "mean must have shape [M]");
    TORCH_CHECK(rstd_contig.numel() == M, "rstd must have shape [M]");

    // Optional per-row scale (mask fold): y = LN(x)*rs. fp32 [M]; nullptr disables it.
    const float* rs_ptr = nullptr;
    torch::Tensor rs_contig;
    if (rowscale.has_value() && rowscale->defined()) {
        rs_contig = rowscale->to(torch::kFloat32).contiguous();
        TORCH_CHECK(rs_contig.numel() == M, "rowscale must have shape [M]");
        rs_ptr = rs_contig.data_ptr<float>();
    }

    auto dx_2d = torch::empty_like(dy_contig);
    auto dw = torch::empty_like(w_contig);
    auto db = torch::empty_like(w_contig);

    constexpr int block = LN_BWD_BLOCK_THREADS;
    const int grid = choose_bwd_grid(N);
    const int total_warps = grid * LN_BWD_WARPS_PER_BLOCK;
    auto partial_opts = x_contig.options().dtype(torch::kFloat32);
    auto partial_dw = torch::empty({total_warps, N}, partial_opts);
    auto partial_db = torch::empty({total_warps, N}, partial_opts);
    constexpr int main_smem = 0;
    const auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x_contig.scalar_type(),
        "layer_norm_bwd_cuda",
        [&]() {
            // Template MAX_K to the actual column count (N/32) so small d doesn't pay for
            // 16-slot register arrays + wasted unrolled iterations (d=128 -> MAX_K=4 = parity
            // with Triton; MAX_K=16 was 2.2x slower at d=128).
            const int max_k = (N <= 128) ? 4 : (N <= 256) ? 8 : (N <= 512) ? 16 : 32;
            // Widest coalesced vector transaction that still tiles the row with a full warp:
            // uint4 (16B) needs N % (32 * 16/elt) == 0, else fall back to uint2 (8B).
            const int elt = (int)sizeof(scalar_t);
            const int txb = (N % (32 * (16 / elt)) == 0) ? 16 : 8;
            auto launch = [&](auto mk, auto tx) {
                constexpr int MK = decltype(mk)::value;
                constexpr int TX = decltype(tx)::value;
                layer_norm_bwd_main_kernel<scalar_t, MK, TX><<<grid, block, main_smem, stream>>>(
                    dy_contig.data_ptr<scalar_t>(),
                    x_contig.data_ptr<scalar_t>(),
                    w_contig.data_ptr<scalar_t>(),
                    mean_contig.data_ptr<float>(),
                    rstd_contig.data_ptr<float>(),
                    rs_ptr,
                    dx_2d.data_ptr<scalar_t>(),
                    partial_dw.data_ptr<float>(),
                    partial_db.data_ptr<float>(),
                    M, N);
            };
            auto pick_tx = [&](auto mk) {
                if (txb == 16) launch(mk, std::integral_constant<int, 16>{});
                else           launch(mk, std::integral_constant<int, 8>{});
            };
            if (max_k == 4)       pick_tx(std::integral_constant<int, 4>{});
            else if (max_k == 8)  pick_tx(std::integral_constant<int, 8>{});
            else if (max_k == 16) pick_tx(std::integral_constant<int, 16>{});
            else                  pick_tx(std::integral_constant<int, 32>{});
        });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "layer_norm_bwd_cuda main kernel failed");

    constexpr int reduce_block = 256;
    const int reduce_smem = 2 * reduce_block * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x_contig.scalar_type(),
        "layer_norm_bwd_reduce_cuda",
        [&]() {
            layer_norm_bwd_reduce_kernel<scalar_t><<<N, reduce_block, reduce_smem, stream>>>(
                partial_dw.data_ptr<float>(),
                partial_db.data_ptr<float>(),
                dw.data_ptr<scalar_t>(),
                db.data_ptr<scalar_t>(),
                total_warps,
                N);
        });
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "layer_norm_bwd_cuda reduce kernel failed");

    return {dx_2d.view(out_sizes), dw, db};
}

// ─────────────────────────────────────────────
// 5.  pybind11 module
// ─────────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "layer_norm_fwd",
        &layer_norm_cuda_fwd,
        "LayerNorm forward (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("eps") = 1e-5f);
    m.def(
        "layer_norm_bwd",
        &layer_norm_cuda_bwd,
        "LayerNorm backward (CUDA)",
        py::arg("dy"),
        py::arg("x"),
        py::arg("weight"),
        py::arg("mean"),
        py::arg("rstd"),
        py::arg("rowscale") = c10::optional<torch::Tensor>());
}
