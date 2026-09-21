// OPM fused epilogue (sm_90a) -- v4.
//
// The outer-product GEMM is left to cuBLAS in its GROUPED layout
//     O[(i,c), (j,e)] = sum_s A2[(i,c), s] * BT[(j,e), s]      (one NT GEMM; no transpose exists)
// and this kernel consumes that layout directly, doing in ONE pass what the stock module spends
// four HBM passes on: the [i,j,c,e] -> [(i,j),(c,e)] permute, the / num_mask division, the bf16
// cast and the c_hidden^2 -> c_z projection (+bias).  The division sits in the fp32 accumulator
// (the projection is linear, so (O/n)*W == (O*W)/n) which also drops one bf16 rounding of P.
//
// The mask normaliser arrives in fp32: this engine's OuterProductMean counts the mask in fp32 and
// clamps at 1, and at S > 256 a bf16 count is no longer exact, so taking it in fp32 keeps the fused
// path on the module's own semantics rather than on the upstream module's bf16 num_mask.
//
// v1-v3 used the wmma API and were bound by the L1/MIO pipe, not by bandwidth or by the tensor
// cores: NCU put L1/TEX throughput at 98 % with DRAM at 10 %, and 65 % of the warp stalls were
// short-scoreboard + MIO-throttle.  wmma::load_matrix_sync issues a fragment as many narrow
// per-thread loads, and with one B fragment per mma the loads outnumbered the math.  v4 drops to
// mma.sync + ldmatrix:
//   * A: one ldmatrix.x4 per 16x16 tile per k step.
//   * B: the projection weight is CONSTANT, so it is pre-swizzled on the host into mma fragment
//     order; the kernel reads each fragment as one 8 B per-thread load (no ldmatrix needed).
//   * warp tile 2 row x 4 n tiles: 6 loads per 8 mma, against 5 loads per 4 mma in v3.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>

namespace {

constexpr int CH = 32;             // c_hidden
constexpr int CZ = 128;            // d_pair
constexpr int BI = 4;              // i tokens per CTA
constexpr int BJ = 16;             // j tokens per CTA  (BJ*CH = 512 contiguous columns of O per read)
constexpr int NP = BI * BJ;        // 64 (i,j) pairs per CTA
constexpr int KC = 128;            // K chunk = 4 c values
constexpr int KCP = KC + 16;       // padded shared-memory row stride (bank spread)
constexpr int NCHUNK = CH * CH / KC;
constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;
constexpr int ROWT = NP / 16;      // 4 mma row tiles
constexpr int NTILE = CZ / 8;      // 16 mma n tiles
constexpr int RPW = 2;             // row tiles per warp
constexpr int NPW = 4;             // n tiles per warp   (RPW*NPW = 8 accumulators = 32 registers)
constexpr int KSTEP = KC / 16;     // 8 k steps per chunk
constexpr int VEC = NP * KC / 8 / THREADS;
constexpr int SMEM_STAGE = 2 * NP * KCP * 2;
constexpr int SMEM_BYTES = SMEM_STAGE > NP * CZ * 4 ? SMEM_STAGE : NP * CZ * 4;

__device__ __forceinline__ void stage(const __nv_bfloat16* __restrict__ O, __nv_bfloat16* dst,
                                      int i0, int j0, int kc0, long M, int tid) {
#pragma unroll
  for (int r = 0; r < VEC; ++r) {
    const int v = tid + r * THREADS;
    const int e8 = v & 3;
    const int jl = (v >> 2) & (BJ - 1);
    const int cl = (v >> 6) & 3;
    const int il = v >> 8;
    const long row = (long)(i0 + il) * CH + (kc0 / CH) + cl;
    const long col = (long)j0 * CH + jl * CH + e8 * 8;
    __pipeline_memcpy_async(dst + (long)(il * BJ + jl) * KCP + cl * CH + e8 * 8, O + row * M + col, 16);
  }
  __pipeline_commit();
}

__global__ __launch_bounds__(THREADS) void opm_epilogue_kernel(
    const __nv_bfloat16* __restrict__ O,      // [N*CH, N*CH] grouped GEMM output
    const float* __restrict__ NORM,           // [N, N]  mask counts in fp32, already clamped >= 1
    const __nv_bfloat16* __restrict__ BF,     // [K/16][CZ/8][32][4] pre-swizzled projection weight
    const float* __restrict__ BIAS,           // [CZ]  bf16-rounded bias, held in fp32
    __nv_bfloat16* __restrict__ OUT,          // [N, N, CZ]
    int N, long M) {
  extern __shared__ char smem[];
  __nv_bfloat16* sP = reinterpret_cast<__nv_bfloat16*>(smem);        // [2][NP][KCP]
  float* sAcc = reinterpret_cast<float*>(smem);                      // [NP][CZ] after the K loop
  __shared__ float sInv[NP];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int rg = (warp & 1) * RPW;          // this warp's first row tile
  const int ng = (warp >> 1) * NPW;         // this warp's first n tile
  const int i0 = blockIdx.y * BI;
  const int j0 = blockIdx.x * BJ;

  if (tid < NP) {
    const int il = tid / BJ, jl = tid - il * BJ;
    sInv[tid] = 1.0f / NORM[(long)(i0 + il) * N + (j0 + jl)];
  }

  float acc[RPW][NPW][4];
#pragma unroll
  for (int r = 0; r < RPW; ++r)
#pragma unroll
    for (int n = 0; n < NPW; ++n)
#pragma unroll
      for (int t = 0; t < 4; ++t) acc[r][n][t] = 0.0f;

  stage(O, sP, i0, j0, 0, M, tid);
#pragma unroll 1
  for (int ck = 0; ck < NCHUNK; ++ck) {
    const int buf = ck & 1;
    if (ck + 1 < NCHUNK) stage(O, sP + (1 - buf) * NP * KCP, i0, j0, (ck + 1) * KC, M, tid);
    __pipeline_wait_prior(ck + 1 < NCHUNK ? 1 : 0);
    __syncthreads();
    const __nv_bfloat16* p = sP + buf * NP * KCP;
#pragma unroll
    for (int ks = 0; ks < KSTEP; ++ks) {
      uint32_t a[RPW][4];
#pragma unroll
      for (int r = 0; r < RPW; ++r) {
        // ldmatrix.x4 of the 16x16 tile at rows (rg+r)*16, k columns ks*16
        const __nv_bfloat16* src = p + (long)((rg + r) * 16 + (lane & 15)) * KCP + ks * 16 + (lane >> 4) * 8;
        uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(src));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(a[r][0]), "=r"(a[r][1]), "=r"(a[r][2]), "=r"(a[r][3]) : "r"(s));
      }
      const int kglob = ck * KSTEP + ks;
#pragma unroll
      for (int n = 0; n < NPW; ++n) {
        const uint32_t* bptr = reinterpret_cast<const uint32_t*>(
            BF + ((long)(kglob * NTILE + ng + n) * 32 + lane) * 4);
        uint32_t b0 = bptr[0], b1 = bptr[1];
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                       "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                       : "+f"(acc[r][n][0]), "+f"(acc[r][n][1]), "+f"(acc[r][n][2]), "+f"(acc[r][n][3])
                       : "r"(a[r][0]), "r"(a[r][1]), "r"(a[r][2]), "r"(a[r][3]), "r"(b0), "r"(b1));
        }
      }
    }
    __syncthreads();
  }

  __syncthreads();
  // mma.m16n8k16 accumulator map: {d0,d1} row lane/4, cols (lane%4)*2 + {0,1}; {d2,d3} row lane/4 + 8
#pragma unroll
  for (int r = 0; r < RPW; ++r) {
#pragma unroll
    for (int n = 0; n < NPW; ++n) {
      const int col = (ng + n) * 8 + (lane & 3) * 2;
      const int row0 = (rg + r) * 16 + (lane >> 2);
      sAcc[(long)row0 * CZ + col] = acc[r][n][0];
      sAcc[(long)row0 * CZ + col + 1] = acc[r][n][1];
      sAcc[(long)(row0 + 8) * CZ + col] = acc[r][n][2];
      sAcc[(long)(row0 + 8) * CZ + col + 1] = acc[r][n][3];
    }
  }
  __syncthreads();

  // --- / num_mask and + bias in fp32 -> bf16, one rounding; 8 channels per vector store
  for (int v = tid; v < NP * CZ / 8; v += THREADS) {
    const int z8 = (v & (CZ / 8 - 1)) * 8;
    const int p = v / (CZ / 8);
    const int il = p / BJ, jl = p - il * BJ;
    const float inv = sInv[p];
    uint4 out;
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      float2 f;
      f.x = fmaf(sAcc[(long)p * CZ + z8 + 2 * t], inv, BIAS[z8 + 2 * t]);
      f.y = fmaf(sAcc[(long)p * CZ + z8 + 2 * t + 1], inv, BIAS[z8 + 2 * t + 1]);
      h[t] = __float22bfloat162_rn(f);
    }
    *reinterpret_cast<uint4*>(OUT + ((long)(i0 + il) * N + (j0 + jl)) * CZ + z8) = out;
  }
}
}  // namespace

torch::Tensor opm_epilogue(torch::Tensor O, torch::Tensor norm, torch::Tensor bf, torch::Tensor bias, int64_t N) {
  TORCH_CHECK(O.is_cuda() && O.scalar_type() == torch::kBFloat16 && O.is_contiguous(), "O: contiguous cuda bf16");
  TORCH_CHECK(norm.scalar_type() == torch::kFloat32 && norm.is_contiguous(), "norm: contiguous fp32");
  TORCH_CHECK(bf.scalar_type() == torch::kBFloat16 && bf.is_contiguous() && bf.numel() == CH * CH * CZ, "bf: swizzled [K/16, CZ/8, 32, 4] bf16");
  TORCH_CHECK(bias.scalar_type() == torch::kFloat32 && bias.numel() == CZ, "bias: fp32 [128]");
  TORCH_CHECK(N % BI == 0 && N % BJ == 0, "N must be a multiple of ", BI, " and ", BJ);
  const long M = O.size(1);
  auto out = torch::empty({1, N, N, (long)CZ}, O.options());
  dim3 grid(N / BJ, N / BI);
  auto stream = at::cuda::getCurrentCUDAStream();
  opm_epilogue_kernel<<<grid, THREADS, SMEM_BYTES, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(O.data_ptr<at::BFloat16>()),
      norm.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(bf.data_ptr<at::BFloat16>()),
      bias.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      (int)N, M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("opm_epilogue", &opm_epilogue, "fused OPM epilogue (div-norm + proj_out + bias)"); }
