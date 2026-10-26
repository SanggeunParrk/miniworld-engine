// Chai-1 OuterProductMean (8 groups of 8-wide halves), H100: the pass between the grouped outer-product GEMM and the pair output.
//
//   O[g, i K + k, j K + l]  (cuBLAS: one batched GEMM over the MSA rows)        K = 8, G = 8, P = G K K = 512 products per pair
//   x_p       = O[g, i K + k, j K + l],  p = (g, k, l)
//   xhat_p    = (x_p - mean) rstd         LayerNorm over the 512 products of a pair (eps 0.1)
//   z[i, j]   = residual[i, j] + bf16(sum_p xhat_p W'[o, p] + b'[o])            W' = W_out . gamma (columns), b' = b_out + W_out beta
//
// ``chai_epilogue``: one persistent CTA per (tile of 64 consecutive pairs of a row i, slice of 128 outputs): a producer warp TMA-loads the pair
// tile of O straight into the K-major wgmma layout (a 4-D box (l, k, j, g): for each group g the 64 j x 8 k x 8 l values land as 64 rows of
// 128 B, the 16-byte chunk = k, swizzled) and the 128-wide slice of W' in 64-deep K blocks; four consumer warps (one warpgroup) normalise the
// tile in place (statistics of 512 values per pair, two threads a pair), multiply it by the slice of W' (wgmma m64n128k16, K = 512), add the
// bias and the residual and store the rows.  The pair tile never visits HBM in its [pair, 512] form; with ``save`` the normalised tile is also
// TMA-stored (y, the operand of dW_out) with the statistics, for the backward.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cudaTypedefs.h>
#include "chai_helpers.cuh"

namespace {
typedef __nv_bfloat16 bf16;
using namespace chai;

constexpr int G = 8, KK = 8, P = 512;      // groups, group width, products per pair
constexpr int BM = 64;                     // pairs per tile
constexpr int NZ = 128;                    // output channels per CTA
constexpr int KBN = P / 64;                // K blocks = groups
constexpr int A_EL = BM * P;               // elements of one pair tile
constexpr int B_EL = NZ * 64;              // elements of one W' stage
constexpr int NBS = 2;                     // W' ring stages
constexpr int NAB = 1;                     // pair-tile buffers: one, two CTAs share an SM (the other one hides this one's load latency)
constexpr int OUT_LD = NZ + 8;
constexpr size_t SMEM = size_t(NAB * A_EL + NBS * B_EL + BM * NZ) * 2 + 256;

PFN_cuTensorMapEncodeTiled_v12000 tma_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (fn == nullptr) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult qr;
#if CUDART_VERSION >= 12050
    C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &qr));
#else
    C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &qr));
#endif
    TORCH_CHECK(p != nullptr && qr == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}

void encode(CUtensorMap* map, const void* ptr, int rank, const uint64_t* dims, const uint64_t* strides, const uint32_t* box, bool swz = true) {
  uint32_t elem[5] = {1, 1, 1, 1, 1};
  CUresult r = tma_encode()(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, const_cast<void*>(ptr), dims, strides, box, elem,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, swz ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
}

template <bool SAVE>
__global__ void __launch_bounds__(160, 2) chai_epilogue_kernel(
    const __grid_constant__ CUtensorMap omap, const __grid_constant__ CUtensorMap wmap, const __grid_constant__ CUtensorMap ymap,
    const float* __restrict__ bias, const bf16* __restrict__ residual, bf16* __restrict__ out, float* __restrict__ stats,
    int N, int CZ, int tiles, float eps) {
  extern __shared__ __align__(1024) unsigned char raw_[];
  unsigned char* raw = raw_;
  bf16* sA = reinterpret_cast<bf16*>(raw);                    // [2][8 kb][64 j][64]
  bf16* sB = sA + NAB * A_EL;                                 // [NBS][128 n][64]
  bf16* sO = sB + NBS * B_EL;                                 // [64][128], 16-byte chunks XOR row % 8
  float (*sred)[64][2] = reinterpret_cast<float (*)[64][2]>(sO);   // LayerNorm exchange (the staging tile is idle then)
  uint64_t* bars = reinterpret_cast<uint64_t*>(sO + BM * NZ + 0);
  uint64_t* a_full = bars; uint64_t* a_empty = a_full + NAB; uint64_t* b_full = a_empty + NAB; uint64_t* b_empty = b_full + NBS;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int tpr = N / 32;                                     // a tile: 2 consecutive rows i x 32 consecutive columns j
  const int z0 = blockIdx.y * NZ;
  if (tid == 0) {
    for (int k = 0; k < NAB; ++k) { mb_init(a_full + k, 1); mb_init(a_empty + k, 1); }
    for (int k = 0; k < NBS; ++k) { mb_init(b_full + k, 1); mb_init(b_empty + k, 4); }
    fence_init();
  }
  __syncthreads();

  if (warp == 4) {                                            // producer
    if (lane == 0) {
      auto load_a = [&](int t, int n) {
        const int slot = n % NAB, i0 = (t / tpr) * 2, j0 = (t % tpr) * 32;
        if (n >= NAB) mb_wait(a_empty + slot, ((n / NAB) - 1) & 1);
        mb_expect(a_full + slot, A_EL * 2);
#pragma unroll
        for (int g = 0; g < G; ++g) tma_ld4(&omap, sA + slot * A_EL + g * (BM * 64), a_full + slot, j0 * 8, i0, 0, g);
      };
      int n = 0, gb = 0;
      if (blockIdx.x < tiles) load_a(blockIdx.x, 0);
      for (int t = blockIdx.x; t < tiles; t += gridDim.x, ++n) {
        for (int kb = 0; kb < KBN; ++kb, ++gb) {
          if (NAB > 1 && kb == 0 && t + (int)gridDim.x < tiles) load_a(t + gridDim.x, n + 1);
          const int slot = gb % NBS;
          if (gb >= NBS) mb_wait(b_empty + slot, ((gb / NBS) - 1) & 1);
          mb_expect(b_full + slot, B_EL * 2);
          tma_ld2(&wmap, sB + slot * B_EL, b_full + slot, kb * 64, z0);
        }
        if (NAB == 1 && t + (int)gridDim.x < tiles) load_a(t + gridDim.x, n + 1);
      }
    }
  } else {                                                    // consumers: one warpgroup
    int n = 0, gb = 0;
    for (int t = blockIdx.x; t < tiles; t += gridDim.x, ++n) {
      const int slot = n % NAB, i0 = (t / tpr) * 2, j0 = (t % tpr) * 32;
      const long pairA = (long)i0 * N + j0;                   // pair of tile row m: pairA + (m >> 5) * N + (m & 31)
      bf16* A = sA + slot * A_EL;
      mb_wait(a_full + slot, (n / NAB) & 1);
      // ---- LayerNorm of the 512 products of every pair, in place: two threads a pair (groups 4 h .. 4 h + 3), the chunk (g, j, k) at
      //      off_n(64, g, j, k) holds the 8 values l of (g, k)
      {
        const int j = tid & 63, h = tid >> 6;                 // a warp reads 32 consecutive chunks (512 B): conflict-free
        const bf16* A0 = A + j * 8;
        const float x0 = __bfloat162float(A[j * 8]);          // shift: the sums below are of (x - x0), x0 is a value of this pair
        float s1 = 0.f, s2 = 0.f;
#pragma unroll 8
        for (int q = 0; q < 32; ++q) {
          const uint4 u = *reinterpret_cast<const uint4*>(A0 + ((h * 4 + (q >> 3)) * 512 + (q & 7) * 64) * 8);
          const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float2 f = __bfloat1622float2(p2[e]);
            const float a = f.x - x0, b = f.y - x0;
            s1 += a + b; s2 += a * a + b * b;
          }
        }
        sred[h][j][0] = s1; sred[h][j][1] = s2;
        bar_sync(1, 128);
        s1 += sred[h ^ 1][j][0]; s2 += sred[h ^ 1][j][1];
        const float m1 = s1 * (1.f / P);
        const float mean = x0 + m1;
        const float rstd = rsqrtf(fmaxf(s2 * (1.f / P) - m1 * m1, 0.f) + eps);
#pragma unroll 8
        for (int q = 0; q < 32; ++q) {
          uint4* ptr = reinterpret_cast<uint4*>(const_cast<bf16*>(A0) + ((h * 4 + (q >> 3)) * 512 + (q & 7) * 64) * 8);
          uint4 u = *ptr;
          __nv_bfloat162* p2 = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float2 f = __bfloat1622float2(p2[e]);
            p2[e] = __floats2bfloat162_rn((f.x - mean) * rstd, (f.y - mean) * rstd);
          }
          *ptr = u;
        }
        if (SAVE && h == 0 && stats != nullptr) { const long pr = pairA + (j >> 5) * N + (j & 31); stats[pr * 2] = mean; stats[pr * 2 + 1] = rstd; }
      }
      proxy_fence();
      bar_sync(1, 128);
      // ---- the slice of W': acc[64 pairs x 128 outputs] = A (K = 512) . W'^T
      float acc[64];
      wg_fence();
      for (int kb = 0; kb < KBN; ++kb, ++gb) {
        const int bslot = gb % NBS;
        mb_wait(b_full + bslot, (gb / NBS) & 1);
        const bf16* B = sB + bslot * B_EL;
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
          mma_m64n128k16(wg_desc_ns(A + kb * (BM * 64) + ks * 1024, 1024, 128), wg_desc(B + off_n(128, 0, 0, ks * 2)), acc, (kb | ks) != 0);
        wg_commit();
        wg_wait<0>();
        if (lane == 0) mb_arrive(b_empty + bslot);
      }
      bar_sync(1, 128);
      if (SAVE && blockIdx.y == 0) {                          // the normalised tile is the operand of dW_out: re-lay it in place as [8 g][64 m][8 k chunks] (128 B swizzle), TMA-store it
#pragma unroll 1
        for (int g0 = 0; g0 < G; g0 += 4) {
          uint4 r[16];
#pragma unroll
          for (int u = 0; u < 16; ++u) r[u] = reinterpret_cast<const uint4*>(A + (g0 + (u >> 2)) * 4096)[tid + 128 * (u & 3)];
          bar_sync(1, 128);
#pragma unroll
          for (int u = 0; u < 16; ++u) {
            const int c = tid + 128 * (u & 3), k = c >> 6, m = c & 63;
            reinterpret_cast<uint4*>(A + (g0 + (u >> 2)) * 4096)[m * 8 + (k ^ (m & 7))] = r[u];
          }
          bar_sync(1, 128);
        }
        proxy_fence();
        bar_sync(1, 128);
        if (tid == 0) {
#pragma unroll
          for (int g = 0; g < KBN; ++g) {
            tma_st2(&ymap, A + g * (BM * 64), g * 64, (int)pairA);
            tma_st2(&ymap, A + g * (BM * 64) + 32 * 64, g * 64, (int)(pairA + N));
          }
          tma_commit();
          tma_wait_read<0>();
        }
      }
      if (tid == 0) {
        mb_arrive(a_empty + slot);
      }
      // ---- + bias, rounded to bf16 into the staging tile, then + residual and the rows out
      {
        const int r0 = warp * 16 + (lane >> 2);
#pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
          const int c = nt * 8 + (lane & 3) * 2;
          const float2 b = *reinterpret_cast<const float2*>(bias + z0 + c);
          *reinterpret_cast<__nv_bfloat162*>(sO + r0 * NZ + ((nt ^ (r0 & 7)) << 3) + (c & 7)) = __floats2bfloat162_rn(acc[nt * 4] + b.x, acc[nt * 4 + 1] + b.y);
          *reinterpret_cast<__nv_bfloat162*>(sO + (r0 + 8) * NZ + ((nt ^ ((r0 + 8) & 7)) << 3) + (c & 7)) = __floats2bfloat162_rn(acc[nt * 4 + 2] + b.x, acc[nt * 4 + 3] + b.y);
        }
      }
      bar_sync(1, 128);
#pragma unroll
      for (int q = 0; q < (BM * (NZ / 8)) / 128; ++q) {
        const int idx = tid + q * 128, row = idx >> 4, ch = idx & 15;
        uint4 u = *reinterpret_cast<const uint4*>(sO + row * NZ + ((ch ^ (row & 7)) << 3));
        const size_t g = (size_t)(pairA + (row >> 5) * N + (row & 31)) * CZ + z0 + ch * 8;
        if (residual != nullptr) {
          const uint4 rv = *reinterpret_cast<const uint4*>(residual + g);
          __nv_bfloat162* a2 = reinterpret_cast<__nv_bfloat162*>(&u);
          const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&rv);
#pragma unroll
          for (int e = 0; e < 4; ++e) a2[e] = __hadd2(a2[e], b2[e]);
        }
        *reinterpret_cast<uint4*>(out + g) = u;
      }
      bar_sync(1, 128);
    }

  }
}
}  // namespace

// O [G, N K, N K] bf16 (the grouped outer product), wfold [CZ, 512] bf16 (W_out . gamma), bias [CZ] fp32 (b_out + W_out beta), residual [1, N, N, CZ]
// bf16 (optional)  ->  [z [1, N, N, CZ] bf16, y [N N, 512] bf16 and stats [N N, 2] fp32 when ``save``]
std::vector<at::Tensor> chai_epilogue(at::Tensor O, at::Tensor wfold, at::Tensor bias, c10::optional<at::Tensor> residual, bool save, double eps) {
  TORCH_CHECK(O.is_cuda() && O.is_contiguous() && O.scalar_type() == at::kBFloat16 && O.dim() == 3 && O.size(0) == G && O.size(1) == O.size(2)
              && O.size(1) % (KK * BM) == 0, "O: contiguous bf16 [8, N K, N K] with N a multiple of 64 and K = 8");
  const int64_t N = O.size(1) / KK, CZ = wfold.size(0);
  TORCH_CHECK(wfold.is_cuda() && wfold.is_contiguous() && wfold.scalar_type() == at::kBFloat16 && wfold.dim() == 2 && wfold.size(1) == P && CZ % NZ == 0,
              "wfold: contiguous bf16 [CZ, 512], CZ a multiple of 128");
  TORCH_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == at::kFloat && bias.numel() == CZ, "bias: fp32 [CZ]");
  if (residual.has_value())
    TORCH_CHECK(residual->is_cuda() && residual->is_contiguous() && residual->scalar_type() == at::kBFloat16 && residual->numel() == N * N * CZ,
                "residual: contiguous bf16 [1, N, N, CZ]");
  const at::cuda::OptionalCUDAGuard guard(O.device());
  auto out = at::empty({1, N, N, CZ}, O.options());
  at::Tensor y, st;
  if (save) {
    y = at::empty({N * N, P}, O.options());
    st = at::empty({N * N, 2}, O.options().dtype(at::kFloat));
  } else {
    y = at::empty({0}, O.options());
    st = at::empty({0}, O.options().dtype(at::kFloat));
  }
  alignas(64) CUtensorMap omap{}, wmap{}, ymap{};
  {
    const uint64_t nk = N * KK;
    uint64_t dims[4] = {(uint64_t)nk, (uint64_t)N, KK, G};
    uint64_t strides[3] = {KK * nk * 2, nk * 2, nk * nk * 2};
    uint32_t box[4] = {256, 2, KK, 1};
    encode(&omap, O.data_ptr(), 4, dims, strides, box, false);
  }
  {
    uint64_t dims[2] = {P, (uint64_t)CZ};
    uint64_t strides[1] = {P * 2};
    uint32_t box[2] = {64, NZ};
    encode(&wmap, wfold.data_ptr(), 2, dims, strides, box);
  }
  if (save) {
    uint64_t dims[2] = {P, (uint64_t)(N * N)};
    uint64_t strides[1] = {P * 2};
    uint32_t box[2] = {64, 32};
    encode(&ymap, y.data_ptr(), 2, dims, strides, box, true);
  }
  const int tiles = (int)(N * N / BM);
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int zs = (int)(CZ / NZ);
  const dim3 grid(std::max(1, std::min(tiles, 2 * sms / zs)), zs);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto kernel) {
    const size_t smem = SMEM;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    kernel<<<grid, 160, smem, stream>>>(omap, wmap, ymap, bias.data_ptr<float>(),
                                         residual.has_value() ? reinterpret_cast<const bf16*>(residual->data_ptr()) : nullptr,
                                         reinterpret_cast<bf16*>(out.data_ptr()), save ? st.data_ptr<float>() : nullptr, (int)N, (int)CZ, tiles,
                                         (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };
  if (save) launch(chai_epilogue_kernel<true>);
  else launch(chai_epilogue_kernel<false>);
  return {out, y, st};
}



// ---------------------------------------------------------------------------------------------------------------------------------------
// Chai-1 prologue: LayerNorm of the MSA rows, both projections (mma.sync m16n8k16), the mask, stored group-major: a[g, t, k] = left channel
// 8 g + k of token row t (the layout the per-group batched GEMM reads as [S, N K] matrices).  A block = 4 warps x 16 token rows.
namespace {
constexpr int WLD = 72;     // padded row of the stacked projection weight in shared memory (bank-conflict-free fragment loads)

__device__ __forceinline__ void mma16816(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(128) chai_proj_kernel(const bf16* __restrict__ X, const unsigned char* __restrict__ mask, const float* __restrict__ gamma,
                                                         const float* __restrict__ beta, const bf16* __restrict__ W, bf16* __restrict__ A,
                                                         bf16* __restrict__ B, float* __restrict__ stats, long T, float eps) {
  __shared__ __align__(16) bf16 sW[128 * WLD];
  __shared__ float sg[64], sb[64];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < 128 * 8; i += 128) {
    const int r = i >> 3, c = i & 7;
    *reinterpret_cast<uint4*>(sW + r * WLD + c * 8) = *reinterpret_cast<const uint4*>(W + r * 64 + c * 8);
  }
  if (tid < 64) { sg[tid] = gamma[tid]; sb[tid] = beta[tid]; }
  __syncthreads();
  const long t0 = (long)blockIdx.x * 64 + warp * 16;
  const int r0 = lane >> 2, q = lane & 3;
  const long ra = t0 + r0, rb = ra + 8;
  uint32_t xa[4][2], xb[4][2];                            // [k step][k, k + 8] pairs of bf16 of row ra / rb
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) {
    const int c = ks * 16 + q * 2;
    xa[ks][0] = ra < T ? *reinterpret_cast<const uint32_t*>(X + ra * 64 + c) : 0u;
    xa[ks][1] = ra < T ? *reinterpret_cast<const uint32_t*>(X + ra * 64 + c + 8) : 0u;
    xb[ks][0] = rb < T ? *reinterpret_cast<const uint32_t*>(X + rb * 64 + c) : 0u;
    xb[ks][1] = rb < T ? *reinterpret_cast<const uint32_t*>(X + rb * 64 + c + 8) : 0u;
  }
  float sa_ = 0.f, sb_ = 0.f;
#pragma unroll
  for (int ks = 0; ks < 4; ++ks)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float2 fa = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xa[ks][h])), fb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xb[ks][h]));
      sa_ += fa.x + fa.y; sb_ += fb.x + fb.y;
    }
  sa_ += __shfl_xor_sync(0xffffffffu, sa_, 1); sa_ += __shfl_xor_sync(0xffffffffu, sa_, 2);
  sb_ += __shfl_xor_sync(0xffffffffu, sb_, 1); sb_ += __shfl_xor_sync(0xffffffffu, sb_, 2);
  const float ma = sa_ * (1.f / 64), mb = sb_ * (1.f / 64);
  float va = 0.f, vb = 0.f;
#pragma unroll
  for (int ks = 0; ks < 4; ++ks)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float2 fa = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xa[ks][h])), fb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xb[ks][h]));
      va += (fa.x - ma) * (fa.x - ma) + (fa.y - ma) * (fa.y - ma); vb += (fb.x - mb) * (fb.x - mb) + (fb.y - mb) * (fb.y - mb);
    }
  va += __shfl_xor_sync(0xffffffffu, va, 1); va += __shfl_xor_sync(0xffffffffu, va, 2);
  vb += __shfl_xor_sync(0xffffffffu, vb, 1); vb += __shfl_xor_sync(0xffffffffu, vb, 2);
  const float rsa = rsqrtf(va * (1.f / 64) + eps), rsb = rsqrtf(vb * (1.f / 64) + eps);
  if (stats != nullptr && q == 0) {
    if (ra < T) { stats[ra * 2] = ma; stats[ra * 2 + 1] = rsa; }
    if (rb < T) { stats[rb * 2] = mb; stats[rb * 2 + 1] = rsb; }
  }
  uint32_t fa_[4][4];                                      // A fragments of the normalised rows (bf16, as the module's LayerNorm output)
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) {
    const int c = ks * 16 + q * 2;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float2 fa = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xa[ks][h])), fb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xb[ks][h]));
      const int cc = c + 8 * h;
      const __nv_bfloat162 ya = __floats2bfloat162_rn((fa.x - ma) * rsa * sg[cc] + sb[cc], (fa.y - ma) * rsa * sg[cc + 1] + sb[cc + 1]);
      const __nv_bfloat162 yb = __floats2bfloat162_rn((fb.x - mb) * rsb * sg[cc] + sb[cc], (fb.y - mb) * rsb * sg[cc + 1] + sb[cc + 1]);
      fa_[ks][2 * h] = *reinterpret_cast<const uint32_t*>(&ya);        // a0 / a2: row ra, k / k + 8
      fa_[ks][2 * h + 1] = *reinterpret_cast<const uint32_t*>(&yb);    // a1 / a3: row rb
    }
  }
  float acc[16][4];
#pragma unroll
  for (int nt = 0; nt < 16; ++nt) { acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f; }
#pragma unroll
  for (int ks = 0; ks < 4; ++ks)
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      const bf16* wr = sW + (nt * 8 + r0) * WLD + ks * 16 + q * 2;
      const uint32_t b0 = *reinterpret_cast<const uint32_t*>(wr), b1 = *reinterpret_cast<const uint32_t*>(wr + 8);
      mma16816(acc[nt], fa_[ks], b0, b1);
    }
  const bool ka = ra < T && (mask == nullptr || mask[ra] != 0), kb = rb < T && (mask == nullptr || mask[rb] != 0);
#pragma unroll
  for (int nt = 0; nt < 16; ++nt) {
    bf16* dst = nt < 8 ? A : B;
    const int g = nt & 7;
    if (ra < T)
      *reinterpret_cast<__nv_bfloat162*>(dst + ((long)g * T + ra) * 8 + q * 2) = ka ? __floats2bfloat162_rn(acc[nt][0], acc[nt][1]) : __floats2bfloat162_rn(0.f, 0.f);
    if (rb < T)
      *reinterpret_cast<__nv_bfloat162*>(dst + ((long)g * T + rb) * 8 + q * 2) = kb ? __floats2bfloat162_rn(acc[nt][2], acc[nt][3]) : __floats2bfloat162_rn(0.f, 0.f);
  }
}
}  // namespace

// x [T, 64] bf16 (T = S N rows), mask [T] bool or None, gamma / beta fp32 [64], w [128, 64] bf16 (left | right projection) -> [a, b [8, T, 8] bf16, stats [T, 2] fp32]
std::vector<at::Tensor> chai_proj(at::Tensor x, c10::optional<at::Tensor> mask, at::Tensor gamma, at::Tensor beta, at::Tensor w, double eps, bool save) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == 64, "x: contiguous bf16 [T, 64]");
  TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == at::kBFloat16 && w.size(0) == 128 && w.size(1) == 64, "w: bf16 [128, 64]");
  TORCH_CHECK(gamma.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat && gamma.numel() == 64 && beta.numel() == 64, "gamma / beta: fp32 [64]");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  const long T = x.size(0);
  auto a = at::empty({G, T, KK}, x.options()), b = at::empty({G, T, KK}, x.options());
  auto st = at::empty({save ? T : 0, 2}, x.options().dtype(at::kFloat));
  const unsigned char* mp = nullptr;
  if (mask.has_value()) {
    TORCH_CHECK(mask->is_cuda() && mask->is_contiguous() && mask->scalar_type() == at::kBool && mask->numel() == T, "mask: bool [T]");
    mp = reinterpret_cast<const unsigned char*>(mask->data_ptr());
  }
  chai_proj_kernel<<<(unsigned)((T + 63) / 64), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(x.data_ptr()), mp, gamma.data_ptr<float>(), beta.data_ptr<float>(), reinterpret_cast<const bf16*>(w.data_ptr()),
      reinterpret_cast<bf16*>(a.data_ptr()), reinterpret_cast<bf16*>(b.data_ptr()), save ? st.data_ptr<float>() : nullptr, T, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {a, b, st};
}


// ---------------------------------------------------------------------------------------------------------------------------------------
// Chai-1 backward of the product LayerNorm: dy = dz W' (cuBLAS, [N N, 512]) and the saved y (normalised products) and rstd give, per pair,
//   dO[p] = rstd (dy[p] - mean_p dy - y[p] mean_p (dy y)),
// scattered into the grouped layout dO[g, i K + k, j K + l] that the two gradient GEMMs read.  A tile = one row i x 32 columns j: dy and y arrive
// by TMA as 8 [32 pairs x 64] blocks (128 B swizzle), dO leaves by TMA from a [k][j][l] tile (the load map of the forward).
namespace {
constexpr int BM2 = 32;
constexpr size_t SMEM_BWD = size_t(3 * BM2 * P) * 2 + 1024;

__global__ void __launch_bounds__(128, 2) chai_bwd_kernel(const __grid_constant__ CUtensorMap dymap, const __grid_constant__ CUtensorMap ymap,
                                                           const __grid_constant__ CUtensorMap omap, const float* __restrict__ stats, int N, int tiles) {
  extern __shared__ __align__(1024) unsigned char raw_[];
  bf16* sDY = reinterpret_cast<bf16*>(raw_);                  // [8 g][32 m][64]  (128 B rows, chunk XOR m % 8)
  bf16* sY = sDY + BM2 * P;
  bf16* sD = sY + BM2 * P;                                    // [8 g][8 k][32 j][8 l]
  __shared__ __align__(8) uint64_t bar;
  __shared__ float red[4][32][2];
  const int tid = threadIdx.x, m = tid & 31, qd = tid >> 5;
  const int tpr = N / BM2;
  if (tid == 0) { mb_init(&bar, 1); fence_init(); }
  __syncthreads();
  int phase = 0;
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int i = t / tpr, j0 = (t % tpr) * BM2;
    const long pair0 = (long)i * N + j0;
    if (tid == 0) {
      mb_expect(&bar, 2 * BM2 * P * 2);
#pragma unroll
      for (int g = 0; g < G; ++g) {
        tma_ld2(&dymap, sDY + g * (BM2 * 64), &bar, g * 64, (int)pair0);
        tma_ld2(&ymap, sY + g * (BM2 * 64), &bar, g * 64, (int)pair0);
      }
    }
    mb_wait(&bar, phase);
    phase ^= 1;
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int gg = 0; gg < 2; ++gg) {
      const int g = qd * 2 + gg;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int off = ((g * 32 + m) * 8 + (k ^ (m & 7))) * 8;
        const uint4 ud = *reinterpret_cast<const uint4*>(sDY + off), uy = *reinterpret_cast<const uint4*>(sY + off);
        const __nv_bfloat162* pd = reinterpret_cast<const __nv_bfloat162*>(&ud);
        const __nv_bfloat162* py = reinterpret_cast<const __nv_bfloat162*>(&uy);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 d = __bfloat1622float2(pd[e]), y = __bfloat1622float2(py[e]);
          s1 += d.x + d.y; s2 += d.x * y.x + d.y * y.y;
        }
      }
    }
    red[qd][m][0] = s1; red[qd][m][1] = s2;
    __syncthreads();
    s1 = red[0][m][0] + red[1][m][0] + red[2][m][0] + red[3][m][0];
    s2 = red[0][m][1] + red[1][m][1] + red[2][m][1] + red[3][m][1];
    const float c1 = s1 * (1.f / P), c2 = s2 * (1.f / P), rstd = stats[(pair0 + m) * 2 + 1];
#pragma unroll
    for (int gg = 0; gg < 2; ++gg) {
      const int g = qd * 2 + gg;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int off = ((g * 32 + m) * 8 + (k ^ (m & 7))) * 8;
        const uint4 ud = *reinterpret_cast<const uint4*>(sDY + off), uy = *reinterpret_cast<const uint4*>(sY + off);
        const __nv_bfloat162* pd = reinterpret_cast<const __nv_bfloat162*>(&ud);
        const __nv_bfloat162* py = reinterpret_cast<const __nv_bfloat162*>(&uy);
        uint4 uo;
        __nv_bfloat162* po = reinterpret_cast<__nv_bfloat162*>(&uo);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 d = __bfloat1622float2(pd[e]), y = __bfloat1622float2(py[e]);
          po[e] = __floats2bfloat162_rn(rstd * (d.x - c1 - y.x * c2), rstd * (d.y - c1 - y.y * c2));
        }
        *reinterpret_cast<uint4*>(sD + ((g * 8 + k) * 32 + m) * 8) = uo;
      }
    }
    proxy_fence();
    __syncthreads();
    if (tid == 0) {
#pragma unroll
      for (int g = 0; g < G; ++g) tma_st4(&omap, sD + g * (8 * 32 * 8), j0 * 8, i, 0, g);
      tma_commit();
      tma_wait_read<0>();
    }
    __syncthreads();
  }
  if (tid == 0) tma_wait_all<0>();
}
}  // namespace

// dy [N N, 512] bf16 (= dz W'), y [N N, 512] bf16 and stats [N N, 2] fp32 (the forward's) -> dO [8, N K, N K] bf16
at::Tensor chai_bwd(at::Tensor dy, at::Tensor y, at::Tensor stats, int64_t N) {
  TORCH_CHECK(dy.is_cuda() && dy.is_contiguous() && dy.scalar_type() == at::kBFloat16 && dy.sizes() == y.sizes() && y.is_contiguous() && dy.size(1) == P
              && dy.size(0) == N * N && N % BM2 == 0, "dy / y: contiguous bf16 [N N, 512]");
  const at::cuda::OptionalCUDAGuard guard(dy.device());
  const int64_t nk = N * KK;
  auto dO = at::empty({G, nk, nk}, dy.options());
  alignas(64) CUtensorMap dymap{}, ymap{}, omap{};
  {
    uint64_t dims[2] = {P, (uint64_t)(N * N)};
    uint64_t strides[1] = {P * 2};
    uint32_t box[2] = {64, BM2};
    encode(&dymap, dy.data_ptr(), 2, dims, strides, box, true);
    encode(&ymap, y.data_ptr(), 2, dims, strides, box, true);
  }
  {
    uint64_t dims[4] = {(uint64_t)nk, (uint64_t)N, KK, G};
    uint64_t strides[3] = {KK * (uint64_t)nk * 2, (uint64_t)nk * 2, (uint64_t)nk * nk * 2};
    uint32_t box[4] = {256, 1, KK, 1};
    encode(&omap, dO.data_ptr(), 4, dims, strides, box, false);
  }
  const int tiles = (int)(N * N / BM2);
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  cudaFuncSetAttribute(chai_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BWD);
  chai_bwd_kernel<<<std::min(tiles, 2 * sms), 128, SMEM_BWD, at::cuda::getCurrentCUDAStream()>>>(dymap, ymap, omap, stats.data_ptr<float>(), (int)N, tiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dO;
}


// ---------------------------------------------------------------------------------------------------------------------------------------
// Chai-1 backward of the prologue: dA, dB [8, T, 8] (the gradients of the group-major projections) -> dcat [T, 128] (masked, the operand of dW), the
// LayerNorm output ln16 [T, 64] (bf16), dx [T, 64] and per-block sums of dgamma / dbeta; dln = dcat Wlr (mma.sync), then the LayerNorm backward on
// the accumulator rows (statistics of the forward).  A block = 4 warps x 16 rows.
namespace {
__global__ void __launch_bounds__(128) chai_projbwd_kernel(const bf16* __restrict__ X, const unsigned char* __restrict__ mask, const bf16* __restrict__ DA,
                                                            const bf16* __restrict__ DB, const float* __restrict__ gamma, const float* __restrict__ beta,
                                                            const float* __restrict__ stats, const bf16* __restrict__ Wlr, bf16* __restrict__ DCAT,
                                                            bf16* __restrict__ LN16, bf16* __restrict__ DX, float* __restrict__ part, long T) {
  __shared__ __align__(16) bf16 sWT[64 * 136];             // [n][k] of the stacked projection weight (k = 128 channels of dcat)
  __shared__ float sg[64], sb[64], sred[4][2][64];
  extern __shared__ __align__(16) bf16 dynS[];                                 // per warp: dcat, ln16, dx rows, flushed as 16-byte rows
  bf16* const sStage[4] = {dynS, dynS + 8960 / 2, dynS + 2 * (8960 / 2), dynS + 3 * (8960 / 2)};
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  for (int i = tid; i < 128 * 64; i += 128) {              // Wlr [128 k][64 n] -> sWT[n][k]
    const int k = i >> 6, n = i & 63;
    sWT[n * 136 + k] = Wlr[i];
  }
  if (tid < 64) { sg[tid] = gamma[tid]; sb[tid] = beta[tid]; }
  __syncthreads();
  const int r0 = lane >> 2, q = lane & 3;
  const long ta = (long)blockIdx.x * 64 + warp * 16 + r0, tb = ta + 8;
  const bool va = ta < T, vb = tb < T;
  const bool ka = va && (mask == nullptr || mask[ta] != 0), kb = vb && (mask == nullptr || mask[tb] != 0);
  float acc[8][4];
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) { acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f; }
#pragma unroll
  for (int ks = 0; ks < 8; ++ks) {                         // k step: channels 16 ks .. 16 ks + 15 of dcat = [left 0..63 | right 0..63]
    const bf16* src = ks < 4 ? DA : DB;
    const int g0 = (ks & 3) * 2;
    uint32_t a[4];
    a[0] = ka ? *reinterpret_cast<const uint32_t*>(src + ((long)g0 * T + ta) * 8 + q * 2) : 0u;
    a[1] = kb ? *reinterpret_cast<const uint32_t*>(src + ((long)g0 * T + tb) * 8 + q * 2) : 0u;
    a[2] = ka ? *reinterpret_cast<const uint32_t*>(src + ((long)(g0 + 1) * T + ta) * 8 + q * 2) : 0u;
    a[3] = kb ? *reinterpret_cast<const uint32_t*>(src + ((long)(g0 + 1) * T + tb) * 8 + q * 2) : 0u;
    const int c = ks * 16 + q * 2;                         // channel of dcat held (and written below)
    bf16* sDc = sStage[warp]; bf16* sLn = sDc + 16 * 136; bf16* sDx = sLn + 16 * 72;
    *reinterpret_cast<uint32_t*>(sDc + r0 * 136 + c) = a[0]; *reinterpret_cast<uint32_t*>(sDc + r0 * 136 + c + 8) = a[2];
    *reinterpret_cast<uint32_t*>(sDc + (r0 + 8) * 136 + c) = a[1]; *reinterpret_cast<uint32_t*>(sDc + (r0 + 8) * 136 + c + 8) = a[3];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const bf16* wr = sWT + (nt * 8 + r0) * 136 + ks * 16 + q * 2;
      mma16816(acc[nt], a, *reinterpret_cast<const uint32_t*>(wr), *reinterpret_cast<const uint32_t*>(wr + 8));
    }
  }
  const float ma = va ? stats[ta * 2] : 0.f, ra = va ? stats[ta * 2 + 1] : 0.f, mb = vb ? stats[tb * 2] : 0.f, rb = vb ? stats[tb * 2 + 1] : 0.f;
  float xh[8][4];
  float s1a = 0.f, s2a = 0.f, s1b = 0.f, s2b = 0.f;
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) {
    const int c = nt * 8 + q * 2;
    float2 xa = make_float2(0.f, 0.f), xb = xa;
    if (va) xa = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(X + ta * 64 + c));
    if (vb) xb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(X + tb * 64 + c));
    xh[nt][0] = (xa.x - ma) * ra; xh[nt][1] = (xa.y - ma) * ra; xh[nt][2] = (xb.x - mb) * rb; xh[nt][3] = (xb.y - mb) * rb;
    *reinterpret_cast<__nv_bfloat162*>(sStage[warp] + 16 * 136 + r0 * 72 + c) = __floats2bfloat162_rn(xh[nt][0] * sg[c] + sb[c], xh[nt][1] * sg[c + 1] + sb[c + 1]);
    *reinterpret_cast<__nv_bfloat162*>(sStage[warp] + 16 * 136 + (r0 + 8) * 72 + c) = __floats2bfloat162_rn(xh[nt][2] * sg[c] + sb[c], xh[nt][3] * sg[c + 1] + sb[c + 1]);
    const float g0 = sg[c], g1 = sg[c + 1];
    s1a += acc[nt][0] * g0 + acc[nt][1] * g1; s2a += acc[nt][0] * g0 * xh[nt][0] + acc[nt][1] * g1 * xh[nt][1];
    s1b += acc[nt][2] * g0 + acc[nt][3] * g1; s2b += acc[nt][2] * g0 * xh[nt][2] + acc[nt][3] * g1 * xh[nt][3];
  }
  s1a += __shfl_xor_sync(0xffffffffu, s1a, 1); s1a += __shfl_xor_sync(0xffffffffu, s1a, 2);
  s2a += __shfl_xor_sync(0xffffffffu, s2a, 1); s2a += __shfl_xor_sync(0xffffffffu, s2a, 2);
  s1b += __shfl_xor_sync(0xffffffffu, s1b, 1); s1b += __shfl_xor_sync(0xffffffffu, s1b, 2);
  s2b += __shfl_xor_sync(0xffffffffu, s2b, 1); s2b += __shfl_xor_sync(0xffffffffu, s2b, 2);
  s1a *= (1.f / 64); s2a *= (1.f / 64); s1b *= (1.f / 64); s2b *= (1.f / 64);
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) {
    const int c = nt * 8 + q * 2;
    const float g0 = sg[c], g1 = sg[c + 1];
    *reinterpret_cast<__nv_bfloat162*>(sStage[warp] + 16 * 136 + 16 * 72 + r0 * 72 + c) = __floats2bfloat162_rn(ra * (acc[nt][0] * g0 - s1a - xh[nt][0] * s2a), ra * (acc[nt][1] * g1 - s1a - xh[nt][1] * s2a));
    *reinterpret_cast<__nv_bfloat162*>(sStage[warp] + 16 * 136 + 16 * 72 + (r0 + 8) * 72 + c) = __floats2bfloat162_rn(rb * (acc[nt][2] * g0 - s1b - xh[nt][2] * s2b), rb * (acc[nt][3] * g1 - s1b - xh[nt][3] * s2b));
    // dgamma = sum dln xhat, dbeta = sum dln over the warp's rows (rows r0 and r0 + 8, then the 8 row groups)
    float pg0 = acc[nt][0] * xh[nt][0] + acc[nt][2] * xh[nt][2], pg1 = acc[nt][1] * xh[nt][1] + acc[nt][3] * xh[nt][3];
    float pb0 = acc[nt][0] + acc[nt][2], pb1 = acc[nt][1] + acc[nt][3];
#pragma unroll
    for (int m = 4; m <= 16; m <<= 1) {
      pg0 += __shfl_xor_sync(0xffffffffu, pg0, m); pg1 += __shfl_xor_sync(0xffffffffu, pg1, m);
      pb0 += __shfl_xor_sync(0xffffffffu, pb0, m); pb1 += __shfl_xor_sync(0xffffffffu, pb1, m);
    }
    if (r0 == 0) { sred[warp][0][c] = pg0; sred[warp][0][c + 1] = pg1; sred[warp][1][c] = pb0; sred[warp][1][c + 1] = pb1; }
  }
  __syncwarp();
  {
    const long t0 = (long)blockIdx.x * 64 + warp * 16;
    const bf16* sDc = sStage[warp]; const bf16* sLn = sDc + 16 * 136; const bf16* sDx = sLn + 16 * 72;
#pragma unroll
    for (int u = 0; u < 8; ++u) {                          // dcat: 16 rows x 16 chunks of 16 B
      const int idx = lane + 32 * u, row = idx >> 4, ch = idx & 15;
      if (t0 + row < T) *reinterpret_cast<uint4*>(DCAT + (t0 + row) * 128 + ch * 8) = *reinterpret_cast<const uint4*>(sDc + row * 136 + ch * 8);
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {                          // ln16, dx: 16 rows x 8 chunks
      const int idx = lane + 32 * u, row = idx >> 3, ch = idx & 7;
      if (t0 + row < T) {
        *reinterpret_cast<uint4*>(LN16 + (t0 + row) * 64 + ch * 8) = *reinterpret_cast<const uint4*>(sLn + row * 72 + ch * 8);
        *reinterpret_cast<uint4*>(DX + (t0 + row) * 64 + ch * 8) = *reinterpret_cast<const uint4*>(sDx + row * 72 + ch * 8);
      }
    }
  }
  __syncthreads();
  if (tid < 128) {
    const int w = tid >> 6, c = tid & 63;
    part[(long)blockIdx.x * 128 + tid] = sred[0][w][c] + sred[1][w][c] + sred[2][w][c] + sred[3][w][c];
  }
}
}  // namespace

// x [T, 64] bf16, mask [T] bool | None, dA / dB [8, T, 8] bf16, gamma / beta fp32 [64], stats [T, 2] fp32, wlr [128, 64] bf16
//   -> [dcat [T, 128], ln16 [T, 64], dx [T, 64] bf16, partials [T / 64, 128] fp32 (dgamma | dbeta per block)]
std::vector<at::Tensor> chai_projbwd(at::Tensor x, c10::optional<at::Tensor> mask, at::Tensor da, at::Tensor db, at::Tensor gamma, at::Tensor beta,
                                     at::Tensor stats, at::Tensor wlr) {
  const long T = x.size(0);
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.size(1) == 64 && da.is_contiguous() && db.is_contiguous()
              && da.sizes() == at::IntArrayRef({8, T, 8}) && db.sizes() == da.sizes() && wlr.is_contiguous() && wlr.size(0) == 128 && wlr.size(1) == 64
              && stats.is_contiguous() && stats.numel() == T * 2, "chai_projbwd: shapes");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto dcat = at::empty({T, 128}, x.options()), ln16 = at::empty({T, 64}, x.options()), dx = at::empty({T, 64}, x.options());
  const long blocks = (T + 63) / 64;
  auto part = at::empty({blocks, 128}, x.options().dtype(at::kFloat));
  const unsigned char* mp = nullptr;
  if (mask.has_value()) mp = reinterpret_cast<const unsigned char*>(mask->data_ptr());
  cudaFuncSetAttribute(chai_projbwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * 8960);
  chai_projbwd_kernel<<<(unsigned)blocks, 128, 4 * 8960, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(x.data_ptr()), mp, reinterpret_cast<const bf16*>(da.data_ptr()), reinterpret_cast<const bf16*>(db.data_ptr()),
      gamma.data_ptr<float>(), beta.data_ptr<float>(), stats.data_ptr<float>(), reinterpret_cast<const bf16*>(wlr.data_ptr()),
      reinterpret_cast<bf16*>(dcat.data_ptr()), reinterpret_cast<bf16*>(ln16.data_ptr()), reinterpret_cast<bf16*>(dx.data_ptr()), part.data_ptr<float>(), T);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dcat, ln16, dx, part};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("chai_epilogue", &chai_epilogue);
  m.def("chai_proj", &chai_proj);
  m.def("chai_bwd", &chai_bwd);
  m.def("chai_projbwd", &chai_projbwd);
}
