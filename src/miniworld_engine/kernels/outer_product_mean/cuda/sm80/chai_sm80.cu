// Chai-1 OuterProductMean (8 groups of 8-wide halves), A100 (sm_80): the passes of ``chai.cu`` (the H100 build) on mma.sync / cp.async.
//
//   O[g, i K + k, j K + l]  (cuBLAS: one batched GEMM over the MSA rows)        K = 8, G = 8, P = G K K = 512 products per pair
//   xhat_p    = (x_p - mean) rstd         LayerNorm over the 512 products of a pair (eps 0.1),  p = (g, k, l)
//   z[i, j]   = residual[i, j] + bf16(sum_p xhat_p W'[o, p] + b'[o])           W' = W_out . gamma (columns), b' = b_out + W_out beta
//
// ``chai_epilogue``: a CTA task = 64 consecutive pairs of a row i x 256 outputs (8 warps: 2 x 32 pairs, 4 x 64 outputs; two CTAs per SM).  The pair tile arrives by
// cp.async straight from the grouped O in K blocks of four (g, k) runs (64 pairs x 8 l: one contiguous 1 KiB run each) together with the W' block; the
// GEMM runs on the raw products, the LayerNorm statistics are summed from the same blocks and the normalisation is the epilogue's affine
// (z = rstd (x W'^T - mean colsum W') + b'; with save y = bf16(xhat) from the L2-resident runs).  ``chai_bwd``: the product LayerNorm's backward per 32-pair tile (dy, y staged in shared memory), dO written
// in the grouped layout (for each (g, k) the 32 pairs are one contiguous 512 B run).  ``chai_proj`` / ``chai_projbwd`` (LayerNorm + projections and their
// backward, mma.sync) are ``chai.cu``'s, which use no sm_90 feature.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "sm80_common.cuh"

namespace {
typedef __nv_bfloat16 bf16;
using namespace opm80;

constexpr int G = 8, KK = 8, P = 512;
constexpr int BM = 64;                     // pairs per epilogue tile
#ifdef MWE_SM120   // the RTX 5090 (99 KiB): a 2-stage ring, two epilogue CTAs per SM (4 stages fit one)
constexpr int KB = 32, NKB = P / KB, NS = 2;
#else
constexpr int KB = 32, NKB = P / KB, NS = 4;   // K blocks of 32 products (4 chunks: four (g, k) runs), ring stages
#endif

DEVI uint32_t r64(int row, int chunk) { return row * 64u + (((chunk ^ (row >> 1)) & 3u) << 4); }      // 64-byte rows (4 chunks)

template <int ZT>   // outputs per CTA task: 256, the pair tile read from O once (128-output tasks at L128 / 256 measured slower)
struct EpiCfg {
  static constexpr int NTHR = 256, WM = BM / 2, MT = WM / 16, WN = ZT / 4, NT = WN / 8;   // 2 x 4 warps: 32 pairs x 64 outputs each
  static constexpr int A_ST = BM * KB * 2, B_ST = ZT * KB * 2, STAGE = A_ST + B_ST;   // 4 + 16 KiB
  static constexpr int OUT = BM * ZT * 2;                                             // the staged output tile (reuses the ring)
  static constexpr int SMEM = NS * STAGE + BM * 2 * 4;                                // + the rows' (mean, rstd)
  static_assert(OUT <= NS * STAGE, "the output tile reuses the ring");
  static_assert(2 * SMEM <= 166912, "two CTAs per SM");
};

// stage s of task (i, j0): the A block = products 32 kb .. 32 kb + 31 of the 64 pairs (the (g, k) runs 4 kb .. 4 kb + 3 of O: 64 pairs x 16 B each), the B block = W'[:, 32 kb ..]
template <int ZT>
DEVI void epi_load(uint32_t st, const bf16* O, const bf16* wf, long nk, int i, int j0, int kb, int tid) {
  {
    const int j = tid & 63, c = tid >> 6, gk = kb * 4 + c, g = gk >> 3, k = gk & 7;
    cp_async16(st + r64(j, c), O + ((long)g * nk + (long)i * KK + k) * nk + (long)(j0 + j) * KK);
  }
#pragma unroll
  for (int it = 0; it < ZT * 4 / 256; ++it) {
    const int v = it * 256 + tid, o = v >> 2, c = v & 3;
    cp_async16(st + EpiCfg<ZT>::A_ST + r64(o, c), wf + (long)o * P + kb * KB + c * 8);
  }
}

// z = rstd (x W'^T - mean colsum(W')) + b': the GEMM runs on the raw products (streamed with W' in K blocks: no pair tile has to sit in shared memory before
// the product), the LayerNorm statistics are summed from the same blocks, and the normalisation is the epilogue's affine.  With SAVE, y = bf16(xhat) and
// the statistics are written for the backward (the tile's O runs re-read, from L2).
template <int ZT, bool SAVE>
__global__ void __launch_bounds__(256, 2) chai_epilogue_kernel(const bf16* __restrict__ O, const bf16* __restrict__ wf, const float* __restrict__ bias,
                                                               const float* __restrict__ colsum, const bf16* __restrict__ residual, bf16* __restrict__ out,
                                                               bf16* __restrict__ y, float* __restrict__ stats, int N, int CZ, int tasks, float eps) {
  using C = EpiCfg<ZT>;
  extern __shared__ __align__(128) unsigned char smem[];
  const uint32_t s0 = smem_u32(smem);
  float* sst = reinterpret_cast<float*>(smem + NS * C::STAGE);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gq = lane >> 2, q = lane & 3;
  const int wm = warp & 1, wn = warp >> 1;
  const int sr = warp * 8 + (lane & 7), sc = lane >> 3;          // statistics: row sr, chunk sc of each A block (the 8 lanes of a phase on 8 rows)
  const long nk = (long)N * KK;
  const int halves = CZ / ZT;
  for (int t = blockIdx.x; t < tasks; t += gridDim.x) {
    const int tile = t / halves, n0 = (t % halves) * ZT;
    const int tpr = N / BM, i = tile / tpr, j0 = (tile % tpr) * BM;
    const long pair0 = (long)i * N + j0;
    const bf16* wz = wf + (long)n0 * P;
    __syncthreads();                                     // the previous task's output staging (in the ring) is stored
#pragma unroll
    for (int s = 0; s < NS - 1; ++s) { epi_load<ZT>(s0 + s * C::STAGE, O, wz, nk, i, j0, s, tid); cp_async_commit(); }
    float acc[C::MT][C::NT][4];
#pragma unroll
    for (int mt = 0; mt < C::MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < C::NT; ++nt) acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f;
    float sum = 0.f, sq = 0.f;
#pragma unroll 1
    for (int kb = 0; kb < NKB; ++kb) {
      cp_async_wait<NS - 2>();
      __syncthreads();                                   // block kb landed; slot (kb - 1) % NS is free
      if (kb + NS - 1 < NKB) epi_load<ZT>(s0 + ((kb + NS - 1) % NS) * C::STAGE, O, wz, nk, i, j0, kb + NS - 1, tid);
      cp_async_commit();
      const uint32_t sa = s0 + (kb % NS) * C::STAGE, sb = sa + C::A_ST;
      {                                                  // the statistics' sums (fp32, of the bf16 products)
        const uint4 r = lds128(sa + r64(sr, sc));
        const uint32_t w[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) { const float a = bf16lo(w[e]), b = bf16hi(w[e]); sum += a + b; sq = fmaf(a, a, fmaf(b, b, sq)); }
      }
#pragma unroll
      for (int ks = 0; ks < KB / 16; ++ks) {
        uint32_t af[C::MT][4];
#pragma unroll
        for (int mt = 0; mt < C::MT; ++mt) ldsm_x4(af[mt], sa + r64(wm * C::WM + mt * 16 + (lane & 15), 2 * ks + (lane >> 4)));
#pragma unroll
        for (int np = 0; np < C::NT / 2; ++np) {
          uint32_t bf[4];
          ldsm_x4(bf, sb + r64(wn * C::WN + np * 16 + (lane & 7) + ((lane >> 4) << 3), 2 * ks + ((lane >> 3) & 1)));
#pragma unroll
          for (int mt = 0; mt < C::MT; ++mt) {
            mma16816(acc[mt][2 * np], af[mt], bf[0], bf[1]);
            mma16816(acc[mt][2 * np + 1], af[mt], bf[2], bf[3]);
          }
        }
      }
    }
    sum += __shfl_xor_sync(0xffffffffu, sum, 8); sum += __shfl_xor_sync(0xffffffffu, sum, 16);
    sq += __shfl_xor_sync(0xffffffffu, sq, 8); sq += __shfl_xor_sync(0xffffffffu, sq, 16);
    if (sc == 0) {
      const float mean = sum * (1.f / P), var = fmaxf(sq * (1.f / P) - mean * mean, 0.f);
      sst[sr * 2] = mean;
      sst[sr * 2 + 1] = rsqrtf(var + eps);
    }
    cp_async_wait<0>();
    __syncthreads();                                     // the statistics are visible; every warp is past its reads of the ring (it becomes the output tile)
    if (SAVE && n0 == 0) {                               // y = bf16(xhat) [pair][p] and the statistics, from the tile's O runs (L2)
#pragma unroll 4
      for (int it = 0; it < BM * 64 / 256; ++it) {
        const int v = it * 256 + tid, j = v & 63, gk = v >> 6, g = gk >> 3, k = gk & 7;
        const uint4 r = ldg128(O + ((long)g * nk + (long)i * KK + k) * nk + (long)(j0 + j) * KK);
        const float mean = sst[j * 2], rs = sst[j * 2 + 1];
        const uint32_t w[4] = {r.x, r.y, r.z, r.w};
        uint32_t o4[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) o4[e] = pack_bf16((bf16lo(w[e]) - mean) * rs, (bf16hi(w[e]) - mean) * rs);
        stg128(y + (pair0 + j) * P + gk * 8, make_uint4(o4[0], o4[1], o4[2], o4[3]));
      }
      if (tid < BM) { stats[(pair0 + tid) * 2] = sst[tid * 2]; stats[(pair0 + tid) * 2 + 1] = sst[tid * 2 + 1]; }
    }
    // ---- rstd (acc - mean colsum) + b', bf16, into the [pairs][ZT] output tile (512-byte rows, chunk XOR row), then 16-byte rows (+ residual)
    constexpr int OCH = ZT / 8;
#pragma unroll
    for (int nt = 0; nt < C::NT; ++nt) {
      const int o = wn * C::WN + nt * 8 + 2 * q;
      const float b0 = bias[n0 + o], b1 = bias[n0 + o + 1], c0 = colsum[n0 + o], c1 = colsum[n0 + o + 1];
#pragma unroll
      for (int mt = 0; mt < C::MT; ++mt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int row = wm * C::WM + mt * 16 + gq + 8 * hh;
          const float mean = sst[row * 2], rs = sst[row * 2 + 1];
          sts32(s0 + row * (ZT * 2) + (((o >> 3) ^ (row & 7)) << 4) + (o & 7) * 2,
                pack_bf16(fmaf(rs, acc[mt][nt][2 * hh] - mean * c0, b0), fmaf(rs, acc[mt][nt][2 * hh + 1] - mean * c1, b1)));
        }
    }
    __syncthreads();
#pragma unroll 4
    for (int it = 0; it < BM * OCH / 256; ++it) {
      const int v = it * 256 + tid, row = v / OCH, ch = v % OCH;
      uint4 u = lds128(s0 + row * (ZT * 2) + ((ch ^ (row & 7)) << 4));
      const long g = (pair0 + row) * CZ + n0 + ch * 8;
      if (residual != nullptr) {
        const uint4 rv = *reinterpret_cast<const uint4*>(residual + g);
        __nv_bfloat162* a2 = reinterpret_cast<__nv_bfloat162*>(&u);
        const __nv_bfloat162* r2 = reinterpret_cast<const __nv_bfloat162*>(&rv);
#pragma unroll
        for (int e = 0; e < 4; ++e) a2[e] = __hadd2(a2[e], r2[e]);
      }
      stg128(out + g, u);
    }
  }
}

// ---- the product LayerNorm's backward: a CTA = 32 pairs (row i, j0 ..), dy and y staged (32 KiB each, chunk XOR row), thread (m, g) = pair m, group g
DEVI uint32_t b_off(int row, int chunk) { return row * 1024 + ((chunk ^ (row & 7)) << 4); }

__global__ void __launch_bounds__(256) chai_bwd_kernel(const bf16* __restrict__ dy, const bf16* __restrict__ y, const float* __restrict__ stats,
                                                       bf16* __restrict__ dO, int N, int tiles) {
  extern __shared__ __align__(128) unsigned char smem[];
  __shared__ float red[8][32][2];
  const uint32_t sD = smem_u32(smem), sY = sD + 32 * 1024;
  const int tid = threadIdx.x, m = tid & 31, g = tid >> 5;
  const long nk = (long)N * KK;
  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int tpr = N / 32, i = t / tpr, j0 = (t % tpr) * 32;
    const long pair0 = (long)i * N + j0;
#pragma unroll
    for (int it = 0; it < 32 * 64 / 256; ++it) {                 // 32 rows x 64 chunks, contiguous in global memory
      const int v = it * 256 + tid, row = v >> 6, c = v & 63;
      cp_async16(sD + b_off(row, c), dy + (pair0 + row) * P + c * 8);
      cp_async16(sY + b_off(row, c), y + (pair0 + row) * P + c * 8);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const uint4 ud = lds128(sD + b_off(m, g * 8 + k)), uy = lds128(sY + b_off(m, g * 8 + k));
      const uint32_t d4[4] = {ud.x, ud.y, ud.z, ud.w}, y4[4] = {uy.x, uy.y, uy.z, uy.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        s1 += bf16lo(d4[e]) + bf16hi(d4[e]);
        s2 = fmaf(bf16lo(d4[e]), bf16lo(y4[e]), fmaf(bf16hi(d4[e]), bf16hi(y4[e]), s2));
      }
    }
    red[g][m][0] = s1; red[g][m][1] = s2;
    __syncthreads();
    s1 = s2 = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) { s1 += red[w][m][0]; s2 += red[w][m][1]; }
    const float c1 = s1 * (1.f / P), c2 = s2 * (1.f / P), rstd = stats[(pair0 + m) * 2 + 1];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const uint4 ud = lds128(sD + b_off(m, g * 8 + k)), uy = lds128(sY + b_off(m, g * 8 + k));
      const uint32_t d4[4] = {ud.x, ud.y, ud.z, ud.w}, y4[4] = {uy.x, uy.y, uy.z, uy.w};
      uint32_t o4[4];
#pragma unroll
      for (int e = 0; e < 4; ++e)
        o4[e] = pack_bf16(rstd * (bf16lo(d4[e]) - c1 - bf16lo(y4[e]) * c2), rstd * (bf16hi(d4[e]) - c1 - bf16hi(y4[e]) * c2));
      stg128(dO + ((long)g * nk + (long)i * KK + k) * nk + (long)(j0 + m) * KK, make_uint4(o4[0], o4[1], o4[2], o4[3]));
    }
    __syncthreads();                                             // the staged tiles and red are rewritten by the next tile
  }
}
}  // namespace

// O [G, N K, N K] bf16, wfold [CZ, 512] bf16, bias [CZ] fp32, residual [1, N, N, CZ] bf16 (optional) -> [z, y [N N, 512], stats [N N, 2] (with save)]
#ifdef MWE_SM120   // the RTX 5090: the column sums of W' come from the caller (cached with the folded weights: two kernels less a call)
std::vector<at::Tensor> chai_epilogue(at::Tensor O, at::Tensor wfold, at::Tensor bias, c10::optional<at::Tensor> residual, bool save, double eps,
                                      at::Tensor colsum_in) {
#else
std::vector<at::Tensor> chai_epilogue(at::Tensor O, at::Tensor wfold, at::Tensor bias, c10::optional<at::Tensor> residual, bool save, double eps) {
#endif
  TORCH_CHECK(O.is_cuda() && O.is_contiguous() && O.scalar_type() == at::kBFloat16 && O.dim() == 3 && O.size(0) == G && O.size(1) == O.size(2)
              && O.size(1) % (KK * BM) == 0, "O: contiguous bf16 [8, N K, N K] with N a multiple of 64 and K = 8");
  const int64_t N = O.size(1) / KK, CZ = wfold.size(0);
  TORCH_CHECK(wfold.is_cuda() && wfold.is_contiguous() && wfold.scalar_type() == at::kBFloat16 && wfold.dim() == 2 && wfold.size(1) == P && CZ % 256 == 0,
              "wfold: contiguous bf16 [CZ, 512], CZ a multiple of 256");
  TORCH_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == at::kFloat && bias.numel() == CZ, "bias: fp32 [CZ]");
  if (residual.has_value())
    TORCH_CHECK(residual->is_cuda() && residual->is_contiguous() && residual->scalar_type() == at::kBFloat16 && residual->numel() == N * N * CZ,
                "residual: contiguous bf16 [1, N, N, CZ]");
  const at::cuda::OptionalCUDAGuard guard(O.device());
  auto out = at::empty({1, N, N, CZ}, O.options());
  at::Tensor y = save ? at::empty({N * N, P}, O.options()) : at::empty({0}, O.options());
  at::Tensor st = save ? at::empty({N * N, 2}, O.options().dtype(at::kFloat)) : at::empty({0}, O.options().dtype(at::kFloat));
  const int zt = 256;
  const int tasks = (int)(N * N / BM * (CZ / zt));
#ifdef MWE_SM120
  TORCH_CHECK(colsum_in.is_cuda() && colsum_in.scalar_type() == at::kFloat && colsum_in.is_contiguous() && colsum_in.numel() == CZ, "colsum: fp32 [CZ]");
  const at::Tensor& colsum = colsum_in;
#else
  const at::Tensor colsum = wfold.to(at::kFloat).sum(1).contiguous();      // the epilogue's mean term: sum_p W'[o, p]
#endif
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  auto launch = [&](auto kernel, int smem) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<std::min(tasks, 2 * sms), 256, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const bf16*>(O.data_ptr()), reinterpret_cast<const bf16*>(wfold.data_ptr()), bias.data_ptr<float>(),
        colsum.data_ptr<float>(),
        residual.has_value() ? reinterpret_cast<const bf16*>(residual->data_ptr()) : nullptr, reinterpret_cast<bf16*>(out.data_ptr()),
        save ? reinterpret_cast<bf16*>(y.data_ptr()) : nullptr, save ? st.data_ptr<float>() : nullptr, (int)N, (int)CZ, tasks, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };
  if (save) launch(chai_epilogue_kernel<256, true>, EpiCfg<256>::SMEM);
  else launch(chai_epilogue_kernel<256, false>, EpiCfg<256>::SMEM);
  return {out, y, st};
}

// dy [N N, 512] bf16 (= dz W'), y [N N, 512] bf16 and stats [N N, 2] fp32 (the forward's) -> dO [8, N K, N K] bf16
at::Tensor chai_bwd(at::Tensor dy, at::Tensor y, at::Tensor stats, int64_t N) {
  TORCH_CHECK(dy.is_cuda() && dy.is_contiguous() && dy.scalar_type() == at::kBFloat16 && dy.sizes() == y.sizes() && y.is_contiguous() && dy.size(1) == P
              && dy.size(0) == N * N && N % 32 == 0, "dy / y: contiguous bf16 [N N, 512]");
  const at::cuda::OptionalCUDAGuard guard(dy.device());
  const int64_t nk = N * KK;
  auto dO = at::empty({G, nk, nk}, dy.options());
  const int tiles = (int)(N * N / 32);
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  C10_CUDA_CHECK(cudaFuncSetAttribute(chai_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
  chai_bwd_kernel<<<std::min(tiles, 3 * sms), 256, 64 * 1024, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(dy.data_ptr()), reinterpret_cast<const bf16*>(y.data_ptr()), stats.data_ptr<float>(),
      reinterpret_cast<bf16*>(dO.data_ptr()), (int)N, tiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dO;
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// Chai-1 prologue: LayerNorm of the MSA rows, both projections (mma.sync m16n8k16), the mask, stored group-major: a[g, t, k] = left channel
// 8 g + k of token row t (the layout the per-group batched GEMM reads as [S, N K] matrices).  A block = 4 warps x 16 token rows.
namespace {
constexpr int WLD = 72;     // padded row of the stacked projection weight in shared memory (bank-conflict-free fragment loads)

__device__ __forceinline__ void mma16816p(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

#ifdef MWE_SM120   // the RTX 5090: 8 warps x 16 rows a CTA (the projection weights staged once per 128 rows)
#define CHAI_PJ_NW 8
__global__ void __launch_bounds__(256) chai_proj_kernel(const bf16* __restrict__ X, const unsigned char* __restrict__ mask, const float* __restrict__ gamma,
#else
#define CHAI_PJ_NW 4
__global__ void __launch_bounds__(128) chai_proj_kernel(const bf16* __restrict__ X, const unsigned char* __restrict__ mask, const float* __restrict__ gamma,
#endif
                                                         const float* __restrict__ beta, const bf16* __restrict__ W, bf16* __restrict__ A,
                                                         bf16* __restrict__ B, float* __restrict__ stats, long T, float eps) {
  __shared__ __align__(16) bf16 sW[128 * WLD];
  __shared__ float sg[64], sb[64];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
#ifdef MWE_SM120
  for (int i = tid; i < 128 * 8; i += 32 * CHAI_PJ_NW) {
#else
  for (int i = tid; i < 128 * 8; i += 128) {
#endif
    const int r = i >> 3, c = i & 7;
    *reinterpret_cast<uint4*>(sW + r * WLD + c * 8) = *reinterpret_cast<const uint4*>(W + r * 64 + c * 8);
  }
  if (tid < 64) { sg[tid] = gamma[tid]; sb[tid] = beta[tid]; }
  __syncthreads();
#ifdef MWE_SM120
  const long t0 = (long)blockIdx.x * (16 * CHAI_PJ_NW) + warp * 16;
#else
  const long t0 = (long)blockIdx.x * 64 + warp * 16;
#endif
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
      mma16816p(acc[nt], fa_[ks], b0, b1);
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
#ifdef MWE_SM120
  chai_proj_kernel<<<(unsigned)((T + 16 * CHAI_PJ_NW - 1) / (16 * CHAI_PJ_NW)), 32 * CHAI_PJ_NW, 0, at::cuda::getCurrentCUDAStream()>>>(
#else
  chai_proj_kernel<<<(unsigned)((T + 63) / 64), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
#endif
      reinterpret_cast<const bf16*>(x.data_ptr()), mp, gamma.data_ptr<float>(), beta.data_ptr<float>(), reinterpret_cast<const bf16*>(w.data_ptr()),
      reinterpret_cast<bf16*>(a.data_ptr()), reinterpret_cast<bf16*>(b.data_ptr()), save ? st.data_ptr<float>() : nullptr, T, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {a, b, st};
}



// ---------------------------------------------------------------------------------------------------------------------------------------
// Chai-1 backward of the prologue: dA, dB [8, T, 8] (the gradients of the group-major projections) -> dcat [T, 128] (masked, the operand of dW), the
// LayerNorm output ln16 [T, 64] (bf16), dx [T, 64] and per-block sums of dgamma / dbeta; dln = dcat Wlr (mma.sync), then the LayerNorm backward on
// the accumulator rows (statistics of the forward).  A block = 4 warps x 16 rows.
namespace {
#ifdef MWE_SM120   // the RTX 5090 (99 KiB: one CTA per SM): 8 warps x 16 rows a block (two of the 64-row groups, each with its own partial row), the same sums
#define CHAI_PB_NW 8
#else
#define CHAI_PB_NW 4
#endif
#ifdef MWE_SM120
__global__ void __launch_bounds__(256) chai_projbwd_kernel(const bf16* __restrict__ X, const unsigned char* __restrict__ mask, const bf16* __restrict__ DA,
#else
__global__ void __launch_bounds__(128) chai_projbwd_kernel(const bf16* __restrict__ X, const unsigned char* __restrict__ mask, const bf16* __restrict__ DA,
#endif
                                                            const bf16* __restrict__ DB, const float* __restrict__ gamma, const float* __restrict__ beta,
                                                            const float* __restrict__ stats, const bf16* __restrict__ Wlr, bf16* __restrict__ DCAT,
                                                            bf16* __restrict__ LN16, bf16* __restrict__ DX, float* __restrict__ part, long T) {
  __shared__ __align__(16) bf16 sWT[64 * 136];             // [n][k] of the stacked projection weight (k = 128 channels of dcat)
#ifdef MWE_SM120
  __shared__ float sg[64], sb[64], sred[CHAI_PB_NW][2][64];
#else
  __shared__ float sg[64], sb[64], sred[4][2][64];
#endif
  extern __shared__ __align__(16) bf16 dynS[];                                 // per warp: dcat, ln16, dx rows, flushed as 16-byte rows
#ifdef MWE_SM120
  bf16* const sStage[8] = {dynS, dynS + 8960 / 2, dynS + 2 * (8960 / 2), dynS + 3 * (8960 / 2), dynS + 4 * (8960 / 2), dynS + 5 * (8960 / 2), dynS + 6 * (8960 / 2),
                           dynS + 7 * (8960 / 2)};
#else
  bf16* const sStage[4] = {dynS, dynS + 8960 / 2, dynS + 2 * (8960 / 2), dynS + 3 * (8960 / 2)};
#endif
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
#ifdef MWE_SM120
  for (int i = tid; i < 128 * 64; i += 32 * CHAI_PB_NW) {
#else
  for (int i = tid; i < 128 * 64; i += 128) {              // Wlr [128 k][64 n] -> sWT[n][k]
#endif
    const int k = i >> 6, n = i & 63;
    sWT[n * 136 + k] = Wlr[i];
  }
  if (tid < 64) { sg[tid] = gamma[tid]; sb[tid] = beta[tid]; }
  __syncthreads();
  const int r0 = lane >> 2, q = lane & 3;
#ifdef MWE_SM120
  const long ta = (long)blockIdx.x * (16 * CHAI_PB_NW) + warp * 16 + r0, tb = ta + 8;
#else
  const long ta = (long)blockIdx.x * 64 + warp * 16 + r0, tb = ta + 8;
#endif
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
      mma16816p(acc[nt], a, *reinterpret_cast<const uint32_t*>(wr), *reinterpret_cast<const uint32_t*>(wr + 8));
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
#ifdef MWE_SM120
    const long t0 = (long)blockIdx.x * (16 * CHAI_PB_NW) + warp * 16;
#else
    const long t0 = (long)blockIdx.x * 64 + warp * 16;
#endif
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
#ifdef MWE_SM120
  {                                                        // one partial row per 64-row group (4 warps), summed in the same order as the 4-warp block
    const int grp = tid >> 7, w = (tid >> 6) & 1, c = tid & 63;
    const long row = (long)blockIdx.x * (CHAI_PB_NW / 4) + grp;
    if (row < (T + 63) / 64)
      part[row * 128 + (tid & 127)] = sred[4 * grp][w][c] + sred[4 * grp + 1][w][c] + sred[4 * grp + 2][w][c] + sred[4 * grp + 3][w][c];
  }
#else
  if (tid < 128) {
    const int w = tid >> 6, c = tid & 63;
    part[(long)blockIdx.x * 128 + tid] = sred[0][w][c] + sred[1][w][c] + sred[2][w][c] + sred[3][w][c];
  }
#endif
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
#ifdef MWE_SM120
  const long ctas = (T + 16 * CHAI_PB_NW - 1) / (16 * CHAI_PB_NW);
  cudaFuncSetAttribute(chai_projbwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, CHAI_PB_NW * 8960);
  chai_projbwd_kernel<<<(unsigned)ctas, 32 * CHAI_PB_NW, CHAI_PB_NW * 8960, at::cuda::getCurrentCUDAStream()>>>(
#else
  cudaFuncSetAttribute(chai_projbwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * 8960);
  chai_projbwd_kernel<<<(unsigned)blocks, 128, 4 * 8960, at::cuda::getCurrentCUDAStream()>>>(
#endif
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
