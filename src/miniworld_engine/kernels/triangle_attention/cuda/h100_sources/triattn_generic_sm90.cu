// Width-generic TriangleAttention glue for H100 (sm_90a): the hand-written kernels around the attention core (triattn_sm90.cu /
// triattn_bwd_sm90.cu) for every (d_pair C, heads H x 32) the fused d_pair-128 kernels (triattn_mod_sm90.cu) do not cover.  The
// projections and the weight gradients are cuBLAS GEMMs (bf16 in, fp32 accumulate); everything between them is here:
//
//   forward   ln_fwd         x -> y = LN(x) (bf16) + mean / rstd (fp32)
//             front_bias     the bias GEMM's [M, Hp] output -> head-major sbias [B, H, L, L] = bf16(bias / scale), masked keys -inf
//             ln_bias_fwd    (fp16 inference, C <= 256, H <= 8) ln_fwd + the bias projection + front_bias in one pass: y fp16 and sbias
//             (core)         triattn_fwd with `gate`: u = sigmoid(g) o O in its epilogue (O itself only when the backward needs it)
//             residual       out = x + drop o t        (t = u Wo^T from cuBLAS; drop = [B, L, C] broadcast over the rows i)
//   backward  gate_bwd_in    ddrop = dy o drop
//             gate_bwd_out   du -> do = du sigmoid(g), dg = du o sigmoid'(g) (into dall), Delta = sum do o O per head
//             (core)         triattn_bwd_kv / triattn_bwd_q write dk, dv, dq straight into their columns of dall
//             assemble       dbias (fp32 [B, H, L, L]) -> the bf16 bias columns of dall
//             ln_bwd         dx = dy + LN backward of (dall Wcat), and the per-CTA partial sums of d gamma / d beta
//
// Row m = (b, i, j) of the [M, .] matrices; dall is [M, 4 HD + Hp] (HD = 32 H, Hp = H rounded up to 8): the columns dq | dk | dv | dg |
// dbias, the layout of the packed weights Wcat = [Wq; Wk; Wv; Wg; Wb] so that dyln = dall Wcat and dW = dall^T y are one GEMM each.
// A row of C channels is handled by C / 8 lanes (C = 64, 128: a warp takes several rows) or by one warp (C <= 512).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "sm90.cuh"
#include <cstdint>
#include <tuple>
#include <type_traits>

namespace {
using bf16 = __nv_bfloat16;
constexpr int MAXV = 2;            // 16-byte vectors per lane and row: C <= 512

__device__ __forceinline__ float bf_lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ uint32_t pk(float a, float b) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a));
  return r;
}
__device__ __forceinline__ void unpack8(const uint4& v, float (&f)[8]) {
  f[0] = bf_lo(v.x); f[1] = bf_hi(v.x); f[2] = bf_lo(v.y); f[3] = bf_hi(v.y);
  f[4] = bf_lo(v.z); f[5] = bf_hi(v.z); f[6] = bf_lo(v.w); f[7] = bf_hi(v.w);
}
__device__ __forceinline__ uint4 pack8(const float (&f)[8]) { return make_uint4(pk(f[0], f[1]), pk(f[2], f[3]), pk(f[4], f[5]), pk(f[6], f[7])); }
__device__ __forceinline__ uint32_t pkh(float a, float b) {     // fp16 pair, a -> low half
  uint32_t r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a));
  return r;
}
__device__ __forceinline__ uint4 pack8h(const float (&f)[8]) { return make_uint4(pkh(f[0], f[1]), pkh(f[2], f[3]), pkh(f[4], f[5]), pkh(f[6], f[7])); }
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + __expf(-x)); }
// ending node: the logical row (b, i, j) of the starting layout is the physical row (b, j, i) of the pair tensor
__device__ __forceinline__ long phys_row(long row, int L, int tr) {
  if (!tr) return row;
  const long ll = (long)L * L, b = row / ll, rem = row - b * ll, i = rem / L, j = rem - i * L;
  return b * ll + j * L + i;
}
auto bp(const torch::Tensor& t) { return reinterpret_cast<bf16*>(t.data_ptr<at::BFloat16>()); }
auto cbp(const torch::Tensor& t) { return reinterpret_cast<const bf16*>(t.data_ptr<at::BFloat16>()); }
}  // namespace

// ---------------------------------------------------------------------------------------------------------------------------------
// y = LN(x) * w + b: LPR lanes per row (LPR = C / 8 for C = 64 / 128: a warp takes 32 / LPR rows; 32 lanes, up to two vectors each, above),
// two-pass statistics in registers.
template <int LPR>
__device__ __forceinline__ float grp_sum(float v) {
#pragma unroll
  for (int o = LPR / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
template <int LPR>
__global__ void __launch_bounds__(256) ln_fwd_k(const bf16* __restrict__ x, const float* __restrict__ w, const float* __restrict__ bs,
                                                bf16* __restrict__ y, float* __restrict__ mean_o, float* __restrict__ rstd_o, long M, int C,
                                                float eps, int L, int tr, int h16) {     // h16: y as fp16 bits (the fp16 core's GEMM input)
  constexpr int RPW = 32 / LPR, NA = LPR == 32 ? MAXV : 1;
  const int lane = threadIdx.x & 31, sub = lane % LPR, grp = lane / LPR;
  const int nv = C >> 3;
  for (long row0 = (((long)blockIdx.x * 256 + threadIdx.x) >> 5) * RPW; row0 < M; row0 += (long)gridDim.x * 8 * RPW) {
    const long row = row0 + grp;
    const bool ok = row < M;
    float v[NA][8];
    float s = 0.f;
#pragma unroll
    for (int a = 0; a < NA; ++a) {
      const int idx = sub + LPR * a;
      if (idx < nv && ok) {
        unpack8(*reinterpret_cast<const uint4*>(x + phys_row(row, L, tr) * C + idx * 8), v[a]);
#pragma unroll
        for (int e = 0; e < 8; ++e) s += v[a][e];
      }
    }
    const float mean = grp_sum<LPR>(s) / C;
    float q = 0.f;
#pragma unroll
    for (int a = 0; a < NA; ++a) {
      const int idx = sub + LPR * a;
      if (idx < nv && ok) {
#pragma unroll
        for (int e = 0; e < 8; ++e) { const float d = v[a][e] - mean; q = fmaf(d, d, q); }
      }
    }
    const float rstd = rsqrtf(grp_sum<LPR>(q) / C + eps);
#pragma unroll
    for (int a = 0; a < NA; ++a) {
      const int idx = sub + LPR * a;
      if (idx < nv && ok) {
        const float4 w0 = *reinterpret_cast<const float4*>(w + idx * 8), w1 = *reinterpret_cast<const float4*>(w + idx * 8 + 4);
        const float4 b0 = *reinterpret_cast<const float4*>(bs + idx * 8), b1 = *reinterpret_cast<const float4*>(bs + idx * 8 + 4);
        const float ww[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w}, bb[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
        float o[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) o[e] = fmaf((v[a][e] - mean) * rstd, ww[e], bb[e]);
        *reinterpret_cast<uint4*>(y + row * C + idx * 8) = h16 ? pack8h(o) : pack8(o);
      }
    }
    if (sub == 0 && ok) { mean_o[row] = mean; rstd_o[row] = rstd; }
  }
}

// fp16-core inference front, LayerNorm and bias in one pass: y = LN(x) * w + b as fp16 (the q | k | v | g GEMM's input, the values ln_fwd_k
// writes) and the head-major scaled bias sbias[b, h, i, j] = fp16(sum_c y16[c] Wb[h, c] / scale) (finite values clamped to the fp16 range,
// -inf where mask[b, j] is false) from the same registers: the bias GEMM re-read all of y for 8 output columns, and front_bias was a third
// launch.  A block = the 128 logical rows (b, i, j0 .. j0 + 127): they arrive in shared memory by bulk copies (one 64 KiB copy; the ending
// node's strided rows one copy each), each warp normalises 16 of them (one row per warp, 8 channels per lane, C <= 256), keeps its [8 heads]
// [8 channels] slice of Wb in registers and reduce-scatters the 8 dot products over the warp (9 shuffles: lane l ends with head l / 4); the
// block's [8 heads][128 keys] bias tile leaves through shared memory in 16-byte stores.
constexpr int LNB_ROWS = 128;
__global__ void __launch_bounds__(256, 2) ln_bias_fwd_k(const bf16* __restrict__ x, const float* __restrict__ w, const float* __restrict__ bs,
                                                        const bf16* __restrict__ wb, const bool* __restrict__ mask, __half* __restrict__ y,
                                                        __half* __restrict__ sbias, int C, float eps, int L, int tr, int H, float inv_scale) {
  extern __shared__ __align__(128) unsigned char lnb_sm[];
  bf16* xs = reinterpret_cast<bf16*>(lnb_sm);                                        // [128 rows][C]
  __half* sb = reinterpret_cast<__half*>(lnb_sm + LNB_ROWS * C * 2);                 // [8 heads][128 keys]
  uint64_t* bar = reinterpret_cast<uint64_t*>(lnb_sm + LNB_ROWS * C * 2 + 8 * LNB_ROWS * 2);
  const uint32_t sbar = (uint32_t)__cvta_generic_to_shared(bar), sxs = (uint32_t)__cvta_generic_to_shared(xs);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const long row0 = (long)blockIdx.x * LNB_ROWS, LL = (long)L * L;
  const long b = row0 / LL, rem = row0 - b * LL, i = rem / L, j0 = rem - i * L;
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sbar) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == 0) {
    const uint32_t rb = (uint32_t)C * 2;
    if (lane == 0) asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(sbar), "r"(LNB_ROWS * rb) : "memory");
    __syncwarp();
    if (!tr) {
      if (lane == 0)
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(sxs), "l"(x + row0 * C), "r"(LNB_ROWS * rb), "r"(sbar) : "memory");
    } else {
      for (int r = lane; r < LNB_ROWS; r += 32)
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(sxs + r * rb), "l"(x + phys_row(row0 + r, L, tr) * C), "r"(rb), "r"(sbar) : "memory");
    }
  }
  const bool act = lane < (C >> 3);
  float wr[8][8], ww[8], bb[8];
#pragma unroll
  for (int h = 0; h < 8; ++h) {
    if (act) unpack8(*reinterpret_cast<const uint4*>(wb + (long)h * C + lane * 8), wr[h]);
    else {
#pragma unroll
      for (int e = 0; e < 8; ++e) wr[h][e] = 0.f;
    }
  }
#pragma unroll
  for (int e = 0; e < 8; ++e) { ww[e] = act ? w[lane * 8 + e] : 0.f; bb[e] = act ? bs[lane * 8 + e] : 0.f; }
  asm volatile("{\n.reg .pred p;\nLNB_WAIT:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n@!p bra LNB_WAIT;\n}\n" :: "r"(sbar) : "memory");
#pragma unroll 2
  for (int k = 0; k < LNB_ROWS / 8; ++k) {
    const int r = warp + 8 * k;
    const long row = row0 + r;
    float v[8];
    float s = 0.f;
    if (act) {
      unpack8(*reinterpret_cast<const uint4*>(xs + r * C + lane * 8), v);
#pragma unroll
      for (int e = 0; e < 8; ++e) s += v[e];
    }
    const float mean = warp_sum(s) / C;
    float q = 0.f;
    if (act) {
#pragma unroll
      for (int e = 0; e < 8; ++e) { const float d = v[e] - mean; q = fmaf(d, d, q); }
    }
    const float rstd = rsqrtf(warp_sum(q) / C + eps);
    float part[8];
#pragma unroll
    for (int h = 0; h < 8; ++h) part[h] = 0.f;
    if (act) {
      float o[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) o[e] = fmaf((v[e] - mean) * rstd, ww[e], bb[e]);
      const uint4 yv = pack8h(o);
      *reinterpret_cast<uint4*>(y + row * C + lane * 8) = yv;
      const uint32_t yw[4] = {yv.x, yv.y, yv.z, yv.w};
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {      // the bias GEMM's input: y as rounded to fp16
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&yw[kk]));
#pragma unroll
        for (int h = 0; h < 8; ++h) part[h] = fmaf(f.y, wr[h][2 * kk + 1], fmaf(f.x, wr[h][2 * kk], part[h]));
      }
    }
    // reduce-scatter: halves of the heads over lane bit 4, then bit 3, then bit 2; bits 1 and 0 sum the last one
    const bool b4 = lane & 16, b3 = lane & 8, b2 = lane & 4;
    float a4[4], a2[2];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) a4[kk] = (b4 ? part[kk + 4] : part[kk]) + __shfl_xor_sync(0xffffffffu, b4 ? part[kk] : part[kk + 4], 16);
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) a2[kk] = (b3 ? a4[kk + 2] : a4[kk]) + __shfl_xor_sync(0xffffffffu, b3 ? a4[kk] : a4[kk + 2], 8);
    float a1 = (b2 ? a2[1] : a2[0]) + __shfl_xor_sync(0xffffffffu, b2 ? a2[0] : a2[1], 4);
    a1 += __shfl_xor_sync(0xffffffffu, a1, 2);
    a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
    if ((lane & 3) == 0) {
      const bool keep = mask == nullptr || mask[b * L + j0 + r];
      sb[(lane >> 2) * LNB_ROWS + r] = __float2half_rn(keep ? fmaxf(fminf(a1 * inv_scale, 6.0e4f), -6.0e4f) : -INFINITY);
    }
  }
  __syncthreads();
  if (tid < 8 * LNB_ROWS / 8) {      // 16 bytes = 8 keys per thread
    const int h = tid / (LNB_ROWS / 8), seg = tid % (LNB_ROWS / 8);
    if (h < H)
      *reinterpret_cast<uint4*>(sbias + ((b * H + h) * L + i) * L + j0 + seg * 8) = *reinterpret_cast<const uint4*>(sb + h * LNB_ROWS + seg * 8);
  }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// fp16-core inference front at d_pair 256 (q | k | v | g of 4 HD = 64 nch columns, at most 8 heads): y = LN(x) (fp16, never stored),
// [q | k | v | g] = y W^T (fp16 out, the [M, 4 HD] layout of the projection GEMM it replaces) and the head-major scaled bias of the fp16
// core, in one kernel -- it replaces ln_fwd / ln_bias_fwd, the projection GEMM (which re-read y) and front_bias.
// Tile = 128 rows (one (b, i), 128 consecutive j).  384 threads: warpgroup 0 = TMA producer, warpgroups 1 / 2 = rows 0-63 / 64-127.
// A consumer warp ldmatrix-loads its 16 rows of x straight into the wgmma A-fragment layout (16 k16 blocks), releases the x stage and
// normalises in registers (a row's 256 channels sit on the 4 lanes of a quad: the statistics are two quad sums; y = fp16 of the fp32
// LayerNorm, the value ln_fwd_k writes).  The fp16 weights [4 HD][256] do not fit beside it (512 KiB), so they stream through a ring
// of 64-column chunks (32 KiB) shared by both consumer warpgroups and by the two CTAs of a cluster (each CTA's producer loads half of
// every chunk, multicast to both: the L2 serves each chunk once per 256 rows); every chunk is 16 m64n64k16 RS wgmma per warpgroup,
// double-buffered so chunk c + 1 runs on the tensor core while chunk c is stored.  The weight rows are permuted within each chunk so
// that a lane's 16 accumulator columns of a row are 2 x 8 consecutive output columns: every lane stores 16 contiguous bytes, a warp
// store covers 8 rows x 64 bytes (whole sectors), no staging.  The bias (8 resident Wb rows, one m64n8 chain) leaves through shared
// memory as [8 heads][128 keys] in 16-byte stores.  The kernel is HBM-bound: it writes four times what it reads.
namespace fc {
constexpr int C = 256, BM = 128, NC = 64, NW = 4;
constexpr int XT = BM * C;                                // x tile [4 k blocks][128 rows][64], 128B swizzle (64 KiB)
constexpr int WT = NC * C;                                // weight chunk [4 k blocks][64 rows][64] fp16 (32 KiB)
constexpr int WBT = 8 * C;                                // Wb [4 k blocks][8 rows][64] fp16 (4 KiB)
constexpr int THREADS = 384;
constexpr int SMEM = 1024 + (XT + NW * WT + WBT) * 2 + 2 * C * 4 + 8 * BM * 2 + 256;
static_assert(SMEM <= 232448, "smem");
__device__ __forceinline__ void mma_n64(float (&d)[32], const uint32_t (&a)[4], uint64_t b, uint32_t scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 " SM90_L32 ", {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
               : SM90_D32(0) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(scale_d));
}
__device__ __forceinline__ void mma_n8(float (&d)[4], const uint32_t (&a)[4], uint64_t b, uint32_t scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %9, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(scale_d));
}
__device__ __forceinline__ uint32_t h2(float a, float b) {            // fp16 pair, a -> low half
  uint32_t r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(b), "f"(a));
  return r;
}
__device__ __forceinline__ uint32_t cta_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ void cluster_sync() { asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;\n" ::: "memory"); }
__device__ __forceinline__ void arrive_rank(uint32_t bar, uint32_t rank) {      // arrive on the barrier at this offset in CTA `rank`
  // (default .release.cta semantics: the stage was read by the wgmma, retired before; .release.cluster put a MEMBAR behind every
  // chunk's stores -- 35 % of the stall samples)
  asm volatile("{\n.reg .b32 ra;\nmapa.shared::cluster.u32 ra, %0, %1;\nmbarrier.arrive.shared::cluster.b64 _, [ra];\n}\n"
               :: "r"(bar), "r"(rank) : "memory");
}
__device__ __forceinline__ void load_2d_mc(const void* map, uint32_t dst, uint32_t bar, int c0, int c1, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;\n"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "h"(mask) : "memory");
}
}  // namespace fc

// NCH: 64-column chunks of q | k | v | g (unrolled: no wgmma may be in flight across a branch or a back edge); MC: launched as clusters
// of two CTAs that share every weight chunk (multicast), else each CTA loads its own
template <int NCH, bool MC>
__global__ void __launch_bounds__(fc::THREADS, 1) front_c256_sm90(
    int L, int ntiles, float eps, float inv_scale, int tr, int H,
    const __grid_constant__ CUtensorMap xmap,             // x [R][256] bf16, box (64, 128), 128B swizzle (tr: the 4-D transposed pair map)
    const __grid_constant__ CUtensorMap wmap,             // W [nch 64][256] fp16 (rows permuted per chunk), box (64, 64), 128B swizzle
    const __grid_constant__ CUtensorMap wbmap,            // Wb [8][256] fp16 (rows >= H zero), box (64, 8), 128B swizzle
    const float* __restrict__ LNW, const float* __restrict__ LNB,
    const bool* __restrict__ MASK,                        // [B][L] key mask or nullptr
    __half* __restrict__ OUT,                             // [R][nch 64] fp16: q | k | v | g
    __half* __restrict__ SBIAS) {                         // [B][H][L][L] fp16
  using namespace fc;
  using sm90::sa;
  constexpr int nch = NCH;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  const uint32_t sX = sa(smb), sW = sX + XT * 2, sWb = sW + NW * WT * 2, sGB = sWb + WBT * 2, sSB = sGB + 2 * C * 4, bar0 = sSB + 8 * BM * 2;
  float* gb = reinterpret_cast<float*>(smb + (sGB - sX));                    // gamma [256] | beta [256]
  __half* sbt = reinterpret_cast<__half*>(smb + (sSB - sX));                 // the tile's bias [8 heads][128 keys]
  const uint32_t xf = bar0, xe = bar0 + 8, wbf = bar0 + 16;                 // xe: count 8 (every consumer warp holds its fragments)
  auto wf = [&](int s) { return bar0 + 24 + 8 * s; };
  auto we = [&](int s) { return bar0 + 24 + 8 * (NW + s); };                // count 16: every consumer warp of both CTAs
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  const uint32_t rank = MC ? cta_rank() : 0u, peer = rank ^ 1u;
  for (int c = tid; c < 2 * C; c += THREADS) gb[c] = c < C ? LNW[c] : LNB[c - C];
  if (tid == 0) {
    uint64_t* bars = reinterpret_cast<uint64_t*>(smb + (bar0 - sX));
    sm90::bar_init(bars, 1); sm90::bar_init(bars + 1, 8); sm90::bar_init(bars + 2, 1);
    for (int s = 0; s < NW; ++s) { sm90::bar_init(bars + 3 + s, 1); sm90::bar_init(bars + 3 + NW + s, MC ? 16 : 8); }
    sm90::bar_init_fence();
  }
  __syncthreads();
  if (MC) cluster_sync();                                 // the peer's barriers exist before a multicast or a remote arrive reaches them

  if (wg == 0) {
    sm90::reg_dealloc<40>();
    if (tid == 0) {
      sm90::prefetch_map(&xmap); sm90::prefetch_map(&wmap); sm90::prefetch_map(&wbmap);
      sm90::expect_tx(wbf, WBT * 2);
      for (int kb = 0; kb < 4; ++kb) sm90::load_2d(&wbmap, sWb + kb * 8 * 128, wbf, kb * 64, 0);
      int ws = 0, wph = 0, xph = 0;
      const int t0 = blockIdx.x;
      if (t0 < ntiles) {
        sm90::expect_tx(xf, XT * 2);
        for (int kb = 0; kb < 4; ++kb) sm90::load_pair(&xmap, sX + kb * BM * 128, xf, kb * 64, t0 * BM, L, tr);
      }
      for (int t = t0; t < ntiles; t += gridDim.x) {
        for (int c = 0; c < nch; ++c) {
          if (c == (nch > 2 ? 2 : nch - 1) && t + (int)gridDim.x < ntiles) {     // the next tile's x, once the consumers hold this one's fragments
            sm90::wait(xe, xph); xph ^= 1;
            sm90::expect_tx(xf, XT * 2);
            for (int kb = 0; kb < 4; ++kb) sm90::load_pair(&xmap, sX + kb * BM * 128, xf, kb * 64, (t + gridDim.x) * BM, L, tr);
          }
          sm90::wait(we(ws), wph ^ 1);                     // the stage is free in both CTAs
          sm90::expect_tx(wf(ws), WT * 2);
          if (MC) {
            for (int kb = 2 * (int)rank; kb < 2 * (int)rank + 2; ++kb)
              load_2d_mc(&wmap, sW + ws * WT * 2 + kb * NC * 128, wf(ws), kb * 64, c * NC, (uint16_t)3);
          } else {
            for (int kb = 0; kb < 4; ++kb) sm90::load_2d(&wmap, sW + ws * WT * 2 + kb * NC * 128, wf(ws), kb * 64, c * NC);
          }
          if (++ws == NW) { ws = 0; wph ^= 1; }
        }
      }
    }
  } else {
    sm90::reg_alloc<232>();
    const int w = wg - 1, T = tid - 128, wi = (tid & 127) >> 5, g8 = lane >> 2, t4 = lane & 3;
    const int lrow = 64 * w + 16 * wi + (lane & 7) + 8 * ((lane >> 3) & 1), lcol8 = lane >> 4;
    sm90::wait(wbf, 0);
    int ws = 0, wph = 0, xph = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
      sm90::wait(xf, xph); xph ^= 1;
      auto chunk = [&](int c16) { return sX + (c16 >> 3) * BM * 128 + lrow * 128 + (((c16 & 7) ^ (lrow & 7)) << 4); };
      uint32_t ya[16][4];
#pragma unroll
      for (int kb = 0; kb < 16; ++kb) sm90::ldsm_x4(chunk(2 * kb + lcol8), ya[kb]);
#pragma unroll
      for (int kb = 0; kb < 16; ++kb) sm90::fence_regs(ya[kb]);
      sm90::fence_proxy_async();
      __syncwarp();
      if (lane == 0) sm90::arrive(xe);
      // LayerNorm of rows g (words 0, 2 of each block) and g + 8 (words 1, 3), fp32
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int kb = 0; kb < 16; ++kb) {
        const float2 a = sm90::bf2f(ya[kb][0]), b = sm90::bf2f(ya[kb][2]), c = sm90::bf2f(ya[kb][1]), d = sm90::bf2f(ya[kb][3]);
        s0 += (a.x + a.y) + (b.x + b.y); s1 += (c.x + c.y) + (d.x + d.y);
      }
      const float mu0 = sm90::quad_sum(s0) * (1.f / C), mu1 = sm90::quad_sum(s1) * (1.f / C);
      float v0 = 0.f, v1 = 0.f;
#pragma unroll
      for (int kb = 0; kb < 16; ++kb) {
        const float2 a = sm90::bf2f(ya[kb][0]), b = sm90::bf2f(ya[kb][2]), c = sm90::bf2f(ya[kb][1]), d = sm90::bf2f(ya[kb][3]);
        v0 += (a.x - mu0) * (a.x - mu0) + (a.y - mu0) * (a.y - mu0) + (b.x - mu0) * (b.x - mu0) + (b.y - mu0) * (b.y - mu0);
        v1 += (c.x - mu1) * (c.x - mu1) + (c.y - mu1) * (c.y - mu1) + (d.x - mu1) * (d.x - mu1) + (d.y - mu1) * (d.y - mu1);
      }
      const float r0 = rsqrtf(sm90::quad_sum(v0) * (1.f / C) + eps), r1 = rsqrtf(sm90::quad_sum(v1) * (1.f / C) + eps);
#pragma unroll
      for (int kb = 0; kb < 16; ++kb) {
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {                      // channels 16 kb + 8 hf + 2 t4 + {0, 1}
          const int k = 16 * kb + 8 * hf + 2 * t4;
          const float2 gm = *reinterpret_cast<const float2*>(gb + k), bt = *reinterpret_cast<const float2*>(gb + C + k);
          const float2 a = sm90::bf2f(ya[kb][2 * hf]), c = sm90::bf2f(ya[kb][2 * hf + 1]);
          ya[kb][2 * hf] = h2(fmaf((a.x - mu0) * r0, gm.x, bt.x), fmaf((a.y - mu0) * r0, gm.y, bt.y));
          ya[kb][2 * hf + 1] = h2(fmaf((c.x - mu1) * r1, gm.x, bt.x), fmaf((c.y - mu1) * r1, gm.y, bt.y));
        }
      }
      const long row = (long)t * BM + 64 * w + 16 * wi + g8;      // this lane's first row; its second is row + 8
      const int ldo = nch * NC;
      float acc0[32], acc1[32];
      auto take = [&](float (&acc)[32]) -> int {               // the next ring stage: wait for its chunk, issue its GEMM
        const int s = ws;
        sm90::wait(wf(s), wph);
        sm90::wg_fence();
#pragma unroll
        for (int kb = 0; kb < 16; ++kb)
          mma_n64(acc, ya[kb], sm90::desc_k128(sW + s * WT * 2 + (kb >> 2) * NC * 128 + (kb & 3) * 32), kb ? 1u : 0u);
        sm90::wg_commit();
        if (++ws == NW) { ws = 0; wph ^= 1; }
        return s;
      };
      // chunk c retired: release its stage in both CTAs, store it (a lane's two runs of 8 columns of rows g, g + 8)
      auto finish = [&](float (&acc)[32], int s, int c) {
        sm90::fence_regs(acc);
        __syncwarp();
        if (lane == 0) {
          if (MC) { arrive_rank(we(s), rank); arrive_rank(we(s), peer); }
          else sm90::arrive(we(s));
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
          for (int hf = 0; hf < 2; ++hf) {
            uint4 v;
            v.x = h2(acc[4 * (4 * r + 0) + 2 * hf], acc[4 * (4 * r + 0) + 2 * hf + 1]);
            v.y = h2(acc[4 * (4 * r + 1) + 2 * hf], acc[4 * (4 * r + 1) + 2 * hf + 1]);
            v.z = h2(acc[4 * (4 * r + 2) + 2 * hf], acc[4 * (4 * r + 2) + 2 * hf + 1]);
            v.w = h2(acc[4 * (4 * r + 3) + 2 * hf], acc[4 * (4 * r + 3) + 2 * hf + 1]);
            *reinterpret_cast<uint4*>(OUT + (row + 8 * hf) * ldo + c * NC + 32 * r + 8 * t4) = v;
          }
        }
      };
      // chunks in pairs (nch is even), so each accumulator set is a fixed register range: chunk c + 1 runs while chunk c is stored
      int stg0 = take(acc0);
#pragma unroll
      for (int c = 0; c < nch; c += 2) {
        const int stg1 = take(acc1);
        sm90::wg_wait<1>();
        finish(acc0, stg0, c);
        if (c + 2 < nch) {
          stg0 = take(acc0);
          sm90::wg_wait<1>();
        } else {
          sm90::wg_wait<0>();
        }
        finish(acc1, stg1, c + 1);
      }
      // the bias: one m64n8 chain over the resident Wb; lanes hold heads 2 t4, 2 t4 + 1 of rows g, g + 8
      float bacc[4];
      sm90::wg_fence();
#pragma unroll
      for (int kb = 0; kb < 16; ++kb) mma_n8(bacc, ya[kb], sm90::desc_k128(sWb + (kb >> 2) * 8 * 128 + (kb & 3) * 32), kb ? 1u : 0u);
      sm90::wg_commit();
      sm90::wg_wait<0>();
      sm90::fence_regs(bacc);
      const long bi = (long)t * BM / L, b = bi / L, i = bi - b * L;
      const int j0 = (int)((long)t * BM - bi * L);
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int r = 64 * w + 16 * wi + g8 + 8 * hf;           // row in the tile = key j0 + r
        const bool keep = MASK == nullptr || MASK[b * L + j0 + r];
#pragma unroll
        for (int e = 0; e < 2; ++e)
          sbt[(2 * t4 + e) * BM + r] = __float2half_rn(keep ? fmaxf(fminf(bacc[2 * hf + e] * inv_scale, 6.0e4f), -6.0e4f) : -INFINITY);
      }
      sm90::named_sync(1, 256);
      if (T < 8 * BM / 8) {                                     // 16 bytes = 8 keys per thread
        const int h = T / (BM / 8), seg = T % (BM / 8);
        if (h < H) *reinterpret_cast<uint4*>(SBIAS + ((b * H + h) * L + i) * L + j0 + seg * 8) = *reinterpret_cast<const uint4*>(sbt + h * BM + seg * 8);
      }
      sm90::named_sync(1, 256);                                 // the staging is read before the next tile writes it
    }
  }
  if (MC) cluster_sync();                                 // no CTA leaves while its peer may still multicast into it or arrive on it
}

// sbias[b, h, i, j] = T(qkvb[(b, i, j), bias_col + h] / scale), -inf where mask[b, j] is false (T = bf16, or fp16 for the fp16 inference core:
// finite values clamped to its range).  A block = 128 consecutive j of one (b, i)
template <typename T>
__global__ void __launch_bounds__(256) front_bias_k(const T* __restrict__ qkvb, const bool* __restrict__ mask, T* __restrict__ sbias,
                                                    long ld, int bias_col, int H, int L, float inv_scale) {
  const long row0 = (long)blockIdx.x * 128;                  // (b * L + i) * L + j0
  const long bi = row0 / L;
  const int j0 = (int)(row0 - bi * L), b = (int)(bi / L), i = (int)(bi - (long)b * L);
  for (int idx = threadIdx.x; idx < 128 * H; idx += 256) {
    const int h = idx >> 7, r = idx & 127;
    const bool keep = mask == nullptr || mask[(long)b * L + j0 + r];
    const T in = qkvb[(row0 + r) * ld + bias_col + h];
    float f;
    if constexpr (std::is_same_v<T, __half>) f = __half2float(in); else f = __bfloat162float(in);
    const float val = keep ? f * inv_scale : -INFINITY;
    if constexpr (std::is_same_v<T, __half>) sbias[(((long)b * H + h) * L + i) * L + j0 + r] = __float2half_rn(keep ? fmaxf(fminf(val, 6.0e4f), -6.0e4f) : val);
    else sbias[(((long)b * H + h) * L + i) * L + j0 + r] = __float2bfloat16(val);
  }
}

// u = sigmoid(g) o o (g: the qkvb column block, o contiguous [M, HD]); u [M, HD]
__global__ void __launch_bounds__(256) gate_u_k(const bf16* __restrict__ qkvb, const bf16* __restrict__ o, bf16* __restrict__ u, long M,
                                                int HD, long ld, int g_col) {
  const int nv = HD >> 3;
  const long total = M * nv;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < total; i += (long)gridDim.x * 256) {
    const long row = i / nv;
    const int c = (int)(i - row * nv) * 8;
    float g[8], ov[8], r[8];
    unpack8(*reinterpret_cast<const uint4*>(qkvb + row * ld + g_col + c), g);
    unpack8(*reinterpret_cast<const uint4*>(o + row * HD + c), ov);
#pragma unroll
    for (int e = 0; e < 8; ++e) r[e] = sigmoidf_(g[e]) * ov[e];
    *reinterpret_cast<uint4*>(u + row * HD + c) = pack8(r);
  }
}

// out = x + drop o t; drop [B, L, C] (index (b, j)) or nullptr
__global__ void __launch_bounds__(256) residual_k(const bf16* __restrict__ x, const bf16* __restrict__ t, const bf16* __restrict__ drop,
                                                  bf16* __restrict__ out, long M, int C, int L, int tr) {
  const int nv = C >> 3;
  const long total = M * nv;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < total; i += (long)gridDim.x * 256) {
    const long row = i / nv;
    const long prow = phys_row(row, L, tr);
    const int c = (int)(i - row * nv) * 8;
    float xv[8], tv[8], r[8];
    unpack8(*reinterpret_cast<const uint4*>(x + prow * C + c), xv);
    unpack8(*reinterpret_cast<const uint4*>(t + row * C + c), tv);
    if (drop != nullptr) {
      const long bj = (row / ((long)L * L)) * L + row % L;
      float dv[8];
      unpack8(*reinterpret_cast<const uint4*>(drop + bj * C + c), dv);
#pragma unroll
      for (int e = 0; e < 8; ++e) tv[e] *= dv[e];
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) r[e] = xv[e] + tv[e];
    *reinterpret_cast<uint4*>(out + prow * C + c) = pack8(r);
  }
}

// ddrop = dy o drop
__global__ void __launch_bounds__(256) gate_bwd_in_k(const bf16* __restrict__ dy, const bf16* __restrict__ drop, bf16* __restrict__ dd,
                                                     long M, int C, int L, int tr) {
  const int nv = C >> 3;
  const long total = M * nv;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < total; i += (long)gridDim.x * 256) {
    const long row = i / nv;
    const int c = (int)(i - row * nv) * 8;
    float v[8];
    unpack8(*reinterpret_cast<const uint4*>(dy + phys_row(row, L, tr) * C + c), v);
    if (drop != nullptr) {
      const long bj = (row / ((long)L * L)) * L + row % L;
      float dv[8];
      unpack8(*reinterpret_cast<const uint4*>(drop + bj * C + c), dv);
#pragma unroll
      for (int e = 0; e < 8; ++e) v[e] *= dv[e];
    }
    *reinterpret_cast<uint4*>(dd + row * C + c) = pack8(v);
  }
}

// do = du sigmoid(g) -> do [M, HD];  dg = du o sigmoid'(g) -> dall[:, g_col ...];  delta[b, i, h, j] = sum_c do o o (the core backward's Delta;
// the 4 lanes of a head are consecutive threads)
__global__ void __launch_bounds__(256) gate_bwd_out_k(const bf16* __restrict__ du, const bf16* __restrict__ qkvb, const bf16* __restrict__ o,
                                                      bf16* __restrict__ dout, bf16* __restrict__ dall, float* __restrict__ delta, long M, int HD,
                                                      long ld, int g_col, long ld_all, int H, int L) {
  const int nv = HD >> 3;
  const long total = M * nv;
  for (long i0 = (long)blockIdx.x * 256; i0 < total; i0 += (long)gridDim.x * 256) {
    const long i = i0 + threadIdx.x;
    const bool ok = i < total;
    const long row = ok ? i / nv : 0;
    const int cv = ok ? (int)(i - row * nv) : 0, c = cv * 8;
    float d[8] = {0, 0, 0, 0, 0, 0, 0, 0}, g[8] = {0, 0, 0, 0, 0, 0, 0, 0}, ov[8] = {0, 0, 0, 0, 0, 0, 0, 0}, a[8], b[8];
    if (ok) {
      unpack8(*reinterpret_cast<const uint4*>(du + row * HD + c), d);
      unpack8(*reinterpret_cast<const uint4*>(qkvb + row * ld + g_col + c), g);
      unpack8(*reinterpret_cast<const uint4*>(o + row * HD + c), ov);
    }
    float part = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float s = sigmoidf_(g[e]);
      a[e] = d[e] * s;
      b[e] = d[e] * ov[e] * s * (1.f - s);
    }
    const uint4 ao = pack8(a);
    const uint32_t aw[4] = {ao.x, ao.y, ao.z, ao.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) { part = fmaf(bf_lo(aw[e]), ov[2 * e], part); part = fmaf(bf_hi(aw[e]), ov[2 * e + 1], part); }
    part += __shfl_xor_sync(0xffffffffu, part, 1);
    part += __shfl_xor_sync(0xffffffffu, part, 2);
    if (ok) {
      *reinterpret_cast<uint4*>(dout + row * HD + c) = ao;
      *reinterpret_cast<uint4*>(dall + row * ld_all + g_col + c) = pack8(b);
      if ((cv & 3) == 0) delta[((row / L) * H + (cv >> 2)) * L + row % L] = part;
    }
  }
}

// dall[:, 4 HD + h] <- bf16(dbias[b, h, i, j]) (zero for h >= H); the first 4 HD columns are written by the core backward and the gate backward.
// One thread per 16 bytes of a row's bias block.
__global__ void __launch_bounds__(256) assemble_k(const float* __restrict__ dbias, bf16* __restrict__ dall, long M, int HD, long ld, int H, int Hp,
                                                  int L) {
  const int nb = Hp >> 3;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < M * nb; i += (long)gridDim.x * 256) {
    const long row = i / nb;
    const int h0 = (int)(i - row * nb) * 8;
    const long bi = row / L;
    const int j = (int)(row - bi * L), b = (int)(bi / L), ii = (int)(bi - (long)b * L);
    float f[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) f[e] = h0 + e < H ? dbias[(((long)b * H + h0 + e) * L + ii) * L + j] : 0.f;
    *reinterpret_cast<uint4*>(dall + row * ld + 4 * HD + h0) = pack8(f);
  }
}

__device__ __forceinline__ void load8(const float* p, float (&f)[8]) {
  const float4 a = *reinterpret_cast<const float4*>(p), b = *reinterpret_cast<const float4*>(p + 4);
  f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w; f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
}
__device__ __forceinline__ void load8(const bf16* p, float (&f)[8]) { unpack8(*reinterpret_cast<const uint4*>(p), f); }
template <int LPR, typename TD>
__global__ void __launch_bounds__(256) ln_bwd_k(const TD* __restrict__ dyln, const bf16* __restrict__ x, const bf16* __restrict__ dy,
                                                const float* __restrict__ w, const float* __restrict__ mean_i, const float* __restrict__ rstd_i,
                                                bf16* __restrict__ dx, float* __restrict__ part, long M, int C, int L, int tr) {
  constexpr int RPW = 32 / LPR, NA = LPR == 32 ? MAXV : 1;
  __shared__ float red[8192];                              // [8 warps x RPW][2][C]
  const int lane = threadIdx.x & 31, wi = threadIdx.x >> 5, sub = lane % LPR, grp = lane / LPR;
  const int nv = C >> 3;
  float pg[NA][8], pb[NA][8];
#pragma unroll
  for (int a = 0; a < NA; ++a)
#pragma unroll
    for (int e = 0; e < 8; ++e) { pg[a][e] = 0.f; pb[a][e] = 0.f; }
  for (long row0 = (((long)blockIdx.x * 256 + threadIdx.x) >> 5) * RPW; row0 < M; row0 += (long)gridDim.x * 8 * RPW) {
    const long row = row0 + grp;
    const bool ok = row < M;
    const float mean = ok ? mean_i[row] : 0.f, rstd = ok ? rstd_i[row] : 0.f;
    float g[NA][8], xh[NA][8];
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int a = 0; a < NA; ++a) {
      const int idx = sub + LPR * a;
      if (idx < nv && ok) {
        float xv[8];
        unpack8(*reinterpret_cast<const uint4*>(x + phys_row(row, L, tr) * C + idx * 8), xv);
        float dd[8];
        load8(dyln + row * C + idx * 8, dd);
        const float4 w0 = *reinterpret_cast<const float4*>(w + idx * 8), w1 = *reinterpret_cast<const float4*>(w + idx * 8 + 4);
        const float ww[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          xh[a][e] = (xv[e] - mean) * rstd;
          g[a][e] = dd[e] * ww[e];
          pg[a][e] = fmaf(dd[e], xh[a][e], pg[a][e]);
          pb[a][e] += dd[e];
          s1 += g[a][e];
          s2 = fmaf(g[a][e], xh[a][e], s2);
        }
      }
    }
    const float m1 = grp_sum<LPR>(s1) / C, m2 = grp_sum<LPR>(s2) / C;
#pragma unroll
    for (int a = 0; a < NA; ++a) {
      const int idx = sub + LPR * a;
      if (idx < nv && ok) {
        float r[8];
        unpack8(*reinterpret_cast<const uint4*>(dy + phys_row(row, L, tr) * C + idx * 8), r);
#pragma unroll
        for (int e = 0; e < 8; ++e) r[e] += rstd * (g[a][e] - m1 - xh[a][e] * m2);
        *reinterpret_cast<uint4*>(dx + phys_row(row, L, tr) * C + idx * 8) = pack8(r);
      }
    }
  }
#pragma unroll
  for (int a = 0; a < NA; ++a) {
    const int idx = sub + LPR * a;
    if (idx < nv)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        red[((wi * RPW + grp) * 2) * C + idx * 8 + e] = pg[a][e];
        red[((wi * RPW + grp) * 2 + 1) * C + idx * 8 + e] = pb[a][e];
      }
  }
  __syncthreads();
  for (int c = threadIdx.x; c < 2 * C; c += 256) {
    const int k = c / C, cc = c - k * C;
    float s = 0.f;
    for (int sl = 0; sl < 8 * RPW; ++sl) s += red[(sl * 2 + k) * C + cc];
    part[((long)blockIdx.x * 2 + k) * C + cc] = s;
  }
}

// ---------------------------------------------------------------------------------------------------------------------------------
namespace {
int grid_for(long work, int per_block) { return (int)std::min<long>((work + per_block - 1) / per_block, 132L * 16); }
cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }
}  // namespace

// x [M, C] bf16 -> (y bf16 [M, C], mean fp32 [M], rstd fp32 [M])
std::vector<torch::Tensor> g_ln_fwd(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps, int64_t L, bool trans, bool h16) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && x.scalar_type() == torch::kBFloat16, "x [M, C] bf16 contiguous");
  const long M = x.size(0);
  const int C = (int)x.size(1);
  TORCH_CHECK(C % 8 == 0 && C <= 256 * MAXV, "C % 8 == 0 and C <= 512");
  auto y = h16 ? torch::empty_like(x, x.options().dtype(torch::kHalf)) : torch::empty_like(x);
  auto mean = torch::empty({M}, x.options().dtype(torch::kFloat32)), rstd = torch::empty({M}, x.options().dtype(torch::kFloat32));
  const int lpr = C == 64 ? 8 : C == 128 ? 16 : 32;
  const int grid = grid_for(M, 8 * (32 / lpr));
  auto args = std::make_tuple(cbp(x), w.data_ptr<float>(), b.data_ptr<float>(), reinterpret_cast<bf16*>(y.data_ptr()), mean.data_ptr<float>(), rstd.data_ptr<float>(), M, C, (float)eps, (int)L, (int)trans,
                              (int)h16);
  if (lpr == 8) std::apply([&](auto... a) { ln_fwd_k<8><<<grid, 256, 0, stream()>>>(a...); }, args);
  else if (lpr == 16) std::apply([&](auto... a) { ln_fwd_k<16><<<grid, 256, 0, stream()>>>(a...); }, args);
  else std::apply([&](auto... a) { ln_fwd_k<32><<<grid, 256, 0, stream()>>>(a...); }, args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, mean, rstd};
}

// x [M, C] bf16 (C <= 256), wb [8, C] bf16 (heads >= H zero) -> (y fp16 [M, C], sbias fp16 [B, H, L, L] = bias / scale, masked keys -inf)
std::vector<torch::Tensor> g_ln_bias_fwd(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor wb, torch::Tensor mask, double eps,
                                         int64_t B, int64_t L, int64_t H, bool trans, double scale) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && x.scalar_type() == torch::kBFloat16, "x [M, C] bf16 contiguous");
  const long M = x.size(0);
  const int C = (int)x.size(1);
  TORCH_CHECK(C % 8 == 0 && C <= 256 && M == B * L * L && H >= 1 && H <= 8 && L % LNB_ROWS == 0, "C % 8 == 0, C <= 256, M = B L L, H <= 8, L % 128 == 0");
  TORCH_CHECK(wb.is_cuda() && wb.is_contiguous() && wb.scalar_type() == torch::kBFloat16 && wb.size(0) == 8 && wb.size(1) == C, "wb [8, C] bf16");
  TORCH_CHECK(w.is_contiguous() && b.is_contiguous() && w.scalar_type() == torch::kFloat32 && b.scalar_type() == torch::kFloat32, "w, b fp32");
  auto y = torch::empty_like(x, x.options().dtype(torch::kHalf));
  auto sbias = torch::empty({B, H, L, L}, x.options().dtype(torch::kHalf));
  torch::Tensor mk;
  const bool* mp = nullptr;
  if (mask.numel() > 0) { mk = mask.to(torch::kBool).contiguous(); TORCH_CHECK(mk.numel() == B * L, "mask [B, L]"); mp = mk.data_ptr<bool>(); }
  const int smem = LNB_ROWS * C * 2 + 8 * LNB_ROWS * 2 + 16;
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(ln_bias_fwd_k, cudaFuncAttributeMaxDynamicSharedMemorySize, LNB_ROWS * 256 * 2 + 8 * LNB_ROWS * 2 + 16)); attr = true; }
  ln_bias_fwd_k<<<(unsigned)(M / LNB_ROWS), 256, smem, stream()>>>(cbp(x), w.data_ptr<float>(), b.data_ptr<float>(), cbp(wb), mp,
                                                                   reinterpret_cast<__half*>(y.data_ptr()), reinterpret_cast<__half*>(sbias.data_ptr()),
                                                                   C, (float)eps, (int)L, (int)trans, (int)H, (float)(1.0 / scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, sbias};
}

// x [M, 256] bf16 (the pair's physical rows), w16 [4 HD, 256] fp16 (rows permuted within each 64-row chunk, see front_c256_sm90),
// wb16 [8, 256] fp16 (heads >= H zero) -> (q | k | v | g fp16 [M, 4 HD], sbias fp16 [B, H, L, L] = bias / scale, masked keys -inf)
std::vector<torch::Tensor> g_front_c256(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor w16, torch::Tensor wb16, torch::Tensor mask,
                                        double eps, int64_t B, int64_t L, int64_t H, bool trans, double scale, bool mc) {
  using namespace sm90;
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && x.scalar_type() == torch::kBFloat16 && x.size(1) == fc::C, "x [M, 256] bf16");
  const long M = x.size(0);
  TORCH_CHECK(M == B * L * L && L % fc::BM == 0 && H >= 1 && H <= 8, "M = B L L, L % 128 == 0, H <= 8");
  TORCH_CHECK(w16.is_contiguous() && w16.scalar_type() == torch::kHalf && w16.size(1) == fc::C && w16.size(0) % fc::NC == 0, "w16 [64 n, 256] fp16");
  TORCH_CHECK(w16.size(0) == 16 * fc::NC, "4 HD = 1024 (8 heads x 32)");
  TORCH_CHECK(wb16.is_contiguous() && wb16.scalar_type() == torch::kHalf && wb16.size(0) == 8 && wb16.size(1) == fc::C, "wb16 [8, 256] fp16");
  TORCH_CHECK(w.is_contiguous() && b.is_contiguous() && w.scalar_type() == torch::kFloat32 && b.scalar_type() == torch::kFloat32, "w, b fp32");
  const int nch = (int)(w16.size(0) / fc::NC);
  auto out = torch::empty({M, (long)nch * fc::NC}, x.options().dtype(torch::kHalf));
  auto sbias = torch::empty({B, H, L, L}, x.options().dtype(torch::kHalf));
  torch::Tensor mk;
  const bool* mp = nullptr;
  if (mask.numel() > 0) { mk = mask.to(torch::kBool).contiguous(); TORCH_CHECK(mk.numel() == B * L, "mask [B, L]"); mp = mk.data_ptr<bool>(); }
  const uint64_t C = fc::C;
  CUtensorMap xm = trans ? make_map<4>(x.data_ptr(), {C, (uint64_t)L, (uint64_t)L, (uint64_t)B}, {(uint64_t)L * C, C, (uint64_t)L * L * C},
                                       {64, 128, 1, 1}, CU_TENSOR_MAP_SWIZZLE_128B, "x")
                         : make_map<2>(x.data_ptr(), {C, (uint64_t)M}, {C}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B, "x");
  CUtensorMap wm = make_map<2>(w16.data_ptr(), {C, (uint64_t)w16.size(0)}, {C}, {64, 64}, CU_TENSOR_MAP_SWIZZLE_128B, "w16",
                               CU_TENSOR_MAP_DATA_TYPE_FLOAT16);
  CUtensorMap wbm = make_map<2>(wb16.data_ptr(), {C, 8}, {C}, {64, 8}, CU_TENSOR_MAP_SWIZZLE_128B, "wb16", CU_TENSOR_MAP_DATA_TYPE_FLOAT16);
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(front_c256_sm90<16, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, fc::SMEM));
    C10_CUDA_CHECK(cudaFuncSetAttribute(front_c256_sm90<16, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, fc::SMEM));
    attr = true;
  }
  const int ntiles = (int)(M / fc::BM);
  const int grid = std::min(ntiles, num_sms(x.device().index())) & ~1;     // clusters of two CTAs; ntiles is even (L % 128 == 0)
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid); cfg.blockDim = dim3(fc::THREADS); cfg.dynamicSmemBytes = fc::SMEM; cfg.stream = stream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 2; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = mc ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, mc ? front_c256_sm90<16, true> : front_c256_sm90<16, false>, (int)L, ntiles, (float)eps, (float)(1.0 / scale),
                                    (int)trans, (int)H, xm, wm, wbm, (const float*)w.data_ptr<float>(), (const float*)b.data_ptr<float>(), mp,
                                    reinterpret_cast<__half*>(out.data_ptr()), reinterpret_cast<__half*>(sbias.data_ptr())));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, sbias};
}

// qkvb [M, ld] bf16 -> sbias [B, H, L, L] bf16; mask [B, L] bool or empty; L % 128 == 0
torch::Tensor g_front_bias(torch::Tensor qkvb, torch::Tensor mask, int64_t B, int64_t L, int64_t H, int64_t bias_col, double scale) {
  TORCH_CHECK(L % 128 == 0, "L % 128 == 0");
  auto sbias = torch::empty({B, H, L, L}, qkvb.options());
  torch::Tensor mk;
  const bool* mp = nullptr;
  if (mask.numel() > 0) { mk = mask.to(torch::kBool).contiguous(); TORCH_CHECK(mk.numel() == B * L, "mask [B, L]"); mp = mk.data_ptr<bool>(); }
  if (qkvb.scalar_type() == torch::kHalf)          // fp16 GEMM output -> fp16 sbias (the fp16 inference core)
    front_bias_k<__half><<<(unsigned)(B * L * L / 128), 256, 0, stream()>>>(reinterpret_cast<const __half*>(qkvb.data_ptr()), mp,
        reinterpret_cast<__half*>(sbias.data_ptr()), qkvb.stride(0), (int)bias_col, (int)H, (int)L, (float)(1.0 / scale));
  else
    front_bias_k<bf16><<<(unsigned)(B * L * L / 128), 256, 0, stream()>>>(cbp(qkvb), mp, bp(sbias), qkvb.stride(0), (int)bias_col, (int)H, (int)L,
                                                                        (float)(1.0 / scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return sbias;
}

torch::Tensor g_gate_u(torch::Tensor qkvb, torch::Tensor o, int64_t g_col) {
  const long M = o.size(0);
  const int HD = (int)o.size(1);
  auto u = torch::empty_like(o);
  gate_u_k<<<grid_for(M * (HD / 8), 256), 256, 0, stream()>>>(cbp(qkvb), cbp(o), bp(u), M, HD, qkvb.stride(0), (int)g_col);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return u;
}

torch::Tensor g_residual(torch::Tensor x, torch::Tensor t, torch::Tensor drop, int64_t L, bool trans) {
  const long M = x.size(0);
  const int C = (int)x.size(1);
  auto out = torch::empty_like(x);
  residual_k<<<grid_for(M * (C / 8), 256), 256, 0, stream()>>>(cbp(x), cbp(t), drop.numel() > 0 ? cbp(drop) : nullptr, bp(out), M, C, (int)L, (int)trans);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor g_gate_bwd_in(torch::Tensor dy, torch::Tensor drop, int64_t L, bool trans) {
  const long M = dy.size(0);
  const int C = (int)dy.size(1);
  auto dd = torch::empty_like(dy);
  gate_bwd_in_k<<<grid_for(M * (C / 8), 256), 256, 0, stream()>>>(cbp(dy), drop.numel() > 0 ? cbp(drop) : nullptr, bp(dd), M, C, (int)L, (int)trans);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dd;
}

// du [M, HD], qkvg [M, ld] (g at columns g_col ..), o [M, HD] -> (do [M, HD], delta fp32 [B, L, H, L]); the dg columns of dall [M, ld_all]
// (g_col ..) are filled, the rest is written by g_assemble
std::vector<torch::Tensor> g_gate_bwd_out(torch::Tensor du, torch::Tensor qkvb, torch::Tensor o, torch::Tensor dall, int64_t g_col, int64_t H,
                                          int64_t L) {
  const long M = du.size(0);
  const int HD = (int)du.size(1);
  auto dout = torch::empty_like(du);
  auto delta = torch::empty({M / L / L, L, H, L}, du.options().dtype(torch::kFloat32));
  gate_bwd_out_k<<<grid_for(M * (HD / 8), 256), 256, 0, stream()>>>(cbp(du), cbp(qkvb), cbp(o), bp(dout), bp(dall), delta.data_ptr<float>(), M, HD,
                                                                    qkvb.stride(0), (int)g_col, dall.stride(0), (int)H, (int)L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dout, delta};
}

void g_assemble(torch::Tensor dbias, torch::Tensor dall, int64_t H, int64_t L, int64_t HD) {
  const long M = dall.size(0);
  const int Hp = (int)(dall.size(1) - 4 * HD);
  TORCH_CHECK(Hp % 8 == 0 && Hp >= H && dall.stride(0) == dall.size(1), "dall [M, 4 HD + Hp]");
  assemble_k<<<grid_for(M * (Hp / 8), 256), 256, 0, stream()>>>(dbias.data_ptr<float>(), bp(dall), M, (int)HD, dall.stride(0), (int)H, Hp, (int)L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// -> (dx bf16 [M, C], partial sums fp32 [grid, 2, C])
std::vector<torch::Tensor> g_ln_bwd(torch::Tensor dyln, torch::Tensor x, torch::Tensor dy, torch::Tensor w, torch::Tensor mean, torch::Tensor rstd,
                                    int64_t L, bool trans) {
  const long M = x.size(0);
  const int C = (int)x.size(1);
  TORCH_CHECK((dyln.scalar_type() == torch::kFloat32 || dyln.scalar_type() == torch::kBFloat16) && dyln.is_contiguous() && C % 8 == 0 && C <= 256 * MAXV,
              "dyln fp32 or bf16 [M, C]");
  const int lpr = C == 64 ? 8 : C == 128 ? 16 : 32;
  const int grid = grid_for(M, 8 * (32 / lpr));
  auto dx = torch::empty_like(x);
  auto part = torch::empty({grid, 2, C}, x.options().dtype(torch::kFloat32));
  auto launch = [&](auto* dptr) {
    using TD = std::remove_cv_t<std::remove_pointer_t<decltype(dptr)>>;
    auto args = std::make_tuple(dptr, cbp(x), cbp(dy), w.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(), bp(dx), part.data_ptr<float>(), M, C, (int)L, (int)trans);
    if (lpr == 8) std::apply([&](auto... a) { ln_bwd_k<8, TD><<<grid, 256, 0, stream()>>>(a...); }, args);
    else if (lpr == 16) std::apply([&](auto... a) { ln_bwd_k<16, TD><<<grid, 256, 0, stream()>>>(a...); }, args);
    else std::apply([&](auto... a) { ln_bwd_k<32, TD><<<grid, 256, 0, stream()>>>(a...); }, args);
  };
  if (dyln.scalar_type() == torch::kFloat32) launch(dyln.data_ptr<float>());
  else launch(cbp(dyln));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dx, part};
}

// ---- weight pack: [Wq; Wk; Wv; Wg; Wb] (heads padded to 32 channels, Wb rows zero-padded to 64) and Wo [C, HD] in one launch
template <typename T> __device__ __forceinline__ bf16 to_bf(T v);
template <> __device__ __forceinline__ bf16 to_bf<float>(float v) { return __float2bfloat16_rn(v); }
template <> __device__ __forceinline__ bf16 to_bf<bf16>(bf16 v) { return v; }
template <typename T>
__global__ void __launch_bounds__(256) pack_generic_k(const T* __restrict__ wq, const T* __restrict__ wk, const T* __restrict__ wv, const T* __restrict__ wg,
                                                      const T* __restrict__ wb, const T* __restrict__ wo, bf16* __restrict__ wall, bf16* __restrict__ wop,
                                                      int C, int H, int hd) {
  const long HD = 32L * H, n1 = (4 * HD + 64) * C, n = n1 + (long)C * HD;
  for (long i = (long)blockIdx.x * 256 + threadIdx.x; i < n; i += (long)gridDim.x * 256) {
    bf16 v = __float2bfloat16_rn(0.f);
    if (i < n1) {
      const long row = i / C; const int col = (int)(i - row * C);
      if (row < 4 * HD) {
        const int part = (int)(row / HD), r = (int)(row - part * HD), h = r >> 5, d = r & 31;
        if (d < hd) {
          const T* w = part == 0 ? wq : part == 1 ? wk : part == 2 ? wv : wg;
          v = to_bf<T>(w[(long)(h * hd + d) * C + col]);
        }
      } else if (row - 4 * HD < H) {
        v = to_bf<T>(wb[(row - 4 * HD) * C + col]);
      }
      wall[i] = v;
    } else {
      const long j = i - n1, c = j / HD; const int r = (int)(j - c * HD), h = r >> 5, d = r & 31;
      if (d < hd) v = to_bf<T>(wo[c * (long)H * hd + h * hd + d]);
      wop[j] = v;
    }
  }
}

// -> (wall [4 HD + 64, C] bf16, wo_p [C, HD] bf16); weights fp32 or bf16, contiguous: wq .. wg [H hd, C], wb [H, C], wo [C, H hd]
std::vector<torch::Tensor> g_pack(torch::Tensor wq, torch::Tensor wk, torch::Tensor wv, torch::Tensor wg, torch::Tensor wb, torch::Tensor wo) {
  const int H = (int)wb.size(0), C = (int)wq.size(1), hd = (int)(wq.size(0) / H);
  TORCH_CHECK(hd >= 1 && hd <= 32 && wq.size(0) == (long)H * hd, "heads of at most 32 channels");
  for (const auto* t : {&wq, &wk, &wv, &wg, &wb, &wo}) TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->scalar_type() == wq.scalar_type(), "contiguous, one dtype");
  const long HD = 32L * H;
  auto wall = torch::empty({4 * HD + 64, C}, wq.options().dtype(torch::kBFloat16));
  auto wop = torch::empty({C, HD}, wq.options().dtype(torch::kBFloat16));
  const long n = (4 * HD + 64) * C + (long)C * HD;
  const int grid = (int)std::min<long>((n + 255) / 256, 1024);
  if (wq.scalar_type() == torch::kFloat32)
    pack_generic_k<float><<<grid, 256, 0, stream()>>>(wq.data_ptr<float>(), wk.data_ptr<float>(), wv.data_ptr<float>(), wg.data_ptr<float>(),
                                                      wb.data_ptr<float>(), wo.data_ptr<float>(), bp(wall), bp(wop), C, H, hd);
  else {
    TORCH_CHECK(wq.scalar_type() == torch::kBFloat16, "fp32 or bf16 weights");
    pack_generic_k<bf16><<<grid, 256, 0, stream()>>>(cbp(wq), cbp(wk), cbp(wv), cbp(wg), cbp(wb), cbp(wo), bp(wall), bp(wop), C, H, hd);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {wall, wop};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ln_fwd", &g_ln_fwd, py::arg("x"), py::arg("w"), py::arg("b"), py::arg("eps"), py::arg("L"), py::arg("trans"), py::arg("h16") = false);
  m.def("ln_bias_fwd", &g_ln_bias_fwd, py::arg("x"), py::arg("w"), py::arg("b"), py::arg("wb"), py::arg("mask"), py::arg("eps"), py::arg("B"),
        py::arg("L"), py::arg("H"), py::arg("trans"), py::arg("scale"));
  m.def("front_c256", &g_front_c256, py::arg("x"), py::arg("w"), py::arg("b"), py::arg("w16"), py::arg("wb16"), py::arg("mask"), py::arg("eps"),
        py::arg("B"), py::arg("L"), py::arg("H"), py::arg("trans"), py::arg("scale"), py::arg("mc") = true);
  m.def("front_bias", &g_front_bias);
  m.def("gate_u", &g_gate_u);
  m.def("residual", &g_residual);
  m.def("gate_bwd_in", &g_gate_bwd_in);
  m.def("gate_bwd_out", &g_gate_bwd_out);
  m.def("assemble", &g_assemble);
  m.def("ln_bwd", &g_ln_bwd);
  m.def("pack", &g_pack);
}
