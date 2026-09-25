// OPM fused kernels for B200 (sm_100a).  Same fusion algorithm and the same tensors / layouts as the
// H100 `opm_epilogue.cu`; the tensor-core work moves from wgmma to tcgen05 (TMEM accumulators, one
// issuing thread) and every operand tile is fetched by TMA.
//
//   opm_epilogue:  z[i,j,:] = (O[(i,c),(j,e)] reshaped to [(i,j),(c,e)]) . Wo^T / n[i,j] + bias (+ residual)
//
// The epilogue is a GEMM with M = 128 (i,j) pairs per tile, N = c_z = 128, K = c_hidden^2 = 1024.
// The [i,j,c,e] -> [(i,j),(c,e)] permute is NOT done by threads: a 4-D TMA box over O with the dims
// ordered (e, j, i, c) lands a k-chunk of two c values as two [(i,j) rows][32 e] tiles -- each exactly
// the canonical 64-byte-swizzled K-major operand -- because TMA does not require monotonic strides.
// (A 128-byte-swizzled box whose inner dim is 64 bytes faults on sm_100, so the two c values of a
// chunk are separate k-blocks of 32 rather than one 128-byte row.)
//
// Persistent, warp-specialized: warp 0 streams (O tile, Wo chunk) stages by TMA, warp 1 issues the
// tcgen05.mma chain into one of two TMEM accumulators, warps 2..5 drain the other accumulator
// (/ n, + bias, bf16, + residual) through a swizzled staging tile and one TMA store per tile.
#include <torch/extension.h>
#include "sm100.cuh"

using namespace sm100;

namespace {
constexpr int CH = 32, CZ = 128, NCH = CH * CH;

namespace ep {
constexpr int BI = 4, BJ = 32, NP = BI * BJ;            // 128 pairs per tile
constexpr int KC = 64;                                  // k per stage = 2 c values
constexpr int NK = NCH / KC;                            // 16 stages per tile
constexpr int NST = 5;
constexpr int ATILE = NP * KC, BTILE = CZ * KC;         // elements
constexpr int STAGE = ATILE + BTILE;
constexpr int OUTT = NP * CZ;                           // staging: [BI][2 halves][BJ][64]
constexpr int THREADS = 192;
constexpr int SMEM = 1024 + (NST * STAGE + OUTT) * 2 + 256;
constexpr uint32_t IDESC = idesc_bf16(128, CZ);
}  // namespace ep

__global__ void __launch_bounds__(ep::THREADS, 1) opm_epilogue_sm100(
    int NI, int NJ, int ntiles,
    const __grid_constant__ CUtensorMap amap,     // O as (e 32, j NJ, i NI, c 32), box (32, BJ, BI, 2), 64B swizzle
    const __grid_constant__ CUtensorMap bmap,     // Wo [CZ][NCH], box (64, 128)
    const __grid_constant__ CUtensorMap zmap,     // z as (64 z, NJ j, 2 halves, NI i), box (64, BJ, 2, BI)
    const __grid_constant__ CUtensorMap rmap,     // the residual, same layout
    const float* __restrict__ NORM,               // [NI][NJ] mask counts (clamped >= 1), or nullptr: counted from BITS
    const uint32_t* __restrict__ BITS, int W,     // the prologue's bit mask [N][W = S/32]
    const void* __restrict__ BIAS, int bias_bf16, int has_res) {
  using namespace ep;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sStage = reinterpret_cast<__nv_bfloat16*>(smb);
  __nv_bfloat16* sOut = sStage + NST * STAGE;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sOut + OUTT);
  uint64_t* full = bars;            // [NST]
  uint64_t* empty = full + NST;     // [NST]
  uint64_t* accf = empty + NST;     // [2]
  uint64_t* acce = accf + 2;        // [2]
  uint64_t* resf = acce + 2;        // [1]
  uint32_t* tslot = reinterpret_cast<uint32_t*>(resf + 1);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { bar_init(full + s, 1); bar_init(empty + s, 1); }
    for (int b = 0; b < 2; ++b) { bar_init(accf + b, 1); bar_init(acce + b, 4); }
    bar_init(resf, 1);
    bar_init_fence();
    prefetch_map(&amap); prefetch_map(&bmap); prefetch_map(&zmap); if (has_res) prefetch_map(&rmap);
  }
  if (warp == 1) tmem_alloc(tslot, 256);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  const int njb = NJ / BJ;

  if (warp == 0) {
    if (lane == 0) {
      int g = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
        const int i0 = (t / njb) * BI, j0 = (t % njb) * BJ;
        for (int kc = 0; kc < NK; ++kc, ++g) {
          const int s = g % NST;
          if (g >= NST) wait(empty + s, ((g / NST) - 1) & 1);
          __nv_bfloat16* st = sStage + s * STAGE;
          expect_tx(full + s, STAGE * 2);
          load_4d(&amap, st, full + s, 0, j0, i0, kc * 2);
          load_2d(&bmap, st + ATILE, full + s, kc * KC, 0);
        }
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      int g = 0, lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int b = lt & 1;
        if (lt >= 2) wait(acce + b, ((lt >> 1) - 1) & 1);
        tc_fence_after();
        const uint32_t d = tmem + b * CZ;
        for (int kc = 0; kc < NK; ++kc, ++g) {
          const int s = g % NST;
          wait(full + s, (g / NST) & 1);
          tc_fence_after();
          const __nv_bfloat16* st = sStage + s * STAGE;
#pragma unroll
          for (int ks = 0; ks < KC / 16; ++ks)
            mma_ss(d, desc_k64(st + (ks >> 1) * (NP * 32) + (ks & 1) * 16), desc_k128(st + ATILE + ks * 16), IDESC, (kc | ks) ? 1u : 0u);
          mma_commit(empty + s);
        }
        mma_commit(accf + b);
      }
    }
  } else {
    // epilogue: warp w owns TMEM lanes 32 * (w % 4) .. +31 = tile rows (pairs) p
    const int q = warp & 3;
    const int p = q * 32 + lane;
    const int il = p / BJ, jl = p % BJ;
    const bool leader = (warp == 2 && lane == 0);
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int i0 = (t / njb) * BI, j0 = (t % njb) * BJ;
      const int b = lt & 1;
      if (leader) {
        bulk_wait_read<0>();                     // the previous tile's store has finished reading sOut
        if (has_res) {
          expect_tx(resf, OUTT * 2);
          load_4d(&rmap, sOut, resf, 0, j0, 0, i0);
        }
      }
      // the mask count of this thread's pair, n = max(1, popc(bits_i & bits_j)): exact, and it replaces a
      // separate [N, N] count kernel (its loads go out before the accumulator wait, under the main loop)
      float inv;
      if (NORM != nullptr) {
        inv = 1.0f / __ldg(NORM + (size_t)(i0 + il) * NJ + (j0 + jl));
      } else {
        const uint4* bi = reinterpret_cast<const uint4*>(BITS + (size_t)(i0 + il) * W);
        const uint4* bj = reinterpret_cast<const uint4*>(BITS + (size_t)(j0 + jl) * W);
        int cnt = 0;
        for (int w = 0; w < W / 4; ++w) {
          const uint4 a = __ldg(bi + w), c = __ldg(bj + w);
          cnt += __popc(a.x & c.x) + __popc(a.y & c.y) + __popc(a.z & c.z) + __popc(a.w & c.w);
        }
        inv = 1.0f / (float)max(cnt, 1);
      }
      wait(accf + b, (lt >> 1) & 1);
      tc_fence_after();
      if (has_res) wait(resf, lt & 1);
      named_sync(1, 128);                        // (no residual) every warp sees the previous store retired
      const uint32_t tb = tmem_at(tmem + b * CZ, q * 32, 0);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float v[64];
        tmem_ld32(tb + h * 64, v);
        tmem_ld32(tb + h * 64 + 32, v + 32);
        tmem_wait_ld();
        if (h == 1) { tc_fence_before(); if (lane == 0) arrive(acce + b); }   // accumulator drained
        __nv_bfloat16* row = sOut + ((il * 2 + h) * BJ + jl) * 64;
#pragma unroll
        for (int c8 = 0; c8 < 8; ++c8) {
          const int sw = (c8 ^ (jl & 7)) << 3;
          uint4 o;
          uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
          uint4 r = make_uint4(0, 0, 0, 0);
          if (has_res) r = *reinterpret_cast<const uint4*>(row + sw);
          const __nv_bfloat162* rh = reinterpret_cast<const __nv_bfloat162*>(&r);
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const int n = h * 64 + c8 * 8 + u * 2;
            // the stock module's bias is added in fp32 after a bf16 rounding (the cuBLASLt epilogue): a bf16 bias is exact
            const float2 bb = bias_bf16 ? __bfloat1622float2(__ldg(reinterpret_cast<const __nv_bfloat162*>(BIAS) + (n >> 1)))
                                        : __ldg(reinterpret_cast<const float2*>(BIAS) + (n >> 1));
            __nv_bfloat162 x = __float22bfloat162_rn(make_float2(fmaf(v[c8 * 8 + u * 2], inv, bb.x), fmaf(v[c8 * 8 + u * 2 + 1], inv, bb.y)));
            if (has_res) x = __hadd2(x, rh[u]);  // round the update first: bf16 OPM + bf16 residual
            ow[u] = *reinterpret_cast<uint32_t*>(&x);
          }
          *reinterpret_cast<uint4*>(row + sw) = o;
        }
      }
      fence_proxy_async();
      named_sync(1, 128);
      if (leader) { store_4d(&zmap, sOut, 0, j0, 0, i0); bulk_commit(); }
    }
    if (leader) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 256);
}


// ---------------------------------------------------------------------------------------------
// PROLOGUE: y = LN(m[s,i,:]) -> bf16, [a | b] = y . [Wa; Wb]^T, -> bf16, * mask, written straight into
// the grouped-GEMM layouts A2[(i,c), s] and BT[(j,e), s] (s contiguous).  The same fusion as the Triton
// prologue.  One tile = one token i x 128 MSA rows; tile t = (token t / NSB, s block t % NSB).
//
// The projection runs TRANSPOSED on the tensor core: D[ch][s] = W[ch][:] . y[s][:], so TMEM lane = output
// channel and a thread drains CONTIGUOUS s values of one channel -- the orientation A2 / BT want --
// straight into a 128-byte-swizzled staging tile with 16-byte stores.  M must be 128, so the 64 stacked
// channels (Wa rows, then Wb rows) appear twice: the second copy lets two more warps drain the second s half.
// The mask is applied to y, not to the projection: a masked MSA row is written as a zero row of y, so its
// column of D is exactly 0 -- bf16(a) * 0 -- and the drain is a plain convert-and-store.  (The one
// difference is the sign of a zero, which no later product or sum can see.)  The LayerNorm statistics
// (training) go out as [N][S] (mean, rstd), s-contiguous so the stores coalesce; the mask leaves as a bit
// mask [N][S/32], which is all the mask count needs.
//
// The LayerNorm is latency-bound, so every role is its own warp and nothing waits on a CTA-wide barrier:
// warp 0 streams x tiles (and their mask bytes) by TMA; warp 1 issues the GEMMs; warps 2-9 run the LayerNorm
// in place, TWO threads per row (a half-warp pair combined by one shuffle: half the serial chain per row and
// twice the rows in flight); y overwrites x in its stage, released by the GEMM's commit, and the LN warps
// report y-ready on an mbarrier; warps 10-13 drain the two TMEM accumulators through two staging tiles and
// store them by TMA.
namespace pro {
constexpr int CM = 64, BS = 128;
constexpr int NST = 3;
constexpr int NACC = 2;                              // TMEM accumulators (and staging tiles) per CTA
constexpr int CTAS = 2;                              // CTAs per SM: shared memory, TMEM (CTAS x NACC x 128 <= 512 columns)
constexpr int XT = BS * CM;                          // one x / y tile, bf16 elements (16 KiB)
constexpr int TT = 2 * 64 * 64;                      // staging [2 s halves][64 channels][64 s], swizzled
constexpr int THREADS = 448;
constexpr int MT = BS * 16;                          // the tile's mask bytes: [128 s][16 tokens] (TMA's 16-byte minimum row)
constexpr int SMEM = 1024 + (NST * XT + NACC * TT + 128 * CM) * 2 + NST * MT + 2 * CM * 4 + 512;
constexpr int TMEM_COLS = NACC * 128;
constexpr uint32_t IDESC = idesc_bf16(128, BS);
static_assert(CTAS * (SMEM + 1024) <= 233472, "CTAS per SM (each also pays the 1 KiB per-block reserve)");
static_assert(CTAS * TMEM_COLS <= 512, "TMEM columns");
}  // namespace pro

template <typename WT>
__global__ void __launch_bounds__(pro::THREADS, pro::CTAS) opm_prologue_sm100(
    int N, int S, int ntiles, float eps,
    const __grid_constant__ CUtensorMap xmap,      // m [S][N][64]: (64, N, S), box (64, 1, 128), 128B swizzle
    const __grid_constant__ CUtensorMap amap,      // A2 [N*32][S], box (64, 32), 128B swizzle
    const __grid_constant__ CUtensorMap bmap,      // BT [N*32][S], box (64, 32), 128B swizzle
    const __grid_constant__ CUtensorMap mmap,      // mask [S][N] bytes, box (16, 128)
    const __grid_constant__ CUtensorMap wamap,     // Wa [32][64], box (64, 32), 128B swizzle
    const __grid_constant__ CUtensorMap wbmap,     // Wb [32][64]
    const WT* __restrict__ LNW, const WT* __restrict__ LNB,
    float2* __restrict__ STATS,                    // [N][S] (mean, rstd) or nullptr
    uint32_t* __restrict__ BITS) {                 // [N][S/32]
  using namespace pro;
  const int NSB = S / BS;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sX = reinterpret_cast<__nv_bfloat16*>(smb);   // [NST][128 s][64] swizzled: x, then y in place (MMA B operand)
  __nv_bfloat16* sT = sX + NST * XT;                             // [NACC][2 halves][64 ch][64 s] swizzled
  __nv_bfloat16* sW = sT + NACC * TT;                               // [128 m][64 k] swizzled: Wa, Wb, Wa, Wb (32 rows each)
  uint8_t* sM = reinterpret_cast<uint8_t*>(sW + 128 * CM);        // [NST][128 s][16 tokens] mask bytes
  float* sLN = reinterpret_cast<float*>(sM + NST * MT);          // gamma[64], beta[64]
  uint64_t* bars = reinterpret_cast<uint64_t*>(sLN + 2 * CM);
  uint64_t* xfull = bars;              // [NST]  TMA landed
  uint64_t* yfull = xfull + NST;       // [NST]  count 8: the LN warps wrote y
  uint64_t* xempty = yfull + NST;      // [NST]  the GEMM's commit: y has been read
  uint64_t* accf = xempty + NST;       // [NACC]
  uint64_t* acce = accf + NACC;        // [NACC] count 4: the epilogue warps drained it
  uint64_t* wful = acce + NACC;        // [1]    the stacked weight landed
  uint32_t* tslot = reinterpret_cast<uint32_t*>(wful + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { bar_init(xfull + s, 1); bar_init(yfull + s, 8); bar_init(xempty + s, 1); }
    for (int b = 0; b < NACC; ++b) { bar_init(accf + b, 1); bar_init(acce + b, 4); }
    bar_init(wful, 1);
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, TMEM_COLS);
  if (tid < CM) { sLN[tid] = to_f(LNW[tid]); sLN[CM + tid] = to_f(LNB[tid]); }
  fence_proxy_async();
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;

  if (warp == 0) {
    if (lane == 0) {
      expect_tx(wful, 128 * CM * 2);             // the stacked weight: Wa, Wb, Wa, Wb, by TMA
      for (int k = 0; k < 4; ++k) load_2d((k & 1) ? &wbmap : &wamap, sW + k * 32 * 64, wful, 0, 0);
      int g = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++g) {
        const int i = t / NSB, s0 = (t % NSB) * BS;
        const int st = g % NST;
        if (g >= NST) wait(xempty + st, ((g / NST) - 1) & 1);
        expect_tx(xfull + st, XT * 2 + MT);
        load_3d(&xmap, sX + st * XT, xfull + st, 0, i, s0);
        load_2d(&mmap, sM + st * MT, xfull + st, i & ~15, s0);
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {                             // GEMM issue: y of tile lt ready, accumulator lt & 1 drained
      wait(wful, 0);
      int lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int st = lt % NST, b = lt % NACC;
        wait(yfull + st, (lt / NST) & 1);
        if (lt >= NACC) wait(acce + b, ((lt / NACC) - 1) & 1);
        tc_fence_after();
        const __nv_bfloat16* y = sX + st * XT;
#pragma unroll
        for (int ks = 0; ks < CM / 16; ++ks)
          mma_ss(tmem + b * BS, desc_k128(sW + ks * 16), desc_k128(y + ks * 16), IDESC, ks ? 1u : 0u);
        mma_commit(accf + b);
        mma_commit(xempty + st);
      }
    }
  } else if (warp <= 9) {
    // ---------------- LayerNorm: row r, channel half hf (lanes 0-15 / 16-31 of a warp share 16 rows) ----------------
    const int r = (warp - 2) * 16 + (lane & 15), hf = lane >> 4;
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int i = t / NSB, s0 = (t % NSB) * BS, st = lt % NST;
      wait(xfull + st, (lt / NST) & 1);
      const bool mkb = sM[st * MT + r * 16 + (i & 15)] != 0;   // the mask came with x: a global byte load here
                                                               // took microseconds behind the tile's traffic
      __nv_bfloat16* xr = sX + st * XT + r * 64;
      float2 x[16];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int c8 = hf * 4 + k;
        const uint4 u = *reinterpret_cast<const uint4*>(xr + ((c8 ^ (r & 7)) << 3));
        x[k * 4 + 0] = bf2f(u.x); x[k * 4 + 1] = bf2f(u.y); x[k * 4 + 2] = bf2f(u.z); x[k * 4 + 3] = bf2f(u.w);
      }
      float2 sa = make_float2(0.f, 0.f), sb = sa;
#pragma unroll
      for (int k = 0; k < 16; k += 2) { sa = add2(sa, x[k]); sb = add2(sb, x[k + 1]); }
      const float2 sm = add2(sa, sb);
      float sum = sm.x + sm.y;
      sum += __shfl_xor_sync(0xffffffffu, sum, 16);
      const float mean = sum * (1.f / CM);
      const float2 nm = make_float2(-mean, -mean);
      float2 va = make_float2(0.f, 0.f), vb = va;
#pragma unroll
      for (int k = 0; k < 16; k += 2) {
        x[k] = add2(x[k], nm); x[k + 1] = add2(x[k + 1], nm);
        va = fma2(x[k], x[k], va); vb = fma2(x[k + 1], x[k + 1], vb);
      }
      const float2 vs = add2(va, vb);
      float var = vs.x + vs.y;
      var += __shfl_xor_sync(0xffffffffu, var, 16);
      const float rstd = 1.f / sqrtf(var * (1.f / CM) + eps);
      const float2 rs = make_float2(rstd, rstd);
      // y = (x - mean) * rstd * gamma + beta -> bf16 over this thread's own half row; a masked row becomes zeros
      // (an AND on the packed words: a branch here serialised the loop behind the gamma / beta loads)
      const uint32_t keep = mkb ? 0xffffffffu : 0u;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int c8 = hf * 4 + k;
        const float4 g0 = *reinterpret_cast<const float4*>(sLN + c8 * 8), g1 = *reinterpret_cast<const float4*>(sLN + c8 * 8 + 4);
        const float4 b0 = *reinterpret_cast<const float4*>(sLN + CM + c8 * 8), b1 = *reinterpret_cast<const float4*>(sLN + CM + c8 * 8 + 4);
        const float2 y0 = fma2(mul2(x[k * 4 + 0], rs), make_float2(g0.x, g0.y), make_float2(b0.x, b0.y));
        const float2 y1 = fma2(mul2(x[k * 4 + 1], rs), make_float2(g0.z, g0.w), make_float2(b0.z, b0.w));
        const float2 y2 = fma2(mul2(x[k * 4 + 2], rs), make_float2(g1.x, g1.y), make_float2(b1.x, b1.y));
        const float2 y3 = fma2(mul2(x[k * 4 + 3], rs), make_float2(g1.z, g1.w), make_float2(b1.z, b1.w));
        uint4 o;
        o.x = pack2(y0.x, y0.y) & keep; o.y = pack2(y1.x, y1.y) & keep; o.z = pack2(y2.x, y2.y) & keep; o.w = pack2(y3.x, y3.y) & keep;
        *reinterpret_cast<uint4*>(xr + ((c8 ^ (r & 7)) << 3)) = o;
      }
      if (STATS != nullptr && hf == 0) STATS[(size_t)i * S + s0 + r] = make_float2(mean, rstd);
      const unsigned bal = __ballot_sync(0xffffffffu, mkb);    // lanes 0-15: this warp's 16 rows
      if (lane == 0) reinterpret_cast<uint16_t*>(BITS)[((size_t)i * S + s0 + r) / 16] = (uint16_t)(bal & 0xffffu);
      fence_proxy_async();                       // y (ordinary stores) -> the tensor core's async proxy
      __syncwarp();
      if (lane == 0) arrive(yfull + st);
    }
  } else {
    // ---------------- drain (barrier 2): TMEM lane = channel; q & 1 picks a | b, q >> 1 the s half ----------------
    const int q = warp & 3;
    const int ch = (q & 1) * 32 + lane, sh = q >> 1;
    const bool leader = (warp == 10 && lane == 0);
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int i = t / NSB, s0 = (t % NSB) * BS, b = lt % NACC;
      wait(accf + b, (lt / NACC) & 1);
      tc_fence_after();
      if (leader) bulk_wait_read<NACC - 1>();    // the stores of tile lt - NACC have left staging buffer b
      named_sync(2, 128);
      __nv_bfloat16* row = sT + b * TT + (sh * 64 + ch) * 64;
#pragma unroll
      for (int h = 0; h < 2; ++h) {              // 32 columns = 32 s at a time
        float acc[32];
        tmem_ld32(tmem_at(tmem + b * BS, q * 32, sh * 64 + h * 32), acc);
        tmem_wait_ld();
        if (h == 1) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(acce + b); }
#pragma unroll
        for (int c8 = 0; c8 < 4; ++c8) {
          uint4 o;
          o.x = pack2(acc[c8 * 8 + 0], acc[c8 * 8 + 1]); o.y = pack2(acc[c8 * 8 + 2], acc[c8 * 8 + 3]);
          o.z = pack2(acc[c8 * 8 + 4], acc[c8 * 8 + 5]); o.w = pack2(acc[c8 * 8 + 6], acc[c8 * 8 + 7]);
          *reinterpret_cast<uint4*>(row + (((h * 4 + c8) ^ (ch & 7)) << 3)) = o;
        }
      }
      fence_proxy_async();
      named_sync(2, 128);
      if (leader) {
        const __nv_bfloat16* tb = sT + b * TT;
        for (int hh = 0; hh < 2; ++hh) {
          store_2d(&amap, tb + hh * 64 * 64, s0 + hh * 64, i * CH);
          store_2d(&bmap, tb + hh * 64 * 64 + CH * 64, s0 + hh * 64, i * CH);
        }
        bulk_commit();
      }
    }
    if (leader) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, TMEM_COLS);
}

// the mask count n[i,j] = max(1, sum_s mask[s,i] mask[s,j]) from the prologue's bit mask, fp32 (exact)
__global__ void __launch_bounds__(256) opm_norm_sm100(const uint32_t* __restrict__ BITS, float* __restrict__ NORM, int N, int W) {
  extern __shared__ uint32_t nb[];                 // [32 i][W + 1] then [32 j][W + 1]: the pad spreads a column over the banks
  const int WP = W + 1;
  const int i0 = blockIdx.y * 32, j0 = blockIdx.x * 32;
  for (int v = threadIdx.x; v < 32 * W; v += 256) {
    const int t = v / W, w = v - t * W;
    nb[t * WP + w] = BITS[(size_t)i0 * W + v];
    nb[(32 + t) * WP + w] = BITS[(size_t)j0 * W + v];
  }
  __syncthreads();
  const int jl = threadIdx.x & 31, ib = threadIdx.x >> 5;
  const uint32_t* bj = nb + (32 + jl) * WP;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int il = ib * 4 + k;
    const uint32_t* bi = nb + il * WP;
    int c = 0;
    for (int w = 0; w < W; ++w) c += __popc(bi[w] & bj[w]);
    NORM[(size_t)(i0 + il) * N + j0 + jl] = (float)max(c, 1);
  }
}
}  // namespace

// `norm`: fp32 [ni, nj] mask counts, or int32 [N, S/32] -- the prologue's bit mask, counted per pair in-kernel
torch::Tensor opm_epilogue(torch::Tensor O, torch::Tensor norm, torch::Tensor wo, torch::Tensor bias,
                           int64_t ni, int64_t nj, c10::optional<torch::Tensor> residual) {
  using namespace ep;
  TORCH_CHECK(O.is_cuda() && O.scalar_type() == torch::kBFloat16 && O.is_contiguous(), "O: contiguous cuda bf16");
  TORCH_CHECK(O.size(0) == ni * CH && O.size(1) == nj * CH, "O: [ni*32, nj*32]");
  const bool bits = norm.scalar_type() == torch::kInt32;
  TORCH_CHECK(norm.is_contiguous() && (bits ? (norm.dim() == 2 && norm.size(0) >= std::max(ni, nj) && norm.size(1) % 4 == 0)
                                            : (norm.scalar_type() == torch::kFloat32 && norm.numel() == ni * nj)),
              "norm: fp32 [ni, nj] counts, or the int32 [N, S/32] bit mask");
  TORCH_CHECK(wo.scalar_type() == torch::kBFloat16 && wo.is_contiguous() && wo.numel() == CZ * NCH, "wo: bf16 [128, 1024]");
  TORCH_CHECK((bias.scalar_type() == torch::kFloat32 || bias.scalar_type() == torch::kBFloat16) && bias.is_contiguous() && bias.numel() == CZ,
              "bias: fp32 or bf16 [128]");
  TORCH_CHECK(ni % BI == 0 && nj % BJ == 0, "the epilogue tiles ", BI, " x ", BJ, " tokens");
  const bool has_res = residual.has_value() && residual->numel() > 0;
  auto out = torch::empty({1, ni, nj, (long)CZ}, O.options());
  if (has_res)
    TORCH_CHECK(residual->scalar_type() == torch::kBFloat16 && residual->is_contiguous() && residual->numel() == ni * nj * CZ, "residual: bf16 [1, ni, nj, 128]");
  const uint64_t M = (uint64_t)nj * CH;
  CUtensorMap am = make_map<4>(O.data_ptr(), {32, (uint64_t)nj, (uint64_t)ni, 32}, {32, 32 * M, M}, {32, BJ, BI, 2}, CU_TENSOR_MAP_SWIZZLE_64B, "O");
  CUtensorMap bm = make_map<2>(wo.data_ptr(), {(uint64_t)NCH, (uint64_t)CZ}, {(uint64_t)NCH}, {64, CZ}, CU_TENSOR_MAP_SWIZZLE_128B, "Wo");
  CUtensorMap zm = make_map<4>(out.data_ptr(), {64, (uint64_t)nj, 2, (uint64_t)ni}, {(uint64_t)CZ, 64, (uint64_t)nj * CZ}, {64, BJ, 2, BI}, CU_TENSOR_MAP_SWIZZLE_128B, "z");
  CUtensorMap rm = has_res ? make_map<4>(residual->data_ptr(), {64, (uint64_t)nj, 2, (uint64_t)ni}, {(uint64_t)CZ, 64, (uint64_t)nj * CZ}, {64, BJ, 2, BI}, CU_TENSOR_MAP_SWIZZLE_128B, "res") : zm;
  const int ntiles = (int)((ni / BI) * (nj / BJ));
  const int grid = std::min(ntiles, num_sms(O.device().index()));
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(opm_epilogue_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
  opm_epilogue_sm100<<<grid, THREADS, SMEM, at::cuda::getCurrentCUDAStream()>>>(
      (int)ni, (int)nj, ntiles, am, bm, zm, rm, bits ? nullptr : norm.data_ptr<float>(),
      bits ? reinterpret_cast<const uint32_t*>(norm.data_ptr<int>()) : nullptr, bits ? (int)norm.size(1) : 0,
      bias.data_ptr(), bias.scalar_type() == torch::kBFloat16 ? 1 : 0, has_res ? 1 : 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> opm_prologue(torch::Tensor m, torch::Tensor mask, torch::Tensor lnw, torch::Tensor lnb, double eps,
                                        torch::Tensor wa, torch::Tensor wb, bool save_stats, bool want_norm) {
  using namespace pro;
  TORCH_CHECK(m.is_cuda() && m.scalar_type() == torch::kBFloat16 && m.is_contiguous() && m.dim() == 3 && m.size(2) == CM, "m: [S, N, 64] bf16");
  const long S = m.size(0), N = m.size(1);
  TORCH_CHECK(S % BS == 0, "S must be a multiple of ", BS);
  TORCH_CHECK(N % 32 == 0, "N must be a multiple of 32");
  TORCH_CHECK(mask.scalar_type() == torch::kBool && mask.is_contiguous() && mask.numel() == S * N, "mask: bool [S, N]");
  TORCH_CHECK(wa.scalar_type() == torch::kBFloat16 && wa.is_contiguous() && wa.numel() == CH * CM, "wa: bf16 [32, 64]");
  TORCH_CHECK(wb.scalar_type() == torch::kBFloat16 && wb.is_contiguous() && wb.numel() == CH * CM, "wb: bf16 [32, 64]");
  TORCH_CHECK(lnw.scalar_type() == lnb.scalar_type() && lnw.is_contiguous() && lnb.is_contiguous(), "LayerNorm affine");
  auto A2 = torch::empty({N * CH, S}, m.options());
  auto BT = torch::empty({N * CH, S}, m.options());
  auto bits = torch::empty({N, S / 32}, m.options().dtype(torch::kInt32));
  auto stats = save_stats ? torch::empty({N, S, 2}, m.options().dtype(torch::kFloat32)) : torch::empty({0}, m.options().dtype(torch::kFloat32));
  CUtensorMap xm = make_map<3>(m.data_ptr(), {(uint64_t)CM, (uint64_t)N, (uint64_t)S}, {(uint64_t)CM, (uint64_t)N * CM}, {64, 1, BS}, CU_TENSOR_MAP_SWIZZLE_128B, "m");
  CUtensorMap am = make_map<2>(A2.data_ptr(), {(uint64_t)S, (uint64_t)N * CH}, {(uint64_t)S}, {64, CH}, CU_TENSOR_MAP_SWIZZLE_128B, "A2");
  CUtensorMap bm = make_map<2>(BT.data_ptr(), {(uint64_t)S, (uint64_t)N * CH}, {(uint64_t)S}, {64, CH}, CU_TENSOR_MAP_SWIZZLE_128B, "BT");
  const int ntiles = (int)(N * (S / BS));
  auto st = at::cuda::getCurrentCUDAStream();
  float2* sp = save_stats ? reinterpret_cast<float2*>(stats.data_ptr<float>()) : nullptr;
  auto* bp = reinterpret_cast<uint32_t*>(bits.data_ptr<int>());
  CUtensorMap mm = make_map<2>(mask.data_ptr(), {(uint64_t)N, (uint64_t)S}, {(uint64_t)N}, {16, BS}, CU_TENSOR_MAP_SWIZZLE_NONE, "mask",
                               CU_TENSOR_MAP_DATA_TYPE_UINT8, 1);
  CUtensorMap wam = make_map<2>(wa.data_ptr(), {(uint64_t)CM, (uint64_t)CH}, {(uint64_t)CM}, {64, CH}, CU_TENSOR_MAP_SWIZZLE_128B, "wa");
  CUtensorMap wbm = make_map<2>(wb.data_ptr(), {(uint64_t)CM, (uint64_t)CH}, {(uint64_t)CM}, {64, CH}, CU_TENSOR_MAP_SWIZZLE_128B, "wb");
  if (lnw.scalar_type() == torch::kFloat32) {
    static bool a = false;
    if (!a) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(opm_prologue_sm100<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
      C10_CUDA_CHECK(cudaFuncSetAttribute(opm_prologue_sm100<float>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      a = true;
    }
    const int grid = std::min(ntiles, CTAS * num_sms(m.device().index()));
    opm_prologue_sm100<float><<<grid, THREADS, SMEM, st>>>((int)N, (int)S, ntiles, (float)eps, xm, am, bm, mm, wam, wbm,
        lnw.data_ptr<float>(), lnb.data_ptr<float>(), sp, bp);
  } else {
    TORCH_CHECK(lnw.scalar_type() == torch::kBFloat16, "LayerNorm affine: fp32 or bf16");
    static bool a = false;
    if (!a) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(opm_prologue_sm100<__nv_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
      C10_CUDA_CHECK(cudaFuncSetAttribute(opm_prologue_sm100<__nv_bfloat16>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      a = true;
    }
    const int grid = std::min(ntiles, CTAS * num_sms(m.device().index()));
    opm_prologue_sm100<__nv_bfloat16><<<grid, THREADS, SMEM, st>>>((int)N, (int)S, ntiles, (float)eps, xm, am, bm, mm, wam, wbm,
        reinterpret_cast<const __nv_bfloat16*>(lnw.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(lnb.data_ptr<at::BFloat16>()), sp, bp);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto norm = torch::empty({want_norm ? N : 0, N}, m.options().dtype(torch::kFloat32));
  if (want_norm) {                                   // the epilogue / dgrad count from the bits themselves
    const int W = (int)(S / 32);
    opm_norm_sm100<<<dim3(N / 32, N / 32), 256, 64 * (W + 1) * 4, st>>>(bp, norm.data_ptr<float>(), (int)N, W);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {A2, BT, norm, stats, bits};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("opm_epilogue", &opm_epilogue, "sm100 fused OPM epilogue (div-norm + proj_out + bias + optional residual)",
        py::arg("O"), py::arg("norm"), py::arg("wo"), py::arg("bias"), py::arg("ni"), py::arg("nj"), py::arg("residual") = py::none());
  m.def("opm_prologue", &opm_prologue, "sm100 fused OPM prologue: LN + both projections + mask -> A2, BT (grouped layouts), mask count, LN stats [N][S]",
        py::arg("m"), py::arg("mask"), py::arg("lnw"), py::arg("lnb"), py::arg("eps"), py::arg("wa"), py::arg("wb"), py::arg("save_stats") = true, py::arg("want_norm") = false);
}
