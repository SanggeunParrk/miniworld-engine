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

// ---------------------------------------------------------------------------------------------
// BACKWARD, dgrad: dO[(i,c),(j,e)] = (sum_z dz[i,j,z] Wo[z,(c,e)]) / n[i,j] straight into the GROUPED layout
// (so the [N,N,c_hidden^2] permute never exists), dz/n alongside it for dWo, and dbias = sum_(i,j) dz.
// The H100 kernel's roles, on tcgen05: a tile is 128 (i,j) pairs; the tensor core runs dz . Wo in eight
// 128-column chunks into a ring of three TMEM accumulators, and the drain applies 1/n in fp32 before the
// single bf16 rounding (the H100 kernel rounded dz/n first) and stores each chunk as one 4-D TMA box of
// dO.  dbias is a second tensor-core product, dz^T . 1, accumulated in TMEM across the CTA's tiles.
// dz/n leaves at the START of a tile, straight from registers (a warp's 32 rows are 8 KiB contiguous), so the dz stage
// is free as soon as its GEMMs retire; the dz tiles and the Wo chunks have one producer thread each, so the next
// tile's dz is never queued behind this tile's Wo chunks (it was: a ~10K-cycle bubble at every tile start).
namespace dg {
constexpr int BI = 4, BJ = 32, NP = BI * BJ;
constexpr int NC = 128;                                   // output columns per chunk: 4 c values
constexpr int NCK = NCH / NC;                             // 8 chunks per tile
constexpr int DZT = NP * CZ;                              // dz tile [2 z halves][128 pairs][64], swizzled
constexpr int WOT = CZ * NC;                              // one Wo chunk [2 n blocks][128 z][64 n]
constexpr int OUT = NP * NC;                              // one dO box [4 i][4 c][32 j][32 e], 64B swizzle
constexpr int NDZ = 2, NWO = 2, NACC = 3, NOUT = 2;
constexpr int ONES = 16 * NP;                             // [2 k blocks][16][64]
constexpr int THREADS = 352;                              // warp 0: Wo chunks, warp 1: MMA, warps 2-9: drain, warp 10: dz tiles
constexpr int SMEM = 1024 + (NDZ * DZT + NWO * WOT + NOUT * OUT + ONES) * 2 + 512;
constexpr uint32_t IDESC = idesc_bf16(128, NC, 0, 1);     // A = dz K-major, B = Wo MN-major
constexpr uint32_t IDESC_BO = idesc_bf16(128, 16, 1, 0);  // A = dz^T (MN-major), B = ones K-major
constexpr int BO_COL = NACC * NC;
static_assert(SMEM <= 232448, "one CTA per SM");
}  // namespace dg

__global__ void __launch_bounds__(dg::THREADS, 1) opm_dgrad_sm100(
    int NI, int NJ, int ntiles,
    const __grid_constant__ CUtensorMap dzmap,    // dz  as (64 z, NJ j, 2 halves, NI i), box (64, BJ, 1, BI), 128B swizzle
    const __grid_constant__ CUtensorMap womap,    // Wo  [CZ][NCH], box (64 n, 128 z), 128B swizzle
    const __grid_constant__ CUtensorMap domap,    // dO  as (32 e, NJ j, 32 c, NI i), box (32, BJ, 4, BI), 64B swizzle
    __nv_bfloat16* __restrict__ DZP,              // dz/n [NI][NJ][CZ]
    const uint32_t* __restrict__ BITS, int W, float* __restrict__ DBO) {
  using namespace dg;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sDZ = reinterpret_cast<__nv_bfloat16*>(smb);
  __nv_bfloat16* sWO = sDZ + NDZ * DZT;
  __nv_bfloat16* sOUT = sWO + NWO * WOT;
  __nv_bfloat16* sONE = sOUT + NOUT * OUT;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sONE + ONES);
  uint64_t* dzf = bars;              // [NDZ] dz landed
  uint64_t* dze = dzf + NDZ;         // [NDZ] count 9: every GEMM reading dz retired (commit) and the 8 drain warps read it
  uint64_t* wof = dze + NDZ;         // [NWO]
  uint64_t* woe = wof + NWO;         // [NWO]
  uint64_t* accf = woe + NWO;        // [NACC]
  uint64_t* acce = accf + NACC;      // [NACC] count 8
  uint64_t* bod = acce + NACC;       // [1] the last dbias GEMM retired
  uint32_t* tslot = reinterpret_cast<uint32_t*>(bod + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int k = 0; k < NDZ; ++k) { bar_init(dzf + k, 1); bar_init(dze + k, 9); }
    for (int k = 0; k < NWO; ++k) { bar_init(wof + k, 1); bar_init(woe + k, 1); }
    for (int k = 0; k < NACC; ++k) { bar_init(accf + k, 1); bar_init(acce + k, 8); }
    bar_init(bod, 1);
    bar_init_fence();
  }
  // the ones operand of the dbias GEMM: B[n][k] = 1 for n == 0 (the other 15 columns are zero), K-major swizzled
  for (int v = tid; v < ONES / 8; v += THREADS) {
    const int row = (v / 8) % 16;
    uint4 o = make_uint4(0u, 0u, 0u, 0u);
    if (row == 0) o = make_uint4(0x3f803f80u, 0x3f803f80u, 0x3f803f80u, 0x3f803f80u);
    reinterpret_cast<uint4*>(sONE)[v] = o;       // row 0 is all ones, so the swizzle of its chunks is irrelevant
  }
  if (warp == 1) tmem_alloc(tslot, 512);
  fence_proxy_async();
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  const int njb = NJ / BJ;

  if (warp == 10) {
    if (lane == 0) {                             // the dz tiles (their own warp: two producer loops in one warp starve each other)
      int lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int i0 = (t / njb) * BI, j0 = (t % njb) * BJ, d = lt % NDZ;
        if (lt >= NDZ) wait(dze + d, ((lt / NDZ) - 1) & 1);
        expect_tx(dzf + d, DZT * 2);
        for (int h = 0; h < 2; ++h) load_4d(&dzmap, sDZ + d * DZT + h * NP * 64, dzf + d, 0, j0, h, i0);
      }
    }
  } else if (warp == 0) {
    if (lane == 0) {                             // the Wo chunks (L2-resident)
      int g = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x)
        for (int k = 0; k < NCK; ++k, ++g) {
          const int w = g % NWO;
          if (g >= NWO) wait(woe + w, ((g / NWO) - 1) & 1);
          expect_tx(wof + w, WOT * 2);
          for (int b = 0; b < 2; ++b) load_2d(&womap, sWO + w * WOT + b * CZ * 64, wof + w, k * NC + b * 64, 0);
        }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      int lt = 0, g = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int d = lt % NDZ;
        wait(dzf + d, (lt / NDZ) & 1);
        tc_fence_after();
        const __nv_bfloat16* dz = sDZ + d * DZT;
        for (int k = 0; k < NCK; ++k, ++g) {
          const int w = g % NWO, a = g % NACC;
          wait(wof + w, (g / NWO) & 1);
          if (g >= NACC) wait(acce + a, ((g / NACC) - 1) & 1);
          tc_fence_after();
          const __nv_bfloat16* wo = sWO + w * WOT;
#pragma unroll
          for (int kk = 0; kk < CZ / 16; ++kk)
            mma_ss(tmem + a * NC, desc_k128(dz + (kk >> 2) * NP * 64 + (kk & 3) * 16),
                   desc_mn128(wo + kk * 16 * 64, CZ * 64 * 2), IDESC, kk ? 1u : 0u);
          mma_commit(woe + w);
          mma_commit(accf + a);
        }
        // dbias += dz^T . 1 over this tile's 128 pairs (raw dz, before the mask count divides it)
#pragma unroll
        for (int kk = 0; kk < NP / 16; ++kk)
          mma_ss(tmem + BO_COL, desc_mn128(dz + kk * 16 * 64, NP * 64 * 2),
                 desc_k128(sONE + (kk >> 2) * 16 * 64 + (kk & 3) * 16), IDESC_BO, (lt | kk) ? 1u : 0u);
        mma_commit(dze + d);
      }
      mma_commit(bod);
    }
  } else {
    // eight drain warps: warps q and q + 4 read the same TMEM lanes (pairs), each taking half of every chunk's columns
    // (and half of dz/n); they meet on one 256-thread barrier per TMA store.  Four warps were the kernel's bottleneck.
    const int q = warp & 3, grp = (warp - 2) >> 2;
    const int p = q * 32 + lane, il = p / BJ, jl = p % BJ;
    const bool leader = (warp == 2 && lane == 0);
    int lt = 0, g = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int i0 = (t / njb) * BI, j0 = (t % njb) * BJ, d = lt % NDZ;
      const uint4* bi = reinterpret_cast<const uint4*>(BITS + (size_t)(i0 + il) * W);
      const uint4* bj = reinterpret_cast<const uint4*>(BITS + (size_t)(j0 + jl) * W);
      int cnt = 0;
      for (int w = 0; w < W / 4; ++w) {
        const uint4 x = __ldg(bi + w), y = __ldg(bj + w);
        cnt += __popc(x.x & y.x) + __popc(x.y & y.y) + __popc(x.z & y.z) + __popc(x.w & y.w);
      }
      const float inv = 1.0f / (float)max(cnt, 1);
      const float2 inv2 = make_float2(inv, inv);
      {                                          // dz/n for dWo: this pair's 128 z, straight to global
        wait(dzf + d, (lt / NDZ) & 1);
        const __nv_bfloat16* dz = sDZ + d * DZT;
        uint4* dst = reinterpret_cast<uint4*>(DZP + ((size_t)(i0 + il) * NJ + j0 + jl) * CZ);
        {
          const int h = grp;                     // this warp's 64 of the pair's 128 z
          const __nv_bfloat16* rp = dz + h * NP * 64 + p * 64;
#pragma unroll
          for (int c8 = 0; c8 < 8; ++c8) {
            uint4 u = *reinterpret_cast<const uint4*>(rp + ((c8 ^ (p & 7)) << 3));
            const float2 x0 = mul2(bf2f(u.x), inv2), x1 = mul2(bf2f(u.y), inv2), x2 = mul2(bf2f(u.z), inv2), x3 = mul2(bf2f(u.w), inv2);
            u.x = pack2(x0.x, x0.y); u.y = pack2(x1.x, x1.y); u.z = pack2(x2.x, x2.y); u.w = pack2(x3.x, x3.y);
            dst[h * 8 + c8] = u;
          }
        }
        __syncwarp();
        if (lane == 0) arrive(dze + d);
      }
      for (int k = 0; k < NCK; ++k, ++g) {
        const int a = g % NACC, ob = g % NOUT;
        wait(accf + a, (g / NACC) & 1);
        tc_fence_after();
        if (leader) bulk_wait_read<NOUT - 1>();  // the store of chunk g - NOUT has left staging buffer ob
        named_sync(1, 256);
        __nv_bfloat16* box = sOUT + ob * OUT;
#pragma unroll
        for (int cc = 0; cc < 2; ++cc) {         // one c value = 32 e per pass; this warp's two of the chunk's four
          const int cl = grp * 2 + cc;
          float v[32];
          tmem_ld32(tmem_at(tmem + a * NC, q * 32, cl * 32), v);
          tmem_wait_ld();
          if (cc == 1) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(acce + a); }
          const int row = (il * 4 + cl) * BJ + jl;             // box [i][c][j][32 e], 64-byte rows
          __nv_bfloat16* rp = box + row * 32;
#pragma unroll
          for (int c8 = 0; c8 < 4; ++c8) {
            const float2 x0 = mul2(make_float2(v[c8 * 8 + 0], v[c8 * 8 + 1]), inv2), x1 = mul2(make_float2(v[c8 * 8 + 2], v[c8 * 8 + 3]), inv2);
            const float2 x2 = mul2(make_float2(v[c8 * 8 + 4], v[c8 * 8 + 5]), inv2), x3 = mul2(make_float2(v[c8 * 8 + 6], v[c8 * 8 + 7]), inv2);
            uint4 o;
            o.x = pack2(x0.x, x0.y); o.y = pack2(x1.x, x1.y); o.z = pack2(x2.x, x2.y); o.w = pack2(x3.x, x3.y);
            *reinterpret_cast<uint4*>(rp + ((c8 ^ ((row >> 1) & 3)) << 3)) = o;
          }
        }
        fence_proxy_async();
        named_sync(1, 256);
        if (leader) { store_4d(&domap, box, 0, j0, k * 4, i0); bulk_commit(); }
      }
    }
    // this CTA's dbias partial: TMEM lane = z, column BO_COL
    wait(bod, 0);
    tc_fence_after();
    float v[8];
    tmem_ld8(tmem_at(tmem + BO_COL, q * 32, 0), v);
    tmem_wait_ld();
    if (grp == 0) DBO[(size_t)blockIdx.x * CZ + q * 32 + lane] = v[0];
    if (leader) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}

// ---------------------------------------------------------------------------------------------
// BACKWARD, dWo[z,(c,e)] = sum_(i,j) (dz/n)[i,j,z] . O[(i,c),(j,e)], read off the GROUPED O -- no permute.
// A GEMM with M = c_z = 128, N = c_hidden^2 and K = the N^2 pairs.  One CTA owns one half of N (512
// columns: all of TMEM, one accumulator for the CTA's whole K range) and a contiguous run of 64-pair
// k-steps; both operands are MN-major tiles exactly as TMA lands them (dz/n: [z half][pairs][64 z],
// 128B swizzle; O: [c][pairs][32 e], 64B swizzle), so no thread touches the operands.  The K splits
// leave fp32 partials that one column-sum GEMV folds in split order: deterministic.
namespace dw {
constexpr int BI = 2, BJ = 32, NP = BI * BJ;              // 64 pairs per k-step
constexpr int NPART = 4;                                  // column parts per K split: fewer, longer splits halve the fp32
constexpr int NHALF = NCH / NPART;                        // partials (their HBM round trip); dz/n re-reads are L2 hits
constexpr int NCB = NHALF / 32;                           // c blocks per CTA
constexpr int AT = NP * CZ;                               // dz/n tile [2 z blocks][64 pairs][64 z]
constexpr int BT = NP * NHALF;                            // O tile [NCB c][64 pairs][32 e]
constexpr int STAGE = AT + BT;
constexpr int NST = 3;
constexpr int THREADS = 192;
constexpr int SMEM = 1024 + NST * STAGE * 2 + 256;
constexpr int MN = NHALF < 256 ? NHALF : 256;               // MMA N
constexpr uint32_t IDESC = idesc_bf16(128, MN, 1, 1);
static_assert(SMEM <= 232448, "one CTA per SM");
}  // namespace dw

__global__ void __launch_bounds__(dw::THREADS, 1) opm_dwo_sm100(
    int NI, int NJ, int ntiles, int nsplit,
    const __grid_constant__ CUtensorMap amap,     // dz/n as (64 z, NJ j, NI i, 2 halves), box (64, BJ, BI, 2), 128B swizzle
    const __grid_constant__ CUtensorMap bmap,     // O    as (32 e, NJ j, NI i, 32 c), box (32, BJ, BI, 16), 64B swizzle
    const __grid_constant__ CUtensorMap pmap) {   // the partials [nsplit * CZ][NCH] fp32, box (32, 128), 128B swizzle
  using namespace dw;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sStage = reinterpret_cast<__nv_bfloat16*>(smb);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sStage + NST * STAGE);
  uint64_t* full = bars;             // [NST]
  uint64_t* empty = full + NST;      // [NST]
  uint64_t* done = empty + NST;      // [1]
  uint32_t* tslot = reinterpret_cast<uint32_t*>(done + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int half = blockIdx.y, split = blockIdx.x;
  const int k0 = (int)((long)ntiles * split / nsplit), k1 = (int)((long)ntiles * (split + 1) / nsplit);
  const int njb = NJ / BJ;
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { bar_init(full + s, 1); bar_init(empty + s, 1); }
    bar_init(done, 1);
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, NHALF < 32 ? 32 : NHALF);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;

  if (warp == 0) {
    if (lane == 0) {
      for (int k = k0, g = 0; k < k1; ++k, ++g) {
        const int s = g % NST, i0 = (k / njb) * BI, j0 = (k % njb) * BJ;
        if (g >= NST) wait(empty + s, ((g / NST) - 1) & 1);
        __nv_bfloat16* st = sStage + s * STAGE;
        expect_tx(full + s, STAGE * 2);
        load_4d(&amap, st, full + s, 0, j0, i0, 0);
        load_4d(&bmap, st + AT, full + s, 0, j0, i0, half * NCB);
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      for (int k = k0, g = 0; k < k1; ++k, ++g) {
        const int s = g % NST;
        wait(full + s, (g / NST) & 1);
        tc_fence_after();
        const __nv_bfloat16* a = sStage + s * STAGE;
        const __nv_bfloat16* b = a + AT;
#pragma unroll
        for (int ks = 0; ks < NP / 16; ++ks)
#pragma unroll
          for (int h = 0; h < NHALF / MN; ++h)   // MN columns = MN / 32 c blocks of 32 e
            mma_ss(tmem + h * MN, desc_mn128(a + ks * 16 * 64, NP * 64 * 2),
                   sdesc(sa(b + h * (MN / 32) * NP * 32 + ks * 16 * 32), NP * 32 * 2, 512, 4), IDESC, (g | ks) ? 1u : 0u);
        mma_commit(empty + s);
      }
      mma_commit(done);
    }
  } else {
    // drain: TMEM lane = z; 512 columns = this CTA's half of (c,e).  The partial leaves by TMA from 128B-swizzled
    // [128 z][32] fp32 tiles in the (now idle) stage buffers: per-thread row stores of it were slow partial-line writes.
    const int q = warp & 3, z = q * 32 + lane;
    const bool leader = (warp == 2 && lane == 0);
    wait(done, 0);
    tc_fence_after();
    const bool any = k1 > k0;
    float* stg = reinterpret_cast<float*>(sStage);                 // 4 tiles of [128][32] fp32 (16 KiB each)
#pragma unroll 1
    for (int c0 = 0; c0 < NHALF; c0 += 32) {
      const int bufi = (c0 / 32) & 3;
      if (bufi == 0) { if (leader) bulk_wait_read<0>(); named_sync(1, 128); }
      float v[32];
      tmem_ld32(tmem_at(tmem, q * 32, c0), v);
      tmem_wait_ld();
      float* row = stg + bufi * 128 * 32 + z * 32;
#pragma unroll
      for (int k = 0; k < 8; ++k)
        *reinterpret_cast<float4*>(row + ((k ^ (z & 7)) << 2)) = any ? make_float4(v[4 * k], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3]) : make_float4(0.f, 0.f, 0.f, 0.f);
      if (bufi == 3) {
        fence_proxy_async();
        named_sync(1, 128);
        if (leader) {
          for (int bb = 0; bb < 4; ++bb) store_2d(&pmap, stg + bb * 128 * 32, half * NHALF + c0 - 96 + bb * 32, split * CZ);
          bulk_commit();
        }
      }
    }
    if (leader) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, NHALF < 32 ? 32 : NHALF);
}

// ---------------------------------------------------------------------------------------------
// BACKWARD prologue, fused (the H100 kernel's math): per MSA row (s, i)
//     g   = mask . ([da db] . Wf)          Wf = [gamma . Wa ; gamma . Wb] (64 x 64): g = dL/dxh
//     dm  = rstd (g - mean(g) - xh mean(g xh))
//     R  += [da db]^T (mask . xh),  ssa += mask . [da db]      (dWa, dWb, dgamma, dbeta on the host)
// Tile = one token x 128 MSA rows.  da / db land by TMA as two [128 s][32] 64B-swizzled tiles, which are
// at once the K-major A operand of the dy GEMM and the MN-major B operand of the dW GEMM; the dW GEMM's
// A operand is (mask . xh) with one extra 64-wide block whose first column is the mask, so the masked
// column sums ssa are row 64 of the same accumulator.  Two compute groups take alternate tiles.
namespace pb {
constexpr int CM = 64, BS = 128;
// The tile loads are latency-bound, so what matters is how long a stage is held.  Two rings: da | db are the dW GEMM's
// operand and stay until it retires (a tile later), but x, the stats and the mask are read once at the start of the
// compute and leave at once -- in one ring they were all held for the dW GEMM.
constexpr int NSA = 4, NSX = 2;
constexpr int DAT = BS * CH;                               // da or db tile [128 s][32]
constexpr int XT = BS * CM;                                // x tile [128 s][64]
constexpr int STAT = BS * 2;                               // (mean, rstd) per row, fp32
constexpr int MKT = BS * 16;                               // mask bytes [128 s][16 tokens]
constexpr int STAGE_A = 2 * DAT * 2;                       // bytes: da | db
constexpr int STAGE_X = 20480;                             // bytes: x | stats | mask, padded to 1 KiB
constexpr int A2T = 2 * XT;                                // [mask . xh | mask column block], [2][128 s][64]
constexpr int THREADS = 352;                               // warp 0: da/db, warp 1: MMA, warps 2-9: two compute groups, warp 10: x
constexpr int SMEM = 1024 + NSA * STAGE_A + NSX * STAGE_X + (2 * A2T + 2 * XT + CM * CM) * 2 + 256;
static_assert(XT * 2 + STAT * 4 + MKT <= STAGE_X && STAGE_A % 1024 == 0, "stage layout");
// the two compute groups take alternate tiles: an x slot must belong to one group, or a group can reach a slot's
// barrier two phases ahead and its parity wait passes on stale data
static_assert(NSX % 2 == 0, "x ring depth: a multiple of the two compute groups");
constexpr uint32_t IDESC_DY = idesc_bf16(128, CM, 0, 1);   // A = [da|db] K-major, B = Wf MN-major
constexpr uint32_t IDESC_DW = idesc_bf16(128, CM, 1, 1);   // A = (mask . xh)^T MN-major, B = [da db] MN-major
constexpr int D2 = 128;                                    // TMEM column of the dW accumulator (dy: 0 and 64)
static_assert(SMEM <= 232448, "one CTA per SM");
}  // namespace pb

template <typename WT_>
__global__ void __launch_bounds__(pb::THREADS, 1) opm_prologue_bwd_sm100(
    int N, int S, int ntiles,
    const __grid_constant__ CUtensorMap damap,    // dA [S][N*32] as (32, N, S), box (32, 1, 128), 64B swizzle
    const __grid_constant__ CUtensorMap dbmap,    // dB, same
    const __grid_constant__ CUtensorMap xmap,     // m [S][N][64] as (64, N, S), box (64, 1, 128), 128B swizzle
    const __grid_constant__ CUtensorMap smap,     // stats [N][S][2] fp32 as (2S, N), box (256, 1)
    const __grid_constant__ CUtensorMap mmap,     // mask [S][N] bytes, box (16, 128)
    const __nv_bfloat16* __restrict__ WA, const __nv_bfloat16* __restrict__ WB, const WT_* __restrict__ GAM,
    const __grid_constant__ CUtensorMap dmmap,    // dm, same layout as m
    float* __restrict__ PART) {                   // [grid][65][64]: rows k = R^T, row 64 = ssa
  using namespace pb;
  const int NSB = S / BS;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  unsigned char* sStage = smb;                                                    // [NSA][da | db]
  unsigned char* sXs = smb + NSA * STAGE_A;                                        // [NSX][x | stats | mask]
  __nv_bfloat16* sA2 = reinterpret_cast<__nv_bfloat16*>(sXs + NSX * STAGE_X);     // [2 groups][A2T]
  __nv_bfloat16* sDM = sA2 + 2 * A2T;                                             // [2 groups][XT]
  __nv_bfloat16* sWF = sDM + 2 * XT;                                              // [64][64]
  uint64_t* bars = reinterpret_cast<uint64_t*>(sWF + CM * CM);
  uint64_t* full = bars;             // [NSA]
  uint64_t* empty = full + NSA;      // [NSA]  the dW GEMM of the stage's tile retired
  uint64_t* xfull = empty + NSA;     // [NSX]
  uint64_t* xempty = xfull + NSX;    // [NSX]  count 4: the compute group has x / stats / mask in registers
  uint64_t* d1f = xempty + NSX;      // [2]    dy accumulator of group gr
  uint64_t* d1e = d1f + 2;           // [2]    count 4
  uint64_t* a2f = d1e + 2;           // [2]    count 4: group gr wrote (mask . xh)
  uint64_t* a2e = a2f + 2;           // [2]    the dW GEMM reading it retired
  uint64_t* done = a2e + 2;          // [1]
  uint32_t* tslot = reinterpret_cast<uint32_t*>(done + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int k = 0; k < NSA; ++k) { bar_init(full + k, 1); bar_init(empty + k, 1); }
    for (int k = 0; k < NSX; ++k) { bar_init(xfull + k, 1); bar_init(xempty + k, 4); }
    for (int k = 0; k < 2; ++k) { bar_init(d1f + k, 1); bar_init(d1e + k, 4); bar_init(a2f + k, 4); bar_init(a2e + k, 1); }
    bar_init(done, 1);
    bar_init_fence();
  }
  // Wf = [gamma . Wa ; gamma . Wb] in bf16 (the projection backward then hands back g = dout . gamma = dL/dxh),
  // built here rather than by four host ops per call: [64 (c|e) rows][64 k], 128B swizzle, the dy GEMM's MN-major B
  for (int v = tid; v < CM * CM / 2; v += THREADS) {
    const int row = v / (CM / 2), k = (v % (CM / 2)) * 2;
    const __nv_bfloat16* w = row < CH ? WA + row * CM : WB + (row - CH) * CM;
    const float2 wv = bf2f(*reinterpret_cast<const uint32_t*>(w + k));
    *reinterpret_cast<uint32_t*>(sWF + sw128(row, k)) = pack2(wv.x * to_f(GAM[k]), wv.y * to_f(GAM[k + 1]));
  }
  // the mask-column blocks: zero once; each tile rewrites only the chunk holding column 0 of its rows
  for (int v = tid; v < 2 * XT / 8; v += THREADS) {
    const int gr = v / (XT / 8), rest = v % (XT / 8);
    reinterpret_cast<uint4*>(sA2 + gr * A2T + XT)[rest] = make_uint4(0u, 0u, 0u, 0u);
  }
  if (warp == 1) tmem_alloc(tslot, 256);
  fence_proxy_async();
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  auto stage_ptr = [&](int st) { return sStage + st * STAGE_A; };
  auto xstage_ptr = [&](int st) { return sXs + st * STAGE_X; };
  // stage layouts: da [128][32] | db [128][32];  x [128][64] | stats [128][2] fp32 | mask [128][16] bytes

  if (warp == 0) {
    if (lane == 0) {
      int g = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++g) {
        const int i = t / NSB, s0 = (t % NSB) * BS, st = g % NSA;
        if (g >= NSA) wait(empty + st, ((g / NSA) - 1) & 1);
        unsigned char* p = stage_ptr(st);
        expect_tx(full + st, STAGE_A);
        load_3d(&damap, p, full + st, 0, i, s0);
        load_3d(&dbmap, p + DAT * 2, full + st, 0, i, s0);
      }
    }
  } else if (warp == 10) {
    if (lane == 0) {
      int g = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++g) {
        const int i = t / NSB, s0 = (t % NSB) * BS, st = g % NSX;
        if (g >= NSX) wait(xempty + st, ((g / NSX) - 1) & 1);
        unsigned char* p = xstage_ptr(st);
        expect_tx(xfull + st, XT * 2 + STAT * 4 + MKT);
        load_3d(&xmap, p, xfull + st, 0, i, s0);
        load_2d(&smap, p + XT * 2, xfull + st, 2 * s0, i);
        load_2d(&mmap, p + XT * 2 + STAT * 4, xfull + st, i & ~15, s0);
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      auto dw_gemm = [&](int lt) {             // dW += (mask . xh | mask)^T-block . [da db] for tile lt
        const int gr = lt & 1, st = lt % NSA;
        wait(a2f + gr, (lt >> 1) & 1);
        tc_fence_after();
        const __nv_bfloat16* a2 = sA2 + gr * A2T;
        const __nv_bfloat16* dab = reinterpret_cast<const __nv_bfloat16*>(stage_ptr(st));
#pragma unroll
        for (int ks = 0; ks < BS / 16; ++ks)
          mma_ss(tmem + D2, desc_mn128(a2 + ks * 16 * 64, XT * 2), sdesc(sa(dab + ks * 16 * 32), DAT * 2, 512, 4), IDESC_DW, (lt | ks) ? 1u : 0u);
        mma_commit(empty + st);
        mma_commit(a2e + gr);
      };
      int lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int st = lt % NSA, gr = lt & 1;
        wait(full + st, (lt / NSA) & 1);
        if (lt >= 2) wait(d1e + gr, ((lt >> 1) - 1) & 1);
        tc_fence_after();
        const __nv_bfloat16* dab = reinterpret_cast<const __nv_bfloat16*>(stage_ptr(st));
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)           // K = 64: two k steps in da, two in db
          mma_ss(tmem + gr * 64, desc_k64(dab + (ks >> 1) * DAT + (ks & 1) * 16), desc_mn128(sWF + ks * 16 * 64, CM * 64 * 2),
                 IDESC_DY, ks ? 1u : 0u);
        mma_commit(d1f + gr);
        if (lt >= 1) dw_gemm(lt - 1);
      }
      if (lt >= 1) dw_gemm(lt - 1);
      mma_commit(done);
    }
  } else {
    const int gr = (warp - 2) >> 2, q = warp & 3, r = q * 32 + lane;
    const bool leader = (lane == 0 && q == 2);    // warps 2 and 6 (q == 2) lead their group
    const int bar_id = 1 + gr;
    int lt = gr;
    for (int t = blockIdx.x + gr * gridDim.x; t < ntiles; t += 2 * gridDim.x, lt += 2) {
      const int i = t / NSB, s0 = (t % NSB) * BS, sx = lt % NSX;
      // x, the stats and the mask first (they do not wait for the tensor core), then their stage is free
      wait(xfull + sx, (lt / NSX) & 1);
      const unsigned char* px = xstage_ptr(sx);
      const __nv_bfloat16* xr = reinterpret_cast<const __nv_bfloat16*>(px) + r * 64;
      const float2 stt = reinterpret_cast<const float2*>(px + XT * 2)[r];
      const float mk = px[XT * 2 + STAT * 4 + r * 16 + (i & 15)] ? 1.f : 0.f;
      uint4 xu[8];
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) xu[c8] = *reinterpret_cast<const uint4*>(xr + ((c8 ^ (r & 7)) << 3));
      __syncwarp();
      if (lane == 0) arrive(xempty + sx);
      wait(d1f + gr, (lt >> 1) & 1);
      tc_fence_after();
      float gv[CM];
      tmem_ld32(tmem_at(tmem + gr * 64, q * 32, 0), gv);
      tmem_ld32(tmem_at(tmem + gr * 64, q * 32, 32), gv + 32);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) arrive(d1e + gr);
      const float mean = stt.x, rstd = stt.y;
      // xh, and the two row sums of g = mask . dy
      float xh[CM];
      float gs = 0.f, gx = 0.f;
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) {
        const uint4 u = xu[c8];
        const float2 f0 = bf2f(u.x), f1 = bf2f(u.y), f2 = bf2f(u.z), f3 = bf2f(u.w);
        const float xs[8] = {f0.x, f0.y, f1.x, f1.y, f2.x, f2.y, f3.x, f3.y};
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int c = c8 * 8 + k;
          xh[c] = (xs[k] - mean) * rstd;
          gv[c] *= mk;
          gs += gv[c];
          gx = fmaf(gv[c], xh[c], gx);
        }
      }
      if (lt >= 2) wait(a2e + gr, ((lt >> 1) - 1) & 1);   // the dW GEMM of this group's previous tile retired
      if (leader) bulk_wait_read<0>();                     // this group's previous dm store has left sDM
      named_sync(bar_id, 128);
      // dm = rstd * (g - mean(g) - xh * mean(g xh)), and (mask . xh) into the dW operand
      const float ra = -rstd * gs * (1.f / CM), rb = rstd * gx * (1.f / CM);
      __nv_bfloat16* dmr = sDM + gr * XT + r * 64;
      __nv_bfloat16* a2r = sA2 + gr * A2T + r * 64;
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) {
        uint4 d, a;
        uint32_t* dw_ = reinterpret_cast<uint32_t*>(&d);
        uint32_t* aw = reinterpret_cast<uint32_t*>(&a);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int c = c8 * 8 + 2 * k;
          dw_[k] = pack2(fmaf(-xh[c], rb, fmaf(gv[c], rstd, ra)), fmaf(-xh[c + 1], rb, fmaf(gv[c + 1], rstd, ra)));
          aw[k] = pack2(xh[c] * mk, xh[c + 1] * mk);
        }
        const int off = (c8 ^ (r & 7)) << 3;
        *reinterpret_cast<uint4*>(dmr + off) = d;
        *reinterpret_cast<uint4*>(a2r + off) = a;
      }
      // the mask column: element (r, 0) of the second block, i.e. its chunk 0 ^ (r & 7)
      *reinterpret_cast<uint4*>(a2r + XT + ((0 ^ (r & 7)) << 3)) = make_uint4(mk != 0.f ? 0x3f80u : 0u, 0u, 0u, 0u);
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(a2f + gr);
      named_sync(bar_id, 128);
      if (leader) { store_3d(&dmmap, sDM + gr * XT, 0, i, s0); bulk_commit(); }
    }
    if (leader) bulk_wait<0>();
    if (gr == 0) {                                 // the dW accumulator: TMEM lane = k (0-63) or the mask row (64)
      wait(done, 0);
      tc_fence_after();
      if (q < 3) {
        float v[32];
        const int row = q * 32 + lane;
        for (int c0 = 0; c0 < CM; c0 += 32) {
          tmem_ld32(tmem_at(tmem + D2, q * 32, c0), v);
          tmem_wait_ld();
          if (row <= CM) {
            float* out = PART + ((size_t)blockIdx.x * (CM + 1) + row) * CM + c0;
#pragma unroll
            for (int k = 0; k < 32; k += 4) *reinterpret_cast<float4*>(out + k) = make_float4(v[k], v[k + 1], v[k + 2], v[k + 3]);
          }
        }
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 256);
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

std::vector<torch::Tensor> opm_dgrad(torch::Tensor dz, torch::Tensor bits, torch::Tensor wo, int64_t ni, int64_t nj, int64_t grad_bf16) {
  using namespace dg;
  TORCH_CHECK(dz.is_cuda() && dz.scalar_type() == torch::kBFloat16 && dz.is_contiguous() && dz.numel() == ni * nj * CZ, "dz: contiguous bf16 [ni, nj, 128]");
  TORCH_CHECK(bits.scalar_type() == torch::kInt32 && bits.is_contiguous() && bits.dim() == 2 && bits.size(1) % 4 == 0, "bits: int32 [N, S/32]");
  TORCH_CHECK(wo.scalar_type() == torch::kBFloat16 && wo.is_contiguous() && wo.numel() == CZ * NCH, "wo: bf16 [128, 1024]");
  TORCH_CHECK(ni % BI == 0 && nj % BJ == 0, "the kernel tiles ", BI, " x ", BJ, " tokens");
  auto dO = torch::empty({ni * CH, nj * CH}, dz.options());
  auto dzp = torch::empty({ni, nj, (long)CZ}, dz.options());
  const int ntiles = (int)((ni / BI) * (nj / BJ));
  const int grid = std::min(ntiles, num_sms(dz.device().index()));
  auto dbo_part = torch::empty({grid, (long)CZ}, dz.options().dtype(torch::kFloat32));
  const uint64_t M = (uint64_t)nj * CH;
  CUtensorMap dzm = make_map<4>(dz.data_ptr(), {64, (uint64_t)nj, 2, (uint64_t)ni}, {(uint64_t)CZ, 64, (uint64_t)nj * CZ}, {64, BJ, 1, BI}, CU_TENSOR_MAP_SWIZZLE_128B, "dz");
  CUtensorMap wom = make_map<2>(wo.data_ptr(), {(uint64_t)NCH, (uint64_t)CZ}, {(uint64_t)NCH}, {64, CZ}, CU_TENSOR_MAP_SWIZZLE_128B, "Wo");
  CUtensorMap dom = make_map<4>(dO.data_ptr(), {32, (uint64_t)nj, 32, (uint64_t)ni}, {32, M, 32 * M}, {32, BJ, 4, BI}, CU_TENSOR_MAP_SWIZZLE_64B, "dO");
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(opm_dgrad_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
  opm_dgrad_sm100<<<grid, THREADS, SMEM, at::cuda::getCurrentCUDAStream()>>>((int)ni, (int)nj, ntiles, dzm, wom, dom,
      reinterpret_cast<__nv_bfloat16*>(dzp.data_ptr<at::BFloat16>()),
      reinterpret_cast<const uint32_t*>(bits.data_ptr<int>()), (int)bits.size(1), dbo_part.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dO, dzp, colsum(dbo_part, grad_bf16 ? torch::kBFloat16 : torch::kFloat32)};
}

torch::Tensor opm_dwo(torch::Tensor dzp, torch::Tensor O, int64_t ni, int64_t nj, int64_t grad_bf16) {
  using namespace dw;
  TORCH_CHECK(dzp.is_contiguous() && dzp.scalar_type() == torch::kBFloat16 && dzp.numel() == ni * nj * CZ, "dzp: bf16 [ni, nj, 128]");
  TORCH_CHECK(O.is_contiguous() && O.scalar_type() == torch::kBFloat16 && O.size(0) == ni * CH && O.size(1) == nj * CH, "O: bf16 [ni*32, nj*32]");
  TORCH_CHECK(ni % BI == 0 && nj % BJ == 0, "the kernel steps ", BI, " x ", BJ, " tokens");
  const int ntiles = (int)((ni / BI) * (nj / BJ));
  const int nsplit = std::max(1, std::min(ntiles, num_sms(O.device().index()) / NPART));
  auto part = torch::empty({nsplit, (long)CZ, (long)NCH}, O.options().dtype(torch::kFloat32));
  const uint64_t M = (uint64_t)nj * CH;
  CUtensorMap am = make_map<4>(dzp.data_ptr(), {64, (uint64_t)nj, (uint64_t)ni, 2}, {(uint64_t)CZ, (uint64_t)nj * CZ, 64}, {64, BJ, BI, 2}, CU_TENSOR_MAP_SWIZZLE_128B, "dzp");
  CUtensorMap bm = make_map<4>(O.data_ptr(), {32, (uint64_t)nj, (uint64_t)ni, 32}, {32, 32 * M, M}, {32, BJ, BI, NCB}, CU_TENSOR_MAP_SWIZZLE_64B, "O");
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(opm_dwo_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
  CUtensorMap pm = make_map<2>(part.data_ptr(), {(uint64_t)NCH, (uint64_t)nsplit * CZ}, {(uint64_t)NCH}, {32, CZ}, CU_TENSOR_MAP_SWIZZLE_128B, "part",
                               CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4);
  opm_dwo_sm100<<<dim3(nsplit, NPART), THREADS, SMEM, at::cuda::getCurrentCUDAStream()>>>((int)ni, (int)nj, ntiles, nsplit, am, bm, pm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return colsum(part, grad_bf16 ? torch::kBFloat16 : torch::kFloat32);                 // the splits, in split order
}


namespace {
// dWa = gamma . R_a + ssa_a (x) beta, dWb likewise, dgamma = sum_k Wf . R / gamma, dbeta = Wf^T ssa / gamma
// (R = [da db]^T (mask . xh): 64 x 64, ssa = the masked column sums): the H100 host formulas, in one launch
template <typename GT, typename OT>
__device__ __forceinline__ void pbwd_finalize_body(const float* __restrict__ RED, const __nv_bfloat16* __restrict__ WA,
                                                   const __nv_bfloat16* __restrict__ WB, const GT* __restrict__ GAM,
                                                   const GT* __restrict__ BET, OT* __restrict__ DWA, OT* __restrict__ DWB,
                                                   GT* __restrict__ DGAM, GT* __restrict__ DBET) {
  constexpr int CM = 64, NR = (CM + 1) * CM;
  __shared__ float sR[NR], sG[CM], sB[CM];
  __shared__ __nv_bfloat16 sW[2 * CH * CM];
  const int t = threadIdx.x;
  // stage the inputs with every load in flight at once: the loops below chained one L2 round trip per step (~16 us)
  {
    float r[(NR + 255) / 256];
#pragma unroll
    for (int u = 0; u < (NR + 255) / 256; ++u) { const int v = t + u * 256; r[u] = v < NR ? __ldcg(RED + v) : 0.f; }   // other blocks wrote it
    __nv_bfloat16 w[2 * CH * CM / 256];
#pragma unroll
    for (int u = 0; u < CH * CM / 256; ++u) { w[u] = WA[t + u * 256]; w[CH * CM / 256 + u] = WB[t + u * 256]; }
    const float gg = t < CM ? to_f(GAM[t]) : 0.f, bb = t < CM ? to_f(BET[t]) : 0.f;
#pragma unroll
    for (int u = 0; u < (NR + 255) / 256; ++u) { const int v = t + u * 256; if (v < NR) sR[v] = r[u]; }
#pragma unroll
    for (int u = 0; u < CH * CM / 256; ++u) { sW[t + u * 256] = w[u]; sW[CH * CM + t + u * 256] = w[CH * CM / 256 + u]; }
    if (t < CM) { sG[t] = gg; sB[t] = bb; }
  }
  __syncthreads();
  // RED rows 0..63: R^T [k][ce]; row 64: ssa[ce]
  for (int v = t; v < 2 * CH * CM; v += 256) {
    const int ce = v / CM, k = v % CM;
    const float r = sR[k * CM + ce], ss = sR[CM * CM + ce];
    const float val = sG[k] * r + ss * sB[k];
    if (ce < CH) DWA[ce * CM + k] = from_f<OT>(val); else DWB[(ce - CH) * CM + k] = from_f<OT>(val);
  }
  if (t < CM) {
    const int k = t;
    const float g = sG[k];
    float dg = 0.f, db = 0.f;
    for (int ce = 0; ce < 2 * CH; ++ce) {                // the same serial order as before: bit-identical
      const float wf = __bfloat162float(__float2bfloat16_rn(__bfloat162float(sW[ce * CM + k]) * g));   // exactly the bf16 Wf the kernel multiplied by
      dg += wf * sR[k * CM + ce];
      db += wf * sR[CM * CM + ce];
    }
    DGAM[k] = from_f<GT>(dg / g); DBET[k] = from_f<GT>(db / g);
  }
}

// The prologue backward's tail in ONE launch: column blocks sum the per-CTA partials [splits][65][64] in a fixed order
// (8 split groups, then the groups in order: deterministic), and the last block to finish runs the finalize.  Three
// serial latency-bound launches (a two-pass column sum and a one-block finalize) were ~15 us of the training step.
template <typename GT, typename OT>
__global__ void __launch_bounds__(256) opm_pbwd_reduce(const float* __restrict__ PART, int splits, float* __restrict__ RED, int* __restrict__ CNT,
                                                       const __nv_bfloat16* __restrict__ WA, const __nv_bfloat16* __restrict__ WB,
                                                       const GT* __restrict__ GAM, const GT* __restrict__ BET, OT* __restrict__ DWA,
                                                       OT* __restrict__ DWB, GT* __restrict__ DGAM, GT* __restrict__ DBET) {
  constexpr int W = 65 * 64, VC = W / 4;
  __shared__ float4 red[8][32];
  __shared__ int last;
  const int v4 = blockIdx.x * 32 + (threadIdx.x & 31), g = threadIdx.x >> 5;
  float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
  if (v4 < VC) {
#pragma unroll 4
    for (int k = g; k < splits; k += 8) {
      const float4 q = __ldg(reinterpret_cast<const float4*>(PART + (size_t)k * W) + v4);
      s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
    }
  }
  red[g][threadIdx.x & 31] = s;
  __syncthreads();
  if (g == 0 && v4 < VC) {
    float4 t = red[0][threadIdx.x];
#pragma unroll
    for (int k = 1; k < 8; ++k) { const float4 q = red[k][threadIdx.x]; t.x += q.x; t.y += q.y; t.z += q.z; t.w += q.w; }
    reinterpret_cast<float4*>(RED)[v4] = t;
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(CNT, 1) == (int)gridDim.x - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  pbwd_finalize_body<GT, OT>(RED, WA, WB, GAM, BET, DWA, DWB, DGAM, DBET);
  if (threadIdx.x == 0) *CNT = 0;              // ready for the next call (and the next graph replay)
}
}  // namespace

// dA, dB: [S, N*32] (a row's channels contiguous); m: [S, N, 64]; stats: [N, S, 2] from the forward prologue;
// mask: bool [S, N].  Returns dm [1, S, N, 64] and dWa, dWb [32, 64] (in wa's dtype), dgamma, dbeta [64] (in gamma's dtype).
std::vector<torch::Tensor> opm_prologue_bwd(torch::Tensor dA, torch::Tensor dB, torch::Tensor m, torch::Tensor stats, torch::Tensor mask,
                                            torch::Tensor gamma, torch::Tensor beta, torch::Tensor wa, torch::Tensor wb) {
  using namespace pb;
  TORCH_CHECK(m.is_contiguous() && m.scalar_type() == torch::kBFloat16 && m.dim() == 3 && m.size(2) == CM, "m: [S, N, 64] bf16");
  const long S = m.size(0), N = m.size(1);
  TORCH_CHECK(S % BS == 0, "S must be a multiple of ", BS);
  TORCH_CHECK(dA.is_contiguous() && dB.is_contiguous() && dA.numel() == S * N * CH && dB.numel() == S * N * CH, "dA, dB: [S, N*32]");
  TORCH_CHECK(stats.is_contiguous() && stats.scalar_type() == torch::kFloat32 && stats.numel() == S * N * 2, "stats: fp32 [N, S, 2]");
  TORCH_CHECK(mask.is_contiguous() && mask.scalar_type() == torch::kBool && mask.numel() == S * N, "mask: bool [S, N]");
  TORCH_CHECK(wa.scalar_type() == torch::kBFloat16 && wb.scalar_type() == torch::kBFloat16 && wa.is_contiguous() && wb.is_contiguous(), "wa, wb: bf16 [32, 64]");
  TORCH_CHECK(gamma.scalar_type() == beta.scalar_type() && gamma.is_contiguous() && beta.is_contiguous(), "gamma, beta");
  const bool gf = gamma.scalar_type() == torch::kFloat32;
  auto dm = torch::empty({1, S, N, (long)CM}, m.options());
  const int ntiles = (int)(N * (S / BS));
  const int grid = std::min(ntiles, num_sms(m.device().index()));
  auto part = torch::empty({grid, (long)CM + 1, (long)CM}, m.options().dtype(torch::kFloat32));
  CUtensorMap dam = make_map<3>(dA.data_ptr(), {(uint64_t)CH, (uint64_t)N, (uint64_t)S}, {(uint64_t)CH, (uint64_t)N * CH}, {CH, 1, BS}, CU_TENSOR_MAP_SWIZZLE_64B, "dA");
  CUtensorMap dbm = make_map<3>(dB.data_ptr(), {(uint64_t)CH, (uint64_t)N, (uint64_t)S}, {(uint64_t)CH, (uint64_t)N * CH}, {CH, 1, BS}, CU_TENSOR_MAP_SWIZZLE_64B, "dB");
  CUtensorMap xm = make_map<3>(m.data_ptr(), {(uint64_t)CM, (uint64_t)N, (uint64_t)S}, {(uint64_t)CM, (uint64_t)N * CM}, {64, 1, BS}, CU_TENSOR_MAP_SWIZZLE_128B, "m");
  CUtensorMap dmm = make_map<3>(dm.data_ptr(), {(uint64_t)CM, (uint64_t)N, (uint64_t)S}, {(uint64_t)CM, (uint64_t)N * CM}, {64, 1, BS}, CU_TENSOR_MAP_SWIZZLE_128B, "dm");
  CUtensorMap sm = make_map<2>(stats.data_ptr(), {(uint64_t)(2 * S), (uint64_t)N}, {(uint64_t)(2 * S)}, {2 * BS, 1}, CU_TENSOR_MAP_SWIZZLE_NONE, "stats",
                               CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4);
  CUtensorMap mm = make_map<2>(mask.data_ptr(), {(uint64_t)N, (uint64_t)S}, {(uint64_t)N}, {16, BS}, CU_TENSOR_MAP_SWIZZLE_NONE, "mask",
                               CU_TENSOR_MAP_DATA_TYPE_UINT8, 1);
  auto st = at::cuda::getCurrentCUDAStream();
  const auto* wap = reinterpret_cast<const __nv_bfloat16*>(wa.data_ptr<at::BFloat16>());
  const auto* wbp = reinterpret_cast<const __nv_bfloat16*>(wb.data_ptr<at::BFloat16>());
  auto dwa = torch::empty({(long)CH, (long)CM}, wa.options()), dwb = torch::empty({(long)CH, (long)CM}, wb.options());
  auto dgam = torch::empty({(long)CM}, gamma.options()), dbet = torch::empty({(long)CM}, beta.options());
  if (gf) {
    static bool attr = false;
    if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(opm_prologue_bwd_sm100<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
    opm_prologue_bwd_sm100<float><<<grid, THREADS, SMEM, st>>>((int)N, (int)S, ntiles, dam, dbm, xm, sm, mm, wap, wbp, gamma.data_ptr<float>(), dmm, part.data_ptr<float>());
  } else {
    static bool attr = false;
    if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(opm_prologue_bwd_sm100<__nv_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
    opm_prologue_bwd_sm100<__nv_bfloat16><<<grid, THREADS, SMEM, st>>>((int)N, (int)S, ntiles, dam, dbm, xm, sm, mm, wap, wbp,
        reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr<at::BFloat16>()), dmm, part.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto red = torch::empty({(long)CM + 1, (long)CM}, m.options().dtype(torch::kFloat32));
  static torch::Tensor cnt[16];                                      // the last-block ticket, one per device (self-resetting)
  const int dev = m.device().index();
  if (!cnt[dev].defined()) cnt[dev] = torch::zeros({1}, m.options().dtype(torch::kInt32));
  auto* dwap = reinterpret_cast<__nv_bfloat16*>(dwa.data_ptr<at::BFloat16>());
  auto* dwbp = reinterpret_cast<__nv_bfloat16*>(dwb.data_ptr<at::BFloat16>());
  const int rblocks = ((CM + 1) * CM / 4 + 31) / 32;
  if (gf) opm_pbwd_reduce<float, __nv_bfloat16><<<rblocks, 256, 0, st>>>(part.data_ptr<float>(), grid, red.data_ptr<float>(), cnt[dev].data_ptr<int>(),
      wap, wbp, gamma.data_ptr<float>(), beta.data_ptr<float>(), dwap, dwbp, dgam.data_ptr<float>(), dbet.data_ptr<float>());
  else opm_pbwd_reduce<__nv_bfloat16, __nv_bfloat16><<<rblocks, 256, 0, st>>>(part.data_ptr<float>(), grid, red.data_ptr<float>(), cnt[dev].data_ptr<int>(), wap, wbp,
      reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr<at::BFloat16>()), dwap, dwbp,
      reinterpret_cast<__nv_bfloat16*>(dgam.data_ptr<at::BFloat16>()), reinterpret_cast<__nv_bfloat16*>(dbet.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dm, dwa, dwb, dgam, dbet};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("opm_epilogue", &opm_epilogue, "sm100 fused OPM epilogue (div-norm + proj_out + bias + optional residual)",
        py::arg("O"), py::arg("norm"), py::arg("wo"), py::arg("bias"), py::arg("ni"), py::arg("nj"), py::arg("residual") = py::none());
  m.def("opm_dgrad", &opm_dgrad, "sm100 fused OPM dO: (dz @ Wo) / n straight into the grouped layout, dz/n and dbias alongside",
        py::arg("dz"), py::arg("bits"), py::arg("wo"), py::arg("ni"), py::arg("nj"), py::arg("grad_bf16") = 0);
  m.def("opm_prologue_bwd", &opm_prologue_bwd, "sm100 fused OPM prologue backward: mask, both projections and the LayerNorm in one pass");
  m.def("opm_dwo", &opm_dwo, "sm100 dWo straight off the grouped outer product, no permute", py::arg("dzp"), py::arg("O"), py::arg("ni"), py::arg("nj"), py::arg("grad_bf16") = 0);
  m.def("opm_prologue", &opm_prologue, "sm100 fused OPM prologue: LN + both projections + mask -> A2, BT (grouped layouts), mask count, LN stats [N][S]",
        py::arg("m"), py::arg("mask"), py::arg("lnw"), py::arg("lnb"), py::arg("eps"), py::arg("wa"), py::arg("wb"), py::arg("save_stats") = true, py::arg("want_norm") = false);
}
