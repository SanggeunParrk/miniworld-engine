// OPM kernels for B200 (sm_100a) at the FoldForge widths: d_msa 64 / 128, d_pair 128 / 256 / 384 (d_hidden 32), RoseTTAFold3's
// projection biases.  The same three stages and tensors as MiniWorld's kernels (opm_sm100.cu, d_msa 64 / d_pair 128):
//
//   prologue   LN + both projections (+ the projection biases, masked) + mask -> A2 [(i,c), s], BT [(j,e), s], bits, stats
//   (cuBLAS)   O = A2 . BT^T, the grouped outer product
//   epilogue   z = O permuted . Wo^T / n + bias (+ residual), one tile = 128 pairs x a 256 (c_z 256) / 192 (c_z 384) column slice
//   dgrad      dO = (dz . Wo) / n into the grouped layout, dz / n, dbias; K = c_z streamed in 64-wide Wo sub-chunks
//   dwo        dWo = (dz/n)^T . O, every c_z slice of a column part in one CTA (O is read once)
//   prologue backward (d_msa 128)   dm, R = [da db]^T (mask . xh) (the 64 / 128 channel kernel of opm_sm100.cu at d_msa 64)
//
// What changes against the d_msa 64 / d_pair 128 kernels, and why:
//   * d_msa 128: an x tile is two 128-byte-swizzled 64-channel blocks (one 4-D TMA box), the projection runs K = 128 in eight k
//     steps; the LayerNorm keeps two threads per row (64 channels each); one CTA per SM (shared memory), four x stages.
//   * the projection biases ride the drain: the LayerNorm warps also leave the tile's mask bits in a small shared ring, and the
//     drain adds b[ch] to the columns (MSA rows) whose bit is set -- a masked row keeps its exact zero, as the module's (proj + b) * mask.
//   * c_z > 128: the epilogue's accumulator is c_z wide (256) or half of it (192; TMEM holds two of them), so an O stage serves
//     all of a slice's columns; dgrad streams Wo in [64 z x 128 n] sub-chunks against a dz tile kept in a ring of 128-z units;
//     dwo splits TMEM between the 128-wide c_z slices of its column part.
#include <torch/extension.h>
#include "sm100.cuh"

using namespace sm100;

namespace {
constexpr int CH = 32, NCH = CH * CH;

// ---------------------------------------------------------------------------------------------
// PROLOGUE at d_msa CM (64 / 128).  The structure of opm_prologue_sm100: warp 0 TMA, warp 1 MMA (D[ch][s] = W[ch][:] . y[s][:]),
// warps 2-9 LayerNorm (two threads a row), warps 10-13 drain to the grouped layouts.
template <int CM>
struct WPro {
  static constexpr int BS = 128, KB = CM / 64;
  static constexpr int NST = CM == 64 ? 3 : 4;
  static constexpr int NACC = 2;
  static constexpr int CTAS = CM == 64 ? 2 : 1;
  static constexpr int XT = BS * CM;
  static constexpr int TT = 2 * 64 * 64;
  static constexpr int THREADS = 448;
  static constexpr int MT = BS * 16;
  static constexpr int NRING = 8;                     // mask-bit ring (tiles): > NST + NACC, the LayerNorm's lead over the drain
  static constexpr int SMEM = 1024 + (NST * XT + NACC * TT + 128 * CM) * 2 + NST * MT + 2 * CM * 4 + 64 * 4 + NRING * 16 + 512;
  static constexpr int TMEM_COLS = NACC * 128;
  static constexpr uint32_t IDESC = idesc_bf16(128, BS);
  static_assert(CTAS * (SMEM + 1024) <= 233472, "CTAS per SM");
  static_assert(NRING >= NST + NACC + 1, "mask-bit ring depth");
};

template <int CM, typename WT>
__global__ void __launch_bounds__(448, WPro<CM>::CTAS) opm_wprologue_sm100(
    int N, int S, int ntiles, float eps,
    const __grid_constant__ CUtensorMap xmap,      // m [S][N][CM] as (64, N, S, CM / 64), box (64, 1, 128, CM / 64), 128B swizzle
    const __grid_constant__ CUtensorMap amap,      // A2 [N*32][S], box (64, 32), 128B swizzle
    const __grid_constant__ CUtensorMap bmap,      // BT [N*32][S]
    const __grid_constant__ CUtensorMap mmap,      // mask [S][N] bytes, box (16, 128)
    const __grid_constant__ CUtensorMap wamap,     // Wa [32][CM], box (64, 32), 128B swizzle
    const __grid_constant__ CUtensorMap wbmap,     // Wb [32][CM]
    const WT* __restrict__ LNW, const WT* __restrict__ LNB,
    const float* __restrict__ PB,                  // the projection biases [left 32 | right 32] (fp32), or nullptr
    float2* __restrict__ STATS,                    // [N][S] (mean, rstd) or nullptr
    uint32_t* __restrict__ BITS) {                 // [N][S/32]
  using C = WPro<CM>;
  constexpr int BS = C::BS, NST = C::NST, NACC = C::NACC, XT = C::XT, TT = C::TT, MT = C::MT, KB = C::KB, NRING = C::NRING;
  const int NSB = S / BS;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sX = reinterpret_cast<__nv_bfloat16*>(smb);   // [NST][KB][128 s][64] swizzled: x, then y in place (MMA B operand)
  __nv_bfloat16* sT = sX + NST * XT;                             // [NACC][2 halves][64 ch][64 s] swizzled
  __nv_bfloat16* sW = sT + NACC * TT;                            // [KB][128 m][64 k] swizzled: Wa, Wb, Wa, Wb (32 rows each)
  uint8_t* sM = reinterpret_cast<uint8_t*>(sW + 128 * CM);       // [NST][128 s][16 tokens] mask bytes
  float* sLN = reinterpret_cast<float*>(sM + NST * MT);          // gamma[CM], beta[CM]
  float* sPB = sLN + 2 * CM;                                     // projection biases [64] (zeros without)
  uint16_t* sRB = reinterpret_cast<uint16_t*>(sPB + 64);         // [NRING][8 warps] the tile's mask bits, 16 rows a warp
  uint64_t* bars = reinterpret_cast<uint64_t*>(sRB + NRING * 8);
  uint64_t* xfull = bars;
  uint64_t* yfull = xfull + NST;
  uint64_t* xempty = yfull + NST;
  uint64_t* accf = xempty + NST;
  uint64_t* acce = accf + NACC;
  uint64_t* wful = acce + NACC;
  uint32_t* tslot = reinterpret_cast<uint32_t*>(wful + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const bool has_bias = PB != nullptr;
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { bar_init(xfull + s, 1); bar_init(yfull + s, 8); bar_init(xempty + s, 1); }
    for (int b = 0; b < NACC; ++b) { bar_init(accf + b, 1); bar_init(acce + b, 4); }
    bar_init(wful, 1);
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, C::TMEM_COLS);
  if (tid < CM) { sLN[tid] = to_f(LNW[tid]); sLN[CM + tid] = to_f(LNB[tid]); }
  if (tid < 64) sPB[tid] = has_bias ? PB[tid] : 0.f;
  fence_proxy_async();
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;

  if (warp == 0) {
    if (lane == 0) {
      expect_tx(wful, 128 * CM * 2);             // the stacked weight: Wa, Wb, Wa, Wb, by TMA, per 64-channel k block
      for (int kb = 0; kb < KB; ++kb)
        for (int k = 0; k < 4; ++k) load_2d((k & 1) ? &wbmap : &wamap, sW + kb * 128 * 64 + k * 32 * 64, wful, kb * 64, 0);
      int g = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++g) {
        const int i = t / NSB, s0 = (t % NSB) * BS;
        const int st = g % NST;
        if (g >= NST) wait(xempty + st, ((g / NST) - 1) & 1);
        expect_tx(xfull + st, XT * 2 + MT);
        load_4d(&xmap, sX + st * XT, xfull + st, 0, i, s0, 0);
        load_2d(&mmap, sM + st * MT, xfull + st, i & ~15, s0);
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
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
          mma_ss(tmem + b * BS, desc_k128(sW + (ks >> 2) * 128 * 64 + (ks & 3) * 16), desc_k128(y + (ks >> 2) * BS * 64 + (ks & 3) * 16),
                 C::IDESC, ks ? 1u : 0u);
        mma_commit(accf + b);
        mma_commit(xempty + st);
      }
    }
  } else if (warp <= 9) {
    // ---------------- LayerNorm: row r, channel half hf (lanes 0-15 / 16-31 of a warp share 16 rows), CM / 2 channels each ----------------
    constexpr int NC = CM / 16;                  // 16-byte chunks per thread: 4 (d_msa 64) or 8 (d_msa 128, one whole k block)
    const int r = (warp - 2) * 16 + (lane & 15), hf = lane >> 4;
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int i = t / NSB, s0 = (t % NSB) * BS, st = lt % NST;
      wait(xfull + st, (lt / NST) & 1);
      const bool mkb = sM[st * MT + r * 16 + (i & 15)] != 0;
      __nv_bfloat16* xr = sX + st * XT + (KB == 1 ? 0 : hf * BS * 64) + r * 64;
      float2 x[NC * 4];
#pragma unroll
      for (int k = 0; k < NC; ++k) {
        const int c8 = KB == 1 ? hf * 4 + k : k;
        const uint4 u = *reinterpret_cast<const uint4*>(xr + ((c8 ^ (r & 7)) << 3));
        x[k * 4 + 0] = bf2f(u.x); x[k * 4 + 1] = bf2f(u.y); x[k * 4 + 2] = bf2f(u.z); x[k * 4 + 3] = bf2f(u.w);
      }
      float2 sa = make_float2(0.f, 0.f), sb = sa;
#pragma unroll
      for (int k = 0; k < NC * 4; k += 2) { sa = add2(sa, x[k]); sb = add2(sb, x[k + 1]); }
      const float2 sm = add2(sa, sb);
      float sum = sm.x + sm.y;
      sum += __shfl_xor_sync(0xffffffffu, sum, 16);
      const float mean = sum * (1.f / CM);
      const float2 nm = make_float2(-mean, -mean);
      float2 va = make_float2(0.f, 0.f), vb = va;
#pragma unroll
      for (int k = 0; k < NC * 4; k += 2) {
        x[k] = add2(x[k], nm); x[k + 1] = add2(x[k + 1], nm);
        va = fma2(x[k], x[k], va); vb = fma2(x[k + 1], x[k + 1], vb);
      }
      const float2 vs = add2(va, vb);
      float var = vs.x + vs.y;
      var += __shfl_xor_sync(0xffffffffu, var, 16);
      const float rstd = 1.f / sqrtf(var * (1.f / CM) + eps);
      const float2 rs = make_float2(rstd, rstd);
      const uint32_t keep = mkb ? 0xffffffffu : 0u;
#pragma unroll
      for (int k = 0; k < NC; ++k) {
        const int c8 = KB == 1 ? hf * 4 + k : k;
        const int cb = KB == 1 ? c8 * 8 : hf * 64 + k * 8;     // this chunk's first channel
        const float4 g0 = *reinterpret_cast<const float4*>(sLN + cb), g1 = *reinterpret_cast<const float4*>(sLN + cb + 4);
        const float4 b0 = *reinterpret_cast<const float4*>(sLN + CM + cb), b1 = *reinterpret_cast<const float4*>(sLN + CM + cb + 4);
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
      if (lane == 0) {
        reinterpret_cast<uint16_t*>(BITS)[((size_t)i * S + s0 + r) / 16] = (uint16_t)(bal & 0xffffu);
        sRB[(lt % NRING) * 8 + (warp - 2)] = (uint16_t)(bal & 0xffffu);   // for the drain's bias mask (read after the accumulator)
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(yfull + st);
    }
  } else {
    // ---------------- drain (barrier 2): TMEM lane = channel; q & 1 picks a | b, q >> 1 the s half ----------------
    const int q = warp & 3;
    const int ch = (q & 1) * 32 + lane, sh = q >> 1;
    const bool leader = (warp == 10 && lane == 0);
    const float pb = sPB[ch];
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int i = t / NSB, s0 = (t % NSB) * BS, b = lt % NACC;
      wait(accf + b, (lt / NACC) & 1);
      tc_fence_after();
      // the mask bits of this half's 64 rows (written by the LayerNorm warps before the GEMM that filled this accumulator was issued)
      uint32_t mb0 = 0, mb1 = 0;
      if (has_bias) {
        const uint32_t* rb = reinterpret_cast<const uint32_t*>(sRB + (lt % NRING) * 8);
        mb0 = rb[sh * 2]; mb1 = rb[sh * 2 + 1];
      }
      if (leader) bulk_wait_read<NACC - 1>();
      named_sync(2, 128);
      __nv_bfloat16* row = sT + b * TT + (sh * 64 + ch) * 64;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float acc[32];
        tmem_ld32(tmem_at(tmem + b * BS, q * 32, sh * 64 + h * 32), acc);
        tmem_wait_ld();
        if (h == 1) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(acce + b); }
        if (has_bias) {
          const uint32_t mb = h ? mb1 : mb0;
#pragma unroll
          for (int k = 0; k < 32; ++k) acc[k] += ((mb >> k) & 1u) ? pb : 0.f;
        }
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
  if (warp == 1) tmem_dealloc(tmem, C::TMEM_COLS);
}

// ---------------------------------------------------------------------------------------------
// EPILOGUE at c_z = 128 / 256 / 384: opm_epilogue_sm100's tile (128 pairs, K = 1024 in 16 stages of two c values) with an
// accumulator NW = 256 (c_z 256) or 192 (c_z 384, two slices) columns wide, so a stage's O chunk serves every output column of
// the slice at once: the per-tile traffic is the O tile plus the slice of Wo, and O is read once (c_z 256) or twice (c_z 384,
// the two slices of a pair tile on neighbouring CTAs at the same time: the second read is an L2 hit).  Two accumulators of NW
// columns (TMEM 512 / 384), the drain of one under the main loop of the other.
template <int NW>
struct WEp {
  static constexpr int BI = 4, BJ = 32, NP = BI * BJ;
  static constexpr int KC = 64, NK = NCH / KC;
  static constexpr int ATILE = NP * KC, BTILE = NW * KC, STAGE = ATILE + BTILE;
  static constexpr int NST = NW == 256 ? 3 : 4;
  static constexpr int OUTT = NP * NW;                       // staging: [BI][NW / 64 blocks][BJ][64]
  static constexpr int THREADS = 192;
  static constexpr int SMEM = 1024 + (NST * STAGE + OUTT) * 2 + 256;
  static constexpr uint32_t IDESC = idesc_bf16(128, NW);
  static constexpr int TMEM_COLS = 2 * NW <= 256 ? 256 : 512;
  static_assert(SMEM <= 232448, "one CTA per SM");
};

template <int NW>
__global__ void __launch_bounds__(192, 1) opm_wepilogue_sm100(
    int NI, int NJ, int ntiles, int NZB,
    const __grid_constant__ CUtensorMap amap,     // O as (e 32, j NJ, i NI, c 32), box (32, BJ, BI, 2), 64B swizzle
    const __grid_constant__ CUtensorMap bmap,     // Wo [CZ][NCH], box (64, NW)
    const __grid_constant__ CUtensorMap zmap,     // z as (64 z, NJ j, CZ / 64, NI i), box (64, BJ, NW / 64, BI)
    const __grid_constant__ CUtensorMap rmap,     // the residual, same layout
    const uint32_t* __restrict__ BITS, int W,     // the prologue's bit mask [N][W = S/32]
    const void* __restrict__ BIAS, int bias_bf16, int has_res, int after) {
  using C = WEp<NW>;
  constexpr int BI = C::BI, BJ = C::BJ, NP = C::NP, KC = C::KC, NK = C::NK, ATILE = C::ATILE, STAGE = C::STAGE, NST = C::NST;
  constexpr int OUTT = C::OUTT, NB64 = NW / 64;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sStage = reinterpret_cast<__nv_bfloat16*>(smb);
  __nv_bfloat16* sOut = sStage + NST * STAGE;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sOut + OUTT);
  uint64_t* full = bars;
  uint64_t* empty = full + NST;
  uint64_t* accf = empty + NST;
  uint64_t* acce = accf + 2;
  uint64_t* resf = acce + 2;
  uint32_t* tslot = reinterpret_cast<uint32_t*>(resf + 1);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { bar_init(full + s, 1); bar_init(empty + s, 1); }
    for (int b = 0; b < 2; ++b) { bar_init(accf + b, 1); bar_init(acce + b, 4); }
    bar_init(resf, 1);
    bar_init_fence();
    prefetch_map(&amap); prefetch_map(&bmap); prefetch_map(&zmap); if (has_res) prefetch_map(&rmap);
  }
  if (warp == 1) tmem_alloc(tslot, C::TMEM_COLS);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  const int njb = NJ / BJ;

  if (warp == 0) {
    if (lane == 0) {
      int g = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
        const int pt = t / NZB, zb = t - pt * NZB;
        const int i0 = (pt / njb) * BI, j0 = (pt % njb) * BJ;
        for (int kc = 0; kc < NK; ++kc, ++g) {
          const int s = g % NST;
          if (g >= NST) wait(empty + s, ((g / NST) - 1) & 1);
          __nv_bfloat16* st = sStage + s * STAGE;
          expect_tx(full + s, STAGE * 2);
          load_4d(&amap, st, full + s, 0, j0, i0, kc * 2);
          load_2d(&bmap, st + ATILE, full + s, kc * KC, zb * NW);
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
        const uint32_t d = tmem + b * NW;
        for (int kc = 0; kc < NK; ++kc, ++g) {
          const int s = g % NST;
          wait(full + s, (g / NST) & 1);
          tc_fence_after();
          const __nv_bfloat16* st = sStage + s * STAGE;
#pragma unroll
          for (int ks = 0; ks < KC / 16; ++ks)
            mma_ss(d, desc_k64(st + (ks >> 1) * (NP * 32) + (ks & 1) * 16), desc_k128(st + ATILE + ks * 16), C::IDESC, (kc | ks) ? 1u : 0u);
          mma_commit(empty + s);
        }
        mma_commit(accf + b);
      }
    }
  } else {
    const int q = warp & 3;
    const int p = q * 32 + lane;
    const int il = p / BJ, jl = p % BJ;
    const bool leader = (warp == 2 && lane == 0);
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int pt = t / NZB, zb = t - pt * NZB;
      const int i0 = (pt / njb) * BI, j0 = (pt % njb) * BJ;
      const int b = lt & 1;
      if (leader) {
        bulk_wait_read<0>();
        if (has_res) {
          expect_tx(resf, OUTT * 2);
          load_4d(&rmap, sOut, resf, 0, j0, zb * NB64, i0);
        }
      }
      const uint4* bi = reinterpret_cast<const uint4*>(BITS + (size_t)(i0 + il) * W);
      const uint4* bj = reinterpret_cast<const uint4*>(BITS + (size_t)(j0 + jl) * W);
      int cnt = 0;
      for (int w = 0; w < W / 4; ++w) {
        const uint4 a = __ldg(bi + w), c = __ldg(bj + w);
        cnt += __popc(a.x & c.x) + __popc(a.y & c.y) + __popc(a.z & c.z) + __popc(a.w & c.w);
      }
      const float inv = 1.0f / (float)max(cnt, 1);
      wait(accf + b, (lt >> 1) & 1);
      tc_fence_after();
      if (has_res) wait(resf, lt & 1);
      named_sync(1, 128);
      const uint32_t tb = tmem_at(tmem + b * NW, q * 32, 0);
#pragma unroll 1
      for (int h = 0; h < NB64; ++h) {
        float v[64];
        tmem_ld32(tb + h * 64, v);
        tmem_ld32(tb + h * 64 + 32, v + 32);
        tmem_wait_ld();
        if (h == NB64 - 1) { tc_fence_before(); if (lane == 0) arrive(acce + b); }
        __nv_bfloat16* row = sOut + ((il * NB64 + h) * BJ + jl) * 64;
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
            const int n = zb * NW + h * 64 + c8 * 8 + u * 2;
            const float2 bb = bias_bf16 ? __bfloat1622float2(__ldg(reinterpret_cast<const __nv_bfloat162*>(BIAS) + (n >> 1)))
                                        : __ldg(reinterpret_cast<const float2*>(BIAS) + (n >> 1));
            const float a0 = v[c8 * 8 + u * 2], a1 = v[c8 * 8 + u * 2 + 1];
            __nv_bfloat162 x = after ? __float22bfloat162_rn(make_float2((a0 + bb.x) * inv, (a1 + bb.y) * inv))
                                     : __float22bfloat162_rn(make_float2(fmaf(a0, inv, bb.x), fmaf(a1, inv, bb.y)));
            if (has_res) x = __hadd2(x, rh[u]);
            ow[u] = *reinterpret_cast<uint32_t*>(&x);
          }
          *reinterpret_cast<uint4*>(row + sw) = o;
        }
      }
      fence_proxy_async();
      named_sync(1, 128);
      if (leader) { store_4d(&zmap, sOut, 0, j0, zb * NB64, i0); bulk_commit(); }
    }
    if (leader) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, C::TMEM_COLS);
}

// ---------------------------------------------------------------------------------------------
// DGRAD at c_z = CZ: dO[(i,c),(j,e)] = (sum_z dz[i,j,z] Wo[z,(c,e)]) / n[i,j] in the grouped layout, dz/n, dbias = sum dz.
// opm_dgrad_sm100's roles and drain; K = c_z no longer fits one stage, so a pair tile's dz stays in a ring of 128-z units (each the
// [2 halves][128 pairs][64 z] tile of the c_z 128 kernel) for the tile's eight 128-column chunks, and Wo streams as [64 z x 128 n]
// sub-chunks.  A unit is released by the commit after the tile's last chunk read it and by the eight drain warps (dz/n); the ring
// holds one unit more than a tile, so the next tile's first unit lands under the current tile's last chunk.
template <int CZ>
struct WDg {
  static constexpr int BI = 4, BJ = 32, NP = BI * BJ;
  static constexpr int NC = 128, NCK = NCH / NC;
  static constexpr int NZB = CZ / 128, NZS = CZ / 64;
  static constexpr int UNIT = NP * 128;              // elements
  static constexpr int NB = NZB + 1;
  static constexpr int WOS = 128 * 64;               // [2 n blocks][64 z][64 n]
  static constexpr int NWO = 3;
  static constexpr int OUT = NP * NC;
  static constexpr int NOUT = CZ >= 384 ? 1 : 2;
  static constexpr int NACC = 3;
  static constexpr int ONES = 16 * NP;
  static constexpr int THREADS = 352;
  static constexpr int SMEM = 1024 + (NB * UNIT + NWO * WOS + NOUT * OUT + ONES) * 2 + 512;
  static constexpr uint32_t IDESC = idesc_bf16(128, NC, 0, 1);
  static constexpr uint32_t IDESC_BO = idesc_bf16(128, 16, 1, 0);
  static constexpr int BO_COL = NACC * NC;
  static_assert(SMEM <= 232448, "one CTA per SM");
  static_assert(BO_COL + NZB * 16 <= 512, "TMEM columns");
};

template <int CZ>
__global__ void __launch_bounds__(352, 1) opm_wdgrad_sm100(
    int NI, int NJ, int ntiles,
    const __grid_constant__ CUtensorMap dzmap,    // dz as (64 z, NJ j, CZ / 64, NI i), box (64, BJ, 1, BI), 128B swizzle
    const __grid_constant__ CUtensorMap womap,    // Wo [CZ][NCH] as (NCH n, CZ z), box (64 n, 64 z), 128B swizzle
    const __grid_constant__ CUtensorMap domap,    // dO as (32 e, NJ j, 32 c, NI i), box (32, BJ, 4, BI), 64B swizzle
    __nv_bfloat16* __restrict__ DZP,              // dz/n [NI][NJ][CZ]
    const uint32_t* __restrict__ BITS, int W, float* __restrict__ DBO) {
  using C = WDg<CZ>;
  constexpr int BJ = C::BJ, BI = C::BI, NP = C::NP, NC = C::NC, NCK = C::NCK, NZB = C::NZB, NZS = C::NZS, UNIT = C::UNIT, NB = C::NB;
  constexpr int WOS = C::WOS, NWO = C::NWO, OUT = C::OUT, NOUT = C::NOUT, NACC = C::NACC, BO_COL = C::BO_COL;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sDZ = reinterpret_cast<__nv_bfloat16*>(smb);
  __nv_bfloat16* sWO = sDZ + NB * UNIT;
  __nv_bfloat16* sOUT = sWO + NWO * WOS;
  __nv_bfloat16* sONE = sOUT + NOUT * OUT;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sONE + C::ONES);
  uint64_t* dzf = bars;              // [NB] unit landed
  uint64_t* dze = dzf + NB;          // [NB] count 9: the tile's last chunk read it (commit) and the 8 drain warps took dz/n
  uint64_t* wof = dze + NB;          // [NWO]
  uint64_t* woe = wof + NWO;         // [NWO]
  uint64_t* accf = woe + NWO;        // [NACC]
  uint64_t* acce = accf + NACC;      // [NACC] count 8
  uint64_t* bod = acce + NACC;       // [1]
  uint32_t* tslot = reinterpret_cast<uint32_t*>(bod + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int k = 0; k < NB; ++k) { bar_init(dzf + k, 1); bar_init(dze + k, 9); }
    for (int k = 0; k < NWO; ++k) { bar_init(wof + k, 1); bar_init(woe + k, 1); }
    for (int k = 0; k < NACC; ++k) { bar_init(accf + k, 1); bar_init(acce + k, 8); }
    bar_init(bod, 1);
    bar_init_fence();
  }
  for (int v = tid; v < C::ONES / 8; v += C::THREADS) {
    const int row = (v / 8) % 16;
    uint4 o = make_uint4(0u, 0u, 0u, 0u);
    if (row == 0) o = make_uint4(0x3f803f80u, 0x3f803f80u, 0x3f803f80u, 0x3f803f80u);
    reinterpret_cast<uint4*>(sONE)[v] = o;
  }
  if (warp == 1) tmem_alloc(tslot, 512);
  fence_proxy_async();
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  const int njb = NJ / BJ;

  if (warp == 10) {
    if (lane == 0) {                             // the dz units
      int u = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
        const int i0 = (t / njb) * BI, j0 = (t % njb) * BJ;
        for (int zu = 0; zu < NZB; ++zu, ++u) {
          const int d = u % NB;
          if (u >= NB) wait(dze + d, ((u / NB) - 1) & 1);
          expect_tx(dzf + d, UNIT * 2);
          for (int h = 0; h < 2; ++h) load_4d(&dzmap, sDZ + d * UNIT + h * NP * 64, dzf + d, 0, j0, zu * 2 + h, i0);
        }
      }
    }
  } else if (warp == 0) {
    if (lane == 0) {                             // the Wo sub-chunks (L2-resident)
      int g = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x)
        for (int k = 0; k < NCK; ++k)
          for (int zs = 0; zs < NZS; ++zs, ++g) {
            const int w = g % NWO;
            if (g >= NWO) wait(woe + w, ((g / NWO) - 1) & 1);
            expect_tx(wof + w, WOS * 2);
            for (int b = 0; b < 2; ++b) load_2d(&womap, sWO + w * WOS + b * 64 * 64, wof + w, k * NC + b * 64, zs * 64);
          }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      int lt = 0, g = 0, gw = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int u0 = lt * NZB;
        for (int k = 0; k < NCK; ++k, ++g) {
          const int a = g % NACC;
          if (g >= NACC) wait(acce + a, ((g / NACC) - 1) & 1);
          tc_fence_after();
          for (int zs = 0; zs < NZS; ++zs, ++gw) {
            const int w = gw % NWO, d = (u0 + (zs >> 1)) % NB;
            if (k == 0 && !(zs & 1)) {           // a unit's first use: wait for it, and dbias += dz^T . 1 over it (raw dz)
              const int zu = zs >> 1;
              wait(dzf + d, ((u0 + zu) / NB) & 1);
              tc_fence_after();
              const __nv_bfloat16* du = sDZ + d * UNIT;
#pragma unroll
              for (int kk = 0; kk < NP / 16; ++kk)
                mma_ss(tmem + BO_COL + zu * 16, desc_mn128(du + kk * 16 * 64, NP * 64 * 2),
                       desc_k128(sONE + (kk >> 2) * 16 * 64 + (kk & 3) * 16), C::IDESC_BO, (lt | kk) ? 1u : 0u);
            }
            wait(wof + w, (gw / NWO) & 1);
            tc_fence_after();
            const __nv_bfloat16* dz = sDZ + d * UNIT + (zs & 1) * NP * 64;
            const __nv_bfloat16* wo = sWO + w * WOS;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
              mma_ss(tmem + a * NC, desc_k128(dz + kk * 16), desc_mn128(wo + kk * 16 * 64, 64 * 64 * 2), C::IDESC, (zs | kk) ? 1u : 0u);
            mma_commit(woe + w);
            if (k == NCK - 1 && (zs & 1)) mma_commit(dze + d);   // the tile's last chunk has read this unit
          }
          mma_commit(accf + a);
        }
      }
      mma_commit(bod);
    }
  } else {
    const int q = warp & 3, grp = (warp - 2) >> 2;
    const int p = q * 32 + lane, il = p / BJ, jl = p % BJ;
    const bool leader = (warp == 2 && lane == 0);
    int lt = 0, g = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int i0 = (t / njb) * BI, j0 = (t % njb) * BJ, u0 = lt * NZB;
      const uint4* bi = reinterpret_cast<const uint4*>(BITS + (size_t)(i0 + il) * W);
      const uint4* bj = reinterpret_cast<const uint4*>(BITS + (size_t)(j0 + jl) * W);
      int cnt = 0;
      for (int w = 0; w < W / 4; ++w) {
        const uint4 x = __ldg(bi + w), y = __ldg(bj + w);
        cnt += __popc(x.x & y.x) + __popc(x.y & y.y) + __popc(x.z & y.z) + __popc(x.w & y.w);
      }
      const float inv = 1.0f / (float)max(cnt, 1);
      const float2 inv2 = make_float2(inv, inv);
      uint4* dst = reinterpret_cast<uint4*>(DZP + ((size_t)(i0 + il) * NJ + j0 + jl) * CZ);
      for (int zu = 0; zu < NZB; ++zu) {         // dz/n for dWo: this warp's 64 z of each unit, straight to global
        const int d = (u0 + zu) % NB;
        wait(dzf + d, ((u0 + zu) / NB) & 1);
        const __nv_bfloat16* rp = sDZ + d * UNIT + grp * NP * 64 + p * 64;
#pragma unroll
        for (int c8 = 0; c8 < 8; ++c8) {
          uint4 u = *reinterpret_cast<const uint4*>(rp + ((c8 ^ (p & 7)) << 3));
          const float2 x0 = mul2(bf2f(u.x), inv2), x1 = mul2(bf2f(u.y), inv2), x2 = mul2(bf2f(u.z), inv2), x3 = mul2(bf2f(u.w), inv2);
          u.x = pack2(x0.x, x0.y); u.y = pack2(x1.x, x1.y); u.z = pack2(x2.x, x2.y); u.w = pack2(x3.x, x3.y);
          dst[zu * 16 + grp * 8 + c8] = u;
        }
        __syncwarp();
        if (lane == 0) arrive(dze + d);
      }
      for (int k = 0; k < NCK; ++k, ++g) {
        const int a = g % NACC, ob = g % NOUT;
        wait(accf + a, (g / NACC) & 1);
        tc_fence_after();
        if (leader) bulk_wait_read<NOUT - 1>();
        named_sync(1, 256);
        __nv_bfloat16* box = sOUT + ob * OUT;
#pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
          const int cl = grp * 2 + cc;
          float v[32];
          tmem_ld32(tmem_at(tmem + a * NC, q * 32, cl * 32), v);
          tmem_wait_ld();
          if (cc == 1) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(acce + a); }
          const int row = (il * 4 + cl) * BJ + jl;
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
    // this CTA's dbias partial: TMEM lane = z of a unit, column BO_COL + 16 unit
    wait(bod, 0);
    tc_fence_after();
#pragma unroll 1
    for (int zu = 0; zu < NZB; ++zu) {
      float v[8];
      tmem_ld8(tmem_at(tmem + BO_COL + zu * 16, q * 32, 0), v);
      tmem_wait_ld();
      if (grp == 0) DBO[(size_t)blockIdx.x * CZ + zu * 128 + q * 32 + lane] = v[0];
    }
    if (leader) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}

// ---------------------------------------------------------------------------------------------
// dWo at c_z = CZ: dWo[z,(c,e)] = sum_(i,j) (dz/n)[i,j,z] O[(i,c),(j,e)], off the grouped O.  opm_dwo_sm100 with M = c_z split into
// 128-lane slices that all live in one CTA's TMEM (NZB x NHALF columns), so a column part reads its O columns once; the parts
// re-read dz/n (L2).  K splits leave fp32 partials summed in split order (deterministic).
template <int CZ>
struct WDw {
  static constexpr int BI = 2, BJ = 32, NP = BI * BJ;
  static constexpr int NZB = CZ / 128;
  static constexpr int NPART = NZB <= 2 ? 4 : 8;
  static constexpr int NHALF = NCH / NPART;          // 256 / 128 columns
  static constexpr int NCB = NHALF / 32;
  static constexpr int AT = NP * CZ;                 // dz/n tile [CZ / 64 z blocks][64 pairs][64 z]
  static constexpr int BT = NP * NHALF;              // O tile [NCB c][64 pairs][32 e]
  static constexpr int STAGE = AT + BT;
  static constexpr int NST = 3;
  static constexpr int THREADS = 192;
  static constexpr int SMEM = 1024 + NST * STAGE * 2 + 256;
  static constexpr int MN = NHALF < 256 ? NHALF : 256;
  static constexpr uint32_t IDESC = idesc_bf16(128, MN, 1, 1);
  static_assert(SMEM <= 232448, "one CTA per SM");
  static_assert(NZB * NHALF <= 512 && NHALF % 128 == 0, "TMEM columns");
};

template <int CZ>
__global__ void __launch_bounds__(192, 1) opm_wdwo_sm100(
    int NI, int NJ, int ntiles, int nsplit,
    const __grid_constant__ CUtensorMap amap,     // dz/n as (64 z, NJ j, NI i, CZ / 64), box (64, BJ, BI, CZ / 64), 128B swizzle
    const __grid_constant__ CUtensorMap bmap,     // O    as (32 e, NJ j, NI i, 32 c), box (32, BJ, BI, NCB), 64B swizzle
    const __grid_constant__ CUtensorMap pmap) {   // the partials [nsplit * CZ][NCH] fp32, box (32, 128), 128B swizzle
  using C = WDw<CZ>;
  constexpr int BI = C::BI, BJ = C::BJ, NP = C::NP, NZB = C::NZB, NHALF = C::NHALF, NCB = C::NCB, AT = C::AT, STAGE = C::STAGE;
  constexpr int NST = C::NST, MN = C::MN;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sStage = reinterpret_cast<__nv_bfloat16*>(smb);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sStage + NST * STAGE);
  uint64_t* full = bars;
  uint64_t* empty = full + NST;
  uint64_t* done = empty + NST;
  uint32_t* tslot = reinterpret_cast<uint32_t*>(done + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int part = blockIdx.y, split = blockIdx.x;
  const int k0 = (int)((long)ntiles * split / nsplit), k1 = (int)((long)ntiles * (split + 1) / nsplit);
  const int njb = NJ / BJ;
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { bar_init(full + s, 1); bar_init(empty + s, 1); }
    bar_init(done, 1);
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, 512);
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
        load_4d(&bmap, st + AT, full + s, 0, j0, i0, part * NCB);
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
          for (int zb = 0; zb < NZB; ++zb)
#pragma unroll
            for (int h = 0; h < NHALF / MN; ++h)
              mma_ss(tmem + zb * NHALF + h * MN, desc_mn128(a + zb * 2 * NP * 64 + ks * 16 * 64, NP * 64 * 2),
                     sdesc(sa(b + h * (MN / 32) * NP * 32 + ks * 16 * 32), NP * 32 * 2, 512, 4), C::IDESC, (g | ks) ? 1u : 0u);
        mma_commit(empty + s);
      }
      mma_commit(done);
    }
  } else {
    const int q = warp & 3, z = q * 32 + lane;
    const bool leader = (warp == 2 && lane == 0);
    wait(done, 0);
    tc_fence_after();
    const bool any = k1 > k0;
    float* stg = reinterpret_cast<float*>(sStage);                 // 4 tiles of [128][32] fp32 (16 KiB each)
#pragma unroll 1
    for (int zb = 0; zb < NZB; ++zb) {
#pragma unroll 1
      for (int c0 = 0; c0 < NHALF; c0 += 32) {
        const int bufi = (c0 / 32) & 3;
        if (bufi == 0) { if (leader) bulk_wait_read<0>(); named_sync(1, 128); }
        float v[32];
        tmem_ld32(tmem_at(tmem + zb * NHALF, q * 32, c0), v);
        tmem_wait_ld();
        float* row = stg + bufi * 128 * 32 + z * 32;
#pragma unroll
        for (int k = 0; k < 8; ++k)
          *reinterpret_cast<float4*>(row + ((k ^ (z & 7)) << 2)) = any ? make_float4(v[4 * k], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3]) : make_float4(0.f, 0.f, 0.f, 0.f);
        if (bufi == 3) {
          fence_proxy_async();
          named_sync(1, 128);
          if (leader) {
            for (int bb = 0; bb < 4; ++bb) store_2d(&pmap, stg + bb * 128 * 32, part * NHALF + c0 - 96 + bb * 32, split * CZ + zb * 128);
            bulk_commit();
          }
        }
      }
    }
    if (leader) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}

// ---------------------------------------------------------------------------------------------
// PROLOGUE BACKWARD at d_msa 128: per MSA row (s, i), with Wf = [gamma . Wa ; gamma . Wb] (64 x 128)
//     g   = mask . ([da db] . Wf)          (dL/dxh)
//     dm  = rstd (g - mean(g) - xh mean(g xh))
//     R  += (mask . xh)^T [da db]           (dWa, dWb, dgamma with the masked column sums ssa of [da db], taken by the host)
// opm_prologue_bwd_sm100's tile (one token x 128 MSA rows) and roles.  A row is 128 channels, which a thread cannot keep with its
// TMEM gradient row, so the eight compute warps form ONE group: warps q and q + 4 take the same 32 rows, each 64 channels, and
// combine their two row sums through shared memory; the dy accumulator is double-buffered, so the next tile's dy GEMM runs
// under this tile's compute.  The dW GEMM's A operand (mask . xh) is exactly M = 128 channels: the masked column sums leave the
// kernel (the host's GEMV over dA / dB, which also gives RoseTTAFold3's projection-bias gradients).
namespace wpb {
constexpr int CM = 128, BS = 128, KB = 2;
constexpr int NSA = 4, NSX = 2;
constexpr int DAT = BS * CH;                               // da or db tile [128 s][32]
constexpr int XT = BS * CM;                                // x tile [2 kb][128 s][64]
constexpr int STAT = BS * 2;
constexpr int MKT = BS * 16;
constexpr int STAGE_A = 2 * DAT * 2;
constexpr int STAGE_X = 36864;                             // bytes: x | stats | mask, padded to 1 KiB
constexpr int THREADS = 352;                               // warp 0: da/db, warp 1: MMA, warps 2-9: compute, warp 10: x
constexpr int SMEM = 1024 + NSA * STAGE_A + NSX * STAGE_X + (XT + XT + 64 * CM) * 2 + 2 * BS * 8 + 256;
static_assert(XT * 2 + STAT * 4 + MKT <= STAGE_X && STAGE_A % 1024 == 0 && STAGE_X % 1024 == 0, "stage layout");
constexpr uint32_t IDESC_DY = idesc_bf16(128, CM, 0, 1);   // A = [da|db] K-major, B = Wf MN-major (N = 128 channels)
constexpr uint32_t IDESC_DW = idesc_bf16(128, 64, 1, 1);   // A = (mask . xh)^T MN-major (M = 128 channels), B = [da db] MN-major
constexpr int D2 = 2 * CM;                                 // TMEM column of the dW accumulator (dy: 0 and 128)
static_assert(SMEM <= 232448, "one CTA per SM");
}  // namespace wpb

template <typename WT_>
__global__ void __launch_bounds__(wpb::THREADS, 1) opm_wprologue_bwd_sm100(
    int N, int S, int ntiles,
    const __grid_constant__ CUtensorMap damap,    // dA [S][N*32] as (32, N, S), box (32, 1, 128), 64B swizzle
    const __grid_constant__ CUtensorMap dbmap,
    const __grid_constant__ CUtensorMap xmap,     // m [S][N][128] as (64, N, S, 2), box (64, 1, 128, 2), 128B swizzle
    const __grid_constant__ CUtensorMap smap,     // stats [N][S][2] fp32 as (2S, N), box (256, 1)
    const __grid_constant__ CUtensorMap mmap,     // mask [S][N] bytes, box (16, 128)
    const __nv_bfloat16* __restrict__ WA, const __nv_bfloat16* __restrict__ WB, const WT_* __restrict__ GAM,
    const __grid_constant__ CUtensorMap dmmap,    // dm, same layout as m
    float* __restrict__ PART) {                   // [grid][128 k][64 ce] = R
  using namespace wpb;
  const int NSB = S / BS;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  unsigned char* sStage = smb;                                                    // [NSA][da | db]
  unsigned char* sXs = smb + NSA * STAGE_A;                                        // [NSX][x | stats | mask]
  __nv_bfloat16* sA2 = reinterpret_cast<__nv_bfloat16*>(sXs + NSX * STAGE_X);     // [2 kb][128 s][64]: mask . xh
  __nv_bfloat16* sDM = sA2 + XT;                                                  // [2 kb][128 s][64]
  __nv_bfloat16* sWF = sDM + XT;                                                  // [2 n blocks][64 ce][64 k]
  float2* sRED = reinterpret_cast<float2*>(sWF + 64 * CM);                        // [2 halves][128 rows] (gs, gx)
  uint64_t* bars = reinterpret_cast<uint64_t*>(sRED + 2 * BS);
  uint64_t* full = bars;             // [NSA]
  uint64_t* empty = full + NSA;      // [NSA]  the dW GEMM of the stage's tile retired
  uint64_t* xfull = empty + NSA;     // [NSX]
  uint64_t* xempty = xfull + NSX;    // [NSX]  count 8
  uint64_t* d1f = xempty + NSX;      // [2]
  uint64_t* d1e = d1f + 2;           // [2]    count 8
  uint64_t* a2f = d1e + 2;           // [1]    count 8
  uint64_t* a2e = a2f + 1;           // [1]
  uint64_t* done = a2e + 1;          // [1]
  uint32_t* tslot = reinterpret_cast<uint32_t*>(done + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int k = 0; k < NSA; ++k) { bar_init(full + k, 1); bar_init(empty + k, 1); }
    for (int k = 0; k < NSX; ++k) { bar_init(xfull + k, 1); bar_init(xempty + k, 8); }
    for (int k = 0; k < 2; ++k) { bar_init(d1f + k, 1); bar_init(d1e + k, 8); }
    bar_init(a2f, 8); bar_init(a2e, 1);
    bar_init(done, 1);
    bar_init_fence();
  }
  // Wf [ce][k] = W[ce][k] gamma[k] in bf16, MN-major for the dy GEMM: n block k / 64, row ce, 128B swizzle
  for (int v = tid; v < 64 * CM / 2; v += THREADS) {
    const int row = v / (CM / 2), k = (v % (CM / 2)) * 2;
    const __nv_bfloat16* w = row < CH ? WA + row * CM : WB + (row - CH) * CM;
    const float2 wv = bf2f(*reinterpret_cast<const uint32_t*>(w + k));
    *reinterpret_cast<uint32_t*>(sWF + (k >> 6) * 64 * 64 + sw128(row, k & 63)) = pack2(wv.x * to_f(GAM[k]), wv.y * to_f(GAM[k + 1]));
  }
  if (warp == 1) tmem_alloc(tslot, 512);
  fence_proxy_async();
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  auto stage_ptr = [&](int st) { return sStage + st * STAGE_A; };
  auto xstage_ptr = [&](int st) { return sXs + st * STAGE_X; };

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
        load_4d(&xmap, p, xfull + st, 0, i, s0, 0);
        load_2d(&smap, p + XT * 2, xfull + st, 2 * s0, i);
        load_2d(&mmap, p + XT * 2 + STAT * 4, xfull + st, i & ~15, s0);
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      auto dw_gemm = [&](int lt) {             // R += (mask . xh)^T . [da db] for tile lt
        const int st = lt % NSA;
        wait(a2f, lt & 1);
        tc_fence_after();
        const __nv_bfloat16* dab = reinterpret_cast<const __nv_bfloat16*>(stage_ptr(st));
#pragma unroll
        for (int ks = 0; ks < BS / 16; ++ks)
          mma_ss(tmem + D2, desc_mn128(sA2 + ks * 16 * 64, BS * 64 * 2), sdesc(sa(dab + ks * 16 * 32), DAT * 2, 512, 4), IDESC_DW, (lt | ks) ? 1u : 0u);
        mma_commit(empty + st);
        mma_commit(a2e);
      };
      int lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int st = lt % NSA, b = lt & 1;
        wait(full + st, (lt / NSA) & 1);
        if (lt >= 2) wait(d1e + b, ((lt >> 1) - 1) & 1);
        tc_fence_after();
        const __nv_bfloat16* dab = reinterpret_cast<const __nv_bfloat16*>(stage_ptr(st));
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)           // K = 64: two k steps in da, two in db
          mma_ss(tmem + b * CM, desc_k64(dab + (ks >> 1) * DAT + (ks & 1) * 16), desc_mn128(sWF + ks * 16 * 64, 64 * 64 * 2),
                 IDESC_DY, ks ? 1u : 0u);
        mma_commit(d1f + b);
        if (lt >= 1) dw_gemm(lt - 1);
      }
      if (lt >= 1) dw_gemm(lt - 1);
      mma_commit(done);
    }
  } else {
    const int q = warp & 3, wh = (warp - 2) >> 2, r = q * 32 + lane;
    const bool leader = (warp == 2 && lane == 0);
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int i = t / NSB, s0 = (t % NSB) * BS, sx = lt % NSX, b = lt & 1;
      wait(xfull + sx, (lt / NSX) & 1);
      const unsigned char* px = xstage_ptr(sx);
      const __nv_bfloat16* xr = reinterpret_cast<const __nv_bfloat16*>(px) + wh * BS * 64 + r * 64;
      const float2 stt = reinterpret_cast<const float2*>(px + XT * 2)[r];
      const float mk = px[XT * 2 + STAT * 4 + r * 16 + (i & 15)] ? 1.f : 0.f;
      uint4 xu[8];
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) xu[c8] = *reinterpret_cast<const uint4*>(xr + ((c8 ^ (r & 7)) << 3));
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(xempty + sx);
      wait(d1f + b, (lt >> 1) & 1);
      tc_fence_after();
      float gv[64];
      tmem_ld32(tmem_at(tmem + b * CM + wh * 64, q * 32, 0), gv);
      tmem_ld32(tmem_at(tmem + b * CM + wh * 64, q * 32, 32), gv + 32);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) arrive(d1e + b);
      const float mean = stt.x, rstd = stt.y;
      const float2 nm = make_float2(-mean, -mean), rs = make_float2(rstd, rstd), m2 = make_float2(mk, mk);
      float2 gsa = make_float2(0.f, 0.f), gsb = gsa, gxa = gsa, gxb = gsa;
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) {
        const uint32_t uw[4] = {xu[c8].x, xu[c8].y, xu[c8].z, xu[c8].w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int e = c8 * 4 + k;
          const float2 xh = mul2(add2(bf2f(uw[k]), nm), rs);
          const float2 g2 = mul2(make_float2(gv[2 * e], gv[2 * e + 1]), m2);
          if (k & 1) { gsb = add2(gsb, g2); gxb = fma2(g2, xh, gxb); }
          else { gsa = add2(gsa, g2); gxa = fma2(g2, xh, gxa); }
        }
      }
      gsa = add2(gsa, gsb); gxa = add2(gxa, gxb);
      sRED[wh * BS + r] = make_float2(gsa.x + gsa.y, gxa.x + gxa.y);
      if (lt >= 1) wait(a2e, (lt - 1) & 1);      // the dW GEMM of the previous tile retired: sA2 is free
      if (leader) bulk_wait_read<0>();           // the previous dm store has left sDM
      named_sync(1, 256);
      const float2 p0 = sRED[r], p1 = sRED[BS + r];   // the two halves in a fixed order: both threads of a row agree bitwise
      const float gs = p0.x + p1.x, gx = p0.y + p1.y;
      const float ra = -rstd * gs * (1.f / CM), rb = rstd * gx * (1.f / CM);
      __nv_bfloat16* dmr = sDM + wh * BS * 64 + r * 64;
      __nv_bfloat16* a2r = sA2 + wh * BS * 64 + r * 64;
      {
        const float2 ra2 = make_float2(ra, ra), nrb = make_float2(-rb, -rb);
#pragma unroll
        for (int c8 = 0; c8 < 8; ++c8) {
          const uint32_t uw[4] = {xu[c8].x, xu[c8].y, xu[c8].z, xu[c8].w};
          uint4 d, a;
          uint32_t* dw_ = reinterpret_cast<uint32_t*>(&d);
          uint32_t* aw = reinterpret_cast<uint32_t*>(&a);
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const int e = c8 * 4 + k;
            const float2 xh = mul2(add2(bf2f(uw[k]), nm), rs);
            const float2 g2 = mul2(make_float2(gv[2 * e], gv[2 * e + 1]), m2);
            const float2 dd = fma2(xh, nrb, fma2(g2, rs, ra2));
            const float2 aa = mul2(xh, m2);
            dw_[k] = pack2(dd.x, dd.y);
            aw[k] = pack2(aa.x, aa.y);
          }
          const int off = (c8 ^ (r & 7)) << 3;
          *reinterpret_cast<uint4*>(dmr + off) = d;
          *reinterpret_cast<uint4*>(a2r + off) = a;
        }
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(a2f);
      named_sync(1, 256);
      if (leader) { store_4d(&dmmap, sDM, 0, i, s0, 0); bulk_commit(); }
    }
    if (leader) bulk_wait<0>();
    if (wh == 0) {                                 // R: TMEM lane = k (0-127), 64 columns ce
      wait(done, 0);
      tc_fence_after();
      float v[32];
      for (int c0 = 0; c0 < 64; c0 += 32) {
        tmem_ld32(tmem_at(tmem + D2, q * 32, c0), v);
        tmem_wait_ld();
        float* out = PART + ((size_t)blockIdx.x * CM + r) * 64 + c0;
#pragma unroll
        for (int k = 0; k < 32; k += 4) *reinterpret_cast<float4*>(out + k) = make_float4(v[k], v[k + 1], v[k + 2], v[k + 3]);
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}

// dWa = gamma . R_a + ssa_a (x) beta, dWb likewise, dgamma[k] = sum_ce W[ce][k] R[k][ce], dbeta[k] = sum_ce W[ce][k] ssa[ce]
// (R [CM][64] summed over the CTAs, ssa [64] the masked column sums of [da db]), one block.
template <int CM, typename GT>
__global__ void __launch_bounds__(256) opm_wpbwd_finalize(const float* __restrict__ RED, const float* __restrict__ SSA,
                                                          const __nv_bfloat16* __restrict__ WA, const __nv_bfloat16* __restrict__ WB,
                                                          const GT* __restrict__ GAM, const GT* __restrict__ BET, float* __restrict__ DWA,
                                                          float* __restrict__ DWB, GT* __restrict__ DGAM, GT* __restrict__ DBET) {
  __shared__ float sR[CM * 64], sS[64], sG[CM], sB[CM];
  const int t = threadIdx.x;
  for (int v = t; v < CM * 64; v += 256) sR[v] = __ldcg(RED + v);
  if (t < 64) sS[t] = __ldcg(SSA + t);
  if (t < CM) { sG[t] = to_f(GAM[t]); sB[t] = to_f(BET[t]); }
  __syncthreads();
  for (int v = t; v < 64 * CM; v += 256) {
    const int ce = v / CM, k = v % CM;
    const float val = sG[k] * sR[k * 64 + ce] + sS[ce] * sB[k];
    if (ce < CH) DWA[ce * CM + k] = val; else DWB[(ce - CH) * CM + k] = val;
  }
  if (t < CM) {
    float dg = 0.f, db = 0.f;
    for (int ce = 0; ce < 64; ++ce) {
      const float w = __bfloat162float(ce < CH ? WA[ce * CM + t] : WB[(ce - CH) * CM + t]);
      dg += w * sR[t * 64 + ce];
      db += w * sS[ce];
    }
    DGAM[t] = from_f<GT>(dg); DBET[t] = from_f<GT>(db);
  }
}
}  // namespace

// ------------------------------------------------------------------------------------------------- host
template <int CM>
static void prologue_launch(torch::Tensor& m, torch::Tensor& mask, torch::Tensor& lnw, torch::Tensor& lnb, double eps, torch::Tensor& wa,
                            torch::Tensor& wb, const float* pb, float2* sp, uint32_t* bp, torch::Tensor& A2, torch::Tensor& BT) {
  using C = WPro<CM>;
  const long S = m.size(0), N = m.size(1);
  CUtensorMap xm = make_map<4>(m.data_ptr(), {64, (uint64_t)N, (uint64_t)S, (uint64_t)C::KB}, {(uint64_t)CM, (uint64_t)N * CM, 64},
                               {64, 1, (uint32_t)C::BS, (uint32_t)C::KB}, CU_TENSOR_MAP_SWIZZLE_128B, "m");
  CUtensorMap am = make_map<2>(A2.data_ptr(), {(uint64_t)S, (uint64_t)N * CH}, {(uint64_t)S}, {64, CH}, CU_TENSOR_MAP_SWIZZLE_128B, "A2");
  CUtensorMap bm = make_map<2>(BT.data_ptr(), {(uint64_t)S, (uint64_t)N * CH}, {(uint64_t)S}, {64, CH}, CU_TENSOR_MAP_SWIZZLE_128B, "BT");
  CUtensorMap mm = make_map<2>(mask.data_ptr(), {(uint64_t)N, (uint64_t)S}, {(uint64_t)N}, {16, (uint32_t)C::BS}, CU_TENSOR_MAP_SWIZZLE_NONE, "mask",
                               CU_TENSOR_MAP_DATA_TYPE_UINT8, 1);
  CUtensorMap wam = make_map<2>(wa.data_ptr(), {(uint64_t)CM, (uint64_t)CH}, {(uint64_t)CM}, {64, CH}, CU_TENSOR_MAP_SWIZZLE_128B, "wa");
  CUtensorMap wbm = make_map<2>(wb.data_ptr(), {(uint64_t)CM, (uint64_t)CH}, {(uint64_t)CM}, {64, CH}, CU_TENSOR_MAP_SWIZZLE_128B, "wb");
  const int ntiles = (int)(N * (S / C::BS));
  const int grid = std::min(ntiles, C::CTAS * num_sms(m.device().index()));
  auto st = at::cuda::getCurrentCUDAStream();
  if (lnw.scalar_type() == torch::kFloat32) {
    static bool a = false;
    if (!a) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(opm_wprologue_sm100<CM, float>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
      C10_CUDA_CHECK(cudaFuncSetAttribute(opm_wprologue_sm100<CM, float>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      a = true;
    }
    opm_wprologue_sm100<CM, float><<<grid, C::THREADS, C::SMEM, st>>>((int)N, (int)S, ntiles, (float)eps, xm, am, bm, mm, wam, wbm,
        lnw.data_ptr<float>(), lnb.data_ptr<float>(), pb, sp, bp);
  } else {
    TORCH_CHECK(lnw.scalar_type() == torch::kBFloat16, "LayerNorm affine: fp32 or bf16");
    static bool a = false;
    if (!a) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(opm_wprologue_sm100<CM, __nv_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
      C10_CUDA_CHECK(cudaFuncSetAttribute(opm_wprologue_sm100<CM, __nv_bfloat16>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      a = true;
    }
    opm_wprologue_sm100<CM, __nv_bfloat16><<<grid, C::THREADS, C::SMEM, st>>>((int)N, (int)S, ntiles, (float)eps, xm, am, bm, mm, wam, wbm,
        reinterpret_cast<const __nv_bfloat16*>(lnw.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(lnb.data_ptr<at::BFloat16>()),
        pb, sp, bp);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// m [S, N, CM] bf16 (CM 64 / 128), mask bool [S, N], LayerNorm affine fp32 | bf16, wa / wb bf16 [32, CM], pbias fp32 [64] or empty
// -> [A2 [N*32, S], BT [N*32, S], stats [N, S, 2] (save_stats) | empty, bits int32 [N, S/32]]
std::vector<torch::Tensor> opm_wprologue(torch::Tensor m, torch::Tensor mask, torch::Tensor lnw, torch::Tensor lnb, double eps,
                                         torch::Tensor wa, torch::Tensor wb, torch::Tensor pbias, bool save_stats) {
  TORCH_CHECK(m.is_cuda() && m.scalar_type() == torch::kBFloat16 && m.is_contiguous() && m.dim() == 3, "m: [S, N, CM] bf16");
  const long S = m.size(0), N = m.size(1), CM = m.size(2);
  TORCH_CHECK(CM == 64 || CM == 128, "d_msa 64 or 128");
  TORCH_CHECK(S % 128 == 0 && N % 32 == 0, "S % 128, N % 32");
  TORCH_CHECK(mask.scalar_type() == torch::kBool && mask.is_contiguous() && mask.numel() == S * N, "mask: bool [S, N]");
  TORCH_CHECK(wa.scalar_type() == torch::kBFloat16 && wa.is_contiguous() && wa.numel() == CH * CM, "wa: bf16 [32, d_msa]");
  TORCH_CHECK(wb.scalar_type() == torch::kBFloat16 && wb.is_contiguous() && wb.numel() == CH * CM, "wb: bf16 [32, d_msa]");
  TORCH_CHECK(lnw.scalar_type() == lnb.scalar_type() && lnw.is_contiguous() && lnb.is_contiguous() && lnw.numel() == CM, "LayerNorm affine");
  const bool has_pb = pbias.numel() > 0;
  TORCH_CHECK(!has_pb || (pbias.scalar_type() == torch::kFloat32 && pbias.is_contiguous() && pbias.numel() == 2 * CH), "pbias: fp32 [64]");
  auto A2 = torch::empty({N * CH, S}, m.options());
  auto BT = torch::empty({N * CH, S}, m.options());
  auto bits = torch::empty({N, S / 32}, m.options().dtype(torch::kInt32));
  auto stats = save_stats ? torch::empty({N, S, 2}, m.options().dtype(torch::kFloat32)) : torch::empty({0}, m.options().dtype(torch::kFloat32));
  float2* sp = save_stats ? reinterpret_cast<float2*>(stats.data_ptr<float>()) : nullptr;
  auto* bp = reinterpret_cast<uint32_t*>(bits.data_ptr<int>());
  const float* pb = has_pb ? pbias.data_ptr<float>() : nullptr;
  if (CM == 64) prologue_launch<64>(m, mask, lnw, lnb, eps, wa, wb, pb, sp, bp, A2, BT);
  else prologue_launch<128>(m, mask, lnw, lnb, eps, wa, wb, pb, sp, bp, A2, BT);
  return {A2, BT, stats, bits};
}

template <int NW>
static void epilogue_launch(torch::Tensor& O, torch::Tensor& bits, torch::Tensor& wo, torch::Tensor& bias, int64_t ni, int64_t nj,
                            const torch::Tensor* res, torch::Tensor& out, int64_t after) {
  using C = WEp<NW>;
  const long CZ = wo.size(0);
  TORCH_CHECK(ni % C::BI == 0 && nj % C::BJ == 0, "the epilogue tiles ", C::BI, " x ", C::BJ, " tokens");
  const uint64_t M = (uint64_t)nj * CH;
  CUtensorMap am = make_map<4>(O.data_ptr(), {32, (uint64_t)nj, (uint64_t)ni, 32}, {32, 32 * M, M}, {32, (uint32_t)C::BJ, (uint32_t)C::BI, 2},
                               CU_TENSOR_MAP_SWIZZLE_64B, "O");
  CUtensorMap bm = make_map<2>(wo.data_ptr(), {(uint64_t)NCH, (uint64_t)CZ}, {(uint64_t)NCH}, {64, (uint32_t)NW}, CU_TENSOR_MAP_SWIZZLE_128B, "Wo");
  CUtensorMap zm = make_map<4>(out.data_ptr(), {64, (uint64_t)nj, (uint64_t)(CZ / 64), (uint64_t)ni}, {(uint64_t)CZ, 64, (uint64_t)nj * CZ},
                               {64, (uint32_t)C::BJ, (uint32_t)(NW / 64), (uint32_t)C::BI}, CU_TENSOR_MAP_SWIZZLE_128B, "z");
  CUtensorMap rm = res ? make_map<4>(res->data_ptr(), {64, (uint64_t)nj, (uint64_t)(CZ / 64), (uint64_t)ni}, {(uint64_t)CZ, 64, (uint64_t)nj * CZ},
                                     {64, (uint32_t)C::BJ, (uint32_t)(NW / 64), (uint32_t)C::BI}, CU_TENSOR_MAP_SWIZZLE_128B, "res") : zm;
  const int nzb = (int)(CZ / NW);
  const int ntiles = (int)((ni / C::BI) * (nj / C::BJ)) * nzb;
  const int grid = std::min(ntiles, num_sms(O.device().index()));
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(opm_wepilogue_sm100<NW>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM)); attr = true; }
  opm_wepilogue_sm100<NW><<<grid, C::THREADS, C::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      (int)ni, (int)nj, ntiles, nzb, am, bm, zm, rm, reinterpret_cast<const uint32_t*>(bits.data_ptr<int>()), (int)bits.size(1),
      bias.data_ptr(), bias.scalar_type() == torch::kBFloat16 ? 1 : 0, res ? 1 : 0, (int)after);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// O [ni*32, nj*32] bf16, bits int32 [N, S/32], wo bf16 [CZ, 1024] (CZ 128 / 256 / 384), bias fp32 | bf16 [CZ] -> z [1, ni, nj, CZ] (+ residual)
torch::Tensor opm_wepilogue(torch::Tensor O, torch::Tensor bits, torch::Tensor wo, torch::Tensor bias, int64_t ni, int64_t nj,
                            c10::optional<torch::Tensor> residual, int64_t after) {
  TORCH_CHECK(O.is_cuda() && O.scalar_type() == torch::kBFloat16 && O.is_contiguous(), "O: contiguous cuda bf16");
  TORCH_CHECK(O.size(0) == ni * CH && O.size(1) == nj * CH, "O: [ni*32, nj*32]");
  TORCH_CHECK(bits.scalar_type() == torch::kInt32 && bits.is_contiguous() && bits.dim() == 2 && bits.size(0) >= std::max(ni, nj) &&
              bits.size(1) % 4 == 0, "bits: int32 [N, S/32]");
  TORCH_CHECK(wo.scalar_type() == torch::kBFloat16 && wo.is_contiguous() && wo.dim() == 2 && wo.size(1) == NCH, "wo: bf16 [c_z, 1024]");
  const long CZ = wo.size(0);
  TORCH_CHECK(CZ == 128 || CZ == 256 || CZ == 384, "c_z 128, 256 or 384");
  TORCH_CHECK((bias.scalar_type() == torch::kFloat32 || bias.scalar_type() == torch::kBFloat16) && bias.is_contiguous() && bias.numel() == CZ,
              "bias: fp32 or bf16 [c_z]");
  const bool has_res = residual.has_value() && residual->numel() > 0;
  if (has_res)
    TORCH_CHECK(residual->scalar_type() == torch::kBFloat16 && residual->is_contiguous() && residual->numel() == ni * nj * CZ, "residual: bf16 [1, ni, nj, c_z]");
  auto out = torch::empty({1, ni, nj, CZ}, O.options());
  const torch::Tensor* res = has_res ? &residual.value() : nullptr;
  if (CZ == 128) epilogue_launch<128>(O, bits, wo, bias, ni, nj, res, out, after);
  else if (CZ == 256) epilogue_launch<256>(O, bits, wo, bias, ni, nj, res, out, after);
  else epilogue_launch<192>(O, bits, wo, bias, ni, nj, res, out, after);
  return out;
}

template <int CZ>
static std::vector<torch::Tensor> dgrad_launch(torch::Tensor& dz, torch::Tensor& bits, torch::Tensor& wo, int64_t ni, int64_t nj, int64_t grad_bf16) {
  using C = WDg<CZ>;
  TORCH_CHECK(ni % C::BI == 0 && nj % C::BJ == 0, "the kernel tiles ", C::BI, " x ", C::BJ, " tokens");
  auto dO = torch::empty({ni * CH, nj * CH}, dz.options());
  auto dzp = torch::empty({ni, nj, (long)CZ}, dz.options());
  const int ntiles = (int)((ni / C::BI) * (nj / C::BJ));
  const int grid = std::min(ntiles, num_sms(dz.device().index()));
  auto dbo_part = torch::empty({grid, (long)CZ}, dz.options().dtype(torch::kFloat32));
  const uint64_t M = (uint64_t)nj * CH;
  CUtensorMap dzm = make_map<4>(dz.data_ptr(), {64, (uint64_t)nj, (uint64_t)(CZ / 64), (uint64_t)ni}, {(uint64_t)CZ, 64, (uint64_t)nj * CZ},
                                {64, (uint32_t)C::BJ, 1, (uint32_t)C::BI}, CU_TENSOR_MAP_SWIZZLE_128B, "dz");
  CUtensorMap wom = make_map<2>(wo.data_ptr(), {(uint64_t)NCH, (uint64_t)CZ}, {(uint64_t)NCH}, {64, 64}, CU_TENSOR_MAP_SWIZZLE_128B, "Wo");
  CUtensorMap dom = make_map<4>(dO.data_ptr(), {32, (uint64_t)nj, 32, (uint64_t)ni}, {32, M, 32 * M}, {32, (uint32_t)C::BJ, 4, (uint32_t)C::BI},
                                CU_TENSOR_MAP_SWIZZLE_64B, "dO");
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(opm_wdgrad_sm100<CZ>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM)); attr = true; }
  opm_wdgrad_sm100<CZ><<<grid, C::THREADS, C::SMEM, at::cuda::getCurrentCUDAStream()>>>((int)ni, (int)nj, ntiles, dzm, wom, dom,
      reinterpret_cast<__nv_bfloat16*>(dzp.data_ptr<at::BFloat16>()),
      reinterpret_cast<const uint32_t*>(bits.data_ptr<int>()), (int)bits.size(1), dbo_part.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dO, dzp, colsum(dbo_part, grad_bf16 ? torch::kBFloat16 : torch::kFloat32)};
}

// dz [ni, nj, CZ] bf16 (CZ 256 / 384), bits, wo bf16 [CZ, 1024] -> [dO [ni*32, nj*32], dz/n [ni, nj, CZ], dbias [CZ] = sum dz]
std::vector<torch::Tensor> opm_wdgrad(torch::Tensor dz, torch::Tensor bits, torch::Tensor wo, int64_t ni, int64_t nj, int64_t grad_bf16) {
  TORCH_CHECK(wo.scalar_type() == torch::kBFloat16 && wo.is_contiguous() && wo.dim() == 2 && wo.size(1) == NCH, "wo: bf16 [c_z, 1024]");
  const long CZ = wo.size(0);
  TORCH_CHECK(dz.is_cuda() && dz.scalar_type() == torch::kBFloat16 && dz.is_contiguous() && dz.numel() == ni * nj * CZ, "dz: contiguous bf16 [ni, nj, c_z]");
  TORCH_CHECK(bits.scalar_type() == torch::kInt32 && bits.is_contiguous() && bits.dim() == 2 && bits.size(1) % 4 == 0, "bits: int32 [N, S/32]");
  if (CZ == 128) return dgrad_launch<128>(dz, bits, wo, ni, nj, grad_bf16);
  if (CZ == 256) return dgrad_launch<256>(dz, bits, wo, ni, nj, grad_bf16);
  TORCH_CHECK(CZ == 384, "c_z 128, 256 or 384");
  return dgrad_launch<384>(dz, bits, wo, ni, nj, grad_bf16);
}

template <int CZ>
static torch::Tensor dwo_launch(torch::Tensor& dzp, torch::Tensor& O, int64_t ni, int64_t nj, int64_t grad_bf16) {
  using C = WDw<CZ>;
  TORCH_CHECK(ni % C::BI == 0 && nj % C::BJ == 0, "the kernel steps ", C::BI, " x ", C::BJ, " tokens");
  const int ntiles = (int)((ni / C::BI) * (nj / C::BJ));
  const int nsplit = std::max(1, std::min(ntiles, num_sms(O.device().index()) / C::NPART));
  auto part = torch::empty({nsplit, (long)CZ, (long)NCH}, O.options().dtype(torch::kFloat32));
  const uint64_t M = (uint64_t)nj * CH;
  CUtensorMap am = make_map<4>(dzp.data_ptr(), {64, (uint64_t)nj, (uint64_t)ni, (uint64_t)(CZ / 64)}, {(uint64_t)CZ, (uint64_t)nj * CZ, 64},
                               {64, (uint32_t)C::BJ, (uint32_t)C::BI, (uint32_t)(CZ / 64)}, CU_TENSOR_MAP_SWIZZLE_128B, "dzp");
  CUtensorMap bm = make_map<4>(O.data_ptr(), {32, (uint64_t)nj, (uint64_t)ni, 32}, {32, 32 * M, M}, {32, (uint32_t)C::BJ, (uint32_t)C::BI, (uint32_t)C::NCB},
                               CU_TENSOR_MAP_SWIZZLE_64B, "O");
  CUtensorMap pm = make_map<2>(part.data_ptr(), {(uint64_t)NCH, (uint64_t)nsplit * CZ}, {(uint64_t)NCH}, {32, 128}, CU_TENSOR_MAP_SWIZZLE_128B, "part",
                               CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4);
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(opm_wdwo_sm100<CZ>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM)); attr = true; }
  opm_wdwo_sm100<CZ><<<dim3(nsplit, C::NPART), C::THREADS, C::SMEM, at::cuda::getCurrentCUDAStream()>>>((int)ni, (int)nj, ntiles, nsplit, am, bm, pm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return colsum(part, grad_bf16 ? torch::kBFloat16 : torch::kFloat32);
}

// dz/n [ni, nj, CZ] bf16, O [ni*32, nj*32] bf16 -> dWo [CZ, 1024] (fp32, or bf16 with grad_bf16)
torch::Tensor opm_wdwo(torch::Tensor dzp, torch::Tensor O, int64_t ni, int64_t nj, int64_t grad_bf16) {
  TORCH_CHECK(dzp.is_contiguous() && dzp.scalar_type() == torch::kBFloat16 && dzp.dim() == 3, "dzp: bf16 [ni, nj, c_z]");
  TORCH_CHECK(O.is_contiguous() && O.scalar_type() == torch::kBFloat16 && O.size(0) == ni * CH && O.size(1) == nj * CH, "O: bf16 [ni*32, nj*32]");
  const long CZ = dzp.size(2);
  if (CZ == 256) return dwo_launch<256>(dzp, O, ni, nj, grad_bf16);
  TORCH_CHECK(CZ == 384, "c_z 256 or 384");
  return dwo_launch<384>(dzp, O, ni, nj, grad_bf16);
}

// d_msa 128: dA, dB [S, N*32], m [S, N, 128], stats [N, S, 2], mask bool [S, N], gamma / beta, wa / wb bf16 [32, 128],
// ssa fp32 [64] (the masked column sums of [dA dB]) -> [dm [1, S, N, 128], dWa, dWb fp32 [32, 128], dgamma, dbeta (gamma's dtype)]
std::vector<torch::Tensor> opm_wprologue_bwd(torch::Tensor dA, torch::Tensor dB, torch::Tensor m, torch::Tensor stats, torch::Tensor mask,
                                             torch::Tensor gamma, torch::Tensor beta, torch::Tensor wa, torch::Tensor wb, torch::Tensor ssa) {
  using namespace wpb;
  TORCH_CHECK(m.is_contiguous() && m.scalar_type() == torch::kBFloat16 && m.dim() == 3 && m.size(2) == CM, "m: [S, N, 128] bf16");
  const long S = m.size(0), N = m.size(1);
  TORCH_CHECK(S % BS == 0, "S must be a multiple of ", BS);
  TORCH_CHECK(dA.is_contiguous() && dB.is_contiguous() && dA.numel() == S * N * CH && dB.numel() == S * N * CH, "dA, dB: [S, N*32]");
  TORCH_CHECK(stats.is_contiguous() && stats.scalar_type() == torch::kFloat32 && stats.numel() == S * N * 2, "stats: fp32 [N, S, 2]");
  TORCH_CHECK(mask.is_contiguous() && mask.scalar_type() == torch::kBool && mask.numel() == S * N, "mask: bool [S, N]");
  TORCH_CHECK(wa.scalar_type() == torch::kBFloat16 && wb.scalar_type() == torch::kBFloat16 && wa.is_contiguous() && wb.is_contiguous() &&
              wa.numel() == CH * CM && wb.numel() == CH * CM, "wa, wb: bf16 [32, 128]");
  TORCH_CHECK(gamma.scalar_type() == beta.scalar_type() && gamma.is_contiguous() && beta.is_contiguous(), "gamma, beta");
  TORCH_CHECK(ssa.scalar_type() == torch::kFloat32 && ssa.is_contiguous() && ssa.numel() == 2 * CH, "ssa: fp32 [64]");
  const bool gf = gamma.scalar_type() == torch::kFloat32;
  auto dm = torch::empty({1, S, N, (long)CM}, m.options());
  const int ntiles = (int)(N * (S / BS));
  const int grid = std::min(ntiles, num_sms(m.device().index()));
  auto part = torch::empty({grid, (long)CM, 64}, m.options().dtype(torch::kFloat32));
  CUtensorMap dam = make_map<3>(dA.data_ptr(), {(uint64_t)CH, (uint64_t)N, (uint64_t)S}, {(uint64_t)CH, (uint64_t)N * CH}, {CH, 1, BS}, CU_TENSOR_MAP_SWIZZLE_64B, "dA");
  CUtensorMap dbm = make_map<3>(dB.data_ptr(), {(uint64_t)CH, (uint64_t)N, (uint64_t)S}, {(uint64_t)CH, (uint64_t)N * CH}, {CH, 1, BS}, CU_TENSOR_MAP_SWIZZLE_64B, "dB");
  CUtensorMap xm = make_map<4>(m.data_ptr(), {64, (uint64_t)N, (uint64_t)S, (uint64_t)KB}, {(uint64_t)CM, (uint64_t)N * CM, 64}, {64, 1, BS, KB},
                               CU_TENSOR_MAP_SWIZZLE_128B, "m");
  CUtensorMap dmm = make_map<4>(dm.data_ptr(), {64, (uint64_t)N, (uint64_t)S, (uint64_t)KB}, {(uint64_t)CM, (uint64_t)N * CM, 64}, {64, 1, BS, KB},
                                CU_TENSOR_MAP_SWIZZLE_128B, "dm");
  CUtensorMap sm = make_map<2>(stats.data_ptr(), {(uint64_t)(2 * S), (uint64_t)N}, {(uint64_t)(2 * S)}, {2 * BS, 1}, CU_TENSOR_MAP_SWIZZLE_NONE, "stats",
                               CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4);
  CUtensorMap mm = make_map<2>(mask.data_ptr(), {(uint64_t)N, (uint64_t)S}, {(uint64_t)N}, {16, BS}, CU_TENSOR_MAP_SWIZZLE_NONE, "mask",
                               CU_TENSOR_MAP_DATA_TYPE_UINT8, 1);
  auto st = at::cuda::getCurrentCUDAStream();
  const auto* wap = reinterpret_cast<const __nv_bfloat16*>(wa.data_ptr<at::BFloat16>());
  const auto* wbp = reinterpret_cast<const __nv_bfloat16*>(wb.data_ptr<at::BFloat16>());
  if (gf) {
    static bool attr = false;
    if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(opm_wprologue_bwd_sm100<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
    opm_wprologue_bwd_sm100<float><<<grid, THREADS, SMEM, st>>>((int)N, (int)S, ntiles, dam, dbm, xm, sm, mm, wap, wbp, gamma.data_ptr<float>(), dmm,
        part.data_ptr<float>());
  } else {
    TORCH_CHECK(gamma.scalar_type() == torch::kBFloat16, "gamma: fp32 or bf16");
    static bool attr = false;
    if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(opm_wprologue_bwd_sm100<__nv_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
    opm_wprologue_bwd_sm100<__nv_bfloat16><<<grid, THREADS, SMEM, st>>>((int)N, (int)S, ntiles, dam, dbm, xm, sm, mm, wap, wbp,
        reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr<at::BFloat16>()), dmm, part.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto red = colsum(part);                                           // [CM * 64], the CTAs' partials in split order
  auto dwa = torch::empty({(long)CH, (long)CM}, wa.options().dtype(torch::kFloat32)), dwb = torch::empty({(long)CH, (long)CM}, wb.options().dtype(torch::kFloat32));
  auto dgam = torch::empty({(long)CM}, gamma.options()), dbet = torch::empty({(long)CM}, beta.options());
  if (gf) opm_wpbwd_finalize<CM, float><<<1, 256, 0, st>>>(red.data_ptr<float>(), ssa.data_ptr<float>(), wap, wbp, gamma.data_ptr<float>(),
      beta.data_ptr<float>(), dwa.data_ptr<float>(), dwb.data_ptr<float>(), dgam.data_ptr<float>(), dbet.data_ptr<float>());
  else opm_wpbwd_finalize<CM, __nv_bfloat16><<<1, 256, 0, st>>>(red.data_ptr<float>(), ssa.data_ptr<float>(), wap, wbp,
      reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr<at::BFloat16>()),
      dwa.data_ptr<float>(), dwb.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(dgam.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(dbet.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dm, dwa, dwb, dgam, dbet};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("opm_wprologue", &opm_wprologue, "sm100 OPM prologue at d_msa 64 / 128 with the optional projection biases",
        py::arg("m"), py::arg("mask"), py::arg("lnw"), py::arg("lnb"), py::arg("eps"), py::arg("wa"), py::arg("wb"), py::arg("pbias"),
        py::arg("save_stats"));
  m.def("opm_wepilogue", &opm_wepilogue, "sm100 OPM epilogue at any c_z (128-wide slices)", py::arg("O"), py::arg("bits"), py::arg("wo"),
        py::arg("bias"), py::arg("ni"), py::arg("nj"), py::arg("residual") = py::none(), py::arg("after") = 0);
  m.def("opm_wdgrad", &opm_wdgrad, "sm100 OPM dO at c_z 128 / 256 / 384", py::arg("dz"), py::arg("bits"), py::arg("wo"), py::arg("ni"), py::arg("nj"),
        py::arg("grad_bf16") = 0);
  m.def("opm_wdwo", &opm_wdwo, "sm100 OPM dWo at c_z 256 / 384", py::arg("dzp"), py::arg("O"), py::arg("ni"), py::arg("nj"), py::arg("grad_bf16") = 0);
  m.def("opm_wprologue_bwd", &opm_wprologue_bwd, "sm100 OPM prologue backward at d_msa 128");
}
