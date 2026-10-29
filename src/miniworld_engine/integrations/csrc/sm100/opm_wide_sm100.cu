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
  // d_msa 256 (IntelliFold-v2): a 64 KiB x tile, so two x stages and one accumulator (the drain of a tile is short next to its loads)
  static constexpr int NST = CM == 64 ? 3 : CM == 128 ? 4 : 2;
  static constexpr int NACC = CM == 256 ? 1 : 2;
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

// One LayerNorm row of an x tile [KB][128 s][64] (128B-swizzled k blocks) in place -> y in bf16 (zeros for a masked row), two threads a
// row (lanes l and l ^ 16 of a warp: channel half hf, CM / 2 channels each).  The statistics leave in mean / rstd.
template <int CM>
__device__ __forceinline__ void ln_half_row(__nv_bfloat16* tile, int r, int hf, const float* sLN, bool mkb, float eps, float& mean_out,
                                        float& rstd_out) {
  constexpr int KB = CM / 64, BS = 128, NC = CM / 16;
  __nv_bfloat16* xr = tile + (KB == 1 ? 0 : hf * BS * 64) + r * 64;
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
  mean_out = mean;
  rstd_out = rstd;
}

// d_msa 256: the same row split (two threads a row, 128 channels each = two of the four 64-channel k blocks), but 128 channels do not
// fit a thread's registers as fp32 pairs: three passes over the swizzled rows in shared memory (sum; centred squares; normalise and
// write), 16 channels at a time.  The same statistics as `ln_half_row` (mean, then the centred variance).
__device__ __forceinline__ void ln_half_row256(__nv_bfloat16* tile, int r, int hf, const float* sLN, bool mkb, float eps, float& mean_out,
                                               float& rstd_out) {
  constexpr int BS = 128, CM = 256;
  auto chunk = [&](int k) -> __nv_bfloat16* {      // k = 0..15: k block hf * 2 + k / 8, 16-byte chunk k % 8 of row r
    return tile + (hf * 2 + (k >> 3)) * BS * 64 + r * 64 + (((k & 7) ^ (r & 7)) << 3);
  };
  float2 sa = make_float2(0.f, 0.f), sb = sa;
#pragma unroll 8
  for (int k = 0; k < 16; ++k) {
    const uint4 u = *reinterpret_cast<const uint4*>(chunk(k));
    sa = add2(sa, add2(bf2f(u.x), bf2f(u.y)));
    sb = add2(sb, add2(bf2f(u.z), bf2f(u.w)));
  }
  const float2 sm = add2(sa, sb);
  float sum = sm.x + sm.y;
  sum += __shfl_xor_sync(0xffffffffu, sum, 16);
  const float mean = sum * (1.f / CM);
  const float2 nm = make_float2(-mean, -mean);
  float2 va = make_float2(0.f, 0.f), vb = va;
  asm volatile("" ::: "memory");                   // re-read the row: 64 registers of it held across the passes spilled
#pragma unroll 8
  for (int k = 0; k < 16; ++k) {
    const uint4 u = *reinterpret_cast<const uint4*>(chunk(k));
    const float2 x0 = add2(bf2f(u.x), nm), x1 = add2(bf2f(u.y), nm), x2 = add2(bf2f(u.z), nm), x3 = add2(bf2f(u.w), nm);
    va = fma2(x0, x0, fma2(x1, x1, va));
    vb = fma2(x2, x2, fma2(x3, x3, vb));
  }
  const float2 vs = add2(va, vb);
  float var = vs.x + vs.y;
  var += __shfl_xor_sync(0xffffffffu, var, 16);
  const float rstd = 1.f / sqrtf(var * (1.f / CM) + eps);
  const float2 rs = make_float2(rstd, rstd);
  const uint32_t keep = mkb ? 0xffffffffu : 0u;
  asm volatile("" ::: "memory");
#pragma unroll 4
  for (int k = 0; k < 16; ++k) {
    uint4* p = reinterpret_cast<uint4*>(chunk(k));
    const uint4 u = *p;
    const int cb = hf * 128 + k * 8;
    const float4 g0 = *reinterpret_cast<const float4*>(sLN + cb), g1 = *reinterpret_cast<const float4*>(sLN + cb + 4);
    const float4 b0 = *reinterpret_cast<const float4*>(sLN + CM + cb), b1 = *reinterpret_cast<const float4*>(sLN + CM + cb + 4);
    const float2 y0 = fma2(mul2(add2(bf2f(u.x), nm), rs), make_float2(g0.x, g0.y), make_float2(b0.x, b0.y));
    const float2 y1 = fma2(mul2(add2(bf2f(u.y), nm), rs), make_float2(g0.z, g0.w), make_float2(b0.z, b0.w));
    const float2 y2 = fma2(mul2(add2(bf2f(u.z), nm), rs), make_float2(g1.x, g1.y), make_float2(b1.x, b1.y));
    const float2 y3 = fma2(mul2(add2(bf2f(u.w), nm), rs), make_float2(g1.z, g1.w), make_float2(b1.z, b1.w));
    uint4 o;
    o.x = pack2(y0.x, y0.y) & keep; o.y = pack2(y1.x, y1.y) & keep; o.z = pack2(y2.x, y2.y) & keep; o.w = pack2(y3.x, y3.y) & keep;
    *p = o;
  }
  mean_out = mean;
  rstd_out = rstd;
}

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
    const int r = (warp - 2) * 16 + (lane & 15), hf = lane >> 4;
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int i = t / NSB, s0 = (t % NSB) * BS, st = lt % NST;
      wait(xfull + st, (lt / NST) & 1);
      const bool mkb = sM[st * MT + r * 16 + (i & 15)] != 0;
      float mean, rstd;
      if constexpr (CM == 256) ln_half_row256(sX + st * XT, r, hf, sLN, mkb, eps, mean, rstd);
      else ln_half_row<CM>(sX + st * XT, r, hf, sLN, mkb, eps, mean, rstd);
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
// CTA-pair (cluster of two, cta_group::2) helpers
__device__ __forceinline__ uint32_t cl_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ void cl_sync() { asm volatile("barrier.cluster.arrive.aligned;\nbarrier.cluster.wait.aligned;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t cl_map(uint32_t a, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(rank));
  return r;
}
__device__ __forceinline__ void wait_cluster(uint64_t* b, uint32_t parity) {   // acquire at cluster scope: the peer's stores are visible
  asm volatile("{\n.reg .pred p;\nWCLU_%=:\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n@!p bra WCLU_%=;\n}\n"
               :: "r"(sa(b)), "r"(parity) : "memory");
}
// TMA into this CTA's shared memory, the bytes counted on the LEADER's barrier at the same offset (peer bit of the address cleared)
__device__ __forceinline__ void load_2d_2sm(const void* map, void* dst, uint64_t* bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
               :: "r"(sa(dst)), "l"(map), "r"(sa(bar) & 0xFEFFFFFFu), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void load_4d_2sm(const void* map, void* dst, uint64_t* bar, int c0, int c1, int c2, int c3) {
  asm volatile("cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];\n"
               :: "r"(sa(dst)), "l"(map), "r"(sa(bar) & 0xFEFFFFFFu), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}
__device__ __forceinline__ void mma2_ss(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
               :: "r"(d), "l"(a), "l"(b), "r"(idesc), "r"(acc) : "memory");
}
// arrives on the barrier at this offset in BOTH CTAs once every MMA issued so far has completed
__device__ __forceinline__ void commit2(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n"
               :: "r"(sa(bar)), "h"((uint16_t)3) : "memory");
}
// a (relaxed, cluster scope) arrive on the leader's barrier at this offset: for TMEM-drained signals, whose ordering
// tcgen05.fence::before_thread_sync carries
__device__ __forceinline__ void arrive_leader(uint64_t* bar) {
  asm volatile("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];\n" :: "r"(cl_map(sa(bar), 0)) : "memory");
}
__device__ __forceinline__ void tmem_alloc2(uint32_t* slot, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;\n" :: "r"(sa(slot)), "r"(ncols) : "memory");
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc2(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;\n" :: "r"(taddr), "r"(ncols) : "memory");
}

// ---------------------------------------------------------------------------------------------
// EPILOGUE at c_z = 128 / 256 / 384 / 512 on CTA PAIRS: opm_epilogue_sm100's tile (128 pairs, K = 1024 in 16 stages of two c values)
// with an accumulator NW = 256 (c_z 256 / 512) or 192 (c_z 384) or 128 columns wide, so a stage's O chunk serves every output column of
// the slice at once.  A cluster of two CTAs takes two neighbouring 128-pair tiles (same i, j and j + 32) and one cta_group::2 product
// M 256 x N NW per k step: each CTA lands its own pairs' O chunk (A) and HALF of the slice of Wo (B, NW / 2 rows), so a CTA streams
// half the Wo bytes per tile of the single-CTA kernel (the L2 -> SM intake bounded it: 768 KiB a tile at c_z 256, now 512).  Two
// accumulators of NW columns per CTA (TMEM lane = this CTA's pair), drained by each CTA's own four warps under the next main loop.
// Roles: warp 0 lane 0 TMA (both CTAs, the bytes counted on the leader's barrier), warp 1 TMEM allocation and, in the leader, the
// MMA issue, warps 2-5 drain (count, bias, either order, the interchain mask, residual) through a staging tile and one TMA store.
template <int NW>
struct WEp {
  static constexpr int BI = 4, BJ = 32, NP = BI * BJ;
  static constexpr int KC = 64, NK = NCH / KC;
  static constexpr int ATILE = NP * KC, BTILE = (NW / 2) * KC, STAGE = ATILE + BTILE;
  static constexpr int NB64 = NW / 64, OBLK = NP * 64;      // staging: NB64 blocks [BI][BJ][64], one per 64 output columns
  static constexpr int OUTT = NB64 * OBLK;
  static constexpr int FREE = 232448 - 1024 - 256 - NW * 4 - OUTT * 2;
  static constexpr int NST = FREE / (STAGE * 2) < 6 ? FREE / (STAGE * 2) : 6;
  static constexpr int THREADS = 192;
  static constexpr int SMEM = 1024 + (NST * STAGE + OUTT) * 2 + NW * 4 + 256;   // + this item's bias slice (fp32)
  static constexpr uint32_t IDESC = idesc_bf16(256, NW);
  static constexpr int TMEM_COLS = 2 * NW <= 256 ? 256 : 512;
  static_assert(SMEM <= 232448 && NST >= 3, "one CTA per SM, at least three stages");
};

template <int NW>
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(192, 1) opm_wepilogue_sm100(
    int NI, int NJ, int nitems, int NZB,
    const __grid_constant__ CUtensorMap amap,     // O as (e 32, j NJ, i NI, c 32), box (32, BJ, BI, 2), 64B swizzle
    const __grid_constant__ CUtensorMap bmap,     // Wo [CZ][NCH], box (64, NW / 2)
    const __grid_constant__ CUtensorMap zmap,     // z as (64 z, NJ j, CZ / 64, NI i), box (64, BJ, 1, BI): one 64-column block
    const __grid_constant__ CUtensorMap rmap,     // the residual, same layout
    const float* __restrict__ NORM,               // the mask counts [NI][NJ] (clamped >= 1; opm_sm100.cu's opm_norm off the prologue's bits)
    const void* __restrict__ BIAS, int bias_bf16, int has_res, int after,
    const int* __restrict__ ASYM) {               // chain ids [N] (the interchain mask), or nullptr
  using C = WEp<NW>;
  constexpr int BI = C::BI, BJ = C::BJ, NP = C::NP, KC = C::KC, NK = C::NK, ATILE = C::ATILE, STAGE = C::STAGE, NST = C::NST;
  constexpr int OUTT = C::OUTT, NB64 = C::NB64, OBLK = C::OBLK;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sStage = reinterpret_cast<__nv_bfloat16*>(smb);
  __nv_bfloat16* sOut = sStage + NST * STAGE;
  float* sBias = reinterpret_cast<float*>(sOut + OUTT);   // [NW] this item's bias slice
  uint64_t* bars = reinterpret_cast<uint64_t*>(sBias + NW);
  uint64_t* full = bars;             // [NST] (the leader's counts both CTAs' bytes)
  uint64_t* empty = full + NST;      // [NST] (both CTAs, by the leader's multicast commit)
  uint64_t* accf = empty + NST;      // [2]   (both CTAs)
  uint64_t* acce = accf + 2;         // [2]   the leader's: count 8, four drain warps of each CTA
  uint64_t* resf = acce + 2;         // [NB64] a staging block's residual landed
  uint32_t* tslot = reinterpret_cast<uint32_t*>(resf + NB64);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rank = (int)cl_rank();
  const bool leader = rank == 0;
  const int cl = blockIdx.x >> 1, ncl = gridDim.x >> 1;
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { bar_init(full + s, 1); bar_init(empty + s, 1); }
    for (int b = 0; b < 2; ++b) { bar_init(accf + b, 1); bar_init(acce + b, 8); }
    for (int h = 0; h < NB64; ++h) bar_init(resf + h, 1);
    bar_init_fence();
    prefetch_map(&amap); prefetch_map(&bmap); prefetch_map(&zmap); if (has_res) prefetch_map(&rmap);
  }
  if (warp == 1) tmem_alloc2(tslot, C::TMEM_COLS);
  tc_fence_before();
  __syncthreads();
  cl_sync();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  const int njb = NJ / BJ;
  // item t: the tile pair t / NZB (this CTA's 128-pair tile 2 (t / NZB) + rank), c_z slice t % NZB
  auto tile = [&](int t, int& i0, int& j0, int& zb) {
    const int pt = 2 * (t / NZB) + rank;
    zb = t % NZB;
    i0 = (pt / njb) * BI;
    j0 = (pt % njb) * BJ;
  };

  if (warp == 0) {
    if (lane == 0) {
      int g = 0;
      for (int t = cl; t < nitems; t += ncl) {
        int i0, j0, zb;
        tile(t, i0, j0, zb);
        for (int kc = 0; kc < NK; ++kc, ++g) {
          const int s = g % NST;
          if (g >= NST) wait(empty + s, ((g / NST) - 1) & 1);
          __nv_bfloat16* st = sStage + s * STAGE;
          if (leader) expect_tx(full + s, 2 * STAGE * 2);
          load_4d_2sm(&amap, st, full + s, 0, j0, i0, kc * 2);
          load_2d_2sm(&bmap, st + ATILE, full + s, kc * KC, zb * NW + rank * (NW / 2));
        }
      }
    }
  } else if (warp == 1) {
    if (leader && lane == 0) {
      int g = 0, lt = 0;
      for (int t = cl; t < nitems; t += ncl, ++lt) {
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
            mma2_ss(d, desc_k64(st + (ks >> 1) * (NP * 32) + (ks & 1) * 16), desc_k128(st + ATILE + ks * 16), C::IDESC, (kc | ks) ? 1u : 0u);
          commit2(empty + s);
        }
        commit2(accf + b);
      }
    }
  } else {
    // drain: TMEM lane = this CTA's pair p (warp q, lane), one 64-column block h at a time through its own staging block, stored by
    // one TMA box.  The residual of a block is loaded ONE ITEM AHEAD, as soon as the block's store has left the staging block (its
    // load latency sat on this critical path when the item's residual came in at the start of its drain); the bias slice goes to
    // shared memory once per item (per-element __ldg of the bias cost 90 of 393 us at c_z 256, L768 S1024)
    const int q = warp & 3;
    const int p = q * 32 + lane;
    const int il = p / BJ, jl = p % BJ;
    const bool lead_thr = (warp == 2 && lane == 0);
    // this thread's pair count, loaded one item ahead
    auto count_of = [&](int t) { int i0_, j0_, zb_; tile(t, i0_, j0_, zb_); return __ldg(NORM + (size_t)(i0_ + il) * NJ + j0_ + jl); };
    auto res_load = [&](int t, int h) {
      int i1, j1, z1;
      tile(t, i1, j1, z1);
      expect_tx(resf + h, OBLK * 2);
      load_4d(&rmap, sOut + h * OBLK, resf + h, 0, j1, z1 * NB64 + h, i1);
    };
    if (lead_thr && has_res && cl < nitems)
      for (int h = 0; h < NB64; ++h) res_load(cl, h);
    float nrm_next = cl < nitems ? count_of(cl) : 1.f;
    int lt = 0;
    for (int t = cl; t < nitems; t += ncl, ++lt) {
      int i0, j0, zb;
      tile(t, i0, j0, zb);
      const int b = lt & 1;
      const float inv = 1.0f / nrm_next;
      if (t + ncl < nitems) nrm_next = count_of(t + ncl);
      const uint32_t keep = (ASYM == nullptr || __ldg(ASYM + i0 + il) == __ldg(ASYM + j0 + jl)) ? 0xffffffffu : 0u;   // interchain: no update
      for (int c = p; c < NW; c += 128)
        sBias[c] = bias_bf16 ? __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(BIAS)[zb * NW + c])
                             : reinterpret_cast<const float*>(BIAS)[zb * NW + c];
      wait(accf + b, (lt >> 1) & 1);
      tc_fence_after();
      named_sync(1, 128);                        // the bias slice is in; (no residual) the previous item's stores have left staging
      const uint32_t tb = tmem_at(tmem + b * NW, q * 32, 0);
#pragma unroll 1
      for (int h = 0; h < NB64; ++h) {
        float v[64];
        tmem_ld32(tb + h * 64, v);
        tmem_ld32(tb + h * 64 + 32, v + 32);
        tmem_wait_ld();
        if (h == NB64 - 1) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive_leader(acce + b); }
        if (has_res) wait(resf + h, lt & 1);
        __nv_bfloat16* row = sOut + h * OBLK + (il * BJ + jl) * 64;
        const float* bh = sBias + h * 64;
#pragma unroll
        for (int c8 = 0; c8 < 8; ++c8) {
          const int sw = (c8 ^ (jl & 7)) << 3;
          uint4 o;
          uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
          uint4 r = make_uint4(0, 0, 0, 0);
          if (has_res) r = *reinterpret_cast<const uint4*>(row + sw);
          const __nv_bfloat162* rh = reinterpret_cast<const __nv_bfloat162*>(&r);
          const float4 b0 = *reinterpret_cast<const float4*>(bh + c8 * 8), b1 = *reinterpret_cast<const float4*>(bh + c8 * 8 + 4);
          const float bb[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const float a0 = v[c8 * 8 + u * 2], a1 = v[c8 * 8 + u * 2 + 1];
            __nv_bfloat162 x = after ? __float22bfloat162_rn(make_float2((a0 + bb[2 * u]) * inv, (a1 + bb[2 * u + 1]) * inv))
                                     : __float22bfloat162_rn(make_float2(fmaf(a0, inv, bb[2 * u]), fmaf(a1, inv, bb[2 * u + 1])));
            *reinterpret_cast<uint32_t*>(&x) &= keep;
            if (has_res) x = __hadd2(x, rh[u]);
            ow[u] = *reinterpret_cast<uint32_t*>(&x);
          }
          *reinterpret_cast<uint4*>(row + sw) = o;
        }
        fence_proxy_async();
        named_sync(1, 128);
        if (lead_thr) {
          store_4d(&zmap, sOut + h * OBLK, 0, j0, zb * NB64 + h, i0);
          bulk_commit();
          bulk_wait_read<0>();                   // the block has left shared memory: the next item's residual block may land in it
          if (has_res && t + ncl < nitems) res_load(t + ncl, h);
        }
      }
      named_sync(1, 128);                        // the bias slice is free for the next item
    }
    if (lead_thr) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  cl_sync();                                       // the peer's last arrives / commits have landed; TMEM is freed by both
  if (warp == 1) { tc_fence_after(); tmem_dealloc2(tmem, C::TMEM_COLS); }
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
  static constexpr int NB = CZ <= 384 ? NZB + 1 : NZB;   // c_z 512: four 32 KiB units fill the ring (the next tile's first unit
                                                       // lands once the last chunk has read it, under the chunk's other units)
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
    const float* __restrict__ NORM, float* __restrict__ DBO,   // the mask counts [NI][NJ] (clamped >= 1); the dbias partials
    const int* __restrict__ ASYM) {               // chain ids [N] (the interchain mask), or nullptr
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
  uint64_t* dzm = bod + 1;           // [NB] count 8: the drain warps took dz/n off the unit and zeroed its interchain rows
  uint32_t* tslot = reinterpret_cast<uint32_t*>(dzm + NB);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int k = 0; k < NB; ++k) { bar_init(dzf + k, 1); bar_init(dze + k, 9); bar_init(dzm + k, 8); }
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
            if (k == 0 && !(zs & 1)) {           // a unit's first use: wait for it
              wait(dzf + d, ((u0 + (zs >> 1)) / NB) & 1);
              tc_fence_after();
            }
            if (k == NCK - 1 && !(zs & 1)) {     // its last: dbias += dz^T . 1 over it (raw dz; interchain rows zeroed by the drain, which
              const int zu = zs >> 1;            // took this tile's dz/n long before: this GEMM waits for nothing in practice)
              wait(dzm + d, ((u0 + zu) / NB) & 1);
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
    // this thread's pair count, loaded one tile ahead (off the drain's critical path)
    auto count_of = [&](int t) { return __ldg(NORM + (size_t)((t / njb) * BI + il) * NJ + (t % njb) * BJ + jl); };
    float nrm_next = blockIdx.x < ntiles ? count_of(blockIdx.x) : 1.f;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int i0 = (t / njb) * BI, j0 = (t % njb) * BJ, u0 = lt * NZB;
      const float nrm = nrm_next;
      if (t + (int)gridDim.x < ntiles) nrm_next = count_of(t + gridDim.x);
      // the interchain mask: a pair of two chains passes no gradient (inv 0 for dO and dz/n; its dz rows zeroed for dbias)
      const bool same = ASYM == nullptr || __ldg(ASYM + i0 + il) == __ldg(ASYM + j0 + jl);
      const float inv = same ? 1.0f / nrm : 0.f;
      const float2 inv2 = make_float2(inv, inv);
      uint4* dst = reinterpret_cast<uint4*>(DZP + ((size_t)(i0 + il) * NJ + j0 + jl) * CZ);
      for (int zu = 0; zu < NZB; ++zu) {         // dz/n for dWo: this warp's 64 z of each unit, straight to global
        const int d = (u0 + zu) % NB;
        wait(dzf + d, ((u0 + zu) / NB) & 1);
        __nv_bfloat16* rp = sDZ + d * UNIT + grp * NP * 64 + p * 64;
#pragma unroll
        for (int c8 = 0; c8 < 8; ++c8) {
          uint4 u = *reinterpret_cast<const uint4*>(rp + ((c8 ^ (p & 7)) << 3));
          const float2 x0 = mul2(bf2f(u.x), inv2), x1 = mul2(bf2f(u.y), inv2), x2 = mul2(bf2f(u.z), inv2), x3 = mul2(bf2f(u.w), inv2);
          u.x = pack2(x0.x, x0.y); u.y = pack2(x1.x, x1.y); u.z = pack2(x2.x, x2.y); u.w = pack2(x3.x, x3.y);
          dst[zu * 16 + grp * 8 + c8] = u;
        }
        // the chunk GEMMs may read this row before or after the zeros land: its dO row is multiplied by inv = 0 either way
        if (!same)
#pragma unroll
          for (int c8 = 0; c8 < 8; ++c8) *reinterpret_cast<uint4*>(rp + c8 * 8) = make_uint4(0u, 0u, 0u, 0u);
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) { arrive(dze + d); arrive(dzm + d); }
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
  static constexpr int NST = CZ >= 512 ? 2 : 3;      // c_z 512: an 80 KiB stage
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
// under this tile's compute.  d_msa 256 (IntelliFold-v2): a CTA PAIR (cluster of two) takes each tile, each CTA 128 of the channels --
// its own x half, Wf columns, dm half and rows of R -- so every per-CTA buffer is the d_msa 128 kernel's; the four partial row sums
// (two warp halves x two CTAs) are exchanged through distributed shared memory (each thread stores its pair into the peer's ring and
// arrives on the peer's barrier) and added in one fixed order in both CTAs.  The dW GEMM's A operand (mask . xh) is exactly M = 128 channels: the masked column sums leave the
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
constexpr int SMEM = 1024 + NSA * STAGE_A + NSX * STAGE_X + (XT + XT + 64 * CM) * 2 + 2 * 4 * BS * 8 + 256;
static_assert(XT * 2 + STAT * 4 + MKT <= STAGE_X && STAGE_A % 1024 == 0 && STAGE_X % 1024 == 0, "stage layout");
constexpr uint32_t IDESC_DY = idesc_bf16(128, CM, 0, 1);   // A = [da|db] K-major, B = Wf MN-major (N = 128 channels)
constexpr uint32_t IDESC_DW = idesc_bf16(128, 64, 1, 1);   // A = (mask . xh)^T MN-major (M = 128 channels), B = [da db] MN-major
constexpr int D2 = 2 * CM;                                 // TMEM column of the dW accumulator (dy: 0 and 128)
static_assert(SMEM <= 232448, "one CTA per SM");
}  // namespace wpb

template <typename WT_, int NCTA>
__global__ void __launch_bounds__(wpb::THREADS, 1) opm_wprologue_bwd_sm100(
    int N, int S, int ntiles,
    const __grid_constant__ CUtensorMap damap,    // dA [S][N*32] as (32, N, S), box (32, 1, 128), 64B swizzle
    const __grid_constant__ CUtensorMap dbmap,
    const __grid_constant__ CUtensorMap xmap,     // m [S][N][128 NCTA] as (64, N, S, 2 NCTA), box (64, 1, 128, 2), 128B swizzle
    const __grid_constant__ CUtensorMap smap,     // stats [N][S][2] fp32 as (2S, N), box (256, 1)
    const __grid_constant__ CUtensorMap mmap,     // mask [S][N] bytes, box (16, 128)
    const __nv_bfloat16* __restrict__ WA, const __nv_bfloat16* __restrict__ WB, const WT_* __restrict__ GAM,
    const __grid_constant__ CUtensorMap dmmap,    // dm, same layout as m
    float* __restrict__ PART) {                   // [clusters][128 NCTA k][64 ce] = R
  using namespace wpb;
  constexpr int CMT = CM * NCTA;                  // the row's channels; this CTA's are rank * 128 .. + 127
  const int NSB = S / BS;
  const int rank = NCTA == 2 ? (int)cl_rank() : 0;
  const int cid = blockIdx.x / NCTA, ncl = gridDim.x / NCTA;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  unsigned char* sStage = smb;                                                    // [NSA][da | db]
  unsigned char* sXs = smb + NSA * STAGE_A;                                        // [NSX][x | stats | mask]
  __nv_bfloat16* sA2 = reinterpret_cast<__nv_bfloat16*>(sXs + NSX * STAGE_X);     // [2 kb][128 s][64]: mask . xh
  __nv_bfloat16* sDM = sA2 + XT;                                                  // [2 kb][128 s][64]
  __nv_bfloat16* sWF = sDM + XT;                                                  // [2 n blocks][64 ce][64 k]
  float2* sRED = reinterpret_cast<float2*>(sWF + 64 * CM);                        // [2 tiles][2 CTAs][2 halves][128 rows] (gs, gx)
  uint64_t* bars = reinterpret_cast<uint64_t*>(sRED + 2 * 4 * BS);
  uint64_t* full = bars;             // [NSA]
  uint64_t* empty = full + NSA;      // [NSA]  the dW GEMM of the stage's tile retired
  uint64_t* xfull = empty + NSA;     // [NSX]
  uint64_t* xempty = xfull + NSX;    // [NSX]  count 8
  uint64_t* d1f = xempty + NSX;      // [2]
  uint64_t* d1e = d1f + 2;           // [2]    count 8
  uint64_t* a2f = d1e + 2;           // [1]    count 8
  uint64_t* a2e = a2f + 1;           // [1]
  uint64_t* done = a2e + 1;          // [1]
  uint64_t* xred = done + 1;         // [2] count 256: the peer's compute threads stored their row sums here (NCTA 2)
  uint32_t* tslot = reinterpret_cast<uint32_t*>(xred + 2);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int k = 0; k < NSA; ++k) { bar_init(full + k, 1); bar_init(empty + k, 1); }
    for (int k = 0; k < NSX; ++k) { bar_init(xfull + k, 1); bar_init(xempty + k, 8); }
    for (int k = 0; k < 2; ++k) { bar_init(d1f + k, 1); bar_init(d1e + k, 8); bar_init(xred + k, 256); }
    bar_init(a2f, 8); bar_init(a2e, 1);
    bar_init(done, 1);
    bar_init_fence();
  }
  // Wf [ce][k] = W[ce][k] gamma[k] in bf16 (this CTA's 128 k), MN-major for the dy GEMM: n block k / 64, row ce, 128B swizzle
  for (int v = tid; v < 64 * CM / 2; v += THREADS) {
    const int row = v / (CM / 2), k = (v % (CM / 2)) * 2, kg = rank * CM + k;
    const __nv_bfloat16* w = row < CH ? WA + row * CMT : WB + (row - CH) * CMT;
    const float2 wv = bf2f(*reinterpret_cast<const uint32_t*>(w + kg));
    *reinterpret_cast<uint32_t*>(sWF + (k >> 6) * 64 * 64 + sw128(row, k & 63)) = pack2(wv.x * to_f(GAM[kg]), wv.y * to_f(GAM[kg + 1]));
  }
  if (warp == 1) tmem_alloc(tslot, 512);
  fence_proxy_async();
  tc_fence_before();
  __syncthreads();
  if constexpr (NCTA == 2) cl_sync();             // the peer's barriers are initialised before any remote arrive
  tc_fence_after();
  const uint32_t tmem = *tslot;
  auto stage_ptr = [&](int st) { return sStage + st * STAGE_A; };
  auto xstage_ptr = [&](int st) { return sXs + st * STAGE_X; };

  if (warp == 0) {
    if (lane == 0) {
      int g = 0;
      for (int t = cid; t < ntiles; t += ncl, ++g) {
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
      for (int t = cid; t < ntiles; t += ncl, ++g) {
        const int i = t / NSB, s0 = (t % NSB) * BS, st = g % NSX;
        if (g >= NSX) wait(xempty + st, ((g / NSX) - 1) & 1);
        unsigned char* p = xstage_ptr(st);
        expect_tx(xfull + st, XT * 2 + STAT * 4 + MKT);
        load_4d(&xmap, p, xfull + st, 0, i, s0, rank * KB);
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
      for (int t = cid; t < ntiles; t += ncl, ++lt) {
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
    for (int t = cid; t < ntiles; t += ncl, ++lt) {
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
      // the gradient row is read from TMEM twice (here and for dm): not held across the row-sum exchange, where 64 more live
      // registers spilled; the dy buffer is released after the second read (the next dy GEMM uses the other buffer)
      // the gradient row 32 channels at a time, read from TMEM twice (row sums, then dm): a thread holding all 64 next to its x row
      // spilled; the dy buffer is released after the second read (the next dy GEMM uses the other buffer)
      const float mean = stt.x, rstd = stt.y;
      const float2 nm = make_float2(-mean, -mean), rs = make_float2(rstd, rstd), m2 = make_float2(mk, mk);
      const uint32_t gt = tmem_at(tmem + b * CM + wh * 64, q * 32, 0);
      float2 gsa = make_float2(0.f, 0.f), gsb = gsa, gxa = gsa, gxb = gsa;
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        float gv[32];
        tmem_ld32(gt + hh * 32, gv);
        tmem_wait_ld();
#pragma unroll
        for (int c4 = 0; c4 < 4; ++c4) {
          const int c8 = hh * 4 + c4;
          const uint32_t uw[4] = {xu[c8].x, xu[c8].y, xu[c8].z, xu[c8].w};
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const int e = c4 * 4 + k;
            const float2 xh = mul2(add2(bf2f(uw[k]), nm), rs);
            const float2 g2 = mul2(make_float2(gv[2 * e], gv[2 * e + 1]), m2);
            if (k & 1) { gsb = add2(gsb, g2); gxb = fma2(g2, xh, gxb); }
            else { gsa = add2(gsa, g2); gxa = fma2(g2, xh, gxa); }
          }
        }
      }
      gsa = add2(gsa, gsb); gxa = add2(gxa, gxb);
      float2* red = sRED + (NCTA == 2 ? (lt & 1) * 4 * BS : 0);   // a CTA pair: double-buffered (the peer runs up to a tile ahead)
      const float2 mine = make_float2(gsa.x + gsa.y, gxa.x + gxa.y);
      red[(rank * 2 + wh) * BS + r] = mine;
      if constexpr (NCTA == 2) {                 // and into the peer's ring, then arrive on the peer's barrier (release: the store first)
        const uint32_t peer = (uint32_t)(rank ^ 1);
        const uint32_t ra_ = cl_map(sa(red + (rank * 2 + wh) * BS + r), peer);
        asm volatile("st.shared::cluster.v2.f32 [%0], {%1, %2};\n" :: "r"(ra_), "f"(mine.x), "f"(mine.y) : "memory");
        asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];\n" :: "r"(cl_map(sa(xred + (lt & 1)), peer)) : "memory");
      }
      if (lt >= 1) wait(a2e, (lt - 1) & 1);      // the dW GEMM of the previous tile retired: sA2 is free
      if (leader) bulk_wait_read<0>();           // the previous dm store has left sDM
      named_sync(1, 256);
      if constexpr (NCTA == 2) wait_cluster(xred + (lt & 1), (lt >> 1) & 1);
      // the partial sums in one fixed order (CTA 0's halves, then CTA 1's): every thread of a row, in both CTAs, agrees bitwise
      float2 p0 = red[r];
#pragma unroll
      for (int h = 1; h < 2 * NCTA; ++h) { const float2 ph = red[h * BS + r]; p0.x += ph.x; p0.y += ph.y; }
      const float ra = -rstd * p0.x * (1.f / CMT), rb = rstd * p0.y * (1.f / CMT);
      __nv_bfloat16* dmr = sDM + wh * BS * 64 + r * 64;
      __nv_bfloat16* a2r = sA2 + wh * BS * 64 + r * 64;
      {
        const float2 ra2 = make_float2(ra, ra), nrb = make_float2(-rb, -rb);
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          float gv[32];
          tmem_ld32(gt + hh * 32, gv);
          tmem_wait_ld();
          if (hh == 1) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(d1e + b); }
#pragma unroll
          for (int c4 = 0; c4 < 4; ++c4) {
            const int c8 = hh * 4 + c4;
            const uint32_t uw[4] = {xu[c8].x, xu[c8].y, xu[c8].z, xu[c8].w};
            uint4 d, a;
            uint32_t* dw_ = reinterpret_cast<uint32_t*>(&d);
            uint32_t* aw = reinterpret_cast<uint32_t*>(&a);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              const int e = c4 * 4 + k;
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
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(a2f);
      named_sync(1, 256);
      if (leader) { store_4d(&dmmap, sDM, 0, i, s0, rank * KB); bulk_commit(); }
    }
    if (leader) bulk_wait<0>();
    if (wh == 0) {                                 // R: TMEM lane = k (0-127 of this CTA's), 64 columns ce
      wait(done, 0);
      tc_fence_after();
      float v[32];
      for (int c0 = 0; c0 < 64; c0 += 32) {
        tmem_ld32(tmem_at(tmem + D2, q * 32, c0), v);
        tmem_wait_ld();
        float* out = PART + ((size_t)cid * CMT + rank * CM + r) * 64 + c0;
#pragma unroll
        for (int k = 0; k < 32; k += 4) *reinterpret_cast<float4*>(out + k) = make_float4(v[k], v[k + 1], v[k + 2], v[k + 3]);
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if constexpr (NCTA == 2) cl_sync();             // no CTA leaves while its peer may still store into it
  if (warp == 1) tmem_dealloc(tmem, 512);
}

// dWa = gamma . R_a + ssa_a (x) beta, dWb likewise, dgamma[k] = sum_ce W[ce][k] R[k][ce], dbeta[k] = sum_ce W[ce][k] ssa[ce]
// (R [CM][CE] summed over the CTAs, ssa [CE] the masked column sums of [da db]; CE = 64 (OPM) or 128 (Chai-1) channels, the first half
// Wa's), one block.
template <int CM, int CE, typename GT>
__global__ void __launch_bounds__(256) opm_wpbwd_finalize(const float* __restrict__ RED, const float* __restrict__ SSA,
                                                          const __nv_bfloat16* __restrict__ WA, const __nv_bfloat16* __restrict__ WB,
                                                          const GT* __restrict__ GAM, const GT* __restrict__ BET, float* __restrict__ DWA,
                                                          float* __restrict__ DWB, GT* __restrict__ DGAM, GT* __restrict__ DBET) {
  constexpr int HALF = CE / 2;
  constexpr bool STAGED = CM * CE <= 8192;        // d_msa 256: R (64 KiB) is read from global memory instead
  __shared__ float sRs[STAGED ? CM * CE : 1], sS[CE], sG[CM], sB[CM];
  const int t = threadIdx.x;
  if (STAGED) for (int v = t; v < CM * CE; v += 256) sRs[v] = __ldcg(RED + v);
  const float* sR = STAGED ? sRs : RED;
  if (t < CE) sS[t] = __ldcg(SSA + t);
  if (t < CM) { sG[t] = to_f(GAM[t]); sB[t] = to_f(BET[t]); }
  __syncthreads();
  for (int v = t; v < CE * CM; v += 256) {
    const int ce = v / CM, k = v % CM;
    const float val = sG[k] * sR[k * CE + ce] + sS[ce] * sB[k];
    if (ce < HALF) DWA[ce * CM + k] = val; else DWB[(ce - HALF) * CM + k] = val;
  }
  if (t < CM) {
    float dg = 0.f, db = 0.f;
    for (int ce = 0; ce < CE; ++ce) {
      const float w = __bfloat162float(ce < HALF ? WA[ce * CM + t] : WB[(ce - HALF) * CM + t]);
      dg += w * sR[t * CE + ce];
      db += w * sS[ce];
    }
    DGAM[t] = from_f<GT>(dg); DBET[t] = from_f<GT>(db);
  }
}

// =============================================================================================================================
// CHAI-1: 8 groups of 8-wide projections (d_msa 64, d_hidden 64), per group the outer product over the MSA rows (cuBLAS batched
// GEMM, O [8, N 8, N 8]), the 512 products of a pair LayerNorm'd (eps 0.1) before the out-projection, no count division.
//
//   prologue       LayerNorm + both projections + mask -> a, b [8, S N, 8] group-major (the batched GEMM's [S, N 8] operands)
//   epilogue       per pair: LayerNorm of the 512 products in shared memory, z = xhat . W'^T + b' (+ residual); W' = W_out . gamma
//                  (columns), b' = b_out + W_out beta (the product norm's affine folded on the host); y (xhat) and its statistics saved
//   LN backward    dO = rstd (dy - mean dy - y mean(dy y)) into the grouped layout (dy = dz W' by cuBLAS)
//   prologue bwd   dm, R = (mask . xh | mask)^T [da db] (the opm prologue backward with 64 + 64 channels)
__device__ __forceinline__ void load_5d(const void* map, void* dst, uint64_t* bar, int c0, int c1, int c2, int c3, int c4) {
  asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6, %7}], [%2];\n"
               :: "r"(sa(dst)), "l"(map), "r"(sa(bar)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4) : "memory");
}

__device__ __forceinline__ int nctaid() { int v; asm volatile("mov.u32 %0, %%nctaid.x;" : "=r"(v)); return v; }
__device__ __forceinline__ void store_5d(const void* map, const void* src, int c0, int c1, int c2, int c3, int c4) {
  asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4, %5, %6}], [%1];\n"
               :: "l"(map), "r"(sa(src)), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4) : "memory");
}

// ---------------------------------------------------------------------------------------------
// CHAI PROLOGUE.  The opm prologue's roles and LayerNorm; a tile is 8 tokens x 16 MSA rows (row r = 8 s + i), so that a group's 8
// channels of the 8 tokens of one MSA row are 128 contiguous bytes of a / b.  The projection runs as D[row][ch] = y[row][:] . W[ch][:]
// (TMEM lane = row, column = one of the 128 channels: left 0-63, right 64-127); a drain thread holds the 8 channels of a group
// contiguously and writes them as one 16-byte chunk of a [side][g][128 rows][8 k] staging tile, stored by one 3-D TMA box a side.
namespace cpro {
constexpr int BS = 128, NST = 4, NACC = 2;
constexpr int XT = BS * 64;
constexpr int TT = 2 * 8 * BS * 8;
constexpr int THREADS = 448, MT = BS * 16;
constexpr int SMEM = 1024 + (NST * XT + NACC * TT + 128 * 64) * 2 + NST * MT + 2 * 64 * 4 + 512;
constexpr uint32_t IDESC = idesc_bf16(128, 128);
static_assert(SMEM <= 232448, "one CTA per SM");
}  // namespace cpro

template <typename WT>
__global__ void __launch_bounds__(448, 1) opm_chai_prologue_sm100(
    int N, int S, int ntiles, float eps,
    const __grid_constant__ CUtensorMap xmap,      // m [S][N][64] as (64, N, S), box (64, 8, 16), 128B swizzle
    const __grid_constant__ CUtensorMap amap,      // a [8][S N][8] as ((i, k) 8 N, S s, 8 g), box (64, 16, 8)
    const __grid_constant__ CUtensorMap bmap,      // b, same
    const __grid_constant__ CUtensorMap mmap,      // mask [S][N] bytes, box (16, 16)
    const __grid_constant__ CUtensorMap wamap,     // W_left [64][64], box (64, 64), 128B swizzle
    const __grid_constant__ CUtensorMap wbmap,     // W_right
    const WT* __restrict__ LNW, const WT* __restrict__ LNB,
    float2* __restrict__ STATS) {                  // [N][S] (mean, rstd) or nullptr
  using namespace cpro;
  const int NSB = S / 16;                        // tile t: tokens 8 (t / NSB) .. + 7, MSA rows 16 (t % NSB) .. + 15; row r = 8 s + i
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sX = reinterpret_cast<__nv_bfloat16*>(smb);
  __nv_bfloat16* sT = sX + NST * XT;
  __nv_bfloat16* sW = sT + NACC * TT;                            // [128 ch][64 k] swizzled: W_left, W_right
  uint8_t* sM = reinterpret_cast<uint8_t*>(sW + 128 * 64);
  float* sLN = reinterpret_cast<float*>(sM + NST * MT);
  uint64_t* bars = reinterpret_cast<uint64_t*>(sLN + 2 * 64);
  uint64_t* xfull = bars;
  uint64_t* yfull = xfull + NST;
  uint64_t* xempty = yfull + NST;
  uint64_t* accf = xempty + NST;
  uint64_t* acce = accf + NACC;
  uint64_t* wful = acce + NACC;
  uint32_t* tslot = reinterpret_cast<uint32_t*>(wful + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { bar_init(xfull + s, 1); bar_init(yfull + s, 8); bar_init(xempty + s, 1); }
    for (int b = 0; b < NACC; ++b) { bar_init(accf + b, 1); bar_init(acce + b, 4); }
    bar_init(wful, 1);
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, NACC * 128);
  if (tid < 64) { sLN[tid] = to_f(LNW[tid]); sLN[64 + tid] = to_f(LNB[tid]); }
  fence_proxy_async();
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;

  if (warp == 0) {
    if (lane == 0) {
      expect_tx(wful, 128 * 64 * 2);
      load_2d(&wamap, sW, wful, 0, 0);
      load_2d(&wbmap, sW + 64 * 64, wful, 0, 0);
      int g = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++g) {
        const int i0 = (t / NSB) * 8, s0 = (t % NSB) * 16;
        const int st = g % NST;
        if (g >= NST) wait(xempty + st, ((g / NST) - 1) & 1);
        expect_tx(xfull + st, XT * 2 + 256);
        load_3d(&xmap, sX + st * XT, xfull + st, 0, i0, s0);
        load_2d(&mmap, sM + st * MT, xfull + st, i0 & ~15, s0);
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
        for (int ks = 0; ks < 4; ++ks) mma_ss(tmem + b * 128, desc_k128(y + ks * 16), desc_k128(sW + ks * 16), IDESC, ks ? 1u : 0u);
        mma_commit(accf + b);
        mma_commit(xempty + st);
      }
    }
  } else if (warp <= 9) {
    const int r = (warp - 2) * 16 + (lane & 15), hf = lane >> 4;
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int i0 = (t / NSB) * 8, s0 = (t % NSB) * 16, st = lt % NST;
      wait(xfull + st, (lt / NST) & 1);
      const bool mkb = sM[st * MT + (r >> 3) * 16 + (i0 & 15) + (r & 7)] != 0;
      float mean, rstd;
      ln_half_row<64>(sX + st * XT, r, hf, sLN, mkb, eps, mean, rstd);
      if (STATS != nullptr && hf == 0) STATS[(size_t)(i0 + (r & 7)) * S + s0 + (r >> 3)] = make_float2(mean, rstd);
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(yfull + st);
    }
  } else {
    // drain: TMEM lane = MSA row s (q * 32 + lane), 128 columns = channels (g, k) of the left (0-63) and right (64-127) projections
    const int q = warp & 3, sl = q * 32 + lane;
    const bool leader = (warp == 10 && lane == 0);
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int i0 = (t / NSB) * 8, s0 = (t % NSB) * 16, b = lt % NACC;
      wait(accf + b, (lt / NACC) & 1);
      tc_fence_after();
      if (leader) bulk_wait_read<NACC - 1>();
      named_sync(2, 128);
#pragma unroll
      for (int c = 0; c < 4; ++c) {              // 32 channels: side c >> 1, groups (c & 1) * 4 .. + 3
        float acc[32];
        tmem_ld32(tmem_at(tmem + b * 128, q * 32, c * 32), acc);
        tmem_wait_ld();
        if (c == 3) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(acce + b); }
#pragma unroll
        for (int gg = 0; gg < 4; ++gg) {
          const int g = (c & 1) * 4 + gg;
          uint4 o;
          o.x = pack2(acc[gg * 8 + 0], acc[gg * 8 + 1]); o.y = pack2(acc[gg * 8 + 2], acc[gg * 8 + 3]);
          o.z = pack2(acc[gg * 8 + 4], acc[gg * 8 + 5]); o.w = pack2(acc[gg * 8 + 6], acc[gg * 8 + 7]);
          *reinterpret_cast<uint4*>(sT + b * TT + (((c >> 1) * 8 + g) * BS + sl) * 8) = o;
        }
      }
      fence_proxy_async();
      named_sync(2, 128);
      if (leader) {
        store_3d(&amap, sT + b * TT, i0 * 8, s0, 0);
        store_3d(&bmap, sT + b * TT + 8 * BS * 8, i0 * 8, s0, 0);
        bulk_commit();
      }
    }
    if (leader) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, NACC * 128);
}

// ---------------------------------------------------------------------------------------------
// CHAI EPILOGUE.  Tile = 128 pairs (4 i x 32 j).  The pair tile of O lands by eight 4-D TMA boxes ((j % 8, l), k, j / 8, g N + i), one
// per group, in the no-swizzle K-major core-matrix layout [16 pair octets][8 k][8 pairs][8 l] (a core matrix = 8 pairs x 16 bytes, 128
// contiguous bytes, which are also 128 contiguous bytes of O): the K-major operand with the [pair, (k, l)] gather done by the copy
// engine; eight warps
// normalise it in place (two threads a pair, 256 products each: shifted sums as the blocks land, then (x - mean) rstd in bf16, the
// rounding the module's bf16 product norm has); one thread multiplies it by W' (N = c_z, K = 512, W' streamed in 64-deep blocks)
// into one of two TMEM accumulators, four warps add b', round, add the residual and write the rows.  With `save` the normalising
// threads also write y (the operand of dW', 512 contiguous bytes of a pair's row each) and the statistics.  The tile buffer is single
// but tracked per group: a group's next block loads once the GEMM has read this one, the statistics accumulate as the blocks land.
template <int CZ>
struct CEp {
  static constexpr int BI = 4, BJ = 32, NP = 128;
  static constexpr int AT = NP * 512;
  static constexpr int BST = CZ * 64;
  static constexpr int NBS = 2;
  static constexpr int NB64 = CZ / 64, OBLK = NP * 64;   // output staging: a ring of two [BI][BJ][64] blocks (64 columns each)
  static constexpr int THREADS = 512;              // 0: O, 1: MMA, 2-9: LayerNorm, 10-13: drain, 14: W', 15: y stores
  static constexpr int SMEM = 1024 + (AT + NBS * BST + 2 * OBLK) * 2 + CZ * 4 + 512;
  static constexpr uint32_t IDESC = idesc_bf16(128, CZ);
  static constexpr int TMEM_COLS = 2 * CZ <= 256 ? 256 : 512;
  static_assert(SMEM <= 232448, "one CTA per SM");
};

template <int CZ, bool SAVE>
__global__ void __launch_bounds__(512, 1) opm_chai_epilogue_sm100(
    int N, int ntiles, float eps,
    const __grid_constant__ CUtensorMap omap,      // O [8][N 8][N 8] as ((j % 8, l) 64, k 8, j / 8, g N + i), box (64, 8, 4, 4), no swizzle
    const __grid_constant__ CUtensorMap wmap,      // W' [CZ][512], box (64, CZ), 128B swizzle
    const __grid_constant__ CUtensorMap zmap,      // z [N][N][CZ] as (64, N j, CZ / 64, N i), box (64, BJ, 1, BI), 128B swizzle
    const __grid_constant__ CUtensorMap rmap,      // the residual, same layout
    const __grid_constant__ CUtensorMap ymap,      // y [N N][512] as (l 8, j % 8, (g, k) 64, j / 8, i), box (8, 8, 8, 4, 4): a group block
    const float* __restrict__ BIAS,                // b' [CZ] fp32
    int has_res, float* __restrict__ STATS,
    const int* __restrict__ ASYM,                  // chain ids [N] (the interchain mask: a pair of two chains gets no update), or nullptr
    __nv_bfloat16* __restrict__ WS) {              // training with ASYM: [N N] 1 / 0 (same chain or not), the dbias GEMV's weights
  using C = CEp<CZ>;
  constexpr int BI = C::BI, BJ = C::BJ, NP = C::NP, BST = C::BST, NBS = C::NBS, NB64 = C::NB64, OBLK = C::OBLK;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sA = reinterpret_cast<__nv_bfloat16*>(smb);     // [8 g][128 pairs][64]
  __nv_bfloat16* sB = sA + C::AT;                                 // [NBS][CZ][64]
  __nv_bfloat16* sOut = sB + NBS * BST;                           // [2][BI][BJ][64]
  float* sBias = reinterpret_cast<float*>(sOut + 2 * OBLK);       // [CZ]
  uint64_t* bars = reinterpret_cast<uint64_t*>(sBias + CZ);
  // per group g: the O block landed / normalised (pairs of groups (gg, gg + 4): count 8 LN warps) / read by the GEMM / (training) y
  // stored from it.  Groups go in the order 0, 4, 1, 5, 2, 6, 3, 7 (the two thread halves of a row normalise gg and gg + 4 at
  // once), so the next tile's first blocks load while this tile's last ones are normalised and multiplied
  uint64_t* afull = bars;            // [8]
  uint64_t* anorm = afull + 8;       // [4] count 8
  uint64_t* aempty = anorm + 4;      // [8]
  uint64_t* ystore = aempty + 8;     // [8] (training) the group block's y store has left shared memory
  uint64_t* bfull = ystore + 8;      // [NBS]
  uint64_t* bempty = bfull + NBS;    // [NBS]
  uint64_t* accf = bempty + NBS;     // [2]
  uint64_t* acce = accf + 2;         // [2] count 4
  uint64_t* resf = acce + 2;         // [2] a staging block's residual landed
  uint32_t* tslot = reinterpret_cast<uint32_t*>(resf + 2);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int njb = N / BJ;
  if (tid == 0) {
    for (int k = 0; k < 8; ++k) { bar_init(afull + k, 1); bar_init(aempty + k, 1); bar_init(ystore + k, 1); }
    for (int k = 0; k < 4; ++k) bar_init(anorm + k, 8);
    for (int k = 0; k < NBS; ++k) { bar_init(bfull + k, 1); bar_init(bempty + k, 1); }
    for (int k = 0; k < 2; ++k) { bar_init(accf + k, 1); bar_init(acce + k, 4); bar_init(resf + k, 1); }
    bar_init_fence();
  }
  for (int c = tid; c < CZ; c += C::THREADS) sBias[c] = BIAS[c];   // b' in shared memory (per-element __ldg of it sat on the drain's path)
  if (warp == 1) tmem_alloc(tslot, C::TMEM_COLS);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;

  if (warp == 0) {
    if (lane == 0) {
      int lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int i0 = (t / njb) * BI, j0 = (t % njb) * BJ;
        for (int q = 0; q < 8; ++q) {
          const int g = (q & 1) * 4 + (q >> 1);
          if (lt >= 1) {
            wait(aempty + g, (lt - 1) & 1);
            if (SAVE) wait(ystore + g, (lt - 1) & 1);
          }
          expect_tx(afull + g, NP * 64 * 2);
          load_4d(&omap, sA + g * NP * 64, afull + g, 0, 0, j0 >> 3, g * N + i0);
        }
      }
    }
  } else if (warp == 14) {
    if (lane == 0) {
      int gb = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x)
        for (int q = 0; q < 8; ++q, ++gb) {
          const int g = (q & 1) * 4 + (q >> 1), s = gb % NBS;
          if (gb >= NBS) wait(bempty + s, ((gb / NBS) - 1) & 1);
          expect_tx(bfull + s, BST * 2);
          load_2d(&wmap, sB + s * BST, bfull + s, g * 64, 0);
        }
    }
  } else if (warp == 15) {
    // (training) y = the normalised group blocks, stored by TMA straight from the GEMM's operand (one 5-D box a group: 16-byte rows
    // of 8 products); the threads wrote y as rows of their own before, 32 lines per warp store (~200 of 544 us at L768)
    if (SAVE && lane == 0) {
      int lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int i0 = (t / njb) * BI, j0 = (t % njb) * BJ;
        for (int gg = 0; gg < 4; ++gg) {
          wait(anorm + gg, lt & 1);
          store_5d(&ymap, sA + gg * NP * 64, 0, 0, gg * 8, j0 >> 3, i0);
          store_5d(&ymap, sA + (gg + 4) * NP * 64, 0, 0, (gg + 4) * 8, j0 >> 3, i0);
          bulk_commit();
          bulk_wait_read<0>();
          arrive(ystore + gg);
          arrive(ystore + gg + 4);
        }
      }
      bulk_wait<0>();
    }
  } else if (warp == 1) {
    if (lane == 0) {
      int lt = 0, gb = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int b = lt & 1;
        if (lt >= 2) wait(acce + b, ((lt >> 1) - 1) & 1);
        for (int q = 0; q < 8; ++q, ++gb) {
          const int g = (q & 1) * 4 + (q >> 1), s = gb % NBS;
          if (!(q & 1)) wait(anorm + (q >> 1), lt & 1);
          wait(bfull + s, (gb / NBS) & 1);
          tc_fence_after();
#pragma unroll
          for (int ks = 0; ks < 4; ++ks)
            mma_ss(tmem + b * CZ, sdesc(sa(sA + g * NP * 64 + ks * 2 * 64), 128, 1024, 0), desc_k128(sB + s * BST + ks * 16), C::IDESC,
                   (q | ks) ? 1u : 0u);
          mma_commit(bempty + s);
          mma_commit(aempty + g);
        }
        mma_commit(accf + b);
      }
    }
  } else if (warp <= 9) {
    // LayerNorm: pair row r, groups hf * 4 .. + 3 (lanes l and l ^ 16 share a row).  The statistics in one pass as the blocks land,
    // shifted by the pair's first product (sums of x - c and (x - c)^2 in fp32), then the normalisation group by group
    const int r = (warp - 2) * 16 + (lane & 15), hf = lane >> 4;
    const int roff = (r >> 3) * 512 + (r & 7) * 8;
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int i0 = (t / njb) * BI, j0 = (t % njb) * BJ;
      wait(afull, lt & 1);                         // group 0: the shift
      const float c0 = __bfloat162float(sA[roff]);
      const float2 nc = make_float2(-c0, -c0);
      float2 sa = make_float2(0.f, 0.f), sb = sa, va = sa, vb = sa;
#pragma unroll 1
      for (int gg = 0; gg < 4; ++gg) {
        const int g = hf * 4 + gg;
        wait(afull + g, lt & 1);
        const __nv_bfloat16* row = sA + g * NP * 64 + roff;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          const uint4 u = *reinterpret_cast<const uint4*>(row + c * 64);
          const float2 x0 = add2(bf2f(u.x), nc), x1 = add2(bf2f(u.y), nc), x2 = add2(bf2f(u.z), nc), x3 = add2(bf2f(u.w), nc);
          sa = add2(sa, add2(x0, x1)); sb = add2(sb, add2(x2, x3));
          va = fma2(x0, x0, fma2(x1, x1, va)); vb = fma2(x2, x2, fma2(x3, x3, vb));
        }
      }
      float s1 = sa.x + sa.y + sb.x + sb.y, s2 = va.x + va.y + vb.x + vb.y;
      s1 += __shfl_xor_sync(0xffffffffu, s1, 16);
      s2 += __shfl_xor_sync(0xffffffffu, s2, 16);
      const float d1 = s1 * (1.f / 512.f);
      const float mean = c0 + d1;
      const float rstd = rsqrtf(fmaxf(s2 * (1.f / 512.f) - d1 * d1, 0.f) + eps);
      const float2 nm = make_float2(-mean, -mean), rs = make_float2(rstd, rstd);
      const long pr = (long)(i0 + (r >> 5)) * N + j0 + (r & 31);
      // the interchain mask in training: a pair of two chains is normalised to zeros (its y, its GEMM row: the drain keeps the residual
      // there anyway), rstd 0 (its product-norm gradient vanishes) and weight 0 in the dbias sum
      const uint32_t keep = (!SAVE || ASYM == nullptr || __ldg(ASYM + i0 + (r >> 5)) == __ldg(ASYM + j0 + (r & 31))) ? 0xffffffffu : 0u;
#pragma unroll 1
      for (int gg = 0; gg < 4; ++gg) {
        __nv_bfloat16* row = sA + (hf * 4 + gg) * NP * 64 + roff;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          uint4 u = *reinterpret_cast<const uint4*>(row + c * 64);
          const float2 x0 = mul2(add2(bf2f(u.x), nm), rs), x1 = mul2(add2(bf2f(u.y), nm), rs);
          const float2 x2 = mul2(add2(bf2f(u.z), nm), rs), x3 = mul2(add2(bf2f(u.w), nm), rs);
          u.x = pack2(x0.x, x0.y) & keep; u.y = pack2(x1.x, x1.y) & keep; u.z = pack2(x2.x, x2.y) & keep; u.w = pack2(x3.x, x3.y) & keep;
          *reinterpret_cast<uint4*>(row + c * 64) = u;
        }
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) arrive(anorm + gg);       // groups gg (lanes 0-15) and gg + 4 (lanes 16-31) of this warp's 16 pairs
      }
      if (SAVE && hf == 0) {
        reinterpret_cast<float2*>(STATS)[pr] = make_float2(mean, keep ? rstd : 0.f);
        if (WS != nullptr) WS[pr] = __float2bfloat16_rn(keep ? 1.f : 0.f);
      }
    }
  } else {
    // drain: TMEM lane = pair p = q * 32 + lane (i0 + p / 32, j0 + p % 32), columns = c_z, 64 at a time through a ring of two staging
    // blocks stored by TMA; a block's residual is loaded as soon as the block two before it has left the ring slot (so the next tile's
    // first blocks land under this tile's last ones).  Rows of z / the residual per thread (32 lines a warp access) were ~190 of 544 us
    const int q = warp & 3, p = q * 32 + lane, il = p >> 5, jl = p & 31;
    const bool lead_thr = (warp == 10 && lane == 0);
    const int nmy = (int)blockIdx.x < ntiles ? (ntiles - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
    auto res_load = [&](int k) {                   // the k-th block of this CTA (tile k / NB64, columns 64 (k % NB64))
      const int t = (int)blockIdx.x + (k / NB64) * (int)gridDim.x, h = k % NB64;
      expect_tx(resf + (k & 1), OBLK * 2);
      load_4d(&rmap, sOut + (k & 1) * OBLK, resf + (k & 1), 0, (t % njb) * BJ, h, (t / njb) * BI);
    };
    if (lead_thr && has_res) for (int k = 0; k < 2 && k < nmy * NB64; ++k) res_load(k);
    int lt = 0, kb = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int i0 = (t / njb) * BI, j0 = (t % njb) * BJ, b = lt & 1;
      const uint32_t keep = (ASYM == nullptr || __ldg(ASYM + i0 + il) == __ldg(ASYM + j0 + jl)) ? 0xffffffffu : 0u;
      wait(accf + b, (lt >> 1) & 1);
      tc_fence_after();
#pragma unroll 1
      for (int h = 0; h < NB64; ++h, ++kb) {
        const int s = kb & 1;
        float v[64];
        tmem_ld32(tmem_at(tmem + b * CZ, q * 32, h * 64), v);
        tmem_ld32(tmem_at(tmem + b * CZ, q * 32, h * 64 + 32), v + 32);
        tmem_wait_ld();
        if (h == NB64 - 1) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(acce + b); }
        if (has_res) wait(resf + s, (kb >> 1) & 1);   // (either way the store two blocks back has left this slot: the leader waited
                                                      // for it before the last block's barrier)
        __nv_bfloat16* row = sOut + s * OBLK + p * 64;
        const float* bh = sBias + h * 64;
#pragma unroll
        for (int c8 = 0; c8 < 8; ++c8) {
          const int sw = (c8 ^ (jl & 7)) << 3;
          uint4 o;
          uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
          uint4 rr = make_uint4(0, 0, 0, 0);
          if (has_res) rr = *reinterpret_cast<const uint4*>(row + sw);
          const __nv_bfloat162* rh = reinterpret_cast<const __nv_bfloat162*>(&rr);
          const float4 b0 = *reinterpret_cast<const float4*>(bh + c8 * 8), b1 = *reinterpret_cast<const float4*>(bh + c8 * 8 + 4);
          const float bb[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            __nv_bfloat162 x = __float22bfloat162_rn(make_float2(v[c8 * 8 + u * 2] + bb[2 * u], v[c8 * 8 + u * 2 + 1] + bb[2 * u + 1]));
            *reinterpret_cast<uint32_t*>(&x) &= keep;
            if (has_res) x = __hadd2(x, rh[u]);
            ow[u] = *reinterpret_cast<uint32_t*>(&x);
          }
          *reinterpret_cast<uint4*>(row + sw) = o;
        }
        fence_proxy_async();
        named_sync(1, 128);
        if (lead_thr) {
          store_4d(&zmap, sOut + s * OBLK, 0, j0, h, i0);
          bulk_commit();
          bulk_wait_read<0>();                     // the block has left its slot: the block two ahead may land in it
          if (has_res && kb + 2 < nmy * NB64) res_load(kb + 2);
        }
      }
    }
    if (lead_thr) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, C::TMEM_COLS);
}

// ---------------------------------------------------------------------------------------------
// CHAI LN BACKWARD: dO[g, i 8 + k, j 8 + l] = rstd (dy[p] - mean_p dy - y[p] mean_p (dy y)), p = (g, k, l).  A block = one i x 32 j (32
// pairs, whose dy and y rows are 32 KiB contiguous each): the rows come in by coalesced 16-byte loads into shared memory (chunk
// c of group g at c ^ g, so the eight threads of a pair read eight banks), eight threads a pair (one group each) take the two pair
// sums by three shuffles and write their dO values to a [g][k][32 j][8 l] tile, which leaves by coalesced 16-byte stores (each
// (g, k) row is 512 contiguous bytes of the grouped layout).  Memory-bound: dy and y read once, dO written once.
__global__ void __launch_bounds__(256, 2) opm_chai_lnbwd_sm100(const __nv_bfloat16* __restrict__ DY, const __nv_bfloat16* __restrict__ Y,
                                                               const float* __restrict__ ST, __nv_bfloat16* __restrict__ DO, int N) {
  extern __shared__ __align__(16) unsigned char lsm[];
  uint4* sDY = reinterpret_cast<uint4*>(lsm);              // [32 pairs][8 g][8 chunks] (chunk swizzled by g)
  uint4* sY = sDY + 32 * 64;
  uint4* sO = sY + 32 * 64;                                // [8 g][8 k][32 j] chunks of 8 l
  const int t = threadIdx.x, i = blockIdx.y, j0 = blockIdx.x * 32;
  const size_t NK = (size_t)N * 8;
  const uint4* gdy = reinterpret_cast<const uint4*>(DY + ((size_t)i * N + j0) * 512);
  const uint4* gy = reinterpret_cast<const uint4*>(Y + ((size_t)i * N + j0) * 512);
#pragma unroll
  for (int u = 0; u < 8; ++u) {                            // 2048 chunks of each: (pair, g, c) = v / 64, (v / 8) % 8, v % 8
    const int v = t + u * 256, pr = v >> 6, g = (v >> 3) & 7, c = v & 7;
    const int d = (pr * 8 + g) * 8 + (c ^ g);
    sDY[d] = __ldg(gdy + v);
    sY[d] = __ldg(gy + v);
  }
  __syncthreads();
  const int pl = t >> 3, g = t & 7;
  const uint4* rd = sDY + (pl * 8 + g) * 8;
  const uint4* ry = sY + (pl * 8 + g) * 8;
  float2 s1 = make_float2(0.f, 0.f), s2 = s1;
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    const uint4 du = rd[c ^ g], yu = ry[c ^ g];
    const uint32_t dw[4] = {du.x, du.y, du.z, du.w}, yw[4] = {yu.x, yu.y, yu.z, yu.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) { const float2 d = bf2f(dw[e]), y = bf2f(yw[e]); s1 = add2(s1, d); s2 = fma2(d, y, s2); }
  }
  float a = s1.x + s1.y, b = s2.x + s2.y;
#pragma unroll
  for (int m = 1; m < 8; m <<= 1) { a += __shfl_xor_sync(0xffffffffu, a, m); b += __shfl_xor_sync(0xffffffffu, b, m); }
  const float c1 = a * (1.f / 512.f), c2 = b * (1.f / 512.f), rstd = __ldg(ST + ((size_t)i * N + j0 + pl) * 2 + 1);
  const float2 r2 = make_float2(rstd, rstd), nc1 = make_float2(-c1, -c1), nc2 = make_float2(-c2, -c2);
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const uint4 du = rd[k ^ g], yu = ry[k ^ g];
    const uint32_t dw[4] = {du.x, du.y, du.z, du.w}, yw[4] = {yu.x, yu.y, yu.z, yu.w};
    uint4 o;
    uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 v = mul2(r2, fma2(bf2f(yw[e]), nc2, add2(bf2f(dw[e]), nc1)));
      ow[e] = pack2(v.x, v.y);
    }
    sO[(g * 8 + k) * 32 + pl] = o;
  }
  __syncthreads();
#pragma unroll
  for (int u = 0; u < 8; ++u) {                            // 64 rows (g, k) of 32 chunks: a warp stores one 512-byte row
    const int v = t + u * 256, row = v >> 5, jj = v & 31, gg = row >> 3, kk = row & 7;
    reinterpret_cast<uint4*>(DO + ((size_t)gg * NK + (size_t)i * 8 + kk) * NK + (size_t)j0 * 8)[jj] = sO[v];
  }
}

// ---------------------------------------------------------------------------------------------
// CHAI PROLOGUE BACKWARD: opm_prologue_bwd_sm100 (opm_sm100.cu) with 64 + 64 projection channels.  The gradients come transposed,
// dA^T / dB^T [8][N 8][S] (the batched GEMMs write them so), and land per 64 MSA rows as [64 (g, k)][64 s] 128-byte-swizzled tiles
// (4-D boxes (s, k, g, i)): stacked [s block][side][64 ce][64 s] they are the MN-major A of the dy GEMM (M = 128 s, K = 128 ce,
// B = Wf [128 ce][64 k]) and the K-major B of the dW GEMM (N = 128 ce, K = s; R = (mask . xh | mask)^T [da db], whose row 64 is the
// masked column sums).  Two compute groups take alternate tiles.
namespace cpb {
constexpr int CM = 64, CE = 128, BS = 128;
constexpr int NSA = 2, NSX = 2;
constexpr int DAT = 64 * 64;                    // one [64 ce][64 s] tile; a stage = [2 s blocks][2 sides] of them
constexpr int XT = BS * CM;
constexpr int STAT = BS * 2, MKT = BS * 16;
constexpr int STAGE_A = 4 * DAT * 2;
constexpr int STAGE_X = 20480;
constexpr int A2T = 2 * XT;
constexpr int THREADS = 352;
constexpr int SMEM = 1024 + NSA * STAGE_A + NSX * STAGE_X + (2 * A2T + 2 * XT + CE * CM) * 2 + 256;
static_assert(XT * 2 + STAT * 4 + MKT <= STAGE_X && STAGE_A % 1024 == 0, "stage layout");
static_assert(NSX % 2 == 0, "x ring depth: a multiple of the two compute groups");
constexpr uint32_t IDESC_DY = idesc_bf16(128, CM, 1, 1);   // A = [da db]^T MN-major (M = s), B = Wf MN-major
constexpr uint32_t IDESC_DW = idesc_bf16(128, CE, 1, 0);   // A = (mask . xh | mask)^T MN-major, B = [da db] K-major (rows ce, K = s)
constexpr int D2 = 128;
static_assert(SMEM <= 232448, "one CTA per SM");
}  // namespace cpb

template <typename WT_>
__global__ void __launch_bounds__(cpb::THREADS, 1) opm_chai_prologue_bwd_sm100(
    int N, int S, int ntiles,
    const __grid_constant__ CUtensorMap damap,    // dA^T [8][N 8][S] as (S s, 8 k, 8 g, N i), box (64, 8, 8, 1), 128B swizzle
    const __grid_constant__ CUtensorMap dbmap,
    const __grid_constant__ CUtensorMap xmap,     // m [S][N][64] as (64, N, S), box (64, 1, 128), 128B swizzle
    const __grid_constant__ CUtensorMap smap,     // stats [N][S][2] fp32 as (2S, N), box (256, 1)
    const __grid_constant__ CUtensorMap mmap,     // mask [S][N] bytes, box (16, 128)
    const __nv_bfloat16* __restrict__ WA, const __nv_bfloat16* __restrict__ WB, const WT_* __restrict__ GAM,
    const __grid_constant__ CUtensorMap dmmap,    // dm, same layout as m
    float* __restrict__ PART) {                   // [grid][65][128]: rows k = R^T, row 64 = ssa
  using namespace cpb;
  const int NSB = S / BS;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  unsigned char* sStage = smb;
  unsigned char* sXs = smb + NSA * STAGE_A;
  __nv_bfloat16* sA2 = reinterpret_cast<__nv_bfloat16*>(sXs + NSX * STAGE_X);
  __nv_bfloat16* sDM = sA2 + 2 * A2T;
  __nv_bfloat16* sWF = sDM + 2 * XT;                                              // [128 ce][64 k]
  uint64_t* bars = reinterpret_cast<uint64_t*>(sWF + CE * CM);
  uint64_t* full = bars;
  uint64_t* empty = full + NSA;
  uint64_t* xfull = empty + NSA;
  uint64_t* xempty = xfull + NSX;
  uint64_t* d1f = xempty + NSX;
  uint64_t* d1e = d1f + 2;
  uint64_t* a2f = d1e + 2;
  uint64_t* a2e = a2f + 2;
  uint64_t* done = a2e + 2;
  uint32_t* tslot = reinterpret_cast<uint32_t*>(done + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int k = 0; k < NSA; ++k) { bar_init(full + k, 1); bar_init(empty + k, 1); }
    for (int k = 0; k < NSX; ++k) { bar_init(xfull + k, 1); bar_init(xempty + k, 4); }
    for (int k = 0; k < 2; ++k) { bar_init(d1f + k, 1); bar_init(d1e + k, 4); bar_init(a2f + k, 4); bar_init(a2e + k, 1); }
    bar_init(done, 1);
    bar_init_fence();
  }
  for (int v = tid; v < CE * CM / 2; v += THREADS) {
    const int row = v / (CM / 2), k = (v % (CM / 2)) * 2;
    const __nv_bfloat16* w = row < 64 ? WA + row * CM : WB + (row - 64) * CM;
    const float2 wv = bf2f(*reinterpret_cast<const uint32_t*>(w + k));
    *reinterpret_cast<uint32_t*>(sWF + sw128(row, k)) = pack2(wv.x * to_f(GAM[k]), wv.y * to_f(GAM[k + 1]));
  }
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

  if (warp == 0) {
    if (lane == 0) {
      int g = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++g) {
        const int i = t / NSB, s0 = (t % NSB) * BS, st = g % NSA;
        if (g >= NSA) wait(empty + st, ((g / NSA) - 1) & 1);
        unsigned char* p = (sStage + st * STAGE_A);
        expect_tx(full + st, STAGE_A);
        for (int sb = 0; sb < 2; ++sb) {
          load_4d(&damap, p + (sb * 2) * DAT * 2, full + st, s0 + sb * 64, 0, 0, i);
          load_4d(&dbmap, p + (sb * 2 + 1) * DAT * 2, full + st, s0 + sb * 64, 0, 0, i);
        }
      }
    }
  } else if (warp == 10) {
    if (lane == 0) {
      int g = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++g) {
        const int i = t / NSB, s0 = (t % NSB) * BS, st = g % NSX;
        if (g >= NSX) wait(xempty + st, ((g / NSX) - 1) & 1);
        unsigned char* p = sXs + st * STAGE_X;
        expect_tx(xfull + st, XT * 2 + STAT * 4 + MKT);
        load_3d(&xmap, p, xfull + st, 0, i, s0);
        load_2d(&smap, p + XT * 2, xfull + st, 2 * s0, i);
        load_2d(&mmap, p + XT * 2 + STAT * 4, xfull + st, i & ~15, s0);
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      // tile k's dy GEMM, then tile k - 1's dW GEMM (its (mask . xh) is written once the group has tile k - 1's dy)
      const int nt = blockIdx.x < ntiles ? (ntiles - blockIdx.x + gridDim.x - 1) / gridDim.x : 0;
      for (int k = 0; k <= nt; ++k) {
        if (k < nt) {
          const int st = k % NSA, gr = k & 1;
          wait(full + st, (k / NSA) & 1);
          if (k >= 2) wait(d1e + gr, ((k >> 1) - 1) & 1);
          tc_fence_after();
          const __nv_bfloat16* dab = reinterpret_cast<const __nv_bfloat16*>(sStage + st * STAGE_A);
#pragma unroll
          for (int ks = 0; ks < 8; ++ks)         // K = 128 ce: four k steps in da, four in db; M = 128 s across the two s blocks
            mma_ss(tmem + gr * 64, desc_mn128(dab + (ks >> 2) * DAT + (ks & 3) * 16 * 64, 2 * DAT * 2), desc_mn128(sWF + ks * 16 * 64, CE * 64 * 2),
                   IDESC_DY, ks ? 1u : 0u);
          mma_commit(d1f + gr);
        }
        if (k >= 1) {
          const int lt = k - 1, gr = lt & 1, st = lt % NSA;
          wait(a2f + gr, (lt >> 1) & 1);
          tc_fence_after();
          const __nv_bfloat16* a2 = sA2 + gr * A2T;
          const __nv_bfloat16* dab = reinterpret_cast<const __nv_bfloat16*>(sStage + st * STAGE_A);
#pragma unroll
          for (int ks = 0; ks < BS / 16; ++ks)
            mma_ss(tmem + D2, desc_mn128(a2 + ks * 16 * 64, XT * 2), desc_k128(dab + (ks >> 2) * 2 * DAT + (ks & 3) * 16), IDESC_DW,
                   (lt | ks) ? 1u : 0u);
          mma_commit(empty + st);
          mma_commit(a2e + gr);
        }
      }
      mma_commit(done);
    }
  } else {
    const int gr = (warp - 2) >> 2, q = warp & 3, r = q * 32 + lane;
    const bool leader = (lane == 0 && q == 2);
    int lt = gr;
    for (int t = blockIdx.x + gr * gridDim.x; t < ntiles; t += 2 * nctaid(), lt += 2) {
      const int i = t / NSB, s0 = (t % NSB) * BS, sx = lt % NSX;
      // the row's swizzle, opaque per tile: hoisted out of the tile loop, its 24 chunk addresses spilled
      int r7 = r & 7;
      asm volatile("" : "+r"(r7));
      wait(xfull + sx, (lt / NSX) & 1);
      const unsigned char* px = (sXs + sx * STAGE_X);
      const __nv_bfloat16* xr = reinterpret_cast<const __nv_bfloat16*>(px) + r * 64;
      const float2 stt = reinterpret_cast<const float2*>(px + XT * 2)[r];
      const float mk = px[XT * 2 + STAT * 4 + r * 16 + (i & 15)] ? 1.f : 0.f;
      uint4 xu[8];
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) xu[c8] = *reinterpret_cast<const uint4*>(xr + ((c8 ^ r7) << 3));
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(xempty + sx);
      wait(d1f + gr, (lt >> 1) & 1);
      tc_fence_after();
      const float mean = stt.x, rstd = stt.y;
      // xh and g = mask . dy are recomputed in the second pass from x (packed bf16) and the gradient row (a thread keeping both as
      // fp32 pairs spilled)
      const float2 nm = make_float2(-mean, -mean), rs = make_float2(rstd, rstd), m2 = make_float2(mk, mk);
      float2 gsa = make_float2(0.f, 0.f), gsb = gsa, gxa = gsa, gxb = gsa;
      float gv[32];                              // the gradient row, 32 channels at a time
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) {
        if ((c8 & 3) == 0) { tmem_ld32(tmem_at(tmem + gr * 64, q * 32, c8 * 8), gv); tmem_wait_ld(); }
        const uint32_t uw[4] = {xu[c8].x, xu[c8].y, xu[c8].z, xu[c8].w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int e = (c8 & 3) * 4 + k;
          const float2 xh = mul2(add2(bf2f(uw[k]), nm), rs);
          const float2 g2 = mul2(make_float2(gv[2 * e], gv[2 * e + 1]), m2);
          if (k & 1) { gsb = add2(gsb, g2); gxb = fma2(g2, xh, gxb); }
          else { gsa = add2(gsa, g2); gxa = fma2(g2, xh, gxa); }
        }
      }
      gsa = add2(gsa, gsb); gxa = add2(gxa, gxb);
      const float gs = gsa.x + gsa.y, gx = gxa.x + gxa.y;
      if (lt >= 2) wait(a2e + gr, ((lt >> 1) - 1) & 1);
      if (leader) bulk_wait_read<0>();
      if (gr) named_sync(2, 128); else named_sync(1, 128);
      const float ra = -rstd * gs * (1.f / CM), rb = rstd * gx * (1.f / CM);
      __nv_bfloat16* dmr = sDM + gr * XT + r * 64;
      __nv_bfloat16* a2r = sA2 + gr * A2T + r * 64;
      {
        const float2 ra2 = make_float2(ra, ra), nrb = make_float2(-rb, -rb);
        float gh[32];                            // the gradient row again, 32 channels at a time (TMEM): it is not kept across the barrier
#pragma unroll
        for (int c8 = 0; c8 < 8; ++c8) {
          if ((c8 & 3) == 0) {
            tmem_ld32(tmem_at(tmem + gr * 64, q * 32, c8 * 8), gh);
            tmem_wait_ld();
            if (c8 == 4) { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(d1e + gr); }
          }
          const uint32_t uw[4] = {xu[c8].x, xu[c8].y, xu[c8].z, xu[c8].w};
          uint4 d, a;
          uint32_t* dw_ = reinterpret_cast<uint32_t*>(&d);
          uint32_t* aw = reinterpret_cast<uint32_t*>(&a);
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const int e = c8 * 4 + k;
            const float2 xh = mul2(add2(bf2f(uw[k]), nm), rs);
            const int ee = (c8 & 3) * 4 + k;
            const float2 g2 = mul2(make_float2(gh[2 * ee], gh[2 * ee + 1]), m2);
            const float2 dd = fma2(xh, nrb, fma2(g2, rs, ra2));
            const float2 aa = mul2(xh, m2);
            dw_[k] = pack2(dd.x, dd.y);
            aw[k] = pack2(aa.x, aa.y);
          }
          const int off = (c8 ^ r7) << 3;
          *reinterpret_cast<uint4*>(dmr + off) = d;
          *reinterpret_cast<uint4*>(a2r + off) = a;
          asm volatile("" ::: "memory");        // store each chunk as it is made (batched, the 16 results spilled)
        }
      }
      *reinterpret_cast<uint4*>(a2r + XT + (r7 << 3)) = make_uint4(mk != 0.f ? 0x3f80u : 0u, 0u, 0u, 0u);
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(a2f + gr);
      if (gr) named_sync(2, 128); else named_sync(1, 128);
      if (leader) {
        int off = gr * XT;                       // re-derived here, not kept across the tile (it was spilled)
        asm volatile("" : "+r"(off));
        store_3d(&dmmap, sDM + off, 0, i, s0);
        bulk_commit();
      }
    }
    if (leader) bulk_wait<0>();
    if (gr == 0) {                                 // R: TMEM lane = k (0-63) or the mask row (64), 128 columns ce
      wait(done, 0);
      tc_fence_after();
      if (q < 3) {
        float v[32];
        const int row = q * 32 + lane;
#pragma unroll 1
        for (int c0 = 0; c0 < CE; c0 += 32) {
          tmem_ld32(tmem_at(tmem + D2, q * 32, c0), v);
          tmem_wait_ld();
          if (row <= CM) {
            float* out = PART + ((size_t)blockIdx.x * (CM + 1) + row) * CE + c0;
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
  TORCH_CHECK(CM == 64 || CM == 128 || CM == 256, "d_msa 64, 128 or 256");
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
  else if (CM == 128) prologue_launch<128>(m, mask, lnw, lnb, eps, wa, wb, pb, sp, bp, A2, BT);
  else prologue_launch<256>(m, mask, lnw, lnb, eps, wa, wb, pb, sp, bp, A2, BT);
  return {A2, BT, stats, bits};
}

static const int* asym_ptr(const c10::optional<torch::Tensor>& asym, int64_t n) {
  if (!asym.has_value() || asym->numel() == 0) return nullptr;
  TORCH_CHECK(asym->scalar_type() == torch::kInt32 && asym->is_contiguous() && asym->numel() >= n, "asym: int32 [N] chain ids");
  return asym->data_ptr<int>();
}

template <int NW>
static void epilogue_launch(torch::Tensor& O, torch::Tensor& norm, torch::Tensor& wo, torch::Tensor& bias, int64_t ni, int64_t nj,
                            const torch::Tensor* res, torch::Tensor& out, int64_t after, const int* asym) {
  using C = WEp<NW>;
  const long CZ = wo.size(0);
  TORCH_CHECK(ni % C::BI == 0 && nj % (2 * C::BJ) == 0, "the epilogue tiles ", C::BI, " x ", 2 * C::BJ, " tokens per CTA pair");
  const uint64_t M = (uint64_t)nj * CH;
  CUtensorMap am = make_map<4>(O.data_ptr(), {32, (uint64_t)nj, (uint64_t)ni, 32}, {32, 32 * M, M}, {32, (uint32_t)C::BJ, (uint32_t)C::BI, 2},
                               CU_TENSOR_MAP_SWIZZLE_64B, "O");
  CUtensorMap bm = make_map<2>(wo.data_ptr(), {(uint64_t)NCH, (uint64_t)CZ}, {(uint64_t)NCH}, {64, (uint32_t)(NW / 2)}, CU_TENSOR_MAP_SWIZZLE_128B, "Wo");
  CUtensorMap zm = make_map<4>(out.data_ptr(), {64, (uint64_t)nj, (uint64_t)(CZ / 64), (uint64_t)ni}, {(uint64_t)CZ, 64, (uint64_t)nj * CZ},
                               {64, (uint32_t)C::BJ, 1, (uint32_t)C::BI}, CU_TENSOR_MAP_SWIZZLE_128B, "z");
  CUtensorMap rm = res ? make_map<4>(res->data_ptr(), {64, (uint64_t)nj, (uint64_t)(CZ / 64), (uint64_t)ni}, {(uint64_t)CZ, 64, (uint64_t)nj * CZ},
                                     {64, (uint32_t)C::BJ, 1, (uint32_t)C::BI}, CU_TENSOR_MAP_SWIZZLE_128B, "res") : zm;
  const int nzb = (int)(CZ / NW);
  const int nitems = (int)((ni / C::BI) * (nj / C::BJ) / 2) * nzb;   // (tile pair, c_z slice)
  static bool attr = false;
  static int max_clusters = 0;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(opm_wepilogue_sm100<NW>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(2 * (num_sms(O.device().index()) / 2));
    cfg.blockDim = dim3(C::THREADS);
    cfg.dynamicSmemBytes = C::SMEM;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&max_clusters, (void*)opm_wepilogue_sm100<NW>, &cfg));
    TORCH_CHECK(max_clusters > 0, "opm_wepilogue_sm100: no CTA pair fits an SM pair");
    attr = true;
  }
  const int clusters = std::min(nitems, max_clusters);
  opm_wepilogue_sm100<NW><<<2 * clusters, C::THREADS, C::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      (int)ni, (int)nj, nitems, nzb, am, bm, zm, rm, norm.data_ptr<float>(),
      bias.data_ptr(), bias.scalar_type() == torch::kBFloat16 ? 1 : 0, res ? 1 : 0, (int)after, asym);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// O [ni*32, nj*32] bf16, norm fp32 [ni, nj] (the mask counts, clamped >= 1), wo bf16 [CZ, 1024] (CZ 128 / 256 / 384), bias fp32 | bf16 [CZ] -> z [1, ni, nj, CZ] (+ residual)
torch::Tensor opm_wepilogue(torch::Tensor O, torch::Tensor norm, torch::Tensor wo, torch::Tensor bias, int64_t ni, int64_t nj,
                            c10::optional<torch::Tensor> residual, int64_t after, c10::optional<torch::Tensor> asym) {
  TORCH_CHECK(O.is_cuda() && O.scalar_type() == torch::kBFloat16 && O.is_contiguous(), "O: contiguous cuda bf16");
  TORCH_CHECK(O.size(0) == ni * CH && O.size(1) == nj * CH, "O: [ni*32, nj*32]");
  TORCH_CHECK(norm.scalar_type() == torch::kFloat32 && norm.is_contiguous() && norm.numel() == ni * nj, "norm: fp32 [ni, nj] mask counts");
  TORCH_CHECK(wo.scalar_type() == torch::kBFloat16 && wo.is_contiguous() && wo.dim() == 2 && wo.size(1) == NCH, "wo: bf16 [c_z, 1024]");
  const long CZ = wo.size(0);
  TORCH_CHECK(CZ == 128 || CZ == 256 || CZ == 384 || CZ == 512, "c_z 128, 256, 384 or 512");
  TORCH_CHECK((bias.scalar_type() == torch::kFloat32 || bias.scalar_type() == torch::kBFloat16) && bias.is_contiguous() && bias.numel() == CZ,
              "bias: fp32 or bf16 [c_z]");
  const bool has_res = residual.has_value() && residual->numel() > 0;
  if (has_res)
    TORCH_CHECK(residual->scalar_type() == torch::kBFloat16 && residual->is_contiguous() && residual->numel() == ni * nj * CZ, "residual: bf16 [1, ni, nj, c_z]");
  auto out = torch::empty({1, ni, nj, CZ}, O.options());
  const torch::Tensor* res = has_res ? &residual.value() : nullptr;
  const int* ap = asym_ptr(asym, std::max(ni, nj));
  if (CZ == 128) epilogue_launch<128>(O, norm, wo, bias, ni, nj, res, out, after, ap);
  else if (CZ == 256 || CZ == 512) epilogue_launch<256>(O, norm, wo, bias, ni, nj, res, out, after, ap);   // c_z 512: two N = 256 slices
  else epilogue_launch<192>(O, norm, wo, bias, ni, nj, res, out, after, ap);
  return out;
}

template <int CZ>
static std::vector<torch::Tensor> dgrad_launch(torch::Tensor& dz, torch::Tensor& norm, torch::Tensor& wo, int64_t ni, int64_t nj, int64_t grad_bf16,
                                               const int* asym) {
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
      norm.data_ptr<float>(), dbo_part.data_ptr<float>(), asym);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dO, dzp, colsum(dbo_part, grad_bf16 ? torch::kBFloat16 : torch::kFloat32)};
}

// dz [ni, nj, CZ] bf16 (CZ 128 / 256 / 384 / 512), norm fp32 [ni, nj], wo bf16 [CZ, 1024] -> [dO [ni*32, nj*32], dz/n [ni, nj, CZ], dbias [CZ]]
std::vector<torch::Tensor> opm_wdgrad(torch::Tensor dz, torch::Tensor norm, torch::Tensor wo, int64_t ni, int64_t nj, int64_t grad_bf16,
                                      c10::optional<torch::Tensor> asym) {
  TORCH_CHECK(wo.scalar_type() == torch::kBFloat16 && wo.is_contiguous() && wo.dim() == 2 && wo.size(1) == NCH, "wo: bf16 [c_z, 1024]");
  const long CZ = wo.size(0);
  TORCH_CHECK(dz.is_cuda() && dz.scalar_type() == torch::kBFloat16 && dz.is_contiguous() && dz.numel() == ni * nj * CZ, "dz: contiguous bf16 [ni, nj, c_z]");
  TORCH_CHECK(norm.scalar_type() == torch::kFloat32 && norm.is_contiguous() && norm.numel() == ni * nj, "norm: fp32 [ni, nj] mask counts");
  const int* ap = asym_ptr(asym, std::max(ni, nj));
  if (CZ == 128) return dgrad_launch<128>(dz, norm, wo, ni, nj, grad_bf16, ap);
  if (CZ == 256) return dgrad_launch<256>(dz, norm, wo, ni, nj, grad_bf16, ap);
  if (CZ == 512) return dgrad_launch<512>(dz, norm, wo, ni, nj, grad_bf16, ap);
  TORCH_CHECK(CZ == 384, "c_z 128, 256, 384 or 512");
  return dgrad_launch<384>(dz, norm, wo, ni, nj, grad_bf16, ap);
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
  if (CZ == 512) return dwo_launch<512>(dzp, O, ni, nj, grad_bf16);
  TORCH_CHECK(CZ == 384, "c_z 256, 384 or 512");
  return dwo_launch<384>(dzp, O, ni, nj, grad_bf16);
}

template <typename WT_, int NCTA>
static void wpbwd_launch(int N, int S, int ntiles, int grid, const CUtensorMap& dam, const CUtensorMap& dbm, const CUtensorMap& xm,
                         const CUtensorMap& sm, const CUtensorMap& mm, const __nv_bfloat16* wap, const __nv_bfloat16* wbp, const WT_* gp,
                         const CUtensorMap& dmm, float* part, cudaStream_t st) {
  using namespace wpb;
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(opm_wprologue_bwd_sm100<WT_, NCTA>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = SMEM;
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = NCTA; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, opm_wprologue_bwd_sm100<WT_, NCTA>, N, S, ntiles, dam, dbm, xm, sm, mm, wap, wbp, gp, dmm, part));
}

// the clusters of NCTA CTAs that can be resident at once (a pair takes two SMs of one GPC: fewer than half the SMs)
template <typename WT_, int NCTA>
static int wpbwd_clusters(int dev) {
  static int n = 0;
  if (n == 0) {
    if (NCTA == 1) { n = num_sms(dev); return n; }
    C10_CUDA_CHECK(cudaFuncSetAttribute(opm_wprologue_bwd_sm100<WT_, NCTA>, cudaFuncAttributeMaxDynamicSharedMemorySize, wpb::SMEM));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(NCTA * (num_sms(dev) / NCTA));
    cfg.blockDim = dim3(wpb::THREADS);
    cfg.dynamicSmemBytes = wpb::SMEM;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = NCTA; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&n, (void*)opm_wprologue_bwd_sm100<WT_, NCTA>, &cfg));
    TORCH_CHECK(n > 0, "opm_wprologue_bwd_sm100: no CTA pair fits an SM pair");
  }
  return n;
}

// d_msa 128 / 256: dA, dB [S, N*32], m [S, N, CM], stats [N, S, 2], mask bool [S, N], gamma / beta, wa / wb bf16 [32, CM],
// ssa fp32 [64] (the masked column sums of [dA dB]) -> [dm [1, S, N, CM], dWa, dWb fp32 [32, CM], dgamma, dbeta (gamma's dtype)]
std::vector<torch::Tensor> opm_wprologue_bwd(torch::Tensor dA, torch::Tensor dB, torch::Tensor m, torch::Tensor stats, torch::Tensor mask,
                                             torch::Tensor gamma, torch::Tensor beta, torch::Tensor wa, torch::Tensor wb, torch::Tensor ssa) {
  using namespace wpb;
  TORCH_CHECK(m.is_contiguous() && m.scalar_type() == torch::kBFloat16 && m.dim() == 3 && (m.size(2) == 128 || m.size(2) == 256),
              "m: [S, N, 128 | 256] bf16");
  const long S = m.size(0), N = m.size(1), CMT = m.size(2);
  const int ncta = (int)(CMT / CM);
  TORCH_CHECK(S % BS == 0, "S must be a multiple of ", BS);
  TORCH_CHECK(dA.is_contiguous() && dB.is_contiguous() && dA.numel() == S * N * CH && dB.numel() == S * N * CH, "dA, dB: [S, N*32]");
  TORCH_CHECK(stats.is_contiguous() && stats.scalar_type() == torch::kFloat32 && stats.numel() == S * N * 2, "stats: fp32 [N, S, 2]");
  TORCH_CHECK(mask.is_contiguous() && mask.scalar_type() == torch::kBool && mask.numel() == S * N, "mask: bool [S, N]");
  TORCH_CHECK(wa.scalar_type() == torch::kBFloat16 && wb.scalar_type() == torch::kBFloat16 && wa.is_contiguous() && wb.is_contiguous() &&
              wa.numel() == CH * CMT && wb.numel() == CH * CMT, "wa, wb: bf16 [32, d_msa]");
  TORCH_CHECK(gamma.scalar_type() == beta.scalar_type() && gamma.is_contiguous() && beta.is_contiguous() && gamma.numel() == CMT, "gamma, beta");
  TORCH_CHECK(ssa.scalar_type() == torch::kFloat32 && ssa.is_contiguous() && ssa.numel() == 2 * CH, "ssa: fp32 [64]");
  const bool gf = gamma.scalar_type() == torch::kFloat32;
  TORCH_CHECK(gf || gamma.scalar_type() == torch::kBFloat16, "gamma: fp32 or bf16");
  const int dev = m.device().index();
  auto dm = torch::empty({1, S, N, CMT}, m.options());
  const int ntiles = (int)(N * (S / BS));
  const int maxcl = ncta == 1 ? num_sms(dev) : (gf ? wpbwd_clusters<float, 2>(dev) : wpbwd_clusters<__nv_bfloat16, 2>(dev));
  const int clusters = std::min(ntiles, maxcl);
  auto part = torch::empty({clusters, CMT, 64}, m.options().dtype(torch::kFloat32));
  CUtensorMap dam = make_map<3>(dA.data_ptr(), {(uint64_t)CH, (uint64_t)N, (uint64_t)S}, {(uint64_t)CH, (uint64_t)N * CH}, {CH, 1, BS}, CU_TENSOR_MAP_SWIZZLE_64B, "dA");
  CUtensorMap dbm = make_map<3>(dB.data_ptr(), {(uint64_t)CH, (uint64_t)N, (uint64_t)S}, {(uint64_t)CH, (uint64_t)N * CH}, {CH, 1, BS}, CU_TENSOR_MAP_SWIZZLE_64B, "dB");
  CUtensorMap xm = make_map<4>(m.data_ptr(), {64, (uint64_t)N, (uint64_t)S, (uint64_t)(CMT / 64)}, {(uint64_t)CMT, (uint64_t)N * CMT, 64}, {64, 1, BS, KB},
                               CU_TENSOR_MAP_SWIZZLE_128B, "m");
  CUtensorMap dmm = make_map<4>(dm.data_ptr(), {64, (uint64_t)N, (uint64_t)S, (uint64_t)(CMT / 64)}, {(uint64_t)CMT, (uint64_t)N * CMT, 64}, {64, 1, BS, KB},
                                CU_TENSOR_MAP_SWIZZLE_128B, "dm");
  CUtensorMap sm = make_map<2>(stats.data_ptr(), {(uint64_t)(2 * S), (uint64_t)N}, {(uint64_t)(2 * S)}, {2 * BS, 1}, CU_TENSOR_MAP_SWIZZLE_NONE, "stats",
                               CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4);
  CUtensorMap mm = make_map<2>(mask.data_ptr(), {(uint64_t)N, (uint64_t)S}, {(uint64_t)N}, {16, BS}, CU_TENSOR_MAP_SWIZZLE_NONE, "mask",
                               CU_TENSOR_MAP_DATA_TYPE_UINT8, 1);
  auto st = at::cuda::getCurrentCUDAStream();
  const auto* wap = reinterpret_cast<const __nv_bfloat16*>(wa.data_ptr<at::BFloat16>());
  const auto* wbp = reinterpret_cast<const __nv_bfloat16*>(wb.data_ptr<at::BFloat16>());
  const int grid = clusters * ncta;
  float* pp = part.data_ptr<float>();
  if (gf) {
    if (ncta == 1) wpbwd_launch<float, 1>((int)N, (int)S, ntiles, grid, dam, dbm, xm, sm, mm, wap, wbp, gamma.data_ptr<float>(), dmm, pp, st);
    else wpbwd_launch<float, 2>((int)N, (int)S, ntiles, grid, dam, dbm, xm, sm, mm, wap, wbp, gamma.data_ptr<float>(), dmm, pp, st);
  } else {
    const auto* gp = reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr<at::BFloat16>());
    if (ncta == 1) wpbwd_launch<__nv_bfloat16, 1>((int)N, (int)S, ntiles, grid, dam, dbm, xm, sm, mm, wap, wbp, gp, dmm, pp, st);
    else wpbwd_launch<__nv_bfloat16, 2>((int)N, (int)S, ntiles, grid, dam, dbm, xm, sm, mm, wap, wbp, gp, dmm, pp, st);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto red = colsum(part);                                           // [CMT * 64], the clusters' partials in split order
  auto dwa = torch::empty({(long)CH, CMT}, wa.options().dtype(torch::kFloat32)), dwb = torch::empty({(long)CH, CMT}, wb.options().dtype(torch::kFloat32));
  auto dgam = torch::empty({CMT}, gamma.options()), dbet = torch::empty({CMT}, beta.options());
  if (gf) {
    if (CMT == 128) opm_wpbwd_finalize<128, 64, float><<<1, 256, 0, st>>>(red.data_ptr<float>(), ssa.data_ptr<float>(), wap, wbp, gamma.data_ptr<float>(),
        beta.data_ptr<float>(), dwa.data_ptr<float>(), dwb.data_ptr<float>(), dgam.data_ptr<float>(), dbet.data_ptr<float>());
    else opm_wpbwd_finalize<256, 64, float><<<1, 256, 0, st>>>(red.data_ptr<float>(), ssa.data_ptr<float>(), wap, wbp, gamma.data_ptr<float>(),
        beta.data_ptr<float>(), dwa.data_ptr<float>(), dwb.data_ptr<float>(), dgam.data_ptr<float>(), dbet.data_ptr<float>());
  } else {
    const auto* gp = reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr<at::BFloat16>());
    const auto* bp = reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr<at::BFloat16>());
    auto* dgp = reinterpret_cast<__nv_bfloat16*>(dgam.data_ptr<at::BFloat16>());
    auto* dbp = reinterpret_cast<__nv_bfloat16*>(dbet.data_ptr<at::BFloat16>());
    if (CMT == 128) opm_wpbwd_finalize<128, 64, __nv_bfloat16><<<1, 256, 0, st>>>(red.data_ptr<float>(), ssa.data_ptr<float>(), wap, wbp, gp, bp,
        dwa.data_ptr<float>(), dwb.data_ptr<float>(), dgp, dbp);
    else opm_wpbwd_finalize<256, 64, __nv_bfloat16><<<1, 256, 0, st>>>(red.data_ptr<float>(), ssa.data_ptr<float>(), wap, wbp, gp, bp,
        dwa.data_ptr<float>(), dwb.data_ptr<float>(), dgp, dbp);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dm, dwa, dwb, dgam, dbet};
}

// ------------------------------------------------------------------------------------------------- Chai-1 host
// m [S, N, 64] bf16, mask bool [S, N], LayerNorm affine fp32 | bf16, wl / wr bf16 [64, 64] -> [a, b [8, S N, 8] bf16 (group-major),
// stats [N, S, 2] fp32 (save_stats) | empty]
std::vector<torch::Tensor> opm_chai_prologue(torch::Tensor m, torch::Tensor mask, torch::Tensor lnw, torch::Tensor lnb, double eps,
                                             torch::Tensor wl, torch::Tensor wr, bool save_stats) {
  using namespace cpro;
  TORCH_CHECK(m.is_cuda() && m.scalar_type() == torch::kBFloat16 && m.is_contiguous() && m.dim() == 3 && m.size(2) == 64, "m: [S, N, 64] bf16");
  const long S = m.size(0), N = m.size(1);
  TORCH_CHECK(S % 16 == 0 && N % 16 == 0, "S % 16, N % 16");
  TORCH_CHECK(mask.scalar_type() == torch::kBool && mask.is_contiguous() && mask.numel() == S * N, "mask: bool [S, N]");
  TORCH_CHECK(wl.scalar_type() == torch::kBFloat16 && wl.is_contiguous() && wl.numel() == 64 * 64, "wl: bf16 [64, 64]");
  TORCH_CHECK(wr.scalar_type() == torch::kBFloat16 && wr.is_contiguous() && wr.numel() == 64 * 64, "wr: bf16 [64, 64]");
  TORCH_CHECK(lnw.scalar_type() == lnb.scalar_type() && lnw.is_contiguous() && lnb.is_contiguous() && lnw.numel() == 64, "LayerNorm affine");
  auto a = torch::empty({8, S * N, 8}, m.options());
  auto b = torch::empty({8, S * N, 8}, m.options());
  auto stats = save_stats ? torch::empty({N, S, 2}, m.options().dtype(torch::kFloat32)) : torch::empty({0}, m.options().dtype(torch::kFloat32));
  CUtensorMap xm = make_map<3>(m.data_ptr(), {64, (uint64_t)N, (uint64_t)S}, {64, (uint64_t)N * 64}, {64, 8, 16}, CU_TENSOR_MAP_SWIZZLE_128B, "m");
  CUtensorMap am = make_map<3>(a.data_ptr(), {(uint64_t)N * 8, (uint64_t)S, 8}, {(uint64_t)N * 8, (uint64_t)(S * N * 8)}, {64, 16, 8},
                               CU_TENSOR_MAP_SWIZZLE_NONE, "a");
  CUtensorMap bm = make_map<3>(b.data_ptr(), {(uint64_t)N * 8, (uint64_t)S, 8}, {(uint64_t)N * 8, (uint64_t)(S * N * 8)}, {64, 16, 8},
                               CU_TENSOR_MAP_SWIZZLE_NONE, "b");
  CUtensorMap mm = make_map<2>(mask.data_ptr(), {(uint64_t)N, (uint64_t)S}, {(uint64_t)N}, {16, 16}, CU_TENSOR_MAP_SWIZZLE_NONE, "mask",
                               CU_TENSOR_MAP_DATA_TYPE_UINT8, 1);
  CUtensorMap wam = make_map<2>(wl.data_ptr(), {64, 64}, {64}, {64, 64}, CU_TENSOR_MAP_SWIZZLE_128B, "wl");
  CUtensorMap wbm = make_map<2>(wr.data_ptr(), {64, 64}, {64}, {64, 64}, CU_TENSOR_MAP_SWIZZLE_128B, "wr");
  const int ntiles = (int)((N / 8) * (S / 16));
  const int grid = std::min(ntiles, num_sms(m.device().index()));
  auto st = at::cuda::getCurrentCUDAStream();
  float2* sp = save_stats ? reinterpret_cast<float2*>(stats.data_ptr<float>()) : nullptr;
  if (lnw.scalar_type() == torch::kFloat32) {
    static bool attr = false;
    if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(opm_chai_prologue_sm100<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
    opm_chai_prologue_sm100<float><<<grid, THREADS, SMEM, st>>>((int)N, (int)S, ntiles, (float)eps, xm, am, bm, mm, wam, wbm,
        lnw.data_ptr<float>(), lnb.data_ptr<float>(), sp);
  } else {
    TORCH_CHECK(lnw.scalar_type() == torch::kBFloat16, "LayerNorm affine: fp32 or bf16");
    static bool attr = false;
    if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(opm_chai_prologue_sm100<__nv_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
    opm_chai_prologue_sm100<__nv_bfloat16><<<grid, THREADS, SMEM, st>>>((int)N, (int)S, ntiles, (float)eps, xm, am, bm, mm, wam, wbm,
        reinterpret_cast<const __nv_bfloat16*>(lnw.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(lnb.data_ptr<at::BFloat16>()), sp);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {a, b, stats};
}

template <int CZ, bool SAVE>
static void chai_epilogue_launch(torch::Tensor& O, torch::Tensor& wf, torch::Tensor& bias, const torch::Tensor* res, torch::Tensor& z,
                                 torch::Tensor& y, torch::Tensor& stats, double eps, const int* asym, torch::Tensor& ws) {
  using C = CEp<CZ>;
  const long N = O.size(1) / 8;
  const uint64_t NK = (uint64_t)N * 8;
  // O[g][i 8 + k][j 8 + l]: ((j % 8, l) 64, k, j / 8, g N + i) -- 8 consecutive j are 128 contiguous bytes, g and i fold into one
  // dimension (row block g N + i, stride 8 N 8); the box lands as [16 pair octets][8 k][8 pairs][8 l]: no-swizzle core matrices
  CUtensorMap om = make_map<4>(O.data_ptr(), {64, 8, (uint64_t)(N / 8), (uint64_t)(8 * N)}, {NK, 64, 8 * NK}, {64, 8, 4, 4},
                               CU_TENSOR_MAP_SWIZZLE_NONE, "O");
  CUtensorMap wm = make_map<2>(wf.data_ptr(), {512, (uint64_t)CZ}, {512}, {64, (uint32_t)CZ}, CU_TENSOR_MAP_SWIZZLE_128B, "wf");
  CUtensorMap zm = make_map<4>(z.data_ptr(), {64, (uint64_t)N, (uint64_t)(CZ / 64), (uint64_t)N}, {(uint64_t)CZ, 64, (uint64_t)N * CZ},
                               {64, (uint32_t)C::BJ, 1, (uint32_t)C::BI}, CU_TENSOR_MAP_SWIZZLE_128B, "z");
  CUtensorMap rm = res ? make_map<4>(res->data_ptr(), {64, (uint64_t)N, (uint64_t)(CZ / 64), (uint64_t)N}, {(uint64_t)CZ, 64, (uint64_t)N * CZ},
                                     {64, (uint32_t)C::BJ, 1, (uint32_t)C::BI}, CU_TENSOR_MAP_SWIZZLE_128B, "res") : zm;
  // y [N N][512]: element (pair (i, j), g, k, l) at (i N + j) 512 + 64 g + 8 k + l, as (l, j % 8, (g, k), j / 8, i) -- the box of one
  // group lands / leaves as the GEMM operand's [i][j / 8][k][j % 8][l]
  CUtensorMap ym = SAVE ? make_map<5>(y.data_ptr(), {8, 8, 64, (uint64_t)(N / 8), (uint64_t)N}, {512, 8, 8 * 512, (uint64_t)N * 512}, {8, 8, 8, 4, 4},
                                      CU_TENSOR_MAP_SWIZZLE_NONE, "y") : zm;
  const int ntiles = (int)((N / C::BI) * (N / C::BJ));
  const int grid = std::min(ntiles, num_sms(O.device().index()));
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(opm_chai_epilogue_sm100<CZ, SAVE>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM)); attr = true; }
  opm_chai_epilogue_sm100<CZ, SAVE><<<grid, C::THREADS, C::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      (int)N, ntiles, (float)eps, om, wm, zm, rm, ym, bias.data_ptr<float>(), res ? 1 : 0, SAVE ? stats.data_ptr<float>() : nullptr, asym,
      ws.numel() ? reinterpret_cast<__nv_bfloat16*>(ws.data_ptr<at::BFloat16>()) : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// O [8, N 8, N 8] bf16 (the grouped products), wf [CZ, 512] bf16 (W_out . gamma), bias fp32 [CZ] (b_out + W_out beta), residual bf16
// [1, N, N, CZ] | None, asym int32 [N] | None (the interchain mask) -> [z [1, N, N, CZ], y [N N, 512] bf16 and stats [N N, 2] fp32 with
// `save`, the per-pair dbias weights [N N] bf16 with `save` and asym]
std::vector<torch::Tensor> opm_chai_epilogue(torch::Tensor O, torch::Tensor wf, torch::Tensor bias, c10::optional<torch::Tensor> residual, bool save,
                                             double eps, c10::optional<torch::Tensor> asym) {
  TORCH_CHECK(O.is_cuda() && O.is_contiguous() && O.scalar_type() == torch::kBFloat16 && O.dim() == 3 && O.size(0) == 8 && O.size(1) == O.size(2) &&
              O.size(1) % 256 == 0, "O: contiguous bf16 [8, N 8, N 8] with N a multiple of 32");
  const long N = O.size(1) / 8, CZ = wf.size(0);
  TORCH_CHECK(wf.is_contiguous() && wf.scalar_type() == torch::kBFloat16 && wf.dim() == 2 && wf.size(1) == 512 && (CZ == 128 || CZ == 256),
              "wf: bf16 [c_z, 512], c_z 128 or 256");
  TORCH_CHECK(bias.is_contiguous() && bias.scalar_type() == torch::kFloat32 && bias.numel() == CZ, "bias: fp32 [c_z]");
  const bool has_res = residual.has_value() && residual->numel() > 0;
  if (has_res)
    TORCH_CHECK(residual->is_contiguous() && residual->scalar_type() == torch::kBFloat16 && residual->numel() == N * N * CZ, "residual: bf16 [1, N, N, c_z]");
  auto z = torch::empty({1, N, N, CZ}, O.options());
  auto y = save ? torch::empty({N * N, 512}, O.options()) : torch::empty({0}, O.options());
  auto stats = save ? torch::empty({N * N, 2}, O.options().dtype(torch::kFloat32)) : torch::empty({0}, O.options().dtype(torch::kFloat32));
  const torch::Tensor* res = has_res ? &residual.value() : nullptr;
  const int* ap = asym_ptr(asym, N);
  auto ws = (save && ap) ? torch::empty({N * N}, O.options()) : torch::empty({0}, O.options());
  if (CZ == 256) {
    if (save) chai_epilogue_launch<256, true>(O, wf, bias, res, z, y, stats, eps, ap, ws);
    else chai_epilogue_launch<256, false>(O, wf, bias, res, z, y, stats, eps, ap, ws);
  } else {
    if (save) chai_epilogue_launch<128, true>(O, wf, bias, res, z, y, stats, eps, ap, ws);
    else chai_epilogue_launch<128, false>(O, wf, bias, res, z, y, stats, eps, ap, ws);
  }
  return {z, y, stats, ws};
}

// dy [N N, 512] bf16 (dz W'), y [N N, 512] bf16, stats [N N, 2] fp32 -> dO [8, N 8, N 8] bf16
torch::Tensor opm_chai_lnbwd(torch::Tensor dy, torch::Tensor y, torch::Tensor stats, int64_t N) {
  TORCH_CHECK(dy.is_contiguous() && y.is_contiguous() && dy.scalar_type() == torch::kBFloat16 && y.scalar_type() == torch::kBFloat16 &&
              dy.numel() == N * N * 512 && y.numel() == N * N * 512, "dy, y: bf16 [N N, 512]");
  TORCH_CHECK(stats.is_contiguous() && stats.scalar_type() == torch::kFloat32 && stats.numel() == N * N * 2, "stats: fp32 [N N, 2]");
  TORCH_CHECK(N % 32 == 0, "N % 32");
  auto dO = torch::empty({8, N * 8, N * 8}, dy.options());
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(opm_chai_lnbwd_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, 3 * 32 * 1024)); attr = true; }
  opm_chai_lnbwd_sm100<<<dim3((unsigned)(N / 32), (unsigned)N), 256, 3 * 32 * 1024, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(dy.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
      stats.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(dO.data_ptr<at::BFloat16>()), (int)N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dO;
}

// dA^T, dB^T [8, N 8, S] bf16 (the gradients of the group-major projections, transposed), m [S, N, 64], stats [N, S, 2], mask bool [S, N], gamma / beta,
// wl / wr bf16 [64, 64] -> [dm [1, S, N, 64], dW_left, dW_right fp32 [64, 64], dgamma, dbeta (gamma's dtype)]
std::vector<torch::Tensor> opm_chai_prologue_bwd(torch::Tensor dA, torch::Tensor dB, torch::Tensor m, torch::Tensor stats, torch::Tensor mask,
                                                 torch::Tensor gamma, torch::Tensor beta, torch::Tensor wl, torch::Tensor wr) {
  using namespace cpb;
  TORCH_CHECK(m.is_contiguous() && m.scalar_type() == torch::kBFloat16 && m.dim() == 3 && m.size(2) == CM, "m: [S, N, 64] bf16");
  const long S = m.size(0), N = m.size(1);
  TORCH_CHECK(S % BS == 0 && N % 16 == 0, "S % 128, N % 16");
  TORCH_CHECK(dA.is_contiguous() && dB.is_contiguous() && dA.numel() == S * N * 64 && dB.numel() == S * N * 64, "dA^T, dB^T: [8, N 8, S]");
  TORCH_CHECK(stats.is_contiguous() && stats.scalar_type() == torch::kFloat32 && stats.numel() == S * N * 2, "stats: fp32 [N, S, 2]");
  TORCH_CHECK(mask.is_contiguous() && mask.scalar_type() == torch::kBool && mask.numel() == S * N, "mask: bool [S, N]");
  TORCH_CHECK(wl.scalar_type() == torch::kBFloat16 && wr.scalar_type() == torch::kBFloat16 && wl.is_contiguous() && wr.is_contiguous() &&
              wl.numel() == 64 * 64 && wr.numel() == 64 * 64, "wl, wr: bf16 [64, 64]");
  TORCH_CHECK(gamma.scalar_type() == beta.scalar_type() && gamma.is_contiguous() && beta.is_contiguous(), "gamma, beta");
  const bool gf = gamma.scalar_type() == torch::kFloat32;
  auto dm = torch::empty({1, S, N, (long)CM}, m.options());
  const int ntiles = (int)(N * (S / BS));
  const int grid = std::min(ntiles, num_sms(m.device().index()));
  auto part = torch::empty({grid, (long)CM + 1, (long)CE}, m.options().dtype(torch::kFloat32));
  // dA^T[g][i 8 + k][s]: (s, k, g, i)
  const uint64_t SS = (uint64_t)S, GS = (uint64_t)(N * 8) * SS;
  CUtensorMap dam = make_map<4>(dA.data_ptr(), {SS, 8, 8, (uint64_t)N}, {SS, GS, 8 * SS}, {64, 8, 8, 1}, CU_TENSOR_MAP_SWIZZLE_128B, "dA");
  CUtensorMap dbm = make_map<4>(dB.data_ptr(), {SS, 8, 8, (uint64_t)N}, {SS, GS, 8 * SS}, {64, 8, 8, 1}, CU_TENSOR_MAP_SWIZZLE_128B, "dB");
  CUtensorMap xm = make_map<3>(m.data_ptr(), {(uint64_t)CM, (uint64_t)N, (uint64_t)S}, {(uint64_t)CM, (uint64_t)N * CM}, {64, 1, BS}, CU_TENSOR_MAP_SWIZZLE_128B, "m");
  CUtensorMap dmm = make_map<3>(dm.data_ptr(), {(uint64_t)CM, (uint64_t)N, (uint64_t)S}, {(uint64_t)CM, (uint64_t)N * CM}, {64, 1, BS}, CU_TENSOR_MAP_SWIZZLE_128B, "dm");
  CUtensorMap sm = make_map<2>(stats.data_ptr(), {(uint64_t)(2 * S), (uint64_t)N}, {(uint64_t)(2 * S)}, {2 * BS, 1}, CU_TENSOR_MAP_SWIZZLE_NONE, "stats",
                               CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4);
  CUtensorMap mm = make_map<2>(mask.data_ptr(), {(uint64_t)N, (uint64_t)S}, {(uint64_t)N}, {16, BS}, CU_TENSOR_MAP_SWIZZLE_NONE, "mask",
                               CU_TENSOR_MAP_DATA_TYPE_UINT8, 1);
  auto st = at::cuda::getCurrentCUDAStream();
  const auto* wlp = reinterpret_cast<const __nv_bfloat16*>(wl.data_ptr<at::BFloat16>());
  const auto* wrp = reinterpret_cast<const __nv_bfloat16*>(wr.data_ptr<at::BFloat16>());
  if (gf) {
    static bool attr = false;
    if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(opm_chai_prologue_bwd_sm100<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
    opm_chai_prologue_bwd_sm100<float><<<grid, THREADS, SMEM, st>>>((int)N, (int)S, ntiles, dam, dbm, xm, sm, mm, wlp, wrp, gamma.data_ptr<float>(), dmm,
        part.data_ptr<float>());
  } else {
    TORCH_CHECK(gamma.scalar_type() == torch::kBFloat16, "gamma: fp32 or bf16");
    static bool attr = false;
    if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(opm_chai_prologue_bwd_sm100<__nv_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
    opm_chai_prologue_bwd_sm100<__nv_bfloat16><<<grid, THREADS, SMEM, st>>>((int)N, (int)S, ntiles, dam, dbm, xm, sm, mm, wlp, wrp,
        reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr<at::BFloat16>()), dmm, part.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto red = colsum(part);                                           // [65, 128]: R^T rows, then ssa
  const float* rp = red.data_ptr<float>();
  auto dwl = torch::empty({64, 64}, wl.options().dtype(torch::kFloat32)), dwr = torch::empty({64, 64}, wr.options().dtype(torch::kFloat32));
  auto dgam = torch::empty({(long)CM}, gamma.options()), dbet = torch::empty({(long)CM}, beta.options());
  if (gf) opm_wpbwd_finalize<CM, CE, float><<<1, 256, 0, st>>>(rp, rp + CM * CE, wlp, wrp, gamma.data_ptr<float>(), beta.data_ptr<float>(),
      dwl.data_ptr<float>(), dwr.data_ptr<float>(), dgam.data_ptr<float>(), dbet.data_ptr<float>());
  else opm_wpbwd_finalize<CM, CE, __nv_bfloat16><<<1, 256, 0, st>>>(rp, rp + CM * CE, wlp, wrp,
      reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr<at::BFloat16>()),
      dwl.data_ptr<float>(), dwr.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(dgam.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(dbet.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dm, dwl, dwr, dgam, dbet};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("opm_wprologue", &opm_wprologue, "sm100 OPM prologue at d_msa 64 / 128 with the optional projection biases",
        py::arg("m"), py::arg("mask"), py::arg("lnw"), py::arg("lnb"), py::arg("eps"), py::arg("wa"), py::arg("wb"), py::arg("pbias"),
        py::arg("save_stats"));
  m.def("opm_wepilogue", &opm_wepilogue, "sm100 OPM epilogue at c_z 128 / 256 / 384 / 512 on CTA pairs", py::arg("O"), py::arg("norm"), py::arg("wo"),
        py::arg("bias"), py::arg("ni"), py::arg("nj"), py::arg("residual") = py::none(), py::arg("after") = 0, py::arg("asym") = py::none());
  m.def("opm_wdgrad", &opm_wdgrad, "sm100 OPM dO at c_z 128 / 256 / 384 / 512", py::arg("dz"), py::arg("norm"), py::arg("wo"), py::arg("ni"),
        py::arg("nj"), py::arg("grad_bf16") = 0, py::arg("asym") = py::none());
  m.def("opm_wdwo", &opm_wdwo, "sm100 OPM dWo at c_z 256 / 384", py::arg("dzp"), py::arg("O"), py::arg("ni"), py::arg("nj"), py::arg("grad_bf16") = 0);
  m.def("opm_wprologue_bwd", &opm_wprologue_bwd, "sm100 OPM prologue backward at d_msa 128 / 256");
  m.def("opm_chai_prologue", &opm_chai_prologue, "sm100 Chai-1 OPM prologue: LN + both projections + mask, group-major");
  m.def("opm_chai_epilogue", &opm_chai_epilogue, "sm100 Chai-1 OPM epilogue: product LayerNorm + out-projection (+ residual), y / stats saved",
        py::arg("O"), py::arg("wf"), py::arg("bias"), py::arg("residual") = py::none(), py::arg("save") = false, py::arg("eps") = 0.1,
        py::arg("asym") = py::none());
  m.def("opm_chai_lnbwd", &opm_chai_lnbwd, "sm100 Chai-1 product LayerNorm backward into the grouped layout");
  m.def("opm_chai_prologue_bwd", &opm_chai_prologue_bwd, "sm100 Chai-1 OPM prologue backward");
}
