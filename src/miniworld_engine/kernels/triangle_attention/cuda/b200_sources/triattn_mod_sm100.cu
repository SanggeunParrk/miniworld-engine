// TriangleAttention module kernels around the attention core for B200 (sm_100a) -- the H100 kernel boundaries (LN + projections,
// gate + out-projection, gate backward + delta, projection dgrad + LN backward), developed on tcgen05 / TMEM / TMA.
//
// front:  y = LN(x) (never stored); q | k | v | g = y . W^T  (four 128 x 128 projections, bf16, the [rows][H*32] projection layout
//         the attention kernels read); bias[b, h, j, k] = y . Wb[h]  (head-major for the attention, masked keys = -inf); with BT
//         (AlphaFold 3's ending node: the bias of the pair as it is) row (b, j, k) writes bias[b, h, k, j] instead (its key j)
#include <torch/extension.h>
#include <cstring>
#include "sm100.cuh"
#include "tailg_sm100.cuh"
using namespace sm100;
#ifndef FR_TCB
#define FR_TCB 1   // B200 N2: the q / k / v front's bias projection on the tensor core (N = 16 GEMM with bf16 hi / lo weights) instead of the LN warps' FMAs
#endif
#ifndef FR_NOBP
#define FR_NOBP 0  // timing only: no bias projection on the CUDA cores in the front LayerNorm (WRONG bias)
#endif
#ifndef FR_TL
#define FR_TL 0    // timing only: per-tile stage stamps of front CTA 10 into g_fr_tl (tri_front_tl)
#endif
__device__ unsigned long long g_fr_tl[32][16];
#define FTLS(W, EV, IDX) do { if (FR_TL && warp == (W) && lane == 0 && blockIdx.x == 10 && (IDX) < 32) g_fr_tl[(IDX)][(EV)] = (unsigned long long)tailg::gtimer_tg(); } while (0)

namespace {
constexpr int C = 128;                                    // d_pair
constexpr int NH = 4;                                     // heads (bias outputs)

#ifndef FR_DW
#define FR_DW 4          // drain warps: 4 (one per TMEM lane quadrant) or 8 (two per quadrant, each half of the columns)
#endif
#ifndef FR_WB
#define FR_WB 3          // per-drain-warp staging ring: [32 rows][32] bf16 tiles (2 KiB), each stored by its own warp
#endif
namespace fr {
// Tile = 128 rows (tokens).  warp 0: TMA (x tiles, the weights once); warp 1: tcgen05 issue (q|k then v|g, two TMEM halves of 2 HD
// columns so the drain of one overlaps the other's GEMMs); warps 2-9: LayerNorm in place (two threads a row) + the bias projection on
// the CUDA cores (<= 4 outputs a row); warps 10 ..: drain (TMEM lane = row) -> per-warp 64B-swizzled [32][32] staging -> TMA stores.
// (The staging must not be the tile's x buffer: the v|g GEMMs still read y there while q|k drain.)  The weights stay resident:
// [CX / 64 K chunks][4 HD outputs][64]; the x buffer is released by the tile's last GEMM commit.
// CX = d_pair (128: the fused path; 64: the wide path's d_pair 64 rows), HD = heads x 32 (128, or 64 at CX 64).
template <int CX, int HD> struct Cfg {
  static constexpr int BM = 128, NST = 2;
  static constexpr int XT = BM * CX;                      // x / y tile [CX / 64][128][64]
  static constexpr int WT = 4 * HD * CX;                  // q|k|v|g weights
  static constexpr int DW = FR_DW;                        // drain warps
  static constexpr int THREADS = 320 + 32 * DW;
  static constexpr int DWB = FR_WB, WBT = 32 * 32;
  static constexpr int SMEM = 1024 + (NST * XT + WT + DW * DWB * WBT + 16 * CX) * 2 + (2 * CX + NH * CX) * 4 + 512;
  static constexpr uint32_t IDESC = idesc_bf16(128, HD, 0, 0);
  static constexpr int CPT = HD / 32;                     // 32-column chunks a tensor
  static constexpr int NCH = 2 * CPT * 4 / DW;            // chunks a half per drain warp
  static_assert(SMEM <= 232448, "smem");
  static_assert((CX == 64 || CX == 128) && (HD == 64 || HD == 128), "front widths");
};
}  // namespace fr

// NOUT: 4 (q, k, v, g), or 3 (q, k, v: the inference tail tri_tailg_sm100 projects the gate itself from x, as the H100's does)
template <int CX, int HD, int NOUT>
__global__ void __launch_bounds__(fr::Cfg<CX, HD>::THREADS, 1) tri_front_sm100(
    int R, int L, int ntiles, float eps,
    const __grid_constant__ CUtensorMap xmap,             // x [R][CX] bf16, box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap wmap,             // W [4 HD][CX] bf16, box (64, min(4 HD, 256)), 128B swizzle
    const __grid_constant__ CUtensorMap qmap,             // q / k / v / g [R][HD], box (32, 32), 64B swizzle (TMA stores)
    const __grid_constant__ CUtensorMap kmap,
    const __grid_constant__ CUtensorMap vmap,
    const __grid_constant__ CUtensorMap gmap,
    const float* __restrict__ LNW, const float* __restrict__ LNB,  // [CX] fp32
    const float* __restrict__ WB, const __nv_bfloat16* __restrict__ WB16,   // the bias rows [NHB][CX]: fp32, or (WB null) bf16
    const int NHB,                                        // heads (bias outputs), <= 4
    const bool* __restrict__ MASK,                        // [B][KL] key mask or nullptr
    __nv_bfloat16* __restrict__ BIAS,                     // [B][NHB][L][L]
    float2* __restrict__ STATS,                           // (mean, rstd) [R] for the wide backward, or nullptr
    const int TRANS,                                      // geometry of x / y / dy / dx (see load_rows)
    const int BT,                                         // 1: row (b, j, k) -> bias[b, h, k, j] (see the header)
    const int KL,                                         // the caller's length (L, or less for a ragged length)
    const int V1) {                                       // 1: v's channel 16 of every 32-channel head slot = 1.0 (16-channel heads, inference: the TA_D16 core's row sum)
  using T = fr::Cfg<CX, HD>;
  constexpr int BM = T::BM, NST = T::NST, XT = T::XT, WT = T::WT, DWB = T::DWB, WBT = T::WBT, CPT = T::CPT, NCH = T::NCH;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sX = reinterpret_cast<__nv_bfloat16*>(smb);   // [NST][XT]
  __nv_bfloat16* sW = sX + NST * XT;                             // [CX / 64][4 HD][64]
  __nv_bfloat16* sO = sW + WT;                                   // [DW drain warps][DWB][32][32] staging
  __nv_bfloat16* sWb = sO + T::DW * DWB * WBT;                   // NOUT 1: the bias rows as a bf16 B operand [CX / 64][16][64] (128B-swizzled)
  float* sP = reinterpret_cast<float*>(sWb + 16 * CX);           // lnw [CX] | lnb [CX] | Wb [4][CX]
  uint64_t* bars = reinterpret_cast<uint64_t*>(sP + 2 * CX + NH * CX);
  uint64_t* xf = bars;               // [NST] x landed
  uint64_t* yf = xf + NST;           // [NST] count 8: y written
  uint64_t* xe = yf + NST;           // [NST] count 1: the tile's GEMMs retired (the buffer takes the tile after next)
  uint64_t* af = xe + NST;           // [2]   accumulator half ready
  uint64_t* ae = af + 2;             // [2]   count DW: drained
  uint64_t* wf = ae + 2;             // [1]
  uint32_t* tslot = reinterpret_cast<uint32_t*>(wf + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int i = 0; i < NST; ++i) { bar_init(xf + i, 1); bar_init(yf + i, 8); bar_init(xe + i, 1); }
    for (int i = 0; i < 2; ++i) { bar_init(af + i, 1); bar_init(ae + i, T::DW); }
    bar_init(wf, 1);
    bar_init_fence();
  }
  for (int i = tid; i < 2 * CX + NH * CX; i += blockDim.x) {
    const int j = i - 2 * CX;
    sP[i] = i < CX ? LNW[i] : i < 2 * CX ? LNB[i - CX] : j >= NHB * CX ? 0.f : WB != nullptr ? WB[j] : __bfloat162float(WB16[j]);
  }
  // B200 N2 (FR_TCB): the q | k | v front's bias projection (4 heads x 128 channels a row = 8 of the ~22 LayerNorm instructions a channel pair, the
  // LN warps' critical path: 2.8 of the 3.7 us a tile) on the tensor core: one N = 16 GEMM [y] x [Wb hi | Wb lo] (rows 0..3 the bf16 weights, rows 4..7 their
  // bf16 residue: fp32-grade weights), fp32 in the g tensor's TMEM columns (unused at NOUT 3), summed (hi + lo) by the drain.
  constexpr bool BTC = (FR_TCB != 0) && NOUT == 3;
  if (NOUT == 1 || BTC) {   // the bias rows (bf16) as the B operand of one N = 16 GEMM: row n, channel c at the 128B-swizzled [c / 64][n][64] slot
    for (int i = tid; i < 16 * CX; i += blockDim.x) {
      const int n = i / CX, c = i % CX;
      const int nr = BTC ? (n & 3) : n;
      float w = (nr >= NHB || (BTC && n >= 8)) ? 0.f : WB != nullptr ? WB[nr * CX + c] : __bfloat162float(WB16[nr * CX + c]);
      if (BTC) { const float hi = __bfloat162float(__float2bfloat16_rn(w)); w = n < 4 ? hi : w - hi; }
      sWb[(c >> 6) * 16 * 64 + n * 64 + ((((c & 63) >> 3) ^ (n & 7)) << 3) + (c & 7)] = __float2bfloat16_rn(w);
    }
    fence_proxy_async();
  }
  if (warp == 1) tmem_alloc(tslot, 512);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  pdl_trigger();                                   // the core (PDL) may start its set-up on SMs this grid leaves free / frees

  if (warp == 0) {
    if (lane == 0) {
      constexpr int WR = 4 * HD < 256 ? 4 * HD : 256;     // weight rows a load
      expect_tx(wf, WT * 2);
      for (int kh = 0; kh < CX / 64; ++kh)
        for (int nh = 0; nh < 4 * HD / WR; ++nh) load_2d(&wmap, sW + kh * 4 * HD * 64 + nh * WR * 64, wf, kh * 64, nh * WR);
      int lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int st = lt % NST;
        if (lt >= NST) wait(xe + st, ((lt / NST) - 1) & 1);
        FTLS(0, 0, lt);
        expect_tx(xf + st, XT * 2);
        for (int kh = 0; kh < CX / 64; ++kh) load_rows(&xmap, sX + st * XT + kh * BM * 64, xf + st, kh * 64, t * BM, L, TRANS);
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      wait(wf, 0);
      int lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int st = lt % NST;
        wait(yf + st, (lt / NST) & 1);
        FTLS(1, 3, lt);
        const __nv_bfloat16* y = sX + st * XT;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          if (NOUT == 1 && half == 0) continue;     // NOUT 1 (v plane only): no q | k half
          if (lt >= 1) wait(ae + half, (lt - 1) & 1);
          tc_fence_after();
#pragma unroll
          for (int tt = 0; tt < ((NOUT <= 3 && half == 1) ? 1 : 2); ++tt) {   // NOUT 3 / 1: no g GEMM
            const int ten = half * 2 + tt;
#pragma unroll
            for (int ks = 0; ks < CX / 16; ++ks) {
              const int kh = ks >> 2, kk = (ks & 3) * 16;
              mma_ss(tmem + ten * HD, desc_k128(y + kh * BM * 64 + kk), desc_k128(sW + kh * 4 * HD * 64 + ten * HD * 64 + kk), T::IDESC, ks ? 1u : 0u);
            }
          }
          if (NOUT == 1 || (BTC && half == 1)) {   // the bias GEMM: acc [128][16] fp32 at TMEM column 3 HD (the g tensor's slot, unused)
            constexpr uint32_t IDB = idesc_bf16(128, 16, 0, 0);
#pragma unroll
            for (int ks = 0; ks < CX / 16; ++ks) {
              const int kh = ks >> 2, kk = (ks & 3) * 16;
              mma_ss(tmem + 3 * HD, desc_k128(y + kh * BM * 64 + kk), desc_k128(sWb + kh * 16 * 64 + kk), IDB, ks ? 1u : 0u);
            }
          }
          mma_commit(af + half);
        }
        mma_commit(xe + st);
        FTLS(1, 4, lt);
      }
    }
  } else if (warp < 10) {
    // ---- LayerNorm: row r, channel half hf (CX / 2 channels = CX / 16 chunks of 16 B of the row's 128B-swizzled [128][64] K chunks) ----
    constexpr int NC = CX / 16;
    const int r = (warp - 2) * 16 + (lane & 15), hf = lane >> 4;
    auto xoff = [&](int c8) {
      const int c = hf * (CX / 2) + c8 * 8;
      return (c >> 6) * BM * 64 + r * 64 + ((((c & 63) >> 3) ^ (r & 7)) << 3);
    };
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int st = lt % NST;
      wait(xf + st, (lt / NST) & 1);
      FTLS(2, 1, lt);
      __nv_bfloat16* xt = sX + st * XT;
      float2 x[NC * 4];
#pragma unroll
      for (int c8 = 0; c8 < NC; ++c8) {
        const uint4 u = *reinterpret_cast<const uint4*>(xt + xoff(c8));
        x[c8 * 4 + 0] = bf2f(u.x); x[c8 * 4 + 1] = bf2f(u.y); x[c8 * 4 + 2] = bf2f(u.z); x[c8 * 4 + 3] = bf2f(u.w);
      }
      float2 s0 = make_float2(0.f, 0.f), s1 = s0;
#pragma unroll
      for (int i = 0; i < NC * 4; i += 2) { s0 = add2(s0, x[i]); s1 = add2(s1, x[i + 1]); }
      float sum = s0.x + s0.y + s1.x + s1.y;
      sum += __shfl_xor_sync(0xffffffffu, sum, 16);
      const float mean = sum * (1.f / CX);
      const float2 nm = make_float2(-mean, -mean);
      float2 v0 = make_float2(0.f, 0.f), v1 = v0;
#pragma unroll
      for (int i = 0; i < NC * 4; i += 2) {
        x[i] = add2(x[i], nm); x[i + 1] = add2(x[i + 1], nm);
        v0 = fma2(x[i], x[i], v0); v1 = fma2(x[i + 1], x[i + 1], v1);
      }
      float var = v0.x + v0.y + v1.x + v1.y;
      var += __shfl_xor_sync(0xffffffffu, var, 16);
      const float rstd = rsqrtf(var * (1.f / CX) + eps);
      const float2 rs = make_float2(rstd, rstd);
      float bacc[NH] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int c8 = 0; c8 < NC; ++c8) {
        const int c0 = hf * (CX / 2) + c8 * 8;
        uint32_t o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int c = c0 + 2 * e;
          const float2 yv = fma2(mul2(x[c8 * 4 + e], rs), make_float2(sP[c], sP[c + 1]), make_float2(sP[CX + c], sP[CX + c + 1]));
          o[e] = pack2(yv.x, yv.y);
          const float2 yb = bf2f(o[e]);           // the bias projection sees the bf16 y the GEMMs see
          if (NOUT != 1 && !FR_NOBP && !BTC) {
#pragma unroll
            for (int h = 0; h < NH; ++h) bacc[h] = fmaf(yb.x, sP[2 * CX + h * CX + c], fmaf(yb.y, sP[2 * CX + h * CX + c + 1], bacc[h]));
          }
        }
        *reinterpret_cast<uint4*>(xt + xoff(c8)) = make_uint4(o[0], o[1], o[2], o[3]);
      }
#pragma unroll
      for (int h = 0; h < NH; ++h) bacc[h] += __shfl_xor_sync(0xffffffffu, bacc[h], 16);
      if (hf == 0 && NOUT != 1 && (!BTC || STATS != nullptr)) {
        const long row = (long)t * BM + r;       // (b, j, k)
        if (STATS != nullptr) STATS[row] = make_float2(mean, rstd);
        if (!BTC) {
        const int kx = (int)(row % L);
        const long bj = row / L;
        const int j = (int)(bj % L), b = (int)(bj / L);
        const int qi = BT ? kx : j, ki = BT ? j : kx;
        // keys beyond the caller's length KL are a ragged length's pad: masked (KL a parameter: derived from TRANS it spilled)
        const bool keep = ki < KL && (MASK == nullptr || MASK[(long)b * KL + ki]);
#pragma unroll
        for (int h = 0; h < NH; ++h)
          if (h < NHB)
            BIAS[(((long)b * NHB + h) * L + qi) * L + ki] = keep ? __float2bfloat16_rn(bacc[h]) : __float2bfloat16_rn(-3.3895313892515355e38f);
        }
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(yf + st);
      FTLS(2, 2, lt);
    }
  } else {
    // ---- drain: TMEM lane = row; each warp owns its 32 rows end to end: a [32][32] chunk -> its own staging ring -> its own TMA store
    // (no block-wide barrier per chunk); the next chunk's TMEM load is in flight while this one is packed ----
    const int qq = warp & 3, dw = warp - 10, part = dw >> 2;   // part: which chunks of each half (DW = 8: every other one)
    const CUtensorMap* maps[4] = {&qmap, &kmap, &vmap, &gmap};
    __nv_bfloat16* ring = sO + dw * DWB * WBT;
    int nst = 0;                                  // this warp's stores issued
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        if (NOUT == 1 && half == 0) continue;
        wait(af + half, lt & 1);
        FTLS(10, 5 + half, lt);
        tc_fence_after();
        if (NOUT == 1 || (BTC && half == 1)) {   // the bias GEMM's accumulator (column 3 HD): row (b, j, k) -> bias[b, h, qi, ki], masked keys at the finite -inf
          float bv[32];
          tmem_ld32(tmem_at(tmem + 3 * HD, qq * 32, 0), bv);
          tmem_wait_ld();
          const long row = (long)t * BM + qq * 32 + lane;
          const int kx = (int)(row % L);
          const long bj = row / L;
          const int j = (int)(bj % L), b = (int)(bj / L);
          const int qi = BT ? kx : j, ki = BT ? j : kx;
          const bool keep = ki < KL && (MASK == nullptr || MASK[(long)b * KL + ki]);
#pragma unroll
          for (int h = 0; h < NH; ++h)
            if (h < NHB)
              BIAS[(((long)b * NHB + h) * L + qi) * L + ki] = keep ? __float2bfloat16_rn(BTC ? bv[h] + bv[4 + h] : bv[h]) : __float2bfloat16_rn(-3.3895313892515355e38f);
        }
        float v[2][32];
        auto chunk = [&](int k0) { return T::DW == 4 ? k0 : k0 * 2 + part; };
        constexpr int NCH1 = NOUT <= 3 ? NCH / 2 : NCH;   // NOUT 3 / 1: the second half holds v only (chunks k < CPT)
        const int nch = half ? NCH1 : NCH;
        tmem_ld32(tmem_at(tmem + half * 2 * HD + chunk(0) * 32, qq * 32, 0), v[0]);
#pragma unroll
        for (int k0 = 0; k0 < nch; ++k0) {        // chunk k: tensor half * 2 + k / CPT, columns 32 (k % CPT)
          const int k = chunk(k0);
          tmem_wait_ld();
          if (k0 + 1 < nch) tmem_ld32(tmem_at(tmem + half * 2 * HD + chunk(k0 + 1) * 32, qq * 32, 0), v[(k0 + 1) & 1]);
          else { tc_fence_before(); __syncwarp(); if (lane == 0) arrive(ae + half); }
          __nv_bfloat16* buf = ring + (nst % DWB) * WBT;
          if (lane == 0 && nst >= DWB) bulk_wait_read<DWB - 1>();   // the store DWB back has read this buffer
          __syncwarp();
          if (NOUT == 1) {   // v plane [HD][R] (channel-major): lane = token; staging [32 channels][32 tokens] unswizzled, one TMA store
#pragma unroll
            for (int i = 0; i < 32; ++i) buf[i * 32 + lane] = __float2bfloat16_rn(v[k0 & 1][i]);
            fence_proxy_async();
            __syncwarp();
            if (lane == 0) {
              store_2d(&vmap, buf, t * BM + qq * 32, (k % CPT) * 32);
              bulk_commit();
            }
            ++nst;
            continue;
          }
          uint32_t w[16];
#pragma unroll
          for (int i = 0; i < 16; ++i) w[i] = pack2(v[k0 & 1][2 * i], v[k0 & 1][2 * i + 1]);
          if (V1 && half == 1 && k / CPT == 0) w[8] = (w[8] & 0xffff0000u) | 0x3f80u;   // (the padded channel 16 is 0)
          unsigned char* rowp = reinterpret_cast<unsigned char*>(buf) + lane * 64;
#pragma unroll
          for (int c = 0; c < 4; ++c)
            *reinterpret_cast<uint4*>(rowp + ((c ^ ((lane >> 1) & 3)) << 4)) = make_uint4(w[c * 4], w[c * 4 + 1], w[c * 4 + 2], w[c * 4 + 3]);
          fence_proxy_async();
          __syncwarp();
          if (lane == 0) {
            store_2d(maps[half * 2 + k / CPT], buf, (k % CPT) * 32, t * BM + qq * 32);
            bulk_commit();
          }
          ++nst;
        }
      }
    }
    if (lane == 0) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
}
// ---------------------------------------------------------------------------------------------------------------------------------
// tail:  out = x + drop o ((sigmoid(g) o o) . Wo^T)   (the gate, the out-projection, the dropout scale and the residual in one pass;
//         drop [B][L (the row's last index)][128] bf16 = the module's broadcast dropout scale, or none)
// Tile = 128 rows.  warp 0: TMA (g | o in a two-stage ring, the residual x single-buffered, Wo once); warp 1: tcgen05 (N = 128,
// K = 128; double-buffered TMEM); warps 2-9: u = sigmoid(g) o o written over g (the GEMM's A tile), two threads a row; warps 10-13:
// drain + x -> 64B-swizzled [128][32] staging -> TMA stores.
__device__ __forceinline__ void reduce_add_2d_bf(const void* map, const void* src, int c0, int c1) {   // bf16 add into global
  asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%2, %3}], [%1];\n"
               :: "l"(map), "r"(sa(src)), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ float sigmoid_t(float x) {   // 0.5 + 0.5 tanh(x / 2): one MUFU op
  float t;
  asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * x));
  return fmaf(0.5f, t, 0.5f);
}
namespace tl {
constexpr int BM = 128, NST = 2;
constexpr int XT = BM * C;                                // [2 halves][128][64] 128B-swizzled (32 KiB)
constexpr int OT = BM * 32;
constexpr int THREADS = 448;
constexpr int SMEM = 1024 + (NST * 2 * XT + XT + C * C + 2 * OT) * 2 + 512;
constexpr uint32_t IDESC = idesc_bf16(128, 128, 0, 0);
static_assert(SMEM <= 232448, "smem");
}  // namespace tl

__global__ void __launch_bounds__(tl::THREADS, 1) tri_tail_sm100(
    int ntiles, int L, const __nv_bfloat16* __restrict__ DROP,
    const __grid_constant__ CUtensorMap gmap,             // g / o / x [R][128] bf16, box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap omap,
    const __grid_constant__ CUtensorMap xmap,
    const __grid_constant__ CUtensorMap wmap,             // Wo [128][128], box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap ymap,             // out [R][128], box (32, 128), 64B swizzle (TMA stores)
    const int TRANS) {                                    // geometry of x / y / dy / dx (see load_rows)
  using namespace tl;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sG = reinterpret_cast<__nv_bfloat16*>(smb);   // [NST][g | o]
  __nv_bfloat16* sX = sG + NST * 2 * XT;                         // residual tile
  __nv_bfloat16* sW = sX + XT;                                   // [2 K halves][128][64]
  __nv_bfloat16* sO = sW + C * C;                                // [2][OT]
  uint64_t* bars = reinterpret_cast<uint64_t*>(sO + 2 * OT);
  uint64_t* gf = bars;               // [NST] g | o landed
  uint64_t* uf = gf + NST;           // [NST] count 8: u written
  uint64_t* ge = uf + NST;           // [NST] the GEMM retired (the stage takes the tile after next)
  uint64_t* xf = ge + NST;           // [1]   residual landed
  uint64_t* xe = xf + 1;             // [1]   count 4: residual read
  uint64_t* af = xe + 1;             // [2]
  uint64_t* ae = af + 2;             // [2]   count 4
  uint64_t* wf = ae + 2;             // [1]
  uint32_t* tslot = reinterpret_cast<uint32_t*>(wf + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int i = 0; i < NST; ++i) { bar_init(gf + i, 1); bar_init(uf + i, 8); bar_init(ge + i, 1); }
    for (int i = 0; i < 2; ++i) { bar_init(af + i, 1); bar_init(ae + i, 4); }
    bar_init(xf, 1); bar_init(xe, 4); bar_init(wf, 1);
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, 256);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;

  if (warp == 0) {
    if (lane == 0) {
      expect_tx(wf, C * C * 2);
      for (int kh = 0; kh < 2; ++kh) load_2d(&wmap, sW + kh * C * 64, wf, kh * 64, 0);
      pdl_wait();                                  // PDL: g / o (and the dropout rows) after the core and the front completed
      int lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int st = lt % NST;
        if (lt >= NST) wait(ge + st, ((lt / NST) - 1) & 1);
        expect_tx(gf + st, 2 * XT * 2);
        for (int kh = 0; kh < 2; ++kh) {
          load_2d(&gmap, sG + st * 2 * XT + kh * BM * 64, gf + st, kh * 64, t * BM);
          load_2d(&omap, sG + st * 2 * XT + XT + kh * BM * 64, gf + st, kh * 64, t * BM);
        }
        if (lt >= 1) wait(xe, (lt - 1) & 1);
        expect_tx(xf, XT * 2);
        for (int kh = 0; kh < 2; ++kh) load_rows(&xmap, sX + kh * BM * 64, xf, kh * 64, t * BM, L, TRANS);
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      wait(wf, 0);
      int lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int st = lt % NST, b = lt & 1;
        wait(uf + st, (lt / NST) & 1);
        if (lt >= 2) wait(ae + b, ((lt >> 1) - 1) & 1);
        tc_fence_after();
        const __nv_bfloat16* u = sG + st * 2 * XT;
#pragma unroll
        for (int ks = 0; ks < C / 16; ++ks) {
          const int kh = ks >> 2, kk = (ks & 3) * 16;
          mma_ss(tmem + b * 128, desc_k128(u + kh * BM * 64 + kk), desc_k128(sW + kh * C * 64 + kk), IDESC, ks ? 1u : 0u);
        }
        mma_commit(af + b);
        mma_commit(ge + st);
      }
    }
  } else if (warp < 10) {
    const int r = (warp - 2) * 16 + (lane & 15), hf = lane >> 4;
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int st = lt % NST;
      wait(gf + st, (lt / NST) & 1);
      __nv_bfloat16* gr = sG + st * 2 * XT + hf * BM * 64 + r * 64;
      const __nv_bfloat16* orow = gr + XT;
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) {
        const int off = (c8 ^ (r & 7)) << 3;
        const uint4 a = *reinterpret_cast<const uint4*>(gr + off), o = *reinterpret_cast<const uint4*>(orow + off);
        const uint32_t aa[4] = {a.x, a.y, a.z, a.w}, oo[4] = {o.x, o.y, o.z, o.w};
        uint32_t w[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 gv = bf2f(aa[e]), ov = bf2f(oo[e]);
          w[e] = pack2(sigmoid_t(gv.x) * ov.x, sigmoid_t(gv.y) * ov.y);
        }
        *reinterpret_cast<uint4*>(gr + off) = make_uint4(w[0], w[1], w[2], w[3]);
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(uf + st);
    }
  } else {
    pdl_wait();                                    // PDL: the stores after the core completed
    const int qq = warp & 3, r = qq * 32 + lane;
    const bool leader = warp == 10 && lane == 0;
    int nst = 0, lt = 0;
    // dropout: row (b, i, j) takes scale row (b, j); its 32-channel chunks are loaded a chunk ahead (the first one before the wait for
    // the tile's GEMM), so most of their L2 latency is off the drain's path (loaded in the chunk itself: 1.18x the kernel time at L768;
    // issuing the next chunk's loads at the top of each chunk, a register copy of the current one, was 1.4x slower than this)
    uint4 dn[4];
    auto drop_load = [&](int t, int ch) {
      const uint4* sp = reinterpret_cast<const uint4*>(DROP + drop_row((long)t * BM + r, L, TRANS, false) * C + ch * 32);
#pragma unroll
      for (int c = 0; c < 4; ++c) dn[c] = __ldg(sp + c);
    };
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const int b = lt & 1;
      if (DROP != nullptr) drop_load(t, 0);
      wait(af + b, (lt >> 1) & 1);
      wait(xf, lt & 1);
      tc_fence_after();
#pragma unroll 1
      for (int ch = 0; ch < 4; ++ch) {
        float v[32];
        tmem_ld32(tmem_at(tmem + b * 128 + ch * 32, qq * 32, 0), v);
        // residual: channels 32 ch .. 32 ch + 31 of row r = 16-byte chunks 4 (ch & 1) .. +3 of half ch >> 1
        const __nv_bfloat16* xr = sX + (ch >> 1) * BM * 64 + r * 64;
        uint32_t xw[16];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const uint4 u = *reinterpret_cast<const uint4*>(xr + ((((ch & 1) * 4 + c) ^ (r & 7)) << 3));
          xw[c * 4] = u.x; xw[c * 4 + 1] = u.y; xw[c * 4 + 2] = u.z; xw[c * 4 + 3] = u.w;
        }
        tmem_wait_ld();
        if (ch == 3) {                            // (xe: the residual tile's reads ordered before its release by the proxy fence -- an arrive
          fence_proxy_async();                    // alone does not wait for outstanding LDS, and the next tile's TMA load overwrote them)
          tc_fence_before(); __syncwarp(); if (lane == 0) { arrive(ae + b); arrive(xe); }
        }
        uint32_t w[16];
        if (DROP != nullptr) {
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const uint32_t sw[4] = {dn[c].x, dn[c].y, dn[c].z, dn[c].w};
#pragma unroll
            for (int e = 0; e < 4; ++e) { const float2 sv = bf2f(sw[e]); v[c * 8 + 2 * e] *= sv.x; v[c * 8 + 2 * e + 1] *= sv.y; }
          }
          if (ch < 3) drop_load(t, ch + 1);
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) { const float2 xv = bf2f(xw[i]); w[i] = pack2(v[2 * i] + xv.x, v[2 * i + 1] + xv.y); }
        __nv_bfloat16* buf = sO + (nst & 1) * OT;
        if (leader && nst >= 2) bulk_wait_read<1>();
        named_sync(1, 128);
        unsigned char* rowp = reinterpret_cast<unsigned char*>(buf) + r * 64;
#pragma unroll
        for (int c = 0; c < 4; ++c)
          *reinterpret_cast<uint4*>(rowp + ((c ^ ((r >> 1) & 3)) << 4)) = make_uint4(w[c * 4], w[c * 4 + 1], w[c * 4 + 2], w[c * 4 + 3]);
        fence_proxy_async();
        named_sync(1, 128);
        if (leader) { store_rows(&ymap, buf, ch * 32, t * BM, L, TRANS); bulk_commit(); }
        ++nst;
      }
    }
    if (leader) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 256);
}
// ---------------------------------------------------------------------------------------------------------------------------------
// gate backward:  du = dy . Wo;  s = sigmoid(g);  do = du o s;  dg = du o o o s (1 - s);  delta[b, i, h, j] = sum_{c in h} do . o;
//                 dWo += dy^T . (s o o)   (accumulated in TMEM over the CTA's tiles; at the end added into the fp32 dWo with
//                 atomics, or (deterministic) stored as the CTA's partial for tri_slot_sum to add in CTA order)
// Tile = 128 rows.  warp 0: TMA; warp 1: tcgen05 (du: A = dy K-major, B = Wo MN-major; dWo: A = dy^T, B = u^T, both MN-major
// straight from the row tiles); warps 2-5 / 6-9: two groups, one 64-channel half each (a 32-channel chunk = one head: delta
// falls out per chunk); they write do into its own tile, dg over g and u = s o o over o, and TMA-store do / dg.
// dy and o are double-buffered (the next tile's pair lands while this one computes); g / do single (32 KiB each) + Wo: 224 KiB.
// dy seeds dpair (the residual's share) by a TMA store from the dy buffer right after it lands.
#ifndef GB_NOSEED
#define GB_NOSEED 0
#endif
namespace gb {
constexpr int BM = 128, XT = BM * C;
constexpr int THREADS = 320;
constexpr int SMEM = 1024 + (6 * XT + C * C) * 2 + 512;
constexpr uint32_t ID_DU = idesc_bf16(128, 128, 0, 1);   // A = dy K-major, B = Wo MN-major (N = in contiguous)
constexpr uint32_t ID_DW = idesc_bf16(128, 128, 1, 1);   // A = dy^T, B = u^T: both MN-major
static_assert(SMEM <= 232448, "smem");
}  // namespace gb

__global__ void __launch_bounds__(gb::THREADS, 1) tri_gate_bwd_sm100(
    int L, int ntiles,
    const __grid_constant__ CUtensorMap dymap,            // dy / g / o [R][128] bf16, box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap gmap,
    const __grid_constant__ CUtensorMap omap,
    const __grid_constant__ CUtensorMap wmap,             // Wo [128 out][128 in], box (64, 128), 128B swizzle
    const __grid_constant__ CUtensorMap domap,            // do / dg [R][128], box (64, 128), 128B swizzle (TMA stores)
    const __grid_constant__ CUtensorMap dgmap,
    const __grid_constant__ CUtensorMap seedmap,          // dpair [R][128] <- dy (the residual's share; head_bwd adds dx), box (64, 128), 128B
    float* __restrict__ DELTA,                            // [B, L (i), 4, L (j)]
    float* __restrict__ DWO,                              // [128 out][128 in] fp32, accumulated (zeroed); deterministic: [grid][128][128]
                                                          // partials, this CTA's at DWO + blockIdx.x * C * C
    unsigned* __restrict__ TICKET,                        // zeroed; the last CTA converts dWo and resets it
    __nv_bfloat16* __restrict__ DWO16,
    const __nv_bfloat16* __restrict__ DROP,               // [B][L][128] dropout scale or nullptr: du / dWo see dy o drop, the seed dy
    const int DET,                                        // 1: per-CTA partials (deterministic), 0: fp32 atomics into DWO
    const int TRANS,                                      // geometry of x / y / dy / dx (see load_rows)
    const __nv_bfloat16* __restrict__ BIAS, float* __restrict__ BT, const int S, const int NBT) {   // (see below) or null
  using namespace gb;
  const bool drop = DROP != nullptr;
  extern __shared__ __align__(1024) unsigned char raw[];
  if ((int)blockIdx.x >= ntiles) {
    // a CTA beyond the tiles (fewer tiles than SMs, B200 2026-11-17): the attention core's bias^T * log2e fp32 [b, h, k, q] for its
    // key-side kernel (tri_bwd_pre's 32 x 32 tiles, the same bits) on the otherwise idle SMs -- one launch fewer in the backward
    if (BT == nullptr) return;
    constexpr float L2E_ = 1.4426950408889634f;
    float (*t)[33] = reinterpret_cast<float (*)[33]>(raw);
    const int nt = S / 32, tx = threadIdx.x & 31, ty = threadIdx.x >> 5, nw = (int)blockDim.x >> 5;
    for (int bx = (int)blockIdx.x - ntiles; bx < NBT; bx += (int)gridDim.x - ntiles) {
      const int k0 = (bx % nt) * 32, q0 = ((bx / nt) % nt) * 32, bh = bx / (nt * nt);
      const __nv_bfloat16* src = BIAS + (size_t)bh * S * S;
      for (int i = ty; i < 32; i += nw) t[i][tx] = __bfloat162float(src[(size_t)(q0 + i) * S + k0 + tx]) * L2E_;
      __syncthreads();
      float* dst = BT + (size_t)bh * S * S;
      for (int i = ty; i < 32; i += nw) dst[(size_t)(k0 + i) * S + q0 + tx] = t[tx][i];
      __syncthreads();
    }
    return;
  }
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  __nv_bfloat16* sDy = reinterpret_cast<__nv_bfloat16*>(smb);   // [2 buffers][2 halves][128][64]
  __nv_bfloat16* sG = sDy + 2 * XT;                              // g -> dg
  __nv_bfloat16* sO = sG + XT;                                   // [2 buffers] o -> u
  __nv_bfloat16* sD = sO + 2 * XT;                               // do
  __nv_bfloat16* sW = sD + XT;                                   // Wo
  uint64_t* bars = reinterpret_cast<uint64_t*>(sW + C * C);
  uint64_t* af = bars;               // [2] dy | o landed (per buffer)
  uint64_t* bfb = af + 2;            // g landed
  uint64_t* wf = bfb + 1;            // Wo landed
  uint64_t* duf = wf + 1;            // du ready
  uint64_t* due = duf + 1;           // count 8: du read
  uint64_t* uf = due + 1;            // count 8: u written (both halves)
  uint64_t* dwd = uf + 1;            // [2] the tile's dWo GEMM retired (its dy / u buffer free)
  uint64_t* sd = dwd + 2;            // count 2: the do / dg stores read their tiles (g / do buffers free)
  uint64_t* fin = sd + 1;            // the last dWo GEMM retired
  uint64_t* sr = fin + 1;            // (dropout) the seed store read dy
  uint64_t* dys = sr + 1;            // (dropout) count 8: dy scaled in place
  uint32_t* tslot = reinterpret_cast<uint32_t*>(dys + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int i = 0; i < 2; ++i) { bar_init(af + i, 1); bar_init(dwd + i, 1); }
    bar_init(bfb, 1); bar_init(wf, 1); bar_init(duf, 1); bar_init(due, 8); bar_init(uf, 8);
    bar_init(sd, 2); bar_init(fin, 1); bar_init(sr, 1); bar_init(dys, 8);
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, 256);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;                    // du at 0, dWo at 128
  const int mytiles = (int)blockIdx.x < ntiles ? (ntiles - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;

  if (warp == 0) {
    if (lane == 0) {
      expect_tx(wf, C * C * 2);
      for (int kh = 0; kh < 2; ++kh) load_2d(&wmap, sW + kh * C * 64, wf, kh * 64, 0);
      // dy | o run one tile ahead of g: tile lt + 1's pair lands, seeds dpair and (dropout) is released to the grad warps (sr) while
      // they work on tile lt, so they scale it in place mid-tile and its du GEMM follows tile lt's dWo GEMM without waiting for them
      // (with dy scaled at the start of its own tile, the scale and the du GEMM ran in series with every tile: 1.5x the kernel time)
      auto pair_in = [&](int t, int lt) {
        const int b = lt & 1;
        __nv_bfloat16* dy = sDy + b * XT;
        if (lt >= 2) {                              // buffer b held tile lt - 2: its dWo GEMM retired and its seed store read it
          wait(dwd + b, ((lt >> 1) - 1) & 1);
          bulk_wait_read<1>();                      // (the seed of lt - 1 may still be pending)
        }
        expect_tx(af + b, 2 * XT * 2);
        for (int kh = 0; kh < 2; ++kh) {
          load_rows(&dymap, dy + kh * BM * 64, af + b, kh * 64, t * BM, L, TRANS);
          load_2d(&omap, sO + b * XT + kh * BM * 64, af + b, kh * 64, t * BM);
        }
        wait(af + b, (lt >> 1) & 1);                // dy landed: it seeds dpair
        if (!GB_NOSEED) for (int kh = 0; kh < 2; ++kh) store_rows(&seedmap, dy + kh * BM * 64, kh * 64, t * BM, L, TRANS);
        bulk_commit();
        if (drop) { bulk_wait_read<0>(); arrive(sr); }   // the grad warps scale dy in place next
      };
      if ((int)blockIdx.x < ntiles) pair_in(blockIdx.x, 0);
      int lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        if (lt >= 1) wait(sd, (lt - 1) & 1);
        expect_tx(bfb, XT * 2);
        for (int kh = 0; kh < 2; ++kh) load_2d(&gmap, sG + kh * BM * 64, bfb, kh * 64, t * BM);
        if (t + (int)gridDim.x < ntiles) pair_in(t + gridDim.x, lt + 1);
      }
      bulk_wait<0>();
    }
  } else if (warp == 1) {
    if (lane == 0) {
      wait(wf, 0);
      int lt = 0;
      for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
        const int b = lt & 1;
        const __nv_bfloat16* dy = sDy + b * XT;
        wait(af + b, (lt >> 1) & 1);
        if (drop) wait(dys, lt & 1);                // dy o drop
        if (lt >= 1) wait(due, (lt - 1) & 1);      // the previous du was read
        tc_fence_after();
#pragma unroll
        for (int ks = 0; ks < C / 16; ++ks) {       // du = dy . Wo  (K = out: 16 rows of Wo per step)
          const int kh = ks >> 2, kk = (ks & 3) * 16;
          mma_ss(tmem, desc_k128(dy + kh * BM * 64 + kk), desc_mn128(sW + ks * 16 * 64, C * 128), ID_DU, ks ? 1u : 0u);
        }
        mma_commit(duf);
        wait(uf, lt & 1);
        tc_fence_after();
#pragma unroll
        for (int ks = 0; ks < BM / 16; ++ks)        // dWo += dy^T . u  (K = rows)
          mma_ss(tmem + 128, desc_mn128(dy + ks * 16 * 64, BM * 128), desc_mn128(sO + b * XT + ks * 16 * 64, BM * 128), ID_DW, (lt | ks) ? 1u : 0u);
        mma_commit(dwd + b);
        if (lt == mytiles - 1) mma_commit(fin);
      }
    }
  } else {
    const int qq = warp & 3, r = qq * 32 + lane;   // TMEM lane = row
    const int gi = (warp - 2) >> 2;                // channel half
    const bool lead = qq == 2 && lane == 0;
    // dropout: dy <- dy o drop in place (row r, channel half gi) after the seed store read it, a tile ahead (see the producer); the
    // scale rows are loaded at the top of the previous tile, so their L2 latency hides behind its work
    auto drop_load = [&](int t, uint4* sv) {
      const uint4* sp = reinterpret_cast<const uint4*>(DROP + drop_row((long)t * BM + r, L, TRANS, false) * C + gi * 64);
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) sv[c8] = __ldg(sp + c8);
    };
    auto drop_scale = [&](int lt, const uint4* sv) {
      wait(sr, lt & 1);
      __nv_bfloat16* dr = sDy + (lt & 1) * XT + gi * BM * 64 + r * 64;
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) {
        uint4* pp = reinterpret_cast<uint4*>(dr + ((c8 ^ (r & 7)) << 3));
        const uint4 dv = *pp;
        const uint32_t d4[4] = {dv.x, dv.y, dv.z, dv.w}, s4[4] = {sv[c8].x, sv[c8].y, sv[c8].z, sv[c8].w};
        uint32_t o4[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) { const float2 a = bf2f(d4[e]), m = bf2f(s4[e]); o4[e] = pack2(a.x * m.x, a.y * m.y); }
        *pp = make_uint4(o4[0], o4[1], o4[2], o4[3]);
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(dys);
    };
    if (drop && (int)blockIdx.x < ntiles) { uint4 sv[8]; drop_load(blockIdx.x, sv); drop_scale(0, sv); }
    int lt = 0;
    for (int t = blockIdx.x; t < ntiles; t += gridDim.x, ++lt) {
      const bool nxt = drop && t + (int)gridDim.x < ntiles;
      uint4 sv[8];
      if (nxt) drop_load(t + gridDim.x, sv);
      wait(duf, lt & 1);
      wait(bfb, lt & 1);
      tc_fence_after();
      float du[2][32];
      tmem_ld32(tmem_at(tmem + gi * 64, qq * 32, 0), du[0]);
      tmem_ld32(tmem_at(tmem + gi * 64 + 32, qq * 32, 0), du[1]);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) arrive(due);
      if (lt >= 1 && lead) bulk_wait_read<0>();    // the previous tile's do store read sD (dg left with g's buffer)
      named_sync(1 + gi, 128);
      const long row = (long)t * BM + r;           // (b, i, j)
      const int j = (int)(row % L);
      const long bi = row / L;
      __nv_bfloat16* sOb = sO + (lt & 1) * XT;
#pragma unroll
      for (int cc = 0; cc < 2; ++cc) {
        const int c = gi * 2 + cc;                 // head
        const int boff = gi * BM * 64 + r * 64;
        float dsum = 0.f;
#pragma unroll
        for (int e4 = 0; e4 < 4; ++e4) {
          const int off = boff + (((cc * 4 + e4) ^ (r & 7)) << 3);
          const uint4 ga = *reinterpret_cast<const uint4*>(sG + off), oa = *reinterpret_cast<const uint4*>(sOb + off);
          const uint32_t gw[4] = {ga.x, ga.y, ga.z, ga.w}, ow[4] = {oa.x, oa.y, oa.z, oa.w};
          uint32_t dow[4], dgw[4], uw[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float2 gv = bf2f(gw[e]), ov = bf2f(ow[e]);
            const float s0 = sigmoid_t(gv.x), s1 = sigmoid_t(gv.y);
            const float u0 = du[cc][e4 * 8 + e * 2], u1 = du[cc][e4 * 8 + e * 2 + 1];
            const float d0 = u0 * s0, d1 = u1 * s1;
            dsum = fmaf(d0, ov.x, fmaf(d1, ov.y, dsum));
            dow[e] = pack2(d0, d1);
            dgw[e] = pack2(u0 * ov.x * s0 * (1.f - s0), u1 * ov.y * s1 * (1.f - s1));
            uw[e] = pack2(s0 * ov.x, s1 * ov.y);
          }
          *reinterpret_cast<uint4*>(sD + off) = make_uint4(dow[0], dow[1], dow[2], dow[3]);
          *reinterpret_cast<uint4*>(sG + off) = make_uint4(dgw[0], dgw[1], dgw[2], dgw[3]);
          *reinterpret_cast<uint4*>(sOb + off) = make_uint4(uw[0], uw[1], uw[2], uw[3]);
        }
        DELTA[(bi * 4 + c) * L + j] = dsum;
      }
      fence_proxy_async();
      named_sync(1 + gi, 128);
      __syncwarp();
      if (lane == 0) arrive(uf);
      if (lead) {
        store_2d(&domap, sD + gi * BM * 64, gi * 64, t * BM);
        store_2d(&dgmap, sG + gi * BM * 64, gi * 64, t * BM);
        bulk_commit();
      }
      // the next tile's dy o drop while the stores drain and the next g loads: its du GEMM follows this tile's dWo GEMM (scaled at
      // the start of its own tile, the scale and the du GEMM ran in series with every tile: 1.55x the kernel time at L768)
      if (nxt) drop_scale(lt + 1, sv);
      if (lead) {
        bulk_wait_read<0>();
        arrive(sd);                                // (both groups: count 2) g / do buffers free
      }
    }
    // ---- dWo: TMEM lane = out, 128 in columns; group 0 adds them into the fp32 dWo (or stores the CTA's partial into its slot) ----
    if (gi == 0 && mytiles > 0) {
      wait(fin, 0);
      tc_fence_after();
#pragma unroll 1
      for (int ch = 0; ch < 4; ++ch) {
        float v[32];
        tmem_ld32(tmem_at(tmem + 128 + ch * 32, qq * 32, 0), v);
        tmem_wait_ld();
        if (DET) {
          float4* dst = reinterpret_cast<float4*>(DWO + (long)blockIdx.x * C * C + (long)r * C + ch * 32);
#pragma unroll
          for (int i = 0; i < 8; ++i) dst[i] = make_float4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);
        } else {
          float* dst = DWO + (long)r * C + ch * 32;
#pragma unroll
          for (int i = 0; i < 32; i += 4)
            asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" :: "l"(dst + i), "f"(v[i]), "f"(v[i + 1]), "f"(v[i + 2]), "f"(v[i + 3]) : "memory");
        }
      }
    }
    if (lead) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 256);
  (void)TICKET; (void)DWO16;                        // (a last-CTA bf16 conversion here cost 15 us: tri_param_finish does it)
}
// ---------------------------------------------------------------------------------------------------------------------------------
// head backward:  gy = sum_t d_t . W_t + sum_h db_h Wb_h   (the five projections' input gradient: q | k | v | g via tcgen05, the
//                 4-wide bias one on the CUDA cores);  LN backward from the recomputed statistics;  dpair += dx (a bf16 TMA
//                 reduce-add into the seed gate_bwd wrote, in place).  The weight / LN-parameter gradients are tri_wgrad's.
// Tile = 128 rows.  warp 0: TMA (x double-buffered; the d_t half-tiles [128][64] in a ring; W once); warp 1: tcgen05 (M = 128 rows,
// N = 128, K = 512 into one of two gy accumulators); warps 2-5 / 6-9: two groups, one 64-channel half each, a row a thread (TMEM
// lane); the row sums go between the groups through shared.  dx overwrites x (the reduce-add's source).
struct HbParams { float lnw[C], lnb[C], wb[NH][C]; };
__constant__ HbParams c_hb;                                // uniform constant-bank operands (filled by an async D2D copy: graph-safe)
namespace hb {
constexpr int BM = 128, XT = BM * C, HT = BM * 64;         // x tile (32 KiB), a d half-tile [128][64] (16 KiB)
#define TLS(k, i) ((void)0)
#define TLX(i) ((void)0)
#ifndef HB_NR
#define HB_NR 2
#endif
#ifndef HB_GIC
#define HB_GIC 0
#endif
#ifndef HB_BOX3
#define HB_BOX3 0
#endif
static_assert(!HB_BOX3 || HB_NR % 2 == 0, "HB_BOX3: slot pairs");
#ifndef HB_NX
#define HB_NX 2
#endif
constexpr int NR = HB_NR, NX = HB_NX;                      // d ring, x buffers
constexpr int THREADS = 320;
// no 1024 B alignment slack: the dynamic shared window starts 1 KiB aligned on this part (measured; the kernel traps otherwise)
constexpr int SMEM = (4 * C * C + NR * HT + NX * XT) * 2 + 2 * 2 * BM * 4 + 256;
constexpr uint32_t ID_DG = idesc_bf16(128, 128, 0, 1);    // A = d_t K-major, B = W_t MN-major
static_assert(SMEM <= 232448, "smem");
}  // namespace hb

// (round 9) the body takes its CTA index / count (BID, NBLK): tri_head_bwd_sm100 runs it on the whole grid, tri_hw_bwd_sm100 on the
// grid's first NBLK CTAs beside tri_wgrad's body on the others
__device__ __forceinline__ void head_bwd_body(const int BID, const int NBLK,
    int L, int ntiles, float eps,
    const CUtensorMap& d0map,                             // dq / dk / dv / dg [R][128], box (64, 128), 128B swizzle
    const CUtensorMap& d1map,
    const CUtensorMap& d2map,
    const CUtensorMap& d3map,
    const CUtensorMap& xmap,                              // x [R][128], box (64, 128), 128B swizzle
    const CUtensorMap& wmap,                              // W4 [512][128], box (64, 256), 128B swizzle
    const CUtensorMap& rmap,                              // dpair [R][128] (the seed; reduce-add target), box (64, 128), 128B swizzle
    const float* __restrict__ DB,                         // dbias [B, 4, L, L] fp32, or (SL) the summed dbias written here
    const int TRANS) {                                    // geometry of x / y / dy / dx (see load_rows)
  using namespace hb;
  extern __shared__ __align__(1024) unsigned char raw[];
  if (BID >= ntiles) return;   // (HEAD_DWO, measured and rejected: the dWo slot sum on CTAs beyond the tiles; see b200_triattn.py)
  if (threadIdx.x == 0) TLS(0, 0);
  if (sa(raw) & 1023u) __trap();
  unsigned char* smb = raw;
  __nv_bfloat16* sW = reinterpret_cast<__nv_bfloat16*>(smb);    // [2 c halves][512 o][64]
  __nv_bfloat16* sD = sW + 4 * C * C;                            // [NR][128][64]
  __nv_bfloat16* sX = sD + NR * HT;                              // [NX] x -> dx   [2 halves][128][64]
  float* sRow = reinterpret_cast<float*>(sX + NX * XT);          // [2 groups][2][128] row partial sums
  uint64_t* bars = reinterpret_cast<uint64_t*>(sRow + 2 * 2 * BM);
  uint64_t* df = bars;               // [NR]
  uint64_t* de = df + NR;            // [NR]
  uint64_t* xf = de + NR;            // [NX]
  uint64_t* xe = xf + NX;            // [NX] the dx reduce read the tile
  uint64_t* af = xe + NX;            // [2]  gy ready
  uint64_t* ae = af + 2;             // [2]  count 8: gy read
  uint64_t* wf = ae + 2;
  uint32_t* tslot = reinterpret_cast<uint32_t*>(wf + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int i = 0; i < NR; ++i) { bar_init(df + i, 1); bar_init(de + i, 1); }
    for (int i = 0; i < NX; ++i) { bar_init(xf + i, 1); bar_init(xe + i, 1); }
    for (int i = 0; i < 2; ++i) { bar_init(af + i, 1); bar_init(ae + i, 8); }
    bar_init(wf, 1);
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, 256);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  const CUtensorMap* dmaps[4] = {&d0map, &d1map, &d2map, &d3map};
  if (threadIdx.x == 0) TLS(0, 1);

  if (warp == 0) {
    if (lane == 0) {
      expect_tx(wf, 4 * C * C * 2);
      for (int kh = 0; kh < 2; ++kh)
        for (int nh = 0; nh < 2; ++nh) load_2d(&wmap, sW + kh * 4 * C * 64 + nh * 256 * 64, wf, kh * 64, nh * 256);
      int lt = 0, y = 0;
      for (int t = BID; t < ntiles; t += NBLK, ++lt) {
        const int xb = lt % NX;
        if (lt >= NX) wait(xe + xb, ((lt / NX) - 1) & 1);
        expect_tx(xf + xb, XT * 2);
        for (int kh = 0; kh < 2; ++kh) load_rows(&xmap, sX + xb * XT + kh * BM * 64, xf + xb, kh * 64, t * BM, L, TRANS);
#if HB_BOX3
        // (round 9) both K halves of a tensor in ONE 3-D box (64 cols x 128 rows x 2 halves = 32 KiB) into a pair of ring slots, on the
        // even slot's barrier (fewer, larger TMA boxes: the per-box issue cost, docs/b200/transition.md part 14)
        for (int q8 = 0; q8 < 8; q8 += 2, y += 2) {
          const int sl = y % NR;
          if (y >= NR) { wait(de + sl, ((y / NR) - 1) & 1); wait(de + sl + 1, ((y / NR) - 1) & 1); }
          expect_tx(df + sl, 2 * HT * 2);
          load_3d(dmaps[q8 >> 1], sD + sl * HT, df + sl, 0, t * BM, 0);
        }
#else
        for (int q8 = 0; q8 < 8; ++q8, ++y) {      // (tensor q8 >> 1, K half q8 & 1)
          const int sl = y % NR;
          if (y >= NR) wait(de + sl, ((y / NR) - 1) & 1);
          expect_tx(df + sl, HT * 2);
          load_2d(dmaps[q8 >> 1], sD + sl * HT, df + sl, (q8 & 1) * 64, t * BM);
        }
#endif
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      wait(wf, 0);
      TLS(0, 2);
      int lt = 0, y = 0;
      for (int t = BID; t < ntiles; t += NBLK, ++lt) {
        const int b = lt & 1;
        if (lt >= 2) wait(ae + b, ((lt >> 1) - 1) & 1);
        for (int q8 = 0; q8 < 8; ++q8, ++y) {
          const int sl = y % NR, ten = q8 >> 1, kh = q8 & 1;
          if (!HB_BOX3 || (q8 & 1) == 0) wait(df + sl, (y / NR) & 1);
          if (y == 0) TLS(0, 3);
          tc_fence_after();
          const __nv_bfloat16* dt = sD + sl * HT;
#pragma unroll
          for (int kk = 0; kk < 4; ++kk)            // gy += d_t[:, 64 kh + 16 kk ..] . W_t[64 kh + 16 kk .., :]
            mma_ss(tmem + b * 128, desc_k128(dt + kk * 16), desc_mn128(sW + (ten * C + kh * 64 + kk * 16) * 64, 4 * C * 128), ID_DG, (q8 | kk) ? 1u : 0u);
          mma_commit(de + sl);
        }
        mma_commit(af + b);
      }
    }
  } else {
    const int qq = warp & 3, r = qq * 32 + lane;   // TMEM lane = row
    // (round 9, HB_GIC) the channel half as a compile-time constant: c_hb's Wb / gamma become immediate constant-bank operands of the
    // FFMAs (with gi from the warp index every one needed an indexed constant load)
#if HB_GIC
    auto epi = [&](auto gic) {
    constexpr int gi = decltype(gic)::value;
#else
    {
    const int gi = (warp - 2) >> 2;                // channel half
#endif
    const bool leader = warp == 2 && lane == 0;
    int lt = 0;
    for (int t = BID; t < ntiles; t += NBLK, ++lt) {
      const int xb = lt % NX, b = lt & 1;
      __nv_bfloat16* xr = sX + xb * XT + gi * BM * 64 + r * 64;
      const long row = (long)t * BM + r;           // (b, r1, r2): dbias[b, h, r1, r2]
      const long bb = row / ((long)L * L), rr = row % ((long)L * L);
      float dbv[NH];
#pragma unroll
      for (int h = 0; h < NH; ++h) dbv[h] = DB[(bb * NH + h) * (long)L * L + rr];
      wait(xf + xb, (lt / NX) & 1);
      if (leader && lt == 0) TLS(0, 4);
      // ---- statistics (the two halves' partial sums through shared) ----
      float xv[64];
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) {
        const uint4 u = *reinterpret_cast<const uint4*>(xr + ((c8 ^ (r & 7)) << 3));
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) { const float2 v = bf2f(w[e]); xv[c8 * 8 + e * 2] = v.x; xv[c8 * 8 + e * 2 + 1] = v.y; }
      }
      float s1 = 0.f, s2 = 0.f;
#pragma unroll
      for (int i = 0; i < 64; ++i) { s1 += xv[i]; s2 = fmaf(xv[i], xv[i], s2); }
      sRow[(gi * 2) * BM + r] = s1; sRow[(gi * 2 + 1) * BM + r] = s2;
      named_sync(1, 256);
      s1 += sRow[((gi ^ 1) * 2) * BM + r]; s2 += sRow[((gi ^ 1) * 2 + 1) * BM + r];
      named_sync(1, 256);                           // (the buffer is reused for the second exchange)
      const float mean = s1 * (1.f / C), rstd = rsqrtf(fmaxf(s2 * (1.f / C) - mean * mean, 0.f) + eps);
#pragma unroll
      for (int i = 0; i < 64; ++i) xv[i] = (xv[i] - mean) * rstd;   // xhat
      // ---- gy = TMEM + db . Wb; the row sums ----
      wait(af + b, (lt >> 1) & 1);
      if (leader && lt == 0) TLS(0, 5);
      tc_fence_after();
      float gy[64];
      tmem_ld32(tmem_at(tmem + b * 128 + gi * 64, qq * 32, 0), gy);
      tmem_ld32(tmem_at(tmem + b * 128 + gi * 64 + 32, qq * 32, 0), gy + 32);
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) arrive(ae + b);
      float sg = 0.f, sgx = 0.f;
#pragma unroll
      for (int i = 0; i < 64; ++i) {
        const int c = gi * 64 + i;
        float v = gy[i];
#pragma unroll
#pragma unroll
        for (int h = 0; h < NH; ++h) v = fmaf(dbv[h], c_hb.wb[h][c], v);
        gy[i] = v * c_hb.lnw[c];
        sg += gy[i]; sgx = fmaf(gy[i], xv[i], sgx);
      }
      sRow[(gi * 2) * BM + r] = sg; sRow[(gi * 2 + 1) * BM + r] = sgx;
      named_sync(1, 256);
      sg += sRow[((gi ^ 1) * 2) * BM + r]; sgx += sRow[((gi ^ 1) * 2 + 1) * BM + r];
      const float mg = sg * (1.f / C), mgx = sgx * (1.f / C);
      // ---- dx over x (in place) -> one TMA reduce-add into dpair ----
#pragma unroll
      for (int c8 = 0; c8 < 8; ++c8) {
        uint32_t w[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int i = c8 * 8 + e * 2;
          w[e] = pack2(rstd * (gy[i] - mg - xv[i] * mgx), rstd * (gy[i + 1] - mg - xv[i + 1] * mgx));
        }
        *reinterpret_cast<uint4*>(xr + ((c8 ^ (r & 7)) << 3)) = make_uint4(w[0], w[1], w[2], w[3]);
      }
      fence_proxy_async();
      named_sync(1, 256);
      if (leader) {
        for (int kh = 0; kh < 2; ++kh) reduce_rows(&rmap, sX + xb * XT + kh * BM * 64, kh * 64, t * BM, L, TRANS);
        if (lt == 0) TLS(0, 6);
        bulk_commit();
        bulk_wait_read<0>();
        arrive(xe + xb);
      }
    }
    if (leader) { bulk_wait<0>(); TLS(0, 7); }
#if HB_GIC
    };
    if (warp < 6) epi(std::integral_constant<int, 0>{}); else epi(std::integral_constant<int, 1>{});
#else
    }
#endif
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 256);
}

__global__ void __launch_bounds__(hb::THREADS, 1) tri_head_bwd_sm100(
    int L, int ntiles, float eps,
    const __grid_constant__ CUtensorMap d0map, const __grid_constant__ CUtensorMap d1map,
    const __grid_constant__ CUtensorMap d2map, const __grid_constant__ CUtensorMap d3map,
    const __grid_constant__ CUtensorMap xmap, const __grid_constant__ CUtensorMap wmap, const __grid_constant__ CUtensorMap rmap,
    const float* __restrict__ DB, const int TRANS) {
  head_bwd_body((int)blockIdx.x, (int)gridDim.x, L, ntiles, eps, d0map, d1map, d2map, d3map, xmap, wmap, rmap, DB, TRANS);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// weight / LN-parameter gradients.  With xhat the normalized rows (y = gamma o xhat + beta) and d_t the four projections' output
// gradients, everything reduces to accumulations the tensor core and a column sum can do:
//   M_t = xhat^T d_t (tcgen05: M = 128 c, N = 64 o per d half-tile, K = 128 rows; four [128][128] accumulators = all of TMEM),
//   cs_t = 1^T d_t, B_h = sum_rows db_h xhat, cdb_h = sum_rows db_h         (CUDA cores)
// and then (host-side algebra on [128][128] matrices):  dW_t^T = gamma o M_t + beta (x) cs_t;
//   dgamma = sum_t diag(W_t^T M_t^T) + sum_h Wb_h o B_h;  dbeta = sum_t cs_t W_t + sum_h cdb_h Wb_h;  dWb_h = gamma o B_h + beta cdb_h.
// Tile = 128 rows.  warp 0: TMA (x double-buffered, the d half-tiles in a ring, db rows); warp 1: tcgen05; warps 2-5: a row a thread
// (statistics, xhat over x in place = the GEMMs' A tile); warps 6-9: a channel a thread (column sums of the d half-tiles, B_h, cdb).
#ifndef WG_ROT
#define WG_ROT 0         // rotate each CTA's flush start across the 16 column chunks (spreads the same-address reductions)
#endif
#ifndef WG_VEC
#define WG_VEC 1         // column sums / B_h on 16-byte loads, a thread per (8-column chunk, 8-row group); 0 = a thread per channel
#endif
namespace wg {
constexpr int BM = 128, XT = BM * C, HT = BM * 64;
#ifndef WG_NR
#define WG_NR 4
#endif
#ifndef WG_DST
#define WG_DST 0
#endif
constexpr int NR = WG_NR, NX = 2;
constexpr int THREADS = 320;
constexpr int NVEC = 8 * C + NH, RED = 4 * 1056;             // the channel warps' end-of-kernel partials [4 warps][NVEC (padded)]
constexpr int NVECP = 1056;                                  // a CTA's slot of the vector partials (NVEC padded to 128 B)
constexpr int SCR = 1024;                                    // (round 10) floats of scratch: the folded dWo sum's lane partials
constexpr int SMEM = 1024 + (NX * XT + NR * HT) * 2 + NX * BM * NH * 4 + (WG_VEC ? RED * 4 : 0) + SCR * 4 + 256;
constexpr uint32_t IDESC = idesc_bf16(128, 64, 1, 1);     // A = xhat^T, B = d_t^T (a 64-wide half): both MN-major
static_assert(SMEM <= 232448, "smem");
static_assert(WG_VEC, "the vector sums run on 16-byte loads only");
// tensor groups per row tile (see tri_wgrad_sm100): 2 while the tiles are few (a CTA runs half the projections over twice the tiles:
// half the deterministic partials). Measured d128 training, wgrad + slot sum us, NG 1 / 2 / 4: L128 24.0 / 22.5 / 22.6, L256 41.9 /
// 41.1 / 48.9, L768 183.5 / 210.7 / 294.3 (B200 2026-11-17)
inline int wg_groups(int ntiles, int sms) { return ntiles <= 8 * sms ? 2 : 1; }
}  // namespace wg

// (round 10) The deterministic tails folded into tri_wgrad (no tri_slot_sum / tri_param_finish launches).  The wgrad grid is at most
// one CTA an SM (nsp <= SMs / NG, 512 TMEM columns a CTA), so every CTA is resident once the previous kernel drained: a grid barrier is
// safe.  (1) The gate backward's dWo partials (ready before this kernel) are summed in CTA order by the row warps 2-5 after their xhat
// loop (idle while the MMAs and the flush run) and written in the parameters' dtype.  (2) After the flush: a self-resetting grid barrier,
// then CTA u (of 128 units: tensor t, 8 outputs o, 64 channels c) sums its [64 c][8 o] block of M_t over the slots in slot order, its 8
// cs_t (and one float4 of B_h / cdb) and writes dW_t[o][c] = gamma_c M + beta_c cs directly, plus its share of dgamma / dbeta
// (sum_o W_t[o][c] M[c][o], sum_o W_t[o][c] cs[o]) into PART; (3) the last CTA to arrive (a self-resetting ticket) adds the 64 shares
// a channel in a fixed order and writes dgamma | dbeta | dWb.  Every sum in a fixed order: the same bits every run.
// Taken (the default, host fold = 2): (1) alone (F.mts null) -- the slot sum keeps the M_t / vector slots, the finish runs as before.
// (2) + (3) (fold = 1) measured slower than the separate launches (L128 +0.4 us, L256 / 384 +6 us; docs/b200/triattn.md round 10): opt-in.
#ifndef WF_UN
#define WF_UN 8          // slot loads in flight a thread (the folded M_t sums)
#endif
struct WFold {
  unsigned* bar;                 // [4] zeroed once: arrivals, generation, finish ticket (all self-resetting); null = off
  const float* dwo_in;           // gate_bwd's dWo partials [n_dwo][C][C] (or null)
  int n_dwo;
  void* dwo_out;                 // [C][C] fp32 / bf16
  float* part;                   // [2][64][C] the units' dgamma / dbeta shares
  float* bhs;                    // [4 C + 4] B_h | cdb summed
  const float* mts;              // the M_t slots [nsp][4][C][C]
  const float* vecs;             // the vector slots [nsp][NVECP]
  const __nv_bfloat16* w4;       // [4][C o][C c]
  const float* lnw; const float* lnb; const float* wbf;
  void* dw;                      // [4][C o][C c]
  void* dgb;                     // dgamma | dbeta | dWb [4][C]
  int f32;                       // outputs fp32 (else bf16)
  int nsp;
};
__device__ __forceinline__ void st_any(void* p, long i, float v, int f32) {
  if (f32) reinterpret_cast<float*>(p)[i] = v; else reinterpret_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16_rn(v);
}
__device__ __forceinline__ unsigned ld_acquire_u32(const unsigned* p) {
  unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void st_release_u32(unsigned* p, unsigned v) { asm volatile("st.release.gpu.global.u32 [%0], %1;\n" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ float4 ldcg4(const float* p) { return __ldcg(reinterpret_cast<const float4*>(p)); }
__device__ __forceinline__ void add4(float4& a, const float4& v) { a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w; }
// all n CTAs (resident) meet (CUTLASS's generic barrier form: release by fence.acq_rel + a relaxed reduction, acquire polls, no
// returned atomics or SC fences on the path): cnt is reset by the finish ticket's last CTA (wf_ticket), after every CTA passed
__device__ __forceinline__ void wf_arrive_wait(unsigned* cnt, unsigned n) {
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
    asm volatile("red.relaxed.gpu.global.add.u32 [%0], 1;\n" :: "l"(cnt) : "memory");
    while (ld_acquire_u32(cnt) < n) {}
  }
  __syncthreads();
}
// the CTA's ticket (arrival order) after its writes: release + acquire
__device__ __forceinline__ unsigned wf_ticket(unsigned* p) {
  unsigned v;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void wgrad_body(const int BID, const int NBLK,
    int L, int ntiles, float eps,
    const CUtensorMap& d0map, const CUtensorMap& d1map,
    const CUtensorMap& d2map, const CUtensorMap& d3map,
    const CUtensorMap& xmap,
    const float* __restrict__ DB,                         // dbias [B, 4, L, L]
    float* __restrict__ MT,                               // (unused: the partials leave through mtmap)
    float* __restrict__ VEC,                              // cs [4][128] | B [4][128] | cdb [4], accumulated (zeroed); deterministic:
                                                          // [grid][NVECP], this CTA's partial
    unsigned* __restrict__ TICKET,                        // zeroed; the last CTA finishes and resets it
    const __nv_bfloat16* __restrict__ W4, const float* __restrict__ LNW, const float* __restrict__ LNB, const float* __restrict__ WBF,
    __nv_bfloat16* __restrict__ DW,                       // dW_q|k|v|g [4][128 o][128 c]
    __nv_bfloat16* __restrict__ DGB,                      // dgamma [128] | dbeta [128] | dWb [4][128]
    const int want_finish,                                // bit 1: deterministic (per-CTA partials: VEC slots, mtmap over the M_t slots)
    const CUtensorMap& mtmap,                             // MT fp32 [4*128 c][128 o] (reduce-add target) or the slots [grid*4*128 c][128 o]; box (32, 128), 128B swizzle
    const int TRANS,                                      // geometry of x / y / dy / dx (see load_rows)
    const int NG,                                         // tensor groups (1, 2, 4): see below
    const WFold& F) {                                     // (round 10) the folded tails (F.bar null: off)
  using namespace wg;
  extern __shared__ __align__(1024) unsigned char raw[];
  unsigned char* smb = raw + ((1024u - (sa(raw) & 1023u)) & 1023u);
  if (threadIdx.x == 0) TLS(1, 0);
  __nv_bfloat16* sX = reinterpret_cast<__nv_bfloat16*>(smb);   // [NX] x -> xhat
  __nv_bfloat16* sD = sX + NX * XT;                              // [NR][128][64]
  float* sDb = reinterpret_cast<float*>(sD + NR * HT);           // [NX][128 rows][4]
  float* sRed = sDb + NX * BM * NH;                              // [4][1056] (WG_VEC)
  float* sScr = sRed + (WG_VEC ? RED : 0);                       // [SCR] (round 10) fold scratch
  uint64_t* bars = reinterpret_cast<uint64_t*>(sScr + SCR);
  uint64_t* xf = bars;               // [NX]
  uint64_t* hf = xf + NX;            // [NX] count 4: xhat written
  uint64_t* he = hf + NX;            // [NX] count 5: the tile's GEMMs retired (commit) + the channel warps read xhat / db
  uint64_t* df = he + NX;            // [NR]
  uint64_t* de = df + NR;            // [NR] count 5: the GEMM + the channel warps
  uint64_t* fin = de + NR;
  uint32_t* tslot = reinterpret_cast<uint32_t*>(fin + 1);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int i = 0; i < NX; ++i) { bar_init(xf + i, 1); bar_init(hf + i, 4); bar_init(he + i, 5); }
    for (int i = 0; i < NR; ++i) { bar_init(df + i, 1); bar_init(de + i, 5); }
    bar_init(fin, 1);
    bar_init_fence();
  }
  if (warp == 1) tmem_alloc(tslot, 512);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tslot;
  const CUtensorMap* dmaps[4] = {&d0map, &d1map, &d2map, &d3map};
  // NG tensor groups: CTA (split, grp) = blockIdx (/ NG, % NG) owns the TS = 4 / NG projections T0 .. T0 + TS - 1 over the row tiles
  // split, split + nsp, ...  (TMEM: the local tensor l at column 128 l; group 0 also sums B_h / cdb)
  const int TS = 4 / NG, grp = BID % NG, split = BID / NG, nsp = NBLK / NG, T0 = grp * TS;
  const int mytiles = split < ntiles ? (ntiles - 1 - split) / nsp + 1 : 0;

  if (warp == 0) {
    if (lane == 0) {
      int lt = 0, y = 0;
      for (int t = split; t < ntiles; t += nsp, ++lt) {
        const int xb = lt % NX;
        if (lt >= NX) wait(he + xb, ((lt / NX) - 1) & 1);
        expect_tx(xf + xb, XT * 2 + BM * NH * 4);
        for (int kh = 0; kh < 2; ++kh) load_rows(&xmap, sX + xb * XT + kh * BM * 64, xf + xb, kh * 64, t * BM, L, TRANS);
        {                                           // the tile's db rows: 4 planes x 128 contiguous floats (512 B each)
          const long row0 = (long)t * BM, bb = row0 / ((long)L * L), rr = row0 % ((long)L * L);
          for (int h = 0; h < NH; ++h)
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                         :: "r"(sa(sDb + (xb * NH + h) * BM)), "l"(DB + (bb * NH + h) * (long)L * L + rr), "r"(BM * 4), "r"(sa(xf + xb)) : "memory");
        }
        for (int q8 = 0; q8 < 2 * TS; ++q8, ++y) {
          const int sl = y % NR;
          if (y >= NR) wait(de + sl, ((y / NR) - 1) & 1);
          expect_tx(df + sl, HT * 2);
          load_2d(dmaps[T0 + (q8 >> 1)], sD + sl * HT, df + sl, (q8 & 1) * 64, t * BM);
        }
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      int lt = 0, y = 0;
      for (int t = split; t < ntiles; t += nsp, ++lt) {
        const int xb = lt % NX;
        wait(hf + xb, (lt / NX) & 1);
        if (lt == 0) TLS(1, 3);
        for (int q8 = 0; q8 < 2 * TS; ++q8, ++y) {
          const int sl = y % NR, ten = q8 >> 1, kh = q8 & 1;   // ten: local
          wait(df + sl, (y / NR) & 1);
          tc_fence_after();
#pragma unroll
          for (int ks = 0; ks < BM / 16; ++ks)      // M_t[:, 64 kh ..] += xhat^T . d_t-half   (K = rows)
            mma_ss(tmem + ten * 128 + kh * 64, desc_mn128(sX + xb * XT + ks * 16 * 64, BM * 128), desc_mn128(sD + sl * HT + ks * 16 * 64, BM * 128),
                   IDESC, (lt | ks) ? 1u : 0u);
          mma_commit(de + sl);
        }
        mma_commit(he + xb);
        if (lt == mytiles - 1) mma_commit(fin);
      }
    }
  } else if (warp < 6) {
    // ---- a row a thread: statistics, xhat (bf16) over x ----
    const int r = (warp - 2) * 32 + lane;
    int lt = 0;
    for (int t = split; t < ntiles; t += nsp, ++lt) {
      const int xb = lt % NX;
      wait(xf + xb, (lt / NX) & 1);
      if (lt == 0 && r == 0) TLS(1, 2);
      float s1 = 0.f, s2 = 0.f;
#pragma unroll
      for (int c8 = 0; c8 < 16; ++c8) {
        const uint4 u = *reinterpret_cast<const uint4*>(sX + xb * XT + (c8 >> 3) * BM * 64 + r * 64 + (((c8 & 7) ^ (r & 7)) << 3));
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) { const float2 v = bf2f(w[e]); s1 += v.x + v.y; s2 = fmaf(v.x, v.x, fmaf(v.y, v.y, s2)); }
      }
      const float mean = s1 * (1.f / C), rstd = rsqrtf(fmaxf(s2 * (1.f / C) - mean * mean, 0.f) + eps);
#pragma unroll
      for (int c8 = 0; c8 < 16; ++c8) {
        __nv_bfloat16* p = sX + xb * XT + (c8 >> 3) * BM * 64 + r * 64 + (((c8 & 7) ^ (r & 7)) << 3);
        const uint4 u = *reinterpret_cast<const uint4*>(p);
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
        uint32_t o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) { const float2 v = bf2f(w[e]); o[e] = pack2((v.x - mean) * rstd, (v.y - mean) * rstd); }
        *reinterpret_cast<uint4*>(p) = make_uint4(o[0], o[1], o[2], o[3]);
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) arrive(hf + xb);
    }
    // (round 10) the gate backward's dWo partials, this CTA's float4 columns: 32 columns x 4 slot lanes (lane g: slots g, g + 4, ..),
    // the lane sums added in lane order
    if (F.bar != nullptr && F.dwo_in != nullptr) {
      const int r = tid - 64, col = r & 31, sl = r >> 5, ncol = C * C / 4, per = (ncol + NBLK - 1) / NBLK;
      const int cb0 = BID * per, cb1 = min(ncol, cb0 + per);
      float4* scr = reinterpret_cast<float4*>(sScr);
      for (int cb = cb0; cb < cb1; cb += 32) {
        const int c4 = cb + col;
        float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
        if (c4 < cb1) {
          const float* p = F.dwo_in + 4L * c4;
          int s = sl;
          for (; s + 28 < F.n_dwo; s += 32) {
            float4 v[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) v[u] = __ldg(reinterpret_cast<const float4*>(p + (long)(s + 4 * u) * C * C));
#pragma unroll
            for (int u = 0; u < 8; ++u) add4(a, v[u]);
          }
          for (; s < F.n_dwo; s += 4) add4(a, __ldg(reinterpret_cast<const float4*>(p + (long)s * C * C)));
        }
        if (sl > 0) scr[(sl - 1) * 32 + col] = a;
        named_sync(4, 128);
        if (sl == 0 && c4 < cb1) {
#pragma unroll
          for (int g = 0; g < 3; ++g) add4(a, scr[g * 32 + col]);
          st_any(F.dwo_out, 4L * c4, a.x, F.f32); st_any(F.dwo_out, 4L * c4 + 1, a.y, F.f32);
          st_any(F.dwo_out, 4L * c4 + 2, a.z, F.f32); st_any(F.dwo_out, 4L * c4 + 3, a.w, F.f32);
        }
        named_sync(4, 128);
      }
    }
    if (tid == 64) TLS(1, 1);
  } else {
    // ---- column sums of the d half-tiles, B_h = sum db_h xhat_c, cdb_h: a thread per (8-column chunk j, 8-row group rg), 16-byte
    // swizzled loads, fp32x2 accumulation in registers over all the CTA's tiles; one shuffle / atomic reduction at the end ----
    const int t = tid - 192, j = t & 7, rg = t >> 3;                    // rg 0..15: rows 8 rg ..
    float2 cs2[8][4], bh2[NH][2][4];
    float cdb[NH] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int a8 = 0; a8 < 8; ++a8)
#pragma unroll
      for (int e = 0; e < 4; ++e) cs2[a8][e] = make_float2(0.f, 0.f);
#pragma unroll
    for (int h = 0; h < NH; ++h)
#pragma unroll
      for (int kh = 0; kh < 2; ++kh)
#pragma unroll
        for (int e = 0; e < 4; ++e) bh2[h][kh][e] = make_float2(0.f, 0.f);
    int lt = 0, y = 0;
    for (int tt = split; tt < ntiles; tt += nsp, ++lt) {
      const int xb = lt % NX;
      wait(hf + xb, (lt / NX) & 1);                 // xhat written (and x / db landed)
      const float* dbp = sDb + xb * NH * BM;
#pragma unroll 2
      for (int i = 0; i < (grp ? 0 : 8); ++i) {
        const int rr = rg * 8 + i;
        float dbv[NH];
#pragma unroll
        for (int h = 0; h < NH; ++h) { dbv[h] = dbp[h * BM + rr]; cdb[h] += dbv[h]; }
#pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
          const uint4 u = *reinterpret_cast<const uint4*>(sX + xb * XT + kh * BM * 64 + rr * 64 + ((j ^ (rr & 7)) << 3));
          const float2 xv[4] = {bf2f(u.x), bf2f(u.y), bf2f(u.z), bf2f(u.w)};
#pragma unroll
          for (int h = 0; h < NH; ++h) {
            const float2 d2 = make_float2(dbv[h], dbv[h]);
#pragma unroll
            for (int e = 0; e < 4; ++e) bh2[h][kh][e] = fma2(xv[e], d2, bh2[h][kh][e]);
          }
        }
      }
      __syncwarp();
      if (lane == 0) arrive(he + xb);
#pragma unroll
      for (int q8 = 0; q8 < 8; ++q8) {               // unrolled: cs2[q8] must stay in registers (a runtime index spills it to local)
        if (q8 >= 2 * TS) continue;
        const int sl = y % NR;
        wait(df + sl, (y / NR) & 1);
        const __nv_bfloat16* dt = sD + sl * HT;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int rr = rg * 8 + i;
          const uint4 u = *reinterpret_cast<const uint4*>(dt + rr * 64 + ((j ^ (rr & 7)) << 3));
          cs2[q8][0] = add2(cs2[q8][0], bf2f(u.x)); cs2[q8][1] = add2(cs2[q8][1], bf2f(u.y));
          cs2[q8][2] = add2(cs2[q8][2], bf2f(u.z)); cs2[q8][3] = add2(cs2[q8][3], bf2f(u.w));
        }
        __syncwarp();
        if (lane == 0) arrive(de + sl);
        ++y;
      }
    }
    // sum the warp's four row groups (lane bits 3, 4) -> lanes 0..7 write the warp's partial to shared; the four warps' partials are
    // summed there and every address gets one coalesced atomic from this CTA
#pragma unroll
    for (int o = 8; o <= 16; o <<= 1) {
#pragma unroll
      for (int a8 = 0; a8 < 8; ++a8)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          cs2[a8][e].x += __shfl_xor_sync(0xffffffffu, cs2[a8][e].x, o); cs2[a8][e].y += __shfl_xor_sync(0xffffffffu, cs2[a8][e].y, o);
        }
#pragma unroll
      for (int h = 0; h < NH; ++h) {
        cdb[h] += __shfl_xor_sync(0xffffffffu, cdb[h], o);
#pragma unroll
        for (int kh = 0; kh < 2; ++kh)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            bh2[h][kh][e].x += __shfl_xor_sync(0xffffffffu, bh2[h][kh][e].x, o); bh2[h][kh][e].y += __shfl_xor_sync(0xffffffffu, bh2[h][kh][e].y, o);
          }
      }
    }
    float* red = sRed + (warp - 6) * 1056;
    if (lane < 8) {
#pragma unroll
      for (int a8 = 0; a8 < 8; ++a8) {
        if (a8 >= 2 * TS) continue;
        float* dst = red + (T0 + (a8 >> 1)) * C + (a8 & 1) * 64 + j * 8;
#pragma unroll
        for (int e = 0; e < 4; ++e) *reinterpret_cast<float2*>(dst + 2 * e) = cs2[a8][e];
      }
#pragma unroll
      for (int h = 0; h < NH; ++h)
#pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
          float* dst = red + 4 * C + h * C + kh * 64 + j * 8;
#pragma unroll
          for (int e = 0; e < 4; ++e) *reinterpret_cast<float2*>(dst + 2 * e) = bh2[h][kh][e];
        }
      if (lane == 0) {
#pragma unroll
        for (int h = 0; h < NH; ++h) red[8 * C + h] = cdb[h];
      }
    }
    named_sync(3, 128);
    for (int i = t; i < NVEC; i += 128) {
      if (i < 4 * C ? (unsigned)(i / C - T0) >= (unsigned)TS : grp != 0) continue;   // only this CTA's sums (the slot's other entries are the other groups')
      const float a = sRed[i] + sRed[1056 + i] + sRed[2 * 1056 + i] + sRed[3 * 1056 + i];
      if (want_finish & 2) VEC[(long)split * NVECP + i] = a;
      else atomicAdd(VEC + i, a);
    }
    // ---- the CTA's M_t: TMEM lane = c, 512 columns -> global ----
    // 128B-swizzled [128 c][32 o] fp32 staging (conflict-free) -> TMA stores of 16 KiB tiles into the CTA's slot (tri_slot_sum adds the
    // slots in CTA order: the same bits every run); the x / d rings are free once the GEMMs retired (fin). A round = one tensor (4
    // tiles, 4 buffers).
    // (round 9, WG_DST) deterministic slots: straight from the registers -- a thread holds row c of M_t, 32 consecutive o a TMEM load
    // = 128 contiguous bytes (whole lines across the warp) -- no staging, no named barriers, no TMA store round trip
    if (WG_DST && (want_finish & 2) && MT != nullptr && mytiles > 0) {
      wait(fin, 0);
      if (t == 0) TLS(1, 4);
      tc_fence_after();
      const int qq = warp & 3, rl = qq * 32 + lane;
#pragma unroll 1
      for (int ten = 0; ten < TS; ++ten) {
        float* dst = MT + ((long)(split * 4 + T0 + ten) * 128 + rl) * C;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          float v[32];
          tmem_ld32(tmem_at(tmem + ten * 128 + q * 32, qq * 32, 0), v);
          tmem_wait_ld();
#pragma unroll
          for (int i = 0; i < 8; ++i)
            reinterpret_cast<float4*>(dst + q * 32)[i] = make_float4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);
        }
      }
      if (t == 0) TLS(1, 5);
    } else if (mytiles > 0) {
      wait(fin, 0);
      if (t == 0) TLS(1, 4);
      tc_fence_after();
      const int qq = warp & 3, rl = qq * 32 + lane;
      const bool lead = t == 0;
      float* stg = reinterpret_cast<float*>(sX);                   // [4][128][32] fp32 = 64 KiB (the x double buffer)
#pragma unroll 1
      for (int r4 = 0; r4 < TS; ++r4) {
        const int ten = (r4 + split) % TS;            // local
        if (r4 > 0) { if (lead) bulk_wait_read<0>(); named_sync(3, 128); }
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          float v[32];
          tmem_ld32(tmem_at(tmem + ten * 128 + q * 32, qq * 32, 0), v);
          tmem_wait_ld();
          float* row = stg + q * 4096 + rl * 32;
#pragma unroll
          for (int i = 0; i < 8; ++i)
            *reinterpret_cast<float4*>(row + ((i ^ (rl & 7)) << 2)) = make_float4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);
        }
        fence_proxy_async();
        named_sync(3, 128);
        if (lead) {
#pragma unroll
          if (want_finish & 2) { for (int q = 0; q < 4; ++q) store_2d(&mtmap, stg + q * 4096, q * 32, (split * 4 + T0 + ten) * 128); }
          else { for (int q = 0; q < 4; ++q) reduce_add_2d_bf(&mtmap, stg + q * 4096, q * 32, (T0 + ten) * 128); }
          bulk_commit();
        }
      }
      if (lead) {
        bulk_wait<0>();
        if (F.bar != nullptr) asm volatile("fence.proxy.async.global;\n" ::: "memory");   // the slot (async-proxy writes) -> other CTAs' loads
        TLS(1, 5);
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) tmem_dealloc(tmem, 512);
  if (F.bar != nullptr && F.mts != nullptr) {
    // ---- (round 10) the slot sums and the parameter-gradient finish (see WFold) ----
    wf_arrive_wait(F.bar, (unsigned)NBLK);
    if (tid == 0) TLS(1, 6);
    float* sc = reinterpret_cast<float*>(sX);                    // scratch: [128] float4 lane-1 partials | the groups' cs
    float4* sc4 = reinterpret_cast<float4*>(sc);
    // this CTA's contiguous range of the 16384 float4 columns j = (t, o8, c, hf) of the M_t sums (an even count: hf pairs together)
    constexpr int NCOL = 4 * 16 * C * 2;
    const int per = ((NCOL + NBLK - 1) / NBLK + 1) & ~1, j0 = min(NCOL, BID * per), j1 = min(NCOL, j0 + per);
    const int g0 = j0 >> 8, ngr = j1 > j0 ? ((j1 - 1) >> 8) - g0 + 1 : 0;   // (t, o8) groups touched: their cs_t[o8 .. + 8]
    if (warp >= 8) {
      // jobs: the groups' cs halves, then B_h float4s BID, BID + NBLK, .. (< 128), then cdb on CTA 0; warps 8 / 9 alternate
      const int w = warp - 8, nbh = BID < 128 ? (127 - BID) / NBLK + 1 : 0, njob = 2 * ngr + nbh + (BID == 0 ? 1 : 0);
      for (int job = w; job < njob; job += 2) {
        long off; float* dst = nullptr;
        if (job < 2 * ngr) { const int gg = g0 + (job >> 1); off = (long)(gg >> 4) * C + (gg & 15) * 8 + (job & 1) * 4; }
        else if (job < 2 * ngr + nbh) { const int b4 = BID + (job - 2 * ngr) * NBLK; off = 4L * C + 4L * b4; dst = F.bhs + 4L * b4; }
        else { off = 8L * C; dst = F.bhs + 4L * C; }
        float4 b = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int s = lane; s < F.nsp; s += 32) add4(b, ldcg4(F.vecs + (long)s * NVECP + off));
#pragma unroll
        for (int x = 16; x >= 1; x >>= 1) {
          b.x += __shfl_xor_sync(0xffffffffu, b.x, x); b.y += __shfl_xor_sync(0xffffffffu, b.y, x);
          b.z += __shfl_xor_sync(0xffffffffu, b.z, x); b.w += __shfl_xor_sync(0xffffffffu, b.w, x);
        }
        if (lane == 0) { if (dst == nullptr) sc4[128 + job] = b; else __stcg(reinterpret_cast<float4*>(dst), b); }
      }
    }
    for (int jb = j0; jb < j1; jb += 128) {
      const int k = tid & 127, sl = tid >> 7, j = jb + k;
      const int t = j >> 12, o8 = (j >> 8) & 15, c = (j >> 1) & (C - 1), hf = j & 1, o = o8 * 8 + hf * 4;
      float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
      if (warp < 8 && j < j1) {                                  // M_t[c][o .. o + 3] over the slots: lane sl sums slots sl, sl + 2, ..
        const float* p = F.mts + ((long)t * C + c) * C + o;
        const long ss = 4L * C * C;
        int s = sl;
        for (; s + 2 * (WF_UN - 1) < F.nsp; s += 2 * WF_UN) {
          float4 v[WF_UN];
#pragma unroll
          for (int q = 0; q < WF_UN; ++q) v[q] = ldcg4(p + (long)(s + 2 * q) * ss);
#pragma unroll
          for (int q = 0; q < WF_UN; ++q) add4(a, v[q]);
        }
        for (; s < F.nsp; s += 2) add4(a, ldcg4(p + (long)s * ss));
      }
      if (warp < 8 && sl == 1) sc4[k] = a;
      __syncthreads();
      if (warp < 4) {
        add4(a, sc4[k]);
        const bool live = j < j1;
        const float4 cs = live ? sc4[128 + ((j >> 8) - g0) * 2 + hf] : make_float4(0.f, 0.f, 0.f, 0.f);
        const float g = live ? F.lnw[c] : 0.f, be = live ? F.lnb[c] : 0.f;
        const float m4[4] = {a.x, a.y, a.z, a.w}, c4v[4] = {cs.x, cs.y, cs.z, cs.w};
        float pg = 0.f, pb = 0.f;
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          const long wi = ((long)t * C + o + jj) * C + c;
          if (live) {
            st_any(F.dw, wi, fmaf(g, m4[jj], be * c4v[jj]), F.f32);
            const float w = __bfloat162float(F.w4[wi]);
            pg = fmaf(w, m4[jj], pg);
            pb = fmaf(w, c4v[jj], pb);
          }
        }
        const float qg = __shfl_xor_sync(0xffffffffu, pg, 1), qb = __shfl_xor_sync(0xffffffffu, pb, 1);
        if (live && hf == 0) {
          __stcg(F.part + (long)(t * 16 + o8) * C + c, pg + qg);
          __stcg(F.part + (long)(64 + t * 16 + o8) * C + c, pb + qb);
        }
      }
      __syncthreads();
    }
    if (tid == 0) TLS(1, 7);
    // ---- the last CTA to arrive: dgamma | dbeta | dWb (and it resets the counters for the next launch) ----
    unsigned* flag = reinterpret_cast<unsigned*>(sc + 1024);
    __syncthreads();
    if (tid == 0) {
      const unsigned tk = wf_ticket(F.bar + 2);
      const unsigned last = tk == (unsigned)NBLK - 1;
      if (last) { F.bar[0] = 0u; F.bar[2] = 0u; TLX(0); }
      *flag = last;
    }
    __syncthreads();
    if (*flag) {
      // 256 threads: plane q (dgamma / dbeta shares), 32 float4 channel columns x 4 lanes of 16 share rows each
      if (tid < 256) {
        const int q = tid >> 7, k = tid & 127, c4 = k & 31, ln = k >> 5;
        float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 v[16];
#pragma unroll
        for (int r = 0; r < 16; ++r) v[r] = ldcg4(F.part + (long)(q * 64 + ln * 16 + r) * C + 4 * c4);
#pragma unroll
        for (int r = 0; r < 16; ++r) add4(a, v[r]);
        sc4[(q * 4 + ln) * 32 + c4] = a;
      }
      __syncthreads();
      if (tid < 256) {
        const int q = tid >> 7, c = tid & 127;
        float s = 0.f;
#pragma unroll
        for (int ln = 0; ln < 4; ++ln) s += sc[((q * 4 + ln) * 32 + (c >> 2)) * 4 + (c & 3)];
        const float* bh = F.bhs; const float* cdb = F.bhs + 4 * C;
        if (q == 0) {
          const float g = F.lnw[c], be = F.lnb[c];
#pragma unroll
          for (int h = 0; h < NH; ++h) {
            const float b = __ldcg(bh + h * C + c);
            s = fmaf(F.wbf[h * C + c], b, s);
            st_any(F.dgb, 2L * C + h * C + c, fmaf(g, b, be * __ldcg(cdb + h)), F.f32);
          }
          st_any(F.dgb, c, s, F.f32);
        } else {
#pragma unroll
          for (int h = 0; h < NH; ++h) s = fmaf(F.wbf[h * C + c], __ldcg(cdb + h), s);
          st_any(F.dgb, C + c, s, F.f32);
        }
      }
      if (tid == 0) TLX(1);
    }
  }
  (void)want_finish; (void)TICKET; (void)W4; (void)LNW; (void)LNB; (void)WBF; (void)DW; (void)DGB;
}

__global__ void __launch_bounds__(wg::THREADS, 1) tri_wgrad_sm100(
    int L, int ntiles, float eps,
    const __grid_constant__ CUtensorMap d0map, const __grid_constant__ CUtensorMap d1map,
    const __grid_constant__ CUtensorMap d2map, const __grid_constant__ CUtensorMap d3map,
    const __grid_constant__ CUtensorMap xmap, const float* __restrict__ DB, float* __restrict__ MT, float* __restrict__ VEC,
    unsigned* __restrict__ TICKET, const __nv_bfloat16* __restrict__ W4, const float* __restrict__ LNW, const float* __restrict__ LNB,
    const float* __restrict__ WBF, __nv_bfloat16* __restrict__ DW, __nv_bfloat16* __restrict__ DGB, const int want_finish,
    const __grid_constant__ CUtensorMap mtmap, const int TRANS, const int NG, const WFold F) {
  wgrad_body((int)blockIdx.x, (int)gridDim.x, L, ntiles, eps, d0map, d1map, d2map, d3map, xmap, DB, MT, VEC, TICKET, W4, LNW, LNB, WBF,
             DW, DGB, want_finish, mtmap, TRANS, NG, F);
}

// (round 9) tri_head_bwd and tri_wgrad in ONE launch, split by CTA: neither reads the other's outputs (both read dq | dk | dv | dg, the
// summed dbias and x), and neither fits beside the other on an SM (head_bwd 226 KiB of shared memory, wgrad all 512 TMEM columns), so
// at L128 -- one tile a CTA, each kernel ~11-16 us of mostly per-CTA latency on 128-148 SMs -- they ran one after the other.  Here CTAs
// 0 .. NH - 1 run head_bwd's tiles (round robin over NH) and CTAs NH .. grid - 1 wgrad's (NG tensor groups over the rest): both chains
// on disjoint SMs at once, and the fewer wgrad CTAs write fewer M_t slots for the slot sum.  dpair: the same bits (a tile's rows are
// written by one CTA whatever the CTA count); the parameter gradients: deterministic (the per-CTA partials summed in CTA order), their
// grouping follows the wgrad CTA count.
__global__ void __launch_bounds__(hb::THREADS, 1) tri_hw_bwd_sm100(
    const int NHC, int L, int ntiles, float eps,
    const __grid_constant__ CUtensorMap d0map, const __grid_constant__ CUtensorMap d1map,
    const __grid_constant__ CUtensorMap d2map, const __grid_constant__ CUtensorMap d3map,
    const __grid_constant__ CUtensorMap xmap, const __grid_constant__ CUtensorMap wmap, const __grid_constant__ CUtensorMap rmap,
    const float* __restrict__ DB, float* __restrict__ MTS, float* __restrict__ VEC, const __grid_constant__ CUtensorMap mtmap, const int TRANS, const int NG, const WFold F) {
  static_assert(hb::THREADS == wg::THREADS, "one block shape");
  if ((int)blockIdx.x < NHC)
    head_bwd_body((int)blockIdx.x, NHC, L, ntiles, eps, d0map, d1map, d2map, d3map, xmap, wmap, rmap, DB, TRANS);
  else
    wgrad_body((int)blockIdx.x - NHC, (int)gridDim.x - NHC, L, ntiles, eps, d0map, d1map, d2map, d3map, xmap, DB, MTS, VEC, nullptr,
               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 2, mtmap, TRANS, NG, F);
}

// The per-CTA partials of tri_gate_bwd (dWo) and tri_wgrad (M_t, the vectors) summed in CTA order (a fixed order: the parameter
// gradients are the same bits every run): out[i] = sum_s slots[s][i] for three (slots, width, out) segments, one launch.
struct SlotSeg { const float* in; float* out; long width, stride; int n; };   // n slots of `width` floats at `stride`
// A block = 32 float4 columns x 16 slot lanes: lane g sums slots g, g + 16, g + 32, ... (in that order, 8 loads in flight), then lane 0
// adds the lane sums in lane order -- a fixed order (the same bits every run) with enough loads in flight to read the partials from L2
// at its bandwidth (B200 2026-11-17: 8 lanes x 4 loads -> 16 x 8: L128 d_pair 128 training 8.1 -> 6.8 us, L256 9.8 -> 8.3).
#ifndef TB_SS_COLS
#define TB_SS_COLS 32
#endif
#ifndef TB_SS_LANES
#define TB_SS_LANES 16
#endif
#ifndef TB_SS_UN
#define TB_SS_UN 8
#endif
constexpr int SS_COLS = TB_SS_COLS, SS_LANES = TB_SS_LANES, SS_UN = TB_SS_UN;
__global__ void __launch_bounds__(SS_COLS * SS_LANES) tri_slot_sum(SlotSeg s0, SlotSeg s1, SlotSeg s2) {
  const int col = threadIdx.x % SS_COLS, lane = threadIdx.x / SS_COLS;
  long i = ((long)blockIdx.x * SS_COLS + col) * 4;
  SlotSeg sg;
  bool live = true;
  if (i < s0.width) sg = s0;
  else if ((i -= s0.width) < s1.width) sg = s1;
  else if ((i -= s1.width) < s2.width) sg = s2;
  else { live = false; sg = s0; i = 0; }
  float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
  if (live) {
    const float4* p = reinterpret_cast<const float4*>(sg.in + i);
    const long st = sg.stride / 4;
    int k = lane;
    for (; k + (SS_UN - 1) * SS_LANES < sg.n; k += SS_UN * SS_LANES) {
      float4 v[SS_UN];
#pragma unroll
      for (int u = 0; u < SS_UN; ++u) v[u] = __ldg(p + (long)(k + u * SS_LANES) * st);
#pragma unroll
      for (int u = 0; u < SS_UN; ++u) { a.x += v[u].x; a.y += v[u].y; a.z += v[u].z; a.w += v[u].w; }
    }
    for (; k < sg.n; k += SS_LANES) { const float4 v = __ldg(p + (long)k * st); a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w; }
  }
  __shared__ float4 part[SS_LANES][SS_COLS];
  part[lane][col] = a;
  __syncthreads();
  if (lane == 0 && live) {
    float4 r = part[0][col];
#pragma unroll
    for (int g = 1; g < SS_LANES; ++g) { const float4 v = part[g][col]; r.x += v.x; r.y += v.y; r.z += v.z; r.w += v.w; }
    *reinterpret_cast<float4*>(sg.out + i) = r;
  }
}

// The parameter gradients from tri_wgrad's sums (the algebra in its header comment), one launch: blocks 0..63 write dW_t[o][c] through
// 32 x 32 shared transposes (M_t is [c][o]); blocks 64..191 one channel c each (dgamma, dbeta, dWb[:, c]), a block-wide reduction.
__device__ __forceinline__ void st_out(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
__device__ __forceinline__ void st_out(float* p, float v) { *p = v; }

// OutT = the parameters' dtype: bf16 parameters get bf16 gradients, fp32 (master) parameters fp32 ones -- no cast kernels after
template <typename OutT>
__global__ void __launch_bounds__(256) tri_param_finish(const float* __restrict__ MT, const float* __restrict__ VEC,
    const __nv_bfloat16* __restrict__ W4, const float* __restrict__ LNW, const float* __restrict__ LNB, const float* __restrict__ WBF,
    OutT* __restrict__ DW, OutT* __restrict__ DGB, const float* __restrict__ DWO, OutT* __restrict__ DWO16) {
  const int tid = threadIdx.x;
  if (blockIdx.x >= 64 + C) {                      // blocks 192..207: gate_bwd's dWo fp32 -> bf16
    const int i0 = (blockIdx.x - 64 - C) * 1024;
    for (int i = tid; i < 1024; i += 256) if (DWO != nullptr) st_out(DWO16 + i0 + i, DWO[i0 + i]);
    return;
  }
  const float* cs = VEC; const float* bh = VEC + 4 * C; const float* cdb = VEC + 8 * C;
  if (blockIdx.x < 64) {
    __shared__ float tile[32][33];
    const int tt = blockIdx.x >> 4, c0 = ((blockIdx.x >> 2) & 3) * 32, o0 = (blockIdx.x & 3) * 32;
    for (int i = tid; i < 32 * 32; i += 256) { const int cc = i >> 5, oo = i & 31; tile[cc][oo] = MT[((long)tt * C + c0 + cc) * C + o0 + oo]; }
    __syncthreads();
    for (int i = tid; i < 32 * 32; i += 256) {
      const int oo = i >> 5, cc = i & 31, c = c0 + cc, o = o0 + oo;
      st_out(DW + ((long)tt * C + o) * C + c, LNW[c] * tile[cc][oo] + LNB[c] * cs[tt * C + o]);
    }
    return;
  }
  const int c = blockIdx.x - 64;
  __shared__ float red[2][8];
  float dgam = 0.f, dbet = 0.f;
  for (int i = tid; i < 4 * C; i += 256) {        // (t, o)
    const int tt = i >> 7, o = i & (C - 1);
    const float w = __bfloat162float(W4[((long)tt * C + o) * C + c]);
    dgam = fmaf(w, MT[((long)tt * C + c) * C + o], dgam);
    dbet = fmaf(w, cs[tt * C + o], dbet);
  }
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1) { dgam += __shfl_xor_sync(0xffffffffu, dgam, o); dbet += __shfl_xor_sync(0xffffffffu, dbet, o); }
  if ((tid & 31) == 0) { red[0][tid >> 5] = dgam; red[1][tid >> 5] = dbet; }
  __syncthreads();
  if (tid == 0) {
    float g = 0.f, b = 0.f;
    for (int w = 0; w < 8; ++w) { g += red[0][w]; b += red[1][w]; }
    for (int h = 0; h < NH; ++h) {
      g = fmaf(WBF[h * C + c], bh[h * C + c], g);
      b = fmaf(WBF[h * C + c], cdb[h], b);
      st_out(DGB + 2 * C + h * C + c, LNW[c] * bh[h * C + c] + LNB[c] * cdb[h]);
    }
    st_out(DGB + c, g);
    st_out(DGB + C + c, b);
  }
}

// ---- parameter pack (one launch per forward): [Wq; Wk; Wv; Wg] -> w4 bf16 [512, 128], Wo -> bf16 [128, 128],
// lnw | lnb | Wb -> fp32 [6, 128] (the layout head_bwd's constant block and every kernel's fp32 parameters use); any float dtype in
__device__ __forceinline__ float ld_any(const void* p, long i, int f32) {
  return f32 ? reinterpret_cast<const float*>(p)[i] : __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(p)[i]);
}
struct PackIn { const void* w[5]; int w32[5]; const void* lnw; const void* lnb; const void* wb; int lnw32, lnb32, wb32; };
__global__ void __launch_bounds__(256) tri_pack(PackIn in, __nv_bfloat16* __restrict__ W4, __nv_bfloat16* __restrict__ WO, float* __restrict__ PRM) {
  const long i = (long)blockIdx.x * 256 + threadIdx.x;
  if (i < 5L * C * C) {
    const int t = (int)(i / (C * C)); const long j = i % (C * C);
    const float v = ld_any(in.w[t], j, in.w32[t]);
    if (t < 4) W4[i] = __float2bfloat16_rn(v); else WO[j] = __float2bfloat16_rn(v);
  } else if (i < 5L * C * C + 6 * C) {
    const int j = (int)(i - 5L * C * C);
    PRM[j] = j < C ? ld_any(in.lnw, j, in.lnw32) : j < 2 * C ? ld_any(in.lnb, j - C, in.lnb32) : ld_any(in.wb, j - 2 * C, in.wb32);
  }
}

}  // namespace

// wq, wk, wv, wg, wo [128, 128], lnw, lnb [128], wb [4, 128] (bf16 or fp32 each) -> (w4 bf16 [512, 128], wo bf16 [128, 128],
// prm fp32 [6, 128] = lnw | lnb | Wb)
std::vector<torch::Tensor> tri_pack_params(torch::Tensor wq, torch::Tensor wk, torch::Tensor wv, torch::Tensor wg, torch::Tensor wo,
                                           torch::Tensor lnw, torch::Tensor lnb, torch::Tensor wb) {
  PackIn in{};
  const torch::Tensor* ws[5] = {&wq, &wk, &wv, &wg, &wo};
  auto ok = [](const torch::Tensor& t, long n) {
    return t.is_cuda() && t.is_contiguous() && t.numel() == n && (t.scalar_type() == torch::kBFloat16 || t.scalar_type() == torch::kFloat32); };
  for (int t = 0; t < 5; ++t) {
    TORCH_CHECK(ok(*ws[t], (long)C * C), "weights: contiguous [128, 128] bf16 / fp32");
    in.w[t] = ws[t]->data_ptr(); in.w32[t] = ws[t]->scalar_type() == torch::kFloat32;
  }
  TORCH_CHECK(ok(lnw, C) && ok(lnb, C) && ok(wb, NH * C), "lnw / lnb [128], wb [4, 128]: contiguous bf16 / fp32");
  in.lnw = lnw.data_ptr(); in.lnb = lnb.data_ptr(); in.wb = wb.data_ptr();
  in.lnw32 = lnw.scalar_type() == torch::kFloat32; in.lnb32 = lnb.scalar_type() == torch::kFloat32; in.wb32 = wb.scalar_type() == torch::kFloat32;
  auto opt = wq.options().dtype(torch::kBFloat16);
  auto w4 = torch::empty({4 * C, C}, opt), wo16 = torch::empty({C, C}, opt), prm = torch::empty({2 + NH, C}, opt.dtype(torch::kFloat32));
  const long n = 5L * C * C + 6 * C;
  tri_pack<<<(unsigned)((n + 255) / 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(in, reinterpret_cast<__nv_bfloat16*>(w4.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(wo16.data_ptr<at::BFloat16>()), prm.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {w4, wo16, prm};
}

// dq, dk, dv, dg [R, 128] bf16; db fp32 [B, 4, L, L]; x [R, 128] (the LN input); w4 [512, 128]; wb [4, 128]; dpair [R, 128] bf16 (the
// seed gate_bwd wrote; dx is added in place)
torch::Tensor tri_head_bwd(torch::Tensor dq, torch::Tensor dk, torch::Tensor dv, torch::Tensor dg, torch::Tensor db, torch::Tensor x,
                  torch::Tensor w4, torch::Tensor wb, torch::Tensor lnw, torch::Tensor lnb, double eps, torch::Tensor dpair, int64_t B, int64_t L,
                  bool trans, int64_t lr, bool consts_ready, torch::Tensor slabs, torch::Tensor smeta, torch::Tensor dwo_slots) {
  const long R = B * L * L;
  const int geo = rows_geo(trans, L, lr);
  const long Lr = geo_rows_len(L, geo), Rr = B * Lr * Lr;   // x and dpair: the caller's [B, Lr, Lr, 128]
  for (const auto* t : {&dq, &dk, &dv, &dg}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16 && t->is_contiguous() && t->numel() == R * C, "[R, 128] bf16 inputs");
  for (const auto* t : {&x, &dpair}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16 && t->is_contiguous() && t->numel() == Rr * C, "x / dpair: [B, Lr, Lr, 128] bf16");
  TORCH_CHECK(db.scalar_type() == torch::kFloat32 && db.is_contiguous() && db.numel() == B * NH * L * L, "db fp32 [B, 4, L, L]");
  TORCH_CHECK(w4.scalar_type() == torch::kBFloat16 && w4.is_contiguous() && w4.size(0) == 4 * C, "w4 [512, 128] bf16");
  const bool packed = lnw.scalar_type() == torch::kFloat32 && lnb.scalar_type() == torch::kFloat32 && wb.scalar_type() == torch::kFloat32 &&
      lnb.data_ptr<float>() == lnw.data_ptr<float>() + C && wb.data_ptr<float>() == lnw.data_ptr<float>() + 2 * C && wb.is_contiguous();
  torch::Tensor prm;
  if (!packed) prm = torch::cat({lnw.to(torch::kFloat32).reshape(-1), lnb.to(torch::kFloat32).reshape(-1), wb.to(torch::kFloat32).reshape(-1)}).contiguous();
  TORCH_CHECK((long)(lnw.numel() + lnb.numel() + wb.numel()) * 4 == (long)sizeof(HbParams), "lnw / lnb / wb sizes");
  auto stream = at::cuda::getCurrentCUDAStream();
  // the constant bank, unless consts_ready (tri_hb_consts already did). (Read from global through the read-only path instead -- the
  // kernel's PRM argument -- head_bwd took 11.1 -> 15.1 us at L128 against the 1.3 us upload: measured and not used, B200 2026-11-17)
  if (!consts_ready)
    C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(c_hb, packed ? lnw.data_ptr<float>() : prm.data_ptr<float>(), sizeof(HbParams), 0,
                                           cudaMemcpyDeviceToDevice, stream));
  auto m2 = [&](const torch::Tensor& t, uint64_t rows, uint32_t br, const char* what) {
    return make_map<2>(t.data_ptr(), {(uint64_t)C, rows}, {(uint64_t)C}, {64, br}, CU_TENSOR_MAP_SWIZZLE_128B, what); };
#if HB_BOX3
  // [2 halves][128 rows][64] in one box: dims (64 cols, rows, 2 halves), half stride 64 elements
  auto m3d = [&](const torch::Tensor& t, const char* what) {
    return make_map<3>(t.data_ptr(), {64, (uint64_t)R, 2}, {(uint64_t)C, 64}, {64, 128, 2}, CU_TENSOR_MAP_SWIZZLE_128B, what); };
  CUtensorMap m0 = m3d(dq, "dq"), m1 = m3d(dk, "dk"), m2_ = m3d(dv, "dv"), m3 = m3d(dg, "dg");
#else
  CUtensorMap m0 = m2(dq, R, 128, "dq"), m1 = m2(dk, R, 128, "dk"), m2_ = m2(dv, R, 128, "dv"), m3 = m2(dg, R, 128, "dg");
#endif
  CUtensorMap xm = make_rows_map(x.data_ptr(), C, R, C, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, "x", L, geo);
  CUtensorMap wm = m2(w4, 4 * C, 256, "w4");
  CUtensorMap rm = make_rows_map(dpair.data_ptr(), C, R, C, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, "dpair", L, geo);
  static bool attr = false;
  // slabs (with smeta = (NG, per), CPU): the attention core's unsummed dbias slabs (triattn_bwd raw_slabs); summed here, the sum written
  // into db for tri_wgrad
  const bool sl = slabs.defined() && slabs.numel() > 0;
  int sng = 0, sper = 1;
  if (sl) {
    TORCH_CHECK(slabs.scalar_type() == torch::kFloat32 && slabs.is_contiguous() && slabs.numel() % (B * NH * L * L) == 0 && smeta.numel() == 2 && !smeta.is_cuda(),
                "slabs fp32 [P, B*4*L*L], smeta CPU (NG, per)");
    sng = (int)smeta[0].item<int64_t>(); sper = (int)smeta[1].item<int64_t>();
  }
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(tri_head_bwd_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, hb::SMEM)); attr = true; }
  const int ntiles = (int)(R / hb::BM);
  // dwo_slots: tri_gate_bwd's dWo partials fp32 [n, 128, 128] (or empty): with fewer tiles than SMs the idle CTAs sum them in slot order
  // -> returned [128, 128] fp32 (else an empty tensor)
  const int sms_ = num_sms(x.device().index());
  const bool wo = dwo_slots.defined() && dwo_slots.numel() > 0 && ntiles < sms_;
  if (wo) TORCH_CHECK(dwo_slots.scalar_type() == torch::kFloat32 && dwo_slots.is_contiguous() && dwo_slots.numel() % (C * C) == 0, "dwo_slots fp32 [n, 128, 128]");
  auto dwo = wo ? torch::empty({C, C}, dwo_slots.options()) : torch::empty({0}, x.options().dtype(torch::kFloat32));
  tri_head_bwd_sm100<<<wo ? sms_ : std::min(ntiles, sms_), hb::THREADS, hb::SMEM, stream>>>(
      (int)L, ntiles, (float)eps, m0, m1, m2_, m3, xm, wm, rm, db.data_ptr<float>(), geo);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  (void)sng; (void)sper;
  return dwo;
}

// head_bwd's constant block (lnw | lnb | Wb) uploaded on the current stream: prm fp32 [6, 128] (tri_pack_params' third output), for a
// tri_head_bwd(consts_ready = true) launched after it (B200 2026-11-17: the upload on a side stream during the attention backward)
void tri_hb_consts(torch::Tensor prm) {
  TORCH_CHECK(prm.is_cuda() && prm.scalar_type() == torch::kFloat32 && prm.is_contiguous() && prm.numel() * 4 == (long)sizeof(HbParams), "prm fp32 [6, 128]");
  C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(c_hb, prm.data_ptr<float>(), sizeof(HbParams), 0, cudaMemcpyDeviceToDevice, at::cuda::getCurrentCUDAStream()));
}

// out = sum_s slots[s] (fp32 [n, ...] -> [...]) in slot order: tri_slot_sum on one segment (the gate backward's dWo partials, summed on a
// side stream while the attention backward runs)
torch::Tensor tri_slot_sum1(torch::Tensor slots) {
  TORCH_CHECK(slots.is_cuda() && slots.scalar_type() == torch::kFloat32 && slots.is_contiguous() && slots.dim() >= 2 && slots.numel() > 0, "slots fp32 [n, ...]");
  const long w = slots.numel() / slots.size(0);
  TORCH_CHECK(w % 4 == 0, "slot width % 4");
  auto out = torch::empty(slots.sizes().slice(1), slots.options());
  const SlotSeg s0{slots.data_ptr<float>(), out.data_ptr<float>(), w, w, (int)slots.size(0)}, z{nullptr, nullptr, 0, 0, 0};
  tri_slot_sum<<<(unsigned)((w / 4 + SS_COLS - 1) / SS_COLS), SS_COLS * SS_LANES, 0, at::cuda::getCurrentCUDAStream()>>>(s0, z, z);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// (round 10) the folded tails' arguments (WFold): bar = a per-device zeroed counter block (self-resetting), ws = the shares / B_h sums
static WFold wfold_args(const torch::Tensor& like, const torch::Tensor& mts, const torch::Tensor& vecs, int nsp, const torch::Tensor& dwo32,
                        torch::Tensor& ws, const torch::Tensor& dwo_out, const void* w4, const float* lw, const float* lb, const float* wbf,
                        const torch::Tensor& dw, const torch::Tensor& dgb, bool f32) {
  static torch::Tensor bar[64];
  const int dev = like.device().index();
  if (!bar[dev].defined()) bar[dev] = torch::zeros({4}, like.options().dtype(torch::kInt32));
  ws = torch::empty({2L * 64 * C + 4L * C + 4}, like.options().dtype(torch::kFloat32));
  const bool wo = dwo32.defined() && dwo32.numel() > 0;
  if (wo) TORCH_CHECK(dwo32.scalar_type() == torch::kFloat32 && dwo32.is_contiguous() && dwo32.numel() % (C * C) == 0, "dwo32: fp32 [n, 128, 128] partials");
  WFold F{};
  F.bar = reinterpret_cast<unsigned*>(bar[dev].data_ptr<int>());
  F.dwo_in = wo ? dwo32.data_ptr<float>() : nullptr;
  F.n_dwo = wo ? (int)(dwo32.numel() / (C * C)) : 0;
  F.dwo_out = wo ? dwo_out.data_ptr() : nullptr;
  F.part = ws.data_ptr<float>();
  F.bhs = F.part + 2L * 64 * C;
  F.mts = mts.data_ptr<float>(); F.vecs = vecs.data_ptr<float>();
  F.w4 = reinterpret_cast<const __nv_bfloat16*>(w4); F.lnw = lw; F.lnb = lb; F.wbf = wbf;
  F.dw = dw.data_ptr(); F.dgb = dgb.data_ptr(); F.f32 = f32 ? 1 : 0; F.nsp = nsp;
  return F;
}

// finish: -> (dW_q|k|v|g [4, 128, 128] bf16 ([o, c] each, the Linear weight layout), dgamma | dbeta | dWb [6, 128] bf16, dWo);
// raw (finish = false): -> (M [4, 128 c, 128 o], cs [4, 128], B [4, 128], cdb [4]) fp32; see tri_wgrad_sm100.
// dwo32: tri_gate_bwd's per-CTA dWo partials [n, 128, 128] fp32 (or empty). Every partial (theirs and this kernel's per-CTA M_t and
// vector sums) is added in CTA order by tri_slot_sum: the parameter gradients are the same bits every run.
std::vector<torch::Tensor> tri_wgrad(torch::Tensor dq, torch::Tensor dk, torch::Tensor dv, torch::Tensor dg, torch::Tensor db, torch::Tensor x,
                                     double eps, int64_t B, int64_t L, torch::Tensor w4, torch::Tensor wb, torch::Tensor lnw, torch::Tensor lnb, bool finish,
                                     torch::Tensor dwo32, torch::Tensor ws, bool fp32_out, bool det, bool trans, int64_t lr, int64_t fold) {
  const long R = B * L * L;
  const int geo = rows_geo(trans, L, lr);
  const long Lr = geo_rows_len(L, geo);
  for (const auto* t : {&dq, &dk, &dv, &dg}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16 && t->is_contiguous() && t->numel() == R * C, "[R, 128] bf16 inputs");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.is_contiguous() && x.numel() == B * Lr * Lr * C, "x: [B, Lr, Lr, 128] bf16");
  TORCH_CHECK(db.scalar_type() == torch::kFloat32 && db.is_contiguous() && db.numel() == B * NH * L * L, "db fp32 [B, 4, L, L]");
  auto opt = x.options().dtype(torch::kFloat32);
  // tensor groups: few row tiles (small L) -> each CTA owns 4 / NG of the projections over a longer run of tiles (fewer, smaller partials)
  const int ntiles = (int)(R / wg::BM), sms = num_sms(x.device().index());
  const int NG = wg::wg_groups(ntiles, sms);
  int nsp = std::min(ntiles, sms / NG);
  const int fmode = det && finish ? (int)fold : 0;   // (round 10) 1: every tail folded in; 2: only the dWo sum (wgrad's idle row warps)
  if (fmode == 1) { const int per = (ntiles + nsp - 1) / nsp; nsp = (ntiles + per - 1) / per; }   // the same tiles a CTA on fewer CTAs: fewer slots
  const int grid = nsp * NG;
  // det: the per-CTA partials (every slot written whole by its CTA: no zeroing), summed into mt / vec by tri_slot_sum.
  // Otherwise ws: an already-zeroed fp32 buffer of >= 4*C*C + 8*C + NH floats (mt | vec) the CTAs reduce into, else fresh zeros.
  const long nws = 4L * C * C + 8 * C + NH;
  const bool use_ws = !det && ws.defined() && ws.numel() >= nws && ws.scalar_type() == torch::kFloat32 && ws.is_contiguous();
  torch::Tensor mts, vecs, mt, vec;
  if (det) {
    mts = torch::empty({(long)nsp * 4 * C * C}, opt); vecs = torch::empty({(long)nsp * wg::NVECP}, opt);
    mt = torch::empty({4, C, C}, opt); vec = torch::empty({wg::NVEC}, opt);
  } else {
    mt = use_ws ? ws.slice(0, 0, 4L * C * C).view({4, C, C}) : torch::zeros({4, C, C}, opt);
    vec = use_ws ? ws.slice(0, 4L * C * C, nws) : torch::zeros({8 * C + NH}, opt);
  }
  static torch::Tensor ticket;
  if (!ticket.defined() || ticket.device() != x.device()) ticket = torch::zeros({1}, x.options().dtype(torch::kInt32));
  auto oopt = finish && fp32_out ? opt : x.options();
  auto dw = torch::empty({4, C, C}, oopt), dgb = torch::empty({2 + NH, C}, oopt);
  torch::Tensor lw, lb, wbf;
  if (finish) {
    TORCH_CHECK(w4.scalar_type() == torch::kBFloat16 && w4.is_contiguous() && w4.numel() == 4 * C * C, "w4 [512, 128] bf16");
    lw = lnw.to(torch::kFloat32).contiguous(); lb = lnb.to(torch::kFloat32).contiguous(); wbf = wb.to(torch::kFloat32).contiguous();
  }
  auto m2 = [&](const torch::Tensor& t, const char* what) {
    return make_map<2>(t.data_ptr(), {(uint64_t)C, (uint64_t)R}, {(uint64_t)C}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B, what); };
  CUtensorMap m0 = m2(dq, "dq"), m1 = m2(dk, "dk"), m2_ = m2(dv, "dv"), m3 = m2(dg, "dg");
  CUtensorMap xm = make_rows_map(x.data_ptr(), C, R, C, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, "x", L, geo);
  CUtensorMap mtm = make_map<2>((det ? mts : mt).data_ptr(), {(uint64_t)C, (uint64_t)(det ? nsp : 1) * 4 * C}, {(uint64_t)C}, {32, 128},
                                CU_TENSOR_MAP_SWIZZLE_128B, "mt", CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4);
  const bool wof = dwo32.defined() && dwo32.numel() > 0;
  torch::Tensor fws, dwo_f;
  WFold F{};
  if (fmode) {
    dwo_f = torch::empty({wof ? C : 0, C}, fp32_out ? opt : x.options());
    F = wfold_args(x, mts, vecs, nsp, dwo32, fws, dwo_f, w4.data_ptr(), lw.data_ptr<float>(), lb.data_ptr<float>(), wbf.data_ptr<float>(), dw, dgb, fp32_out);
    if (fmode == 2) F.mts = nullptr;
  }
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(tri_wgrad_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, wg::SMEM)); attr = true; }
  tri_wgrad_sm100<<<grid, wg::THREADS, wg::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      (int)L, ntiles, (float)eps, m0, m1, m2_, m3, xm, db.data_ptr<float>(), (det ? mts : mt).data_ptr<float>(), (det ? vecs : vec).data_ptr<float>(),
      reinterpret_cast<unsigned*>(ticket.data_ptr<int>()), finish ? reinterpret_cast<const __nv_bfloat16*>(w4.data_ptr<at::BFloat16>()) : nullptr,
      finish ? lw.data_ptr<float>() : nullptr, finish ? lb.data_ptr<float>() : nullptr, finish ? wbf.data_ptr<float>() : nullptr,
      finish && fp32_out ? nullptr : reinterpret_cast<__nv_bfloat16*>(dw.data_ptr<at::BFloat16>()),
      finish && fp32_out ? nullptr : reinterpret_cast<__nv_bfloat16*>(dgb.data_ptr<at::BFloat16>()), det ? 2 : 0, mtm, geo, NG, F);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (fmode == 1) return {dw, dgb, dwo_f};
  if (fmode == 2) dwo32 = torch::empty({0}, opt);
  const bool wo_ = dwo32.defined() && dwo32.numel() > 0;
  if (wo_) TORCH_CHECK(dwo32.scalar_type() == torch::kFloat32 && dwo32.is_contiguous() && dwo32.numel() % (C * C) == 0 && (det || dwo32.numel() == C * C),
                       "dwo32: fp32 [128, 128] (det: [n, 128, 128] partials)");
  torch::Tensor dwo_sum = det ? torch::empty({wo_ ? C : 0, C}, opt) : (wo_ ? dwo32.view({C, C}) : torch::empty({0, C}, opt));
  if (det) {
    const SlotSeg s0{mts.data_ptr<float>(), mt.data_ptr<float>(), 4L * C * C, 4L * C * C, nsp};
    const SlotSeg s1{vecs.data_ptr<float>(), vec.data_ptr<float>(), (long)wg::NVEC, (long)wg::NVECP, nsp};
    const SlotSeg s2{wo_ ? dwo32.data_ptr<float>() : nullptr, dwo_sum.data_ptr<float>(), wo_ ? (long)C * C : 0L, (long)C * C,
                     wo_ ? (int)(dwo32.numel() / (C * C)) : 0};
    const long n = (s0.width + s1.width + s2.width) / 4;     // float4 columns (every width a multiple of 4)
    tri_slot_sum<<<(unsigned)((n + SS_COLS - 1) / SS_COLS), SS_COLS * SS_LANES, 0, at::cuda::getCurrentCUDAStream()>>>(s0, s1, s2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (finish) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* w4p = reinterpret_cast<const __nv_bfloat16*>(w4.data_ptr<at::BFloat16>());
    if (fp32_out) {                                  // fp32 parameters: dWo is the fp32 sum itself
      tri_param_finish<float><<<64 + C, 256, 0, stream>>>(mt.data_ptr<float>(), vec.data_ptr<float>(), w4p, lw.data_ptr<float>(),
          lb.data_ptr<float>(), wbf.data_ptr<float>(), dw.data_ptr<float>(), dgb.data_ptr<float>(), nullptr, nullptr);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return {dw, dgb, fmode == 2 ? dwo_f : dwo_sum};
    }
    auto dwo16 = torch::empty({wo_ ? C : 0, C}, x.options());
    tri_param_finish<__nv_bfloat16><<<64 + C + (wo_ ? 16 : 0), 256, 0, stream>>>(mt.data_ptr<float>(), vec.data_ptr<float>(), w4p,
        lw.data_ptr<float>(), lb.data_ptr<float>(), wbf.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(dw.data_ptr<at::BFloat16>()), reinterpret_cast<__nv_bfloat16*>(dgb.data_ptr<at::BFloat16>()),
        wo_ ? dwo_sum.data_ptr<float>() : nullptr, wo_ ? reinterpret_cast<__nv_bfloat16*>(dwo16.data_ptr<at::BFloat16>()) : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dw, dgb, fmode == 2 ? dwo_f : dwo16};
  }
  return {mt, vec.slice(0, 0, 4 * C).view({4, C}), vec.slice(0, 4 * C, 8 * C).view({NH, C}), vec.slice(0, 8 * C, 8 * C + NH)};
}

// (round 9) tri_head_bwd + tri_wgrad (deterministic, finished) as one split launch (tri_hw_bwd_sm100), then the slot sum and the finish:
// -> (dW_q|k|v|g, dgamma | dbeta | dWb, dWo) as tri_wgrad(finish = true, det = true); dx added into dpair in place as tri_head_bwd.
// nhc: head_bwd's CTAs (0: by the tile count); nwc: wgrad's CTAs (0: the rest of the SMs, a multiple of its tensor groups).
std::vector<torch::Tensor> tri_hw_bwd(torch::Tensor dq, torch::Tensor dk, torch::Tensor dv, torch::Tensor dg, torch::Tensor db, torch::Tensor x,
                                      torch::Tensor w4, torch::Tensor prm, double eps, torch::Tensor dpair, int64_t B, int64_t L, bool trans,
                                      int64_t lr, torch::Tensor dwo32, bool fp32_out, int64_t nhc, int64_t nwc, int64_t ng, int64_t fold) {
  const long R = B * L * L;
  const int geo = rows_geo(trans, L, lr);
  const long Lr = geo_rows_len(L, geo), Rr = B * Lr * Lr;
  for (const auto* t : {&dq, &dk, &dv, &dg}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16 && t->is_contiguous() && t->numel() == R * C, "[R, 128] bf16 inputs");
  for (const auto* t : {&x, &dpair}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16 && t->is_contiguous() && t->numel() == Rr * C, "x / dpair: [B, Lr, Lr, 128] bf16");
  TORCH_CHECK(db.scalar_type() == torch::kFloat32 && db.is_contiguous() && db.numel() == B * NH * L * L, "db fp32 [B, 4, L, L]");
  TORCH_CHECK(w4.scalar_type() == torch::kBFloat16 && w4.is_contiguous() && w4.numel() == 4 * C * C, "w4 [512, 128] bf16");
  TORCH_CHECK(prm.is_cuda() && prm.scalar_type() == torch::kFloat32 && prm.is_contiguous() && prm.numel() * 4 == (long)sizeof(HbParams), "prm fp32 [6, 128]");
  TORCH_CHECK(!HB_BOX3, "tri_hw_bwd: the 2-D d maps (HB_BOX3 is head_bwd-only)");
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(c_hb, prm.data_ptr<float>(), sizeof(HbParams), 0, cudaMemcpyDeviceToDevice, stream));
  auto opt = x.options().dtype(torch::kFloat32);
  const int ntiles = (int)(R / hb::BM), sms = num_sms(x.device().index());
  const int NG = ng > 0 ? (int)ng : 2;
  int nh = nhc > 0 ? (int)nhc : std::max(1, std::min(ntiles, sms / 2));
  nh = std::min(nh, ntiles);
  int nw = nwc > 0 ? (int)nwc : sms - nh;
  int nsp = std::max(1, std::min(ntiles, nw / NG));
  if (fold == 1) { const int per = (ntiles + nsp - 1) / nsp; nsp = (ntiles + per - 1) / per; }
  nw = nsp * NG;
  auto mts = torch::empty({(long)nsp * 4 * C * C}, opt), vecs = torch::empty({(long)nsp * wg::NVECP}, opt);
  auto mt = torch::empty({4, C, C}, opt), vec = torch::empty({wg::NVEC}, opt);
  auto m2 = [&](const torch::Tensor& t, uint64_t rows, uint32_t br, const char* what) {
    return make_map<2>(t.data_ptr(), {(uint64_t)C, rows}, {(uint64_t)C}, {64, br}, CU_TENSOR_MAP_SWIZZLE_128B, what); };
  CUtensorMap m0 = m2(dq, R, 128, "dq"), m1 = m2(dk, R, 128, "dk"), m2_ = m2(dv, R, 128, "dv"), m3 = m2(dg, R, 128, "dg");
  CUtensorMap xm = make_rows_map(x.data_ptr(), C, R, C, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, "x", L, geo);
  CUtensorMap wm = m2(w4, 4 * C, 256, "w4");
  CUtensorMap rm = make_rows_map(dpair.data_ptr(), C, R, C, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, "dpair", L, geo);
  CUtensorMap mtm = make_map<2>(mts.data_ptr(), {(uint64_t)C, (uint64_t)nsp * 4 * C}, {(uint64_t)C}, {32, 128}, CU_TENSOR_MAP_SWIZZLE_128B, "mt",
                                CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4);
  torch::Tensor fws, dwo_f, fdw, fdgb;
  WFold F{};
  if (fold) {
    const bool wo = dwo32.defined() && dwo32.numel() > 0;
    auto oo = fp32_out ? opt : x.options();
    dwo_f = torch::empty({wo ? C : 0, C}, oo); fdw = torch::empty({4, C, C}, oo); fdgb = torch::empty({2 + NH, C}, oo);
    const float* lw_ = prm.data_ptr<float>();
    F = wfold_args(x, mts, vecs, nsp, dwo32, fws, dwo_f, w4.data_ptr(), lw_, lw_ + C, lw_ + 2 * C, fdw, fdgb, fp32_out);
    if (fold == 2) F.mts = nullptr;
  }
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(tri_hw_bwd_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, hb::SMEM)); attr = true; }
  tri_hw_bwd_sm100<<<nh + nw, hb::THREADS, hb::SMEM, stream>>>(nh, (int)L, ntiles, (float)eps, m0, m1, m2_, m3, xm, wm, rm,
      db.data_ptr<float>(), mts.data_ptr<float>(), vecs.data_ptr<float>(), mtm, geo, NG, F);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (fold == 1) return {fdw, fdgb, dwo_f};
  if (fold == 2) dwo32 = torch::empty({0}, opt);
  const bool wo_ = dwo32.defined() && dwo32.numel() > 0;
  if (wo_) TORCH_CHECK(dwo32.scalar_type() == torch::kFloat32 && dwo32.is_contiguous() && dwo32.numel() % (C * C) == 0, "dwo32: fp32 [n, 128, 128] partials");
  auto dwo_sum = torch::empty({wo_ ? C : 0, C}, opt);
  {
    const SlotSeg s0{mts.data_ptr<float>(), mt.data_ptr<float>(), 4L * C * C, 4L * C * C, nsp};
    const SlotSeg s1{vecs.data_ptr<float>(), vec.data_ptr<float>(), (long)wg::NVEC, (long)wg::NVECP, nsp};
    const SlotSeg s2{wo_ ? dwo32.data_ptr<float>() : nullptr, dwo_sum.data_ptr<float>(), wo_ ? (long)C * C : 0L, (long)C * C,
                     wo_ ? (int)(dwo32.numel() / (C * C)) : 0};
    const long n = (s0.width + s1.width + s2.width) / 4;
    tri_slot_sum<<<(unsigned)((n + SS_COLS - 1) / SS_COLS), SS_COLS * SS_LANES, 0, stream>>>(s0, s1, s2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const float* lw = prm.data_ptr<float>(); const float* lb = lw + C; const float* wbf = lw + 2 * C;
  const auto* w4p = reinterpret_cast<const __nv_bfloat16*>(w4.data_ptr<at::BFloat16>());
  auto oopt = fp32_out ? opt : x.options();
  auto dw = torch::empty({4, C, C}, oopt), dgb = torch::empty({2 + NH, C}, oopt);
  if (fp32_out) {
    tri_param_finish<float><<<64 + C, 256, 0, stream>>>(mt.data_ptr<float>(), vec.data_ptr<float>(), w4p, lw, lb, wbf, dw.data_ptr<float>(),
                                                        dgb.data_ptr<float>(), nullptr, nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dw, dgb, fold == 2 ? dwo_f : dwo_sum};
  }
  auto dwo16 = torch::empty({wo_ ? C : 0, C}, x.options());
  tri_param_finish<__nv_bfloat16><<<64 + C + (wo_ ? 16 : 0), 256, 0, stream>>>(mt.data_ptr<float>(), vec.data_ptr<float>(), w4p, lw, lb, wbf,
      reinterpret_cast<__nv_bfloat16*>(dw.data_ptr<at::BFloat16>()), reinterpret_cast<__nv_bfloat16*>(dgb.data_ptr<at::BFloat16>()),
      wo_ ? dwo_sum.data_ptr<float>() : nullptr, wo_ ? reinterpret_cast<__nv_bfloat16*>(dwo16.data_ptr<at::BFloat16>()) : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dw, dgb, fold == 2 ? dwo_f : dwo16};
}

// dy, g, o [R, 128] bf16 (rows (b, i, j)), wo [128, 128] -> (do, dg [R, 128] bf16, delta fp32 [B, L, 4, L], dWo partials fp32 [n, 128, 128]
// (one per CTA; tri_wgrad adds them in CTA order and finishes),
// dpair [R, 128] bf16 = dy: the seed head_bwd adds dx into)
std::vector<torch::Tensor> tri_gate_bwd(torch::Tensor dy, torch::Tensor g, torch::Tensor o, torch::Tensor wo, int64_t B, int64_t L,
                                        torch::Tensor dwo_ws, torch::Tensor drop, bool det, bool trans, int64_t lr, torch::Tensor bias) {
  for (const auto* t : {&g, &o}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16 && t->is_contiguous() && t->numel() == B * L * L * C, "g / o: [R, 128] bf16");
  const long R = B * L * L;
  const int geo = rows_geo(trans, L, lr);
  const long Lr = geo_rows_len(L, geo), Rr = B * Lr * Lr;   // dy and the dx seed: the caller's [B, Lr, Lr, 128]
  TORCH_CHECK(dy.is_cuda() && dy.scalar_type() == torch::kBFloat16 && dy.is_contiguous() && dy.numel() == Rr * C, "dy: [B, Lr, Lr, 128] bf16");
  TORCH_CHECK(L % gb::BM == 0, "L % 128");
  if (drop.defined() && drop.numel() > 0)
    TORCH_CHECK(drop.scalar_type() == torch::kBFloat16 && drop.is_contiguous() && drop.numel() == B * Lr * C, "drop: [B, Lr, 128] bf16 contiguous");
  auto opt = dy.options();
  auto dO = torch::empty({R, C}, opt), dG = torch::empty({R, C}, opt);
  auto delta = torch::empty({B, L, 4, L}, opt.dtype(torch::kFloat32));
  // det: the per-CTA dWo partials (each slot written whole by its CTA: no zeroing); otherwise dwo_ws: an already-zeroed fp32 [C * C]
  // accumulator (one memset with the other gradient accumulators), else fresh zeros
  const int ntiles_ = (int)(R / gb::BM), grid_ = std::min(ntiles_, num_sms(dy.device().index()));
  auto dwo = det ? torch::empty({grid_, C, C}, opt.dtype(torch::kFloat32))
           : (dwo_ws.defined() && dwo_ws.numel() == C * C && dwo_ws.scalar_type() == torch::kFloat32) ? dwo_ws.view({C, C})
                                                                                                    : torch::zeros({C, C}, opt.dtype(torch::kFloat32));
  auto dwo16 = torch::empty({C, C}, opt), seed = torch::empty({Rr, C}, opt);
  static torch::Tensor ticket;
  if (!ticket.defined() || ticket.device() != dy.device()) ticket = torch::zeros({1}, opt.dtype(torch::kInt32));
  auto m2 = [&](const torch::Tensor& t, uint64_t rows, const char* what) {
    return make_map<2>(t.data_ptr(), {(uint64_t)C, rows}, {(uint64_t)C}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B, what); };
  CUtensorMap gm = m2(g, R, "g"), om = m2(o, R, "o"), wm = m2(wo, C, "wo"), dom = m2(dO, R, "do"), dgm = m2(dG, R, "dg");
  // dy and the seed of dx (= dy, the residual's share) in the caller's layout
  CUtensorMap dym = make_rows_map(dy.data_ptr(), C, R, C, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, "dy", L, geo);
  CUtensorMap sm_ = make_rows_map(seed.data_ptr(), C, R, C, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, "seed", L, geo);
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(tri_gate_bwd_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, gb::SMEM)); attr = true; }
  // bias (bf16 [B, 4, L, L], or empty): with fewer tiles than SMs the idle CTAs make the core's bias^T * log2e (returned; else empty)
  const int sms_ = num_sms(dy.device().index());
  const bool bt = bias.defined() && bias.numel() == B * NH * L * L && bias.scalar_type() == torch::kBFloat16 && bias.is_contiguous() &&
                  ntiles_ < sms_ && L % 32 == 0;
  auto bT = bt ? torch::empty({B, NH, L, L}, opt.dtype(torch::kFloat32)) : torch::empty({0}, opt.dtype(torch::kFloat32));
  tri_gate_bwd_sm100<<<bt ? sms_ : grid_, gb::THREADS, gb::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      (int)L, ntiles_, dym, gm, om, wm, dom, dgm, sm_, delta.data_ptr<float>(), dwo.data_ptr<float>(), reinterpret_cast<unsigned*>(ticket.data_ptr<int>()),
      reinterpret_cast<__nv_bfloat16*>(dwo16.data_ptr<at::BFloat16>()),
      drop.defined() && drop.numel() > 0 ? reinterpret_cast<const __nv_bfloat16*>(drop.data_ptr<at::BFloat16>()) : nullptr, det ? 1 : 0,
      geo,
      bt ? reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr<at::BFloat16>()) : nullptr, bt ? bT.data_ptr<float>() : nullptr, (int)L,
      (int)(B * NH * (L / 32) * (L / 32)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dO, dG, delta, dwo, seed, bT};
}

// out = x + (sigmoid(g) o o) . Wo^T; g, o, x [R, 128] bf16, wo [128, 128] bf16; out_: a contiguous [R, 128] bf16 tensor (or a view of
// one, e.g. a chunk's rows of a batch's output) to write the result into, or empty (a fresh tensor)
torch::Tensor tri_tail(torch::Tensor g, torch::Tensor o, torch::Tensor x, torch::Tensor wo, torch::Tensor drop, int64_t L, torch::Tensor out_,
                       bool trans, int64_t lr) {
  for (const auto* t : {&g, &o}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == torch::kBFloat16 && t->is_contiguous() && t->numel() == g.numel(), "g / o: [R, 128] bf16");
  const long R = g.numel() / C;
  TORCH_CHECK(R % tl::BM == 0 && R % (L * L) == 0, "rows % 128, rows = B L^2");
  const int geo = rows_geo(trans, L, lr);
  const long Lr = geo_rows_len(L, geo), Rr = R / (L * L) * Lr * Lr;   // x and the output: the caller's [B, Lr, Lr, 128]
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.is_contiguous() && x.numel() == Rr * C, "x: [B, Lr, Lr, 128] bf16");
  TORCH_CHECK(wo.scalar_type() == torch::kBFloat16 && wo.is_contiguous() && wo.numel() == C * C, "wo [128, 128] bf16");
  const bool given = out_.defined() && out_.numel() > 0;
  if (given) TORCH_CHECK(out_.scalar_type() == torch::kBFloat16 && out_.is_contiguous() && out_.numel() == Rr * C && out_.device() == g.device(),
                         "out: contiguous bf16 with B Lr^2 x 128 elements");
  auto out = given ? out_.view({Rr, C}) : torch::empty({Rr, C}, g.options());
  auto m2 = [&](const torch::Tensor& t, uint64_t rows, const char* what) {
    return make_map<2>(t.data_ptr(), {(uint64_t)C, rows}, {(uint64_t)C}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B, what); };
  CUtensorMap gm = m2(g, R, "g"), om = m2(o, R, "o"), wm = m2(wo, C, "wo");
  // x and the output in the caller's layout (the ending node's [B, L, L, 128] read and written through 4-D maps)
  CUtensorMap xm = make_rows_map(x.data_ptr(), C, R, C, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, "x", L, geo);
  CUtensorMap ym = make_rows_map(out.data_ptr(), C, R, C, 32, 128, CU_TENSOR_MAP_SWIZZLE_64B, "out", L, geo);
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(tri_tail_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, tl::SMEM)); attr = true; }
  const int ntiles = (int)(R / tl::BM);
  const bool dr = drop.defined() && drop.numel() > 0;
  if (dr) TORCH_CHECK(drop.scalar_type() == torch::kBFloat16 && drop.is_contiguous() && drop.numel() == R / (L * L) * Lr * C,
                      "drop: [B, Lr, 128] bf16 contiguous");
  launch_pdl(tri_tail_sm100, dim3(std::min(ntiles, num_sms(g.device().index()))), dim3(tl::THREADS), tl::SMEM, at::cuda::getCurrentCUDAStream(), ntiles, (int)L,
      dr ? reinterpret_cast<const __nv_bfloat16*>(drop.data_ptr<at::BFloat16>()) : nullptr, gm, om, xm, wm, ym, geo);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// out = x + (sigmoid(LN(x) . Wg^T [+ gb]) o o) . Wo^T [+ bo]: o [R, 128] bf16 (the kernels' rows); x the caller's [B, Lr, Lr, 128] bf16; wo, wg
// [128, 128] bf16 (wg: rows 3C.. of the front's w4); lnw / lnb fp32 [128]; gb / ob fp32 [128] or empty; out_ as in tri_tail
torch::Tensor tri_tailg(torch::Tensor o, torch::Tensor x, torch::Tensor wo, torch::Tensor wg, torch::Tensor lnw, torch::Tensor lnb, double eps,
                        int64_t L, torch::Tensor out_, bool trans, int64_t lr, torch::Tensor gb, torch::Tensor ob) {
  return tailg::launch<C, C>(o, x, wo, wg, lnw, lnb, eps, L, out_, trans, lr, gb, ob);
}

// the same tail with o a channel-major plane [128, R] bf16 (the bias-only chain's PV output, frame layout)
torch::Tensor tri_tailg_pl(torch::Tensor o, torch::Tensor x, torch::Tensor wo, torch::Tensor wg, torch::Tensor lnw, torch::Tensor lnb, double eps,
                           int64_t L, torch::Tensor out_, bool trans, int64_t lr) {
  return tailg::launch<C, C, 1>(o, x, wo, wg, lnw, lnb, eps, L, out_, trans, lr, torch::Tensor(), torch::Tensor());
}

// the front over the kernels' rows R = B L^2: x the caller's [B, Lr, Lr, CX] (geometry geo); W [4 HD (+ the bias rows), CX] bf16 (rows
// 0 .. 4 HD - 1 = [Wq; Wk; Wv; Wg]); the bias rows fp32 (wb) or bf16 (wb16), NHB of them -> (q, k, v, g [R, HD], bias [B, NHB, L, L])
template <int CX, int HD>
static std::vector<torch::Tensor> launch_front(const torch::Tensor& x, const torch::Tensor& lw, const torch::Tensor& lb, double eps,
                                               const torch::Tensor& W, const float* wb, const __nv_bfloat16* wb16, int NHB,
                                               const torch::Tensor& mask, long B, long L, int geo, bool bt, bool g3, float2* stats, int v1 = 0, bool vpl = false) {
  using T = fr::Cfg<CX, HD>;
  const long R = B * L * L, Lr = geo_rows_len(L, geo);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.is_contiguous() && x.numel() == B * Lr * Lr * CX && L % T::BM == 0,
              "x: the caller's B Lr^2 rows of CX, L % 128 == 0");
  TORCH_CHECK(W.scalar_type() == torch::kBFloat16 && W.is_contiguous() && W.size(0) >= 4 * HD && W.size(1) == CX, "W [>= 4 HD, CX] bf16");
  TORCH_CHECK(NHB >= 1 && NHB <= NH, "1 .. 4 bias heads");
  auto opt = x.options();
  auto q = torch::empty({vpl ? 0 : R, HD}, opt), k = torch::empty({vpl ? 0 : R, HD}, opt), v = vpl ? torch::empty({HD, R}, opt) : torch::empty({R, HD}, opt), g = torch::empty({g3 || vpl ? 0 : R, HD}, opt);
  auto bias = torch::empty({B, NHB, L, L}, opt);
  const bool* mp = nullptr;
  torch::Tensor mk;
  if (mask.numel() > 0) { mk = mask.to(torch::kBool).contiguous(); TORCH_CHECK(mk.numel() == B * Lr, "mask [B, Lr]"); mp = mk.data_ptr<bool>(); }
  constexpr uint32_t WR = 4 * HD < 256 ? 4 * HD : 256;
  CUtensorMap xm = make_rows_map(x.data_ptr(), CX, R, CX, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, "x", L, geo);
  CUtensorMap wm = make_map<2>(W.data_ptr(), {(uint64_t)CX, (uint64_t)(4 * HD)}, {(uint64_t)CX}, {64, WR}, CU_TENSOR_MAP_SWIZZLE_128B, "W");
  auto o2 = [&](const torch::Tensor& t, const char* what) {
    return make_map<2>(t.data_ptr(), {(uint64_t)HD, (uint64_t)R}, {(uint64_t)HD}, {32, 32}, CU_TENSOR_MAP_SWIZZLE_64B, what); };
  CUtensorMap vm = vpl ? make_map<2>(v.data_ptr(), {(uint64_t)R, (uint64_t)HD}, {(uint64_t)R}, {32, 32}, CU_TENSOR_MAP_SWIZZLE_NONE, "vplane") : o2(v, "v");
  CUtensorMap qm = vpl ? vm : o2(q, "q"), km = vpl ? vm : o2(k, "k"), gm = vpl ? vm : o2(g3 ? v : g, "g");
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(tri_front_sm100<CX, HD, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM));
    C10_CUDA_CHECK(cudaFuncSetAttribute(tri_front_sm100<CX, HD, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM));
    C10_CUDA_CHECK(cudaFuncSetAttribute(tri_front_sm100<CX, HD, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM));
    attr = true;
  }
  const int ntiles = (int)(R / T::BM);
  (vpl ? tri_front_sm100<CX, HD, 1> : g3 ? tri_front_sm100<CX, HD, 3> : tri_front_sm100<CX, HD, 4>)<<<std::min(ntiles, num_sms(x.device().index())), T::THREADS, T::SMEM,
                                                                    at::cuda::getCurrentCUDAStream()>>>(
      (int)R, (int)L, ntiles, (float)eps, xm, wm, qm, km, vm, gm, lw.data_ptr<float>(), lb.data_ptr<float>(), wb, wb16, NHB, mp,
      reinterpret_cast<__nv_bfloat16*>(bias.data_ptr<at::BFloat16>()), stats, geo, bt ? 1 : 0, (int)Lr, v1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, k, v, g, bias};
}

// x [R, 128] bf16 (R = B * L * L rows (b, j, k)); w4 = [Wq; Wk; Wv; Wg] [512, 128] bf16; wb [4, 128]; mask [B, L] bool or empty
// -> (q, k, v, g [R, 128] bf16, bias [B, 4, L, L] bf16 with masked keys at -inf)
// g3: q, k, v only (g comes back empty): the inference tail tri_tailg projects the gate from x
std::vector<torch::Tensor> tri_front(torch::Tensor x, torch::Tensor lnw, torch::Tensor lnb, double eps, torch::Tensor w4, torch::Tensor wb,
                                     torch::Tensor mask, int64_t B, int64_t L, bool trans, bool bt, int64_t lr, bool g3, bool vpl) {
  TORCH_CHECK(w4.size(0) == 4 * C, "w4 [512, 128] bf16");
  auto lw = lnw.to(torch::kFloat32).contiguous(), lb = lnb.to(torch::kFloat32).contiguous(), wbf = wb.to(torch::kFloat32).contiguous();
  TORCH_CHECK(wbf.numel() == NH * C, "wb [4, 128]");
  return launch_front<C, C>(x, lw, lb, eps, w4, wbf.data_ptr<float>(), nullptr, NH, mask, B, L, rows_geo(trans, L, lr), bt, g3, nullptr, 0, vpl);
}

// the wide path's d_pair 64 rows (heads x 32 = HDp 64 / 128, <= 4 heads) on this front: x the caller's [B, Lr, Lr, 64]; W = the packed
// [Wq; Wk; Wv; Wg; Wb (16 rows)] [4 HDp + 16, 64] bf16 (its rows 4 HDp .. + NH the bias); stats fp32 [R, 2] (mean, rstd) or empty;
// nog: q, k, v only (the inference tail tri_wtailg projects the gate) -> (q, k, v, g [R, HDp], bias [B, NH, L, L])
std::vector<torch::Tensor> tri_front64(torch::Tensor x, torch::Tensor lnw, torch::Tensor lnb, double eps, torch::Tensor W, int64_t NHB,
                                       torch::Tensor mask, int64_t B, int64_t L, torch::Tensor stats, bool trans, bool bt, int64_t lr, bool nog, bool v1) {
  TORCH_CHECK(W.dim() == 2 && W.size(1) == 64 && (W.size(0) - 16) % 4 == 0, "W [4 HDp + 16, 64]");
  const long HDp = (W.size(0) - 16) / 4;
  auto lw = lnw.to(torch::kFloat32).contiguous(), lb = lnb.to(torch::kFloat32).contiguous();
  float2* stp = nullptr;
  if (stats.numel() > 0) {
    TORCH_CHECK(stats.scalar_type() == torch::kFloat32 && stats.is_contiguous() && stats.numel() == 2 * B * L * L, "stats: fp32 [R, 2]");
    stp = reinterpret_cast<float2*>(stats.data_ptr<float>());
  }
  const auto* wb16 = reinterpret_cast<const __nv_bfloat16*>(W.data_ptr<at::BFloat16>()) + 4 * HDp * 64;
  const int geo = rows_geo(trans, L, lr);
  if (HDp == 64) return launch_front<64, 64>(x, lw, lb, eps, W, nullptr, wb16, (int)NHB, mask, B, L, geo, bt, nog, stp, v1);
  if (HDp == 128) return launch_front<64, 128>(x, lw, lb, eps, W, nullptr, wb16, (int)NHB, mask, B, L, geo, bt, nog, stp, v1);
  TORCH_CHECK(false, "tri_front64: heads x 32 in (64, 128)");
  return {};
}


// B200 N2: the tail's per-CTA %globaltimer stamps (TG_STAMP builds): [4][512] int64 = start, end, rank, tiles
torch::Tensor tri_tail_stamps() {
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  auto t = torch::empty({4, 512}, torch::dtype(torch::kInt64).device(torch::kCPU));
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(t.data_ptr(), tailg::g_tg_stamp, 4 * 512 * sizeof(long long)));
  return t;
}
// B200 N2: the tail's per-tile stage stamps (TG_TL builds, CTA 10): [32][16] int64
torch::Tensor tri_tail_tl() {
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  auto t = torch::empty({32, 16}, torch::dtype(torch::kInt64).device(torch::kCPU));
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(t.data_ptr(), tailg::g_tg_tl, 32 * 16 * sizeof(long long)));
  return t;
}
torch::Tensor tri_front_tl() {
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  auto t = torch::empty({32, 16}, torch::dtype(torch::kInt64).device(torch::kCPU));
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(t.data_ptr(), g_fr_tl, 32 * 16 * sizeof(long long)));
  return t;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tri_head_bwd", &tri_head_bwd, "sm100 TriangleAttention head backward: projection dgrad + LN backward + residual (added into dpair in place)",
        py::arg("dq"), py::arg("dk"), py::arg("dv"), py::arg("dg"), py::arg("db"), py::arg("x"), py::arg("w4"), py::arg("wb"), py::arg("lnw"), py::arg("lnb"),
        py::arg("eps"), py::arg("dpair"), py::arg("B"), py::arg("L"), py::arg("trans"), py::arg("lr") = 0, py::arg("consts_ready") = false, py::arg("slabs") = torch::Tensor(), py::arg("smeta") = torch::Tensor(),
        py::arg("dwo_slots") = torch::Tensor());
  m.def("tri_hb_consts", &tri_hb_consts, "upload tri_head_bwd's constant block (prm fp32 [6, 128]) on the current stream");
  m.def("tri_slot_sum1", &tri_slot_sum1, "sum of fp32 slots [n, ...] in slot order");
  m.def("tri_wgrad", &tri_wgrad, "sm100 TriangleAttention parameter gradients: finish -> (dW_q|k|v|g [4,128,128], dgamma|dbeta|dWb [6,128]) bf16; raw -> (M, cs, B, cdb)",
        py::arg("dq"), py::arg("dk"), py::arg("dv"), py::arg("dg"), py::arg("db"), py::arg("x"), py::arg("eps"), py::arg("B"), py::arg("L"),
        py::arg("w4"), py::arg("wb"), py::arg("lnw"), py::arg("lnb"), py::arg("finish") = true, py::arg("dwo32") = torch::Tensor(),
        py::arg("ws") = torch::Tensor(), py::arg("fp32_out") = false, py::arg("det") = false, py::arg("trans") = false, py::arg("lr") = 0, py::arg("fold") = 0);
  m.def("tri_hw_bwd", &tri_hw_bwd, "tri_head_bwd + deterministic tri_wgrad in one CTA-split launch (+ slot sum, finish) -> (dW, dgb, dWo)");
  m.def("tri_gate_bwd", &tri_gate_bwd, "sm100 TriangleAttention gate backward: (do, dg, delta, dWo)", py::arg("dy"), py::arg("g"), py::arg("o"), py::arg("wo"), py::arg("B"), py::arg("L"),
        py::arg("dwo_ws"), py::arg("drop"), py::arg("det") = false, py::arg("trans") = false, py::arg("lr") = 0, py::arg("bias") = torch::Tensor());
  m.def("tri_tail", &tri_tail, "sm100 TriangleAttention tail: x + (sigmoid(g) o o) . Wo^T", py::arg("g"), py::arg("o"), py::arg("x"), py::arg("wo"), py::arg("drop"), py::arg("L"),
        py::arg("out") = torch::Tensor(), py::arg("trans") = false, py::arg("lr") = 0);
  m.def("tri_pack_params", &tri_pack_params, "one-launch parameter pack: (w4 bf16 [512,128], wo bf16 [128,128], lnw|lnb|Wb fp32 [6,128])");
  m.def("tri_front", &tri_front, "sm100 TriangleAttention front: LN + q|k|v|g projections + head-major (masked) bias",
        py::arg("x"), py::arg("lnw"), py::arg("lnb"), py::arg("eps"), py::arg("w4"), py::arg("wb"), py::arg("mask"), py::arg("B"), py::arg("L"),
        py::arg("trans") = false, py::arg("bt") = false, py::arg("lr") = 0, py::arg("g3") = false, py::arg("vpl") = false);
  m.def("tri_front64", &tri_front64, "the front at d_pair 64 (heads x 32 = 64 / 128, <= 4 heads) for the wide path",
        py::arg("x"), py::arg("lnw"), py::arg("lnb"), py::arg("eps"), py::arg("W"), py::arg("NH"), py::arg("mask"), py::arg("B"), py::arg("L"),
        py::arg("stats"), py::arg("trans"), py::arg("bt"), py::arg("lr"), py::arg("nog"), py::arg("v1") = false);
  m.def("tri_tail_stamps", &tri_tail_stamps, "TG_STAMP: per-CTA tail stamps");
  m.def("tri_tail_tl", &tri_tail_tl, "TG_TL: per-tile tail stage stamps");
  m.def("tri_front_tl", &tri_front_tl, "FR_TL: per-tile front stage stamps");
  m.def("tri_tailg_pl", &tri_tailg_pl, "tri_tailg with o a channel-major plane [128, R]",
        py::arg("o"), py::arg("x"), py::arg("wo"), py::arg("wg"), py::arg("lnw"), py::arg("lnb"), py::arg("eps"), py::arg("L"), py::arg("out"),
        py::arg("trans"), py::arg("lr"));
  m.def("tri_tailg", &tri_tailg, "sm100 TriangleAttention inference tail: x + (sigmoid(LN(x) . Wg^T [+ gb]) o o) . Wo^T [+ bo], the gate projected here",
        py::arg("o"), py::arg("x"), py::arg("wo"), py::arg("wg"), py::arg("lnw"), py::arg("lnb"), py::arg("eps"), py::arg("L"), py::arg("out"),
        py::arg("trans"), py::arg("lr"), py::arg("gb"), py::arg("ob"));
}
