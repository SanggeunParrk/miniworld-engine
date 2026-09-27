// pwa_ctr.cuh -- PWA split path, K_C (contraction only; the gate and the out-projection run in pwa_out2):
//   o_h[s, i, :] = sum_j w[h, i, j] v[h][j][s 32 + :]      forward: o [S*L][256] in fragment order (pwa_out2's operand layout)
//   dv_h[s, j, :] = sum_i w[h, i, j] do[h][i][s 32 + :]    backward (TA, NAT): dv [S*L][256] natural order
// Tile = one head x 128 i (M) x 4 s (N = 128); 4 warps of 64 x 64; 2 CTAs / SM. K in 64-deep chunks through a 2-slot cp.async ring:
// A chunk = w [128 i][64 j] (128 B rows) -- or, for TA, w rows [64 i][128 j] (256 B) read with ldmatrix.trans -- and B chunk =
// v (or do) [64 j][128 = 4 s x 32 d] (256 B rows, ldmatrix.trans). Key compaction: K runs to n_pad (cnt; a partial last chunk runs
// fewer k16 steps on its own code path), and for TA the M range is the compacted keys (mcnt, output token j = midx[k]).
// v6: PERSISTENT. A CTA walks tiles blockIdx.x, + gridDim.x, ... and the ring runs straight through tile boundaries: the next tile's
// first chunk is in flight during the current tile's last chunk and epilogue, which stages the output through the slot that chunk
// just freed. (v5: one tile per CTA, 11 chunks at L768; ~30% of the stall samples sat outside the main loop -- the first chunk's wait,
// the epilogue -- and stage-dependent addresses were recomputed with IMAD every iteration.)
#pragma once
#include "sm80_common.cuh"

namespace a100 {

struct PwaCtrParams {
  const __nv_bfloat16* y;     // unused (the gate-in-ctr path is retired)
  const __nv_bfloat16* w;     // [8][L][ldw]
  const __nv_bfloat16* v;     // [8][ldw][S*32] (forward: v, backward: do with ldw = L)
  const uint4* wgf;           // unused
  const float* bg;            // unused
  __nv_bfloat16* u;           // [S*L, 256] output (o or dv)
  const int* cnt;             // [2] n, n_pad of the compacted keys for K, or nullptr (K = L)
  const int* midx;            // TA + compaction: output token j = midx[k] for M index k (nullptr: dense)
  const int* mcnt;            // [2] n, n_pad for the TA M range
  int ldw;
  int S, L;
  float eps;
};

template <int NS_, int JC_, bool TA_ = false, bool NAT_ = false>
struct PwaCCfg {
  static_assert(NS_ == 2 && JC_ == 64, "the persistent ring is 2 x 64");
  static constexpr int NS = 2, NTHR = 128, TI = 128, TS = 4, JC = 64;
  static constexpr bool TA = TA_, NAT = NAT_;
  static constexpr int STAGE_A = 128 * JC * 2, STAGE = STAGE_A + JC * 256;   // 16 KB + 16 KB
  static constexpr int SMEM = NS * STAGE;
};

DEVI uint32_t swz64c(int row, int gr) { return row * 64 + ((gr ^ ((row >> 1) & 3)) << 4); }
DEVI uint32_t swz128c(int row, int gr) { return row * 128 + ((gr ^ (row & 7)) << 4); }
DEVI uint32_t swz256c(int row, int gr) { return row * 256 + ((gr ^ (row & 7)) << 4); }

template <class G>
__global__ void __launch_bounds__(G::NTHR, 2) pwa_ctr_kernel(PwaCtrParams p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sb = smem_u32(smem);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, q = lane & 3, wm = warp & 1, wn = warp >> 1, l4 = lane >> 4;
  const int L = p.L, ldw = p.ldw, nsb = p.S / G::TS;
  const int kpad = p.cnt ? __ldg(p.cnt + 1) : L, nJ = (kpad + G::JC - 1) / G::JC;
  const int mval = (G::TA && p.mcnt) ? __ldg(p.mcnt) : L;
  const int nIv = (G::TA && p.mcnt) ? (__ldg(p.mcnt + 1) + G::TI - 1) / G::TI : L / G::TI;
  const int ntiles = nIv * 8 * nsb;
  if ((int)blockIdx.x >= ntiles) return;
  const int ntl = (ntiles - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;   // this CTA's tiles
  const int gtotal = ntl * nJ;
  const size_t ldv = (size_t)p.S * 32;
  // copy slots (8 per operand per chunk): A: non-TA granule (tid & 7) of rows (tid >> 3) + 16 k (128 B rows); TA granule (tid & 15) of
  // rows (tid >> 4) + 8 k (256 B rows). B: granule (tid & 15) of rows (tid >> 4) + 8 k.
  const uint32_t a_dst = G::TA ? swz256c(tid >> 4, tid & 15) : swz128c(tid >> 3, tid & 7);
  const uint32_t b_dst = G::STAGE_A + swz256c(tid >> 4, tid & 15);
  constexpr int AROWB = G::TA ? 256 : 128, AROWS = G::TA ? 8 : 16;
  auto issue = [&](int gc, int stage) {
    const int tl = gc / nJ, jc = gc - tl * nJ;
    const int t = (int)blockIdx.x + tl * (int)gridDim.x;
    const int ib = t % nIv, h = (t / nIv) & 7, s0 = (t / (nIv * 8)) * G::TS, i0 = ib * G::TI;
    const __nv_bfloat16* asrc = G::TA ? p.w + ((size_t)(h * L + jc * G::JC + (tid >> 4)) * ldw + i0 + (tid & 15) * 8)
                                      : p.w + ((size_t)(h * L + i0 + (tid >> 3)) * ldw + jc * G::JC + (tid & 7) * 8);
    const __nv_bfloat16* bsrc = p.v + ((size_t)h * ldw + jc * G::JC + (tid >> 4)) * ldv + (size_t)s0 * 32 + (tid & 15) * 8;
    const size_t astep = (size_t)AROWS * ldw, bstep = (size_t)8 * ldv;
    const uint32_t st = sb + stage * G::STAGE;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      cp_async16(st + a_dst + k * AROWS * AROWB, asrc + k * astep);
      cp_async16(st + b_dst + k * 8 * 256, bsrc + k * bstep);
    }
  };

  uint32_t a_off[4], b_off[4];
#pragma unroll
  for (int mt = 0; mt < 4; ++mt) {
    if (G::TA) {          // .trans: matrix (lane >> 3): bit 0 = m half, bit 1 = k half; row = k, granule = m / 8
      const int mat = lane >> 3;
      a_off[mt] = swz256c(8 * (mat >> 1) + (lane & 7), 8 * wm + 2 * mt + (mat & 1));      // + kk 16 rows
    } else {
      a_off[mt] = swz128c(64 * wm + 16 * mt + (lane & 15), l4);                             // ^ (kk << 5)
    }
  }
  const int r0 = (lane & 7) + (((lane >> 3) & 1) << 3);
#pragma unroll
  for (int np = 0; np < 4; ++np) b_off[np] = G::STAGE_A + swz256c(r0, 8 * wn + 2 * np + l4);

  issue(0, 0);
  cp_async_commit();
  int gc = 0, stage = 0;
  for (int tl = 0; tl < ntl; ++tl) {
    const int t = (int)blockIdx.x + tl * (int)gridDim.x;
    const int ib = t % nIv, h = (t / nIv) & 7, s0 = (t / (nIv * 8)) * G::TS, i0 = ib * G::TI;
    float acc[4][8][4];
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
      for (int b = 0; b < 8; ++b)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
    for (int jc = 0; jc < nJ; ++jc, ++gc) {
      cp_async_wait<0>();
      __syncthreads();
      if (gc + 1 < gtotal) issue(gc + 1, stage ^ 1);
      cp_async_commit();
      const uint32_t st = sb + stage * G::STAGE;
      stage ^= 1;
      const int kkn = min(G::JC / 16, (kpad - jc * G::JC) / 16);
      auto kstep = [&](int kk) {
        uint32_t af[4][4], bf[4][4];
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
          if (G::TA) ldsm_x4_t(af[mt], st + a_off[mt] + kk * 16 * 256);
          else ldsm_x4(af[mt], st + (a_off[mt] ^ (kk << 5)));
        }
#pragma unroll
        for (int np = 0; np < 4; ++np) ldsm_x4_t(bf[np], st + b_off[np] + kk * 16 * 256);
#pragma unroll
        for (int mt = 0; mt < 4; ++mt)
#pragma unroll
          for (int nt = 0; nt < 8; ++nt) mma16816(acc[mt][nt], af[mt], bf[nt >> 1][(nt & 1) * 2], bf[nt >> 1][(nt & 1) * 2 + 1]);
      };
      if (kkn == G::JC / 16) {
#pragma unroll
        for (int kk = 0; kk < G::JC / 16; ++kk) kstep(kk);
      } else {
#pragma unroll 1
        for (int kk = 0; kk < kkn; ++kk) kstep(kk);
      }
    }
    // epilogue through the slot the last chunk used (stage ^ 1 now): free once every warp is past its mma
    __syncthreads();
    const uint32_t eb = sb + (stage ^ 1) * G::STAGE;
    // token row r = 128 sl + il; warp rows i = 64 wm + 16 mt + g (+8), s = 2 wn + nt / 4
    if (G::NAT) {
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const int row = 128 * (2 * wn + (nt >> 2)) + 64 * wm + 16 * mt + g + 8 * hh;
            sts32(eb + swz64c(row, nt & 3) + 4 * q, pack_bf16(acc[mt][nt][2 * hh], acc[mt][nt][2 * hh + 1]));
          }
      __syncthreads();
#pragma unroll 4
      for (int k = 0; k < 16; ++k) {
        const int idx = tid + 128 * k, row = idx >> 2, gr = idx & 3, m = i0 + (row & 127);
        if (m >= mval) continue;
        const size_t tok = (size_t)(s0 + (row >> 7)) * L + (p.midx ? __ldg(p.midx + m) : m);
        stg128(p.u + tok * 256 + 32 * h + gr * 8, lds128(eb + swz64c(row, gr)));
      }
    } else {
      // FRAGMENT ORDER within a head: position 8 q + 2 dt + e holds d = 8 dt + 2 q + e (pwa_out2's lane q reads its 16 B). Staging slot
      // dt ^ ((row >> 1) & 3) keeps the stores conflict-free; the read-out undoes it with two conditional swaps.
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const int row = 128 * (2 * wn + (nt >> 2)) + 64 * wm + 16 * mt + g + 8 * hh;
            sts32(eb + row * 64 + q * 16 + 4 * ((nt & 3) ^ ((row >> 1) & 3)), pack_bf16(acc[mt][nt][2 * hh], acc[mt][nt][2 * hh + 1]));
          }
      __syncthreads();
#pragma unroll 4
      for (int k = 0; k < 16; ++k) {
        const int idx = tid + 128 * k, row = idx >> 2, gr = idx & 3, hx = (row >> 1) & 3;
        const size_t tok = (size_t)(s0 + (row >> 7)) * L + i0 + (row & 127);
        uint4 v = lds128(eb + row * 64 + gr * 16);
        if (hx & 1) { uint32_t tt = v.x; v.x = v.y; v.y = tt; tt = v.z; v.z = v.w; v.w = tt; }
        if (hx & 2) { uint32_t tt = v.x; v.x = v.z; v.z = tt; tt = v.y; v.y = v.w; v.w = tt; }
        stg128(p.u + tok * 256 + 32 * h + gr * 8, v);
      }
    }
    // the next tile's first __syncthreads (after its wait) orders these reads of slot eb before it is refilled
  }
  cp_async_wait<0>();
}

}  // namespace a100
