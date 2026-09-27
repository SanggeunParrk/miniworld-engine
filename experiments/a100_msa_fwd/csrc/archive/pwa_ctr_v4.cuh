// pwa_ctr.cuh -- PWA split path, K_C: per head h, u[s, i, 32 h + d] = sigmoid(y[s, i, :] Wg_h'^T + bg_h) .* o_h[s, i, d],
//   o_h[s, i, :] = sum_j w[h, i, j] v[h][j][s 32 + :],  y = LN(msa) without affine (folded into Wg', bg).
// The out-projection (sum over heads) is NOT in this kernel: u goes to memory [S*L][256] and one GEMM (+ residual) follows. That frees
// the 64 fp32 per token the fused kernel kept across heads, so the contraction gets a cutlass-shaped tile: CTA = one head x 128 i x
// 4 s (N = 128), 4 warps of 64 i x 64 (2 s x 32 d), 2 CTAs / SM. Per k16 step a warp issues 8 ldmatrix for 32 mma (fused kernel: 4 for 8).
// K = L in 32-j chunks through an NS-stage cp.async ring: w chunk [128 i][32 j] (64 B rows), v chunk [32 j][128 = 4 s x 32 d] (256 B rows,
// ldmatrix.trans). Epilogue: the y tile [4 s][128 i][64] (LN(msa), written once by pwa_value; v1 re-normalized msa here for every
// head: FADD / FMUL / FFMA 21% of instructions) is staged into the idle ring, and the gate GEMM runs per (s, m-tile pair) with Wg_h'
// fragments pre-packed per lane (8 x 16 B global loads).
// v3: JC templated (64 j per stage halves the barriers), u leaves through the y rows it replaces (16 B coalesced stores instead of 64
// 4-byte ones per thread; the epilogue was 30% of v2's stall samples), 32-bit element offsets in the copy addressing (LEA 12% in v2).
#pragma once
#include "sm80_common.cuh"

namespace a100 {

struct PwaCtrParams {
  const __nv_bfloat16* y;     // [S*L, 64] LN(msa) without affine (from pwa_value)
  const __nv_bfloat16* w;     // [8][L][L]
  const __nv_bfloat16* v;     // [8][L][S*32]
  const uint4* wgf;           // [8 h][32 lanes][8 x 16 B]: Wg' B fragments, e = (dt 4 + kc) 2 + half
  const float* bg;            // [256]
  __nv_bfloat16* u;           // [S*L, 256]
  const int* cnt;             // [2] n, n_pad of the compacted keys, or nullptr (dense: K = L)
  int ldw;                    // w row stride and v rows per head (L dense, >= n_pad compacted)
  int S, L;
  float eps;
};

template <int NS_, int JC_>
struct PwaCCfg {
  static constexpr int NS = NS_, NTHR = 128, TI = 128, TS = 4, JC = JC_;
  static constexpr int AGR = JC / 8, NCP = JC / 8;                           // A granules per row; copies per thread per operand
  static constexpr int STAGE_A = 128 * JC * 2, STAGE = STAGE_A + JC * 256;
  DEVI static uint32_t swzA(int row, int gr) {
    return row * (JC * 2) + ((JC == 32 ? (gr ^ ((row >> 1) & 3)) : (gr ^ (row & 7))) << 4);
  }
  static constexpr int YB = TS * 128 * 128;                                   // staged y tile, 64 KB
  static constexpr int SMEM = NS * STAGE > YB ? NS * STAGE : YB;
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
  // K = n_pad (a multiple of 16) in JC chunks; the last chunk may run fewer k16 steps. Its unused rows are loaded (inside the
  // [.., L, ..] buffers, L % JC == 0) but never multiplied. (Fragment double-buffering of the k16 steps was tried: +-0, removed.)
  const int L = p.L, nI = L / G::TI, kpad = p.cnt ? __ldg(p.cnt + 1) : L, nJ = (kpad + G::JC - 1) / G::JC, ldw = p.ldw;
  // grid order: i-block fastest, then head, then s-block (the y tile and the v slice of an s-block stay L2-resident)
  const int ib = blockIdx.x % nI, h = (blockIdx.x / nI) & 7, sblk = blockIdx.x / (nI * 8);
  const int i0 = ib * G::TI, s0 = sblk * G::TS;
  const size_t ldv = (size_t)p.S * 32;

  // copy slots: A granule (tid % AGR) of rows tid / AGR + (128 / NCP) k; B granule (tid & 15) of rows (tid >> 4) + 8 k; k < NCP.
  // 32-bit element offsets (w: 8 L^2, v: 8 L S 32 elements, both < 2^31 at the supported sizes)
  constexpr int AROWS = 128 / G::NCP;
  const uint32_t a_dst = G::swzA(tid / G::AGR, tid % G::AGR), b_dst = G::STAGE_A + swz256c(tid >> 4, tid & 15);
  const __nv_bfloat16* wbase = p.w;
  const __nv_bfloat16* vbase = p.v;
  uint32_t aoff = (uint32_t)((h * L + i0 + tid / G::AGR) * ldw + (tid % G::AGR) * 8);
  uint32_t boff = (uint32_t)(((size_t)h * ldw + (tid >> 4)) * ldv + (size_t)s0 * 32 + (tid & 15) * 8);
  const uint32_t astep = (uint32_t)(AROWS * ldw), bstep = (uint32_t)(8 * ldv), bchunk = (uint32_t)(G::JC * ldv);
  auto issue = [&](int stage) {       // the next chunk; offsets advance by one chunk
    const uint32_t st = sb + stage * G::STAGE;
#pragma unroll
    for (int k = 0; k < G::NCP; ++k) {
      cp_async16(st + a_dst + k * AROWS * (G::JC * 2), wbase + (aoff + k * astep));
      cp_async16(st + b_dst + k * 8 * 256, vbase + (boff + k * bstep));
    }
    aoff += G::JC;
    boff += bchunk;
  };

  uint32_t a_off[4], b_off[4];
#pragma unroll
  for (int mt = 0; mt < 4; ++mt) a_off[mt] = G::swzA(64 * wm + 16 * mt + (lane & 15), l4);
  const int r0 = (lane & 7) + (((lane >> 3) & 1) << 3);
#pragma unroll
  for (int np = 0; np < 4; ++np) b_off[np] = G::STAGE_A + swz256c(r0, 8 * wn + 2 * np + l4);

  float acc[4][8][4];
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 8; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;

#pragma unroll
  for (int s = 0; s < G::NS - 1; ++s) { if (s < nJ) issue(s); cp_async_commit(); }
  int stage = 0, lstage = G::NS - 1;
  for (int jc = 0; jc < nJ; ++jc) {
    cp_async_wait<G::NS - 2>();
    __syncthreads();
    if (jc + G::NS - 1 < nJ) issue(lstage);
    cp_async_commit();
    if (++lstage == G::NS) lstage = 0;
    const uint32_t st = sb + stage * G::STAGE;
    if (++stage == G::NS) stage = 0;
    const int kkn = min(G::JC / 16, (kpad - jc * G::JC) / 16);
    auto kstep = [&](int kk) {
      uint32_t af[4][4], bf[4][4];
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) ldsm_x4(af[mt], st + (a_off[mt] ^ (kk << 5)));
#pragma unroll
      for (int np = 0; np < 4; ++np) ldsm_x4_t(bf[np], st + b_off[np] + kk * 16 * 256);
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) mma16816(acc[mt][nt], af[mt], bf[nt >> 1][(nt & 1) * 2], bf[nt >> 1][(nt & 1) * 2 + 1]);
    };
    if (kkn == G::JC / 16) {        // full chunk: straight-line code
#pragma unroll
      for (int kk = 0; kk < G::JC / 16; ++kk) kstep(kk);
    } else {                        // the partial last chunk of a compacted K
#pragma unroll 1
      for (int kk = 0; kk < kkn; ++kk) kstep(kk);
    }
  }
  cp_async_wait<0>();
  __syncthreads();

#ifndef PWA_C_GATE
  // o only (the gate runs in pwa_out2, the memory-bound kernel with idle tensor time): acc -> bf16 [512 tokens][64 B] in the ring,
  // then 16 B coalesced stores into o[tok][32 h ..]. Token row r = 128 sl + il; warp rows i = 64 wm + 16 mt + g (+8), s = 2 wn + nt / 4.
  // Within a head, o is stored in FRAGMENT ORDER: position 8 q + 2 dt + e holds d = 8 dt + 2 q + e, so pwa_out2's lane q finds its
  // eight accumulator-layout values (dt = 0..3) as one 16 B vector. Staging: granule q of the 64 B row, u32 slot dt ^ ((row >> 1) & 3)
  // -- the XOR spreads the 32 lanes of one store over 32 banks (plain slot dt: 4-way conflict); the read-out undoes it with two
  // conditional swaps. (A register transpose on the read-out, one row per thread, broke global-store coalescing: +13%.)
#ifndef PWA_C_STAGE_CONFLICT
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int row = 128 * (2 * wn + (nt >> 2)) + 64 * wm + 16 * mt + g + 8 * hh;
        sts32(sb + row * 64 + q * 16 + 4 * ((nt & 3) ^ ((row >> 1) & 3)), pack_bf16(acc[mt][nt][2 * hh], acc[mt][nt][2 * hh + 1]));
      }
  __syncthreads();
#pragma unroll 4
  for (int k = 0; k < 16; ++k) {
    const int idx = tid + 128 * k, row = idx >> 2, gr = idx & 3, hx = (row >> 1) & 3;
    const size_t tok = (size_t)(s0 + (row >> 7)) * L + i0 + (row & 127);
    uint4 v = lds128(sb + row * 64 + gr * 16);
    if (hx & 1) { uint32_t t = v.x; v.x = v.y; v.y = t; t = v.z; v.z = v.w; v.w = t; }
    if (hx & 2) { uint32_t t = v.x; v.x = v.z; v.z = t; t = v.y; v.y = v.w; v.w = t; }
    stg128(p.u + tok * 256 + 32 * h + gr * 8, v);
  }
#else
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int row = 128 * (2 * wn + (nt >> 2)) + 64 * wm + 16 * mt + g + 8 * hh;
        sts32(sb + swz64c(row, q) + 4 * (nt & 3), pack_bf16(acc[mt][nt][2 * hh], acc[mt][nt][2 * hh + 1]));
      }
  __syncthreads();
#pragma unroll 4
  for (int k = 0; k < 16; ++k) {
    const int idx = tid + 128 * k, row = idx >> 2, gr = idx & 3;
    const size_t tok = (size_t)(s0 + (row >> 7)) * L + i0 + (row & 127);
    stg128(p.u + tok * 256 + 32 * h + gr * 8, lds128(sb + swz64c(row, gr)));
  }
#endif
#else
  // y tile -> smem (row r = 128 sl + il)
  for (int idx = tid; idx < G::TS * 128 * 8; idx += G::NTHR) {
    const int row = idx >> 3, gr = idx & 7;
    cp_async16(sb + swz128c(row, gr), p.y + ((size_t)(s0 + (row >> 7)) * L + i0 + (row & 127)) * 64 + gr * 8);
  }
  cp_async_commit();
  uint4 wg4[8];
  const uint4* wsrc4 = p.wgf + ((size_t)h * 32 + lane) * 8;
#pragma unroll
  for (int e = 0; e < 8; ++e) wg4[e] = __ldg(wsrc4 + e);
  float2 bgv[4];
#pragma unroll
  for (int dt = 0; dt < 4; ++dt) bgv[dt] = __ldg(reinterpret_cast<const float2*>(p.bg + 32 * h + 8 * dt + 2 * q));
  cp_async_wait<0>();
  __syncthreads();

  // gate + u: warp tokens are i = i0 + 64 wm + 16 mt + g (+8), s = s0 + 2 wn + sn; acc n-tile nt = 4 sn + dt (d = 8 dt + 2 q)
  const uint32_t* wg = reinterpret_cast<const uint32_t*>(wg4);
#pragma unroll
  for (int sn = 0; sn < 2; ++sn)
#pragma unroll
    for (int mp = 0; mp < 2; ++mp) {
      float gacc[2][4][4];
#pragma unroll
      for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int dt = 0; dt < 4; ++dt) {
          gacc[m][dt][0] = gacc[m][dt][2] = bgv[dt].x;
          gacc[m][dt][1] = gacc[m][dt][3] = bgv[dt].y;
        }
#pragma unroll
      for (int kc = 0; kc < 4; ++kc) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          uint32_t af[4];
          const int row = 128 * (2 * wn + sn) + 64 * wm + 16 * (2 * mp + m) + (lane & 15);
          ldsm_x4(af, sb + swz128c(row, 2 * kc + l4));
#pragma unroll
          for (int dt = 0; dt < 4; ++dt) mma16816(gacc[m][dt], af, wg[(dt * 4 + kc) * 2], wg[(dt * 4 + kc) * 2 + 1]);
        }
      }
      __syncwarp();      // every lane's y reads of these rows are done: u (64 B) replaces the first half of each row
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const int mt = 2 * mp + m;
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int row = 128 * (2 * wn + sn) + 64 * wm + 16 * mt + g + 8 * hh;
#pragma unroll
          for (int dt = 0; dt < 4; ++dt) {
            const float* o = acc[mt][4 * sn + dt];
            const float* gg = gacc[m][dt];
            sts32(sb + swz128c(row, dt) + 4 * q, pack_bf16(sigmoid(gg[2 * hh]) * o[2 * hh], sigmoid(gg[2 * hh + 1]) * o[2 * hh + 1]));
          }
        }
      }
    }
  __syncthreads();
  // u tile: 512 token rows x 64 B -> u[tok][32 h .. 32 h + 31]
#pragma unroll 4
  for (int k = 0; k < 16; ++k) {
    const int idx = tid + 128 * k, row = idx >> 2, gr = idx & 3;
    const size_t tok = (size_t)(s0 + (row >> 7)) * L + i0 + (row & 127);
    stg128(p.u + tok * 256 + 32 * h + gr * 8, lds128(sb + swz128c(row, gr)));
  }
#endif
}

}  // namespace a100
