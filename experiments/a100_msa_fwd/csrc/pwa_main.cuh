// pwa_main.cuh -- PWA K_M: out[s, i, :] = msa[s, i, :] + sum_h (sigmoid(y Wg_h'^T + bg_h) .* o_h) Wo_h^T,
//   o_h[s, i, :] = sum_j w[h, i, j] v[h][j][s 32 + :],   y = (msa - mu) rstd (LN affine folded into Wg', bg).
// CTA = 128 i x 2 s (256 tokens), 8 warps = 4 (32 i) x 2 (one s). One cp.async ring over the (head, 64-j chunk) sequence:
// stage = w chunk [128 i][64 j] + v chunk [64 j][64 = 2 s x 32 d] (both 128 B rows, 16 B granules XOR-swizzled by row & 7; v read with
// ldmatrix.trans), so head h+1's loads overlap head h's gate / out-projection. Every smem offset is precomputed per thread (the k16 step
// only XORs the granule bits), and the chunk walk is counters, not divisions (v1 was integer-instruction bound: HMMA 9% of issue).
// Resident: y (the staged msa tile normalized in place) and Wg'. Wo arrives as per-lane pre-packed B fragments (8 x 16 B global loads
// per head). Epilogue adds the residual msa.
// v3 (the H100 pwa_fwd3 stage shape): JC = 128 j per stage (w rows 256 B), so one barrier per 128 j (v2: barrier 11% of stall samples);
// Wo fragments and bg are loaded at the start of a head's last chunk, not after it (v2: 6% long-scoreboard at the head boundary);
// the ldmatrix fragments are double-buffered across k16 steps (PWA_M_NOFRAGDB turns it off: +1% time).
#pragma once
#include "sm80_common.cuh"

namespace a100 {

struct PwaMainParams {
  const __nv_bfloat16* msa;   // [S*L, 64]
  const __nv_bfloat16* w;     // [8][L][L]
  const __nv_bfloat16* v;     // [8][L][S*32]
  const __nv_bfloat16* wg;    // [256, 64] gamma folded
  const float* bg;            // [256]
  const uint4* wof;           // [8 h][32 lanes][8 x 16 B]: Wo B fragments, e = (nt 2 + kc) 2 + half
  __nv_bfloat16* out;         // [S*L, 64]
  int S, L;
  float eps;
};

template <int NS_>
struct PwaMCfg {
  static constexpr int NS = NS_, NTHR = 256, TI = 128, JC = 128;
  static constexpr int Y = 0, WG = 256 * 128, RING = WG + 256 * 128;
  static constexpr int STAGE_W = 128 * JC * 2, STAGE = STAGE_W + JC * 128;
  static constexpr int SMEM = RING + NS * STAGE;
};

DEVI uint32_t swz128(int row, int gr) { return row * 128 + ((gr ^ (row & 7)) << 4); }
DEVI uint32_t swz256(int row, int gr) { return row * 256 + ((gr ^ (row & 7)) << 4); }

template <class G>
__global__ void __launch_bounds__(G::NTHR, 1) pwa_main_kernel(PwaMainParams p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sb = smem_u32(smem);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, q = lane & 3, wi = warp & 3, ws = warp >> 2, l4 = lane >> 4;
  const int L = p.L, nI = L / G::TI, nJ = L / G::JC;
  const int i0 = (blockIdx.x % nI) * G::TI, s0 = (blockIdx.x / nI) * 2;
  const size_t ldv = (size_t)p.S * 32;

  // resident: msa tile (row r = 128 ws + il) and Wg' (row 32 h + d)
  for (int idx = tid; idx < 2048; idx += G::NTHR) {
    const int row = idx >> 3, gr = idx & 7;
    cp_async16(sb + G::Y + swz128(row, gr), p.msa + ((size_t)(s0 + (row >> 7)) * L + i0 + (row & 127)) * 64 + gr * 8);
    cp_async16(sb + G::WG + swz128(row, gr), p.wg + row * 64 + gr * 8);
  }
  cp_async_commit();

  // loads: w: granule (tid & 15) of rows (tid >> 4) + 16 k, k = 0..7 (256 B rows); v: granule (tid & 7) of rows (tid >> 3) + 32 k, k = 0..3
  const uint32_t wdst = (uint32_t)swz256(tid >> 4, tid & 15), ldst = (uint32_t)swz128(tid >> 3, tid & 7);
  const __nv_bfloat16* wsrc = p.w + (size_t)(i0 + (tid >> 4)) * L + (tid & 15) * 8;
  const __nv_bfloat16* vsrc = p.v + (size_t)(tid >> 3) * ldv + (size_t)s0 * 32 + (tid & 7) * 8;
  size_t wofs = 0, vofs = 0;       // next chunk to load: w  h L L + j0, v  (h L + j0) ldv
  int lj = 0, lstage = 0;
  auto issue = [&]() {
    const uint32_t st = sb + G::RING + lstage * G::STAGE;
#pragma unroll
    for (int k = 0; k < 8; ++k) cp_async16(st + wdst + k * 4096, wsrc + wofs + (size_t)k * 16 * L);
#pragma unroll
    for (int k = 0; k < 4; ++k) cp_async16(st + G::STAGE_W + ldst + k * 4096, vsrc + vofs + (size_t)k * 32 * ldv);
    wofs += G::JC;
    vofs += (size_t)G::JC * ldv;
    if (++lj == nJ) { lj = 0; wofs += (size_t)L * L - L; }
    if (++lstage == G::NS) lstage = 0;
  };
  const int NC = 8 * nJ;
#pragma unroll
  for (int s = 0; s < G::NS - 1; ++s) { issue(); cp_async_commit(); }

  // LayerNorm of the staged msa rows in place (thread = row)
  cp_async_wait<G::NS - 1>();
  __syncthreads();
  {
    const int row = tid;
    uint4 x[8];
    float sum = 0.f;
#pragma unroll
    for (int gr = 0; gr < 8; ++gr) {
      x[gr] = lds128(sb + G::Y + swz128(row, gr));
      const uint32_t* u = reinterpret_cast<const uint32_t*>(&x[gr]);
#pragma unroll
      for (int e = 0; e < 4; ++e) sum += bf16lo(u[e]) + bf16hi(u[e]);
    }
    const float mu = sum * (1.f / 64);
    float var = 0.f;
#pragma unroll
    for (int gr = 0; gr < 8; ++gr) {
      const uint32_t* u = reinterpret_cast<const uint32_t*>(&x[gr]);
#pragma unroll
      for (int e = 0; e < 4; ++e) { const float a = bf16lo(u[e]) - mu, b = bf16hi(u[e]) - mu; var += a * a + b * b; }
    }
    const float r = rsqrtf(var * (1.f / 64) + p.eps);
#pragma unroll
    for (int gr = 0; gr < 8; ++gr) {
      uint32_t* u = reinterpret_cast<uint32_t*>(&x[gr]);
#pragma unroll
      for (int e = 0; e < 4; ++e) u[e] = pack_bf16((bf16lo(u[e]) - mu) * r, (bf16hi(u[e]) - mu) * r);
      sts128(sb + G::Y + swz128(row, gr), x[gr]);
    }
  }

  // fragment offsets (k16 step kk: granule 2 kk + l4 -> XOR (kk << 5) into the in-row part)
  uint32_t a_off[2], y_off[2], b_off[2], g_off[2];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    const int r = 32 * wi + 16 * mt + (lane & 15);
    a_off[mt] = swz256(r, l4);
    y_off[mt] = G::Y + swz128(128 * ws + r, l4);
  }
  const int r0 = (lane & 7) + (((lane >> 3) & 1) << 3);
#pragma unroll
  for (int np = 0; np < 2; ++np) {
    b_off[np] = G::STAGE_W + swz128(r0, 4 * ws + 2 * np + l4);                                           // + kk 2048
    g_off[np] = G::WG + swz128(16 * np + (lane & 7) + (l4 << 3), (lane >> 3) & 1);                       // + 32 h 128, ^ (kc << 5)
  }

  float acc[2][4][4], oacc[2][8][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[mt][nt][e] = 0.f;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) oacc[mt][nt][e] = 0.f;
  }

  int c = 0, stage = 0;
  uint4 wo4[8];
  float2 bgv[4];
  for (int h = 0; h < 8; ++h) {
    for (int jc = 0; jc < nJ; ++jc, ++c) {
      if (jc == nJ - 1) {           // this head's Wo fragments / gate bias: in flight during the last chunk
        const uint4* wsrc4 = p.wof + ((size_t)h * 32 + lane) * 8;
#pragma unroll
        for (int e = 0; e < 8; ++e) wo4[e] = __ldg(wsrc4 + e);
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) bgv[nt] = __ldg(reinterpret_cast<const float2*>(p.bg + 32 * h + 8 * nt + 2 * q));
      }
      cp_async_wait<G::NS - 2>();
      __syncthreads();
      if (c + G::NS - 1 < NC) issue();
      cp_async_commit();
      const uint32_t st = sb + G::RING + stage * G::STAGE;
      if (++stage == G::NS) stage = 0;
#ifndef PWA_M_NOFRAGDB
      uint32_t af[2][2][4], bf[2][2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) ldsm_x4(af[0][mt], st + a_off[mt]);
#pragma unroll
      for (int np = 0; np < 2; ++np) ldsm_x4_t(bf[0][np], st + b_off[np]);
#pragma unroll
      for (int kk = 0; kk < G::JC / 16; ++kk) {
        const int cb = kk & 1, nb = cb ^ 1;
        if (kk + 1 < G::JC / 16) {
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) ldsm_x4(af[nb][mt], st + (a_off[mt] ^ ((kk + 1) << 5)));
#pragma unroll
          for (int np = 0; np < 2; ++np) ldsm_x4_t(bf[nb][np], st + b_off[np] + (kk + 1) * 2048);
        }
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int nt = 0; nt < 4; ++nt)
            mma16816(acc[mt][nt], af[cb][mt], bf[cb][nt >> 1][(nt & 1) * 2], bf[cb][nt >> 1][(nt & 1) * 2 + 1]);
      }
#else
#pragma unroll
      for (int kk = 0; kk < G::JC / 16; ++kk) {
        uint32_t af[2][4], bf[2][4];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) ldsm_x4(af[mt], st + (a_off[mt] ^ (kk << 5)));
#pragma unroll
        for (int np = 0; np < 2; ++np) ldsm_x4_t(bf[np], st + b_off[np] + kk * 2048);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) mma16816(acc[mt][nt], af[mt], bf[nt >> 1][(nt & 1) * 2], bf[nt >> 1][(nt & 1) * 2 + 1]);
      }
#endif
    }
    // head boundary: gate GEMM, u = sigmoid(g) o, out-projection
    float gacc[2][4][4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const float2 b = bgv[nt];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) { gacc[mt][nt][0] = gacc[mt][nt][2] = b.x; gacc[mt][nt][1] = gacc[mt][nt][3] = b.y; }
    }
#pragma unroll
    for (int kc = 0; kc < 4; ++kc) {
      uint32_t af[2][4], bf[2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) ldsm_x4(af[mt], sb + (y_off[mt] ^ (kc << 5)));
#pragma unroll
      for (int np = 0; np < 2; ++np) ldsm_x4(bf[np], sb + ((g_off[np] + 32 * h * 128) ^ (kc << 5)));
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) mma16816(gacc[mt][nt], af[mt], bf[nt >> 1][(nt & 1) * 2], bf[nt >> 1][(nt & 1) * 2 + 1]);
    }
    uint32_t ua[2][2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int kc = 0; kc < 2; ++kc) {
        float (&x0)[4] = acc[mt][2 * kc];
        float (&x1)[4] = acc[mt][2 * kc + 1];
        float (&g0)[4] = gacc[mt][2 * kc];
        float (&g1)[4] = gacc[mt][2 * kc + 1];
        ua[mt][kc][0] = pack_bf16(sigmoid(g0[0]) * x0[0], sigmoid(g0[1]) * x0[1]);
        ua[mt][kc][1] = pack_bf16(sigmoid(g0[2]) * x0[2], sigmoid(g0[3]) * x0[3]);
        ua[mt][kc][2] = pack_bf16(sigmoid(g1[0]) * x1[0], sigmoid(g1[1]) * x1[1]);
        ua[mt][kc][3] = pack_bf16(sigmoid(g1[2]) * x1[2], sigmoid(g1[3]) * x1[3]);
      }
    const uint32_t* wo = reinterpret_cast<const uint32_t*>(wo4);
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int kc = 0; kc < 2; ++kc)
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) mma16816(oacc[mt][nt], ua[mt][kc], wo[(nt * 2 + kc) * 2], wo[(nt * 2 + kc) * 2 + 1]);
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[mt][nt][e] = 0.f;
  }
  cp_async_wait<0>();

  // residual + store: rows i = i0 + 32 wi + 16 mt + g (+8) of s = s0 + ws, cols 8 nt + 2 q
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const size_t base = ((size_t)(s0 + ws) * L + i0 + 32 * wi + 16 * mt + g + 8 * hh) * 64;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int cc = 8 * nt + 2 * q;
        const uint32_t r = __ldg(reinterpret_cast<const unsigned int*>(p.msa + base + cc));
        stg32(p.out + base + cc, pack_bf16(bf16lo(r) + oacc[mt][nt][2 * hh], bf16hi(r) + oacc[mt][nt][2 * hh + 1]));
      }
    }
}

}  // namespace a100
