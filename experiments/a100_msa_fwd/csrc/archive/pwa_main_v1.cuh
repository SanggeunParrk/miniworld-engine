// pwa_main.cuh -- PWA K_M: out[s, i, :] = msa[s, i, :] + sum_h (sigmoid(y Wg_h'^T + bg_h) .* o_h) Wo_h^T,
//   o_h[s, i, :] = sum_j w[h, i, j] v[h][j][s 32 + :],   y = (msa - mu) rstd (LN affine folded into Wg', bg).
// CTA = 128 i x 2 s (256 tokens), 8 warps = 4 (32 i) x 2 (one s). One cp.async ring over the flattened (head, 32-j chunk) sequence:
// stage = w chunk [128 i][32 j] (64 B rows) + v chunk [32 j][64 = 2 s x 32 d] (128 B rows, read with ldmatrix.trans), so the loads of
// head h+1 overlap head h's gate / out-projection. The msa tile is staged once and normalized in place (y); Wg' and Wo stay resident.
// At a head boundary: gate GEMM from y, u = sigmoid(g) o packed from the accumulators straight into A fragments, out-projection
// accumulated over heads in registers. Epilogue adds the residual msa.
#pragma once
#include "sm80_common.cuh"

namespace a100 {

struct PwaMainParams {
  const __nv_bfloat16* msa;   // [S*L, 64]
  const __nv_bfloat16* w;     // [8][L][L]
  const __nv_bfloat16* v;     // [8][L][S*32]
  const __nv_bfloat16* wg;    // [256, 64] gamma folded
  const float* bg;            // [256]
  const __nv_bfloat16* wo;    // [64, 256]
  __nv_bfloat16* out;         // [S*L, 64]
  int S, L;
  float eps;
};

template <int NS_>
struct PwaMCfg {
  static constexpr int NS = NS_, NTHR = 256, TI = 128;
  static constexpr int Y = 0, WG = 256 * 128, WO = WG + 256 * 128, RING = WO + 64 * 512;
  static constexpr int STAGE_W = 128 * 64, STAGE = STAGE_W + 32 * 128;
  static constexpr int SMEM = RING + NS * STAGE;
};

DEVI uint32_t swz128(int row, int gr) { return row * 128 + ((gr ^ (row & 7)) << 4); }
DEVI uint32_t swz512(int row, int gr) { return row * 512 + ((gr ^ (row & 7)) << 4); }
DEVI uint32_t swz64m(int row, int gr) { return row * 64 + ((gr ^ ((row >> 1) & 3)) << 4); }

template <class G>
__global__ void __launch_bounds__(G::NTHR, 1) pwa_main_kernel(PwaMainParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t sb = smem_u32(smem);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, q = lane & 3, wi = warp & 3, ws = warp >> 2;
  const int L = p.L, nI = L / G::TI, nJ = L / 32;
  const int i0 = (blockIdx.x % nI) * G::TI, s0 = (blockIdx.x / nI) * 2;
  const size_t ldv = (size_t)p.S * 32;

  // resident: msa tile (row r = 128 ws + il), Wg' (row 32 h + d), Wo (row c, 512 B)
  for (int idx = tid; idx < 2048; idx += G::NTHR) {
    const int row = idx >> 3, gr = idx & 7;
    cp_async16(sb + G::Y + swz128(row, gr), p.msa + ((size_t)(s0 + (row >> 7)) * L + i0 + (row & 127)) * 64 + gr * 8);
    cp_async16(sb + G::WG + swz128(row, gr), p.wg + row * 64 + gr * 8);
    const int ro = idx >> 5, go = idx & 31;
    cp_async16(sb + G::WO + swz512(ro, go), p.wo + ro * 256 + go * 8);
  }
  cp_async_commit();

  auto load = [&](int c, int stage) {
    const int h = c / nJ, j0 = (c % nJ) * 32;
    const uint32_t st = sb + G::RING + stage * G::STAGE;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int idx = tid + 256 * k, row = idx >> 2, gr = idx & 3;
      cp_async16(st + swz64m(row, gr), p.w + ((size_t)h * L + i0 + row) * L + j0 + gr * 8);
    }
    const int row = tid >> 3, gr = tid & 7;
    cp_async16(st + G::STAGE_W + swz128(row, gr), p.v + ((size_t)h * L + j0 + row) * ldv + (size_t)s0 * 32 + gr * 8);
  };
  const int NC = 8 * nJ;
#pragma unroll
  for (int s = 0; s < G::NS - 1; ++s) { load(s, s); cp_async_commit(); }

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

  for (int c = 0; c < NC; ++c) {
    cp_async_wait<G::NS - 2>();
    __syncthreads();
    if (c + G::NS - 1 < NC) load(c + G::NS - 1, (c + G::NS - 1) % G::NS);
    cp_async_commit();
    const uint32_t st = sb + G::RING + (c % G::NS) * G::STAGE, sv = st + G::STAGE_W;
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      uint32_t af[2][4], bf[2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) ldsm_x4(af[mt], st + swz64m(32 * wi + 16 * mt + (lane & 15), 2 * kk + (lane >> 4)));
#pragma unroll
      for (int np = 0; np < 2; ++np)
        ldsm_x4_t(bf[np], sv + swz128(16 * kk + (lane & 7) + (((lane >> 3) & 1) << 3), 4 * ws + 2 * np + (lane >> 4)));
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) mma16816(acc[mt][nt], af[mt], bf[nt >> 1][(nt & 1) * 2], bf[nt >> 1][(nt & 1) * 2 + 1]);
    }
    if (c % nJ == nJ - 1) {           // head boundary: gate, u = sigmoid(g) o, out-projection
      const int h = c / nJ;
      float gacc[2][4][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int d = 8 * nt + 2 * q;
          gacc[mt][nt][0] = gacc[mt][nt][2] = p.bg[32 * h + d];
          gacc[mt][nt][1] = gacc[mt][nt][3] = p.bg[32 * h + d + 1];
        }
#pragma unroll
      for (int kc = 0; kc < 4; ++kc) {
        uint32_t af[2][4], bf[2][4];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) ldsm_x4(af[mt], sb + G::Y + swz128(128 * ws + 32 * wi + 16 * mt + (lane & 15), 2 * kc + (lane >> 4)));
#pragma unroll
        for (int np = 0; np < 2; ++np)
          ldsm_x4(bf[np], sb + G::WG + swz128(32 * h + 16 * np + (lane & 7) + ((lane >> 4) << 3), 2 * kc + ((lane >> 3) & 1)));
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
#pragma unroll
      for (int kc = 0; kc < 2; ++kc)
#pragma unroll
        for (int np = 0; np < 4; ++np) {
          uint32_t bf[4];
          ldsm_x4(bf, sb + G::WO + swz512(16 * np + (lane & 7) + ((lane >> 4) << 3), 4 * h + 2 * kc + ((lane >> 3) & 1)));
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            mma16816(oacc[mt][2 * np], ua[mt][kc], bf[0], bf[1]);
            mma16816(oacc[mt][2 * np + 1], ua[mt][kc], bf[2], bf[3]);
          }
        }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[mt][nt][e] = 0.f;
    }
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
