// pwa_bwd.cuh -- PWA backward kernels (A100, sm_80), split-path forward: v = LN(msa) Wv^T (head-major), o_h = w_h v_h (pwa_ctr, fragment
// order), u = sigmoid(LN(msa) Wg^T) .* o, out = msa + keep / (1 - p) .* (u Wo^T). Given dres:
//   pwa_bglue : dout' = dres .* keep / (1 - p); per head: g (recomputed), du = dout' Wo_h, do = du g (head-major [h][i][(s, d)]),
//               dgp = du o g (1 - g) ([T][256]), dWo += dout'^T (g o) (in registers: warp h owns dWo[:, 32 h ..])
//   (pwa_ctr with transposed A) dv = w^T do;  (cuBLAS) dw = do v^T
//   pwa_bproj : dy = dgp Wg + dv Wv, dWg / dWv, LayerNorm backward (+ dres) -> dmsa, dgamma_m, dbeta_m
//   pwa_bpair : dlogit = w (dw - sum_j w dw), dWb, LN_z backward -> dpair, dgamma_z, dbeta_z
#pragma once
#include "sm80_common.cuh"

namespace a100 {

DEVI uint32_t bswz64(int row, int gr) { return row * 64 + ((gr ^ ((row >> 1) & 3)) << 4); }
DEVI uint32_t bswz128(int row, int gr) { return row * 128 + ((gr ^ (row & 7)) << 4); }
DEVI uint32_t bswz512(int row, int gr) { return row * 512 + ((gr ^ (row & 7)) << 4); }

// ---------------------------------------------------------------------------------------------------- pwa_bglue
struct PwaBglueParams {
  const __nv_bfloat16* msa;   // [T, 64]
  const __nv_bfloat16* dres;  // [T, 64]
  const __nv_bfloat16* o;     // [T, 256] fragment order (pwa_ctr)
  const __nv_bfloat16* keep;  // [L, 64] or nullptr
  float scale;
  const __nv_bfloat16* wg;    // [256, 64] gamma folded
  const float* bg;            // [256]
  const __nv_bfloat16* wo;    // [64, 256]
  __nv_bfloat16* dout_h;      // [8][L][S * 32]  do, head-major
  __nv_bfloat16* dgp;         // [T, 256] natural order
  float* dwo;                 // [64, 256] accumulated
  int S, L;
  float eps;
};

// Persistent CTA of 8 warps; warp h = head h. Tile = 32 tokens: 32 consecutive s at one i (so do's head-major pieces are 2 KB
// contiguous). Staged (double-buffered cp.async): msa (normalized in place by all threads), dres (scaled in place by keep / (1 - p)),
// o. Resident: Wg', Wo, bg.
constexpr int PB_WG = 0, PB_WO = 256 * 128, PB_BG = PB_WO + 64 * 512, PB_BUF = PB_BG + 1024;
constexpr int PB_X = 0, PB_D = 32 * 128, PB_O = 2 * 32 * 128, PB_SLOT = PB_O + 32 * 512;        // 24 KB per slot
constexpr int PB_DGP = PB_BUF + 2 * PB_SLOT, PB_WS = PB_DGP + 32 * 512;                          // + 16 KB dgp staging
constexpr int PB_WSZ = 2 * 32 * 64;                                                              // per warp: do + u staging, 4 KB
constexpr int PB_SMEM = PB_WS + 8 * PB_WSZ;

__global__ void __launch_bounds__(256, 1) pwa_bglue_kernel(PwaBglueParams p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sb = smem_u32(smem);
  const int tid = threadIdx.x, lane = tid & 31, h = tid >> 5;
  const int g = lane >> 2, q = lane & 3, l4 = lane >> 4;
  const int L = p.L, nsb = p.S / 32, ntiles = L * nsb;
  const size_t ldh = (size_t)p.S * 32;
  for (int idx = tid; idx < 2048; idx += 256) {
    const int row = idx >> 3, gr = idx & 7;
    cp_async16(sb + PB_WG + bswz128(row, gr), p.wg + row * 64 + gr * 8);
    const int ro = idx >> 5, go = idx & 31;
    cp_async16(sb + PB_WO + bswz512(ro, go), p.wo + ro * 256 + go * 8);
  }
  float* sbg = reinterpret_cast<float*>(smem + PB_BG);
  for (int idx = tid; idx < 256; idx += 256) sbg[idx] = p.bg[idx];

  auto issue = [&](int tile, int slot) {
    const int i = tile % L, s0 = (tile / L) * 32;
    const uint32_t st = sb + PB_BUF + slot * PB_SLOT;
    const int row = tid >> 3, gr = tid & 7;
    const size_t t = (size_t)(s0 + row) * L + i;
    cp_async16(st + PB_X + bswz128(row, gr), p.msa + t * 64 + gr * 8);
    cp_async16(st + PB_D + bswz128(row, gr), p.dres + t * 64 + gr * 8);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int idx = tid + 256 * k, r = idx >> 5, go = idx & 31;
      cp_async16(st + PB_O + bswz512(r, go), p.o + ((size_t)(s0 + r) * L + i) * 256 + go * 8);
    }
  };
  float dw[4][4][4];
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) dw[a][b][c] = 0.f;

  int tile = blockIdx.x, slot = 0;
  if (tile < ntiles) issue(tile, 0);
  cp_async_commit();
  const uint32_t wsd = sb + PB_WS + h * PB_WSZ, wsu = wsd + 32 * 64;
  for (; tile < ntiles; tile += gridDim.x, slot ^= 1) {
    const int i = tile % L, s0 = (tile / L) * 32;
    cp_async_wait<0>();
    __syncthreads();                                  // this slot has landed; every warp is done with the other slot and the stagings
    if (tile + (int)gridDim.x < ntiles) issue(tile + gridDim.x, slot ^ 1);
    cp_async_commit();
    const uint32_t st = sb + PB_BUF + slot * PB_SLOT;
    {   // prep: LN of the msa rows in place (8 threads per row), dres scaled by the keep-mask row of token i
      const int row = tid >> 3, gr = tid & 7;
      uint4 xv = lds128(st + PB_X + bswz128(row, gr));
      uint32_t* u = reinterpret_cast<uint32_t*>(&xv);
      float sum = 0.f;
#pragma unroll
      for (int e = 0; e < 4; ++e) sum += bf16lo(u[e]) + bf16hi(u[e]);
      sum += __shfl_xor_sync(0xffffffffu, sum, 1); sum += __shfl_xor_sync(0xffffffffu, sum, 2); sum += __shfl_xor_sync(0xffffffffu, sum, 4);
      const float mu = sum * (1.f / 64);
      float var = 0.f;
#pragma unroll
      for (int e = 0; e < 4; ++e) { const float a = bf16lo(u[e]) - mu, b = bf16hi(u[e]) - mu; var += a * a + b * b; }
      var += __shfl_xor_sync(0xffffffffu, var, 1); var += __shfl_xor_sync(0xffffffffu, var, 2); var += __shfl_xor_sync(0xffffffffu, var, 4);
      const float r = rsqrtf(var * (1.f / 64) + p.eps);
#pragma unroll
      for (int e = 0; e < 4; ++e) u[e] = pack_bf16((bf16lo(u[e]) - mu) * r, (bf16hi(u[e]) - mu) * r);
      sts128(st + PB_X + bswz128(row, gr), xv);
      if (p.keep) {
        uint4 dv = lds128(st + PB_D + bswz128(row, gr));
        const uint4 kv = __ldg(reinterpret_cast<const uint4*>(p.keep + (size_t)i * 64) + gr);
        uint32_t* d = reinterpret_cast<uint32_t*>(&dv);
        const uint32_t* k = reinterpret_cast<const uint32_t*>(&kv);
#pragma unroll
        for (int e = 0; e < 4; ++e) d[e] = pack_bf16(bf16lo(d[e]) * bf16lo(k[e]) * p.scale, bf16hi(d[e]) * bf16hi(k[e]) * p.scale);
        sts128(st + PB_D + bswz128(row, gr), dv);
      }
    }
    __syncthreads();
    uint32_t xf[2][4][4], df[2][4][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int kc = 0; kc < 4; ++kc) {
        ldsm_x4(xf[mt][kc], st + PB_X + bswz128(16 * mt + (lane & 15), 2 * kc + l4));
        ldsm_x4(df[mt][kc], st + PB_D + bswz128(16 * mt + (lane & 15), 2 * kc + l4));
      }
    // gate and du for head h: rows = tokens (s), cols = d
    float ga[2][4][4], da[2][4][4];
#pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
      const float2 b = *reinterpret_cast<const float2*>(sbg + 32 * h + 8 * dt + 2 * q);
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        ga[mt][dt][0] = ga[mt][dt][2] = b.x; ga[mt][dt][1] = ga[mt][dt][3] = b.y;
        da[mt][dt][0] = da[mt][dt][1] = da[mt][dt][2] = da[mt][dt][3] = 0.f;
      }
    }
#pragma unroll
    for (int kc = 0; kc < 4; ++kc)
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        uint32_t bgf[4], bof[4];
        ldsm_x4(bgf, sb + PB_WG + bswz128(32 * h + 16 * np + (lane & 7) + (l4 << 3), 2 * kc + ((lane >> 3) & 1)));
        ldsm_x4_t(bof, sb + PB_WO + bswz512(16 * kc + (lane & 7) + (((lane >> 3) & 1) << 3), 4 * h + 2 * np + l4));
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          mma16816(ga[mt][2 * np], xf[mt][kc], bgf[0], bgf[1]);
          mma16816(ga[mt][2 * np + 1], xf[mt][kc], bgf[2], bgf[3]);
          mma16816(da[mt][2 * np], df[mt][kc], bof[0], bof[1]);
          mma16816(da[mt][2 * np + 1], df[mt][kc], bof[2], bof[3]);
        }
      }
    // o (fragment order: lane q's 16 B of head h = dt 0..3), do / dgp / u
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int row = 16 * mt + g + 8 * hh;
        const uint4 ov = lds128(st + PB_O + bswz512(row, 4 * h + q));
        const uint32_t oo[4] = {ov.x, ov.y, ov.z, ov.w};
#pragma unroll
        for (int dt = 0; dt < 4; ++dt) {
          float dd[2], gp[2], uu[2];
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float gv = sigmoid(ga[mt][dt][2 * hh + e]), ovf = e ? bf16hi(oo[dt]) : bf16lo(oo[dt]), du = da[mt][dt][2 * hh + e];
            dd[e] = du * gv;
            gp[e] = du * ovf * gv * (1.f - gv);
            uu[e] = gv * ovf;
          }
          sts32(wsd + bswz64(row, dt) + 4 * q, pack_bf16(dd[0], dd[1]));
          sts32(wsu + bswz64(row, dt) + 4 * q, pack_bf16(uu[0], uu[1]));
          sts32(sb + PB_DGP + bswz512(row, 4 * h + dt) + 4 * q, pack_bf16(gp[0], gp[1]));
        }
      }
    __syncwarp();
    {   // do -> head-major rows (h, i), 32 s x 64 B = 2 KB contiguous
      __nv_bfloat16* dst = p.dout_h + ((size_t)h * L + i) * ldh + (size_t)s0 * 32;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int idx = lane + 32 * k, row = idx >> 2, gr = idx & 3;
        stg128(dst + row * 32 + gr * 8, lds128(wsd + bswz64(row, gr)));
      }
    }
    // dWo[:, 32 h ..] += dout'^T u : A[m = c][k = t] from the dres tile (ldmatrix.trans), B[k = t][n = d] from the u staging (.trans)
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      uint32_t bu[2][4];
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        const int row = 16 * kk + (lane & 7) + (((lane >> 3) & 1) << 3);
        ldsm_x4_t(bu[np], wsu + bswz64(row, 2 * np + l4));
      }
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) {
        uint32_t af[4];
        const int mat = lane >> 3, row = 16 * kk + 8 * (mat >> 1) + (lane & 7);
        ldsm_x4_t(af, st + PB_D + bswz128(row, 2 * mt + (mat & 1)));
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) mma16816(dw[mt][nt], af, bu[nt >> 1][(nt & 1) * 2], bu[nt >> 1][(nt & 1) * 2 + 1]);
      }
    }
    __syncthreads();                                  // dgp staging complete
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int idx = tid + 256 * k, row = idx >> 5, gr = idx & 31;
      stg128(p.dgp + ((size_t)(s0 + row) * L + i) * 256 + gr * 8, lds128(sb + PB_DGP + bswz512(row, gr)));
    }
  }
  cp_async_wait<0>();
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int c = 16 * mt + g + 8 * hh;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int d = 32 * h + 8 * nt + 2 * q;
        atomicAdd(p.dwo + c * 256 + d, dw[mt][nt][2 * hh]);
        atomicAdd(p.dwo + c * 256 + d + 1, dw[mt][nt][2 * hh + 1]);
      }
    }
}

// ---------------------------------------------------------------------------------------------------- pwa_bproj
struct PwaBprojParams {
  const __nv_bfloat16* msa;   // [T, 64]
  const __nv_bfloat16* dres;  // [T, 64]
  const __nv_bfloat16* dgp;   // [T, 256]
  const __nv_bfloat16* dv;    // [T, 256]
  const __nv_bfloat16* w;     // [512, 64]: rows 0..255 to_gate, 256..511 to_value (raw)
  const int* posinv;          // [L] or nullptr: with key compaction pwa_ctr_dv writes no dv for masked keys j (it is 0)
  int L;
  const float* gamma;         // [64]
  const float* beta;          // [64]
  __nv_bfloat16* dmsa;        // [T, 64]
  float* dw;                  // [512, 64] accumulated
  float* dgamma;              // [64]
  float* dbeta;               // [64]
  int T;
  float eps;
};

// CTA of 8 warps over 64-token tiles (single-buffered). Tile row = [dgp 512 B | dv 512 B] (1 KB, granules XOR row & 7).
//  A: warp w = (m-tile w & 3, K half w >> 2): partial dy over 256 of the 512 inputs; the upper half's partials go through smem.
//  LN: warps 0..3 finish dy, recompute x^ / y = x^ gamma + beta (y -> smem), LayerNorm backward + dres -> dmsa, dgamma / dbeta.
//  B: warp w owns dW rows 64 w .. + 63 over the tile's 64 tokens (both operands ldmatrix.trans).
constexpr int BP_W = 0, BP_G = 512 * 128, BP_X = BP_G + 64 * 1024, BP_Y = BP_X + 64 * 128, BP_R = BP_Y + 64 * 128;
constexpr int BP_GB = BP_R + 4 * 16 * 64 * 4, BP_SMEM = BP_GB + 2 * 64 * 4;

DEVI uint32_t bswz1k(int row, int gr) { return row * 1024 + ((gr ^ (row & 7)) << 4); }

__global__ void __launch_bounds__(256, 1) pwa_bproj_kernel(PwaBprojParams p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sb = smem_u32(smem);
  float* red = reinterpret_cast<float*>(smem + BP_R);
  float* sgam = reinterpret_cast<float*>(smem + BP_GB);
  float* sbet = sgam + 64;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, q = lane & 3, l4 = lane >> 4, mt = warp & 3, kh = warp >> 2;
  for (int idx = tid; idx < 512 * 8; idx += 256) {
    const int row = idx >> 3, gr = idx & 7;
    cp_async16(sb + BP_W + bswz128(row, gr), p.w + row * 64 + gr * 8);
  }
  if (tid < 64) { sgam[tid] = p.gamma[tid]; sbet[tid] = p.beta[tid]; }
  cp_async_commit();

  float dwa[4][8][4];
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 8; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) dwa[a][b][c] = 0.f;
  float dg[8][2], dbt[8][2];                      // warps 0..3: per (nt, col pair), summed over this lane's two rows
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) { dg[nt][0] = dg[nt][1] = dbt[nt][0] = dbt[nt][1] = 0.f; }

  const int ntiles = p.T / 64;
  const uint32_t a_off = bswz1k(16 * mt + (lane & 15), 32 * kh + l4);                 // ^ (kk << 5)
  for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
    const size_t t0 = (size_t)tile * 64;
    // key validity of the tile's 64 tokens (L % 64 == 0: a tile never wraps a row of the MSA), as a warp-uniform bitmask
    uint64_t valid = ~0ull;
    if (p.posinv) {
      const int j0 = (int)(t0 % (size_t)p.L);
      const uint32_t lo = __ballot_sync(0xffffffffu, __ldg(p.posinv + j0 + lane) >= 0);
      const uint32_t hi = __ballot_sync(0xffffffffu, __ldg(p.posinv + j0 + 32 + lane) >= 0);
      valid = ((uint64_t)hi << 32) | lo;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 16; ++k) {                 // dgp | dv: 64 rows x 64 granules
      const int idx = tid + 256 * k, row = idx >> 6, gr = idx & 63;
      const __nv_bfloat16* src = (gr < 32 ? p.dgp : p.dv) + (t0 + row) * 256 + (gr & 31) * 8;
      // with key compaction pwa_ctr_dv writes no dv row for a masked key: zero-fill it in the copy (src_bytes 0)
      const bool zero = gr >= 32 && !((valid >> row) & 1ull);
      cp_async16(sb + BP_G + bswz1k(row, gr), src, zero ? 0u : 16u);
    }
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int idx = tid + 256 * k, row = idx >> 3, gr = idx & 7;
      cp_async16(sb + BP_X + bswz128(row, gr), p.msa + (t0 + row) * 64 + gr * 8);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    // ---- A: partial dy[16 tokens][64] over inputs 256 kh .. + 255
    float dy[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) dy[nt][e] = 0.f;
#pragma unroll 4
    for (int kk = 0; kk < 16; ++kk) {
      uint32_t af[4];
      ldsm_x4(af, sb + BP_G + (a_off ^ (kk << 5)));
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t bf[4];
        ldsm_x4_t(bf, sb + BP_W + bswz128(256 * kh + 16 * kk + (lane & 7) + (((lane >> 3) & 1) << 3), 2 * np + l4));
        mma16816(dy[2 * np], af, bf[0], bf[1]);
        mma16816(dy[2 * np + 1], af, bf[2], bf[3]);
      }
    }
    float* rm = red + mt * 16 * 64;
    if (kh == 1) {
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int c = 8 * nt + 2 * q;
        *reinterpret_cast<float2*>(rm + g * 64 + (c ^ (g << 3))) = make_float2(dy[nt][0], dy[nt][1]);
        *reinterpret_cast<float2*>(rm + (g + 8) * 64 + (c ^ (g << 3))) = make_float2(dy[nt][2], dy[nt][3]);
      }
    }
    __syncthreads();
    if (kh == 0) {
      // ---- dy complete; x^ from the msa fragments (A layout == accumulator layout), y -> smem, LayerNorm backward, dmsa
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int c = 8 * nt + 2 * q;
        const float2 a0 = *reinterpret_cast<const float2*>(rm + g * 64 + (c ^ (g << 3)));
        const float2 a1 = *reinterpret_cast<const float2*>(rm + (g + 8) * 64 + (c ^ (g << 3)));
        dy[nt][0] += a0.x; dy[nt][1] += a0.y; dy[nt][2] += a1.x; dy[nt][3] += a1.y;
      }
      const int wr = 16 * mt;
      uint32_t xf[4][4];
#pragma unroll
      for (int kc = 0; kc < 4; ++kc) ldsm_x4(xf[kc], sb + BP_X + bswz128(wr + (lane & 15), 2 * kc + l4));
      float xh[4][4][2];
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int kc = 0; kc < 4; ++kc)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          xh[kc][r][0] = bf16lo(xf[kc][r]); xh[kc][r][1] = bf16hi(xf[kc][r]);
          if (r & 1) s1 += xh[kc][r][0] + xh[kc][r][1]; else s0 += xh[kc][r][0] + xh[kc][r][1];
        }
      const float mu0 = quad_sum(s0) * (1.f / 64), mu1 = quad_sum(s1) * (1.f / 64);
      float v0 = 0.f, v1 = 0.f;
#pragma unroll
      for (int kc = 0; kc < 4; ++kc)
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float d = xh[kc][r][e] - ((r & 1) ? mu1 : mu0);
            if (r & 1) v1 += d * d; else v0 += d * d;
          }
      const float r0 = rsqrtf(quad_sum(v0) * (1.f / 64) + p.eps), r1 = rsqrtf(quad_sum(v1) * (1.f / 64) + p.eps);
#pragma unroll
      for (int kc = 0; kc < 4; ++kc)
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
          for (int e = 0; e < 2; ++e) xh[kc][r][e] = (xh[kc][r][e] - ((r & 1) ? mu1 : mu0)) * ((r & 1) ? r1 : r0);
#pragma unroll
      for (int kc = 0; kc < 4; ++kc)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const int col = 16 * kc + 8 * (r >> 1) + 2 * q, row = wr + g + 8 * (r & 1);
          sts32(sb + BP_Y + bswz128(row, col >> 3) + (col & 7) * 2,
                pack_bf16(fmaf(xh[kc][r][0], sgam[col], sbet[col]), fmaf(xh[kc][r][1], sgam[col + 1], sbet[col + 1])));
        }
      float m1[2] = {0.f, 0.f}, m2[2] = {0.f, 0.f};
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int col = 8 * nt + 2 * q + (e & 1), hh = e >> 1;
          const float x = xh[nt >> 1][2 * (nt & 1) + hh][e & 1];
          const float dx = dy[nt][e] * sgam[col];
          m1[hh] += dx; m2[hh] += dx * x;
          dg[nt][e & 1] += dy[nt][e] * x;
          dbt[nt][e & 1] += dy[nt][e];
        }
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) { m1[hh] = quad_sum(m1[hh]) * (1.f / 64); m2[hh] = quad_sum(m2[hh]) * (1.f / 64); }
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int c0 = 8 * nt + 2 * q, row = wr + g + 8 * hh;
          const float rsd = hh ? r1 : r0;
          const uint32_t rr = __ldg(reinterpret_cast<const unsigned int*>(p.dres + (t0 + row) * 64 + c0));
          float o2[2];
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float x = xh[nt >> 1][2 * (nt & 1) + hh][e];
            o2[e] = rsd * (dy[nt][2 * hh + e] * sgam[c0 + e] - m1[hh] - x * m2[hh]) + (e ? bf16hi(rr) : bf16lo(rr));
          }
          sts32(sb + BP_X + bswz128(row, nt) + 4 * q, pack_bf16(o2[0], o2[1]));
        }
    }
    __syncthreads();                                   // y complete, dmsa staged
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int idx = tid + 256 * k, row = idx >> 3, gr = idx & 7;
      stg128(p.dmsa + (t0 + row) * 64 + gr * 8, lds128(sb + BP_X + bswz128(row, gr)));
    }
    // ---- B: dW rows o = 64 warp + 16 m + .., K = 64 tokens
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      uint32_t bf[4][4];
#pragma unroll
      for (int np = 0; np < 4; ++np)
        ldsm_x4_t(bf[np], sb + BP_Y + bswz128(16 * kt + (lane & 7) + (((lane >> 3) & 1) << 3), 2 * np + l4));
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        uint32_t af[4];
        const int mat = lane >> 3, row = 16 * kt + 8 * (mat >> 1) + (lane & 7);
        ldsm_x4_t(af, sb + BP_G + bswz1k(row, 8 * warp + 2 * m + (mat & 1)));
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) mma16816(dwa[m][nt], af, bf[nt >> 1][(nt & 1) * 2], bf[nt >> 1][(nt & 1) * 2 + 1]);
      }
    }
  }
#pragma unroll
  for (int m = 0; m < 4; ++m)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int o = 64 * warp + 16 * m + g + 8 * hh;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        atomicAdd(p.dw + o * 64 + 8 * nt + 2 * q, dwa[m][nt][2 * hh]);
        atomicAdd(p.dw + o * 64 + 8 * nt + 2 * q + 1, dwa[m][nt][2 * hh + 1]);
      }
    }
  if (kh == 0) {
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        float a = dg[nt][e], b = dbt[nt][e];
#pragma unroll
        for (int o = 4; o < 32; o <<= 1) { a += __shfl_xor_sync(0xffffffffu, a, o); b += __shfl_xor_sync(0xffffffffu, b, o); }
        if (g == 0) { atomicAdd(p.dgamma + 8 * nt + 2 * q + e, a); atomicAdd(p.dbeta + 8 * nt + 2 * q + e, b); }
      }
  }
}

// ---------------------------------------------------------------------------------------------------- pwa_bpair
struct PwaBpairParams {
  const __nv_bfloat16* z;     // [L*L, 128]
  const __nv_bfloat16* w;     // [8, L, L] (compacted: [.., k] for the keys of idx)
  const float* dw;            // [ksplit][8, L, L] fp32 split-K partials of pwa_dw
  int ksplit;
  const int* posinv;          // [L] key j -> compacted k (-1: masked), or nullptr (dense)
  const int* cnt;             // [2] n, n_pad or nullptr
  const uint8_t* mask;        // [L] key mask or nullptr: a masked key's logit is a constant in the module -> no gradient to z
                              // (this also covers an all-masked row, where compaction lists every key and w is uniform)
  const float* wb;            // [8, 128] raw
  const float* gamma;         // [128]
  const float* beta;          // [128]
  __nv_bfloat16* dz;          // [L*L, 128]
  float* dwb;                 // [8, 128] accumulated
  float* dgamma;              // [128]
  float* dbeta;               // [128]
  int L;
  float eps;
};

// One CTA (8 warps) per pair row i. Warp h: dlogit[h][j] = w (dw - sum_j w dw) -> smem. Then warps walk j: lane = 4 channels;
// LN_z recomputed (warp sums), dzy = dlogit[:, j] . Wb, LayerNorm backward -> dz; dWb, dgamma, dbeta in registers.
__global__ void __launch_bounds__(256) pwa_bpair_kernel(PwaBpairParams p) {
  extern __shared__ __align__(16) uint8_t smem[];
  float* dl = reinterpret_cast<float*>(smem);            // [8][L]
  float* redb = dl + 8 * p.L;                            // [8 warps][8 h + 2][128]
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int L = p.L, i = blockIdx.x;
  const int n = p.cnt ? __ldg(p.cnt) : L;
  {
    const __nv_bfloat16* wr = p.w + ((size_t)warp * L + i) * L;
    const size_t pstride = (size_t)8 * L * L;
    const float* dr = p.dw + ((size_t)warp * L + i) * L;
    float s = 0.f;
    for (int k = lane; k < n; k += 32) {
      float d = 0.f;
      for (int sp = 0; sp < p.ksplit; ++sp) d += dr[sp * pstride + k];
      dl[warp * L + k] = d;
      s += __bfloat162float(wr[k]) * d;
    }
    s = warp_sum(s);
    for (int k = lane; k < n; k += 32) dl[warp * L + k] = __bfloat162float(wr[k]) * (dl[warp * L + k] - s);
  }
  __syncthreads();
  const int c0 = 4 * lane;
  float wb[8][4], ga[4], be[4];
#pragma unroll
  for (int hh = 0; hh < 8; ++hh)
#pragma unroll
    for (int e = 0; e < 4; ++e) wb[hh][e] = p.wb[hh * 128 + c0 + e];
#pragma unroll
  for (int e = 0; e < 4; ++e) { ga[e] = p.gamma[c0 + e]; be[e] = p.beta[c0 + e]; }
  float awb[8][4], ag[4], ab[4];
#pragma unroll
  for (int hh = 0; hh < 8; ++hh)
#pragma unroll
    for (int e = 0; e < 4; ++e) awb[hh][e] = 0.f;
#pragma unroll
  for (int e = 0; e < 4; ++e) { ag[e] = 0.f; ab[e] = 0.f; }
  // j walk: 4 keys per warp iteration, their z rows requested before any is used (one dependent load per key was long-scoreboard bound)
  constexpr int JU = 4;
  for (int j0 = warp * JU; j0 < L; j0 += 8 * JU) {
    uint2 zv[JU];
    int kp[JU];
#pragma unroll
    for (int u = 0; u < JU; ++u) {
      kp[u] = p.posinv ? __ldg(p.posinv + j0 + u) : j0 + u;
      if (p.mask && !p.mask[j0 + u]) kp[u] = -1;
      zv[u] = kp[u] >= 0 ? *reinterpret_cast<const uint2*>(p.z + ((size_t)i * L + j0 + u) * 128 + c0) : make_uint2(0u, 0u);
    }
#pragma unroll
    for (int u = 0; u < JU; ++u) {
      const int j = j0 + u;
      float d8[8];
      bool any = false;
#pragma unroll
      for (int hh = 0; hh < 8; ++hh) { d8[hh] = kp[u] >= 0 ? dl[hh * L + kp[u]] : 0.f; any |= d8[hh] != 0.f; }
      const size_t base = ((size_t)i * L + j) * 128 + c0;
      if (!any) {                                         // masked key (w = 0): no gradient
        *reinterpret_cast<uint2*>(p.dz + base) = make_uint2(0u, 0u);
        continue;
      }
      float x[4] = {bf16lo(zv[u].x), bf16hi(zv[u].x), bf16lo(zv[u].y), bf16hi(zv[u].y)};
      const float mu = warp_sum(x[0] + x[1] + x[2] + x[3]) * (1.f / 128);
      float vs = 0.f;
#pragma unroll
      for (int e = 0; e < 4; ++e) { x[e] -= mu; vs += x[e] * x[e]; }
      const float rs = rsqrtf(warp_sum(vs) * (1.f / 128) + p.eps);
      float dzy[4], m1 = 0.f, m2 = 0.f;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        x[e] *= rs;                                       // x^
        const float zy = fmaf(x[e], ga[e], be[e]);
        float d = 0.f;
#pragma unroll
        for (int hh = 0; hh < 8; ++hh) { d = fmaf(d8[hh], wb[hh][e], d); awb[hh][e] = fmaf(d8[hh], zy, awb[hh][e]); }
        dzy[e] = d;
        ag[e] = fmaf(d, x[e], ag[e]);
        ab[e] += d;
        const float dx = d * ga[e];
        m1 += dx; m2 += dx * x[e];
      }
      m1 = warp_sum(m1) * (1.f / 128);
      m2 = warp_sum(m2) * (1.f / 128);
      float o[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) o[e] = rs * (dzy[e] * ga[e] - m1 - x[e] * m2);
      *reinterpret_cast<uint2*>(p.dz + base) = make_uint2(pack_bf16(o[0], o[1]), pack_bf16(o[2], o[3]));
    }
  }
  // reduce the 8 warps' partials, then atomics
#pragma unroll
  for (int hh = 0; hh < 8; ++hh)
#pragma unroll
    for (int e = 0; e < 4; ++e) redb[(warp * 10 + hh) * 128 + c0 + e] = awb[hh][e];
#pragma unroll
  for (int e = 0; e < 4; ++e) { redb[(warp * 10 + 8) * 128 + c0 + e] = ag[e]; redb[(warp * 10 + 9) * 128 + c0 + e] = ab[e]; }
  __syncthreads();
  for (int idx = tid; idx < 10 * 128; idx += 256) {
    float s = 0.f;
#pragma unroll
    for (int wv = 0; wv < 8; ++wv) s += redb[wv * 10 * 128 + idx];
    const int r = idx >> 7, c = idx & 127;
    if (r < 8) atomicAdd(p.dwb + r * 128 + c, s);
    else if (r == 8) atomicAdd(p.dgamma + c, s);
    else atomicAdd(p.dbeta + c, s);
  }
}

// ---------------------------------------------------------------------------------------------------- pwa_dw
// dw partial [split][h][i][k] (fp32) = sum over K range `split` of do_h[i][K] v[h][k][K], K = S * 32 (both operands K-contiguous: an NT
// GEMM). CTA 128 i x 128 k, 4 warps of 64 x 64 (the pwa_ctr shape), 64-deep K chunks through a 2-stage cp.async ring, 2 CTAs / SM.
// The split-K partials are summed by pwa_bpair (no atomics). With compaction, k-blocks past n_pad exit.
struct PwaDwParams {
  const __nv_bfloat16* a;     // do_h [8][L][K]
  const __nv_bfloat16* b;     // v (compacted rows) [8][L][K]
  float* out;                 // [ksplit][8][L][L]
  const int* cnt;             // [2] or nullptr
  int L, K, ksplit;
};
constexpr int DW_STAGE = 2 * 128 * 128, DW_SMEM = 2 * DW_STAGE;

__global__ void __launch_bounds__(128, 2) pwa_dw_kernel(PwaDwParams p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sb = smem_u32(smem);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, q = lane & 3, wm = warp & 1, wn = warp >> 1, l4 = lane >> 4;
  const int nb = p.L / 128;
  int x = blockIdx.x;
  const int ib = x % nb; x /= nb;
  const int kb = x % nb; x /= nb;
  const int split = x % p.ksplit; x /= p.ksplit;
  const int h = x;
  const int i0 = ib * 128, k0 = kb * 128;
  if (p.cnt && k0 >= __ldg(p.cnt + 1)) return;
  const int krange = p.K / p.ksplit, nch = krange / 64;
  const size_t ka = (size_t)split * krange;
  // copy slots: granule (tid & 7) of rows (tid >> 3) + 16 k, k = 0..7 (both operands, 128 B rows)
  const uint32_t dst = bswz128(tid >> 3, tid & 7);
  const __nv_bfloat16* asrc = p.a + ((size_t)h * p.L + i0 + (tid >> 3)) * p.K + ka + (tid & 7) * 8;
  const __nv_bfloat16* bsrc = p.b + ((size_t)h * p.L + k0 + (tid >> 3)) * p.K + ka + (tid & 7) * 8;
  const size_t rstep = (size_t)16 * p.K;
  auto issue = [&](int c, int stage) {
    const uint32_t st = sb + stage * DW_STAGE;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      cp_async16(st + dst + k * 16 * 128, asrc + k * rstep + (size_t)c * 64);
      cp_async16(st + 128 * 128 + dst + k * 16 * 128, bsrc + k * rstep + (size_t)c * 64);
    }
  };
  uint32_t a_off[4], b_off[4];
#pragma unroll
  for (int mt = 0; mt < 4; ++mt) a_off[mt] = bswz128(64 * wm + 16 * mt + (lane & 15), l4);                     // ^ (kk << 5)
#pragma unroll
  for (int np = 0; np < 4; ++np) b_off[np] = 128 * 128 + bswz128(64 * wn + 16 * np + (lane & 7) + (l4 << 3), (lane >> 3) & 1);
  float acc[4][8][4];
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 8; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
  issue(0, 0);
  cp_async_commit();
  for (int c = 0; c < nch; ++c) {
    cp_async_wait<0>();
    __syncthreads();
    if (c + 1 < nch) issue(c + 1, (c + 1) & 1);
    cp_async_commit();
    const uint32_t st = sb + (c & 1) * DW_STAGE;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      uint32_t af[4][4], bf[4][4];
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) ldsm_x4(af[mt], st + (a_off[mt] ^ (kk << 5)));
#pragma unroll
      for (int np = 0; np < 4; ++np) ldsm_x4(bf[np], st + (b_off[np] ^ (kk << 5)));
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) mma16816(acc[mt][nt], af[mt], bf[nt >> 1][(nt & 1) * 2], bf[nt >> 1][(nt & 1) * 2 + 1]);
    }
  }
  float* o = p.out + (((size_t)split * 8 + h) * p.L) * p.L;
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int i = i0 + 64 * wm + 16 * mt + g + 8 * hh;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
        *reinterpret_cast<float2*>(o + (size_t)i * p.L + k0 + 64 * wn + 8 * nt + 2 * q) = make_float2(acc[mt][nt][2 * hh], acc[mt][nt][2 * hh + 1]);
    }
}

// ---------------------------------------------------------------------------------------------------- pwa_bglue2 (dgp consumed in place)
// As pwa_bglue, but the gate's gradient never goes to memory: warp h also computes dWg[32 h ..] += dgp_h^T y and the gate's share of
// dy, dy_g = dgp Wg (K = all 256 gate channels), written as bf16 [T][64] (128 B / token instead of dgp's 512 B written and re-read by
// the projection backward). dgp is staged per tile [32 tokens][256] in smem; dy_g is one small GEMM over it split across the 8 warps
// (v1 summed per-head partials with shared fp32 atomics: sm_80 runs those as CAS loops -- 3x slower kernel). The gate is recomputed from y = x^ gamma + beta with the RAW Wg (no bias), so
// dy_g is the gradient w.r.t. y itself -- what the LayerNorm backward needs for dgamma / dbeta.
struct PwaBglue2Params {
  const __nv_bfloat16* msa;   // [T, 64]
  const __nv_bfloat16* dres;  // [T, 64]
  const __nv_bfloat16* o;     // [T, 256] fragment order (pwa_ctr)
  const __nv_bfloat16* keep;  // [L, 64] or nullptr
  float scale;
  const __nv_bfloat16* wg;    // [256, 64] RAW to_gate.weight
  const float* gamma;         // [64] ln_msa
  const float* beta;          // [64]
  const __nv_bfloat16* wo;    // [64, 256]
  __nv_bfloat16* dout_h;      // [8][L][S * 32]
  __nv_bfloat16* dyg;         // [T, 64]  gate part of dL/dy
  float* dwo;                 // [64, 256] accumulated
  float* dwg;                 // [256, 64] accumulated
  int S, L;
  float eps;
};
constexpr int PB2_WG = 0, PB2_WO = 256 * 128, PB2_GB = PB2_WO + 64 * 512, PB2_BUF = PB2_GB + 512;
constexpr int PB2_SLOT = PB_SLOT;                                                  // y 4 KB | dout' 4 KB | o 16 KB
constexpr int PB2_DGP = PB2_BUF + 2 * PB2_SLOT;                                    // [32 tokens][256] bf16 (512 B rows, swizzled)
constexpr int PB2_WS = PB2_DGP + 32 * 512;
constexpr int PB2_WSZ = 2 * 32 * 64;                                               // per warp: do | u
constexpr int PB2_SMEM = PB2_WS + 8 * PB2_WSZ;


__global__ void __launch_bounds__(256, 1) pwa_bglue2_kernel(PwaBglue2Params p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sb = smem_u32(smem);
  const int tid = threadIdx.x, lane = tid & 31, h = tid >> 5;
  const int g = lane >> 2, q = lane & 3, l4 = lane >> 4;
  const int L = p.L, nsb = p.S / 32, ntiles = L * nsb;
  const size_t ldh = (size_t)p.S * 32;
  for (int idx = tid; idx < 2048; idx += 256) {
    const int row = idx >> 3, gr = idx & 7;
    cp_async16(sb + PB2_WG + bswz128(row, gr), p.wg + row * 64 + gr * 8);
    const int ro = idx >> 5, go = idx & 31;
    cp_async16(sb + PB2_WO + bswz512(ro, go), p.wo + ro * 256 + go * 8);
  }
  float* sgam = reinterpret_cast<float*>(smem + PB2_GB);
  float* sbet = sgam + 64;
  if (tid < 64) { sgam[tid] = p.gamma[tid]; sbet[tid] = p.beta[tid]; }

  auto issue = [&](int tile, int slot) {
    const int i = tile % L, s0 = (tile / L) * 32;
    const uint32_t st = sb + PB2_BUF + slot * PB2_SLOT;
    const int row = tid >> 3, gr = tid & 7;
    const size_t t = (size_t)(s0 + row) * L + i;
    cp_async16(st + PB_X + bswz128(row, gr), p.msa + t * 64 + gr * 8);
    cp_async16(st + PB_D + bswz128(row, gr), p.dres + t * 64 + gr * 8);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int idx = tid + 256 * k, r = idx >> 5, go = idx & 31;
      cp_async16(st + PB_O + bswz512(r, go), p.o + ((size_t)(s0 + r) * L + i) * 256 + go * 8);
    }
  };
  float dwo[4][4][4], dwg[2][8][4];
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) dwo[a][b][c] = 0.f;
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int b = 0; b < 8; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) dwg[a][b][c] = 0.f;

  int tile = blockIdx.x, slot = 0;
  if (tile < ntiles) issue(tile, 0);
  cp_async_commit();
  const uint32_t wsd = sb + PB2_WS + h * PB2_WSZ, wsx = wsd + 32 * 64;
  for (; tile < ntiles; tile += gridDim.x, slot ^= 1) {
    const int i = tile % L, s0 = (tile / L) * 32;
    cp_async_wait<0>();
    __syncthreads();
    if (tile + (int)gridDim.x < ntiles) issue(tile + gridDim.x, slot ^ 1);
    cp_async_commit();
    const uint32_t st = sb + PB2_BUF + slot * PB2_SLOT;
    {   // prep: y = LN(msa) gamma + beta in place (8 threads per row), dres scaled by the keep-mask row of token i
      const int row = tid >> 3, gr = tid & 7;
      uint4 xv = lds128(st + PB_X + bswz128(row, gr));
      uint32_t* u = reinterpret_cast<uint32_t*>(&xv);
      float sum = 0.f;
#pragma unroll
      for (int e = 0; e < 4; ++e) sum += bf16lo(u[e]) + bf16hi(u[e]);
      sum += __shfl_xor_sync(0xffffffffu, sum, 1); sum += __shfl_xor_sync(0xffffffffu, sum, 2); sum += __shfl_xor_sync(0xffffffffu, sum, 4);
      const float mu = sum * (1.f / 64);
      float var = 0.f;
#pragma unroll
      for (int e = 0; e < 4; ++e) { const float a = bf16lo(u[e]) - mu, b = bf16hi(u[e]) - mu; var += a * a + b * b; }
      var += __shfl_xor_sync(0xffffffffu, var, 1); var += __shfl_xor_sync(0xffffffffu, var, 2); var += __shfl_xor_sync(0xffffffffu, var, 4);
      const float r = rsqrtf(var * (1.f / 64) + p.eps);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int c = gr * 8 + 2 * e;
        u[e] = pack_bf16(fmaf((bf16lo(u[e]) - mu) * r, sgam[c], sbet[c]), fmaf((bf16hi(u[e]) - mu) * r, sgam[c + 1], sbet[c + 1]));
      }
      sts128(st + PB_X + bswz128(row, gr), xv);
      if (p.keep) {
        uint4 dv = lds128(st + PB_D + bswz128(row, gr));
        const uint4 kv = __ldg(reinterpret_cast<const uint4*>(p.keep + (size_t)i * 64) + gr);
        uint32_t* d = reinterpret_cast<uint32_t*>(&dv);
        const uint32_t* k = reinterpret_cast<const uint32_t*>(&kv);
#pragma unroll
        for (int e = 0; e < 4; ++e) d[e] = pack_bf16(bf16lo(d[e]) * bf16lo(k[e]) * p.scale, bf16hi(d[e]) * bf16hi(k[e]) * p.scale);
        sts128(st + PB_D + bswz128(row, gr), dv);
      }
    }
    __syncthreads();
    float gp[2][4][4];                                // dgp of this warp's head, both m-tiles (kept until staged)
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      uint32_t xf[4][4], df[4][4];
#pragma unroll
      for (int kc = 0; kc < 4; ++kc) {
        ldsm_x4(xf[kc], st + PB_X + bswz128(16 * mt + (lane & 15), 2 * kc + l4));
        ldsm_x4(df[kc], st + PB_D + bswz128(16 * mt + (lane & 15), 2 * kc + l4));
      }
      float ga[4][4], da[4][4];
#pragma unroll
      for (int dt = 0; dt < 4; ++dt)
#pragma unroll
        for (int e = 0; e < 4; ++e) { ga[dt][e] = 0.f; da[dt][e] = 0.f; }
#pragma unroll
      for (int kc = 0; kc < 4; ++kc)
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          uint32_t bgf[4], bof[4];
          ldsm_x4(bgf, sb + PB2_WG + bswz128(32 * h + 16 * np + (lane & 7) + (l4 << 3), 2 * kc + ((lane >> 3) & 1)));
          ldsm_x4_t(bof, sb + PB2_WO + bswz512(16 * kc + (lane & 7) + (((lane >> 3) & 1) << 3), 4 * h + 2 * np + l4));
          mma16816(ga[2 * np], xf[kc], bgf[0], bgf[1]);
          mma16816(ga[2 * np + 1], xf[kc], bgf[2], bgf[3]);
          mma16816(da[2 * np], df[kc], bof[0], bof[1]);
          mma16816(da[2 * np + 1], df[kc], bof[2], bof[3]);
        }
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int row = 16 * mt + g + 8 * hh;
        const uint4 ov = lds128(st + PB_O + bswz512(row, 4 * h + q));
        const uint32_t oo[4] = {ov.x, ov.y, ov.z, ov.w};
#pragma unroll
        for (int dt = 0; dt < 4; ++dt) {
          float dd[2], uu[2];
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float gv = sigmoid(ga[dt][2 * hh + e]), ovf = e ? bf16hi(oo[dt]) : bf16lo(oo[dt]), du = da[dt][2 * hh + e];
            dd[e] = du * gv;
            gp[mt][dt][2 * hh + e] = du * ovf * gv * (1.f - gv);
            uu[e] = gv * ovf;
          }
          sts32(wsd + bswz64(row, dt) + 4 * q, pack_bf16(dd[0], dd[1]));
          sts32(wsx + bswz64(row, dt) + 4 * q, pack_bf16(uu[0], uu[1]));
        }
      }
    }
    __syncwarp();
    {   // do -> head-major rows (h, i), 32 s x 64 B = 2 KB contiguous
      __nv_bfloat16* dst = p.dout_h + ((size_t)h * L + i) * ldh + (size_t)s0 * 32;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int idx = lane + 32 * k, row = idx >> 2, gr = idx & 3;
        stg128(dst + row * 32 + gr * 8, lds128(wsd + bswz64(row, gr)));
      }
    }
    // dWo[:, 32 h ..] += dout'^T u
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      uint32_t bu[2][4];
#pragma unroll
      for (int np = 0; np < 2; ++np) ldsm_x4_t(bu[np], wsx + bswz64(16 * kk + (lane & 7) + (((lane >> 3) & 1) << 3), 2 * np + l4));
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) {
        uint32_t af[4];
        const int mat = lane >> 3, row = 16 * kk + 8 * (mat >> 1) + (lane & 7);
        ldsm_x4_t(af, st + PB_D + bswz128(row, 2 * mt + (mat & 1)));
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) mma16816(dwo[mt][nt], af, bu[nt >> 1][(nt & 1) * 2], bu[nt >> 1][(nt & 1) * 2 + 1]);
      }
    }
    // dgp of head h -> the shared [32][256] tile (natural order: channel 32 h + 8 dt + 2 q + e)
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int row = 16 * mt + g + 8 * hh;
#pragma unroll
        for (int dt = 0; dt < 4; ++dt)
          sts32(sb + PB2_DGP + bswz512(row, 4 * h + dt) + 4 * q, pack_bf16(gp[mt][dt][2 * hh], gp[mt][dt][2 * hh + 1]));
      }
    __syncthreads();                                  // the whole dgp tile is in
    // dWg[32 h + d][k] += dgp_h^T y : A[m = d][k = t] from the dgp tile (.trans), B[k = t][n] = y (.trans)
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      uint32_t by[4][4];
#pragma unroll
      for (int np = 0; np < 4; ++np) ldsm_x4_t(by[np], st + PB_X + bswz128(16 * kk + (lane & 7) + (((lane >> 3) & 1) << 3), 2 * np + l4));
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        uint32_t af[4];
        const int mat = lane >> 3, row = 16 * kk + 8 * (mat >> 1) + (lane & 7);
        ldsm_x4_t(af, sb + PB2_DGP + bswz512(row, 4 * h + 2 * mt + (mat & 1)));
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) mma16816(dwg[mt][nt], af, by[nt >> 1][(nt & 1) * 2], by[nt >> 1][(nt & 1) * 2 + 1]);
      }
    }
    // dy_g = dgp . Wg (K = 256): warp h takes tokens 16 (h & 1) .. + 15, columns 16 (h >> 1) .. + 15
    {
      const int mr = 16 * (h & 1), nc = 16 * (h >> 1);
      float dy[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
      const uint32_t aoff = bswz512(mr + (lane & 15), l4);                                   // ^ (kk << 5)
#pragma unroll 4
      for (int kk = 0; kk < 16; ++kk) {
        uint32_t af[4], bf[4];
        ldsm_x4(af, sb + PB2_DGP + (aoff ^ (kk << 5)));
        ldsm_x4_t(bf, sb + PB2_WG + bswz128(16 * kk + (lane & 7) + (((lane >> 3) & 1) << 3), (nc >> 3) + l4));
        mma16816(dy[0], af, bf[0], bf[1]);
        mma16816(dy[1], af, bf[2], bf[3]);
      }
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const size_t tok = (size_t)(s0 + mr + g + 8 * hh) * L + i;
          stg32(p.dyg + tok * 64 + nc + 8 * nt + 2 * q, pack_bf16(dy[nt][2 * hh], dy[nt][2 * hh + 1]));
        }
    }
    // the next tile's __syncthreads (after its wait) orders these reads of the dgp tile before it is rewritten
  }
  cp_async_wait<0>();
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int c = 16 * mt + g + 8 * hh;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int d = 32 * h + 8 * nt + 2 * q;
        atomicAdd(p.dwo + c * 256 + d, dwo[mt][nt][2 * hh]);
        atomicAdd(p.dwo + c * 256 + d + 1, dwo[mt][nt][2 * hh + 1]);
      }
    }
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int d = 32 * h + 16 * mt + g + 8 * hh;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        atomicAdd(p.dwg + d * 64 + 8 * nt + 2 * q, dwg[mt][nt][2 * hh]);
        atomicAdd(p.dwg + d * 64 + 8 * nt + 2 * q + 1, dwg[mt][nt][2 * hh + 1]);
      }
    }
}

// ---------------------------------------------------------------------------------------------------- pwa_bproj2 (value side only)
// dy = dy_g (from pwa_bglue2) + dv Wv, dWv += dv^T y, LayerNorm backward + dres -> dmsa, dgamma, dbeta. 64-token tiles, double-buffered
// (dv 32 KB + msa 8 KB per slot); 8 warps: A = (m-tile w & 3, K half w >> 2) over the 256 dv inputs, the upper half's partials through
// smem; LN by warps 0..3; B = dWv rows 32 w .. + 31 over the tile's 64 tokens.
struct PwaBproj2Params {
  const __nv_bfloat16* msa;   // [T, 64]
  const __nv_bfloat16* dres;  // [T, 64]
  const __nv_bfloat16* dyg;   // [T, 64]
  const __nv_bfloat16* dv;    // [T, 256]
  const __nv_bfloat16* w;     // [256, 64] RAW to_value.weight
  const float* gamma;
  const float* beta;
  const int* posinv;          // [L] or nullptr (compacted keys: dv of a masked key is 0, never written)
  __nv_bfloat16* dmsa;
  float* dw;                  // [256, 64]
  float* dgamma;
  float* dbeta;
  int T, L;
  float eps;
};
constexpr int BP2_W = 0, BP2_BUF = 256 * 128, BP2_G = 0, BP2_X = 64 * 512, BP2_SLOT = BP2_X + 64 * 128;   // 40 KB per slot
constexpr int BP2_Y = BP2_BUF + 2 * BP2_SLOT, BP2_R = BP2_Y + 64 * 128, BP2_GB = BP2_R + 4 * 16 * 64 * 4;
constexpr int BP2_SMEM = BP2_GB + 2 * 64 * 4;

__global__ void __launch_bounds__(256, 1) pwa_bproj2_kernel(PwaBproj2Params p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sb = smem_u32(smem);
  float* red = reinterpret_cast<float*>(smem + BP2_R);
  float* sgam = reinterpret_cast<float*>(smem + BP2_GB);
  float* sbet = sgam + 64;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, q = lane & 3, l4 = lane >> 4, mt = warp & 3, kh = warp >> 2;
  for (int idx = tid; idx < 256 * 8; idx += 256) {
    const int row = idx >> 3, gr = idx & 7;
    cp_async16(sb + BP2_W + bswz128(row, gr), p.w + row * 64 + gr * 8);
  }
  if (tid < 64) { sgam[tid] = p.gamma[tid]; sbet[tid] = p.beta[tid]; }
  const int ntiles = p.T / 64;
  auto issue = [&](int tile, int slot) {
    const size_t t0 = (size_t)tile * 64;
    uint64_t valid = ~0ull;
    if (p.posinv) {
      const int j0 = (int)(t0 % (size_t)p.L);
      const uint32_t lo = __ballot_sync(0xffffffffu, __ldg(p.posinv + j0 + lane) >= 0);
      const uint32_t hi = __ballot_sync(0xffffffffu, __ldg(p.posinv + j0 + 32 + lane) >= 0);
      valid = ((uint64_t)hi << 32) | lo;
    }
    const uint32_t st = sb + BP2_BUF + slot * BP2_SLOT;
#pragma unroll
    for (int k = 0; k < 8; ++k) {                 // dv: 64 rows x 32 granules
      const int idx = tid + 256 * k, row = idx >> 5, gr = idx & 31;
      cp_async16(st + BP2_G + bswz512(row, gr), p.dv + (t0 + row) * 256 + gr * 8, ((valid >> row) & 1ull) ? 16u : 0u);
    }
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int idx = tid + 256 * k, row = idx >> 3, gr = idx & 7;
      cp_async16(st + BP2_X + bswz128(row, gr), p.msa + (t0 + row) * 64 + gr * 8);
    }
  };
  float dwa[2][8][4];
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int b = 0; b < 8; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) dwa[a][b][c] = 0.f;
  float dg[8][2], dbt[8][2];
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) { dg[nt][0] = dg[nt][1] = dbt[nt][0] = dbt[nt][1] = 0.f; }
  const uint32_t a_off = bswz512(16 * mt + (lane & 15), 16 * kh + l4);                  // ^ (kk << 5)
  int tile = blockIdx.x, slot = 0;
  if (tile < ntiles) issue(tile, 0);
  cp_async_commit();
  for (; tile < ntiles; tile += gridDim.x, slot ^= 1) {
    const size_t t0 = (size_t)tile * 64;
    cp_async_wait<0>();
    __syncthreads();
    if (tile + (int)gridDim.x < ntiles) issue(tile + gridDim.x, slot ^ 1);
    cp_async_commit();
    const uint32_t st = sb + BP2_BUF + slot * BP2_SLOT;
    float dy[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) dy[nt][e] = 0.f;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      uint32_t af[4];
      ldsm_x4(af, st + BP2_G + (a_off ^ (kk << 5)));
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t bf[4];
        ldsm_x4_t(bf, sb + BP2_W + bswz128(128 * kh + 16 * kk + (lane & 7) + (((lane >> 3) & 1) << 3), 2 * np + l4));
        mma16816(dy[2 * np], af, bf[0], bf[1]);
        mma16816(dy[2 * np + 1], af, bf[2], bf[3]);
      }
    }
    float* rm = red + mt * 16 * 64;
    if (kh == 1) {
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int c = 8 * nt + 2 * q;
        *reinterpret_cast<float2*>(rm + g * 64 + (c ^ (g << 3))) = make_float2(dy[nt][0], dy[nt][1]);
        *reinterpret_cast<float2*>(rm + (g + 8) * 64 + (c ^ (g << 3))) = make_float2(dy[nt][2], dy[nt][3]);
      }
    }
    __syncthreads();
    if (kh == 0) {
      const int wr = 16 * mt;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int c = 8 * nt + 2 * q;
        const float2 a0 = *reinterpret_cast<const float2*>(rm + g * 64 + (c ^ (g << 3)));
        const float2 a1 = *reinterpret_cast<const float2*>(rm + (g + 8) * 64 + (c ^ (g << 3)));
        const uint32_t y0 = __ldg(reinterpret_cast<const unsigned int*>(p.dyg + (t0 + wr + g) * 64 + c));
        const uint32_t y1 = __ldg(reinterpret_cast<const unsigned int*>(p.dyg + (t0 + wr + g + 8) * 64 + c));
        dy[nt][0] += a0.x + bf16lo(y0); dy[nt][1] += a0.y + bf16hi(y0); dy[nt][2] += a1.x + bf16lo(y1); dy[nt][3] += a1.y + bf16hi(y1);
      }
      uint32_t xf[4][4];
#pragma unroll
      for (int kc = 0; kc < 4; ++kc) ldsm_x4(xf[kc], st + BP2_X + bswz128(wr + (lane & 15), 2 * kc + l4));
      float xh[4][4][2];
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int kc = 0; kc < 4; ++kc)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          xh[kc][r][0] = bf16lo(xf[kc][r]); xh[kc][r][1] = bf16hi(xf[kc][r]);
          if (r & 1) s1 += xh[kc][r][0] + xh[kc][r][1]; else s0 += xh[kc][r][0] + xh[kc][r][1];
        }
      const float mu0 = quad_sum(s0) * (1.f / 64), mu1 = quad_sum(s1) * (1.f / 64);
      float v0 = 0.f, v1 = 0.f;
#pragma unroll
      for (int kc = 0; kc < 4; ++kc)
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float d = xh[kc][r][e] - ((r & 1) ? mu1 : mu0);
            if (r & 1) v1 += d * d; else v0 += d * d;
          }
      const float r0 = rsqrtf(quad_sum(v0) * (1.f / 64) + p.eps), r1 = rsqrtf(quad_sum(v1) * (1.f / 64) + p.eps);
#pragma unroll
      for (int kc = 0; kc < 4; ++kc)
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
          for (int e = 0; e < 2; ++e) xh[kc][r][e] = (xh[kc][r][e] - ((r & 1) ? mu1 : mu0)) * ((r & 1) ? r1 : r0);
#pragma unroll
      for (int kc = 0; kc < 4; ++kc)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const int col = 16 * kc + 8 * (r >> 1) + 2 * q, row = wr + g + 8 * (r & 1);
          sts32(sb + BP2_Y + bswz128(row, col >> 3) + (col & 7) * 2,
                pack_bf16(fmaf(xh[kc][r][0], sgam[col], sbet[col]), fmaf(xh[kc][r][1], sgam[col + 1], sbet[col + 1])));
        }
      float m1[2] = {0.f, 0.f}, m2[2] = {0.f, 0.f};
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int col = 8 * nt + 2 * q + (e & 1), hh = e >> 1;
          const float x = xh[nt >> 1][2 * (nt & 1) + hh][e & 1];
          const float dx = dy[nt][e] * sgam[col];
          m1[hh] += dx; m2[hh] += dx * x;
          dg[nt][e & 1] += dy[nt][e] * x;
          dbt[nt][e & 1] += dy[nt][e];
        }
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) { m1[hh] = quad_sum(m1[hh]) * (1.f / 64); m2[hh] = quad_sum(m2[hh]) * (1.f / 64); }
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int c0 = 8 * nt + 2 * q, row = wr + g + 8 * hh;
          const float rsd = hh ? r1 : r0;
          const uint32_t rr = __ldg(reinterpret_cast<const unsigned int*>(p.dres + (t0 + row) * 64 + c0));
          float o2[2];
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float x = xh[nt >> 1][2 * (nt & 1) + hh][e];
            o2[e] = rsd * (dy[nt][2 * hh + e] * sgam[c0 + e] - m1[hh] - x * m2[hh]) + (e ? bf16hi(rr) : bf16lo(rr));
          }
          sts32(st + BP2_X + bswz128(row, nt) + 4 * q, pack_bf16(o2[0], o2[1]));
        }
    }
    __syncthreads();                                   // y complete, dmsa staged in the msa rows
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int idx = tid + 256 * k, row = idx >> 3, gr = idx & 7;
      stg128(p.dmsa + (t0 + row) * 64 + gr * 8, lds128(st + BP2_X + bswz128(row, gr)));
    }
    // dWv rows o = 32 warp + 16 m + .., K = 64 tokens
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      uint32_t bf[4][4];
#pragma unroll
      for (int np = 0; np < 4; ++np)
        ldsm_x4_t(bf[np], sb + BP2_Y + bswz128(16 * kt + (lane & 7) + (((lane >> 3) & 1) << 3), 2 * np + l4));
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        uint32_t af[4];
        const int mat = lane >> 3, row = 16 * kt + 8 * (mat >> 1) + (lane & 7);
        ldsm_x4_t(af, st + BP2_G + bswz512(row, 4 * warp + 2 * m + (mat & 1)));
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) mma16816(dwa[m][nt], af, bf[nt >> 1][(nt & 1) * 2], bf[nt >> 1][(nt & 1) * 2 + 1]);
      }
    }
  }
  cp_async_wait<0>();
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int o = 32 * warp + 16 * m + g + 8 * hh;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        atomicAdd(p.dw + o * 64 + 8 * nt + 2 * q, dwa[m][nt][2 * hh]);
        atomicAdd(p.dw + o * 64 + 8 * nt + 2 * q + 1, dwa[m][nt][2 * hh + 1]);
      }
    }
  if (kh == 0) {
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        float a = dg[nt][e], b = dbt[nt][e];
#pragma unroll
        for (int o = 4; o < 32; o <<= 1) { a += __shfl_xor_sync(0xffffffffu, a, o); b += __shfl_xor_sync(0xffffffffu, b, o); }
        if (g == 0) { atomicAdd(p.dgamma + 8 * nt + 2 * q + e, a); atomicAdd(p.dbeta + 8 * nt + 2 * q + e, b); }
      }
  }
}

}  // namespace a100
