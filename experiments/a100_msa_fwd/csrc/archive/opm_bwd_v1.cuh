// opm_bwd.cuh -- OPM backward kernels (A100, sm_80). Forward: a, b [S][32 L] (masked projections of LN(msa)), O = a^T b in the grouped
// layout [(i, d), (j, e)], P_ij = O_ij / n_ij, z = P . Wo^T + bo (+ residual). Given dz:
//   opm_dgrad : dzn = bf16(dz / n) [L^2, 128], dO = dzn . Wo in the grouped layout (the forward epilogue's permute, run backwards),
//               dbo = sum_ij dz (fp32 atomics)
//   (cuBLAS)  : dA = b . dO^T,  dB = a . dO                                   [S, 32 L] each
//   opm_dwo   : dWo[c, (d, e)] = sum_ij dzn[ij, c] O[(i, d), (j, e)]          split-K over pairs, fp32 atomics
//   opm_pbwd  : mask, dy = [dA | dB] . W, dW = [dA | dB]^T y, LayerNorm backward -> dmsa, dgamma, dbeta
#pragma once
#include "sm80_common.cuh"

namespace a100 {

DEVI uint32_t swz256b(int row, int gr) { return row * 256 + ((gr ^ (row & 7)) << 4); }
DEVI uint32_t swz128b(int row, int gr) { return row * 128 + ((gr ^ (row & 7)) << 4); }

// ---------------------------------------------------------------------------------------------------- opm_dgrad
struct OpmDgradParams {
  const __nv_bfloat16* dz;    // [L*L, 128]
  const __nv_bfloat16* woT;   // [1024, 128]: WoT[n][c] = Wo[c][n]
  const uint32_t* bits;       // [L, nw]
  __nv_bfloat16* dzn;         // [L*L, 128]
  __nv_bfloat16* dO;          // [32 L, 32 L]
  float* dbo;                 // [128], accumulated
  int L, nw;
};

// CTA: 128 pairs (4 i x 32 j), all 1024 outputs in 8 chunks of 128 (= 4 d x 32 e). A = dzn tile [128 pairs][128 c] resident (256 B rows);
// B = WoT chunk [128 n][128 c], double-buffered. 8 warps = 4 (32 pairs) x 2 (64 n). A finished chunk is staged through its own
// (now idle) B buffer as [4 i x 4 d rows][32 j x 32 e] and leaves as 16 rows of 2 KB contiguous grouped-layout dO.
constexpr int OPM_DG_A = 0, OPM_DG_B = 128 * 256, OPM_DG_N = OPM_DG_B + 2 * 128 * 256, OPM_DG_SMEM = OPM_DG_N + 128 * 4 + 16 * 128 * 4;

__global__ void __launch_bounds__(256, 1) opm_dgrad_kernel(OpmDgradParams p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sb = smem_u32(smem);
  float* inv_n = reinterpret_cast<float*>(smem + OPM_DG_N);
  float* colsum = inv_n + 128;                     // [16 row groups][128]
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, q = lane & 3, wm = warp & 3, wn = warp >> 2, l4 = lane >> 4;
  const int nj = p.L / 32;
  const int i0 = (blockIdx.x / nj) * 4, j0 = (blockIdx.x % nj) * 32;
  const size_t ldo = (size_t)32 * p.L;

  auto issue_b = [&](int ch, int buf) {           // WoT rows 128 ch .. + 127: 2048 granules, 8 per thread
    const uint32_t dst = sb + OPM_DG_B + buf * 128 * 256;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int idx = tid + 256 * k, row = idx >> 4, gr = idx & 15;
      cp_async16(dst + swz256b(row, gr), p.woT + (size_t)(128 * ch + row) * 128 + gr * 8);
    }
  };
  issue_b(0, 0);
  cp_async_commit();

  if (tid < 128) {
    const int i = i0 + (tid >> 5), j = j0 + (tid & 31);
    const uint32_t* bi = p.bits + (size_t)i * p.nw;
    const uint32_t* bj = p.bits + (size_t)j * p.nw;
    int c = 0;
    for (int w = 0; w < p.nw; ++w) c += __popc(__ldg(bi + w) & __ldg(bj + w));
    inv_n[tid] = 1.f / (float)max(c, 1);
  }
  __syncthreads();
  // dz tile: thread owns granule gr = tid & 15 (8 channels) of rows (tid >> 4) + 16 k; raw column sums for dbo, dzn to smem + global
  {
    const int gr = tid & 15;
    float cs[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int row = (tid >> 4) + 16 * k;
      const size_t pr = (size_t)(i0 + (row >> 5)) * p.L + j0 + (row & 31);
      const uint4 v = __ldg(reinterpret_cast<const uint4*>(p.dz + pr * 128) + gr);
      const uint32_t* u = reinterpret_cast<const uint32_t*>(&v);
      const float s = inv_n[row];
      uint32_t o4[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float lo = bf16lo(u[e]), hi = bf16hi(u[e]);
        cs[2 * e] += lo; cs[2 * e + 1] += hi;
        o4[e] = pack_bf16(lo * s, hi * s);
      }
      const uint4 w4 = make_uint4(o4[0], o4[1], o4[2], o4[3]);
      sts128(sb + OPM_DG_A + swz256b(row, gr), w4);
      stg128(p.dzn + pr * 128 + gr * 8, w4);
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) colsum[(tid >> 4) * 128 + gr * 8 + e] = cs[e];
  }
  __syncthreads();
  if (tid < 128) {
    float s = 0.f;
#pragma unroll
    for (int r = 0; r < 16; ++r) s += colsum[r * 128 + tid];
    atomicAdd(p.dbo + tid, s);
  }

  uint32_t a_off[2], b_off[4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) a_off[mt] = OPM_DG_A + swz256b(32 * wm + 16 * mt + (lane & 15), l4);          // ^ (kc << 5)
#pragma unroll
  for (int np = 0; np < 4; ++np) b_off[np] = swz256b(64 * wn + 16 * np + (lane & 7) + (l4 << 3), (lane >> 3) & 1);

  for (int ch = 0; ch < 8; ++ch) {
    const int buf = ch & 1;
    if (ch + 1 < 8) issue_b(ch + 1, buf ^ 1);
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();
    const uint32_t bb = sb + OPM_DG_B + buf * 128 * 256;
    float acc[2][8][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
      for (int b = 0; b < 8; ++b)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
#pragma unroll
    for (int kc = 0; kc < 8; ++kc) {
      uint32_t af[2][4], bf[4][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) ldsm_x4(af[mt], sb + (a_off[mt] ^ (kc << 5)));
#pragma unroll
      for (int np = 0; np < 4; ++np) ldsm_x4(bf[np], bb + (b_off[np] ^ (kc << 5)));
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) mma16816(acc[mt][nt], af[mt], bf[nt >> 1][(nt & 1) * 2], bf[nt >> 1][(nt & 1) * 2 + 1]);
    }
    __syncthreads();                               // every warp is done reading this B buffer: stage the chunk into it
    // pair p = 32 wm + 16 mt + g (+8) -> (il = p >> 5, jl = p & 31); n = 64 wn + 8 nt + 2 q -> (dl = n >> 5, e = n & 31)
    // staging row r = 4 il + dl (16 rows of 2 KB: [32 jl][32 e]), 16 B granule (jl * 4 + e / 8) XOR (r & 7)... rows are 2 KB: use
    // a granule swizzle inside 128 B groups: granule gg -> gg ^ (jl & 7)
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int pr = 32 * wm + 16 * mt + g + 8 * hh, il = pr >> 5, jl = pr & 31;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          const int n = 64 * wn + 8 * nt + 2 * q, dl = n >> 5, e = n & 31;
          const int r = 4 * il + dl, gg = jl * 4 + (e >> 3);
          sts32(bb + r * 2048 + ((gg ^ (jl & 7)) << 4) + (e & 7) * 2, pack_bf16(acc[mt][nt][2 * hh], acc[mt][nt][2 * hh + 1]));
        }
      }
    __syncthreads();
    // 16 rows x 128 granules = 2048 granules, 8 per thread; row r -> dO row (i0 + r / 4) * 32 + 4 ch + r % 4, cols j0 * 32 ..
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int idx = tid + 256 * k, r = idx >> 7, gg = idx & 127, jl = gg >> 2;
      const size_t orow = (size_t)(i0 + (r >> 2)) * 32 + 4 * ch + (r & 3);
      stg128(p.dO + orow * ldo + (size_t)j0 * 32 + gg * 8, lds128(bb + r * 2048 + ((gg ^ (jl & 7)) << 4)));
    }
    __syncthreads();                               // the staging reads are done before this buffer is refilled (chunk ch + 2)
  }
}

// ---------------------------------------------------------------------------------------------------- opm_dwo
struct OpmDwoParams {
  const __nv_bfloat16* dzn;   // [L*L, 128]
  const __nv_bfloat16* o;     // [32 L, 32 L] grouped O (unnormalized)
  float* dwo;                 // [128, 1024], accumulated
  int L, chunks_per;          // K chunk = 32 pairs (one i, 32 consecutive j); chunks [split * chunks_per, ...)
};

// CTA: dWo block [128 c][128 n = 4 d x 32 e] (n-block nb = d / 4), split-K over pair chunks. A = dzn^T (M = c, K = pairs) from the
// [pairs][c] tile with ldmatrix.trans; B = O rows (K = pairs, N = (d, e)) staged [32 j][4 d x 32 e] (256 B rows), ldmatrix.trans.
// NS-stage ring of 16 KB stages; 8 warps = 4 (32 c) x 2 (64 n). Grid: n-block fastest, so the 8 CTAs reading one dzn slice co-run.
constexpr int OPM_DW_NS = 4, OPM_DW_STAGE = 2 * 32 * 256, OPM_DW_SMEM = OPM_DW_NS * OPM_DW_STAGE;

__global__ void __launch_bounds__(256, 2) opm_dwo_kernel(OpmDwoParams p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sb = smem_u32(smem);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, q = lane & 3, wm = warp & 3, wn = warp >> 2, l4 = lane >> 4;
  const int nb = blockIdx.x & 7, split = blockIdx.x >> 3;
  const int njb = p.L / 32, nchunks = p.L * njb;
  const int c_begin = split * p.chunks_per, c_end = min(nchunks, c_begin + p.chunks_per);
  if (c_begin >= c_end) return;
  const size_t ldo = (size_t)32 * p.L;
  // copy slots (2 per operand): A granule (idx & 15) of row idx >> 4; B (d = idx >> 7, j = (idx >> 2) & 31, eg = idx & 3)
  auto issue = [&](int ck, int stage) {
    const int i = ck / njb, jb = ck % njb;
    const uint32_t st = sb + stage * OPM_DW_STAGE;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int idx = tid + 256 * k;
      const int ar = idx >> 4, ag = idx & 15;
      cp_async16(st + swz256b(ar, ag), p.dzn + ((size_t)i * p.L + jb * 32 + ar) * 128 + ag * 8);
      const int d = idx >> 7, j = (idx >> 2) & 31, eg = idx & 3;
      cp_async16(st + 32 * 256 + swz256b(j, d * 4 + eg), p.o + (size_t)(i * 32 + 4 * nb + d) * ldo + (size_t)(jb * 32 + j) * 32 + eg * 8);
    }
  };
  float acc[2][8][4];
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int b = 0; b < 8; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
  // A (trans): matrix (lane >> 3): m half = bit 0, k half = bit 1 -> row k = 16 kk + 8 (mat >> 1) + (lane & 7), granule m / 8
  uint32_t a_off[2], b_off[4];
  {
    const int mat = lane >> 3;
    const int kr = 8 * (mat >> 1) + (lane & 7);
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) a_off[mt] = swz256b(kr, (32 * wm + 16 * mt) / 8 + (mat & 1));      // + kk 16 rows (row & 7 kept)
    const int br = (lane & 7) + (((lane >> 3) & 1) << 3);
#pragma unroll
    for (int np = 0; np < 4; ++np) b_off[np] = 32 * 256 + swz256b(br, 8 * wn + 2 * np + l4);
  }
  const int n = c_end - c_begin;
#pragma unroll
  for (int s = 0; s < OPM_DW_NS - 1; ++s) { if (s < n) issue(c_begin + s, s); cp_async_commit(); }
  int stage = 0, lstage = OPM_DW_NS - 1;
  for (int t = 0; t < n; ++t) {
    cp_async_wait<OPM_DW_NS - 2>();
    __syncthreads();
    if (t + OPM_DW_NS - 1 < n) issue(c_begin + t + OPM_DW_NS - 1, lstage);
    cp_async_commit();
    if (++lstage == OPM_DW_NS) lstage = 0;
    const uint32_t st = sb + stage * OPM_DW_STAGE;
    if (++stage == OPM_DW_NS) stage = 0;
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      uint32_t af[2][4], bf[4][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) ldsm_x4_t(af[mt], st + a_off[mt] + kk * 16 * 256);
#pragma unroll
      for (int np = 0; np < 4; ++np) ldsm_x4_t(bf[np], st + b_off[np] + kk * 16 * 256);
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) mma16816(acc[mt][nt], af[mt], bf[nt >> 1][(nt & 1) * 2], bf[nt >> 1][(nt & 1) * 2 + 1]);
    }
  }
  cp_async_wait<0>();
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int c = 32 * wm + 16 * mt + g + 8 * hh;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int nn = 128 * nb + 64 * wn + 8 * nt + 2 * q;
        atomicAdd(p.dwo + c * 1024 + nn, acc[mt][nt][2 * hh]);
        atomicAdd(p.dwo + c * 1024 + nn + 1, acc[mt][nt][2 * hh + 1]);
      }
    }
}

// ---------------------------------------------------------------------------------------------------- opm_pbwd
struct OpmPbwdParams {
  const __nv_bfloat16* msa;   // [T, 64]
  const uint8_t* mask;        // [T] or nullptr
  const __nv_bfloat16* da;    // [T, 32]  (dA [S][32 L] = token-major)
  const __nv_bfloat16* db;    // [T, 32]
  const __nv_bfloat16* w;     // [64 o, 64 k]: rows 0..31 to_left, 32..63 to_right (no gamma folded)
  const float* gamma;         // [64]
  const float* beta;          // [64]
  __nv_bfloat16* dmsa;        // [T, 64]
  float* dw;                  // [64, 64] accumulated
  float* dgamma;              // [64] accumulated
  float* dbeta;               // [64] accumulated
  int T;
  float eps;
};

// Persistent CTA of 4 warps over 64-token tiles (warp = 16 tokens). Per tile: stage msa, dp = mask . [da | db] (both 128 B rows);
// LN statistics from the msa fragments (quad sums), y = x^ gamma + beta -> smem (bf16, the forward's projection operand);
// dy = dp . W (mma, B = W with ldmatrix.trans); LayerNorm backward in the accumulator layout (which coincides with the A-fragment
// layout of x^): dx = rstd (dx^ - mean(dx^) - x^ mean(dx^ x^)), dx^ = dy gamma; dgamma += dy x^, dbeta += dy in registers;
// dW[o, k] += dp^T y: warp w owns rows o = 16 w .. +15 over the tile's 64 tokens (both operands ldmatrix.trans).
constexpr int OPM_PB_W = 0, OPM_PB_X = 64 * 128, OPM_PB_P = OPM_PB_X + 64 * 128, OPM_PB_Y = OPM_PB_P + 64 * 128;
constexpr int OPM_PB_SMEM = OPM_PB_Y + 64 * 128 + 64 * 4 * 2;

__global__ void __launch_bounds__(128) opm_pbwd_kernel(OpmPbwdParams p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sb = smem_u32(smem);
  float* sg = reinterpret_cast<float*>(smem + OPM_PB_Y + 64 * 128);
  float* sbe = sg + 64;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, q = lane & 3, l4 = lane >> 4;
  for (int idx = tid; idx < 64 * 8; idx += 128) {
    const int row = idx >> 3, gr = idx & 7;
    cp_async16(sb + OPM_PB_W + swz128b(row, gr), p.w + row * 64 + gr * 8);
  }
  if (tid < 64) { sg[tid] = p.gamma[tid]; sbe[tid] = p.beta[tid]; }
  cp_async_commit();

  float dwacc[8][4], dg[8][2][2], dbt[8][2][2];     // dg / dbt: [nt][row half][col pair]
#pragma unroll
  for (int nt = 0; nt < 8; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) { dwacc[nt][e] = 0.f; dg[nt][e >> 1][e & 1] = 0.f; dbt[nt][e >> 1][e & 1] = 0.f; }

  const int ntiles = p.T / 64;
  for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
    const int t0 = tile * 64;
    __syncthreads();                                  // the previous tile's smem reads are done
    for (int idx = tid; idx < 64 * 8; idx += 128) {
      const int row = idx >> 3, gr = idx & 7;
      cp_async16(sb + OPM_PB_X + swz128b(row, gr), p.msa + (size_t)(t0 + row) * 64 + gr * 8);
      const __nv_bfloat16* src = (gr < 4 ? p.da : p.db) + (size_t)(t0 + row) * 32 + (gr & 3) * 8;
      cp_async16(sb + OPM_PB_P + swz128b(row, gr), src);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    if (p.mask) {                                      // dp of masked tokens is 0 (the forward multiplied a, b by the mask)
      for (int idx = tid; idx < 64 * 8; idx += 128) {
        const int row = idx >> 3, gr = idx & 7;
        if (!p.mask[t0 + row]) sts128(sb + OPM_PB_P + swz128b(row, gr), make_uint4(0, 0, 0, 0));
      }
      __syncthreads();
    }
    // x^ for this warp's 16 tokens, in the A-fragment (= accumulator) layout: rows g / g + 8, cols 16 kc + 2 q (+1) and + 8
    const int wr = 16 * warp;
    uint32_t xf[4][4];
#pragma unroll
    for (int kc = 0; kc < 4; ++kc) ldsm_x4(xf[kc], sb + OPM_PB_X + swz128b(wr + (lane & 15), 2 * kc + l4));
    float xh[4][4][2];                                 // [kc][reg][lo/hi]
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
    // y = x^ gamma + beta -> y tile (bf16). reg r: row g + 8 (r & 1), col 16 kc + 8 (r >> 1) + 2 q
#pragma unroll
    for (int kc = 0; kc < 4; ++kc)
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int col = 16 * kc + 8 * (r >> 1) + 2 * q, row = wr + g + 8 * (r & 1);
        sts32(sb + OPM_PB_Y + swz128b(row, col >> 3) + (col & 7) * 2,
              pack_bf16(fmaf(xh[kc][r][0], sg[col], sbe[col]), fmaf(xh[kc][r][1], sg[col + 1], sbe[col + 1])));
      }
    // dy = dp . W : A = dp rows (K = o), B[k = o][n] = W[o][n] -> W rows are K: ldmatrix.trans
    float dy[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) dy[nt][e] = 0.f;
#pragma unroll
    for (int kc = 0; kc < 4; ++kc) {
      uint32_t af[4];
      ldsm_x4(af, sb + OPM_PB_P + swz128b(wr + (lane & 15), 2 * kc + l4));
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t bf[4];
        ldsm_x4_t(bf, sb + OPM_PB_W + swz128b(16 * kc + (lane & 7) + (((lane >> 3) & 1) << 3), 2 * np + l4));
        mma16816(dy[2 * np], af, bf[0], bf[1]);
        mma16816(dy[2 * np + 1], af, bf[2], bf[3]);
      }
    }
    // LayerNorm backward. dy[nt][e]: row g + 8 (e >> 1), col 8 nt + 2 q + (e & 1)  <->  xh[nt >> 1][2 (nt & 1) + (e >> 1)][e & 1]
    float m1[2] = {0.f, 0.f}, m2[2] = {0.f, 0.f};
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int col = 8 * nt + 2 * q + (e & 1), h = e >> 1;
        const float x = xh[nt >> 1][2 * (nt & 1) + h][e & 1];
        const float dx = dy[nt][e] * sg[col];
        m1[h] += dx; m2[h] += dx * x;
        dg[nt][h][e & 1] += dy[nt][e] * x;
        dbt[nt][h][e & 1] += dy[nt][e];
      }
#pragma unroll
    for (int h = 0; h < 2; ++h) { m1[h] = quad_sum(m1[h]) * (1.f / 64); m2[h] = quad_sum(m2[h]) * (1.f / 64); }
    // dx -> msa staging rows (the x tile is consumed), then 128 B per token
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int c0 = 8 * nt + 2 * q, row = wr + g + 8 * h;
        const float rs = h ? r1 : r0;
        float o2[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float x = xh[nt >> 1][2 * (nt & 1) + h][e];
          o2[e] = rs * (dy[nt][2 * h + e] * sg[c0 + e] - m1[h] - x * m2[h]);
        }
        sts32(sb + OPM_PB_X + swz128b(row, nt) + 4 * q, pack_bf16(o2[0], o2[1]));
      }
    __syncthreads();                                   // y and the masked dp tile are complete; dmsa staged
    for (int idx = tid; idx < 64 * 8; idx += 128) {
      const int row = idx >> 3, gr = idx & 7;
      stg128(p.dmsa + (size_t)(t0 + row) * 64 + gr * 8, lds128(sb + OPM_PB_X + swz128b(row, gr)));
    }
    // dW rows o = 16 warp + .., K = 64 tokens: A[m = o][k = t] = dp[t][o] (ldmatrix.trans of the dp tile), B[k = t][n] = y[t][n]
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      uint32_t af[4];
      {
        const int mat = lane >> 3, row = 16 * kt + 8 * (mat >> 1) + (lane & 7);
        ldsm_x4_t(af, sb + OPM_PB_P + swz128b(row, 2 * warp + (mat & 1)));
      }
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t bf[4];
        ldsm_x4_t(bf, sb + OPM_PB_Y + swz128b(16 * kt + (lane & 7) + (((lane >> 3) & 1) << 3), 2 * np + l4));
        mma16816(dwacc[2 * np], af, bf[0], bf[1]);
        mma16816(dwacc[2 * np + 1], af, bf[2], bf[3]);
      }
    }
  }
  // dW rows 16 warp + g (+8), cols 8 nt + 2 q; dgamma / dbeta: reduce the 8 row-groups (g) of each quad column, then atomics
#pragma unroll
  for (int nt = 0; nt < 8; ++nt)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int o = 16 * warp + g + 8 * hh, k = 8 * nt + 2 * q;
      atomicAdd(p.dw + o * 64 + k, dwacc[nt][2 * hh]);
      atomicAdd(p.dw + o * 64 + k + 1, dwacc[nt][2 * hh + 1]);
    }
#pragma unroll
  for (int nt = 0; nt < 8; ++nt)
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      float a = dg[nt][0][e] + dg[nt][1][e], b = dbt[nt][0][e] + dbt[nt][1][e];
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) { a += __shfl_xor_sync(0xffffffffu, a, o); b += __shfl_xor_sync(0xffffffffu, b, o); }
      if (g == 0) { atomicAdd(p.dgamma + 8 * nt + 2 * q + e, a); atomicAdd(p.dbeta + 8 * nt + 2 * q + e, b); }
    }
}

}  // namespace a100
