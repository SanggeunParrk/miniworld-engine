// tr_bwd_w_sm80.cuh -- the weight gradients of the Transition backward (role W, see tr_bwd_sm80.cuh), A100 / sm_80:
//
//   dWa = dA^T xn   dWb = dB^T xn   dWs^T = h^T dy          (K = rows, f32 accumulation, per-replica partial sums)
//
// CTA = 8 warps on one 128-unit hidden slice (blockIdx % 4) and one row replica (blockIdx / 4).  The slice's 3 x 128 output rows
// ([dWa; dWb; dWs^T]) are 24 m16 tiles; warp w owns tiles 3 w .. 3 w + 2 x all 128 columns (192 f32 accumulators).  Per 16-row k-step:
//   A = the transposed intermediate: a fragment-native 16 x 16 block (from the stage in shared memory, 16 B per lane) turned into the
//       m16 (hidden) x k16 (rows) A fragment by movmatrix.trans (block words f0..f3 -> A words (f0, f2, f1, f3), each transposed);
//   B = xn or dy rows (k) x columns (n) via ldmatrix.trans.
// Rows arrive in 64-row stages (x | dy | the slice's A blocks, 80 KB) through a 2-deep cp.async pipeline; the x rows are normalised in
// shared memory with the forward's arithmetic (the same xn bits as P used), so xn is never written to global memory.
#pragma once
#include "tr_bwd_sm80.cuh"

namespace a100 {

struct WParams {
  const __nv_bfloat16* x;      // [T][128]
  const __nv_bfloat16* dy;     // [T][128]
  const float2* stats;         // [T] (mean, rstd) from P
  const float* gamma;          // [128]
  const float* beta;           // [128]
  const uint4* ab;             // fragment-native blocks [T / 16][32 K][dA | dB | h][32 lanes]
  float* part;                 // [nrep][3][512][128] f32 partial sums (dWa, dWb, dWs^T)
  int T;
};

struct CfgW {
  static constexpr int NWARP = 8, NTHR = 256, RS = 64, NSTAGE = 2;
  static constexpr int TILE = RS * 256;                        // [64 rows][256 B] swizzled
  static constexpr int ABLK = 3 * 8 * (RS / 16) * 512;         // [matrix][hidden m16][k-step][32 lanes][16 B]
  static constexpr int STAGE = 2 * TILE + ABLK + RS * 8;       // x | dy | A | (mean, rstd) of the rows
  static constexpr int SMEM = NSTAGE * STAGE + 2 * 128 * 4;    // + gamma | beta
  static_assert(SMEM + 1024 <= 167936, "sm_80 shared memory");
};

DEVI uint32_t movtrans(uint32_t v) {
  uint32_t r; asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(r) : "r"(v)); return r;
}
DEVI uint32_t swz_w(uint32_t r, uint32_t G) { return r * 256 + ((G ^ (r & 7u)) << 4); }   // 8 consecutive rows: 8 bank groups

// one 64-row stage of one warp's three m16 tiles.  MODE 0: every tile's B is xn, 2: dy, 1: tile 0 xn, tiles 1 and 2 dy (compile-time,
// so no MMA operand is ever a select)
template <int MODE>
DEVI void w_stage(float (&acc)[3][16][4], uint32_t buf, int g0, int lane) {
  using G = CfgW;
  // ldmatrix.trans lane address: matrix mi = lane / 8 -> (row half mi & 1, column granule half mi >> 1)
  const int lr = (((lane >> 3) & 1) << 3) + (lane & 7), lg = lane >> 4;
#pragma unroll
  for (int kk = 0; kk < G::RS / 16; ++kk) {
    uint32_t a[3][4];
#pragma unroll
    for (int m = 0; m < 3; ++m) {
      const int g = g0 + m, blk = ((g >> 3) * 8 + (g & 7)) * 4 + kk;
      const uint4 f = lds128(buf + 2 * G::TILE + blk * 512 + lane * 16);
      a[m][0] = movtrans(f.x); a[m][1] = movtrans(f.z); a[m][2] = movtrans(f.y); a[m][3] = movtrans(f.w);
    }
    const int row = 16 * kk + lr;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      uint32_t bx[4], by[4];
      if (MODE != 2) ldsm_x4_t(bx, buf + swz_w(row, 2 * j + lg));
      if (MODE != 0) ldsm_x4_t(by, buf + G::TILE + swz_w(row, 2 * j + lg));
#pragma unroll
      for (int m = 0; m < 3; ++m) {
        const bool y = MODE == 2 || (MODE == 1 && m > 0);
        const uint32_t* b = y ? by : bx;
        mma16816(acc[m][2 * j], a[m], b[0], b[1]);
        mma16816(acc[m][2 * j + 1], a[m], b[2], b[3]);
      }
    }
  }
}

__global__ void __launch_bounds__(256, 1) tr_bwd_w_kernel(const WParams p) {
  using G = CfgW;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int sl = blockIdx.x & 3, rr = blockIdx.x >> 2, nrep = gridDim.x >> 2;
  const int nst = p.T / G::RS;                                  // 64-row stages; replica rr takes rr, rr + nrep, ...
  const int n_mine = nst > rr ? (nst - rr + nrep - 1) / nrep : 0;
  const uint32_t s_u = smem_u32(smem);
  float* sGB = reinterpret_cast<float*>(smem + G::NSTAGE * G::STAGE);
  for (int k = tid; k < 128; k += G::NTHR) { sGB[k] = p.gamma[k]; sGB[128 + k] = p.beta[k]; }

  // per-thread copy slots (fixed across stages): 8 of the 2048 x | dy granules (slot k: rows 16 k + tid / 16 of x (k < 4) | dy),
  // 12 of the 3072 A granules (slot k: block warp + 8 k = matrix k / 4, hidden m16 2 (k % 4) + warp / 4, k-step warp % 4), the rows' stats
  const int xr = tid >> 4, xg = tid & 15;
  const uint4* const asrc = p.ab + (8 * sl + (warp >> 2)) * 96 + lane;
  auto load_stage = [&](int i) {
    const int row0 = (rr + i * nrep) * G::RS;
    const uint32_t buf = s_u + (i % G::NSTAGE) * G::STAGE;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int r = 16 * (k & 3) + xr;
      cp_async16_full(buf + (k >> 2) * G::TILE + swz_w(r, xg), ((k >> 2) ? p.dy : p.x) + (size_t)(row0 + r) * 128 + xg * 8);
    }
    const uint4* const a0 = asrc + (size_t)(row0 / 16 + (warp & 3)) * 3072;
#pragma unroll
    for (int k = 0; k < 12; ++k)
      cp_async16_full(buf + 2 * G::TILE + (warp + 8 * k) * 512 + lane * 16, a0 + 2 * (k & 3) * 96 + (k >> 2) * 32);
    if (tid < G::RS / 2) cp_async16_full(buf + 2 * G::TILE + G::ABLK + tid * 16, p.stats + row0 + 2 * tid);
  };
  // x -> xn in place (the forward's LayerNorm arithmetic): 4 granules per thread (rows 16 k + tid / 16, granule tid % 16)
  auto normalise = [&](int i) {
    const uint32_t buf = s_u + (i % G::NSTAGE) * G::STAGE;
    float gm[8], bt[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) { gm[e] = sGB[xg * 8 + e]; bt[e] = sGB[128 + xg * 8 + e]; }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int r = 16 * k + xr;
      const uint2 stw = lds64(buf + 2 * G::TILE + G::ABLK + r * 8);
      const float mean = __uint_as_float(stw.x), rstd = __uint_as_float(stw.y);
      const uint32_t a = buf + swz_w(r, xg);
      const uint4 v = lds128(a);
      const uint32_t wv[4] = {v.x, v.y, v.z, v.w};
      uint32_t o[4];
#pragma unroll
      for (int e = 0; e < 4; ++e)
        o[e] = pack_bf16(fmaf((bf16lo(wv[e]) - mean) * rstd, gm[2 * e], bt[2 * e]), fmaf((bf16hi(wv[e]) - mean) * rstd, gm[2 * e + 1], bt[2 * e + 1]));
      sts128(a, make_uint4(o[0], o[1], o[2], o[3]));
    }
  };

  float acc[3][16][4];
#pragma unroll
  for (int m = 0; m < 3; ++m)
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[m][j][e] = 0.f;
  // m16 tiles 3 warp + m: matrix g / 8 (0 dWa, 1 dWb: B = xn; 2 dWs^T: B = dy), hidden m16 g % 8
  const int g0 = 3 * warp;

  if (n_mine > 0) load_stage(0);
  cp_async_commit();
#pragma unroll 1
  for (int i = 0; i < n_mine; ++i) {
    cp_async_wait<0>();
    __syncthreads();                                            // stage i landed; everyone is done with stage i - 1
    if (i + 1 < n_mine) load_stage(i + 1);
    cp_async_commit();
    normalise(i);
    __syncthreads();
    const uint32_t buf = s_u + (i % G::NSTAGE) * G::STAGE;
    if (g0 + 2 < 16) w_stage<0>(acc, buf, g0, lane);            // all three tiles on xn
    else if (g0 >= 16) w_stage<2>(acc, buf, g0, lane);          // all on dy
    else w_stage<1>(acc, buf, g0, lane);                        // warp 5: tile 15 on xn, 16 and 17 on dy
  }
  cp_async_wait<0>();
  // partial sums: acc[m][J] = rows (hidden) g8, g8 + 8 of m16 tile g x columns 8 J + 2 q + e
  const int g8 = lane >> 2, q = lane & 3;
#pragma unroll
  for (int m = 0; m < 3; ++m) {
    const int g = g0 + m;
    float* out = p.part + (((size_t)rr * 3 + (g >> 3)) * 512 + 128 * sl + 16 * (g & 7)) * 128;
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
      for (int hr = 0; hr < 2; ++hr)
        *reinterpret_cast<float2*>(out + (size_t)(8 * hr + g8) * 128 + 8 * j + 2 * q) = make_float2(acc[m][j][2 * hr], acc[m][j][2 * hr + 1]);
  }
}

}  // namespace a100
