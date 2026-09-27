// tr_bwd_w_sm80.cuh -- the weight gradients of the Transition backward (role W, see tr_bwd_sm80.cuh), A100 / sm_80:
//
//   dWa = dA^T xn   dWb = dB^T xn   dWs^T = h^T dy          (K = rows, f32 accumulation, per-replica partial sums)
//
// CTA = 12 warps (3 per SM sub-partition) on one 128-unit hidden slice (blockIdx % 4) and one row replica (blockIdx / 4): warp w owns
// matrix w / 4 (dWa | dWb | dWs^T) x hidden rows 32 (w % 4) .. + 32 x all 128 columns, 128 f32 accumulators.  Per 16-row k-step:
//   A = the transposed intermediate: two fragment-native 16 x 16 blocks loaded from L2 (16 B per lane, two k-steps ahead) and turned
//       into the m16 (hidden) x k16 (rows) A fragment by movmatrix.trans (block words f0..f3 -> A words (f0, f2, f1, f3) transposed);
//   B = xn or dy rows (k) x columns (n) from a swizzled shared-memory stage via ldmatrix.trans.
// Rows arrive in 64-row stages through a 3-deep cp.async pipeline shared by all 12 warps (one __syncthreads per stage).
#pragma once
#include "tr_bwd_sm80.cuh"

namespace a100 {

struct WParams {
  const __nv_bfloat16* xn;     // [T][128]
  const __nv_bfloat16* dy;     // [T][128]
  const uint4* dA;             // fragment-native blocks [T / 16][32][32]
  const uint4* dB;
  const uint4* hh;
  float* part;                 // [nrep][3][512][128] f32 partial sums (dWa, dWb, dWs^T)
  int T;
};

struct CfgW {
  static constexpr int NWARP = 12, NTHR = 384, RS = 64, NSTAGE = 3;
  static constexpr int TILE = RS * 256, STAGE = 2 * TILE;      // xn | dy, [64 rows][256 B] swizzled
  static constexpr int SMEM = NSTAGE * STAGE;
  static_assert(SMEM + 1024 <= 167936, "sm_80 shared memory");
};

DEVI uint32_t movtrans(uint32_t v) {
  uint32_t r; asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(r) : "r"(v)); return r;
}
DEVI uint32_t swz_w(uint32_t r, uint32_t G) { return r * 256 + ((G ^ (r & 7u)) << 4); }   // 8 consecutive rows: 8 bank groups

__global__ void __launch_bounds__(384, 1) tr_bwd_w_kernel(const WParams p) {
  using G = CfgW;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int sl = blockIdx.x & 3, rr = blockIdx.x >> 2, nrep = gridDim.x >> 2;
  const int mm = warp >> 2, hq = warp & 3;                      // matrix, hidden quarter of the slice
  const int nst = p.T / G::RS;                                  // 64-row stages; replica rr takes rr, rr + nrep, ...
  const int n_mine = nst > rr ? (nst - rr + nrep - 1) / nrep : 0;
  const uint32_t s_u = smem_u32(smem);

  auto load_stage = [&](int i) {                                // stage i of this replica -> buffer i % NSTAGE (xn and dy rows)
    const int row0 = (rr + i * nrep) * G::RS;
    const uint32_t buf = s_u + (i % G::NSTAGE) * G::STAGE;
    for (int c = tid; c < 2 * G::RS * 16; c += G::NTHR) {
      const int m = c >> 10, r = (c >> 4) & (G::RS - 1), gr = c & 15;
      const __nv_bfloat16* src = (m ? p.dy : p.xn) + (size_t)(row0 + r) * 128 + gr * 8;
      cp_async16_full(buf + m * G::TILE + swz_w(r, gr), src);
    }
  };
  const uint4* A = (mm == 0 ? p.dA : mm == 1 ? p.dB : p.hh) + (size_t)(8 * sl + 2 * hq) * 32 + lane;   // + R * 1024 + m * 32
  auto ldA = [&](int i, int kk, uint4 (&f)[2]) {                 // the two blocks (hidden m16 tiles) of k-step kk of stage i
    const size_t R = (size_t)(rr + i * nrep) * (G::RS / 16) + kk;
    f[0] = ldg_nc_na(A + R * 1024);
    f[1] = ldg_nc_na(A + R * 1024 + 32);
  };

  float acc[2][16][4];
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[m][j][e] = 0.f;
  // ldmatrix.trans lane address: matrix mi = lane / 8 -> (row half mi & 1, column granule half mi >> 1)
  const int lr = (((lane >> 3) & 1) << 3) + (lane & 7), lg = lane >> 4;
  const uint32_t tb = (mm == 2 ? G::TILE : 0);

  for (int i = 0; i < G::NSTAGE - 1; ++i) { if (i < n_mine) load_stage(i); cp_async_commit(); }
  uint4 fq[2][2];                                               // A blocks, two k-steps ahead
  if (n_mine > 0) { ldA(0, 0, fq[0]); ldA(0, 1, fq[1]); }
#pragma unroll 1
  for (int i = 0; i < n_mine; ++i) {
    cp_async_wait<G::NSTAGE - 2>();
    __syncthreads();                                            // stage i landed; stage i - 1's buffer is free
    if (i + G::NSTAGE - 1 < n_mine) load_stage(i + G::NSTAGE - 1);
    cp_async_commit();
    const uint32_t buf = s_u + (i % G::NSTAGE) * G::STAGE + tb;
#pragma unroll
    for (int kk = 0; kk < G::RS / 16; ++kk) {
      uint32_t a[2][4];
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const uint4 f = fq[kk & 1][m];
        a[m][0] = movtrans(f.x); a[m][1] = movtrans(f.z); a[m][2] = movtrans(f.y); a[m][3] = movtrans(f.w);
      }
      {                                                         // refill: k-step kk + 2 (possibly of the next stage)
        const int nk = kk + 2, ni = i + (nk >= G::RS / 16), nkk = nk & (G::RS / 16 - 1);
        if (ni < n_mine) ldA(ni, nkk, fq[kk & 1]);
      }
      const int row = 16 * kk + lr;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        uint32_t b[4];
        ldsm_x4_t(b, buf + swz_w(row, 2 * j + lg));
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          mma16816(acc[m][2 * j], a[m], b[0], b[1]);
          mma16816(acc[m][2 * j + 1], a[m], b[2], b[3]);
        }
      }
    }
  }
  cp_async_wait<0>();
  // partial sums: acc[m][J] = rows (hidden) 16 m + g, + 8 x columns 8 J + 2 q + e
  const int g8 = lane >> 2, q = lane & 3;
  float* out = p.part + (((size_t)rr * 3 + mm) * 512 + 128 * sl + 32 * hq) * 128;
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
      for (int hr = 0; hr < 2; ++hr)
        *reinterpret_cast<float2*>(out + (size_t)(16 * m + 8 * hr + g8) * 128 + 8 * j + 2 * q) =
            make_float2(acc[m][j][2 * hr], acc[m][j][2 * hr + 1]);
}

}  // namespace a100
