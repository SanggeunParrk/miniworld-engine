// attn_bwd_dkv_sm80.cuh -- triangle-attention core, backward, the key side, A100 / sm_80: dk and dv.
//
//   P^T  = 2^(scl k.q + bias^T log2 e - lse[q])                   recomputed in the transposed (key x query) space
//   dP^T = V . dO^T,  dS^T = P^T (dP^T - delta[q])
//   dv = sum_q P^T dO        dk = sm_scale sum_q dS^T Q
//
// One CTA task = (z, head h, 128-key tile) of ONE pair row i (grid.y); warp w owns the 16 keys [16 w, 16 w + 16): K and V of the tile sit in registers as the
// A operands of S^T = K Q^T and dP^T = V dO^T, dk / dv accumulate over the query tiles of the row.  The query tiles stream through an NST-stage
// cp.async ring: Q, dO, the row terms lse / delta and the TRANSPOSED bias tile bias^T[k][q] (the host hands the bias transposed: [z h][k][j], so its tile
// reads like the forward's bias tile).  Per query tile (BN = 32): S^T and dP^T (mma; B operands = the Q / dO rows, ldmatrix), P^T and dS^T (fp32),
// dv += P^T dO and dk += dS^T Q (mma; B operands = dO / Q via ldmatrix.trans).  The row terms of a thread's columns are two floats per n tile (shared memory).
#pragma once
#include "attn_fwd_sm80.cuh"

namespace a100 {

struct DkvParams {
  const __nv_bfloat16 *q, *k, *v;   // token-major [z * L * L][ld]
  const __nv_bfloat16* dov;         // [z * L * L][lddo]
  const __nv_bfloat16* bias_t;      // [z][H][L][L] transposed: bias_t[k][j], masked keys = bf16 min
  const float* lse;                 // [z][H][L][L] (row i, query j)
  const float* delta;               // [z][H][L][L] (row i, query j)
  __nv_bfloat16 *dk, *dv;           // token-major [z * L * L][lddk / lddv]
  long long ldq, ldk, ldv, lddo, lddk, lddv;
  int L, H;
  float scl, sm_scale;
};

template <int NST_, int BN_, int MINB_>
struct DkvCfg {
  static constexpr int NST = NST_, BN = BN_, MINB = MINB_, NTHR = 256;
  static_assert(BN == 32, "BN");
  static constexpr int NT = BN / 8, KB = BN / 16;
  static constexpr int KV_BYTES = TA_BM * TA_D * 2;           // the CTA's K (and V) tile: 8 KiB each, resident
  static constexpr int QD_BYTES = BN * TA_D * 2;              // a query tile of Q (and of dO)
  static constexpr int ROWT_BYTES = 2 * BN * 4;               // lse | delta of a query tile
  static constexpr int BT_BYTES = TA_BM * BN * 2;             // the transposed bias tile [128 k][BN q]
  static constexpr int STAGE = 2 * QD_BYTES + ROWT_BYTES + BT_BYTES;
  static constexpr int OFF_K = 0, OFF_V = KV_BYTES, OFF_STAGE = 2 * KV_BYTES;
  static constexpr int SMEM = OFF_STAGE + NST * STAGE;
};

// The cp.async work of one pipeline stage of the key side (query tile s of the pair row: Q, dO, lse | delta, the transposed bias tile) with everything that depends
// on the thread only made once (see KVBiasLoader).
template <class G>
struct QdLoader {
  static constexpr int BN = G::BN, BI = BN / 16;
  uint32_t qd_dst, rt_dst, b_dst[BI];
  const __nv_bfloat16 *qd_src, *b_src[BI];
  const float* rt_src;
  long long qd_ld;
  bool is_q;
  int tid;

  DEVI QdLoader(const __nv_bfloat16* q, const __nv_bfloat16* dov, long long ldq, long long lddo, const __nv_bfloat16* bias_t, const float* lse, const float* delta,
                long long rowtok, long long plane, int row, int kt, int h, int L, int tid_) : tid(tid_) {
    is_q = tid < 128;                                          // threads 0-127: Q (BN tokens x 4 granules), 128-255: dO
    const int token = (tid & 127) >> 2, g = tid & 3;
    qd_dst = (is_q ? 0 : G::QD_BYTES) + swz64(token, g);
    qd_ld = is_q ? ldq : lddo;
    qd_src = (is_q ? q : dov) + (rowtok + token) * qd_ld + h * TA_D + g * 8;
    {                                                          // lse | delta of the BN queries: 2 x BN floats = 16 chunks of 16 B (threads 0-15)
      const int which = (tid >> 3) & 1, c = tid & 7;
      rt_src = (which ? delta : lse) + (plane + row) * L + c * 4;
      rt_dst = 2 * G::QD_BYTES + which * (BN * 4) + c * 16;
    }
#pragma unroll
    for (int i = 0; i < BI; ++i) {                             // the transposed bias tile: 128 rows (keys) x BN / 8 granules
      const int c = tid + 256 * i, krow = c / (BN / 8), gg = c % (BN / 8);
      b_dst[i] = 2 * G::QD_BYTES + G::ROWT_BYTES + bias_off<BN>(krow, gg);
      b_src[i] = bias_t + (plane + kt * TA_BM + krow) * L + gg * 8;
    }
  }

  DEVI void issue(uint32_t sbase, int s, int NJ) const {
    if (s < NJ) {
      const uint32_t st = sbase + G::OFF_STAGE + (s % G::NST) * G::STAGE;
      cp_async16(st + qd_dst, qd_src + (long long)(s * BN) * qd_ld);
      if (tid < 16) cp_async16(st + rt_dst, rt_src + s * BN);
#pragma unroll
      for (int i = 0; i < BI; ++i) cp_async16(st + b_dst[i], b_src[i] + s * BN);
    }
    cp_async_commit();
  }
};

template <class G>
__global__ void __launch_bounds__(256, G::MINB) attn_bwd_dkv_kernel(const DkvParams p) {
  constexpr int NST = G::NST, BN = G::BN, NT = G::NT, KB = G::KB;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sbase = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int L = p.L, NJ = L / BN;
  const int kt = blockIdx.x, row = blockIdx.y;                    // key tile, pair row
  const int h = blockIdx.z % p.H, z = blockIdx.z / p.H;
  const long long tok0 = (long long)z * L * L, rowtok = tok0 + (long long)row * L;
  const long long plane = ((long long)(z * p.H + h) * L);       // (z h) plane base, in rows of L

  // ---- loads: the K | V tile (group 0), then query tile jt = s of the row per group: Q, dO, lse | delta, the transposed bias tile
  const QdLoader<G> loader(p.q, p.dov, p.ldq, p.lddo, p.bias_t, p.lse, p.delta, rowtok, plane, row, kt, h, L, tid);
  auto issue = [&](int s) { loader.issue(sbase, s, NJ); };
#pragma unroll
  for (int i = 0; i < 2; ++i) {                                // K and V of the key tile
    const int c = tid + 256 * i, token = c >> 2, g = c & 3;
    const long long t = rowtok + kt * TA_BM + token;
    cp_async16(sbase + G::OFF_K + swz64(token, g), p.k + t * p.ldk + h * TA_D + g * 8);
    cp_async16(sbase + G::OFF_V + swz64(token, g), p.v + t * p.ldv + h * TA_D + g * 8);
  }
#pragma unroll
  for (int s = 0; s < NST - 1; ++s) issue(s);

  float dkacc[4][4], dvacc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) dkacc[i][0] = dkacc[i][1] = dkacc[i][2] = dkacc[i][3] = dvacc[i][0] = dvacc[i][1] = dvacc[i][2] = dvacc[i][3] = 0.f;

  const int krow = warp * 16 + (lane & 7) + 8 * ((lane >> 3) & 1);       // ldmatrix row of this lane in the K / V tile
  uint32_t kf[2][4], vf[2][4];
  cp_async_wait<NST - 2>();                                    // group 0 (K | V and the first query tile) landed
  __syncthreads();
#pragma unroll
  for (int ks = 0; ks < 2; ++ks) {
    ldsm_x4(kf[ks], sbase + G::OFF_K + swz64(krow, 2 * ks + (lane >> 4)));
    ldsm_x4(vf[ks], sbase + G::OFF_V + swz64(krow, 2 * ks + (lane >> 4)));
  }

  for (int s = 0; s < NJ; ++s) {
    cp_async_wait<NST - 2>();
    __syncthreads();                                           // query tile s landed; every warp is past s - 1
    issue(s + NST - 1);
    const uint32_t st = sbase + G::OFF_STAGE + (s % NST) * G::STAGE;
    const uint32_t qbase = st, dobase = st + G::QD_BYTES, rowt = st + 2 * G::QD_BYTES, btile = st + 2 * G::QD_BYTES + G::ROWT_BYTES;
    // ---- S^T = K Q^T and dP^T = V dO^T  (16 keys x BN queries per warp)
    float sacc[NT][4], dpacc[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      sacc[nt][0] = sacc[nt][1] = sacc[nt][2] = sacc[nt][3] = 0.f;
      dpacc[nt][0] = dpacc[nt][1] = dpacc[nt][2] = dpacc[nt][3] = 0.f;
    }
#pragma unroll
    for (int pp = 0; pp < KB; ++pp)
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        uint32_t qb[4], ob[4];
        ldsm_x4(qb, qbase + swz64(16 * pp + 8 * ((lane >> 4) & 1) + (lane & 7), 2 * ks + ((lane >> 3) & 1)));
        ldsm_x4(ob, dobase + swz64(16 * pp + 8 * ((lane >> 4) & 1) + (lane & 7), 2 * ks + ((lane >> 3) & 1)));
        mma16816(sacc[2 * pp], kf[ks], qb[0], qb[1]);
        mma16816(sacc[2 * pp + 1], kf[ks], qb[2], qb[3]);
        mma16816(dpacc[2 * pp], vf[ks], ob[0], ob[1]);
        mma16816(dpacc[2 * pp + 1], vf[ks], ob[2], ob[3]);
      }
    // ---- P^T in place of S^T (bias^T log2 e, the queries' lse)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const float2 lq = *reinterpret_cast<const float2*>(smem_raw + (rowt - sbase) + (8 * nt + 2 * q4) * 4);
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const uint32_t u = lds32(btile + bias_off<BN>(warp * 16 + g8 + 8 * hh, nt) + q4 * 4);
        sacc[nt][2 * hh] = ex2f(fmaf(sacc[nt][2 * hh], p.scl, bf16lo(u) * TA_L2E) - lq.x);
        sacc[nt][2 * hh + 1] = ex2f(fmaf(sacc[nt][2 * hh + 1], p.scl, bf16hi(u) * TA_L2E) - lq.y);
      }
    }
    // ---- dv += P^T dO
    uint32_t pa[KB][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      pa[nt >> 1][(nt & 1) * 2] = pack_bf16(sacc[nt][0], sacc[nt][1]);
      pa[nt >> 1][(nt & 1) * 2 + 1] = pack_bf16(sacc[nt][2], sacc[nt][3]);
    }
#pragma unroll
    for (int c = 0; c < KB; ++c)
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        uint32_t ob[4];
        ldsm_x4_t(ob, dobase + swz64(16 * c + 8 * ((lane >> 3) & 1) + (lane & 7), 2 * np + (lane >> 4)));
        mma16816(dvacc[2 * np], pa[c], ob[0], ob[1]);
        mma16816(dvacc[2 * np + 1], pa[c], ob[2], ob[3]);
      }
    // ---- dS^T = P^T (dP^T - delta), dk += dS^T Q
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const float2 dq2 = *reinterpret_cast<const float2*>(smem_raw + (rowt - sbase) + BN * 4 + (8 * nt + 2 * q4) * 4);
      sacc[nt][0] *= dpacc[nt][0] - dq2.x;
      sacc[nt][1] *= dpacc[nt][1] - dq2.y;
      sacc[nt][2] *= dpacc[nt][2] - dq2.x;
      sacc[nt][3] *= dpacc[nt][3] - dq2.y;
      pa[nt >> 1][(nt & 1) * 2] = pack_bf16(sacc[nt][0], sacc[nt][1]);
      pa[nt >> 1][(nt & 1) * 2 + 1] = pack_bf16(sacc[nt][2], sacc[nt][3]);
    }
#pragma unroll
    for (int c = 0; c < KB; ++c)
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        uint32_t qb[4];
        ldsm_x4_t(qb, qbase + swz64(16 * c + 8 * ((lane >> 3) & 1) + (lane & 7), 2 * np + (lane >> 4)));
        mma16816(dkacc[2 * np], pa[c], qb[0], qb[1]);
        mma16816(dkacc[2 * np + 1], pa[c], qb[2], qb[3]);
      }
  }

  // ---- epilogue: dk = sm_scale sum dS^T Q, dv, bf16
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    const long long tok = rowtok + kt * TA_BM + warp * 16 + g8 + 8 * hh;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      stg32(p.dk + tok * p.lddk + h * TA_D + 8 * nt + 2 * q4, pack_bf16(dkacc[nt][2 * hh] * p.sm_scale, dkacc[nt][2 * hh + 1] * p.sm_scale));
      stg32(p.dv + tok * p.lddv + h * TA_D + 8 * nt + 2 * q4, pack_bf16(dvacc[nt][2 * hh], dvacc[nt][2 * hh + 1]));
    }
  }
}

// bias_t[zh][k][j] = bias[zh][j][k]: 32 x 32 tiles through shared memory (bf16; L a multiple of 32).
__global__ void __launch_bounds__(256) bias_transpose_kernel(const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ bias_t, int L) {
  __shared__ __nv_bfloat16 tile[32][34];
  const long long base = (long long)blockIdx.z * L * L;
  const int j0 = blockIdx.y * 32, k0 = blockIdx.x * 32, tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
#pragma unroll
  for (int i = ty; i < 32; i += 8) tile[i][tx] = bias[base + (long long)(j0 + i) * L + k0 + tx];
  __syncthreads();
#pragma unroll
  for (int i = ty; i < 32; i += 8) bias_t[base + (long long)(k0 + i) * L + j0 + tx] = tile[tx][i];
}

}  // namespace a100
