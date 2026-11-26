// attn_bwd_dq_sm80.cuh -- triangle-attention core, backward, the query side, A100 / sm_80: dq and the pair bias' gradient.
//
//   P  = 2^(scl q.k + bias log2 e - lse)       recomputed from the forward's log-sum-exp (base 2, scaled domain; masked keys -> 0)
//   dP = dO . V^T,  dS = P (dP - delta)        delta[j] = sum_d o do (the front's backward term, saved by the back's backward)
//   dq = sm_scale  sum_k dS K                  dbias[j, k] = sum_i dS_i[j, k]     (the bias is shared by every pair row i)
//
// The structure of the forward (attn_fwd_sm80.cuh): one CTA task = (z, head h, 128-query tile) x R pair rows, warp w owns 16 queries of every row; the key
// tiles stream through an NKV-stage cp.async ring (K | V of a row's key tile) and the bias tile of a key tile is staged once for the R rows.  Per key tile
// and row (a "sub-step"): S = Q K^T (mma), P, dP = dO V^T (mma; V's rows are the B operand like K's), dS (fp32), dQ += dS K (mma; K via ldmatrix.trans).
// The bias gradient of the R rows of a CTA is summed in registers over the rows of a key tile and written once per key tile as a bf16 PARTIAL
// dbp[row group][z h][j][k]; ``db_reduce_kernel`` adds the L / R partials (fixed order: deterministic).
#pragma once
#include "attn_fwd_sm80.cuh"

namespace a100 {

struct DqParams {
  const __nv_bfloat16 *q, *k, *v;   // token-major [z * L * L][ld] (the front's [T, 512] buffer: row stride 512), head h = columns [32 h, 32 h + 32)
  const __nv_bfloat16* dov;         // [z * L * L][lddo] the attention output's gradient
  const __nv_bfloat16* bias;        // [z][H][L][L] bf16, masked keys = bf16 min
  const float* lse;                 // [z][H][L][L] (row i, query j): the forward's log-sum-exp
  const float* delta;               // [z][H][L][L] (row i, query j): sum_d o do
  __nv_bfloat16* dq;                // token-major [z * L * L][lddq]
  __nv_bfloat16* dbp;               // [L / R][z H][L][L] bf16 partial bias gradients
  long long ldq, ldk, ldv, lddo, lddq;
  int L, H, ZH;
  float scl;                        // log2 e / sqrt(32)
  float sm_scale;                   // 1 / sqrt(32)
};

template <int R_, int NKV_, int BN_, int MINB_>
struct DqCfg {
  static constexpr int R = R_, NKV = NKV_, BN = BN_, MINB = MINB_, NTHR = 256;
  static_assert(BN == 64 || BN == 32, "BN");
  static constexpr int NT = BN / 8, KB = BN / 16;
  static constexpr int Q_BYTES = TA_BM * TA_D * 2;           // one row's Q (and dO) tile: 8 KiB
  static constexpr int KV_BYTES = 2 * BN * TA_D * 2;
  static constexpr int B_BYTES = TA_BM * BN * 2;
  static constexpr int OFF_Q = 0, OFF_DO = OFF_Q + R * Q_BYTES, OFF_KV = OFF_DO + R * Q_BYTES, OFF_B = OFF_KV + NKV * KV_BYTES;
  static constexpr int SMEM = OFF_B + 2 * B_BYTES;
};

template <class G>
__global__ void __launch_bounds__(256, G::MINB) attn_bwd_dq_kernel(const DqParams p) {
  constexpr int R = G::R, NKV = G::NKV, BN = G::BN, NT = G::NT, KB = G::KB;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sbase = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int L = p.L, NJ = L / BN, NS = R * NJ;
  const int qt = blockIdx.x, gi = blockIdx.y, row0 = gi * R;
  const int h = blockIdx.z % p.H, z = blockIdx.z / p.H;
  const long long tok0 = (long long)z * L * L;

  // ---- loads: as the forward: sub-step s = (key tile j = s / R, row r = s % R) brings its K | V, the first row of a key tile also its bias tile
  const KVBiasLoader<G> loader(p.k, p.v, p.ldk, p.ldv, p.bias, L, tok0, row0, h, z, p.H, qt, tid);
  auto issue = [&](int s) { loader.issue(sbase, s, NS, L); };
  // Q and dO of the R rows ride in group 0
#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int c = tid + 256 * i, token = c >> 2, g = c & 3;
      const long long t = tok0 + (long long)(row0 + r) * L + qt * TA_BM + token;
      cp_async16(sbase + G::OFF_Q + r * G::Q_BYTES + swz64(token, g), p.q + t * p.ldq + h * TA_D + g * 8);
      cp_async16(sbase + G::OFF_DO + r * G::Q_BYTES + swz64(token, g), p.dov + t * p.lddo + h * TA_D + g * 8);
    }
  }
#pragma unroll
  for (int s = 0; s < NKV - 1; ++s) issue(s);

  float acc[R][4][4], lsev[R][2], dlt[R][2], b2[NT][4], dbacc[NT][4];
#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[r][i][0] = acc[r][i][1] = acc[r][i][2] = acc[r][i][3] = 0.f;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {                             // the row terms of this thread's two queries
      const long long at = ((long long)(z * p.H + h) * L + (row0 + r)) * L + qt * TA_BM + warp * 16 + g8 + 8 * hh;
      lsev[r][hh] = p.lse[at];
      dlt[r][hh] = p.delta[at];
    }
  }

  const int qrow = warp * 16 + (lane & 7) + 8 * ((lane >> 3) & 1);
  for (int j = 0; j < NJ; ++j) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int s = j * R + r, st = s % NKV;
      cp_async_wait<NKV - 2>();
      __syncthreads();
      issue(s + NKV - 1);
      if (r == 0) {                                              // this key tile's bias fragments (bias log2 e); the bias gradient restarts
        const uint32_t bb = sbase + G::OFF_B + (j & 1) * G::B_BYTES;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const uint32_t u = lds32(bb + bias_off<BN>(warp * 16 + (lane >> 2) + 8 * hh, nt) + (lane & 3) * 4);
            b2[nt][2 * hh] = bf16lo(u) * TA_L2E;
            b2[nt][2 * hh + 1] = bf16hi(u) * TA_L2E;
          }
          dbacc[nt][0] = dbacc[nt][1] = dbacc[nt][2] = dbacc[nt][3] = 0.f;
        }
      }
      const uint32_t kbase = sbase + G::OFF_KV + st * G::KV_BYTES, vbase = kbase + G::KV_BYTES / 2;
      // ---- S = Q K^T and dP = dO V^T
      uint32_t qa[2][4], da[2][4];
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        ldsm_x4(qa[ks], sbase + G::OFF_Q + r * G::Q_BYTES + swz64(qrow, 2 * ks + (lane >> 4)));
        ldsm_x4(da[ks], sbase + G::OFF_DO + r * G::Q_BYTES + swz64(qrow, 2 * ks + (lane >> 4)));
      }
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
          uint32_t kb[4], vb[4];
          ldsm_x4(kb, kbase + swz64(16 * pp + 8 * ((lane >> 4) & 1) + (lane & 7), 2 * ks + ((lane >> 3) & 1)));
          ldsm_x4(vb, vbase + swz64(16 * pp + 8 * ((lane >> 4) & 1) + (lane & 7), 2 * ks + ((lane >> 3) & 1)));
          mma16816(sacc[2 * pp], qa[ks], kb[0], kb[1]);
          mma16816(sacc[2 * pp + 1], qa[ks], kb[2], kb[3]);
          mma16816(dpacc[2 * pp], da[ks], vb[0], vb[1]);
          mma16816(dpacc[2 * pp + 1], da[ks], vb[2], vb[3]);
        }
      // ---- P, dS = P (dP - delta); the bias gradient sums dS over the rows
      uint32_t pa[KB][4];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float pv = ex2f(fmaf(sacc[nt][e], p.scl, b2[nt][e]) - lsev[r][e >> 1]);
          const float dsv = pv * (dpacc[nt][e] - dlt[r][e >> 1]);
          dpacc[nt][e] = dsv;
          dbacc[nt][e] += dsv;
        }
        pa[nt >> 1][(nt & 1) * 2] = pack_bf16(dpacc[nt][0], dpacc[nt][1]);
        pa[nt >> 1][(nt & 1) * 2 + 1] = pack_bf16(dpacc[nt][2], dpacc[nt][3]);
      }
      // ---- dQ += dS K
#pragma unroll
      for (int c = 0; c < KB; ++c)
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          uint32_t kb[4];
          ldsm_x4_t(kb, kbase + swz64(16 * c + 8 * ((lane >> 3) & 1) + (lane & 7), 2 * np + (lane >> 4)));
          mma16816(acc[r][2 * np], pa[c], kb[0], kb[1]);
          mma16816(acc[r][2 * np + 1], pa[c], kb[2], kb[3]);
        }
      if (r == R - 1) {                                          // the bias gradient of this key tile, summed over the R rows: one bf16 partial
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const long long row = (((long long)gi * p.ZH + (z * p.H + h)) * L + qt * TA_BM + warp * 16 + g8 + 8 * hh) * L + j * BN;
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) stg32(p.dbp + row + 8 * nt + 2 * q4, pack_bf16(dbacc[nt][2 * hh], dbacc[nt][2 * hh + 1]));
        }
      }
    }
  }

  // ---- epilogue: dq = sm_scale sum_k dS K, bf16
#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const long long tok = tok0 + (long long)(row0 + r) * L + qt * TA_BM + warp * 16 + g8 + 8 * hh;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
        stg32(p.dq + tok * p.lddq + h * TA_D + 8 * nt + 2 * q4, pack_bf16(acc[r][nt][2 * hh] * p.sm_scale, acc[r][nt][2 * hh + 1] * p.sm_scale));
    }
  }
}

// The bias gradient: db[z h][j][k] = sum over the G row groups of the partials, in a fixed order.  8 consecutive keys per thread (16-byte loads).
__global__ void __launch_bounds__(256) db_reduce_kernel(const __nv_bfloat16* __restrict__ dbp, __nv_bfloat16* __restrict__ db, int groups, long long plane8) {
  const long long i = (long long)blockIdx.x * 256 + threadIdx.x;       // 8 elements each
  if (i >= plane8) return;
  float s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 8
  for (int g = 0; g < groups; ++g) {
    const uint4 u = ldg128(dbp + ((long long)g * plane8 + i) * 8);
    s[0] += bf16lo(u.x); s[1] += bf16hi(u.x); s[2] += bf16lo(u.y); s[3] += bf16hi(u.y);
    s[4] += bf16lo(u.z); s[5] += bf16hi(u.z); s[6] += bf16lo(u.w); s[7] += bf16hi(u.w);
  }
  stg128(db + i * 8, make_uint4(pack_bf16(s[0], s[1]), pack_bf16(s[2], s[3]), pack_bf16(s[4], s[5]), pack_bf16(s[6], s[7])));
}

}  // namespace a100
