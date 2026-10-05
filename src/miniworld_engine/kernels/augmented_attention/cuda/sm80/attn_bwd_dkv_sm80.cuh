// attn_bwd_dkv_sm80.cuh -- attention with a shared pair bias, backward, the key side, A100 / sm_80: dk and dv.
//
//   P^T  = 2^(scl k.q + bscl bias^T - lse[q])                     recomputed in the transposed (key x query) space
//   dP^T = V . dO^T,  dS^T = P^T (dP^T - delta[q])
//   dv = sum_q P^T dO        dk = sm_scale sum_q dS^T Q
//
// One CTA = (head h, 128-key tile) of ONE sample (grid.y): warp w owns the 16 keys [16 w, 16 w + 16): K and V of the tile sit in registers as the A
// operands of S^T = K Q^T and dP^T = V dO^T, dk / dv accumulate over the query tiles of the sample.  The query tiles stream through an NST-stage cp.async
// ring: Q, dO, the row terms lse / delta and the TRANSPOSED bias tile bias^T[k][q] (the host hands the bias transposed: bias_t[h][k][j], so its tile reads like
// the forward's bias tile).  Per query tile (BN = 32): S^T and dP^T (mma; B operands = the Q / dO rows, ldmatrix), P^T and dS^T (fp32), dv += P^T dO and
// dk += dS^T Q (mma; B operands = dO / Q via ldmatrix.trans).  Builds with OB (bf16 gradients) take the bias^T in raw units and start S^T from it through the
// tensor core (see the forward's MODE_PLAIN).
#pragma once
#include "attn_bwd_dq_sm80.cuh"

namespace aa80 {

struct DkvParams {
  const __nv_bfloat16 *q, *k, *v;   // token-major [A * L][ld]
  const __nv_bfloat16* dov;         // [A * L][lddo]
  const __nv_bfloat16* bias_t;      // [H][L][L] transposed: bias_t[h][k][j], natural units, masked keys -inf
  const float* lse;                 // [A][H][L]
  const float* delta;               // [A][H][L]
  void *dk, *dv;                    // [A * L][lddk / lddv]: fp32, or bf16 when the schedule has OB
  const float* kpen;                // KM: per-sample key penalties [A][L] (0 / -inf), sample stride kps floats: the gradients of a masked key are 0
  long long ldq, ldk, ldv, lddo, lddk, lddv, ss, kps;   // ss: tokens between consecutive samples
  int L, H;
  float scl, bscl, sm_scale;
};

template <int HD_, int NST_, int MINB_, int OB_ = 0, int KM_ = 0>
struct DkvCfg {
  static constexpr int HD = HD_, NST = NST_, MINB = MINB_, OB = OB_, KM = KM_, NTHR = 256, BN = 32;   // OB: bf16 gradients; KM: per-sample key mask
  static constexpr int NT = BN / 8, KB = BN / 16, KS = HD / 16, NO = HD / 8;
  using T = Tile<HD>;
  static constexpr int KV_BYTES = BM * T::PITCH;              // the CTA's K (and V) tile, resident
  static constexpr int QD_BYTES = BN * T::PITCH;              // a query tile of Q (and of dO)
  static constexpr int ROWT_BYTES = 2 * BN * 4;               // lse | delta of a query tile
  static constexpr int BT_BYTES = BM * BN * 2;                // the transposed bias tile [128 k][BN q]
  static constexpr int STAGE = 2 * QD_BYTES + ROWT_BYTES + BT_BYTES;
  static constexpr int OFF_K = 0, OFF_V = KV_BYTES, OFF_STAGE = 2 * KV_BYTES;
  static constexpr int SMEM = OFF_STAGE + NST * STAGE;
};

// The cp.async work of one pipeline stage of the key side (query tile s: Q, dO, lse | delta, the transposed bias tile), with everything that depends on the
// thread only made once (see KVBiasLoader).
template <class G>
struct QdLoader {
  static constexpr int HG = G::HD / 8, BN = G::BN, NC = 2 * BN * HG, QN = (NC + 255) / 256, BI = BN / 16;
  uint32_t qd_dst[QN], b_dst[BI], rt_dst;
  const __nv_bfloat16 *qd_src[QN], *b_src[BI];
  const float* rt_src;
  int qd_ld[QN];
  int tid;

  DEVI QdLoader(const __nv_bfloat16* q, const __nv_bfloat16* dov, long long ldq, long long lddo, const __nv_bfloat16* bias_t, const float* lse, const float* delta,
                long long rowtok, long long plane, long long ahl, int kt, int h, int L, int tid_) : tid(tid_) {
#pragma unroll
    for (int it = 0; it < QN; ++it) {                           // Q then dO: BN tokens x HG granules each
      const int c = tid + it * 256;
      const int which = c / (BN * HG), w = c - which * BN * HG, token = w / HG, g = w - token * HG;
      qd_dst[it] = which * G::QD_BYTES + G::T::off(token, g);
      qd_ld[it] = which ? (int)lddo : (int)ldq;
      qd_src[it] = (which ? dov : q) + (rowtok + token) * qd_ld[it] + h * G::HD + g * 8;
    }
    {                                                           // lse | delta of the BN queries: 2 x BN floats = 16 chunks of 16 B (threads 0-15)
      const int which = (tid >> 3) & 1, c = tid & 7;
      rt_src = (which ? delta : lse) + ahl + c * 4;
      rt_dst = 2 * G::QD_BYTES + which * (BN * 4) + c * 16;
    }
#pragma unroll
    for (int i = 0; i < BI; ++i) {                              // the transposed bias tile: 128 rows (keys) x BN / 8 granules
      const int c = tid + 256 * i, krow = c / (BN / 8), gg = c % (BN / 8);
      b_dst[i] = 2 * G::QD_BYTES + G::ROWT_BYTES + swz64(krow, gg);
      b_src[i] = bias_t + (plane + kt * BM + krow) * L + gg * 8;
    }
  }

  DEVI void issue(uint32_t sbase, int s, int NJ) const {
    if (s < NJ) {
      const uint32_t st = sbase + G::OFF_STAGE + (s % G::NST) * G::STAGE;
#pragma unroll
      for (int it = 0; it < QN; ++it)
        if (NC % 256 == 0 || it < QN - 1 || tid + it * 256 < NC) cp_async16(st + qd_dst[it], qd_src[it] + (long long)(s * BN) * qd_ld[it]);
      if (tid < 16) cp_async16(st + rt_dst, rt_src + s * BN);
#pragma unroll
      for (int i = 0; i < BI; ++i) cp_async16(st + b_dst[i], b_src[i] + s * BN);
    }
    cp_async_commit();
  }
};

template <class G>
__global__ void __launch_bounds__(256, G::MINB) attn_bwd_dkv_kernel(const DkvParams p) {
  constexpr int HD = G::HD, NST = G::NST, BN = G::BN, NT = G::NT, KB = G::KB, KS = G::KS, NO = G::NO, HG = HD / 8;
  using T = typename G::T;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sbase = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int L = p.L, NJ = L / BN;
  const int kt = blockIdx.x, a = blockIdx.y, h = blockIdx.z;     // key tile, sample, head
  const long long rowtok = (long long)a * p.ss;
  const long long plane = (long long)h * L;                      // bias_t plane base, in rows of L
  const long long ahl = ((long long)a * p.H + h) * L;            // lse / delta base of (a, h)

  // ---- loads: the K | V tile (group 0), then query tile s per group: Q, dO, lse | delta, the transposed bias tile
  const QdLoader<G> loader(p.q, p.dov, p.ldq, p.lddo, p.bias_t, p.lse, p.delta, rowtok, plane, ahl, kt, h, L, tid);
  auto issue = [&](int s) { loader.issue(sbase, s, NJ); };
  for (int c = tid; c < BM * HG; c += 256) {                     // K and V of the key tile
    const int token = c / HG, g = c - token * HG;
    const long long t = rowtok + kt * BM + token;
    cp_async16(sbase + G::OFF_K + T::off(token, g), p.k + t * p.ldk + h * HD + g * 8);
    cp_async16(sbase + G::OFF_V + T::off(token, g), p.v + t * p.ldv + h * HD + g * 8);
  }
#pragma unroll
  for (int s = 0; s < NST - 1; ++s) issue(s);

  float dkacc[NO][4], dvacc[NO][4];
#pragma unroll
  for (int i = 0; i < NO; ++i) dkacc[i][0] = dkacc[i][1] = dkacc[i][2] = dkacc[i][3] = dvacc[i][0] = dvacc[i][1] = dvacc[i][2] = dvacc[i][3] = 0.f;

  const int krow = warp * 16 + (lane & 7) + 8 * ((lane >> 3) & 1);       // ldmatrix row of this lane in the K / V tile
  // OB (bf16 gradients: the module-level core) takes the bias in raw units (bias / sm_scale) and adds it into S^T through the tensor core, as the forward's MODE_PLAIN
#ifdef AA_NO_RAWB
  constexpr bool RAWB = false;                              // (A/B builds only)
#else
  constexpr bool RAWB = G::OB != 0;
#endif
  uint32_t ba[RAWB ? KB : 1][4];
  const uint32_t idl = bias_identity(lane);
  uint32_t kf[KS][4], vf[KS][4];
  cp_async_wait<NST - 2>();                                    // group 0 (K | V and the first query tile) landed
  __syncthreads();
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    ldsm_x4(kf[ks], sbase + G::OFF_K + T::off(krow, 2 * ks + (lane >> 4)));
    ldsm_x4(vf[ks], sbase + G::OFF_V + T::off(krow, 2 * ks + (lane >> 4)));
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
    if constexpr (RAWB) {
#pragma unroll
      for (int kb = 0; kb < KB; ++kb) ldsm_x4(ba[kb], btile + swz64(krow, 2 * kb + (lane >> 4)));
#pragma unroll
      for (int kb = 0; kb < KB; ++kb) {                          // S^T starts as the bias^T (raw units)
        mma16816(sacc[2 * kb], ba[kb], idl, 0u);
        mma16816(sacc[2 * kb + 1], ba[kb], 0u, idl);
      }
    }
#pragma unroll
    for (int pp = 0; pp < KB; ++pp)
#pragma unroll
      for (int ks = 0; ks < KS; ++ks) {
        uint32_t qb[4], ob[4];
        ldsm_x4(qb, qbase + T::off(16 * pp + 8 * ((lane >> 4) & 1) + (lane & 7), 2 * ks + ((lane >> 3) & 1)));
        ldsm_x4(ob, dobase + T::off(16 * pp + 8 * ((lane >> 4) & 1) + (lane & 7), 2 * ks + ((lane >> 3) & 1)));
        mma16816(sacc[2 * pp], kf[ks], qb[0], qb[1]);
        mma16816(sacc[2 * pp + 1], kf[ks], qb[2], qb[3]);
        mma16816(dpacc[2 * pp], vf[ks], ob[0], ob[1]);
        mma16816(dpacc[2 * pp + 1], vf[ks], ob[2], ob[3]);
      }
    // ---- P^T in place of S^T (bias^T times bscl, the queries' lse)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const float2 lq = *reinterpret_cast<const float2*>(smem_raw + (rowt - sbase) + (8 * nt + 2 * q4) * 4);
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        if constexpr (RAWB) {
          sacc[nt][2 * hh] = ex2f(fmaf(sacc[nt][2 * hh], p.scl, -lq.x));
          sacc[nt][2 * hh + 1] = ex2f(fmaf(sacc[nt][2 * hh + 1], p.scl, -lq.y));
        } else {
          const uint32_t u = lds32(btile + swz64(warp * 16 + g8 + 8 * hh, nt) + q4 * 4);
          sacc[nt][2 * hh] = ex2f(fmaf(sacc[nt][2 * hh], p.scl, bf16lo(u) * p.bscl) - lq.x);
          sacc[nt][2 * hh + 1] = ex2f(fmaf(sacc[nt][2 * hh + 1], p.scl, bf16hi(u) * p.bscl) - lq.y);
        }
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
      for (int np = 0; np < NO / 2; ++np) {
        uint32_t ob[4];
        ldsm_x4_t(ob, dobase + T::off(16 * c + 8 * ((lane >> 3) & 1) + (lane & 7), 2 * np + (lane >> 4)));
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
      for (int np = 0; np < NO / 2; ++np) {
        uint32_t qb[4];
        ldsm_x4_t(qb, qbase + T::off(16 * c + 8 * ((lane >> 3) & 1) + (lane & 7), 2 * np + (lane >> 4)));
        mma16816(dkacc[2 * np], pa[c], qb[0], qb[1]);
        mma16816(dkacc[2 * np + 1], pa[c], qb[2], qb[3]);
      }
  }

  // ---- epilogue: dk = sm_scale sum dS^T Q, dv, fp32
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    const long long tok = rowtok + kt * BM + warp * 16 + g8 + 8 * hh;
    // KM: a masked key has no weight, so its dk / dv are exactly 0 (the loop above ran its row without the penalty: the values are discarded here; rows never mix)
    const bool key_ok = !G::KM || __ldg(p.kpen + (long long)a * p.kps + kt * BM + warp * 16 + g8 + 8 * hh) == 0.f;
#pragma unroll
    for (int nt = 0; nt < NO; ++nt) {
      const float k0 = key_ok ? dkacc[nt][2 * hh] * p.sm_scale : 0.f, k1 = key_ok ? dkacc[nt][2 * hh + 1] * p.sm_scale : 0.f;
      const float v0 = key_ok ? dvacc[nt][2 * hh] : 0.f, v1 = key_ok ? dvacc[nt][2 * hh + 1] : 0.f;
      if constexpr (G::OB) {
        stg32(reinterpret_cast<__nv_bfloat16*>(p.dk) + tok * p.lddk + h * HD + 8 * nt + 2 * q4, pack_bf16(k0, k1));
        stg32(reinterpret_cast<__nv_bfloat16*>(p.dv) + tok * p.lddv + h * HD + 8 * nt + 2 * q4, pack_bf16(v0, v1));
      } else {
        *reinterpret_cast<float2*>(reinterpret_cast<float*>(p.dk) + tok * p.lddk + h * HD + 8 * nt + 2 * q4) = make_float2(k0, k1);
        *reinterpret_cast<float2*>(reinterpret_cast<float*>(p.dv) + tok * p.lddv + h * HD + 8 * nt + 2 * q4) = make_float2(v0, v1);
      }
    }
  }
}

// bias_t[h][k][j] = bias[h][j][k]: 32 x 32 tiles through shared memory (bf16; L a multiple of 32).
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

}  // namespace aa80
