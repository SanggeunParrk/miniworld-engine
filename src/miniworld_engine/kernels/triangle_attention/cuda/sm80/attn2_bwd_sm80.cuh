// attn2_bwd_sm80.cuh -- the backward of attn2_fwd_sm80.cuh (head dim HD = 16 | 32, element strides, optional per-pair-row key mask): the query side (dq and the pair bias' gradient
// partials, as attn_bwd_dq_sm80.cuh) and the key side (dk, dv, as attn_bwd_dkv_sm80.cuh).  A masked key has weight 0 in the recomputed P, so it contributes nothing to dS, dq, dk, dv or
// the bias gradient.
#pragma once
#include <type_traits>

#include "attn2_fwd_sm80.cuh"

namespace a100 {

// ------------------------------------------------------------------------------------------------------------------------------------------------------- query side
struct Dq2Params {
  const __nv_bfloat16 *q, *k, *v;   // strided (Lay lq, lk, lv)
  const __nv_bfloat16* dov;         // strided (Lay ld): the attention output's gradient
  const __nv_bfloat16* bias;        // [z][H][L][L] bf16, masked keys = bf16 min
  const uint8_t* rowmask;           // [z][L][L] or nullptr
  const float* lse;                 // [z][H][L][L] (row i, query j): the forward's log-sum-exp
  const float* delta;               // [z][H][L][L]: sum_d o do
  __nv_bfloat16* dq;                // strided (Lay ldq)
  __nv_bfloat16* dbp;               // [L / R][z H][L][L] bf16 partial bias gradients
  Lay lq, lk, lv, ld, ldq;
  int L, H, ZH;
  float scl;                        // log2 e / sqrt(HD)
  float sm_scale;                   // 1 / sqrt(HD)
};

template <int HD_, int R_, int NKV_, int BN_, int MINB_, bool MASK_>
struct Dq2Cfg {
  using T = Hd<HD_>;
  static constexpr int HD = HD_, R = R_, NKV = NKV_, BN = BN_, MINB = MINB_, NTHR = 256;
  static constexpr bool MASK = MASK_;
  static_assert(BN == 64 || BN == 32, "BN");
  static_assert(!MASK_ || BN == 32, "the masked variants read one mask word per key tile");
  static constexpr int NT = BN / 8, KB = BN / 16;
  static constexpr int Q_BYTES = T2_BM * HD * 2;             // one row's Q (and dO) tile
  static constexpr int KV_BYTES = 2 * BN * HD * 2;
  static constexpr int B_BYTES = T2_BM * BN * 2;
  static constexpr int OFF_Q = 0, OFF_DO = OFF_Q + R * Q_BYTES, OFF_KV = OFF_DO + R * Q_BYTES, OFF_B = OFF_KV + NKV * KV_BYTES, OFF_M = OFF_B + 2 * B_BYTES;
  static constexpr int SMEM = OFF_M + (MASK ? R * MASK_WORDS * 4 : 0);
};

// One sub-step of the query side -- key tile (K | V at kbase) of the pair row whose Q and dO tiles are at qaddr / daddr: S = Q K^T and dP = dO V^T, P, dS = P (dP - delta) (summed into the bias
// gradient's accumulators) and dQ += dS K for the warp's 16 queries.  MK: the tile has a masked key (a masked key has weight 0); the caller branches once per sub-step on the tile's bit word.
template <class G, bool MK>
DEVI void dq_step(const Dq2Params& p, uint32_t qaddr, uint32_t daddr, uint32_t kbase, uint32_t bm, int lane, int warp, const float (&b2)[G::NT][4], const float (&lsev)[2],
                  const float (&dlt)[2], float (&acc)[Hd<G::HD>::NTV][4], float (&dbacc)[G::NT][4]) {
  constexpr int HD = G::HD, NT = G::NT, KB = G::KB, KS = Hd<HD>::KS, NTV = Hd<HD>::NTV;
  using T = Hd<HD>;
  const int qrow = warp * 16 + (lane & 7) + 8 * ((lane >> 3) & 1);
  const uint32_t vbase = kbase + G::KV_BYTES / 2;
  // ---- S = Q K^T and dP = dO V^T
  uint32_t qa[KS][4], da[KS][4];
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    ldsm_x4(qa[ks], qaddr + T::swz(qrow, 2 * ks + (lane >> 4)));
    ldsm_x4(da[ks], daddr + T::swz(qrow, 2 * ks + (lane >> 4)));
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
    for (int ks = 0; ks < KS; ++ks) {
      uint32_t kb[4], vb[4];
      ldsm_x4(kb, kbase + T::swz(16 * pp + 8 * ((lane >> 4) & 1) + (lane & 7), 2 * ks + ((lane >> 3) & 1)));
      ldsm_x4(vb, vbase + T::swz(16 * pp + 8 * ((lane >> 4) & 1) + (lane & 7), 2 * ks + ((lane >> 3) & 1)));
      mma16816(sacc[2 * pp], qa[ks], kb[0], kb[1]);
      mma16816(sacc[2 * pp + 1], qa[ks], kb[2], kb[3]);
      mma16816(dpacc[2 * pp], da[ks], vb[0], vb[1]);
      mma16816(dpacc[2 * pp + 1], da[ks], vb[2], vb[3]);
    }
  // ---- P, dS = P (dP - delta); the bias gradient sums dS over the rows
  [[maybe_unused]] const uint32_t sh = bm >> (2 * (lane & 3));      // columns 8 nt + 2 q4 and + 1 of this thread are bits 8 nt and 8 nt + 1 of bm >> 2 q4
  uint32_t pa[KB][4];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      float pv = ex2f2(fmaf(sacc[nt][e], p.scl, b2[nt][e]) - lsev[e >> 1]);
      if constexpr (MK) pv = ((sh >> (8 * nt + (e & 1))) & 1u) != 0 ? pv : 0.f;
      const float dsv = pv * (dpacc[nt][e] - dlt[e >> 1]);
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
    for (int np = 0; np < NTV / 2; ++np) {
      uint32_t kb[4];
      ldsm_x4_t(kb, kbase + T::swz(16 * c + 8 * ((lane >> 3) & 1) + (lane & 7), 2 * np + (lane >> 4)));
      mma16816(acc[2 * np], pa[c], kb[0], kb[1]);
      mma16816(acc[2 * np + 1], pa[c], kb[2], kb[3]);
    }
}

template <class G>
__global__ void __launch_bounds__(256, G::MINB) attn2_bwd_dq_kernel(const Dq2Params p) {
  constexpr int HD = G::HD, R = G::R, NKV = G::NKV, BN = G::BN, NT = G::NT, NTV = Hd<HD>::NTV, GR = Hd<HD>::GR;
  using T = Hd<HD>;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sbase = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int L = p.L, NJ = L / BN, NS = R * NJ;
  const int qt = blockIdx.x, gi = blockIdx.y, row0 = gi * R;
  const int h = blockIdx.z % p.H, z = blockIdx.z / p.H;

  const KV2Loader<G> loader(p, z, row0, h, qt, tid);
  auto issue = [&](int s) { loader.issue(sbase, s, NS); };
  // Q and dO of the R rows ride in group 0
#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int i = 0; i < (T2_BM * GR + 255) / 256; ++i) {
      const int c = tid + 256 * i, token = c / GR, g = c % GR;
      if (c < T2_BM * GR) {
        const long long tq = (long long)(qt * T2_BM + token);
        cp_async16(sbase + G::OFF_Q + r * G::Q_BYTES + T::swz(token, g), p.q + z * p.lq.z + (row0 + r) * p.lq.row + tq * p.lq.tok + h * p.lq.head + g * 8);
        cp_async16(sbase + G::OFF_DO + r * G::Q_BYTES + T::swz(token, g), p.dov + z * p.ld.z + (row0 + r) * p.ld.row + tq * p.ld.tok + h * p.ld.head + g * 8);
      }
    }
  }
#pragma unroll
  for (int s = 0; s < NKV - 1; ++s) issue(s);
  if constexpr (G::MASK) build_mask_words<R>(sbase + G::OFF_M, p.rowmask, (long long)z * L + row0, L, tid);     // while the first stages are in flight; published by the first barrier of the main loop

  float acc[R][NTV][4], lsev[R][2], dlt[R][2], b2[NT][4], dbacc[NT][4];
#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int i = 0; i < NTV; ++i) acc[r][i][0] = acc[r][i][1] = acc[r][i][2] = acc[r][i][3] = 0.f;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {                             // the row terms of this thread's two queries
      const long long at = ((long long)(z * p.H + h) * L + (row0 + r)) * L + qt * T2_BM + warp * 16 + g8 + 8 * hh;
      lsev[r][hh] = p.lse[at];
      dlt[r][hh] = p.delta[at];
    }
  }

  uint32_t bm_next = ALL_KEYS;
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
            const uint32_t u = lds32(bb + bias_off2<BN>(warp * 16 + (lane >> 2) + 8 * hh, nt) + (lane & 3) * 4);
            b2[nt][2 * hh] = bf16lo(u) * T2_L2E;
            b2[nt][2 * hh + 1] = bf16hi(u) * T2_L2E;
          }
          dbacc[nt][0] = dbacc[nt][1] = dbacc[nt][2] = dbacc[nt][3] = 0.f;
        }
      }
      uint32_t bm = ALL_KEYS;                            // the key mask of this (row, key tile): one bit per key
      if constexpr (G::MASK) {
        if (r == 0 && j == 0) bm_next = lds32(sbase + G::OFF_M);       // the first word, once the barrier has published the words
        bm = bm_next;
        bm_next = lds32(sbase + G::OFF_M + (((r + 1 < R) ? r + 1 : 0) * MASK_WORDS + ((r + 1 < R) ? j : (j + 1 < NJ ? j + 1 : j))) * 4);    // the next sub-step's word, its latency hidden behind this one's products
      }
      const uint32_t qaddr = sbase + G::OFF_Q + r * G::Q_BYTES, daddr = sbase + G::OFF_DO + r * G::Q_BYTES, kbase = sbase + G::OFF_KV + st * G::KV_BYTES;
      if constexpr (G::MASK) {
        if (bm != ALL_KEYS) dq_step<G, true>(p, qaddr, daddr, kbase, bm, lane, warp, b2, lsev[r], dlt[r], acc[r], dbacc);      // a tile with a masked key
        else dq_step<G, false>(p, qaddr, daddr, kbase, bm, lane, warp, b2, lsev[r], dlt[r], acc[r], dbacc);
      } else {
        dq_step<G, false>(p, qaddr, daddr, kbase, bm, lane, warp, b2, lsev[r], dlt[r], acc[r], dbacc);
      }
      if (r == R - 1) {                                          // the bias gradient of this key tile, summed over the R rows: one bf16 partial
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const long long row = (((long long)gi * p.ZH + (z * p.H + h)) * L + qt * T2_BM + warp * 16 + g8 + 8 * hh) * L + j * BN;
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
      const long long tq = (long long)(qt * T2_BM + warp * 16 + g8 + 8 * hh);
      __nv_bfloat16* drow = p.dq + z * p.ldq.z + (row0 + r) * p.ldq.row + tq * p.ldq.tok + h * p.ldq.head;
#pragma unroll
      for (int nt = 0; nt < NTV; ++nt) stg32(drow + 8 * nt + 2 * q4, pack_bf16(acc[r][nt][2 * hh] * p.sm_scale, acc[r][nt][2 * hh + 1] * p.sm_scale));
    }
  }
}

// -------------------------------------------------------------------------------------------------------------------------------------------------------- key side
struct Dkv2Params {
  const __nv_bfloat16 *q, *k, *v;   // strided (Lay lq, lk, lv)
  const __nv_bfloat16* dov;         // strided (Lay ld)
  const __nv_bfloat16* bias_t;      // [z][H][L][L] transposed: bias_t[k][j], masked keys = bf16 min
  const uint8_t* rowmask;           // [z][L][L] or nullptr
  const float* lse;                 // [z][H][L][L] (row i, query j)
  const float* delta;               // [z][H][L][L]
  __nv_bfloat16 *dk, *dv;           // strided (Lay ldk, ldv)
  Lay lq, lk, lv, ld, ldk, ldv;
  int L, H;
  float scl, sm_scale;
};

template <int HD_, int NST_, int BN_, int MINB_, bool MASK_>
struct Dkv2Cfg {
  using T = Hd<HD_>;
  static constexpr int HD = HD_, NST = NST_, BN = BN_, MINB = MINB_, NTHR = 256;
  static constexpr bool MASK = MASK_;
  static_assert(BN == 32, "BN");
  static constexpr int NT = BN / 8, KB = BN / 16;
  static constexpr int KV_BYTES = T2_BM * HD * 2;             // the CTA's K (and V) tile, resident
  static constexpr int QD_BYTES = BN * HD * 2;                // a query tile of Q (and of dO)
  static constexpr int ROWT_BYTES = 2 * BN * 4;               // lse | delta of a query tile
  static constexpr int BT_BYTES = T2_BM * BN * 2;             // the transposed bias tile [128 k][BN q]
  static constexpr int STAGE = 2 * QD_BYTES + ROWT_BYTES + BT_BYTES;
  static constexpr int OFF_K = 0, OFF_V = KV_BYTES, OFF_STAGE = 2 * KV_BYTES;
  static constexpr int SMEM = OFF_STAGE + NST * STAGE;
};

template <class G>
struct Qd2Loader {
  static constexpr int BN = G::BN, BI = BN / 16, GR = Hd<G::HD>::GR, QG = BN * GR;       // granules of a Q (dO) tile
  uint32_t qd_dst, rt_dst, b_dst[BI];
  const __nv_bfloat16 *qd_src, *b_src[BI];
  const float* rt_src;
  long long qd_tok;
  bool is_q, qd_ok;
  int tid;

  DEVI Qd2Loader(const Dkv2Params& p, long long plane, int row, int z, int kt, int h, int L, int tid_) : tid(tid_) {
    is_q = tid < 128;                                          // threads 0-127: Q, 128-255: dO
    const int c = tid & 127;
    qd_ok = c < QG;
    const int token = c / GR, g = c % GR;
    const Lay& l = is_q ? p.lq : p.ld;
    qd_dst = (is_q ? 0 : G::QD_BYTES) + Hd<G::HD>::swz(token, g);
    qd_tok = l.tok;
    qd_src = (is_q ? p.q : p.dov) + z * l.z + row * l.row + token * l.tok + h * l.head + g * 8;
    {                                                          // lse | delta of the BN queries: 2 x BN floats = 16 chunks of 16 B (threads 0-15)
      const int which = (tid >> 3) & 1, cc = tid & 7;
      rt_src = (which ? p.delta : p.lse) + (plane + row) * L + cc * 4;
      rt_dst = 2 * G::QD_BYTES + which * (BN * 4) + cc * 16;
    }
#pragma unroll
    for (int i = 0; i < BI; ++i) {                             // the transposed bias tile: 128 rows (keys) x BN / 8 granules
      const int cb = tid + 256 * i, krow = cb / (BN / 8), gg = cb % (BN / 8);
      b_dst[i] = 2 * G::QD_BYTES + G::ROWT_BYTES + bias_off2<BN>(krow, gg);
      b_src[i] = p.bias_t + (plane + kt * T2_BM + krow) * L + gg * 8;
    }
  }

  DEVI void issue(uint32_t sbase, int s, int NJ) const {
    if (s < NJ) {
      const uint32_t st = sbase + G::OFF_STAGE + (s % G::NST) * G::STAGE;
      if (qd_ok) cp_async16(st + qd_dst, qd_src + (long long)(s * BN) * qd_tok);
      if (tid < 16) cp_async16(st + rt_dst, rt_src + s * BN);
#pragma unroll
      for (int i = 0; i < BI; ++i) cp_async16(st + b_dst[i], b_src[i] + s * BN);
    }
    cp_async_commit();
  }
};

template <class G>
__global__ void __launch_bounds__(256, G::MINB) attn2_bwd_dkv_kernel(const Dkv2Params p) {
  constexpr int HD = G::HD, NST = G::NST, BN = G::BN, NT = G::NT, KB = G::KB, KS = Hd<HD>::KS, NTV = Hd<HD>::NTV, GR = Hd<HD>::GR;
  using T = Hd<HD>;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sbase = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int L = p.L, NJ = L / BN;
  const int kt = blockIdx.x, row = blockIdx.y;                    // key tile, pair row
  const int h = blockIdx.z % p.H, z = blockIdx.z / p.H;
  const long long plane = ((long long)(z * p.H + h) * L);       // (z h) plane base, in rows of L

  const Qd2Loader<G> loader(p, plane, row, z, kt, h, L, tid);
  auto issue = [&](int s) { loader.issue(sbase, s, NJ); };
#pragma unroll
  for (int i = 0; i < (T2_BM * GR + 255) / 256; ++i) {          // K and V of the key tile
    const int c = tid + 256 * i, token = c / GR, g = c % GR;
    if (c < T2_BM * GR) {
      const long long tk = (long long)(kt * T2_BM + token);
      cp_async16(sbase + G::OFF_K + T::swz(token, g), p.k + z * p.lk.z + row * p.lk.row + tk * p.lk.tok + h * p.lk.head + g * 8);
      cp_async16(sbase + G::OFF_V + T::swz(token, g), p.v + z * p.lv.z + row * p.lv.row + tk * p.lv.tok + h * p.lv.head + g * 8);
    }
  }
#pragma unroll
  for (int s = 0; s < NST - 1; ++s) issue(s);

  float dkacc[NTV][4], dvacc[NTV][4];
#pragma unroll
  for (int i = 0; i < NTV; ++i) dkacc[i][0] = dkacc[i][1] = dkacc[i][2] = dkacc[i][3] = dvacc[i][0] = dvacc[i][1] = dvacc[i][2] = dvacc[i][3] = 0.f;

  bool keep[2] = {true, true};                                   // the key mask of this thread's two key rows (the same for every query tile)
  bool all_keys = true;                                          // every key of this warp's 16 is real: the masking below is skipped (a branch the whole warp takes alike)
  if constexpr (G::MASK) {
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) keep[hh] = p.rowmask[((long long)z * L + row) * L + kt * T2_BM + warp * 16 + g8 + 8 * hh] != 0;
    all_keys = __all_sync(0xffffffffu, keep[0] && keep[1]);
  }

  const int krow = warp * 16 + (lane & 7) + 8 * ((lane >> 3) & 1);       // ldmatrix row of this lane in the K / V tile
  uint32_t kf[KS][4], vf[KS][4];
  cp_async_wait<NST - 2>();                                    // group 0 (K | V and the first query tile) landed
  __syncthreads();
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    ldsm_x4(kf[ks], sbase + G::OFF_K + T::swz(krow, 2 * ks + (lane >> 4)));
    ldsm_x4(vf[ks], sbase + G::OFF_V + T::swz(krow, 2 * ks + (lane >> 4)));
  }

  // the query-tile loop, in two copies: with the masking of the keys (some key of this warp's 16 is masked) and without (the code of a kernel without a mask)
  auto run = [&](auto mk) {
  constexpr bool MK = decltype(mk)::value;
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
      for (int ks = 0; ks < KS; ++ks) {
        uint32_t qb[4], ob[4];
        ldsm_x4(qb, qbase + T::swz(16 * pp + 8 * ((lane >> 4) & 1) + (lane & 7), 2 * ks + ((lane >> 3) & 1)));
        ldsm_x4(ob, dobase + T::swz(16 * pp + 8 * ((lane >> 4) & 1) + (lane & 7), 2 * ks + ((lane >> 3) & 1)));
        mma16816(sacc[2 * pp], kf[ks], qb[0], qb[1]);
        mma16816(sacc[2 * pp + 1], kf[ks], qb[2], qb[3]);
        mma16816(dpacc[2 * pp], vf[ks], ob[0], ob[1]);
        mma16816(dpacc[2 * pp + 1], vf[ks], ob[2], ob[3]);
      }
    // ---- P^T in place of S^T (bias^T log2 e, the queries' lse); a masked key row has weight 0
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const float2 lq = *reinterpret_cast<const float2*>(smem_raw + (rowt - sbase) + (8 * nt + 2 * q4) * 4);
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const uint32_t u = lds32(btile + bias_off2<BN>(warp * 16 + g8 + 8 * hh, nt) + q4 * 4);
        sacc[nt][2 * hh] = ex2f2(fmaf(sacc[nt][2 * hh], p.scl, bf16lo(u) * T2_L2E) - lq.x);
        sacc[nt][2 * hh + 1] = ex2f2(fmaf(sacc[nt][2 * hh + 1], p.scl, bf16hi(u) * T2_L2E) - lq.y);
      }
    }
    if constexpr (MK) {
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          sacc[nt][2 * hh] = keep[hh] ? sacc[nt][2 * hh] : 0.f;
          sacc[nt][2 * hh + 1] = keep[hh] ? sacc[nt][2 * hh + 1] : 0.f;
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
      for (int np = 0; np < NTV / 2; ++np) {
        uint32_t ob[4];
        ldsm_x4_t(ob, dobase + T::swz(16 * c + 8 * ((lane >> 3) & 1) + (lane & 7), 2 * np + (lane >> 4)));
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
      for (int np = 0; np < NTV / 2; ++np) {
        uint32_t qb[4];
        ldsm_x4_t(qb, qbase + T::swz(16 * c + 8 * ((lane >> 3) & 1) + (lane & 7), 2 * np + (lane >> 4)));
        mma16816(dkacc[2 * np], pa[c], qb[0], qb[1]);
        mma16816(dkacc[2 * np + 1], pa[c], qb[2], qb[3]);
      }
  }
  };
  if constexpr (G::MASK) {
    if (all_keys) run(std::false_type{});
    else run(std::true_type{});
  } else {
    run(std::false_type{});
  }

  // ---- epilogue: dk = sm_scale sum dS^T Q, dv, bf16
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    const long long tk = (long long)(kt * T2_BM + warp * 16 + g8 + 8 * hh);
    __nv_bfloat16* dkr = p.dk + z * p.ldk.z + row * p.ldk.row + tk * p.ldk.tok + h * p.ldk.head;
    __nv_bfloat16* dvr = p.dv + z * p.ldv.z + row * p.ldv.row + tk * p.ldv.tok + h * p.ldv.head;
#pragma unroll
    for (int nt = 0; nt < NTV; ++nt) {
      stg32(dkr + 8 * nt + 2 * q4, pack_bf16(dkacc[nt][2 * hh] * p.sm_scale, dkacc[nt][2 * hh + 1] * p.sm_scale));
      stg32(dvr + 8 * nt + 2 * q4, pack_bf16(dvacc[nt][2 * hh], dvacc[nt][2 * hh + 1]));
    }
  }
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------- the row term
// delta[z][h][row][query] = sum_d o[z, row, query, h, d] do[z, row, query, h, d] (fp32 sum of the bf16 products): the backward's row term when no back stage computes it (the projected-attention
// leaf).  One thread per (z h, row, query): its head-dim run (HD * 2 bytes) of o and of do through 16-byte loads.
struct DeltaParams {
  const __nv_bfloat16 *o, *dov;     // strided (Lay lo, ld)
  float* delta;                     // [z][H][L rows][L queries]
  Lay lo, ld;
  int L, H;
};

template <int HD>
__global__ void __launch_bounds__(128) attn2_delta_kernel(const DeltaParams p) {
  const int query = blockIdx.x * 128 + threadIdx.x, row = blockIdx.y;
  const int h = blockIdx.z % p.H, z = blockIdx.z / p.H;
  if (query >= p.L) return;
  const __nv_bfloat16* po = p.o + z * p.lo.z + row * p.lo.row + (long long)query * p.lo.tok + h * p.lo.head;
  const __nv_bfloat16* pd = p.dov + z * p.ld.z + row * p.ld.row + (long long)query * p.ld.tok + h * p.ld.head;
  float acc = 0.f;
#pragma unroll
  for (int g = 0; g < HD / 8; ++g) {
    const uint4 a = __ldg(reinterpret_cast<const uint4*>(po) + g), b = __ldg(reinterpret_cast<const uint4*>(pd) + g);
    const uint32_t av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) acc = fmaf(bf16lo(av[i]), bf16lo(bv[i]), fmaf(bf16hi(av[i]), bf16hi(bv[i]), acc));
  }
  p.delta[((long long)(z * p.H + h) * p.L + row) * p.L + query] = acc;
}

}  // namespace a100
