// attn_rf_fwd_sm80.cuh -- triangle-attention core, inference forward, A100 / sm_80, the register-resident schedule ("rf"): head dim 32, bf16
// operands, fp32 softmax and accumulation, bf16 out (optionally the sigmoid-gated operand of the output projection).  The same op and the same
// AttnParams as attn_fwd_sm80.cuh:
//
//   out[z, i, j, h, :] = softmax_k( scale * q[z, i, j, h, :] . k[z, i, k, h, :] + bias[z, h, j, k] ) @ v[z, i, k, h, :]
//
// What differs from attn_fwd_sm80.cuh's schedule (16 queries per warp, Q re-read from shared memory every sub-step, bias converted per element):
//   * one CTA = R = WR * RW pair rows x BM = 32 QG queries of one (z, head) as WR x QG warps; warp (wr, qg) owns queries [32 qg, 32 qg + 32) of
//     the RW consecutive pair rows [RW wr, RW wr + RW).  Two m16 tiles per warp: every K / V fragment loaded by ldmatrix feeds two mma, the Q
//     fragments of the warp's rows stay in registers for the whole key loop (RW x 2 x 2 x 4 registers), as do O and the row sums;
//   * the pair bias enters the score accumulator through the tensor core: S = bias / scale + Q K^T, the C operand being an exact bias-scaling
//     product A_bias B_sel (A_bias = the bf16 bias of 16 queries x 16 keys by ldmatrix, B_sel = diag(s_hi) over its first 8 keys and diag(s_lo)
//     over the second 8, s_hi + s_lo = 1 / scale in two bf16 words; one mma per 16 x 8 tile, shared by the RW rows of the warp) -- no per-element
//     conversion or multiply;
//   * p = 2^(S scale log2 e) without a shift (the fixed-shift "no shift" softmax of the tuned cores: an overflowed, non-finite or vanishing
//     (< 2^-80) row sum sends the whole CTA through the running-max instantiation of the same routine), the row sums from the same bf16 P by a
//     ones-column mma;
//   * K | V | bias stream in 32-key slots through an ST-stage cp.async ring (one __syncthreads per slot); Q rides in through the last slot.
// A flagged CTA stores nothing in the hot pass; the recomputation kernel writes its tile.
#pragma once
#include "attn_fwd_sm80.cuh"

namespace a100 {

template <int WR_, int RW_, int QG_, int ST_, bool GATE_ = false, int HD_ = 32, bool PIPE_ = false>
struct AttnRCfg {
  static constexpr int WR = WR_, RW = RW_, QG = QG_, ST = ST_, HD = HD_;
  static constexpr bool GATE = GATE_;
  static constexpr bool PIPE = PIPE_;                        // the hot loop software-pipelined across slots (one slot less in flight)
  static constexpr int R = WR * RW, BM = 32 * QG, NW = WR * QG, NTHR = 32 * NW;
  static constexpr int KEYS = 32;                            // keys per ring slot
  static constexpr int GR = HD / 8, KK = HD / 16, DN = HD / 8;   // 16-byte granules per Q / K / V / O row, k steps of Q K^T, n tiles of O
  static constexpr int ROWB = 64;                            // bias tile rows: 32 keys (swz64)
  static constexpr int RB = HD * 2;                          // Q / K / V / O tile rows (rf_swz)
  static constexpr int CPR = KEYS * GR;                      // 16-byte chunks of one pair row's K (or V) in a slot
  static constexpr int BIAS_BYTES = BM * ROWB;               // [BM queries][32 keys] bf16
  static constexpr int KV_ROW = KEYS * RB;                   // 32 keys x HD channels of one pair row
  static constexpr int KOFF = BIAS_BYTES, VOFF = KOFF + R * KV_ROW, SLOT = VOFF + R * KV_ROW;
  static constexpr int Q_BYTES = R * BM * RB;
  static constexpr int SMEM = ST * SLOT;
  static constexpr int KV_IT = R * CPR / NTHR;               // 16-byte K (and V) chunks per thread and slot
  static constexpr int B_IT = BM * 4 / NTHR;                 // bias chunks per thread and slot
  static constexpr int Q_IT = R * BM * GR / NTHR;
  static_assert(RW == 1 || RW == 2, "rows per warp");
  static_assert(HD == 16 || HD == 32, "head dim");
  static_assert(NTHR % CPR == 0 && (R * CPR) % NTHR == 0, "whole pair rows of K / V chunks per thread step");
  static_assert(Q_BYTES <= SLOT, "Q rides in through one ring slot");
  static_assert(2 * Q_BYTES <= ST * SLOT, "the epilogue stages O and g in the ring");
  static_assert(ST >= 2 && SMEM <= 163 * 1024, "ring");
};

// mma with a separate C operand (d = a b + c), and the in-place form; not volatile (pure register ops: the scheduler may move them)
DEVI void mma_c(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, const float (&c)[4]) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
// Q / K / V / O tile rows of HD channels: 64-byte rows (HD 32) swz64; 32-byte rows (HD 16): granule g of row r at g ^ ((r >> 2) & 1).  The 8 rows of one
// ldmatrix matrix cover all 32 banks either way; the phase is periodic in 8 rows.
template <int HD>
DEVI uint32_t rf_swz(uint32_t row, uint32_t g) {
  if constexpr (HD == 32) return swz64(row, g);
  else return row * 32u + ((g ^ ((row >> 2) & 1u)) << 4);
}

DEVI void mma_acc(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// One pass over the CTA tile.  SAFE = false: the unshifted softmax, returns whether the CTA must be recomputed; SAFE = true: running max.
template <class G, bool SAFE>
DEVI bool attn_rf_body(const AttnParams& p, float sinv_hi, float sinv_lo, int bx, int by, int bz) {
  constexpr int RW = G::RW, ST = G::ST, HD = G::HD, GR = G::GR, KK = G::KK, DN = G::DN;
  constexpr int LA = (G::PIPE && !SAFE) ? ST - 2 : ST - 1;   // slots in flight ahead of the one consumed
  static_assert(LA >= 1, "ring too short");
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sbase = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wr = warp / G::QG, qg = warp - wr * G::QG;
  const int L = p.L, qt = bx, row0 = by * G::R;
  const int h = bz % p.H, z = bz / p.H;
  const long long tok0 = (long long)z * L * L;
  const int NU = L / G::KEYS;

  // ---- per-thread copy plan of a slot: K / V chunk i = pair row tid / CPR + i NTHR / CPR, key (tid / GR) % 32, granule tid % GR; bias chunk i =
  // query (tid >> 2) + i NTHR / 4, granule tid & 3 (the swizzles are periodic in 8 rows, so chunk i's destination is chunk 0's + a constant)
  const int ck = (tid / GR) % 32, cg = tid % GR, cr = tid / G::CPR;
  constexpr int RSTEP = G::NTHR / G::CPR;                    // pair rows per chunk step
  const __nv_bfloat16* ksrc = p.k + (tok0 + (long long)(row0 + cr) * L + ck) * p.ldk + h * HD + cg * 8;
  const __nv_bfloat16* vsrc = p.v + (tok0 + (long long)(row0 + cr) * L + ck) * p.ldv + h * HD + cg * 8;
  const __nv_bfloat16* bsrc = p.bias + ((long long)(z * p.H + h) * L + qt * G::BM + (tid >> 2)) * L + (tid & 3) * 8;
  const uint32_t kdst = G::KOFF + cr * G::KV_ROW + rf_swz<HD>(ck, cg), bdst = swz64(tid >> 2, tid & 3);
  const long long kstep = (long long)RSTEP * L * p.ldk, vstep = (long long)RSTEP * L * p.ldv;
  const long long kadv = (long long)G::KEYS * p.ldk, vadv = (long long)G::KEYS * p.ldv;
  auto issue = [&](int u) {             // called for u = 0, 1, 2, ... in order: the sources advance by one slot per call
    if (u < NU) {
      const uint32_t s = sbase + (u % ST) * G::SLOT;
#pragma unroll
      for (int i = 0; i < G::KV_IT; ++i) {
        cp_async16(s + kdst + i * RSTEP * G::KV_ROW, ksrc + i * kstep);
        cp_async16(s + kdst + (G::VOFF - G::KOFF) + i * RSTEP * G::KV_ROW, vsrc + i * vstep);
      }
#pragma unroll
      for (int i = 0; i < G::B_IT; ++i) cp_async16(s + bdst + i * (G::NTHR / 4) * G::ROWB, bsrc + (long long)i * (G::NTHR / 4) * L);
      ksrc += kadv; vsrc += vadv; bsrc += G::KEYS;
    }
    cp_async_commit();
  };

  // ---- Q of the R rows through the last ring slot (group 0), then the first ST - 1 slots
  {
    const uint32_t qs = sbase + (ST - 1) * G::SLOT;
#pragma unroll
    for (int i = 0; i < G::Q_IT; ++i) {
      const int c = tid + G::NTHR * i, r = c / (G::BM * GR), q = (c / GR) % G::BM, g = c % GR;
      cp_async16(qs + r * G::BM * G::RB + rf_swz<HD>(q, g), p.q + (tok0 + (long long)(row0 + r) * L + qt * G::BM + q) * p.ldq + h * HD + g * 8);
    }
    cp_async_commit();
  }
#pragma unroll
  for (int u = 0; u < LA; ++u) issue(u);

  const int qw = qg * 32;                                     // the warp's first query in the tile
  const int lrow = (lane & 7) + 8 * ((lane >> 3) & 1);        // ldmatrix A-operand row of this lane in a 16-row tile
  uint32_t qa[RW][2][KK][4];
  cp_async_wait<LA>();
  __syncthreads();
  {
    const uint32_t qs = sbase + (ST - 1) * G::SLOT;
#pragma unroll
    for (int w = 0; w < RW; ++w)
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int kk = 0; kk < KK; ++kk)
          ldsm_x4(qa[w][mt][kk], qs + (wr * RW + w) * G::BM * G::RB + rf_swz<HD>(qw + 16 * mt + lrow, 2 * kk + (lane >> 4)));
  }

  // B operands of the bias-scaling product: b0 = diag(s_hi) (keys 0-7 of the A slice), b1 = diag(s_lo) (keys 8-15); the lane holds rows
  // 2t, 2t+1 of column g.  And the ones column of the row sums.
  const int gq = lane >> 2, tq = lane & 3;
  const uint32_t bs0 = pack_bf16(2 * tq == gq ? sinv_hi : 0.f, 2 * tq + 1 == gq ? sinv_hi : 0.f);
  const uint32_t bs1 = pack_bf16(2 * tq == gq ? sinv_lo : 0.f, 2 * tq + 1 == gq ? sinv_lo : 0.f);
  constexpr uint32_t ONES = 0x3f803f80u;
  const float c1 = p.scl;

  float o[RW][2][DN][4], l4[RW][2][4], m[RW][2][2];
#pragma unroll
  for (int w = 0; w < RW; ++w)
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        l4[w][mt][e] = 0.f;
#pragma unroll
        for (int dn = 0; dn < DN; ++dn) o[w][mt][dn][e] = 0.f;
      }
      m[w][mt][0] = m[w][mt][1] = -INFINITY;
    }

  const uint32_t a_off = swz64(qw + lrow, 0);                // bias A tile rows of mt 0 (mt 1: + 16 rows: same swizzle phase); lanes 16-31 repeat 0-15
  const uint32_t k_off = rf_swz<HD>(lane & 7, (lane >> 3) & (GR - 1));   // K: keys 8jj + (lane & 7), granule (lane >> 3): all HD channels of one 8-key column (HD 16: lanes 0-15)
  uint32_t v_off[GR / 2];                                    // V^T: keys 16kl + .., granule 2np + (lane >> 4)
#pragma unroll
  for (int np = 0; np < GR / 2; ++np) v_off[np] = rf_swz<HD>(8 * ((lane >> 3) & 1) + (lane & 7), 2 * np + (lane >> 4));

  // ---- one slot = two 16-key slices; per slice and row only S of 16 keys is live (RW x 2 x 2 x 4 registers).
  // S of 8-key column jj of the slot into sc[.][.][jj & 1], every row of the warp: bias / scale (one product per m tile, shared by the rows) + Q K^T
  auto s_column = [&](uint32_t sb, int jj, float (&sc)[RW][2][2][4]) {
    float cb[2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      // A = the bias of 16 queries x keys 8jj..8jj+7, twice (k 0-7 and 8-15: the lanes 16-31 address the same rows), so B = [diag(s_hi); diag(s_lo)]
      uint32_t a[4];
      ldsm_x4(a, sb + mt * 16 * G::ROWB + (a_off ^ (jj << 4)));
      const float zero[4] = {0.f, 0.f, 0.f, 0.f};
      mma_c(cb[mt], a, bs0, bs1, zero);
    }
#pragma unroll
    for (int w = 0; w < RW; ++w) {
      uint32_t kb[2 * KK];
      const uint32_t ka = sb + G::KOFF + (wr * RW + w) * G::KV_ROW + jj * 8 * G::RB + k_off;
      if constexpr (KK == 2) ldsm_x4(*reinterpret_cast<uint32_t(*)[4]>(kb), ka);
      else ldsm_x2(*reinterpret_cast<uint32_t(*)[2]>(kb), ka);
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        mma_c(sc[w][mt][jj & 1], qa[w][mt][0], kb[0], kb[1], cb[mt]);
#pragma unroll
        for (int kk = 1; kk < KK; ++kk) mma_acc(sc[w][mt][jj & 1], qa[w][mt][kk], kb[2 * kk], kb[2 * kk + 1]);
      }
    }
  };

  // weights of the slice in sc for row w (SAFE: after raising the running max, rescaling O and the sums), bf16 P, its row sums (ones-column mma)
  auto weights = [&](float (&sc)[RW][2][2][4], int w, uint32_t (&pa)[2][4]) {
    float mu[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
    if constexpr (SAFE) {
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          float mx = fmaxf(fmaxf(sc[w][mt][0][2 * hh], sc[w][mt][0][2 * hh + 1]), fmaxf(sc[w][mt][1][2 * hh], sc[w][mt][1][2 * hh + 1]));
          mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
          mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
          const float mnew = fmaxf(m[w][mt][hh], mx * c1);
          mu[mt][hh] = mnew == -INFINITY ? 0.f : mnew;
          const float alpha = ex2f(m[w][mt][hh] - mu[mt][hh]);   // 0 from the initial -inf
          m[w][mt][hh] = mnew;
          l4[w][mt][2 * hh] *= alpha; l4[w][mt][2 * hh + 1] *= alpha;
#pragma unroll
          for (int dn = 0; dn < DN; ++dn) { o[w][mt][dn][2 * hh] *= alpha; o[w][mt][dn][2 * hh + 1] *= alpha; }
        }
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      float pw[2][4];
#pragma unroll
      for (int c = 0; c < 2; ++c)
#pragma unroll
        for (int e = 0; e < 4; ++e) pw[c][e] = SAFE ? ex2f(fmaf(sc[w][mt][c][e], c1, -mu[mt][e >> 1])) : ex2f(sc[w][mt][c][e] * c1);
      pa[mt][0] = pack_bf16(pw[0][0], pw[0][1]);
      pa[mt][1] = pack_bf16(pw[0][2], pw[0][3]);
      pa[mt][2] = pack_bf16(pw[1][0], pw[1][1]);
      pa[mt][3] = pack_bf16(pw[1][2], pw[1][3]);
      mma_acc(l4[w][mt], pa[mt], ONES, ONES);
    }
  };
  // O += P V of the 16-key slice kl of slot sb, row w
  auto pv = [&](uint32_t sb, int kl, int w, const uint32_t (&pa)[2][4]) {
#pragma unroll
    for (int np = 0; np < GR / 2; ++np) {
      uint32_t vb[4];
      ldsm_x4_t(vb, sb + G::VOFF + (wr * RW + w) * G::KV_ROW + kl * 16 * G::RB + v_off[np]);
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        mma_acc(o[w][mt][2 * np], pa[mt], vb[0], vb[1]);
        mma_acc(o[w][mt][2 * np + 1], pa[mt], vb[2], vb[3]);
      }
    }
  };

  if constexpr (G::PIPE && !SAFE) {
    // Software-pipelined across slots: the P V of a slot's second slice runs in the next slot, next to its first S product and exponentials
    // (one slot less in flight: the deferred slice still reads slot u - 1 while u + ST - 2 lands).
    uint32_t pd[RW][2][4];
    uint32_t sprev = 0;
    for (int u = 0; u < NU; ++u) {
      cp_async_wait<LA - 1>();
      __syncthreads();                                        // slot u landed; every warp finished slot u - 2 (its deferred slice ran in u - 1)
      issue(u + LA);
      const uint32_t sb = sbase + (u % ST) * G::SLOT;
      float sc[RW][2][2][4];
      s_column(sb, 0, sc);
      s_column(sb, 1, sc);
      if (u > 0) {
#pragma unroll
        for (int w = 0; w < RW; ++w) pv(sprev, 1, w, pd[w]);
      }
#pragma unroll
      for (int w = 0; w < RW; ++w) {
        uint32_t pa[2][4];
        weights(sc, w, pa);
        pv(sb, 0, w, pa);
      }
      s_column(sb, 2, sc);
      s_column(sb, 3, sc);
#pragma unroll
      for (int w = 0; w < RW; ++w) weights(sc, w, pd[w]);
      sprev = sb;
    }
#pragma unroll
    for (int w = 0; w < RW; ++w) pv(sprev, 1, w, pd[w]);
  } else {
    for (int u = 0; u < NU; ++u) {
      cp_async_wait<LA - 1>();
      __syncthreads();                                        // slot u landed; every warp is past u - 1, so its slot is free for u + ST - 1
      issue(u + LA);
      const uint32_t sb = sbase + (u % ST) * G::SLOT;
#pragma unroll
      for (int kl = 0; kl < 2; ++kl) {
        float sc[RW][2][2][4];
        s_column(sb, 2 * kl, sc);
        s_column(sb, 2 * kl + 1, sc);
#pragma unroll
        for (int w = 0; w < RW; ++w) {
          uint32_t pa[2][4];
          weights(sc, w, pa);
          pv(sb, kl, w, pa);
        }
      }
    }
  }

  // ---- epilogue through shared memory (the ring is free): the bf16 O tile [R][BM][32] (and, gated, the g tile by cp.async) staged, then
  // 16-byte coalesced stores.  O / l, a zero sum -> zero output; gated: bf16(sigmoid(g) bf16(o)), the module's statement.  A CTA the hot
  // pass flags for recomputation stores nothing.
  cp_async_wait<0>();
  __syncthreads();
  const uint32_t so = sbase, sg = sbase + G::Q_BYTES;
  auto tile_chunk = [&](int i, int& r, int& q, int& g) { const int c = tid + G::NTHR * i; r = c / (G::BM * GR); q = (c / GR) % G::BM; g = c % GR; };
  if constexpr (G::GATE) {
#pragma unroll
    for (int i = 0; i < G::Q_IT; ++i) {
      int r, q, g;
      tile_chunk(i, r, q, g);
      cp_async16(sg + r * G::BM * G::RB + rf_swz<HD>(q, g), p.gate + (tok0 + (long long)(row0 + r) * L + qt * G::BM + q) * p.ldgate + h * HD + g * 8);
    }
    cp_async_commit();
  }
  bool bad = false;
#pragma unroll
  for (int w = 0; w < RW; ++w)
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const float lt = l4[w][mt][2 * hh];
        if constexpr (!SAFE) {
          bad |= !(lt >= 0x1p-80f && lt <= 3.0e38f);
#pragma unroll
          for (int dn = 0; dn < DN; ++dn) bad |= !isfinite(o[w][mt][dn][2 * hh]) || !isfinite(o[w][mt][dn][2 * hh + 1]);
        }
        const float inv = lt > 0.f ? 1.f / lt : 0.f;
        const uint32_t row = so + (wr * RW + w) * G::BM * G::RB, qr = qw + 16 * mt + gq + 8 * hh;
#pragma unroll
        for (int dn = 0; dn < DN; ++dn) sts32(row + rf_swz<HD>(qr, dn) + 4 * tq, pack_bf16(o[w][mt][dn][2 * hh] * inv, o[w][mt][dn][2 * hh + 1] * inv));
      }
  if constexpr (G::GATE) cp_async_wait<0>();
  if constexpr (!SAFE) {
    if (__syncthreads_or(bad)) return true;
  } else {
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < G::Q_IT; ++i) {
    int r, q, g;
    tile_chunk(i, r, q, g);
    const uint32_t a = r * G::BM * G::RB + rf_swz<HD>(q, g);
    const uint4 ov = lds128(so + a);
    const long long tok = tok0 + (long long)(row0 + r) * L + qt * G::BM + q;
    if (!G::GATE || p.out != nullptr) stg128(p.out + tok * p.ldo + h * HD + g * 8, ov);
    if constexpr (G::GATE) {
      const uint4 gv = lds128(sg + a);
      const uint32_t* ow = &ov.x; const uint32_t* gw = &gv.x;
      uint32_t aw[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) aw[e] = pack_bf16(bf16lo(ow[e]) * sigmoid(bf16lo(gw[e])), bf16hi(ow[e]) * sigmoid(bf16hi(gw[e])));
      stg128(p.aout + tok * p.ldao + h * HD + g * 8, make_uint4(aw[0], aw[1], aw[2], aw[3]));
    }
  }
  return false;
}

// The hot pass over every CTA tile.  A tile it cannot finish is appended to the fix list (fix[0] = count, cleared by the launcher; fix[1 + i] = the
// tile's linear index) and recomputed by attn_rf_fix_kernel -- a separate kernel, so the hot kernel's register allocation is its own.  The list
// order may vary; the tiles are disjoint and each is recomputed whole, so the output does not.
template <class G>
__global__ void __launch_bounds__(G::NTHR, 2) attn_rf_fwd_kernel(const AttnParams p, float sinv_hi, float sinv_lo, int* fix) {
  if (attn_rf_body<G, false>(p, sinv_hi, sinv_lo, blockIdx.x, blockIdx.y, blockIdx.z) && threadIdx.x == 0)
    fix[1 + atomicAdd(fix, 1)] = (int)((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x);
}

// The running-max recomputation of the listed tiles: a few persistent CTAs walk the list (none listed: they leave at once).
template <class G>
__global__ void __launch_bounds__(G::NTHR, 1) attn_rf_fix_kernel(const AttnParams p, float sinv_hi, float sinv_lo, const int* fix, int nx, int ny) {
  const int n = fix[0];
  for (int i = blockIdx.x; i < n; i += gridDim.x) {
    const int t = fix[1 + i];
    attn_rf_body<G, true>(p, sinv_hi, sinv_lo, t % nx, (t / nx) % ny, t / (nx * ny));
    __syncthreads();                                          // the next tile's copies reuse the shared memory
  }
}


// Host launch (torch-free): the hot pass over the grid (L / BM, L / R, z H), then the recomputation of the listed tiles.  fix: int32 [1 + tiles]
// (rf_fix_ints) device scratch; its count is cleared here.  scl = sm_scale log2 e; the bias-scaling operand is 1 / sm_scale = log2 e / scl in two
// bf16 words.
template <class G>
inline long long rf_fix_ints(int L, int zh) { return 1 + (long long)(L / G::BM) * (L / G::R) * zh; }

template <class G>
inline cudaError_t attn_rf_launch(const AttnParams& p, int z, double scl, int* fix, cudaStream_t stream, int nsm) {
  static bool set[64] = {};
  int dev = 0;
  cudaError_t e = cudaGetDevice(&dev);
  if (e != cudaSuccess) return e;
  if (!set[dev & 63]) {
    if ((e = cudaFuncSetAttribute(attn_rf_fwd_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM)) != cudaSuccess) return e;
    if ((e = cudaFuncSetAttribute(attn_rf_fix_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM)) != cudaSuccess) return e;
    set[dev & 63] = true;
  }
  const double s = 1.4426950408889634 / scl;
  const float hi = __bfloat162float(__float2bfloat16_rn((float)s)), lo = __bfloat162float(__float2bfloat16_rn((float)(s - hi)));
  const dim3 grid(p.L / G::BM, p.L / G::R, z * p.H);
  const long long ntile = (long long)grid.x * grid.y * grid.z;
  if ((e = cudaMemsetAsync(fix, 0, sizeof(int), stream)) != cudaSuccess) return e;
  attn_rf_fwd_kernel<G><<<grid, G::NTHR, G::SMEM, stream>>>(p, hi, lo, fix);
  if ((e = cudaGetLastError()) != cudaSuccess) return e;
  attn_rf_fix_kernel<G><<<(unsigned)(ntile < nsm ? ntile : nsm), G::NTHR, G::SMEM, stream>>>(p, hi, lo, fix, (int)grid.x, (int)grid.y);
  return cudaGetLastError();
}

}  // namespace a100
