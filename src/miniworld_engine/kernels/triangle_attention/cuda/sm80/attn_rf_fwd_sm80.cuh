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
// Each output element is written by the same thread in either pass, so a recomputed CTA simply overwrites its tile.
#pragma once
#include "attn_fwd_sm80.cuh"

namespace a100 {

template <int WR_, int RW_, int QG_, int ST_, bool GATE_ = false, bool PIPE_ = false>
struct AttnRCfg {
  static constexpr int WR = WR_, RW = RW_, QG = QG_, ST = ST_;
  static constexpr bool GATE = GATE_, PIPE = PIPE_;   // PIPE: the software-pipelined hot loop (one slice deferred into the next slot)
  static constexpr int R = WR * RW, BM = 32 * QG, NW = WR * QG, NTHR = 32 * NW;
  static constexpr int KEYS = 32;                            // keys per ring slot
  static constexpr int ROWB = TA_D * 2;                      // 64-byte rows (32 bf16) everywhere: swz64
  static constexpr int BIAS_BYTES = BM * ROWB;               // [BM queries][32 keys] bf16
  static constexpr int KV_ROW = KEYS * ROWB;                 // 32 keys x 32 channels of one pair row
  static constexpr int KOFF = BIAS_BYTES, VOFF = KOFF + R * KV_ROW, SLOT = VOFF + R * KV_ROW;
  static constexpr int Q_BYTES = R * BM * ROWB;
  static constexpr int SMEM = ST * SLOT;
  static constexpr int KV_IT = R * KEYS * 4 / NTHR;          // 16-byte K (and V) chunks per thread and slot
  static constexpr int B_IT = BM * 4 / NTHR;                 // bias chunks per thread and slot
  static constexpr int Q_IT = R * BM * 4 / NTHR;
  static_assert(RW == 1 || RW == 2, "rows per warp");
  static_assert(NTHR % 128 == 0, "a K / V slot row is 128 chunks");
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
DEVI void mma_acc(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// One pass over the CTA tile.  SAFE = false: the unshifted softmax, returns whether the CTA must be recomputed; SAFE = true: running max.
template <class G, bool SAFE>
DEVI bool attn_rf_body(const AttnParams& p, float sinv_hi, float sinv_lo) {
  constexpr int RW = G::RW, ST = G::ST;
  constexpr int LA = (SAFE || !G::PIPE) ? ST - 1 : ST - 2;              // slots in flight ahead of the one consumed (the hot loop defers one slice: one slot less)
  static_assert(LA >= 1, "ring too short");
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sbase = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wr = warp / G::QG, qg = warp - wr * G::QG;
  const int L = p.L, qt = blockIdx.x, row0 = blockIdx.y * G::R;
  const int h = blockIdx.z % p.H, z = blockIdx.z / p.H;
  const long long tok0 = (long long)z * L * L;
  const int NU = L / G::KEYS;

  // ---- per-thread copy plan of a slot: K / V chunk i = pair row (tid >> 7) + i NTHR / 128, key (tid >> 2) & 31, granule tid & 3; bias chunk
  // i = query (tid >> 2) + i NTHR / 4, granule tid & 3 (the swizzle of a row is periodic in 8 rows, so chunk i's destination is chunk 0's + a constant)
  const int ck = (tid >> 2) & 31, cg = tid & 3, cr = tid >> 7;
  const __nv_bfloat16* ksrc = p.k + (tok0 + (long long)(row0 + cr) * L + ck) * p.ldk + h * TA_D + cg * 8;
  const __nv_bfloat16* vsrc = p.v + (tok0 + (long long)(row0 + cr) * L + ck) * p.ldv + h * TA_D + cg * 8;
  const __nv_bfloat16* bsrc = p.bias + ((long long)(z * p.H + h) * L + qt * G::BM + (tid >> 2)) * L + cg * 8;
  const uint32_t kdst = G::KOFF + cr * G::KV_ROW + swz64(ck, cg), bdst = swz64(tid >> 2, cg);
  const long long kstep = (long long)(G::NTHR / 128) * L * p.ldk, vstep = (long long)(G::NTHR / 128) * L * p.ldv;
  const long long kadv = (long long)G::KEYS * p.ldk, vadv = (long long)G::KEYS * p.ldv;
  auto issue = [&](int u) {             // called for u = 0, 1, 2, ... in order: the sources advance by one slot per call
    if (u < NU) {
      const uint32_t s = sbase + (u % ST) * G::SLOT;
#pragma unroll
      for (int i = 0; i < G::KV_IT; ++i) {
        cp_async16(s + kdst + i * (G::NTHR / 128) * G::KV_ROW, ksrc + i * kstep);
        cp_async16(s + kdst + (G::VOFF - G::KOFF) + i * (G::NTHR / 128) * G::KV_ROW, vsrc + i * vstep);
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
      const int c = tid + G::NTHR * i, r = c / (G::BM * 4), q = (c >> 2) % G::BM, g = c & 3;
      cp_async16(qs + r * G::BM * G::ROWB + swz64(q, g), p.q + (tok0 + (long long)(row0 + r) * L + qt * G::BM + q) * p.ldq + h * TA_D + g * 8);
    }
    cp_async_commit();
  }
#pragma unroll
  for (int u = 0; u < LA; ++u) issue(u);

  const int qw = qg * 32;                                     // the warp's first query in the tile
  const int lrow = (lane & 7) + 8 * ((lane >> 3) & 1);        // ldmatrix A-operand row of this lane in a 16-row tile
  uint32_t qa[RW][2][2][4];
  cp_async_wait<LA>();
  __syncthreads();
  {
    const uint32_t qs = sbase + (ST - 1) * G::SLOT;
#pragma unroll
    for (int w = 0; w < RW; ++w)
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int kk = 0; kk < 2; ++kk)
          ldsm_x4(qa[w][mt][kk], qs + (wr * RW + w) * G::BM * G::ROWB + swz64(qw + 16 * mt + lrow, 2 * kk + (lane >> 4)));
  }

  // B operands of the bias-scaling product: b0 = diag(s_hi) (keys 0-7 of the A slice), b1 = diag(s_lo) (keys 8-15); the lane holds rows
  // 2t, 2t+1 of column g.  And the ones column of the row sums.
  const int gq = lane >> 2, tq = lane & 3;
  const uint32_t bs0 = pack_bf16(2 * tq == gq ? sinv_hi : 0.f, 2 * tq + 1 == gq ? sinv_hi : 0.f);
  const uint32_t bs1 = pack_bf16(2 * tq == gq ? sinv_lo : 0.f, 2 * tq + 1 == gq ? sinv_lo : 0.f);
  constexpr uint32_t ONES = 0x3f803f80u;
  const float c1 = p.scl;

  float o[RW][2][4][4], l4[RW][2][4], m[RW][2][2];
#pragma unroll
  for (int w = 0; w < RW; ++w)
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        l4[w][mt][e] = 0.f;
#pragma unroll
        for (int dn = 0; dn < 4; ++dn) o[w][mt][dn][e] = 0.f;
      }
      m[w][mt][0] = m[w][mt][1] = -INFINITY;
    }

  const uint32_t a_off = swz64(qw + lrow, 0);                // bias A tile rows of mt 0 (mt 1: + 16 rows: same swizzle phase); lanes 16-31 repeat 0-15
  const uint32_t k_off = swz64(lane & 7, lane >> 3);          // K: keys 8jj + (lane & 7), granule lane >> 3 (all 32 channels of one 8-key column)
  const uint32_t v_off = swz64(8 * ((lane >> 3) & 1) + (lane & 7), lane >> 4);   // V^T: keys 16kl + .., granule 2np + (lane >> 4)

  // ---- the per-column blocks of one slot.
  // S of 8-key column jj for every row of the warp: bias / scale (one product per m tile, shared by the rows) + Q K^T
  auto s_column = [&](uint32_t sb, int jj, float (&sc)[RW][2][4][4]) {
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
      uint32_t kb[4];
      ldsm_x4(kb, sb + G::KOFF + (wr * RW + w) * G::KV_ROW + jj * 8 * G::ROWB + k_off);
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        mma_c(sc[w][mt][jj], qa[w][mt][0], kb[0], kb[1], cb[mt]);
        mma_acc(sc[w][mt][jj], qa[w][mt][1], kb[2], kb[3]);
      }
    }
  };
  // unshifted weights of column jj of row w
  auto exp_column = [&](float (&sc)[RW][2][4][4], int w, int jj) {
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int e = 0; e < 4; ++e) sc[w][mt][jj][e] = ex2f(sc[w][mt][jj][e] * c1);
  };
  // bf16 P of the 16-key slice kl of row w and its row sums (ones-column mma)
  auto pack_slice = [&](float (&sc)[RW][2][4][4], int w, int kl, uint32_t (&pa)[2][4]) {
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      pa[mt][0] = pack_bf16(sc[w][mt][2 * kl][0], sc[w][mt][2 * kl][1]);
      pa[mt][1] = pack_bf16(sc[w][mt][2 * kl][2], sc[w][mt][2 * kl][3]);
      pa[mt][2] = pack_bf16(sc[w][mt][2 * kl + 1][0], sc[w][mt][2 * kl + 1][1]);
      pa[mt][3] = pack_bf16(sc[w][mt][2 * kl + 1][2], sc[w][mt][2 * kl + 1][3]);
      mma_acc(l4[w][mt], pa[mt], ONES, ONES);
    }
  };
  // O += P V for the 16-key slice kl of row w (V of slot sb)
  auto pv_slice = [&](uint32_t sb, int w, int kl, const uint32_t (&pa)[2][4]) {
#pragma unroll
    for (int np = 0; np < 2; ++np) {
      uint32_t vb[4];
      ldsm_x4_t(vb, sb + G::VOFF + (wr * RW + w) * G::KV_ROW + kl * 16 * G::ROWB + (v_off ^ (np << 5)));
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        mma_acc(o[w][mt][2 * np], pa[mt], vb[0], vb[1]);
        mma_acc(o[w][mt][2 * np + 1], pa[mt], vb[2], vb[3]);
      }
    }
  };

  if constexpr (!SAFE && G::PIPE) {
    // Software-pipelined: the exponentials of column jj - 1 follow the tensor work of column jj, the products of slice 0 run under the
    // exponentials of columns 2 / 3, and the last slice's P V of slot u - 1 is deferred into slot u (behind its first S columns).
    uint32_t pd[RW][2][4];                                    // the deferred slice: packed P of keys 16-31 of the previous slot
    uint32_t sprev = 0;
    for (int u = 0; u < NU; ++u) {
      cp_async_wait<LA - 1>();
      __syncthreads();                                        // slot u landed; every warp finished slot u - 2 (its deferred slice ran in u - 1)
      issue(u + LA);
      const uint32_t sb = sbase + (u % ST) * G::SLOT;
      float sc[RW][2][4][4];
      s_column(sb, 0, sc);
      if (u > 0) pv_slice(sprev, 0, 1, pd[0]);
      s_column(sb, 1, sc);
#pragma unroll
      for (int w = 0; w < RW; ++w) exp_column(sc, w, 0);
      if (u > 0) {
#pragma unroll
        for (int w = 1; w < RW; ++w) pv_slice(sprev, w, 1, pd[w]);
      }
      s_column(sb, 2, sc);
#pragma unroll
      for (int w = 0; w < RW; ++w) exp_column(sc, w, 1);
#pragma unroll
      for (int w = 0; w < RW; ++w) {
        uint32_t pa[2][4];
        pack_slice(sc, w, 0, pa);
        pv_slice(sb, w, 0, pa);
      }
      s_column(sb, 3, sc);
#pragma unroll
      for (int w = 0; w < RW; ++w) exp_column(sc, w, 2);
#pragma unroll
      for (int w = 0; w < RW; ++w) exp_column(sc, w, 3);
#pragma unroll
      for (int w = 0; w < RW; ++w) pack_slice(sc, w, 1, pd[w]);
      sprev = sb;
    }
#pragma unroll
    for (int w = 0; w < RW; ++w) pv_slice(sprev, w, 1, pd[w]);
  } else {
    for (int u = 0; u < NU; ++u) {
      cp_async_wait<LA - 1>();
      __syncthreads();
      issue(u + LA);
      const uint32_t sb = sbase + (u % ST) * G::SLOT;
      float sc[RW][2][4][4];
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) s_column(sb, jj, sc);
#pragma unroll
      for (int w = 0; w < RW; ++w) {
        float mu[2][2];
        if constexpr (SAFE) {
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            float mx = -INFINITY;
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) mx = fmaxf(mx, fmaxf(sc[w][mt][jj][2 * hh], sc[w][mt][jj][2 * hh + 1]));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float mnew = fmaxf(m[w][mt][hh], mx * c1);
            mu[mt][hh] = mnew == -INFINITY ? 0.f : mnew;
            const float alpha = ex2f(m[w][mt][hh] - mu[mt][hh]);   // 0 from the initial -inf
            m[w][mt][hh] = mnew;
            l4[w][mt][2 * hh] *= alpha; l4[w][mt][2 * hh + 1] *= alpha;
#pragma unroll
            for (int dn = 0; dn < 4; ++dn) { o[w][mt][dn][2 * hh] *= alpha; o[w][mt][dn][2 * hh + 1] *= alpha; }
          }
        }
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int jj = 0; jj < 4; ++jj)
#pragma unroll
            for (int e = 0; e < 4; ++e) sc[w][mt][jj][e] = (SAFE ? ex2f(fmaf(sc[w][mt][jj][e], c1, -mu[mt][e >> 1])) : ex2f(sc[w][mt][jj][e] * c1));
#pragma unroll
        for (int kl = 0; kl < 2; ++kl) {
          uint32_t pa[2][4];
          pack_slice(sc, w, kl, pa);
          pv_slice(sb, w, kl, pa);
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
  auto tile_chunk = [&](int i, int& r, int& q, int& g) { const int c = tid + G::NTHR * i; r = c / (G::BM * 4); q = (c >> 2) % G::BM; g = c & 3; };
  if constexpr (G::GATE) {
#pragma unroll
    for (int i = 0; i < G::Q_IT; ++i) {
      int r, q, g;
      tile_chunk(i, r, q, g);
      cp_async16(sg + r * G::BM * G::ROWB + swz64(q, g), p.gate + (tok0 + (long long)(row0 + r) * L + qt * G::BM + q) * p.ldgate + h * TA_D + g * 8);
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
          for (int dn = 0; dn < 4; ++dn) bad |= !isfinite(o[w][mt][dn][2 * hh]) || !isfinite(o[w][mt][dn][2 * hh + 1]);
        }
        const float inv = lt > 0.f ? 1.f / lt : 0.f;
        const uint32_t row = so + (wr * RW + w) * G::BM * G::ROWB, qr = qw + 16 * mt + gq + 8 * hh;
#pragma unroll
        for (int dn = 0; dn < 4; ++dn) sts32(row + swz64(qr, dn) + 4 * tq, pack_bf16(o[w][mt][dn][2 * hh] * inv, o[w][mt][dn][2 * hh + 1] * inv));
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
    const uint32_t a = r * G::BM * G::ROWB + swz64(q, g);
    const uint4 ov = lds128(so + a);
    const long long tok = tok0 + (long long)(row0 + r) * L + qt * G::BM + q;
    if (!G::GATE || p.out != nullptr) stg128(p.out + tok * p.ldo + h * TA_D + g * 8, ov);
    if constexpr (G::GATE) {
      const uint4 gv = lds128(sg + a);
      const uint32_t* ow = &ov.x; const uint32_t* gw = &gv.x;
      uint32_t aw[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) aw[e] = pack_bf16(bf16lo(ow[e]) * sigmoid(bf16lo(gw[e])), bf16hi(ow[e]) * sigmoid(bf16hi(gw[e])));
      stg128(p.aout + tok * p.ldao + h * TA_D + g * 8, make_uint4(aw[0], aw[1], aw[2], aw[3]));
    }
  }
  return false;
}

template <class G>
__global__ void __launch_bounds__(G::NTHR, 2) attn_rf_fwd_kernel(const AttnParams p, float sinv_hi, float sinv_lo) {
  if (attn_rf_body<G, false>(p, sinv_hi, sinv_lo)) attn_rf_body<G, true>(p, sinv_hi, sinv_lo);
}

}  // namespace a100
