// attn2_fwd_sm80.cuh -- the triangle-attention core, forward, A100 / sm_80, generalised from attn_fwd_sm80.cuh: head dim HD = 16 or 32 (a native 16-channel head: half the tile bytes, one k step
// of the score product, two n tiles of the output), element strides for every operand (token-major packed buffers AND head-major [A, B, H, L, D] tensors read in place), and an optional
// per-pair-row key mask (the augmented-attention contract: mask[z][row][key]).  The arithmetic and the schedule are attn_fwd_sm80.cuh's (see its header): one CTA task =
// (z, head h, 128-query tile) x R pair rows, warp w owns the 16 queries [16 w, 16 w + 16) of every row, K | V stream through an NKV-stage cp.async ring, the bias tile through two
// buffers, fp32 online softmax in base 2, mma.sync for both products.
#pragma once
#include "sm80_common.cuh"

namespace a100 {

constexpr int T2_BM = 128;
constexpr float T2_L2E = 1.4426950408889634f;

// element strides of token j, pair row a, batch z and head h of a [z, a, j, h, d] tensor (the channel stride is 1)
struct Lay {
  long long tok, row, z, head;
};

template <int HD>
struct Hd {
  static_assert(HD == 16 || HD == 32, "head dim");
  static constexpr int GR = HD / 8;                           // 16-byte granules per row of a tile
  static constexpr int ROWB = HD * 2;                         // bytes per row
  static constexpr int KS = HD / 16;                          // 16-wide k steps of the score product
  static constexpr int NTV = HD / 8;                          // n tiles (8 channels) of the output
  // 64 B rows: 16 B granule g of row r at g ^ ((r >> 1) & 3);  32 B rows: g ^ ((r >> 2) & 1): the 8 rows of one ldmatrix matrix cover all 32 banks either way
  static DEVI uint32_t swz(uint32_t row, uint32_t g) {
    if constexpr (HD == 32) return row * 64u + ((g ^ ((row >> 1) & 3u)) << 4);
    else return row * 32u + ((g ^ ((row >> 2) & 1u)) << 4);
  }
};

struct Attn2Params {
  const __nv_bfloat16 *q, *k, *v;   // strided (Lay q, k, v)
  const __nv_bfloat16* bias;        // [z][H][L][L] (query j, key k), bf16, masked keys = bf16 min (the shared key mask folded in)
  const uint8_t* rowmask;           // [z][L rows][L keys] (non-zero = real key) or nullptr: the per-pair-row key mask
  __nv_bfloat16* out;               // strided (Lay o)
  float* lse;                       // [z][H][L][L] (row i, query j): m + log2 l in the scaled base-2 domain, or nullptr
  Lay lq, lk, lv, lo;
  int L, H;
  float scl;                        // log2 e / sqrt(HD)
};

// The per-pair-row key masks as bit words: word w of a row = the mask bytes of its keys [32 w, 32 w + 32) as bits (bit c set = key 32 w + c is real).  A CTA builds the words of its R rows once
// (one warp ballot per word, ahead of the main loop) and reads one word per (row, key tile) with one shared load; a tile whose word is all ones (the usual case: padding is the exception)
// skips the masking arithmetic altogether -- a branch every thread of the CTA takes alike.  L <= 32 MASK_WORDS = 8192.
constexpr int MASK_WORDS = 256;
constexpr uint32_t ALL_KEYS = 0xffffffffu;

template <int R>
DEVI void build_mask_words(uint32_t mw, const uint8_t* rowmask, long long first_row, int L, int tid) {
  const int warp = tid >> 5, lane = tid & 31, nw = L / 32;
  constexpr int UNR = 4;                                     // groups of 8 words (one per warp) in flight: 32 words of a row per pass, i.e. L <= 1024 in one pass
  for (int w0 = 0; w0 < nw; w0 += 8 * UNR) {
    uint32_t v[R][UNR];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int u = 0; u < UNR; ++u) {
        const int w = w0 + 8 * u + warp;
        v[r][u] = w < nw ? (uint32_t)__ldg(rowmask + (first_row + r) * L + w * 32 + lane) : 0u;
      }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int u = 0; u < UNR; ++u) {
        const int w = w0 + 8 * u + warp;
        const uint32_t bits = __ballot_sync(0xffffffffu, v[r][u] != 0u);
        if (lane == 0 && w < nw) sts32(mw + (r * MASK_WORDS + w) * 4, bits);
      }
  }
}

template <int HD_, int R_, int NKV_, int BN_, int MINB_, bool MASK_>
struct Attn2Cfg {
  using T = Hd<HD_>;
  static constexpr int HD = HD_, R = R_, NKV = NKV_, BN = BN_, MINB = MINB_, NTHR = 256;
  static constexpr bool MASK = MASK_;
  static_assert(BN == 64 || BN == 32, "BN");
  static_assert(!MASK_ || BN == 32, "the masked variants read one mask word per key tile");
  static constexpr int NT = BN / 8;                          // S n-tiles (8 keys each)
  static constexpr int KB = BN / 16;                         // 16-key blocks: the PV k-steps
  static constexpr int Q_BYTES = T2_BM * HD * 2;             // one row's Q tile
  static constexpr int KV_BYTES = 2 * BN * HD * 2;           // K | V of one sub-step
  static constexpr int B_BYTES = T2_BM * BN * 2;             // bias tile
  static constexpr int OFF_Q = 0, OFF_KV = OFF_Q + R * Q_BYTES, OFF_B = OFF_KV + NKV * KV_BYTES, OFF_M = OFF_B + 2 * B_BYTES;
  static constexpr int SMEM = OFF_M + (MASK ? R * MASK_WORDS * 4 : 0);
};

// the bias tile's rows are BN * 2 bytes: 128 B (BN = 64, the common 8-row XOR swizzle) or 64 B (BN = 32)
template <int BN>
DEVI uint32_t bias_off2(uint32_t row, uint32_t g) {
  if constexpr (BN == 64) return swz<128>(row, g * 16);
  else return row * 64u + ((g ^ ((row >> 1) & 3u)) << 4);
}

DEVI float ex2f2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }

// The cp.async work of one pipeline stage (forward and the query side of the backward): the K | V tile of sub-step s = (key tile j = s / R, pair row r = s % R) and, for the first row
// of a key tile, its bias tile.  Per-thread offsets are made once in the constructor.  G: Attn2Cfg or Dq2Cfg.
template <class G>
struct KV2Loader {
  static constexpr int BN = G::BN, R = G::R, NKV = G::NKV, BI = BN / 16, GR = Hd<G::HD>::GR;
  static constexpr int KVG = 2 * BN * GR;                                            // 16-byte granules of one K | V stage
  static constexpr int KVN = (KVG + 255) / 256;
  uint32_t kv_dst[KVN], b_dst[BI];
  const __nv_bfloat16 *kv_src[KVN], *b_src[BI];
  long long kv_tok[KVN], kv_row[KVN];

  bool kv_ok[KVN];

  template <class P>
  DEVI KV2Loader(const P& p, int z, int row0, int h, int qt, int tid) {
#pragma unroll
    for (int i = 0; i < KVN; ++i) {                          // K | V: 2 x BN tokens x GR granules of 16 B
      const int c = tid + 256 * i;
      kv_ok[i] = c < KVG;
      const int which = c / (BN * GR), w = c - which * BN * GR, token = w / GR, g = w % GR;
      kv_dst[i] = which * (G::KV_BYTES / 2) + Hd<G::HD>::swz(token, g);
      const Lay& l = which ? p.lv : p.lk;
      kv_tok[i] = l.tok;
      kv_row[i] = l.row;
      kv_src[i] = (which ? p.v : p.k) + z * l.z + row0 * l.row + token * l.tok + h * l.head + g * 8;
    }
#pragma unroll
    for (int i = 0; i < BI; ++i) {                           // bias tile: 128 rows x BN / 8 granules
      const int c = tid + 256 * i, row = c / (BN / 8), gg = c % (BN / 8);
      b_dst[i] = bias_off2<BN>(row, gg);
      b_src[i] = p.bias + ((long long)(z * p.H + h) * p.L + qt * T2_BM + row) * p.L + gg * 8;
    }
  }

  DEVI void issue(uint32_t sbase, int s, int NS) const {
    if (s < NS) {
      const int j = s / R, r = s - j * R, st = s % NKV;
#pragma unroll
      for (int i = 0; i < KVN; ++i)
        if (kv_ok[i]) cp_async16(sbase + G::OFF_KV + st * G::KV_BYTES + kv_dst[i], kv_src[i] + (long long)r * kv_row[i] + (long long)j * BN * kv_tok[i]);
      if (r == 0) {
#pragma unroll
        for (int i = 0; i < BI; ++i) cp_async16(sbase + G::OFF_B + (j & 1) * G::B_BYTES + b_dst[i], b_src[i] + j * BN);
      }
    }
    cp_async_commit();
  }
};

// One sub-step of the forward -- key tile (K | V at kbase) of the pair row whose Q tile is at qaddr: S = Q K^T, the online softmax and O += P V for the warp's 16 queries.  MK: the tile has a
// masked key (its bit word bm is not all ones).  The two variants are two separate straight-line bodies (the caller branches once per sub-step, so the unmasked one is the code of a kernel
// without a mask and the masked one pays only its own selects).
template <class G, bool MK>
DEVI void fwd_step(const Attn2Params& p, uint32_t qaddr, uint32_t kbase, uint32_t bm, int lane, int warp, const float (&b2)[G::NT][4], float (&acc)[Hd<G::HD>::NTV][4], float (&m)[2],
                   float (&l)[2]) {
  constexpr int HD = G::HD, NT = G::NT, KB = G::KB, KS = Hd<HD>::KS, NTV = Hd<HD>::NTV;
  using T = Hd<HD>;
  const int qrow = warp * 16 + (lane & 7) + 8 * ((lane >> 3) & 1);
  // ---- S = Q K^T
  uint32_t qa[KS][4];
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) ldsm_x4(qa[ks], qaddr + T::swz(qrow, 2 * ks + (lane >> 4)));
  float sacc[NT][4];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) sacc[nt][0] = sacc[nt][1] = sacc[nt][2] = sacc[nt][3] = 0.f;
#pragma unroll
  for (int pp = 0; pp < KB; ++pp)
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      uint32_t kb[4];
      ldsm_x4(kb, kbase + T::swz(16 * pp + 8 * ((lane >> 4) & 1) + (lane & 7), 2 * ks + ((lane >> 3) & 1)));
      mma16816(sacc[2 * pp], qa[ks], kb[0], kb[1]);
      mma16816(sacc[2 * pp + 1], qa[ks], kb[2], kb[3]);
    }
  // ---- online softmax (base 2)
  float mx[2] = {-INFINITY, -INFINITY};
  [[maybe_unused]] const uint32_t sh = bm >> (2 * (lane & 3));      // a masked key has logit -inf: columns 8 nt + 2 q4 and + 1 of this thread are bits 8 nt and 8 nt + 1 of bm >> 2 q4
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    sacc[nt][0] = fmaf(sacc[nt][0], p.scl, b2[nt][0]);
    sacc[nt][1] = fmaf(sacc[nt][1], p.scl, b2[nt][1]);
    sacc[nt][2] = fmaf(sacc[nt][2], p.scl, b2[nt][2]);
    sacc[nt][3] = fmaf(sacc[nt][3], p.scl, b2[nt][3]);
    if constexpr (MK) {
      const bool k0 = ((sh >> (8 * nt)) & 1u) != 0, k1 = ((sh >> (8 * nt + 1)) & 1u) != 0;
      sacc[nt][0] = k0 ? sacc[nt][0] : -INFINITY;
      sacc[nt][2] = k0 ? sacc[nt][2] : -INFINITY;
      sacc[nt][1] = k1 ? sacc[nt][1] : -INFINITY;
      sacc[nt][3] = k1 ? sacc[nt][3] : -INFINITY;
    }
    mx[0] = fmaxf(mx[0], fmaxf(sacc[nt][0], sacc[nt][1]));
    mx[1] = fmaxf(mx[1], fmaxf(sacc[nt][2], sacc[nt][3]));
  }
  float mu[2], alpha[2];
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffffu, mx[hh], 1));
    mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffffu, mx[hh], 2));
    const float mnew = fmaxf(m[hh], mx[hh]);
    mu[hh] = mnew == -INFINITY ? 0.f : mnew;         // a fully masked key tile: subtract 0, every weight is 2^-inf = 0
    alpha[hh] = ex2f2(m[hh] - mu[hh]);
    m[hh] = mnew;
  }
  uint32_t pa[KB][4];
  float rs[2] = {0.f, 0.f};
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const float p0 = ex2f2(sacc[nt][0] - mu[0]), p1 = ex2f2(sacc[nt][1] - mu[0]);
    const float p2 = ex2f2(sacc[nt][2] - mu[1]), p3 = ex2f2(sacc[nt][3] - mu[1]);
    rs[0] += p0 + p1;
    rs[1] += p2 + p3;
    pa[nt >> 1][(nt & 1) * 2] = pack_bf16(p0, p1);
    pa[nt >> 1][(nt & 1) * 2 + 1] = pack_bf16(p2, p3);
  }
  l[0] = fmaf(l[0], alpha[0], rs[0]);
  l[1] = fmaf(l[1], alpha[1], rs[1]);
#pragma unroll
  for (int nt = 0; nt < NTV; ++nt) {
    acc[nt][0] *= alpha[0]; acc[nt][1] *= alpha[0];
    acc[nt][2] *= alpha[1]; acc[nt][3] *= alpha[1];
  }
  // ---- O += P V
  const uint32_t vbase = kbase + G::KV_BYTES / 2;
#pragma unroll
  for (int c = 0; c < KB; ++c)
#pragma unroll
    for (int np = 0; np < NTV / 2; ++np) {
      uint32_t vb[4];
      ldsm_x4_t(vb, vbase + T::swz(16 * c + 8 * ((lane >> 3) & 1) + (lane & 7), 2 * np + (lane >> 4)));
      mma16816(acc[2 * np], pa[c], vb[0], vb[1]);
      mma16816(acc[2 * np + 1], pa[c], vb[2], vb[3]);
    }
}

template <class G>
__global__ void __launch_bounds__(256, G::MINB) attn2_fwd_kernel(const Attn2Params p) {
  constexpr int HD = G::HD, R = G::R, NKV = G::NKV, BN = G::BN, NT = G::NT, NTV = Hd<HD>::NTV, GR = Hd<HD>::GR;
  using T = Hd<HD>;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sbase = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int L = p.L, NJ = L / BN, NS = R * NJ;
  const int qt = blockIdx.x, row0 = blockIdx.y * R;
  const int h = blockIdx.z % p.H, z = blockIdx.z / p.H;

  const KV2Loader<G> loader(p, z, row0, h, qt, tid);
  auto issue = [&](int s) { loader.issue(sbase, s, NS); };
  // Q of the R rows rides in group 0
#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int i = 0; i < (T2_BM * GR + 255) / 256; ++i) {
      const int c = tid + 256 * i, token = c / GR, g = c % GR;
      if (c < T2_BM * GR)
        cp_async16(sbase + G::OFF_Q + r * G::Q_BYTES + T::swz(token, g), p.q + z * p.lq.z + (row0 + r) * p.lq.row + (qt * T2_BM + token) * p.lq.tok + h * p.lq.head + g * 8);
    }
  }
#pragma unroll
  for (int s = 0; s < NKV - 1; ++s) issue(s);
  if constexpr (G::MASK) build_mask_words<R>(sbase + G::OFF_M, p.rowmask, (long long)z * L + row0, L, tid);     // while the first stages are in flight; published by the first barrier of the main loop

  float acc[R][NTV][4], m[R][2], l[R][2], b2[NT][4];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    m[r][0] = m[r][1] = -INFINITY;
    l[r][0] = l[r][1] = 0.f;
#pragma unroll
    for (int i = 0; i < NTV; ++i) acc[r][i][0] = acc[r][i][1] = acc[r][i][2] = acc[r][i][3] = 0.f;
  }

  uint32_t bm_next = ALL_KEYS;
  for (int j = 0; j < NJ; ++j) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int s = j * R + r, st = s % NKV;
      cp_async_wait<NKV - 2>();
      __syncthreads();
      issue(s + NKV - 1);
      if (r == 0) {                                      // this key tile's bias fragments: bias * log2 e in the S accumulator layout
        const uint32_t bb = sbase + G::OFF_B + (j & 1) * G::B_BYTES;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const uint32_t u = lds32(bb + bias_off2<BN>(warp * 16 + (lane >> 2) + 8 * hh, nt) + (lane & 3) * 4);
            b2[nt][2 * hh] = bf16lo(u) * T2_L2E;
            b2[nt][2 * hh + 1] = bf16hi(u) * T2_L2E;
          }
      }
      uint32_t bm = ALL_KEYS;                            // the key mask of this (row, key tile): one bit per key
      if constexpr (G::MASK) {
        if (r == 0 && j == 0) bm_next = lds32(sbase + G::OFF_M);       // the first word, once the barrier has published the words
        bm = bm_next;
        bm_next = lds32(sbase + G::OFF_M + (((r + 1 < R) ? r + 1 : 0) * MASK_WORDS + ((r + 1 < R) ? j : (j + 1 < NJ ? j + 1 : j))) * 4);    // the next sub-step's word, its latency hidden behind this one's products
      }
      const uint32_t qaddr = sbase + G::OFF_Q + r * G::Q_BYTES, kbase = sbase + G::OFF_KV + st * G::KV_BYTES;
      if constexpr (G::MASK) {
        if (bm != ALL_KEYS) fwd_step<G, true>(p, qaddr, kbase, bm, lane, warp, b2, acc[r], m[r], l[r]);       // a tile with a masked key
        else fwd_step<G, false>(p, qaddr, kbase, bm, lane, warp, b2, acc[r], m[r], l[r]);
      } else {
        fwd_step<G, false>(p, qaddr, kbase, bm, lane, warp, b2, acc[r], m[r], l[r]);
      }
    }
  }

  // ---- epilogue: O / l (a zero sum -> zero output), bf16 stores, optionally the log-sum-exp
#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      float lt = l[r][hh];
      lt += __shfl_xor_sync(0xffffffffu, lt, 1);
      lt += __shfl_xor_sync(0xffffffffu, lt, 2);
      const float inv = lt > 0.f ? 1.f / lt : 1.f;
      const int query = qt * T2_BM + warp * 16 + (lane >> 2) + 8 * hh;
      __nv_bfloat16* orow = p.out + z * p.lo.z + (row0 + r) * p.lo.row + (long long)query * p.lo.tok + h * p.lo.head;
#pragma unroll
      for (int nt = 0; nt < NTV; ++nt) stg32(orow + 8 * nt + 2 * (lane & 3), pack_bf16(acc[r][nt][2 * hh] * inv, acc[r][nt][2 * hh + 1] * inv));
      if (p.lse != nullptr && (lane & 3) == 0) {
        const float mm = m[r][hh] == -INFINITY ? -1e38f : m[r][hh];
        p.lse[((long long)(z * p.H + h) * L + (row0 + r)) * L + query] = mm + log2f(lt > 0.f ? lt : 1.f);
      }
    }
  }
}

}  // namespace a100
