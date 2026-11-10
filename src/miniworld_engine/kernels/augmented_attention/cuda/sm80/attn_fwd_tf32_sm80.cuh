// attn_fwd_tf32_sm80.cuh -- attention with a shared pair bias, forward, A100 / sm_80, FP32 operands on the TF32 tensor cores (mma.sync m16n8k8, fp32 accumulation): the twin of
// attn_fwd_sm80.cuh for fp32 callers (head dim HD a multiple of 8: 32 or 48).
//
//   o[a, i, h, :] = softmax_k( scl q[a, i, h, :] . k[a, k, h, :] + bscl bias[h, i, k] ) @ v[a, k, h, :]        (base 2: scl = sm_scale log2 e, bscl = log2 e)
//
// Structure of the bf16 kernel (one CTA = (head h, 128-query tile) of one sample, 8 warps x 16 queries, K | V through an NKV-stage cp.async ring, the bias tile of the key tile through
// a two-deep ring, online softmax in registers) with fp32 tiles.  What changes with TF32:
//   * fragments.  A (16 x 8, row): a0 = A[g][t], a1 = A[g + 8][t], a2 = A[g][t + 4], a3 = A[g + 8][t + 4] (g = lane / 4, t = lane % 4); B (8 x 8, col): b0 = B[t][g], b1 = B[t + 4][g];
//     C: c0 = C[g][2 t], c1 = C[g][2 t + 1], c2 = C[g + 8][2 t], c3 = C[g + 8][2 t + 1].  ldmatrix moves 16-bit units, but an fp32 element is exactly the pair a thread gets of a row, so the
//     non-transposed ldmatrix.x4 loads Q's A fragments (and K's B fragments: rows = keys) from an fp32 tile unchanged.  Tile rows are PITCH = HD + 4 floats: an odd number of 16-byte
//     granules (ldmatrix conflict-free) that also makes the scalar V reads conflict-free (see below).
//   * P V.  The S accumulator is already P's A fragment if the logical key index k of the mma is permuted: slot t <-> key 2 t, slot t + 4 <-> key 2 t + 1 of the 8-key tile (a0 = P[g][2 t]
//     = c0, a1 = P[g + 8][2 t] = c2, a2 = P[g][2 t + 1] = c1, a3 = P[g + 8][2 t + 1] = c3).  V's B fragment is then b0 = V[2 t][n], b1 = V[2 t + 1][n]: two scalar shared loads per
//     fragment (an 8 x 8 transpose by ldmatrix does not exist for 32-bit elements); with PITCH = HD + 4 the 32 lanes of such a load fall into 32 different banks.
//   * the bias tile is fp32 [128][BN = 32] (16 KB), swizzled so the float2 fragment reads of a half-warp (4 rows x 32 bytes) cover all 32 banks.
// Operands go to the tensor core as they are (the hardware drops the low 13 mantissa bits: round toward zero); P likewise.  Masked keys: -inf in the bias, or per-sample penalties (KM,
// as attn_fwd_sm80.cuh).  A row without a valid key gets a zero output.
// Epilogues: GATE: sigmoid(g) o written to ``out`` (may alias q: a CTA reads its q tile before any other thread writes it); else o and the log-sum-exp lse[a][h][i] = m + log2 l.
#pragma once
#include "attn_fwd_sm80.cuh"

namespace aa80 {

DEVI void mma1688_tf32(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// [rows][HD] fp32 tile in shared memory, rows of HD + 4 floats (an odd number of 16-byte granules)
template <int HD>
struct Tile32 {
  static constexpr int HG = HD / 4;                        // granules of 4 floats in a row of data
  static constexpr int PITCH = (HD + 4) * 4;               // bytes
  static DEVI uint32_t off(uint32_t row, uint32_t g) { return row * PITCH + g * 16; }
};

// the fp32 bias tile [128 rows][32 keys]: 16-byte chunk c of row r at c ^ ((r & 3) << 1)
DEVI uint32_t swzb32(uint32_t row, uint32_t chunk) { return row * 128u + ((chunk ^ ((row & 3u) << 1)) << 4); }

struct Fwd32Params {
  const float *q, *k, *v;           // token-major [A * L][ld] fp32, head h = columns [HD h, HD h + HD)
  const float* bias;                // [H][L][L] fp32 (query i, key k); masked keys -inf
  const float* gate;                // GATE: g, token-major like q
  float* out;                       // [A * L][ldo] fp32: o, or sigmoid(g) o (GATE: may be q itself)
  float* lse;                       // [A][H][L] (null in GATE builds)
  const float* kpen;                // KM: per-sample key penalties [A][L] (0 / -inf), sample stride kps floats
  long long ldq, ldk, ldv, ldg, ldo, ss, kps;
  int L, H;
  float scl, bscl;
};

template <int HD_, int NKV_, int MINB_, int GATE_, int KM_>
struct Fwd32Cfg {
  static constexpr int HD = HD_, R = 1, NKV = NKV_, MINB = MINB_, GATE = GATE_, KM = KM_, NTHR = 256, BN = 32;
  static_assert(HD % 8 == 0, "HD");
  static constexpr int NT = BN / 8;                          // S n-tiles (8 keys each)
  static constexpr int KS = HD / 8;                          // 8-dim steps of S = Q K^T
  static constexpr int NO = HD / 8;                          // O n-tiles (8 dims each)
  using T = Tile32<HD>;
  static constexpr int Q_BYTES = BM * T::PITCH;
  static constexpr int KV_BYTES = 2 * BN * T::PITCH;         // K | V of one key tile
  static constexpr int PEN_BYTES = KM ? BN * 4 : 0;
  static constexpr int STAGE = KV_BYTES + PEN_BYTES;
  static constexpr int B_BYTES = BM * BN * 4;                // fp32 bias tile
  static constexpr int NBIAS = (NKV - 1) / R + 1;            // the bias tile of key tile j rides with the K | V of its first sample: the tiles in flight
  static constexpr int OFF_Q = 0, OFF_KV = OFF_Q + Q_BYTES, OFF_B = OFF_KV + NKV * STAGE;
  static constexpr int SMEM = OFF_B + NBIAS * B_BYTES;
};

// The cp.async work of one pipeline stage of the forward and of the query-side backward: the K | V tile of sub-step s = (key tile j = s / R, sample r = s % R) (and its penalties) and,
// for the first sample of a key tile, its bias tile.  G: Fwd32Cfg or Dq32Cfg (HD, BN, R, NKV, NBIAS, KV_BYTES, STAGE, B_BYTES, OFF_KV, OFF_B, KM, T).
template <class G>
struct KVBias32Loader {
  static constexpr int HG = G::HD / 4, BN = G::BN, R = G::R, NKV = G::NKV;
  static constexpr int NC = 2 * BN * HG;                    // K | V: 2 x BN tokens x HG granules of 16 B
  static constexpr int KVN = (NC + 255) / 256;
  static constexpr int BI = BN * 4 / 16 * 128 / 256;        // bias tile: 128 rows x 8 chunks / 256 threads
  uint32_t kv_dst[KVN], b_dst[BI];
  const float* kv_src[KVN];
  const float* b_src[BI];
  const float* pen_src;
  long long kv_ld[KVN], ss, kps;
  int tid;

  DEVI KVBias32Loader(const float* k, const float* v, long long ldk, long long ldv, const float* bias, int L, int a0, int h, int qt, int tid_, long long ss_, const float* kpen,
                      long long kps_)
      : ss(ss_), kps(kps_), tid(tid_) {
#pragma unroll
    for (int it = 0; it < KVN; ++it) {
      const int c = tid + it * 256;
      const int which = c / (BN * HG), w = c - which * BN * HG, token = w / HG, g = w - token * HG;
      kv_dst[it] = which * (G::KV_BYTES / 2) + G::T::off(token, g);
      kv_ld[it] = which ? ldv : ldk;
      kv_src[it] = (which ? v : k) + ((long long)a0 * ss + token) * kv_ld[it] + h * G::HD + g * 4;
    }
#pragma unroll
    for (int i = 0; i < BI; ++i) {
      const int c = tid + 256 * i, row = c / 8, ch = c % 8;
      b_dst[i] = swzb32(row, ch);
      b_src[i] = bias + ((long long)h * L + qt * BM + row) * L + ch * 4;
    }
    pen_src = G::KM ? kpen + (long long)a0 * kps_ + (tid & (BN / 4 - 1)) * 4 : nullptr;
  }

  DEVI void issue(uint32_t sbase, int s, int NS) const {
    if (s < NS) {
      const int j = s / R, r = s - j * R, st = s % NKV;
#pragma unroll
      for (int it = 0; it < KVN; ++it)
        if (NC % 256 == 0 || it < KVN - 1 || tid + it * 256 < NC)
          cp_async16(sbase + G::OFF_KV + st * G::STAGE + kv_dst[it], kv_src[it] + ((long long)r * ss + (long long)j * BN) * kv_ld[it]);
      if (G::KM && tid < BN / 4) cp_async16(sbase + G::OFF_KV + st * G::STAGE + G::KV_BYTES + tid * 16, pen_src + r * kps + j * BN);
      if (r == 0) {
#pragma unroll
        for (int i = 0; i < BI; ++i) cp_async16(sbase + G::OFF_B + (j % G::NBIAS) * G::B_BYTES + b_dst[i], b_src[i] + j * BN);
      }
    }
    cp_async_commit();
  }
};

template <class G>
__global__ void __launch_bounds__(256, G::MINB) attn_fwd_tf32_kernel(const Fwd32Params p) {
  constexpr int HD = G::HD, NKV = G::NKV, BN = G::BN, NT = G::NT, KS = G::KS, NO = G::NO, HG = HD / 4;
  using T = typename G::T;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sbase = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int L = p.L, NJ = L / BN;
  const int qt = blockIdx.x, a = blockIdx.y, h = blockIdx.z;

  const KVBias32Loader<G> loader(p.k, p.v, p.ldk, p.ldv, p.bias, L, a, h, qt, tid, p.ss, p.kpen, p.kps);
  // Q rides in group 0
  for (int c = tid; c < BM * HG; c += 256) {
    const int token = c / HG, g = c - token * HG;
    cp_async16(sbase + G::OFF_Q + T::off(token, g), p.q + ((long long)a * p.ss + qt * BM + token) * p.ldq + h * HD + g * 4);
  }
#pragma unroll
  for (int s = 0; s < NKV - 1; ++s) loader.issue(sbase, s, NJ);                  // (R = 1: sub-step = key tile)

  float acc[NO][4], m[2], l[2];
  m[0] = m[1] = -INFINITY;
  l[0] = l[1] = 0.f;
#pragma unroll
  for (int i = 0; i < NO; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;

  const int qrow = warp * 16 + (lane & 7) + 8 * ((lane >> 3) & 1);        // ldmatrix row of this lane in the Q tile (matrices 0 / 1: rows 0-7 / 8-15; 2 / 3 the same rows, +4 columns)
  const int krow = (lane & 7) + 8 * (lane >> 4);                           // ldmatrix row of this lane in a pair of K n-tiles
  for (int j = 0; j < NJ; ++j) {
    cp_async_wait<NKV - 2>();
    __syncthreads();                                   // key tile j landed; every warp is past j - 1, so its stage is free for j + NKV - 1
    loader.issue(sbase, j + NKV - 1, NJ);
    const uint32_t st = sbase + G::OFF_KV + (j % NKV) * G::STAGE;
    const uint32_t bb = sbase + G::OFF_B + (j % G::NBIAS) * G::B_BYTES;
    // ---- S = Q K^T   (fp32 operands, 16 queries x BN keys per warp)
    float sacc[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) sacc[nt][0] = sacc[nt][1] = sacc[nt][2] = sacc[nt][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      uint32_t qa[4];
      ldsm_x4(qa, sbase + G::OFF_Q + T::off(qrow, 2 * ks + (lane >> 4)));
#pragma unroll
      for (int pp = 0; pp < NT / 2; ++pp) {
        uint32_t kb[4];
        ldsm_x4(kb, st + T::off(16 * pp + krow, 2 * ks + ((lane >> 3) & 1)));
        mma1688_tf32(sacc[2 * pp], qa, kb[0], kb[1]);
        mma1688_tf32(sacc[2 * pp + 1], qa, kb[2], kb[3]);
      }
    }
    // ---- scores: scl q.k + bscl bias (+ the sample's key penalties); the online softmax (base 2)
    float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const float2 bv = *reinterpret_cast<const float2*>(smem_raw + (bb - sbase) + swzb32(warp * 16 + g8 + 8 * hh, 2 * nt + (q4 >> 1)) + (q4 & 1) * 8);
        sacc[nt][2 * hh] = fmaf(sacc[nt][2 * hh], p.scl, bv.x * p.bscl);
        sacc[nt][2 * hh + 1] = fmaf(sacc[nt][2 * hh + 1], p.scl, bv.y * p.bscl);
      }
      if constexpr (G::KM) {
        const uint2 pu = lds64(st + G::KV_BYTES + (8 * nt + 2 * q4) * 4);
        const float p0 = __uint_as_float(pu.x), p1 = __uint_as_float(pu.y);
        sacc[nt][0] += p0; sacc[nt][1] += p1; sacc[nt][2] += p0; sacc[nt][3] += p1;
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
      alpha[hh] = ex2f(m[hh] - mu[hh]);
      m[hh] = mnew;
    }
    float rs[2] = {0.f, 0.f};
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      sacc[nt][0] = ex2f(sacc[nt][0] - mu[0]); sacc[nt][1] = ex2f(sacc[nt][1] - mu[0]);
      sacc[nt][2] = ex2f(sacc[nt][2] - mu[1]); sacc[nt][3] = ex2f(sacc[nt][3] - mu[1]);
      rs[0] += sacc[nt][0] + sacc[nt][1];
      rs[1] += sacc[nt][2] + sacc[nt][3];
    }
    l[0] = fmaf(l[0], alpha[0], rs[0]);
    l[1] = fmaf(l[1], alpha[1], rs[1]);
#pragma unroll
    for (int nt = 0; nt < NO; ++nt) {
      acc[nt][0] *= alpha[0]; acc[nt][1] *= alpha[0];
      acc[nt][2] *= alpha[1]; acc[nt][3] *= alpha[1];
    }
    // ---- O += P V: P's A fragment is the S accumulator (key index permuted, see above), V's B fragment two scalar loads
    const uint32_t vbase = st + G::KV_BYTES / 2;
#pragma unroll
    for (int c = 0; c < NT; ++c) {
      const uint32_t pa[4] = {__float_as_uint(sacc[c][0]), __float_as_uint(sacc[c][2]), __float_as_uint(sacc[c][1]), __float_as_uint(sacc[c][3])};
#pragma unroll
      for (int np = 0; np < NO; ++np) {
        const uint32_t addr = vbase + (8 * c + 2 * q4) * T::PITCH + (8 * np + g8) * 4;
        mma1688_tf32(acc[np], pa, lds32(addr), lds32(addr + T::PITCH));
      }
    }
  }

  // ---- epilogue: O / l (a zero sum -> zero output), [* sigmoid(g)], and the log-sum-exp
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    float lt = l[hh];
    lt += __shfl_xor_sync(0xffffffffu, lt, 1);
    lt += __shfl_xor_sync(0xffffffffu, lt, 2);
    const float inv = lt > 0.f ? 1.f / lt : 1.f;
    const int query = qt * BM + warp * 16 + g8 + 8 * hh;
    const long long tok = (long long)a * p.ss + query;
#pragma unroll
    for (int nt = 0; nt < NO; ++nt) {
      const int col = h * HD + 8 * nt + 2 * q4;
      float v0 = acc[nt][2 * hh] * inv, v1 = acc[nt][2 * hh + 1] * inv;
      if constexpr (G::GATE) {
        const float2 gv = *reinterpret_cast<const float2*>(p.gate + tok * p.ldg + col);
        v0 *= sigmoid(gv.x); v1 *= sigmoid(gv.y);
      }
      *reinterpret_cast<float2*>(p.out + tok * p.ldo + col) = make_float2(v0, v1);
    }
    if (!G::GATE && q4 == 0) {
      const float mm = m[hh] == -INFINITY ? -1e38f : m[hh];
      p.lse[((long long)a * p.H + h) * L + query] = mm + log2f(lt > 0.f ? lt : 1.f);
    }
  }
}

}  // namespace aa80
