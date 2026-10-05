// attn_fwd_sm80.cuh -- triangle-attention core, forward, A100 / sm_80: head dim 32, bf16 operands, fp32 softmax and accumulation, bf16 out.
//
//   out[z, i, j, h, :] = softmax_k( scale * q[z, i, j, h, :] . k[z, i, k, h, :] + bias[z, h, j, k] ) @ v[z, i, k, h, :]
//
// i = the pair row (an independent attention problem), j = query, k = key; the bias belongs to (j, k) and is shared by every pair row.
// Masked keys carry the bias bf16-min (the module's masked_fill): their logit is -inf, their weight 0; a query whose keys are all
// masked gets a zero output (as the Triton kernel's l = 0 guard).
//
// One CTA task = (z, head h, 128-query tile) x R pair rows.  The bias tile [128 q][BN k] of a key tile is staged ONCE and used by all R
// rows, and K | V of a row are re-read from L2 only once per 128 queries: per score element 128 / 128 + 2 / R bytes of L2 -> SM traffic
// (a CTA of one row would re-read the bias L per head: 3.6 GB at L = 768).  Warp w owns queries [16 w, 16 w + 16) of every row, so a
// warp's state is R x (O [16 x 32] fp32, running max, running sum).
// Per key tile j and row r (a "sub-step", NS = R NJ of them, key tile outermost):
//   S = Q_r . K_{r,j}^T          BN / 4 mma.sync m16n8k16 (A = Q via ldmatrix, B = K via ldmatrix)
//   x = S * scale log2 e + bias log2 e; online softmax in base 2 (running max, rescale of O)
//   O_r += P . V_{r,j}           BN / 4 mma (A = P from the S accumulators, B = V via ldmatrix.trans)
// K | V of a sub-step stream through an NKV-stage cp.async ring, the bias tile through two buffers (key tile j uses buffer j & 1).
// All smem tiles are XOR-swizzled so every ldmatrix and every bias fragment read is conflict-free.
//
// Schedule knobs (template): R rows per CTA, NKV ring stages, BN keys per tile (64 / 32), MINB = CTAs per SM the register budget is
// bounded for (2 -> at most 128 registers per thread, 16 warps per SM: the softmax of one CTA runs under the tensor work of the other).
#pragma once
#include "sm80_common.cuh"

namespace a100 {

constexpr int TA_D = 32, TA_BM = 128;
constexpr float TA_L2E = 1.4426950408889634f;

struct AttnParams {
  const __nv_bfloat16 *q, *k, *v;   // token-major [z * L * L][ld]; token t = (z L + i) L + j, head h = columns [32 h, 32 h + 32)
  const __nv_bfloat16* bias;        // [z][H][L][L] (query j, key k), bf16, masked keys = bf16 min
  __nv_bfloat16* out;               // token-major [z * L * L][ldo]
  float* lse;                       // [z][H][L][L] (row i, query j): m + log2 l in the scaled base-2 domain, or nullptr
  long long ldq, ldk, ldv, ldo;     // row strides in elements
  int L, H;
  float scl;                        // log2 e / sqrt(32)
  // the gated variant (AttnCfg::GATE): aout = bf16(sigmoid(gate) bf16(out)) as the output projection's operand, gate = the front's g columns; `out` may then be nullptr (inference)
  const __nv_bfloat16* gate;        // token-major [z * L * L][ldgate], head h = columns [32 h, 32 h + 32)
  __nv_bfloat16* aout;              // token-major [z * L * L][ldao]
  long long ldgate, ldao;
  const int *indices = nullptr, *counts = nullptr;  // optional compact key order and live counts, [Z, L] / [Z]
};

template <int R_, int NKV_, int BN_, int MINB_, bool GATE_ = false, bool COMPACT_ = false>
struct AttnCfg {
  static constexpr bool GATE = GATE_, COMPACT = COMPACT_;
  static constexpr int R = R_, NKV = NKV_, BN = BN_, MINB = MINB_, NTHR = 256;
  static_assert(BN == 64 || BN == 48 || BN == 32, "BN");
  static constexpr int NT = BN / 8;                          // S n-tiles (8 keys each)
  static constexpr int KB = BN / 16;                         // 16-key blocks: the PV k-steps
  static constexpr int Q_BYTES = TA_BM * TA_D * 2;           // one row's Q tile: 8 KiB
  static constexpr int KV_BYTES = 2 * BN * TA_D * 2;         // K | V of one sub-step
  static constexpr int B_BYTES = TA_BM * (BN == 48 ? 64 : BN) * 2;  // 48-key rows use the 128-byte swizzle
  static constexpr int OFF_Q = 0, OFF_KV = OFF_Q + R * Q_BYTES, OFF_B = OFF_KV + NKV * KV_BYTES;
  static constexpr int SMEM = OFF_B + 2 * B_BYTES;
};

// 64 B rows (32 bf16): 16 B granule g of row r sits at g ^ ((r >> 1) & 3), so the 8 rows of one ldmatrix matrix cover all 32 banks
DEVI uint32_t swz64(uint32_t row, uint32_t g) { return row * 64u + ((g ^ ((row >> 1) & 3u)) << 4); }

// the bias tile's rows are BN * 2 bytes: 128 B (BN = 64, the common 8-row XOR swizzle) or 64 B (BN = 32, as the K | V tiles)
template <int BN>
DEVI uint32_t bias_off(uint32_t row, uint32_t g) {
  if constexpr (BN >= 48) return swz<128>(row, g * 16);
  else return swz64(row, g);
}

DEVI float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }

// The cp.async work of one pipeline stage of the forward and of the query-side backward: the K | V tile of sub-step s = (key tile j = s / R, pair row r = s % R)
// and, for the first row of a key tile, its bias tile.  Everything that depends on the thread only (shared-memory offsets, the global pointers at (row0, key tile
// 0)) is made once in the constructor, so the loop pays one 64-bit address update per chunk instead of the whole index computation.  G: AttnCfg or DqCfg.
template <class G, bool COMPACT = false>
struct KVBiasLoader {
  static constexpr int BN = G::BN, R = G::R, NKV = G::NKV, KVN = (8 * BN + 255) / 256, BI = BN / 16;
  uint32_t kv_dst[KVN], b_dst[BI];
  const __nv_bfloat16 *kv_src[KVN], *b_src[BI];
  int kv_ld[KVN];
  int tokens[KVN];
  bool kv_ok[KVN];
  const int* indices;

  DEVI KVBiasLoader(const __nv_bfloat16* k, const __nv_bfloat16* v, long long ldk, long long ldv, const __nv_bfloat16* bias, int L, long long tok0, int row0, int h,
                    int z, int H, int qt, int tid, const int* idx = nullptr) : indices(idx) {
#pragma unroll
    for (int i = 0; i < KVN; ++i) {                          // K | V: 2 x BN tokens x 4 granules of 16 B
      const int c = tid + 256 * i, which = c / (4 * BN), w = c - which * 4 * BN, token = w >> 2, g = w & 3;
      tokens[i] = token;
      kv_ok[i] = c < 8 * BN;
      kv_dst[i] = which * (G::KV_BYTES / 2) + swz64(token, g);
      kv_ld[i] = which ? (int)ldv : (int)ldk;
      kv_src[i] = (which ? v : k) + (tok0 + (long long)row0 * L + (COMPACT ? 0 : token)) * kv_ld[i] + h * TA_D + g * 8;
    }
#pragma unroll
    for (int i = 0; i < BI; ++i) {                           // bias tile: 128 rows x BN / 8 granules
      const int c = tid + 256 * i, row = c / (BN / 8), gg = c % (BN / 8);
      b_dst[i] = bias_off<BN>(row, gg);
      b_src[i] = bias + ((long long)(z * H + h) * L + qt * TA_BM + row) * L + gg * 8;
    }
  }

  DEVI void issue(uint32_t sbase, int s, int NS, int L) const {
    if (s < NS) {
      const int j = s / R, r = s - j * R, st = s % NKV;
#pragma unroll
      for (int i = 0; i < KVN; ++i) {
        if constexpr (BN == 48) { if (!kv_ok[i]) continue; }
        if constexpr (COMPACT) {
          const int key = __ldg(indices + j * BN + tokens[i]);
          cp_async16(sbase + G::OFF_KV + st * G::KV_BYTES + kv_dst[i],
                     kv_src[i] + (long long)(r * L + max(key, 0)) * kv_ld[i], key >= 0 ? 16 : 0);
        } else {
          cp_async16(sbase + G::OFF_KV + st * G::KV_BYTES + kv_dst[i], kv_src[i] + (long long)(r * L + j * BN) * kv_ld[i]);
        }
      }
      if (r == 0) {
#pragma unroll
        for (int i = 0; i < BI; ++i) cp_async16(sbase + G::OFF_B + (j & 1) * G::B_BYTES + b_dst[i], b_src[i] + j * BN);
      }
    }
    cp_async_commit();
  }
};

template <class G>
__global__ void __launch_bounds__(256, G::MINB) attn_fwd_kernel(const AttnParams p) {
  constexpr int R = G::R, NKV = G::NKV, BN = G::BN, NT = G::NT, KB = G::KB;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sbase = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int L = p.L;
  const int qt = blockIdx.x, row0 = blockIdx.y * R;
  const int h = blockIdx.z % p.H, z = blockIdx.z / p.H;
  const int NJ = G::COMPACT ? (p.counts[z] + BN - 1) / BN : L / BN, NS = R * NJ;
  const long long tok0 = (long long)z * L * L;

  // ---- loads: sub-step s = (key tile j = s / R, row r = s % R) brings its K | V, and the first row of a key tile also its bias tile
  const KVBiasLoader<G, G::COMPACT> loader(p.k, p.v, p.ldk, p.ldv, p.bias, L, tok0, row0, h, z, p.H, qt, tid,
                                          G::COMPACT ? p.indices + z * L : nullptr);
  auto issue = [&](int s) { loader.issue(sbase, s, NS, L); };
  // Q of the R rows rides in group 0
#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int c = tid + 256 * i, token = c >> 2, g = c & 3;
      cp_async16(sbase + G::OFF_Q + r * G::Q_BYTES + swz64(token, g),
                 p.q + (tok0 + (long long)(row0 + r) * L + qt * TA_BM + token) * p.ldq + h * TA_D + g * 8);
    }
  }
#pragma unroll
  for (int s = 0; s < NKV - 1; ++s) issue(s);

  float acc[R][4][4], m[R][2], l[R][2], b2[NT][4];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    m[r][0] = m[r][1] = -INFINITY;
    l[r][0] = l[r][1] = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[r][i][0] = acc[r][i][1] = acc[r][i][2] = acc[r][i][3] = 0.f;
  }

  const int qrow = warp * 16 + (lane & 7) + 8 * ((lane >> 3) & 1);   // ldmatrix row of this lane: Q rows (V keys, K keys use other lane bits below)
  for (int j = 0; j < NJ; ++j) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int s = j * R + r, st = s % NKV;
      cp_async_wait<NKV - 2>();
      __syncthreads();                                   // sub-step s landed; every warp is past s - 1, so its stage is free for s + NKV - 1
      issue(s + NKV - 1);
      if (r == 0) {                                      // this key tile's bias fragments: bias * log2 e in the S accumulator layout
        const uint32_t bb = sbase + G::OFF_B + (j & 1) * G::B_BYTES;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const uint32_t u = lds32(bb + bias_off<BN>(warp * 16 + (lane >> 2) + 8 * hh, nt) + (lane & 3) * 4);
            b2[nt][2 * hh] = bf16lo(u) * TA_L2E;
            b2[nt][2 * hh + 1] = bf16hi(u) * TA_L2E;
          }
      }
      // ---- S = Q K^T
      uint32_t qa[2][4];
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) ldsm_x4(qa[ks], sbase + G::OFF_Q + r * G::Q_BYTES + swz64(qrow, 2 * ks + (lane >> 4)));
      float sacc[NT][4];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) sacc[nt][0] = sacc[nt][1] = sacc[nt][2] = sacc[nt][3] = 0.f;
      const uint32_t kbase = sbase + G::OFF_KV + st * G::KV_BYTES;
#pragma unroll
      for (int pp = 0; pp < KB; ++pp)
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
          uint32_t kb[4];
          ldsm_x4(kb, kbase + swz64(16 * pp + 8 * ((lane >> 4) & 1) + (lane & 7), 2 * ks + ((lane >> 3) & 1)));
          mma16816(sacc[2 * pp], qa[ks], kb[0], kb[1]);
          mma16816(sacc[2 * pp + 1], qa[ks], kb[2], kb[3]);
        }
      // ---- online softmax (base 2)
      float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        sacc[nt][0] = fmaf(sacc[nt][0], p.scl, b2[nt][0]);
        sacc[nt][1] = fmaf(sacc[nt][1], p.scl, b2[nt][1]);
        sacc[nt][2] = fmaf(sacc[nt][2], p.scl, b2[nt][2]);
        sacc[nt][3] = fmaf(sacc[nt][3], p.scl, b2[nt][3]);
        mx[0] = fmaxf(mx[0], fmaxf(sacc[nt][0], sacc[nt][1]));
        mx[1] = fmaxf(mx[1], fmaxf(sacc[nt][2], sacc[nt][3]));
      }
      float mu[2], alpha[2];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffffu, mx[hh], 1));
        mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffffu, mx[hh], 2));
        const float mnew = fmaxf(m[r][hh], mx[hh]);
        mu[hh] = mnew == -INFINITY ? 0.f : mnew;         // a fully masked key tile: subtract 0, every weight is 2^-inf = 0
        alpha[hh] = ex2f(m[r][hh] - mu[hh]);
        m[r][hh] = mnew;
      }
      uint32_t pa[KB][4];
      float rs[2] = {0.f, 0.f};
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const float p0 = ex2f(sacc[nt][0] - mu[0]), p1 = ex2f(sacc[nt][1] - mu[0]);
        const float p2 = ex2f(sacc[nt][2] - mu[1]), p3 = ex2f(sacc[nt][3] - mu[1]);
        rs[0] += p0 + p1;
        rs[1] += p2 + p3;
        pa[nt >> 1][(nt & 1) * 2] = pack_bf16(p0, p1);
        pa[nt >> 1][(nt & 1) * 2 + 1] = pack_bf16(p2, p3);
      }
      l[r][0] = fmaf(l[r][0], alpha[0], rs[0]);
      l[r][1] = fmaf(l[r][1], alpha[1], rs[1]);
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        acc[r][nt][0] *= alpha[0]; acc[r][nt][1] *= alpha[0];
        acc[r][nt][2] *= alpha[1]; acc[r][nt][3] *= alpha[1];
      }
      // ---- O += P V
      const uint32_t vbase = kbase + G::KV_BYTES / 2;
#pragma unroll
      for (int c = 0; c < KB; ++c)
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          uint32_t vb[4];
          ldsm_x4_t(vb, vbase + swz64(16 * c + 8 * ((lane >> 3) & 1) + (lane & 7), 2 * np + (lane >> 4)));
          mma16816(acc[r][2 * np], pa[c], vb[0], vb[1]);
          mma16816(acc[r][2 * np + 1], pa[c], vb[2], vb[3]);
        }
    }
  }

  // A fully masked batch has no key iterations to wait on the prologue's Q copies.
  if constexpr (G::COMPACT) cp_async_wait<0>();
  // ---- epilogue: O / l (a zero sum -> zero output), bf16 stores, optionally the log-sum-exp
  uint32_t gate_w[G::GATE ? R : 1][2][4];                            // the gated variant reads g first, all of it before the first store (the stores are volatile asm: a load after one would wait for it)
  if constexpr (G::GATE) {
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const long long tok = tok0 + (long long)(row0 + r) * L + qt * TA_BM + warp * 16 + (lane >> 2) + 8 * hh;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) gate_w[r][hh][nt] = *reinterpret_cast<const uint32_t*>(p.gate + tok * p.ldgate + h * TA_D + 8 * nt + 2 * (lane & 3));
      }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      float lt = l[r][hh];
      lt += __shfl_xor_sync(0xffffffffu, lt, 1);
      lt += __shfl_xor_sync(0xffffffffu, lt, 2);
      const float inv = lt > 0.f ? 1.f / lt : 1.f;
      const int query = qt * TA_BM + warp * 16 + (lane >> 2) + 8 * hh;
      const long long tok = tok0 + (long long)(row0 + r) * L + query;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const uint32_t ow = pack_bf16(acc[r][nt][2 * hh] * inv, acc[r][nt][2 * hh + 1] * inv);
        if (!G::GATE || p.out != nullptr) stg32(p.out + tok * p.ldo + h * TA_D + 8 * nt + 2 * (lane & 3), ow);
        if constexpr (G::GATE) {                                  // a = bf16(sigmoid(g) o): the module's sigmoid-gate statement (f3_sm80.cuh's gate_pair)
          const uint32_t gw = gate_w[r][hh][nt];
          stg32(p.aout + tok * p.ldao + h * TA_D + 8 * nt + 2 * (lane & 3), pack_bf16(bf16lo(ow) * sigmoid(bf16lo(gw)), bf16hi(ow) * sigmoid(bf16hi(gw))));
        }
      }
      if (p.lse != nullptr && (lane & 3) == 0) {
        const float mm = m[r][hh] == -INFINITY ? -1e38f : m[r][hh];
        p.lse[((long long)(z * p.H + h) * L + (row0 + r)) * L + query] = mm + log2f(lt > 0.f ? lt : 1.f);
      }
    }
  }
}

}  // namespace a100
