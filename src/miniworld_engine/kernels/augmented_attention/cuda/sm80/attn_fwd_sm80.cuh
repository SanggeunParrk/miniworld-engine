// attn_fwd_sm80.cuh -- attention with a shared pair bias, forward, A100 / sm_80: head dim HD (a multiple of 16; 48 for the token DiT), bf16 operands,
// fp32 softmax and accumulation.
//
//   o[a, i, h, :] = softmax_k( scl q[a, i, h, :] . k[a, k, h, :] + bscl bias[h, i, k] ) @ v[a, k, h, :]        (base 2: scl = sm_scale log2 e, bscl = log2 e)
//
// a = a sample (an independent attention problem), i = query, k = key; the bias belongs to (h, i, k) and is shared by every sample (the diffusion
// samples of one trunk output).  Masked keys carry -inf in the bias; a query whose keys are all masked gets a zero output.  Operands are token-major
// views [A * L][ld] (token = a L + i, head h = columns [HD h, HD h + HD)): q | k | v | g are column blocks of the one projection output, so the
// strides are arguments.
//
// This is the structure of the TriangleAttention forward (triangle_attention/cuda/sm80/attn_fwd_sm80.cuh) with the pair rows replaced by samples:
// one CTA = (head h, 128-query tile) x R samples.  The bias tile [128 q][BN k] of a key tile is staged ONCE and used by all R samples, K | V of a
// sample stream through an NKV-stage cp.async ring, warp w owns the queries [16 w, 16 w + 16) of every sample, its state is R x (O [16 x HD] fp32,
// running max, running sum).  Per key tile and sample: S = Q K^T (mma), online softmax in base 2, O += P V (mma).
// Shared-memory rows are PITCH bytes (an odd number of 16 B granules): the 8 rows of an ldmatrix matrix then fall into 8 different bank groups with
// no swizzle, whatever HD (HD = 48: 96 B of data in 112 B rows).
//
// Epilogues: MODE_GATE writes bf16( sigmoid(g) o ) over q (the inference step: q is dead after its tile is loaded, and no other CTA reads these
// rows of q); MODE_TRAIN writes o as fp32 [A L][H HD] and the log-sum-exp lse[a][h][i] = m + log2 l (the backward's recompute reads it); MODE_PLAIN writes o as
// bf16 [A L][H HD] (no gate) and, when lse is not null, the log-sum-exp (the attention module's core: the gate is applied outside); MODE_PGATE is MODE_PLAIN's
// score path (raw-unit bias through the tensor core) with MODE_GATE's epilogue (bf16 sigmoid(g) o, no lse: the module-level inference step).
//
// The bias of MODE_GATE / MODE_TRAIN is read as fp32 fragments and added in fp32 (times bscl; -inf marks a masked key).  MODE_PLAIN takes it in RAW units, bias /
// sm_scale (the units of q . k), and adds it through the tensor core: S starts as the bias (an mma with the bias block as A and an identity as B), q . k accumulates
// onto it, and the softmax works on S directly (p = 2^(scl S - m)): no unpack, no scale, no add per score.  A masked key is then a very negative FINITE bias
// (-inf times the identity's zeros would be NaN); bscl is unused.
//
// KM (a per-sample key mask): the bias is shared by the samples, a key mask that differs per sample cannot ride in it.  The host hands the penalties as fp32
// [A][L] (0 for a valid key, -inf for a masked one, sample stride kps floats); they travel through the same cp.async ring as the K | V tile (BN floats a
// sub-step) and are added to the scores before the softmax.  A row with no valid key has no weight: its output is 0 (and its lse -1e38, so the backward's
// probabilities are 0 as well).  ``ss`` is the distance in tokens between consecutive samples (L for a packed [A L][ld] operand; B L when the operand holds
// [A, B, L] tokens and the call serves one batch element b through the pointers' offsets).
#pragma once
#include "sm80_common.cuh"

namespace aa80 {

constexpr int BM = 128;                       // queries per CTA
constexpr float L2E = 1.4426950408889634f;
enum Mode { MODE_GATE = 0, MODE_TRAIN = 1, MODE_PLAIN = 2, MODE_PGATE = 3 };

// [rows][HD] bf16 tile in shared memory: HD / 8 granules of 16 B in a row of an odd number of granules
template <int HD>
struct Tile {
  static constexpr int HG = HD / 8;
  static constexpr int PITCH = (HG % 2 ? HG : HG + 1) * 16;
  static DEVI uint32_t off(uint32_t row, uint32_t g) { return row * PITCH + g * 16; }
};

// the bias tile's rows are BN = 32 keys = 64 B: 16 B granule g of row r sits at g ^ ((r >> 1) & 3) (the 8 rows of a matrix cover all 32 banks)
DEVI uint32_t swz64(uint32_t row, uint32_t g) { return row * 64u + ((g ^ ((row >> 1) & 3u)) << 4); }

DEVI float ex2f(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }

// The B fragment of the mma that adds a bias tile into S: with the bias block as the A operand (16 queries x 16 keys), B = the identity on the lo 8 keys (b0 = idl,
// b1 = 0) or on the hi 8 keys (b0 = 0, b1 = idl).  A thread holds B[k][n] for n = lane / 4 and k = 2 (lane % 4) + {0, 1}: bf16 1.0 where k == n.
DEVI uint32_t bias_identity(int lane) {
  const int n = lane >> 2, k0 = 2 * (lane & 3);
  return n == k0 ? 0x00003F80u : n == k0 + 1 ? 0x3F800000u : 0u;
}

struct FwdParams {
  const __nv_bfloat16 *q, *k, *v;   // token-major [A * L][ld], head h = columns [HD h, HD h + HD)
  const __nv_bfloat16* bias;        // [H][L][L] (query i, key k); masked keys -inf
  const __nv_bfloat16* gate;        // MODE_GATE / MODE_PGATE: g, token-major like q
  void* out;                        // MODE_GATE / MODE_PLAIN / MODE_PGATE: bf16 token-major [A * L][ldo] (GATE: over q); MODE_TRAIN: float [A * L][ldo]
  float* lse;                       // MODE_TRAIN / MODE_PLAIN: [A][H][L] (PLAIN: may be null)
  const float* kpen;                // KM: per-sample key penalties [A][L] (0 / -inf), sample stride kps floats
  long long ldq, ldk, ldv, ldg, ldo, ss, kps;   // ss: tokens between consecutive samples
  int L, H;
  float scl;                        // multiplies q . k (1 when q is pre-scaled by sm_scale log2 e)
  float bscl;                       // multiplies the bias (log2 e, or 1 when the bias is already in base 2)
};

// RAWB: the bias rides through the tensor core in raw units (MODE_PLAIN, MODE_PGATE); the other modes add it in fp32 base-2 units
constexpr bool mode_rawb(int mode) { return mode == MODE_PLAIN || mode == MODE_PGATE; }

template <int HD_, int R_, int NKV_, int MINB_, int MODE_, int KM_ = 0>
struct FwdCfg {
  static constexpr int HD = HD_, R = R_, NKV = NKV_, MINB = MINB_, MODE = MODE_, KM = KM_, NTHR = 256, BN = 32;
  static_assert(HD % 16 == 0, "HD");
  static constexpr int NT = BN / 8;                          // S n-tiles (8 keys each)
  static constexpr int KB = BN / 16;                         // 16-key blocks: the PV k-steps
  static constexpr int KS = HD / 16;                         // 16-dim steps of S = Q K^T
  static constexpr int NO = HD / 8;                          // O n-tiles (8 dims each)
  using T = Tile<HD>;
  static constexpr int Q_BYTES = BM * T::PITCH;              // one sample's Q tile
  static constexpr int KV_BYTES = 2 * BN * T::PITCH;         // K | V of one sub-step
  static constexpr int PEN_BYTES = KM ? BN * 4 : 0;          // KM: the key penalties of the sub-step, after K | V
  static constexpr int STAGE = KV_BYTES + PEN_BYTES;         // one stage of the ring
  static constexpr int B_BYTES = BM * BN * 2;                // bias tile
  static constexpr int OFF_Q = 0, OFF_KV = OFF_Q + R * Q_BYTES, OFF_B = OFF_KV + NKV * STAGE;
  // the bias tile of key tile j is issued NKV - 1 sub-steps ahead, i.e. up to (NKV - 1) / R key tiles: a ring of that many + 1 buffers (two when R >= NKV - 1)
  static constexpr int NB = (NKV - 1) / R + 1;
  static constexpr int SMEM = OFF_B + NB * B_BYTES;
};

// The cp.async work of one pipeline stage of the forward and of the query-side backward: the K | V tile of sub-step s = (key tile j = s / R, sample r = s % R)
// and, for the first sample of a key tile, its bias tile.  Everything that depends on the thread only (shared-memory offsets, the global pointers at (sample
// a0, key tile 0)) is made once in the constructor, so the loop pays one 64-bit address update per chunk instead of the whole index computation (the loop
// was ~half integer instructions before).  G: FwdCfg or DqCfg (HD, BN, R, NKV, NB, KV_BYTES, B_BYTES, OFF_KV, OFF_B, T).
template <class G>
struct KVBiasLoader {
  static constexpr int HG = G::HD / 8, BN = G::BN, R = G::R, NKV = G::NKV;
  static constexpr int NC = 2 * BN * HG;                    // K | V: 2 x BN tokens x HG granules of 16 B
  static constexpr int KVN = (NC + 255) / 256;              // chunks per thread (the last one guarded when NC is not a multiple of 256)
  static constexpr int BI = BN / 16;                        // bias tile: 128 rows x BN / 8 granules
  uint32_t kv_dst[KVN], b_dst[BI];
  const __nv_bfloat16* kv_src[KVN];
  const __nv_bfloat16* b_src[BI];
  const float* pen_src;                                     // KM: this thread's granule of the key penalties of sample a0 (threads < BN / 4), key tile 0
  long long ss, kps;
  int kv_ld[KVN];
  int tid;

  DEVI KVBiasLoader(const __nv_bfloat16* k, const __nv_bfloat16* v, long long ldk, long long ldv, const __nv_bfloat16* bias, int L, int a0, int h, int qt, int tid_,
                    long long ss_, const float* kpen = nullptr, long long kps_ = 0)
      : ss(ss_), kps(kps_), tid(tid_) {
#pragma unroll
    for (int it = 0; it < KVN; ++it) {
      const int c = tid + it * 256;
      const int which = c / (BN * HG), w = c - which * BN * HG, token = w / HG, g = w - token * HG;
      kv_dst[it] = which * (G::KV_BYTES / 2) + G::T::off(token, g);
      kv_ld[it] = which ? (int)ldv : (int)ldk;
      kv_src[it] = (which ? v : k) + ((long long)a0 * ss + token) * kv_ld[it] + h * G::HD + g * 8;
    }
#pragma unroll
    for (int i = 0; i < BI; ++i) {
      const int c = tid + 256 * i, row = c / (BN / 8), gg = c % (BN / 8);
      b_dst[i] = swz64(row, gg);
      b_src[i] = bias + ((long long)h * L + qt * BM + row) * L + gg * 8;
    }
    pen_src = G::KM ? kpen + (long long)a0 * kps_ + (tid & (BN / 4 - 1)) * 4 : nullptr;
  }

  DEVI void issue(uint32_t sbase, int s, int NS, int L) const {
    if (s < NS) {
      const int j = s / R, r = s - j * R, st = s % NKV;
#pragma unroll
      for (int it = 0; it < KVN; ++it)
        if (NC % 256 == 0 || it < KVN - 1 || tid + it * 256 < NC)
          cp_async16(sbase + G::OFF_KV + st * G::STAGE + kv_dst[it], kv_src[it] + (long long)r * ss * kv_ld[it] + (long long)(j * BN) * kv_ld[it]);
      if (G::KM && tid < BN / 4) cp_async16(sbase + G::OFF_KV + st * G::STAGE + G::KV_BYTES + tid * 16, pen_src + r * kps + j * BN);
      if (r == 0) {
#pragma unroll
        for (int i = 0; i < BI; ++i) cp_async16(sbase + G::OFF_B + (j % G::NB) * G::B_BYTES + b_dst[i], b_src[i] + j * BN);
      }
    }
    cp_async_commit();
  }
};

// ---- epilogue: O / l (a zero sum -> zero output), per MODE; acc / m / l are the thread's accumulators of the R samples of the CTA
template <class G>
DEVI void fwd_epilogue(const FwdParams& p, const float (&acc)[G::R][G::NO][4], const float (&m)[G::R][2], const float (&l)[G::R][2], int qt, int a0, int h, int warp,
                       int lane) {
  constexpr int HD = G::HD, R = G::R, NO = G::NO;
  const int L = p.L;
#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      float lt = l[r][hh];
      lt += __shfl_xor_sync(0xffffffffu, lt, 1);
      lt += __shfl_xor_sync(0xffffffffu, lt, 2);
      const float inv = lt > 0.f ? 1.f / lt : 1.f;
      const int query = qt * BM + warp * 16 + (lane >> 2) + 8 * hh;
      const long long tok = (long long)(a0 + r) * p.ss + query;
      if constexpr (G::MODE == MODE_GATE || G::MODE == MODE_PGATE) {
        __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(p.out);
#pragma unroll
        for (int nt = 0; nt < NO; ++nt) {
          const int col = h * HD + 8 * nt + 2 * (lane & 3);
          const uint32_t gu = __ldg(reinterpret_cast<const unsigned*>(p.gate + tok * p.ldg + col));
          const float g0 = bf16lo(gu), g1 = bf16hi(gu);
          stg32(out + tok * p.ldo + col, pack_bf16(acc[r][nt][2 * hh] * inv * sigmoid(g0), acc[r][nt][2 * hh + 1] * inv * sigmoid(g1)));
        }
      } else if constexpr (G::MODE == MODE_PLAIN) {
        __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(p.out);
#pragma unroll
        for (int nt = 0; nt < NO; ++nt)
          stg32(out + tok * p.ldo + h * HD + 8 * nt + 2 * (lane & 3), pack_bf16(acc[r][nt][2 * hh] * inv, acc[r][nt][2 * hh + 1] * inv));
        if (p.lse != nullptr && (lane & 3) == 0) {
          const float mm = m[r][hh] == -INFINITY ? -1e38f : m[r][hh];
          p.lse[((long long)(a0 + r) * p.H + h) * L + query] = mm + log2f(lt > 0.f ? lt : 1.f);
        }
      } else {
        float* out = reinterpret_cast<float*>(p.out);
#pragma unroll
        for (int nt = 0; nt < NO; ++nt)
          *reinterpret_cast<float2*>(out + tok * p.ldo + h * HD + 8 * nt + 2 * (lane & 3)) =
              make_float2(acc[r][nt][2 * hh] * inv, acc[r][nt][2 * hh + 1] * inv);
        if ((lane & 3) == 0) {
          const float mm = m[r][hh] == -INFINITY ? -1e38f : m[r][hh];
          p.lse[((long long)(a0 + r) * p.H + h) * L + query] = mm + log2f(lt > 0.f ? lt : 1.f);
        }
      }
    }
  }
}

template <class G>
__global__ void __launch_bounds__(256, G::MINB) attn_fwd_kernel(const FwdParams p) {
  constexpr int HD = G::HD, R = G::R, NKV = G::NKV, BN = G::BN, NT = G::NT, KB = G::KB, KS = G::KS, NO = G::NO, HG = HD / 8;
  using T = typename G::T;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sbase = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int L = p.L, NJ = L / BN, NS = R * NJ;
  const int qt = blockIdx.x, a0 = blockIdx.y * R, h = blockIdx.z;

  // ---- loads: sub-step s = (key tile j = s / R, sample r = s % R) brings its K | V, and the first sample of a key tile also its bias tile
  const KVBiasLoader<G> loader(p.k, p.v, p.ldk, p.ldv, p.bias, L, a0, h, qt, tid, p.ss, p.kpen, p.kps);
  auto issue = [&](int s) { loader.issue(sbase, s, NS, L); };
  // Q of the R samples rides in group 0
#pragma unroll
  for (int r = 0; r < R; ++r)
    for (int c = tid; c < BM * HG; c += 256) {
      const int token = c / HG, g = c - token * HG;
      cp_async16(sbase + G::OFF_Q + r * G::Q_BYTES + T::off(token, g),
                 p.q + ((long long)(a0 + r) * p.ss + qt * BM + token) * p.ldq + h * HD + g * 8);
    }
#pragma unroll
  for (int s = 0; s < NKV - 1; ++s) issue(s);

  // MODE_PLAIN takes the bias in raw units (bias / sm_scale, the units of q . k) and adds it into S through the tensor core (RAWB); the other modes take it in base-2 units
  // (times bscl) and add it in fp32
#ifdef AA_NO_RAWB
  constexpr bool RAWB = false;                              // (A/B builds only)
#else
  constexpr bool RAWB = mode_rawb(G::MODE);
#endif
  float acc[R][NO][4], m[R][2], l[R][2], b2[RAWB ? 1 : NT][4];
  uint32_t ba[RAWB ? KB : 1][4];                            // RAWB: the bias tile of this key tile as mma A fragments (16 queries x 16 keys per block)
  const uint32_t idl = bias_identity(lane);                 // RAWB: the B fragment of the mma that adds the bias into S
#pragma unroll
  for (int r = 0; r < R; ++r) {
    m[r][0] = m[r][1] = -INFINITY;
    l[r][0] = l[r][1] = 0.f;
#pragma unroll
    for (int i = 0; i < NO; ++i) acc[r][i][0] = acc[r][i][1] = acc[r][i][2] = acc[r][i][3] = 0.f;
  }

  const int qrow = warp * 16 + (lane & 7) + 8 * ((lane >> 3) & 1);   // ldmatrix row of this lane in the Q tile
  for (int j = 0; j < NJ; ++j) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int s = j * R + r, st = s % NKV;
      cp_async_wait<NKV - 2>();
      __syncthreads();                                   // sub-step s landed; every warp is past s - 1, so its stage is free for s + NKV - 1
      issue(s + NKV - 1);
      if (r == 0) {
        const uint32_t bb = sbase + G::OFF_B + (j % G::NB) * G::B_BYTES;
        if constexpr (RAWB) {                            // this key tile's bias as mma A fragments
#pragma unroll
          for (int kb = 0; kb < KB; ++kb) ldsm_x4(ba[kb], bb + swz64(qrow, 2 * kb + (lane >> 4)));
        } else {                                         // this key tile's bias fragments, times bscl, in the S accumulator layout
#pragma unroll
          for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
              const uint32_t u = lds32(bb + swz64(warp * 16 + (lane >> 2) + 8 * hh, nt) + (lane & 3) * 4);
              b2[nt][2 * hh] = bf16lo(u) * p.bscl;
              b2[nt][2 * hh + 1] = bf16hi(u) * p.bscl;
            }
        }
      }
      // ---- S = Q K^T
      uint32_t qa[KS][4];
#pragma unroll
      for (int ks = 0; ks < KS; ++ks) ldsm_x4(qa[ks], sbase + G::OFF_Q + r * G::Q_BYTES + T::off(qrow, 2 * ks + (lane >> 4)));
      float sacc[NT][4];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) sacc[nt][0] = sacc[nt][1] = sacc[nt][2] = sacc[nt][3] = 0.f;
      if constexpr (RAWB) {
#pragma unroll
        for (int kb = 0; kb < KB; ++kb) {                // S starts as the bias (raw units): A = the bias block, B = the identity on its lo / hi 8 keys
          mma16816(sacc[2 * kb], ba[kb], idl, 0u);
          mma16816(sacc[2 * kb + 1], ba[kb], 0u, idl);
        }
      }
      const uint32_t kbase = sbase + G::OFF_KV + st * G::STAGE;
#pragma unroll
      for (int pp = 0; pp < KB; ++pp)
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
          uint32_t kb[4];
          ldsm_x4(kb, kbase + T::off(16 * pp + 8 * ((lane >> 4) & 1) + (lane & 7), 2 * ks + ((lane >> 3) & 1)));
          mma16816(sacc[2 * pp], qa[ks], kb[0], kb[1]);
          mma16816(sacc[2 * pp + 1], qa[ks], kb[2], kb[3]);
        }
      // ---- online softmax (base 2)
      float mx[2] = {-INFINITY, -INFINITY};
      // RAWB: S is in raw units (q . k + bias / sm_scale): the maximum is taken raw, m / mu live in the scaled base-2 domain (scl = sm_scale log2 e), p = 2^(scl S - mu)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        if constexpr (!RAWB) {
          sacc[nt][0] = fmaf(sacc[nt][0], p.scl, b2[nt][0]);
          sacc[nt][1] = fmaf(sacc[nt][1], p.scl, b2[nt][1]);
          sacc[nt][2] = fmaf(sacc[nt][2], p.scl, b2[nt][2]);
          sacc[nt][3] = fmaf(sacc[nt][3], p.scl, b2[nt][3]);
        }
        if constexpr (G::KM) {                           // this sample's key penalties (0 / -inf): the columns 8 nt + 2 (lane % 4) + {0, 1}
          const uint2 pu = lds64(kbase + G::KV_BYTES + (8 * nt + 2 * (lane & 3)) * 4);
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
        if constexpr (RAWB) mx[hh] *= p.scl;
        const float mnew = fmaxf(m[r][hh], mx[hh]);
        mu[hh] = mnew == -INFINITY ? 0.f : mnew;         // a fully masked key tile: subtract 0, every weight is 2^-inf = 0
        alpha[hh] = ex2f(m[r][hh] - mu[hh]);
        m[r][hh] = mnew;
      }
      uint32_t pa[KB][4];
      float rs[2] = {0.f, 0.f};
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        float p0, p1, p2, p3;
        if constexpr (RAWB) {
          p0 = ex2f(fmaf(sacc[nt][0], p.scl, -mu[0])); p1 = ex2f(fmaf(sacc[nt][1], p.scl, -mu[0]));
          p2 = ex2f(fmaf(sacc[nt][2], p.scl, -mu[1])); p3 = ex2f(fmaf(sacc[nt][3], p.scl, -mu[1]));
        } else {
          p0 = ex2f(sacc[nt][0] - mu[0]); p1 = ex2f(sacc[nt][1] - mu[0]);
          p2 = ex2f(sacc[nt][2] - mu[1]); p3 = ex2f(sacc[nt][3] - mu[1]);
        }
        rs[0] += p0 + p1;
        rs[1] += p2 + p3;
        pa[nt >> 1][(nt & 1) * 2] = pack_bf16(p0, p1);
        pa[nt >> 1][(nt & 1) * 2 + 1] = pack_bf16(p2, p3);
      }
      l[r][0] = fmaf(l[r][0], alpha[0], rs[0]);
      l[r][1] = fmaf(l[r][1], alpha[1], rs[1]);
#pragma unroll
      for (int nt = 0; nt < NO; ++nt) {
        acc[r][nt][0] *= alpha[0]; acc[r][nt][1] *= alpha[0];
        acc[r][nt][2] *= alpha[1]; acc[r][nt][3] *= alpha[1];
      }
      // ---- O += P V
      const uint32_t vbase = kbase + G::KV_BYTES / 2;
#pragma unroll
      for (int c = 0; c < KB; ++c)
#pragma unroll
        for (int np = 0; np < NO / 2; ++np) {
          uint32_t vb[4];
          ldsm_x4_t(vb, vbase + T::off(16 * c + 8 * ((lane >> 3) & 1) + (lane & 7), 2 * np + (lane >> 4)));
          mma16816(acc[r][2 * np], pa[c], vb[0], vb[1]);
          mma16816(acc[r][2 * np + 1], pa[c], vb[2], vb[3]);
        }
    }
  }

  fwd_epilogue<G>(p, acc, m, l, qt, a0, h, warp, lane);
}

}  // namespace aa80
