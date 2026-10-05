// attn_bwd_dq_sm80.cuh -- attention with a shared pair bias, backward, the query side, A100 / sm_80: dq and the bias gradient.
//
//   P  = 2^(scl q.k + bscl bias - lse)         recomputed from the forward's log-sum-exp (base 2, scaled domain; masked keys -> 0)
//   dP = dO . V^T,  dS = P (dP - delta)        delta[a, h, i] = sum_d o do (the backward's row term, from the gate backward)
//   dq = sm_scale  sum_k dS K                  dbias[h, i, k] = sum_a dS_a[i, k]     (the bias is shared by every sample)
//
// The structure of the forward (attn_fwd_sm80.cuh): one CTA = (head h, 128-query tile) x R samples, warp w owns 16 queries of every sample, the key tiles
// stream through an NKV-stage cp.async ring (K | V of a sample's key tile) and the bias tile of a key tile is staged once for the R samples.  Per key
// tile and sample (a "sub-step"): S = Q K^T (mma), P, dP = dO V^T (mma), dS (fp32), dQ += dS K (mma; K via ldmatrix.trans).  The bias gradient of the R
// samples of a CTA is summed in registers over the samples of a key tile and written once per key tile as a bf16 PARTIAL dbp[sample group][h][i][k];
// ``db_reduce_kernel`` adds the A / R partials in a fixed order (deterministic) into the fp32 gradient.
// Builds with OB (bf16 gradients, the attention module's core) take the bias in RAW units and start S from it through the tensor core, as the forward's MODE_PLAIN
// (then P = 2^(scl S - lse), bscl unused); the fp32-gradient builds (the token DiT's fused training) keep the fp32 add.
#pragma once
#include "attn_fwd_sm80.cuh"

namespace aa80 {

struct DqParams {
  const __nv_bfloat16 *q, *k, *v;   // token-major [A * L][ld], head h = columns [HD h, HD h + HD)
  const __nv_bfloat16* dov;         // [A * L][lddo] the attention output's gradient (bf16)
  const __nv_bfloat16* bias;        // [H][L][L] natural units, masked keys -inf
  const float* lse;                 // [A][H][L] the forward's log-sum-exp (base 2)
  const float* delta;               // [A][H][L] sum_d o do
  void* dq;                         // [A * L][lddq]: fp32, or bf16 when the schedule has OB
  __nv_bfloat16* dbp;               // [A / R][H][L][L] partial bias gradients
  const float* kpen;                // KM: per-sample key penalties [A][L] (0 / -inf), sample stride kps floats (see attn_fwd_sm80.cuh)
  long long ldq, ldk, ldv, lddo, lddq, ss, kps;   // ss: tokens between consecutive samples
  int L, H;
  float scl, bscl, sm_scale;
};

template <int HD_, int R_, int NKV_, int MINB_, int OB_ = 0, int KM_ = 0>
struct DqCfg {
  static constexpr int HD = HD_, R = R_, NKV = NKV_, MINB = MINB_, OB = OB_, KM = KM_, NTHR = 256, BN = 32;   // OB: bf16 gradients; KM: per-sample key mask
  static constexpr int NT = BN / 8, KB = BN / 16, KS = HD / 16, NO = HD / 8;
  using T = Tile<HD>;
  static constexpr int Q_BYTES = BM * T::PITCH;              // one sample's Q (and dO) tile
  static constexpr int KV_BYTES = 2 * BN * T::PITCH;
  static constexpr int PEN_BYTES = KM ? BN * 4 : 0;          // KM: the key penalties of the sub-step, after K | V
  static constexpr int STAGE = KV_BYTES + PEN_BYTES;
  static constexpr int B_BYTES = BM * BN * 2;
  static constexpr int OFF_Q = 0, OFF_DO = OFF_Q + R * Q_BYTES, OFF_KV = OFF_DO + R * Q_BYTES, OFF_B = OFF_KV + NKV * STAGE;
  static constexpr int NB = (NKV - 1) / R + 1;               // the bias ring: see FwdCfg
  static constexpr int SMEM = OFF_B + NB * B_BYTES;
};

template <class G>
__global__ void __launch_bounds__(256, G::MINB) attn_bwd_dq_kernel(const DqParams p) {
  constexpr int HD = G::HD, R = G::R, NKV = G::NKV, BN = G::BN, NT = G::NT, KB = G::KB, KS = G::KS, NO = G::NO, HG = HD / 8;
  using T = typename G::T;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sbase = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int L = p.L, NJ = L / BN, NS = R * NJ;
  const int qt = blockIdx.x, gi = blockIdx.y, a0 = gi * R, h = blockIdx.z;

  // ---- loads: as the forward: sub-step s = (key tile j = s / R, sample r = s % R) brings its K | V, the first sample of a key tile also its bias tile
  const KVBiasLoader<G> loader(p.k, p.v, p.ldk, p.ldv, p.bias, L, a0, h, qt, tid, p.ss, p.kpen, p.kps);
  auto issue = [&](int s) { loader.issue(sbase, s, NS, L); };
  // Q and dO of the R samples ride in group 0
#pragma unroll
  for (int r = 0; r < R; ++r)
    for (int c = tid; c < BM * HG; c += 256) {
      const int token = c / HG, g = c - token * HG;
      const long long t = (long long)(a0 + r) * p.ss + qt * BM + token;
      cp_async16(sbase + G::OFF_Q + r * G::Q_BYTES + T::off(token, g), p.q + t * p.ldq + h * HD + g * 8);
      cp_async16(sbase + G::OFF_DO + r * G::Q_BYTES + T::off(token, g), p.dov + t * p.lddo + h * HD + g * 8);
    }
#pragma unroll
  for (int s = 0; s < NKV - 1; ++s) issue(s);

  // OB (bf16 gradients: the module-level core) takes the bias in raw units (bias / sm_scale) and adds it into S through the tensor core, as the forward's MODE_PLAIN
#ifdef AA_NO_RAWB
  constexpr bool RAWB = false;                              // (A/B builds only)
#else
  constexpr bool RAWB = G::OB != 0;
#endif
  float acc[R][NO][4], lsev[R][2], dlt[R][2], b2[RAWB ? 1 : NT][4], dbacc[NT][4];
  uint32_t ba[RAWB ? KB : 1][4];
  const uint32_t idl = bias_identity(lane);
#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int i = 0; i < NO; ++i) acc[r][i][0] = acc[r][i][1] = acc[r][i][2] = acc[r][i][3] = 0.f;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {                             // the row terms of this thread's two queries
      const long long at = ((long long)(a0 + r) * p.H + h) * L + qt * BM + warp * 16 + g8 + 8 * hh;
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
      if (r == 0) {                                              // this key tile's bias (fragments times bscl, or mma A fragments); the bias gradient restarts
        const uint32_t bb = sbase + G::OFF_B + (j % G::NB) * G::B_BYTES;
        if constexpr (RAWB) {
#pragma unroll
          for (int kb = 0; kb < KB; ++kb) ldsm_x4(ba[kb], bb + swz64(qrow, 2 * kb + (lane >> 4)));
        }
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          if constexpr (!RAWB) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
              const uint32_t u = lds32(bb + swz64(warp * 16 + (lane >> 2) + 8 * hh, nt) + (lane & 3) * 4);
              b2[nt][2 * hh] = bf16lo(u) * p.bscl;
              b2[nt][2 * hh + 1] = bf16hi(u) * p.bscl;
            }
          }
          dbacc[nt][0] = dbacc[nt][1] = dbacc[nt][2] = dbacc[nt][3] = 0.f;
        }
      }
      const uint32_t kbase = sbase + G::OFF_KV + st * G::STAGE, vbase = kbase + G::KV_BYTES / 2;
      // ---- S = Q K^T and dP = dO V^T
      uint32_t qa[KS][4], da[KS][4];
#pragma unroll
      for (int ks = 0; ks < KS; ++ks) {
        ldsm_x4(qa[ks], sbase + G::OFF_Q + r * G::Q_BYTES + T::off(qrow, 2 * ks + (lane >> 4)));
        ldsm_x4(da[ks], sbase + G::OFF_DO + r * G::Q_BYTES + T::off(qrow, 2 * ks + (lane >> 4)));
      }
      float sacc[NT][4], dpacc[NT][4];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        sacc[nt][0] = sacc[nt][1] = sacc[nt][2] = sacc[nt][3] = 0.f;
        dpacc[nt][0] = dpacc[nt][1] = dpacc[nt][2] = dpacc[nt][3] = 0.f;
      }
      if constexpr (RAWB) {
#pragma unroll
        for (int kb = 0; kb < KB; ++kb) {                        // S starts as the bias (raw units)
          mma16816(sacc[2 * kb], ba[kb], idl, 0u);
          mma16816(sacc[2 * kb + 1], ba[kb], 0u, idl);
        }
      }
#pragma unroll
      for (int pp = 0; pp < KB; ++pp)
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
          uint32_t kb[4], vb[4];
          ldsm_x4(kb, kbase + T::off(16 * pp + 8 * ((lane >> 4) & 1) + (lane & 7), 2 * ks + ((lane >> 3) & 1)));
          ldsm_x4(vb, vbase + T::off(16 * pp + 8 * ((lane >> 4) & 1) + (lane & 7), 2 * ks + ((lane >> 3) & 1)));
          mma16816(sacc[2 * pp], qa[ks], kb[0], kb[1]);
          mma16816(sacc[2 * pp + 1], qa[ks], kb[2], kb[3]);
          mma16816(dpacc[2 * pp], da[ks], vb[0], vb[1]);
          mma16816(dpacc[2 * pp + 1], da[ks], vb[2], vb[3]);
        }
      if constexpr (G::KM) {                                     // this sample's key penalties (0 / -inf): the columns 8 nt + 2 (lane % 4) + {0, 1}
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const uint2 pu = lds64(kbase + G::KV_BYTES + (8 * nt + 2 * (lane & 3)) * 4);
          const float p0 = __uint_as_float(pu.x), p1 = __uint_as_float(pu.y);
          sacc[nt][0] += p0; sacc[nt][1] += p1; sacc[nt][2] += p0; sacc[nt][3] += p1;
        }
      }
      // ---- P, dS = P (dP - delta); the bias gradient sums dS over the samples
      uint32_t pa[KB][4];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          float pv;
          if constexpr (RAWB) pv = ex2f(fmaf(sacc[nt][e], p.scl, -lsev[r][e >> 1]));
          else pv = ex2f(fmaf(sacc[nt][e], p.scl, b2[nt][e]) - lsev[r][e >> 1]);
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
        for (int np = 0; np < NO / 2; ++np) {
          uint32_t kb[4];
          ldsm_x4_t(kb, kbase + T::off(16 * c + 8 * ((lane >> 3) & 1) + (lane & 7), 2 * np + (lane >> 4)));
          mma16816(acc[r][2 * np], pa[c], kb[0], kb[1]);
          mma16816(acc[r][2 * np + 1], pa[c], kb[2], kb[3]);
        }
      if (r == R - 1) {                                          // the bias gradient of this key tile, summed over the R samples: one bf16 partial
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const long long row = (((long long)gi * p.H + h) * L + qt * BM + warp * 16 + g8 + 8 * hh) * L + j * BN;
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) stg32(p.dbp + row + 8 * nt + 2 * q4, pack_bf16(dbacc[nt][2 * hh], dbacc[nt][2 * hh + 1]));
        }
      }
    }
  }

  // ---- epilogue: dq = sm_scale sum_k dS K, fp32
#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const long long tok = (long long)(a0 + r) * p.ss + qt * BM + warp * 16 + g8 + 8 * hh;
#pragma unroll
      for (int nt = 0; nt < NO; ++nt) {
        const float v0 = acc[r][nt][2 * hh] * p.sm_scale, v1 = acc[r][nt][2 * hh + 1] * p.sm_scale;
        if constexpr (G::OB) stg32(reinterpret_cast<__nv_bfloat16*>(p.dq) + tok * p.lddq + h * HD + 8 * nt + 2 * q4, pack_bf16(v0, v1));
        else *reinterpret_cast<float2*>(reinterpret_cast<float*>(p.dq) + tok * p.lddq + h * HD + 8 * nt + 2 * q4) = make_float2(v0, v1);
      }
    }
  }
}

// The bias gradient: db[h][i][k] = scale * sum over the G sample groups of the partials, in a fixed order.  8 consecutive keys per thread (16-byte loads); fp32 out.
// (scale: the gradient with respect to a bias stored in raw units is the natural-unit gradient times sm_scale.)
__global__ void __launch_bounds__(256) db_reduce_kernel(const __nv_bfloat16* __restrict__ dbp, float* __restrict__ db, int groups, long long plane8, float scale) {
  const long long i = (long long)blockIdx.x * 256 + threadIdx.x;       // 8 elements each
  if (i >= plane8) return;
  float s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 8
  for (int g = 0; g < groups; ++g) {
    const uint4 u = ldg128(dbp + ((long long)g * plane8 + i) * 8);
    s[0] += bf16lo(u.x); s[1] += bf16hi(u.x); s[2] += bf16lo(u.y); s[3] += bf16hi(u.y);
    s[4] += bf16lo(u.z); s[5] += bf16hi(u.z); s[6] += bf16lo(u.w); s[7] += bf16hi(u.w);
  }
  float4* o = reinterpret_cast<float4*>(db + i * 8);
  o[0] = make_float4(s[0] * scale, s[1] * scale, s[2] * scale, s[3] * scale);
  o[1] = make_float4(s[4] * scale, s[5] * scale, s[6] * scale, s[7] * scale);
}

}  // namespace aa80
