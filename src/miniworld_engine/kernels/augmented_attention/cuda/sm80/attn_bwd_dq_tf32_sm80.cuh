// attn_bwd_dq_tf32_sm80.cuh -- attention with a shared pair bias, backward, the query side, A100 / sm_80, fp32 operands on the TF32 tensor cores: dq and the bias gradient.
// The twin of attn_bwd_dq_sm80.cuh (see it and attn_fwd_tf32_sm80.cuh for the structure and the fragment conventions):
//
//   P  = 2^(scl q.k + bscl bias - lse)         recomputed from the forward's log-sum-exp (base 2, scaled domain; masked keys -> 0)
//   dP = dO . V^T,  dS = P (dP - delta)        delta[a, h, i] = sum_d o do
//   dq = sm_scale  sum_k dS K                  dbias[h, i, k] = sum_a dS_a[i, k]     (the bias is shared by every sample)
//
// One CTA = (head h, 128-query tile) x R samples (the bias tile of a key tile is staged once for the R samples and their dS is summed in registers into ONE fp32 partial
// dbp[sample group][h][row][k]; ``db_reduce32_kernel`` adds the A / R partials in a fixed order).  dS is dQ's A operand straight from the S accumulator (the key index of the mma
// permuted, see the forward) and K's B fragment two scalar shared loads.  ``qt0`` / the partial's row count let the host run the query tiles in chunks (the partials of a chunk are
// [groups][H][chunk rows][L]).
#pragma once
#include "attn_fwd_tf32_sm80.cuh"

namespace aa80 {

struct Dq32Params {
  const float *q, *k, *v, *dov;     // token-major [A * L][ld] fp32
  const float* bias;                // [H][L][L] natural units, masked keys -inf
  const float* lse;                 // [A][H][L] the forward's log-sum-exp (base 2)
  const float* delta;               // [A][H][L] sum_d o do
  float* dq;                        // [A * L][lddq]
  float* dbp;                       // [A / R][H][chunk rows][L] partial bias gradients
  const float* kpen;                // KM: per-sample key penalties [A][L], sample stride kps floats
  long long ldq, ldk, ldv, lddo, lddq, ss, kps;
  int L, H, qt0, rows_chunk;        // query tiles qt0 .. of this call; rows of the partial buffer
  float scl, bscl, sm_scale;
};

template <int HD_, int R_, int NKV_, int MINB_, int KM_>
struct Dq32Cfg {
  static constexpr int HD = HD_, R = R_, NKV = NKV_, MINB = MINB_, KM = KM_, NTHR = 256, BN = 32;
  static constexpr int NT = BN / 8, KS = HD / 8, NO = HD / 8;
  using T = Tile32<HD>;
  static constexpr int Q_BYTES = BM * T::PITCH;              // one sample's Q (and dO) tile
  static constexpr int KV_BYTES = 2 * BN * T::PITCH;
  static constexpr int PEN_BYTES = KM ? BN * 4 : 0;
  static constexpr int STAGE = KV_BYTES + PEN_BYTES;
  static constexpr int B_BYTES = BM * BN * 4;
  static constexpr int NBIAS = (NKV - 1) / R + 1;
  static constexpr int OFF_Q = 0, OFF_DO = OFF_Q + R * Q_BYTES, OFF_KV = OFF_DO + R * Q_BYTES, OFF_B = OFF_KV + NKV * STAGE;
  static constexpr int SMEM = OFF_B + NBIAS * B_BYTES;
};

template <class G>
__global__ void __launch_bounds__(256, G::MINB) attn_bwd_dq_tf32_kernel(const Dq32Params p) {
  constexpr int HD = G::HD, R = G::R, NKV = G::NKV, BN = G::BN, NT = G::NT, KS = G::KS, NO = G::NO, HG = HD / 4;
  using T = typename G::T;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sbase = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int L = p.L, NJ = L / BN, NS = R * NJ;
  const int qt = p.qt0 + blockIdx.x, gi = blockIdx.y, a0 = gi * R, h = blockIdx.z;

  const KVBias32Loader<G> loader(p.k, p.v, p.ldk, p.ldv, p.bias, L, a0, h, qt, tid, p.ss, p.kpen, p.kps);
  // Q and dO of the R samples ride in group 0
#pragma unroll
  for (int r = 0; r < R; ++r)
    for (int c = tid; c < BM * HG; c += 256) {
      const int token = c / HG, g = c - token * HG;
      const long long t = (long long)(a0 + r) * p.ss + qt * BM + token;
      cp_async16(sbase + G::OFF_Q + r * G::Q_BYTES + T::off(token, g), p.q + t * p.ldq + h * HD + g * 4);
      cp_async16(sbase + G::OFF_DO + r * G::Q_BYTES + T::off(token, g), p.dov + t * p.lddo + h * HD + g * 4);
    }
#pragma unroll
  for (int s = 0; s < NKV - 1; ++s) loader.issue(sbase, s, NS);

  float acc[R][NO][4], lsev[R][2], dlt[R][2], b2[NT][4], dbacc[NT][4];
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
  const int krow = (lane & 7) + 8 * (lane >> 4);
  for (int j = 0; j < NJ; ++j) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int s = j * R + r, st = s % NKV;
      cp_async_wait<NKV - 2>();
      __syncthreads();
      loader.issue(sbase, s + NKV - 1, NS);
      const uint32_t kv = sbase + G::OFF_KV + st * G::STAGE, vt = kv + G::KV_BYTES / 2;
      if (r == 0) {                                              // this key tile's bias fragments, times bscl; the bias gradient restarts
        const uint32_t bb = sbase + G::OFF_B + (j % G::NBIAS) * G::B_BYTES;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const float2 bv = *reinterpret_cast<const float2*>(smem_raw + (bb - sbase) + swzb32(warp * 16 + g8 + 8 * hh, 2 * nt + (q4 >> 1)) + (q4 & 1) * 8);
            b2[nt][2 * hh] = bv.x * p.bscl;
            b2[nt][2 * hh + 1] = bv.y * p.bscl;
          }
          dbacc[nt][0] = dbacc[nt][1] = dbacc[nt][2] = dbacc[nt][3] = 0.f;
        }
      }
      // ---- S = Q K^T and dP = dO V^T
      float sacc[NT][4], dpacc[NT][4];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        sacc[nt][0] = sacc[nt][1] = sacc[nt][2] = sacc[nt][3] = 0.f;
        dpacc[nt][0] = dpacc[nt][1] = dpacc[nt][2] = dpacc[nt][3] = 0.f;
      }
#pragma unroll
      for (int ks = 0; ks < KS; ++ks) {
        uint32_t qa[4], da[4];
        ldsm_x4(qa, sbase + G::OFF_Q + r * G::Q_BYTES + T::off(qrow, 2 * ks + (lane >> 4)));
        ldsm_x4(da, sbase + G::OFF_DO + r * G::Q_BYTES + T::off(qrow, 2 * ks + (lane >> 4)));
#pragma unroll
        for (int pp = 0; pp < NT / 2; ++pp) {
          uint32_t kb[4], vb[4];
          ldsm_x4(kb, kv + T::off(16 * pp + krow, 2 * ks + ((lane >> 3) & 1)));
          ldsm_x4(vb, vt + T::off(16 * pp + krow, 2 * ks + ((lane >> 3) & 1)));
          mma1688_tf32(sacc[2 * pp], qa, kb[0], kb[1]);
          mma1688_tf32(sacc[2 * pp + 1], qa, kb[2], kb[3]);
          mma1688_tf32(dpacc[2 * pp], da, vb[0], vb[1]);
          mma1688_tf32(dpacc[2 * pp + 1], da, vb[2], vb[3]);
        }
      }
      // ---- P, dS = P (dP - delta); the bias gradient sums dS over the samples
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        float pen0 = 0.f, pen1 = 0.f;
        if constexpr (G::KM) {                                   // this sample's key penalties (0 / -inf)
          const uint2 pu = lds64(kv + G::KV_BYTES + (8 * nt + 2 * q4) * 4);
          pen0 = __uint_as_float(pu.x); pen1 = __uint_as_float(pu.y);
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float pen = (e & 1) ? pen1 : pen0;
          const float pv = ex2f(fmaf(sacc[nt][e], p.scl, b2[nt][e] + pen) - lsev[r][e >> 1]);
          const float dsv = pv * (dpacc[nt][e] - dlt[r][e >> 1]);
          sacc[nt][e] = dsv;
          dbacc[nt][e] += dsv;
        }
      }
      // ---- dQ += dS K   (dS is the A operand with the permuted key index; K's B fragment: two scalar loads)
#pragma unroll
      for (int c = 0; c < NT; ++c) {
        const uint32_t pa[4] = {__float_as_uint(sacc[c][0]), __float_as_uint(sacc[c][2]), __float_as_uint(sacc[c][1]), __float_as_uint(sacc[c][3])};
#pragma unroll
        for (int np = 0; np < NO; ++np) {
          const uint32_t addr = kv + (8 * c + 2 * q4) * T::PITCH + (8 * np + g8) * 4;
          mma1688_tf32(acc[r][np], pa, lds32(addr), lds32(addr + T::PITCH));
        }
      }
      if (r == R - 1) {                                          // the bias gradient of this key tile, summed over the R samples: one fp32 partial
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const long long row = (((long long)gi * p.H + h) * p.rows_chunk + (blockIdx.x * BM + warp * 16 + g8 + 8 * hh)) * L + j * BN;
#pragma unroll
          for (int nt = 0; nt < NT; ++nt)
            *reinterpret_cast<float2*>(p.dbp + row + 8 * nt + 2 * q4) = make_float2(dbacc[nt][2 * hh], dbacc[nt][2 * hh + 1]);
        }
      }
    }
  }

  // ---- epilogue: dq = sm_scale sum_k dS K
#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const long long tok = (long long)(a0 + r) * p.ss + qt * BM + warp * 16 + g8 + 8 * hh;
#pragma unroll
      for (int nt = 0; nt < NO; ++nt)
        *reinterpret_cast<float2*>(p.dq + tok * p.lddq + h * HD + 8 * nt + 2 * q4) = make_float2(acc[r][nt][2 * hh] * p.sm_scale, acc[r][nt][2 * hh + 1] * p.sm_scale);
    }
  }
}

// The bias gradient: db[h][r0 + r][k] = scale * sum over the `groups` sample groups of dbp[g][h][r][k] (the chunk's rows), in a fixed order.  8 consecutive keys per thread.
__global__ void __launch_bounds__(256) db_reduce32_kernel(const float* __restrict__ dbp, float* __restrict__ db, int groups, int H, int rows_chunk, int L, int r0, float scale) {
  const long long plane8 = (long long)H * rows_chunk * L / 8;
  const long long i = (long long)blockIdx.x * 256 + threadIdx.x;       // 8 elements each
  if (i >= plane8) return;
  const long long e = i * 8;
  const long long hr = e / L;                                          // (h, r)
  const int j = (int)(e - hr * L), h = (int)(hr / rows_chunk), r = (int)(hr - (long long)h * rows_chunk);
  float s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
  for (int g = 0; g < groups; ++g) {
    const float4 a = __ldg(reinterpret_cast<const float4*>(dbp + (long long)g * H * rows_chunk * L + e)), b = __ldg(reinterpret_cast<const float4*>(dbp + (long long)g * H * rows_chunk * L + e) + 1);
    s[0] += a.x; s[1] += a.y; s[2] += a.z; s[3] += a.w; s[4] += b.x; s[5] += b.y; s[6] += b.z; s[7] += b.w;
  }
  float4* o = reinterpret_cast<float4*>(db + ((long long)h * L + r0 + r) * L + j);
  o[0] = make_float4(s[0] * scale, s[1] * scale, s[2] * scale, s[3] * scale);
  o[1] = make_float4(s[4] * scale, s[5] * scale, s[6] * scale, s[7] * scale);
}

}  // namespace aa80
