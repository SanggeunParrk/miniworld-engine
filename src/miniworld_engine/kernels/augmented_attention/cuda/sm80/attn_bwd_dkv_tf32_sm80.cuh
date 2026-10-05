// attn_bwd_dkv_tf32_sm80.cuh -- attention with a shared pair bias, backward, the key side, A100 / sm_80, fp32 operands on the TF32 tensor cores: dk and dv.  The twin of
// attn_bwd_dkv_sm80.cuh (see it and attn_fwd_tf32_sm80.cuh for the structure and the fragment conventions):
//
//   P^T  = 2^(scl k.q + bscl bias^T - lse[q])                     recomputed in the transposed (key x query) space
//   dP^T = V . dO^T,  dS^T = P^T (dP^T - delta[q])
//   dv = sum_q P^T dO        dk = sm_scale sum_q dS^T Q
//
// One CTA = (head h, 128-key tile) of ONE sample (grid.y): warp w owns the keys [16 w, 16 w + 16), K and V of the tile sit in registers as the A operands of S^T = K Q^T and
// dP^T = V dO^T; the query tiles stream through an NST-stage cp.async ring (Q, dO, lse | delta, the transposed bias tile).  P^T / dS^T are the A operands of dv += P^T dO and
// dk += dS^T Q straight from the accumulators (the query index of the mma permuted: slot t <-> query 2 t, t + 4 <-> 2 t + 1), dO's / Q's B fragments two scalar shared loads.
// A masked key (KM) has dk = dv = 0: its row ran without the penalty and is replaced by 0 in the epilogue (rows never mix).
#pragma once
#include "attn_bwd_dq_tf32_sm80.cuh"

namespace aa80 {

struct Dkv32Params {
  const float *q, *k, *v, *dov;     // token-major [A * L][ld] fp32
  const float* bias_t;              // [H][L][L] transposed: bias_t[h][k][j], natural units, masked keys -inf
  const float* lse;                 // [A][H][L]
  const float* delta;               // [A][H][L]
  float *dk, *dv;                   // [A * L][lddk / lddv] fp32
  const float* kpen;                // KM: per-sample key penalties [A][L], sample stride kps floats
  long long ldq, ldk, ldv, lddo, lddk, lddv, ss, kps;
  int L, H;
  float scl, bscl, sm_scale;
};

template <int HD_, int NST_, int MINB_, int KM_>
struct Dkv32Cfg {
  static constexpr int HD = HD_, NST = NST_, MINB = MINB_, KM = KM_, NTHR = 256, BN = 32;
  static constexpr int NT = BN / 8, KS = HD / 8, NO = HD / 8;
  using T = Tile32<HD>;
  static constexpr int KV_BYTES = BM * T::PITCH;              // the CTA's K (and V) tile, resident
  static constexpr int QD_BYTES = BN * T::PITCH;              // a query tile of Q (and of dO)
  static constexpr int ROWT_BYTES = 2 * BN * 4;               // lse | delta of a query tile
  static constexpr int BT_BYTES = BM * BN * 4;                // the transposed bias tile [128 k][BN q]
  static constexpr int STAGE = 2 * QD_BYTES + ROWT_BYTES + BT_BYTES;
  static constexpr int OFF_K = 0, OFF_V = KV_BYTES, OFF_STAGE = 2 * KV_BYTES;
  static constexpr int SMEM = OFF_STAGE + NST * STAGE;
};

// the cp.async work of one pipeline stage of the key side (query tile s: Q, dO, lse | delta, the transposed bias tile)
template <class G>
struct Qd32Loader {
  static constexpr int HG = G::HD / 4, BN = G::BN, NC = 2 * BN * HG, QN = (NC + 255) / 256, BI = BN * 4 / 16 * 128 / 256;
  uint32_t qd_dst[QN], b_dst[BI], rt_dst;
  const float *qd_src[QN], *b_src[BI];
  const float* rt_src;
  long long qd_ld[QN];
  int tid;

  DEVI Qd32Loader(const float* q, const float* dov, long long ldq, long long lddo, const float* bias_t, const float* lse, const float* delta, long long rowtok, long long plane,
                  long long ahl, int kt, int h, int L, int tid_) : tid(tid_) {
#pragma unroll
    for (int it = 0; it < QN; ++it) {                           // Q then dO: BN tokens x HG granules each
      const int c = tid + it * 256;
      const int which = c / (BN * HG), w = c - which * BN * HG, token = w / HG, g = w - token * HG;
      qd_dst[it] = which * G::QD_BYTES + G::T::off(token, g);
      qd_ld[it] = which ? lddo : ldq;
      qd_src[it] = (which ? dov : q) + (rowtok + token) * qd_ld[it] + h * G::HD + g * 4;
    }
    {                                                           // lse | delta of the BN queries: 2 x BN floats = 16 chunks of 16 B (threads 0-15)
      const int which = (tid >> 3) & 1, c = tid & 7;
      rt_src = (which ? delta : lse) + ahl + c * 4;
      rt_dst = 2 * G::QD_BYTES + which * (BN * 4) + c * 16;
    }
#pragma unroll
    for (int i = 0; i < BI; ++i) {                              // the transposed bias tile: 128 rows (keys) x 8 chunks
      const int c = tid + 256 * i, krow = c / 8, ch = c % 8;
      b_dst[i] = 2 * G::QD_BYTES + G::ROWT_BYTES + swzb32(krow, ch);
      b_src[i] = bias_t + (plane + kt * BM + krow) * L + ch * 4;
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
__global__ void __launch_bounds__(256, G::MINB) attn_bwd_dkv_tf32_kernel(const Dkv32Params p) {
  constexpr int HD = G::HD, NST = G::NST, BN = G::BN, NT = G::NT, KS = G::KS, NO = G::NO, HG = HD / 4;
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
  const Qd32Loader<G> loader(p.q, p.dov, p.ldq, p.lddo, p.bias_t, p.lse, p.delta, rowtok, plane, ahl, kt, h, L, tid);
  for (int c = tid; c < BM * HG; c += 256) {                     // K and V of the key tile
    const int token = c / HG, g = c - token * HG;
    const long long t = rowtok + kt * BM + token;
    cp_async16(sbase + G::OFF_K + T::off(token, g), p.k + t * p.ldk + h * HD + g * 4);
    cp_async16(sbase + G::OFF_V + T::off(token, g), p.v + t * p.ldv + h * HD + g * 4);
  }
#pragma unroll
  for (int s = 0; s < NST - 1; ++s) loader.issue(sbase, s, NJ);

  float dkacc[NO][4], dvacc[NO][4];
#pragma unroll
  for (int i = 0; i < NO; ++i) dkacc[i][0] = dkacc[i][1] = dkacc[i][2] = dkacc[i][3] = dvacc[i][0] = dvacc[i][1] = dvacc[i][2] = dvacc[i][3] = 0.f;

  const int krow = warp * 16 + (lane & 7) + 8 * ((lane >> 3) & 1);       // ldmatrix row of this lane in the K / V tile (A fragments)
  const int qrow = (lane & 7) + 8 * (lane >> 4);                         // ldmatrix row of this lane in a pair of query n-tiles (B fragments)
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
    loader.issue(sbase, s + NST - 1, NJ);
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
    for (int ks = 0; ks < KS; ++ks)
#pragma unroll
      for (int pp = 0; pp < NT / 2; ++pp) {
        uint32_t qb[4], ob[4];
        ldsm_x4(qb, qbase + T::off(16 * pp + qrow, 2 * ks + ((lane >> 3) & 1)));
        ldsm_x4(ob, dobase + T::off(16 * pp + qrow, 2 * ks + ((lane >> 3) & 1)));
        mma1688_tf32(sacc[2 * pp], kf[ks], qb[0], qb[1]);
        mma1688_tf32(sacc[2 * pp + 1], kf[ks], qb[2], qb[3]);
        mma1688_tf32(dpacc[2 * pp], vf[ks], ob[0], ob[1]);
        mma1688_tf32(dpacc[2 * pp + 1], vf[ks], ob[2], ob[3]);
      }
    // ---- P^T in place of S^T (bias^T times bscl, the queries' lse), dS^T = P^T (dP^T - delta)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const float2 lq = *reinterpret_cast<const float2*>(smem_raw + (rowt - sbase) + (8 * nt + 2 * q4) * 4);
      const float2 dq2 = *reinterpret_cast<const float2*>(smem_raw + (rowt - sbase) + BN * 4 + (8 * nt + 2 * q4) * 4);
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const float2 bv = *reinterpret_cast<const float2*>(smem_raw + (btile - sbase) + swzb32(warp * 16 + g8 + 8 * hh, 2 * nt + (q4 >> 1)) + (q4 & 1) * 8);
        sacc[nt][2 * hh] = ex2f(fmaf(sacc[nt][2 * hh], p.scl, bv.x * p.bscl) - lq.x);
        sacc[nt][2 * hh + 1] = ex2f(fmaf(sacc[nt][2 * hh + 1], p.scl, bv.y * p.bscl) - lq.y);
        dpacc[nt][2 * hh] = sacc[nt][2 * hh] * (dpacc[nt][2 * hh] - dq2.x);          // dS^T (P^T stays in sacc for dv)
        dpacc[nt][2 * hh + 1] = sacc[nt][2 * hh + 1] * (dpacc[nt][2 * hh + 1] - dq2.y);
      }
    }
    // ---- dv += P^T dO, dk += dS^T Q   (A = the accumulators with the permuted query index; B = two scalar loads of dO / Q)
#pragma unroll
    for (int c = 0; c < NT; ++c) {
      const uint32_t pa[4] = {__float_as_uint(sacc[c][0]), __float_as_uint(sacc[c][2]), __float_as_uint(sacc[c][1]), __float_as_uint(sacc[c][3])};
      const uint32_t da[4] = {__float_as_uint(dpacc[c][0]), __float_as_uint(dpacc[c][2]), __float_as_uint(dpacc[c][1]), __float_as_uint(dpacc[c][3])};
#pragma unroll
      for (int np = 0; np < NO; ++np) {
        const uint32_t ao = dobase + (8 * c + 2 * q4) * T::PITCH + (8 * np + g8) * 4, aq = qbase + (8 * c + 2 * q4) * T::PITCH + (8 * np + g8) * 4;
        mma1688_tf32(dvacc[np], pa, lds32(ao), lds32(ao + T::PITCH));
        mma1688_tf32(dkacc[np], da, lds32(aq), lds32(aq + T::PITCH));
      }
    }
  }

  // ---- epilogue: dk = sm_scale sum dS^T Q, dv
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    const long long tok = rowtok + kt * BM + warp * 16 + g8 + 8 * hh;
    const bool key_ok = !G::KM || __ldg(p.kpen + (long long)a * p.kps + kt * BM + warp * 16 + g8 + 8 * hh) == 0.f;
#pragma unroll
    for (int nt = 0; nt < NO; ++nt) {
      const float k0 = key_ok ? dkacc[nt][2 * hh] * p.sm_scale : 0.f, k1 = key_ok ? dkacc[nt][2 * hh + 1] * p.sm_scale : 0.f;
      const float v0 = key_ok ? dvacc[nt][2 * hh] : 0.f, v1 = key_ok ? dvacc[nt][2 * hh + 1] : 0.f;
      *reinterpret_cast<float2*>(p.dk + tok * p.lddk + h * HD + 8 * nt + 2 * q4) = make_float2(k0, k1);
      *reinterpret_cast<float2*>(p.dv + tok * p.lddv + h * HD + 8 * nt + 2 * q4) = make_float2(v0, v1);
    }
  }
}

// bias_t[h][k][j] = bias[h][j][k] for fp32 planes: 32 x 32 tiles through shared memory (L a multiple of 32).
__global__ void __launch_bounds__(256) bias_transpose32_kernel(const float* __restrict__ bias, float* __restrict__ bias_t, int L) {
  __shared__ float tile[32][33];
  const long long base = (long long)blockIdx.z * L * L;
  const int j0 = blockIdx.y * 32, k0 = blockIdx.x * 32, tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
#pragma unroll
  for (int i = ty; i < 32; i += 8) tile[i][tx] = bias[base + (long long)(j0 + i) * L + k0 + tx];
  __syncthreads();
#pragma unroll
  for (int i = ty; i < 32; i += 8) bias_t[base + (long long)(k0 + i) * L + j0 + tx] = tile[tx][i];
}

// delta[a][h][i] = sum_d dO o  (fp32 dO and o)
template <int HD>
__global__ void __launch_bounds__(128) attn_delta32_kernel(const float* __restrict__ dO, const float* __restrict__ O, float* __restrict__ delta, long long lddo, long long ldo, int L, int H,
                                                           long long tokens) {
  const long long t = (long long)blockIdx.x * 128 + threadIdx.x;
  const int h = blockIdx.y;
  if (t >= tokens) return;
  const float4* d = reinterpret_cast<const float4*>(dO + t * lddo + h * HD);
  const float4* o = reinterpret_cast<const float4*>(O + t * ldo + h * HD);
  float s = 0.f;
#pragma unroll
  for (int c = 0; c < HD / 4; ++c) {
    const float4 a = __ldg(d + c), b = __ldg(o + c);
    s = fmaf(a.x, b.x, fmaf(a.y, b.y, fmaf(a.z, b.z, fmaf(a.w, b.w, s))));
  }
  const long long a = t / L, i = t % L;
  delta[(a * H + h) * L + i] = s;
}

}  // namespace aa80
