// attn_bwd_sm80.cuh -- the sliding-window attention backward of the SWA atom DiT block, A100 / sm_80 (mma.sync, ldmatrix, cp.async): the twins of the Triton ``_swa_attn_bwd_dq_kernel`` and
// ``_swa_attn_bwd_dkv_kernel`` (same statement, padding rows included), laid out like the forward kernel (``attn_fwd_sm80.cuh``: a CTA = 8 warps x 16 rows = 128 rows of one (n, h) with the 256-row span of
// the other operand in shared memory, 80-byte rows).  For a query i and a key j with |i - j| <= HW, i and j < seqused:
//
//   p_ij = exp(scale q_i . k_j - lse_i),  dp_ij = dO_i . v_j,  ds_ij = rn(p_ij (dp_ij - D_i))    (D_i = rowsum(dO_i o_i), computed by the out-projection backward)
//   dQ_i = scale sum_j ds_ij k_j,   dK_j = scale sum_i ds_ij q_i,   dV_j = sum_i rn(p_ij) dO_i
//
//   attn_bwd_dq_kernel    queries on the rows (K | V span in shared memory): S = Q K^T, P, dP = dO V^T, dS, dQ = dS K
//   attn_bwd_dkv_kernel   keys on the rows (Q | dO span in shared memory, with the lse / D of the span): S^T = K Q^T, P^T, dP^T = V dO^T, dS^T, dV = P^T dO, dK = dS^T Q
//
// Q, K, V, dQ, dK, dV are bf16 tensors viewed as [N][S][H][32] with any element strides (``Str3``: head-major [N][4][S][32] planes in the fused block, row-major [N][S][4][32] in the module path); dO is
// row-major [N S][128] (head h in columns 32 h ..); lse and D [N][4][S] fp32.  A warp's 16 rows see 144 columns (18 n8 tiles): its tile of
// scores sits in registers (72 fp32); the band mask (r <= t <= r + 128 for the row r and the column t of the warp's span) only cuts the first two and the last two tiles, and the bounds of the sequence
// matter only for the CTAs at its ends, so inside the sequence the mask is applied to those four tiles alone.
#pragma once
#include "attn_fwd_sm80.cuh"

namespace sw80 {

struct AttnBwdParams {
  const __nv_bfloat16 *q, *k, *v;                    // viewed as [N][S][4][32]
  const __nv_bfloat16* d_o;                          // [N S][128]
  const float* lse;                                  // [N][4][S]
  const float* dvv;                                  // [N][4][S]  D
  const int* seqused;                                // [N]
  __nv_bfloat16 *dq, *dk, *dvo;                      // viewed as [N][S][4][32]
  int S;
  float scale, scl2;                                 // 32^-0.5, scale log2 e
  Str3 qs, ks, vs, dqs, dks, dvs;                    // the strides of q, k, v and of dq, dk, dv
};

// D = rowsum(dO o) per row and head, for a caller whose out-projection backward does not make it (the module path): dO and o [M][128] bf16, D [N][4][S] fp32 (M = N S).  A thread is one (head, row) --
// consecutive threads take consecutive rows of a head, so D is written coalesced -- and reads 2 x 64 contiguous bytes
struct AttnDeltaParams {
  const __nv_bfloat16* d_o;
  const __nv_bfloat16* o;
  float* dvv;
  int M, S;
};

__global__ void __launch_bounds__(256) attn_delta_kernel(const AttnDeltaParams p) {
  const long long t = (long long)blockIdx.x * 256 + threadIdx.x;
  if (t >= (long long)AF_H * p.M) return;
  const int h = (int)(t / p.M), row = (int)(t - (long long)h * p.M);
  const size_t off = (size_t)row * AF_C + h * AF_HD;
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint4 a = ldg128(p.d_o + off + 8 * i), b = ldg128(p.o + off + 8 * i);
    acc = fmaf(bf16lo(a.x), bf16lo(b.x), acc); acc = fmaf(bf16hi(a.x), bf16hi(b.x), acc);
    acc = fmaf(bf16lo(a.y), bf16lo(b.y), acc); acc = fmaf(bf16hi(a.y), bf16hi(b.y), acc);
    acc = fmaf(bf16lo(a.z), bf16lo(b.z), acc); acc = fmaf(bf16hi(a.z), bf16hi(b.z), acc);
    acc = fmaf(bf16lo(a.w), bf16lo(b.w), acc); acc = fmaf(bf16hi(a.w), bf16hi(b.w), acc);
  }
  p.dvv[((size_t)(row / p.S) * AF_H + h) * p.S + row % p.S] = acc;
}

// the band mask of the warp's tiles of scores: row r = g8 + 8 (e >> 1), column t = 8 nt + 2 q4 + (e & 1); ``inside`` (the whole window lies in the sequence) leaves the interior tiles alone
// ``lo(r)``: the first valid column of the row, ``hi(r)``: the last one (r + 128); p = 0 where invalid
template <bool ROWS_ARE_QUERIES>
DEVI bool band_valid(int r, int t, int jb, int su) {
  // rows and columns exchange their roles in the key kernel, the band is the same: r <= t <= r + 128;  the sequence bounds apply to the column index (queries / keys of the span) and the row
  return t >= r && t <= r + 2 * AF_HW && jb + t >= 0 && jb + t < su;
}

template <bool HM>
__global__ void __launch_bounds__(AF_NW * 32, 2) attn_bwd_dq_kernel(const AttnBwdParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sk = smem_u32(smem_raw), sv = sk + AF_SPAN * AF_KP;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int nh = blockIdx.y, n = nh / AF_H, h = nh % AF_H;
  const int S = p.S, su = p.seqused[n], i0 = blockIdx.x * AF_QT;
  const size_t hq = p.qs.base(n, h), hk = p.ks.base(n, h), hv = p.vs.base(n, h), hdq = p.dqs.base(n, h);
  const long long rq = row_stride<HM>(p.qs.s), rk = row_stride<HM>(p.ks.s), rv = row_stride<HM>(p.vs.s), rdq = row_stride<HM>(p.dqs.s);
  if (i0 >= su) {                                                  // every query of the CTA is padding: dQ = 0
    for (int c = tid; c < AF_QT * 4; c += AF_NW * 32)
      if (i0 + (c >> 2) < S) stg128(p.dq + hdq + (size_t)(i0 + (c >> 2)) * rdq + (c & 3) * 8, make_uint4(0u, 0u, 0u, 0u));
    return;
  }

  // ---- K | V span into shared memory (zero rows outside [0, seqused))
  for (int c = tid; c < AF_SPAN * 4; c += AF_NW * 32) {
    const int t = c >> 2, part = c & 3, j = i0 - AF_HW + t;
    const bool ok = j >= 0 && j < su;
    const size_t jj = (size_t)(ok ? j : 0);
    cp_async16(sk + t * AF_KP + part * 16, p.k + hk + jj * rk + part * 8, ok ? 16 : 0);
    cp_async16(sv + t * AF_KP + part * 16, p.v + hv + jj * rv + part * 8, ok ? 16 : 0);
  }
  cp_async_commit();

  // ---- Q and dO fragments of this warp's 16 queries (a query >= seqused reads 0), their lse and D
  const int q0 = i0 + warp * 16, r0 = q0 + g8, r1 = r0 + 8;
  uint32_t qa[2][4], da[2][4];
#pragma unroll
  for (int ks = 0; ks < 2; ++ks) {
    const int col = ks * 16 + 2 * q4;
    qa[ks][0] = r0 < su ? *reinterpret_cast<const uint32_t*>(p.q + hq + (size_t)r0 * rq + col) : 0u;
    qa[ks][1] = r1 < su ? *reinterpret_cast<const uint32_t*>(p.q + hq + (size_t)r1 * rq + col) : 0u;
    qa[ks][2] = r0 < su ? *reinterpret_cast<const uint32_t*>(p.q + hq + (size_t)r0 * rq + col + 8) : 0u;
    qa[ks][3] = r1 < su ? *reinterpret_cast<const uint32_t*>(p.q + hq + (size_t)r1 * rq + col + 8) : 0u;
    const __nv_bfloat16* d0 = p.d_o + ((size_t)n * S + r0) * AF_C + h * AF_HD + col;
    const __nv_bfloat16* d1 = p.d_o + ((size_t)n * S + r1) * AF_C + h * AF_HD + col;
    da[ks][0] = r0 < su ? *reinterpret_cast<const uint32_t*>(d0) : 0u;
    da[ks][1] = r1 < su ? *reinterpret_cast<const uint32_t*>(d1) : 0u;
    da[ks][2] = r0 < su ? *reinterpret_cast<const uint32_t*>(d0 + 8) : 0u;
    da[ks][3] = r1 < su ? *reinterpret_cast<const uint32_t*>(d1 + 8) : 0u;
  }
  const float lse0 = r0 < su ? p.lse[(size_t)nh * S + r0] : 0.f, lse1 = r1 < su ? p.lse[(size_t)nh * S + r1] : 0.f;
  const float dd0 = r0 < su ? p.dvv[(size_t)nh * S + r0] : 0.f, dd1 = r1 < su ? p.dvv[(size_t)nh * S + r1] : 0.f;
  cp_async_wait<0>();
  __syncthreads();

  // ---- S = Q K^T over this warp's 144 keys, P = exp(scale S - lse)
  float s[AF_NKT][4];
#pragma unroll
  for (int nt = 0; nt < AF_NKT; ++nt) { s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f; }
  const uint32_t kbase = sk + (16 * warp) * AF_KP;
#pragma unroll
  for (int nt = 0; nt < AF_NKT; ++nt) {
    uint32_t b[4];
    ldsm_x4(b, kbase + (8 * nt + (lane & 7)) * AF_KP + (lane >> 3) * 16);
    mma16816(s[nt], qa[0], b[0], b[1]);
    mma16816(s[nt], qa[1], b[2], b[3]);
  }
  const int jb = q0 - AF_HW;
  const bool inside = jb >= 0 && jb + 16 + 2 * AF_HW <= su && q0 + 16 <= su;                 // every key of the window and every query of the warp is inside the sequence
  const float l0 = lse0 * 1.4426950408889634f, l1 = lse1 * 1.4426950408889634f;
#pragma unroll
  for (int nt = 0; nt < AF_NKT; ++nt) {
    const bool edge = nt < 2 || nt >= AF_NKT - 2;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float v = ex2f(fmaf(s[nt][e], p.scl2, -(e < 2 ? l0 : l1)));
      if (inside && !edge) {
        s[nt][e] = v;
      } else {
        const int r = g8 + ((e >> 1) << 3), t = 8 * nt + 2 * q4 + (e & 1);
        const bool qok = (e < 2 ? r0 : r1) < su;
        s[nt][e] = (qok && band_valid<true>(r, t, jb, su)) ? v : 0.f;
      }
    }
  }

  // ---- dQ = dS K: per pair of key tiles dP = dO V^T, dS = rn(P (dP - D)), then the product with the 16 keys' K rows
  float dq[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) { dq[i][0] = dq[i][1] = dq[i][2] = dq[i][3] = 0.f; }
  const uint32_t vbase = sv + (16 * warp) * AF_KP;
#pragma unroll
  for (int kb = 0; kb < AF_NKT / 2; ++kb) {
    float dp[2][4];
#pragma unroll
    for (int nn = 0; nn < 2; ++nn) {
      dp[nn][0] = dp[nn][1] = dp[nn][2] = dp[nn][3] = 0.f;
      uint32_t b[4];
      ldsm_x4(b, vbase + (8 * (2 * kb + nn) + (lane & 7)) * AF_KP + (lane >> 3) * 16);
      mma16816(dp[nn], da[0], b[0], b[1]);
      mma16816(dp[nn], da[1], b[2], b[3]);
    }
    const uint32_t a[4] = {pack_bf16(s[2 * kb][0] * (dp[0][0] - dd0), s[2 * kb][1] * (dp[0][1] - dd0)), pack_bf16(s[2 * kb][2] * (dp[0][2] - dd1), s[2 * kb][3] * (dp[0][3] - dd1)),
                           pack_bf16(s[2 * kb + 1][0] * (dp[1][0] - dd0), s[2 * kb + 1][1] * (dp[1][1] - dd0)), pack_bf16(s[2 * kb + 1][2] * (dp[1][2] - dd1), s[2 * kb + 1][3] * (dp[1][3] - dd1))};
#pragma unroll
    for (int nn = 0; nn < 2; ++nn) {
      uint32_t b[4];
      const int m = lane >> 3;
      ldsm_x4_t(b, kbase + (16 * kb + ((m & 1) << 3) + (lane & 7)) * AF_KP + (nn * 16 + (m >> 1) * 8) * 2);
      mma16816(dq[2 * nn], a, b[0], b[1]);
      mma16816(dq[2 * nn + 1], a, b[2], b[3]);
    }
  }

  // ---- dQ = scale dS K
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) {
    const int col = 8 * nt + 2 * q4;
    if (r0 < S) stg32(p.dq + hdq + (size_t)r0 * rdq + col, pack_bf16(dq[nt][0] * p.scale, dq[nt][1] * p.scale));
    if (r1 < S) stg32(p.dq + hdq + (size_t)r1 * rdq + col, pack_bf16(dq[nt][2] * p.scale, dq[nt][3] * p.scale));
  }
}

// Rows = keys, columns = the queries of a span: the span (Q | dO rows and their lse / D) of the CTA's 128 keys is [j0 - HW, j0 + 128 + HW)
constexpr int AB_SMEM = 2 * AF_SPAN * AF_KP + 2 * AF_SPAN * 4;

template <bool HM>
__global__ void __launch_bounds__(AF_NW * 32, 2) attn_bwd_dkv_kernel(const AttnBwdParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sq = smem_u32(smem_raw), sd = sq + AF_SPAN * AF_KP;
  float* lse_s = reinterpret_cast<float*>(smem_raw + 2 * AF_SPAN * AF_KP);
  float* dd_s = lse_s + AF_SPAN;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int nh = blockIdx.y, n = nh / AF_H, h = nh % AF_H;
  const int S = p.S, su = p.seqused[n], j0 = blockIdx.x * AF_QT;
  const size_t hq = p.qs.base(n, h), hk = p.ks.base(n, h), hv = p.vs.base(n, h), hdk = p.dks.base(n, h), hdv = p.dvs.base(n, h);
  const long long rq = row_stride<HM>(p.qs.s), rk = row_stride<HM>(p.ks.s), rv = row_stride<HM>(p.vs.s), rdk = row_stride<HM>(p.dks.s), rdv = row_stride<HM>(p.dvs.s);
  if (j0 >= su) {                                                  // every key of the CTA is padding: dK = dV = 0
    for (int c = tid; c < AF_QT * 4; c += AF_NW * 32)
      if (j0 + (c >> 2) < S) {
        stg128(p.dk + hdk + (size_t)(j0 + (c >> 2)) * rdk + (c & 3) * 8, make_uint4(0u, 0u, 0u, 0u));
        stg128(p.dvo + hdv + (size_t)(j0 + (c >> 2)) * rdv + (c & 3) * 8, make_uint4(0u, 0u, 0u, 0u));
      }
    return;
  }

  // ---- Q | dO span into shared memory (zero rows outside [0, seqused)), with lse and D (0 there)
  for (int c = tid; c < AF_SPAN * 4; c += AF_NW * 32) {
    const int t = c >> 2, part = c & 3, i = j0 - AF_HW + t;
    const bool ok = i >= 0 && i < su;
    cp_async16(sq + t * AF_KP + part * 16, p.q + hq + (size_t)(ok ? i : 0) * rq + part * 8, ok ? 16 : 0);
    cp_async16(sd + t * AF_KP + part * 16, p.d_o + ((size_t)n * S + (ok ? i : 0)) * AF_C + h * AF_HD + part * 8, ok ? 16 : 0);
  }
  cp_async_commit();
  for (int t = tid; t < AF_SPAN; t += AF_NW * 32) {
    const int i = j0 - AF_HW + t;
    const bool ok = i >= 0 && i < su;
    lse_s[t] = ok ? p.lse[(size_t)nh * S + i] : 0.f;
    dd_s[t] = ok ? p.dvv[(size_t)nh * S + i] : 0.f;
  }

  // ---- K and V fragments of this warp's 16 keys (a key >= seqused reads 0)
  const int k0 = j0 + warp * 16, r0 = k0 + g8, r1 = r0 + 8;
  uint32_t ka[2][4], va[2][4];
#pragma unroll
  for (int ks = 0; ks < 2; ++ks) {
    const int col = ks * 16 + 2 * q4;
    ka[ks][0] = r0 < su ? *reinterpret_cast<const uint32_t*>(p.k + hk + (size_t)r0 * rk + col) : 0u;
    ka[ks][1] = r1 < su ? *reinterpret_cast<const uint32_t*>(p.k + hk + (size_t)r1 * rk + col) : 0u;
    ka[ks][2] = r0 < su ? *reinterpret_cast<const uint32_t*>(p.k + hk + (size_t)r0 * rk + col + 8) : 0u;
    ka[ks][3] = r1 < su ? *reinterpret_cast<const uint32_t*>(p.k + hk + (size_t)r1 * rk + col + 8) : 0u;
    va[ks][0] = r0 < su ? *reinterpret_cast<const uint32_t*>(p.v + hv + (size_t)r0 * rv + col) : 0u;
    va[ks][1] = r1 < su ? *reinterpret_cast<const uint32_t*>(p.v + hv + (size_t)r1 * rv + col) : 0u;
    va[ks][2] = r0 < su ? *reinterpret_cast<const uint32_t*>(p.v + hv + (size_t)r0 * rv + col + 8) : 0u;
    va[ks][3] = r1 < su ? *reinterpret_cast<const uint32_t*>(p.v + hv + (size_t)r1 * rv + col + 8) : 0u;
  }
  cp_async_wait<0>();
  __syncthreads();

  // ---- S^T = K Q^T over the 144 queries of this warp's window (query t of the window is row 16 warp + t of the span)
  float s[AF_NKT][4];
#pragma unroll
  for (int nt = 0; nt < AF_NKT; ++nt) { s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f; }
  const uint32_t qbase = sq + (16 * warp) * AF_KP, dbase = sd + (16 * warp) * AF_KP;
#pragma unroll
  for (int nt = 0; nt < AF_NKT; ++nt) {
    uint32_t b[4];
    ldsm_x4(b, qbase + (8 * nt + (lane & 7)) * AF_KP + (lane >> 3) * 16);
    mma16816(s[nt], ka[0], b[0], b[1]);
    mma16816(s[nt], ka[1], b[2], b[3]);
  }
  // P^T = exp(scale S^T - lse_q) (the lse of the thread's columns t = 8 nt + 2 q4 + {0, 1}), zero outside the band and the sequence
  const int ib = k0 - AF_HW;                                       // the sequence index of window query 0
  const bool inside = ib >= 0 && ib + 16 + 2 * AF_HW <= su && k0 + 16 <= su;
  const float* lw = lse_s + 16 * warp;
  const float* dw = dd_s + 16 * warp;
#pragma unroll
  for (int nt = 0; nt < AF_NKT; ++nt) {
    const bool edge = nt < 2 || nt >= AF_NKT - 2;
    const float2 lq = *reinterpret_cast<const float2*>(lw + 8 * nt + 2 * q4);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float v = ex2f(fmaf(s[nt][e], p.scl2, -(e & 1 ? lq.y : lq.x) * 1.4426950408889634f));
      if (inside && !edge) {
        s[nt][e] = v;
      } else {
        const int r = g8 + ((e >> 1) << 3), t = 8 * nt + 2 * q4 + (e & 1);
        s[nt][e] = ((k0 + r) < su && band_valid<false>(r, t, ib, su)) ? v : 0.f;
      }
    }
  }

  // ---- dV = P^T dO and dK = dS^T Q: per pair of query tiles, dP^T = V dO^T, dS^T = rn(P^T (dP^T - D_q))
  float dv[4][4], dk[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) { dv[i][0] = dv[i][1] = dv[i][2] = dv[i][3] = 0.f; dk[i][0] = dk[i][1] = dk[i][2] = dk[i][3] = 0.f; }
#pragma unroll
  for (int kb = 0; kb < AF_NKT / 2; ++kb) {
    float dp[2][4];
#pragma unroll
    for (int nn = 0; nn < 2; ++nn) {
      dp[nn][0] = dp[nn][1] = dp[nn][2] = dp[nn][3] = 0.f;
      uint32_t b[4];
      ldsm_x4(b, dbase + (8 * (2 * kb + nn) + (lane & 7)) * AF_KP + (lane >> 3) * 16);
      mma16816(dp[nn], va[0], b[0], b[1]);
      mma16816(dp[nn], va[1], b[2], b[3]);
    }
    const float2 d0 = *reinterpret_cast<const float2*>(dw + 16 * kb + 2 * q4), d1 = *reinterpret_cast<const float2*>(dw + 16 * kb + 8 + 2 * q4);
    const uint32_t pa[4] = {pack_bf16(s[2 * kb][0], s[2 * kb][1]), pack_bf16(s[2 * kb][2], s[2 * kb][3]), pack_bf16(s[2 * kb + 1][0], s[2 * kb + 1][1]), pack_bf16(s[2 * kb + 1][2], s[2 * kb + 1][3])};
    const uint32_t sa[4] = {pack_bf16(s[2 * kb][0] * (dp[0][0] - d0.x), s[2 * kb][1] * (dp[0][1] - d0.y)), pack_bf16(s[2 * kb][2] * (dp[0][2] - d0.x), s[2 * kb][3] * (dp[0][3] - d0.y)),
                           pack_bf16(s[2 * kb + 1][0] * (dp[1][0] - d1.x), s[2 * kb + 1][1] * (dp[1][1] - d1.y)), pack_bf16(s[2 * kb + 1][2] * (dp[1][2] - d1.x), s[2 * kb + 1][3] * (dp[1][3] - d1.y))};
#pragma unroll
    for (int nn = 0; nn < 2; ++nn) {
      uint32_t b[4], c[4];
      const int m = lane >> 3;
      const uint32_t off = (16 * kb + ((m & 1) << 3) + (lane & 7)) * AF_KP + (nn * 16 + (m >> 1) * 8) * 2;
      ldsm_x4_t(b, dbase + off);
      ldsm_x4_t(c, qbase + off);
      mma16816(dv[2 * nn], pa, b[0], b[1]);
      mma16816(dv[2 * nn + 1], pa, b[2], b[3]);
      mma16816(dk[2 * nn], sa, c[0], c[1]);
      mma16816(dk[2 * nn + 1], sa, c[2], c[3]);
    }
  }

  // ---- dK = scale dS^T Q and dV (rows >= seqused hold zeros)
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) {
    const int col = 8 * nt + 2 * q4;
    if (r0 < S) { stg32(p.dk + hdk + (size_t)r0 * rdk + col, pack_bf16(dk[nt][0] * p.scale, dk[nt][1] * p.scale)); stg32(p.dvo + hdv + (size_t)r0 * rdv + col, pack_bf16(dv[nt][0], dv[nt][1])); }
    if (r1 < S) { stg32(p.dk + hdk + (size_t)r1 * rdk + col, pack_bf16(dk[nt][2] * p.scale, dk[nt][3] * p.scale)); stg32(p.dvo + hdv + (size_t)r1 * rdv + col, pack_bf16(dv[nt][2], dv[nt][3])); }
  }
}

}  // namespace sw80
