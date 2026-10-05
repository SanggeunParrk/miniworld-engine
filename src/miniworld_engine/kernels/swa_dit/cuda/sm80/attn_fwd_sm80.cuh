// attn_fwd_sm80.cuh -- the sliding-window attention forward of the SWA atom DiT block, A100 / sm_80 (mma.sync, ldmatrix, cp.async).
//
//   o[i] = softmax_j( scale q_i . k_j ) v_j       over |i - j| <= HW and j < seqused, per (sample n, head h)
//
// The Triton ``_swa_attn_fwd_kernel``'s statement, padding rows included: a query >= seqused reads q = 0 (so its lse is log(number of valid keys in its
// window), or 0 when there are none) and writes o = 0.  Q / K / V are bf16 tensors viewed as [N][S][H][32] with any element strides (``Str3``: the fused block's head-major
// [N][H][S][32] planes, the module's row-major [N][S][H][32], a slice of the fused qkv projection); O is row-major [N S][C] (head h in columns 32 h ..), bf16; lse [N][H][S] fp32
// (natural log of the softmax denominator of the scaled scores).
//
// One CTA = 8 warps x 16 queries = 128 queries of one (n, h).  Its keys are the span [i0 - HW, i0 + 128 + HW) = 256 rows, loaded once through cp.async
// (zero-filled outside [0, seqused)); a warp's 16 queries need 144 of them, so its scores are 18 n8 tiles in registers (72 fp32) and the softmax is a single
// pass: the row maximum over the whole window, no running rescale.  Only the first and last 16 keys of a warp's window cross the band's edge, but the mask
// (r <= t <= r + 128 for the query row r and the key t of the warp's span, and 0 <= j < seqused) is applied to every score: it is a few integer
// compares.  P goes to bf16 for the PV products, the denominator is the fp32 sum of the unrounded P (as the Triton kernel).
#pragma once
#include "sm80_common.cuh"

namespace sw80 {

constexpr int AF_HD = 32, AF_H = 4, AF_C = 128, AF_HW = 64;
constexpr int AF_NW = 8, AF_QT = AF_NW * 16, AF_SPAN = AF_QT + 2 * AF_HW, AF_KP = 80;     // 80-byte rows (5 granules): ldmatrix is conflict free
constexpr int AF_NKT = (16 + 2 * AF_HW) / 8;                                              // n8 key tiles of a warp's 144-key window
constexpr int AF_SMEM = 2 * AF_SPAN * AF_KP;

// element strides of a tensor viewed as [N][S][H][32]: sample n, row s, head h (the 32 channels of a head are contiguous)
struct Str3 {
  long long n, s, h;
  DEVI size_t base(int sample, int head) const { return (size_t)sample * n + (size_t)head * h; }
};

struct AttnFwdParams {
  const __nv_bfloat16* q;
  const __nv_bfloat16* k;
  const __nv_bfloat16* v;
  const int* seqused;           // [N]
  __nv_bfloat16* o;             // [N S][128]
  float* lse;                   // [N][H][S]
  int S;
  float scale, scl2;            // 32^-0.5, scale log2 e
  Str3 qs, ks, vs;              // the strides of q, k, v
};

// the row stride of a tensor in the kernels: HM (the fused block's head-major [N][H][S][32] planes) makes it the compile-time 32, so the address arithmetic of the DiT's kernels is the one it always was
template <bool HM>
DEVI long long row_stride(long long runtime) { return HM ? (long long)AF_HD : runtime; }

template <bool HM>
__global__ void __launch_bounds__(AF_NW * 32, 2) attn_fwd_kernel(const AttnFwdParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sk = smem_u32(smem_raw), sv = sk + AF_SPAN * AF_KP;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int nh = blockIdx.y, n = nh / AF_H, h = nh % AF_H;
  const int S = p.S, su = p.seqused[n], i0 = blockIdx.x * AF_QT;
  const size_t hq = p.qs.base(n, h), hk = p.ks.base(n, h), hv = p.vs.base(n, h);   // element offsets of this (n, h) plane of q, k, v
  const long long rq = row_stride<HM>(p.qs.s), rk = row_stride<HM>(p.ks.s), rv = row_stride<HM>(p.vs.s);

  // ---- K | V span into shared memory (zero rows outside [0, seqused): the products read them, so they must be finite)
  for (int c = tid; c < AF_SPAN * 4; c += AF_NW * 32) {
    const int t = c >> 2, part = c & 3, j = i0 - AF_HW + t;
    const bool ok = j >= 0 && j < su;
    const size_t jj = (size_t)(ok ? j : 0);
    cp_async16(sk + t * AF_KP + part * 16, p.k + hk + jj * rk + part * 8, ok ? 16 : 0);
    cp_async16(sv + t * AF_KP + part * 16, p.v + hv + jj * rv + part * 8, ok ? 16 : 0);
  }
  cp_async_commit();

  // ---- Q fragments of this warp's 16 queries (a query >= seqused reads 0)
  const int q0 = i0 + warp * 16, r0 = q0 + g8, r1 = r0 + 8;
  uint32_t qa[2][4];
#pragma unroll
  for (int ks = 0; ks < 2; ++ks) {
    const int col = ks * 16 + 2 * q4;
    qa[ks][0] = r0 < su ? *reinterpret_cast<const uint32_t*>(p.q + hq + (size_t)r0 * rq + col) : 0u;
    qa[ks][1] = r1 < su ? *reinterpret_cast<const uint32_t*>(p.q + hq + (size_t)r1 * rq + col) : 0u;
    qa[ks][2] = r0 < su ? *reinterpret_cast<const uint32_t*>(p.q + hq + (size_t)r0 * rq + col + 8) : 0u;
    qa[ks][3] = r1 < su ? *reinterpret_cast<const uint32_t*>(p.q + hq + (size_t)r1 * rq + col + 8) : 0u;
  }
  cp_async_wait<0>();
  __syncthreads();

  // ---- S = Q K^T over this warp's 144 keys: key t of the window is row 16 warp + t of the span
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

  // ---- mask and row maxima: row r (g8 or g8 + 8), key t = 8 nt + 2 q4 + {0, 1}; valid iff r <= t <= r + 128 and 0 <= q0 - HW + t < seqused
  const int jb = q0 - AF_HW;
  // the band (r <= t <= r + 128) only cuts the first two and the last two n8 tiles of the 18, and the sequence bounds only the warps at its ends: inside the sequence the mask is applied to those four tiles
  const bool inside = jb >= 0 && jb + 16 + 2 * AF_HW <= su;
  float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
  for (int nt = 0; nt < AF_NKT; ++nt) {
    const bool edge = nt < 2 || nt >= AF_NKT - 2;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      if (!inside || edge) {
        const int r = g8 + ((e >> 1) << 3), t = 8 * nt + 2 * q4 + (e & 1), j = jb + t;
        if (!(t >= r && t <= r + 2 * AF_HW && j >= 0 && j < su)) s[nt][e] = -INFINITY;
      }
      if (e < 2) mx0 = fmaxf(mx0, s[nt][e]); else mx1 = fmaxf(mx1, s[nt][e]);
    }
  }
  mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
  mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
  const float m0 = mx0 == -INFINITY ? 0.f : mx0, m1 = mx1 == -INFINITY ? 0.f : mx1;

  // ---- P = exp(scale (s - max)), the fp32 row sums
  float l0 = 0.f, l1 = 0.f;
#pragma unroll
  for (int nt = 0; nt < AF_NKT; ++nt) {
    s[nt][0] = ex2f((s[nt][0] - m0) * p.scl2); s[nt][1] = ex2f((s[nt][1] - m0) * p.scl2);
    s[nt][2] = ex2f((s[nt][2] - m1) * p.scl2); s[nt][3] = ex2f((s[nt][3] - m1) * p.scl2);
    l0 += s[nt][0] + s[nt][1];
    l1 += s[nt][2] + s[nt][3];
  }
  l0 = quad_sum(l0);
  l1 = quad_sum(l1);

  // ---- O = P V: 9 k16 steps of keys x 4 n8 tiles of the head dimension (the n16 pair of each ldmatrix.trans)
  float o[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) { o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f; }
  const uint32_t vbase = sv + (16 * warp) * AF_KP;
#pragma unroll
  for (int kb = 0; kb < AF_NKT / 2; ++kb) {
    const uint32_t a[4] = {pack_bf16(s[2 * kb][0], s[2 * kb][1]), pack_bf16(s[2 * kb][2], s[2 * kb][3]),
                           pack_bf16(s[2 * kb + 1][0], s[2 * kb + 1][1]), pack_bf16(s[2 * kb + 1][2], s[2 * kb + 1][3])};
#pragma unroll
    for (int nn = 0; nn < 2; ++nn) {
      uint32_t b[4];
      const int m = lane >> 3;
      ldsm_x4_t(b, vbase + (16 * kb + ((m & 1) << 3) + (lane & 7)) * AF_KP + (nn * 16 + (m >> 1) * 8) * 2);
      mma16816(o[2 * nn], a, b[0], b[1]);
      mma16816(o[2 * nn + 1], a, b[2], b[3]);
    }
  }

  // ---- epilogue: o / l (0 for a query >= seqused), lse
  const float il0 = l0 > 0.f ? 1.f / l0 : 1.f, il1 = l1 > 0.f ? 1.f / l1 : 1.f;
  __nv_bfloat16* orow0 = p.o + ((size_t)n * S + r0) * AF_C + h * AF_HD;
  __nv_bfloat16* orow1 = p.o + ((size_t)n * S + r1) * AF_C + h * AF_HD;
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) {
    const int col = 8 * nt + 2 * q4;
    if (r0 < S) stg32(orow0 + col, r0 < su ? pack_bf16(o[nt][0] * il0, o[nt][1] * il0) : 0u);
    if (r1 < S) stg32(orow1 + col, r1 < su ? pack_bf16(o[nt][2] * il1, o[nt][3] * il1) : 0u);
  }
  if (q4 == 0) {
    if (r0 < S) p.lse[(size_t)nh * S + r0] = l0 > 0.f ? mx0 * p.scale + logf(l0) : 0.f;
    if (r1 < S) p.lse[(size_t)nh * S + r1] = l1 > 0.f ? mx1 * p.scale + logf(l1) : 0.f;
  }
}

}  // namespace sw80
