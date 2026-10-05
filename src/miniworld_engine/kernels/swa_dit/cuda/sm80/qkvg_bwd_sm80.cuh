// qkvg_bwd_sm80.cuh -- the backward of the first stage of the SWA atom DiT block (adaLN-modulated RMSNorm, the projections q | k | v | g, per-head RMS norm and RoPE of q and k), A100 / sm_80:
// the twin of the Triton ``_swa_qkvg_bwd_kernel`` with its rounding points.  For a row (rn = round to bf16, all else fp32):
//
//   q, k:   dP_q = rn(head_rms_bwd(p_q, unrope(dQ)))      p_q = the saved rounded pre-norm projection, head_rms_bwd(p, d) = r (d - yh mean_D(d yh)), yh = p r, r = 1 / sqrt(mean_D(p^2) + qk_eps)
//   v, g:   dP_v = dV, dP_g = dG (already bf16)
//   dx = dP W (W = Wqkv | Wg stacked, K = 512)                                       -> dP is stored too (the dW operand of the weight-gradient GEMMs)
//   xh = q rstd (q: the block input, rstd = 1 / sqrt(mean(q^2) + eps)):  d scale_a = dx xh, d shift_a = dx                -> dmod columns 128 .. 255 | 0 .. 127
//   dq = rn(dq1 + rstd (dxh - xh mean(dxh xh))),  dxh = dx (1 + scale_a)
//
// The structure of ``ffn_bwd_dy_kernel``: Wt = (Wqkv | Wg)^T [128 input channels][512 outputs] stays in shared memory in the f1 row order, one warp per 16-row tile (tiles dealt round-robin, see
// ``bwd_rows_sm80.cuh`` for the two tilings), K in 16 blocks of 32 (q | k | v | g, 4 heads each) whose A fragments are produced on the registers: a thread holds the 8 channels 8 q4 .. 8 q4 + 7 of a head,
// so the head norm is a quad reduction, the RoPE partner d +- 16 is the lane two over, and its 16-byte loads of the head-major dQ / dK / dV are contiguous.
#pragma once
#include "adaln_bwd_sm80.cuh"
#include "ffn_bwd_sm80.cuh"      // wut_off, qf_channel, rn

namespace sw80 {

struct QkvgBwdParams {
  const __nv_bfloat16* qi;                           // [M][128] the block input q
  const __nv_bfloat16 *pq, *pk;                      // [M][128] the saved rounded pre-norm projections
  const __nv_bfloat16 *dqh, *dkh, *dvh;              // [N][4][S][32] the gradients of the head-major Q / K / V
  const __nv_bfloat16* dg;                           // [M][128] the gradient of G
  const __nv_bfloat16* dq1;                          // [M][128] the gradient of q1
  const float* mod;                                  // [B S][768]
  const float *cos, *sin;                            // [B S][16]
  const __nv_bfloat16* wt;                           // [128][512] = (Wqkv | Wg)^T
  __nv_bfloat16 *dq, *dp;                            // [M][128], [M][512]
  float* dmod;                                       // MODE_SINGLE: [B S][768]; MODE_HOIST: [nblk][B S][768].  Columns 0 .. 255 are d shift_a | d scale_a
  int M, S, B;
  float eps, qk_eps;
};

template <int NW_>
struct QkvgBwdCfg {
  static constexpr int NW = NW_, NTHR = NW_ * 32, SMEM = 128 * 1024;
};

template <class G, int MODE>
__global__ void __launch_bounds__(G::NTHR, 1) qkvg_bwd_kernel(const QkvgBwdParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sw = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  for (int i = tid; i < 128 * 64; i += G::NTHR) cp_async16(sw + wut_off(i >> 6, i & 63), p.wt + (size_t)qf_channel(i >> 6) * 512 + (i & 63) * 8);
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  const uint32_t wlane = g8 * 1024u + q4 * 16u, gpar = g8 & 1u;    // the B fragment of output block kb (32 outputs) in row g8 of a group of 8 rows is at wlane + 64 (kb ^ gpar)

  const int ntile = bwd_ntile<MODE>(p.M, p.S, p.B);
  for (int base = blockIdx.x * G::NW; base < ntile; base += G::NW * gridDim.x) {
    const int tile = base + warp;
    const RowMap rm = row_map<MODE>(tile, p.M, p.S, p.B, g8);
    float cs[2][8], sn[2][8];                                      // RoPE tables of the thread's frequencies f = 8 (q4 & 1) .. + 7
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const float* cp = p.cos + (size_t)rm.mrow[hh] * 16 + 8 * (q4 & 1);
      const float* sp = p.sin + (size_t)rm.mrow[hh] * 16 + 8 * (q4 & 1);
      const float4 c0 = rm.rok[hh] ? ldg_f4b(cp) : make_float4(1.f, 1.f, 1.f, 1.f), c1 = rm.rok[hh] ? ldg_f4b(cp + 4) : make_float4(1.f, 1.f, 1.f, 1.f);
      const float4 s0 = rm.rok[hh] ? ldg_f4b(sp) : make_float4(0.f, 0.f, 0.f, 0.f), s1 = rm.rok[hh] ? ldg_f4b(sp + 4) : make_float4(0.f, 0.f, 0.f, 0.f);
      cs[hh][0] = c0.x; cs[hh][1] = c0.y; cs[hh][2] = c0.z; cs[hh][3] = c0.w; cs[hh][4] = c1.x; cs[hh][5] = c1.y; cs[hh][6] = c1.z; cs[hh][7] = c1.w;
      sn[hh][0] = s0.x; sn[hh][1] = s0.y; sn[hh][2] = s0.z; sn[hh][3] = s0.w; sn[hh][4] = s1.x; sn[hh][5] = s1.y; sn[hh][6] = s1.z; sn[hh][7] = s1.w;
    }

    float acc[4][4][4];
#pragma unroll
    for (int gq = 0; gq < 4; ++gq)
#pragma unroll
      for (int j = 0; j < 4; ++j) { acc[gq][j][0] = acc[gq][j][1] = acc[gq][j][2] = acc[gq][j][3] = 0.f; }

    // ---- dx = dP W, one block of 32 outputs (a head of q, k, v or g) at a time: its dP (8 channels per thread and row) is produced, stored and multiplied.  The loads of a block (the gradient slice
    // and, for q and k, the saved projection) are requested one block ahead.
    uint4 cd[2], cp_[2], nd[2], np_[2], md[2], mp_[2];
    auto load_block = [&](int kbl, uint4 (&dd)[2], uint4 (&pp)[2]) {
      const int part = kbl >> 2, h = kbl & 3;
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        dd[hh] = pp[hh] = make_uint4(0u, 0u, 0u, 0u);
        if (rm.rok[hh]) {
          const size_t hm = (((size_t)rm.nrow[hh] * 4 + h) * p.S + rm.sat[hh]) * 32 + 8 * q4;     // the head-major offset of the thread's 8 channels
          if (part < 2) {
            dd[hh] = ldg128((part == 0 ? p.dqh : p.dkh) + hm);
            pp[hh] = ldg128((part == 0 ? p.pq : p.pk) + (size_t)rm.rr[hh] * 128 + 32 * h + 8 * q4);
          } else {
            dd[hh] = part == 2 ? ldg128(p.dvh + hm) : ldg128(p.dg + (size_t)rm.rr[hh] * 128 + 32 * h + 8 * q4);
          }
        }
      }
    };
    load_block(0, cd, cp_);
    load_block(1, nd, np_);
#pragma unroll 1
    for (int kb = 0; kb < 16; ++kb) {
      const int part = kb >> 2, h = kb & 3;                        // 0 q, 1 k, 2 v, 3 g
      if (kb + 2 < 16) load_block(kb + 2, md, mp_);
      uint32_t a[2][4];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        uint4 u = cd[hh];
        if (part < 2) {
          const uint4 ud = cd[hh], up = cp_[hh];
          float f[8] = {bf16lo(ud.x), bf16hi(ud.x), bf16lo(ud.y), bf16hi(ud.y), bf16lo(ud.z), bf16hi(ud.z), bf16lo(ud.w), bf16hi(ud.w)};
          float t[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) {                            // the transpose of RoPE: dx1 = dy1 c + dy2 s, dx2 = dy2 c - dy1 s (the partner d +- 16 is the lane two over)
            const float o = __shfl_xor_sync(0xffffffffu, f[i], 2);
            t[i] = q4 < 2 ? fmaf(o, sn[hh][i], f[i] * cs[hh][i]) : fmaf(-o, sn[hh][i], f[i] * cs[hh][i]);
          }
          const float pv[8] = {bf16lo(up.x), bf16hi(up.x), bf16lo(up.y), bf16hi(up.y), bf16lo(up.z), bf16hi(up.z), bf16lo(up.w), bf16hi(up.w)};
          float ssq = 0.f;
#pragma unroll
          for (int i = 0; i < 8; ++i) ssq = fmaf(pv[i], pv[i], ssq);
          const float r = 1.f / sqrtf(quad_sum(ssq) * (1.f / 32.f) + p.qk_eps);
          float dd = 0.f;
#pragma unroll
          for (int i = 0; i < 8; ++i) dd = fmaf(t[i], pv[i] * r, dd);
          dd = quad_sum(dd) * (1.f / 32.f);
          float d[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) d[i] = r * (t[i] - (pv[i] * r) * dd);
          u = make_uint4(pack_bf16(d[0], d[1]), pack_bf16(d[2], d[3]), pack_bf16(d[4], d[5]), pack_bf16(d[6], d[7]));
        }
        a[0][hh] = u.x; a[0][2 + hh] = u.y; a[1][hh] = u.z; a[1][2 + hh] = u.w;
        if (rm.rok[hh]) stg128(p.dp + (size_t)rm.rr[hh] * 512 + 128 * part + 32 * h + 8 * q4, u);
      }
#pragma unroll
      for (int nt = 0; nt < 16; ++nt) {
        const uint4 b = lds128_ro(sw + nt * 8 * 1024 + wlane + 64u * (kb ^ gpar));
        mma16816(acc[nt >> 2][nt & 3], a[0], b.x, b.y);
        mma16816(acc[nt >> 2][nt & 3], a[1], b.z, b.w);
      }
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) { cd[hh] = nd[hh]; cp_[hh] = np_[hh]; nd[hh] = md[hh]; np_[hh] = mp_[hh]; }
    }

    // ---- the adaLN backward of the input RMS norm: d shift_a -> dmod columns 0 .., d scale_a -> 128 ..
    adaln_bwd_epilogue<MODE>(acc, rm, lane, p.qi, p.dq1, p.mod, 128, p.dq, p.dmod, 0, p.B, p.S, p.eps);
  }
}

}  // namespace sw80
