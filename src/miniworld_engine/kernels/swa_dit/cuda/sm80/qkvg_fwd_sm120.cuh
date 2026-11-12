// qkvg_fwd_sm120.cuh -- the RTX 5090 (sm_120a) variant of qkvg_fwd_sm80.cuh's stage (the same statements and rounding points): the A100 kernel keeps the four
// [128][128] weights (128 KB) in shared memory, over the 5090's 99 KB a block.  Here the CTAs come in pairs: CTA 2c + h holds HALF h of the packed weight rows
// (h 0: q | k, h 1: v | g -- 64 KB) and computes those 4 output blocks of every tile it takes (the RMS norm / modulation of the tile is computed by both CTAs of a
// pair).  ``nsplit`` (1 / 2 / 4) splits a half's 4 blocks over that many warps.  Built for sm_120a only (fused_sm120.cu).
#pragma once
#include "qkvg_fwd_sm80.cuh"

namespace sw80 {

template <int NW_, int MT_>
struct QkvgHalfCfg {
  static constexpr int NW = NW_, NTHR = NW_ * 32, MT = MT_, ROWS = 16 * MT_;
  static constexpr int SMEM = (QF_ROWS / 2) * 256;
  static_assert(SMEM <= 101376, "the RTX 5090's 99 KB of dynamic shared memory per block");
};

template <class G>
__global__ void __launch_bounds__(G::NTHR, 1) qkvg_fwd_half_kernel(const QkvgFwdParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int half = blockIdx.x & 1, cta = blockIdx.x >> 1, ncta = gridDim.x >> 1;

  // ---- this CTA's half of the weights (packed rows 256 half ..) into shared memory, in the kernel's row order
  for (int i = tid; i < (QF_ROWS / 2) * 16; i += G::NTHR) {
    const int row = i >> 4, c = qf_channel(256 * half + row);
    const __nv_bfloat16* src = c < 384 ? p.wqkv + (size_t)c * 128 : p.wg + (size_t)(c - 384) * 128;
    cp_async16(sb + qf_woff(row, i & 15), src + (i & 15) * 8);
  }
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  uint32_t wb[4];
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) wb[kb] = sb + qf_woff(g8, 4 * kb + q4);

  const int ntile = (p.M + G::ROWS - 1) / G::ROWS, nsplit = p.nsplit, nitem = ntile * nsplit, bper = 4 / nsplit;
  for (int item = warp * ncta + cta; item < nitem; item += G::NW * ncta) {
    const int tile = item / nsplit, bgrp = item - tile * nsplit;
    const int t0 = tile * G::ROWS;
    const bool first = half == 0 && bgrp == 0;

    // ---- rows (mt, hh): r = t0 + 16 mt + g8 + 8 hh;  n, s, and the modulation row
    int rr[G::MT][2], mrow[G::MT][2], sidx[G::MT][2], nidx[G::MT][2];
    bool rok[G::MT][2];
#pragma unroll
    for (int mt = 0; mt < G::MT; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int r = t0 + mt * 16 + g8 + 8 * hh;
        rr[mt][hh] = r;
        rok[mt][hh] = r < p.M;
        const int n = rok[mt][hh] ? r / p.S : 0, s = rok[mt][hh] ? r - n * p.S : 0;
        nidx[mt][hh] = n;
        sidx[mt][hh] = s;
        mrow[mt][hh] = (n % p.B) * p.S + s;
      }

    // ---- RMS norm + modulation on the registers: a thread holds 32 of the 128 channels of its row, a quad the whole row
    uint32_t a[G::MT][8][4];
#pragma unroll
    for (int mt = 0; mt < G::MT; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        uint4 xr[4];
#pragma unroll
        for (int kb = 0; kb < 4; ++kb)
          xr[kb] = rok[mt][hh] ? ldg128(p.x + (size_t)rr[mt][hh] * 128 + 32 * kb + 8 * q4) : make_uint4(0u, 0u, 0u, 0u);
        float ss = 0.f;
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
          const uint4 u = xr[kb];
          float e;
          e = bf16lo(u.x); ss = fmaf(e, e, ss); e = bf16hi(u.x); ss = fmaf(e, e, ss);
          e = bf16lo(u.y); ss = fmaf(e, e, ss); e = bf16hi(u.y); ss = fmaf(e, e, ss);
          e = bf16lo(u.z); ss = fmaf(e, e, ss); e = bf16hi(u.z); ss = fmaf(e, e, ss);
          e = bf16lo(u.w); ss = fmaf(e, e, ss); e = bf16hi(u.w); ss = fmaf(e, e, ss);
        }
        const float rstd = 1.f / sqrtf(quad_sum(ss) * (1.f / 128.f) + p.eps);
        const float* mrp = p.mod + (size_t)mrow[mt][hh] * 768;
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
          const uint4 u = xr[kb];
          const float4 sh0 = *reinterpret_cast<const float4*>(mrp + 32 * kb + 8 * q4), sh1 = *reinterpret_cast<const float4*>(mrp + 32 * kb + 8 * q4 + 4);
          const float4 sc0 = *reinterpret_cast<const float4*>(mrp + 128 + 32 * kb + 8 * q4), sc1 = *reinterpret_cast<const float4*>(mrp + 128 + 32 * kb + 8 * q4 + 4);
          const float xv[8] = {bf16lo(u.x), bf16hi(u.x), bf16lo(u.y), bf16hi(u.y), bf16lo(u.z), bf16hi(u.z), bf16lo(u.w), bf16hi(u.w)};
          const float sh[8] = {sh0.x, sh0.y, sh0.z, sh0.w, sh1.x, sh1.y, sh1.z, sh1.w};
          const float sc[8] = {sc0.x, sc0.y, sc0.z, sc0.w, sc1.x, sc1.y, sc1.z, sc1.w};
          float y[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) y[i] = fmaf(xv[i] * rstd, 1.f + sc[i], sh[i]);
          const uint32_t n0 = pack_bf16(y[0], y[1]), n1 = pack_bf16(y[2], y[3]), n2 = pack_bf16(y[4], y[5]), n3 = pack_bf16(y[6], y[7]);
          a[mt][2 * kb][hh] = n0;      a[mt][2 * kb][2 + hh] = n1;
          a[mt][2 * kb + 1][hh] = n2;  a[mt][2 * kb + 1][2 + hh] = n3;
          if (p.xs != nullptr && first && rok[mt][hh])
            stg128(p.xs + (size_t)rr[mt][hh] * 128 + 32 * kb + 8 * q4, make_uint4(n0, n1, n2, n3));
        }
      }

    // ---- 8 blocks of 64 output channels: q | k | v | g, 2 heads (og) per block
#pragma unroll 1
    for (int blk = 4 * half + bgrp * bper; blk < 4 * half + (bgrp + 1) * bper; ++blk) {
      float acc[G::MT][8][4];
#pragma unroll
      for (int mt = 0; mt < G::MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) { acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f; }
      const uint32_t wblk = 64u * (blk - 4 * half) * 256u;            // the block within this CTA's half
      uint4 bn = lds128_ro(wb[0] + wblk);                        // the B fragments, one n tile ahead of the mma that uses them
#pragma unroll
      for (int kb = 0; kb < 4; ++kb)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          const uint4 bc = bn;
          if (kb * 8 + nt < 31) bn = lds128_ro(wb[(kb * 8 + nt + 1) >> 3] + wblk + ((kb * 8 + nt + 1) & 7) * 8 * 256);
#pragma unroll
          for (int mt = 0; mt < G::MT; ++mt) {
            mma16816(acc[mt][nt], a[mt][2 * kb], bc.x, bc.y);
            mma16816(acc[mt][nt], a[mt][2 * kb + 1], bc.z, bc.w);
          }
        }

      // ---- epilogue: per head (og), row (mt, hh): 8 consecutive channels d = 8 q4 ..
      const int proj = blk >> 1;                                  // 0 q, 1 k, 2 v, 3 g
#pragma unroll
      for (int og = 0; og < 2; ++og)
#pragma unroll
        for (int mt = 0; mt < G::MT; ++mt)
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {          // every lane runs the quad reductions (a row >= M only skips its stores)
            float v[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) { v[2 * j] = acc[mt][4 * og + j][2 * hh]; v[2 * j + 1] = acc[mt][4 * og + j][2 * hh + 1]; }
            const int head = 2 * (blk & 1) + og;
            const int col = 64 * (blk & 1) + 32 * og + 8 * q4;      // channel within the projection
            const size_t row = (size_t)rr[mt][hh];
            if (proj >= 2) {                                          // v (head-major) and g (row-major): bf16 of the accumulators
              uint4 u;
              u.x = pack_bf16(v[0], v[1]); u.y = pack_bf16(v[2], v[3]); u.z = pack_bf16(v[4], v[5]); u.w = pack_bf16(v[6], v[7]);
              if (rok[mt][hh]) {
                if (proj == 2) stg128(p.vh + qkv_off(p, rr[mt][hh], nidx[mt][hh], sidx[mt][hh], head) + 8 * q4, u);
                else stg128(p.g + row * 128 + col, u);
              }
            } else {                                                  // q, k: round, head RMS norm, round, RoPE, round
              float pb[8];
#pragma unroll
              for (int i = 0; i < 8; ++i) pb[i] = rn(v[i]);
              if (p.xs != nullptr && rok[mt][hh]) {
                uint4 u;
                u.x = pack_bf16(pb[0], pb[1]); u.y = pack_bf16(pb[2], pb[3]); u.z = pack_bf16(pb[4], pb[5]); u.w = pack_bf16(pb[6], pb[7]);
                stg128((proj == 0 ? p.pqs : p.pks) + row * 128 + col, u);
              }
              float ssq = 0.f;
#pragma unroll
              for (int i = 0; i < 8; ++i) ssq = fmaf(pb[i], pb[i], ssq);
              const float rh = 1.f / sqrtf(quad_sum(ssq) * (1.f / 32.f) + p.qk_eps);
              float x[8];
#pragma unroll
              for (int i = 0; i < 8; ++i) x[i] = rn(pb[i] * rh);
              float cs[8], sn[8];
              const float* cp = p.cos + (size_t)mrow[mt][hh] * 16 + 8 * (q4 & 1);
              const float* sp = p.sin + (size_t)mrow[mt][hh] * 16 + 8 * (q4 & 1);
              {
                const float4 c0 = *reinterpret_cast<const float4*>(cp), c1 = *reinterpret_cast<const float4*>(cp + 4);
                const float4 s0 = *reinterpret_cast<const float4*>(sp), s1 = *reinterpret_cast<const float4*>(sp + 4);
                cs[0] = c0.x; cs[1] = c0.y; cs[2] = c0.z; cs[3] = c0.w; cs[4] = c1.x; cs[5] = c1.y; cs[6] = c1.z; cs[7] = c1.w;
                sn[0] = s0.x; sn[1] = s0.y; sn[2] = s0.z; sn[3] = s0.w; sn[4] = s1.x; sn[5] = s1.y; sn[6] = s1.z; sn[7] = s1.w;
              }
              rope8(x, cs, sn, q4);
              uint4 u;
              u.x = pack_bf16(x[0], x[1]); u.y = pack_bf16(x[2], x[3]); u.z = pack_bf16(x[4], x[5]); u.w = pack_bf16(x[6], x[7]);
              if (rok[mt][hh]) stg128((proj == 0 ? p.qh : p.kh) + qkv_off(p, rr[mt][hh], nidx[mt][hh], sidx[mt][hh], head) + 8 * q4, u);
            }
          }
    }
  }
}

}  // namespace sw80
