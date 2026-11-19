// bwd_stream_sm120.cuh -- the FFN backward's dy kernel and the qkvg backward kernel of the SWA atom DiT block for the RTX 5090 (sm_120a, 99 KB of shared memory per block).
// The sm_80 kernels (``ffn_bwd_dy_kernel``, ``qkvg_bwd_kernel``) keep the whole 128 KB weight W^T [128 channels][512] resident in shared memory; here it STREAMS through a
// cp.async ring instead, 8 KB per block of 32 of the 512 K units (a row of the f1 order, 64 B), shared by the NW warps of a CTA (one 16-row tile each), one stream through all of
// the CTA's tile groups.  The statements, the order of the K blocks and the epilogue (``adaln_bwd_epilogue``) are the sm_80 kernels': the same values, bitwise, as their one-warp-per-tile
// form.  The sm_80 headers are not changed (A100 / H100 outputs are untouched).
#pragma once
#include "ffn_bwd_sm80.cuh"
#include "qkvg_bwd_sm80.cuh"
#include "bwd_epi_sm120.cuh"

namespace sw80 {

// NW warps per CTA (one tile each), KPS K blocks (of 32 units) per ring stage, NST ring stages, MINB CTAs per SM
template <int NW_, int KPS_, int NST_, int MINB_, int DBG_ = 0>
struct BwdStreamCfg {
  static constexpr int NW = NW_, NTHR = NW_ * 32, KPS = KPS_, NST = NST_, MINB = MINB_, DBG = DBG_;
  static constexpr int NSG = 16 / KPS_;                                   // ring stages per tile group
  static constexpr int SUB = 8192, STAGE = KPS_ * SUB, SMEM = NST_ * STAGE;
};

// stage of K blocks 32 (kg KPS + j) .. + 31, j < KPS: sub-block j, smem row r (the f1 order: output channel qf_channel(r)) of 64 B at r 64 -- the B fragment of row g8 of n tile nt, chunk q4 is at
// nt 512 + g8 64 + q4 16 (a quarter warp reads 128 contiguous bytes: no bank conflicts)
template <class G>
DEVI void bs_load_stage(const __nv_bfloat16* wt, uint32_t stage, int kg, int tid) {
  for (int i = tid; i < G::KPS * 512; i += G::NTHR) {
    const int j = i >> 9, r = (i >> 2) & 127, c = i & 3;
    cp_async16(stage + j * 8192 + r * 64 + c * 16, wt + (size_t)qf_channel(r) * 512 + (kg * G::KPS + j) * 32 + c * 8);
  }
}

template <class G, int MODE>
__global__ void __launch_bounds__(G::NTHR, G::MINB) ffn_bwd_dy_stream_kernel(const FfnBwdDyParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sring = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const uint32_t wlane = g8 * 64u + q4 * 16u;

  const int ntile = bwd_ntile<MODE>(p.M, p.S, p.B);
  const int ngrp = (ntile + G::NW - 1) / G::NW;
  const int mine = (int)blockIdx.x < ngrp ? (ngrp - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  const int total = mine * G::NSG;
#pragma unroll
  for (int s = 0; s < G::NST - 1; ++s) {
    if (s < total) bs_load_stage<G>(p.wut, sring + s * G::STAGE, s % G::NSG, tid);
    cp_async_commit();
  }

  float acc[4][4][4];
  RowMap rm;
  EpiPre epre;
  uint4 u0[2], u1[2], u2[2], u3[2];
  auto load_dab = [&](int kbl, uint4 (&dst)[2]) {
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) dst[hh] = (G::DBG == 5 || !rm.rok[hh]) ? make_uint4(0u, 0u, 0u, 0u) : ldg128(p.dab + (size_t)rm.rr[hh] * 512 + 32 * kbl + 8 * q4);
  };
  for (int ci = 0; ci < total; ++ci) {
    cp_async_wait<G::NST - 2>();
    __syncthreads();
    {
      const int cn = ci + G::NST - 1;
      if (cn < total) bs_load_stage<G>(p.wut, sring + (cn % G::NST) * G::STAGE, cn % G::NSG, tid);
      cp_async_commit();
    }
    const int kg = ci % G::NSG;
    if (kg == 0) {
      const int tile = ((int)blockIdx.x + (ci / G::NSG) * (int)gridDim.x) * G::NW + warp;
      rm = row_map<MODE>(tile, p.M, p.S, p.B, g8);
      epi_prefetch(epre, rm, p.q1, p.dy, q4);
#pragma unroll
      for (int gq = 0; gq < 4; ++gq)
#pragma unroll
        for (int j = 0; j < 4; ++j) { acc[gq][j][0] = acc[gq][j][1] = acc[gq][j][2] = acc[gq][j][3] = 0.f; }
      load_dab(0, u0);
      load_dab(1, u1);
      load_dab(2, u2);
    }
    const uint32_t stg = sring + (ci % G::NST) * G::STAGE;
#pragma unroll 1
    for (int j = 0; j < G::KPS; ++j) {
      const int kb = kg * G::KPS + j;
      if (kb + 3 < 16) load_dab(kb + 3, u3);
      uint32_t a[2][4];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) { a[0][hh] = u0[hh].x; a[0][2 + hh] = u0[hh].y; a[1][hh] = u0[hh].z; a[1][2 + hh] = u0[hh].w; }
#pragma unroll
      for (int nt = 0; nt < (G::DBG == 6 ? 0 : 16); ++nt) {
        const uint4 b = lds128_ro(stg + j * 8192 + nt * 512 + wlane);
        mma16816(acc[nt >> 2][nt & 3], a[0], b.x, b.y);
        mma16816(acc[nt >> 2][nt & 3], a[1], b.z, b.w);
      }
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) { u0[hh] = u1[hh]; u1[hh] = u2[hh]; u2[hh] = u3[hh]; }
    }
    if (kg == G::NSG - 1 && G::DBG != 1) {
      if (G::DBG == 3) adaln_bwd_epilogue<MODE>(acc, rm, lane, p.q1, p.dy, p.mod, 512, p.dq1, p.dmod, 384, p.B, p.S, p.eps);
      else epi_run<G::DBG>(acc, rm, epre, q4, p.mod, 512, p.dq1, p.dmod, 384, p.eps);
    }
  }
}

template <class G, int MODE>
__global__ void __launch_bounds__(G::NTHR, G::MINB) qkvg_bwd_stream_kernel(const QkvgBwdParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sring = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const uint32_t wlane = g8 * 64u + q4 * 16u;

  const int ntile = bwd_ntile<MODE>(p.M, p.S, p.B);
  const int ngrp = (ntile + G::NW - 1) / G::NW;
  const int mine = (int)blockIdx.x < ngrp ? (ngrp - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  const int total = mine * G::NSG;
#pragma unroll
  for (int s = 0; s < G::NST - 1; ++s) {
    if (s < total) bs_load_stage<G>(p.wt, sring + s * G::STAGE, s % G::NSG, tid);
    cp_async_commit();
  }

  float acc[4][4][4];
  float cs[2][8], sn[2][8];
  RowMap rm;
  EpiPre epre;
  uint4 cd[2], cp_[2], nd[2], np_[2], md[2], mp_[2];
  auto load_block = [&](int kbl, uint4 (&dd)[2], uint4 (&pp)[2]) {
    const int part = kbl >> 2, h = kbl & 3;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      dd[hh] = pp[hh] = make_uint4(0u, 0u, 0u, 0u);
      if (rm.rok[hh]) {
        const size_t hm = (((size_t)rm.nrow[hh] * 4 + h) * p.S + rm.sat[hh]) * 32 + 8 * q4;
        if (part < 2) {
          dd[hh] = ldg128((part == 0 ? p.dqh : p.dkh) + hm);
          pp[hh] = ldg128((part == 0 ? p.pq : p.pk) + (size_t)rm.rr[hh] * 128 + 32 * h + 8 * q4);
        } else {
          dd[hh] = part == 2 ? ldg128(p.dvh + hm) : ldg128(p.dg + (size_t)rm.rr[hh] * 128 + 32 * h + 8 * q4);
        }
      }
    }
  };
  for (int ci = 0; ci < total; ++ci) {
    cp_async_wait<G::NST - 2>();
    __syncthreads();
    {
      const int cn = ci + G::NST - 1;
      if (cn < total) bs_load_stage<G>(p.wt, sring + (cn % G::NST) * G::STAGE, cn % G::NSG, tid);
      cp_async_commit();
    }
    const int kg = ci % G::NSG;
    if (kg == 0) {
      const int tile = ((int)blockIdx.x + (ci / G::NSG) * (int)gridDim.x) * G::NW + warp;
      rm = row_map<MODE>(tile, p.M, p.S, p.B, g8);
      epi_prefetch(epre, rm, p.qi, p.dq1, q4);
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const float* cp = p.cos + (size_t)rm.mrow[hh] * 16 + 8 * (q4 & 1);
        const float* sp = p.sin + (size_t)rm.mrow[hh] * 16 + 8 * (q4 & 1);
        const float4 c0 = rm.rok[hh] ? ldg_f4b(cp) : make_float4(1.f, 1.f, 1.f, 1.f), c1 = rm.rok[hh] ? ldg_f4b(cp + 4) : make_float4(1.f, 1.f, 1.f, 1.f);
        const float4 s0 = rm.rok[hh] ? ldg_f4b(sp) : make_float4(0.f, 0.f, 0.f, 0.f), s1 = rm.rok[hh] ? ldg_f4b(sp + 4) : make_float4(0.f, 0.f, 0.f, 0.f);
        cs[hh][0] = c0.x; cs[hh][1] = c0.y; cs[hh][2] = c0.z; cs[hh][3] = c0.w; cs[hh][4] = c1.x; cs[hh][5] = c1.y; cs[hh][6] = c1.z; cs[hh][7] = c1.w;
        sn[hh][0] = s0.x; sn[hh][1] = s0.y; sn[hh][2] = s0.z; sn[hh][3] = s0.w; sn[hh][4] = s1.x; sn[hh][5] = s1.y; sn[hh][6] = s1.z; sn[hh][7] = s1.w;
      }
#pragma unroll
      for (int gq = 0; gq < 4; ++gq)
#pragma unroll
        for (int j = 0; j < 4; ++j) { acc[gq][j][0] = acc[gq][j][1] = acc[gq][j][2] = acc[gq][j][3] = 0.f; }
      load_block(0, cd, cp_);
      load_block(1, nd, np_);
    }
    const uint32_t stg = sring + (ci % G::NST) * G::STAGE;
#pragma unroll 1
    for (int j = 0; j < G::KPS; ++j) {
      const int kb = kg * G::KPS + j;
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
          for (int i = 0; i < 8; ++i) {
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
        const uint4 b = lds128_ro(stg + j * 8192 + nt * 512 + wlane);
        mma16816(acc[nt >> 2][nt & 3], a[0], b.x, b.y);
        mma16816(acc[nt >> 2][nt & 3], a[1], b.z, b.w);
      }
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) { cd[hh] = nd[hh]; cp_[hh] = np_[hh]; nd[hh] = md[hh]; np_[hh] = mp_[hh]; }
    }
    if (kg == G::NSG - 1 && G::DBG != 1) {
      if (G::DBG == 3) adaln_bwd_epilogue<MODE>(acc, rm, lane, p.qi, p.dq1, p.mod, 128, p.dq, p.dmod, 0, p.B, p.S, p.eps);
      else epi_run<G::DBG>(acc, rm, epre, q4, p.mod, 128, p.dq, p.dmod, 0, p.eps);
    }
  }
}

}  // namespace sw80
