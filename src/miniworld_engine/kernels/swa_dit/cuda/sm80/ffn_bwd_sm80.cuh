// ffn_bwd_sm80.cuh -- the backward of the FFN half of the SWA atom DiT block, A100 / sm_80: the twin of the Triton ``_swa_ffn_bwd_kernel`` (with materialised dW operands) in two kernels.
// Forward:  y = rn(rn(q1 rstd) (1 + scale_f) + shift_f),  a | b = y Wu^T,  h = rn(a sigmoid(a) b),  ffn = rn(h Wd^T),  q2 = q1 + gate_f ffn.  For a row (rn = round to bf16, all else fp32), dq2 = dy:
//
//   ffn_bwd_gate_kernel   dffn = rn(dq2 rn(gate_f))                                   -> ``dffn``       (the dWd operand)
//                         d gate_f = dq2 ffn                                          -> dmod columns 640 .. 767
//                         a, b = y Wu^T (recomputed), sa = sigmoid(a), h = rn(a sa b)  -> ``hh``       (the dWd operand)
//                         dh = dffn Wd,  da = rn(dh b sa (1 + a (1 - sa))),  db = rn(dh a sa)  -> ``dab`` = da | db  (the dWu operand)
//   ffn_bwd_dy_kernel     dy_ = da Wu_a + db Wu_b = dab Wu,  xh = q1 rstd,  rstd = 1 / sqrt(mean(q1^2) + eps)
//                         d scale_f = dy_ xh, d shift_f = dy_                          -> dmod columns 512 .. 639 | 384 .. 511
//                         dq1 = rn(dq2 + rstd (dy_ (1 + scale_f) - xh mean(dy_ (1 + scale_f) xh)))
//
// Both keep their weights in the rows-are-outputs form the mma B fragments are read in (``qf_woff`` swizzle, the f1 row order: a thread's accumulator pair over the 4 n tiles of a group of
// 32 is 8 consecutive channels, which are also the channels of its A fragments and of its 16-byte loads and stores):
//   gate: Wu rows (hidden units) and Wd^T rows (hidden units) stream through a 3-deep cp.async ring in chunks of 32 hidden units, shared by the warps of the CTA;
//   dy:   Wu^T [128 channels][512 hidden] stays in shared memory (128 KB).
#pragma once
#include "adaln_bwd_sm80.cuh"    // bwd_rows_sm80.cuh, ldg_f4b, adaln_bwd_epilogue
#include "qkvg_fwd_sm80.cuh"     // qf_channel / qf_woff, rn

namespace sw80 {

// ================================================================== gate
struct FfnBwdGateParams {
  const __nv_bfloat16* dy;                           // [M][128] dq2
  const __nv_bfloat16* y;                            // [M][128]
  const __nv_bfloat16* ffn;                          // [M][128]
  const float* mod;                                  // [B S][768]
  const __nv_bfloat16* wu;                           // [512][128]: Wu_a (rows 0..255) | Wu_b
  const __nv_bfloat16* wdt;                          // [256][128] = Wd^T
  __nv_bfloat16 *dffn, *hh, *dab;                    // [M][128], [M][256], [M][512]
  float* dmod;                                       // [B S][768]; columns 640 .. 767 are d gate_f
  int M, S, B;
};

template <int NW_, int MINB_>
struct FfnBwdGateCfg {
  static constexpr int NW = NW_, NTHR = NW_ * 32, MINB = MINB_;
  static constexpr int HC = 32, NCH = 256 / HC, NST = 3;                    // hidden units per chunk, chunks per tile, ring depth
  static constexpr int STAGE = 3 * HC * 256, SMEM = NST * STAGE;           // Wu_a | Wu_b | Wd^T rows of a chunk
};

template <class G>
DEVI void bg_load_chunk(const FfnBwdGateParams& p, uint32_t stage, int j, int tid) {
  // smem row r of a chunk holds hidden unit 32 j + qf_channel(r) (r < 32: the within-group permutation of f1)
  for (int i = tid; i < G::HC * 16; i += G::NTHR) {
    const int r = i >> 4, c = i & 15, hid = G::HC * j + qf_channel(r);
    const uint32_t off = qf_woff(r, c);
    cp_async16(stage + off, p.wu + (size_t)hid * 128 + c * 8);
    cp_async16(stage + G::HC * 256 + off, p.wu + (size_t)(256 + hid) * 128 + c * 8);
    cp_async16(stage + 2 * G::HC * 256 + off, p.wdt + (size_t)hid * 128 + c * 8);
  }
}

template <class G, int MODE>
__global__ void __launch_bounds__(G::NTHR, G::MINB) ffn_bwd_gate_kernel(const FfnBwdGateParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sring = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  uint32_t wbq[4];                                               // B-fragment address of channel chunk kb in row g8 of a group of 8 rows
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) wbq[kb] = qf_woff(g8, 4 * kb + q4);

  const int ntile_all = bwd_ntile<MODE>(p.M, p.S, p.B);
  const int ngrp = (ntile_all + G::NW - 1) / G::NW;                // groups of NW tiles (one per warp): the CTA's iterations
  const int mine = (int)blockIdx.x < ngrp ? (ngrp - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  const int total = mine * G::NCH;                                 // chunks of all this CTA's iterations, one stream through the ring

#pragma unroll
  for (int s = 0; s < G::NST - 1; ++s) {
    if (s < total) bg_load_chunk<G>(p, sring + s * G::STAGE, s % G::NCH, tid);
    cp_async_commit();
  }

  uint32_t ay[8][4], adff[8][4];                                   // A fragments of y and of dffn (channel chunk kb = 32 kb + 8 q4 .. + 7 of row g8 + 8 hh)
  RowMap rm;
  for (int ci = 0; ci < total; ++ci) {
    cp_async_wait<G::NST - 2>();
    __syncthreads();                                               // chunk ci is in shared memory; the stage of chunk ci - 1 is free
    const int j = ci % G::NCH, it = ci / G::NCH;
    {
      const int cn = ci + G::NST - 1;
      if (cn < total) bg_load_chunk<G>(p, sring + (cn % G::NST) * G::STAGE, cn % G::NCH, tid);
      cp_async_commit();
    }
    if (j == 0) {                                                  // ---- a new tile: y -> A fragments, dffn = rn(dq2 rn(gate_f)) -> A fragments + store, d gate_f
      const int tile = (blockIdx.x + it * gridDim.x) * G::NW + warp;
      rm = row_map<MODE>(tile, p.M, p.S, p.B, g8);
      float dgr[4] = {0.f, 0.f, 0.f, 0.f};                         // MODE_HOIST: this lane's share of d gate_f of block kb, summed over the row halves
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const float* mp = p.mod + (size_t)rm.mrow[hh] * 768 + 640;
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
          uint4 uy = make_uint4(0u, 0u, 0u, 0u), ud = uy, uf = uy;
          if (rm.rok[hh]) {
            const size_t off = (size_t)rm.rr[hh] * 128 + 32 * kb + 8 * q4;
            uy = ldg128(p.y + off);
            ud = ldg128(p.dy + off);
            uf = ldg128(p.ffn + off);
          }
          ay[2 * kb][hh] = uy.x;      ay[2 * kb][2 + hh] = uy.y;
          ay[2 * kb + 1][hh] = uy.z;  ay[2 * kb + 1][2 + hh] = uy.w;
          const float4 g0 = ldg_f4b(mp + 32 * kb + 8 * q4), g1 = ldg_f4b(mp + 32 * kb + 8 * q4 + 4);
          const float gfv[8] = {rn(g0.x), rn(g0.y), rn(g0.z), rn(g0.w), rn(g1.x), rn(g1.y), rn(g1.z), rn(g1.w)};
          const float dq[8] = {bf16lo(ud.x), bf16hi(ud.x), bf16lo(ud.y), bf16hi(ud.y), bf16lo(ud.z), bf16hi(ud.z), bf16lo(ud.w), bf16hi(ud.w)};
          const float ff[8] = {bf16lo(uf.x), bf16hi(uf.x), bf16lo(uf.y), bf16hi(uf.y), bf16lo(uf.z), bf16hi(uf.z), bf16lo(uf.w), bf16hi(uf.w)};
          float t[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) t[i] = dq[i] * gfv[i];
          const uint32_t n0 = pack_bf16(t[0], t[1]), n1 = pack_bf16(t[2], t[3]), n2 = pack_bf16(t[4], t[5]), n3 = pack_bf16(t[6], t[7]);
          adff[2 * kb][hh] = n0;      adff[2 * kb][2 + hh] = n1;
          adff[2 * kb + 1][hh] = n2;  adff[2 * kb + 1][2 + hh] = n3;
          if (rm.rok[hh]) stg128(p.dffn + (size_t)rm.rr[hh] * 128 + 32 * kb + 8 * q4, make_uint4(n0, n1, n2, n3));
          if (MODE == MODE_SINGLE) {
            if (rm.rok[hh]) {
              float* dm = p.dmod + (size_t)rm.mrow[hh] * 768 + 640 + 32 * kb + 8 * q4;
              *reinterpret_cast<float4*>(dm) = make_float4(dq[0] * ff[0], dq[1] * ff[1], dq[2] * ff[2], dq[3] * ff[3]);
              *reinterpret_cast<float4*>(dm + 4) = make_float4(dq[4] * ff[4], dq[5] * ff[5], dq[6] * ff[6], dq[7] * ff[7]);
            }
          } else {
            float v[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = dq[i] * ff[i];
            reduce_scatter_g8<8>(v, lane);
            dgr[kb] += v[0];
          }
        }
      }
      if (MODE == MODE_HOIST && rm.rok[0]) {
        const int i0 = g8_base(lane, 8);
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) p.dmod[((size_t)rm.blk * p.B * p.S + rm.mrow[0]) * 768 + 640 + 32 * kb + 8 * q4 + i0] = dgr[kb];
      }
    }

    // ---- chunk j: a, b = y Wu^T and dh = dffn Wd of hidden units 32 j .. 32 j + 31 (4 n tiles each; thread: units 32 j + 8 q4 .. + 7)
    const uint32_t stg = sring + (ci % G::NST) * G::STAGE, wa = stg, wb = stg + G::HC * 256, wd = stg + 2 * G::HC * 256;
    float ac[4][4], bc[4][4], dc[4][4];
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      ac[t][0] = ac[t][1] = ac[t][2] = ac[t][3] = 0.f;
      bc[t][0] = bc[t][1] = bc[t][2] = bc[t][3] = 0.f;
      dc[t][0] = dc[t][1] = dc[t][2] = dc[t][3] = 0.f;
    }
#pragma unroll
    for (int kb = 0; kb < 4; ++kb)
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const uint4 b1 = lds128_ro(wa + t * 8 * 256 + wbq[kb]);
        const uint4 b2 = lds128_ro(wb + t * 8 * 256 + wbq[kb]);
        const uint4 b3 = lds128_ro(wd + t * 8 * 256 + wbq[kb]);
        mma16816(ac[t], ay[2 * kb], b1.x, b1.y);
        mma16816(ac[t], ay[2 * kb + 1], b1.z, b1.w);
        mma16816(bc[t], ay[2 * kb], b2.x, b2.y);
        mma16816(bc[t], ay[2 * kb + 1], b2.z, b2.w);
        mma16816(dc[t], adff[2 * kb], b3.x, b3.y);
        mma16816(dc[t], adff[2 * kb + 1], b3.z, b3.w);
      }
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      float hv[8], dav[8], dbv[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {                                // unit 8 q4 + i = 2 t + e of the quad's pair: n tile t = i / 2, column e = i & 1
        const float av = ac[i >> 1][2 * hh + (i & 1)], bv = bc[i >> 1][2 * hh + (i & 1)], dh = dc[i >> 1][2 * hh + (i & 1)];
        const float sa = sigmoidf(av);
        hv[i] = (av * sa) * bv;
        dav[i] = ((dh * bv) * sa) * (1.f + av * (1.f - sa));
        dbv[i] = (dh * av) * sa;
      }
      if (rm.rok[hh]) {
        const size_t u = (size_t)G::HC * j + 8 * q4;
        stg128(p.hh + (size_t)rm.rr[hh] * 256 + u, make_uint4(pack_bf16(hv[0], hv[1]), pack_bf16(hv[2], hv[3]), pack_bf16(hv[4], hv[5]), pack_bf16(hv[6], hv[7])));
        stg128(p.dab + (size_t)rm.rr[hh] * 512 + u, make_uint4(pack_bf16(dav[0], dav[1]), pack_bf16(dav[2], dav[3]), pack_bf16(dav[4], dav[5]), pack_bf16(dav[6], dav[7])));
        stg128(p.dab + (size_t)rm.rr[hh] * 512 + 256 + u, make_uint4(pack_bf16(dbv[0], dbv[1]), pack_bf16(dbv[2], dbv[3]), pack_bf16(dbv[4], dbv[5]), pack_bf16(dbv[6], dbv[7])));
      }
    }
  }
}

// ================================================================== dy
struct FfnBwdDyParams {
  const __nv_bfloat16* dab;                          // [M][512] da | db
  const __nv_bfloat16* q1;                           // [M][128]
  const __nv_bfloat16* dy;                           // [M][128] dq2
  const float* mod;                                  // [B S][768]
  const __nv_bfloat16* wut;                          // [128 channels][512 hidden] = Wu^T
  __nv_bfloat16* dq1;                                // [M][128]
  float* dmod;                                       // [B S][768]; columns 384 .. 511 are d shift_f, 512 .. 639 d scale_f
  int M, S, B;
  float eps;
};

template <int NW_>
struct FfnBwdDyCfg {
  static constexpr int NW = NW_, NTHR = NW_ * 32, SMEM = 128 * 1024;
};

// byte offset of 16-byte chunk `chunk` (0 .. 63) of row `row` of the [128][1024 B] Wu^T tile: the chunk index is XOR-ed with 4 on odd rows (as qf_woff)
DEVI uint32_t wut_off(uint32_t row, uint32_t chunk) { return row * 1024u + ((chunk ^ ((row & 1u) << 2)) << 4); }

template <class G, int MODE>
__global__ void __launch_bounds__(G::NTHR, 1) ffn_bwd_dy_kernel(const FfnBwdDyParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sw = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  // Wu^T into shared memory in the f1 row order (smem row r holds output channel qf_channel(r)), once per CTA
  for (int i = tid; i < 128 * 64; i += G::NTHR) cp_async16(sw + wut_off(i >> 6, i & 63), p.wut + (size_t)qf_channel(i >> 6) * 512 + (i & 63) * 8);
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  const uint32_t wlane = g8 * 1024u + q4 * 16u, gpar = g8 & 1u;    // the B fragment of hidden chunk kb (32 units) in row g8 of a group of 8 rows is at wlane + 64 (kb ^ gpar) (wut_off, the XOR folded)

  const int ntile = bwd_ntile<MODE>(p.M, p.S, p.B);
  for (int base = blockIdx.x * G::NW; base < ntile; base += G::NW * gridDim.x) {
    const int tile = base + warp;
    const RowMap rm = row_map<MODE>(tile, p.M, p.S, p.B, g8);

    // ---- dy_ = dab Wu: the f1 order (4 groups x 4 n tiles); K = 512 hidden units in 16 chunks of 32 (thread: units 32 kb + 8 q4 .. + 7)
    float acc[4][4][4];
#pragma unroll
    for (int gq = 0; gq < 4; ++gq)
#pragma unroll
      for (int j = 0; j < 4; ++j) { acc[gq][j][0] = acc[gq][j][1] = acc[gq][j][2] = acc[gq][j][3] = 0.f; }
    // the dab fragments of a block of 32 hidden units are requested three blocks ahead
    uint4 u0[2], u1[2], u2[2], u3[2];
    auto load_dab = [&](int kbl, uint4 (&dst)[2]) {
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) dst[hh] = rm.rok[hh] ? ldg128(p.dab + (size_t)rm.rr[hh] * 512 + 32 * kbl + 8 * q4) : make_uint4(0u, 0u, 0u, 0u);
    };
    load_dab(0, u0);
    load_dab(1, u1);
    load_dab(2, u2);
#pragma unroll 1
    for (int kb = 0; kb < 16; ++kb) {
      if (kb + 3 < 16) load_dab(kb + 3, u3);
      uint32_t a[2][4];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) { a[0][hh] = u0[hh].x; a[0][2 + hh] = u0[hh].y; a[1][hh] = u0[hh].z; a[1][2 + hh] = u0[hh].w; }
#pragma unroll
      for (int nt = 0; nt < 16; ++nt) {
        const uint4 b = lds128_ro(sw + nt * 8 * 1024 + wlane + 64u * (kb ^ gpar));
        mma16816(acc[nt >> 2][nt & 3], a[0], b.x, b.y);
        mma16816(acc[nt >> 2][nt & 3], a[1], b.z, b.w);
      }
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) { u0[hh] = u1[hh]; u1[hh] = u2[hh]; u2[hh] = u3[hh]; }
    }

    // ---- the adaLN backward: d scale_f -> dmod columns 512 .., d shift_f -> 384 .., dq1 = rn(dq2 + ...)
    adaln_bwd_epilogue<MODE>(acc, rm, lane, p.q1, p.dy, p.mod, 512, p.dq1, p.dmod, 384, p.B, p.S, p.eps);
  }
}

}  // namespace sw80
