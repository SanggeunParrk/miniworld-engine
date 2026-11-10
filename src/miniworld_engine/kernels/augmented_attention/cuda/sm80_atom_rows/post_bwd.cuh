// post_bwd.cuh -- the second part of the AF3 local atom DiT block's post-attention backward on A100 / sm_80 (the bf16 module's autograd rounding points):
//   dx2 = rn(rn(da Wa) + rn(db Wb))                                        (DAB = [da | db] from tr_bwd.cuh)
//   x2 = rn(rn(s2 xn) + bi2), xn = rn(LN(a2)):  dbi2 = dx2;  ds2 = rn(dx2 xn);  dxn = rn(dx2 s2);  dsc2 = rn(rn(ds2 rn(1 - s2)) s2)
//   da2 = rn(dy + rn(LN_bwd(dxn)))                                         (out = a2 + st t: the residual)
//   a2 = x + rn(so u):  DA = da2 (pre_bwd adds the AdaLN1 path);  dso = rn(da2 u);  du = rn(da2 so);  dos = rn(rn(dso rn(1 - so)) so)
//   u = rn(gated Wo^T):  dgated = rn(du Wo);  gated = rn(sg o):  dsg = rn(dgated o);  dO = rn(dgated sg);  dg = rn(rn(dsg rn(1 - sg)) sg)
//   D[a, h, n] = the sum over head h's 32 channels of dO o (fp32, the attention backward's row term)
// -> DA [M, 128]; dmod columns 512 .. (dsc2), 640 .. (dbi2), 768 .. (dos); DU, GATED (dWo = DU^T GATED); DO; dg into dP's block [128, 256); D; the column sums of dsc2 and dos (atomics).
// Structure (the SWA ffn_bwd dy kernel's): Wu^T [128 channels][512 hidden] and Wo^T stay in shared memory (160 KB), a warp owns tiles of 16 rows; the dab fragments of a block of 32 hidden
// units are requested three blocks ahead; the products' accumulators are the row stages' operands (the f1 order: a thread's 8 consecutive channels of each 32).
#pragma once
#include "rows_layout.cuh"

namespace lr80 {

struct PostBwdParams {
  const __nv_bfloat16* dab;                          // [M][512] da | db
  const __nv_bfloat16* dy;                           // [M][128]
  const __nv_bfloat16* a2;                           // [M][128]
  const __nv_bfloat16* mod;                          // [M][1024]
  const __nv_bfloat16* u;                            // [M][128]
  const __nv_bfloat16* sg;                           // [M][128]
  const __nv_bfloat16* o;                            // [M][128]
  const __nv_bfloat16* wut;                          // [128 channels][512 hidden] = Wu^T
  const __nv_bfloat16* wot;                          // [128][128] = Wo^T
  __nv_bfloat16 *da, *du, *gated, *dO;               // [M][128]
  __nv_bfloat16* dp;                                 // [M][512]: dg goes to columns 128 .. 255
  __nv_bfloat16* dmod;                               // [M][1024]
  float* dD;                                         // [A][4][N]
  float* dbias;                                      // [1024]: d bsc2 at 512 .., d bos at 768 ..
  int M, N;
  float eps;
  int prefetch;                                      // 1: the next tile's rows are requested into L2
};

// STREAM_ = false: Wu^T stays in shared memory (160 KB with Wo^T, one CTA of 8 warps a SM); true: Wu^T streams through a three-stage ring of 64-unit chunks (16 KB), 80 KB with Wo^T,
// two CTAs of 4 warps a SM (two groups that drift apart)
template <int NW_, bool STREAM_ = false, int MINB_ = 1>
struct PostBwdCfg {
  static constexpr int NW = NW_, NTHR = NW_ * 32, STREAM = STREAM_, MINB = MINB_, NST = 3, W_BYTES = 128 * 128;
  static constexpr int WU_BYTES = STREAM_ ? NST * W_BYTES : 128 * 1024, SMEM = WU_BYTES + 128 * 256;
};

// byte offset of 16-byte chunk `chunk` (0 .. 63) of row `row` of the [128][1024 B] Wu^T tile: the chunk index is XOR-ed with 4 on odd rows (as qf_woff)
DEVI uint32_t wut_off(uint32_t row, uint32_t chunk) { return row * 1024u + ((chunk ^ ((row & 1u) << 2)) << 4); }

DEVI uint32_t omin(uint32_t s) { return pk2(1.f - bf16lo(s), 1.f - bf16hi(s)); }   // rn(1 - s) per half

// the rest of a tile once dx2 is known: the adaLN 2 backward, the residual, the gates and the product dgated = rn(du Wo)
DEVI void pb_tail(const PostBwdParams& p, const uint32_t (&dx2p)[2][4][4], const int (&rr)[2], const bool (&rok)[2], uint32_t swo, const uint32_t (&wbq)[4], int lane, float (&bsc)[4][8],
                  float (&bos)[4][8]) {
  const int g8 = lane >> 2, q4 = lane & 3;
    // ---- the adaLN 2 backward and the residual: da2 = rn(dy + rn(LN_bwd(s2 dx2))), per row half
    uint32_t dup[2][4][4];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const size_t row = (size_t)rr[hh];
      float xh[4][8];
      float s = 0.f;
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) {
        const uint4 ua = rok[hh] ? ldg128(p.a2 + row * 128 + 32 * gq + 8 * q4) : make_uint4(0u, 0u, 0u, 0u);
        unpack8(ua, xh[gq]);
#pragma unroll
        for (int i = 0; i < 8; ++i) s += xh[gq][i];
      }
      const float mean = quad_sum(s) * (1.f / 128.f);
      float vs = 0.f;
#pragma unroll
      for (int gq = 0; gq < 4; ++gq)
#pragma unroll
        for (int i = 0; i < 8; ++i) { xh[gq][i] -= mean; vs = fmaf(xh[gq][i], xh[gq][i], vs); }
      const float rstd = 1.f / sqrtf(quad_sum(vs) * (1.f / 128.f) + p.eps);
      float sa = 0.f, sb = 0.f;                                    // sum dxn, sum dxn xhat
      uint32_t dxnp[4][4];
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) {
        const __nv_bfloat16* mp = p.mod + row * 1024 + 512 + 32 * gq + 8 * q4;
        const uint4 s2v = rok[hh] ? ldg128(mp) : make_uint4(0u, 0u, 0u, 0u);
        const uint32_t dx[4] = {dx2p[hh][gq][0], dx2p[hh][gq][1], dx2p[hh][gq][2], dx2p[hh][gq][3]};
        const uint32_t sc[4] = {s2v.x, s2v.y, s2v.z, s2v.w};
        uint4 dbi, dsc;
        uint32_t xn[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          xh[gq][2 * e] *= rstd; xh[gq][2 * e + 1] *= rstd;
          xn[e] = pk2(xh[gq][2 * e], xh[gq][2 * e + 1]);
        }
        const uint32_t ds2[4] = {pmul(dx[0], xn[0]), pmul(dx[1], xn[1]), pmul(dx[2], xn[2]), pmul(dx[3], xn[3])};
        dsc = make_uint4(pmul(pmul(ds2[0], omin(sc[0])), sc[0]), pmul(pmul(ds2[1], omin(sc[1])), sc[1]), pmul(pmul(ds2[2], omin(sc[2])), sc[2]), pmul(pmul(ds2[3], omin(sc[3])), sc[3]));
        dbi = make_uint4(dx[0], dx[1], dx[2], dx[3]);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          dxnp[gq][e] = pmul(dx[e], sc[e]);
          sa += bf16lo(dxnp[gq][e]) + bf16hi(dxnp[gq][e]);
          sb = fmaf(bf16lo(dxnp[gq][e]), xh[gq][2 * e], fmaf(bf16hi(dxnp[gq][e]), xh[gq][2 * e + 1], sb));
        }
        if (rok[hh]) {
          stg128(p.dmod + row * 1024 + 512 + 32 * gq + 8 * q4, dsc);
          stg128(p.dmod + row * 1024 + 640 + 32 * gq + 8 * q4, dbi);
          const uint32_t w[4] = {dsc.x, dsc.y, dsc.z, dsc.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) { bsc[gq][2 * e] += bf16lo(w[e]); bsc[gq][2 * e + 1] += bf16hi(w[e]); }
        }
      }
      const float m1 = quad_sum(sa) * (1.f / 128.f), m2 = quad_sum(sb) * (1.f / 128.f);
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) {
        uint32_t lnp[4];
#pragma unroll
        for (int e = 0; e < 4; ++e)
          lnp[e] = pk2(rstd * (bf16lo(dxnp[gq][e]) - m1 - xh[gq][2 * e] * m2), rstd * (bf16hi(dxnp[gq][e]) - m1 - xh[gq][2 * e + 1] * m2));
        const uint4 udy = rok[hh] ? ldg128(p.dy + row * 128 + 32 * gq + 8 * q4) : make_uint4(0u, 0u, 0u, 0u);
        const uint4 uu = rok[hh] ? ldg128(p.u + row * 128 + 32 * gq + 8 * q4) : make_uint4(0u, 0u, 0u, 0u);
        const uint4 uso = rok[hh] ? ldg128(p.mod + row * 1024 + 768 + 32 * gq + 8 * q4) : make_uint4(0u, 0u, 0u, 0u);
        const uint32_t dyw[4] = {udy.x, udy.y, udy.z, udy.w}, uw[4] = {uu.x, uu.y, uu.z, uu.w}, sow[4] = {uso.x, uso.y, uso.z, uso.w};
        uint32_t da2w[4], duw[4], dosw[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          da2w[e] = padd(dyw[e], lnp[e]);                          // da2 = rn(dy + rn(LN_bwd))
          const uint32_t dso = pmul(da2w[e], uw[e]);
          duw[e] = pmul(da2w[e], sow[e]);
          dup[hh][gq][e] = duw[e];
          dosw[e] = pmul(pmul(dso, omin(sow[e])), sow[e]);
        }
        if (rok[hh]) {
          stg128(p.da + row * 128 + 32 * gq + 8 * q4, make_uint4(da2w[0], da2w[1], da2w[2], da2w[3]));
          stg128(p.du + row * 128 + 32 * gq + 8 * q4, make_uint4(duw[0], duw[1], duw[2], duw[3]));
          stg128(p.dmod + row * 1024 + 768 + 32 * gq + 8 * q4, make_uint4(dosw[0], dosw[1], dosw[2], dosw[3]));
#pragma unroll
          for (int e = 0; e < 4; ++e) { bos[gq][2 * e] += bf16lo(dosw[e]); bos[gq][2 * e + 1] += bf16hi(dosw[e]); }
        }
      }
    }

    // ---- dgated = rn(du Wo): the A fragments are du (the f1 order), the accumulators dgated
    uint32_t a[8][4];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) put_a(a, gq, hh, make_uint4(dup[hh][gq][0], dup[hh][gq][1], dup[hh][gq][2], dup[hh][gq][3]));
    float acc2[4][4][4];
#pragma unroll
    for (int gq = 0; gq < 4; ++gq)
#pragma unroll
      for (int j = 0; j < 4; ++j) { acc2[gq][j][0] = acc2[gq][j][1] = acc2[gq][j][2] = acc2[gq][j][3] = 0.f; }
#pragma unroll
    for (int kb = 0; kb < 4; ++kb)
#pragma unroll
      for (int nt = 0; nt < 16; ++nt) {
        const uint4 b = lds128_ro(swo + nt * 8 * 256 + wbq[kb]);
        mma16816(acc2[nt >> 2][nt & 3], a[2 * kb], b.x, b.y);
        mma16816(acc2[nt >> 2][nt & 3], a[2 * kb + 1], b.z, b.w);
      }
    // ---- dsg = rn(dgated o), dO = rn(dgated sg), dg = rn(rn(dsg rn(1 - sg)) sg), gated = rn(sg o); D = the head sums of dO o
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const size_t row = (size_t)rr[hh];
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) {
        const uint4 uo = rok[hh] ? ldg128(p.o + row * 128 + 32 * gq + 8 * q4) : make_uint4(0u, 0u, 0u, 0u);
        const uint4 us = rok[hh] ? ldg128(p.sg + row * 128 + 32 * gq + 8 * q4) : make_uint4(0u, 0u, 0u, 0u);
        const uint32_t ow[4] = {uo.x, uo.y, uo.z, uo.w}, sgw[4] = {us.x, us.y, us.z, us.w};
        uint32_t dgw[4], dow[4], gtw[4];
        float dd = 0.f;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const uint32_t dgated = pk2(acc2[gq][e][2 * hh], acc2[gq][e][2 * hh + 1]);
          const uint32_t dsg = pmul(dgated, ow[e]);
          dow[e] = pmul(dgated, sgw[e]);
          dgw[e] = pmul(pmul(dsg, omin(sgw[e])), sgw[e]);
          gtw[e] = pmul(sgw[e], ow[e]);
          dd = fmaf(bf16lo(dow[e]), bf16lo(ow[e]), fmaf(bf16hi(dow[e]), bf16hi(ow[e]), dd));
        }
        dd = quad_sum(dd);
        if (rok[hh]) {
          stg128(p.dO + row * 128 + 32 * gq + 8 * q4, make_uint4(dow[0], dow[1], dow[2], dow[3]));
          stg128(p.dp + row * 512 + 128 + 32 * gq + 8 * q4, make_uint4(dgw[0], dgw[1], dgw[2], dgw[3]));
          stg128(p.gated + row * 128 + 32 * gq + 8 * q4, make_uint4(gtw[0], gtw[1], gtw[2], gtw[3]));
          if (q4 == 0) {
            const int sample = (int)(row / p.N), pos = (int)(row - (size_t)sample * p.N);
            p.dD[((size_t)sample * 4 + gq) * p.N + pos] = dd;
          }
        }
      }
    }
}

template <class G>
__global__ void __launch_bounds__(G::NTHR, G::MINB) post_bwd_kernel(const PostBwdParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sw = smem_u32(smem_raw), swo = sw + G::WU_BYTES;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  uint32_t wbq[4];
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) wbq[kb] = qf_woff(g8, 4 * kb + q4);

  float bsc[4][8], bos[4][8];                                      // column sums of dsc2 and dos over the rows this lane saw
#pragma unroll
  for (int kb = 0; kb < 4; ++kb)
#pragma unroll
    for (int i = 0; i < 8; ++i) { bsc[kb][i] = 0.f; bos[kb][i] = 0.f; }

  const int ntile = (p.M + 15) / 16;
  if constexpr (!G::STREAM) {
    // Wu^T and Wo^T into shared memory in the f1 row order (smem row r holds output channel qf_channel(r)), once per CTA
    for (int i = tid; i < 128 * 64; i += G::NTHR) cp_async16(sw + wut_off(i >> 6, i & 63), p.wut + (size_t)qf_channel(i >> 6) * 512 + (i & 63) * 8);
    for (int i = tid; i < 128 * 16; i += G::NTHR) cp_async16(swo + qf_woff(i >> 4, i & 15), p.wot + (size_t)qf_channel(i >> 4) * 128 + (i & 15) * 8);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    const uint32_t wlane = g8 * 1024u + q4 * 16u, gpar = g8 & 1u;    // the B fragment of hidden block kb (32 units) in row g8 of a group of 8 rows is at wlane + 64 (kb ^ gpar)
    for (int base = blockIdx.x * G::NW; base < ntile; base += G::NW * gridDim.x) {
      const int tile = base + warp;
      int rr[2];
      bool rok[2];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) { rr[hh] = 16 * tile + g8 + 8 * hh; rok[hh] = tile < ntile && rr[hh] < p.M; }
      // ---- dx2 = rn(rn(da Wa) + rn(db Wb)): two products of K = 256 hidden units in 8 blocks of 32 (thread: units 32 kb + 8 q4 .. + 7)
      uint32_t dx2p[2][4][4];                                        // [hh][group][j]: bf16 pairs of channels 32 gq + 8 q4 + 2 j + {0, 1}
      uint4 u0[2], u1[2], u2[2], u3[2];
      auto load_dab = [&](int kbl, uint4 (&dst)[2]) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) dst[hh] = rok[hh] ? ldg128(p.dab + (size_t)rr[hh] * 512 + 32 * kbl + 8 * q4) : make_uint4(0u, 0u, 0u, 0u);
      };
      load_dab(0, u0);
      load_dab(1, u1);
      load_dab(2, u2);
#pragma unroll 1
      for (int pass = 0; pass < 2; ++pass) {
        float acc[4][4][4];
#pragma unroll
        for (int gq = 0; gq < 4; ++gq)
#pragma unroll
          for (int j = 0; j < 4; ++j) { acc[gq][j][0] = acc[gq][j][1] = acc[gq][j][2] = acc[gq][j][3] = 0.f; }
#pragma unroll 1
        for (int kbl = 0; kbl < 8; ++kbl) {
          const int kb = 8 * pass + kbl;
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
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
#pragma unroll
          for (int gq = 0; gq < 4; ++gq)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const uint32_t v = pk2(acc[gq][j][2 * hh], acc[gq][j][2 * hh + 1]);
              dx2p[hh][gq][j] = pass == 0 ? v : padd(dx2p[hh][gq][j], v);
            }
      }
      pb_tail(p, dx2p, rr, rok, swo, wbq, lane, bsc, bos);
    }
  } else {
    // Wo^T once; Wu^T streams: chunk j of a tile = units 64 j .. 64 j + 63 (j < 4: da, j >= 4: db) of all 128 channels, 16 KB, shared by the CTA's warps
    for (int i = tid; i < 128 * 16; i += G::NTHR) cp_async16(swo + qf_woff(i >> 4, i & 15), p.wot + (size_t)qf_channel(i >> 4) * 128 + (i & 15) * 8);
    const int ngrp = (ntile + G::NW - 1) / G::NW;
    const int mine = (int)blockIdx.x < ngrp ? (ngrp - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
    const int total = mine * 8;
    auto load_w = [&](uint32_t stage, int j) {
      for (int i = tid; i < 128 * 8; i += G::NTHR) {
        const int r = i >> 3, ch = i & 7;
        cp_async16(stage + w128_off(r, ch), p.wut + (size_t)qf_channel(r) * 512 + 64 * j + ch * 8);
      }
    };
#pragma unroll
    for (int s = 0; s < G::NST - 1; ++s) {
      if (s < total) load_w(sw + s * G::W_BYTES, s & 7);
      cp_async_commit();
    }
    uint32_t wb2[2];
#pragma unroll
    for (int kbp = 0; kbp < 2; ++kbp) wb2[kbp] = w128_off(g8, 4 * kbp + q4);
    int rr[2];
    bool rok[2];
    uint32_t dx2p[2][4][4];
    float acc[4][4][4];
    uint4 an[2][2];                                                // the A fragments of the next chunk: [kb'][hh]
    auto load_a = [&](int j, uint4 (&dst)[2][2]) {
#pragma unroll
      for (int kbp = 0; kbp < 2; ++kbp)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) dst[kbp][hh] = rok[hh] ? ldg128(p.dab + (size_t)rr[hh] * 512 + 64 * j + 32 * kbp + 8 * q4) : make_uint4(0u, 0u, 0u, 0u);
    };
    for (int ci = 0; ci < total; ++ci) {
      cp_async_wait<G::NST - 2>();
      __syncthreads();                                             // chunk ci (and Wo^T) are in shared memory; the stage of chunk ci - 1 is free
      const int j = ci & 7, tl = ci >> 3, tile = ((int)blockIdx.x + tl * (int)gridDim.x) * G::NW + warp;
      {
        const int cn = ci + G::NST - 1;
        if (cn < total) load_w(sw + (cn % G::NST) * G::W_BYTES, cn & 7);
        cp_async_commit();
      }
      if (j == 0) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) { rr[hh] = 16 * tile + g8 + 8 * hh; rok[hh] = tile < ntile && rr[hh] < p.M; }
        load_a(0, an);
      }
      if (j == 0 || j == 4) {
#pragma unroll
        for (int gq = 0; gq < 4; ++gq)
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) acc[gq][jj][0] = acc[gq][jj][1] = acc[gq][jj][2] = acc[gq][jj][3] = 0.f;
      }
      uint4 au[2][2];
#pragma unroll
      for (int kbp = 0; kbp < 2; ++kbp)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) au[kbp][hh] = an[kbp][hh];
      if (j < 7) load_a(j + 1, an);                                // the next chunk's fragments (a new tile's first chunk is requested with its rows)
      const uint32_t wst = sw + (ci % G::NST) * G::W_BYTES;
#pragma unroll
      for (int kbp = 0; kbp < 2; ++kbp) {
        uint32_t a[2][4];
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) { a[0][hh] = au[kbp][hh].x; a[0][2 + hh] = au[kbp][hh].y; a[1][hh] = au[kbp][hh].z; a[1][2 + hh] = au[kbp][hh].w; }
#pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
          const uint4 b = lds128_ro(wst + nt * 8 * 128 + wb2[kbp]);
          mma16816(acc[nt >> 2][nt & 3], a[0], b.x, b.y);
          mma16816(acc[nt >> 2][nt & 3], a[1], b.z, b.w);
        }
      }
      if (j == 3 || j == 7) {                                      // a pass (da Wa, then db Wb) is complete
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
#pragma unroll
          for (int gq = 0; gq < 4; ++gq)
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
              const uint32_t v = pk2(acc[gq][jj][2 * hh], acc[gq][jj][2 * hh + 1]);
              dx2p[hh][gq][jj] = j == 3 ? v : padd(dx2p[hh][gq][jj], v);
            }
      }
      if (j == 7) pb_tail(p, dx2p, rr, rok, swo, wbq, lane, bsc, bos);
    }
  }

  // ---- column sums: d bsc2 -> dbias[512 ..], d bos -> dbias[768 ..] (the lane sums over the 8 rows of the tile, then one atomic per channel and warp)
#pragma unroll
  for (int kb = 0; kb < 4; ++kb)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float v = bsc[kb][i], w = bos[kb][i];
      v += __shfl_xor_sync(0xffffffffu, v, 4); w += __shfl_xor_sync(0xffffffffu, w, 4);
      v += __shfl_xor_sync(0xffffffffu, v, 8); w += __shfl_xor_sync(0xffffffffu, w, 8);
      v += __shfl_xor_sync(0xffffffffu, v, 16); w += __shfl_xor_sync(0xffffffffu, w, 16);
      if (g8 == 0) { atomicAdd(p.dbias + 512 + 32 * kb + 8 * q4 + i, v); atomicAdd(p.dbias + 768 + 32 * kb + 8 * q4 + i, w); }
    }
}

}  // namespace lr80
