// pre_bwd.cuh -- the attention-input backward of the AF3 local atom DiT block on A100 / sm_80 (the bf16 module's autograd rounding points), dP = [dq | dg | dk | dv] [M, 512]:
//   dxq = rn(dq Wq + dg Wg)  (one fp32 sum, one rounding: K = 256)       dxkv = rn(dk Wk + dv Wv)  (K = 256)
//   cross: xkv = rn(rn(skv xk) + bikv), xk = rn(LN(x1)):  dbikv = dxkv;  dskv = rn(dxkv xk);  dxhat = rn(dxkv skv);  dsckv = rn(rn(dskv rn(1 - skv)) skv);  dx1 = rn(dxq + rn(LN_bwd(dxhat)))
//   otherwise (K and V from x1 itself): dx1 = rn(dxq + dxkv)
//   x1 = rn(rn(s1 xn) + bi1), xn = rn(LN(x)):  dbi1 = dx1;  ds1 = rn(dx1 xn);  dxn = rn(dx1 s1);  dsc1 = rn(rn(ds1 rn(1 - s1)) s1);  d single = rn(DA + rn(LN_bwd(dxn)))
// -> d single [M, 128]; dmod columns 0 .. (dsc1), 128 .. (dbi1), 256 .. (dsckv), 384 .. (dbikv); the column sums of dsc1, dsckv (dbias[0 ..], dbias[256 ..]) and of dq (dbq).
// Structure (pre_fwd.cuh's frame): Wq^T | Wg^T and Wk^T | Wv^T ([128 channels][256 units] each, 64 KB) stay in shared memory; a warp owns tiles of 16 rows; the dP fragments of a block of 32 units are
// requested three blocks ahead; the accumulators are the row stages' operands (the f1 order).
#pragma once
#include "rows_layout.cuh"

namespace lr80 {

struct PreBwdParams {
  const __nv_bfloat16* dp;                           // [M][512]
  const __nv_bfloat16* x;                            // [M][128]
  const __nv_bfloat16* x1;                           // [M][128] (cross)
  const __nv_bfloat16* da;                           // [M][128]
  const __nv_bfloat16* mod;                          // [M][1024]
  const __nv_bfloat16* wqgt;                         // [128][256] = [Wq^T | Wg^T]
  const __nv_bfloat16* wkvt;                         // [128][256] = [Wk^T | Wv^T]
  __nv_bfloat16* ds;                                 // [M][128]
  __nv_bfloat16* dmod;                               // [M][1024]
  float* dbias;                                      // [1024]
  float* dbq;                                        // [128]
  int M, cross;
  float eps;
  int prefetch;                                      // 1: the next tile's rows are requested into L2
};

template <int NW_>
struct PreBwdCfg {
  static constexpr int NW = NW_, NTHR = NW_ * 32, SMEM = 2 * 128 * 512;
};

// byte offset of 16-byte chunk `chunk` (0 .. 31) of row `row` of a [128][512 B] tile (the chunk index is XOR-ed with 4 on odd rows)
DEVI uint32_t w512_off(uint32_t row, uint32_t chunk) { return row * 512u + ((chunk ^ ((row & 1u) << 2)) << 4); }

// acc = the 16 rows of dP columns [col0, col0 + 256) times the packed weight at ``wbase`` (K = 256 in 8 blocks of 32 units); optionally also the column sums of columns [0, 128) into dbq
template <bool SUMS>
DEVI void pb_product(const PreBwdParams& p, uint32_t wbase, int col0, const int (&rr)[2], const bool (&rok)[2], int g8, int q4, float (&acc)[4][4][4], float (&dbq)[4][8]) {
#pragma unroll
  for (int gq = 0; gq < 4; ++gq)
#pragma unroll
    for (int j = 0; j < 4; ++j) { acc[gq][j][0] = acc[gq][j][1] = acc[gq][j][2] = acc[gq][j][3] = 0.f; }
  const uint32_t wlane = g8 * 512u + q4 * 16u, gpar = g8 & 1u;
  uint4 u0[2], u1[2], u2[2], u3[2];
  auto load_blk = [&](int kb, uint4 (&dst)[2]) {
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) dst[hh] = rok[hh] ? ldg128(p.dp + (size_t)rr[hh] * 512 + col0 + 32 * kb + 8 * q4) : make_uint4(0u, 0u, 0u, 0u);
  };
  load_blk(0, u0);
  load_blk(1, u1);
  load_blk(2, u2);
#pragma unroll 1
  for (int kb = 0; kb < 8; ++kb) {
    if (kb + 3 < 8) load_blk(kb + 3, u3);
    if (SUMS && kb < 4) {
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        float v[8];
        unpack8(u0[hh], v);
#pragma unroll
        for (int i = 0; i < 8; ++i) dbq[kb][i] += v[i];
      }
    }
    uint32_t a[2][4];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) { a[0][hh] = u0[hh].x; a[0][2 + hh] = u0[hh].y; a[1][hh] = u0[hh].z; a[1][2 + hh] = u0[hh].w; }
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      const uint4 b = lds128_ro(wbase + nt * 8 * 512 + wlane + 64u * (kb ^ gpar));
      mma16816(acc[nt >> 2][nt & 3], a[0], b.x, b.y);
      mma16816(acc[nt >> 2][nt & 3], a[1], b.z, b.w);
    }
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) { u0[hh] = u1[hh]; u1[hh] = u2[hh]; u2[hh] = u3[hh]; }
  }
}

template <class G>
__global__ void __launch_bounds__(G::NTHR, 1) pre_bwd_kernel(const PreBwdParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sw = smem_u32(smem_raw), sk = sw + 128 * 512;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  for (int i = tid; i < 128 * 32; i += G::NTHR) {
    cp_async16(sw + w512_off(i >> 5, i & 31), p.wqgt + (size_t)qf_channel(i >> 5) * 256 + (i & 31) * 8);
    cp_async16(sk + w512_off(i >> 5, i & 31), p.wkvt + (size_t)qf_channel(i >> 5) * 256 + (i & 31) * 8);
  }
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  float bsc1[4][8], bskv[4][8], dbq[4][8];
#pragma unroll
  for (int kb = 0; kb < 4; ++kb)
#pragma unroll
    for (int i = 0; i < 8; ++i) { bsc1[kb][i] = 0.f; bskv[kb][i] = 0.f; dbq[kb][i] = 0.f; }

  const int ntile = (p.M + 15) / 16;
  for (int base = blockIdx.x * G::NW; base < ntile; base += G::NW * gridDim.x) {
    const int tile = base + warp;
    int rr[2];
    bool rok[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) { rr[hh] = 16 * tile + g8 + 8 * hh; rok[hh] = tile < ntile && rr[hh] < p.M; }
    if (p.prefetch) {
      const int nt = tile + G::NW * (int)gridDim.x;
      if (nt < ntile) {
        pf_rows(p.dp, 1024, 0, 8, 16 * nt, 16, p.M, lane);
        pf_rows(p.x, 256, 0, 2, 16 * nt, 16, p.M, lane);
        pf_rows(p.da, 256, 0, 2, 16 * nt, 16, p.M, lane);
        pf_rows(p.mod, 2048, 0, 2, 16 * nt, 16, p.M, lane);
        if (p.cross) {
          pf_rows(p.x1, 256, 0, 2, 16 * nt, 16, p.M, lane);
          pf_rows(p.mod, 2048, 512, 2, 16 * nt, 16, p.M, lane);
        }
      }
    }

    uint32_t dx1[2][4][4];                                         // [hh][group][j] bf16 pairs
    {
      uint32_t dxq[2][4][4], dxk[2][4][4];
      {
        float acc[4][4][4];
        pb_product<true>(p, sw, 0, rr, rok, g8, q4, acc, dbq);     // dxq = rn(dq Wq + dg Wg)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
#pragma unroll
          for (int gq = 0; gq < 4; ++gq)
#pragma unroll
            for (int j = 0; j < 4; ++j) dxq[hh][gq][j] = pk2(acc[gq][j][2 * hh], acc[gq][j][2 * hh + 1]);
        pb_product<false>(p, sk, 256, rr, rok, g8, q4, acc, dbq);  // dxkv = rn(dk Wk + dv Wv)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
#pragma unroll
          for (int gq = 0; gq < 4; ++gq)
#pragma unroll
            for (int j = 0; j < 4; ++j) dxk[hh][gq][j] = pk2(acc[gq][j][2 * hh], acc[gq][j][2 * hh + 1]);
      }
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const size_t row = (size_t)rr[hh];
        if (p.cross) {                                             // the key / value AdaLN's backward
          float xh[4][8];
#pragma unroll
          for (int gq = 0; gq < 4; ++gq) unpack8(rok[hh] ? ldg128(p.x1 + row * 128 + 32 * gq + 8 * q4) : make_uint4(0u, 0u, 0u, 0u), xh[gq]);
          const float rstd = ln_normalize(xh, p.eps);
          uint32_t dxhat[4][4];
#pragma unroll
          for (int gq = 0; gq < 4; ++gq) {
            const uint4 skv = rok[hh] ? ldg128(p.mod + row * 1024 + 256 + 32 * gq + 8 * q4) : make_uint4(0u, 0u, 0u, 0u);
            const uint32_t sc[4] = {skv.x, skv.y, skv.z, skv.w};
            uint32_t dsc[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const uint32_t xk = pk2(xh[gq][2 * e], xh[gq][2 * e + 1]);
              const uint32_t dskv = pmul(dxk[hh][gq][e], xk);
              dsc[e] = pmul(pmul(dskv, omin2(sc[e])), sc[e]);
              dxhat[gq][e] = pmul(dxk[hh][gq][e], sc[e]);
            }
            if (rok[hh]) {
              stg128(p.dmod + row * 1024 + 256 + 32 * gq + 8 * q4, make_uint4(dsc[0], dsc[1], dsc[2], dsc[3]));
              stg128(p.dmod + row * 1024 + 384 + 32 * gq + 8 * q4, make_uint4(dxk[hh][gq][0], dxk[hh][gq][1], dxk[hh][gq][2], dxk[hh][gq][3]));
#pragma unroll
              for (int e = 0; e < 4; ++e) { bskv[gq][2 * e] += bf16lo(dsc[e]); bskv[gq][2 * e + 1] += bf16hi(dsc[e]); }
            }
          }
          uint32_t lnk[4][4];
          ln_backward(xh, rstd, dxhat, lnk);
#pragma unroll
          for (int gq = 0; gq < 4; ++gq)
#pragma unroll
            for (int e = 0; e < 4; ++e) dx1[hh][gq][e] = padd(dxq[hh][gq][e], lnk[gq][e]);
        } else {
#pragma unroll
          for (int gq = 0; gq < 4; ++gq)
#pragma unroll
            for (int e = 0; e < 4; ++e) dx1[hh][gq][e] = padd(dxq[hh][gq][e], dxk[hh][gq][e]);
        }
      }
    }

    // ---- the AdaLN 1 backward and the residual: d single = rn(DA + rn(LN_bwd(s1 dx1)))
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const size_t row = (size_t)rr[hh];
      float xh[4][8];
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) unpack8(rok[hh] ? ldg128(p.x + row * 128 + 32 * gq + 8 * q4) : make_uint4(0u, 0u, 0u, 0u), xh[gq]);
      const float rstd = ln_normalize(xh, p.eps);
      uint32_t dxn[4][4];
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) {
        const uint4 s1v = rok[hh] ? ldg128(p.mod + row * 1024 + 32 * gq + 8 * q4) : make_uint4(0u, 0u, 0u, 0u);
        const uint32_t sc[4] = {s1v.x, s1v.y, s1v.z, s1v.w};
        uint32_t dsc[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const uint32_t xn = pk2(xh[gq][2 * e], xh[gq][2 * e + 1]);
          const uint32_t ds1 = pmul(dx1[hh][gq][e], xn);
          dsc[e] = pmul(pmul(ds1, omin2(sc[e])), sc[e]);
          dxn[gq][e] = pmul(dx1[hh][gq][e], sc[e]);
        }
        if (rok[hh]) {
          stg128(p.dmod + row * 1024 + 32 * gq + 8 * q4, make_uint4(dsc[0], dsc[1], dsc[2], dsc[3]));
          stg128(p.dmod + row * 1024 + 128 + 32 * gq + 8 * q4, make_uint4(dx1[hh][gq][0], dx1[hh][gq][1], dx1[hh][gq][2], dx1[hh][gq][3]));
#pragma unroll
          for (int e = 0; e < 4; ++e) { bsc1[gq][2 * e] += bf16lo(dsc[e]); bsc1[gq][2 * e + 1] += bf16hi(dsc[e]); }
        }
      }
      uint32_t ln[4][4];
      ln_backward(xh, rstd, dxn, ln);
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) {
        const uint4 uda = rok[hh] ? ldg128(p.da + row * 128 + 32 * gq + 8 * q4) : make_uint4(0u, 0u, 0u, 0u);
        if (rok[hh]) stg128(p.ds + row * 128 + 32 * gq + 8 * q4, make_uint4(padd(uda.x, ln[gq][0]), padd(uda.y, ln[gq][1]), padd(uda.z, ln[gq][2]), padd(uda.w, ln[gq][3])));
      }
    }
  }

  // ---- column sums: d bsc1 -> dbias[0 ..], d bskv -> dbias[256 ..], d bq (the lane sums over the 8 rows of the tile, then one atomic per channel and warp)
#pragma unroll
  for (int kb = 0; kb < 4; ++kb)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float v = bsc1[kb][i], w = bskv[kb][i], z = dbq[kb][i];
      v += __shfl_xor_sync(0xffffffffu, v, 4); w += __shfl_xor_sync(0xffffffffu, w, 4); z += __shfl_xor_sync(0xffffffffu, z, 4);
      v += __shfl_xor_sync(0xffffffffu, v, 8); w += __shfl_xor_sync(0xffffffffu, w, 8); z += __shfl_xor_sync(0xffffffffu, z, 8);
      v += __shfl_xor_sync(0xffffffffu, v, 16); w += __shfl_xor_sync(0xffffffffu, w, 16); z += __shfl_xor_sync(0xffffffffu, z, 16);
      if (g8 == 0) {
        atomicAdd(p.dbias + 32 * kb + 8 * q4 + i, v);
        atomicAdd(p.dbias + 256 + 32 * kb + 8 * q4 + i, w);
        atomicAdd(p.dbq + 32 * kb + 8 * q4 + i, z);
      }
    }
}

}  // namespace lr80
