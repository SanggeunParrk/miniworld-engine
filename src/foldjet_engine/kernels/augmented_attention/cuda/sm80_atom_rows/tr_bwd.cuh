// tr_bwd.cuh -- the first part of the AF3 local atom DiT block's ConditionedTransition backward on A100 / sm_80 (the bf16 module's autograd rounding points; forward
// out = rn(a2 + rn(st t)), t = rn(h Ws^T), h = rn(rn(silu(a)) b), a | b = rn(x2 Wa^T) | rn(x2 Wb^T)):
//   dt = rn(dy st);  dst = rn(dy t);  dts = rn(rn(dst rn(1 - st)) st)                       (st = rn(sigmoid(ts)) of cond_fwd's mod, columns 896 ..)
//   a, b = recomputed from x2;  dh = rn(dt Ws);  sa = sigmoid(a);  sl = rn(a sa);  h = rn(sl b)
//   ds = rn(dh b);  db = rn(dh sl);  da = rn(ds sa (1 + a (1 - sa)))
//   -> DT [M, 128], HH [M, 256] (dWs = DT^T HH), DAB = [da | db] [M, 512] (dWu = DAB^T x2, and post_bwd's input), dts into dmod columns 896 .., d bts = the column sums of dts.
// Structure (the SWA ffn_bwd gate kernel's): a warp owns tiles of 16 rows; the A fragments of x2 and of dt stay in registers; Wa | Wb | Ws^T stream through a three-stage cp.async ring in
// chunks of 32 hidden units shared by the warps of a CTA (a thread's units 32 j + 8 q4 .. + 7: 4 n tiles of the f1 order).
#pragma once
#include "rows_layout.cuh"

namespace lr80 {

struct TrBwdParams {
  const __nv_bfloat16* dy;                           // [M][128]
  const __nv_bfloat16* t;                            // [M][128]
  const __nv_bfloat16* x2;                           // [M][128]
  const __nv_bfloat16* mod;                          // [M][1024]
  const __nv_bfloat16* wu;                           // [512][128]: Wa (rows 0..255) | Wb
  const __nv_bfloat16* wst;                          // [256][128] = Ws^T
  __nv_bfloat16 *dt, *hh, *dab;                      // [M][128], [M][256], [M][512]
  __nv_bfloat16* dmod;                               // [M][1024]; columns 896 .. 1023 are dts
  float* dbts;                                       // [128], accumulated (atomics, once per warp)
  int M;
  int prefetch;                                      // 1: the next tile's rows are requested into L2
};

template <int NW_, int MINB_>
struct TrBwdCfg {
  static constexpr int NW = NW_, NTHR = NW_ * 32, MINB = MINB_;
  static constexpr int HC = 32, NCH = 256 / HC, NST = 3;                    // hidden units per chunk, chunks per tile, ring depth
  static constexpr int STAGE = 3 * HC * 256, SMEM = NST * STAGE;           // Wa | Wb | Ws^T rows of a chunk
};

// the chunk loader's index math, once per thread: thread tid copies the 16-byte pieces i = tid + k NTHR of a chunk (row i >> 4 of the chunk, piece i & 15)
template <class G>
struct TbLoad {
  static constexpr int KN = G::HC * 16 / G::NTHR;
  int src[KN];                                                   // element offset of the piece in Wa's chunk-0 rows (hidden unit of the row * 128 + piece * 8)
  uint32_t dst[KN];                                              // byte offset in a stage
  DEVI void init(int tid) {
#pragma unroll
    for (int k = 0; k < KN; ++k) {
      const int i = tid + k * G::NTHR, r = i >> 4, c = i & 15;
      src[k] = qf_channel(r) * 128 + c * 8;
      dst[k] = qf_woff(r, c);
    }
  }
  DEVI void load(const TrBwdParams& p, uint32_t stage, int j) const {
    const size_t base = (size_t)G::HC * j * 128;
#pragma unroll
    for (int k = 0; k < KN; ++k) {
      cp_async16(stage + dst[k], p.wu + base + src[k]);
      cp_async16(stage + G::HC * 256 + dst[k], p.wu + (size_t)256 * 128 + base + src[k]);
      cp_async16(stage + 2 * G::HC * 256 + dst[k], p.wst + base + src[k]);
    }
  }
};

DEVI uint32_t one_minus(uint32_t s) {                  // rn(1 - s) per half
  return pk2(1.f - bf16lo(s), 1.f - bf16hi(s));
}

template <class G>
__global__ void __launch_bounds__(G::NTHR, G::MINB) tr_bwd_kernel(const TrBwdParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sring = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  TbLoad<G> ld;
  ld.init(tid);
  uint32_t wbq[4];                                               // B-fragment address of channel chunk kb in row g8 of a group of 8 rows
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) wbq[kb] = qf_woff(g8, 4 * kb + q4);

  const int ntile_all = (p.M + 15) / 16;
  const int ngrp = (ntile_all + G::NW - 1) / G::NW;                // groups of NW tiles (one per warp): the CTA's iterations
  const int mine = (int)blockIdx.x < ngrp ? (ngrp - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  const int total = mine * G::NCH;                                 // chunks of all this CTA's iterations, one stream through the ring
#pragma unroll
  for (int s = 0; s < G::NST - 1; ++s) {
    if (s < total) ld.load(p, sring + s * G::STAGE, s % G::NCH);
    cp_async_commit();
  }

  float dbs[4][8];                                                 // column sums of dts over the rows this lane saw (channels 32 kb + 8 q4 + i)
#pragma unroll
  for (int kb = 0; kb < 4; ++kb)
#pragma unroll
    for (int i = 0; i < 8; ++i) dbs[kb][i] = 0.f;

  uint32_t ay[8][4], adt[8][4];                                    // A fragments of x2 and of dt
  int rr[2];
  bool rok[2];
  __nv_bfloat16 *phh[2], *pdab[2];                                 // the tile's output row pointers (hh at the lane's units, dab = da | db)
  for (int ci = 0; ci < total; ++ci) {
    cp_async_wait<G::NST - 2>();
    __syncthreads();                                               // chunk ci is in shared memory; the stage of chunk ci - 1 is free
    const int j = ci % G::NCH, it = ci / G::NCH;
    {
      const int cn = ci + G::NST - 1;
      if (cn < total) ld.load(p, sring + (cn % G::NST) * G::STAGE, cn % G::NCH);
      cp_async_commit();
    }
    if (j == 0) {                                                  // ---- a new tile: x2, dt -> A fragments, dt / dts stored
      const int tile = ((int)blockIdx.x + it * (int)gridDim.x) * G::NW + warp;
      if (p.prefetch) {
        const int nt = tile + G::NW * (int)gridDim.x;
        if (nt < ntile_all) {
          pf_rows(p.dy, 256, 0, 2, 16 * nt, 16, p.M, lane);
          pf_rows(p.t, 256, 0, 2, 16 * nt, 16, p.M, lane);
          pf_rows(p.x2, 256, 0, 2, 16 * nt, 16, p.M, lane);
          pf_rows(p.mod, 2048, 1792, 2, 16 * nt, 16, p.M, lane);
        }
      }
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        rr[hh] = 16 * tile + g8 + 8 * hh;
        rok[hh] = tile < ntile_all && rr[hh] < p.M;
        phh[hh] = p.hh + (size_t)rr[hh] * 256 + 8 * q4;
        pdab[hh] = p.dab + (size_t)rr[hh] * 512 + 8 * q4;
      }
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
          uint4 ux = make_uint4(0u, 0u, 0u, 0u), ud = ux, ut = ux, us = ux;
          if (rok[hh]) {
            const size_t off = (size_t)rr[hh] * 128 + 32 * kb + 8 * q4;
            ux = ldg128(p.x2 + off);
            ud = ldg128(p.dy + off);
            ut = ldg128(p.t + off);
            us = ldg128(p.mod + (size_t)rr[hh] * 1024 + 896 + 32 * kb + 8 * q4);
          }
          put_a(ay, kb, hh, ux);
          const uint4 dtv = make_uint4(pmul(ud.x, us.x), pmul(ud.y, us.y), pmul(ud.z, us.z), pmul(ud.w, us.w));
          put_a(adt, kb, hh, dtv);
          const uint4 dst = make_uint4(pmul(ud.x, ut.x), pmul(ud.y, ut.y), pmul(ud.z, ut.z), pmul(ud.w, ut.w));
          const uint4 dts = make_uint4(pmul(pmul(dst.x, one_minus(us.x)), us.x), pmul(pmul(dst.y, one_minus(us.y)), us.y),
                                       pmul(pmul(dst.z, one_minus(us.z)), us.z), pmul(pmul(dst.w, one_minus(us.w)), us.w));
          if (rok[hh]) {
            stg128(p.dt + (size_t)rr[hh] * 128 + 32 * kb + 8 * q4, dtv);
            stg128(p.dmod + (size_t)rr[hh] * 1024 + 896 + 32 * kb + 8 * q4, dts);
            const uint32_t w[4] = {dts.x, dts.y, dts.z, dts.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) { dbs[kb][2 * e] += bf16lo(w[e]); dbs[kb][2 * e + 1] += bf16hi(w[e]); }
          }
        }
    }

    // ---- chunk j: a, b = x2 Wa^T, x2 Wb^T and dh = dt Ws of hidden units 32 j .. 32 j + 31 (4 n tiles each; thread: units 32 j + 8 q4 .. + 7)
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
        mma16816(bc[t], ay[2 * kb], b2.x, b2.y);
        mma16816(dc[t], adt[2 * kb], b3.x, b3.y);
        mma16816(ac[t], ay[2 * kb + 1], b1.z, b1.w);
        mma16816(bc[t], ay[2 * kb + 1], b2.z, b2.w);
        mma16816(dc[t], adt[2 * kb + 1], b3.z, b3.w);
      }
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      uint32_t hp[4], dap[4], dbp[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {                                // units 8 q4 + 2 e, + 1 of the quad's 8: n tile e, accumulator pair hh
        const uint32_t ar = pk2(ac[e][2 * hh], ac[e][2 * hh + 1]), br = pk2(bc[e][2 * hh], bc[e][2 * hh + 1]), dhr = pk2(dc[e][2 * hh], dc[e][2 * hh + 1]);
        const float a0 = bf16lo(ar), a1 = bf16hi(ar);
        const float s0 = sigm(a0), s1 = sigm(a1);
        const uint32_t sl = pk2(a0 * s0, a1 * s1);                 // rn(a sa)
        hp[e] = pmul(sl, br);                                      // h = rn(sl b)
        const uint32_t ds = pmul(dhr, br);                         // ds = rn(dh b)
        dbp[e] = pmul(dhr, sl);                                    // db = rn(dh sl)
        dap[e] = pk2(bf16lo(ds) * s0 * (1.f + a0 * (1.f - s0)), bf16hi(ds) * s1 * (1.f + a1 * (1.f - s1)));   // da = rn(ds sa (1 + a (1 - sa)))
      }
      if (rok[hh]) {
        stg128(phh[hh] + G::HC * j, make_uint4(hp[0], hp[1], hp[2], hp[3]));
        stg128(pdab[hh] + G::HC * j, make_uint4(dap[0], dap[1], dap[2], dap[3]));
        stg128(pdab[hh] + 256 + G::HC * j, make_uint4(dbp[0], dbp[1], dbp[2], dbp[3]));
      }
    }
  }

  // ---- d bts: the lane sums over the 8 rows of the tile (lanes with the same q4), then one atomic per channel and warp
#pragma unroll
  for (int kb = 0; kb < 4; ++kb)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float v = dbs[kb][i];
      v += __shfl_xor_sync(0xffffffffu, v, 4);
      v += __shfl_xor_sync(0xffffffffu, v, 8);
      v += __shfl_xor_sync(0xffffffffu, v, 16);
      if (g8 == 0) atomicAdd(p.dbts + 32 * kb + 8 * q4 + i, v);
    }
}

}  // namespace lr80
