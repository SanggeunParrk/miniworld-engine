// ct_atom_bwd.cuh -- the backward of the ConditionedTransition tail at the atom width (d_hidden = d_cond = 128, hidden 256), bf16 rows, A100 / sm_80, in two kernels (the forward is
// ``ct_atom_fwd.cuh``).  Forward, per row: [a | b] = xa [Wa; Wb]^T, h = rn(silu(a) b), z = h Ws^T (the forward stores rn(z)), g = cond Wsc^T + bsc, y = x + sigmoid(g) z.  With dy:
//
//   ct_atom_bwd_gate_kernel   g = sigmoid(cond Wsc^T + bsc)          recomputed (the forward's product, same instruction order)
//                             dz = rn(g dy)           -> ``dz``       (the A operand of dh, the weight-gradient operand of Ws)
//                             dg = rn(dy z g (1 - g)) -> ``dg``       (the weight-gradient operand of Wsc), its column sums -> d bsc
//                             dcond2 = dg Wsc         -> ``dcond2``   (the gate's share of d cond: the cond LayerNorm backward adds it)
//                             a, b = xa [Wa; Wb]^T recomputed in chunks of 32 hidden units (fp32, the forward's bits), dh = dz Ws, sa = sigmoid(a)
//                             h = rn(a sa b) -> ``hh`` (the weight-gradient operand of Ws),  da = rn(dh b sa (1 + a (1 - sa))), db = rn(dh a sa)  -> ``dab`` = da | db
//   ct_atom_bwd_dxa_kernel    dxa = rn(dab [Wa; Wb]) -> ``dxa``       (the gradient of the AdaLN's output: the AdaLN backward kernel takes it as its dy)
//
// The weight gradients dWa | dWb = dab^T xa, dWs = dz^T hh, dWsc = dg^T cond are cuBLAS GEMMs on these operands.  The gate kernel is the SWA atom DiT's FFN-backward gate kernel
// (``kernels/swa_dit/cuda/sm80/ffn_bwd_sm80.cuh``) with this block's gate in front; the dxa kernel is its dy kernel without the adaLN epilogue.
#pragma once
#include "adaln_atom_bwd.cuh"      // reduce_scatter_g8, g8_base (adaln_mma.cuh)

namespace adl {

// ================================================================== gate
struct CtAtomBwdGateParams {
  const bf* dy;                  // [M][128]
  const bf* z;                   // [M][128]  rn(z)
  const bf* cond;                // [M][128]
  const bf* xa;                  // [M][128]
  const bf* wab;                 // [512][128]  Wa | Wb  [hidden][in]
  const bf* wdt;                 // [256][128]  Ws^T: row = hidden unit, 128 output channels
  const bf* wsc;                 // [128][128]  gate weight [out][in]
  const bf* wsct;                // [128][128]  Wsc^T
  const bf* bsc;                 // [128]
  bf *dz, *dg, *dcond2;          // [M][128]
  bf* hh;                        // [M][256]
  bf* dab;                       // [M][512]
  float* pbsc;                   // [CTAs x NW][128] per-warp partial column sums of dg
  long M;
};

template <int NW_>
struct CtAtomBwdGateCfg {
  static constexpr int NW = NW_, NTHR = NW_ * 32, MINB = 1;
  static constexpr int HC = 32, NCH = 256 / HC, NST = 3;                    // hidden units per chunk, chunks per tile, ring depth
  static constexpr int STAGE = 3 * HC * 256;                                // Wa | Wb | Ws^T rows of a chunk
  static constexpr int SMEM = 2 * QF_ROWS * 256 + NST * STAGE;              // + the gate weight and its transpose
};

template <class G>
ADL_DEVI void bg_load_chunk(const CtAtomBwdGateParams& p, uint32_t stage, int j, int tid) {
  // smem row r of a chunk holds hidden unit 32 j + qf_channel(r) (r < 32: the within-group permutation of the f1 order)
  for (int i = tid; i < G::HC * 16; i += G::NTHR) {
    const int r = i >> 4, c = i & 15, hid = G::HC * j + qf_channel(r);
    const uint32_t off = qf_woff(r, c);
    cp_async16(stage + off, p.wab + (size_t)hid * 128 + c * 8);
    cp_async16(stage + G::HC * 256 + off, p.wab + (size_t)(256 + hid) * 128 + c * 8);
    cp_async16(stage + 2 * G::HC * 256 + off, p.wdt + (size_t)hid * 128 + c * 8);
  }
}

template <class G>
__global__ void __launch_bounds__(G::NTHR, G::MINB) ct_atom_bwd_gate_kernel(const CtAtomBwdGateParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sbase = smem_u32(smem_raw), swsc = sbase, swsct = sbase + QF_ROWS * 256, sring = sbase + 2 * QF_ROWS * 256;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  load_weight128(swsc, p.wsc, tid, G::NTHR);
  load_weight128(swsct, p.wsct, tid, G::NTHR);
  cp_async_commit();                                                          // group 0: the gate's weights

  uint32_t wbq[4];                                                            // B-fragment address of channel chunk kb in row g8 of a group of 8 rows
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) wbq[kb] = qf_woff(g8, 4 * kb + q4);

  const long ntile_all = (p.M + 15) / 16;
  const long ngrp = (ntile_all + G::NW - 1) / G::NW;                          // groups of NW tiles (one per warp): the CTA's iterations
  const long mine = (long)blockIdx.x < ngrp ? (ngrp - (long)blockIdx.x + (long)gridDim.x - 1) / (long)gridDim.x : 0;
  const long total = mine * G::NCH;                                           // chunks of all this CTA's iterations, one stream through the ring

#pragma unroll
  for (int s = 0; s < G::NST - 1; ++s) {
    if (s < total) bg_load_chunk<G>(p, sring + s * G::STAGE, s % G::NCH, tid);
    cp_async_commit();
  }

  float accbs[4] = {0.f, 0.f, 0.f, 0.f};                                      // d bsc: this lane's channel 32 gq + 8 q4 + g8_base(lane, 8), summed over the warp's tiles
  uint32_t ay[8][4], adz[8][4];                                               // A fragments of xa and of dz
  long rr[2] = {0, 0};
  bool rok[2] = {false, false};
  for (long ci = 0; ci < total; ++ci) {
    cp_async_wait<G::NST - 2>();
    __syncthreads();                                                          // chunk ci (and the gate weights) are in shared memory; the stage of chunk ci - 1 is free
    const int j = (int)(ci % G::NCH);
    const long it = ci / G::NCH;
    {
      const long cn = ci + G::NST - 1;
      if (cn < total) bg_load_chunk<G>(p, sring + (cn % G::NST) * G::STAGE, (int)(cn % G::NCH), tid);
      cp_async_commit();
    }
    if (j == 0) {                                                             // ---- a new tile: the gate, dz, dg, dcond2
      const long tile = ((long)blockIdx.x + it * gridDim.x) * G::NW + warp;
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) { rr[hh] = tile * 16 + g8 + 8 * hh; rok[hh] = tile < ntile_all && rr[hh] < p.M; }
      uint4 udy[2][4], uz[2][4], uc[2][4], uxa[2][4];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int gq = 0; gq < 4; ++gq) {
          udy[hh][gq] = uz[hh][gq] = uc[hh][gq] = uxa[hh][gq] = make_uint4(0u, 0u, 0u, 0u);
          if (rok[hh]) {
            const long off = rr[hh] * 128 + 32 * gq + 8 * q4;
            udy[hh][gq] = ldg128(p.dy + off);
            uz[hh][gq] = ldg128(p.z + off);
            uc[hh][gq] = ldg128(p.cond + off);
            uxa[hh][gq] = ldg128(p.xa + off);
          }
        }
      a_from_vec(ay, uxa);
      uint32_t aca[8][4], adg[8][4];
      a_from_vec(aca, uc);
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) {
        float gacc[4][4];
#pragma unroll
        for (int t = 0; t < 4; ++t) { gacc[t][0] = gacc[t][1] = gacc[t][2] = gacc[t][3] = 0.f; }
#pragma unroll
        for (int kb = 0; kb < 4; ++kb)
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            const uint4 b = lds128_ro(swsc + (4 * gq + t) * 8 * 256 + wbq[kb]);
            mma16816(gacc[t], aca[2 * kb], b.x, b.y);
            mma16816(gacc[t], aca[2 * kb + 1], b.z, b.w);
          }
        float bs[8], ds8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        unpack8(ldg128(p.bsc + 32 * gq + 8 * q4), bs);
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          float dyv[8], zv[8], dzv[8], dgv[8];
          unpack8(udy[hh][gq], dyv);
          unpack8(uz[hh][gq], zv);
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const float g = sigmoidf(gacc[i >> 1][2 * hh + (i & 1)] + bs[i]);
            dzv[i] = dyv[i] * g;
            dgv[i] = dyv[i] * zv[i] * g * (1.f - g);
            ds8[i] += dgv[i];
          }
          const uint4 udz = pack8(dzv), udg = pack8(dgv);
          adz[2 * gq][hh] = udz.x;      adz[2 * gq][2 + hh] = udz.y;
          adz[2 * gq + 1][hh] = udz.z;  adz[2 * gq + 1][2 + hh] = udz.w;
          adg[2 * gq][hh] = udg.x;      adg[2 * gq][2 + hh] = udg.y;
          adg[2 * gq + 1][hh] = udg.z;  adg[2 * gq + 1][2 + hh] = udg.w;
          if (rok[hh]) {
            stg128(p.dz + rr[hh] * 128 + 32 * gq + 8 * q4, udz);
            stg128(p.dg + rr[hh] * 128 + 32 * gq + 8 * q4, udg);
          }
        }
        reduce_scatter_g8<8>(ds8, lane);
        accbs[gq] += ds8[0];
      }
      // dcond2 = dg Wsc: the products over the d channels of dg with Wsc^T's rows (the cond channels in the f1 order)
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) {
        float acc2[4][4];
#pragma unroll
        for (int t = 0; t < 4; ++t) { acc2[t][0] = acc2[t][1] = acc2[t][2] = acc2[t][3] = 0.f; }
#pragma unroll
        for (int kb = 0; kb < 4; ++kb)
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            const uint4 b = lds128_ro(swsct + (4 * gq + t) * 8 * 256 + wbq[kb]);
            mma16816(acc2[t], adg[2 * kb], b.x, b.y);
            mma16816(acc2[t], adg[2 * kb + 1], b.z, b.w);
          }
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          float o[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) o[i] = acc2[i >> 1][2 * hh + (i & 1)];
          if (rok[hh]) stg128(p.dcond2 + rr[hh] * 128 + 32 * gq + 8 * q4, pack8(o));
        }
      }
    }

    // ---- chunk j: a, b = xa [Wa; Wb]^T and dh = dz Ws of hidden units 32 j .. 32 j + 31 (4 n tiles each; thread: units 32 j + 8 q4 .. + 7)
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
        mma16816(dc[t], adz[2 * kb], b3.x, b3.y);
        mma16816(dc[t], adz[2 * kb + 1], b3.z, b3.w);
      }
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      float hv[8], dav[8], dbv[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {                                            // unit 8 q4 + i = 2 t + e of the quad's pair: n tile t = i / 2, column e = i & 1
        const float av = ac[i >> 1][2 * hh + (i & 1)], bv = bc[i >> 1][2 * hh + (i & 1)], dh = dc[i >> 1][2 * hh + (i & 1)];
        const float sa = sigmoidf(av);
        hv[i] = (av * sa) * bv;
        dav[i] = ((dh * bv) * sa) * (1.f + av * (1.f - sa));
        dbv[i] = (dh * av) * sa;
      }
      if (rok[hh]) {
        const long u = (long)G::HC * j + 8 * q4;
        stg128(p.hh + rr[hh] * 256 + u, pack8(hv));
        stg128(p.dab + rr[hh] * 512 + u, pack8(dav));
        stg128(p.dab + rr[hh] * 512 + 256 + u, pack8(dbv));
      }
    }
  }
  cp_async_wait<0>();

  // the warps' column sums of dg, added through shared memory (all warps are done with it): one partial row per CTA
  __syncthreads();
  float* red = reinterpret_cast<float*>(smem_raw);                              // [NW][128]
  const int base = g8_base(lane, 8);
#pragma unroll
  for (int gq = 0; gq < 4; ++gq) red[warp * 128 + 32 * gq + 8 * q4 + base] = accbs[gq];
  __syncthreads();
  for (int i = tid; i < 128; i += G::NTHR) {
    float t = 0.f;
#pragma unroll
    for (int w = 0; w < G::NW; ++w) t += red[w * 128 + i];
    p.pbsc[(long)blockIdx.x * 128 + i] = t;
  }
}

// ================================================================== dxa
struct CtAtomBwdDxaParams {
  const bf* dab;                 // [M][512]  da | db
  const bf* wabt;                // [128 channels][512 hidden] = [Wa; Wb]^T
  bf* dxa;                       // [M][128]
  long M;
};

template <int NW_>
struct CtAtomBwdDxaCfg {
  static constexpr int NW = NW_, NTHR = NW_ * 32, MINB = 1, SMEM = 128 * 1024;
};

// byte offset of 16-byte chunk `chunk` (0 .. 63) of row `row` of the [128][1024 B] [Wa; Wb]^T tile: the chunk index is XOR-ed with 4 on odd rows (as qf_woff)
ADL_DEVI uint32_t wut_off(uint32_t row, uint32_t chunk) { return row * 1024u + ((chunk ^ ((row & 1u) << 2)) << 4); }

template <class G>
__global__ void __launch_bounds__(G::NTHR, G::MINB) ct_atom_bwd_dxa_kernel(const CtAtomBwdDxaParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sw = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  // [Wa; Wb]^T into shared memory in the f1 row order (smem row r holds output channel qf_channel(r)), once per CTA
  for (int i = tid; i < 128 * 64; i += G::NTHR) cp_async16(sw + wut_off(i >> 6, i & 63), p.wabt + (size_t)qf_channel(i >> 6) * 512 + (i & 63) * 8);
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  const uint32_t wlane = g8 * 1024u + q4 * 16u, gpar = g8 & 1u;               // the B fragment of hidden chunk kb (32 units) in row g8 of a group of 8 rows is at wlane + 64 (kb ^ gpar)

  const long ntile = (p.M + 15) / 16;
  for (long tile = (long)blockIdx.x * G::NW + warp; tile < ntile; tile += (long)gridDim.x * G::NW) {
    long rr[2];
    bool rok[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) { rr[hh] = tile * 16 + g8 + 8 * hh; rok[hh] = rr[hh] < p.M; }

    float acc[4][4][4];                                                       // the f1 order: 4 groups x 4 n tiles
#pragma unroll
    for (int gq = 0; gq < 4; ++gq)
#pragma unroll
      for (int j = 0; j < 4; ++j) { acc[gq][j][0] = acc[gq][j][1] = acc[gq][j][2] = acc[gq][j][3] = 0.f; }
    // the dab fragments of a block of 32 hidden units are requested three blocks ahead
    uint4 u0[2], u1[2], u2[2], u3[2];
    auto load_dab = [&](int kbl, uint4 (&dst)[2]) {
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) dst[hh] = rok[hh] ? ldg128(p.dab + rr[hh] * 512 + 32 * kbl + 8 * q4) : make_uint4(0u, 0u, 0u, 0u);
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
#pragma unroll
    for (int gq = 0; gq < 4; ++gq)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        float o[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = acc[gq][i >> 1][2 * hh + (i & 1)];
        if (rok[hh]) stg128(p.dxa + rr[hh] * 128 + 32 * gq + 8 * q4, pack8(o));
      }
  }
}

}  // namespace adl
