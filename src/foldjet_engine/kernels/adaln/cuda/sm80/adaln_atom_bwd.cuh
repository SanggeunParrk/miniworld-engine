// adaln_atom_bwd.cuh -- AdaLN backward at the atom width (128 / 128), bf16 rows, A100 / sm_80, ONE kernel (the forward is ``adaln_atom_fwd.cuh``).  Per row, with the saved row
// statistics (mean, rstd) of x and of cond:
//
//   aff = rn(chat w), chat = (cond - mean_c) rstd_c                 recomputed, and stored: the operand of both weight gradients (cuBLAS)
//   scale = aff Ws^T (the forward's product, same instruction order), g = sigmoid(scale + sb), xh = (x - mean_x) rstd_x
//   dscale = dy xh g (1 - g)   (stored as bf16: the operand of dWs),   dxh = dy g
//   dx = rstd_x (dxh - mean(dxh) - xh mean(dxh xh)) (+ dres)
//   dcond_aff = dscale Ws + dy Wb                                  two products on the tensor cores, fp32 accumulators: never rounded, never stored
//   dcond = rstd_c (dcond_aff w - mean(dcond_aff w) - chat mean(dcond_aff w chat)) (+ dextra)
//
// The column sums leave as per-CTA partial rows: d sb = sum dscale, d w = sum dcond_aff chat (fp32; the host adds the rows).  dWs = dscale^T aff and dWb = dy^T aff are GEMMs over all
// rows and stay with cuBLAS.  One CTA of 8 warps per SM, each warp loops over tiles of 16 rows (the layout of adaln_mma.cuh); the three weights (Ws for the recomputed gate, Ws^T and
// Wb^T for the product with dscale / dy) live in shared memory in the f1 row order, and dxh waits in a per-warp shared buffer between the row sums and the dx pass.
#pragma once
#include "adaln_mma.cuh"

namespace adl {

// Sums the NV values of every lane over the 8 lanes that share lane & 3 (lane bits 2, 3, 4: the rows g8 of an mma tile) and scatters the sums: afterwards v[0 .. NV / 8) of a lane hold the
// sums of the values base .. base + NV / 8 - 1, base = NV / 2 b2 + NV / 4 b3 + NV / 8 b4 (b2, b3, b4: the lane's bits 2, 3, 4).  NV - NV / 8 shuffles per lane.
template <int NV>
ADL_DEVI void reduce_scatter_g8(float (&v)[NV], int lane) {
  static_assert(NV % 8 == 0, "NV must be a multiple of 8");
  const bool b2 = (lane & 4) != 0, b3 = (lane & 8) != 0, b4 = (lane & 16) != 0;
#pragma unroll
  for (int i = 0; i < NV / 2; ++i) {
    const float lo = v[i], hi = v[NV / 2 + i];
    v[i] = (b2 ? hi : lo) + __shfl_xor_sync(0xffffffffu, b2 ? lo : hi, 4);
  }
#pragma unroll
  for (int i = 0; i < NV / 4; ++i) {
    const float lo = v[i], hi = v[NV / 4 + i];
    v[i] = (b3 ? hi : lo) + __shfl_xor_sync(0xffffffffu, b3 ? lo : hi, 8);
  }
#pragma unroll
  for (int i = 0; i < NV / 8; ++i) {
    const float lo = v[i], hi = v[NV / 8 + i];
    v[i] = (b4 ? hi : lo) + __shfl_xor_sync(0xffffffffu, b4 ? lo : hi, 16);
  }
}
ADL_DEVI int g8_base(int lane, int nv) { return (nv / 2) * ((lane >> 2) & 1) + (nv / 4) * ((lane >> 3) & 1) + (nv / 8) * ((lane >> 4) & 1); }

struct AdalnAtomBwdParams {
  const bf* dy;                  // [M][128]
  const bf* x;                   // [M][128]
  const bf* cond;                // [M][128]
  const float2* xst;             // [M] (mean, rstd)
  const float2* cst;             // [M]
  const float* lnw;              // [128]
  const bf* ws;                  // [128][128] to_scale.weight [out][in]
  const bf* wt1;                 // [128][128] = Ws^T  ([in][out]: row c holds Ws[:, c])
  const bf* wt2;                 // [128][128] = Wb^T
  const bf* sb;                  // [128]
  const bf* dres;                // [M][128] or nullptr: added to dx
  const bf* dextra;              // [M][128] or nullptr: added to dcond
  bf *dx, *dcond, *dsc, *aff;    // [M][128]
  float *psb, *plnw;             // [CTAs x NW][128]: per-warp partial column sums
  long M;
};

template <int NW_>
struct AdalnAtomBwdCfg {
  static constexpr int NW = NW_, NTHR = NW_ * 32, MINB = 1;
  static constexpr int WBYTES = 3 * QF_ROWS * 256, SPILL = 8192;                 // Ws | Ws^T | Wb^T; per warp 16 rows x 128 fp32 (dxh)
  static constexpr int SMEM = WBYTES + NW_ * SPILL;
};

template <class G>
__global__ void __launch_bounds__(G::NTHR, 1) adaln_atom_bwd_kernel(const AdalnAtomBwdParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sbase = smem_u32(smem_raw), sws = sbase, swt1 = sbase + QF_ROWS * 256, swt2 = sbase + 2 * QF_ROWS * 256;
  float4* spill = reinterpret_cast<float4*>(smem_raw + G::WBYTES + (threadIdx.x >> 5) * G::SPILL);       // this warp's dxh: slot (gq, hh, half) of lane l at [(2 (2 gq + hh) + half) 32 + l]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  load_weight128(sws, p.ws, tid, G::NTHR);
  load_weight128(swt1, p.wt1, tid, G::NTHR);
  load_weight128(swt2, p.wt2, tid, G::NTHR);
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  uint32_t wbq[4];
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) wbq[kb] = qf_woff(g8, 4 * kb + q4);

  float accsb[4] = {0.f, 0.f, 0.f, 0.f}, acclw[4] = {0.f, 0.f, 0.f, 0.f};      // this lane's channel 32 gq + 8 q4 + g8_base(lane, 8) of d sb / d w, summed over the warp's tiles
  const long ntile = (p.M + 15) / 16;
  for (long tile = (long)blockIdx.x * G::NW + warp; tile < ntile; tile += (long)gridDim.x * G::NW) {
    long rr[2];
    bool rok[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) { rr[hh] = tile * 16 + g8 + 8 * hh; rok[hh] = rr[hh] < p.M; }
    // only dy stays in registers for the whole tile (it is the A operand of the last product); x and cond are read again where they are needed (L2 hits: the registers they would hold
    // were what made the 255-register kernel spill)
    uint4 udy[2][4];
    float2 xs[2], cs[2];
    auto ldrow = [&](const bf* base, int hh, int gq) -> uint4 { return rok[hh] ? ldg128(base + rr[hh] * 128 + 32 * gq + 8 * q4) : make_uint4(0u, 0u, 0u, 0u); };
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      xs[hh] = cs[hh] = make_float2(0.f, 1.f);
      if (rok[hh]) { xs[hh] = p.xst[rr[hh]]; cs[hh] = p.cst[rr[hh]]; }
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) udy[hh][gq] = ldrow(p.dy, hh, gq);
    }

    // ---- aff = rn(chat w): stored, and the A fragments of the recomputed forward product
    uint32_t aaff[8][4];
    {
      uint4 uaff[2][4];
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) {
        const float4 w0 = ldg_f4(p.lnw + 32 * gq + 8 * q4), w1 = ldg_f4(p.lnw + 32 * gq + 8 * q4 + 4);
        const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          float v[8], t[8];
          unpack8(ldrow(p.cond, hh, gq), v);
#pragma unroll
          for (int i = 0; i < 8; ++i) t[i] = (v[i] - cs[hh].x) * cs[hh].y * wv[i];
          uaff[hh][gq] = pack8(t);
          if (rok[hh]) stg128(p.aff + rr[hh] * 128 + 32 * gq + 8 * q4, uaff[hh][gq]);
        }
      }
      a_from_vec(aaff, uaff);
    }

    // ---- pass 1, per group of 32 channels: scale = aff Ws^T, the gate, dscale (-> A fragments of the next product, global), dxh (-> shared), the row sums of dxh and dxh xh
    uint32_t ads[8][4];
    float s1[2] = {0.f, 0.f}, s2[2] = {0.f, 0.f};
#pragma unroll
    for (int gq = 0; gq < 4; ++gq) {
      float as[4][4];
#pragma unroll
      for (int t = 0; t < 4; ++t) { as[t][0] = as[t][1] = as[t][2] = as[t][3] = 0.f; }
#pragma unroll
      for (int kb = 0; kb < 4; ++kb)
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          const uint4 b = lds128_ro(sws + (4 * gq + t) * 8 * 256 + wbq[kb]);
          mma16816(as[t], aaff[2 * kb], b.x, b.y);
          mma16816(as[t], aaff[2 * kb + 1], b.z, b.w);
        }
      float sbv[8];
      unpack8(ldg128(p.sb + 32 * gq + 8 * q4), sbv);
      float ds8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        float dyv[8], xv[8], dsc[8], dxh[8];
        unpack8(udy[hh][gq], dyv);
        unpack8(ldrow(p.x, hh, gq), xv);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float g = sigmoidf(as[i >> 1][2 * hh + (i & 1)] + sbv[i]);
          const float xh = (xv[i] - xs[hh].x) * xs[hh].y;
          dxh[i] = dyv[i] * g;
          dsc[i] = dyv[i] * xh * g * (1.f - g);
          s1[hh] += dxh[i];
          s2[hh] = fmaf(dxh[i], xh, s2[hh]);
          ds8[i] += dsc[i];
        }
        const uint4 uds = pack8(dsc);
        ads[2 * gq][hh] = uds.x;      ads[2 * gq][2 + hh] = uds.y;
        ads[2 * gq + 1][hh] = uds.z;  ads[2 * gq + 1][2 + hh] = uds.w;
        if (rok[hh]) stg128(p.dsc + rr[hh] * 128 + 32 * gq + 8 * q4, uds);
        spill[(2 * (2 * gq + hh) + 0) * 32 + lane] = make_float4(dxh[0], dxh[1], dxh[2], dxh[3]);
        spill[(2 * (2 * gq + hh) + 1) * 32 + lane] = make_float4(dxh[4], dxh[5], dxh[6], dxh[7]);
      }
      reduce_scatter_g8<8>(ds8, lane);
      accsb[gq] += ds8[0];
    }
    float m1[2], m2[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) { m1[hh] = quad_sum(s1[hh]) * (1.f / 128.f); m2[hh] = quad_sum(s2[hh]) * (1.f / 128.f); }
    __syncwarp();

    // ---- pass 2: dx = rstd (dxh - m1 - xh m2) (+ dres)
#pragma unroll
    for (int gq = 0; gq < 4; ++gq)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const float4 d0 = spill[(2 * (2 * gq + hh) + 0) * 32 + lane], d1 = spill[(2 * (2 * gq + hh) + 1) * 32 + lane];
        const float dxh[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
        float xv[8], rv[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, o[8];
        unpack8(ldrow(p.x, hh, gq), xv);
        if (p.dres != nullptr && rok[hh]) unpack8(ldg128(p.dres + rr[hh] * 128 + 32 * gq + 8 * q4), rv);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float xh = (xv[i] - xs[hh].x) * xs[hh].y;
          o[i] = fmaf(xs[hh].y, dxh[i] - m1[hh] - xh * m2[hh], rv[i]);
        }
        if (rok[hh]) stg128(p.dx + rr[hh] * 128 + 32 * gq + 8 * q4, pack8(o));
      }

    // ---- dcond_aff = dscale Ws + dy Wb: fp32 accumulators in the f1 order; the cond rows are requested first
    uint4 uc2[2][4];                                                           // the cond rows again, for the LayerNorm backward after the products
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
      for (int gq = 0; gq < 4; ++gq) uc2[hh][gq] = ldrow(p.cond, hh, gq);
    uint32_t ady[8][4];
    a_from_vec(ady, udy);
    float acc[4][4][4];
#pragma unroll
    for (int gq = 0; gq < 4; ++gq)
#pragma unroll
      for (int t = 0; t < 4; ++t) { acc[gq][t][0] = acc[gq][t][1] = acc[gq][t][2] = acc[gq][t][3] = 0.f; }
#pragma unroll
    for (int gq = 0; gq < 4; ++gq)
#pragma unroll
      for (int kb = 0; kb < 4; ++kb)
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          const uint32_t off = (4 * gq + t) * 8 * 256 + wbq[kb];
          const uint4 b1 = lds128_ro(swt1 + off), b2 = lds128_ro(swt2 + off);
          mma16816(acc[gq][t], ads[2 * kb], b1.x, b1.y);
          mma16816(acc[gq][t], ads[2 * kb + 1], b1.z, b1.w);
          mma16816(acc[gq][t], ady[2 * kb], b2.x, b2.y);
          mma16816(acc[gq][t], ady[2 * kb + 1], b2.z, b2.w);
        }

    // ---- the cond LayerNorm backward: dh = dcond_aff w, row sums of dh and dh chat, dcond (+ dextra), the column sums of dcond_aff chat
    float sdh[2] = {0.f, 0.f}, sdc[2] = {0.f, 0.f};
#pragma unroll
    for (int gq = 0; gq < 4; ++gq) {
      const float4 w0 = ldg_f4(p.lnw + 32 * gq + 8 * q4), w1 = ldg_f4(p.lnw + 32 * gq + 8 * q4 + 4);
      const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        float cv[8];
        unpack8(uc2[hh][gq], cv);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float dh = acc[gq][i >> 1][2 * hh + (i & 1)] * wv[i];
          sdh[hh] += dh;
          sdc[hh] = fmaf(dh, (cv[i] - cs[hh].x) * cs[hh].y, sdc[hh]);
        }
      }
    }
    float n1[2], n2[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) { n1[hh] = quad_sum(sdh[hh]) * (1.f / 128.f); n2[hh] = quad_sum(sdc[hh]) * (1.f / 128.f); }
#pragma unroll
    for (int gq = 0; gq < 4; ++gq) {
      const float4 w0 = ldg_f4(p.lnw + 32 * gq + 8 * q4), w1 = ldg_f4(p.lnw + 32 * gq + 8 * q4 + 4);
      const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
      float lw8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        float cv[8], ev[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, o[8];
        unpack8(uc2[hh][gq], cv);
        if (p.dextra != nullptr && rok[hh]) unpack8(ldg128(p.dextra + rr[hh] * 128 + 32 * gq + 8 * q4), ev);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float dca = acc[gq][i >> 1][2 * hh + (i & 1)];
          const float chat = (cv[i] - cs[hh].x) * cs[hh].y;
          o[i] = fmaf(cs[hh].y, dca * wv[i] - n1[hh] - chat * n2[hh], ev[i]);
          lw8[i] += dca * chat;
        }
        if (rok[hh]) stg128(p.dcond + rr[hh] * 128 + 32 * gq + 8 * q4, pack8(o));
      }
      reduce_scatter_g8<8>(lw8, lane);
      acclw[gq] += lw8[0];
    }
    __syncwarp();                                                              // the spill is reused by the warp's next tile
  }

  // the warps' column sums, added through shared memory (all warps are done with it): one partial row per CTA
  __syncthreads();
  float* red = reinterpret_cast<float*>(smem_raw);                              // [NW][2][128]
  const int base = g8_base(lane, 8);
#pragma unroll
  for (int gq = 0; gq < 4; ++gq) {
    red[(warp * 2) * 128 + 32 * gq + 8 * q4 + base] = accsb[gq];
    red[(warp * 2 + 1) * 128 + 32 * gq + 8 * q4 + base] = acclw[gq];
  }
  __syncthreads();
  for (int i = tid; i < 256; i += G::NTHR) {
    const int which = i >> 7, ch = i & 127;
    float t = 0.f;
#pragma unroll
    for (int w = 0; w < G::NW; ++w) t += red[(w * 2 + which) * 128 + ch];
    (which ? p.plnw : p.psb)[(long)blockIdx.x * 128 + ch] = t;
  }
}

}  // namespace adl
