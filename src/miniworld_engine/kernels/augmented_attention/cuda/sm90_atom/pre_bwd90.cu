// pre_bwd90.cu — backward of the q | gate and k | v projections and of the two AdaLNs (sm_90a, wgmma), per 16-row warp tile:
//   dxkv = dk Wk + dv Wv          dx1 = dg Wg + dq Wq        (dp = [dg | dq | dk | dv], K = 128 each)
//   n1 = LN(x1)  sk = sigmoid(pre + bs):  d shift = dxkv -> dmkv[:, 128:256],  d pre = dxkv n1 sk (1 - sk) -> dmkv[:, 0:128]
//   dx1 += LN'(dxkv sk)
//   n0 = LN(x):  d bi1 = dx1 -> dmod[:, 128:256],  d s1pre = dx1 n0 s1 (1 - s1) -> dmod[:, 0:128]
//   dx = da2 + LN'(dx1 s1)
// SPDX-License-Identifier: Apache-2.0
#include "row90.cuh"
using namespace r90;

// dn -> LN backward given the normalised row n: out = rstd (dn - mean(dn) - n mean(dn n))
DEVI void ln_bwd(Tile& out, const Tile& dn, const Tile& n, float r0, float r1) {
  float p0 = 0.f, q0 = 0.f, p1 = 0.f, q1 = 0.f;
#pragma unroll
  for (int nt = 0; nt < 16; ++nt) {
    p0 += dn.f[nt][0] + dn.f[nt][1]; q0 += dn.f[nt][0] * n.f[nt][0] + dn.f[nt][1] * n.f[nt][1];
    p1 += dn.f[nt][2] + dn.f[nt][3]; q1 += dn.f[nt][2] * n.f[nt][2] + dn.f[nt][3] * n.f[nt][3];
  }
  p0 = quad_sum(p0) * (1.f / 128.f); q0 = quad_sum(q0) * (1.f / 128.f);
  p1 = quad_sum(p1) * (1.f / 128.f); q1 = quad_sum(q1) * (1.f / 128.f);
#pragma unroll
  for (int nt = 0; nt < 16; ++nt) {
    out.f[nt][0] = r0 * (dn.f[nt][0] - p0 - n.f[nt][0] * q0);
    out.f[nt][1] = r0 * (dn.f[nt][1] - p0 - n.f[nt][1] * q0);
    out.f[nt][2] = r1 * (dn.f[nt][2] - p1 - n.f[nt][2] * q1);
    out.f[nt][3] = r1 * (dn.f[nt][3] - p1 - n.f[nt][3] * q1);
  }
}

extern "C" __global__ __launch_bounds__(NTHR, MINB) void pre_bwd90(
    const T* __restrict__ dp, const T* __restrict__ x1, const T* __restrict__ x, const T* __restrict__ mod, const T* __restrict__ mkv,
    const T* __restrict__ da2, const T* __restrict__ wbt /*[4 kb: Wg Wq Wk Wv][2 half][64,128]: W^T tiles*/, const float* __restrict__ bs,
    T* __restrict__ dmod, T* __restrict__ dmkv, T* __restrict__ dx_out, float* __restrict__ colpart, int M, int nmod, float eps, float* __restrict__ dmod_acc, float* __restrict__ dmkv_acc, int spc, int natoms) {
  pdl_wait();
  extern __shared__ uint8_t sm_raw[];
  constexpr int UNIT = 64 * 128 * ES;
  const uint32_t base = (su(sm_raw) + 1023u) & ~1023u;
  float* red = reinterpret_cast<float*>(sm_raw + (base - su(sm_raw)) + 2 * UNIT);  // [3 sets][8 warps][128]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const bool shared = spc > 0;   // grouped-sample mode (spc > 0: shared conditioning, samples a multiple of 4): warp w = sample w & 3 of a group of four, summed in the CTA
  const size_t atom0 = (size_t)blockIdx.x * ROWS + warp * 16;
  for (int i = tid; i < NWARP * 128 * 3; i += NTHR) red[i] = 0.f;
  __syncthreads();
  // step order: kb = 2 (dk Wk), 3 (dv Wv), then 0 (dg Wg), 1 (dq Wq); 2 halves each
  auto issue = [&](int s) {
    const int kb = s < 4 ? 2 + (s >> 1) : (s - 4) >> 1, hf = s & 1;
    load_w(base + (s & 1) * UNIT, wbt + ((size_t)kb * 2 + hf) * 64 * 128, 128, 64, 128, tid, NTHR);
    cp_commit();
  };
  const int nsmp = 1;   // grouped mode: the samples are the warps of the CTA (see put_grp)
  for (int smp = 0; smp < nsmp; ++smp) {
  const int nacg = shared ? natoms / (16 * (NWARP / 4)) : 1;
  const int acg = (int)blockIdx.x % nacg, sg4 = (int)blockIdx.x / nacg;
  const size_t row0 = shared ? (size_t)(sg4 * 4 + (warp & 3)) * natoms + (size_t)(acg * (NWARP / 4) + (warp >> 2)) * 16 : atom0;
  const size_t mrow = row0 % (size_t)nmod;   // rows of the modulation tables (nmod = M: per sample; = N: shared)
  const size_t prow = (size_t)sg4 * natoms + (size_t)acg * (NWARP / 4) * 16;   // partial row of this CTA's first atom block
  T* const gred = reinterpret_cast<T*>(sm_raw + (base - su(sm_raw)) + 2 * UNIT + 3 * NWARP * 128 * 4);
  int gpar = 0;
#define GFLUSH(DST, LD, COL) if (shared) { grp_flush(reinterpret_cast<T*>(DST), prow, LD, COL, gred + (size_t)gpar * FBUF, tid); gpar ^= GNB - 1; }
  issue(0);
  issue(1);
  Tile c, lkeep;
  float acc0[32], acc1[32];
#pragma unroll
  for (int ph = 0; ph < 4; ++ph) {  // ph 0, 1: kb 2, 3; ph 2, 3: kb 0, 1
    const int kb = ph < 2 ? 2 + ph : ph - 2;
    ld_tile(c, dp + row0 * 512, 512, 128 * kb, g, t);
    if (ph == 3) col_partial(c, red + 2 * NWARP * 128, warp, lane, t);  // dq: d bias of the query projection
    rna(c);
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const int s = 2 * ph + hf;
      cp_wait<1>();
      proxy_fence();
      __syncthreads();
      wg_gemm<64, KSTEPS>(hf ? acc1 : acc0, c, base + (s & 1) * UNIT, (ph & 1) > 0);
      __syncthreads();
      if (s + 2 < 8) issue(s + 2); else cp_commit();
    }
    if (ph == 1) {  // dxkv complete -> the K / V AdaLN backward
      // c (the A tile of dv) is dead: it holds dxkv
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) { c.f[nt][e] = round_store(acc0[4 * nt + e]); c.f[8 + nt][e] = round_store(acc1[4 * nt + e]); }
      Tile n1;
      ld_tile(n1, x1 + row0 * 128, 128, 0, g, t);
      float m0, r0, m1, r1;
      ln_stats(n1, eps, m0, r0, m1, r1);
      float p0 = 0.f, q0 = 0.f, p1 = 0.f, q1 = 0.f;
#pragma unroll
      for (int H = 0; H < 4; ++H) {
        float sk[4][4];
        ld_nt<4>(sk, mkv + mrow * 256, 256, 32 * H, g, t);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int nt = 4 * H + i;
          const float2 bb = colv(bs, nt, t);
          const float bx[4] = {bb.x, bb.y, bb.x, bb.y};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float sv = sigm(sk[i][e] + bx[e]);
            const float nv = (n1.f[nt][e] - (e < 2 ? m0 : m1)) * (e < 2 ? r0 : r1);
            n1.f[nt][e] = nv;
            const float dn = c.f[nt][e] * sv;
            if (e < 2) { p0 += dn; q0 += dn * nv; } else { p1 += dn; q1 += dn * nv; }
          }
        }
      }
      p0 = quad_sum(p0) * (1.f / 128.f); q0 = quad_sum(q0) * (1.f / 128.f);
      p1 = quad_sum(p1) * (1.f / 128.f); q1 = quad_sum(q1) * (1.f / 128.f);
      if (shared) grp_put<16>(c.f, gred + (size_t)gpar * FBUF, 0, warp, g, t); else st_nt<16, false>(c.f, dmkv + row0 * 256, 256, 128, g, t);
      GFLUSH(dmkv_acc, 256, 128);  // d shift
#pragma unroll
      for (int H = 0; H < 4; ++H) {
        float sk[4][4], dpre[4][4];
        ld_nt<4>(sk, mkv + mrow * 256, 256, 32 * H, g, t);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int nt = 4 * H + i;
          const float2 bb = colv(bs, nt, t);
          const float bx[4] = {bb.x, bb.y, bb.x, bb.y};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float sv = sigm(sk[i][e] + bx[e]);
            const float nv = n1.f[nt][e], dx = c.f[nt][e];
            dpre[i][e] = dx * nv * sv * (1.f - sv);
            lkeep.f[nt][e] = (e < 2 ? r0 : r1) * (dx * sv - (e < 2 ? p0 : p1) - nv * (e < 2 ? q0 : q1));
          }
        }
        if (shared) grp_put<4>(dpre, gred + (size_t)gpar * FBUF, 32 * H, warp, g, t); else st_nt<4, false>(dpre, dmkv + row0 * 256, 256, 32 * H, g, t);
        col_partial_nt<4>(dpre, 4 * H, red + NWARP * 128, warp, lane, t);
      }
      GFLUSH(dmkv_acc, 256, 0);
    }
  }
  // acc0 / acc1 = dg Wg + dq Wq halves; dx1 = that + the K / V branch's contribution (into c)
#pragma unroll
  for (int nt = 0; nt < 8; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      c.f[nt][e] = round_store(acc0[4 * nt + e] + lkeep.f[nt][e]);
      c.f[8 + nt][e] = round_store(acc1[4 * nt + e] + lkeep.f[8 + nt][e]);
    }
  {  // AdaLN 1
    Tile n0;
    ld_tile(n0, x + row0 * 128, 128, 0, g, t);
    float m0, r0, m1, r1;
    ln_stats(n0, eps, m0, r0, m1, r1);
    float p0 = 0.f, q0 = 0.f, p1 = 0.f, q1 = 0.f;
#pragma unroll
    for (int H = 0; H < 4; ++H) {
      float s1[4][4];
      ld_nt<4>(s1, mod + mrow * 768, 768, 32 * H, g, t);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int nt = 4 * H + i;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float nv = (n0.f[nt][e] - (e < 2 ? m0 : m1)) * (e < 2 ? r0 : r1);
          n0.f[nt][e] = nv;
          const float dn = c.f[nt][e] * s1[i][e];
          if (e < 2) { p0 += dn; q0 += dn * nv; } else { p1 += dn; q1 += dn * nv; }
        }
      }
    }
    p0 = quad_sum(p0) * (1.f / 128.f); q0 = quad_sum(q0) * (1.f / 128.f);
    p1 = quad_sum(p1) * (1.f / 128.f); q1 = quad_sum(q1) * (1.f / 128.f);
    if (shared) grp_put<16>(c.f, gred + (size_t)gpar * FBUF, 0, warp, g, t); else st_nt<16, false>(c.f, dmod + row0 * 768, 768, 128, g, t);
    GFLUSH(dmod_acc, 768, 128);  // d bi1
#pragma unroll
    for (int H = 0; H < 4; ++H) {
      float s1[4][4], d2[4][4], ds1[4][4], o[4][4];
      ld_nt<4>(s1, mod + mrow * 768, 768, 32 * H, g, t);
      ld_nt<4>(d2, da2 + row0 * 128, 128, 32 * H, g, t);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int nt = 4 * H + i;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float nv = n0.f[nt][e], sv = s1[i][e], dx = c.f[nt][e];
          ds1[i][e] = dx * nv * sv * (1.f - sv);
          o[i][e] = d2[i][e] + (e < 2 ? r0 : r1) * (dx * sv - (e < 2 ? p0 : p1) - nv * (e < 2 ? q0 : q1));
        }
      }
      if (shared) grp_put<4>(ds1, gred + (size_t)gpar * FBUF, 32 * H, warp, g, t); else st_nt<4, false>(ds1, dmod + row0 * 768, 768, 32 * H, g, t);
      col_partial_nt<4>(ds1, 4 * H, red, warp, lane, t);
      st_nt<4, false>(o, dx_out + row0 * 128, 128, 32 * H, g, t);
    }
    GFLUSH(dmod_acc, 768, 0);
  }
  __syncthreads();
  }  // samples
  col_finish(red, colpart + ((size_t)blockIdx.y * gridDim.x + blockIdx.x) * 384, tid);
  col_finish(red + NWARP * 128, colpart + ((size_t)blockIdx.y * gridDim.x + blockIdx.x) * 384 + 128, tid);
  col_finish(red + 2 * NWARP * 128, colpart + ((size_t)blockIdx.y * gridDim.x + blockIdx.x) * 384 + 256, tid);
}
