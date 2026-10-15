// post_bwd90.cu — backward of AdaLN 2, the residual and the gated out-projection (sm_90a, wgmma), per 16-row warp tile:
//   dx2 = [da | db] [Wa; Wb]              (K = 512)
//   n = LN(a2)            d s2pre = dx2 n s2 (1 - s2)  -> dmod[:, 384:512]        d bi2 = dx2 -> dmod[:, 512:640]
//   da2 = dy + LN'(dx2 s2)                du = da2 so   d sopre = da2 u so (1 - so) -> dmod[:, 256:384]
//   dgated = du Wo      do = dgated sg      dg = dgated o sg (1 - sg)  -> dp[:, 0:128]      gated = o sg
//   D[a, h, i] = sum_{c in head h} do o
// Outputs: da2, du, gated, do [M, 128], dp's dg block, dmod blocks 2-4, dd [A, 4, N].
// SPDX-License-Identifier: Apache-2.0
#include "row90.cuh"
using namespace r90;
#ifndef SPLITK
#define SPLITK 0   // 1: the split-K build for small row counts (grid.y CTAs per tile)
#endif

extern "C" __global__ __launch_bounds__(NTHR, MINB) void post_bwd90(
    const T* __restrict__ dab, const T* __restrict__ dy, const T* __restrict__ a2, const T* __restrict__ mod, const T* __restrict__ u,
    const T* __restrict__ sg, const T* __restrict__ o, const T* __restrict__ wabt /*[4 kb][2 half][64,128]*/, const T* __restrict__ wot /*[2][64,128]*/,
    T* __restrict__ da2_out, T* __restrict__ du_out, T* __restrict__ gated_out, T* __restrict__ do_out, T* __restrict__ dp, T* __restrict__ dmod,
    float* __restrict__ dd, float* __restrict__ colpart, int M, int nmod, int N, float eps, float* __restrict__ dmod_acc, int spc, int natoms,
    float* __restrict__ part, int* __restrict__ ctr) {
  pdl_wait();
  extern __shared__ uint8_t sm_raw[];
  constexpr int UNIT = 64 * 128 * ES;
  const uint32_t base = (su(sm_raw) + 1023u) & ~1023u;
  float* red = reinterpret_cast<float*>(sm_raw + (base - su(sm_raw)) + 2 * UNIT);  // [2 sets][8 warps][128]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const bool shared = spc > 0;   // grouped-sample mode (spc > 0: shared conditioning, samples a multiple of 4): warp w = sample w & 3 of a group of four, summed in the CTA
  const size_t atom0 = (size_t)blockIdx.x * ROWS + warp * 16;
  for (int i = tid; i < NWARP * 128 * 2; i += NTHR) red[i] = 0.f;
  __syncthreads();
  // split-K over the four dab blocks (small row counts: grid.y = ns CTAs per tile, each its 4 / ns blocks; the last to finish sums the fp32
  // partials in a fixed order and runs the rest alone). Local unit ul: this CTA's wabt units, then the two Wo units (the finisher's).
#if SPLITK
  const int ns = (int)gridDim.y, cy = (int)blockIdx.y, kbper = 4 / ns, kb0 = cy * kbper;
#else
  constexpr int ns = 1, cy = 0, kbper = 4, kb0 = 0;
#endif
  auto issue = [&](int ul) {
    const int s = ul < 2 * kbper ? 2 * kb0 + ul : 8 + (ul - 2 * kbper);
    const T* src = s < 8 ? wabt + (size_t)s * 64 * 128 : wot + (size_t)(s - 8) * 64 * 128;
    load_w(base + (ul & 1) * UNIT, src, 128, 64, 128, tid, NTHR);
    cp_commit();
  };
  const int nsmp = 1;   // grouped mode: the samples are the warps of the CTA (see put_grp)
  for (int smp = 0; smp < nsmp; ++smp) {
  const int nacg = shared ? natoms / (16 * (NWARP / 4)) : 1;
  const int acg = (int)blockIdx.x % nacg, sg4 = (int)blockIdx.x / nacg;
  const size_t row0 = shared ? (size_t)(sg4 * 4 + (warp & 3)) * natoms + (size_t)(acg * (NWARP / 4) + (warp >> 2)) * 16 : atom0;
  const size_t mrow = row0 % (size_t)nmod;   // rows of the modulation tables (nmod = M: per sample; = N: shared)
  const size_t prow = (size_t)sg4 * natoms + (size_t)acg * (NWARP / 4) * 16;   // partial row of this CTA's first atom block
  T* const gred = reinterpret_cast<T*>(sm_raw + (base - su(sm_raw)) + 2 * UNIT + 2 * NWARP * 128 * 4);
  int gpar = 0;
#define GFLUSH(DST, LD, COL) if (shared) { grp_flush(reinterpret_cast<T*>(DST), prow, LD, COL, gred + (size_t)gpar * FBUF, tid); gpar ^= GNB - 1; }
  issue(0);
  issue(1);
  Tile c;  // A operand / working tile
  float acc0[32], acc1[32];
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) {
    if (kb >= kbper) break;
    ld_tile(c, dab + row0 * 512, 512, 128 * (kb0 + kb), g, t);
    rna(c);
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const int ul = 2 * kb + hf;
      cp_wait<1>();
      proxy_fence();
      __syncthreads();
      wg_gemm<64, KSTEPS>(hf ? acc1 : acc0, c, base + (ul & 1) * UNIT, kb > 0);
      __syncthreads();
      if (ul + 2 < 2 * kbper || ns == 1) issue(ul + 2); else cp_commit();
    }
  }
  if (ns > 1) {
    float* pw = part + ((size_t)(blockIdx.x * ns + cy) * 16) * 512;
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      *reinterpret_cast<float4*>(pw + (size_t)q * 512 + tid * 4) = make_float4(acc0[4 * q], acc0[4 * q + 1], acc0[4 * q + 2], acc0[4 * q + 3]);
      *reinterpret_cast<float4*>(pw + (size_t)(8 + q) * 512 + tid * 4) = make_float4(acc1[4 * q], acc1[4 * q + 1], acc1[4 * q + 2], acc1[4 * q + 3]);
    }
    __threadfence();
    __syncthreads();
    __shared__ int s_last;
    if (tid == 0) s_last = atomicAdd(ctr + blockIdx.x, 1) == ns - 1;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    issue(2 * kbper);       // the Wo units load while the partials are summed
    issue(2 * kbper + 1);
    for (int y = 0; y < ns; ++y) {
      const float* pr = part + ((size_t)(blockIdx.x * ns + y) * 16) * 512;
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const float4 a = __ldcg(reinterpret_cast<const float4*>(pr + (size_t)q * 512 + tid * 4));
        const float4 b = __ldcg(reinterpret_cast<const float4*>(pr + (size_t)(8 + q) * 512 + tid * 4));
        if (y == 0) {
          acc0[4 * q] = a.x; acc0[4 * q + 1] = a.y; acc0[4 * q + 2] = a.z; acc0[4 * q + 3] = a.w;
          acc1[4 * q] = b.x; acc1[4 * q + 1] = b.y; acc1[4 * q + 2] = b.z; acc1[4 * q + 3] = b.w;
        } else {
          acc0[4 * q] += a.x; acc0[4 * q + 1] += a.y; acc0[4 * q + 2] += a.z; acc0[4 * q + 3] += a.w;
          acc1[4 * q] += b.x; acc1[4 * q + 1] += b.y; acc1[4 * q + 2] += b.z; acc1[4 * q + 3] += b.w;
        }
      }
    }
  }
  // dx2 -> c
#pragma unroll
  for (int nt = 0; nt < 8; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      c.f[nt][e] = round_store(acc0[4 * nt + e]);
      c.f[8 + nt][e] = round_store(acc1[4 * nt + e]);
    }
  // LN2 backward: n from a2, s2 (read in halves of the row), dy, u, so
  {
    Tile an;
    ld_tile(an, a2 + row0 * 128, 128, 0, g, t);
    float m0, r0, m1, r1;
    ln_stats(an, eps, m0, r0, m1, r1);
    float p0 = 0.f, q0 = 0.f, p1 = 0.f, q1 = 0.f;
#pragma unroll
    for (int H = 0; H < 4; ++H) {
      float s2[4][4];
      ld_nt<4>(s2, mod + mrow * 768, 768, 384 + 32 * H, g, t);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int nt = 4 * H + i;
        const float n0 = (an.f[nt][0] - m0) * r0, n1 = (an.f[nt][1] - m0) * r0, n2 = (an.f[nt][2] - m1) * r1, n3 = (an.f[nt][3] - m1) * r1;
        an.f[nt][0] = n0; an.f[nt][1] = n1; an.f[nt][2] = n2; an.f[nt][3] = n3;  // n
        const float d0 = c.f[nt][0] * s2[i][0], d1 = c.f[nt][1] * s2[i][1], d2 = c.f[nt][2] * s2[i][2], d3 = c.f[nt][3] * s2[i][3];
        p0 += d0 + d1; q0 += d0 * n0 + d1 * n1; p1 += d2 + d3; q1 += d2 * n2 + d3 * n3;
      }
    }
    p0 = quad_sum(p0) * (1.f / 128.f); q0 = quad_sum(q0) * (1.f / 128.f);
    p1 = quad_sum(p1) * (1.f / 128.f); q1 = quad_sum(q1) * (1.f / 128.f);
    // dmod blocks 3 (d s2pre) and 4 (d bi2); da2 = dy + rstd (dn - p - n q)
    if (shared) grp_put<16>(c.f, gred + (size_t)gpar * FBUF, 0, warp, g, t); else st_nt<16, false>(c.f, dmod + row0 * 768, 768, 512, g, t);
    GFLUSH(dmod_acc, 768, 512);
#pragma unroll
    for (int H = 0; H < 4; ++H) {
      float s2[4][4], y[4][4], ds2[4][4];
      ld_nt<4>(s2, mod + mrow * 768, 768, 384 + 32 * H, g, t);
      ld_nt<4>(y, dy + row0 * 128, 128, 32 * H, g, t);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int nt = 4 * H + i;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float nv = an.f[nt][e], sv = s2[i][e], dx = c.f[nt][e];
          ds2[i][e] = dx * nv * sv * (1.f - sv);
          c.f[nt][e] = round_store(y[i][e] + (e < 2 ? r0 : r1) * (dx * sv - (e < 2 ? p0 : p1) - nv * (e < 2 ? q0 : q1)));
        }
      }
      if (shared) grp_put<4>(ds2, gred + (size_t)gpar * FBUF, 32 * H, warp, g, t); else st_nt<4, false>(ds2, dmod + row0 * 768, 768, 384 + 32 * H, g, t);
      col_partial_nt<4>(ds2, 4 * H, red + NWARP * 128, warp, lane, t);
    }
    GFLUSH(dmod_acc, 768, 384);
  }
  // c = da2: du = da2 so, d sopre = da2 u so (1 - so)
  st_tile(c, da2_out + row0 * 128, 128, 0, g, t);
#pragma unroll
  for (int H = 0; H < 4; ++H) {
    float so[4][4], uu[4][4], dso[4][4], du[4][4];
    ld_nt<4>(so, mod + mrow * 768, 768, 256 + 32 * H, g, t);
    ld_nt<4>(uu, u + row0 * 128, 128, 32 * H, g, t);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int nt = 4 * H + i;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float sv = so[i][e], d = c.f[nt][e];
        dso[i][e] = d * uu[i][e] * sv * (1.f - sv);
        du[i][e] = round_store(d * sv);
        c.f[nt][e] = du[i][e];  // du: the A operand of dgated = du Wo
      }
    }
    if (shared) grp_put<4>(dso, gred + (size_t)gpar * FBUF, 32 * H, warp, g, t); else st_nt<4, false>(dso, dmod + row0 * 768, 768, 256 + 32 * H, g, t);
    col_partial_nt<4>(dso, 4 * H, red, warp, lane, t);
    st_nt<4, false>(du, du_out + row0 * 128, 128, 32 * H, g, t);
  }
  GFLUSH(dmod_acc, 768, 256);
  rna(c);  // du: the A operand of dgated = du Wo
  float g0[32], g1[32];
#pragma unroll
  for (int hf = 0; hf < 2; ++hf) {
    const int s = 8 + hf;
    cp_wait<1>();
    proxy_fence();
    __syncthreads();
    wg_gemm<64, KSTEPS>(hf ? g1 : g0, c, base + (s & 1) * UNIT, 0);
    __syncthreads();
    cp_commit();
  }
  {
    float dh0[4], dh1[4];
#pragma unroll
    for (int h = 0; h < 4; ++h) dh0[h] = dh1[h] = 0.f;
#pragma unroll
    for (int H = 0; H < 4; ++H) {
      float oo[4][4], ss[4][4], gd[4][4], dg[4][4], ga[4][4];
      ld_nt<4>(oo, o + row0 * 128, 128, 32 * H, g, t);
      ld_nt<4>(ss, sg + row0 * 128, 128, 32 * H, g, t);
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float dgv = round_store((H >= 2 ? g1 : g0)[4 * (i + 4 * (H & 1)) + e]);
          const float sv = ss[i][e], ov = oo[i][e];
          const float dov = round_store(dgv * sv);
          gd[i][e] = dov;
          dg[i][e] = dgv * ov * sv * (1.f - sv);
          ga[i][e] = round_store(ov * sv);
          if (e < 2) dh0[H] += dov * ov; else dh1[H] += dov * ov;
        }
      st_nt<4, false>(gd, do_out + row0 * 128, 128, 32 * H, g, t);
      st_nt<4, false>(dg, dp + row0 * 512, 512, 32 * H, g, t);
      st_nt<4, false>(ga, gated_out + row0 * 128, 128, 32 * H, g, t);
    }
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      const float v0 = quad_sum(dh0[h]), v1 = quad_sum(dh1[h]);
      if (t == 0) {
        const size_t r0 = row0 + g, r1 = row0 + g + 8;
        dd[((r0 / N) * 4 + h) * N + (r0 % N)] = v0;
        dd[((r1 / N) * 4 + h) * N + (r1 % N)] = v1;
      }
    }
  }
  __syncthreads();
  }  // samples
done:
  // only the finishing CTA of a tile gets here (it is whichever arrives last: a fixed slot per tile keeps the later sum order fixed)
  col_finish(red, colpart + (size_t)blockIdx.x * 256, tid);
  col_finish(red + NWARP * 128, colpart + (size_t)blockIdx.x * 256 + 128, tid);
}
