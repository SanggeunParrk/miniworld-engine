// cond_bwd90.cu — backward of the conditioning tables (sm_90a, wgmma), per 16-row warp tile:
//   dn = dmod[:, 0:128] Ws1' + dmod[:, 128:256] Wb1' + dmod[:, 384:512] Ws2' + dmod[:, 512:640] Wb2' + dmkv[:, 0:128] Wkp' + dmkv[:, 128:256] Wks'
//   dcraw = dmod[:, 256:384] Wso + dmod[:, 640:768] Wst
//   dc = LN'(dn) + dcraw          (the LayerNorm weights are folded into W'; their gradients come from the weight gradients)
// SPDX-License-Identifier: Apache-2.0
#include "row90.cuh"
using namespace r90;
#ifndef SPLITK
#define SPLITK 0   // 1: the split-K build for small row counts (grid.y CTAs per tile)
#endif

// Split-K for small row counts (grid.y = 4 CTAs per tile: kb pairs {0, 1}, {2, 3}, {4, 5} of dn and {6, 7} of dcraw). Each writes its fp32
// accumulators; the last to finish sums dn = p0 + p1 + p2 in a fixed order, runs the LayerNorm backward and adds dcraw = p3.
extern "C" __global__ __launch_bounds__(NTHR, MINB) void cond_bwd90(const T* __restrict__ dmod, const T* __restrict__ dmkv, const T* __restrict__ c,
                                                                const T* __restrict__ wt /*[8 kb][2 half][64,128]: W^T tiles*/, T* __restrict__ dc_out, T* __restrict__ n_out,
                                                                int M, float eps, float* __restrict__ part, int* __restrict__ ctr) {
  pdl_wait();
  extern __shared__ uint8_t sm_raw[];
  constexpr int UNIT = 64 * 128 * ES;
  const uint32_t base = (su(sm_raw) + 1023u) & ~1023u;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const size_t row0 = (size_t)blockIdx.x * ROWS + warp * 16;
#if SPLITK
  const int ns = (int)gridDim.y, cy = (int)blockIdx.y, kbn = 8 / ns, kb0 = cy * kbn;   // ns = 4
#else
  constexpr int ns = 1, cy = 0, kbn = 8, kb0 = 0;
#endif
  auto issue = [&](int ul) {
    load_w(base + (ul & 1) * UNIT, wt + (size_t)(2 * kb0 + ul) * 64 * 128, 128, 64, 128, tid, NTHR);
    cp_commit();
  };
  issue(0);
  issue(1);
  Tile a, lc;
  float acc0[32], acc1[32];
  // lc <- LN'(dn) with n(c) (also writes n(c))
  auto ln_bwd = [&]() {
    Tile n;
    ld_tile(n, c + row0 * 128, 128, 0, g, t);
    float m0, r0, m1, r1;
    ln_stats(n, eps, m0, r0, m1, r1);
    float p0 = 0.f, q0 = 0.f, p1 = 0.f, q1 = 0.f;
#pragma unroll
    for (int nt = 0; nt < 16; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float nv = (n.f[nt][e] - (e < 2 ? m0 : m1)) * (e < 2 ? r0 : r1);
        n.f[nt][e] = nv;
        if (e < 2) { p0 += lc.f[nt][e]; q0 += lc.f[nt][e] * nv; } else { p1 += lc.f[nt][e]; q1 += lc.f[nt][e] * nv; }
      }
    p0 = quad_sum(p0) * (1.f / 128.f); q0 = quad_sum(q0) * (1.f / 128.f);
    p1 = quad_sum(p1) * (1.f / 128.f); q1 = quad_sum(q1) * (1.f / 128.f);
    st_tile(n, n_out + row0 * 128, 128, 0, g, t);  // n(c): the operand of the gamma-folded weights' gradients
#pragma unroll
    for (int nt = 0; nt < 16; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e)
        lc.f[nt][e] = (e < 2 ? r0 : r1) * (lc.f[nt][e] - (e < 2 ? p0 : p1) - n.f[nt][e] * (e < 2 ? q0 : q1));
  };
#pragma unroll
  for (int kl = 0; kl < 8; ++kl) {
    if (kl >= kbn) break;
    const int kb = kb0 + kl;
    if (kb == 6 && ns == 1) {  // dn complete (into lc)
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) { lc.f[nt][e] = acc0[4 * nt + e]; lc.f[8 + nt][e] = acc1[4 * nt + e]; }
      ln_bwd();
    }
    if (kb < 4) ld_tile(a, dmod + row0 * 768, 768, kb < 2 ? 128 * kb : 128 * (kb + 1), g, t);  // blocks 0, 1, 3, 4
    else if (kb < 6) ld_tile(a, dmkv + row0 * 256, 256, 128 * (kb - 4), g, t);
    else ld_tile(a, dmod + row0 * 768, 768, kb == 6 ? 256 : 640, g, t);                         // blocks 2, 5
    rna(a);
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const int ul = 2 * kl + hf;
      cp_wait<1>();
      proxy_fence();
      __syncthreads();
      wg_gemm<64, KSTEPS>(hf ? acc1 : acc0, a, base + (ul & 1) * UNIT, kl > 0 && !(kb == 6 && ns == 1));
      __syncthreads();
      if (ul + 2 < 2 * kbn) issue(ul + 2); else cp_commit();
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
    const float* pr0 = part + ((size_t)(blockIdx.x * ns) * 16) * 512;
#pragma unroll
    for (int q = 0; q < 8; ++q) {   // dn = p0 + p1 + p2
      float4 x0 = __ldcg(reinterpret_cast<const float4*>(pr0 + (size_t)q * 512 + tid * 4));
      float4 y0 = __ldcg(reinterpret_cast<const float4*>(pr0 + (size_t)(8 + q) * 512 + tid * 4));
#pragma unroll
      for (int y = 1; y < 3; ++y) {
        const float* pr = pr0 + (size_t)y * 16 * 512;
        const float4 x1 = __ldcg(reinterpret_cast<const float4*>(pr + (size_t)q * 512 + tid * 4));
        const float4 y1 = __ldcg(reinterpret_cast<const float4*>(pr + (size_t)(8 + q) * 512 + tid * 4));
        x0.x += x1.x; x0.y += x1.y; x0.z += x1.z; x0.w += x1.w;
        y0.x += y1.x; y0.y += y1.y; y0.z += y1.z; y0.w += y1.w;
      }
      lc.f[q][0] = x0.x; lc.f[q][1] = x0.y; lc.f[q][2] = x0.z; lc.f[q][3] = x0.w;
      lc.f[8 + q][0] = y0.x; lc.f[8 + q][1] = y0.y; lc.f[8 + q][2] = y0.z; lc.f[8 + q][3] = y0.w;
    }
    ln_bwd();
    const float* pr3 = pr0 + (size_t)3 * 16 * 512;   // dcraw
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const float4 x3 = __ldcg(reinterpret_cast<const float4*>(pr3 + (size_t)q * 512 + tid * 4));
      const float4 y3 = __ldcg(reinterpret_cast<const float4*>(pr3 + (size_t)(8 + q) * 512 + tid * 4));
      acc0[4 * q] = x3.x; acc0[4 * q + 1] = x3.y; acc0[4 * q + 2] = x3.z; acc0[4 * q + 3] = x3.w;
      acc1[4 * q] = y3.x; acc1[4 * q + 1] = y3.y; acc1[4 * q + 2] = y3.z; acc1[4 * q + 3] = y3.w;
    }
  }
#pragma unroll
  for (int nt = 0; nt < 8; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      lc.f[nt][e] = round_store(lc.f[nt][e] + acc0[4 * nt + e]);
      lc.f[8 + nt][e] = round_store(lc.f[8 + nt][e] + acc1[4 * nt + e]);
    }
  st_tile(lc, dc_out + row0 * 128, 128, 0, g, t);
}
