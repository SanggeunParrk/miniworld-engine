// kqln_bwd90.cu — backward of RoseTTAFold3's affine LayerNorm of the flat q / k (pre90 KQLN), per 16-row warp tile, in place on dp:
//   x = the raw projection (qkr [M, 256] = [q | k]), xh = LN(x), dn = dp's dq (or dk) block (the gradient of LN(x) w + b)
//   dw += dn xh, db += dn (column partials)          dx = rstd (dn w - mean(dn w) - xh mean(dn w xh))  -> dp's block
// dp = [dg | dq | dk | dv]: dq is block 1, dk block 2. colpart [blocks][512] = dwq | dbq | dwk | dbk.
// SPDX-License-Identifier: Apache-2.0
#include "row90.cuh"
using namespace r90;

extern "C" __global__ __launch_bounds__(NTHR, 1) void kqln_bwd90(T* __restrict__ dp, const T* __restrict__ qkr, const float* __restrict__ kq,
                                                                float* __restrict__ colpart, int M, float eps) {
  pdl_wait();
  __shared__ float red[4 * NWARP * 128];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const size_t row0 = (size_t)blockIdx.x * ROWS + warp * 16;
  for (int i = tid; i < 4 * NWARP * 128; i += NTHR) red[i] = 0.f;
  __syncthreads();
#pragma unroll 1
  for (int which = 0; which < 2; ++which) {   // q, then k
    Tile x, dn, tmp;
    ld_tile(x, qkr + row0 * 256, 256, 128 * which, g, t);
    ld_tile(dn, dp + row0 * 512, 512, 128 * (1 + which), g, t);
    float m0, r0, m1, r1;
    ln_stats(x, eps, m0, r0, m1, r1);
    const float* w = kq + 256 * which;
    float p0 = 0.f, q0 = 0.f, p1 = 0.f, q1 = 0.f;
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      const float2 ww = colv(w, nt, t);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float xh = (x.f[nt][e] - (e < 2 ? m0 : m1)) * (e < 2 ? r0 : r1);
        x.f[nt][e] = xh;
        tmp.f[nt][e] = dn.f[nt][e] * xh;                       // d w
        const float d = dn.f[nt][e] * ((e & 1) ? ww.y : ww.x);   // d xh
        if (e < 2) { p0 += d; q0 += d * xh; } else { p1 += d; q1 += d * xh; }
      }
    }
    col_partial(tmp, red + (2 * which) * NWARP * 128, warp, lane, t);
    col_partial(dn, red + (2 * which + 1) * NWARP * 128, warp, lane, t);
    p0 = quad_sum(p0) * (1.f / 128.f); q0 = quad_sum(q0) * (1.f / 128.f);
    p1 = quad_sum(p1) * (1.f / 128.f); q1 = quad_sum(q1) * (1.f / 128.f);
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      const float2 ww = colv(w, nt, t);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float d = dn.f[nt][e] * ((e & 1) ? ww.y : ww.x);
        dn.f[nt][e] = (e < 2 ? r0 : r1) * (d - (e < 2 ? p0 : p1) - x.f[nt][e] * (e < 2 ? q0 : q1));
      }
    }
    st_tile(dn, dp + row0 * 512, 512, 128 * (1 + which), g, t);
  }
  __syncthreads();
#pragma unroll 1
  for (int s = 0; s < 4; ++s) col_finish(red + s * NWARP * 128, colpart + (size_t)blockIdx.x * 512 + 128 * s, tid);
}
