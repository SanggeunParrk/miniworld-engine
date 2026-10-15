// cond_bwd_sh90.cu — backward of the conditioning tables for a conditioning shared by the samples (sm_90a, wgmma), per 16-row warp tile of the
// N shared atoms: the modulation gradients arrive as P partial rows (the sample groups of the block backward), are summed in fp32 here (the
// sums are written out in fp32: the operands of the weight-gradient GEMMs) and run through the same products as cond_bwd90:
//   dn = ds[:, 0:128] Ws1' + ds[:, 128:256] Wb1' + ds[:, 384:512] Ws2' + ds[:, 512:640] Wb2' + dk[:, 0:128] Wkp' + dk[:, 128:256] Wks'
//   dcraw = ds[:, 256:384] Wso + ds[:, 640:768] Wst        dc = (LN'(dn) + dcraw) * inv_a
// SPDX-License-Identifier: Apache-2.0
#include "row90.cuh"
using namespace r90;

#ifndef PUNR
#define PUNR 2
#endif
#define STR_(x) #x
#define PRAG2(n) _Pragma(STR_(unroll n))
#define PRAG(n) PRAG2(n)

template <int NT>
DEVI void st_f32(const float (*v)[4], float* p, size_t ld, int col, int g, int t) {
  float* r0 = p + (size_t)g * ld + col;
  float* r1 = r0 + 8 * ld;
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    const int pc = pcol(i, t);
    *reinterpret_cast<float2*>(r0 + pc) = make_float2(v[i][0], v[i][1]);
    *reinterpret_cast<float2*>(r1 + pc) = make_float2(v[i][2], v[i][3]);
  }
}

extern "C" __global__ __launch_bounds__(NTHR, MINB) void cond_bwd_sh90(
    const T* __restrict__ dmodp /*[P][N][768]*/, const T* __restrict__ dmkvp /*[P][N][256]*/, const T* __restrict__ c /*[N][128]*/,
    const T* __restrict__ wt /*[8 kb][2 half][64,128]: W^T tiles*/, float* __restrict__ ds_out, float* __restrict__ dk_out,
    float* __restrict__ n_out, float* __restrict__ dc_out, int N, int P, float eps, float inv_a) {
  pdl_wait();
  extern __shared__ uint8_t sm_raw[];
  constexpr int UNIT = 64 * 128 * ES;
  const uint32_t base = (su(sm_raw) + 1023u) & ~1023u;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const size_t row0 = (size_t)blockIdx.x * ROWS + warp * 16;
  auto issue = [&](int s) {
    load_w(base + (s & 1) * UNIT, wt + (size_t)s * 64 * 128, 128, 64, 128, tid, NTHR);
    cp_commit();
  };
  issue(0);
  issue(1);
  Tile a, lc;
  float acc0[32], acc1[32];
#pragma unroll
  for (int kb = 0; kb < 8; ++kb) {
    if (kb == 6) {  // dn complete (into lc): LN backward with n(c), in place
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) { lc.f[nt][e] = acc0[4 * nt + e]; lc.f[8 + nt][e] = acc1[4 * nt + e]; }
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
      st_f32<16>(n.f, n_out + row0 * 128, 128, 0, g, t);
#pragma unroll
      for (int nt = 0; nt < 16; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e)
          lc.f[nt][e] = (e < 2 ? r0 : r1) * (lc.f[nt][e] - (e < 2 ? p0 : p1) - n.f[nt][e] * (e < 2 ? q0 : q1));
    }
    const bool kv = kb == 4 || kb == 5;
    const T* src = kv ? dmkvp : dmodp;
    const int ld = kv ? 256 : 768;
    const int col = kb < 4 ? (kb < 2 ? 128 * kb : 128 * (kb + 1)) : kv ? 128 * (kb - 4) : (kb == 6 ? 256 : 640);
    ld_tile(a, src + row0 * ld, ld, col, g, t);
PRAG(PUNR)
    for (int p = 1; p < P; ++p) {
      Tile b;
      ld_tile(b, src + ((size_t)p * N + row0) * ld, ld, col, g, t);
#pragma unroll
      for (int nt = 0; nt < 16; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) a.f[nt][e] += b.f[nt][e];
    }
    if (kv) st_f32<16>(a.f, dk_out + row0 * 256, 256, col, g, t);
    else st_f32<16>(a.f, ds_out + row0 * 768, 768, col, g, t);
    rna(a);
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const int s = 2 * kb + hf;
      cp_wait<1>();
      proxy_fence();
      __syncthreads();
      wg_gemm<64, KSTEPS>(hf ? acc1 : acc0, a, base + (s & 1) * UNIT, kb > 0 && kb != 6);
      __syncthreads();
      if (s + 2 < 16) issue(s + 2); else cp_commit();
    }
  }
#pragma unroll
  for (int nt = 0; nt < 8; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      lc.f[nt][e] = (lc.f[nt][e] + acc0[4 * nt + e]) * inv_a;
      lc.f[8 + nt][e] = (lc.f[8 + nt][e] + acc1[4 * nt + e]) * inv_a;
    }
  st_f32<16>(lc.f, dc_out + row0 * 128, 128, 0, g, t);
}
