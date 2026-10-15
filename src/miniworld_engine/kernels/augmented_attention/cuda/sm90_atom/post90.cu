// post90.cu — gated out-projection, residual, AdaLN 2, SwiGLU, squeeze and residual of the AF3 atom block (sm_90a, wgmma).
//
//   u   = (o * sg) Wo^T            a2  = a + so * u          x2 = LN(a2) * s2 + bi2
//   h_j = silu(x2 Wa_j^T) * (x2 Wb_j^T)                      (hidden 256 in chunks of HB = 128 B / sizeof(T): 64 bf16, 32 fp32)
//   t   = sum_j h_j Wd_j^T          out = a2 + st * t
// mod columns: so = [256, 384), s2 = [384, 512), bi2 = [512, 640), st = [640, 768). The residual stream a2 and x2 stay in registers;
// SAVE writes u, a2, x2 and t for the backward. Weights stream through shared memory as swizzled K-major tiles: Wo in two 64-row
// units, then per chunk [Wa_j; Wb_j] (2 HB rows x 128) and Wd_j (128 rows x HB).
// SPDX-License-Identifier: Apache-2.0
#include "row90.cuh"
using namespace r90;

extern "C" __global__ __launch_bounds__(NTHR, 1) void post90_fwd(const T* __restrict__ a, const T* __restrict__ o, const T* __restrict__ sg,
                                                                const T* __restrict__ mod, const T* __restrict__ wo /*[128,128]*/,
                                                                const T* __restrict__ wab /*[NCH][2 HB,128]*/, const T* __restrict__ wd /*[NCH][128,HB]*/,
                                                                T* __restrict__ out, T* __restrict__ us, T* __restrict__ a2s, T* __restrict__ x2s,
                                                                T* __restrict__ ts, int M, int nmod, float eps, int save) {
  pdl_wait();
  extern __shared__ uint8_t sm_raw[];
  constexpr int NCH = 256 / HB;
  constexpr int ABB = 2 * HB * 128 * ES, DBB = 128 * 128, BUF = ABB + DBB;   // chunk buffer: [Wa_j; Wb_j] then Wd_j
  static_assert(BUF >= 64 * 128 * ES, "buffer holds a Wo unit");
  const uint32_t base = (su(sm_raw) + 1023u) & ~1023u;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const size_t row0 = (size_t)blockIdx.x * ROWS + warp * 16;
  const size_t mrow = row0 % (size_t)nmod;   // rows of the modulation tables (nmod = M: per sample; = N: one conditioning shared by the samples)
  // step s: 0, 1 = Wo units; 2 + j = chunk j
  auto issue = [&](int s) {
    const uint32_t d = base + (s & 1) * BUF;
    if (s < 2) {
      load_w(d, wo + (size_t)s * 64 * 128, 128, 64, 128, tid, NTHR);
    } else {
      const int j = s - 2;
      load_w(d, wab + (size_t)j * 2 * HB * 128, 128, 2 * HB, 128, tid, NTHR);
      load_w(d + ABB, wd + (size_t)j * 128 * HB, HB, 128, HB, tid, NTHR);
    }
    cp_commit();
  };
  issue(0);
  issue(1);
  Tile c, r;
  {  // gated = o * sg
    ld_tile(c, o + row0 * 128, 128, 0, g, t);
    ld_tile(r, sg + row0 * 128, 128, 0, g, t);
#pragma unroll
    for (int nt = 0; nt < 16; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) c.f[nt][e] = round_store(c.f[nt][e] * r.f[nt][e]);
    rna(c);
  }
  Tile a2;
  for (int s = 0; s < 2; ++s) {  // u = gated Wo^T, in two 64-column halves
    cp_wait<1>();
    proxy_fence();
    __syncthreads();
    float acc[32];
    wg_gemm<64, KSTEPS>(acc, c, base + (s & 1) * BUF, 0);
    __syncthreads();
    issue(s + 2);
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) a2.f[8 * s + nt][e] = acc[4 * nt + e];
  }
  {  // a2 = a + so * u  (u rounded as stored)
    Tile aa, so;
    ld_tile(aa, a + row0 * 128, 128, 0, g, t);
    ld_tile(so, mod + mrow * 768, 768, 256, g, t);
#pragma unroll
    for (int nt = 0; nt < 16; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float u = round_store(a2.f[nt][e]);
        a2.f[nt][e] = round_store(aa.f[nt][e] + round_store(so.f[nt][e] * u));
        c.f[nt][e] = u;
      }
    if (save) st_tile(c, us + row0 * 128, 128, 0, g, t);
  }
  if (save) st_tile(a2, a2s + row0 * 128, 128, 0, g, t);
  {  // x2 = LN(a2) * s2 + bi2   (into c)
    Tile s2, b2;
    ld_tile(s2, mod + mrow * 768, 768, 384, g, t);
    ld_tile(b2, mod + mrow * 768, 768, 512, g, t);
    float m0, r0, m1, r1;
    ln_stats(a2, eps, m0, r0, m1, r1);
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      c.f[nt][0] = round_store((a2.f[nt][0] - m0) * r0 * s2.f[nt][0] + b2.f[nt][0]);
      c.f[nt][1] = round_store((a2.f[nt][1] - m0) * r0 * s2.f[nt][1] + b2.f[nt][1]);
      c.f[nt][2] = round_store((a2.f[nt][2] - m1) * r1 * s2.f[nt][2] + b2.f[nt][2]);
      c.f[nt][3] = round_store((a2.f[nt][3] - m1) * r1 * s2.f[nt][3] + b2.f[nt][3]);
    }
    if (save) st_tile(c, x2s + row0 * 128, 128, 0, g, t);
    rna(c);
  }
  float o4[64];
  for (int j = 0; j < NCH; ++j) {
    cp_wait<1>();
    proxy_fence();
    __syncthreads();
    const uint32_t wa = base + ((j + 2) & 1) * BUF;
    float acc[HB];  // [a (HB / 2 regs ... ) | b]: wgmma n = 2 HB, acc = HB regs
    wg_gemm<2 * HB, KSTEPS>(acc, c, wa, 0);
    // h = silu(a) * b : a = n-tiles 0 .. HB / 8 - 1, b = the next HB / 8 -> a 16 x HB tile
    Tile h;
#pragma unroll
    for (int nt = 0; nt < HB / 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float av = round_store(acc[4 * nt + e]), bv = round_store(acc[4 * (nt + HB / 8) + e]);
        h.f[nt][e] = round_store(round_store(silu(av)) * bv);
      }
    rna(h);
    wg_gemm<128, HB / (TF32 ? 8 : 16)>(o4, h, wa + ABB, j > 0);
    __syncthreads();
    if (j + 4 < NCH + 2) issue(j + 4); else cp_commit();
  }
  {  // out = a2 + st * t
    Tile st, tt;
    ld_tile(st, mod + mrow * 768, 768, 640, g, t);
#pragma unroll
    for (int nt = 0; nt < 16; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float tv = round_store(o4[4 * nt + e]);
        tt.f[nt][e] = tv;
        a2.f[nt][e] = round_store(a2.f[nt][e] + round_store(st.f[nt][e] * tv));
      }
    st_tile(a2, out + row0 * 128, 128, 0, g, t);
    if (save) st_tile(tt, ts + row0 * 128, 128, 0, g, t);
  }
}
