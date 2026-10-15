// tr_bwd90.cu — backward of the conditioned transition's output gate and SwiGLU (sm_90a, wgmma), per 16-row warp tile:
//   dt = dy * st            dts = dy * t * st (1 - st)   (-> dmod[:, 640:768], the st pre-sigmoid gradient)
//   a, b = x2 Wa^T, x2 Wb^T (recomputed)        dh = dt Wd
//   h = silu(a) b           da = dh b silu'(a)           db = dh silu(a)
// -> dt [M, 128], hh = h [M, 256], dab = [da | db] [M, 512]. Per hidden chunk of HB columns the weights [Wa_j; Wb_j] and Wd^T_j stream
// through shared memory.
// SPDX-License-Identifier: Apache-2.0
#include "row90.cuh"
using namespace r90;

extern "C" __global__ __launch_bounds__(NTHR, 1) void tr_bwd90(const T* __restrict__ dy, const T* __restrict__ tsv, const T* __restrict__ mod,
                                                              const T* __restrict__ x2, const T* __restrict__ wab /*[NCH][2 HB,128]*/,
                                                              const T* __restrict__ wdt /*[NCH][HB,128]*/, T* __restrict__ dt_out,
                                                              T* __restrict__ dmod, T* __restrict__ hh, T* __restrict__ dab, float* __restrict__ colpart, int M, int nmod, float* __restrict__ dmod_acc, int spc, int natoms) {
  pdl_wait();
  extern __shared__ uint8_t sm_raw[];
  constexpr int NCH = 256 / HB;
  constexpr int ABB = 2 * HB * 128 * ES, DBB = HB * 128 * ES, BUF = ABB + DBB;
  const uint32_t base = (su(sm_raw) + 1023u) & ~1023u;
  float* red = reinterpret_cast<float*>(sm_raw + (base - su(sm_raw)) + 2 * BUF);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const bool shared = spc > 0;   // grouped-sample mode (spc > 0: shared conditioning, samples a multiple of 4): warp w = sample w & 3 of a group of four, summed in the CTA
  const size_t atom0 = (size_t)blockIdx.x * ROWS + warp * 16;
  for (int i = tid; i < NWARP * 128 * 1; i += NTHR) red[i] = 0.f;
  __syncthreads();
  auto issue = [&](int j) {
    const uint32_t d = base + (j & 1) * BUF;
    load_w(d, wab + (size_t)j * 2 * HB * 128, 128, 2 * HB, 128, tid, NTHR);
    load_w(d + ABB, wdt + (size_t)j * HB * 128, 128, HB, 128, tid, NTHR);
    cp_commit();
  };
  const int nsmp = 1;   // grouped mode: the samples are the warps of the CTA (see put_grp)
  for (int smp = 0; smp < nsmp; ++smp) {
  const int nacg = shared ? natoms / (16 * (NWARP / 4)) : 1;
  const int acg = (int)blockIdx.x % nacg, sg4 = (int)blockIdx.x / nacg;
  const size_t row0 = shared ? (size_t)(sg4 * 4 + (warp & 3)) * natoms + (size_t)(acg * (NWARP / 4) + (warp >> 2)) * 16 : atom0;
  const size_t mrow = row0 % (size_t)nmod;   // rows of the modulation tables (nmod = M: per sample; = N: shared)
  const size_t prow = (size_t)sg4 * natoms + (size_t)acg * (NWARP / 4) * 16;   // partial row of this CTA's first atom block
  T* const gred = reinterpret_cast<T*>(sm_raw + (base - su(sm_raw)) + 2 * BUF + 1 * NWARP * 128 * 4);
  int gpar = 0;
#define GFLUSH(DST, LD, COL) if (shared) { grp_flush(reinterpret_cast<T*>(DST), prow, LD, COL, gred + (size_t)gpar * FBUF, tid); gpar ^= GNB - 1; }
  const int ncs = (int)gridDim.y, cy = (int)blockIdx.y, jper = NCH / ncs, j0 = cy * jper;   // hidden chunks of this CTA (small row counts: grid.y > 1)
  issue(j0);
  if (jper > 1) issue(j0 + 1); else cp_commit();
  Tile dtt, xt;
  {
    Tile y, tt, st;
    ld_tile(y, dy + row0 * 128, 128, 0, g, t);
    ld_tile(tt, tsv + row0 * 128, 128, 0, g, t);
    ld_tile(st, mod + mrow * 768, 768, 640, g, t);
    Tile ds;
#pragma unroll
    for (int nt = 0; nt < 16; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float s = st.f[nt][e], yv = y.f[nt][e];
        dtt.f[nt][e] = round_store(yv * s);
        ds.f[nt][e] = yv * tt.f[nt][e] * s * (1.f - s);
      }
    if (cy == 0) st_tile(dtt, dt_out + row0 * 128, 128, 0, g, t);
    rna(dtt);
    if (cy == 0) {   // the chunk-split CTAs of a tile share the gate gradient: the first writes it
      if (shared) grp_put<16>(ds.f, gred + (size_t)gpar * FBUF, 0, warp, g, t); else st_nt<16, false>(ds.f, dmod + row0 * 768, 768, 640, g, t);
      GFLUSH(dmod_acc, 768, 640);
      col_partial(ds, red, warp, lane, t);  // d bias of the st gate
    }
  }
  ld_tile(xt, x2 + row0 * 128, 128, 0, g, t);
  rna(xt);
  for (int jl = 0; jl < jper; ++jl) {
    const int j = j0 + jl;
    cp_wait<1>();
    proxy_fence();
    __syncthreads();
    const uint32_t wa = base + (j & 1) * BUF;
    float ab[HB];
    float dh[HB / 2];
    wg_gemm<2 * HB, KSTEPS>(ab, xt, wa, 0);
    wg_gemm<HB, KSTEPS>(dh, dtt, wa + ABB, 0);
    __syncthreads();
    if (jl + 2 < jper) issue(j + 2); else cp_commit();
    float h4[HB / 8][4], da4[HB / 8][4], db4[HB / 8][4];
#pragma unroll
    for (int nt = 0; nt < HB / 8; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float a = round_store(ab[4 * nt + e]), b = round_store(ab[4 * (nt + HB / 8) + e]), d = round_store(dh[4 * nt + e]);
        const float s = sigm(a), sl = round_store(a * s);
        h4[nt][e] = sl * b;
        da4[nt][e] = d * b * s * (1.f + a * (1.f - s));
        db4[nt][e] = d * sl;
      }
    }
    st_nt<HB / 8, false>(h4, hh + row0 * 256, 256, j * HB, g, t);
    st_nt<HB / 8, false>(da4, dab + row0 * 512, 512, j * HB, g, t);
    st_nt<HB / 8, false>(db4, dab + row0 * 512, 512, 256 + j * HB, g, t);
  }
    __syncthreads();
  }  // samples
  col_finish(red, colpart + ((size_t)blockIdx.y * gridDim.x + blockIdx.x) * 128, tid);
}
