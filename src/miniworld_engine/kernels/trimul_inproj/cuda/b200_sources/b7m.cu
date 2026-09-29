// b7m.cu -- TriMul B7 (input-side backward) on B200 for d_pair C = 64, one CTA per 128-token tile, end to end.
//
// At C = 64 the whole packed W1 (NCH chunks x [128 rows][64], <= 64 KB) and Wg stay in shared memory and dW1 (NCH x 64
// TMEM columns) stays in TMEM, so the b7r / b7s source -> L2 ring -> consumer hand-over is not needed: each persistent CTA
//   for every chunk c of its tile:  pre = xn W1_c^T (TMEM) -> dg / dp (8 warps, fragment layout, as b7r) -> dgp smem tile
//                                   dxn += dgp W1_c,  dW1_c += dgp^T xn          (tcgen05, fp32 accumulate)
//   last:                           dxn += dGout Wg -> 4 epilogue warps (one row per thread): LN_in backward + residual -> dx,
//                                   dgi / dbi; they overlap the next tile's chunks (dxn double buffered in TMEM).
// Same math and rounding points as b7r.cu: dxn is one bf16 rounding of the fp32 sum over K = 64 + 128 NCH.
#include "sm100.cuh"
#include "tmap.h"

using namespace sm100;

namespace b7m {

constexpr int C = 64, TOK = 128;
constexpr int XT = TOK * C * 2;            // 16 KB: [128][64] SW128 tile (xn, dGout, x, dy, a W1 chunk half ... )
constexpr int DAT = 64 * TOK * 2;          // 16 KB: dA chunk [64 ch][128 tok], two 64-token halves
constexpr int GT = TOK * 128 * 2;          // 32 KB: dgp tile [128 tok][gate 64 | proj 64]
template <int NCH>
struct Geo {
  static constexpr int S_W1 = 0, S_WG = S_W1 + NCH * XT, S_X = S_WG + 8192, S_DA = S_X + 2 * XT, S_DGP = S_DA + 2 * DAT,
                       S_GO = S_DGP + GT, S_XI = S_GO + XT, S_DY = S_XI + XT, O_BAR = S_DY + XT, O_GI = O_BAR + 512, SMEM = O_GI + 256;
  static_assert(SMEM <= 232448, "smem");
  static constexpr int NPB = NCH == 4 ? 1 : 2;               // pre-activation TMEM buffers
  static constexpr uint32_t T_DW = 0, T_PRE = NCH * 64, T_ACC = 384;
  static_assert(T_PRE + NPB * 128 <= T_ACC, "tmem");
};
constexpr uint32_t ID_PRE = idesc_bf16_mj(128, 128, 0, 0);
constexpr uint32_t ID_DW = idesc_bf16_mj(128, C, 1, 1);
constexpr uint32_t ID_DX = idesc_bf16_mj(128, C, 0, 1);

struct Bars {
  uint64_t w_full, x_full[2], x_free[2], da_full[2], da_free[2], pre_full[2], pre_free[2], dgp_ready, dgp_free, go_full, go_free,
      acc_full[2], acc_empty[2], xi_full, xi_free;
  uint32_t tmem;
};

DEV float ex2_ftz(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float rcp_ftz(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float rsqrt_ftz(float x) { float y; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV uint32_t sw128(uint32_t r, uint32_t q) { return r * 128u + ((q ^ (r & 7u)) << 4); }
DEV float2 up(uint32_t w) { return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u)); }
DEV void ldsm_x4_trans(uint32_t addr, uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr) : "memory");
}
DEV void tmem_ld16x256_x4(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
                 "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr) : "memory");
}
DEV uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory");
  return v;
}
DEV float2 lds64g(uint32_t ad) { float2 v; asm volatile("ld.shared.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(ad) : "memory"); return v; }
DEV void red_add(float* p, float v) { asm volatile("red.global.add.f32 [%0], %1;" ::"l"(p), "f"(v) : "memory"); }
DEV void red_add4(float* p, float a, float b, float c, float d) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}

struct Args {
  const float *mask, *gi;
  float *dw1, *lnpart;      // lnpart [grid][dgi 64 | dbi 64]: this CTA's row (summed on the host: deterministic)
  int tiles;
  float eps;
};

template <int NCH>
__global__ void __launch_bounds__(512, 1)
    b7m_kernel(const __grid_constant__ CUtensorMap mxn, const __grid_constant__ CUtensorMap mda, const __grid_constant__ CUtensorMap mw1,
               const __grid_constant__ CUtensorMap mgo, const __grid_constant__ CUtensorMap mwg, const __grid_constant__ CUtensorMap mdx,
               const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mdy, const Args a) {
  using G_ = Geo<NCH>;
  constexpr int NPB = G_::NPB;
  constexpr uint32_t T_DW = G_::T_DW, T_PRE = G_::T_PRE, T_ACC = G_::T_ACC;
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + G_::O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, G = gridDim.x;
  const int n_local = a.tiles > cta ? (a.tiles - cta + G - 1) / G : 0;   // tiles t = cta + G i
  const int n_chunks = n_local * NCH;
  if (tid == 0 && (su & 1023u)) asm volatile("trap;");
  if (tid == 0) {
    mbar_init(&B.w_full, 1);
    for (int b = 0; b < 2; ++b) {
      mbar_init(&B.x_full[b], 1); mbar_init(&B.x_free[b], 1); mbar_init(&B.da_full[b], 1); mbar_init(&B.da_free[b], 8);
      mbar_init(&B.pre_full[b], 1); mbar_init(&B.pre_free[b], 8); mbar_init(&B.acc_full[b], 1); mbar_init(&B.acc_empty[b], 4);
    }
    mbar_init(&B.dgp_ready, 1); mbar_init(&B.dgp_free, 1); mbar_init(&B.go_full, 1); mbar_init(&B.go_free, 1);
    mbar_init(&B.xi_full, 1); mbar_init(&B.xi_free, 1);
    fence_mbar_init();
  }
  if (tid < C) reinterpret_cast<float*>(sm + G_::O_GI)[tid] = a.gi[tid];
  if (warp == 2) { tmem_alloc(&B.tmem, 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  PROF_BEGIN

  if (warp == 0) {
    if (lane == 0) {
      // weights once; then per tile: xn, the NCH dA chunks, dGout
      mbar_expect_tx(&B.w_full, NCH * XT + 8192);
      for (int c = 0; c < NCH; ++c) tma_load_2d(sm + G_::S_W1 + c * XT, &mw1, &B.w_full, 0, 128 * c, EVICT_LAST);
      tma_load_2d(sm + G_::S_WG, &mwg, &B.w_full, 0, 0, EVICT_LAST);
      for (int i = 0; i < n_local; ++i) {
        const int row0 = (cta + G * i) * TOK, b = i & 1;
        if (i >= 2) PW(0, mbar_wait(&B.x_free[b], ((i >> 1) - 1) & 1));
        mbar_expect_tx(&B.x_full[b], XT);
        tma_load_2d(sm + G_::S_X + b * XT, &mxn, &B.x_full[b], 0, row0, EVICT_NORMAL);
        for (int c = 0; c < NCH; ++c) {
          const int u = i * NCH + c, db = u & 1;
          if (u >= 2) PW(1, mbar_wait(&B.da_free[db], ((u >> 1) - 1) & 1));
          mbar_expect_tx(&B.da_full[db], DAT);
          tma_load_2d(sm + G_::S_DA + db * DAT, &mda, &B.da_full[db], row0, 64 * c, EVICT_FIRST);
          tma_load_2d(sm + G_::S_DA + db * DAT + DAT / 2, &mda, &B.da_full[db], row0 + 64, 64 * c, EVICT_FIRST);
        }
        if (i >= 1) PW(2, mbar_wait(&B.go_free, (i - 1) & 1));
        mbar_expect_tx(&B.go_full, XT);
        tma_load_2d(sm + G_::S_GO, &mgo, &B.go_full, 0, row0, EVICT_FIRST);
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      mbar_wait(&B.w_full, 0);
      const uint32_t sw1 = su + G_::S_W1, sdgp = su + G_::S_DGP;
      auto issue_pre = [&](int u) {
        const int i = u / NCH, c = u % NCH, pb = u % NPB;
        PW(0, mbar_wait(&B.x_full[i & 1], (i >> 1) & 1));
        if (u >= NPB) PW(1, mbar_wait(&B.pre_free[pb], ((u / NPB) - 1) & 1));
        tc_fence_after();
        const uint32_t sx = su + G_::S_X + (i & 1) * XT, sw = sw1 + c * XT;
#pragma unroll
        for (int k = 0; k < 4; ++k) umma_ss(tmem + T_PRE + pb * 128, desc_k_sw128(sx + k * 32), desc_k_sw128(sw + k * 32), ID_PRE, k != 0);
        umma_commit(&B.pre_full[pb]);
      };
      if (n_chunks > 0) issue_pre(0);
      for (int i = 0; i < n_local; ++i) {
        const int acc = i & 1;
        const uint32_t sx = su + G_::S_X + acc * XT, dacc = tmem + T_ACC + acc * 64;
        if (i >= 2) PW(2, mbar_wait(&B.acc_empty[acc], ((i >> 1) - 1) & 1));
        for (int c = 0; c < NCH; ++c) {
          const int u = i * NCH + c;
          if (u + 1 < n_chunks) issue_pre(u + 1);            // runs while the dg / dp warps work on chunk u
          PW(3, mbar_wait(&B.dgp_ready, u & 1));
          tc_fence_after();
          const uint32_t sw = sw1 + c * XT;
          // dxn [128 tok][64] += dgp [tok][128 k] . W1_c [128 k][64]   (A K-major: two 64-column K-blocks; B MN-major)
#pragma unroll
          for (int k = 0; k < 8; ++k)
            umma_ss(dacc, desc_k_sw128(sdgp + (k >> 2) * (GT / 2) + (k & 3) * 32), desc_mn_sw128(sw + k * 2048, 8192), ID_DX, (c | k) != 0);
          // dW1_c [128 rows][64] += dgp^T [rows][128 tok] . xn [tok][64]
#pragma unroll
          for (int k = 0; k < 8; ++k)
            umma_ss(tmem + T_DW + c * 64, desc_mn_sw128(sdgp + k * 2048, GT / 2), desc_mn_sw128(sx + k * 2048, XT), ID_DW, (i | k) != 0);
          umma_commit(&B.dgp_free);
        }
        umma_commit(&B.x_free[acc]);
        PW(4, mbar_wait(&B.go_full, i & 1));
        tc_fence_after();
        const uint32_t sgo = su + G_::S_GO, swg = su + G_::S_WG;
#pragma unroll
        for (int k = 0; k < 4; ++k) umma_ss(dacc, desc_k_sw128(sgo + k * 32), desc_mn_sw128(swg + k * 2048, 8192), ID_DX, 1);
        umma_commit(&B.go_free);
        umma_commit(&B.acc_full[acc]);
      }
    }
  } else if (warp == 3) {
    if (lane == 0) {
      // x / dy of each tile for the epilogue (refilled once dx has left the x tile)
      for (int i = 0; i < n_local; ++i) {
        const int row0 = (cta + G * i) * TOK;
        if (i >= 1) PW(0, mbar_wait(&B.xi_free, (i - 1) & 1));
        mbar_expect_tx(&B.xi_full, 2 * XT);
        tma_load_2d(sm + G_::S_XI, &mx, &B.xi_full, 0, row0, EVICT_FIRST);
        tma_load_2d(sm + G_::S_DY, &mdy, &B.xi_full, 0, row0, EVICT_FIRST);
      }
    }
  } else if (warp >= 4 && warp < 12) {
    // dg / dp of chunk u (as b7r.cu): warps 4-7 token group hh 0, warps 8-11 hh 1 of each warp quarter
    const int q = warp & 3, hh = (warp - 4) >> 2, tr = lane >> 2;
    const int tok0 = q * 32 + hh * 16;
    float mA = 0.f, mB = 0.f;
    for (int u = 0; u < n_chunks; ++u) {
      const int i = u / NCH, pb = u % NPB, db = u & 1;
      if (u % NCH == 0) {
        const int r0 = (cta + G * i) * TOK + tok0 + tr;
        mA = __ldg(a.mask + r0); mB = __ldg(a.mask + r0 + 8);
      }
      PW(0, mbar_wait(&B.pre_full[pb], (u / NPB) & 1));
      PW(1, mbar_wait(&B.da_full[db], (u >> 1) & 1));
      tc_fence_after();
      const uint32_t ta = tmem + T_PRE + pb * 128 + ((uint32_t)tok0 << 16);
      uint32_t da[16];
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const int mm = lane >> 3, ch = 16 * jj + 8 * (mm & 1) + (lane & 7), qc = ((tok0 & 63) >> 3) + (mm >> 1);
        ldsm_x4_trans(su + G_::S_DA + db * DAT + (tok0 >> 6) * (DAT / 2) + sw128(ch, qc), da[4 * jj], da[4 * jj + 1], da[4 * jj + 2],
                      da[4 * jj + 3]);
      }
      uint32_t gall[2][16], pall[2][16];
      tmem_ld16x256_x4(ta, gall[0]);
      tmem_ld16x256_x4(ta + 64, pall[0]);
      tmem_wait_ld();
      tmem_ld16x256_x4(ta + 32, gall[1]);
      tmem_ld16x256_x4(ta + 96, pall[1]);
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.da_free[db]);
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        if (half == 1) {
          tmem_wait_ld();
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive(&B.pre_free[pb]);
        }
        uint32_t odg[8], odp[8];
#pragma unroll
        for (int jx = 0; jx < 4; ++jx) {
          const int J = 4 * half + jx;
#pragma unroll
          for (int tb = 0; tb < 2; ++tb) {
            const float2 dA = up(da[4 * (J >> 1) + (J & 1) + 2 * tb]);
            const float mkk = tb ? mB : mA;
            const uint32_t gw = pack_bf16(__uint_as_float(gall[half][4 * jx + 2 * tb]), __uint_as_float(gall[half][4 * jx + 2 * tb + 1]));
            const uint32_t pw = pack_bf16(__uint_as_float(pall[half][4 * jx + 2 * tb]), __uint_as_float(pall[half][4 * jx + 2 * tb + 1]));
            const float2 t = __fmul2_rn(up(gw), make_float2(-1.4426950408889634f, -1.4426950408889634f));
            const float2 dn = __fadd2_rn(make_float2(ex2_ftz(t.x), ex2_ftz(t.y)), make_float2(1.f, 1.f));
            const float2 gg = make_float2(rcp_ftz(dn.x), rcp_ftz(dn.y));
            const float2 m = __fmul2_rn(dA, make_float2(mkk, mkk));
            const float2 rg = __fmul2_rn(__fmul2_rn(__fmul2_rn(m, up(pw)), gg), __fadd2_rn(make_float2(1.f, 1.f), make_float2(-gg.x, -gg.y)));
            const float2 rp = __fmul2_rn(m, gg);
            odg[2 * jx + tb] = pack_bf16(rg.x, rg.y);
            odp[2 * jx + tb] = pack_bf16(rp.x, rp.y);
          }
        }
        if (half == 0 && u >= 1) PW(2, mbar_wait(&B.dgp_free, (u - 1) & 1));   // chunk u-1's MMAs have read the dgp tile
#pragma unroll
        for (int x2 = 0; x2 < 2; ++x2) {
          const int m = lane >> 3, rr = lane & 7;
          const int tok = tok0 + rr + 8 * (m & 1), J = 4 * half + 2 * x2 + (m >> 1);
          stsm_x4(su + G_::S_DGP + sw128(tok, J), odg[4 * x2], odg[4 * x2 + 1], odg[4 * x2 + 2], odg[4 * x2 + 3]);
          stsm_x4(su + G_::S_DGP + GT / 2 + sw128(tok, J), odp[4 * x2], odp[4 * x2 + 1], odp[4 * x2 + 2], odp[4 * x2 + 3]);
        }
      }
      fence_async_smem();
      PW(3, named_bar_sync(1, 256));
      if (tid == 128) mbar_arrive(&B.dgp_ready);
    }
    // flush dW1: chunk c rows 128c + row, this warp's 32-column half
    tc_fence_before();
    named_bar_sync(1, 256);
    if (n_local > 0) PW(4, mbar_wait(&B.x_free[(n_local - 1) & 1], ((n_local - 1) >> 1) & 1));   // the last dW1 MMAs are done
    tc_fence_after();
    if (n_local > 0) {
      const int part = hh, row = q * 32 + lane;
      for (int n = 0; n < NCH; ++n) {
        const int c = (n + cta) % NCH;
        uint32_t v[32];
        tmem_ld32(tmem + T_DW + ((uint32_t)(q * 32) << 16) + c * 64 + part * 32, v);
        tmem_wait_ld();
        float* dst = a.dw1 + (size_t)(128 * c + row) * C + part * 32;
#pragma unroll
        for (int k = 0; k < 32; k += 4)
          red_add4(dst + k, __uint_as_float(v[k]), __uint_as_float(v[k + 1]), __uint_as_float(v[k + 2]), __uint_as_float(v[k + 3]));
      }
    }
  } else if (warp >= 12) {
    // LN_in backward + residual, one row per thread (r = 32 (warp % 4) + lane, all 64 columns): x / dy from smem, dxn from TMEM
    const int q = warp & 3, r = q * 32 + lane;
    const uint32_t gbase = su + G_::O_GI;
    float agi[4], abi[4];
#pragma unroll
    for (int m = 0; m < 4; ++m) { agi[m] = abi[m] = 0.f; }
    auto xa = [&](int ch) { return su + G_::S_XI + sw128(r, ch); };
    auto ya = [&](int ch) { return su + G_::S_DY + sw128(r, ch); };
    for (int i = 0; i < n_local; ++i) {
      const int row0 = (cta + G * i) * TOK, acc = i & 1;
      PW(0, mbar_wait(&B.xi_full, i & 1));
      float2 sa = make_float2(0.f, 0.f), sb = make_float2(0.f, 0.f);
#pragma unroll
      for (int ch = 0; ch < 8; ++ch) {
        const uint4 v = lds128(xa(ch));
        sa = __fadd2_rn(sa, __fadd2_rn(up(v.x), up(v.y))); sb = __fadd2_rn(sb, __fadd2_rn(up(v.z), up(v.w)));
      }
      float mean;
      {
        const float2 s2 = __fadd2_rn(sa, sb);
        mean = __fmul_rn(__fadd_rn(s2.x, s2.y), 1.f / C);
      }
      const float2 nmean = make_float2(-mean, -mean);
      sa = sb = make_float2(0.f, 0.f);
#pragma unroll
      for (int ch = 0; ch < 8; ++ch) {
        const uint4 v = lds128(xa(ch));
        float2 d = __fadd2_rn(up(v.x), nmean); sa = __ffma2_rn(d, d, sa);
        d = __fadd2_rn(up(v.y), nmean); sb = __ffma2_rn(d, d, sb);
        d = __fadd2_rn(up(v.z), nmean); sa = __ffma2_rn(d, d, sa);
        d = __fadd2_rn(up(v.w), nmean); sb = __ffma2_rn(d, d, sb);
      }
      float rstd;
      {
        const float2 s2 = __fadd2_rn(sa, sb);
        rstd = rsqrt_ftz(__fadd_rn(__fmul_rn(__fadd_rn(s2.x, s2.y), 1.f / C), a.eps));
      }
      const float2 rs2 = make_float2(rstd, rstd);
      PW(1, mbar_wait(&B.acc_full[acc], (i >> 1) & 1));
      tc_fence_after();
      const uint32_t trow = tmem + T_ACC + acc * 64 + ((uint32_t)(q * 32) << 16);
      uint32_t dpk[32];
      float c1, c2;
      {
        float2 c1a[2] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f)}, c2a[2] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f)};
#pragma unroll
        for (int h32 = 0; h32 < 2; ++h32) {
          uint32_t dv[32];
          tmem_ld32(trow + 32 * h32, dv);
          tmem_wait_ld();
          if (h32 == 1) { tc_fence_before(); __syncwarp(); if (lane == 0) mbar_arrive(&B.acc_empty[acc]); }
#pragma unroll
          for (int c4 = 0; c4 < 4; ++c4) {
            const int ch = 4 * h32 + c4;
            const uint4 xv = lds128(xa(ch));
            const uint32_t xs[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int w = 4 * ch + e;
              dpk[w] = pack_bf16(__uint_as_float(dv[8 * c4 + 2 * e]), __uint_as_float(dv[8 * c4 + 2 * e + 1]));
              const float2 d = up(dpk[w]);
              const float2 xh = __fmul2_rn(__fadd2_rn(up(xs[e]), nmean), rs2);
              const float2 dg = __fmul2_rn(d, lds64g(gbase + 8 * w));
              c1a[e & 1] = __ffma2_rn(dg, xh, c1a[e & 1]);
              c2a[e & 1] = __fadd2_rn(c2a[e & 1], dg);
            }
          }
        }
        const float2 c1s = __fadd2_rn(c1a[0], c1a[1]), c2s = __fadd2_rn(c2a[0], c2a[1]);
        c1 = __fmul_rn(__fadd_rn(c1s.x, c1s.y), 1.f / C); c2 = __fmul_rn(__fadd_rn(c2s.x, c2s.y), 1.f / C);
      }
      const float2 c1b = make_float2(c1, c1), c2b = make_float2(c2, c2);
#pragma unroll
      for (int hq = 0; hq < 4; ++hq) {                     // 16 columns (two chunks) at a time
        float2 pg[8], pb[8];
#pragma unroll
        for (int h2 = 0; h2 < 2; ++h2) {
          const int ch = 2 * hq + h2;
          const uint4 xv = lds128(xa(ch)), yv = lds128(ya(ch));
          const uint32_t xs[4] = {xv.x, xv.y, xv.z, xv.w}, ys[4] = {yv.x, yv.y, yv.z, yv.w};
          uint32_t o[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int w = 4 * ch + e;
            const float2 d = up(dpk[w]);
            const float2 xh = __fmul2_rn(__fadd2_rn(up(xs[e]), nmean), rs2);
            const float2 uu = __fadd2_rn(__fmul2_rn(xh, c1b), c2b);
            const float2 v = __fadd2_rn(__fmul2_rn(d, lds64g(gbase + 8 * w)), make_float2(-uu.x, -uu.y));
            const uint32_t lw = pack_bf16(__fmul_rn(rstd, v.x), __fmul_rn(rstd, v.y));
            const float2 tt = __fadd2_rn(up(lw), up(ys[e]));
            o[e] = pack_bf16(tt.x, tt.y);
            pg[4 * h2 + e] = __fmul2_rn(d, xh); pb[4 * h2 + e] = d;
          }
          sts128(xa(ch), o[0], o[1], o[2], o[3]);
        }
#pragma unroll
        for (int st = 0, msk = 16, nh = 4; st < 3; ++st, msk >>= 1, nh >>= 1) {
          const bool hi = (lane & msk) != 0;
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) {
            if (kk < nh) {
              const float2 sg_ = hi ? pg[kk] : pg[kk + nh], kg = hi ? pg[kk + nh] : pg[kk];
              const float2 sb_ = hi ? pb[kk] : pb[kk + nh], kb = hi ? pb[kk + nh] : pb[kk];
              pg[kk] = __fadd2_rn(kg, make_float2(__shfl_xor_sync(0xffffffffu, sg_.x, msk), __shfl_xor_sync(0xffffffffu, sg_.y, msk)));
              pb[kk] = __fadd2_rn(kb, make_float2(__shfl_xor_sync(0xffffffffu, sb_.x, msk), __shfl_xor_sync(0xffffffffu, sb_.y, msk)));
            }
          }
        }
        {
          const bool hi = (lane & 2) != 0;
          float g_ = hi ? pg[0].x : pg[0].y, kg = hi ? pg[0].y : pg[0].x;
          float b_ = hi ? pb[0].x : pb[0].y, kb = hi ? pb[0].y : pb[0].x;
          kg += __shfl_xor_sync(0xffffffffu, g_, 2); kb += __shfl_xor_sync(0xffffffffu, b_, 2);
          kg += __shfl_xor_sync(0xffffffffu, kg, 1); kb += __shfl_xor_sync(0xffffffffu, kb, 1);
          agi[hq] += kg; abi[hq] += kb;
        }
      }
      fence_async_smem();
      PW(2, named_bar_sync(3, 128));
      if (tid == 384) {
        tma_store_2d(&mdx, sm + G_::S_XI, 0, row0);
        bulk_commit();
        bulk_wait_read<0>();
        mbar_arrive(&B.xi_free);
      }
    }
    if (tid == 384) bulk_wait<0>();
    // dgi / dbi: the 4 warps' column sums combined in a fixed order (the dy tile is no longer read), one row per CTA
    float* xs = reinterpret_cast<float*>(sm + G_::S_DY);
    if ((lane & 1) == 0) {
#pragma unroll
      for (int h = 0; h < 4; ++h) {
        const int col = 16 * h + 2 * (lane >> 2) + ((lane >> 1) & 1);
        xs[q * 128 + col] = agi[h]; xs[q * 128 + 64 + col] = abi[h];
      }
    }
    named_bar_sync(3, 128);
    {
      const int c = tid - 384;
      a.lnpart[(size_t)cta * 128 + c] = ((xs[c] + xs[128 + c]) + xs[256 + c]) + xs[384 + c];
    }
  }
  if (lane == 0 && (warp < 5 || warp == 12)) PROF_END(warp == 12 ? 5 : warp);
  tc_fence_before();
  __syncthreads();
  if (warp == 2) tmem_dealloc(tmem, 512);
}

}  // namespace b7m

// x / xn / dy / dgout / dx [M, 64]; dpl [64 NCH, M] plane gradient; mask [M] fp32 pair mask; w1 [128 NCH, 64] packed
// (chunk c = 64 gate rows then 64 projection rows); wg [64, 64]; gi [64] fp32; dw1 [128 NCH, 64] fp32 accumulated;
// lnpart [>= min(tiles, SMs), 128] fp32: per-CTA (dg_i | db_i) rows, written (sum them for dg_i / db_i).
void b7m_backward(torch::Tensor x, torch::Tensor xn, torch::Tensor dy, torch::Tensor dpl, torch::Tensor dgout, torch::Tensor mask,
                  torch::Tensor w1, torch::Tensor wg, torch::Tensor gi, torch::Tensor dx, torch::Tensor dw1, torch::Tensor lnpart,
                  double eps) {
  using namespace b7m;
  const int64_t M = x.size(0);
  const int P = (int)dpl.size(0), NCH = P / 64;
  TORCH_CHECK(M % TOK == 0 && x.size(1) == C && dpl.numel() == (int64_t)P * M && (NCH == 2 || NCH == 4) && w1.size(0) == 128 * NCH);
  int nsm = 0, dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  const int tiles = (int)(M / TOK), grid = tiles < nsm ? tiles : nsm;
  auto bf = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  auto tok_map = [&](const torch::Tensor& t, int64_t rows) {
    return tmap::make(t.data_ptr(), bf, {(uint64_t)C, (uint64_t)rows}, {(uint64_t)C * 2}, {64, TOK}, CU_TENSOR_MAP_SWIZZLE_128B);
  };
  auto mxn = tok_map(xn, M), mgo = tok_map(dgout, M), mdx = tok_map(dx, M), mx = tok_map(x, M), mdy = tok_map(dy, M);
  auto mw1 = tok_map(w1, w1.size(0));
  auto mwg = tmap::make(wg.data_ptr(), bf, {(uint64_t)C, (uint64_t)C}, {(uint64_t)C * 2}, {64, 64}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mda = tmap::make(dpl.data_ptr(), bf, {(uint64_t)M, (uint64_t)P}, {(uint64_t)M * 2}, {64, 64}, CU_TENSOR_MAP_SWIZZLE_128B);
  Args a;
  a.mask = mask.data_ptr<float>(); a.gi = gi.data_ptr<float>();
  TORCH_CHECK(lnpart.numel() >= (int64_t)grid * 2 * C);
  a.dw1 = dw1.data_ptr<float>(); a.lnpart = lnpart.data_ptr<float>();
  a.tiles = tiles; a.eps = (float)eps;
  auto st = at::cuda::getCurrentCUDAStream();
  static bool attr2 = false, attr4 = false;
  if (NCH == 2) {
    if (!attr2) { C10_CUDA_CHECK(cudaFuncSetAttribute(b7m_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, Geo<2>::SMEM)); attr2 = true; }
    b7m_kernel<2><<<grid, 512, Geo<2>::SMEM, st>>>(mxn, mda, mw1, mgo, mwg, mdx, mx, mdy, a);
  } else {
    if (!attr4) { C10_CUDA_CHECK(cudaFuncSetAttribute(b7m_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, Geo<4>::SMEM)); attr4 = true; }
    b7m_kernel<4><<<grid, 512, Geo<4>::SMEM, st>>>(mxn, mda, mw1, mgo, mwg, mdx, mx, mdy, a);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor b7m_prof() {
#ifdef PROF
  auto h = torch::empty({160, 8, 8}, torch::kInt64);
  cudaMemcpyFromSymbol(h.data_ptr(), g_prof, sizeof(unsigned long long) * 160 * 64);
  return h;
#else
  return torch::Tensor();
#endif
}
