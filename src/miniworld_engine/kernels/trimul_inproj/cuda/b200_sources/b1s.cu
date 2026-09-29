// b1s.cu -- TriMul B1 (output-side backward) on B200 for d_pair C = 64: b1r.cu's front / LN-backward CTA roles and L2 ring,
// generalised over the contraction width H (64: one direction, 128: bidirectional).
//
//   front CTAs (NF):  G = xn Wg^T, norm = LN_out(tri), P = norm Wp^T, dP / dG epilogue; dG -> global, dP -> ring slot;
//                     weight gradients as ONE stacked operand: A = [dP | dG] (M = 128: dP's 64 rows, then dG's), so
//                     [dP | dG]^T xn   rows 64..127 = dWg  and  [dP | dG]^T norm  rows 0..63 = dWp  (M = 64 UMMA avoided;
//                     the unused halves cost one extra K = 128 MMA each)
//   LN-bwd CTAs (NL): dP from the ring, dn = bf16(dP Wp) (TMEM, double buffered), LN_out backward -> dTri, dgo / dbo.
// Ring protocol as b1r.cu (front CTA f owns slots f RSF .. f RSF + RSF - 1; flag = generation; flags are 0 at kernel end).
#include "sm100.cuh"
#include "tmap.h"

using namespace sm100;

namespace b1s {

constexpr int C = 64, TOK = 128;
constexpr int XT = TOK * C * 2;            // 16 KB: one SW128 K-block [128 tok][64]
constexpr int WGT = C * C * 2;             // 8 KB
template <int H>
struct Geo {
  static constexpr int TT = H * TOK * 2;   // tri tile [H ch][128 tok] as two 64-token halves of TT / 2
  static constexpr int WPT = C * H * 2;    // Wp [C][H]: H / 64 K-blocks of C * 128 bytes
  static constexpr int WKB = C * 128;
  // front: Wp | two input sets of xn | dP | dG | tri   (dP and dG adjacent: the stacked A operand, LBO = XT)
  static constexpr int O_WP = 0, O_FB = O_WP + WPT, FB = 3 * XT + TT, F_END = O_FB + 2 * FB;
  // LN backward: Wp | dP | tri x 2
  static constexpr int O_DPB = O_WP + WPT, O_T2 = O_DPB + XT, L_END = O_T2 + 2 * TT;
  static constexpr int O_GB = F_END > L_END ? F_END : L_END;   // go [H], bo [H]
  static constexpr int O_BAR = O_GB + 8 * H;
  static constexpr int SMEM = O_BAR + 512;
  static_assert(SMEM <= 232448, "smem");
  static_assert(8 * 2 * H * 4 <= WPT, "LN-bwd partials live in the Wp buffer");
  // front TMEM: stacked dW products, then the G / P accumulator
  static constexpr uint32_t T_DX = 0, T_DT = 64, T0 = 256;   // T0: two 64-column G / P accumulators
  static constexpr uint32_t ID_G = idesc_bf16_mj(128, C, 0, 0);
  static constexpr uint32_t ID_P = idesc_bf16_mj(128, C, 1, 0);
  static constexpr uint32_t ID_DX = idesc_bf16_mj(128, C, 1, 1);
  static constexpr uint32_t ID_DT = idesc_bf16_mj(128, H, 1, 0);
  static constexpr uint32_t ID_DN = idesc_bf16_mj(128, H, 0, 1);
};

struct Bars {
  uint64_t wp_full, in_full[2], g_full[2], p_full[2], dwg_done[2], dg_stored[2], dp_stored[2], dp_ready[2], front_ready;
  uint64_t dpb_full, dpb_free, tl_full[2], tl_free[2], acc_full[2], acc_empty[2], dtri_ready[2];
  uint32_t tmem;
};

DEV float ex2_ftz(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float rcp_ftz(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float sigmoid_kit(float g) { return rcp_ftz(__fadd_rn(1.f, ex2_ftz(__fmul_rn(-1.4426950408889634f, g)))); }
DEV float2 f2(float a) { return make_float2(a, a); }
DEV uint32_t pk2(float2 v) { return pack_bf16(v.x, v.y); }
DEV float2 rbf2(float2 v) { const uint32_t w = pk2(v); return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u)); }
DEV float2 lds64f(uint32_t a) { float2 v; asm volatile("ld.shared.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(a) : "memory"); return v; }
DEV float2 sigmoid2(float2 g) { return make_float2(sigmoid_kit(g.x), sigmoid_kit(g.y)); }
DEV uint32_t sw128(uint32_t r, uint32_t q) { return r * 128u + ((q ^ (r & 7u)) << 4); }
DEV uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory");
  return v;
}
DEV void ldsm_x4_trans(uint32_t addr, uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr) : "memory");
}
DEV void stsm_x4_trans(uint32_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}
DEV void tmem_ld16x256_x4(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
                 "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr) : "memory");
}
DEV float2 up(uint32_t w) { return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u)); }
DEV void red_add(float* p, float v) { asm volatile("red.global.add.f32 [%0], %1;" ::"l"(p), "f"(v) : "memory"); }
DEV void red_add4(float* p, float a, float b, float c, float d) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}
DEV uint32_t ld_acquire(const uint32_t* p) { uint32_t v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
DEV void st_release(uint32_t* p, uint32_t v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
DEV void st_relaxed(uint32_t* p, uint32_t v) { asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
DEV void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
DEV void spin_until(const uint32_t* p, uint32_t v) { while (ld_acquire(p) != v) __nanosleep(32); }

// fragment addressing of a [H ch][128 tok] tile (two 64-token SW128 halves of TT / 2), as b1r.cu
template <int H>
DEV uint32_t frag_addr(uint32_t tbase, int tok0, int jj, int lane) {
  const int mm = lane >> 3, ch = 16 * jj + 8 * (mm & 1) + (lane & 7), qc = ((tok0 & 63) >> 3) + (mm >> 1);
  return tbase + (tok0 >> 6) * (Geo<H>::TT / 2) + sw128(ch, qc);
}

struct Args {
  const __nv_bfloat16* ds;
  const float *mean_o, *rs_o, *go, *bo;
  float *dwg, *dwp, *lnpart;   // lnpart [LN CTAs][dgo H | dbo H]: one row per LN-backward CTA (summed on the host)
  uint32_t* flags;
  int L, tiles, NF, RSF;
};

template <int H>
__global__ void __launch_bounds__(512, 1)
    b1s_kernel(const __grid_constant__ CUtensorMap mxn, const __grid_constant__ CUtensorMap mtri, const __grid_constant__ CUtensorMap mwp,
               const __grid_constant__ CUtensorMap mwg, const __grid_constant__ CUtensorMap mdg, const __grid_constant__ CUtensorMap mdtri,
               const __grid_constant__ CUtensorMap mdy, const __grid_constant__ CUtensorMap mring, const Args a) {
  using G_ = Geo<H>;
  constexpr int TT = G_::TT, WPT = G_::WPT, WKB = G_::WKB;
  constexpr int O_WP = G_::O_WP, O_FB = G_::O_FB, FB = G_::FB, O_DPB = G_::O_DPB, O_T2 = G_::O_T2;
  constexpr int O_GB = G_::O_GB;
  constexpr uint32_t T_DX = G_::T_DX, T_DT = G_::T_DT, T0 = G_::T0;
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + G_::O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, G = gridDim.x, NF = a.NF, NL = G - NF;
  const bool front = cta < NF;
  if (tid == 0 && (su & 1023u)) asm volatile("trap;");
  if (tid == 0) {
    mbar_init(&B.wp_full, 1); mbar_init(&B.front_ready, 1);
    for (int b = 0; b < 2; ++b) {
      mbar_init(&B.in_full[b], 1); mbar_init(&B.g_full[b], 1); mbar_init(&B.p_full[b], 1); mbar_init(&B.dwg_done[b], 1);
      mbar_init(&B.dg_stored[b], 1); mbar_init(&B.dp_stored[b], 1); mbar_init(&B.dp_ready[b], 1);
    }
    mbar_init(&B.dpb_full, 1); mbar_init(&B.dpb_free, 1);
    for (int b = 0; b < 2; ++b) {
      mbar_init(&B.tl_full[b], 1); mbar_init(&B.tl_free[b], 1); mbar_init(&B.acc_full[b], 1); mbar_init(&B.acc_empty[b], 8);
      mbar_init(&B.dtri_ready[b], 8);
    }
    fence_mbar_init();
  }
  if (tid < H) { reinterpret_cast<float*>(sm + O_GB)[tid] = a.go[tid]; reinterpret_cast<float*>(sm + O_GB)[H + tid] = a.bo[tid]; }
  if (warp == 2) { tmem_alloc(&B.tmem, 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  const int LC = a.L / TOK;
  PROF_BEGIN

  if (front) {
    // ======================================================================= FRONT (inputs double buffered: set b = i & 1)
    const int jc = cta % LC;
    auto tile_of = [&](int i) { return cta + NF * i; };
    int n_local = 0;
    while (tile_of(n_local) < a.tiles) ++n_local;
    auto fb = [&](int b) { return O_FB + b * FB; };          // set b: xn | dP (dy) | dG (Wg, G) | tri
    if (warp == 0) {
      if (lane == 0) {
        mbar_expect_tx(&B.wp_full, WPT);
        for (int kb = 0; kb < H / 64; ++kb) tma_load_2d(sm + O_WP + kb * WKB, &mwp, &B.wp_full, kb * 64, 0, EVICT_LAST);
        for (int i = 0; i < n_local; ++i) {
          const int row0 = tile_of(i) * TOK, b = i & 1;
          if (i >= 2) {                                        // tile i-2 of this set: MMAs, dG store and ring store done
            const uint32_t ph = ((i >> 1) - 1) & 1;
            PW(0, mbar_wait(&B.dwg_done[b], ph)); PW(1, mbar_wait(&B.dg_stored[b], ph)); PW(2, mbar_wait(&B.dp_stored[b], ph));
          }
          uint8_t* s = sm + fb(b);
          mbar_expect_tx(&B.in_full[b], 2 * XT + WGT + TT);
          tma_load_2d(s, &mxn, &B.in_full[b], 0, row0, EVICT_FIRST);
          tma_load_2d(s + 2 * XT, &mwg, &B.in_full[b], 0, 0, EVICT_LAST);
          tma_load_2d(s + 3 * XT, &mtri, &B.in_full[b], row0, 0, EVICT_NORMAL);
          tma_load_2d(s + 3 * XT + TT / 2, &mtri, &B.in_full[b], row0 + 64, 0, EVICT_NORMAL);
          tma_load_2d(s + XT, &mdy, &B.in_full[b], 0, row0, EVICT_FIRST);
        }
      }
    } else if (warp == 1) {
      if (lane == 0) {
        mbar_wait(&B.wp_full, 0);
        const uint32_t swp = su + O_WP;
        auto issue_g = [&](int i) {                            // G(i) = xn Wg^T into T0 + 64 b
          const int b = i & 1;
          const uint32_t sx = su + fb(b), sdg = sx + 2 * XT;
          PW(0, mbar_wait(&B.in_full[b], (i >> 1) & 1));
          tc_fence_after();
#pragma unroll
          for (int k = 0; k < 4; ++k) umma_ss(tmem + T0 + 64 * b, desc_k_sw128(sx + k * 32), desc_k_sw128(sdg + k * 32), G_::ID_G, k != 0);
          umma_commit(&B.g_full[b]);
        };
        if (n_local > 0) issue_g(0);
        for (int i = 0; i < n_local; ++i) {
          const int b = i & 1;
          const uint32_t sx = su + fb(b), sdp = sx + XT, st = sx + 3 * XT;
          PW(1, mbar_wait(&B.front_ready, i & 1));
          tc_fence_after();
#pragma unroll
          for (int k = 0; k < H / 16; ++k)
            umma_ss(tmem + T0 + 64 * b, desc_mn_sw128(st + k * 2048, TT / 2), desc_k_sw128(swp + (k >> 2) * WKB + (k & 3) * 32), G_::ID_P,
                    k != 0);
          umma_commit(&B.p_full[b]);
          if (i + 1 < n_local) issue_g(i + 1);                 // the next tile's G while the workers finish this one
          PW(2, mbar_wait(&B.dp_ready[b], (i >> 1) & 1));   // per set: the workers may run two tiles ahead
          tc_fence_after();
#pragma unroll
          for (int k = 0; k < 8; ++k)
            umma_ss(tmem + T_DX, desc_mn_sw128(sdp + k * 2048, XT), desc_mn_sw128(sx + k * 2048, XT), G_::ID_DX, (i | k) != 0);
#pragma unroll
          for (int k = 0; k < 8; ++k)
            umma_ss(tmem + T_DT, desc_mn_sw128(sdp + k * 2048, XT), desc_k_sw128(st + (k >> 2) * (TT / 2) + (k & 3) * 32), G_::ID_DT,
                    (i | k) != 0);
          umma_commit(&B.dwg_done[b]);                         // set b free for tile i + 2
        }
      }
    } else if (warp == 3) {
      if (lane == 0) {
        for (int i = 0; i < n_local; ++i) {
          const int slot = cta * a.RSF + i % a.RSF, b = i & 1;
          PW(0, spin_until(a.flags + slot, 0u));
          PW(1, mbar_wait(&B.dp_ready[b], (i >> 1) & 1));
          tma_store_2d(&mring, sm + fb(b) + XT, 0, slot * TOK);
          bulk_commit();
          PW(2, bulk_wait_read<0>());
          mbar_arrive(&B.dp_stored[b]);
          PW(3, bulk_wait<0>());
          fence_proxy_async_global();
          st_release(a.flags + slot, (uint32_t)(i / a.RSF + 1));
        }
      }
    } else if (warp >= 4) {
      // 12 worker warps: gate write, LN_out forward recompute, dP / dG epilogue
      const int q = warp & 3, kq = (warp - 4) >> 2, wi = warp - 4, t4 = lane & 3, tr = lane >> 2;
      const int c_lo = kq * 4 / 3, c_hi = (kq + 1) * 4 / 3;     // 16-column chunks 0..3: (1, 1, 2)
      uint4 dsv[4];
      {
        const uint4* dsr = reinterpret_cast<const uint4*>(a.ds + (size_t)(jc * TOK + q * 32 + lane) * C);
#pragma unroll
        for (int k = 0; k < 4; ++k)
          if (c_lo * 2 + k < c_hi * 2) dsv[k] = __ldg(dsr + c_lo * 2 + k);
      }
      constexpr int NU = H / 8, HQ = H / 64;       // LN units: 8 token groups x H / 64 channel quarters
      for (int i = 0; i < n_local; ++i) {
        const int row0 = tile_of(i) * TOK, b = i & 1;
        const uint32_t trow = tmem + ((uint32_t)(q * 32) << 16) + T0 + 64 * b;
        const int sdp = fb(b) + XT, sdg = fb(b) + 2 * XT, stt_ = fb(b) + 3 * XT;
        float stt[2][4];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const int u = wi + 12 * k, g = u / HQ;
          if (u < NU) {
            stt[k][0] = __ldg(a.mean_o + row0 + g * 16 + tr); stt[k][1] = __ldg(a.rs_o + row0 + g * 16 + tr);
            stt[k][2] = __ldg(a.mean_o + row0 + g * 16 + tr + 8); stt[k][3] = __ldg(a.rs_o + row0 + g * 16 + tr + 8);
          }
        }
        PW(0, mbar_wait(&B.g_full[b], (i >> 1) & 1));
        tc_fence_after();
        for (int c = c_lo; c < c_hi; ++c) {
          uint32_t v[16];
          tmem_ld16(trow + c * 16, v);
          tmem_wait_ld();
          const int r = q * 32 + lane;
#pragma unroll
          for (int h2 = 0; h2 < 2; ++h2) {
            const int qq = c * 2 + h2;
            const uint32_t* f = v + h2 * 8;
            sts128(su + sdg + sw128(r, qq), pack_bf16(__uint_as_float(f[0]), __uint_as_float(f[1])),
                   pack_bf16(__uint_as_float(f[2]), __uint_as_float(f[3])), pack_bf16(__uint_as_float(f[4]), __uint_as_float(f[5])),
                   pack_bf16(__uint_as_float(f[6]), __uint_as_float(f[7])));
          }
        }
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const int u = wi + 12 * k;
          if (u < NU) {
            const int tok0 = (u / HQ) * 16, jq = u % HQ;
            const float2 nmA = f2(-stt[k][0]), rsA = f2(stt[k][1]), nmB = f2(-stt[k][2]), rsB = f2(stt[k][3]);
#pragma unroll
            for (int j4 = 0; j4 < 4; ++j4) {
              const int jj = jq * 4 + j4;
              const uint32_t ad = frag_addr<H>(su + stt_, tok0, jj, lane);
              const uint32_t ca = su + O_GB + (16 * jj + 2 * t4) * 4;
              const float2 g0 = lds64f(ca), g1 = lds64f(ca + 32), b0 = lds64f(ca + 4 * H), b1 = lds64f(ca + 4 * H + 32);
              uint32_t r[4];
              ldsm_x4_trans(ad, r[0], r[1], r[2], r[3]);
#pragma unroll
              for (int e = 0; e < 4; ++e)
                r[e] = pk2(__ffma2_rn(__fmul2_rn(__fadd2_rn(up(r[e]), (e & 2) ? nmB : nmA), (e & 2) ? rsB : rsA), (e & 1) ? g1 : g0,
                                      (e & 1) ? b1 : b0));
              stsm_x4_trans(ad, r[0], r[1], r[2], r[3]);
            }
          }
        }
        tc_fence_before();
        fence_async_smem();
        PW(1, named_bar_sync(1, 384));
        if (tid == 128) mbar_arrive(&B.front_ready);
        {
          const int r = q * 32 + lane;
          PW(2, mbar_wait(&B.p_full[b], (i >> 1) & 1));
          tc_fence_after();
#pragma unroll
          for (int cc = 0; cc < 2; ++cc) {
            const int c = c_lo + cc;
            if (c < c_hi) {
              uint32_t v[16];
              tmem_ld16(trow + c * 16, v);
              tmem_wait_ld();
#pragma unroll
              for (int h2 = 0; h2 < 2; ++h2) {
                const int qq = c * 2 + h2;
                const uint32_t ga = su + sdg + sw128(r, qq), ya = ga - sdg + sdp;
                const uint4 gw = lds128(ga), dyw = lds128(ya), dsw = dsv[cc * 2 + h2];
                const uint32_t gv[4] = {gw.x, gw.y, gw.z, gw.w}, yv[4] = {dyw.x, dyw.y, dyw.z, dyw.w}, sv[4] = {dsw.x, dsw.y, dsw.z, dsw.w};
                uint32_t odp[4], odg[4];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                  const float2 aa = __fmul2_rn(up(yv[k]), up(sv[k]));
                  const float2 g = sigmoid2(up(gv[k]));
                  const float2 p = rbf2(make_float2(__uint_as_float(v[h2 * 8 + 2 * k]), __uint_as_float(v[h2 * 8 + 2 * k + 1])));
                  odp[k] = pk2(__fmul2_rn(aa, g));
                  odg[k] = pk2(__fmul2_rn(__fmul2_rn(__fmul2_rn(aa, p), g), __fadd2_rn(f2(1.f), make_float2(-g.x, -g.y))));
                }
                sts128(ga, odg[0], odg[1], odg[2], odg[3]);
                sts128(ya, odp[0], odp[1], odp[2], odp[3]);
              }
            }
          }
          tc_fence_before();
          fence_async_smem();
          PW(3, named_bar_sync(2, 384));
          if (tid == 256) {
            mbar_arrive(&B.dp_ready[b]);
            tma_store_2d(&mdg, sm + sdg, 0, row0);
            bulk_commit();
            bulk_wait_read<0>();
            mbar_arrive(&B.dg_stored[b]);
          }
        }
      }
      if (tid == 256) bulk_wait<0>();
    }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    if (warp >= 8) {
      // flush: lanes 64..127 of T_DX = dWg [o][i] (2 x 32 columns), lanes 0..63 of T_DT = dWp [o][h] (H / 32 blocks)
      const int q = warp & 3, part = (warp - 8) >> 2, o = q * 32 + lane;
      const uint32_t ta = tmem + ((uint32_t)(q * 32) << 16);
      if (q >= 2) {
        uint32_t v[32];
        tmem_ld32(ta + T_DX + part * 32, v);
        tmem_wait_ld();
        float* dst = a.dwg + (size_t)(o - 64) * C + part * 32;
#pragma unroll
        for (int k = 0; k < 32; k += 4)
          red_add4(dst + k, __uint_as_float(v[k]), __uint_as_float(v[k + 1]), __uint_as_float(v[k + 2]), __uint_as_float(v[k + 3]));
      } else {
        for (int n = 0; n < H / 64; ++n) {
          const int cb = 2 * ((n + cta) % (H / 64)) + part;
          uint32_t v[32];
          tmem_ld32(ta + T_DT + cb * 32, v);
          tmem_wait_ld();
          float* dst = a.dwp + (size_t)o * H + cb * 32;
#pragma unroll
          for (int k = 0; k < 32; k += 4)
            red_add4(dst + k, __uint_as_float(v[k]), __uint_as_float(v[k + 1]), __uint_as_float(v[k + 2]), __uint_as_float(v[k + 3]));
        }
      }
    }
  } else {
    // ======================================================================= LN BACKWARD
    const int l = cta - NF;
    int n_local = 0;
    while (l + NL * n_local < a.tiles) ++n_local;
    float* acc_s = reinterpret_cast<float*>(sm + O_WP);     // after the last dn product: [8 warps][dgo H, dbo H]
    if (warp == 0) {
      if (lane == 0) {
        mbar_expect_tx(&B.wp_full, WPT);
        for (int kb = 0; kb < H / 64; ++kb) tma_load_2d(sm + O_WP + kb * WKB, &mwp, &B.wp_full, kb * 64, 0, EVICT_LAST);
        int prev_slot = -1;
        for (int j = 0; j < n_local; ++j) {
          const int t = l + NL * j, b = j & 1, row0 = t * TOK, fi = t / NF, slot = (t % NF) * a.RSF + fi % a.RSF;
          if (j >= 2) PW(0, mbar_wait(&B.tl_free[b], ((j >> 1) - 1) & 1));
          mbar_expect_tx(&B.tl_full[b], TT);
          tma_load_2d(sm + O_T2 + b * TT, &mtri, &B.tl_full[b], row0, 0, EVICT_FIRST);
          tma_load_2d(sm + O_T2 + b * TT + TT / 2, &mtri, &B.tl_full[b], row0 + 64, 0, EVICT_FIRST);
          PW(1, spin_until(a.flags + slot, (uint32_t)(fi / a.RSF + 1)));
          fence_proxy_async_global();
          if (j >= 1) {
            PW(2, mbar_wait(&B.dpb_free, (j - 1) & 1));
            st_relaxed(a.flags + prev_slot, 0u);
          }
          mbar_expect_tx(&B.dpb_full, XT);
          tma_load_2d(sm + O_DPB, &mring, &B.dpb_full, 0, slot * TOK, EVICT_FIRST);
          prev_slot = slot;
        }
        if (n_local > 0) {
          mbar_wait(&B.dpb_free, (n_local - 1) & 1);
          st_relaxed(a.flags + prev_slot, 0u);
        }
      }
    } else if (warp == 1) {
      if (lane == 0) {
        mbar_wait(&B.wp_full, 0);
        const uint32_t sdp = su + O_DPB, swp = su + O_WP;
        for (int j = 0; j < n_local; ++j) {
          const int b = j & 1;
          PW(0, mbar_wait(&B.dpb_full, j & 1));
          if (j >= 2) PW(1, mbar_wait(&B.acc_empty[b], ((j >> 1) - 1) & 1));
          tc_fence_after();
          // dn [128 tok][H] = dP [tok][64 o] . Wp [64 o][H]  (B N-major: H / 64 channel blocks WKB apart)
#pragma unroll
          for (int k = 0; k < 4; ++k)
            umma_ss(tmem + b * 256, desc_k_sw128(sdp + k * 32), desc_mn_sw128(swp + k * 2048, WKB), G_::ID_DN, k != 0);
          umma_commit(&B.acc_full[b]);
          umma_commit(&B.dpb_free);
        }
      }
    } else if (warp == 3) {
      if (lane == 0) {
        for (int j = 0; j < n_local; ++j) {
          const int t = l + NL * j, b = j & 1, row0 = t * TOK;
          PW(0, mbar_wait(&B.dtri_ready[b], (j >> 1) & 1));
          tma_store_2d(&mdtri, sm + O_T2 + b * TT, row0, 0);
          tma_store_2d(&mdtri, sm + O_T2 + b * TT + TT / 2, row0 + 64, 0);
          bulk_commit();
          PW(1, bulk_wait_read<0>());
          mbar_arrive(&B.tl_free[b]);
        }
        bulk_wait<0>();
      }
    } else if (warp >= 4 && warp < 12) {
      constexpr int NB = H / 64;
      const int q = warp & 3, hh = (warp - 4) >> 2, t4 = lane & 3, tr = lane >> 2, tok0 = q * 32 + hh * 16;
      float dacc[NB][4];
#pragma unroll
      for (int x = 0; x < NB; ++x)
#pragma unroll
        for (int k = 0; k < 4; ++k) dacc[x][k] = 0.f;
      for (int j = 0; j < n_local; ++j) {
        const int t = l + NL * j, b = j & 1, row0 = t * TOK;
        const uint32_t tb = su + O_T2 + b * TT;
        const uint32_t ta = tmem + ((uint32_t)tok0 << 16) + b * 256;
        const float2 nmA = f2(-__ldg(a.mean_o + row0 + tok0 + tr)), rsA = f2(__ldg(a.rs_o + row0 + tok0 + tr));
        const float2 nmB = f2(-__ldg(a.mean_o + row0 + tok0 + tr + 8)), rsB = f2(__ldg(a.rs_o + row0 + tok0 + tr + 8));
        PW(0, mbar_wait(&B.acc_full[b], (j >> 1) & 1));
        PW(1, mbar_wait(&B.tl_full[b], (j >> 1) & 1));
        tc_fence_after();
        auto dn_pair = [&](int jj, int e, const uint32_t (&tv)[16]) {
          const int jx = 2 * (jj & 1) + (e & 1), r0 = 4 * jx + 2 * (e >> 1);
          return rbf2(make_float2(__uint_as_float(tv[r0]), __uint_as_float(tv[r0 + 1])));
        };
        float2 s1A = f2(0.f), s2A = f2(0.f), s1B = f2(0.f), s2B = f2(0.f);
#pragma unroll 1
        for (int blk = 0; blk < NB; ++blk) {
          float2 v[16];
#pragma unroll
          for (int k = 0; k < 16; ++k) v[k] = f2(0.f);
          uint32_t tv[16];
#pragma unroll
          for (int j4 = 0; j4 < 4; ++j4) {
            const int jj = 4 * blk + j4;
            if ((j4 & 1) == 0) { tmem_ld16x256_x4(ta + (jj >> 1) * 32, tv); tmem_wait_ld(); }
            uint32_t tt[4];
            ldsm_x4_trans(frag_addr<H>(tb, tok0, jj, lane), tt[0], tt[1], tt[2], tt[3]);
            const uint32_t ca = su + O_GB + (16 * jj + 2 * t4) * 4;
            const float2 g0 = lds64f(ca), g1 = lds64f(ca + 32);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const float2 dn = dn_pair(jj, e, tv);
              const float2 xh = __fmul2_rn(__fadd2_rn(up(tt[e]), (e & 2) ? nmB : nmA), (e & 2) ? rsB : rsA);
              const float2 dg = __fmul2_rn(dn, (e & 1) ? g1 : g0);
              if (e & 2) { s1B = __ffma2_rn(dg, xh, s1B); s2B = __fadd2_rn(s2B, dg); } else { s1A = __ffma2_rn(dg, xh, s1A); s2A = __fadd2_rn(s2A, dg); }
              const int b4 = j4 * 4 + (e & 1) * 2;
              v[b4] = __ffma2_rn(dn, xh, v[b4]);
              v[b4 + 1] = __fadd2_rn(v[b4 + 1], dn);
            }
          }
          float* vf = reinterpret_cast<float*>(v);
#pragma unroll
          for (int s = 0, msk = 16, nh = 16; s < 3; ++s, msk >>= 1, nh >>= 1) {
            const bool hi = (lane & msk) != 0;
#pragma unroll
            for (int k = 0; k < 16; ++k) {
              if (k < nh) {
                const float send = hi ? vf[k] : vf[k + nh];
                const float keep = hi ? vf[k + nh] : vf[k];
                vf[k] = keep + __shfl_xor_sync(0xffffffffu, send, msk);
              }
            }
          }
#pragma unroll
          for (int k = 0; k < 4; ++k) dacc[blk][k] += vf[k];
        }
        float c1A = s1A.x + s1A.y, c2A = s2A.x + s2A.y, c1B = s1B.x + s1B.y, c2B = s2B.x + s2B.y;
#pragma unroll
        for (int o = 1; o <= 2; o <<= 1) {
          c1A += __shfl_xor_sync(0xffffffffu, c1A, o); c2A += __shfl_xor_sync(0xffffffffu, c2A, o);
          c1B += __shfl_xor_sync(0xffffffffu, c1B, o); c2B += __shfl_xor_sync(0xffffffffu, c2B, o);
        }
        const float2 nc1A = f2(-c1A * (1.f / H)), nc2A = f2(-c2A * (1.f / H)), nc1B = f2(-c1B * (1.f / H)), nc2B = f2(-c2B * (1.f / H));
        uint32_t tv[16];
#pragma unroll 2
        for (int jj = 0; jj < H / 16; ++jj) {
          if ((jj & 1) == 0) { tmem_ld16x256_x4(ta + (jj >> 1) * 32, tv); tmem_wait_ld(); }
          const uint32_t ad = frag_addr<H>(tb, tok0, jj, lane);
          uint32_t tt[4];
          ldsm_x4_trans(ad, tt[0], tt[1], tt[2], tt[3]);
          const uint32_t ca = su + O_GB + (16 * jj + 2 * t4) * 4;
          const float2 g0 = lds64f(ca), g1 = lds64f(ca + 32);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float2 dn = dn_pair(jj, e, tv);
            const float2 xh = __fmul2_rn(__fadd2_rn(up(tt[e]), (e & 2) ? nmB : nmA), (e & 2) ? rsB : rsA);
            const float2 tq = __ffma2_rn(xh, (e & 2) ? nc1B : nc1A, __ffma2_rn(dn, (e & 1) ? g1 : g0, (e & 2) ? nc2B : nc2A));
            tt[e] = pk2(__fmul2_rn(tq, (e & 2) ? rsB : rsA));
          }
          stsm_x4_trans(ad, tt[0], tt[1], tt[2], tt[3]);
        }
        tc_fence_before();
        fence_async_smem();
        __syncwarp();
        if (lane == 0) { mbar_arrive(&B.acc_empty[b]); mbar_arrive(&B.dtri_ready[b]); }
      }
#pragma unroll
      for (int blk = 0; blk < NB; ++blk) {
        const int ch = 16 * (4 * blk + (tr >> 1)) + 8 * (tr & 1) + 2 * t4;
        float* pw = acc_s + (hh * 4 + q) * 2 * H;
        *reinterpret_cast<float2*>(pw + ch) = make_float2(dacc[blk][0], dacc[blk][1]);
        *reinterpret_cast<float2*>(pw + H + ch) = make_float2(dacc[blk][2], dacc[blk][3]);
      }
    }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    if (tid < H) {
      float sgo = 0.f, sbo = 0.f;
#pragma unroll
      for (int w = 0; w < 8; ++w) { sgo += acc_s[w * 2 * H + tid]; sbo += acc_s[w * 2 * H + H + tid]; }
      a.lnpart[(size_t)l * 2 * H + tid] = sgo; a.lnpart[(size_t)l * 2 * H + H + tid] = sbo;
    }
  }
  if (lane == 0 && warp < 5) PROF_END(warp);
  tc_fence_before();
  __syncthreads();
  if (warp == 2) tmem_dealloc(tmem, 512);
}

}  // namespace b1s

// dy / xn / dg [M, 64]; tri / dtri [H, M] channel-major; ds [L, 64] bf16; mean_o / rs_o [M]; wg [64, 64], wp [64, H] bf16;
// go / bo [H] fp32; dwg [64, 64], dwp [64, H] fp32 accumulated; lnpart [>= SMs - nf, 2H] fp32: per-LN-CTA (dg_o | db_o)
// rows, written (sum them); ring [nf RSF 128, 64] bf16; flags [nf RSF] zero.
void b1s_backward(torch::Tensor dy, torch::Tensor xn, torch::Tensor tri, torch::Tensor ds, torch::Tensor mean_o, torch::Tensor rs_o,
                  torch::Tensor wg, torch::Tensor wp, torch::Tensor go, torch::Tensor bo, torch::Tensor dg, torch::Tensor dtri,
                  torch::Tensor dwg, torch::Tensor dwp, torch::Tensor lnpart, torch::Tensor ring, torch::Tensor flags,
                  int64_t L, int64_t nf) {
  using namespace b1s;
  const int64_t M = dy.size(0);
  const int H = (int)tri.size(0);
  TORCH_CHECK(dy.size(1) == C && M == L * L && L % TOK == 0 && tri.numel() == H * M && (H == 64 || H == 128));
  int nsm = 0, dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  const int RS = (int)(ring.size(0) / TOK), RSF = RS / (int)nf;
  TORCH_CHECK(nf > 0 && nf < nsm && flags.numel() >= RS && RSF >= 1 && nf % (L / TOK) == 0);
  auto bf = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  auto tok_map = [&](const torch::Tensor& t, int64_t rows) {
    return tmap::make(t.data_ptr(), bf, {(uint64_t)C, (uint64_t)rows}, {(uint64_t)C * 2}, {64, TOK}, CU_TENSOR_MAP_SWIZZLE_128B);
  };
  auto ch_map = [&](const torch::Tensor& t) {
    return tmap::make(t.data_ptr(), bf, {(uint64_t)M, (uint64_t)H}, {(uint64_t)M * 2}, {64, (uint32_t)H}, CU_TENSOR_MAP_SWIZZLE_128B);
  };
  auto mxn = tok_map(xn, M), mdg = tok_map(dg, M), mdy = tok_map(dy, M), mring = tok_map(ring, ring.size(0));
  auto mtri = ch_map(tri), mdtri = ch_map(dtri);
  auto mwp = tmap::make(wp.data_ptr(), bf, {(uint64_t)H, (uint64_t)C}, {(uint64_t)H * 2}, {64, (uint32_t)C}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mwg = tmap::make(wg.data_ptr(), bf, {(uint64_t)C, (uint64_t)C}, {(uint64_t)C * 2}, {64, 64}, CU_TENSOR_MAP_SWIZZLE_128B);
  Args a;
  a.ds = reinterpret_cast<const __nv_bfloat16*>(ds.data_ptr());
  a.mean_o = mean_o.data_ptr<float>(); a.rs_o = rs_o.data_ptr<float>(); a.go = go.data_ptr<float>(); a.bo = bo.data_ptr<float>();
  TORCH_CHECK(lnpart.numel() >= (int64_t)(nsm - nf) * 2 * H);
  a.dwg = dwg.data_ptr<float>(); a.dwp = dwp.data_ptr<float>(); a.lnpart = lnpart.data_ptr<float>();
  a.flags = reinterpret_cast<uint32_t*>(flags.data_ptr<int32_t>());
  a.L = (int)L; a.tiles = (int)(M / TOK); a.NF = (int)nf; a.RSF = RSF;
  void* params[] = {(void*)&mxn, (void*)&mtri, (void*)&mwp, (void*)&mwg, (void*)&mdg, (void*)&mdtri, (void*)&mdy, (void*)&mring, (void*)&a};
  auto launch = [&](const void* fn, int smem, bool& attr) {
    if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); attr = true; }
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(fn, dim3(nsm), dim3(512), params, smem, at::cuda::getCurrentCUDAStream()));
  };
  static bool attr64 = false, attr128 = false;
  if (H == 64) launch((const void*)b1s_kernel<64>, Geo<64>::SMEM, attr64);
  else launch((const void*)b1s_kernel<128>, Geo<128>::SMEM, attr128);
}

torch::Tensor b1s_prof() {
#ifdef PROF
  auto h = torch::empty({160, 8, 8}, torch::kInt64);
  cudaMemcpyFromSymbol(h.data_ptr(), g_prof, sizeof(unsigned long long) * 160 * 64);
  return h;
#else
  return torch::Tensor();
#endif
}
