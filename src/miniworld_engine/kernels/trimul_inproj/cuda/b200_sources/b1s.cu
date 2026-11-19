// b1s.cu -- TriMul B1 (output-side backward) on B200, d_pair C = 64, contraction width H (128 bidirectional / one direction with
// hidden 2 D, 64 one direction). b1g's design (see b1g.cu) at C = 64, where every per-tile operand fits double buffered:
//
// LN_out's affine is folded into the projection: Wp' = bf16(Wp diag(g_o)), u = rowsum(Wp'), c = Wp b_o; with Q = tri Wp'^T on the
// raw contraction output
//   P = rs (Q - mean u) + c,      dP = dy ds sig(G),  dG = dy ds P sig(G) (1 - sig(G)),  dPr = bf16(dP rs)
//   dnr = dPr Wp' (= rs g_o dn),  dTri = dnr - S2 / H - (tri - mean) rs^2 S1 / H,
//   S1 = sum_o dPr (Q - mean u),  S2 = sum_o dPr u   (formed by the front epilogue)
//   A = sum_tok dPr^T tri,  v = sum_tok dPr mean,  s = sum_tok dPr / rs:
//   dWp = g_o (A - v 1^T) + s b_o^T,   dg_o = colsum(Wp o (A - v 1^T)),   db_o = Wp^T s
// The M = 64 products run as M = 128 MMAs with a padding block of rows nothing reads: dWg = [xn | dG]^T xn (dWg in TMEM lanes
// 64..127), A = dPr^T tri with rows 64..127 reading the shared memory after dPr (A in lanes 0..63). Every padding read stays
// inside the CTA's shared memory.
//
// Roles (cooperative grid, one CTA per SM; both form Wp' in shared memory at start):
//   front CTAs (NF): per tile set b = i & 1 (tri, xn, dy double buffered): Q = tri Wp'^T and G = xn Wg^T (TMEM, double
//                    buffered: the next tile's run during this tile's epilogue), P / dP / dG / S1 / S2 epilogue: dPr and (S1, S2)
//                    stored from registers into this CTA's ring slot, dG in place of dy -> global (TMA), dWg += dG^T xn (TMEM);
//                    the flag (generation) is published by warp 2.
//   LN CTAs (NL):    dPr / S and tri double buffered; A += dPr^T tri, dnr = dPr Wp' (TMEM, double buffered), LN backward in one
//                    pass -> dTri in place (TMA store); v / s on four warps; dWp, dg_o, db_o at the end.
// tri / dTri: [H, M] (2-D maps) or sample-major [H / 64, B, 64, L, L] (4-D maps, tmap::make_chmajor; a tile never crosses a sample).
// Partials (plain stores; grad_reduce sums them in a fixed order): part row r < max(NF, NL) = (dWg of front CTA r | dWp of LN CTA r),
// zeros where that CTA does not exist; lnpart row l = (dg_o | db_o) of LN CTA l. Deterministic.
// Ring: front CTA f takes tiles t = f + NF i (NF a multiple of L / 128: one j class, so its ds rows are fixed per sample) and owns
// slots f RSF .. f RSF + RSF - 1: tile i -> slot f RSF + i % RSF, generation i / RSF + 1. Flags are 0 at kernel end. The ring holds
// RS = NF RSF dPr tiles of 16 KB in the SWIZZLE_NONE K-major operand layout [8 column chunks][128 tok][8 o], then RS (S1 [128] |
// S2 [128]) fp32 blocks of 1 KB. Hand-over (generic stores -> async-proxy bulk copies on another SM): epilogue threads' st.global
// -> fence.proxy.async.global (each writer) -> bar.sync -> mbarrier dp_ready -> published -> flag warp st.release.gpu;
// LN producer ld.acquire.gpu -> fence.proxy.async.global -> cp.async.bulk.
#include "sm100.cuh"
#include "tmap.h"
#include "pt_tl.cuh"

using namespace sm100;

namespace b1s {

constexpr int C = 64, TOK = 128;
constexpr int XT = TOK * C * 2;            // 16 KB: a [128 tok][64] tile, one SW128 K-block
constexpr int KB = C * 128;                // 8 KB: one 64-channel block of Wp' [C o][H ch]; also Wg [C][C]
constexpr int SBB = 2 * TOK * 4;           // 1 KB: a tile's per-token (S1 | S2)
template <int H>
struct Geo {
  static constexpr int TT = H * TOK * 2;   // tri tile [H ch][128 tok], two 64-token halves (TT / 2 apart)
  static constexpr int WPT = C * H * 2;    // Wp' (both roles, at 0)
  // front: Wp' | Wg | 2 x (tri | xn | dy -> dG) | u, c | S partials
  static constexpr int F_WG = WPT, F_SET = F_WG + KB, SET = TT + 2 * XT, F_UC = F_SET + 2 * SET, F_SB = F_UC + 2 * C * 4;
  static constexpr int F_END = F_SB + SBB;
  // LN: Wp' | 2 x dPr | 2 x tri | 2 x (S1 | S2) | v, s
  static constexpr int L_DP = WPT, L_T = L_DP + 2 * XT, L_SB = L_T + 2 * TT, L_VF = L_SB + 2 * SBB, L_END = L_VF + 2 * C * 4;
  static constexpr int E_LD = C + 1;       // LN tail: E [H][C + 1] fp32 at 0
  static constexpr int O_BAR = F_END > L_END ? F_END : L_END;
  static constexpr int SMEM = O_BAR + 256;
  static_assert(SMEM <= 232448, "smem");
  static_assert(H * E_LD * 4 <= L_VF, "LN tail scratch");
  static_assert(F_SET % 1024 == 0 && SET % 1024 == 0 && L_T % 1024 == 0 && TT % 1024 == 0, "SW128 bases");
  static constexpr uint32_t T_DWG = 0, T_G = 64, T_Q = 192;   // front TMEM: dWg (lanes 64..127), G x 2, Q x 2 (64 columns each)
  static constexpr uint32_t T_D = H;                          // LN TMEM: A at 0 (H columns, rows 0..63), dnr x 2 at H
  static constexpr uint32_t ID_GQ = idesc_bf16_mj(128, C, 0, 0);    // G: A = xn (K-major), B = Wg (K-major)
  static constexpr uint32_t ID_Q = idesc_bf16_mj(128, C, 1, 0);     // Q: A = tri^T (MN-major), B = Wp' (K-major)
  static constexpr uint32_t ID_DWG = idesc_bf16_mj(128, C, 1, 1);   // dWg: A = [xn | dG]^T (MN-major, rows 0..63 padding), B = xn
  static constexpr uint32_t ID_A = idesc_bf16_mj(128, H, 1, 0);     // A: A = dPr^T (MN-major, rows 64.. padding), B = tri (K-major)
  static constexpr uint32_t ID_DN = idesc_bf16_mj(128, H, 0, 1);    // dnr: A = dPr (K-major), B = Wp' (MN-major)
};

struct FBars {
  uint64_t wg_full, in_full[2], in_free[2], qg_full[2], tm_empty[2], dp_ready, slot_ok, published, rel_done, dwg_last;
};
struct LBars {
  uint64_t tl_full[2], tl_free[2], dpb_full[2], dpb_free[2], a_done[2], dn_full[2], dn_empty[2], dtri_ready[2];
};
struct Bars {
  uint64_t wp_full;
  union { FBars f; LBars n; };
  uint32_t tmem;
};

DEV void tmem_ld8(uint32_t taddr, uint32_t (&r)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr) : "memory");
}
DEV float tanh_ap(float x) { float y; asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float rcp_ftz(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float2 f2(float a) { return make_float2(a, a); }
DEV uint32_t pk2(float2 v) { return pack_bf16(v.x, v.y); }
DEV float2 up(uint32_t w) { return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u)); }
DEV float2 rbf2(float2 v) { return up(pk2(v)); }
DEV float2 sig2(float2 g) { return __ffma2_rn(make_float2(tanh_ap(0.5f * g.x), tanh_ap(0.5f * g.y)), f2(0.5f), f2(0.5f)); }
DEV float2 sigh2(float2 h) { return __ffma2_rn(make_float2(tanh_ap(h.x), tanh_ap(h.y)), f2(0.5f), f2(0.5f)); }   // sigmoid(2 h)
DEV uint32_t sw128(uint32_t r, uint32_t q) { return r * 128u + ((q ^ (r & 7u)) << 4); }
DEV uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory");
  return v;
}
DEV float4 ldsf4(uint32_t a) {
  float4 v;
  asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a) : "memory");
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
DEV uint32_t ld_acquire(const uint32_t* p) { uint32_t v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
DEV void st_release(uint32_t* p, uint32_t v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
DEV void st_relaxed(uint32_t* p, uint32_t v) { asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
DEV void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
DEV void spin_until(const uint32_t* p, uint32_t v) { while (ld_acquire(p) != v) __nanosleep(32); }
DEV void bulk_load(uint32_t sdst, const void* gsrc, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(sdst), "l"(gsrc), "r"(bytes),
               "r"(smem_u32(bar)) : "memory");
}
// SWIZZLE_NONE operand: 8-row x 16-byte core matrices; lbo = byte stride between core matrices along K, sbo = along M / N
DEV uint64_t desc_nosw(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((saddr >> 4) & 0x3FFFu) | ((uint64_t)((lbo >> 4) & 0x3FFFu) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFFu) << 32) |
         ((uint64_t)1 << 46);
}

// ldmatrix / stmatrix fragment address of a [H ch][128 tok] tile (two 64-token SW128 halves): channel group jj (16 channels)
template <int H>
DEV uint32_t frag_addr(uint32_t tbase, int tok0, int jj, int lane) {
  const int mm = lane >> 3, ch = 16 * jj + 8 * (mm & 1) + (lane & 7), qc = ((tok0 & 63) >> 3) + (mm >> 1);
  return tbase + (tok0 >> 6) * (Geo<H>::TT / 2) + sw128(ch, qc);
}
DEV float2 dn_pair(int jj, int e, const uint32_t (&tv)[16]) {
  const int jx = 2 * (jj & 1) + (e & 1), r0 = 4 * jx + 2 * (e >> 1);
  return make_float2(__uint_as_float(tv[r0]), __uint_as_float(tv[r0 + 1]));
}

struct Args {
  const __nv_bfloat16 *ds, *wp, *wpf;   // wpf [C][H] = bf16(Wp g_o) from the stored parameter (trimul_forms)
  const float *mean_o, *rs_o, *go, *bo;
  float *part, *lnpart;     // part [max(NF, NL)][C][C + H] (row o = dWg[o] | dWp[o]); lnpart [NL][dg_o H | db_o H]
  uint32_t* flags;          // [NF RSF]
  float* sring;             // [NF RSF][S1 128 | S2 128]
  __nv_bfloat16* ring;      // [NF RSF][8][128][8] dPr tiles
  int L, tiles, NF, RSF;
  int n2;                   // > 0: tri / dtri sample-major (4-D maps), 0: [H, M]
};

#ifndef B1U_T1
#define B1U_T1 0   // (2026-11-18 part 11) b1u, sample-major tri: the tile's two 64-token halves in ONE TMA box (5-D map); bitwise, measured level (D64 L128 out 0.0518 / 0.0516 ms, in 0.0539 / 0.0532-0.0539, L256 level): off
#endif
#if B1U_T1
#define B1U_T1_ON 1
#else
#define B1U_T1_ON 0
#endif
// one box = the 128-token tri tile as two SW128 [64 ch][64 tok] halves (map tri5_map: (token mod 64, channel, token / 64, sample, chunk))
DEV void tri_load1(void* dst, const CUtensorMap* m, uint64_t* bar, int row0, int n2, uint64_t hint) {
  int tq, ts;
  tok_split(row0, n2, tq, ts);
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;" ::"r"(
          smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(m)), "r"(smem_u32(bar)), "r"(0), "r"(0), "r"(tq / 64), "r"(ts), "r"(0), "l"(hint)
      : "memory");
}
template <int SMJ>
DEV void tri_load(void* dst, const CUtensorMap* m, uint64_t* bar, int row0, int n2, uint64_t hint) {
  if constexpr (SMJ != 0) {
    int tq, ts;
    tok_split(row0, n2, tq, ts);
    tma_load_4d(dst, m, bar, tq, 0, ts, 0, hint);
  } else {
    tma_load_2d(dst, m, bar, row0, 0, hint);
  }
}
template <int SMJ>
DEV void tri_store(const CUtensorMap* m, const void* src, int row0, int n2) {
  if constexpr (SMJ != 0) {
    int tq, ts;
    tok_split(row0, n2, tq, ts);
    tma_store_4d(m, src, tq, 0, ts, 0);
  } else {
    tma_store_2d(m, src, row0, 0);
  }
}

#ifndef B1S_PDL
#define B1S_PDL 1   // cooperative + programmatic dependent of k3g: set-up, TMEM, the Wp' formation overlap k3g's last tiles (with K3_TRIG)
#endif
template <int H, int SMJ>
__global__ void __launch_bounds__(512, 1)
    b1s_kernel(const __grid_constant__ CUtensorMap mxn, const __grid_constant__ CUtensorMap mtri, const __grid_constant__ CUtensorMap mwp,
               const __grid_constant__ CUtensorMap mwg, const __grid_constant__ CUtensorMap mdg, const __grid_constant__ CUtensorMap mdtri,
               const __grid_constant__ CUtensorMap mdy, const Args a) {
  using G_ = Geo<H>;
  constexpr int TT = G_::TT, WPT = G_::WPT, SET = G_::SET;
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + G_::O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, NF = a.NF, NL = (int)gridDim.x - NF;
  const bool front = cta < NF;
  if (tid == 0 && (su & 1023u)) asm volatile("trap;");
  if (tid == 0) {
    mbar_init(&B.wp_full, 1);
    if (front) {
      FBars& F = B.f;
      mbar_init(&F.wg_full, 1); mbar_init(&F.dp_ready, 1); mbar_init(&F.slot_ok, 1); mbar_init(&F.published, 1);
      mbar_init(&F.rel_done, 1); mbar_init(&F.dwg_last, 1);
      for (int b = 0; b < 2; ++b) {
        mbar_init(&F.in_full[b], 1); mbar_init(&F.in_free[b], 2); mbar_init(&F.qg_full[b], 1); mbar_init(&F.tm_empty[b], 1);
      }
    } else {
      LBars& N = B.n;
      for (int b = 0; b < 2; ++b) {
        mbar_init(&N.tl_full[b], 1); mbar_init(&N.tl_free[b], 1); mbar_init(&N.dpb_full[b], 1); mbar_init(&N.dpb_free[b], 1 + 4 + 8);
        mbar_init(&N.a_done[b], 1); mbar_init(&N.dn_full[b], 1); mbar_init(&N.dn_empty[b], 8); mbar_init(&N.dtri_ready[b], 8);
      }
    }
    fence_mbar_init();
    mbar_expect_tx(&B.wp_full, WPT);
    for (int kb = 0; kb < H / 64; ++kb) tma_load_2d(sm + kb * KB, &mwp, &B.wp_full, kb * 64, 0, EVICT_LAST);
  }
  if (warp == 2) { tmem_alloc(&B.tmem, 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  // ---- Wp' = wpf (bf16(Wp g_o) of the stored Wp, trimul_forms) in place (8 threads per row o); front: u = rowsum(Wp'), c = Wp b_o
  mbar_wait(&B.wp_full, 0);
  {
    constexpr int CPT = H / 8;
    const int o = tid >> 3, pq = tid & 7;
    float u = 0.f, cb = 0.f;
#pragma unroll 1
    for (int m = 0; m < CPT / 8; ++m) {
      const int ch0 = pq * CPT + 8 * m;
      const uint32_t ad = su + (ch0 >> 6) * KB + sw128(o, (ch0 & 63) >> 3);
      const uint4 w = lds128(ad);
      const uint4 wq = __ldg(reinterpret_cast<const uint4*>(a.wpf + (size_t)o * H + ch0));   // bf16(Wp g_o), one rounding (trimul_forms)
      const float4 ba = __ldg(reinterpret_cast<const float4*>(a.bo + ch0)), bb = __ldg(reinterpret_cast<const float4*>(a.bo + ch0 + 4));
      const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
      const uint32_t qv[4] = {wq.x, wq.y, wq.z, wq.w};
      const float2 bv[4] = {make_float2(ba.x, ba.y), make_float2(ba.z, ba.w), make_float2(bb.x, bb.y), make_float2(bb.z, bb.w)};
      uint32_t ov[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float2 wf = up(wv[k]);
        const float2 sc = up(qv[k]);
        u += sc.x + sc.y;
        cb = fmaf(wf.x, bv[k].x, cb);
        cb = fmaf(wf.y, bv[k].y, cb);
        ov[k] = pk2(sc);
      }
      sts128(ad, ov[0], ov[1], ov[2], ov[3]);
    }
#pragma unroll
    for (int x = 1; x <= 4; x <<= 1) { u += __shfl_xor_sync(0xffffffffu, u, x); cb += __shfl_xor_sync(0xffffffffu, cb, x); }
    if (front && pq == 0) {
      reinterpret_cast<float*>(sm + G_::F_UC)[o] = u;
      reinterpret_cast<float*>(sm + G_::F_UC)[C + o] = cb;
    }
  }
  fence_async_smem();
  __syncthreads();
  if (B1S_PDL) pdl_wait();   // xn, t, the LN_out statistics and the zeroed ring flags are the previous kernel's (k3g) output
  const int LC = a.L / TOK;
  PROF_BEGIN

  if (front) {
    // ======================================================================= FRONT (set b = i & 1: tri | xn | dy -> dG)
    FBars& F = B.f;
    const int jc = cta % LC;
    auto tile_of = [&](int i) { return cta + NF * i; };
    int n_local = 0;
    while (tile_of(n_local) < a.tiles) ++n_local;
    auto set_of = [&](int b) { return (uint32_t)(G_::F_SET + b * SET); };   // tri at +0, xn at +TT, dy / dG at +TT + XT
    if (warp == 0) {
      if (lane == 0) {
        mbar_expect_tx(&F.wg_full, KB);
        tma_load_2d(sm + G_::F_WG, &mwg, &F.wg_full, 0, 0, EVICT_LAST);
        for (int i = 0; i < n_local; ++i) {
          const int row0 = tile_of(i) * TOK, b = i & 1;
          if (i >= 2) PW(0, mbar_wait(&F.in_free[b], ((i >> 1) - 1) & 1));  // tile i-2: dWg read xn / dG, dG stored
          uint8_t* s = sm + set_of(b);
          mbar_expect_tx(&F.in_full[b], TT + 2 * XT);
          tri_load<SMJ>(s, &mtri, &F.in_full[b], row0, a.n2, EVICT_NORMAL);
          tri_load<SMJ>(s + TT / 2, &mtri, &F.in_full[b], row0 + 64, a.n2, EVICT_NORMAL);
          tma_load_2d(s + TT, &mxn, &F.in_full[b], 0, row0, EVICT_FIRST);
          tma_load_2d(s + TT + XT, &mdy, &F.in_full[b], 0, row0, EVICT_FIRST);
        }
      }
    } else if (warp == 1) {
      if (lane == 0) {
        const uint32_t swg = su + G_::F_WG, swp = su;
        // Q_i [128 tok][64 o] = tri^T . Wp'^T (K = H), G_i = xn . Wg^T (K = 64), into TMEM set b
        auto mma_qg = [&](int i) {
          const int b = i & 1;
          const uint32_t s = su + set_of(b);
          PW(0, mbar_wait(&F.in_full[b], (i >> 1) & 1));
          if (i >= 2) PW(0, mbar_wait(&F.tm_empty[b], ((i >> 1) - 1) & 1));
          tc_fence_after();
#pragma unroll
          for (int k = 0; k < H / 16; ++k)
            umma_ss(tmem + G_::T_Q + 64 * b, desc_mn_sw128(s + k * 2048, TT / 2), desc_k_sw128(swp + (k >> 2) * KB + (k & 3) * 32), G_::ID_Q, k != 0);
#pragma unroll
          for (int k = 0; k < 4; ++k)
            umma_ss(tmem + G_::T_G + 64 * b, desc_k_sw128(s + TT + k * 32), desc_k_sw128(swg + k * 32), G_::ID_GQ, k != 0);
          umma_commit(&F.qg_full[b]);
        };
        if (n_local > 0) { mbar_wait(&F.wg_full, 0); mma_qg(0); }
        for (int i = 0; i < n_local; ++i) {
          if (i + 1 < n_local) mma_qg(i + 1);
          const int b = i & 1;
          const uint32_t s = su + set_of(b);
          PW(1, mbar_wait(&F.dp_ready, i & 1));
          tc_fence_after();
          // dWg [(64 padding rows +) 64 o][64 i] += dG^T [o][tok] . xn [tok][i]: the M = 128 A operand's first 64-row block is xn (the
          // padding, TMEM lanes 0..63), its second dG = xn + XT (lanes 64..127)
#pragma unroll
          for (int k = 0; k < 8; ++k)
            umma_ss(tmem + G_::T_DWG, desc_mn_sw128(s + TT + k * 2048, XT), desc_mn_sw128(s + TT + k * 2048, XT), G_::ID_DWG, (i | k) != 0);
          umma_commit(&F.in_free[b]);
          if (i == n_local - 1) umma_commit(&F.dwg_last);
        }
        if (n_local > 0) mbar_wait(&F.dwg_last, 0);
      }
    } else if (warp == 2) {
      if (lane == 0) {
        for (int i = 0; i < n_local; ++i) {
          const int slot = cta * a.RSF + i % a.RSF;
          PW(0, mbar_wait(&F.published, i & 1));
          PW(1, st_release(a.flags + slot, (uint32_t)(i / a.RSF + 1)));
          mbar_arrive(&F.rel_done);
        }
      }
    } else if (warp == 3) {
      if (lane == 0) {
        // ring slot free (its previous tile consumed) -> the epilogue may store into it; dG -> global
        for (int i = 0; i < n_local; ++i) {
          const int row0 = tile_of(i) * TOK, slot = cta * a.RSF + i % a.RSF, b = i & 1;
          PW(1, spin_until(a.flags + slot, 0u));
          mbar_arrive(&F.slot_ok);
          PW(0, mbar_wait(&F.dp_ready, i & 1));
          tma_store_2d(&mdg, sm + set_of(b) + TT + XT, 0, row0);
          bulk_commit();
          if (i >= 1) PW(4, mbar_wait(&F.rel_done, (i - 1) & 1));
          mbar_arrive(&F.published);
          PW(2, bulk_wait_read<0>());
          mbar_arrive(&F.in_free[b]);
        }
        bulk_wait<0>();
      }
    } else if (warp >= 4) {
      // 12 epilogue warps: lane quarter q (tokens 32 q ..), 8-column chunks c_lo .. c_hi - 1 of the 64 outputs
      const int q = warp & 3, kq = (warp - 4) >> 2, r = q * 32 + lane;
      const uint32_t trow = tmem + ((uint32_t)(q * 32) << 16);
      const int c_lo = kq * 3, c_hi = kq == 2 ? 8 : kq * 3 + 3;
      const uint32_t suc = su + G_::F_UC;
      float* sb = reinterpret_cast<float*>(sm + G_::F_SB);
      uint4 dsv[3];
      int cur_sb = -1;
      for (int i = 0; i < n_local; ++i) {
        const int row0 = tile_of(i) * TOK, b = i & 1;
        {
          const int sbi = tile_of(i) / (a.L * LC);
          if (sbi != cur_sb) {
            const uint4* dsr = reinterpret_cast<const uint4*>(a.ds + ((size_t)sbi * a.L + jc * TOK + r) * C);
#pragma unroll
            for (int cc = 0; cc < 3; ++cc)
              if (c_lo + cc < c_hi) dsv[cc] = __ldg(dsr + c_lo + cc);
            cur_sb = sbi;
          }
        }
        const float mean = __ldg(a.mean_o + row0 + r), rs = __ldg(a.rs_o + row0 + r);
        const int slot = cta * a.RSF + i % a.RSF;
        uint4* dpg = reinterpret_cast<uint4*>(a.ring) + (size_t)slot * (XT / 16) + r;   // + qq TOK: chunk qq of token r
        PW(0, mbar_wait(&F.qg_full[b], (i >> 1) & 1));
        PW(1, mbar_wait(&F.in_full[b], (i >> 1) & 1));
        PW(2, mbar_wait(&F.slot_ok, i & 1));
        tc_fence_after();
        const float2 rs2 = f2(rs), nmean2 = f2(-mean);
        float2 s1 = f2(0.f), s2 = f2(0.f);
        const uint32_t sdy = su + set_of(b) + TT + XT;
#pragma unroll
        for (int cc = 0; cc < 3; ++cc) {
          const int c = c_lo + cc;
          if (c < c_hi) {
            uint32_t vq[8], vg[8];
            tmem_ld8(trow + G_::T_Q + 64 * b + c * 8, vq);
            tmem_ld8(trow + G_::T_G + 64 * b + c * 8, vg);
            const uint32_t ya = sdy + sw128(r, c);
            const uint4 dyw = lds128(ya), dsw = dsv[cc];
            const float4 u0 = ldsf4(suc + 32 * c), u1 = ldsf4(suc + 32 * c + 16);
            const float4 c0 = ldsf4(suc + 4 * C + 32 * c), c1 = ldsf4(suc + 4 * C + 32 * c + 16);
            const float2 uu[4] = {make_float2(u0.x, u0.y), make_float2(u0.z, u0.w), make_float2(u1.x, u1.y), make_float2(u1.z, u1.w)};
            const float2 cv[4] = {make_float2(c0.x, c0.y), make_float2(c0.z, c0.w), make_float2(c1.x, c1.y), make_float2(c1.z, c1.w)};
            const uint32_t yv[4] = {dyw.x, dyw.y, dyw.z, dyw.w}, sv[4] = {dsw.x, dsw.y, dsw.z, dsw.w};
            tmem_wait_ld();
            uint32_t odp[4], odg[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              const float2 aa = __fmul2_rn(up(yv[k]), up(sv[k]));
              const float2 s = sig2(rbf2(make_float2(__uint_as_float(vg[2 * k]), __uint_as_float(vg[2 * k + 1]))));
              const float2 q2 = make_float2(__uint_as_float(vq[2 * k]), __uint_as_float(vq[2 * k + 1]));
              const float2 qc = __ffma2_rn(nmean2, uu[k], q2);
              const float2 p = __ffma2_rn(rs2, qc, cv[k]);
              odp[k] = pk2(__fmul2_rn(__fmul2_rn(aa, s), rs2));
              const float2 dpr = up(odp[k]);
              s1 = __ffma2_rn(dpr, qc, s1);
              s2 = __ffma2_rn(dpr, uu[k], s2);
              odg[k] = pk2(__fmul2_rn(__fmul2_rn(aa, p), __fmul2_rn(s, __fadd2_rn(f2(1.f), make_float2(-s.x, -s.y)))));
            }
            sts128(ya, odg[0], odg[1], odg[2], odg[3]);
            dpg[c * TOK] = make_uint4(odp[0], odp[1], odp[2], odp[3]);
          }
        }
        tc_fence_before();
        fence_async_smem();
        // (S1 | S2) of token r over the 64 outputs: the three column partials summed in a fixed order (kq 2, 1, 0) through sb
        const float p1 = s1.x + s1.y, p2 = s2.x + s2.y;
        if (kq == 2) {
          sb[r] = p1; sb[TOK + r] = p2;
          named_bar_sync(2, 256);
        } else if (kq == 1) {
          named_bar_sync(2, 256);
          sb[r] += p1; sb[TOK + r] += p2;
          named_bar_sync(3, 256);
        } else {
          named_bar_sync(3, 256);
          float* sg = a.sring + (size_t)slot * 2 * TOK;
          sg[r] = sb[r] + p1; sg[TOK + r] = sb[TOK + r] + p2;
        }
        fence_proxy_async_global();                                     // this thread's dPr / S stores before the async-proxy reads
        named_bar_sync(1, 384);
        if (tid == 128) { mbar_arrive(&F.tm_empty[b]); mbar_arrive(&F.dp_ready); }
      }
    }
    if (lane == 0 && warp == 0) PROF_END(0);
    if (lane == 0 && warp == 1) PROF_END(1);
    if (lane == 0 && warp == 3) PROF_END(2);
    if (lane == 0 && warp == 4) PROF_END(3);
    if (lane == 0 && warp == 2) PROF_END(4);
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    if ((warp & 3) >= 2) {
      // dWg partial (TMEM lanes 64..127 = o): 8 warps, warp (q, cg) -> o = 32 (q - 2) + lane, columns 16 cg ..; zero dWp columns when
      // this row has no LN CTA
      const int q = warp & 3, cg = warp >> 2, o = (q - 2) * 32 + lane;
      float* dst = a.part + ((size_t)cta * C + o) * (C + H);
#pragma unroll
      for (int h8 = 0; h8 < 2; ++h8) {
        uint32_t v[8];
        tmem_ld8(tmem + ((uint32_t)(q * 32) << 16) + G_::T_DWG + cg * 16 + h8 * 8, v);
        tmem_wait_ld();
#pragma unroll
        for (int k = 0; k < 8; k += 4)
          *reinterpret_cast<float4*>(dst + cg * 16 + h8 * 8 + k) =
              n_local > 0 ? make_float4(__uint_as_float(v[k]), __uint_as_float(v[k + 1]), __uint_as_float(v[k + 2]), __uint_as_float(v[k + 3]))
                          : make_float4(0.f, 0.f, 0.f, 0.f);
      }
      if (cta >= NL)
        for (int k = 0; k < H / 4; k += 4) *reinterpret_cast<float4*>(dst + C + cg * (H / 4) + k) = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  } else {
    // ======================================================================= LN BACKWARD (set b = j & 1)
    LBars& N = B.n;
    const int l = cta - NF;
    int n_local = 0;
    while (l + NL * n_local < a.tiles) ++n_local;
    if (warp == 0) {
      if (lane == 0) {
        int prev_slot = -1;
        for (int j = 0; j < n_local; ++j) {
          const int t = l + NL * j, b = j & 1, row0 = t * TOK, fi = t / NF, slot = (t % NF) * a.RSF + fi % a.RSF;
          if (j >= 2) PW(0, mbar_wait(&N.tl_free[b], ((j >> 1) - 1) & 1));
          mbar_expect_tx(&N.tl_full[b], TT);
          tri_load<SMJ>(sm + G_::L_T + b * TT, &mtri, &N.tl_full[b], row0, a.n2, EVICT_FIRST);
          tri_load<SMJ>(sm + G_::L_T + b * TT + TT / 2, &mtri, &N.tl_full[b], row0 + 64, a.n2, EVICT_FIRST);
          PW(1, spin_until(a.flags + slot, (uint32_t)(fi / a.RSF + 1)));
          fence_proxy_async_global();
          if (j >= 2) PW(2, mbar_wait(&N.dpb_free[b], ((j >> 1) - 1) & 1));   // tile j-2's products, v / s and LN warps are done
          mbar_expect_tx(&N.dpb_full[b], XT + SBB);
          bulk_load(su + G_::L_DP + b * XT, a.ring + (size_t)slot * TOK * C, XT, &N.dpb_full[b]);
          bulk_load(su + G_::L_SB + b * SBB, a.sring + (size_t)slot * 2 * TOK, SBB, &N.dpb_full[b]);
          if (j >= 1) {                                                // tile j-1's slot has landed in shared memory: it goes back
            PW(3, mbar_wait(&N.dpb_full[b ^ 1], ((j - 1) >> 1) & 1));
            st_relaxed(a.flags + prev_slot, 0u);
          }
          prev_slot = slot;
        }
        if (n_local > 0) {
          mbar_wait(&N.dpb_full[(n_local - 1) & 1], ((n_local - 1) >> 1) & 1);
          st_relaxed(a.flags + prev_slot, 0u);
        }
      }
    } else if (warp == 1) {
      if (lane == 0) {
        const uint32_t swp = su;
        for (int j = 0; j < n_local; ++j) {
          const int b = j & 1;
          const uint32_t st = su + G_::L_T + b * TT, sdp = su + G_::L_DP + b * XT;
          PW(0, mbar_wait(&N.dpb_full[b], (j >> 1) & 1));
          PW(0, mbar_wait(&N.tl_full[b], (j >> 1) & 1));
          tc_fence_after();
          // A [64 o (+ 64 padding rows)][H ch] += dPr^T [o][128 tok] . tri [ch][tok]
#pragma unroll
          for (int k = 0; k < 8; ++k)
            umma_ss(tmem, desc_nosw(sdp + k * 256, 128, 2048), desc_k_sw128(st + (k >> 2) * (TT / 2) + (k & 3) * 32), G_::ID_A, (j | k) != 0);
          umma_commit(&N.a_done[b]);
          if (j >= 2) PW(1, mbar_wait(&N.dn_empty[b], ((j >> 1) - 1) & 1));
          tc_fence_after();
          // dnr [128 tok][H ch] = dPr [tok][64 o] . Wp' [o][ch]   (B N-major: H / 64 blocks of 64 channels, KB apart)
#pragma unroll
          for (int k = 0; k < 4; ++k)
            umma_ss(tmem + G_::T_D + H * b, desc_nosw(sdp + k * 4096, 2048, 128), desc_mn_sw128(swp + k * 2048, KB), G_::ID_DN, k != 0);
          umma_commit(&N.dn_full[b]);
          umma_commit(&N.dpb_free[b]);
        }
        if (n_local > 0) mbar_wait(&N.a_done[(n_local - 1) & 1], ((n_local - 1) >> 1) & 1);
        if (n_local > 0) mbar_wait(&N.dn_full[(n_local - 1) & 1], ((n_local - 1) >> 1) & 1);
      }
    } else if (warp == 3) {
      if (lane == 0) {
        for (int j = 0; j < n_local; ++j) {
          const int t = l + NL * j, b = j & 1, row0 = t * TOK;
          PW(0, mbar_wait(&N.dtri_ready[b], (j >> 1) & 1));
          tri_store<SMJ>(&mdtri, sm + G_::L_T + b * TT, row0, a.n2);
          tri_store<SMJ>(&mdtri, sm + G_::L_T + b * TT + TT / 2, row0 + 64, a.n2);
          bulk_commit();
          PW(1, bulk_wait_read<0>());
          mbar_arrive(&N.tl_free[b]);
        }
        bulk_wait<0>();
      }
    } else if (warp >= 4 && warp < 12) {
      // 8 warps: LN backward of the 16-token group (q, hh) in the fragment layout, one pass: dnr from TMEM, tri by ldmatrix
      const int q = warp & 3, hh = (warp - 4) >> 2, tr = lane >> 2, tok0 = q * 32 + hh * 16;
      constexpr int JH = H / 16;
      for (int j = 0; j < n_local; ++j) {
        const int t = l + NL * j, b = j & 1, row0 = t * TOK;
        const uint32_t tb = su + G_::L_T + b * TT;
        const float* sbl = reinterpret_cast<const float*>(sm + G_::L_SB + b * SBB);
        const float mA = __ldg(a.mean_o + row0 + tok0 + tr), rA = __ldg(a.rs_o + row0 + tok0 + tr);
        const float mB = __ldg(a.mean_o + row0 + tok0 + tr + 8), rB = __ldg(a.rs_o + row0 + tok0 + tr + 8);
        const float2 nmA = f2(-mA), nmB = f2(-mB);
        PW(0, mbar_wait(&N.dpb_full[b], (j >> 1) & 1));
        // dTri = dnr + (tri - mean) K1 + K0,  K1 = -rs^2 S1 / H,  K0 = -S2 / H
        const float2 k1A = f2(-rA * rA * sbl[tok0 + tr] * (1.f / H)), k0A = f2(-sbl[TOK + tok0 + tr] * (1.f / H));
        const float2 k1B = f2(-rB * rB * sbl[tok0 + tr + 8] * (1.f / H)), k0B = f2(-sbl[TOK + tok0 + tr + 8] * (1.f / H));
        PW(1, mbar_wait(&N.a_done[b], (j >> 1) & 1));            // A has read tri: it may be overwritten
        __syncwarp();
        if (lane == 0) mbar_arrive(&N.dpb_free[b]);
        PW(0, mbar_wait(&N.tl_full[b], (j >> 1) & 1));
        PW(2, mbar_wait(&N.dn_full[b], (j >> 1) & 1));
        tc_fence_after();
        const uint32_t ta = tmem + ((uint32_t)tok0 << 16) + G_::T_D + H * b;
        uint32_t tv[16];
#pragma unroll 2
        for (int jj = 0; jj < JH; ++jj) {
          if ((jj & 1) == 0) { tmem_ld16x256_x4(ta + (jj >> 1) * 32, tv); tmem_wait_ld(); }
          const uint32_t ad = frag_addr<H>(tb, tok0, jj, lane);
          uint32_t tt[4];
          ldsm_x4_trans(ad, tt[0], tt[1], tt[2], tt[3]);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float2 dn = dn_pair(jj, e, tv);
            const float2 d = __fadd2_rn(up(tt[e]), (e & 2) ? nmB : nmA);
            tt[e] = pk2(__fadd2_rn(__ffma2_rn(d, (e & 2) ? k1B : k1A, (e & 2) ? k0B : k0A), dn));
          }
          stsm_x4_trans(ad, tt[0], tt[1], tt[2], tt[3]);
        }
        tc_fence_before();
        fence_async_smem();
        __syncwarp();
        if (lane == 0) { mbar_arrive(&N.dn_empty[b]); mbar_arrive(&N.dtri_ready[b]); }
      }
    } else if (warp >= 12) {
      // 4 warps: v[o] = sum_tok dPr[tok][o] mean[tok], s[o] = sum_tok dPr[tok][o] / rs[tok] for o = 16 wv .. 16 wv + 15 (chunks
      // 2 wv, 2 wv + 1): lane = token (mod 32) while the tiles run, x[c 16 + 2 e + {v, s}]; one butterfly at the end
      const int wv = warp - 12;
      float x[32];
#pragma unroll
      for (int k = 0; k < 32; ++k) x[k] = 0.f;
      for (int j = 0; j < n_local; ++j) {
        const int b = j & 1, row0 = (l + NL * j) * TOK;
        const uint32_t base = su + G_::L_DP + b * XT + (2 * wv) * 2048;
        PW(0, mbar_wait(&N.dpb_full[b], (j >> 1) & 1));
#pragma unroll 1
        for (int m = 0; m < 4; ++m) {
          const int tok = lane + 32 * m;
          const float mv = __ldg(a.mean_o + row0 + tok), sd = rcp_ftz(__ldg(a.rs_o + row0 + tok));
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            const uint4 w = lds128(base + c * 2048 + tok * 16);
            const uint32_t wq[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              const float2 f = up(wq[k]);
              x[c * 16 + 4 * k] = fmaf(f.x, mv, x[c * 16 + 4 * k]);
              x[c * 16 + 4 * k + 1] = fmaf(f.x, sd, x[c * 16 + 4 * k + 1]);
              x[c * 16 + 4 * k + 2] = fmaf(f.y, mv, x[c * 16 + 4 * k + 2]);
              x[c * 16 + 4 * k + 3] = fmaf(f.y, sd, x[c * 16 + 4 * k + 3]);
            }
          }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&N.dpb_free[b]);
      }
      // reduce-scatter over the 32 lanes (fixed order): lane ends with x[0] = index lane: c = lane >> 4, e = (lane >> 1) & 7, w = lane & 1
#pragma unroll
      for (int st = 0, msk = 16, nh = 16; st < 5; ++st, msk >>= 1, nh >>= 1) {
        const bool hi = (lane & msk) != 0;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          if (k < nh) {
            const float send = hi ? x[k] : x[k + nh];
            const float keep = hi ? x[k + nh] : x[k];
            x[k] = keep + __shfl_xor_sync(0xffffffffu, send, msk);
          }
        }
      }
      float* vfo = reinterpret_cast<float*>(sm + G_::L_VF);
      const int o = (2 * wv + (lane >> 4)) * 8 + ((lane >> 1) & 7);
      vfo[(lane & 1) * C + o] = x[0];
    }
    if (lane == 0 && warp == 0) PROF_END(0);
    if (lane == 0 && warp == 1) PROF_END(1);
    if (lane == 0 && warp == 3) PROF_END(2);
    if (lane == 0 && warp == 4) PROF_END(3);
    if (lane == 0 && warp == 12) PROF_END(4);
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    // ---- LN tail: dWp = g_o (A - v) + s b_o -> part row l; E = Wp o (A - v) -> smem; dg_o = colsum E, db_o = Wp^T s
    const float* vf = reinterpret_cast<const float*>(sm + G_::L_VF);   // v [C], s [C]
    float* E = reinterpret_cast<float*>(sm);
    if ((warp & 3) < 2) {
      // TMEM lanes 0..63 = o: 8 warps, warp (q, cg) -> o = 32 q + lane, channels cg H / 4 ..
      const int q = warp & 3, cg = warp >> 2, o = q * 32 + lane;
      const float vo = vf[o], so = vf[C + o];
      float* dst = a.part + ((size_t)l * C + o) * (C + H) + C;
#pragma unroll 1
      for (int m = 0; m < H / 32; ++m) {
        const int col = cg * (H / 4) + 8 * m;
        uint32_t v[8];
        tmem_ld8(tmem + ((uint32_t)(q * 32) << 16) + col, v);
        const uint4 w = __ldg(reinterpret_cast<const uint4*>(a.wp + (size_t)o * H + col));
        const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
        const float4 ga = __ldg(reinterpret_cast<const float4*>(a.go + col)), gb = __ldg(reinterpret_cast<const float4*>(a.go + col + 4));
        const float4 ba = __ldg(reinterpret_cast<const float4*>(a.bo + col)), bb = __ldg(reinterpret_cast<const float4*>(a.bo + col + 4));
        const float gv[8] = {ga.x, ga.y, ga.z, ga.w, gb.x, gb.y, gb.z, gb.w}, bv[8] = {ba.x, ba.y, ba.z, ba.w, bb.x, bb.y, bb.z, bb.w};
        float outv[8];
        tmem_wait_ld();
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const float d = n_local > 0 ? __uint_as_float(v[k]) - vo : 0.f;
          const float wpv = (k & 1) ? up(wv[k >> 1]).y : up(wv[k >> 1]).x;
          outv[k] = fmaf(gv[k], d, bv[k] * so);
          E[(col + k) * G_::E_LD + o] = wpv * d;
        }
        *reinterpret_cast<float4*>(dst + col) = make_float4(outv[0], outv[1], outv[2], outv[3]);
        *reinterpret_cast<float4*>(dst + col + 4) = make_float4(outv[4], outv[5], outv[6], outv[7]);
      }
      if (l >= NF) {
        float* z = a.part + ((size_t)l * C + o) * (C + H) + cg * 16;
        for (int k = 0; k < 16; k += 4) *reinterpret_cast<float4*>(z + k) = make_float4(0.f, 0.f, 0.f, 0.f);
      }
    }
    __syncthreads();
    if (tid < H) {
      float sgo = 0.f, sbo = 0.f;
#pragma unroll 8
      for (int o = 0; o < C; ++o) {
        sgo += E[tid * G_::E_LD + o];
        sbo = fmaf(__bfloat162float(a.wp[(size_t)o * H + tid]), vf[C + o], sbo);
      }
      a.lnpart[(size_t)l * 2 * H + tid] = sgo;
      a.lnpart[(size_t)l * 2 * H + H + tid] = sbo;
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) tmem_dealloc(tmem, 512);
}

}  // namespace b1s

// ================================================================================================ b1u: unified B1, H = 64
// (2026-11-17, round 3 training pass) One CTA does the whole output-side backward of its tiles -- the front epilogue (P / dP / dG / S1 /
// S2) AND the LN_out backward / v / s -- so dPr and (S1 | S2) stay in shared memory: no L2 ring, no flags, no cross-CTA wait and so no
// cooperative launch. At H = 64 every operand fits one CTA: TMEM dWg 64 | A 64 | G 2 x 64 | Q 2 x 64 | dnr 2 x 64 = 512 columns;
// shared memory Wp' | Wg | dPr x 2 | NT tri slots | NX (xn | dy -> dG) slots | (S1 | S2) x 2. Same arithmetic per element as b1s (the
// (S1 | S2) column partials are two here, added in a fixed order). Persistent grid G = min(tiles, SMs), tiles t = cta + G i; partial
// row cta = (dWg | dWp) of this CTA, lnpart row cta = (dg_o | db_o): grad_reduce sums G rows of each (deterministic).
// Warps: 0 loads, 1 MMA, 2 dG store, 3 dTri store, 4-11 front epilogue (lane quarter q, column half kq), 12-15 LN backward (two
// 16-token groups each) then v / s of the tile.
namespace b1u {
using namespace b1s;
constexpr int H = 64, TT = H * TOK * 2;
// tri slots (freed by the dTri store, the last use); (xn | dy) slots (freed by the dWg MMAs and the dG store, right after the front
// epilogue)
constexpr int NT = 4, NX = 3, XS = 2 * XT;
constexpr int S_WP = 0, S_WG = KB, S_DP = 2 * KB, S_T = S_DP + 2 * XT, S_X = S_T + NT * TT, S_SB = S_X + NX * XS, S_UC = S_SB + 2 * SBB,
              S_VF = S_UC + 2 * C * 4, O_BAR = S_VF + 2 * C * 4, SMEM = O_BAR + 512;
static_assert(SMEM <= 232448, "smem");
static_assert(S_T % 1024 == 0 && S_X % 1024 == 0 && TT % 1024 == 0 && S_DP % 1024 == 0, "SW128 bases");
static_assert(2 * H * (C + 1) * 4 <= S_T, "LN tail scratch (E, F) below the sets");
constexpr uint32_t T_DWG = 0, T_A = 64, T_G = 128, T_Q = 256, T_DN = 384;
struct UBars {
  uint64_t wp_full, wg_full, t_full[NT], t_free[NT], x_full[NX], x_free[NX], qg_full[2], tm_empty[2], dp_ready[4], a_done[2],
      dn_full[2], dn_empty[2], dtri_ready[4], dpb_free[2], last;
  uint32_t tmem;
};
static_assert(sizeof(UBars) <= 512, "bars");
// dp_ready / dtri_ready are 4 deep (tile i & 3): the dG / dTri store warps may lag the epilogue / LN warps by up to NX / NT - 1 tiles
// (their slot refills bound it), and a barrier may not complete twice before a waiter of the earlier phase looks at it.
static_assert(NT <= 4 && NX <= 4 && NX >= 2 && NT >= 2, "ring depths");
#ifndef B1U_PDL
#define B1U_PDL 1
#endif
#ifndef B1U_ML
#define B1U_ML 1   // b1u / b1v loaders: the tile's loads issued by four lanes (2026-11-17 part 5: D64 hidden 128 L128 0.0687 -> 0.0667, D64 L256 0.0954 -> 0.0934 ms; the rest level)
#endif
#ifndef B1U_TRIG
#define B1U_TRIG 1   // b1u / b1v: griddepcontrol.launch_dependents at entry
#endif

#if defined(PT_TL) && PT_TL
// phase trace (-DPT_TL=1): 0 entry (globaltimer), 1 set-up done, 2 Wp' formed + griddepcontrol.wait returned, 3 MMA has tile 0's
// operands, 4 epilogue has Q / G of tile 0, 5 tile 0 dPr handed over, 6 LN warps have dnr of tile 0, 7 last dTri handed over,
// 8 last dTri store read, 9 last dG store read, 10 v / s done, 11 tail start, 12 partials written, 14 before TMEM release, 15 exit
__device__ unsigned long long g_pt1[160][16];
#define P1(s) if constexpr (!FU) PTM(g_pt1, s)
#define P1G(s) if constexpr (!FU) PT_G(g_pt1, s)
#else
#define P1(s) ((void)0)
#define P1G(s) ((void)0)
#endif
// The kernel's body as a device function (2026-11-17 part 5): b1u_kernel calls it with (blockIdx.x, gridDim.x); the fused small-L
// backward (b7m.cu, b7f) calls it with FU = true as its first phase, with a virtual CTA id and the TMEM it allocated itself (no
// dependent trigger, no TMEM allocation here).
template <int SMJ, bool FU>
DEV void b1u_body(const CUtensorMap& mxn, const CUtensorMap& mtri, const CUtensorMap& mwp, const CUtensorMap& mwg, const CUtensorMap& mdg,
                  const CUtensorMap& mdtri, const CUtensorMap& mdy, const Args& a, uint8_t* sm, const int cta, const int G,
                  const uint32_t tmem_fu) {
  using G_ = Geo<H>;
  const uint32_t su = smem_u32(sm);
  UBars& B = *reinterpret_cast<UBars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int n_local = a.tiles > cta ? (a.tiles - cta + G - 1) / G : 0;
  PT_BEGIN
  if (tid == 0) P1G(0);
  if (!FU && B1U_TRIG) pdl_trigger();   // the dependent (cgb / the contraction backward) is scheduled now: its CTAs start on SMs this grid leaves free
  if (tid == 0 && (su & 1023u)) asm volatile("trap;");
  if (tid == 0) {
    mbar_init(&B.wp_full, 1); mbar_init(&B.wg_full, 1); mbar_init(&B.last, 1);
    for (int s = 0; s < NT; ++s) { mbar_init(&B.t_full[s], 1); mbar_init(&B.t_free[s], 1); }
    for (int s = 0; s < NX; ++s) { mbar_init(&B.x_full[s], 1); mbar_init(&B.x_free[s], 2); }
    for (int s = 0; s < 4; ++s) { mbar_init(&B.dp_ready[s], 1); mbar_init(&B.dtri_ready[s], 4); }
    for (int b = 0; b < 2; ++b) {
      mbar_init(&B.qg_full[b], 1); mbar_init(&B.tm_empty[b], 1); mbar_init(&B.a_done[b], 1);
      mbar_init(&B.dn_full[b], 1); mbar_init(&B.dn_empty[b], 4); mbar_init(&B.dpb_free[b], 1 + 4);
    }
    fence_mbar_init();
    mbar_expect_tx(&B.wp_full, KB);
    tma_load_2d(sm + S_WP, &mwp, &B.wp_full, 0, 0, EVICT_LAST);
    mbar_expect_tx(&B.wg_full, KB);
    tma_load_2d(sm + S_WG, &mwg, &B.wg_full, 0, 0, EVICT_LAST);
  }
  if (!FU && warp == 2) { tmem_alloc(&B.tmem, 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = FU ? tmem_fu : B.tmem;
  if (tid == 0) P1(1);
  // ---- Wp' = wpf in place (8 threads per row o, 8 channels each); u = rowsum(Wp'), c = Wp b_o (as b1s)
  mbar_wait(&B.wp_full, 0);
  {
    const int o = tid >> 3, pq = tid & 7, ch0 = pq * 8;
    const uint32_t ad = su + S_WP + sw128(o, ch0 >> 3);
    const uint4 w = lds128(ad);
    const uint4 wq = __ldg(reinterpret_cast<const uint4*>(a.wpf + (size_t)o * H + ch0));
    const float4 ba = __ldg(reinterpret_cast<const float4*>(a.bo + ch0)), bb = __ldg(reinterpret_cast<const float4*>(a.bo + ch0 + 4));
    const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
    const uint32_t qv[4] = {wq.x, wq.y, wq.z, wq.w};
    const float2 bv[4] = {make_float2(ba.x, ba.y), make_float2(ba.z, ba.w), make_float2(bb.x, bb.y), make_float2(bb.z, bb.w)};
    float u = 0.f, cb = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float2 wf = up(wv[k]);
      const float2 sc = up(qv[k]);
      u += sc.x + sc.y;
      cb = fmaf(wf.x, bv[k].x, cb);
      cb = fmaf(wf.y, bv[k].y, cb);
    }
    sts128(ad, wq.x, wq.y, wq.z, wq.w);
    {  // Wg / 2 in place (exact: a power of two), so G' = G / 2 and bf16(G') = bf16(G) / 2: the gate's tanh(g / 2) needs no multiply
      mbar_wait(&B.wg_full, 0);
      const uint32_t wa = su + S_WG + tid * 16;
      const uint4 g = lds128(wa);
      const uint32_t gv[4] = {g.x, g.y, g.z, g.w};
      uint32_t hv[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) hv[k] = pk2(__fmul2_rn(up(gv[k]), f2(0.5f)));
      sts128(wa, hv[0], hv[1], hv[2], hv[3]);
    }
#pragma unroll
    for (int x = 1; x <= 4; x <<= 1) { u += __shfl_xor_sync(0xffffffffu, u, x); cb += __shfl_xor_sync(0xffffffffu, cb, x); }
    if (pq == 0) {
      reinterpret_cast<float*>(sm + S_UC)[o] = u;
      reinterpret_cast<float*>(sm + S_UC)[C + o] = cb;
    }
  }
  fence_async_smem();
  __syncthreads();
  if (B1U_PDL) pdl_wait();   // xn, t, the LN_out statistics are the previous kernel's (k3g) output
  if (tid == 0) P1(2);
  const int LC = a.L / TOK;
  auto t_addr = [&](int i) { return (uint32_t)(S_T + (i % NT) * TT); };   // tri [64 ch][128 tok]
  auto x_addr = [&](int i) { return (uint32_t)(S_X + (i % NX) * XS); };   // xn, then dy -> dG
  PROF_BEGIN

  if (warp == 0 && B1U_ML) {
    // (2026-11-17 part 5) the tile's four loads (tri halves, xn, dy) issued by four lanes (one thread issues ~one TMA per 600 clk)
    for (int i = 0; i < n_local; ++i) {
      const int row0 = (cta + G * i) * TOK, ts = i % NT, xs = i % NX;
      if (lane == 0) {
        if (i >= NT) PW(0, mbar_wait(&B.t_free[ts], ((i / NT) - 1) & 1));
        mbar_expect_tx(&B.t_full[ts], TT);
        if (i >= NX) PW(1, mbar_wait(&B.x_free[xs], ((i / NX) - 1) & 1));
        mbar_expect_tx(&B.x_full[xs], XS);
      }
      __syncwarp();
      uint8_t* pt = sm + t_addr(i);
      uint8_t* px = sm + x_addr(i);
      if (SMJ != 0 && B1U_T1_ON) { if (lane == 0) tri_load1(pt, &mtri, &B.t_full[ts], row0, a.n2, EVICT_FIRST); }
      else if (lane < 2) tri_load<SMJ>(pt + lane * (TT / 2), &mtri, &B.t_full[ts], row0 + 64 * lane, a.n2, EVICT_FIRST);
      if (lane == 2) tma_load_2d(px, &mxn, &B.x_full[xs], 0, row0, EVICT_NORMAL);
      else if (lane == 3) tma_load_2d(px + XT, &mdy, &B.x_full[xs], 0, row0, EVICT_NORMAL);
      __syncwarp();
    }
  } else if (warp == 0) {
    if (lane == 0) {
      for (int i = 0; i < n_local; ++i) {
        const int row0 = (cta + G * i) * TOK, ts = i % NT, xs = i % NX;
        if (i >= NT) PW(0, mbar_wait(&B.t_free[ts], ((i / NT) - 1) & 1));   // tile i - NT: dTri stored
        uint8_t* pt = sm + t_addr(i);
        mbar_expect_tx(&B.t_full[ts], TT);
        if (SMJ != 0 && B1U_T1_ON) {
          tri_load1(pt, &mtri, &B.t_full[ts], row0, a.n2, EVICT_FIRST);
        } else {
          tri_load<SMJ>(pt, &mtri, &B.t_full[ts], row0, a.n2, EVICT_FIRST);
          tri_load<SMJ>(pt + TT / 2, &mtri, &B.t_full[ts], row0 + 64, a.n2, EVICT_FIRST);
        }
        if (i >= NX) PW(1, mbar_wait(&B.x_free[xs], ((i / NX) - 1) & 1));   // tile i - NX: dWg read xn / dG, dG stored
        uint8_t* px = sm + x_addr(i);
        mbar_expect_tx(&B.x_full[xs], XS);
        tma_load_2d(px, &mxn, &B.x_full[xs], 0, row0, EVICT_NORMAL);
        tma_load_2d(px + XT, &mdy, &B.x_full[xs], 0, row0, EVICT_NORMAL);
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      mbar_wait(&B.wg_full, 0);
      const uint32_t swp = su + S_WP, swg = su + S_WG;
      auto mma_qg = [&](int i) {
        const int b = i & 1;
        const uint32_t pt = su + t_addr(i), px = su + x_addr(i);
        PW(0, mbar_wait(&B.t_full[i % NT], (i / NT) & 1));
        PW(0, mbar_wait(&B.x_full[i % NX], (i / NX) & 1));
        if (i == 0) P1(3);
        if (i >= 2) PW(1, mbar_wait(&B.tm_empty[b], ((i >> 1) - 1) & 1));
        tc_fence_after();
#pragma unroll
        for (int k = 0; k < H / 16; ++k)
          umma_ss(tmem + T_Q + 64 * b, desc_mn_sw128(pt + k * 2048, TT / 2), desc_k_sw128(swp + k * 32), G_::ID_Q, k != 0);
#pragma unroll
        for (int k = 0; k < 4; ++k)
          umma_ss(tmem + T_G + 64 * b, desc_k_sw128(px + k * 32), desc_k_sw128(swg + k * 32), G_::ID_GQ, k != 0);
        umma_commit(&B.qg_full[b]);
      };
      if (n_local > 0) mma_qg(0);
      for (int i = 0; i < n_local; ++i) {
        if (i + 1 < n_local) mma_qg(i + 1);
        const int b = i & 1;
        const uint32_t pt = su + t_addr(i), px = su + x_addr(i), sdp = su + S_DP + b * XT;
        PW(2, mbar_wait(&B.dp_ready[i & 3], (i >> 2) & 1));
        tc_fence_after();
        // dWg [(64 padding rows: xn +) 64 o][64 i] += dG^T xn (lanes 64..127)
#pragma unroll
        for (int k = 0; k < 8; ++k)
          umma_ss(tmem + T_DWG, desc_mn_sw128(px + k * 2048, XT), desc_mn_sw128(px + k * 2048, XT), G_::ID_DWG, (i | k) != 0);
        umma_commit(&B.x_free[i % NX]);
        // A [64 o (+ 64 padding rows: the next 16 KB of shared memory)][H ch] += dPr^T tri
#pragma unroll
        for (int k = 0; k < 8; ++k)
          umma_ss(tmem + T_A, desc_nosw(sdp + k * 256, 128, 2048), desc_k_sw128(pt + (k >> 2) * (TT / 2) + (k & 3) * 32), G_::ID_A,
                  (i | k) != 0);
        umma_commit(&B.a_done[b]);
        if (i >= 2) PW(3, mbar_wait(&B.dn_empty[b], ((i >> 1) - 1) & 1));
        tc_fence_after();
        // dnr [128 tok][H ch] = dPr [tok][64 o] . Wp' [o][ch]
#pragma unroll
        for (int k = 0; k < 4; ++k)
          umma_ss(tmem + T_DN + 64 * b, desc_nosw(sdp + k * 4096, 2048, 128), desc_mn_sw128(swp + k * 2048, KB), G_::ID_DN, k != 0);
        umma_commit(&B.dn_full[b]);
        umma_commit(&B.dpb_free[b]);
      }
      if (n_local > 0) { umma_commit(&B.last); mbar_wait(&B.last, 0); }
    }
  } else if (warp == 2) {
    if (lane == 0) {
      for (int i = 0; i < n_local; ++i) {
        const int row0 = (cta + G * i) * TOK;
        PW(0, mbar_wait(&B.dp_ready[i & 3], (i >> 2) & 1));
        tma_store_2d(&mdg, sm + x_addr(i) + XT, 0, row0);
        bulk_commit();
        PW(1, bulk_wait_read<0>());
        mbar_arrive(&B.x_free[i % NX]);
        if (i == n_local - 1) P1(9);
      }
      BULK_END();
    }
  } else if (warp == 3) {
    if (lane == 0) {
      for (int i = 0; i < n_local; ++i) {
        const int row0 = (cta + G * i) * TOK;
        PW(0, mbar_wait(&B.dtri_ready[i & 3], (i >> 2) & 1));
        tri_store<SMJ>(&mdtri, sm + t_addr(i), row0, a.n2);
        tri_store<SMJ>(&mdtri, sm + t_addr(i) + TT / 2, row0 + 64, a.n2);
        bulk_commit();
        PW(1, bulk_wait_read<0>());
        mbar_arrive(&B.t_free[i % NT]);
        if (i == n_local - 1) P1(8);
      }
      BULK_END();
    }
  } else if (warp < 12) {
    // 8 front-epilogue warps: lane quarter q (tokens 32 q ..), 8-column chunks 4 kq .. 4 kq + 3
    const int q = warp & 3, kq = (warp - 4) >> 2, r = q * 32 + lane, c_lo = 4 * kq;
    const uint32_t trow = tmem + ((uint32_t)(q * 32) << 16);
    const uint32_t suc = su + S_UC;
    float* sbs = reinterpret_cast<float*>(sm + S_SB);
    // the per-token operands (row-dropout scale, LN_out mean / rstd) of the NEXT tile are loaded while this one is worked on
    uint4 dsn[4];
    float mean_n = 0.f, rs_n = 0.f;
    auto ld_tok = [&](int i) {
      const int t = cta + G * i, sbi = t / (a.L * LC), jc = t % LC;
      const uint4* dsr = reinterpret_cast<const uint4*>(a.ds + ((size_t)sbi * a.L + jc * TOK + r) * C);
#pragma unroll
      for (int cc = 0; cc < 4; ++cc) dsn[cc] = __ldg(dsr + c_lo + cc);
      mean_n = __ldg(a.mean_o + t * TOK + r); rs_n = __ldg(a.rs_o + t * TOK + r);
    };
    if (n_local > 0) ld_tok(0);
    for (int i = 0; i < n_local; ++i) {
      const int b = i & 1;
      uint4 dsv[4];
#pragma unroll
      for (int cc = 0; cc < 4; ++cc) dsv[cc] = dsn[cc];
      const float mean = mean_n, rs = rs_n;
      if (i + 1 < n_local) ld_tok(i + 1);
      const uint32_t sdp = su + S_DP + b * XT;
      PW(0, mbar_wait(&B.qg_full[b], (i >> 1) & 1));
      if (tid == 128 && i == 0) P1(4);
      PW(1, mbar_wait(&B.x_full[i % NX], (i / NX) & 1));
      if (i >= 2) PW(2, mbar_wait(&B.dpb_free[b], ((i >> 1) - 1) & 1));   // tile i-2: A / dnr read dPr, LN / v / s read (S1 | S2) and dPr
      tc_fence_after();
      const float2 rs2 = f2(rs), nmean2 = f2(-mean);
      float2 s1 = f2(0.f), s2 = f2(0.f);
      const uint32_t sdy = su + x_addr(i) + XT;
      {
        // the four chunks' TMEM loads (Q, G) and dy words issued together, one wait (measured neutral against one chunk per wait: b1u is
        // bound by its memory pipeline, not this arithmetic -- removing the epilogue and LN math altogether took 13 of ~48 us at L512)
        constexpr int c0 = 0;
        uint32_t vq[4][8], vg[4][8];
        uint4 dyw[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int c = c_lo + c0 + e;
          tmem_ld8(trow + T_Q + 64 * b + c * 8, vq[e]);
          tmem_ld8(trow + T_G + 64 * b + c * 8, vg[e]);
          dyw[e] = lds128(sdy + sw128(r, c));
        }
        tmem_wait_ld();
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int c = c_lo + c0 + e;
          const uint32_t ya = sdy + sw128(r, c);
          const uint4 dsw = dsv[c0 + e];
          const float4 u0 = ldsf4(suc + 32 * c), u1 = ldsf4(suc + 32 * c + 16);
          const float4 cq0 = ldsf4(suc + 4 * C + 32 * c), cq1 = ldsf4(suc + 4 * C + 32 * c + 16);
          const float2 uu[4] = {make_float2(u0.x, u0.y), make_float2(u0.z, u0.w), make_float2(u1.x, u1.y), make_float2(u1.z, u1.w)};
          const float2 cv[4] = {make_float2(cq0.x, cq0.y), make_float2(cq0.z, cq0.w), make_float2(cq1.x, cq1.y), make_float2(cq1.z, cq1.w)};
          const uint32_t yv[4] = {dyw[e].x, dyw[e].y, dyw[e].z, dyw[e].w}, sv[4] = {dsw.x, dsw.y, dsw.z, dsw.w};
          uint32_t odp[4], odg[4];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const float2 aa = __fmul2_rn(up(yv[k]), up(sv[k]));
            const float2 sg = sigh2(rbf2(make_float2(__uint_as_float(vg[e][2 * k]), __uint_as_float(vg[e][2 * k + 1]))));
            const float2 q2 = make_float2(__uint_as_float(vq[e][2 * k]), __uint_as_float(vq[e][2 * k + 1]));
            const float2 qc = __ffma2_rn(nmean2, uu[k], q2);
            const float2 pp = __ffma2_rn(rs2, qc, cv[k]);
            odp[k] = pk2(__fmul2_rn(__fmul2_rn(aa, sg), rs2));
            const float2 dpr = up(odp[k]);
            s1 = __ffma2_rn(dpr, qc, s1);
            s2 = __ffma2_rn(dpr, uu[k], s2);
            odg[k] = pk2(__fmul2_rn(__fmul2_rn(aa, pp), __fmul2_rn(sg, __fadd2_rn(f2(1.f), make_float2(-sg.x, -sg.y)))));
          }
          sts128(ya, odg[0], odg[1], odg[2], odg[3]);
          sts128(sdp + c * 2048 + r * 16, odp[0], odp[1], odp[2], odp[3]);   // [8 chunks][128 tok][8 o], the ring's layout
        }
      }
      tc_fence_before();
      // (S1 | S2) of token r: the two column halves added in a fixed order (kq 1, then kq 0)
      const float p1 = s1.x + s1.y, p2 = s2.x + s2.y;
      float* sb = sbs + b * 2 * TOK;
      if (kq == 1) {
        sb[r] = p1; sb[TOK + r] = p2;
        PW(4, named_bar_sync(2, 256));
      } else {
        PW(4, named_bar_sync(2, 256));
        sb[r] += p1; sb[TOK + r] += p2;
      }
      fence_async_smem();
      PW(5, named_bar_sync(1, 256));
      if (tid == 128) { mbar_arrive(&B.tm_empty[b]); mbar_arrive(&B.dp_ready[i & 3]); if (i == 0) P1(5); }
    }
  } else {
    // 4 LN warps: the LN_out backward of tokens 32 q .. 32 q + 31 (two 16-token fragment groups), then v / s of the tile's
    // 8-column chunks 2 wv, 2 wv + 1 (lane = token mod 32), accumulated over the tiles
    const int q = warp & 3, wv = warp - 12, tr = lane >> 2;
    const float* sbs = reinterpret_cast<const float*>(sm + S_SB);
    float x[32];
#pragma unroll
    for (int k = 0; k < 32; ++k) x[k] = 0.f;
    for (int i = 0; i < n_local; ++i) {
      const int row0 = (cta + G * i) * TOK, b = i & 1;
      const uint32_t tb = su + t_addr(i), sdp = su + S_DP + b * XT;
      float mv[2][2], rv[2][2];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int tok = row0 + q * 32 + hh * 16 + tr;
        mv[hh][0] = __ldg(a.mean_o + tok); rv[hh][0] = __ldg(a.rs_o + tok);
        mv[hh][1] = __ldg(a.mean_o + tok + 8); rv[hh][1] = __ldg(a.rs_o + tok + 8);
      }
      float vm[4], vs[4];
#pragma unroll
      for (int m = 0; m < 4; ++m) { vm[m] = __ldg(a.mean_o + row0 + lane + 32 * m); vs[m] = __ldg(a.rs_o + row0 + lane + 32 * m); }
      PW(0, mbar_wait(&B.dp_ready[i & 3], (i >> 2) & 1));
      PW(1, mbar_wait(&B.a_done[b], (i >> 1) & 1));     // A has read tri: it may be overwritten
      PW(2, mbar_wait(&B.dn_full[b], (i >> 1) & 1));
      if (tid == 384 && i == 0) P1(6);
      tc_fence_after();
      const float* sbl = sbs + b * 2 * TOK;
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int tok0 = q * 32 + hh * 16;
        const float rA = rv[hh][0], rB = rv[hh][1];
        const float2 nmA = f2(-mv[hh][0]), nmB = f2(-mv[hh][1]);
        const float2 k1A = f2(-rA * rA * sbl[tok0 + tr] * (1.f / H)), k0A = f2(-sbl[TOK + tok0 + tr] * (1.f / H));
        const float2 k1B = f2(-rB * rB * sbl[tok0 + tr + 8] * (1.f / H)), k0B = f2(-sbl[TOK + tok0 + tr + 8] * (1.f / H));
        const uint32_t ta = tmem + ((uint32_t)tok0 << 16) + T_DN + 64 * b;
        // the group's two dnr TMEM blocks and four tri fragments loaded before one wait
        uint32_t tv2[2][16], tt4[4][4];
        tmem_ld16x256_x4(ta, tv2[0]);
        tmem_ld16x256_x4(ta + 32, tv2[1]);
#pragma unroll
        for (int jj = 0; jj < H / 16; ++jj)
          ldsm_x4_trans(frag_addr<H>(tb, tok0, jj, lane), tt4[jj][0], tt4[jj][1], tt4[jj][2], tt4[jj][3]);
        tmem_wait_ld();
#pragma unroll
        for (int jj = 0; jj < H / 16; ++jj) {
          uint32_t* tt = tt4[jj];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float2 dn = dn_pair(jj, e, tv2[jj >> 1]);
            const float2 d = __fadd2_rn(up(tt[e]), (e & 2) ? nmB : nmA);
            tt[e] = pk2(__fadd2_rn(__ffma2_rn(d, (e & 2) ? k1B : k1A, (e & 2) ? k0B : k0A), dn));
          }
          stsm_x4_trans(frag_addr<H>(tb, tok0, jj, lane), tt[0], tt[1], tt[2], tt[3]);
        }
      }
      tc_fence_before();
      fence_async_smem();
      __syncwarp();
      if (lane == 0) { mbar_arrive(&B.dn_empty[b]); mbar_arrive(&B.dtri_ready[i & 3]); }
      if (tid == 384 && i == n_local - 1) P1(7);
      const uint32_t base = sdp + (2 * wv) * 2048;
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const int tok = lane + 32 * m;
        const float mvv = vm[m], sd = rcp_ftz(vs[m]);
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          const uint4 w = lds128(base + c * 2048 + tok * 16);
          const uint32_t wq[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const float2 f = up(wq[k]);
            x[c * 16 + 4 * k] = fmaf(f.x, mvv, x[c * 16 + 4 * k]);
            x[c * 16 + 4 * k + 1] = fmaf(f.x, sd, x[c * 16 + 4 * k + 1]);
            x[c * 16 + 4 * k + 2] = fmaf(f.y, mvv, x[c * 16 + 4 * k + 2]);
            x[c * 16 + 4 * k + 3] = fmaf(f.y, sd, x[c * 16 + 4 * k + 3]);
          }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.dpb_free[b]);
    }
#pragma unroll
    for (int st = 0, msk = 16, nh = 16; st < 5; ++st, msk >>= 1, nh >>= 1) {
      const bool hi = (lane & msk) != 0;
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        if (k < nh) {
          const float send = hi ? x[k] : x[k + nh];
          const float keep = hi ? x[k + nh] : x[k];
          x[k] = keep + __shfl_xor_sync(0xffffffffu, send, msk);
        }
      }
    }
    float* vfo = reinterpret_cast<float*>(sm + S_VF);
    const int o = (2 * wv + (lane >> 4)) * 8 + ((lane >> 1) & 7);
    vfo[(lane & 1) * C + o] = x[0];
    if (tid == 384) P1(10);
  }
  if (lane == 0 && warp <= 4) PROF_END(warp);   // probes (-DPROF): 0 loader, 1 MMA, 2 dG store, 3 dTri store, 4 epilogue, 5 LN
  if (lane == 0 && warp == 12) PROF_END(5);
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  if (tid == 0) P1(11);
  // ---- tail, every global store coalesced (2026-11-17 part 3; was one row per lane and 64-long column loops on 64 threads):
  //   1. dWg (TMEM lanes 64..127 = o) -> smem Gs [o][i]; D = A - v (TMEM lanes 0..63 = o) -> smem Ds [o][ch]
  //   2. thread (column, 8-output group): part row cta [o][col] = dWg, [o][C + col] = g_o D + b_o s; dg_o = sum_o Wp D,
  //      db_o = sum_o Wp s per group in o order, the eight groups added in order
#ifndef B1U_OLD_TAIL
  const float* vf = reinterpret_cast<const float*>(sm + S_VF);   // v [C], s [C]
  constexpr int T_LD = 68;
  float* Gs = reinterpret_cast<float*>(sm);
  float* Ds = Gs + C * T_LD;
  float* Q8 = Ds + C * T_LD;                                       // [2][8][H]
  static_assert((2 * C * T_LD + 16 * H) * 4 <= S_SB, "tail scratch");
  const int tcol = tid & 63, trg = tid >> 6;
  float wpr[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) wpr[k] = __bfloat162float(a.wp[(size_t)(trg * 8 + k) * H + tcol]);
  {
    const int q = warp & 3, cg = warp >> 2;
    const bool g = q >= 2;
    const int o = g ? (q - 2) * 32 + lane : q * 32 + lane;
    const float vo = g ? 0.f : vf[o];
    float* row = (g ? Gs : Ds) + o * T_LD + cg * 16;
#pragma unroll
    for (int h8 = 0; h8 < 2; ++h8) {
      uint32_t v[8];
      tmem_ld8(tmem + ((uint32_t)(q * 32) << 16) + (g ? T_DWG : T_A) + cg * 16 + h8 * 8, v);
      tmem_wait_ld();
#pragma unroll
      for (int k = 0; k < 8; k += 4)
        *reinterpret_cast<float4*>(row + h8 * 8 + k) =
            n_local > 0 ? make_float4(__uint_as_float(v[k]) - vo, __uint_as_float(v[k + 1]) - vo, __uint_as_float(v[k + 2]) - vo,
                                      __uint_as_float(v[k + 3]) - vo)
                        : make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }
  __syncthreads();
  {
    const float gch = __ldg(a.go + tcol), bch = __ldg(a.bo + tcol);
    float sg = 0.f, sbo = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int o = trg * 8 + k;
      float* dst = a.part + ((size_t)cta * C + o) * (C + H);
      const float d = Ds[o * T_LD + tcol], so = vf[C + o];
      dst[tcol] = Gs[o * T_LD + tcol];
      dst[C + tcol] = fmaf(gch, d, bch * so);
      sg = fmaf(wpr[k], d, sg);
      sbo = fmaf(wpr[k], so, sbo);
    }
    Q8[trg * H + tcol] = sg;
    Q8[8 * H + trg * H + tcol] = sbo;
  }
  __syncthreads();
  if (tid < 2 * H) {
    const float* s8 = Q8 + (tid < H ? 0 : 8 * H);
    const int ch = tid & (H - 1);
    float acc = s8[ch];
#pragma unroll
    for (int k = 1; k < 8; ++k) acc += s8[k * H + ch];
    a.lnpart[(size_t)cta * 2 * H + tid] = acc;
  }
  if (tid == 0) P1(12);
#else
  const float* vf = reinterpret_cast<const float*>(sm + S_VF);   // v [C], s [C]
  float* E = reinterpret_cast<float*>(sm);
  float* F = E + H * G_::E_LD;   // (2026-11-17: db_o = Wp^T s from shared memory -- the column loop over Wp in global memory was
                                 // ~3 us of b1u's tail at one tile per CTA)
  if ((warp & 3) >= 2) {
    // dWg partial (TMEM lanes 64..127 = o): 8 warps, warp (q, cg) -> o = 32 (q - 2) + lane, columns 16 cg ..
    const int q = warp & 3, cg = warp >> 2, o = (q - 2) * 32 + lane;
    float* dst = a.part + ((size_t)cta * C + o) * (C + H);
#pragma unroll
    for (int h8 = 0; h8 < 2; ++h8) {
      uint32_t v[8];
      tmem_ld8(tmem + ((uint32_t)(q * 32) << 16) + T_DWG + cg * 16 + h8 * 8, v);
      tmem_wait_ld();
#pragma unroll
      for (int k = 0; k < 8; k += 4)
        *reinterpret_cast<float4*>(dst + cg * 16 + h8 * 8 + k) =
            n_local > 0 ? make_float4(__uint_as_float(v[k]), __uint_as_float(v[k + 1]), __uint_as_float(v[k + 2]), __uint_as_float(v[k + 3]))
                        : make_float4(0.f, 0.f, 0.f, 0.f);
    }
  } else {
    // LN tail: dWp = g_o (A - v) + s b_o -> columns C.. of part row cta; E = Wp o (A - v) and F = Wp o s -> smem (TMEM lanes 0..63 = o)
    const int q = warp & 3, cg = warp >> 2, o = q * 32 + lane;
    const float vo = vf[o], so = vf[C + o];
    float* dst = a.part + ((size_t)cta * C + o) * (C + H) + C;
#pragma unroll 1
    for (int m = 0; m < H / 32; ++m) {
      const int col = cg * (H / 4) + 8 * m;
      uint32_t v[8];
      tmem_ld8(tmem + ((uint32_t)(q * 32) << 16) + T_A + col, v);
      const uint4 w = __ldg(reinterpret_cast<const uint4*>(a.wp + (size_t)o * H + col));
      const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
      const float4 ga = __ldg(reinterpret_cast<const float4*>(a.go + col)), gb = __ldg(reinterpret_cast<const float4*>(a.go + col + 4));
      const float4 ba = __ldg(reinterpret_cast<const float4*>(a.bo + col)), bb = __ldg(reinterpret_cast<const float4*>(a.bo + col + 4));
      const float gv[8] = {ga.x, ga.y, ga.z, ga.w, gb.x, gb.y, gb.z, gb.w}, bv[8] = {ba.x, ba.y, ba.z, ba.w, bb.x, bb.y, bb.z, bb.w};
      float outv[8];
      tmem_wait_ld();
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const float d = n_local > 0 ? __uint_as_float(v[k]) - vo : 0.f;
        const float wpv = (k & 1) ? up(wv[k >> 1]).y : up(wv[k >> 1]).x;
        outv[k] = fmaf(gv[k], d, bv[k] * so);
        E[(col + k) * G_::E_LD + o] = wpv * d;
        F[(col + k) * G_::E_LD + o] = wpv * so;
      }
      *reinterpret_cast<float4*>(dst + col) = make_float4(outv[0], outv[1], outv[2], outv[3]);
      *reinterpret_cast<float4*>(dst + col + 4) = make_float4(outv[4], outv[5], outv[6], outv[7]);
    }
  }
  __syncthreads();
  if (tid < H) {
    float sgo = 0.f, sbo = 0.f;
#pragma unroll 8
    for (int o = 0; o < C; ++o) {
      sgo += E[tid * G_::E_LD + o];
      sbo += F[tid * G_::E_LD + o];
    }
    a.lnpart[(size_t)cta * 2 * H + tid] = sgo;
    a.lnpart[(size_t)cta * 2 * H + H + tid] = sbo;
  }
#endif
  tc_fence_before();
  __syncthreads();
  if (tid == 0) P1(14);
  if (!FU && warp == 2) tmem_dealloc(tmem, 512);
  if (tid == 0) P1G(15);
}

template <int SMJ>
__global__ void __launch_bounds__(512, 1)
    b1u_kernel(const __grid_constant__ CUtensorMap mxn, const __grid_constant__ CUtensorMap mtri, const __grid_constant__ CUtensorMap mwp,
               const __grid_constant__ CUtensorMap mwg, const __grid_constant__ CUtensorMap mdg, const __grid_constant__ CUtensorMap mdtri,
               const __grid_constant__ CUtensorMap mdy, const Args a) {
  extern __shared__ __align__(1024) uint8_t sm[];
  b1u_body<SMJ, false>(mxn, mtri, mwp, mwg, mdg, mdtri, mdy, a, sm, (int)blockIdx.x, (int)gridDim.x, 0u);
}

}  // namespace b1u

// ================================================================================================ b1v: unified B1, H = 128
// (2026-11-17, round 3 training pass 2) b1u for the contraction width H = 128 (D64 one direction with hidden 128, bidirectional D64):
// one CTA does the front epilogue AND the LN_out backward of its tiles, dPr / (S1 | S2) in shared memory -- no L2 ring, ring flags or
// cooperative launch (the ring b1s read tri twice and moved dPr through L2). b1u's TMEM plan does not fit at H = 128 (A [64 o][128 ch]
// and two whole-tile dnr buffers of 128 columns: 704 columns), so:
//   A^T = tri dPr  [128 ch lanes][64 o]: M = 128 channels (no padding rows), tri K-major as loaded, dPr the MN-major B operand;
//   dnr in a two-slot ring of 64-channel halves (slot = half: half 1 of tile i runs while the LN warps work on half 0, half 0 of tile
//   i + 1 while they work on half 1).
// TMEM: dWg 64 | A^T 64 | G 2 x 64 | Q 2 x 64 | dnr 2 x 64 = 512. Shared memory: Wp' (two 64-channel blocks) | Wg | dPr x 2 | NT tri
// slots [128 ch][128 tok] | NX (xn | dy -> dG) slots | (S1 | S2) x 2 | u, c | v, s. Per element the arithmetic of b1u / b1s (the
// front epilogue is b1u's; Q sums K = 128 channels; the LN constants use 1 / H, H = 128). Partials: part row cta = (dWg | dWp [64 o]
// [128 ch]), lnpart row cta = (dg_o | db_o) [2 x 128] -- the ring b1s layout, summed by grad_reduce in a fixed order.
// Warps: 0 loads, 1 MMA, 2 dG store, 3 dTri store, 4-11 front epilogue, 12-15 LN backward (both halves) then v / s.
namespace b1v {
using namespace b1s;
constexpr int H = 128, TT = H * TOK * 2, WPB = H / 64;   // tri tile 32 KB: two 64-token halves [128 ch][64 tok] (TT / 2 apart)
// 3 tri slots, 2 (xn | dy) slots (2 / 3 measured 5-8 % slower at L384 / L768; 4 / 2 does not fit)
constexpr int NT = 3, NX = 2, XS = 2 * XT;
constexpr int S_WP = 0, S_WG = WPB * KB, S_DP = S_WG + KB, S_T = S_DP + 2 * XT, S_X = S_T + NT * TT, S_SB = S_X + NX * XS,
              S_UC = S_SB + 2 * SBB, S_VF = S_UC + 2 * C * 4, O_BAR = S_VF + 2 * C * 4, SMEM = O_BAR + 512;
static_assert(SMEM <= 232448, "smem");
static_assert(S_T % 1024 == 0 && S_X % 1024 == 0 && TT % 1024 == 0 && S_DP % 1024 == 0, "SW128 bases");
static_assert(2 * 4 * H * 4 <= NT * TT, "tail scratch in the tri slots");
constexpr uint32_t T_DWG = 0, T_A = 64, T_G = 128, T_Q = 256, T_DN = 384;
constexpr uint32_t ID_GQ = idesc_bf16_mj(128, C, 0, 0);    // G = xn Wg^T
constexpr uint32_t ID_Q = idesc_bf16_mj(128, C, 1, 0);     // Q = tri^T Wp'^T (A MN-major, K = 128 channels)
constexpr uint32_t ID_DWG = idesc_bf16_mj(128, C, 1, 1);   // dWg = [xn | dG]^T xn (rows 0..63 padding)
constexpr uint32_t ID_AT = idesc_bf16_mj(128, C, 0, 1);    // A^T = tri . dPr (A K-major [ch][tok], B = dPr MN-major [tok][o])
constexpr uint32_t ID_DN = idesc_bf16_mj(128, 64, 0, 1);   // dnr half = dPr Wp'_half
struct VBars {
  uint64_t wp_full, wg_full, t_full[NT], t_free[NT], x_full[NX], x_free[NX], qg_full[2], tm_empty[2], dp_ready[4], a_done[2],
      dn_full[2], dn_empty[2], dtri_ready[4], dpb_free[2], last;
  uint32_t tmem;
};
static_assert(sizeof(VBars) <= 512, "bars");
static_assert(NT <= 4 && NX <= 4 && NX >= 2 && NT >= 2, "ring depths");
#ifndef B1V_PDL
#define B1V_PDL 1
#endif

template <int SMJ>
__global__ void __launch_bounds__(512, 1)
    b1v_kernel(const __grid_constant__ CUtensorMap mxn, const __grid_constant__ CUtensorMap mtri, const __grid_constant__ CUtensorMap mwp,
               const __grid_constant__ CUtensorMap mwg, const __grid_constant__ CUtensorMap mdg, const __grid_constant__ CUtensorMap mdtri,
               const __grid_constant__ CUtensorMap mdy, const Args a) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  VBars& B = *reinterpret_cast<VBars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, G = gridDim.x;
  const int n_local = a.tiles > cta ? (a.tiles - cta + G - 1) / G : 0;
  if (B1U_TRIG) pdl_trigger();
  if (tid == 0 && (su & 1023u)) asm volatile("trap;");
  if (tid == 0) {
    mbar_init(&B.wp_full, 1); mbar_init(&B.wg_full, 1); mbar_init(&B.last, 1);
    for (int s = 0; s < NT; ++s) { mbar_init(&B.t_full[s], 1); mbar_init(&B.t_free[s], 1); }
    for (int s = 0; s < NX; ++s) { mbar_init(&B.x_full[s], 1); mbar_init(&B.x_free[s], 2); }
    for (int s = 0; s < 4; ++s) { mbar_init(&B.dp_ready[s], 1); mbar_init(&B.dtri_ready[s], 4); }
    for (int b = 0; b < 2; ++b) {
      mbar_init(&B.qg_full[b], 1); mbar_init(&B.tm_empty[b], 1); mbar_init(&B.a_done[b], 1);
      mbar_init(&B.dn_full[b], 1); mbar_init(&B.dn_empty[b], 4); mbar_init(&B.dpb_free[b], 1 + 4);
    }
    fence_mbar_init();
    mbar_expect_tx(&B.wp_full, WPB * KB);
    for (int kb = 0; kb < WPB; ++kb) tma_load_2d(sm + S_WP + kb * KB, &mwp, &B.wp_full, kb * 64, 0, EVICT_LAST);
    mbar_expect_tx(&B.wg_full, KB);
    tma_load_2d(sm + S_WG, &mwg, &B.wg_full, 0, 0, EVICT_LAST);
  }
  if (warp == 2) { tmem_alloc(&B.tmem, 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  // ---- Wp' = wpf in place (8 threads per row o, 8 channels of each 64-channel block); u = rowsum(Wp'), c = Wp b_o; Wg / 2 (as b1u)
  mbar_wait(&B.wp_full, 0);
  {
    const int o = tid >> 3, pq = tid & 7;
    float u = 0.f, cb = 0.f;
#pragma unroll
    for (int m = 0; m < WPB; ++m) {
      const int ch0 = 64 * m + pq * 8;
      const uint32_t ad = su + S_WP + m * KB + sw128(o, pq);
      const uint4 w = lds128(ad);
      const uint4 wq = __ldg(reinterpret_cast<const uint4*>(a.wpf + (size_t)o * H + ch0));
      const float4 ba = __ldg(reinterpret_cast<const float4*>(a.bo + ch0)), bb = __ldg(reinterpret_cast<const float4*>(a.bo + ch0 + 4));
      const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
      const uint32_t qv[4] = {wq.x, wq.y, wq.z, wq.w};
      const float2 bv[4] = {make_float2(ba.x, ba.y), make_float2(ba.z, ba.w), make_float2(bb.x, bb.y), make_float2(bb.z, bb.w)};
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float2 wf = up(wv[k]);
        const float2 sc = up(qv[k]);
        u += sc.x + sc.y;
        cb = fmaf(wf.x, bv[k].x, cb);
        cb = fmaf(wf.y, bv[k].y, cb);
      }
      sts128(ad, wq.x, wq.y, wq.z, wq.w);
    }
    {
      mbar_wait(&B.wg_full, 0);
      const uint32_t wa = su + S_WG + tid * 16;
      const uint4 g = lds128(wa);
      const uint32_t gv[4] = {g.x, g.y, g.z, g.w};
      uint32_t hv[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) hv[k] = pk2(__fmul2_rn(up(gv[k]), f2(0.5f)));
      sts128(wa, hv[0], hv[1], hv[2], hv[3]);
    }
#pragma unroll
    for (int x = 1; x <= 4; x <<= 1) { u += __shfl_xor_sync(0xffffffffu, u, x); cb += __shfl_xor_sync(0xffffffffu, cb, x); }
    if (pq == 0) {
      reinterpret_cast<float*>(sm + S_UC)[o] = u;
      reinterpret_cast<float*>(sm + S_UC)[C + o] = cb;
    }
  }
  fence_async_smem();
  __syncthreads();
  if (B1V_PDL) pdl_wait();
  const int LC = a.L / TOK;
  auto t_addr = [&](int i) { return (uint32_t)(S_T + (i % NT) * TT); };   // tri [128 ch][128 tok]
  auto x_addr = [&](int i) { return (uint32_t)(S_X + (i % NX) * XS); };   // xn, then dy -> dG

  if (warp == 0 && B1U_ML) {   // as b1u: four lanes issue the tile's loads
    for (int i = 0; i < n_local; ++i) {
      const int row0 = (cta + G * i) * TOK, ts = i % NT, xs = i % NX;
      if (lane == 0) {
        if (i >= NT) mbar_wait(&B.t_free[ts], ((i / NT) - 1) & 1);
        mbar_expect_tx(&B.t_full[ts], TT);
        if (i >= NX) mbar_wait(&B.x_free[xs], ((i / NX) - 1) & 1);
        mbar_expect_tx(&B.x_full[xs], XS);
      }
      __syncwarp();
      uint8_t* pt = sm + t_addr(i);
      uint8_t* px = sm + x_addr(i);
      if (lane < 2) tri_load<SMJ>(pt + lane * (TT / 2), &mtri, &B.t_full[ts], row0 + 64 * lane, a.n2, EVICT_FIRST);
      else if (lane == 2) tma_load_2d(px, &mxn, &B.x_full[xs], 0, row0, EVICT_NORMAL);
      else if (lane == 3) tma_load_2d(px + XT, &mdy, &B.x_full[xs], 0, row0, EVICT_NORMAL);
      __syncwarp();
    }
  } else if (warp == 0) {
    if (lane == 0) {
      for (int i = 0; i < n_local; ++i) {
        const int row0 = (cta + G * i) * TOK, ts = i % NT, xs = i % NX;
        if (i >= NT) mbar_wait(&B.t_free[ts], ((i / NT) - 1) & 1);
        uint8_t* pt = sm + t_addr(i);
        mbar_expect_tx(&B.t_full[ts], TT);
        tri_load<SMJ>(pt, &mtri, &B.t_full[ts], row0, a.n2, EVICT_FIRST);
        tri_load<SMJ>(pt + TT / 2, &mtri, &B.t_full[ts], row0 + 64, a.n2, EVICT_FIRST);
        if (i >= NX) mbar_wait(&B.x_free[xs], ((i / NX) - 1) & 1);
        uint8_t* px = sm + x_addr(i);
        mbar_expect_tx(&B.x_full[xs], XS);
        tma_load_2d(px, &mxn, &B.x_full[xs], 0, row0, EVICT_NORMAL);
        tma_load_2d(px + XT, &mdy, &B.x_full[xs], 0, row0, EVICT_NORMAL);
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      mbar_wait(&B.wg_full, 0);
      const uint32_t swp = su + S_WP, swg = su + S_WG;
      auto mma_qg = [&](int i) {
        const int b = i & 1;
        const uint32_t pt = su + t_addr(i), px = su + x_addr(i);
        mbar_wait(&B.t_full[i % NT], (i / NT) & 1);
        mbar_wait(&B.x_full[i % NX], (i / NX) & 1);
        if (i >= 2) mbar_wait(&B.tm_empty[b], ((i >> 1) - 1) & 1);
        tc_fence_after();
#pragma unroll
        for (int k = 0; k < H / 16; ++k)
          umma_ss(tmem + T_Q + 64 * b, desc_mn_sw128(pt + k * 2048, TT / 2), desc_k_sw128(swp + (k >> 2) * KB + (k & 3) * 32), ID_Q, k != 0);
#pragma unroll
        for (int k = 0; k < 4; ++k)
          umma_ss(tmem + T_G + 64 * b, desc_k_sw128(px + k * 32), desc_k_sw128(swg + k * 32), ID_GQ, k != 0);
        umma_commit(&B.qg_full[b]);
      };
      if (n_local > 0) mma_qg(0);
      for (int i = 0; i < n_local; ++i) {
        if (i + 1 < n_local) mma_qg(i + 1);
        const int b = i & 1;
        const uint32_t pt = su + t_addr(i), px = su + x_addr(i), sdp = su + S_DP + b * XT;
        mbar_wait(&B.dp_ready[i & 3], (i >> 2) & 1);
        tc_fence_after();
        // dWg [(64 padding rows: xn +) 64 o][64 i] += dG^T xn (lanes 64..127)
#pragma unroll
        for (int k = 0; k < 8; ++k)
          umma_ss(tmem + T_DWG, desc_mn_sw128(px + k * 2048, XT), desc_mn_sw128(px + k * 2048, XT), ID_DWG, (i | k) != 0);
        umma_commit(&B.x_free[i % NX]);
        // A^T [128 ch][64 o] += tri [ch][128 tok] . dPr [tok][o]
#pragma unroll
        for (int k = 0; k < 8; ++k)
          umma_ss(tmem + T_A, desc_k_sw128(pt + (k >> 2) * (TT / 2) + (k & 3) * 32), desc_nosw(sdp + k * 256, 128, 2048), ID_AT, (i | k) != 0);
        umma_commit(&B.a_done[b]);
        // dnr half hc [128 tok][64 ch] = dPr [tok][64 o] . Wp' [o][64 hc ..], into slot hc
#pragma unroll
        for (int hc = 0; hc < WPB; ++hc) {
          if (i >= 1) mbar_wait(&B.dn_empty[hc], (i - 1) & 1);
          tc_fence_after();
#pragma unroll
          for (int k = 0; k < 4; ++k)
            umma_ss(tmem + T_DN + 64 * hc, desc_nosw(sdp + k * 4096, 2048, 128), desc_mn_sw128(swp + hc * KB + k * 2048, KB), ID_DN, k != 0);
          umma_commit(&B.dn_full[hc]);
        }
        umma_commit(&B.dpb_free[b]);
      }
      if (n_local > 0) { umma_commit(&B.last); mbar_wait(&B.last, 0); }
    }
  } else if (warp == 2) {
    if (lane == 0) {
      for (int i = 0; i < n_local; ++i) {
        const int row0 = (cta + G * i) * TOK;
        mbar_wait(&B.dp_ready[i & 3], (i >> 2) & 1);
        tma_store_2d(&mdg, sm + x_addr(i) + XT, 0, row0);
        bulk_commit();
        bulk_wait_read<0>();
        mbar_arrive(&B.x_free[i % NX]);
      }
      BULK_END();
    }
  } else if (warp == 3) {
    if (lane == 0) {
      for (int i = 0; i < n_local; ++i) {
        const int row0 = (cta + G * i) * TOK;
        mbar_wait(&B.dtri_ready[i & 3], (i >> 2) & 1);
        tri_store<SMJ>(&mdtri, sm + t_addr(i), row0, a.n2);
        tri_store<SMJ>(&mdtri, sm + t_addr(i) + TT / 2, row0 + 64, a.n2);
        bulk_commit();
        bulk_wait_read<0>();
        mbar_arrive(&B.t_free[i % NT]);
      }
      BULK_END();
    }
  } else if (warp < 12) {
    // 8 front-epilogue warps (b1u's): lane quarter q (tokens 32 q ..), 8-column chunks 4 kq .. 4 kq + 3
    const int q = warp & 3, kq = (warp - 4) >> 2, r = q * 32 + lane, c_lo = 4 * kq;
    const uint32_t trow = tmem + ((uint32_t)(q * 32) << 16);
    const uint32_t suc = su + S_UC;
    float* sbs = reinterpret_cast<float*>(sm + S_SB);
    uint4 dsn[4];
    float mean_n = 0.f, rs_n = 0.f;
    auto ld_tok = [&](int i) {
      const int t = cta + G * i, sbi = t / (a.L * LC), jc = t % LC;
      const uint4* dsr = reinterpret_cast<const uint4*>(a.ds + ((size_t)sbi * a.L + jc * TOK + r) * C);
#pragma unroll
      for (int cc = 0; cc < 4; ++cc) dsn[cc] = __ldg(dsr + c_lo + cc);
      mean_n = __ldg(a.mean_o + t * TOK + r); rs_n = __ldg(a.rs_o + t * TOK + r);
    };
    if (n_local > 0) ld_tok(0);
    for (int i = 0; i < n_local; ++i) {
      const int b = i & 1;
      uint4 dsv[4];
#pragma unroll
      for (int cc = 0; cc < 4; ++cc) dsv[cc] = dsn[cc];
      const float mean = mean_n, rs = rs_n;
      if (i + 1 < n_local) ld_tok(i + 1);
      const uint32_t sdp = su + S_DP + b * XT;
      mbar_wait(&B.qg_full[b], (i >> 1) & 1);
      mbar_wait(&B.x_full[i % NX], (i / NX) & 1);
      if (i >= 2) mbar_wait(&B.dpb_free[b], ((i >> 1) - 1) & 1);
      tc_fence_after();
      const float2 rs2 = f2(rs), nmean2 = f2(-mean);
      float2 s1 = f2(0.f), s2 = f2(0.f);
      const uint32_t sdy = su + x_addr(i) + XT;
      uint32_t vq[4][8], vg[4][8];
      uint4 dyw[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int c = c_lo + e;
        tmem_ld8(trow + T_Q + 64 * b + c * 8, vq[e]);
        tmem_ld8(trow + T_G + 64 * b + c * 8, vg[e]);
        dyw[e] = lds128(sdy + sw128(r, c));
      }
      tmem_wait_ld();
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int c = c_lo + e;
        const uint32_t ya = sdy + sw128(r, c);
        const uint4 dsw = dsv[e];
        const float4 u0 = ldsf4(suc + 32 * c), u1 = ldsf4(suc + 32 * c + 16);
        const float4 cq0 = ldsf4(suc + 4 * C + 32 * c), cq1 = ldsf4(suc + 4 * C + 32 * c + 16);
        const float2 uu[4] = {make_float2(u0.x, u0.y), make_float2(u0.z, u0.w), make_float2(u1.x, u1.y), make_float2(u1.z, u1.w)};
        const float2 cv[4] = {make_float2(cq0.x, cq0.y), make_float2(cq0.z, cq0.w), make_float2(cq1.x, cq1.y), make_float2(cq1.z, cq1.w)};
        const uint32_t yv[4] = {dyw[e].x, dyw[e].y, dyw[e].z, dyw[e].w}, sv[4] = {dsw.x, dsw.y, dsw.z, dsw.w};
        uint32_t odp[4], odg[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float2 aa = __fmul2_rn(up(yv[k]), up(sv[k]));
          const float2 sg = sigh2(rbf2(make_float2(__uint_as_float(vg[e][2 * k]), __uint_as_float(vg[e][2 * k + 1]))));
          const float2 q2 = make_float2(__uint_as_float(vq[e][2 * k]), __uint_as_float(vq[e][2 * k + 1]));
          const float2 qc = __ffma2_rn(nmean2, uu[k], q2);
          const float2 pp = __ffma2_rn(rs2, qc, cv[k]);
          odp[k] = pk2(__fmul2_rn(__fmul2_rn(aa, sg), rs2));
          const float2 dpr = up(odp[k]);
          s1 = __ffma2_rn(dpr, qc, s1);
          s2 = __ffma2_rn(dpr, uu[k], s2);
          odg[k] = pk2(__fmul2_rn(__fmul2_rn(aa, pp), __fmul2_rn(sg, __fadd2_rn(f2(1.f), make_float2(-sg.x, -sg.y)))));
        }
        sts128(ya, odg[0], odg[1], odg[2], odg[3]);
        sts128(sdp + c * 2048 + r * 16, odp[0], odp[1], odp[2], odp[3]);
      }
      tc_fence_before();
      const float p1 = s1.x + s1.y, p2 = s2.x + s2.y;
      float* sb = sbs + b * 2 * TOK;
      if (kq == 1) {
        sb[r] = p1; sb[TOK + r] = p2;
        named_bar_sync(2, 256);
      } else {
        named_bar_sync(2, 256);
        sb[r] += p1; sb[TOK + r] += p2;
      }
      fence_async_smem();
      named_bar_sync(1, 256);
      if (tid == 128) { mbar_arrive(&B.tm_empty[b]); mbar_arrive(&B.dp_ready[i & 3]); }
    }
  } else {
    // 4 LN warps: the LN_out backward of tokens 32 q .. 32 q + 31 over both 64-channel halves (dnr slot = half), then v / s
    const int q = warp & 3, wv = warp - 12, tr = lane >> 2;
    const float* sbs = reinterpret_cast<const float*>(sm + S_SB);
    float x[32];
#pragma unroll
    for (int k = 0; k < 32; ++k) x[k] = 0.f;
    for (int i = 0; i < n_local; ++i) {
      const int row0 = (cta + G * i) * TOK, b = i & 1;
      const uint32_t tb = su + t_addr(i), sdp = su + S_DP + b * XT;
      float mv[2][2], rv[2][2];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int tok = row0 + q * 32 + hh * 16 + tr;
        mv[hh][0] = __ldg(a.mean_o + tok); rv[hh][0] = __ldg(a.rs_o + tok);
        mv[hh][1] = __ldg(a.mean_o + tok + 8); rv[hh][1] = __ldg(a.rs_o + tok + 8);
      }
      float vm[4], vs[4];
#pragma unroll
      for (int m = 0; m < 4; ++m) { vm[m] = __ldg(a.mean_o + row0 + lane + 32 * m); vs[m] = __ldg(a.rs_o + row0 + lane + 32 * m); }
      mbar_wait(&B.dp_ready[i & 3], (i >> 2) & 1);
      mbar_wait(&B.a_done[b], (i >> 1) & 1);     // A^T has read tri: it may be overwritten
      const float* sbl = sbs + b * 2 * TOK;
      float2 kk1[2][2], kk0[2][2], nm[2][2];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int tok0 = q * 32 + hh * 16;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float rr = rv[hh][e];
          kk1[hh][e] = f2(-rr * rr * sbl[tok0 + tr + 8 * e] * (1.f / H));
          kk0[hh][e] = f2(-sbl[TOK + tok0 + tr + 8 * e] * (1.f / H));
          nm[hh][e] = f2(-mv[hh][e]);
        }
      }
#pragma unroll 1
      for (int hc = 0; hc < WPB; ++hc) {
        mbar_wait(&B.dn_full[hc], i & 1);
        tc_fence_after();
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int tok0 = q * 32 + hh * 16;
          const uint32_t ta = tmem + ((uint32_t)tok0 << 16) + T_DN + 64 * hc;
          uint32_t tv2[2][16], tt4[4][4];
          tmem_ld16x256_x4(ta, tv2[0]);
          tmem_ld16x256_x4(ta + 32, tv2[1]);
#pragma unroll
          for (int jj = 0; jj < 4; ++jj)
            ldsm_x4_trans(frag_addr<H>(tb, tok0, 4 * hc + jj, lane), tt4[jj][0], tt4[jj][1], tt4[jj][2], tt4[jj][3]);
          tmem_wait_ld();
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) {
            uint32_t* tt = tt4[jj];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const float2 dn = dn_pair(jj, e, tv2[jj >> 1]);
              const float2 d = __fadd2_rn(up(tt[e]), nm[hh][e >> 1]);
              tt[e] = pk2(__fadd2_rn(__ffma2_rn(d, kk1[hh][e >> 1], kk0[hh][e >> 1]), dn));
            }
            stsm_x4_trans(frag_addr<H>(tb, tok0, 4 * hc + jj, lane), tt[0], tt[1], tt[2], tt[3]);
          }
        }
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.dn_empty[hc]);
      }
      fence_async_smem();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.dtri_ready[i & 3]);
      const uint32_t base = sdp + (2 * wv) * 2048;
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const int tok = lane + 32 * m;
        const float mvv = vm[m], sd = rcp_ftz(vs[m]);
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          const uint4 w = lds128(base + c * 2048 + tok * 16);
          const uint32_t wq[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const float2 f = up(wq[k]);
            x[c * 16 + 4 * k] = fmaf(f.x, mvv, x[c * 16 + 4 * k]);
            x[c * 16 + 4 * k + 1] = fmaf(f.x, sd, x[c * 16 + 4 * k + 1]);
            x[c * 16 + 4 * k + 2] = fmaf(f.y, mvv, x[c * 16 + 4 * k + 2]);
            x[c * 16 + 4 * k + 3] = fmaf(f.y, sd, x[c * 16 + 4 * k + 3]);
          }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.dpb_free[b]);
    }
#pragma unroll
    for (int st = 0, msk = 16, nh = 16; st < 5; ++st, msk >>= 1, nh >>= 1) {
      const bool hi = (lane & msk) != 0;
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        if (k < nh) {
          const float send = hi ? x[k] : x[k + nh];
          const float keep = hi ? x[k + nh] : x[k];
          x[k] = keep + __shfl_xor_sync(0xffffffffu, send, msk);
        }
      }
    }
    float* vfo = reinterpret_cast<float*>(sm + S_VF);
    const int o = (2 * wv + (lane >> 4)) * 8 + ((lane >> 1) & 7);
    vfo[(lane & 1) * C + o] = x[0];
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const float* vf = reinterpret_cast<const float*>(sm + S_VF);   // v [C], s [C]
  if ((warp & 3) >= 2) {
    // dWg partial (TMEM lanes 64..127 = o), b1u's
    const int q = warp & 3, cg = warp >> 2, o = (q - 2) * 32 + lane;
    float* dst = a.part + ((size_t)cta * C + o) * (C + H);
#pragma unroll
    for (int h8 = 0; h8 < 2; ++h8) {
      uint32_t v[8];
      tmem_ld8(tmem + ((uint32_t)(q * 32) << 16) + T_DWG + cg * 16 + h8 * 8, v);
      tmem_wait_ld();
#pragma unroll
      for (int k = 0; k < 8; k += 4)
        *reinterpret_cast<float4*>(dst + cg * 16 + h8 * 8 + k) =
            n_local > 0 ? make_float4(__uint_as_float(v[k]), __uint_as_float(v[k + 1]), __uint_as_float(v[k + 2]), __uint_as_float(v[k + 3]))
                        : make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }
  {
    // LN tail on A^T (TMEM lane = channel, column = o): every warp (q, cg) -> ch = 32 q + lane, o = 16 cg .. 16 cg + 15:
    // dWp[o][ch] = g_o (A - v[o]) + s[o] b_o -> columns C + ch of part rows o; per thread the partial dg_o = sum_o Wp (A - v) and
    // db_o = sum_o Wp s over its 16 o, then the 4 column groups added in a fixed order (cg 0 .. 3)
    const int q = warp & 3, cg = warp >> 2, ch = q * 32 + lane;
    const float gch = __ldg(a.go + ch), bch = __ldg(a.bo + ch);
    float* scr = reinterpret_cast<float*>(sm + S_T);   // [2][4 cg][H]
    float sgo = 0.f, sbo = 0.f;
#pragma unroll
    for (int h8 = 0; h8 < 2; ++h8) {
      uint32_t v[8];
      tmem_ld8(tmem + ((uint32_t)(q * 32) << 16) + T_A + cg * 16 + h8 * 8, v);
      tmem_wait_ld();
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int o = cg * 16 + h8 * 8 + k;
        const float d = n_local > 0 ? __uint_as_float(v[k]) - vf[o] : 0.f;
        const float so = vf[C + o];
        const float wpv = __bfloat162float(a.wp[(size_t)o * H + ch]);
        a.part[((size_t)cta * C + o) * (C + H) + C + ch] = fmaf(gch, d, bch * so);
        sgo = fmaf(wpv, d, sgo);
        sbo = fmaf(wpv, so, sbo);
      }
    }
    scr[cg * H + ch] = sgo;
    scr[4 * H + cg * H + ch] = sbo;
  }
  __syncthreads();
  if (tid < H) {
    const float* scr = reinterpret_cast<const float*>(sm + S_T);
    float sgo = 0.f, sbo = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) { sgo += scr[k * H + tid]; sbo += scr[4 * H + k * H + tid]; }
    a.lnpart[(size_t)cta * 2 * H + tid] = sgo;
    a.lnpart[(size_t)cta * 2 * H + H + tid] = sbo;
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) tmem_dealloc(tmem, 512);
}

}  // namespace b1v

void b1s_backward(torch::Tensor dy, torch::Tensor xn, torch::Tensor tri, torch::Tensor ds, torch::Tensor mean_o, torch::Tensor rs_o,
                  torch::Tensor wg, torch::Tensor wp, torch::Tensor wpf, torch::Tensor go, torch::Tensor bo, torch::Tensor dg, torch::Tensor dtri,
                  torch::Tensor part, torch::Tensor lnpart, torch::Tensor ring, torch::Tensor flags, int64_t L, int64_t nf) {
  using namespace b1s;
  const int64_t M = dy.size(0);
  const int H = (int)tmap::chmajor_channels(tri);   // tri / dtri [H, ..] or sample-major [H / 64, B, 64, L, L], the same layout
  TORCH_CHECK(dy.size(1) == C && M % (L * L) == 0 && L % TOK == 0 && tri.numel() == H * M && (H == 64 || H == 128));   // M = B L L, tokens b-major
  TORCH_CHECK(ds.numel() == (M / (L * L)) * L * C, "b1s_backward: ds must be [B, L, 64]");
  TORCH_CHECK(wp.is_contiguous() && wp.numel() == (int64_t)C * H && go.numel() == H && bo.numel() == H);
  int nsm = 0, dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  // nf == 0: the unified kernel (b1u at H = 64, b1v at H = 128; no ring, no flags): grid min(tiles, SMs), part / lnpart one row per CTA.
  // nf > 0: ring RS dPr tiles of 128 rows (128 B each), then RS (S1 | S2) blocks of 1 KB (8 rows)
  const bool uni = nf == 0;
  const int RS = uni ? 0 : (int)(ring.size(0) / (TOK + SBB / (C * 2))), RSF = uni ? 0 : RS / (int)nf;
  TORCH_CHECK(uni || (nf > 0 && nf < nsm && flags.numel() >= RS && RSF >= 1 && nf % (L / TOK) == 0 && ring.is_contiguous() &&
              ring.size(0) == (int64_t)RS * (TOK + SBB / (C * 2))), "b1s_backward: ring [RS (128 + 8), 64]");
  auto bf = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  auto tok_map = [&](const torch::Tensor& t, int64_t rows) {
    return tmap::make(t.data_ptr(), bf, {(uint64_t)C, (uint64_t)rows}, {(uint64_t)C * 2}, {64, TOK}, CU_TENSOR_MAP_SWIZZLE_128B);
  };
  TORCH_CHECK(dtri.dim() == tri.dim() && dtri.numel() == tri.numel(), "b1s_backward: dtri has tri's layout");
  const int n2 = tri.dim() == 5 ? (int)(L * L) : 0;
  auto ch_map = [&](const torch::Tensor& t) {
    return n2 ? tmap::make_chmajor(t, n2, 64, (uint32_t)H, CU_TENSOR_MAP_SWIZZLE_128B)
              : tmap::make(t.data_ptr(), bf, {(uint64_t)M, (uint64_t)H}, {(uint64_t)M * 2}, {64, (uint32_t)H}, CU_TENSOR_MAP_SWIZZLE_128B);
  };
  auto mxn = tok_map(xn, M), mdg = tok_map(dg, M), mdy = tok_map(dy, M);
  auto mtri = ch_map(tri), mdtri = ch_map(dtri);
  auto mwp = tmap::make(wp.data_ptr(), bf, {(uint64_t)H, (uint64_t)C}, {(uint64_t)H * 2}, {64, (uint32_t)C}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mwg = tmap::make(wg.data_ptr(), bf, {(uint64_t)C, (uint64_t)C}, {(uint64_t)C * 2}, {64, 64}, CU_TENSOR_MAP_SWIZZLE_128B);
  Args a;
  a.ds = reinterpret_cast<const __nv_bfloat16*>(ds.data_ptr());
  a.wp = reinterpret_cast<const __nv_bfloat16*>(wp.data_ptr());
  TORCH_CHECK(wpf.is_contiguous() && wpf.scalar_type() == torch::kBFloat16 && wpf.numel() == (int64_t)C * H, "wpf [C, H] bf16");
  a.wpf = reinterpret_cast<const __nv_bfloat16*>(wpf.data_ptr());
  a.mean_o = mean_o.data_ptr<float>(); a.rs_o = rs_o.data_ptr<float>(); a.go = go.data_ptr<float>(); a.bo = bo.data_ptr<float>();
  const int64_t tiles = M / TOK, gu = tiles < nsm ? tiles : nsm;
  const int64_t nl = uni ? gu : nsm - nf, rows = uni ? gu : (nf > nl ? nf : nl);
  TORCH_CHECK(lnpart.numel() >= nl * 2 * H);
  TORCH_CHECK(part.numel() >= rows * C * (C + H) && part.scalar_type() == torch::kFloat32, "b1s_backward: part holds max(nf, nsm - nf) rows");
  a.part = part.data_ptr<float>(); a.lnpart = lnpart.data_ptr<float>();
  if (uni) {
    a.flags = nullptr; a.ring = nullptr; a.sring = nullptr;
    a.L = (int)L; a.tiles = (int)tiles; a.NF = 0; a.RSF = 0; a.n2 = n2;
    if (H == 128) {
      static bool attrv = false, attrvs = false;
      auto kv = n2 ? b1v::b1v_kernel<1> : b1v::b1v_kernel<0>;
      bool& av = n2 ? attrvs : attrv;
      if (!av) { C10_CUDA_CHECK(cudaFuncSetAttribute(kv, cudaFuncAttributeMaxDynamicSharedMemorySize, b1v::SMEM)); av = true; }
      tmap::launch_pdl(kv, dim3((unsigned)gu), dim3(512), b1v::SMEM, at::cuda::getCurrentCUDAStream(), B1V_PDL, mxn, mtri, mwp, mwg, mdg,
                       mdtri, mdy, a);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
    if (n2 && B1U_T1_ON) {   // b1u: the tri tile in one 5-D box (tri [1, B, 64, L, L])
      TORCH_CHECK(tri.dim() == 5 && tri.is_contiguous() && tri.size(0) == 1 && tri.size(2) == 64 && n2 % 64 == 0, "b1u: sample-major tri");
      mtri = tmap::make(tri.data_ptr(), bf, {64, 64, (uint64_t)(n2 / 64), (uint64_t)tri.size(1), 1},
                        {(uint64_t)n2 * 2, 128, (uint64_t)(64 * n2) * 2, (uint64_t)(tri.size(1) * 64 * n2) * 2}, {64, 64, 2, 1, 1},
                        CU_TENSOR_MAP_SWIZZLE_128B);
    }
    static bool attru = false, attrus = false;
    auto kern = n2 ? b1u::b1u_kernel<1> : b1u::b1u_kernel<0>;
    bool& at = n2 ? attrus : attru;
    if (!at) { C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, b1u::SMEM)); at = true; }
    tmap::launch_pdl(kern, dim3((unsigned)gu), dim3(512), b1u::SMEM, at::cuda::getCurrentCUDAStream(), B1U_PDL, mxn, mtri, mwp, mwg, mdg,
                     mdtri, mdy, a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#ifdef PROF
    {  // eager probe runs only (synchronizes): per role, the mean over CTAs of each wait slot and the total, in cycles
      static unsigned long long hp[160][8][8];
      C10_CUDA_CHECK(cudaDeviceSynchronize());
      C10_CUDA_CHECK(cudaMemcpyFromSymbol(hp, g_prof, sizeof(hp)));
      for (int r = 0; r < 6; ++r) {
        printf("[b1u prof] role %d:", r);
        for (int k = 0; k < 8; ++k) { double s = 0; for (int c = 0; c < gu; ++c) s += (double)hp[c][r][k]; printf(" %8.0f", s / gu); }
        printf("\n");
      }
      fflush(stdout);
    }
#endif
    return;
  }
  a.flags = reinterpret_cast<uint32_t*>(flags.data_ptr<int32_t>());
  a.ring = reinterpret_cast<__nv_bfloat16*>(ring.data_ptr());
  a.sring = reinterpret_cast<float*>(reinterpret_cast<__nv_bfloat16*>(ring.data_ptr()) + (size_t)RS * TOK * C);
  a.L = (int)L; a.tiles = (int)(M / TOK); a.NF = (int)nf; a.RSF = RSF; a.n2 = n2;
  void* params[] = {(void*)&mxn, (void*)&mtri, (void*)&mwp, (void*)&mwg, (void*)&mdg, (void*)&mdtri, (void*)&mdy, (void*)&a};
  auto launch = [&](const void* fn, int smem, bool& attr) {
    if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); attr = true; }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(nsm); cfg.blockDim = dim3(512); cfg.dynamicSmemBytes = smem; cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute la[2];
    int na = 0;
    la[na].id = cudaLaunchAttributeCooperative; la[na].val.cooperative = 1; ++na;   // the ring flags need every CTA resident
    if (B1S_PDL) { la[na].id = cudaLaunchAttributeProgrammaticStreamSerialization; la[na].val.programmaticStreamSerializationAllowed = 1; ++na; }
    cfg.attrs = la; cfg.numAttrs = na;
    C10_CUDA_CHECK(cudaLaunchKernelExC(&cfg, fn, params));
  };
  static bool attr64 = false, attr128 = false, attr64s = false, attr128s = false;
  if (H == 64) launch(n2 ? (const void*)b1s_kernel<64, 1> : (const void*)b1s_kernel<64, 0>, Geo<64>::SMEM, n2 ? attr64s : attr64);
  else launch(n2 ? (const void*)b1s_kernel<128, 1> : (const void*)b1s_kernel<128, 0>, Geo<128>::SMEM, n2 ? attr128s : attr128);
}
// b1u phase trace (-DPT_TL=1, see g_pt1): [160][16]; an empty tensor in other builds
torch::Tensor b1u_tl() {
#if defined(PT_TL) && PT_TL
  auto h = torch::empty({160, 16}, torch::kInt64);
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(h.data_ptr(), b1u::g_pt1, sizeof(unsigned long long) * 160 * 16));
  return h;
#else
  return torch::Tensor();
#endif
}
