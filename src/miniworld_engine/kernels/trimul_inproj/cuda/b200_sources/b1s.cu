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
  const __nv_bfloat16 *ds, *wp;
  const float *mean_o, *rs_o, *go, *bo;
  float *part, *lnpart;     // part [max(NF, NL)][C][C + H] (row o = dWg[o] | dWp[o]); lnpart [NL][dg_o H | db_o H]
  uint32_t* flags;          // [NF RSF]
  float* sring;             // [NF RSF][S1 128 | S2 128]
  __nv_bfloat16* ring;      // [NF RSF][8][128][8] dPr tiles
  int L, tiles, NF, RSF;
  int n2;                   // > 0: tri / dtri sample-major (4-D maps), 0: [H, M]
};

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
  // ---- Wp' = bf16(Wp g_o) in place (8 threads per row o); front: u = rowsum(Wp'), c = Wp b_o
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
      const float4 ga = __ldg(reinterpret_cast<const float4*>(a.go + ch0)), gb = __ldg(reinterpret_cast<const float4*>(a.go + ch0 + 4));
      const float4 ba = __ldg(reinterpret_cast<const float4*>(a.bo + ch0)), bb = __ldg(reinterpret_cast<const float4*>(a.bo + ch0 + 4));
      const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
      const float2 gg[4] = {make_float2(ga.x, ga.y), make_float2(ga.z, ga.w), make_float2(gb.x, gb.y), make_float2(gb.z, gb.w)};
      const float2 bv[4] = {make_float2(ba.x, ba.y), make_float2(ba.z, ba.w), make_float2(bb.x, bb.y), make_float2(bb.z, bb.w)};
      uint32_t ov[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float2 wf = up(wv[k]);
        const float2 sc = rbf2(__fmul2_rn(wf, gg[k]));
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

void b1s_backward(torch::Tensor dy, torch::Tensor xn, torch::Tensor tri, torch::Tensor ds, torch::Tensor mean_o, torch::Tensor rs_o,
                  torch::Tensor wg, torch::Tensor wp, torch::Tensor go, torch::Tensor bo, torch::Tensor dg, torch::Tensor dtri,
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
  // ring: RS dPr tiles of 128 rows (128 B each), then RS (S1 | S2) blocks of 1 KB (8 rows)
  const int RS = (int)(ring.size(0) / (TOK + SBB / (C * 2))), RSF = RS / (int)nf;
  TORCH_CHECK(nf > 0 && nf < nsm && flags.numel() >= RS && RSF >= 1 && nf % (L / TOK) == 0 && ring.is_contiguous() &&
              ring.size(0) == (int64_t)RS * (TOK + SBB / (C * 2)), "b1s_backward: ring [RS (128 + 8), 64]");
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
  a.mean_o = mean_o.data_ptr<float>(); a.rs_o = rs_o.data_ptr<float>(); a.go = go.data_ptr<float>(); a.bo = bo.data_ptr<float>();
  const int64_t nl = nsm - nf, rows = nf > nl ? nf : nl;
  TORCH_CHECK(lnpart.numel() >= nl * 2 * H);
  TORCH_CHECK(part.numel() >= rows * C * (C + H) && part.scalar_type() == torch::kFloat32, "b1s_backward: part holds max(nf, nsm - nf) rows");
  a.part = part.data_ptr<float>(); a.lnpart = lnpart.data_ptr<float>();
  a.flags = reinterpret_cast<uint32_t*>(flags.data_ptr<int32_t>());
  a.ring = reinterpret_cast<__nv_bfloat16*>(ring.data_ptr());
  a.sring = reinterpret_cast<float*>(reinterpret_cast<__nv_bfloat16*>(ring.data_ptr()) + (size_t)RS * TOK * C);
  a.L = (int)L; a.tiles = (int)(M / TOK); a.NF = (int)nf; a.RSF = RSF; a.n2 = n2;
  void* params[] = {(void*)&mxn, (void*)&mtri, (void*)&mwp, (void*)&mwg, (void*)&mdg, (void*)&mdtri, (void*)&mdy, (void*)&a};
  auto launch = [&](const void* fn, int smem, bool& attr) {
    if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); attr = true; }
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(fn, dim3(nsm), dim3(512), params, smem, at::cuda::getCurrentCUDAStream()));
  };
  static bool attr64 = false, attr128 = false, attr64s = false, attr128s = false;
  if (H == 64) launch(n2 ? (const void*)b1s_kernel<64, 1> : (const void*)b1s_kernel<64, 0>, Geo<64>::SMEM, n2 ? attr64s : attr64);
  else launch(n2 ? (const void*)b1s_kernel<128, 1> : (const void*)b1s_kernel<128, 0>, Geo<128>::SMEM, n2 ? attr128s : attr128);
}
