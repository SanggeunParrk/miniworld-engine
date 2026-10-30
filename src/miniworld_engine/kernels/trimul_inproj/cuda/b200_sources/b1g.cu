// b1g.cu -- TriMul B1 (output-side backward) on B200, d_pair C = 128, contraction width H (256 bidirectional, 128 one direction).
//
// LN_out's affine is folded into the projection (as k3w does in the forward of the wide path): Wp' = bf16(Wp diag(g_o)),
// u = rowsum(Wp'), c = Wp b_o. With Q = tri Wp'^T on the raw contraction output (no normalised tile is formed):
//   P = rs (Q - mean u) + c,      dP = dy ds sig(G),  dG = dy ds P sig(G) (1 - sig(G)),  dPr = bf16(dP rs)
//   dnr = dPr Wp' (= rs g_o dn),  dTri = dnr - S2 / H - (tri - mean) rs^2 S1 / H  with the per-token sums
//   S1 = sum_ch dnr (tri - mean) = sum_o dPr (Q - mean u),  S2 = sum_ch dnr = sum_o dPr u   (formed by the front epilogue)
//   A = sum_tok dPr^T tri,  v = sum_tok dPr mean,  s = sum_tok dPr / rs:
//   dWp = g_o (A - v 1^T) + s b_o^T,   dg_o = colsum(Wp o (A - v 1^T)),   db_o = Wp^T s
// so no role runs a LayerNorm-forward pass, the LayerNorm backward is one pass, and no per-tile reduction over tokens runs on
// the CUDA cores (A on the tensor cores; v / s on four spare warps).
//
// Roles (cooperative grid, one CTA per SM; both form Wp' in shared memory at start):
//   front CTAs (NF): Q = tri Wp'^T (double-buffered TMEM: the next tile's Q runs during this tile's epilogue), G = xn Wg^T
//                    (Wg resident), P / dP / dG / S1 / S2 epilogue per 64-output block: dPr and (S1, S2) stored from registers
//                    straight into this CTA's ring slot, dG in place of dy -> global (TMA) and dWg^T += xn^T dG (TMEM, per
//                    block, so the next tile's dy refills a block as soon as both have read it); the flag (generation) is
//                    published by warp 2 (a gpu-scope release costs ~2k clk). tri's channel halves and xn share three 32 KB
//                    slots (a tri half lives until its Q product, xn until dWg^T), so the next tile's operands load during
//                    this tile's epilogue.
//   LN CTAs (NL):    A += dPr^T tri, then per channel half h dnr_h = dPr Wp'_h (TMEM halves: the next tile's half 0 runs while
//                    this tile's half 1 is finished), LN backward in one pass -> dTri in place (TMA store); v / s; at the end
//                    dWp, dg_o, db_o from A, v, s.
// Partials (plain stores; grad_reduce sums them in a fixed order): part row r < max(NF, NL) = (dWg of front CTA r | dWp of LN CTA r),
// zeros where that CTA does not exist; lnpart row l = (dg_o | db_o) of LN CTA l. Deterministic.
// Ring: front CTA f takes tiles t = f + NF i (NF a multiple of L / 128: one j class, so its ds rows are fixed per sample) and owns
// slots f RSF .. f RSF + RSF - 1: tile i -> slot f RSF + i % RSF, generation i / RSF + 1. Flags are 0 at kernel end. The ring holds
// RS = NF RSF dPr tiles of 32 KB, each in the SWIZZLE_NONE K-major operand layout [16 column chunks][128 tok][8 o] (a front warp
// stores a chunk of its 32 tokens as 512 contiguous bytes; the LN CTA bulk-copies the tile and feeds it to its MMAs as is), then
// RS (S1 [128] | S2 [128]) fp32 blocks of 1 KB.
// Ring hand-over (generic-proxy global stores -> async-proxy bulk copies on another SM), in order:
//   epilogue threads: st.global dPr / S -> fence.proxy.async.global (each writer, orders its generic stores before later
//   async-proxy accesses) -> bar.sync (all 384 epilogue threads) -> mbarrier arrive dp_ready (release.cta) -> store warp:
//   arrive published -> flag warp: st.release.gpu flag = generation;  LN producer: ld.acquire.gpu flag -> fence.proxy.async.global
//   -> cp.async.bulk of the slot. The flag release is cumulative over the writers' stores through the CTA barrier / mbarrier
//   chain. (dG and dTri go out by TMA stores; tile operands come in by TMA loads.)
#include "sm100.cuh"
#include "tmap.h"

using namespace sm100;

namespace b1g {

constexpr int C = 128, TOK = 128;
constexpr int XT = TOK * C * 2;            // 32 KB: a [128 tok][128] tile, two SW128 K-blocks of 64 columns (XT / 2 apart)
constexpr int KB = C * 128;                // 16 KB: one 64-channel block of Wp' [C o][H ch]
constexpr int SBB = 2 * TOK * 4;           // 1 KB: a tile's per-token (S1 | S2)
template <int H>
struct Geo {
  static_assert(H * TOK <= TOK * C * 2, "a tri channel half fits an operand slot");
  static constexpr int TT = H * TOK * 2;   // tri tile [H ch][128 tok], two 64-token halves (TT / 2 apart)
  static constexpr int WPT = C * H * 2;    // Wp' (both roles, at 0)
  // front: Wp' | 3 operand slots (tri channel halves, xn) | dy (-> dG in place) | Wg | u, c | S partials
  static constexpr int SLOT = XT;          // >= TT / 2
  static constexpr int F_S = WPT, F_DY = F_S + 3 * SLOT, F_WG = F_DY + XT, F_UC = F_WG + XT, F_SB = F_UC + 2 * C * 4;
  static constexpr int F_END = F_SB + SBB;
  // LN: Wp' | dPr | tri x 2 | S1, S2 | v, s (written by the v / s warps when they finish)
  static constexpr int L_DP = WPT, L_T = L_DP + XT, L_SB = L_T + 2 * TT, L_VF = L_SB + SBB, L_END = L_VF + 2 * C * 4;
  // LN tail scratch (everything below L_VF is dead by then): E [H][C + 1] fp32 at 0
  static constexpr int E_LD = C + 1;
  static constexpr int O_BAR = F_END > L_END ? F_END : L_END;
  static constexpr int SMEM = O_BAR + 256;
  static_assert(SMEM <= 232448, "smem");
  static_assert(H * E_LD * 4 <= L_VF, "LN tail scratch");
  static constexpr uint32_t ID_DN = idesc_bf16_mj(128, H / 2, 0, 1);   // dnr half: A = dPr (K-major), B = Wp' (MN-major)
  static constexpr uint32_t ID_A = idesc_bf16_mj(128, H / 2, 1, 0);    // A half: A = dPr^T (MN-major), B = tri (K-major)
};
constexpr uint32_t T_DWG = 0, T_G = 128, T_Q = 256;                     // front TMEM; T_Q: two 128-column buffers
constexpr uint32_t ID_G = idesc_bf16_mj(128, 128, 0, 0);
constexpr uint32_t ID_Q = idesc_bf16_mj(128, 128, 1, 0);
constexpr uint32_t ID_DWGT = idesc_bf16_mj(128, 64, 1, 1);         // dWg^T block: A = xn^T (MN-major), B = dG block (MN-major)

struct FBars {
  uint64_t s_full[3], s_free[3], wg_full, dy_full[2], dwg_done[2], dg_ready[2], g_full, g_empty, q_full[2], q_empty[2], dp_ready, slot_ok,
      published, rel_done;
};
struct LBars {
  uint64_t tl_full[2], tl_free[2], dpb_full, dpb_mma, vs_done, s_free, a_done, dn_full[2], dn_empty[2], dtri_ready[2];
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
// sigmoid as 0.5 tanh(g / 2) + 0.5 (one MUFU op; as k3g's forward)
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
DEV uint32_t lds32(uint32_t a) { uint32_t v; asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }
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
DEV void bulk_store(void* gdst, uint32_t ssrc, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(gdst), "r"(ssrc), "r"(bytes) : "memory");
}
DEV void bulk_load(uint32_t sdst, const void* gsrc, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(sdst), "l"(gsrc), "r"(bytes),
               "r"(smem_u32(bar)) : "memory");
}
// SWIZZLE_NONE operand: 8-row x 16-byte core matrices; lbo = byte stride between core matrices along K, sbo = along M / N (for
// both K-major and MN-major operands, as measured here)
DEV uint64_t desc_nosw(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((saddr >> 4) & 0x3FFFu) | ((uint64_t)((lbo >> 4) & 0x3FFFu) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFFu) << 32) |
         ((uint64_t)1 << 46);
}
DEV void prefetch_l2(const CUtensorMap* m, int c0, int c1) {
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" ::"l"(reinterpret_cast<uint64_t>(m)), "r"(c0), "r"(c1) : "memory");
}

// ldmatrix / stmatrix fragment address of a [H ch][128 tok] tile (two 64-token SW128 halves): channel group jj (16 channels)
template <int H>
DEV uint32_t frag_addr(uint32_t tbase, int tok0, int jj, int lane) {
  const int mm = lane >> 3, ch = 16 * jj + 8 * (mm & 1) + (lane & 7), qc = ((tok0 & 63) >> 3) + (mm >> 1);
  return tbase + (tok0 >> 6) * (Geo<H>::TT / 2) + sw128(ch, qc);
}
// dnr pair (channels 2 t4, 2 t4 + 1 [+ 8 if e & 1], token tr [+ 8 if e & 2]) of channel group jj from the 16x256b.x4 load of
// columns 32 (jj / 2) .. 32 (jj / 2) + 31
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
  __nv_bfloat16* ring;      // [NF RSF 128][128] dPr tiles
  int L, tiles, NF, RSF;
};

template <int H>
__global__ void __launch_bounds__(512, 1)
    b1g_kernel(const __grid_constant__ CUtensorMap mxn, const __grid_constant__ CUtensorMap mtri, const __grid_constant__ CUtensorMap mwp,
               const __grid_constant__ CUtensorMap mwg, const __grid_constant__ CUtensorMap mdg, const __grid_constant__ CUtensorMap mdtri,
               const __grid_constant__ CUtensorMap mdy, const __grid_constant__ CUtensorMap mring, const __grid_constant__ CUtensorMap mtric,
               const Args a) {
  using G_ = Geo<H>;
  constexpr int TT = G_::TT, WPT = G_::WPT;
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
      for (int k = 0; k < 3; ++k) { mbar_init(&F.s_full[k], 1); mbar_init(&F.s_free[k], 1); }
      mbar_init(&F.wg_full, 1);
      for (int k = 0; k < 2; ++k) { mbar_init(&F.dy_full[k], 1); mbar_init(&F.dwg_done[k], 1); mbar_init(&F.dg_ready[k], 1); }
      mbar_init(&F.g_full, 1); mbar_init(&F.g_empty, 12); mbar_init(&F.dp_ready, 1);
      mbar_init(&F.slot_ok, 1); mbar_init(&F.published, 1); mbar_init(&F.rel_done, 1);
      for (int b = 0; b < 2; ++b) { mbar_init(&F.q_full[b], 1); mbar_init(&F.q_empty[b], 1); }
    } else {
      LBars& N = B.n;
      mbar_init(&N.dpb_full, 1); mbar_init(&N.dpb_mma, 1); mbar_init(&N.vs_done, 4); mbar_init(&N.s_free, 8); mbar_init(&N.a_done, 1);
      for (int b = 0; b < 2; ++b) {
        mbar_init(&N.tl_full[b], 1); mbar_init(&N.tl_free[b], 1); mbar_init(&N.dn_full[b], 1); mbar_init(&N.dn_empty[b], 8);
        mbar_init(&N.dtri_ready[b], 8);
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
  // ---- Wp' = bf16(Wp g_o) in place (4 threads per row o); front: u = rowsum(Wp'), c = Wp b_o
  mbar_wait(&B.wp_full, 0);
  {
    constexpr int CPT = H / 4;
    const int o = tid >> 2, pq = tid & 3;
    float u = 0.f, cb = 0.f;
#pragma unroll 2
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
    u += __shfl_xor_sync(0xffffffffu, u, 1); u += __shfl_xor_sync(0xffffffffu, u, 2);
    cb += __shfl_xor_sync(0xffffffffu, cb, 1); cb += __shfl_xor_sync(0xffffffffu, cb, 2);
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
    // ======================================================================= FRONT
    FBars& F = B.f;
    const int jc = cta % LC;
    auto tile_of = [&](int i) { return cta + NF * i; };
    int n_local = 0;
    while (tile_of(n_local) < a.tiles) ++n_local;
    // operand slots: tri's two channel halves (T0, T1) and xn (X) of tile i: T1_i -> slot i % 3, T0_i then X_i -> slot (i + 2) % 3;
    // per slot the uses run T1_i, T0_{i+1}, X_{i+1}, T1_{i+3}, ...: use u = i - (s == 1) (T0, T1), u + 1 (X)
    auto slot_t1 = [](int i, int& s, int& u) { s = i % 3; u = i - (s == 1); };
    auto slot_t0 = [](int i, int& s, int& u) { s = (i + 2) % 3; u = i - (s == 1); };
    const uint32_t sslot = su + G_::F_S;
    if (warp == 0) {
      if (lane == 0) {
        mbar_expect_tx(&F.wg_full, XT);
        tma_load_2d(sm + G_::F_WG, &mwg, &F.wg_full, 0, 0, EVICT_LAST);
        tma_load_2d(sm + G_::F_WG + XT / 2, &mwg, &F.wg_full, 64, 0, EVICT_LAST);
        auto load_t = [&](int i, int c, int s, int u) {
          const int row0 = tile_of(i) * TOK;
          if (u >= 1) PW(0, mbar_wait(&F.s_free[s], (u - 1) & 1));
          mbar_expect_tx(&F.s_full[s], TT / 2);
          tma_load_2d(sm + G_::F_S + s * G_::SLOT, &mtric, &F.s_full[s], row0, c * (H / 2), EVICT_NORMAL);
          tma_load_2d(sm + G_::F_S + s * G_::SLOT + TT / 4, &mtric, &F.s_full[s], row0 + 64, c * (H / 2), EVICT_NORMAL);
        };
        for (int i = 0; i < n_local; ++i) {
          const int row0 = tile_of(i) * TOK;
          int s0, u0, s1, u1;
          slot_t0(i, s0, u0);
          slot_t1(i, s1, u1);
          load_t(i, 0, s0, u0);                                         // T0_i: after Q_{i-1} has read T1_{i-1}
          load_t(i, 1, s1, u1);                                         // T1_i: after dWg_{i-2} has read X_{i-2}
          PW(1, mbar_wait(&F.s_free[s0], u0 & 1));                      // X_i: after Q_i has read T0_i
          mbar_expect_tx(&F.s_full[s0], XT);
          tma_load_2d(sm + G_::F_S + s0 * G_::SLOT, &mxn, &F.s_full[s0], 0, row0, EVICT_FIRST);
          tma_load_2d(sm + G_::F_S + s0 * G_::SLOT + XT / 2, &mxn, &F.s_full[s0], 64, row0, EVICT_FIRST);
          if (i + 1 < n_local) {                                        // the next tile's xn into L2
            const int nr0 = tile_of(i + 1) * TOK;
            prefetch_l2(&mxn, 0, nr0); prefetch_l2(&mxn, 64, nr0);
          }
        }
      }
    } else if (warp == 1) {
      if (lane == 0) {
        const uint32_t sdg = su + G_::F_DY, swg = su + G_::F_WG, swp = su;
        // Q_i [128 tok][128 o] = sum over the channel halves c of T_c^T (MN-major, token halves TT / 4 apart) . Wp'[:, half c]
        auto mma_q = [&](int i) {
          const int b = i & 1;
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            int s, u;
            if (c == 0) slot_t0(i, s, u); else slot_t1(i, s, u);
            PW(0, mbar_wait(&F.s_full[s], u & 1));
            if (c == 0 && i >= 2) PW(0, mbar_wait(&F.q_empty[b], ((i >> 1) - 1) & 1));
            tc_fence_after();
#pragma unroll
            for (int kk = 0; kk < H / 32; ++kk) {
              const int kg = c * (H / 32) + kk;
              umma_ss(tmem + T_Q + 128 * b, desc_mn_sw128(sslot + s * G_::SLOT + kk * 2048, TT / 4),
                      desc_k_sw128(swp + (kg >> 2) * KB + (kg & 3) * 32), ID_Q, kg != 0);
            }
            if (c == 1) umma_commit(&F.q_full[b]);
            umma_commit(&F.s_free[s]);
          }
        };
        // G_i [128 tok][128 o] = xn_i . Wg^T
        auto mma_g = [&](int i) {
          int s, u;
          slot_t0(i, s, u);
          const uint32_t sx = sslot + s * G_::SLOT;
          PW(1, mbar_wait(&F.s_full[s], (u + 1) & 1));
          if (i >= 1) PW(1, mbar_wait(&F.g_empty, (i - 1) & 1));
          tc_fence_after();
#pragma unroll
          for (int k = 0; k < 8; ++k)
            umma_ss(tmem + T_G, desc_k_sw128(sx + (k >> 2) * (XT / 2) + (k & 3) * 32), desc_k_sw128(swg + (k >> 2) * (XT / 2) + (k & 3) * 32),
                    ID_G, k != 0);
          umma_commit(&F.g_full);
        };
        if (n_local > 0) { mma_q(0); mbar_wait(&F.wg_full, 0); mma_g(0); }
        for (int i = 0; i < n_local; ++i) {
          if (i + 1 < n_local) { mma_q(i + 1); mma_g(i + 1); }
          int s, u;
          slot_t0(i, s, u);
          const uint32_t sx = sslot + s * G_::SLOT;
          // dWg^T [128 i][o block] += xn^T [i][tok] . dG [tok][o block], per 64-output block as the epilogue finishes it
#pragma unroll
          for (int blk = 0; blk < 2; ++blk) {
            PW(2, mbar_wait(&F.dg_ready[blk], i & 1));
            tc_fence_after();
#pragma unroll
            for (int k = 0; k < 8; ++k)
              umma_ss(tmem + T_DWG + 64 * blk, desc_mn_sw128(sx + k * 2048, XT / 2), desc_mn_sw128(sdg + blk * (XT / 2) + k * 2048, XT / 2),
                      ID_DWGT, (i | k) != 0);
            umma_commit(&F.dwg_done[blk]);
          }
          umma_commit(&F.s_free[s]);                                    // X_i
        }
        if (n_local > 0) mbar_wait(&F.dwg_done[1], (n_local - 1) & 1);
      }
    } else if (warp == 2) {
      if (lane == 0) {
        // flag publisher: a gpu-scope release per tile, off the store loop
        for (int i = 0; i < n_local; ++i) {
          const int slot = cta * a.RSF + i % a.RSF;
          PW(0, mbar_wait(&F.published, i & 1));
          PW(1, st_release(a.flags + slot, (uint32_t)(i / a.RSF + 1)));
          mbar_arrive(&F.rel_done);
        }
      }
    } else if (warp == 3) {
      if (lane == 0) {
        // dy loads per 64-column block; ring slot free (its previous tile consumed) -> the epilogue may store into it; dG -> global
        auto load_dy = [&](int i, int blk) {
          const int row0 = tile_of(i) * TOK;
          mbar_expect_tx(&F.dy_full[blk], XT / 2);
          tma_load_2d(sm + G_::F_DY + blk * (XT / 2), &mdy, &F.dy_full[blk], blk * 64, row0, EVICT_FIRST);
          if (i + 1 < n_local) prefetch_l2(&mdy, blk * 64, tile_of(i + 1) * TOK);
        };
        if (n_local > 0) { load_dy(0, 0); load_dy(0, 1); }
        for (int i = 0; i < n_local; ++i) {
          const int row0 = tile_of(i) * TOK, slot = cta * a.RSF + i % a.RSF;
          PW(1, spin_until(a.flags + slot, 0u));
          mbar_arrive(&F.slot_ok);
#pragma unroll 1
          for (int blk = 0; blk < 2; ++blk) {
            PW(0, mbar_wait(&F.dg_ready[blk], i & 1));
            tma_store_2d(&mdg, sm + G_::F_DY + blk * (XT / 2), blk * 64, row0);
            bulk_commit();
            PW(2, bulk_wait_read<0>());
            if (i + 1 < n_local) {
              PW(3, mbar_wait(&F.dwg_done[blk], i & 1));                // dWg^T has read this block of dG_i
              load_dy(i + 1, blk);
            }
          }
          PW(0, mbar_wait(&F.dp_ready, i & 1));
          if (i >= 1) PW(4, mbar_wait(&F.rel_done, (i - 1) & 1));
          mbar_arrive(&F.published);                                   // dPr / S of tile i (generic stores) -> warp 2 releases
        }
        bulk_wait<0>();
      }
    } else if (warp >= 4) {
      // 12 epilogue warps: lane quarter q (tokens 32 q ..); in each 64-output block, 8-column chunks c_lo .. c_hi - 1
      const int q = warp & 3, kq = (warp - 4) >> 2, r = q * 32 + lane;
      const uint32_t trow = tmem + ((uint32_t)(q * 32) << 16);
      const int c_lo = kq * 3, c_hi = kq == 2 ? 8 : kq * 3 + 3;
      const uint32_t suc = su + G_::F_UC;
      float* sb = reinterpret_cast<float*>(sm + G_::F_SB);
      uint4 dsv[2][3];
      int cur_sb = -1;
      auto load_ds = [&](int sbi) {
        const uint4* dsr = reinterpret_cast<const uint4*>(a.ds + ((size_t)sbi * a.L + jc * TOK + r) * C);
#pragma unroll
        for (int blk = 0; blk < 2; ++blk)
#pragma unroll
          for (int cc = 0; cc < 3; ++cc)
            if (c_lo + cc < c_hi) dsv[blk][cc] = __ldg(dsr + blk * 8 + c_lo + cc);
      };
      for (int i = 0; i < n_local; ++i) {
        const int row0 = tile_of(i) * TOK, b = i & 1;
        {
          const int sbi = tile_of(i) / (a.L * LC);
          if (sbi != cur_sb) { load_ds(sbi); cur_sb = sbi; }
        }
        const float mean = __ldg(a.mean_o + row0 + r), rs = __ldg(a.rs_o + row0 + r);
        // phase A: G -> bf16 registers, then G's TMEM is free for the next tile
        PW(0, mbar_wait(&F.g_full, i & 1));
        tc_fence_after();
        uint32_t gk[2][3][4];
#pragma unroll
        for (int blk = 0; blk < 2; ++blk)
#pragma unroll
          for (int cc = 0; cc < 3; ++cc)
            if (c_lo + cc < c_hi) {
              uint32_t v[8];
              tmem_ld8(trow + T_G + blk * 64 + (c_lo + cc) * 8, v);
              tmem_wait_ld();
#pragma unroll
              for (int k = 0; k < 4; ++k) gk[blk][cc][k] = pack_bf16(__uint_as_float(v[2 * k]), __uint_as_float(v[2 * k + 1]));
            }
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&F.g_empty);
        // phase B, per 64-output block: qc = Q - mean u, P = rs qc + c; dP, dG; dPr = bf16(dP rs) -> ring slot, dG in place of dy;
        // S1 += dPr qc, S2 += dPr u
        const int slot = cta * a.RSF + i % a.RSF;
        uint4* dpg = reinterpret_cast<uint4*>(a.ring) + (size_t)slot * (XT / 16) + r;   // + qq TOK: chunk qq of token r
        PW(1, mbar_wait(&F.slot_ok, i & 1));
        PW(2, mbar_wait(&F.q_full[b], (i >> 1) & 1));
        const float2 rs2 = f2(rs), nmean2 = f2(-mean);
        float2 s1 = f2(0.f), s2 = f2(0.f);
#pragma unroll
        for (int blk = 0; blk < 2; ++blk) {
          PW(1, mbar_wait(&F.dy_full[blk], i & 1));
          tc_fence_after();
#pragma unroll
          for (int cc = 0; cc < 3; ++cc) {
            const int c = c_lo + cc;
            if (c < c_hi) {
              const int qq = blk * 8 + c;
              uint32_t v[8];
              tmem_ld8(trow + T_Q + 128 * b + qq * 8, v);
              const uint32_t ya = su + G_::F_DY + blk * (XT / 2) + sw128(r, c);
              const uint4 dyw = lds128(ya), dsw = dsv[blk][cc];
              const float4 u0 = ldsf4(suc + 32 * qq), u1 = ldsf4(suc + 32 * qq + 16);
              const float4 c0 = ldsf4(suc + 4 * C + 32 * qq), c1 = ldsf4(suc + 4 * C + 32 * qq + 16);
              const float2 uu[4] = {make_float2(u0.x, u0.y), make_float2(u0.z, u0.w), make_float2(u1.x, u1.y), make_float2(u1.z, u1.w)};
              const float2 cv[4] = {make_float2(c0.x, c0.y), make_float2(c0.z, c0.w), make_float2(c1.x, c1.y), make_float2(c1.z, c1.w)};
              const uint32_t yv[4] = {dyw.x, dyw.y, dyw.z, dyw.w}, sv[4] = {dsw.x, dsw.y, dsw.z, dsw.w};
              tmem_wait_ld();
              uint32_t odp[4], odg[4];
#pragma unroll
              for (int k = 0; k < 4; ++k) {
                const float2 aa = __fmul2_rn(up(yv[k]), up(sv[k]));
                const float2 s = sig2(up(gk[blk][cc][k]));
                const float2 q2 = make_float2(__uint_as_float(v[2 * k]), __uint_as_float(v[2 * k + 1]));
                const float2 qc = __ffma2_rn(nmean2, uu[k], q2);
                const float2 p = __ffma2_rn(rs2, qc, cv[k]);
                odp[k] = pk2(__fmul2_rn(__fmul2_rn(aa, s), rs2));
                const float2 dpr = up(odp[k]);
                s1 = __ffma2_rn(dpr, qc, s1);
                s2 = __ffma2_rn(dpr, uu[k], s2);
                odg[k] = pk2(__fmul2_rn(__fmul2_rn(aa, p), __fmul2_rn(s, __fadd2_rn(f2(1.f), make_float2(-s.x, -s.y)))));
              }
              sts128(ya, odg[0], odg[1], odg[2], odg[3]);
              dpg[qq * TOK] = make_uint4(odp[0], odp[1], odp[2], odp[3]);
            }
          }
          tc_fence_before();
          fence_async_smem();
          named_bar_sync(1, 384);
          if (tid == 128) mbar_arrive(&F.dg_ready[blk]);
        }
        // (S1 | S2) of token r over all 128 outputs: the three column partials summed in a fixed order (kq 2, 1, 0) through sb
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
#ifndef B1G_NO_PROXY_FENCE
        fence_proxy_async_global();                                     // this thread's dPr / S stores before the async-proxy reads
#endif
        named_bar_sync(1, 384);
        if (tid == 128) { mbar_arrive(&F.q_empty[b]); mbar_arrive(&F.dp_ready); }
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
    {
      // dWg^T partial (TMEM lanes = i, columns = o): warp (q, cg) -> i = 32 q + lane, o = 32 cg ..; row o of part = dWg[o][:];
      // zero the dWp columns when this row has no LN CTA
      const int q = warp & 3, cg = warp >> 2, ii = q * 32 + lane;
      float* dst = a.part + (size_t)cta * C * (C + H);
      uint32_t v[32];
      tmem_ld32(tmem + ((uint32_t)(q * 32) << 16) + T_DWG + cg * 32, v);
      tmem_wait_ld();
#pragma unroll
      for (int k = 0; k < 32; ++k) dst[(size_t)(cg * 32 + k) * (C + H) + ii] = n_local > 0 ? __uint_as_float(v[k]) : 0.f;
      if (cta >= NL)
        for (int k = 0; k < H / 4; k += 4)
          *reinterpret_cast<float4*>(dst + (size_t)ii * (C + H) + C + cg * (H / 4) + k) = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  } else {
    // ======================================================================= LN BACKWARD
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
          tma_load_2d(sm + G_::L_T + b * TT, &mtri, &N.tl_full[b], row0, 0, EVICT_FIRST);
          tma_load_2d(sm + G_::L_T + b * TT + TT / 2, &mtri, &N.tl_full[b], row0 + 64, 0, EVICT_FIRST);
          PW(1, spin_until(a.flags + slot, (uint32_t)(fi / a.RSF + 1)));
          fence_proxy_async_global();
          if (j >= 1) {
            // tile j-1's products, v / s and LN warps have read dPr and (S1 | S2): its slot goes back
            PW(2, mbar_wait(&N.dpb_mma, (j - 1) & 1));
            PW(2, mbar_wait(&N.vs_done, (j - 1) & 1));
            PW(2, mbar_wait(&N.s_free, (j - 1) & 1));
            st_relaxed(a.flags + prev_slot, 0u);
          }
          mbar_expect_tx(&N.dpb_full, XT + SBB);
          bulk_load(su + G_::L_DP, a.ring + (size_t)slot * TOK * C, XT, &N.dpb_full);
          bulk_load(su + G_::L_SB, a.sring + (size_t)slot * 2 * TOK, SBB, &N.dpb_full);
          prev_slot = slot;
        }
        if (n_local > 0) {
          mbar_wait(&N.dpb_mma, (n_local - 1) & 1);
          mbar_wait(&N.vs_done, (n_local - 1) & 1);
          mbar_wait(&N.s_free, (n_local - 1) & 1);
          st_relaxed(a.flags + prev_slot, 0u);
        }
      }
    } else if (warp == 1) {
      if (lane == 0) {
        const uint32_t sdp = su + G_::L_DP, swp = su;
        for (int j = 0; j < n_local; ++j) {
          const int b = j & 1;
          const uint32_t st = su + G_::L_T + b * TT;
          PW(0, mbar_wait(&N.dpb_full, j & 1));
          PW(0, mbar_wait(&N.tl_full[b], (j >> 1) & 1));
          tc_fence_after();
          // A [128 o][H ch] += dPr^T [o][128 tok] . tri [ch][tok], as two N = H / 2 halves (it reads tri before the LN pass rewrites it)
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int k = 0; k < 8; ++k)
              umma_ss(tmem + h * (H / 2), desc_nosw(sdp + k * 256, 128, 2048),
                      desc_k_sw128(st + (k >> 2) * (TT / 2) + (k & 3) * 32 + h * (H / 2) * 128), G_::ID_A, (j | k) != 0);
          umma_commit(&N.a_done);
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            if (j >= 1) PW(1, mbar_wait(&N.dn_empty[h], (j - 1) & 1));
            tc_fence_after();
            // dnr_h [128 tok][H / 2 ch] = dPr [tok][128 o] . Wp' [o][ch half h]   (B N-major: H / 128 blocks of 64 channels, KB apart)
#pragma unroll
            for (int k = 0; k < 8; ++k)
              umma_ss(tmem + H + h * (H / 2), desc_nosw(sdp + k * 4096, 2048, 128),
                      desc_mn_sw128(swp + h * (H / 128) * KB + k * 2048, KB), G_::ID_DN, k != 0);
            umma_commit(&N.dn_full[h]);
          }
          umma_commit(&N.dpb_mma);
        }
        if (n_local > 0) mbar_wait(&N.dpb_mma, (n_local - 1) & 1);
      }
    } else if (warp == 3) {
      if (lane == 0) {
        // dTri storer
        for (int j = 0; j < n_local; ++j) {
          const int t = l + NL * j, b = j & 1, row0 = t * TOK;
          PW(0, mbar_wait(&N.dtri_ready[b], (j >> 1) & 1));
          tma_store_2d(&mdtri, sm + G_::L_T + b * TT, row0, 0);
          tma_store_2d(&mdtri, sm + G_::L_T + b * TT + TT / 2, row0 + 64, 0);
          bulk_commit();
          PW(1, bulk_wait_read<0>());
          mbar_arrive(&N.tl_free[b]);
        }
        bulk_wait<0>();
      }
    } else if (warp >= 4 && warp < 12) {
      // 8 warps: LN backward of the 16-token group (q, hh) in the fragment layout, one pass: dnr from TMEM, tri by ldmatrix
      const int q = warp & 3, hh = (warp - 4) >> 2, tr = lane >> 2, tok0 = q * 32 + hh * 16;
      const float* sbl = reinterpret_cast<const float*>(sm + G_::L_SB);
      constexpr int JH = H / 32;                           // 16-channel groups per half
      for (int j = 0; j < n_local; ++j) {
        const int t = l + NL * j, b = j & 1, row0 = t * TOK;
        const uint32_t tb = su + G_::L_T + b * TT;
        const float mA = __ldg(a.mean_o + row0 + tok0 + tr), rA = __ldg(a.rs_o + row0 + tok0 + tr);
        const float mB = __ldg(a.mean_o + row0 + tok0 + tr + 8), rB = __ldg(a.rs_o + row0 + tok0 + tr + 8);
        const float2 nmA = f2(-mA), nmB = f2(-mB);
        PW(0, mbar_wait(&N.dpb_full, j & 1));
        // dTri = dnr + (tri - mean) K1 + K0,  K1 = -rs^2 S1 / H,  K0 = -S2 / H
        const float2 k1A = f2(-rA * rA * sbl[tok0 + tr] * (1.f / H)), k0A = f2(-sbl[TOK + tok0 + tr] * (1.f / H));
        const float2 k1B = f2(-rB * rB * sbl[tok0 + tr + 8] * (1.f / H)), k0B = f2(-sbl[TOK + tok0 + tr + 8] * (1.f / H));
        PW(1, mbar_wait(&N.a_done, j & 1));                  // A has read tri: it may be overwritten
        __syncwarp();
        if (lane == 0) mbar_arrive(&N.s_free);
        PW(0, mbar_wait(&N.tl_full[b], (j >> 1) & 1));
#pragma unroll 1
        for (int h = 0; h < 2; ++h) {
          const uint32_t ta = tmem + ((uint32_t)tok0 << 16) + H + h * (H / 2);
          PW(2, mbar_wait(&N.dn_full[h], j & 1));
          tc_fence_after();
          uint32_t tv[16];
#pragma unroll 2
          for (int jj = 0; jj < JH; ++jj) {
            if ((jj & 1) == 0) { tmem_ld16x256_x4(ta + (jj >> 1) * 32, tv); tmem_wait_ld(); }
            const uint32_t ad = frag_addr<H>(tb, tok0, h * JH + jj, lane);
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
          __syncwarp();
          if (lane == 0) mbar_arrive(&N.dn_empty[h]);
        }
        fence_async_smem();
        __syncwarp();
        if (lane == 0) mbar_arrive(&N.dtri_ready[b]);
      }
    } else if (warp >= 12) {
      // 4 warps: v[o] = sum_tok dPr[tok][o] mean[tok], s[o] = sum_tok dPr[tok][o] / rs[tok] for o = 32 wv .. 32 wv + 31 (column chunks
      // 4 wv ..): lane = token (mod 32) while the tiles run, x[c 16 + 2 e + {v, s}] for chunk c, element e; one butterfly at the end
      const int wv = warp - 12;
      const uint32_t base = su + G_::L_DP + (4 * wv) * 2048;
      float x[64];
#pragma unroll
      for (int k = 0; k < 64; ++k) x[k] = 0.f;
      for (int j = 0; j < n_local; ++j) {
        const int row0 = (l + NL * j) * TOK;
        PW(0, mbar_wait(&N.dpb_full, j & 1));
#pragma unroll 1
        for (int m = 0; m < 4; ++m) {
          const int tok = lane + 32 * m;
          const float mv = __ldg(a.mean_o + row0 + tok), sd = rcp_ftz(__ldg(a.rs_o + row0 + tok));
#pragma unroll
          for (int c = 0; c < 4; ++c) {
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
        if (lane == 0) mbar_arrive(&N.vs_done);
      }
      // reduce-scatter over the 32 lanes (fixed order): lane ends with x[0] = v, x[1] = s of index 2 lane -> o = 32 wv + lane
#pragma unroll
      for (int st = 0, msk = 16, nh = 32; st < 5; ++st, msk >>= 1, nh >>= 1) {
        const bool hi = (lane & msk) != 0;
#pragma unroll
        for (int k = 0; k < 32; ++k) {
          if (k < nh) {
            const float send = hi ? x[k] : x[k + nh];
            const float keep = hi ? x[k + nh] : x[k];
            x[k] = keep + __shfl_xor_sync(0xffffffffu, send, msk);
          }
        }
      }
      float* vfo = reinterpret_cast<float*>(sm + G_::L_VF);
      vfo[32 * wv + lane] = x[0];
      vfo[C + 32 * wv + lane] = x[1];
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
    {
      const int q = warp & 3, cg = warp >> 2, o = q * 32 + lane;
      const float vo = vf[o], so = vf[C + o];
      float* dst = a.part + ((size_t)l * C + o) * (C + H) + C;
#pragma unroll 1
      for (int cb = 0; cb < H / 128; ++cb) {
        const int col = cg * (H / 4) + cb * 32;
        const uint4* wrow = reinterpret_cast<const uint4*>(a.wp + (size_t)o * H + col);
#pragma unroll 1
        for (int m = 0; m < 4; ++m) {
          uint32_t v[8];
          tmem_ld8(tmem + ((uint32_t)(q * 32) << 16) + col + 8 * m, v);
          const uint4 w = __ldg(wrow + m);
          const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
          const float4 ga = __ldg(reinterpret_cast<const float4*>(a.go + col + 8 * m)), gb = __ldg(reinterpret_cast<const float4*>(a.go + col + 8 * m + 4));
          const float4 ba = __ldg(reinterpret_cast<const float4*>(a.bo + col + 8 * m)), bb = __ldg(reinterpret_cast<const float4*>(a.bo + col + 8 * m + 4));
          const float gv[8] = {ga.x, ga.y, ga.z, ga.w, gb.x, gb.y, gb.z, gb.w}, bv[8] = {ba.x, ba.y, ba.z, ba.w, bb.x, bb.y, bb.z, bb.w};
          float outv[8];
          tmem_wait_ld();
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            const float d = n_local > 0 ? __uint_as_float(v[k]) - vo : 0.f;
            const float wpv = (k & 1) ? up(wv[k >> 1]).y : up(wv[k >> 1]).x;
            outv[k] = fmaf(gv[k], d, bv[k] * so);
            E[(col + 8 * m + k) * G_::E_LD + o] = wpv * d;
          }
          *reinterpret_cast<float4*>(dst + col + 8 * m) = make_float4(outv[0], outv[1], outv[2], outv[3]);
          *reinterpret_cast<float4*>(dst + col + 8 * m + 4) = make_float4(outv[4], outv[5], outv[6], outv[7]);
        }
      }
      if (l >= NF) {
        float* z = a.part + ((size_t)l * C + o) * (C + H) + cg * 32;
        for (int k = 0; k < 32; k += 4) *reinterpret_cast<float4*>(z + k) = make_float4(0.f, 0.f, 0.f, 0.f);
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

}  // namespace b1g

void b1g_backward(torch::Tensor dy, torch::Tensor xn, torch::Tensor tri, torch::Tensor ds, torch::Tensor mean_o, torch::Tensor rs_o,
                  torch::Tensor wg, torch::Tensor wp, torch::Tensor go, torch::Tensor bo, torch::Tensor dg, torch::Tensor dtri,
                  torch::Tensor part, torch::Tensor lnpart, torch::Tensor ring, torch::Tensor flags, int64_t L, int64_t nf) {
  using namespace b1g;
  const int64_t M = dy.size(0);
  const int H = (int)tri.size(0);
  TORCH_CHECK(dy.size(1) == C && M % (L * L) == 0 && M % TOK == 0 && L % TOK == 0 && tri.numel() == H * M && (H == 128 || H == 256));
  TORCH_CHECK(ds.numel() == (M / (L * L)) * L * C, "b1g_backward: ds must be [B, L, 128]");   // M = B L L, tokens b-major
  TORCH_CHECK(wp.is_contiguous() && wp.numel() == (int64_t)C * H && go.numel() == H && bo.numel() == H);
  int nsm = 0, dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  // ring: RS dPr tiles of 128 rows, then RS (S1 | S2) blocks of 1 KB (4 rows of 128 bf16)
  const int RS = (int)(ring.size(0) / (TOK + SBB / (C * 2))), RSF = RS / (int)nf;
  TORCH_CHECK(nf > 0 && nf < nsm && flags.numel() >= RS && RSF >= 1 && nf % (L / TOK) == 0 && ring.is_contiguous() &&
              ring.size(0) == (int64_t)RS * (TOK + SBB / (C * 2)), "b1g_backward: ring [RS (128 + 4), 128]");
  auto bf = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  auto tok_map = [&](const torch::Tensor& t, int64_t rows) {
    return tmap::make(t.data_ptr(), bf, {(uint64_t)C, (uint64_t)rows}, {(uint64_t)C * 2}, {64, TOK}, CU_TENSOR_MAP_SWIZZLE_128B);
  };
  auto ch_map = [&](const torch::Tensor& t) {
    return tmap::make(t.data_ptr(), bf, {(uint64_t)M, (uint64_t)H}, {(uint64_t)M * 2}, {64, (uint32_t)H}, CU_TENSOR_MAP_SWIZZLE_128B);
  };
  auto mxn = tok_map(xn, M), mdg = tok_map(dg, M), mdy = tok_map(dy, M), mring = tok_map(ring, (int64_t)RS * TOK);
  auto mtri = ch_map(tri), mdtri = ch_map(dtri);
  auto mtric = tmap::make(tri.data_ptr(), bf, {(uint64_t)M, (uint64_t)H}, {(uint64_t)M * 2}, {64, (uint32_t)(H / 2)}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mwp = tmap::make(wp.data_ptr(), bf, {(uint64_t)H, (uint64_t)C}, {(uint64_t)H * 2}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mwg = tmap::make(wg.data_ptr(), bf, {(uint64_t)C, (uint64_t)C}, {(uint64_t)C * 2}, {64, 128}, CU_TENSOR_MAP_SWIZZLE_128B);
  Args a;
  a.ds = reinterpret_cast<const __nv_bfloat16*>(ds.data_ptr());
  a.wp = reinterpret_cast<const __nv_bfloat16*>(wp.data_ptr());
  a.mean_o = mean_o.data_ptr<float>(); a.rs_o = rs_o.data_ptr<float>(); a.go = go.data_ptr<float>(); a.bo = bo.data_ptr<float>();
  const int64_t nl = nsm - nf, rows = nf > nl ? nf : nl;
  TORCH_CHECK(lnpart.numel() >= nl * 2 * H);
  TORCH_CHECK(part.numel() >= rows * C * (C + H) && part.scalar_type() == torch::kFloat32, "b1g_backward: part holds max(nf, nsm - nf) rows");
  a.part = part.data_ptr<float>(); a.lnpart = lnpart.data_ptr<float>();
  a.flags = reinterpret_cast<uint32_t*>(flags.data_ptr<int32_t>());
  a.ring = reinterpret_cast<__nv_bfloat16*>(ring.data_ptr());
  a.sring = reinterpret_cast<float*>(reinterpret_cast<__nv_bfloat16*>(ring.data_ptr()) + (size_t)RS * TOK * C);
  a.L = (int)L; a.tiles = (int)(M / TOK); a.NF = (int)nf; a.RSF = RSF;
  void* params[] = {(void*)&mxn, (void*)&mtri, (void*)&mwp, (void*)&mwg, (void*)&mdg, (void*)&mdtri, (void*)&mdy, (void*)&mring, (void*)&mtric, (void*)&a};
  auto launch = [&](const void* fn, int smem, bool& attr) {
    if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); attr = true; }
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(fn, dim3(nsm), dim3(512), params, smem, at::cuda::getCurrentCUDAStream()));
  };
  static bool attr128 = false, attr256 = false;
  if (H == 128) launch((const void*)b1g_kernel<128>, Geo<128>::SMEM, attr128);
  else launch((const void*)b1g_kernel<256>, Geo<256>::SMEM, attr256);
}
