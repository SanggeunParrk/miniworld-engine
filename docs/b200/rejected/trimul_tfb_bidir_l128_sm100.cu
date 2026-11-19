// tfb_sm100.cu -- the BIDIRECTIONAL D64 TriMul inference update at L = 128 in ONE launch (B200, sm_100a only): pair 64, hidden 64
// per direction (planes a_out | b_out | a_in | b_in, t = [t_out | t_in] over 128 channels, LN_out over 128, Wp [64][128]).
//
// The one-direction kernel tf_sm100.cu (docs/b200/trimul.md, round 3 passes 9-10) with both directions in the same grid and the same
// CTA token tiles: 128 CTAs = 32 clusters of 4; a cluster owns a 16 x 32 block of output pair tokens (i in I0.., j in J0..), CTA rank r
// the 8 x 16 sub-block i0 = I0 + 8 (r >> 1), j0 = J0 + 16 (r & 1) (its x tile, token row q = il * 16 + jl) and contraction channels
// 16 r .. 16 r + 15 of BOTH directions.
//   front: x (coalesced ld.global at entry), LN_in -> xn (TMEM), g = xn Wg^T (TMEM to the end), the four gated projections a_out,
//          b_out, a_in, b_in (tcgen05 N 128 each: gate 64 | projection 64) in three TMEM slots (a_in reuses slot 0 after a_out's
//          epilogue loaded it); each written with st.global straight from registers into BLOCKED channel-major planes in the
//          contraction's K-major orientation (outgoing planes as is: 8 rows x 16 k blocks; incoming planes transposed: 16 rows x 8 k
//          blocks), so both directions contract T_c = A_c B_c^T. Two release counters: outgoing planes per 32-row i block (released
//          after a_out / b_out), incoming per 32-column j block (after a_in / b_in).
//   contraction: warp w = 8 + s owns channel pair s of both directions and operand slot s (24 KB: A 2 ch x 16 rows | B 2 ch x 32 rows);
//          pass 1 outgoing (loads issued by warp 0 lanes 0-7 as soon as the outgoing counters pass; slots 4-7 lie beyond the front's
//          weights / x tile, slots 0-3 wait for the front), pass 2 incoming into the same slot (issued by the warp itself);
//          mma.sync m16n8k16, fp32 accumulators, one bf16 rounding. The two results go to the owning CTA's t tile as ONE 1 KB DSMEM
//          bulk copy a destination (t rows interleaved: row 2 q + dir is channel pair q of direction dir).
//   output: LN_out over 128 channels (thread = token row), p = xo Wp^T (tcgen05, K 128; Wp loaded into slot 0's consumed B part),
//          y = x + bf16(sigmoid(g) p) (x re-read from global: TMEM holds no copy here).
#include "sm100.cuh"
#include "tmap.h"

using namespace sm100;

#ifdef TF_TL
static __device__ unsigned long long g_tftl[128][16];
#define TFTL(k) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); g_tftl[blockIdx.x][k] = t_; }
#else
#define TFTL(k) {}
#endif

namespace tfb {
constexpr int C = 64, H = 64, HT = 128, TOK = 128, L = 128;
constexpr int NCTA = 128, CL = 4, CPC = 16;
constexpr int WB = 64 * 64 * 2;                    // one [64][64] bf16 weight box (SW128) = 8 KB
// shared memory. Front: W (9 boxes: wlg0 wl0 | wrg0 wr0 | wlg1 wl1 | wrg1 wr1 | wg) | x tile, inside operand slots 0-3.
// Operand slots s = 0..7 (24 KB: A [2 ch][16 blocks of 256 B] | B [2 ch][32 blocks]); Wp late into slot 0's B part.
constexpr int SLOT = 24576, SLOT_B = 8192;
constexpr int O_W = 0;
constexpr int O_X = O_W + 9 * WB;                   // 72 KB: x tile [128][64] SW128 (16 KB)
constexpr int O_WP = SLOT_B;                        // slot 0's B part (after its pass 2)
constexpr int O_T = 8 * SLOT;                       // t tile [64 rows of channel pairs][128 tok] x 4 B (32 KB)
constexpr int O_GB = O_T + TOK * HT * 2;            // g_in | b_in (64 each) | g_out | b_out (128 each), fp32
constexpr int O_BAR = O_GB + (64 + 64 + 128 + 128) * 4;
constexpr int SMEM = O_BAR + 256 + 1024;            // + alignment slack
static_assert(SMEM <= 232448 && O_X + TOK * C * 2 <= 4 * SLOT, "smem");
// TMEM columns
constexpr uint32_t T_S0 = 0, T_S1 = 128, T_S2 = 256, T_G = 384, T_XN = 448, T_XNU = 480, T_XO = 128, T_P = 192;

struct Maps {
  CUtensorMap w[6];   // wlg, wl, wrg, wr ([128][64]), wg [64][64], wp [64][128]: box 64 x 64 SW128
  CUtensorMap pld;    // planes, A box {128 el, 16 blocks, 2 ch}
  CUtensorMap pldb;   // B box {128 el, 32 blocks, 2 ch}
};

struct Bars {
  uint64_t w_full, x_free, xn_full, acc_full[4], s0_free, ab_full[8], t_full, xo_ready, p_full, wp_full;
  uint32_t tmem;
  uint32_t base;
};

// plane-block counters, monotonic: every launch adds exactly 32 to each (cnt[0..3] outgoing i blocks, cnt[4..7] incoming j blocks);
// a launch's base is read at entry before any increment of its own
__device__ unsigned g_cnt[8];

DEV float tanh_approx(float x) { float y; asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float rsqrt_ftz(float x) { float y; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float2 sigmoid2(float2 g) {
  const float2 h = __fmul2_rn(g, make_float2(0.5f, 0.5f));
  return __ffma2_rn(make_float2(tanh_approx(h.x), tanh_approx(h.y)), make_float2(0.5f, 0.5f), make_float2(0.5f, 0.5f));
}
DEV uint32_t sw128(uint32_t r, uint32_t q) { return r * 128u + ((q ^ (r & 7u)) << 4); }
DEV uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory");
  return v;
}
DEV float4 lds128f(uint32_t a) {
  float4 v;
  asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a));
  return v;
}
DEV void tmem_st8(uint32_t taddr, const uint32_t (&r)[8]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
               :: "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]) : "memory");
}
DEV void ldsm_x4(uint32_t a, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
DEV void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
DEV void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
DEV unsigned ld_acquire(const unsigned* p) {
  unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
DEV void wait_cnt(const unsigned* a, const unsigned* b, unsigned tgt) {
  while ((int)(ld_acquire(a) - tgt) < 0) {}
  while ((int)(ld_acquire(b) - tgt) < 0) {}
  fence_proxy_async_global();
}

// byte offset of row r, 8-element k chunk kc inside a channel's blocked operand (OUTL 1: outgoing 8 rows x 16 k blocks with the
// 16-byte halves swapped when (row >> 2) is odd; 0: incoming 16 rows x 8 k blocks)
template <int OUTL>
DEV uint32_t boff_of(int r, int kc) {
  return OUTL ? (uint32_t)(((r >> 3) * 8 + (kc >> 1)) * 256 + (r & 7) * 32 + (((kc & 1) ^ ((r >> 2) & 1)) << 4))
              : (uint32_t)(((r >> 4) * 16 + kc) * 256 + (r & 15) * 16);
}
// T_c = A_c B_c^T on the 16 x 32 block for the warp's two channels (A: 2 x 4 KB at sA, B: 2 x 8 KB at sB)
template <int OUTL>
DEV void contract(uint32_t sA, uint32_t sB, int lane, float (&acc)[2][4][4]) {
#pragma unroll
  for (int c2 = 0; c2 < 2; ++c2)
#pragma unroll
    for (int n = 0; n < 4; ++n) acc[c2][n][0] = acc[c2][n][1] = acc[c2][n][2] = acc[c2][n][3] = 0.f;
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    const uint32_t aoff = boff_of<OUTL>(lane & 15, 2 * s + (lane >> 4));
    const uint32_t bo0 = boff_of<OUTL>(8 * (lane >> 4) + (lane & 7), 2 * s + ((lane >> 3) & 1));
    const uint32_t bo1 = boff_of<OUTL>(16 + 8 * (lane >> 4) + (lane & 7), 2 * s + ((lane >> 3) & 1));
#pragma unroll
    for (int c2 = 0; c2 < 2; ++c2) {
      uint32_t a0, a1, a2, a3, b0, b1, b2, b3, b4, b5, b6, b7;
      ldsm_x4(sA + c2 * 4096 + aoff, a0, a1, a2, a3);
      ldsm_x4(sB + c2 * 8192 + bo0, b0, b1, b2, b3);
      ldsm_x4(sB + c2 * 8192 + bo1, b4, b5, b6, b7);
      mma16816(acc[c2][0], a0, a1, a2, a3, b0, b1);
      mma16816(acc[c2][1], a0, a1, a2, a3, b2, b3);
      mma16816(acc[c2][2], a0, a1, a2, a3, b4, b5);
      mma16816(acc[c2][3], a0, a1, a2, a3, b6, b7);
    }
  }
}
// front epilogue of one gated projection group for token row r (gate in TMEM columns tq + 0..63, projection tq + 64..127), halves
// h0 .. h1 - 1 of 32 channels; gpl: this token's byte address in the group's channel 0 plane. Two tokens adjacent in the plane block
// (outgoing: rows r, r ^ 1; incoming: r, r ^ 16) are lanes l, l ^ PX: the lower one stores channel c of both, the upper c + 1.
template <int PX>
DEV void plane_epi(uint32_t tq, int h0, int h1, uint8_t* gpl, int lane, uint64_t* freed) {
  const bool odd = lane & PX;
#pragma unroll 1
  for (int h = h0; h < h1; ++h) {
    uint32_t g[32], p[32];
    tmem_ld32(tq + h * 32, g);
    tmem_ld32(tq + 64 + h * 32, p);
    tmem_wait_ld();
    if (freed && h == h1 - 1) {   // the slot's columns are read: the next group's MMA may overwrite them
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(freed);
    }
#pragma unroll
    for (int c = 0; c < 32; c += 2) {
      const float2 s = sigmoid2(make_float2(__uint_as_float(g[c]), __uint_as_float(g[c + 1])));
      const float2 v = __fmul2_rn(s, make_float2(__uint_as_float(p[c]), __uint_as_float(p[c + 1])));
      const uint32_t w = pack_bf16(v.x, v.y), wo = __shfl_xor_sync(0xffffffffu, w, PX);
      const uint32_t word = odd ? __byte_perm(wo, w, 0x7632) : __byte_perm(w, wo, 0x5410);
      *reinterpret_cast<uint32_t*>(gpl - (odd ? 2 : 0) + (size_t)(h * 32 + c + (odd ? 1 : 0)) * (L * L * 2)) = word;
    }
  }
}

template <int MASK>
__global__ void __launch_bounds__(512, 1)
    tfb_kernel(const __grid_constant__ Maps m, const uint8_t* __restrict__ tokmask, const float* __restrict__ g_in,
               const float* __restrict__ b_in, const float* __restrict__ g_out, const float* __restrict__ b_out, float eps,
               __nv_bfloat16* __restrict__ planes, __nv_bfloat16* __restrict__ yg, const __nv_bfloat16* __restrict__ xg) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rank = (int)cluster_ctarank(), cid = blockIdx.x / CL;
  const int I0 = 16 * (cid >> 2), J0 = 32 * (cid & 3);
  const int i0 = I0 + 8 * (rank >> 1), j0 = J0 + 16 * (rank & 1);
  // token row r = il * 16 + jl of the sub-block
  auto tok_of = [&](int r) -> size_t { return (size_t)(i0 + (r >> 4)) * L + j0 + (r & 15); };
  // the x tile by all 512 threads, coalesced (32 B a thread, token row tid / 4), first of all
  uint4 xw0, xw1;
  {
    const uint4* xp = reinterpret_cast<const uint4*>(xg + tok_of(tid >> 2) * C) + (tid & 3) * 2;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(xw0.x), "=r"(xw0.y), "=r"(xw0.z), "=r"(xw0.w) : "l"(xp));
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(xw1.x), "=r"(xw1.y), "=r"(xw1.z), "=r"(xw1.w) : "l"(xp + 1));
  }
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  if (tid == 0) TFTL(0)

  if (tid == 0) {
    mbar_init(&B.w_full, 9); mbar_init(&B.x_free, 1); mbar_init(&B.xn_full, 1);
    for (int q = 0; q < 4; ++q) mbar_init(&B.acc_full[q], 1);
    mbar_init(&B.s0_free, 4);
    for (int q = 0; q < 8; ++q) mbar_init(&B.ab_full[q], 1);
    mbar_init(&B.t_full, 1); mbar_init(&B.xo_ready, 1); mbar_init(&B.p_full, 1); mbar_init(&B.wp_full, 1);
    mbar_expect_tx(&B.t_full, TOK * HT * 2);   // the cluster's 4 CTAs complete 32 KB of bulk copies on it
    fence_mbar_init();
  }
  __syncwarp();
  if (warp == 0 && lane < 9) {
    // the front weights at entry, one box a lane: lane l < 8: matrix l & 3 (wlg, wl, wrg, wr), rows 64 (l >> 2); lane 8: wg
    const CUtensorMap* mp = &m.w[lane < 8 ? (lane & 3) : 4];
    prefetch_tmap(mp);
    mbar_expect_tx(&B.w_full, WB);
    tma_load_2d(sm + O_W + lane * WB, mp, &B.w_full, 0, lane < 8 ? 64 * (lane >> 2) : 0, EVICT_LAST);
  }
  if (warp == 3 && lane == 0) { prefetch_tmap(&m.pld); prefetch_tmap(&m.pldb); prefetch_tmap(&m.w[5]); }
  if (tid < 64) {
    float* gb = reinterpret_cast<float*>(sm + O_GB);
    gb[tid] = g_in[tid]; gb[64 + tid] = b_in[tid];
  }
  if (tid < 128) {
    float* gb = reinterpret_cast<float*>(sm + O_GB);
    gb[128 + tid] = g_out[tid]; gb[256 + tid] = b_out[tid];
  }
  if (warp == 2) { tmem_alloc(&B.tmem, 512); tmem_relinquish(); }
  if (warp == 3 && lane == 1) B.base = ld_acquire(&g_cnt[0]) & ~31u;
  sts128(su + O_X + sw128(tid >> 2, (tid & 3) * 2), xw0.x, xw0.y, xw0.z, xw0.w);
  sts128(su + O_X + sw128(tid >> 2, (tid & 3) * 2 + 1), xw1.x, xw1.y, xw1.z, xw1.w);
  tc_fence_before();
  cluster_arrive();   // every barrier of the cluster initialised before any peer's bulk copy
  cluster_wait();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  const unsigned tgt = B.base + 32u;
  // this CTA's plane blocks: outgoing (rows i, k j: 8 x 16) and incoming (rows j, k i: 16 x 8), 256 B each a channel
  const int blk_o = (i0 >> 3) * 8 + (j0 >> 4), blk_i = (j0 >> 4) * 16 + (i0 >> 3);

  if (warp == 0) {
    if (lane < 8) {
      // pass-1 (outgoing) operands of slot `lane`; slots 0-3 overlap the front's weights / x tile
      if (lane < 4) { mbar_wait(&B.acc_full[3], 0); mbar_wait(&B.x_free, 0); }
      wait_cnt(&g_cnt[I0 >> 5], &g_cnt[J0 >> 5], tgt);
      if (lane == 0) TFTL(2)
      const int ch = CPC * rank + 2 * lane;
      mbar_expect_tx(&B.ab_full[lane], 2 * (4096 + 8192));
      tma_load_3d(sm + lane * SLOT, &m.pld, &B.ab_full[lane], 0, I0, ch, EVICT_FIRST);
      tma_load_3d(sm + lane * SLOT + SLOT_B, &m.pldb, &B.ab_full[lane], 0, J0, 64 + ch, EVICT_FIRST);
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------ MMA (front)
    if (elect_one()) {
      constexpr uint32_t ID128 = idesc_bf16(128, 128), ID64 = idesc_bf16(128, 64);
      mbar_wait(&B.w_full, 0);
      mbar_wait(&B.xn_full, 0);
      tc_fence_after();
#pragma unroll
      for (int k = 0; k < 4; ++k)
        umma_ts(tmem + T_G, tmem + (MASK ? T_XNU : T_XN) + k * 8, desc_k_sw128(su + O_W + 8 * WB + k * 32), ID64, k);
#pragma unroll
      for (int g = 0; g < 3; ++g) {
#pragma unroll
        for (int k = 0; k < 4; ++k)
          umma_ts(tmem + (uint32_t)g * 128u, tmem + T_XN + k * 8, desc_k_sw128(su + O_W + 2 * g * WB + k * 32), ID128, k);
        umma_commit(&B.acc_full[g]);
      }
      mbar_wait(&B.s0_free, 0);
      tc_fence_after();
#pragma unroll
      for (int k = 0; k < 4; ++k) umma_ts(tmem + T_S0, tmem + T_XN + k * 8, desc_k_sw128(su + O_W + 6 * WB + k * 32), ID128, k);
      umma_commit(&B.acc_full[3]);   // every front product done: the operand loads may overwrite every front weight
    }
    __syncwarp();
  } else if (warp >= 4 && warp < 8) {
    // ------------------------------------------------------------------ LN_in, thread = token row r
    const int r = (warp & 3) * 32 + lane;
    const uint32_t trow = tmem + ((uint32_t)((warp & 3) * 32) << 16);
    auto up = [](uint32_t w) { return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u)); };
    const uint32_t xr = su + O_X;
    {
      uint32_t kmask = 0xffffffffu;
      if (MASK) kmask = (tokmask[i0 + (r >> 4)] && tokmask[j0 + (r & 15)]) ? 0xffffffffu : 0u;
      if (r == 0) TFTL(1)
      float2 a0 = make_float2(0.f, 0.f), a1 = a0, a2 = a0, a3 = a0;
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const uint4 t = lds128(xr + sw128(r, q));
        a0 = __fadd2_rn(a0, up(t.x)); a1 = __fadd2_rn(a1, up(t.y)); a2 = __fadd2_rn(a2, up(t.z)); a3 = __fadd2_rn(a3, up(t.w));
      }
      a0 = __fadd2_rn(__fadd2_rn(a0, a1), __fadd2_rn(a2, a3));
      const float mu = __fmul_rn(__fadd_rn(a0.x, a0.y), 1.f / C);
      const float2 nm0 = make_float2(-mu, -mu);
      a0 = a1 = a2 = a3 = make_float2(0.f, 0.f);
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const uint4 t = lds128(xr + sw128(r, q));
        float2 d;
        d = __fadd2_rn(up(t.x), nm0); a0 = __ffma2_rn(d, d, a0);
        d = __fadd2_rn(up(t.y), nm0); a1 = __ffma2_rn(d, d, a1);
        d = __fadd2_rn(up(t.z), nm0); a2 = __ffma2_rn(d, d, a2);
        d = __fadd2_rn(up(t.w), nm0); a3 = __ffma2_rn(d, d, a3);
      }
      a0 = __fadd2_rn(__fadd2_rn(a0, a1), __fadd2_rn(a2, a3));
      const float rs = rsqrt_ftz(__fadd_rn(__fmul_rn(__fadd_rn(a0.x, a0.y), 1.f / C), eps));
      const float2 nm = make_float2(-mu, -mu), rr = make_float2(rs, rs);
#pragma unroll
      for (int q4 = 0; q4 < 4; ++q4) {
        uint32_t o[8];
#pragma unroll
        for (int qq = 0; qq < 2; ++qq) {
          const int q = q4 * 2 + qq;
          const uint4 t = lds128(xr + sw128(r, q));
          const uint32_t gq = su + O_GB + q * 32;
          const float4 ga = lds128f(gq), gb4 = lds128f(gq + 16), ba = lds128f(gq + 256), bb = lds128f(gq + 256 + 16);
          auto aff2 = [&](uint32_t w, float g0, float g1, float b0, float b1) {
            const float2 y = __ffma2_rn(__fmul2_rn(__fadd2_rn(up(w), nm), rr), make_float2(g0, g1), make_float2(b0, b1));
            return pack_bf16(y.x, y.y);
          };
          o[4 * qq + 0] = aff2(t.x, ga.x, ga.y, ba.x, ba.y);
          o[4 * qq + 1] = aff2(t.y, ga.z, ga.w, ba.z, ba.w);
          o[4 * qq + 2] = aff2(t.z, gb4.x, gb4.y, bb.x, bb.y);
          o[4 * qq + 3] = aff2(t.w, gb4.z, gb4.w, bb.z, bb.w);
        }
        if (MASK) {
          tmem_st8(trow + T_XNU + q4 * 8, o);
#pragma unroll
          for (int e = 0; e < 8; ++e) o[e] &= kmask;
        }
        tmem_st8(trow + T_XN + q4 * 8, o);
      }
      tmem_wait_st();
      tc_fence_before();
      named_bar_sync(1, 128);
      if (r == 0) { mbar_arrive(&B.xn_full); mbar_arrive(&B.x_free); TFTL(7) }
    }
    // front epilogue of a_in (slot 2)
    mbar_wait(&B.acc_full[2], 0);
    tc_fence_after();
    plane_epi<16>(tmem + T_S2 + ((uint32_t)((warp & 3) * 32) << 16), 0, 2,
                  reinterpret_cast<uint8_t*>(planes) + ((size_t)128 * (L * L) + (size_t)blk_i * 128 + (r & 15) * 8 + (r >> 4)) * 2,
                  lane, nullptr);
    named_bar_sync(3, 384);
  } else if (warp >= 8) {
    // ------------------------------------------------------------------ front epilogue: warps 8-11 a_out (slot 0), 12-15 b_out (slot 1);
    // then b_in (slot 0 again) halves 0 / 1
    const int side = (warp - 8) >> 2;
    const int r = (warp & 3) * 32 + lane;
    const uint32_t lrow = (uint32_t)((warp & 3) * 32) << 16;
    const uint32_t offo = (uint32_t)((r >> 4) * 32 + ((((r >> 3) & 1) ^ ((r >> 6) & 1)) << 4) + (r & 7) * 2);
    uint8_t* pl = reinterpret_cast<uint8_t*>(planes);
    mbar_wait(&B.acc_full[side], 0);
    tc_fence_after();
    if (warp == 8 && lane == 0) TFTL(8)
    plane_epi<1>(tmem + (side ? T_S1 : T_S0) + lrow, 0, 2, pl + ((size_t)(side * 64) * (L * L) + (size_t)blk_o * 128) * 2 + offo, lane,
                 side ? nullptr : &B.s0_free);
    named_bar_sync(2, 256);
    if (warp == 8 && lane == 0) {
      __threadfence();   // one thread's gpu-scope fence after the barrier (cumulative)
      asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(&g_cnt[i0 >> 5]) : "memory");
      TFTL(9)
    }
    mbar_wait(&B.acc_full[3], 0);
    tc_fence_after();
    plane_epi<16>(tmem + T_S0 + lrow, side, side + 1,
                  pl + ((size_t)192 * (L * L) + (size_t)blk_i * 128 + (r & 15) * 8 + (r >> 4)) * 2, lane, nullptr);
    named_bar_sync(3, 384);
    if (warp == 8 && lane == 0) {
      __threadfence();
      asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(&g_cnt[4 + (j0 >> 5)]) : "memory");
      TFTL(6)
    }
  }
  // ------------------------------------------------------------------ contraction, warps 8-15: warp = channel pair cp of both directions
  if (warp >= 8) {
    const int cp = warp - 8;
    const uint32_t sA = su + cp * SLOT, sB = sA + SLOT_B;
    float acc[2][4][4];
    uint32_t pk[4][2][2];   // the outgoing pair, bf16-packed (channel 2 cp low, 2 cp + 1 high)
    mbar_wait(&B.ab_full[cp], 0);
    if (warp == 8 && lane == 0) TFTL(3)
    contract<1>(sA, sB, lane, acc);
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int k = 0; k < 4; ++k) pk[n][k >> 1][k & 1] = pack_bf16(acc[0][n][k], acc[1][n][k]);
    // pass 2: the incoming operands into the same slot
    fence_async_smem();
    __syncwarp();
    if (lane == 0) {
      wait_cnt(&g_cnt[4 + (I0 >> 5)], &g_cnt[4 + (J0 >> 5)], tgt);
      const int ch = CPC * rank + 2 * cp;
      mbar_expect_tx(&B.ab_full[cp], 2 * (4096 + 8192));
      tma_load_3d(sm + cp * SLOT, &m.pld, &B.ab_full[cp], 0, I0, 128 + ch, EVICT_FIRST);
      tma_load_3d(sm + cp * SLOT + SLOT_B, &m.pldb, &B.ab_full[cp], 0, J0, 192 + ch, EVICT_FIRST);
    }
    mbar_wait(&B.ab_full[cp], 1);
    if (warp == 8 && lane == 0) TFTL(15)
    contract<0>(sA, sB, lane, acc);
    if (warp == 8 && lane == 0) TFTL(10)
    // t exchange: per destination CTA [dir 2][128 tokens] x 4 B (1 KB) staged in the consumed A part, one DSMEM bulk copy each into
    // t rows 2 q, 2 q + 1 (q = 8 rank + cp)
    const uint32_t stw = sA;
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int il = (lane >> 2) + 8 * h, jl = 8 * n + 2 * (lane & 3) + e, k = 2 * h + e;
          const int dst = (il >> 3) * 2 + (jl >> 4);
          const int row = (il & 7) * 16 + (jl & 15);
          asm volatile("st.shared.b32 [%0], %1;" ::"r"(stw + dst * 1024 + row * 4), "r"(pk[n][h][e]) : "memory");
          asm volatile("st.shared.b32 [%0], %1;" ::"r"(stw + dst * 1024 + 512 + row * 4), "r"(pack_bf16(acc[0][n][k], acc[1][n][k])) : "memory");
        }
    fence_async_smem();
    __syncwarp();
    if (lane < 4) {
      const uint32_t dsm = mapa(su + O_T + (uint32_t)(8 * rank + cp) * 1024u, (uint32_t)lane);
      asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], 1024, [%2];"
                   ::"r"(dsm), "r"(stw + lane * 1024), "r"(mapa(smem_u32(&B.t_full), (uint32_t)lane)) : "memory");
    } else if (warp == 8 && lane == 4) {
      // Wp [64][128] (two SW128 boxes) into slot 0's consumed B part
      mbar_expect_tx(&B.wp_full, 2 * WB);
      tma_load_2d(sm + O_WP, &m.w[5], &B.wp_full, 0, 0, EVICT_LAST);
      tma_load_2d(sm + O_WP + WB, &m.w[5], &B.wp_full, 64, 0, EVICT_LAST);
    }
    if (warp == 8 && lane == 0) TFTL(11)
    if (warp >= 12) cluster_arrive();   // early: the exit waits for every arrival of the cluster
  }
  // ------------------------------------------------------------------ output: p = xo Wp^T (warp 1), LN_out + epilogue (warps 4-7)
  if (warp == 1) {
    if (elect_one()) {
      constexpr uint32_t ID64 = idesc_bf16(128, 64);
      mbar_wait(&B.wp_full, 0);
      mbar_wait(&B.xo_ready, 0);
      tc_fence_after();
#pragma unroll
      for (int k = 0; k < 8; ++k)
        umma_ts(tmem + T_P, tmem + T_XO + k * 8, desc_k_sw128(su + O_WP + (k >> 2) * WB + (k & 3) * 32), ID64, k);
      umma_commit(&B.p_full);
    }
    __syncwarp();
  } else if ((warp >= 4 && warp < 8) || warp >= 12) {   // warps 12-15: the upper half of the output channels in the epilogue
    const int r = (warp & 3) * 32 + lane;
    const uint32_t trow = tmem + ((uint32_t)((warp & 3) * 32) << 16);
    auto up = [](uint32_t w) { return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u)); };
    const int cq0 = warp < 8 ? 0 : 2;
    // the residual x (channels 16 cq0 .. 16 cq0 + 31 of row r) from global, before the waits
    uint4 xres[4];
    {
      const uint4* xp = reinterpret_cast<const uint4*>(xg + tok_of(r) * C + cq0 * 16);
#pragma unroll
      for (int q = 0; q < 4; ++q) xres[q] = xp[q];
    }
    if (warp < 8) {
      mbar_wait_cluster(&B.t_full, 0);
      cluster_arrive();   // this CTA's t tile is complete: its peers' copies into it are done
      if (r == 0) TFTL(4)
      const uint32_t trr = su + O_T;
      // channels 8 q .. 8 q + 7 (q < 8 outgoing, >= 8 incoming): t rows 2 (4 q & 31) + (q >> 3) .. step 2
      auto ldt = [&](int q) {
        const uint32_t a = trr + (uint32_t)(2 * ((4 * q) & 31) + (q >> 3)) * 512u + r * 4;
        uint4 v;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v.x) : "r"(a));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v.y) : "r"(a + 1024));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v.z) : "r"(a + 2048));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v.w) : "r"(a + 3072));
        return v;
      };
      float2 a0 = make_float2(0.f, 0.f), a1 = a0, a2 = a0, a3 = a0;
#pragma unroll
      for (int q = 0; q < 16; ++q) {
        const uint4 t = ldt(q);
        a0 = __fadd2_rn(a0, up(t.x)); a1 = __fadd2_rn(a1, up(t.y)); a2 = __fadd2_rn(a2, up(t.z)); a3 = __fadd2_rn(a3, up(t.w));
      }
      a0 = __fadd2_rn(__fadd2_rn(a0, a1), __fadd2_rn(a2, a3));
      const float mu = __fmul_rn(__fadd_rn(a0.x, a0.y), 1.f / HT);
      const float2 nm0 = make_float2(-mu, -mu);
      a0 = a1 = a2 = a3 = make_float2(0.f, 0.f);
#pragma unroll
      for (int q = 0; q < 16; ++q) {
        const uint4 t = ldt(q);
        float2 d;
        d = __fadd2_rn(up(t.x), nm0); a0 = __ffma2_rn(d, d, a0);
        d = __fadd2_rn(up(t.y), nm0); a1 = __ffma2_rn(d, d, a1);
        d = __fadd2_rn(up(t.z), nm0); a2 = __ffma2_rn(d, d, a2);
        d = __fadd2_rn(up(t.w), nm0); a3 = __ffma2_rn(d, d, a3);
      }
      a0 = __fadd2_rn(__fadd2_rn(a0, a1), __fadd2_rn(a2, a3));
      const float rs = rsqrt_ftz(__fadd_rn(__fmul_rn(__fadd_rn(a0.x, a0.y), 1.f / HT), eps));
      const float2 nm = make_float2(-mu, -mu), rr = make_float2(rs, rs);
#pragma unroll
      for (int q4 = 0; q4 < 8; ++q4) {
        uint32_t o[8];
#pragma unroll
        for (int qq = 0; qq < 2; ++qq) {
          const int q = q4 * 2 + qq;
          const uint4 t = ldt(q);
          const uint32_t gq = su + O_GB + 512 + q * 32;
          const float4 ga = lds128f(gq), gb4 = lds128f(gq + 16), ba = lds128f(gq + 512), bb = lds128f(gq + 512 + 16);
          auto aff2 = [&](uint32_t w, float g0, float g1, float b0, float b1) {
            const float2 y = __ffma2_rn(__fmul2_rn(__fadd2_rn(up(w), nm), rr), make_float2(g0, g1), make_float2(b0, b1));
            return pack_bf16(y.x, y.y);
          };
          o[4 * qq + 0] = aff2(t.x, ga.x, ga.y, ba.x, ba.y);
          o[4 * qq + 1] = aff2(t.y, ga.z, ga.w, ba.z, ba.w);
          o[4 * qq + 2] = aff2(t.z, gb4.x, gb4.y, bb.x, bb.y);
          o[4 * qq + 3] = aff2(t.w, gb4.z, gb4.w, bb.z, bb.w);
        }
        tmem_st8(trow + T_XO + q4 * 8, o);
      }
      tmem_wait_st();
      tc_fence_before();
      named_bar_sync(1, 128);
      if (r == 0) { mbar_arrive(&B.xo_ready); TFTL(12) }
    }
    // epilogue: y = bf16(x + bf16(sigmoid(g) p)) (k3g inference), st.global
    {
      mbar_wait(&B.p_full, 0);
      tc_fence_after();
      if (r == 0) TFTL(13)
      __nv_bfloat16* ytok = yg + tok_of(r) * C;
#pragma unroll
      for (int cc = 0; cc < 2; ++cc) {
        const int cq = cq0 + cc;
        uint32_t pv[16], gv[16];
        tmem_ld16(trow + T_P + cq * 16, pv);
        tmem_ld16(trow + T_G + cq * 16, gv);
        tmem_wait_ld();
#pragma unroll
        for (int q2 = 0; q2 < 2; ++q2) {
          const int q = cq * 2 + q2;
          const uint4 xq = xres[cc * 2 + q2];
          const uint32_t xw[4] = {xq.x, xq.y, xq.z, xq.w};
          uint32_t o[4];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const int c = q2 * 8 + 2 * k;
            const float2 pp = make_float2(__uint_as_float(pv[c]), __uint_as_float(pv[c + 1]));
            const float2 sg = sigmoid2(make_float2(__uint_as_float(gv[c]), __uint_as_float(gv[c + 1])));
            const float2 pr = __fmul2_rn(sg, pp);
            const uint32_t pb = pack_bf16(pr.x, pr.y);
            const float2 y = __fadd2_rn(make_float2(bf16lo(xw[k]), bf16hi(xw[k])), make_float2(bf16lo(pb), bf16hi(pb)));
            o[k] = pack_bf16(y.x, y.y);
          }
          *reinterpret_cast<uint4*>(ytok + q * 8) = make_uint4(o[0], o[1], o[2], o[3]);
        }
      }
      if (r == 0) TFTL(5)
    }
  }
  // exit only when every CTA of the cluster has its t tile (its DSMEM bulk copies read this CTA's shared memory)
  if (warp < 4 || (warp >= 8 && warp < 12)) cluster_arrive();
  tc_fence_before();
  __syncthreads();
  cluster_wait();
  if (tid == 0) TFTL(14)
  if (warp == 2) tmem_dealloc(tmem, 512);
}

template <int MASK>
void* kfn() { return reinterpret_cast<void*>(tfb_kernel<MASK>); }

}  // namespace tfb

int tfb_clusters() {
  auto kern = tfb::tfb_kernel<0>;
  cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, tfb::SMEM);
  if (e != cudaSuccess) { cudaGetLastError(); return -1000 - (int)e; }
  e = cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  if (e != cudaSuccess) { cudaGetLastError(); return -2000 - (int)e; }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(tfb::NCTA); cfg.blockDim = dim3(512); cfg.dynamicSmemBytes = tfb::SMEM;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = tfb::CL; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  int n = 0;
  e = cudaOccupancyMaxActiveClusters(&n, kern, &cfg);
  if (e != cudaSuccess) { cudaGetLastError(); return -(int)e; }
  return n;
}
bool tfb_supported() {
  static int ok = -1;
  if (ok < 0) ok = tfb_clusters() >= tfb::NCTA / tfb::CL ? 1 : 0;
  return ok == 1;
}

// x [L L, 64] bf16 (one sample, L = 128); wl, wlg, wr, wrg [128, 64] ([outgoing | incoming] rows), wg [64, 64], wp [64, 128] bf16 with
// unit column stride; tokmask [L] uint8 / bool or None; gi, bi fp32 [64], go, bo fp32 [128]; planes [256, L, L] bf16 scratch; y [L L, 64].
void tfb_forward(torch::Tensor x, torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg, torch::Tensor wg,
                 torch::Tensor wp, c10::optional<torch::Tensor> tokmask, torch::Tensor gi, torch::Tensor bi, torch::Tensor go,
                 torch::Tensor bo, torch::Tensor planes, torch::Tensor y, double eps) {
  using namespace tfb;
  TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && planes.is_contiguous() && x.size(0) == L * L && x.size(1) == C &&
              planes.numel() == 4 * H * L * L, "tfb_forward: shapes");
  const auto bf = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  Maps m;
  const torch::Tensor* ws[6] = {&wlg, &wl, &wrg, &wr, &wg, &wp};
  const int64_t rows[6] = {128, 128, 128, 128, 64, 64}, cols[6] = {64, 64, 64, 64, 64, 128};
  for (int k = 0; k < 6; ++k) {
    const torch::Tensor& w = *ws[k];
    TORCH_CHECK(w.scalar_type() == at::kBFloat16 && w.size(0) == rows[k] && w.size(1) == cols[k] && w.stride(1) == 1 &&
                (w.stride(0) * 2) % 16 == 0, "tfb_forward: bf16 weights with unit column stride");
    m.w[k] = tmap::make(w.data_ptr(), bf, {(uint64_t)cols[k], (uint64_t)rows[k]}, {(uint64_t)w.stride(0) * 2}, {64, 64},
                        CU_TENSOR_MAP_SWIZZLE_128B);
  }
  // planes [256][L L] scratch, BLOCKED: (element in a 128-element block, block, channel)
  const std::vector<uint64_t> ps = {256, (uint64_t)L * L * 2};
  const std::vector<uint64_t> pd = {128, (uint64_t)L * L / 128, 256};
  m.pld = tmap::make(planes.data_ptr(), bf, pd, ps, {128, 16, 2}, CU_TENSOR_MAP_SWIZZLE_NONE);
  m.pldb = tmap::make(planes.data_ptr(), bf, pd, ps, {128, 32, 2}, CU_TENSOR_MAP_SWIZZLE_NONE);
  const uint8_t* mk = nullptr;
  if (tokmask.has_value()) {
    TORCH_CHECK(tokmask->numel() == L && tokmask->element_size() == 1 && tokmask->is_contiguous(), "tfb_forward: token mask [L]");
    mk = reinterpret_cast<const uint8_t*>(tokmask->data_ptr());
  }
  void* kern = mk ? kfn<1>() : kfn<0>();
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(NCTA); cfg.blockDim = dim3(512); cfg.dynamicSmemBytes = SMEM; cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute lat[1];
  lat[0].id = cudaLaunchAttributeClusterDimension;
  lat[0].val.clusterDim.x = CL; lat[0].val.clusterDim.y = 1; lat[0].val.clusterDim.z = 1;
  cfg.attrs = lat; cfg.numAttrs = 1;
  const float* gip = gi.data_ptr<float>(); const float* bip = bi.data_ptr<float>();
  const float* gop = go.data_ptr<float>(); const float* bop = bo.data_ptr<float>();
  float e = (float)eps;
  void* plp = planes.data_ptr(); void* ypp = y.data_ptr(); const void* xpp = x.data_ptr();
  void* args[] = {&m, &mk, &gip, &bip, &gop, &bop, &e, &plp, &ypp, &xpp};
  C10_CUDA_CHECK(cudaLaunchKernelExC(&cfg, kern, args));
}

torch::Tensor tfb_tl() {
#ifdef TF_TL
  torch::Tensor h = torch::empty({128, 16}, torch::kInt64);
  cudaMemcpyFromSymbol(h.data_ptr(), g_tftl, sizeof(unsigned long long) * 128 * 16);
  return h;
#else
  return torch::empty({0});
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mm) {
  mm.def("tfb_forward", &tfb_forward);
  mm.def("tfb_supported", &tfb_supported);
  mm.def("tfb_clusters", &tfb_clusters);
  mm.def("tfb_tl", &tfb_tl);
}
