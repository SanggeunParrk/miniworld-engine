// tf_sm100.cu -- the whole D64 TriMul inference update at L = 128 in ONE launch (B200, sm_100a only), one direction, hidden 64.
//
// Why a new design (docs/b200/trimul.md, round 3 part 9): the three-launch chain k1w -> contraction -> k3g is three one-tile
// dependency chains in a row, and the earlier one-launch version of the same three phases (two grid barriers, t through L2) was not
// faster. This kernel removes one of the two all-to-all exchanges: the contraction is computed by the CTAs that own the OUTPUT tokens,
// so t never goes to global memory and the output phase needs no second grid barrier.
//
// Grid: 128 CTAs = 16 clusters of 8. A cluster owns a 32 x 32 block of pair tokens (i in I0 .. I0+31, j in J0 .. J0+31); CTA rank r
// owns the 8 x 16 sub-block i in I0 + 8 (r >> 1) .., j in J0 + 16 (r & 1) .. (128 tokens, its x tile) and contraction channels
// 8 r .. 8 r + 7 of the cluster block.
//   phase 1 (per CTA, its 128 tokens): x tile (TMA, a 3-D box: row = (i, j) of the sub-block), LN_in -> xn (TMEM), the gated
//           projections a | b (tcgen05, N 128 + 128) and the output gate g = xn Wg^T (N 64, kept in TMEM to the end); a, b ->
//           channel-major planes in global (L2) in the contraction's K-major orientation: P[c][i][k] (outgoing: the plane as is;
//           incoming: the plane transposed, written so by the 3-D store box), so both directions contract T_c = A_c B_c^T.
//   barrier: the CTAs whose plane rows this cluster needs (a counter per 32-row block, release / acquire at GPU scope).
//   phase 2 (per CTA): A rows I0.., B rows J0.. of its 8 channels (4 TMA boxes, 128 KB), T_c = A_c B_c^T on the 32 x 32 block with
//           mma.sync m16n8k16 (fp32 accumulators, one bf16 rounding as cuBLAS's bf16 output), each token's 8 channels sent as one
//           16-byte st.async into the owning CTA's t tile (DSMEM, completing on its mbarrier).
//   phase 3 (per CTA, its 128 tokens): LN_out over the 64 channels -> xo (TMEM), p = xo Wp^T (tcgen05), y = x + bf16(sigmoid(g) p)
//           (k3g's inference epilogue arithmetic), y stored with the x tile's box.
// Warps: 0 TMA (lanes 0-6 issue in parallel), 1 MMA, 2 TMEM, 4-7 LN_in / LN_out / epilogue (thread = token row), 8-11 / 12-15 the
// front epilogues of a / b, then all eight the contraction.
#include "sm100.cuh"
#include "tmap.h"

using namespace sm100;

#ifdef TF_TL
static __device__ unsigned long long g_tftl[128][16];
#define TFTL(k) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); g_tftl[blockIdx.x][k] = t_; }
#else
#define TFTL(k) {}
#endif

#ifndef TF_DIRECT
#define TF_DIRECT 1   // planes and y written by the threads with st.global (no shared-memory staging, no TMA store)
#endif
#ifndef TF_XREG
#define TF_XREG 2     // x at entry by ld.global before anything else: 2 all 512 threads coalesced into the swizzled tile (taken), 1 the LN threads' rows into registers (slower), 0 one TMA box
#endif
#ifndef TF_OPL
#define TF_OPL 4      // operand-issuing lanes: 4 (channel quarters) or 8 (channel pairs, a barrier per contraction warp)
#endif
#ifndef TF_EPI
#define TF_EPI 0      // output epilogue: g, x from TMEM before the wait for p, one TMEM wait after it
#endif
#ifndef TF_ONEFENCE
#define TF_ONEFENCE 1 // planes released by one thread's fence after the CTA barrier (no per-thread __threadfence)
#endif
namespace tf {
constexpr int C = 64, H = 64, TOK = 128, L = 128;
// 32 clusters of 4 (at most 15 clusters of 8 / 7 of 16 are co-resident on the 148 SMs; the counter waits need all 128 CTAs resident),
// 16 contraction channels a CTA
constexpr int NCTA = 128, CL = 4, CPC = 16;
// shared memory. Front: W (5 boxes) | x tile | plane staging. After the front the contraction operands overwrite all three (x lives
// in TMEM from LN_in on): A kh0 | A kh1 (32 KB each: 16 rows x 64 k x 16 ch) | B kh0 | B kh1 (64 KB each: 32 rows)
constexpr int WB = 64 * 64 * 2;                    // one [64][64] bf16 weight box (SW128) = 8 KB
constexpr int O_W = 0;                              // wlg | wl | wrg | wr | wg
constexpr int O_X = O_W + 5 * WB;                   // x tile [128][64] SW128 (16 KB)
constexpr int O_STG = O_X + TOK * C * 2;            // plane staging a | b ([64 ch][128 tok] bf16, 16 KB each)
constexpr int O_AB = 0, O_BB = O_AB + 16 * 4096;   // A [16 ch][16 blocks of 256 B], B [16 ch][32 blocks]
constexpr int O_WP = O_BB + 2 * 65536;              // wp (kept to the end)
constexpr int O_T = O_WP + WB;                      // t tile [128 tok][64 ch] SW128 (16 KB), written by the cluster's CTAs
constexpr int O_GB = O_T + TOK * H * 2;             // g_in | b_in | g_out | b_out (fp32)
constexpr int O_BAR = O_GB + 4 * 64 * 4;
constexpr int SMEM = O_BAR + 256 + 1024;            // + alignment slack
static_assert(SMEM <= 232448 && O_STG + 2 * TOK * 64 * 2 <= O_WP, "smem");
// TMEM columns (x: the raw bf16 row, for the residual)
constexpr uint32_t T_A = 0, T_B = 128, T_G = 256, T_XN = 320, T_XNU = 352, T_XO = 384, T_P = 416, T_X = 480;

struct Maps {
  CUtensorMap w[6];   // wlg, wl, wrg, wr, wg, wp: [64][64] row-major, box 64 x 64 SW128
  CUtensorMap x;      // the CTA's 8 x 16 token sub-block (3-D box), SW128; also y
  CUtensorMap y;
  CUtensorMap pst;    // planes, the front's store box (no swizzle)
  CUtensorMap pld;    // planes, the contraction's A box {64 k, 16 rows, 16 ch} SW128
  CUtensorMap pldb;   // B box {64 k, 32 rows, 16 ch}
};

struct Bars {
  uint64_t g_full, x_full, x_free, w_full, xn_full, acc_full[2], ab_full[8], t_full, xo_ready, p_full;
  uint32_t tmem;
  uint32_t base;
};

// plane-row block counters, monotonic: every launch adds exactly 32 to each (one per CTA whose front writes that block); a launch's
// base is read at entry from its own block, before its own increment (kernels of a stream do not overlap: no PDL attribute)
__device__ unsigned g_cnt[4];

DEV float tanh_approx(float x) { float y; asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float rsqrt_ftz(float x) { float y; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float2 sigmoid2(float2 g) {   // k1w / k3g: 0.5 + 0.5 tanh.approx(g / 2)
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
DEV void sts16(uint32_t a, uint32_t v) { asm volatile("st.shared.b16 [%0], %1;" ::"r"(a), "h"((unsigned short)v) : "memory"); }
DEV void tmem_st8(uint32_t taddr, const uint32_t (&r)[8]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
               :: "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]) : "memory");
}
DEV void tmem_ld8(uint32_t taddr, uint32_t (&r)[8]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr));
}
DEV void ldsm_x4(uint32_t a, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
DEV void ldsm_x2(uint32_t a, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(a));
}
DEV void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
DEV void st_async_b32(uint32_t raddr, uint32_t a, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(raddr), "r"(a), "r"(rbar) : "memory");
}
DEV void st_async_v2(uint32_t raddr, uint32_t a, uint32_t b, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];" ::"r"(raddr), "r"(a), "r"(b), "r"(rbar) : "memory");
}
DEV void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
DEV unsigned ld_acquire(const unsigned* p) {
  unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}

// OUT = 1 outgoing (t_c = a_c b_c^T over the second token index), 0 incoming (a_c^T b_c over the first)
template <int OUT, int MASK>
__global__ void __launch_bounds__(512, 1)
    tf_kernel(const __grid_constant__ Maps m, const uint8_t* __restrict__ tokmask, const float* __restrict__ g_in,
              const float* __restrict__ b_in, const float* __restrict__ g_out, const float* __restrict__ b_out, float eps,
              __nv_bfloat16* __restrict__ planes, __nv_bfloat16* __restrict__ yg, const __nv_bfloat16* __restrict__ xg, float rf3) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
#ifdef TF_EMPTY
  if (threadIdx.x < 1u << 20) return;   // launch-floor probe
#endif
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  float rf3s = 1.f;   // RoseTTAFold3 length factor (rf3 >= 0): the constant, or bf16(1 / max(n, 1)) from the token mask counted here
  if (rf3 >= 0.f) {
    if (rf3 > 0.f || tokmask == nullptr) rf3s = rf3;
    else {
      int n = 0;
      for (int i0_ = 0; i0_ < L; i0_ += 512) n += __syncthreads_count(i0_ + tid < L && tokmask[i0_ + tid] != 0);
      rf3s = __bfloat162float(__float2bfloat16_rn(1.0f / (float)max(n, 1)));
    }
  }
  const int rank = (int)cluster_ctarank(), cid = blockIdx.x / CL;
  const int I0 = 16 * (cid >> 2), J0 = 32 * (cid & 3);   // the cluster's 16 x 32 token block
  const int i0 = I0 + 8 * (rank >> 1), j0 = J0 + 16 * (rank & 1);
  // pair-token offset of row r of the sub-block (the x / y box order: outgoing (i, j) j-fastest, incoming i-fastest)
  auto tok_of = [&](int r) -> size_t {
    return OUT ? (size_t)(i0 + (r >> 4)) * L + j0 + (r & 15) : (size_t)(i0 + (r & 7)) * L + j0 + (r >> 3);
  };
#if TF_XREG == 1
  // the LN threads' x rows first of all (a cold DRAM read: nothing waits for a tensormap or a barrier before it is issued)
  uint4 xv[8];
  if (warp >= 4 && warp < 8) {
    const uint4* xp = reinterpret_cast<const uint4*>(xg + tok_of((warp & 3) * 32 + lane) * C);
#pragma unroll
    for (int q = 0; q < 8; ++q)
      asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(xv[q].x), "=r"(xv[q].y), "=r"(xv[q].z), "=r"(xv[q].w) : "l"(xp + q));
  }
#elif TF_XREG == 2
  // the x tile by all 512 threads, coalesced (32 B a thread, token row tid / 4), first of all; into the swizzled tile before the
  // cluster barrier below
  uint4 xw0, xw1;
  {
    const uint4* xp = reinterpret_cast<const uint4*>(xg + tok_of(tid >> 2) * C) + (tid & 3) * 2;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(xw0.x), "=r"(xw0.y), "=r"(xw0.z), "=r"(xw0.w) : "l"(xp));
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(xw1.x), "=r"(xw1.y), "=r"(xw1.z), "=r"(xw1.w) : "l"(xp + 1));
  }
#endif
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  // the x / y box coordinates: (channel, inner token index, outer token index)
  const int xc1 = OUT ? j0 : i0, xc2 = OUT ? i0 : j0;
  const int pkey = OUT ? (i0 >> 5) : (j0 >> 5);   // the plane-row block this CTA's front writes
  if (tid == 0) TFTL(0)

  if (tid == 0) {
    mbar_init(&B.x_full, 1); mbar_init(&B.x_free, 1); mbar_init(&B.w_full, 6); mbar_init(&B.xn_full, 1);
    mbar_init(&B.g_full, 1); mbar_init(&B.acc_full[0], 1); mbar_init(&B.acc_full[1], 1); for (int q = 0; q < TF_OPL; ++q) mbar_init(&B.ab_full[q], 1);
    mbar_init(&B.t_full, 1); mbar_init(&B.xo_ready, 1); mbar_init(&B.p_full, 1);
    mbar_expect_tx(&B.t_full, TOK * H * 2);   // the cluster's 8 CTAs complete 16 KB of st.async on it
    fence_mbar_init();
  }
  __syncwarp();
  if (warp == 0 && lane < 7) {
    // every first load at entry, one TMA a lane (one thread issues ~1 TMA per few hundred clocks)
    if (lane < 6) {
      prefetch_tmap(&m.w[lane]);
      mbar_expect_tx(&B.w_full, WB);
      tma_load_2d(sm + (lane < 5 ? O_W + lane * WB : O_WP), &m.w[lane], &B.w_full, 0, 0, EVICT_LAST);
    } else if (!TF_XREG) {
      prefetch_tmap(&m.x);
      mbar_expect_tx(&B.x_full, TOK * C * 2);
      tma_load_3d(sm + O_X, &m.x, &B.x_full, 0, xc1, xc2, EVICT_NORMAL);
    }
  }
  if (warp == 3 && lane == 0) { prefetch_tmap(&m.pst); prefetch_tmap(&m.pld); prefetch_tmap(&m.y); }
  if (tid < 64) {
    float* gb = reinterpret_cast<float*>(sm + O_GB);
    gb[tid] = g_in[tid]; gb[64 + tid] = b_in[tid]; gb[128 + tid] = g_out[tid]; gb[192 + tid] = b_out[tid];
  }
  if (warp == 2) { tmem_alloc(&B.tmem, 512); tmem_relinquish(); }
  if (warp == 3 && lane == 1) B.base = ld_acquire(&g_cnt[pkey]) & ~31u;
#if TF_XREG == 2
  sts128(su + O_X + sw128(tid >> 2, (tid & 3) * 2), xw0.x, xw0.y, xw0.z, xw0.w);
  sts128(su + O_X + sw128(tid >> 2, (tid & 3) * 2 + 1), xw1.x, xw1.y, xw1.z, xw1.w);
#endif
  tc_fence_before();
  cluster_arrive();   // every barrier of the cluster initialised before any peer's st.async
  cluster_wait();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    if (lane < TF_OPL) {
      // the operands overwrite the front's weights, x tile and staging: after the front products (acc_full[1]) and LN_in (x_free);
      // A rows I0 .. I0 + 15, B rows J0 .. J0 + 31 (their 32-row plane blocks complete)
      mbar_wait(&B.g_full, 0);
      if (TF_XREG != 1) mbar_wait(&B.x_free, 0);
      const unsigned* ca = &g_cnt[I0 >> 5];
      const unsigned* cb = &g_cnt[J0 >> 5];
      const unsigned tgt = B.base + 32u;
      while ((int)(ld_acquire(ca) - tgt) < 0) {}
      while ((int)(ld_acquire(cb) - tgt) < 0) {}
      fence_proxy_async_global();
      if (lane == 0) TFTL(2)
      // lane q: channel quarter q (4 channels) of A and B, on its own barrier (the contraction starts on the first quarter)
      constexpr int CQ = CPC / TF_OPL;   // channels a lane
      mbar_expect_tx(&B.ab_full[lane], CQ * (4096 + 8192));
      tma_load_3d(sm + O_AB + lane * CQ * 4096, &m.pld, &B.ab_full[lane], 0, I0, CPC * rank + CQ * lane, EVICT_FIRST);
      tma_load_3d(sm + O_BB + lane * CQ * 8192, &m.pldb, &B.ab_full[lane], 0, J0, 64 + CPC * rank + CQ * lane, EVICT_FIRST);
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------ MMA
    if (elect_one()) {
      constexpr uint32_t ID128 = idesc_bf16(128, 128), ID64 = idesc_bf16(128, 64);
      mbar_wait(&B.w_full, 0);
      mbar_wait(&B.xn_full, 0);
      tc_fence_after();
#pragma unroll
      for (int k = 0; k < 4; ++k) umma_ts(tmem + T_A, tmem + T_XN + k * 8, desc_k_sw128(su + O_W + k * 32), ID128, k);
      umma_commit(&B.acc_full[0]);
#pragma unroll
      for (int k = 0; k < 4; ++k) umma_ts(tmem + T_B, tmem + T_XN + k * 8, desc_k_sw128(su + O_W + 2 * WB + k * 32), ID128, k);
      umma_commit(&B.acc_full[1]);
#pragma unroll
      for (int k = 0; k < 4; ++k)
        umma_ts(tmem + T_G, tmem + (MASK ? T_XNU : T_XN) + k * 8, desc_k_sw128(su + O_W + 4 * WB + k * 32), ID64, k);
      umma_commit(&B.g_full);   // every front product done: the operand loads may then overwrite every front weight
    }
    __syncwarp();
  } else if (warp >= 4 && warp < 8) {
    // ------------------------------------------------------------------ thread = token row r of the sub-block
    const int r = (warp & 3) * 32 + lane;
    const uint32_t trow = tmem + ((uint32_t)((warp & 3) * 32) << 16);
    auto up = [](uint32_t w) { return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u)); };
    const uint32_t xr = su + O_X;
#if TF_XREG == 1
#define XQ(q) xv[q]
#else
#define XQ(q) lds128(xr + sw128(r, (q)))
#endif
    // LN_in (k1w's arithmetic: two-pass statistics, xn = bf16(fma((x - mean) rstd, g, b)), masked pairs' rows zero)
    {
      uint32_t kmask = 0xffffffffu;
      if (MASK) {
        const int pi = OUT ? (r >> 4) : (r & 7), pj = OUT ? (r & 15) : (r >> 3);
        kmask = (tokmask[i0 + pi] && tokmask[j0 + pj]) ? 0xffffffffu : 0u;
      }
      if (!TF_XREG) mbar_wait(&B.x_full, 0);
      (void)xr;
#if defined(TF_TL) && TF_XREG == 1
      if (r == 0) asm volatile("" :: "r"(xv[7].w));   // the mark after the row arrived
#endif
      if (r == 0) TFTL(1)
      float2 a0 = make_float2(0.f, 0.f), a1 = a0, a2 = a0, a3 = a0;
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const uint4 t = XQ(q);
        a0 = __fadd2_rn(a0, up(t.x)); a1 = __fadd2_rn(a1, up(t.y)); a2 = __fadd2_rn(a2, up(t.z)); a3 = __fadd2_rn(a3, up(t.w));
      }
      a0 = __fadd2_rn(__fadd2_rn(a0, a1), __fadd2_rn(a2, a3));
      const float mu = __fmul_rn(__fadd_rn(a0.x, a0.y), 1.f / C);
      const float2 nm0 = make_float2(-mu, -mu);
      a0 = a1 = a2 = a3 = make_float2(0.f, 0.f);
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const uint4 t = XQ(q);
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
          const uint4 t = XQ(q);
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
        {   // the raw x row, for the residual (the x tile's shared memory goes to the contraction operands)
          const uint4 t0 = XQ(2 * q4), t1 = XQ(2 * q4 + 1);
          const uint32_t xw[8] = {t0.x, t0.y, t0.z, t0.w, t1.x, t1.y, t1.z, t1.w};
          tmem_st8(trow + T_X + q4 * 8, xw);
        }
      }
      tmem_wait_st();
      tc_fence_before();
      named_bar_sync(1, 128);
      if (r == 0) { mbar_arrive(&B.xn_full); mbar_arrive(&B.x_free); TFTL(7) }
    }
  } else if (warp >= 8) {
    // ------------------------------------------------------------------ front epilogue: warps 8-11 plane a, 12-15 plane b
    {
      const int side = (warp - 8) >> 2;
      const int r = (warp & 3) * 32 + lane;
      const uint32_t tq = tmem + (side ? T_B : T_A) + ((uint32_t)((warp & 3) * 32) << 16);
      // the planes are BLOCKED: per channel, 256-byte blocks of the contraction's (row, k) sub-tiles -- outgoing 8 rows x 16 k
      // (16-byte halves of a 32-byte row swapped when (row >> 2) is odd: conflict-free ldmatrix), incoming 16 rows x 8 k; this CTA's
      // tokens are one block of every channel, its staging [64 ch][256 B] one TMA box of 64 rows
      const uint32_t stg = su + O_STG + side * (TOK * 64 * 2) +
                           (OUT ? (uint32_t)((r >> 4) * 32 + ((((r >> 3) & 1) ^ ((r >> 6) & 1)) << 4) + (r & 7) * 2) : (uint32_t)(r * 2));
      const int blk = OUT ? (i0 >> 3) * 8 + (j0 >> 4) : (j0 >> 4) * 16 + (i0 >> 3);
      uint8_t* gpl = reinterpret_cast<uint8_t*>(planes) + ((size_t)(side * 64) * (L * L) + (size_t)blk * 128) * 2 + (stg - (su + O_STG + side * (TOK * 64 * 2)));
      mbar_wait(&B.acc_full[side], 0);
      tc_fence_after();
      if (warp == 8 && lane == 0) TFTL(8)
#pragma unroll 1
      for (int h = 0; h < 2; ++h) {
        uint32_t g[32], p[32];
        tmem_ld32(tq + h * 32, g);
        tmem_ld32(tq + 64 + h * 32, p);
        tmem_wait_ld();
#pragma unroll
        for (int c = 0; c < 32; c += 2) {
          const float2 s = sigmoid2(make_float2(__uint_as_float(g[c]), __uint_as_float(g[c + 1])));
          float2 pv2 = make_float2(__uint_as_float(p[c]), __uint_as_float(p[c + 1]));
          if (rf3 >= 0.f && side == 0) pv2 = __fmul2_rn(pv2, make_float2(rf3s, rf3s));   // RoseTTAFold3: f on the left projection
          const float2 v = __fmul2_rn(s, pv2);
          // tokens r, r ^ 1 are adjacent in the staging block: the even lane stores channel c of both, the odd lane channel c + 1
          const uint32_t w = pack_bf16(v.x, v.y), wo = __shfl_xor_sync(0xffffffffu, w, 1);
          const bool odd = lane & 1;
          const uint32_t word = odd ? __byte_perm(wo, w, 0x7632) : __byte_perm(w, wo, 0x5410);
#if TF_DIRECT
          *reinterpret_cast<uint32_t*>(gpl - (odd ? 2 : 0) + (size_t)(h * 32 + c + (odd ? 1 : 0)) * (L * L * 2)) = word;
#else
          asm volatile("st.shared.b32 [%0], %1;" ::"r"(stg - (odd ? 2u : 0u) + (h * 32 + c + (odd ? 1 : 0)) * (TOK * 2)), "r"(word) : "memory");
#endif
        }
      }
#if TF_DIRECT
      if (!TF_ONEFENCE) __threadfence();   // this thread's plane words performed at GPU scope before the block's release
      named_bar_sync(2, 256);
      if (warp == 8 && lane == 0) {
        TFTL(9)
        // one thread's gpu-scope fence after the barrier (cumulative: the cooperative-groups grid barrier's pattern)
        if (TF_ONEFENCE) __threadfence();
#else
      fence_async_smem();
      named_bar_sync(2, 256);
      if (warp == 8 && lane == 0) {
        TFTL(9)
        tma_store_3d(&m.pst, sm + O_STG, 0, blk, 0);
        tma_store_3d(&m.pst, sm + O_STG + TOK * 64 * 2, 0, blk, 64);
        bulk_commit();
        bulk_wait<0>();
        fence_proxy_async_global();
        __threadfence();
#endif
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(&g_cnt[pkey]) : "memory");
        TFTL(6)
      }
    }
  }
  // ------------------------------------------------------------------ contraction, warps 8-15: warp = local channel pair cp (2 cp,
  // 2 cp + 1; quarter cp / 2 on its own barrier) over the whole 16 x 32 block: every operand byte is read from shared memory once
  // (the ldmatrix traffic bounds this phase); blocked operands (A 16 blocks = 4 KB a channel, B 32 blocks = 8 KB)
  if (warp >= 8) {
    const int cp = warp - 8, qd = cp >> 1;
    float acc[2][4][4];
#pragma unroll
    for (int c2 = 0; c2 < 2; ++c2)
#pragma unroll
      for (int n = 0; n < 4; ++n) acc[c2][n][0] = acc[c2][n][1] = acc[c2][n][2] = acc[c2][n][3] = 0.f;
    // byte offset of row r, 8-element k chunk kc inside a channel's blocked operand
    auto boff_of = [](int r, int kc) -> uint32_t {
      return OUT ? (uint32_t)(((r >> 3) * 8 + (kc >> 1)) * 256 + (r & 7) * 32 + (((kc & 1) ^ ((r >> 2) & 1)) << 4))
                 : (uint32_t)(((r >> 4) * 16 + kc) * 256 + (r & 15) * 16);
    };
    mbar_wait(&B.ab_full[TF_OPL == 8 ? cp : qd], 0);
    if (warp == 8 && lane == 0) TFTL(3)
    const uint32_t sA = su + O_AB + cp * 2 * 4096, sB = su + O_BB + cp * 2 * 8192;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const uint32_t aoff = boff_of(lane & 15, 2 * s + (lane >> 4));
      // B x4: lanes 0-7 / 8-15 the two k halves of n-tile 2 np, lanes 16-23 / 24-31 those of n-tile 2 np + 1
      const uint32_t bo0 = boff_of(8 * (lane >> 4) + (lane & 7), 2 * s + ((lane >> 3) & 1));
      const uint32_t bo1 = boff_of(16 + 8 * (lane >> 4) + (lane & 7), 2 * s + ((lane >> 3) & 1));
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
    if (warp == 8 && lane == 0) TFTL(10)
    // t exchange: this warp's channel pair for each destination CTA (128 tokens x 4 B = 512 B) staged in its own operand slab (read
    // by this warp only, consumed), then one 512-byte DSMEM bulk copy a destination into the t tile, which is [32 channel pairs]
    // [128 tokens] x 4 B: pair 8 rank + cp
    const uint32_t stw = sA;
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int il = (lane >> 2) + 8 * h, jl = 8 * n + 2 * (lane & 3) + e, k = 2 * h + e;
          const int dst = (il >> 3) * 2 + (jl >> 4);
          const int row = OUT ? (il & 7) * 16 + (jl & 15) : (jl & 15) * 8 + (il & 7);
          asm volatile("st.shared.b32 [%0], %1;" ::"r"(stw + dst * 512 + row * 4), "r"(pack_bf16(acc[0][n][k], acc[1][n][k])) : "memory");
        }
    fence_async_smem();
    __syncwarp();
    if (lane < 4) {
      const uint32_t dsm = mapa(su + O_T + (uint32_t)(8 * rank + cp) * 512u, (uint32_t)lane);
      asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], 512, [%2];"
                   ::"r"(dsm), "r"(stw + lane * 512), "r"(mapa(smem_u32(&B.t_full), (uint32_t)lane)) : "memory");
    }
    if (warp == 8 && lane == 0) TFTL(11)
    if (warp >= 12) cluster_arrive();   // early: the exit waits for every arrival of the cluster
  }
  // ------------------------------------------------------------------ output: p = xo Wp^T (warp 1), LN_out + epilogue (warps 4-7)
  if (warp == 1) {
    if (elect_one()) {
      constexpr uint32_t ID64 = idesc_bf16(128, 64);
      mbar_wait(&B.xo_ready, 0);
      tc_fence_after();
#pragma unroll
      for (int k = 0; k < 4; ++k) umma_ts(tmem + T_P, tmem + T_XO + k * 8, desc_k_sw128(su + O_WP + k * 32), ID64, k);
      umma_commit(&B.p_full);
    }
    __syncwarp();
  } else if ((warp >= 4 && warp < 8) || warp >= 12) {   // warps 12-15: the upper half of the output channels in the epilogue
    const int r = (warp & 3) * 32 + lane;
    const uint32_t trow = tmem + ((uint32_t)((warp & 3) * 32) << 16);
    auto up = [](uint32_t w) { return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u)); };
    // LN_out over the 64 channels of the t row (two-pass), xo = bf16(fma((t - mean) rstd, g_out, b_out)) -> TMEM
    if (warp < 8) {
      mbar_wait_cluster(&B.t_full, 0);
      cluster_arrive();   // this CTA's t tile is complete: its peers' copies into it are done (the exit waits for every peer's)
      if (r == 0) TFTL(4)
      const uint32_t trr = su + O_T;
      // channels 8 q .. 8 q + 7 of row r: channel pairs 4 q .. 4 q + 3 of the [32 pairs][128 tokens] t tile
      auto ldt = [&](int q) {
        uint4 v;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v.x) : "r"(trr + (4 * q) * 512 + r * 4));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v.y) : "r"(trr + (4 * q + 1) * 512 + r * 4));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v.z) : "r"(trr + (4 * q + 2) * 512 + r * 4));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v.w) : "r"(trr + (4 * q + 3) * 512 + r * 4));
        return v;
      };
      float2 a0 = make_float2(0.f, 0.f), a1 = a0, a2 = a0, a3 = a0;
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const uint4 t = ldt(q);
        a0 = __fadd2_rn(a0, up(t.x)); a1 = __fadd2_rn(a1, up(t.y)); a2 = __fadd2_rn(a2, up(t.z)); a3 = __fadd2_rn(a3, up(t.w));
      }
      a0 = __fadd2_rn(__fadd2_rn(a0, a1), __fadd2_rn(a2, a3));
      const float mu = __fmul_rn(__fadd_rn(a0.x, a0.y), 1.f / H);
      const float2 nm0 = make_float2(-mu, -mu);
      a0 = a1 = a2 = a3 = make_float2(0.f, 0.f);
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const uint4 t = ldt(q);
        float2 d;
        d = __fadd2_rn(up(t.x), nm0); a0 = __ffma2_rn(d, d, a0);
        d = __fadd2_rn(up(t.y), nm0); a1 = __ffma2_rn(d, d, a1);
        d = __fadd2_rn(up(t.z), nm0); a2 = __ffma2_rn(d, d, a2);
        d = __fadd2_rn(up(t.w), nm0); a3 = __ffma2_rn(d, d, a3);
      }
      a0 = __fadd2_rn(__fadd2_rn(a0, a1), __fadd2_rn(a2, a3));
      const float rs = rsqrt_ftz(__fadd_rn(__fmul_rn(__fadd_rn(a0.x, a0.y), 1.f / H), eps));
      const float2 nm = make_float2(-mu, -mu), rr = make_float2(rs, rs);
#pragma unroll
      for (int q4 = 0; q4 < 4; ++q4) {
        uint32_t o[8];
#pragma unroll
        for (int qq = 0; qq < 2; ++qq) {
          const int q = q4 * 2 + qq;
          const uint4 t = ldt(q);
          const uint32_t gq = su + O_GB + 512 + q * 32;
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
        tmem_st8(trow + T_XO + q4 * 8, o);
      }
      tmem_wait_st();
      tc_fence_before();
      named_bar_sync(1, 128);
      if (r == 0) { mbar_arrive(&B.xo_ready); TFTL(12) }
    }
    // epilogue: y = bf16(x + bf16(sigmoid(g) p)) (k3g inference), staged in the operand window, one TMA store
#if TF_EPI && TF_DIRECT
    {
      // g and x (in TMEM since the front) loaded before the wait for p; after it, p of both channel halves and one wait
      const int c0 = warp < 8 ? 0 : 2;
      uint32_t gv[32], xv16[16], pv[32];
      tmem_ld16(trow + T_G + c0 * 16, *reinterpret_cast<uint32_t(*)[16]>(gv));
      tmem_ld16(trow + T_G + c0 * 16 + 16, *reinterpret_cast<uint32_t(*)[16]>(gv + 16));
      tmem_ld8(trow + T_X + c0 * 8, *reinterpret_cast<uint32_t(*)[8]>(xv16));
      tmem_ld8(trow + T_X + c0 * 8 + 8, *reinterpret_cast<uint32_t(*)[8]>(xv16 + 8));
      tmem_wait_ld();
      float2 sg[16];
#pragma unroll
      for (int c = 0; c < 16; ++c) sg[c] = sigmoid2(make_float2(__uint_as_float(gv[2 * c]), __uint_as_float(gv[2 * c + 1])));
      mbar_wait(&B.p_full, 0);
      tc_fence_after();
      if (r == 0) TFTL(13)
      tmem_ld16(trow + T_P + c0 * 16, *reinterpret_cast<uint32_t(*)[16]>(pv));
      tmem_ld16(trow + T_P + c0 * 16 + 16, *reinterpret_cast<uint32_t(*)[16]>(pv + 16));
      tmem_wait_ld();
      __nv_bfloat16* ytok = yg + tok_of(r) * C;
#pragma unroll
      for (int q4 = 0; q4 < 4; ++q4) {
        uint32_t o[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int c = q4 * 4 + k;
          const float2 pr = __fmul2_rn(sg[c], make_float2(__uint_as_float(pv[2 * c]), __uint_as_float(pv[2 * c + 1])));
          const uint32_t pb = pack_bf16(pr.x, pr.y);
          const float2 y = __fadd2_rn(make_float2(bf16lo(xv16[c]), bf16hi(xv16[c])), make_float2(bf16lo(pb), bf16hi(pb)));
          o[k] = pack_bf16(y.x, y.y);
        }
        *reinterpret_cast<uint4*>(ytok + (c0 * 2 + q4) * 8) = make_uint4(o[0], o[1], o[2], o[3]);
      }
      if (r == 0) TFTL(5)
    }
#else
    {
      mbar_wait(&B.p_full, 0);
      tc_fence_after();
      if (r == 0) TFTL(13)
      __nv_bfloat16* ytok = yg + (size_t)(OUT ? (i0 + (r >> 4)) * L + j0 + (r & 15) : (i0 + (r & 7)) * L + j0 + (r >> 3)) * C;
#pragma unroll 1
      for (int cq = warp < 8 ? 0 : 2; cq < (warp < 8 ? 2 : 4); ++cq) {
        uint32_t pv[16], gv[16], xv8[8];
        tmem_ld16(trow + T_P + cq * 16, pv);
        tmem_ld16(trow + T_G + cq * 16, gv);
        tmem_ld8(trow + T_X + cq * 8, xv8);
        tmem_wait_ld();
#pragma unroll
        for (int q2 = 0; q2 < 2; ++q2) {
          const int q = cq * 2 + q2;
          const uint32_t a = su + O_T + sw128(r, q);   // y staged over the t tile (read by LN_out, complete): other warps may still read the operands
          const uint32_t xw[4] = {xv8[4 * q2], xv8[4 * q2 + 1], xv8[4 * q2 + 2], xv8[4 * q2 + 3]};
          uint32_t o[4];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const int c = q2 * 8 + 2 * k;
            const float2 pp = make_float2(__uint_as_float(pv[c]), __uint_as_float(pv[c + 1]));
            const float2 sg = sigmoid2(make_float2(__uint_as_float(gv[c]), __uint_as_float(gv[c + 1])));
            const float2 pr = __fmul2_rn(sg, pp);
            const float2 prb = make_float2(__uint_as_float(pack_bf16(pr.x, pr.y) << 16), __uint_as_float(pack_bf16(pr.x, pr.y) & 0xffff0000u));
            const float2 y = __fadd2_rn(make_float2(bf16lo(xw[k]), bf16hi(xw[k])), prb);
            o[k] = pack_bf16(y.x, y.y);
          }
#if TF_DIRECT
          *reinterpret_cast<uint4*>(ytok + q * 8) = make_uint4(o[0], o[1], o[2], o[3]);
#else
          sts128(a, o[0], o[1], o[2], o[3]);
#endif
        }
      }
#if TF_DIRECT
      if (r == 0) TFTL(5)
#else
      fence_async_smem();
      named_bar_sync(1, 128);
      if (r == 0) {
        tma_store_3d(&m.y, sm + O_T, 0, xc1, xc2);
        bulk_commit();
        bulk_wait_read<0>();   // the smem read; the kernel end completes the write
        TFTL(5)
      }
#endif
    }
#endif
  }
  // exit only when every CTA of the cluster has its t tile (its DSMEM bulk copies read this CTA's shared memory)
  if (warp < 4 || (warp >= 8 && warp < 12)) cluster_arrive();
  tc_fence_before();
  __syncthreads();
  cluster_wait();
  if (tid == 0) TFTL(14)
  if (warp == 2) tmem_dealloc(tmem, 512);
}

template <int OUT, int MASK>
void* kfn() { return reinterpret_cast<void*>(tf_kernel<OUT, MASK>); }

}  // namespace tf

// Co-residency of the 16 clusters (the counter waits spin): the launch is used only when this holds. tf_clusters: the occupancy
// query's count (negative: the CUDA error of the attribute set / the query).
int tf_clusters() {
  auto kern = tf::tf_kernel<1, 0>;
  cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, tf::SMEM);
  if (e != cudaSuccess) { cudaGetLastError(); return -1000 - (int)e; }
  e = cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  if (e != cudaSuccess) { cudaGetLastError(); return -2000 - (int)e; }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(tf::NCTA); cfg.blockDim = dim3(512); cfg.dynamicSmemBytes = tf::SMEM;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = tf::CL; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  int n = 0;
  e = cudaOccupancyMaxActiveClusters(&n, kern, &cfg);
  if (e != cudaSuccess) { cudaGetLastError(); return -(int)e; }
  return n;
}
bool tf_supported() {
  static int ok = -1;
  if (ok < 0) ok = tf_clusters() >= tf::NCTA / tf::CL ? 1 : 0;
  return ok == 1;
}


// x [L L, 64] bf16 (one sample, L = 128); the four front matrices [64, 64] row-major; wg, wp [64, 64] bf16 contiguous; tokmask [L]
// uint8 / bool or None; LayerNorm affines fp32 [64]; planes [128, L, L] bf16 scratch; y [L L, 64]. dir 1 outgoing, 2 incoming.
void tf_forward(torch::Tensor x, torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg, torch::Tensor wg,
                torch::Tensor wp, c10::optional<torch::Tensor> tokmask, torch::Tensor gi, torch::Tensor bi, torch::Tensor go,
                torch::Tensor bo, torch::Tensor planes, torch::Tensor y, int64_t dir, double eps, double rf3) {
  using namespace tf;
  TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && planes.is_contiguous() && x.size(0) == L * L && x.size(1) == C &&
              planes.numel() == 2 * H * L * L, "tf_forward: shapes");
  TORCH_CHECK(dir == 1 || dir == 2, "tf_forward: one direction");
  const auto bf = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  Maps m;
  const torch::Tensor* ws[6] = {&wlg, &wl, &wrg, &wr, &wg, &wp};
  for (int k = 0; k < 6; ++k) {
    const torch::Tensor& w = *ws[k];
    TORCH_CHECK(w.scalar_type() == at::kBFloat16 && w.size(0) == 64 && w.size(1) == 64 && w.stride(1) == 1 && (w.stride(0) * 2) % 16 == 0,
                "tf_forward: [64, 64] bf16 weights with unit column stride");
    m.w[k] = tmap::make(w.data_ptr(), bf, {64, 64}, {(uint64_t)w.stride(0) * 2}, {64, 64}, CU_TENSOR_MAP_SWIZZLE_128B);
  }
  const bool out = dir == 1;
  // x / y: dims (channel, inner token index, outer token index); outgoing inner = j, incoming inner = i (rows of the box q-major)
  const std::vector<uint64_t> xs = out ? std::vector<uint64_t>{(uint64_t)C * 2, (uint64_t)L * C * 2}
                                       : std::vector<uint64_t>{(uint64_t)L * C * 2, (uint64_t)C * 2};
  const std::vector<uint32_t> xb = out ? std::vector<uint32_t>{64, 16, 8} : std::vector<uint32_t>{64, 8, 16};
  m.x = tmap::make(x.data_ptr(), bf, {(uint64_t)C, (uint64_t)L, (uint64_t)L}, xs, xb, CU_TENSOR_MAP_SWIZZLE_128B);
  m.y = tmap::make(y.data_ptr(), bf, {(uint64_t)C, (uint64_t)L, (uint64_t)L}, xs, xb, CU_TENSOR_MAP_SWIZZLE_128B);
  // planes [128][L L] scratch, BLOCKED (see the front epilogue): (element in a 128-element block, block, channel)
  const std::vector<uint64_t> ps = {256, (uint64_t)L * L * 2};
  const std::vector<uint64_t> pd = {128, (uint64_t)L * L / 128, 128};
  m.pst = tmap::make(planes.data_ptr(), bf, pd, ps, {128, 1, 64}, CU_TENSOR_MAP_SWIZZLE_NONE);
  m.pld = tmap::make(planes.data_ptr(), bf, pd, ps, {128, 16, CPC / TF_OPL}, CU_TENSOR_MAP_SWIZZLE_NONE);
  m.pldb = tmap::make(planes.data_ptr(), bf, pd, ps, {128, 32, CPC / TF_OPL}, CU_TENSOR_MAP_SWIZZLE_NONE);
  const uint8_t* mk = nullptr;
  if (tokmask.has_value()) {
    TORCH_CHECK(tokmask->numel() == L && tokmask->element_size() == 1 && tokmask->is_contiguous(), "tf_forward: token mask [L]");
    mk = reinterpret_cast<const uint8_t*>(tokmask->data_ptr());
  }
  void* kern = out ? (mk ? kfn<1, 1>() : kfn<1, 0>()) : (mk ? kfn<0, 1>() : kfn<0, 0>());
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(NCTA); cfg.blockDim = dim3(512); cfg.dynamicSmemBytes = SMEM; cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute lat[1];
  lat[0].id = cudaLaunchAttributeClusterDimension;
  lat[0].val.clusterDim.x = CL; lat[0].val.clusterDim.y = 1; lat[0].val.clusterDim.z = 1;
  cfg.attrs = lat; cfg.numAttrs = 1;
#ifdef TF_EMPTY
  if (TF_EMPTY >= 2) cfg.dynamicSmemBytes = 1024;
  if (TF_EMPTY >= 3) cfg.numAttrs = 0;
#endif
  const float* gip = gi.data_ptr<float>(); const float* bip = bi.data_ptr<float>();
  const float* gop = go.data_ptr<float>(); const float* bop = bo.data_ptr<float>();
  float e = (float)eps;
  void* plp = planes.data_ptr(); void* ypp = y.data_ptr(); const void* xpp = x.data_ptr();
  float rf3f = (float)rf3;
  void* args[] = {&m, &mk, &gip, &bip, &gop, &bop, &e, &plp, &ypp, &xpp, &rf3f};
  C10_CUDA_CHECK(cudaLaunchKernelExC(&cfg, kern, args));
}

torch::Tensor tf_tl() {
#ifdef TF_TL
  torch::Tensor h = torch::empty({128, 16}, torch::kInt64);
  cudaMemcpyFromSymbol(h.data_ptr(), g_tftl, sizeof(unsigned long long) * 128 * 16);
  return h;
#else
  return torch::empty({0});
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mm) {
  mm.def("tf_forward", &tf_forward);
  mm.def("tf_supported", &tf_supported);
  mm.def("tf_clusters", &tf_clusters);
  mm.def("tf_tl", &tf_tl);
}
