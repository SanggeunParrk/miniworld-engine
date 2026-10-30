// b7g.cu -- b7r.cu (TriMul B7 on B200, d_pair C = 128) generalised over the number of 64-channel plane chunks NCH:
// 8 = bidirectional (b7r), 4 = one direction; the plane gradient comes as one channel-major tensor [64 NCH, M].
// b7r.cu's description, with 8 -> NCH:
// Same math as b7.cu (see there). One cooperative launch, one CTA per SM:
//   source CTAs  (8 per source group; CTA (g, c) owns packed-W1 chunk c = plane channels 64c .. 64c+63, rows 128c .. 128c+127)
//     for every tile of its group: pre = xn W1_c^T, dg/dp (fragment layout) -> dgp tile [128 tok][128] bf16,
//     dW_c += dgp^T xn (TMEM, persistent), dgp -> ring slot (TMA store) + flag = 1.
//   consumer CTAs (pairs, clusters of two): for each of its tiles: NCH watcher lanes poll the tile's chunk
//     flags together and hand them to the loader through mbarriers; stream (dgp_c, W1_c) K-blocks through a 4-stage ring (each
//     CTA of the pair loads its own dgp K-block and one 64-column half of the W1 / Wg K-block, multicast to both: the pair's MMAs
//     both release a stage, tcgen05.commit multicast),
//     dxn = bf16( sum_c dgp_c W1_c + dGout Wg )  (fp32 accumulate over K = 1024 + 128, one rounding, as on H100),
//     LN_in backward + residual -> dx (TMA store), dgi / dbi (reduce-scatter over each warp's rows, register-accumulated);
//     a flag is reset to 0 once its K-blocks have landed in shared memory (a returner lane, per half tile: fence.proxy.async +
//     release pattern; the source acquires the slot and fences the proxy before its TMA store). Odd tile counts (not reached
//     with L % 128 == 0) run one CTA per tile without the pairing. A published
//     flag carries the group-local tile index + 1, so a consumer never takes an older tile still waiting in the same slot;
//     sources publish two tiles per release fence.
// Flags are 0 again when the kernel ends, so launches and CUDA-graph replays need no reset.
#include "sm100.cuh"
#include "tmap.h"

using namespace sm100;

#ifndef B7_WARPMMA
#define B7_WARPMMA 1   // consumer MMA loop run by the whole warp (uniform descriptors), one elected lane issues
#endif
#ifndef B7_SIG
#define B7_SIG 0
#endif
namespace b7g {

constexpr int C = 128, TOK = 128;
constexpr int XT = TOK * C * 2;            // 32 KB tiles ([128][128] bf16, 2 K-blocks)
constexpr int DAT = 64 * TOK * 2;          // 16 KB
#ifndef B7_RD
#define B7_RD 12
#endif
constexpr int RD = B7_RD;                  // ring slots per source group
// source layout
constexpr int NX = 3;                      // source xn buffers (xn is read by pre(j) and, after EW(j), by dW(j))
constexpr int S_W = 0, S_X = S_W + XT, S_DA = S_X + NX * XT, S_DGP = S_DA + 2 * DAT, S_END = S_DGP + 2 * XT;
// consumer layout
#ifndef B7_NST
#define B7_NST 4
#endif
// consumer ring: NST stages of (A K-block 16 KB + B K-block 16 KB); per tile 2 stages (dGout, Wg) then 16 (dgp_c, W1_c) stages
constexpr int NST = B7_NST, STB = XT;
// + two x tiles (dx written in place, then TMA-stored: the next tile's x lands while this one is worked on and stored) and one
// dy tile (read only by the output pass; refilled while the next tile's statistics run)
constexpr int C_ST = 0, C_X = C_ST + NST * STB, C_DY = C_X + 2 * XT, C_END = C_DY + XT;
constexpr int O_BAR = (S_END > C_END ? S_END : C_END);
constexpr int O_GI = O_BAR + 512;          // gi [128] fp32 (consumer)
constexpr int SMEM = O_GI + 512 + 2048;     // + [2][128] x 2 row-sum exchange (consumer epilogue)
static_assert(SMEM <= 232448, "smem");
constexpr uint32_t T_PRE = 0, T_DW = 256, T_ACC = 0;
constexpr uint32_t ID_PRE = idesc_bf16_mj(128, 128, 0, 0);
constexpr uint32_t ID_DW = idesc_bf16_mj(128, 128, 1, 1);
constexpr uint32_t ID_DX = idesc_bf16_mj(128, 128, 0, 1);

struct Bars {
  // source
  uint64_t w_full, x_full[NX], x_free[NX], da_full[2], da_free[2], pre_full[2], pre_free[2], dgp_ready[2], dgp_free[2];
  // consumer
  uint64_t st_full[NST], st_empty[NST], acc_full[2], acc_empty[2], cx_full[2], cx_free[2], dy_full, dy_free, fready[2][8], fdone[2],
      landed[2][2], retdone[2];
  uint32_t tmem;
};
static_assert(sizeof(Bars) <= 512, "barrier block");

DEV float ex2_ftz(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float rcp_ftz(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float rsqrt_ftz(float x) { float y; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float tanh_approx(float x) { float y; asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float sigmoid_kit(float g) { return rcp_ftz(__fadd_rn(1.f, ex2_ftz(__fmul_rn(-1.4426950408889634f, g)))); }
DEV float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
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
DEV float unpack_v(uint32_t w, int hi) {
  uint32_t r;
  if (hi) asm volatile("and.b32 %0, %1, 0xffff0000;" : "=r"(r) : "r"(w)); else asm volatile("shl.b32 %0, %1, 16;" : "=r"(r) : "r"(w));
  return __uint_as_float(r);
}
DEV uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory");
  return v;
}
DEV float4 lds128f(uint32_t ad) {
  float4 v; asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(ad) : "memory"); return v;
}
DEV void red_add(float* p, float v) { asm volatile("red.global.add.f32 [%0], %1;" ::"l"(p), "f"(v) : "memory"); }
DEV void red_add4(float* p, float a, float b, float c, float d) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}
DEV uint32_t ld_acquire(const uint32_t* p) { uint32_t v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
DEV void st_relaxed(uint32_t* p, uint32_t v) { asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
DEV void st_release(uint32_t* p, uint32_t v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
// Returning a ring slot (consumer): after fence.proxy.async.global + fence.acq_rel.gpu (the slot's TMA reads have landed), a
// relaxed store completes the release pattern; the source acquires it and fences the proxy before its TMA store.
DEV void flag_return(uint32_t* p) { asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(0u) : "memory"); }
DEV void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
// multicast to both CTAs of the pair: same smem offset / mbarrier offset in each CTA of ctaMask
DEV void tma_load_3d_mc(void* dst, const CUtensorMap* m, uint64_t* bar, int c0, int c1, int c2, uint16_t mask, uint64_t hint) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
               " [%0], [%1, {%3, %4, %5}], [%2], %6, %7;"
               ::"r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(m)), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "r"(c2), "h"(mask), "l"(hint)
               : "memory");
}
DEV void umma_commit_mc(uint64_t* bar, uint16_t mask) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
               ::"r"(smem_u32(bar)), "h"(mask) : "memory");
}
DEV void fence_acq_rel_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
// a relaxed flag store after fence_acq_rel_gpu: together a release pattern (PTX memory model), so one fence serves several flags
DEV void flag_store(uint32_t* p, uint32_t v) { asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
#ifdef MBAR_DEBUG
__device__ volatile int g_prog[160][8];
#define PROG(k, v) (g_prog[blockIdx.x][k] = (v))
#else
#define PROG(k, v) ((void)0)
#endif
DEV void spin_until(const uint32_t* p, uint32_t v) {
#ifdef MBAR_DEBUG
  long long n = 0;
  while (ld_acquire(p) != v) {
    __nanosleep(64);
    if (n == 3000000 && blockIdx.x == 96) {
      for (int b = 0; b < 148; ++b)
        printf("PROG %d: %d %d %d %d %d %d\n", b, g_prog[b][0], g_prog[b][1], g_prog[b][2], g_prog[b][3], g_prog[b][4], g_prog[b][5]);
    }
    if (++n == 2000000) printf("FLAG HANG blk %d thr %d flag_idx %lld want %u have %u\n", blockIdx.x, threadIdx.x, (long long)(p - (const uint32_t*)0) , v, ld_acquire(p));
  }
#else
  while (ld_acquire(p) != v) __nanosleep(64);
#endif
}

// Pair mask from the [B L] token mask: pair (i, j) of sample s (token (s L + i) L + j, b-major) is kept when tokens s L + i and
// s L + j are. A 128-token tile lies in one row i (L % 128 == 0); (s, i, j0) of a CTA's tile is advanced incrementally (no integer
// division on the elementwise warps' path: three divisions per token cost b7g ~15 %).
struct PairIdx {
  int s, i, j, dq, dr;      // dq, dr: the CTA's tile step (step rows = dq L + dr), split once
  DEV void init(int row0, int step, int L) { s = row0 / (L * L); i = (row0 / L) % L; j = row0 % L; dq = step / L; dr = step % L; }
  DEV void advance(int L) {   // no loop over L-sized steps (b7m steps 148 x 128 tokens: ~25 per tile at L768)
    j += dr; i += dq;
    if (j >= L) { j -= L; ++i; }
    while (i >= L) { i -= L; ++s; }
  }
  // the two tokens' masks of this thread (tile offsets tok, tok + 8): both loads issued together, no short-circuit
  DEV void mask2(const uint8_t* tm, int tok, int L, float& m0, float& m1) const {
    if (tm == nullptr) { m0 = m1 = 1.f; return; }
    const uint8_t* row = tm + s * L;
    const uint32_t ri = __ldg(row + i), a0 = __ldg(row + j + tok), a1 = __ldg(row + j + tok + 8);
    m0 = (ri & a0) ? 1.f : 0.f;
    m1 = (ri & a1) ? 1.f : 0.f;
  }
};

struct Args {
  const __nv_bfloat16 *x, *dy;
  const uint8_t* tmask;     // [B L] token mask (pair (i, j) of sample s kept when tokens s L + i and s L + j are), or null
  const float* gi;
  __nv_bfloat16* dx;
  float *dw1, *lnpart;      // dw1 [SG][128 NCH][128]: one dW1 partial per source group (rows 128 c + r of chunk c), summed in a fixed
                            // order by grad_reduce; lnpart [consumers][dgi 128 | dbi 128]: one row per consumer CTA
  uint32_t* flags;          // [SG][RD][NCH]
  int tiles, L, SG, NC;
  int pair;                 // consumers in CTA pairs sharing the W1 / Wg stream (even tile count); 0: one CTA per tile, own stream
  float eps;
};

#ifdef PROF
#define TLS(ev) { if (blockIdx.x == 0 && (tid & 31) == 0 && (j == 10 || j == 11)) g_prof[153 + j - 10][(ev) >> 3][(ev) & 7] = clock64(); }
#define TLC(ev) { if (k == 0 && (tid & 31) == 0 && jj == 3) g_prof[152][(ev) >> 3][(ev) & 7] = clock64(); }
#define TLD(ev) { if (k == 0 && (tid & 31) == 0 && (jj == 3 || jj == 4)) g_prof[155 + jj - 3][(ev) >> 3][(ev) & 7] = clock64(); }
#else
#define TLC(ev)
#define TLS(ev)
#define TLD(ev)
#endif
template <int NCH>
__global__ void __launch_bounds__(384, 1)
    b7g_kernel(const __grid_constant__ CUtensorMap mxn, const __grid_constant__ CUtensorMap mda,
               const __grid_constant__ CUtensorMap mw1, const __grid_constant__ CUtensorMap mring, const __grid_constant__ CUtensorMap mgo,
               const __grid_constant__ CUtensorMap mwg, const __grid_constant__ CUtensorMap mdx, const __grid_constant__ CUtensorMap mw1k,
               const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mdy, const Args a) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int NS = NCH * a.SG;
  const bool is_src = (int)blockIdx.x < NS;
  if (tid == 0 && (su & 1023u)) asm volatile("trap;");

  if (tid == 0) {
    mbar_init(&B.w_full, 1);
    for (int b = 0; b < 2; ++b) {
      mbar_init(&B.da_full[b], 1); mbar_init(&B.da_free[b], 8);
      mbar_init(&B.pre_full[b], 1); mbar_init(&B.pre_free[b], 8); mbar_init(&B.dgp_ready[b], 1); mbar_init(&B.dgp_free[b], 2);
      mbar_init(&B.acc_full[b], 1); mbar_init(&B.acc_empty[b], 8);
    }
    for (int b = 0; b < NX; ++b) { mbar_init(&B.x_full[b], 1); mbar_init(&B.x_free[b], 1); }
    for (int b = 0; b < NST; ++b) { mbar_init(&B.st_full[b], 1); mbar_init(&B.st_empty[b], a.pair ? 2 : 1); }   // consumer pairs: both CTAs' MMAs release a stage
    for (int b = 0; b < 2; ++b) { mbar_init(&B.cx_full[b], 1); mbar_init(&B.cx_free[b], 1); }
    mbar_init(&B.dy_full, 1); mbar_init(&B.dy_free, 1);
    for (int b = 0; b < 2; ++b) {
      mbar_init(&B.fdone[b], 1);
      for (int c = 0; c < 8; ++c) mbar_init(&B.fready[b][c], 1);
      mbar_init(&B.landed[b][0], 1); mbar_init(&B.landed[b][1], 1); mbar_init(&B.retdone[b], 1);
    }
    fence_mbar_init();
  }
  if (tid < 128) reinterpret_cast<float*>(sm + O_GI)[tid] = a.gi[tid];
  if (warp == 2) { tmem_alloc(&B.tmem, 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  cluster_arrive();                                          // the pair's barriers are initialised before any multicast lands
  cluster_wait();
  const uint32_t tmem = B.tmem;
  PROF_BEGIN

  if (is_src) {
    // ================================================================== SOURCE
    const int g = blockIdx.x / NCH, c = blockIdx.x % NCH;
    const int n_local = a.tiles > g ? (a.tiles - g + a.SG - 1) / a.SG : 0;   // tiles t = g + SG j
    const int chbase = c * 64;
    if (warp == 0) {
      if (lane == 0) {
        mbar_expect_tx(&B.w_full, XT);
        tma_load_2d(sm + S_W, &mw1, &B.w_full, 0, 128 * c, EVICT_LAST);
        tma_load_2d(sm + S_W + XT / 2, &mw1, &B.w_full, 64, 128 * c, EVICT_LAST);
        for (int j = 0; j < n_local; ++j) {
          const int b = j & 1, row0 = (g + a.SG * j) * TOK;
          const int bx = j % NX;
          if (j >= NX) PW(0, mbar_wait(&B.x_free[bx], ((j / NX) - 1) & 1));
          PROG(0, j);
#ifdef B7_NOXN
          if (j >= NX) mbar_arrive(&B.x_full[bx]); else   // energy ablation: stale xn after the first NX tiles
#endif
          {
          mbar_expect_tx(&B.x_full[bx], XT);
          tma_load_2d(sm + S_X + bx * XT, &mxn, &B.x_full[bx], 0, row0, EVICT_NORMAL);
          tma_load_2d(sm + S_X + bx * XT + XT / 2, &mxn, &B.x_full[bx], 64, row0, EVICT_NORMAL);
          }
          if (j >= 2) PW(1, mbar_wait(&B.da_free[b], ((j >> 1) - 1) & 1));
          mbar_expect_tx(&B.da_full[b], DAT);
          tma_load_2d(sm + S_DA + b * DAT, &mda, &B.da_full[b], row0, chbase, EVICT_FIRST);
          tma_load_2d(sm + S_DA + b * DAT + DAT / 2, &mda, &B.da_full[b], row0 + 64, chbase, EVICT_FIRST);
        }
      }
    } else if (warp == 1) {
      if (lane == 0) {
        mbar_wait(&B.w_full, 0);
        const uint32_t sw = su + S_W;
        auto issue_pre = [&](int j) {
          const int b = j & 1;
          const uint32_t sx = su + S_X + (j % NX) * XT;
          PW(0, mbar_wait(&B.x_full[j % NX], (j / NX) & 1));
          if (j >= 2) PW(1, mbar_wait(&B.pre_free[b], ((j >> 1) - 1) & 1));
          tc_fence_after();
#pragma unroll
          for (int k = 0; k < 8; ++k)
            umma_ss(tmem + T_PRE + b * 128, desc_k_sw128(sx + (k >> 2) * (XT / 2) + (k & 3) * 32),
                    desc_k_sw128(sw + (k >> 2) * (XT / 2) + (k & 3) * 32), ID_PRE, k != 0);
          umma_commit(&B.pre_full[b]);
        };
        if (n_local > 0) issue_pre(0);
        for (int j = 0; j < n_local; ++j) {
          const int b = j & 1;
          const uint32_t sx = su + S_X + (j % NX) * XT, sg = su + S_DGP + b * XT;
          if (j + 1 < n_local) issue_pre(j + 1);             // the next tile's pre-activations run while EW works on tile j
          PROG(1, j);
          PW(2, mbar_wait(&B.dgp_ready[b], (j >> 1) & 1)); TLS(7);
          tc_fence_after();
#pragma unroll
          for (int k = 0; k < 8; ++k)
            umma_ss(tmem + T_DW, desc_mn_sw128(sg + k * 2048, XT / 2), desc_mn_sw128(sx + k * 2048, XT / 2), ID_DW, (j | k) != 0);
          umma_commit(&B.x_free[j % NX]);
          umma_commit(&B.dgp_free[b]);
          PROG(2, j);
        }
      }
    } else if (warp == 2 || warp == 3) {
      if (lane == 0) {
        // ring publishers: warp 3 takes the tiles in dgp buffer 0 (even j), warp 2 those in buffer 1 (odd j). Per tile: slot free ->
        // TMA store -> smem read (buffer back to EW) -> ... ; the flag of a tile is set after the publisher's next store is issued,
        // so waiting for the global writes to complete stays off the EW path.
        // One release fence publishes two tiles: every second store, once the two previous stores' writes are complete,
        // fence.acq_rel.gpu then relaxed flag stores (each a release pattern with the fence); a fence costs ~2k clk.
        const int b = warp == 3 ? 0 : 1;
        int ps0 = 0, pj0 = 0, ps1 = 0, pj1 = 0, np = 0;         // stored, not yet published (oldest first)
        for (int j = b, i = 0; j < n_local; j += 2, ++i) {
          const int slot = (g * RD + j % RD) * NCH + c;
          PW(1, spin_until(a.flags + slot, 0u)); TLS(1);            // consumer of tile j - RD has taken the slot (acquire)
          fence_proxy_async_global();                               // ... ordered before this thread's async-proxy writes to it
          PW(0, mbar_wait(&B.dgp_ready[b], (j >> 1) & 1)); TLS(0);
          PW(3, tma_store_2d(&mring, sm + S_DGP + b * XT, 0, slot * TOK));
          PW(3, tma_store_2d(&mring, sm + S_DGP + b * XT + XT / 2, 64, slot * TOK));
          bulk_commit();
          PW(2, bulk_wait_read<0>());
          mbar_arrive(&B.dgp_free[b]); TLS(2);                      // smem read: EW may refill the buffer
          if (np == 2) {                                            // the two stores before this one: writes complete (at L2)
            PW(5, bulk_wait<1>());
            fence_proxy_async_global();
            PW(4, fence_acq_rel_gpu());
            flag_store(a.flags + ps0, (uint32_t)(pj0 + 1));   // generation tag: group-local tile index + 1
            flag_store(a.flags + ps1, (uint32_t)(pj1 + 1));
            np = 0;
          }
          if (np == 0) { ps0 = slot; pj0 = j; } else { ps1 = slot; pj1 = j; }
          ++np;
          TLS(3);
        }
        if (np > 0) {
          bulk_wait<0>();
          fence_proxy_async_global();
          fence_acq_rel_gpu();
          flag_store(a.flags + ps0, (uint32_t)(pj0 + 1));
          if (np == 2) flag_store(a.flags + ps1, (uint32_t)(pj1 + 1));
        }
      }
    } else if (warp >= 4) {
      // dg / dp: warps 4-7 token group hh 0, warps 8-11 hh 1 of each warp quarter
      const int q = warp & 3, hh = (warp - 4) >> 2, t4 = lane & 3, tr = lane >> 2;
      const int tok0 = q * 32 + hh * 16;
      float mk0 = 0.f, mk1 = 0.f;
      PairIdx pix;
      pix.init(g * TOK, a.SG * TOK, a.L);
      if (n_local > 0) pix.mask2(a.tmask, tok0 + tr, a.L, mk0, mk1);
      for (int j = 0; j < n_local; ++j) {
        const int b = j & 1;
        const float mA = mk0, mB = mk1;
        if (warp == 4) TLS(4);
        PW(0, mbar_wait(&B.pre_full[b], (j >> 1) & 1));
        PW(1, mbar_wait(&B.da_full[b], (j >> 1) & 1));
        if (j >= 2) PW(2, mbar_wait(&B.dgp_free[b], ((j >> 1) - 1) & 1));
        if (warp == 4) TLS(5);
        tc_fence_after();
        const uint32_t ta = tmem + T_PRE + b * 128 + ((uint32_t)tok0 << 16);
        uint32_t da[16];
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          const int mm = lane >> 3, ch = 16 * jj + 8 * (mm & 1) + (lane & 7), qc = ((tok0 & 63) >> 3) + (mm >> 1);
          ldsm_x4_trans(su + S_DA + b * DAT + (tok0 >> 6) * (DAT / 2) + sw128(ch, qc), da[4 * jj], da[4 * jj + 1], da[4 * jj + 2], da[4 * jj + 3]);
        }
        if (warp == 4) TLS(8);
        uint32_t gall[2][16], pall[2][16];
        tmem_ld16x256_x4(ta, gall[0]);
        tmem_ld16x256_x4(ta + 64, pall[0]);
        tmem_wait_ld();
        if (warp == 4) TLS(9);
        tmem_ld16x256_x4(ta + 32, gall[1]);                 // half 1 in flight during the half-0 math
        tmem_ld16x256_x4(ta + 96, pall[1]);
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.da_free[b]);          // dA is in registers
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          if (half == 1) {
            tmem_wait_ld();
            tc_fence_before();
            __syncwarp();
            if (lane == 0) mbar_arrive(&B.pre_free[b]);
          }
          uint32_t odg[8], odp[8];
#pragma unroll
          for (int jx = 0; jx < 4; ++jx) {
            const int J = 4 * half + jx;
#pragma unroll
            for (int tb = 0; tb < 2; ++tb) {
              const float2 dA = up(da[4 * (J >> 1) + (J & 1) + 2 * tb]);
              const float mkk = tb ? mB : mA;
              // packed f32x2, same per-lane IEEE sequence: g = sigmoid(bf16(pre_g)), p = bf16(pre_p), m = dA mask
              const uint32_t gw = pack_bf16(__uint_as_float(gall[half][4 * jx + 2 * tb]), __uint_as_float(gall[half][4 * jx + 2 * tb + 1]));
              const uint32_t pw = pack_bf16(__uint_as_float(pall[half][4 * jx + 2 * tb]), __uint_as_float(pall[half][4 * jx + 2 * tb + 1]));
#if B7_SIG == 2
              float2 gg;
              {
                const float2 h = __fmul2_rn(up(gw), make_float2(0.5f, 0.5f));
                gg = __ffma2_rn(make_float2(tanh_approx(h.x), tanh_approx(h.y)), make_float2(0.5f, 0.5f), make_float2(0.5f, 0.5f));
              }
#elif B7_SIG == 0
              const float2 t = __fmul2_rn(up(gw), make_float2(-1.4426950408889634f, -1.4426950408889634f));
#else
              float2 t = __fmul2_rn(up(gw), make_float2(-1.4426950408889634f, -1.4426950408889634f));
              t.x = fminf(t.x, 126.f); t.y = fminf(t.y, 126.f);
#endif
#if B7_SIG != 2
              const float2 dn = __fadd2_rn(make_float2(ex2_ftz(t.x), ex2_ftz(t.y)), make_float2(1.f, 1.f));
#endif
#if B7_SIG == 2
#elif B7_SIG == 0
              const float2 gg = make_float2(rcp_ftz(dn.x), rcp_ftz(dn.y));
#else
              float2 gg = make_float2(__int_as_float(0x7EF311C3 - __float_as_int(dn.x)), __int_as_float(0x7EF311C3 - __float_as_int(dn.y)));
              {
                const float2 nd = make_float2(-dn.x, -dn.y);
#pragma unroll
                for (int it = 0; it < 3; ++it) gg = __fmul2_rn(gg, __ffma2_rn(nd, gg, make_float2(2.f, 2.f)));
              }
#endif
              const float2 m = __fmul2_rn(dA, make_float2(mkk, mkk));
              const float2 rg = __fmul2_rn(__fmul2_rn(__fmul2_rn(m, up(pw)), gg), __fadd2_rn(make_float2(1.f, 1.f), make_float2(-gg.x, -gg.y)));
              const float2 rp = __fmul2_rn(m, gg);
              odg[2 * jx + tb] = pack_bf16(rg.x, rg.y);
              odp[2 * jx + tb] = pack_bf16(rp.x, rp.y);
            }
          }
#pragma unroll
          for (int x2 = 0; x2 < 2; ++x2) {
            const int m = lane >> 3, rr = lane & 7;
            const int tok = tok0 + rr + 8 * (m & 1), J = 4 * half + 2 * x2 + (m >> 1);
            stsm_x4(su + S_DGP + b * XT + sw128(tok, J), odg[4 * x2], odg[4 * x2 + 1], odg[4 * x2 + 2], odg[4 * x2 + 3]);
            stsm_x4(su + S_DGP + b * XT + XT / 2 + sw128(tok, J), odp[4 * x2], odp[4 * x2 + 1], odp[4 * x2 + 2], odp[4 * x2 + 3]);
          }
          if (warp == 4) TLS(10 + half);
        }
        fence_async_smem();
        if (warp == 4) TLS(12);
        named_bar_sync(1, 256);
        if (tid == 128) mbar_arrive(&B.dgp_ready[b]);
        if (warp == 4) TLS(6);
        if (j + 1 < n_local) {                               // next tile's mask (after the proxy fence, which would wait on it)
          pix.advance(a.L);
          pix.mask2(a.tmask, tok0 + tr, a.L, mk0, mk1);
        }
      }
    }
    // flush dW_c
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    if (warp >= 4) {
      const int q = warp & 3, part = (warp - 4) >> 2, row = q * 32 + lane;
      for (int n = 0; n < 2; ++n) {                           // start block rotated by group: the SG CTAs of chunk c spread out
        const int cb = 2 * ((n + g) & 1) + part;
        uint32_t v[32];
        tmem_ld32(tmem + T_DW + ((uint32_t)(q * 32) << 16) + cb * 32, v);
        tmem_wait_ld();
        if (n_local == 0) {                                   // a group without tiles (tiles < SG): its TMEM was never written
#pragma unroll
          for (int e = 0; e < 32; ++e) v[e] = 0u;
        }
        float* dst = a.dw1 + ((size_t)g * (128 * NCH) + 128 * c + row) * C + cb * 32;   // this group's partial (no atomics)
#pragma unroll
        for (int k = 0; k < 32; k += 4)
          *reinterpret_cast<float4*>(dst + k) = make_float4(__uint_as_float(v[k]), __uint_as_float(v[k + 1]), __uint_as_float(v[k + 2]),
                                                             __uint_as_float(v[k + 3]));
      }
    }
  } else {
    // ================================================================== CONSUMER
    const int k = blockIdx.x - NS;
    // consumer pairs (clusters of two): pair pk takes tile pairs u = pk + NP jj, its CTA r = k & 1 tile 2 u + r; the pair streams
    // the W1 / Wg K-blocks once, each CTA loading one 64-column half multicast to both
    // (odd tile count: a.pair = 0, CTA k takes tiles k + NC jj and streams the whole K-blocks itself)
    const bool pair = a.pair != 0;
    const int NP = a.NC >> 1, pk = k >> 1, rk = pair ? (k & 1) : 0, tp = a.tiles >> 1;
    const int n_local = pair ? (tp > pk ? (tp - pk + NP - 1) / NP : 0) : (a.tiles > k ? (a.tiles - k + a.NC - 1) / a.NC : 0);
    auto tile_of = [&](int jj) { return pair ? 2 * (pk + NP * jj) + rk : k + a.NC * jj; };
#ifdef MBAR_DEBUG
    if (tid == 0 && k < 3) printf("CONSUMER blk %d k %d n_local %d NC %d SG %d\n", blockIdx.x, k, n_local, a.NC, a.SG);
#endif
    if (warp == 0) {
      if (lane == 0) {
        int q = 0;
        for (int jj = 0; jj < n_local; ++jj) {
          const int t = tile_of(jj), row0 = t * TOK, g = t % a.SG, s = (t / a.SG) % RD;
          TLD(0);
          for (int kh = 0; kh < 2; ++kh, ++q) {              // dGout K-block kh with Wg rows 64 kh ..: needs no flag
            const int st = q % NST;
            PW(1, mbar_wait(&B.st_empty[st], ((q / NST) & 1) ^ 1));
            mbar_expect_tx(&B.st_full[st], STB);
            const uint32_t d = C_ST + st * STB;
            tma_load_2d(sm + d, &mgo, &B.st_full[st], 64 * kh, row0, EVICT_FIRST);
            if (pair) {
              tma_load_3d_mc(sm + d + STB / 2 + rk * (STB / 4), &mwg, &B.st_full[st], 0, 64 * kh, rk, 3, EVICT_LAST);
            } else {
              tma_load_3d(sm + d + STB / 2, &mwg, &B.st_full[st], 0, 64 * kh, 0, EVICT_LAST);
              tma_load_3d(sm + d + STB / 2 + STB / 4, &mwg, &B.st_full[st], 0, 64 * kh, 1, EVICT_LAST);
            }
          }
          const int fb = jj & 1;
          for (int c = 0; c < NCH; ++c) {
            const int slot = (g * RD + s) * NCH + c;
            PW(0, mbar_wait(&B.fready[fb][c], (jj >> 1) & 1));   // the watcher lane of chunk c has seen its flag
            TLD(1 + c);
            PW(3, fence_proxy_async_global());
            for (int kh = 0; kh < 2; ++kh, ++q) {          // K-block kh of the chunk: W1 rows 128c + 64kh .., dgp columns 64kh ..
              const int st = q % NST;
              PW(1, mbar_wait(&B.st_empty[st], ((q / NST) & 1) ^ 1));
              PROG(0, t * 8 + c);
              // energy ablations (wrong results): B7_NOW1 skips the W1 half, B7_NORING the dgp half of the stage
#if defined(B7_NOW1) && defined(B7_NORING)
              PW(4, mbar_arrive(&B.st_full[st]));
#else
#if defined(B7_NOW1) || defined(B7_NORING)
              PW(4, mbar_expect_tx(&B.st_full[st], STB / 2));
#else
              PW(4, mbar_expect_tx(&B.st_full[st], STB));
#endif
              const uint32_t d = C_ST + st * STB;
#ifndef B7_NORING
              PW(5, tma_load_2d(sm + d, &mring, &B.st_full[st], 64 * kh, slot * TOK, EVICT_FIRST));
#endif
#ifndef B7_NOW1
              if (pair) {
                PW(6, tma_load_3d_mc(sm + d + STB / 2 + rk * (STB / 4), &mw1k, &B.st_full[st], 0, 128 * c + 64 * kh, rk, 3, EVICT_LAST));
              } else {
                tma_load_3d(sm + d + STB / 2, &mw1k, &B.st_full[st], 0, 128 * c + 64 * kh, 0, EVICT_LAST);
                tma_load_3d(sm + d + STB / 2 + STB / 4, &mw1k, &B.st_full[st], 0, 128 * c + 64 * kh, 1, EVICT_LAST);
              }
#endif
#endif
            }
          }
          mbar_arrive(&B.fdone[fb]);                       // the watchers may move on to tile jj + 2
        }
      }
    } else if (warp == 1) {
#if B7_WARPMMA
      {
        const bool ldr = elect_one();
#else
      if (lane == 0) {
        constexpr bool ldr = true;
#endif
        // Lean issue loop: this lane shares its SMSP with two busy epilogue warps, so every instruction between MMAs costs ~3 cycles.
        // Stage index / phase advance incrementally; descriptors are a per-stage base plus constant K16 offsets.
        const uint64_t dA0 = desc_k_sw128(su + C_ST), dB0 = desc_mn_sw128(su + C_ST + STB / 2, STB / 4);
        constexpr uint64_t SD = STB >> 4;                     // stage stride in descriptor address units
        int st = 0;
        uint32_t ph = 0;
        for (int jj = 0; jj < n_local; ++jj) {
          const int t = tile_of(jj), g = t % a.SG, s = (t / a.SG) % RD, acc = jj & 1;
          const int fb = jj & 1;
          PW(1, mbar_wait(&B.acc_empty[acc], ((jj >> 1) & 1) ^ 1));
          tc_fence_after();
          const uint32_t d = tmem + T_ACC + acc * 128;
#pragma unroll 1
          for (int s2 = 0; s2 < 2 + 2 * NCH; ++s2) {         // 2 (dGout, Wg) stages, then NCH chunks x 2 K-blocks of (dgp_c, W1_c)
            PW(0, mbar_wait(&B.st_full[st], ph));
            tc_fence_after();
            // the first / second half of the chunks has landed (their ring slots' global reads are done): hand them to the slot
            // returner (warp 3 lane 16), which fences and releases; nothing here waits for the MMAs
            if (s2 == NCH + 1) {
              if (jj >= 2) mbar_wait(&B.retdone[fb], ((jj >> 1) - 1) & 1);
              if (ldr) mbar_arrive(&B.landed[fb][0]);
            } else if (s2 == 2 * NCH + 1) {
              if (ldr) mbar_arrive(&B.landed[fb][1]);
            }
            const uint64_t da = dA0 + (uint64_t)st * SD, db = dB0 + (uint64_t)st * SD;
            if (ldr) {
              umma_ss(d, da, db, ID_DX, s2 != 0);
              umma_ss(d, da + 2, db + 128, ID_DX, 1);
              umma_ss(d, da + 4, db + 256, ID_DX, 1);
              umma_ss(d, da + 6, db + 384, ID_DX, 1);
              if (pair) umma_commit_mc(&B.st_empty[st], 3);  // the stage is free in both CTAs once both MMAs have read it
              else umma_commit(&B.st_empty[st]);
            }
            if (B7_WARPMMA) __syncwarp();
            if (++st == NST) { st = 0; ph ^= 1; }
          }
          if (ldr) umma_commit(&B.acc_full[acc]);
          if (B7_WARPMMA) __syncwarp();
          TLC(14); TLD(12);
        }

      }
    } else if (warp == 2) {
      if (lane == 0) {
        // x tiles of the consumer tiles (the epilogue's LN_in recompute; dx overwrites them), double buffered: x buffer b is
        // refilled once the dx of the tile two back has left it; dy tiles (the output pass's residual), refilled as soon as the
        // previous output pass has read its tile
        for (int jj = 0; jj < n_local; ++jj) {
          const int row0 = tile_of(jj) * TOK, b = jj & 1;
          if (jj >= 2) PW(0, mbar_wait(&B.cx_free[b], ((jj >> 1) - 1) & 1));
          mbar_expect_tx(&B.cx_full[b], XT);
          tma_load_2d(sm + C_X + b * XT, &mx, &B.cx_full[b], 0, row0, EVICT_FIRST);
          tma_load_2d(sm + C_X + b * XT + XT / 2, &mx, &B.cx_full[b], 64, row0, EVICT_FIRST);
          if (jj >= 1) PW(1, mbar_wait(&B.dy_free, (jj - 1) & 1));
          mbar_expect_tx(&B.dy_full, XT);
          tma_load_2d(sm + C_DY, &mdy, &B.dy_full, 0, row0, EVICT_FIRST);
          tma_load_2d(sm + C_DY + XT / 2, &mdy, &B.dy_full, 64, row0, EVICT_FIRST);
        }
      }
    } else if (warp == 3) {
      if (lane < NCH) {
        // flag watchers: lane c polls chunk c's ring flag of each consumer tile (the NCH polls of a tile in flight together, up to
        // two tiles ahead of the loader) and hands it to the loader through an mbarrier (acquire.gpu here, release / acquire .cta
        // to the loader, which then orders its TMA reads behind them with fence.proxy.async)
        const int c = lane;
        for (int jj = 0; jj < n_local; ++jj) {
          const int t = tile_of(jj), g = t % a.SG, s = (t / a.SG) % RD, fb = jj & 1;
          if (jj >= 2) mbar_wait(&B.fdone[fb], ((jj >> 1) - 1) & 1);
          PW(0, spin_until(a.flags + (g * RD + s) * NCH + c, (uint32_t)(t / a.SG + 1)));
          mbar_arrive(&B.fready[fb][c]);
        }
      } else if (lane == 16) {
        // ring-slot returner: once a half of a tile's chunks has landed in shared memory (the MMA warp's st_full), order those
        // async-proxy reads before the flag writes (fence.proxy.async, then a release pattern) and reset the flags to 0
        for (int jj = 0; jj < n_local; ++jj) {
          const int t = tile_of(jj), g = t % a.SG, s = (t / a.SG) % RD, fb = jj & 1;
          for (int h = 0; h < 2; ++h) {
            mbar_wait(&B.landed[fb][h], (jj >> 1) & 1);
            fence_proxy_async_global();
            fence_acq_rel_gpu();
            for (int c = h * (NCH / 2); c < (h + 1) * (NCH / 2); ++c) flag_return(a.flags + (g * RD + s) * NCH + c);
          }
          mbar_arrive(&B.retdone[fb]);
        }
      }
    } else if (warp >= 4) {
      // LN backward + residual: row r = 32 (warp % 4) + lane, column half hc = 64 columns; the two halves of a row exchange
      // their partial sums through shared memory. The x half row is read from shared memory once per tile into registers (packed
      // bf16) and serves the statistics, the c1 / c2 pass and the output pass; dy comes from its smem tile; dx overwrites x in place.
      const int q = warp & 3, hc = (warp - 4) >> 2, r = q * 32 + lane, et = tid - 128;   // et 0..255
      float* xch = reinterpret_cast<float*>(sm + O_GI + 512);           // [2 halves][128 rows] exchange
      float* xch2 = reinterpret_cast<float*>(sm + O_GI + 1536);
      float agi[4], abi[4];
#pragma unroll
      for (int m = 0; m < 4; ++m) { agi[m] = abi[m] = 0.f; }
      const uint32_t xrow0 = su + C_X + hc * (XT / 2), yrow = su + C_DY + hc * (XT / 2);
      uint32_t xrow = xrow0;
      auto xa = [&](int ch) { return xrow + sw128(r, ch); };            // 16-byte chunk ch (8 columns) of this half row
      auto ya = [&](int ch) { return yrow + sw128(r, ch); };
      const uint32_t gbase = su + O_GI + 64 * hc * 4;                   // gi of this half: 16 float4
      for (int jj = 0; jj < n_local; ++jj) {
        const int t = tile_of(jj), row0 = t * TOK, acc = jj & 1, xb = jj & 1;
        xrow = xrow0 + xb * XT;
        if (warp == 4) { TLC(0); TLD(13); }
        PW(1, mbar_wait(&B.cx_full[xb], (jj >> 1) & 1));
#ifdef PROF
        long long tz = clock64();
#endif
        // ---- input-LN statistics: one pass per half row about its first element (pivot), halves merged exactly (Chan)
        uint32_t xs[32];
#pragma unroll
        for (int ch = 0; ch < 8; ++ch) {
          const uint4 v = lds128(xa(ch));
          xs[4 * ch] = v.x; xs[4 * ch + 1] = v.y; xs[4 * ch + 2] = v.z; xs[4 * ch + 3] = v.w;
        }
        {
          const float pv = __uint_as_float(xs[0] << 16);
          const float2 np2 = make_float2(-pv, -pv);
          float2 s1a = make_float2(0.f, 0.f), s1b = s1a, s2a = s1a, s2b = s1a;
#pragma unroll
          for (int w = 0; w < 32; w += 2) {
            const float2 d0 = __fadd2_rn(up(xs[w]), np2), d1 = __fadd2_rn(up(xs[w + 1]), np2);
            s1a = __fadd2_rn(s1a, d0); s2a = __ffma2_rn(d0, d0, s2a);
            s1b = __fadd2_rn(s1b, d1); s2b = __ffma2_rn(d1, d1, s2b);
          }
          const float2 s1 = __fadd2_rn(s1a, s1b), s2 = __fadd2_rn(s2a, s2b);
          const float S1 = __fadd_rn(s1.x, s1.y), S2 = __fadd_rn(s2.x, s2.y);
          xch[hc * 128 + r] = __fmaf_rn(S1, 1.f / 64, pv);                       // half mean
          xch2[hc * 128 + r] = __fmaf_rn(-S1, __fmul_rn(S1, 1.f / 64), S2);      // half sum of squared deviations
        }
        named_bar_sync(2, 256);
        float mean, rstd;
        {
          const float m0 = xch[r], m1 = xch[128 + r], dm = __fsub_rn(m0, m1);
          mean = __fmul_rn(__fadd_rn(m0, m1), 0.5f);
          const float m2 = __fmaf_rn(__fmul_rn(32.f, dm), dm, __fadd_rn(xch2[r], xch2[128 + r]));
          rstd = rsqrt_ftz(__fadd_rn(__fmul_rn(m2, 1.f / C), a.eps));
        }
        const float2 nmean = make_float2(-mean, -mean), rs2 = make_float2(rstd, rstd);
        if (et == 0 && jj >= 1) {                           // the previous tile's dx has left its x buffer: x of tile jj + 1 may land
          PW(5, bulk_wait_read<0>());
          mbar_arrive(&B.cx_free[xb ^ 1]);
        }
        if (warp == 4 || warp == 11) TLC(warp == 4 ? 2 : 3);
#ifdef PROF
        pacc_[2] += clock64() - tz;
#endif
        PW(0, mbar_wait(&B.acc_full[acc], (jj >> 1) & 1));
        if (warp == 4 || warp == 11) TLC(warp == 4 ? 4 : 5);
        if (warp == 4) TLD(14);
        tc_fence_after();
#ifdef PROF
        tz = clock64();
#endif
        // ---- c1 = sum dxn gi xhat, c2 = sum dxn gi (dxn = bf16 of the fp32 accumulator; the output pass reads it again)
        const uint32_t trow = tmem + T_ACC + acc * 128 + ((uint32_t)(q * 32) << 16) + 64 * hc;
        float c1, c2;
        {
          float2 c1a[2] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f)}, c2a[2] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f)};
#pragma unroll
          for (int h32 = 0; h32 < 2; ++h32) {
            uint32_t dv[32];
            tmem_ld32(trow + 32 * h32, dv);
            tmem_wait_ld();
#pragma unroll
            for (int c4 = 0; c4 < 4; ++c4) {
              const int ch = 4 * h32 + c4;
              const float4 g0 = lds128f(gbase + 32 * ch), g1 = lds128f(gbase + 32 * ch + 16);
              const float2 gw[4] = {make_float2(g0.x, g0.y), make_float2(g0.z, g0.w), make_float2(g1.x, g1.y), make_float2(g1.z, g1.w)};
#pragma unroll
              for (int e = 0; e < 4; ++e) {
                const int w = 4 * ch + e;
                const float2 d = up(pack_bf16(__uint_as_float(dv[8 * c4 + 2 * e]), __uint_as_float(dv[8 * c4 + 2 * e + 1])));
                const float2 xh = __fmul2_rn(__fadd2_rn(up(xs[w]), nmean), rs2);
                const float2 dg = __fmul2_rn(d, gw[e]);
                c1a[e & 1] = __ffma2_rn(dg, xh, c1a[e & 1]);
                c2a[e & 1] = __fadd2_rn(c2a[e & 1], dg);
              }
            }
          }
          const float2 c1s = __fadd2_rn(c1a[0], c1a[1]), c2s = __fadd2_rn(c2a[0], c2a[1]);
          c1 = __fadd_rn(c1s.x, c1s.y); c2 = __fadd_rn(c2s.x, c2s.y);
        }
        if (warp == 4 || warp == 11) TLC(warp == 4 ? 6 : 7);
        named_bar_sync(2, 256);                             // everyone has read this tile's statistics exchange
        xch[hc * 128 + r] = c1;
        xch2[hc * 128 + r] = c2;
        named_bar_sync(2, 256);
        c1 = __fmul_rn(__fadd_rn(xch[r], xch[128 + r]), 1.f / C);
        c2 = __fmul_rn(__fadd_rn(xch2[r], xch2[128 + r]), 1.f / C);
#ifdef PROF
        pacc_[3] += clock64() - tz; tz = clock64();
#endif
        // ---- dx = bf16(bf16(rstd (dxn gi - (xhat c1 + c2))) + dy) in place over x; dgi / dbi column partials
        const float2 c1b = make_float2(c1, c1), c2b = make_float2(c2, c2);
        PW(6, mbar_wait(&B.dy_full, jj & 1));
#pragma unroll
        for (int hq = 0; hq < 4; ++hq) {                   // 16 columns (two chunks) at a time
          float2 pg[8], pb[8];
          uint32_t dv[16];
          tmem_ld16(trow + 16 * hq, dv);
          tmem_wait_ld();
          if (hq == 3) { tc_fence_before(); __syncwarp(); if (lane == 0) mbar_arrive(&B.acc_empty[acc]); }
#pragma unroll
          for (int h2 = 0; h2 < 2; ++h2) {
            const int ch = 2 * hq + h2;
            const uint4 yv = lds128(ya(ch));
            const float4 g0 = lds128f(gbase + 32 * ch), g1 = lds128f(gbase + 32 * ch + 16);
            const float2 gw[4] = {make_float2(g0.x, g0.y), make_float2(g0.z, g0.w), make_float2(g1.x, g1.y), make_float2(g1.z, g1.w)};
            const uint32_t ys[4] = {yv.x, yv.y, yv.z, yv.w};
            uint32_t o[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int w = 4 * ch + e;
              const float2 d = up(pack_bf16(__uint_as_float(dv[8 * h2 + 2 * e]), __uint_as_float(dv[8 * h2 + 2 * e + 1])));
              const float2 xh = __fmul2_rn(__fadd2_rn(up(xs[w]), nmean), rs2);
              const float2 u = __fadd2_rn(__fmul2_rn(xh, c1b), c2b);
              const float2 v = __fadd2_rn(__fmul2_rn(d, gw[e]), make_float2(-u.x, -u.y));
              const uint32_t lw = pack_bf16(__fmul_rn(rstd, v.x), __fmul_rn(rstd, v.y));
              const float2 tt = __fadd2_rn(up(lw), up(ys[e]));
              o[e] = pack_bf16(tt.x, tt.y);
              pg[4 * h2 + e] = __fmul2_rn(d, xh); pb[4 * h2 + e] = d;
            }
            sts128(xa(ch), o[0], o[1], o[2], o[3]);
          }
          // column sums over the 32 rows of the warp: reduce-scatter on float2 pairs (lane bits 4..2), then lane bits 1, 0
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
        if (warp == 4 || warp == 11) TLC(warp == 4 ? 8 : 9);
#ifdef PROF
        pacc_[4] += clock64() - tz; tz = clock64();
#endif
        fence_async_smem();
        named_bar_sync(2, 256);
        if (warp == 4 || warp == 11) TLC(warp == 4 ? 12 : 13);
        if (warp == 4) TLD(15);
        if (et == 0) {
          mbar_arrive(&B.dy_free);                          // the dy tile may take the next consumer tile
          tma_store_2d(&mdx, sm + C_X + xb * XT, 0, row0);
          tma_store_2d(&mdx, sm + C_X + xb * XT + XT / 2, 64, row0);
          bulk_commit();
          PROG(2, t);
        }
#ifdef PROF
        pacc_[5] += clock64() - tz;
#endif
      }
      if (et == 0) bulk_wait<0>();
      // dgi / dbi: the 4 row quarters' column sums combined in a fixed order (the stage ring is drained), one row per CTA
      float* xs = reinterpret_cast<float*>(sm + C_ST);
      if ((lane & 1) == 0) {
#pragma unroll
        for (int h = 0; h < 4; ++h) {
          const int col = 64 * hc + 16 * h + 2 * (lane >> 2) + ((lane >> 1) & 1);
          xs[q * 256 + col] = agi[h]; xs[q * 256 + 128 + col] = abi[h];
        }
      }
      named_bar_sync(2, 256);
      a.lnpart[(size_t)k * 256 + et] = ((xs[et] + xs[256 + et]) + xs[512 + et]) + xs[768 + et];
    }
  }
  if (lane == 0 && warp < 8) PROF_END(warp == 3 ? 2 : (warp >= 4 ? 3 : warp));
  tc_fence_before();
  __syncthreads();
  cluster_arrive();                                          // no CTA leaves while its pair may still multicast into it
  cluster_wait();
  if (warp == 2) tmem_dealloc(tmem, 512);
}

}  // namespace b7g

// x / xn / dy / dgout / dx [M, 128]; dpl [64 NCH, M] plane gradient; tmask [B L] bool token mask or None; w1 [128 NCH, 128] packed;
// wg [128, 128]; gi [128] fp32; dw1 [SG, 128 NCH, 128] fp32 per-group partials, written; lnpart [>= SMs - NCH SG, 256] fp32:
// per-consumer (dg_i | db_i) rows, written (grad_reduce sums both in a fixed order); ring [SG RD NCH 128, 128]; flags zero.
void b7g_backward(torch::Tensor x, torch::Tensor xn, torch::Tensor dy, torch::Tensor dpl, torch::Tensor dgout,
                  c10::optional<torch::Tensor> tmask, torch::Tensor w1, torch::Tensor wg, torch::Tensor gi, torch::Tensor dx,
                  torch::Tensor dw1, torch::Tensor lnpart, torch::Tensor ring, torch::Tensor flags, double eps, int64_t sg, int64_t L) {
  using namespace b7g;
  const int64_t M = x.size(0);
  const int P = (int)dpl.size(0), NCH = P / 64;
  TORCH_CHECK(M % TOK == 0 && x.size(1) == C && dpl.numel() == (int64_t)P * M && (NCH == 4 || NCH == 8) && w1.size(0) == 128 * NCH);
  int nsm = 0, dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  const int SG = (int)sg, NC = nsm - NCH * SG;
  TORCH_CHECK(NC >= 2 && NC % 2 == 0 && ring.size(0) >= (int64_t)SG * RD * NCH * TOK && flags.numel() >= SG * RD * NCH);
  auto bf = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  auto tok_map = [&](const torch::Tensor& t, int64_t rows) {
    return tmap::make(t.data_ptr(), bf, {(uint64_t)C, (uint64_t)rows}, {(uint64_t)C * 2}, {64, TOK}, CU_TENSOR_MAP_SWIZZLE_128B);
  };
  auto mxn = tok_map(xn, M), mgo = tok_map(dgout, M), mdx = tok_map(dx, M), mring = tok_map(ring, ring.size(0));
  auto mw1 = tok_map(w1, w1.size(0)), mx = tok_map(x, M), mdy = tok_map(dy, M);
  // W1 / Wg K-blocks as two 64-column halves (z), one per CTA of a consumer pair
  auto mwg = tmap::make(wg.data_ptr(), bf, {64, 128, 2}, {(uint64_t)C * 2, 128}, {64, 64, 1}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mw1k = tmap::make(w1.data_ptr(), bf, {64, (uint64_t)w1.size(0), 2}, {(uint64_t)C * 2, 128}, {64, 64, 1}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mda = tmap::make(dpl.data_ptr(), bf, {(uint64_t)M, (uint64_t)P}, {(uint64_t)M * 2}, {64, 64}, CU_TENSOR_MAP_SWIZZLE_128B);
  Args a;
  a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  a.dy = reinterpret_cast<const __nv_bfloat16*>(dy.data_ptr());
  TORCH_CHECK(M % (L * L) == 0 && (!tmask.has_value() || (tmask->numel() == M / L && tmask->element_size() == 1 && tmask->is_contiguous())));
  a.tmask = tmask.has_value() ? reinterpret_cast<const uint8_t*>(tmask->data_ptr()) : nullptr;
  a.gi = gi.data_ptr<float>();
  a.dx = reinterpret_cast<__nv_bfloat16*>(dx.data_ptr());
  TORCH_CHECK(lnpart.numel() >= (int64_t)NC * 2 * C && dw1.numel() >= (int64_t)SG * 128 * NCH * C);
  a.dw1 = dw1.data_ptr<float>(); a.lnpart = lnpart.data_ptr<float>();
  a.flags = reinterpret_cast<uint32_t*>(flags.data_ptr<int32_t>());
  a.tiles = (int)(M / TOK); a.L = (int)L; a.SG = SG; a.NC = NC; a.eps = (float)eps;
  a.pair = (a.tiles % 2 == 0) ? 1 : 0;                       // L % 128 == 0 always gives an even count; kept for any M
  void* params[] = {(void*)&mxn, (void*)&mda, (void*)&mw1, (void*)&mring, (void*)&mgo, (void*)&mwg, (void*)&mdx, (void*)&mw1k,
                    (void*)&mx, (void*)&mdy, (void*)&a};
  auto launch = [&](const void* fn, bool& attr) {
    if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM)); attr = true; }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(nsm); cfg.blockDim = dim3(384); cfg.dynamicSmemBytes = SMEM; cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[2];
    at[0].id = cudaLaunchAttributeCooperative; at[0].val.cooperative = 1;   // every CTA resident: the ring flags spin across CTAs
    at[1].id = cudaLaunchAttributeClusterDimension; at[1].val.clusterDim.x = 2; at[1].val.clusterDim.y = 1; at[1].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 2;
    C10_CUDA_CHECK(cudaLaunchKernelExC(&cfg, fn, params));
  };
  static bool attr4 = false, attr8 = false;
  if (NCH == 4) launch((const void*)b7g_kernel<4>, attr4);
  else launch((const void*)b7g_kernel<8>, attr8);
}
