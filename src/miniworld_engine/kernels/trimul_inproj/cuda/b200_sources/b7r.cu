// b7r.cu -- TriMul B7 on B200, v2: source / consumer roles connected by an L2-resident ring (the H100 b7_joint structure).
//
// Same math as b7.cu (see there). One cooperative launch, one CTA per SM:
//   source CTAs  (8 per source group; CTA (g, c) owns packed-W1 chunk c = plane channels 64c .. 64c+63, rows 128c .. 128c+127)
//     for every tile of its group: pre = xn W1_c^T, dg/dp (fragment layout) -> dgp tile [128 tok][128] bf16,
//     dW_c += dgp^T xn (TMEM, persistent), dgp -> ring slot (TMA store) + flag = 1.
//   consumer CTAs: for each of its tiles: wait the 8 flags, stream (dgp_c, W1_c) pairs through a 2-stage ring,
//     dxn = bf16( sum_c dgp_c W1_c + dGout Wg )  (fp32 accumulate over K = 1024 + 128, one rounding, as on H100),
//     LN_in backward + residual -> dx (TMA store), dgi / dbi (reduce-scatter over each warp's rows, register-accumulated);
//     a flag is reset to 0 once its tile has landed in shared memory, which also frees the ring slot. A published flag carries
//     the group-local tile index + 1, so a consumer never takes an older tile still waiting in the same slot.
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
namespace b7r {

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
#define B7_NST 5
#endif
// consumer ring: NST stages of (A K-block 16 KB + B K-block 16 KB); per tile 2 stages (dGout, Wg) then 16 (dgp_c, W1_c) stages
constexpr int NST = B7_NST, STB = XT;
// + the x tile (dx written in place, then TMA-stored) and the dy tile of the current consumer tile
constexpr int C_ST = 0, C_X = C_ST + NST * STB, C_DY = C_X + XT, C_END = C_DY + XT;
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
  uint64_t st_full[NST], st_empty[NST], rel_bar[NST], wg_full, go_full, go_free, acc_full[2], acc_empty[2], out_free, xy_full, xy_free;
  uint32_t tmem;
  int rel_done;             // consumer: ring-slot releases done by warp 3
};

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
DEV float2 lds64g(uint32_t ad) { float2 v; asm volatile("ld.shared.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(ad) : "memory"); return v; }
DEV void red_add(float* p, float v) { asm volatile("red.global.add.f32 [%0], %1;" ::"l"(p), "f"(v) : "memory"); }
DEV void red_add4(float* p, float a, float b, float c, float d) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}
DEV uint32_t ld_acquire(const uint32_t* p) { uint32_t v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
DEV void st_relaxed(uint32_t* p, uint32_t v) { asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
DEV void st_release(uint32_t* p, uint32_t v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
#ifndef B7_RELAXED_FLAG
#define B7_RELAXED_FLAG 0
#endif
// Publishing a ring tile: its TMA-store writes have completed (cp.async.bulk.wait_group, performed at L2) and are ordered before
// generic accesses by fence.proxy.async; B7_RELAXED_FLAG=1 then writes the flag with a relaxed store (st.release.gpu costs ~2k clk).
#ifndef B7_RELAXED_RET
#define B7_RELAXED_RET 1
#endif
// Returning a ring slot (consumer): its TMA reads landed in shared memory before the MMAs that read them ran, so a relaxed store
// suffices (B7_RELAXED_RET=1).
DEV void flag_return(uint32_t* p) {
#if B7_RELAXED_RET
  asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(0u) : "memory");
#else
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(0u) : "memory");
#endif
}
DEV void flag_publish(uint32_t* p, uint32_t v) {
#if B7_RELAXED_FLAG
  asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
#else
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
#endif
}
DEV void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
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

struct Args {
  const __nv_bfloat16 *x, *dy;
  const float *mask, *gi;
  __nv_bfloat16* dx;
  float *dw1, *dgi, *dbi;
  uint32_t* flags;          // [SG][RD][8]
  int tiles, L, SG, NC;
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
__global__ void __launch_bounds__(384, 1)
    b7r_kernel(const __grid_constant__ CUtensorMap mxn, const __grid_constant__ CUtensorMap mdl, const __grid_constant__ CUtensorMap mdr,
               const __grid_constant__ CUtensorMap mw1, const __grid_constant__ CUtensorMap mring, const __grid_constant__ CUtensorMap mgo,
               const __grid_constant__ CUtensorMap mwg, const __grid_constant__ CUtensorMap mdx, const __grid_constant__ CUtensorMap mw1k,
               const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mdy, const Args a) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int NS = 8 * a.SG;
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
    for (int b = 0; b < NST; ++b) { mbar_init(&B.st_full[b], 1); mbar_init(&B.st_empty[b], 1); mbar_init(&B.rel_bar[b], 1); }
    *reinterpret_cast<volatile int*>(&B.rel_done) = 0;     // consumer: ring-slot releases done by warp 3
    mbar_init(&B.wg_full, 1); mbar_init(&B.go_full, 1); mbar_init(&B.go_free, 1); mbar_init(&B.out_free, 1);
    mbar_init(&B.xy_full, 1); mbar_init(&B.xy_free, 1);
    fence_mbar_init();
  }
  if (tid < 128) reinterpret_cast<float*>(sm + O_GI)[tid] = a.gi[tid];
  if (warp == 2) { tmem_alloc(&B.tmem, 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  PROF_BEGIN

  if (is_src) {
    // ================================================================== SOURCE
    const int g = blockIdx.x / 8, c = blockIdx.x % 8;
    const int n_local = a.tiles > g ? (a.tiles - g + a.SG - 1) / a.SG : 0;   // tiles t = g + SG j
    const CUtensorMap* mda = c < 4 ? &mdl : &mdr;
    const int chbase = (c & 3) * 64;
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
          tma_load_2d(sm + S_DA + b * DAT, mda, &B.da_full[b], row0, chbase, EVICT_FIRST);
          tma_load_2d(sm + S_DA + b * DAT + DAT / 2, mda, &B.da_full[b], row0 + 64, chbase, EVICT_FIRST);
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
        const int b = warp == 3 ? 0 : 1;
        int prev_slot = -1, prev_j = 0;
        for (int j = b; j < n_local; j += 2) {
          const int slot = (g * RD + j % RD) * 8 + c;
          PW(1, spin_until(a.flags + slot, 0u)); TLS(1);            // consumer of tile j - RD has taken the slot
          PW(0, mbar_wait(&B.dgp_ready[b], (j >> 1) & 1)); TLS(0);
          PW(3, tma_store_2d(&mring, sm + S_DGP + b * XT, 0, slot * TOK));
          PW(3, tma_store_2d(&mring, sm + S_DGP + b * XT + XT / 2, 64, slot * TOK));
          bulk_commit();
          PW(2, bulk_wait_read<0>());
          mbar_arrive(&B.dgp_free[b]); TLS(2);                      // smem read: EW may refill the buffer
          if (prev_slot >= 0) {
            PW(5, bulk_wait<1>());                                 // the previous tile's global writes are complete (at L2)
            fence_proxy_async_global();
            PW(4, flag_publish(a.flags + prev_slot, (uint32_t)(prev_j + 1)));   // generation tag: group-local tile index + 1
          }
          prev_slot = slot; prev_j = j;
          TLS(3);
        }
        if (prev_slot >= 0) {
          bulk_wait<0>();
          fence_proxy_async_global();
          flag_publish(a.flags + prev_slot, (uint32_t)(prev_j + 1));
        }
      }
    } else if (warp >= 4) {
      // dg / dp: warps 4-7 token group hh 0, warps 8-11 hh 1 of each warp quarter
      const int q = warp & 3, hh = (warp - 4) >> 2, t4 = lane & 3, tr = lane >> 2;
      const int tok0 = q * 32 + hh * 16;
      float mk0 = 0.f, mk1 = 0.f;
      if (n_local > 0) { mk0 = __ldg(a.mask + g * TOK + tok0 + tr); mk1 = __ldg(a.mask + g * TOK + tok0 + tr + 8); }
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
        if (warp == 4) TLS(6);
        if (tid == 128) { mbar_arrive(&B.dgp_ready[b]); PROG(3, j); }
        if (j + 1 < n_local) {                               // next tile's mask (after the proxy fence, which would wait on it)
          const int r1 = (g + a.SG * (j + 1)) * TOK + tok0 + tr;
          mk0 = __ldg(a.mask + r1); mk1 = __ldg(a.mask + r1 + 8);
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
        float* dst = a.dw1 + (size_t)(128 * c + row) * C + cb * 32;
#pragma unroll
        for (int k = 0; k < 32; k += 4)
          red_add4(dst + k, __uint_as_float(v[k]), __uint_as_float(v[k + 1]), __uint_as_float(v[k + 2]), __uint_as_float(v[k + 3]));
      }
    }
  } else {
    // ================================================================== CONSUMER
    const int k = blockIdx.x - NS;
    const int n_local = a.tiles > k ? (a.tiles - k + a.NC - 1) / a.NC : 0;   // tiles t = k + NC jj
#ifdef MBAR_DEBUG
    if (tid == 0 && k < 3) printf("CONSUMER blk %d k %d n_local %d NC %d SG %d\n", blockIdx.x, k, n_local, a.NC, a.SG);
#endif
    if (warp == 0) {
      if (lane == 0) {
        int q = 0;
        for (int jj = 0; jj < n_local; ++jj) {
          const int t = k + a.NC * jj, row0 = t * TOK, g = t % a.SG, s = (t / a.SG) % RD;
          TLD(0);
          for (int kh = 0; kh < 2; ++kh, ++q) {              // dGout K-block kh with Wg rows 64 kh ..: needs no flag
            const int st = q % NST;
            PW(1, mbar_wait(&B.st_empty[st], ((q / NST) & 1) ^ 1));
            mbar_expect_tx(&B.st_full[st], STB);
            const uint32_t d = C_ST + st * STB;
            tma_load_2d(sm + d, &mgo, &B.st_full[st], 64 * kh, row0, EVICT_FIRST);
            tma_load_3d(sm + d + STB / 2, &mwg, &B.st_full[st], 0, 64 * kh, 0, EVICT_LAST);
          }
          for (int c = 0; c < 8; ++c) {
            const int slot = (g * RD + s) * 8 + c;
            PW(0, spin_until(a.flags + slot, (uint32_t)(t / a.SG + 1)));
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
              PW(6, tma_load_3d(sm + d + STB / 2, &mw1k, &B.st_full[st], 0, 128 * c + 64 * kh, 0, EVICT_LAST));
#endif
#endif
            }
          }
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
        int rel_st = -1, rel_f = 0;                            // pending ring-slot return: stage, its phase, flag index
        uint32_t rel_ph = 0;
        int since = 0;
        for (int jj = 0; jj < n_local; ++jj) {
          const int t = k + a.NC * jj, g = t % a.SG, s = (t / a.SG) % RD, acc = jj & 1;
          const int fbase = (g * RD + s) * 8;
          PW(1, mbar_wait(&B.acc_empty[acc], ((jj >> 1) & 1) ^ 1));
          tc_fence_after();
          const uint32_t d = tmem + T_ACC + acc * 128;
#pragma unroll 1
          for (int s2 = 0; s2 < 18; ++s2) {                  // 2 (dGout, Wg) stages, then 8 chunks x 2 K-blocks of (dgp_c, W1_c)
            PW(0, mbar_wait(&B.st_full[st], ph));
            tc_fence_after();
            const uint64_t da = dA0 + (uint64_t)st * SD, db = dB0 + (uint64_t)st * SD;
            if (ldr) {
              umma_ss(d, da, db, ID_DX, s2 != 0);
              umma_ss(d, da + 2, db + 128, ID_DX, 1);
              umma_ss(d, da + 4, db + 256, ID_DX, 1);
              umma_ss(d, da + 6, db + 384, ID_DX, 1);
              umma_commit(&B.st_empty[st]);
            }
            if (B7_WARPMMA) __syncwarp();
            ++since;
            if (rel_st >= 0 && since >= 2) {                   // a chunk's last K-block two stages back: return its ring slot
              PW(3, mbar_wait(&B.st_empty[rel_st], rel_ph));
              if (ldr) flag_return(a.flags + rel_f);           // (relaxed store: the slot's TMA reads landed before its MMAs ran)
              rel_st = -1;
            }
            if (s2 >= 3 && (s2 & 1)) { rel_st = st; rel_ph = ph; rel_f = fbase + ((s2 - 2) >> 1); since = 0; }
            if (++st == NST) { st = 0; ph ^= 1; }
          }
          if (ldr) umma_commit(&B.acc_full[acc]);
          if (B7_WARPMMA) __syncwarp();
          TLC(14); TLD(12);
        }
        if (rel_st >= 0) { mbar_wait(&B.st_empty[rel_st], rel_ph); if (ldr) flag_return(a.flags + rel_f); }

      }
    } else if (warp == 2) {
      if (lane == 0) {
        // x / dy tiles of each consumer tile (the epilogue's LN_in recompute and residual); refilled once dx has left the x tile
        for (int jj = 0; jj < n_local; ++jj) {
          const int row0 = (k + a.NC * jj) * TOK;
          if (jj >= 1) PW(0, mbar_wait(&B.xy_free, (jj - 1) & 1));
          mbar_expect_tx(&B.xy_full, 2 * XT);
          tma_load_2d(sm + C_X, &mx, &B.xy_full, 0, row0, EVICT_FIRST);
          tma_load_2d(sm + C_X + XT / 2, &mx, &B.xy_full, 64, row0, EVICT_FIRST);
          tma_load_2d(sm + C_DY, &mdy, &B.xy_full, 0, row0, EVICT_FIRST);
          tma_load_2d(sm + C_DY + XT / 2, &mdy, &B.xy_full, 64, row0, EVICT_FIRST);
        }
      }
    } else if (warp >= 4) {
      // LN backward + residual: row r = 32 (warp % 4) + lane, column half hc = 64 columns; the two halves of a row exchange
      // their partial sums through shared memory. x / dy come from the smem tiles; dx overwrites x in place.
      const int q = warp & 3, hc = (warp - 4) >> 2, r = q * 32 + lane, et = tid - 128;   // et 0..255
      float* xch = reinterpret_cast<float*>(sm + O_GI + 512);           // [2 halves][128 rows] exchange
      float* xch2 = reinterpret_cast<float*>(sm + O_GI + 1536);
      float agi[4], abi[4];
#pragma unroll
      for (int m = 0; m < 4; ++m) { agi[m] = abi[m] = 0.f; }
      const uint32_t xrow = su + C_X + hc * (XT / 2), yrow = su + C_DY + hc * (XT / 2);
      auto xa = [&](int ch) { return xrow + sw128(r, ch); };            // 16-byte chunk ch (8 columns) of this half row
      auto ya = [&](int ch) { return yrow + sw128(r, ch); };
      const uint32_t gbase = su + O_GI + 64 * hc * 4;
      for (int jj = 0; jj < n_local; ++jj) {
        const int t = k + a.NC * jj, row0 = t * TOK, acc = jj & 1;
        if (warp == 4) { TLC(0); TLD(13); }
        PW(1, mbar_wait(&B.xy_full, jj & 1));
#ifdef PROF
        long long tz = clock64();
#endif
        // ---- input-LN statistics (two-pass, fp32)
        float2 sa = make_float2(0.f, 0.f), sb = make_float2(0.f, 0.f);
#pragma unroll
        for (int ch = 0; ch < 8; ++ch) {
          const uint4 v = lds128(xa(ch));
          sa = __fadd2_rn(sa, __fadd2_rn(up(v.x), up(v.y))); sb = __fadd2_rn(sb, __fadd2_rn(up(v.z), up(v.w)));
        }
        {
          const float2 s2 = __fadd2_rn(sa, sb);
          xch[hc * 128 + r] = __fadd_rn(s2.x, s2.y);
        }
        named_bar_sync(2, 256);
        const float mean = __fmul_rn(__fadd_rn(xch[r], xch[128 + r]), 1.f / C);
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
        {
          const float2 s2 = __fadd2_rn(sa, sb);
          xch2[hc * 128 + r] = __fadd_rn(s2.x, s2.y);
        }
        named_bar_sync(2, 256);
        const float rstd = rsqrt_ftz(__fadd_rn(__fmul_rn(__fadd_rn(xch2[r], xch2[128 + r]), 1.f / C), a.eps));
        const float2 rs2 = make_float2(rstd, rstd);
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
        // ---- c1 = sum dxn gi xhat, c2 = sum dxn gi (dxn = bf16 of the fp32 accumulator, kept packed for the output pass)
        const uint32_t trow = tmem + T_ACC + acc * 128 + ((uint32_t)(q * 32) << 16) + 64 * hc;
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
          c1 = __fadd_rn(c1s.x, c1s.y); c2 = __fadd_rn(c2s.x, c2s.y);
        }
        if (warp == 4 || warp == 11) TLC(warp == 4 ? 6 : 7);
        named_bar_sync(2, 256);
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
#pragma unroll
        for (int hq = 0; hq < 4; ++hq) {                   // 16 columns (two chunks) at a time
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
              const float2 u = __fadd2_rn(__fmul2_rn(xh, c1b), c2b);
              const float2 v = __fadd2_rn(__fmul2_rn(d, lds64g(gbase + 8 * w)), make_float2(-u.x, -u.y));
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
          tma_store_2d(&mdx, sm + C_X, 0, row0);
          tma_store_2d(&mdx, sm + C_X + XT / 2, 64, row0);
          bulk_commit();
          bulk_wait_read<0>();
          mbar_arrive(&B.xy_free);                          // x / dy tiles may take the next consumer tile
          PROG(2, t);
        }
#ifdef PROF
        pacc_[5] += clock64() - tz;
#endif
      }
      if (et == 0) bulk_wait<0>();
      if ((lane & 1) == 0) {
#pragma unroll
        for (int h = 0; h < 4; ++h) {
          const int col = 64 * hc + 16 * h + 2 * (lane >> 2) + ((lane >> 1) & 1);
          red_add(a.dgi + col, agi[h]); red_add(a.dbi + col, abi[h]);
        }
      }
    }
  }
  if (lane == 0 && warp < 8) PROF_END(warp == 3 ? 2 : (warp >= 4 ? 3 : warp));
  tc_fence_before();
  __syncthreads();
  if (warp == 2) tmem_dealloc(tmem, 512);
}

}  // namespace b7r

void b7r_backward(torch::Tensor x, torch::Tensor xn, torch::Tensor dy, torch::Tensor dl, torch::Tensor dr, torch::Tensor dgout,
                  torch::Tensor mask, torch::Tensor w1, torch::Tensor wg, torch::Tensor gi, torch::Tensor dx, torch::Tensor dw1,
                  torch::Tensor dgi, torch::Tensor dbi, torch::Tensor ring, torch::Tensor flags, double eps, int64_t sg) {
  using namespace b7r;
  const int64_t M = x.size(0);
  TORCH_CHECK(M % TOK == 0);
  int nsm = 0, dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  const int SG = (int)sg, NC = nsm - 8 * SG;
  TORCH_CHECK(NC >= 1 && ring.size(0) >= SG * RD * 8 * TOK && flags.numel() >= SG * RD * 8);
  auto bf = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  auto tok_map = [&](const torch::Tensor& t, int64_t rows) {
    return tmap::make(t.data_ptr(), bf, {(uint64_t)C, (uint64_t)rows}, {(uint64_t)C * 2}, {64, TOK}, CU_TENSOR_MAP_SWIZZLE_128B);
  };
  auto mxn = tok_map(xn, M), mgo = tok_map(dgout, M), mdx = tok_map(dx, M), mring = tok_map(ring, ring.size(0)), mw1 = tok_map(w1, 1024);
  auto mx = tok_map(x, M), mdy = tok_map(dy, M);
  auto mwg = tmap::make(wg.data_ptr(), bf, {64, 128, 2}, {(uint64_t)C * 2, 128}, {64, 64, 2}, CU_TENSOR_MAP_SWIZZLE_128B);
  // W1 / Wg K-blocks [64 rows][128 i] as two 64-column MN blocks 8 KB apart: 3D view (i inner 64, rows, i outer 2)
  auto mw1k = tmap::make(w1.data_ptr(), bf, {64, 1024, 2}, {(uint64_t)C * 2, 128}, {64, 64, 2}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mdl = tmap::make(dl.data_ptr(), bf, {(uint64_t)M, 256}, {(uint64_t)M * 2}, {64, 64}, CU_TENSOR_MAP_SWIZZLE_128B);
  auto mdr = tmap::make(dr.data_ptr(), bf, {(uint64_t)M, 256}, {(uint64_t)M * 2}, {64, 64}, CU_TENSOR_MAP_SWIZZLE_128B);
  Args a;
  a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  a.dy = reinterpret_cast<const __nv_bfloat16*>(dy.data_ptr());
  a.mask = mask.data_ptr<float>(); a.gi = gi.data_ptr<float>();
  a.dx = reinterpret_cast<__nv_bfloat16*>(dx.data_ptr());
  a.dw1 = dw1.data_ptr<float>(); a.dgi = dgi.data_ptr<float>(); a.dbi = dbi.data_ptr<float>();
  a.flags = reinterpret_cast<uint32_t*>(flags.data_ptr<int32_t>());
  a.tiles = (int)(M / TOK); a.L = 0; a.SG = SG; a.NC = NC; a.eps = (float)eps;
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(b7r_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM); attr = true; }
  void* params[] = {(void*)&mxn, (void*)&mdl, (void*)&mdr, (void*)&mw1, (void*)&mring, (void*)&mgo, (void*)&mwg, (void*)&mdx, (void*)&mw1k, (void*)&mx, (void*)&mdy, (void*)&a};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)b7r_kernel, dim3(nsm), dim3(384), params, SMEM, at::cuda::getCurrentCUDAStream()));
}

std::vector<double> b7r_prof(int64_t sg) {
#ifndef PROF
  return {};
#else
  std::vector<unsigned long long> h(160 * 64);
  cudaMemcpyFromSymbol(h.data(), g_prof, sizeof(unsigned long long) * 160 * 64);
  std::vector<double> m(128, 0.0);   // [0,64) sources, [64,128) consumers
  const int NS = 8 * (int)sg;
  for (int b = 0; b < 148; ++b) for (int k = 0; k < 64; ++k) m[(b < NS ? 0 : 64) + k] += h[b * 64 + k] / (b < NS ? NS : 148 - NS);
  for (int k = 0; k < 16; ++k) m.push_back((double)(h[152 * 64 + k] - h[152 * 64]));
  for (int r = 0; r < 2; ++r) for (int k = 0; k < 16; ++k) m.push_back((double)(h[(153 + r) * 64 + k] - h[153 * 64 + 4]));
  for (int r = 0; r < 2; ++r) for (int k = 0; k < 16; ++k) m.push_back((double)(h[(155 + r) * 64 + k] - h[155 * 64]));
  return m;
#endif
}
