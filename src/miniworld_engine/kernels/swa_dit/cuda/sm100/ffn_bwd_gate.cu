// ffn_bwd_gate.cu — first half of the SWA atom block's FFN backward on sm_100a (the Triton _ffn_bwd's DFFN / HH / DAB, same rounding):
//   dffn = rn(dq2 rn(gate_f));  a | b = y Wu^T (recomputed);  dh = dffn Wd;  sa = sigmoid(a)
//   h = rn(a sa b);  da = rn(dh b sa (1 + a (1 - sa)));  db = rn(dh a sa)          -> DFFN [M, C], HH [M, 256], DAB = [da | db] [M, 512]
// (dy = DAB Wu and everything after it is ffn_bwd_dy.cu.)
// Transposed, no weights in shared memory: Wu (512 x 128) and Wd^T (256 x 128) live in TMEM as bf16 A operands (384 columns), and
// per 128-hidden chunk a^T = Wu_a y^T, b^T = Wu_b y^T, dh^T = Wd^T dffn^T (M = 128 hidden, N = 32 rows) take y and dffn straight from
// their tiles (K-major B). Tiles of SP augments x AT atoms = 32 rows. dffn: one channel per thread, written in place over the dq2 tile
// (then MMA operand and TMA-store source). Gate: one hidden unit per thread (the TMEM lane), 16 rows per warpgroup; h / da / db go to
// row-major staging tiles (2-byte stores, a warp covers 64 contiguous bytes of a row) that leave by TMA stores.
// TMEM: Wu tiles (a 0-127, a 128-255, b 0-127, b 128-255) at 0 / 64 / 128 / 192, Wd^T tiles at 256 / 320, a^T / b^T / dh^T at 384 / 416 / 448.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

constexpr int C = 128, NH = 256, NSMAX = 8, UPT = 2;
constexpr int SLOT = 8192, KBLK = 4096;                                   // [32 rows][128 ch] tile: 2 x [32][64] SW128
constexpr int HHB = 4 * KBLK, DABB = 8 * KBLK, STG = HHB + DABB;          // staging per tile: HH (16 KB) | DAB (32 KB)
enum { U_Y = 0, U_DQ2 };
constexpr uint32_t T_WU = 0, T_WD = 256, T_A = 384, T_B = 416, T_DH = 448;
constexpr uint32_t I_G = idesc_bf16(128, 32);

struct Bars {
  uint64_t full[NSMAX], rdy[NSMAX], empty[NSMAX], auxfull[2], auxempty[2], accfull, accfree, stfull[2], stfree[2];
  uint32_t tmem;
};
struct RingPos {
  int b = 0; uint32_t p = 0;
  DEVI void get(int u, int ns, int& s, uint32_t& ph) const { s = b + u; ph = p; if (s >= ns) { s -= ns; ph ^= 1u; } }
  DEVI void next(int ns) { b += UPT; if (b >= ns) { b -= ns; p ^= 1u; } }
};
DEVI void mbar_arrive_cnt(uint64_t* b, uint32_t n) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(n) : "memory"); }
DEVI float lds_bf16(uint32_t a) { unsigned short v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a) : "memory"); return __uint_as_float((uint32_t)v << 16); }
DEVI float rnb(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
DEVI void sts_bf16(uint32_t a, float x) {
  const unsigned short v = __bfloat16_as_ushort(__float2bfloat16_rn(x));
  asm volatile("st.shared.u16 [%0], %1;" :: "r"(a), "h"(v) : "memory");
}
DEVI float sigm(float x) { return rcpf(1.f + ex2f(-1.4426950408889634f * x)); }

extern "C" __global__ void __launch_bounds__(384, 1)
swa_ffn_bwd_gate_sm100(const __grid_constant__ CUtensorMap my, const __grid_constant__ CUtensorMap mdq2, const __grid_constant__ CUtensorMap mmod,
                       const __grid_constant__ CUtensorMap mdffn, const __grid_constant__ CUtensorMap mhh, const __grid_constant__ CUtensorMap mdab,
                       int S, int A, int Bn, int SP, int AT, int nab, int nag, int ntile, int NS,
                       const __nv_bfloat16* __restrict__ WU, const __nv_bfloat16* __restrict__ WDT) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int AUXST = (AT * 512 + 1023) / 1024 * 1024;
  const int O_STG = NS * SLOT, O_AUX = O_STG + 2 * STG, O_BAR = O_AUX + 2 * AUXST;
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ntT = (int)blockIdx.x < ntile ? (ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto coords = [&](int T, int& b, int& a0, int& s0) {
    const int t = (int)blockIdx.x + T * (int)gridDim.x;
    const int ab = t % nab, r = t / nab, ag = r % nag;
    b = r / nag; a0 = ag * SP; s0 = ab * AT;
  };

  if (tid == 0) {
    for (int i = 0; i < NS; ++i) { mbar_init(&B.full[i], 1); mbar_init(&B.rdy[i], 2); mbar_init(&B.empty[i], 2); }
    for (int i = 0; i < 2; ++i) { mbar_init(&B.auxfull[i], 1); mbar_init(&B.auxempty[i], 2); mbar_init(&B.stfull[i], 2); mbar_init(&B.stfree[i], 1); }
    mbar_init(&B.accfull, 1); mbar_init(&B.accfree, 8);
    fence_barrier_init();
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  if (warp >= 4) {                                                        // weights -> TMEM: lane l of M-tile t = row 128 t + l
    const int wg = (warp - 4) >> 2, lb = (warp & 3) * 32, l = lb + lane;
#pragma unroll 1
    for (int t = wg * 3; t < wg * 3 + 3; ++t) {                          // tiles 0-3: Wu rows 128 t ..; tiles 4-5: Wd^T rows 128 (t - 4) ..
      const __nv_bfloat16* src = t < 4 ? WU + (size_t)(128 * t + l) * C : WDT + (size_t)(128 * (t - 4) + l) * C;
      const uint32_t col = t < 4 ? T_WU + 64 * t : T_WD + 64 * (t - 4);
#pragma unroll 1
      for (int k = 0; k < 4; ++k) {
        uint32_t v[16];
#pragma unroll
        for (int e = 0; e < 4; ++e) { const uint4 u = ldg128(src + 32 * k + 8 * e); v[4 * e] = u.x; v[4 * e + 1] = u.y; v[4 * e + 2] = u.z; v[4 * e + 3] = u.w; }
        tmem_st16(tmem + ((uint32_t)lb << 16) + col + 16 * k, v);
      }
    }
    tmem_wait_st();
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer: y, dq2, gate_f
    if (lane == 0) {
      const uint32_t tx = (uint32_t)(SP * AT * 256), auxtx = (uint32_t)(AT * 512);
      RingPos rp; int gtot = 0;
      for (int T = 0; T < ntT; ++T, rp.next(NS)) {
        int b, a0, s0; coords(T, b, a0, s0);
        const int xa = T & 1;
        if (T >= 2) mbar_wait(&B.auxempty[xa], ((T >> 1) - 1) & 1);
        mbar_expect_tx(&B.auxfull[xa], auxtx);
        tma_load_3d(su + O_AUX + xa * AUXST, &mmod, &B.auxfull[xa], 0, b * S + s0, 20);   // gate_f: blocks 20..23
        for (int u = 0; u < UPT; ++u, ++gtot) {
          int sl; uint32_t ph; rp.get(u, NS, sl, ph);
          if (gtot >= NS) mbar_wait(&B.empty[sl], ph ^ 1);
          mbar_expect_tx(&B.full[sl], tx);
          for (int kb = 0; kb < 2; ++kb) tma_load_4d(su + sl * SLOT + kb * KBLK, u == U_Y ? &my : &mdq2, &B.full[sl], kb * 64, s0, b, a0);
          if (u == U_Y) mbar_arrive_cnt(&B.rdy[sl], 2);                   // y is used as loaded (keeps the rdy phases in step)
        }
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    RingPos rp; int gc = 0;
    for (int T = 0; T < ntT; ++T, rp.next(NS)) {
      int sy, sd; uint32_t py, pd; rp.get(U_Y, NS, sy, py); rp.get(U_DQ2, NS, sd, pd);
      const uint32_t ya = su + sy * SLOT, da = su + sd * SLOT;
      for (int hc = 0; hc < 2; ++hc, ++gc) {
        if (gc >= 1) mbar_wait(&B.accfree, (gc - 1) & 1);                  // the gate threads have read the previous chunk
        if (hc == 0) mbar_wait(&B.full[sy], py);
        tc_fence_after();
        if (elect_one()) {
#pragma unroll
          for (int ks = 0; ks < 8; ++ks) {
            const uint64_t by = desc_k128(ya + (ks >> 2) * KBLK) + (uint64_t)((ks & 3) * 2);
            umma_ts(tmem + T_A, tmem + T_WU + 64 * hc + ks * 8, by, I_G, ks > 0 ? 1u : 0u);
            umma_ts(tmem + T_B, tmem + T_WU + 64 * (2 + hc) + ks * 8, by, I_G, ks > 0 ? 1u : 0u);
          }
          tc_commit(&B.empty[sy]);                                         // y: one release per chunk (count 2)
        }
        __syncwarp();
        if (hc == 0) mbar_wait(&B.rdy[sd], pd);                            // dffn written over dq2
        tc_fence_after();
        if (elect_one()) {
#pragma unroll
          for (int ks = 0; ks < 8; ++ks)
            umma_ts(tmem + T_DH, tmem + T_WD + 64 * hc + ks * 8, desc_k128(da + (ks >> 2) * KBLK) + (uint64_t)((ks & 3) * 2), I_G, ks > 0 ? 1u : 0u);
          if (hc == 1) tc_commit(&B.empty[sd]);
          tc_commit(&B.accfull);
        }
        __syncwarp();
      }
    }
  } else if (warp == 2) {
    // ------------------------------------------------------------------------------------------------ TMA stores: DFFN, HH, DAB
    if (lane == 0) {
      RingPos rp;
      for (int T = 0; T < ntT; ++T, rp.next(NS)) {
        int b, a0, s0; coords(T, b, a0, s0);
        int sd; uint32_t pd; rp.get(U_DQ2, NS, sd, pd);
        mbar_wait(&B.rdy[sd], pd);
        for (int kb = 0; kb < 2; ++kb) tma_store_4d(&mdffn, su + sd * SLOT + kb * KBLK, kb * 64, s0, b, a0);
        tma_store_commit();
        const int xs = T & 1;
        mbar_wait(&B.stfull[xs], (T >> 1) & 1);
        const uint32_t sg = su + O_STG + xs * STG;
        for (int kb = 0; kb < 4; ++kb) tma_store_4d(&mhh, sg + kb * KBLK, kb * 64, s0, b, a0);
        for (int kb = 0; kb < 8; ++kb) tma_store_4d(&mdab, sg + HHB + kb * KBLK, kb * 64, s0, b, a0);
        tma_store_commit();
        tma_store_wait_read0();
        mbar_arrive(&B.empty[sd]);
        mbar_arrive(&B.stfree[xs]);
      }
      tma_store_wait0();
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ dffn (channel per thread), gate (hidden unit per thread)
    const int wg = (warp - 4) >> 2, qw = warp & 3;
    const uint32_t lb = (uint32_t)qw * 32;
    const int c = (int)lb + lane;                                          // channel (dffn) / hidden unit within the chunk (gate)
    const uint32_t cofs = (uint32_t)(c >> 6) * KBLK + (uint32_t)(c & 7) * 2, cch = (uint32_t)((c & 63) >> 3);
    uint32_t xo[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) xo[k] = (cch ^ (uint32_t)k) << 4;
    RingPos rp; int gc = 0;
    for (int T = 0; T < ntT; ++T, rp.next(NS)) {
      int sd; uint32_t pd; rp.get(U_DQ2, NS, sd, pd);
      const uint32_t ax = O_AUX + (T & 1) * AUXST;
      // ---- dffn = rn(dq2 rn(gate_f)) in place over dq2, rows 16 wg ..
      mbar_wait(&B.auxfull[T & 1], (T >> 1) & 1);
      mbar_wait(&B.full[sd], pd);
      {
        const uint32_t da = su + sd * SLOT + cofs + (uint32_t)(16 * wg) * 128u;
        float dq_[16], gf_[16];                                            // all shared-memory loads first
        int at = (16 * wg) % AT;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const int row = 16 * wg + j, grow = (c >> 5) * AT + at;
          gf_[j] = rnb(*reinterpret_cast<const float*>(sm + ax + grow * 128 + ((((c & 31) >> 2) ^ (grow & 7)) << 4) + (c & 3) * 4));
          dq_[j] = lds_bf16(da + j * 128 + xo[row & 7]);
          at = at + 1 == AT ? 0 : at + 1;
        }
#pragma unroll
        for (int j = 0; j < 16; ++j) sts_bf16(da + j * 128 + xo[(16 * wg + j) & 7], dq_[j] * gf_[j]);
      }
      fence_proxy_async();
      named_bar_sync(1 + wg, 128);
      if (qw == 0 && lane == 0) { mbar_arrive(&B.rdy[sd]); mbar_arrive(&B.auxempty[T & 1]); }
      // ---- gate: two 128-hidden chunks; this thread = hidden unit 128 hc + c, rows 16 wg .. 16 wg + 15
      const int xs = T & 1;
      if (T >= 2) mbar_wait(&B.stfree[xs], ((T >> 1) - 1) & 1);            // staging of tile T - 2 has been stored
      const uint32_t sg = su + O_STG + xs * STG;
      for (int hc = 0; hc < 2; ++hc, ++gc) {
        mbar_wait(&B.accfull, gc & 1);
        tc_fence_after();
        uint32_t av[16], bv[16], dv[16];
        const uint32_t tr = tmem + (lb << 16) + 16 * wg;
        tmem_ld16(tr + T_A, av); tmem_ld16(tr + T_B, bv); tmem_ld16(tr + T_DH, dv);
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.accfree);                            // all 8 gate warps
        const int h = 128 * hc + c;                                        // hidden unit
        const uint32_t hofs = (uint32_t)(h >> 6) * KBLK + (uint32_t)(h & 7) * 2, hch = (uint32_t)((h & 63) >> 3);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const int row = 16 * wg + j;
          const uint32_t ro = (uint32_t)row * 128u + ((hch ^ (uint32_t)(row & 7)) << 4) + hofs;
          const float a = __uint_as_float(av[j]), bq = __uint_as_float(bv[j]), dh = __uint_as_float(dv[j]);
          const float sa = sigm(a);
          sts_bf16(sg + ro, a * sa * bq);                                  // HH: block h / 64
          sts_bf16(sg + HHB + ro, dh * bq * sa * (1.f + a * (1.f - sa)));  // DAB: da at column h
          sts_bf16(sg + HHB + 4 * KBLK + ro, dh * a * sa);                 //      db at column 256 + h
        }
      }
      fence_proxy_async();
      named_bar_sync(1 + wg, 128);
      if (qw == 0 && lane == 0) mbar_arrive(&B.stfull[xs]);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
