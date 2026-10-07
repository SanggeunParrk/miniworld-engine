// ffn_bwd_gate_tf32.cu — first half of the fp32 SWA atom block's FFN backward on sm_100a, TF32 tensor cores (tcgen05.mma kind::tf32,
// fp32 accumulation): the fp32 counterpart of ffn_bwd_gate.cu (same equations, same outputs):
//   dffn = dq2 gate_f;  a | b = y Wu^T (recomputed);  dh = dffn Wd;  sa = sigmoid(a)
//   h = a sa b;  da = dh b sa (1 + a (1 - sa));  db = dh a sa           -> DFFN [M, C], HH [M, 256], DAB = [da | db] [M, 512]
// (dy = DAB Wu and everything after it is ffn_bwd_dy_tf32.cu.) The outputs leave rounded to the nearest TF32 (cvt.rna, low 13 bits
// cleared): they are the MMA operands of the next kernel and of the cuBLAS TF32 weight gradients, which then round nothing further.
//
// Structure of ffn_fwd_tf32.cu: the activations are the A operand in TMEM (y, dffn: 2 x 128 columns, written by the row threads with
// tcgen05.st, thread = row = TMEM lane), the weights stream from L2 through a ring of 8-KB slots as the B operand. The hidden runs in
// eight 32-unit chunks; chunk j needs
//     a | b = y Wab[j]^T      N = 64  (Wab = the host-packed [Wu[32 j ..]; Wu[256 + 32 j ..]], 4 k-blocks of one [64][32] slot)
//     dh    = dffn WdT[j]^T   N = 32  (WdT [256 hidden][128 c] = Wd^T, 4 k-blocks of one [32][32] box)
// into a double-buffered accumulator (a | b | dh = 96 columns), which the row threads turn into h / da / db for the chunk while the
// MMA warp runs chunk j + 1.
//
// Tiles: SP = min(A, 16) augments x AT = 128 / SP atoms of one batch element (<= 128 rows). Activations arrive by TMA through a ring of
// 8 k-blocks ([128 rows][32 fp32] SW128, 16 KB), 12 per tile in the order the row threads consume them: y0 dq2 0 gate_f0 | y1 ... (a
// ring in any other order deadlocks: the row threads hold a k-block until the whole k-block triple has arrived)
// (y, dq2 as 4-D boxes [32 channels, AT atoms, 1, SP augments]; gate_f = columns 5C + 32 kb of mod [B S, 6C], a 2-D box [32, AT]).
// Warps: 0 TMA producer (activations); 1 TMEM allocator + MMA issuer; 3 TMA producer (weights); 4-11 row threads (two warpgroups: wg
// takes channels / hidden units 16 wg .. 16 wg + 15 of every 32).
// Outputs leave by TMA stores: each warpgroup stages its 16 channels / hidden units of a k-block or chunk as [128 rows][64 B] tiles
// (SW64, written one row per thread: conflict-free 16-B stores), then one thread issues the box stores [16, AT, 1, SP] -- the row threads'
// own 64-B pieces (32 distinct sectors half-filled per store) cost 2.9x the time of the stores themselves.
// Shared memory: activation ring 6 x 16 KB | weight ring 10 x 8 KB | staging 2 warpgroups x 3 x 8 KB | barriers.
// TMEM: y 0 | dffn 128 | a|b[2] 256 | dh[2] 384.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

__host__ __device__ constexpr uint32_t idesc_tf32(int M, int N, int a_mn = 0, int b_mn = 0) {
  return (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) | ((uint32_t)(N >> 3) << 17) |
         ((uint32_t)(M >> 4) << 24);
}
DEVI void umma_ts_tf32(uint32_t d_tmem, uint32_t a_tmem, uint64_t b, uint32_t idesc, uint32_t accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, p; }"
               :: "r"(d_tmem), "r"(a_tmem), "l"(b), "r"(idesc), "r"(accumulate) : "memory");
}
// x rounded to the nearest TF32 (ties away from zero), low 13 bits cleared
DEVI uint32_t tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r & 0xffffe000u; }
DEVI float sigm32(float x) { return rcpf(1.f + ex2f(-1.4426950408889634f * x)); }
DEVI float4 u2f4(uint4 u) { return make_float4(__uint_as_float(u.x), __uint_as_float(u.y), __uint_as_float(u.z), __uint_as_float(u.w)); }
DEVI void stg16w(float* p, const uint32_t (&v)[16]) {                       // 64 contiguous bytes
#pragma unroll
  for (int k = 0; k < 4; ++k) stg128(p + 4 * k, make_uint4(v[4 * k], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3]));
}

constexpr int C = 128, NH = 256;
constexpr int KB = 128 * 128;                                              // activation k-block: [128 rows][32 fp32], SW128 (16 KB)
constexpr int WB = 64 * 128;                                               // weight slot: [64 rows][32 fp32], SW128 (8 KB)
constexpr int NA = 6, NW = 10;
constexpr int NAK = 12, NWK = 64;                                          // ring k-blocks / weight slots per tile
constexpr int SGB = 128 * 64;                                              // one staged tile: [128 rows][16 fp32], SW64 (8 KB)
constexpr int O_A = 0, O_W = NA * KB, O_STG = O_W + NW * WB, O_BAR = O_STG + 2 * 3 * SGB;
constexpr int SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory");
static_assert((O_W % 1024) == 0, "1 KB alignment of the 128-B-swizzled tiles");
constexpr uint32_t T_Y = 0, T_DF = 128, T_AB = 256, T_DH = 384;
constexpr uint32_t I_AB = idesc_tf32(128, 64), I_DH = idesc_tf32(128, 32);

struct Bars {
  uint64_t afull[NA], aempty[NA], wfull[NW], wempty[NW], rowsfull, accfull[2], accfree[2];
  uint32_t tmem;
};
static_assert(sizeof(Bars) <= 512, "barriers");

// weight slot k (0 .. 63) of a tile: chunk j = k / 8; r < 4: Wab k-block r (a 64-row slot), else WdT k-block r - 4 (a 32-row box)
DEVI void wslot(int k, bool& dh, int& c0, int& c1) {
  const int j = k >> 3, r = k & 7;
  dh = r >= 4;
  c0 = 32 * (r & 3);
  c1 = dh ? 32 * j : 64 * j;
}

extern "C" __global__ void __launch_bounds__(384, 1)
swa_ffn_bwd_gate_tf32_sm100(const __grid_constant__ CUtensorMap my, const __grid_constant__ CUtensorMap mdq2,
                            const __grid_constant__ CUtensorMap mmod, const __grid_constant__ CUtensorMap mwab,
                            const __grid_constant__ CUtensorMap mwdt, const __grid_constant__ CUtensorMap mdffn,
                            const __grid_constant__ CUtensorMap mhh, const __grid_constant__ CUtensorMap mdab,
                            int S, int A, int Bn, int SP, int AT, int nab, int nag, int ntile) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ntT = (int)blockIdx.x < ntile ? (ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto coords = [&](int T, int& b, int& a0, int& s0) {
    const int t = (int)blockIdx.x + T * (int)gridDim.x;
    const int ab = t % nab, r = t / nab, ag = r % nag;
    b = r / nag; a0 = ag * SP; s0 = ab * AT;
  };

  if (tid == 0) {
    for (int i = 0; i < NA; ++i) { mbar_init(&B.afull[i], 1); mbar_init(&B.aempty[i], 8); }      // 8 = every row warp
    for (int i = 0; i < NW; ++i) { mbar_init(&B.wfull[i], 1); mbar_init(&B.wempty[i], 1); }
    mbar_init(&B.rowsfull, 8);
    for (int i = 0; i < 2; ++i) { mbar_init(&B.accfull[i], 1); mbar_init(&B.accfree[i], 8); }
    fence_barrier_init();
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_launch();

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer: the activation ring
    if (lane == 0) {
      pdl_wait();                                                          // y / dq2 / mod come from the previous kernels
      const uint32_t abytes = (uint32_t)(SP * AT * 128), mbytes = (uint32_t)(AT * 128);   // full boxes (out of range: zero-filled)
      for (int T = 0; T < ntT; ++T) {
        int b, a0, s0; coords(T, b, a0, s0);
        for (int k = 0; k < NAK; ++k) {
          const int i = NAK * T + k, slot = i % NA;
          if (i >= NA) mbar_wait(&B.aempty[slot], ((i / NA) - 1) & 1);
          const uint32_t dst = su + O_A + slot * KB;
          if (k % 3 < 2) {                                                  // k = 3 kb + {0: y, 1: dq2, 2: gate_f}
            mbar_expect_tx(&B.afull[slot], abytes);
            tma_load_4d(dst, k % 3 == 0 ? &my : &mdq2, &B.afull[slot], 32 * (k / 3), s0, b, a0);
          } else {
            mbar_expect_tx(&B.afull[slot], mbytes);
            tma_load_2d(dst, &mmod, &B.afull[slot], 5 * C + 32 * (k / 3), b * S + s0);   // gate_f
          }
        }
      }
    }
  } else if (warp == 3) {
    // ------------------------------------------------------------------------------------------------ TMA producer: weight ring
    if (lane == 0) {
      pdl_wait();                                                          // the weight forms are made by the kernels before
      int j = 0;
      for (int T = 0; T < ntT; ++T)
        for (int k = 0; k < NWK; ++k, ++j) {
          bool dh; int c0, c1; wslot(k, dh, c0, c1);
          const int slot = j % NW;
          if (j >= NW) mbar_wait(&B.wempty[slot], ((j / NW) - 1) & 1);
          mbar_expect_tx(&B.wfull[slot], dh ? WB / 2 : WB);
          tma_load_2d(su + O_W + slot * WB, dh ? &mwdt : &mwab, &B.wfull[slot], c0, c1);
        }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    int jw = 0;                                                            // weight slots consumed
    auto wnext = [&]() -> int {
      const int slot = jw % NW;
      mbar_wait(&B.wfull[slot], (jw / NW) & 1);
      tc_fence_after();
      return slot;
    };
    for (int T = 0; T < ntT; ++T) {
      mbar_wait(&B.rowsfull, T & 1);                                       // y(T), dffn(T) in TMEM
      tc_fence_after();
#pragma unroll 1
      for (int j = 0; j < 8; ++j) {
        const int G = 8 * T + j, jb = G & 1;
        if (G >= 2) { mbar_wait(&B.accfree[jb], ((G >> 1) - 1) & 1); tc_fence_after(); }   // the row threads have read chunk G - 2
        for (int kb = 0; kb < 4; ++kb, ++jw) {                             // a | b = y Wab[j]^T
          const int slot = wnext();
          if (elect_one()) {
            const uint64_t dw = desc_k128(su + O_W + slot * WB);
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
              umma_ts_tf32(tmem + T_AB + 64 * jb, tmem + T_Y + (4 * kb + ks) * 8, dw + (uint64_t)(2 * ks), I_AB, (kb | ks) ? 1u : 0u);
            tc_commit(&B.wempty[slot]);
          }
          __syncwarp();
        }
        for (int kb = 0; kb < 4; ++kb, ++jw) {                             // dh = dffn WdT[j]^T
          const int slot = wnext();
          if (elect_one()) {
            const uint64_t dw = desc_k128(su + O_W + slot * WB);
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
              umma_ts_tf32(tmem + T_DH + 32 * jb, tmem + T_DF + (4 * kb + ks) * 8, dw + (uint64_t)(2 * ks), I_DH, (kb | ks) ? 1u : 0u);
            tc_commit(&B.wempty[slot]);
            if (kb == 3) tc_commit(&B.accfull[jb]);
          }
          __syncwarp();
        }
      }
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ row threads
    const int wg = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const int at = (int)r % AT;
    const bool elected = (warp & 3) == 0 && lane == 0;                     // one thread per warpgroup issues its box stores
    const uint32_t stg = su + O_STG + (uint32_t)wg * 3 * SGB;
    // stage this thread's 16 values of array k (0 .. 2) as its row of the [128][64 B] tile
    auto stage = [&](int k, const uint32_t (&v)[16]) {
#pragma unroll
      for (int q = 0; q < 4; ++q) sts128(stg + k * SGB + sw64(r, q), make_uint4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]));
    };
    auto stage_wait = [&]() { if (elected) tma_store_wait_read0(); named_bar_sync(1 + wg, 128); };   // the previous stores have read the tiles
    auto stage_ready = [&]() { fence_proxy_async(); named_bar_sync(1 + wg, 128); };
    for (int T = 0; T < ntT; ++T) {
      int b, a0, s0; coords(T, b, a0, s0);
      const int ia = NAK * T;
      auto aslot = [&](int k) { return su + O_A + ((ia + k) % NA) * KB; };
      auto await_ = [&](int k) { mbar_wait(&B.afull[(ia + k) % NA], ((ia + k) / NA) & 1); };
      auto arel = [&](int k) { if (lane == 0) mbar_arrive(&B.aempty[(ia + k) % NA]); };
      // ---- y -> TMEM (TF32); dffn = dq2 gate_f -> DFFN (TF32) and TMEM. k-block kb, channels 32 kb + 16 wg .. + 15
#pragma unroll 1
      for (int kb = 0; kb < 4; ++kb) {
        await_(3 * kb);
        await_(3 * kb + 1);
        await_(3 * kb + 2);
        uint32_t yr[16], fr[16];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float4 y4 = u2f4(lds128(aslot(3 * kb) + sw128(r, 4 * wg + e)));
          const float4 d4 = u2f4(lds128(aslot(3 * kb + 1) + sw128(r, 4 * wg + e)));
          const float4 g4 = u2f4(lds128(aslot(3 * kb + 2) + sw128((uint32_t)at, 4 * wg + e)));
          yr[4 * e + 0] = tf32r(y4.x); yr[4 * e + 1] = tf32r(y4.y); yr[4 * e + 2] = tf32r(y4.z); yr[4 * e + 3] = tf32r(y4.w);
          fr[4 * e + 0] = tf32r(d4.x * g4.x); fr[4 * e + 1] = tf32r(d4.y * g4.y);
          fr[4 * e + 2] = tf32r(d4.z * g4.z); fr[4 * e + 3] = tf32r(d4.w * g4.w);
        }
        tmem_st16(trow + T_Y + 32 * kb + 16 * wg, yr);
        tmem_st16(trow + T_DF + 32 * kb + 16 * wg, fr);
        stage_wait();
        stage(0, fr);
        stage_ready();
        if (elected) { tma_store_4d(&mdffn, stg, 32 * kb + 16 * wg, s0, b, a0); tma_store_commit(); }
        __syncwarp();
        arel(3 * kb);
        arel(3 * kb + 1);
        arel(3 * kb + 2);
      }
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.rowsfull);
      // ---- per hidden chunk j: this thread = row r, hidden units 32 j + 16 wg .. + 15
#pragma unroll 1
      for (int j = 0; j < 8; ++j) {
        const int G = 8 * T + j, jb = G & 1;
        mbar_wait(&B.accfull[jb], (G >> 1) & 1);
        tc_fence_after();
        uint32_t av[16], bv[16], dv[16];
        tmem_ld16(trow + T_AB + 64 * jb + 16 * wg, av);
        tmem_ld16(trow + T_AB + 64 * jb + 32 + 16 * wg, bv);
        tmem_ld16(trow + T_DH + 32 * jb + 16 * wg, dv);
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.accfree[jb]);
        uint32_t hh[16], da[16], db[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          const float a = __uint_as_float(av[k]), bq = __uint_as_float(bv[k]), dh = __uint_as_float(dv[k]);
          const float sa = sigm32(a);
          hh[k] = tf32r(a * sa * bq);
          da[k] = tf32r(dh * bq * sa * (1.f + a * (1.f - sa)));
          db[k] = tf32r(dh * a * sa);
        }
        stage_wait();
        stage(0, hh);
        stage(1, da);
        stage(2, db);
        stage_ready();
        if (elected) {
          tma_store_4d(&mdab, stg + SGB, 32 * j + 16 * wg, s0, b, a0);
          tma_store_4d(&mdab, stg + 2 * SGB, NH + 32 * j + 16 * wg, s0, b, a0);
          tma_store_4d(&mhh, stg, 32 * j + 16 * wg, s0, b, a0);
          tma_store_commit();
        }
      }
    }
  }
  if (warp >= 4 && (warp & 3) == 0 && lane == 0) tma_store_wait0();
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
