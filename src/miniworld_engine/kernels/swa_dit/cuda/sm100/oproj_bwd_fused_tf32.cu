// oproj_bwd_fused_tf32.cu — the fp32 SWA atom block's whole out-projection backward in ONE kernel on sm_100a (TF32 tensor cores, tcgen05.mma
// kind::tf32, fp32 accumulation): the former oproj prep kernel + cuBLAS dgated = datt Wo + oproj post kernel. The block output is
// q1 = q + gate_a att,  att = gated Wo^T,  gated = sigmoid(g) o  (g: the raw gate, o: the attention output):
//   datt = dq1 gate_a                       (rounded to TF32: the A operand of dgated and an operand of dWo = datt^T gated)
//   dgated = datt Wo                        (TMEM: M = 128 rows, N = 128, K = 128; Wo^T [128 j][128 c] resident in shared memory, 64 KB)
//   so = sigmoid(g);  dO = dgated so;  dG = dgated o so (1 - so);  gated = so o;  Dv[n, h, s] = sum_{d in head h} dO o
// -> DATT, GATED, DO [M, C] (TF32-rounded), dG into dP [M, 4C] at column 3C (TF32-rounded: dP is the operand of dx = dP W), Dv [N, H, S] fp32.
// (d gate_a = sum_aug dq1 att is the FFN adaLN backward kernel's: it already walks the augments of every atom.)
//
// Tile: SP = min(A, 16) augments x AT = 128 / SP atoms of one batch element (<= 128 rows). Per tile the TMA ring carries, in the order the
// row threads consume them, dq1 k-block kb (4-D box [32 channels, AT atoms, 1, SP augments]) and gate_a k-block kb (2-D box [32, AT] of
// mod [B S, 6C]) for kb = 0..3, then o and g of head h for h = 0..3 (16 ring entries; the weights arrive once, at start, as four 16-KB boxes).
// Warps: 0 TMA producer (ring); 1 TMEM allocator + MMA issuer; 3 TMA producer of Wo^T; 4-11 row threads, two warpgroups, thread = tile row =
// TMEM lane: phase A (both warpgroups, channels 32 kb + 16 wg .. + 15): datt -> TMEM T_A (the MMA A operand) and the DATT store;
// the MMA warp then issues 16 MMAs (4 k-blocks x 4 K = 8 steps) into T_D; phase E (warpgroup wg: heads 2 wg, 2 wg + 1, a whole 32-channel head
// per thread, so the head sum Dv is thread-local): dO, dG, gated, Dv.
// Outputs leave by TMA stores from SW128 / SW64 staging tiles. TMEM: datt (A) 0 | dgated (D) 128. Shared memory: Wo^T 64 KB | ring 4 x 16 KB |
// staging 2 warpgroups x 3 tiles x 16 KB | barriers = 229,888 B.
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
DEVI uint32_t tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
DEVI float rtf(float x) { return __uint_as_float(tf32r(x)); }
DEVI float sigm(float x) { return 1.f / (1.f + __expf(-x)); }
DEVI float4 u2f4(uint4 u) { return make_float4(__uint_as_float(u.x), __uint_as_float(u.y), __uint_as_float(u.z), __uint_as_float(u.w)); }

constexpr int C = 128, H = 4;
constexpr int KB = 128 * 128;                                              // a k-block / staging tile: [128 rows][32 fp32], SW128 (16 KB)
constexpr int NA = 8, NAK = 16;                                            // ring slots; ring entries per tile
constexpr int O_W = 0, O_A = 4 * KB, O_STG = O_A + NA * KB, O_BAR = O_STG + 2 * 1 * KB;
constexpr int SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory");
constexpr uint32_t T_A = 0, T_D = 128;
constexpr uint32_t I_D = idesc_tf32(128, 128);

struct Bars {
  uint64_t afull[NA], aempty[NA], wofull, adone, dfull, dfree;
  uint32_t tmem;
};
static_assert(sizeof(Bars) <= 512, "barriers");

extern "C" __global__ void __launch_bounds__(384, 1)
swa_oproj_bwd_fused_tf32_sm100(const __grid_constant__ CUtensorMap mdq1, const __grid_constant__ CUtensorMap mmod,
                               const __grid_constant__ CUtensorMap mo, const __grid_constant__ CUtensorMap mg,
                               const __grid_constant__ CUtensorMap mwot, const __grid_constant__ CUtensorMap mdatt,
                               const __grid_constant__ CUtensorMap mgated, const __grid_constant__ CUtensorMap mdo,
                               const __grid_constant__ CUtensorMap mdp, float* __restrict__ DV,
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
    mbar_init(&B.wofull, 1); mbar_init(&B.adone, 8); mbar_init(&B.dfull, 1); mbar_init(&B.dfree, 8);
    fence_barrier_init();
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_launch();

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer: the ring
    if (lane == 0) {
      pdl_wait();                                                          // dq1 / o / g / mod come from the previous kernels
      const uint32_t abytes = (uint32_t)(SP * AT * 128), mbytes = (uint32_t)(AT * 128);   // full boxes (out of range: zero-filled)
      for (int T = 0; T < ntT; ++T) {
        int b, a0, s0; coords(T, b, a0, s0);
        for (int k = 0; k < NAK; ++k) {
          const int i = NAK * T + k, slot = i % NA;
          if (i >= NA) mbar_wait(&B.aempty[slot], ((i / NA) - 1) & 1);
          const uint32_t dst = su + O_A + slot * KB;
          if (k < 8) {
            if ((k & 1) == 0) { mbar_expect_tx(&B.afull[slot], abytes); tma_load_4d(dst, &mdq1, &B.afull[slot], 32 * (k >> 1), s0, b, a0); }
            else { mbar_expect_tx(&B.afull[slot], mbytes); tma_load_2d(dst, &mmod, &B.afull[slot], 2 * C + 32 * (k >> 1), b * S + s0); }   // gate_a
          } else {
            mbar_expect_tx(&B.afull[slot], abytes);
            tma_load_4d(dst, ((k - 8) & 1) ? &mg : &mo, &B.afull[slot], 32 * ((k - 8) >> 1), s0, b, a0);
          }
        }
      }
    }
  } else if (warp == 3) {
    // ------------------------------------------------------------------------------------------------ TMA producer: Wo^T, once
    if (lane == 0) {
      pdl_wait();                                                          // the weight form is made by a kernel / torch op before
      mbar_expect_tx(&B.wofull, 4 * KB);
      for (int kb = 0; kb < 4; ++kb) tma_load_2d(su + O_W + kb * KB, &mwot, &B.wofull, 32 * kb, 0);
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    mbar_wait(&B.wofull, 0);
    for (int T = 0; T < ntT; ++T) {
      mbar_wait(&B.adone, T & 1);                                          // datt(T) in T_A
      if (T >= 1) mbar_wait(&B.dfree, (T - 1) & 1);                        // the row threads have read dgated(T - 1) out of T_D
      tc_fence_after();
      if (elect_one()) {
        for (int kb = 0; kb < 4; ++kb) {                                   // dgated = datt Wo: B = Wo^T k-block kb [128 j][32 c]
          const uint64_t dw = desc_k128(su + O_W + kb * KB);
#pragma unroll
          for (int ks = 0; ks < 4; ++ks)
            umma_ts_tf32(tmem + T_D, tmem + T_A + (4 * kb + ks) * 8, dw + (uint64_t)(2 * ks), I_D, (kb | ks) ? 1u : 0u);
        }
        tc_commit(&B.dfull);
      }
      __syncwarp();
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ row threads
    const int wg = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const int sp = (int)r / AT, at = (int)r % AT;
    const bool rowok = (int)r < SP * AT;
    const bool elected = (warp & 3) == 0 && lane == 0;                     // one thread per warpgroup issues its box stores
    const uint32_t stg = su + O_STG + (uint32_t)wg * 1 * KB;               // this warpgroup's three staging tiles (dO | dG | gated)
    auto stage_wait = [&]() { if (elected) tma_store_wait_read0(); named_bar_sync(1 + wg, 128); };   // the previous stores have read the tiles
    auto stage_ready = [&]() { fence_proxy_async(); named_bar_sync(1 + wg, 128); };
    for (int T = 0; T < ntT; ++T) {
      int b, a0, s0; coords(T, b, a0, s0);
      const int n = (a0 + sp) * Bn + b, s = s0 + at;
      const bool ok = rowok && a0 + sp < A && s < S;
      const int ia = NAK * T;
      auto slot_of = [&](int k) { return su + O_A + ((ia + k) % NA) * KB; };
      auto await_ = [&](int k) { mbar_wait(&B.afull[(ia + k) % NA], ((ia + k) / NA) & 1); };
      // ---- phase A: datt = dq1 gate_a (this warpgroup's 16 channels of every k-block) -> TMEM T_A and the DATT store (SW64 tile [128][16])
#pragma unroll 1
      for (int kb = 0; kb < 4; ++kb) {
        await_(2 * kb);
        await_(2 * kb + 1);
        uint32_t dt[16];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float4 d4 = u2f4(lds128(slot_of(2 * kb) + sw128(r, 4 * wg + e)));
          const float4 g4 = u2f4(lds128(slot_of(2 * kb + 1) + sw128((uint32_t)at, 4 * wg + e)));
          dt[4 * e + 0] = tf32r(d4.x * g4.x); dt[4 * e + 1] = tf32r(d4.y * g4.y);
          dt[4 * e + 2] = tf32r(d4.z * g4.z); dt[4 * e + 3] = tf32r(d4.w * g4.w);
        }
        tmem_st16(trow + T_A + 32 * kb + 16 * wg, dt);
        stage_wait();
#pragma unroll
        for (int q = 0; q < 4; ++q) sts128(stg + sw64(r, q), make_uint4(dt[4 * q], dt[4 * q + 1], dt[4 * q + 2], dt[4 * q + 3]));
        stage_ready();
        if (elected) { tma_store_4d(&mdatt, stg, 32 * kb + 16 * wg, s0, b, a0); tma_store_commit(); }
        __syncwarp();
        if (lane == 0) { mbar_arrive(&B.aempty[(ia + 2 * kb) % NA]); mbar_arrive(&B.aempty[(ia + 2 * kb + 1) % NA]); }
      }
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.adone);
      // ---- phase E: dgated is in T_D; warpgroup wg owns heads 2 wg, 2 wg + 1: a whole 32-channel head per thread. The o / g ring entries of
      //      head h are released one head at a time by every warp (the slots' count is 8; the other warpgroup only waits for them to
      //      arrive and releases them, so six slots suffice for the eight entries)
      mbar_wait(&B.dfull, T & 1);
      tc_fence_after();
#pragma unroll 1
      for (int h = 0; h < 4; ++h) {
        const bool mine = (h >> 1) == wg;
        await_(8 + 2 * h);
        await_(9 + 2 * h);
        uint32_t v[32];
        float ov[32], gv[32];
        if (mine) {
          tmem_ld32(trow + T_D + 32 * h, v);
          const uint32_t so_ = slot_of(8 + 2 * h), sg_ = slot_of(9 + 2 * h);
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            const float4 o4 = u2f4(lds128(so_ + sw128(r, q))), g4 = u2f4(lds128(sg_ + sw128(r, q)));
            ov[4 * q] = o4.x; ov[4 * q + 1] = o4.y; ov[4 * q + 2] = o4.z; ov[4 * q + 3] = o4.w;
            gv[4 * q] = g4.x; gv[4 * q + 1] = g4.y; gv[4 * q + 2] = g4.z; gv[4 * q + 3] = g4.w;
          }
          tmem_wait_ld();
          if ((h & 1) == 1) {                                              // both heads read out of T_D: the next tile's MMAs may start
            tc_fence_before();
            __syncwarp();
            if (lane == 0) mbar_arrive(&B.dfree);
          }
        }
        __syncwarp();
        if (lane == 0) { mbar_arrive(&B.aempty[(ia + 8 + 2 * h) % NA]); mbar_arrive(&B.aempty[(ia + 9 + 2 * h) % NA]); }
        if (!mine) continue;
        // in place (96 live registers: the 32-element arrays are overwritten): v -> dO, gv -> dG, ov -> gated; Dv from dO and the old o
        float dvs = 0.f;
#pragma unroll
        for (int c = 0; c < 32; ++c) {
          const float dg = __uint_as_float(v[c]), so = sigm(gv[c]), oc = ov[c];
          const float dov = rtf(dg * so);
          dvs = fmaf(dov, oc, dvs);
          v[c] = __float_as_uint(dov);
          gv[c] = rtf(dg * oc * so * (1.f - so));
          ov[c] = rtf(so * oc);
        }
        if (ok) DV[((size_t)n * H + h) * S + s] = dvs;
        // dO, dG, gated: one SW128 tile, reused three times (a wait for the previous store's smem read in between)
#pragma unroll
        for (int t = 0; t < 3; ++t) {
          stage_wait();
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            uint4 u;
            if (t == 0) u = make_uint4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]);
            else if (t == 1) u = make_uint4(__float_as_uint(gv[4 * q]), __float_as_uint(gv[4 * q + 1]), __float_as_uint(gv[4 * q + 2]), __float_as_uint(gv[4 * q + 3]));
            else u = make_uint4(__float_as_uint(ov[4 * q]), __float_as_uint(ov[4 * q + 1]), __float_as_uint(ov[4 * q + 2]), __float_as_uint(ov[4 * q + 3]));
            sts128(stg + sw128(r, q), u);
          }
          stage_ready();
          if (elected) { tma_store_4d(t == 1 ? &mdp : t == 0 ? &mdo : &mgated, stg, (t == 1 ? 3 * C : 0) + 32 * h, s0, b, a0); tma_store_commit(); }
        }
      }
    }
  }
  if (warp >= 4 && (warp & 3) == 0 && lane == 0) tma_store_wait0();
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
