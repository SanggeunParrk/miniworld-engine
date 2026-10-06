// mod_fwd_tf32.cu — the SWA atom block's adaLN modulation for the fp32 path on sm_100a, one kernel:  mod = silu(c) Wmod^T
// c [R, 128] fp32, Wmod [768, 128] fp32 -> fp32 out [R, 768], on TF32 tensor cores (tcgen05.mma kind::tf32, fp32 accumulation in TMEM).
// silu in fp32 (x / (1 + e^-x): ex2.approx and rcp.approx, a few fp32 ulp -- far below the TF32 rounding that follows) rounded to
// TF32 by cvt.rna; Wmod arrives already rounded to TF32 (the host's
// tf32_fwd round kernel, cached per weight version): both MMA operands are round-to-nearest TF32, not the tensor core's truncation of
// the low mantissa bits (a truncation biases every product toward zero, and that bias does not average out over K = 128).
//
// The product is ~1 GFLOP but writes R x 3 KB of fp32: the kernel is a store stream (5 samples x 6144 atoms: 94 MB, ~15 us at HBM
// speed), and everything else has to hide under it. The first version (one CTA per (128 rows, 128 channels), load -> silu -> 16 MMAs
// -> TMEM -> smem -> TMA store, all serial, 1 CTA / SM) ran ~10x slower than that: nothing overlapped the stores, every row's silu was
// recomputed by all six channel blocks (6x the MUFU work), and each CTA paid its own setup. So now:
//   * persistent CTAs (grid <= #SMs) over items = (128-row tile, channel group); a group is NB = 6 / NG blocks of 128 channels. The
//     host picks NG (1 / 2 / 3 / 6) so that small R still fills the GPU (R = 5120: 40 tiles x 3 groups) and large R computes each
//     row's silu once or a few times (R = 30720: NG = 3, 10 channel blocks per CTA);
//   * warp-specialised pipeline: the c tile (silu'd in place) is double-buffered, Wmod streams from L2 through a ring of 16-KB
//     k-blocks, the [128 rows][128 channels] accumulator is double-buffered in TMEM (block g + 1 accumulates while block g drains),
//     and the epilogue streams 32-channel quarters through two 16-KB staging buffers into TMA stores that stay in flight.
// Warps: 0 TMA producer of the c tiles (lane 0); 1 TMEM allocator + MMA issuer (whole warp waits, elect_one() issues); 2 TMA producer
// of the Wmod ring (lane 0); 3 idle; 4-7 epilogue (thread = tile row = TMEM lane: warp w reads lanes 32 (w % 4) ..); 8-15 silu (thread
// = tile row r, half h: k-blocks 2 h, 2 h + 1).
// Round 2 (measured round 1: ~10 us per item whatever its channel blocks -- the silu was the pipeline's slowest stage): the silu
// ran on 4 warps (one per SMSP) with the IEEE reciprocal (__frcp_rn) and one 16-B chunk per step (the volatile shared-memory accesses
// serialise load -> math -> store), so nothing hid its latency. Now 8 warps, a whole 128-B k-block row (8 chunks) loaded before the
// math (32 independent sigmoids), and the approximate reciprocal.
//   shared memory  c tiles 2 x (4 k-blocks of [128 rows][32 fp32], 128-B swizzle) = 128 KB | Wmod ring NWS = 4 x [128 channels][32
//                  fp32] = 64 KB | out staging 2 x [128 rows][32 fp32] = 32 KB | barriers 512 B  -> 229888 B (1 CTA / SM, 512 threads)
//   TMEM (256)     acc[b] at 128 b: the [128 rows][128 channels] fp32 accumulator of channel block g (b = g & 1)
//   MMA            per channel block: 16 x M128 N128 K8 (SS, both operands K-major, SW128: a 128-B row holds the 32 fp32 of one
//                  k-block, K = 8 is 32 B -> descriptor + 2 per step)
// Registers: one 128-B k-block row (32 fp32) at a time for the silu, 32 fp32 per TMEM load in the epilogue (512 threads: <= 128).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

// ------------------------------------------------------------------ kind::tf32 (local: sm100.cuh is the bf16 kernels' header)
// instruction descriptor: D fp32 (bit 4), A / B tf32 (format 2 at bits 7 / 10), a / b major (0 = K, 1 = MN), N >> 3, M >> 4
__host__ __device__ constexpr uint32_t idesc_tf32(int M, int N, int a_mn = 0, int b_mn = 0) {
  return (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) | ((uint32_t)(N >> 3) << 17) |
         ((uint32_t)(M >> 4) << 24);
}
DEVI void umma_ss_tf32(uint32_t d_tmem, uint64_t a, uint64_t b, uint32_t idesc, uint32_t accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d_tmem), "l"(a), "l"(b), "r"(idesc), "r"(accumulate) : "memory");
}
// fp32 -> the nearest TF32 (ties away from zero), kept in an fp32 container: the MMA then reads it exactly
DEVI uint32_t tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
DEVI float silu32(float x) { return x * rcpf(1.f + ex2f(-1.4426950408889634f * x)); }   // x -> -inf: x * rcp(inf) = -0
// the bulk store group issued two quarters ago has finished READING its staging buffer (at most one group still reading)
DEVI void tma_store_wait_read1() { asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory"); }

constexpr int KB = 128 * 128;                                              // one k-block: [128 rows][32 fp32], 128-B swizzle (16 KB)
constexpr int CT = 4 * KB;                                                 // a c tile: 4 k-blocks (64 KB)
constexpr int NWS = 4;                                                     // Wmod ring: one channel block's four k-blocks
constexpr int O_C = 0, O_W = 2 * CT, O_OUT = O_W + NWS * KB, O_BAR = O_OUT + 2 * KB;
constexpr int SMEM = O_BAR + 512;
static_assert(SMEM <= 232448, "shared memory");
static_assert((O_W % 1024) == 0 && (O_OUT % 1024) == 0 && (O_BAR % 1024) == 0, "1 KB alignment of the 128-B-swizzled tiles");
constexpr uint32_t I_M = idesc_tf32(128, 128);

struct Bars {
  uint64_t cfull[2], cready[2], cfree[2], wfull[NWS], wempty[NWS], accfull[2], accfree[2];
  uint32_t tmem;
};
static_assert(sizeof(Bars) <= 512, "barriers");

extern "C" __global__ void __launch_bounds__(512, 1)
swa_mod_fwd_tf32_sm100(const __grid_constant__ CUtensorMap mc, const __grid_constant__ CUtensorMap mw, const __grid_constant__ CUtensorMap mout,
                       int ntile, int NG) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nitem = ntile * NG, NB = 6 / NG;                               // items, 128-channel blocks per item
  const int ntT = (int)blockIdx.x < nitem ? (nitem - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  auto item = [&](int T, int& r0, int& nb0) {                              // this CTA's item T -> (first row, first channel block)
    const int t = (int)blockIdx.x + T * (int)gridDim.x;
    r0 = (t / NG) * 128;
    nb0 = (t % NG) * NB;
  };

  if (tid == 0) {
    for (int i = 0; i < 2; ++i) {
      mbar_init(&B.cfull[i], 1); mbar_init(&B.cready[i], 8); mbar_init(&B.cfree[i], 1);              // 8 = every silu warp
      mbar_init(&B.accfull[i], 1); mbar_init(&B.accfree[i], 4);                                     // 4 = every epilogue warp
    }
    for (int i = 0; i < NWS; ++i) { mbar_init(&B.wfull[i], 1); mbar_init(&B.wempty[i], 1); }
    fence_barrier_init();
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 256); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_launch();                                                            // PDL: the next kernel may launch now

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ TMA producer: c tiles
    if (lane == 0) {
      pdl_wait();                                                          // c comes from the previous kernels
      for (int T = 0; T < ntT; ++T) {
        int r0, nb0; item(T, r0, nb0);
        const int cb = T & 1;
        if (T >= 2) mbar_wait(&B.cfree[cb], ((T >> 1) - 1) & 1);           // every MMA of item T - 2 has read c[cb]
        mbar_expect_tx(&B.cfull[cb], CT);
        for (int kb = 0; kb < 4; ++kb) tma_load_2d(su + O_C + cb * CT + kb * KB, &mc, &B.cfull[cb], 32 * kb, r0);   // rows >= R: zeros
      }
    }
  } else if (warp == 2) {
    // ------------------------------------------------------------------------------------------------ TMA producer: Wmod ring
    // per item, per channel block nb0 + i, k-blocks 0-3 -- the MMA warp's order
    if (lane == 0) {
      pdl_wait();                                                          // the rounded Wmod may come from the kernel just before
      int j = 0;
      for (int T = 0; T < ntT; ++T) {
        int r0, nb0; item(T, r0, nb0);
        for (int i = 0; i < NB; ++i)
          for (int kb = 0; kb < 4; ++kb, ++j) {
            const int slot = j % NWS;
            if (j >= NWS) mbar_wait(&B.wempty[slot], ((j / NWS) - 1) & 1);
            mbar_expect_tx(&B.wfull[slot], KB);
            tma_load_2d(su + O_W + slot * KB, &mw, &B.wfull[slot], 32 * kb, 128 * (nb0 + i));
          }
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    // channel block g (running count over the CTA's items): acc[g & 1] = silu(c)[T & 1] Wmod[128 (nb0 + i) ..]^T
    int j = 0, g = 0;
    for (int T = 0; T < ntT; ++T) {
      const int cb = T & 1;
      const uint32_t ca = su + O_C + cb * CT;
      mbar_wait(&B.cready[cb], (T >> 1) & 1);                              // silu(c) of item T, rounded, in c[cb]
      tc_fence_after();
      for (int i = 0; i < NB; ++i, ++g) {
        const int ab = g & 1;
        if (g >= 2) { mbar_wait(&B.accfree[ab], ((g >> 1) - 1) & 1); tc_fence_after(); }   // block g - 2 has left acc[ab]
        for (int kb = 0; kb < 4; ++kb, ++j) {
          const int slot = j % NWS;
          mbar_wait(&B.wfull[slot], (j / NWS) & 1);
          tc_fence_after();
          if (elect_one()) {
            const uint64_t da = desc_k128(ca + kb * KB), dw = desc_k128(su + O_W + slot * KB);
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
              umma_ss_tf32(tmem + 128 * ab, da + (uint64_t)(2 * ks), dw + (uint64_t)(2 * ks), I_M, (kb | ks) ? 1u : 0u);
            tc_commit(&B.wempty[slot]);
            if (kb == 3) {
              tc_commit(&B.accfull[ab]);
              if (i == NB - 1) tc_commit(&B.cfree[cb]);                    // every block of item T has read c[cb]
            }
          }
          __syncwarp();
        }
      }
    }
  } else if (warp >= 4 && warp < 8) {
    // ------------------------------------------------------------------------------------------------ epilogue: TMEM -> smem -> TMA store
    const int et = tid - 128;                                              // tile row = TMEM lane
    pdl_wait();                                                            // out may reuse memory an earlier kernel still reads
    const uint32_t trow = tmem + ((uint32_t)((warp & 3) * 32) << 16);
    int g = 0, u = 0;                                                      // channel blocks, 32-channel quarters
    for (int T = 0; T < ntT; ++T) {
      int r0, nb0; item(T, r0, nb0);
      for (int i = 0; i < NB; ++i, ++g) {
        const int ab = g & 1;
        mbar_wait(&B.accfull[ab], (g >> 1) & 1);
        tc_fence_after();
#pragma unroll 1
        for (int q = 0; q < 4; ++q, ++u) {
          uint32_t v[32];
          tmem_ld32(trow + 128 * ab + 32 * q, v);
          tmem_wait_ld();
          if (q == 3) {                                                    // acc[ab] read out: block g + 2 may accumulate
            tc_fence_before();
            __syncwarp();
            if (lane == 0) mbar_arrive(&B.accfree[ab]);
          }
          const uint32_t st = su + O_OUT + (u & 1) * KB;
          if (et == 0) tma_store_wait_read1();                             // the store of quarter u - 2 has read staging[u & 1]
          named_bar_sync(1, 128);
#pragma unroll
          for (int k = 0; k < 8; ++k) sts128(st + sw128((uint32_t)et, k), make_uint4(v[4 * k], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3]));
          fence_proxy_async();
          named_bar_sync(1, 128);
          if (et == 0) {                                                   // clipped at R by the map
            tma_store_2d(&mout, st, 128 * (nb0 + i) + 32 * q, r0);
            tma_store_commit();
          }
        }
      }
    }
    if (et == 0) tma_store_wait0();
  } else if (warp >= 8) {
    // ------------------------------------------------------------------------------------------------ silu in place, row r, half h
    // k-block kb = 2 h + kk: the row's 8 swizzled 16-B chunks loaded first, 32 sigmoids, rounded to TF32, stored back
    const uint32_t r = (uint32_t)(tid - 256) & 127u, h = (uint32_t)(tid - 256) >> 7;
    for (int T = 0; T < ntT; ++T) {
      const int cb = T & 1;
      mbar_wait(&B.cfull[cb], (T >> 1) & 1);
#pragma unroll 1
      for (int kk = 0; kk < 2; ++kk) {
        const uint32_t base = su + O_C + cb * CT + (2 * h + kk) * KB;
        uint4 x[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) x[k] = lds128(base + sw128(r, k));
#pragma unroll
        for (int k = 0; k < 8; ++k)
          x[k] = make_uint4(tf32r(silu32(__uint_as_float(x[k].x))), tf32r(silu32(__uint_as_float(x[k].y))),
                            tf32r(silu32(__uint_as_float(x[k].z))), tf32r(silu32(__uint_as_float(x[k].w))));
#pragma unroll
        for (int k = 0; k < 8; ++k) sts128(base + sw128(r, k), x[k]);
      }
      fence_proxy_async();                                                 // the generic writes, then the MMAs' async-proxy reads
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.cready[cb]);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 256); }
}
