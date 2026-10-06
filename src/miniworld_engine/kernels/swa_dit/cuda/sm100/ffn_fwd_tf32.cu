// ffn_fwd_tf32.cu — the SWA atom block's second forward stage for the fp32 path on sm_100a, TF32 tensor cores (tcgen05.mma kind::tf32,
// fp32 accumulation). Same equations as the Triton _swa_oproj_ffn_fwd_fp32_kernel, every elementwise step in fp32:
//   gated = sigmoid(g) o;  att = gated Wo^T;  q1 = q + gate_a att;  y = RMS(q1) (1 + scale_f) + shift_f;
//   a | b = y Wu^T;  h = silu(a) b;  ffn = h Wd^T;  out = q1 + gate_f ffn
// optionally saving q1, att, y, ffn (fp32, unrounded) for the backward -- the meanings of the bf16 ffn_fwd2 saves. The MMA operands
// gated, y, h are rounded to TF32 by cvt.rna in the kernel, the weights by wprep_tf32.cu (cached per weight version).
//
// Why not the bf16 design (ffn_fwd2: Wu and Wd resident in TMEM as the A operand, Wo in shared memory, 32-row transposed tiles): in fp32
// the three weights are 448 KB -- Wu and Wd alone would need 768 TMEM columns. So the activations are the A operand, in TMEM (written by
// the row threads with tcgen05.st: thread = row = TMEM lane), and the weights stream from L2 through a ring of 8-KB [64 output rows][32
// inputs] slots as the B operand, each slot feeding four M128 N64 K8 MMAs. The SwiGLU hidden runs in eight 32-unit chunks: a | b of
// chunk j (N = 64: the host packs Wu's rows as [Wu[32 j ..]; Wu[256 + 32 j ..]] per chunk, dispatch._pack_ffn's wab), h written over
// a, then ffn += h Wd[:, 32 j ..]^T; the a | b accumulator is double-buffered so chunk j + 1's up-projection runs under chunk j's SwiGLU.
//
// Tiles: SP = min(A, 16) augments x AT = 128 / SP atoms of one batch element (<= 128 rows), the bf16 kernels' tiling. Everything a
// tile reads arrives by TMA through one ring of 8 k-blocks, in the order the row threads consume it (28 per tile):
//   G0 O0 G1 O1 G2 O2 G3 O3 | q0 q1 q2 q3 | ga0-3 | sh0-3 | sc0-3 | gf0-3
// g, o, q as 4-D boxes [32 channels, AT atoms, 1, SP augments]; the modulation columns gate_a | shift_f | scale_f | gate_f as 2-D
// boxes [32 channels, AT rows] of mod [B S, 6C] (row at of the box serves every augment of atom at). Each group is loaded while the
// previous one is consumed: q / gate_a under phase A, shift_f / scale_f once phase B has released q / gate_a, gate_f under phase D,
// the next tile's g / o under phase D / E. (Round 3: the modulation used to be read with 16-B global loads, one row per lane -- 32
// sectors per warp load, an L1 of ~24 KB next to 232 KB of shared memory -- and those loads were ~45% of the row threads' stall
// samples, Nsight Compute.) Outputs leave by plain 16-B global stores (64 B per thread and unit).
// Warps: 0 TMA producer of the activation ring (lane 0); 1 TMEM allocator + MMA issuer (whole warp waits, elect_one() issues); 3 TMA
// producer of the weight ring (lane 0); 4-11 row threads, two warpgroups, thread = tile row r = TMEM lane r:
//   A  gated: k-block by k-block, warpgroup wg takes channels 16 wg .. 16 wg + 15 of each      -> TMEM T_A
//   B  q1 (and the q1 / att saves), warpgroup wg channels 64 wg ..; the row RMS from both halves -> TMEM T_Q1; y -> T_A (over gated)
//   D  per hidden chunk j: h = silu(a) b for hidden units 16 wg .. 16 wg + 15 of the chunk      -> over a
//   E  out = q1 + gate_f ffn (and the ffn save)
//   shared memory  activation ring NA = 8 k-blocks of [128 rows][32 fp32] SW128 (16 KB) = 128 KB | weight ring NW = 12 slots of
//                  [64 rows][32 fp32] SW128 (8 KB) = 96 KB | row sum-of-squares exchange [2 parities][2][128] fp32 (2 KB) | barriers
//                  -> 231936 B (1 CTA / SM)
//   TMEM (512)     T_A 0 (gated, then y: 128 columns) | T_ATT 128 (att, then the ffn accumulator) | T_Q1 256 (q1, thread-private) |
//                  T_AB 384 + 64 b (a | b of hidden chunk with parity b; h over its a columns)
//   MMA per tile   att 32 (Wo: 2 halves x 4 k-blocks x 4), a | b 8 chunks x 16, ffn 8 chunks x 2 halves x 4: 192 MMAs, 56 weight slots
//   registers      one 16-channel unit at a time (16 TMEM columns + 4 float4s per input): designed for <= 128 / thread
// The a | b buffer of chunk j + 2 is the one ffn(j) reads h from: the MMA warp waits on ffn(j)'s commit before issuing a | b (j + 2)
// unless MMA_INORDER=1 relies on in-order tcgen05.mma execution.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef MMA_INORDER
#define MMA_INORDER 0
#endif

// ------------------------------------------------------------------ kind::tf32 (local: sm100.cuh is the bf16 kernels' header)
__host__ __device__ constexpr uint32_t idesc_tf32(int M, int N, int a_mn = 0, int b_mn = 0) {
  return (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) | ((uint32_t)(N >> 3) << 17) |
         ((uint32_t)(M >> 4) << 24);
}
DEVI void umma_ts_tf32(uint32_t d_tmem, uint32_t a_tmem, uint64_t b, uint32_t idesc, uint32_t accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, p; }"
               :: "r"(d_tmem), "r"(a_tmem), "l"(b), "r"(idesc), "r"(accumulate) : "memory");
}
DEVI uint32_t tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
// sigmoid in fp32 with the approximate reciprocal (rcp.approx, ~1 ulp): its only uses (gated, h) are rounded to TF32 right after, so
// the IEEE reciprocal (__frcp_rn: a refinement sequence per element, 48 K sigmoids per tile) bought nothing
DEVI float sigm32(float x) { return rcpf(1.f + ex2f(-1.4426950408889634f * x)); }
DEVI float4 u2f4(uint4 u) { return make_float4(__uint_as_float(u.x), __uint_as_float(u.y), __uint_as_float(u.z), __uint_as_float(u.w)); }
DEVI float4 ldg4(const float* p) { return u2f4(ldg128(p)); }
DEVI void stg16w(float* p, const uint32_t (&v)[16]) {                       // 64 contiguous bytes
#pragma unroll
  for (int k = 0; k < 4; ++k) stg128(p + 4 * k, make_uint4(v[4 * k], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3]));
}

constexpr int C = 128;
constexpr int KB = 128 * 128;                                              // activation k-block: [128 rows][32 fp32], SW128 (16 KB)
constexpr int WB = 64 * 128;                                               // weight slot: [64 output rows][32 fp32], SW128 (8 KB)
constexpr int NA = 8, NW = 12;
constexpr int NAK = 28, NWK = 56;                                          // ring k-blocks / weight slots per tile
constexpr int K_Q = 8, K_GA = 12, K_SH = 16, K_SC = 20, K_GF = 24;         // ring order within a tile (G / O at 0 .. 7)
constexpr int O_A = 0, O_W = NA * KB, O_RED = O_W + NW * WB, O_BAR = O_RED + 2 * 2 * 128 * 4;
constexpr int SMEM_BYTES = O_BAR + 512;
static_assert(SMEM_BYTES <= 232448, "shared memory");
static_assert((O_W % 1024) == 0 && (O_RED % 1024) == 0, "1 KB alignment of the 128-B-swizzled tiles");
constexpr uint32_t T_A = 0, T_ATT = 128, T_Q1 = 256, T_AB = 384;
constexpr uint32_t I_64 = idesc_tf32(128, 64);

struct Bars {
  uint64_t afull[NA], aempty[NA], wfull[NW], wempty[NW], gfull, attfull, yfull, abfull[2], hfull[2], hfree[2], ffull, ffree;
  uint32_t tmem;
};
static_assert(sizeof(Bars) <= 512, "barriers");

// weight slot k (0 .. 55) of a tile, in the MMA warp's order: att (Wo: halves 0 / 1 x k-blocks 0-3), a | b chunks 0 and 1 (Wab: k-blocks
// 0-3), then for hidden chunk j = 0 .. 7: ffn(j) (Wd: halves 0 / 1 of the output channels, hidden 32 j ..) and a | b chunk j + 2 (j < 6).
// which: 0 Wo [128 out][128 in], 1 Wab [512][128], 2 Wd [128 out][256 hidden]; (c0, c1) = (inner column, row) of the TMA box.
DEVI void wslot(int k, int& which, int& c0, int& c1) {
  if (k < 8) { which = 0; c0 = 32 * (k & 3); c1 = 64 * (k >> 2); return; }
  k -= 8;
  if (k < 8) { which = 1; c0 = 32 * (k & 3); c1 = 64 * (k >> 2); return; }
  k -= 8;
  int j, r;
  if (k < 36) { j = k / 6; r = k % 6; } else { j = 6 + ((k - 36) >> 1); r = (k - 36) & 1; }
  if (r < 2) { which = 2; c0 = 32 * j; c1 = 64 * r; }
  else { which = 1; c0 = 32 * (r - 2); c1 = 64 * (j + 2); }
}

extern "C" __global__ void __launch_bounds__(384, 1)
swa_ffn_fwd_tf32_sm100(const __grid_constant__ CUtensorMap mg, const __grid_constant__ CUtensorMap mo, const __grid_constant__ CUtensorMap mq,
                       const __grid_constant__ CUtensorMap mwo, const __grid_constant__ CUtensorMap mwab, const __grid_constant__ CUtensorMap mwd,
                       const __grid_constant__ CUtensorMap mmod,
                       int S, int A, int Bn, int SP, int AT, int nab, int nag, int ntile, float eps, int save,
                       float* __restrict__ OUT, float* __restrict__ Q1s, float* __restrict__ ATTs,
                       float* __restrict__ Ys, float* __restrict__ FFs) {
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
    mbar_init(&B.gfull, 8); mbar_init(&B.attfull, 1); mbar_init(&B.yfull, 8); mbar_init(&B.ffull, 1); mbar_init(&B.ffree, 8);
    for (int i = 0; i < 2; ++i) { mbar_init(&B.abfull[i], 1); mbar_init(&B.hfull[i], 8); mbar_init(&B.hfree[i], 1); }
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
      pdl_wait();                                                          // g / o / q / mod come from the previous kernels
      const uint32_t abytes = (uint32_t)(SP * AT * 128), mbytes = (uint32_t)(AT * 128);   // full boxes (out of range: zero-filled)
      for (int T = 0; T < ntT; ++T) {
        int b, a0, s0; coords(T, b, a0, s0);
        for (int k = 0; k < NAK; ++k) {
          const int i = NAK * T + k, slot = i % NA;
          if (i >= NA) mbar_wait(&B.aempty[slot], ((i / NA) - 1) & 1);
          const uint32_t dst = su + O_A + slot * KB;
          if (k < K_GA) {
            mbar_expect_tx(&B.afull[slot], abytes);
            const CUtensorMap* m = k < K_Q ? ((k & 1) ? &mo : &mg) : &mq;
            tma_load_4d(dst, m, &B.afull[slot], 32 * (k < K_Q ? k >> 1 : k - K_Q), s0, b, a0);
          } else {                                                         // mod columns 256 (gate_a), 384, 512, 640 (gate_f) + 32 kb
            mbar_expect_tx(&B.afull[slot], mbytes);
            tma_load_2d(dst, &mmod, &B.afull[slot], 2 * C + C * ((k - K_GA) >> 2) + 32 * ((k - K_GA) & 3), b * S + s0);
          }
        }
      }
    }
  } else if (warp == 3) {
    // ------------------------------------------------------------------------------------------------ TMA producer: weight ring
    if (lane == 0) {
      pdl_wait();                                                          // Wo / Wab / Wd: the weight-form kernel runs in the chain
      int j = 0;
      for (int T = 0; T < ntT; ++T)
        for (int k = 0; k < NWK; ++k, ++j) {
          int which, c0, c1; wslot(k, which, c0, c1);
          const int slot = j % NW;
          if (j >= NW) mbar_wait(&B.wempty[slot], ((j / NW) - 1) & 1);
          mbar_expect_tx(&B.wfull[slot], WB);
          tma_load_2d(su + O_W + slot * WB, which == 0 ? &mwo : which == 1 ? &mwab : &mwd, &B.wfull[slot], c0, c1);
        }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    // every weight slot feeds four M128 N64 K8 MMAs (D (+)= A[TMEM columns a0 + 8 ks] B[slot]), then its tcgen05.commit releases it
    int jw = 0;                                                            // weight slots consumed
    auto wnext = [&]() -> int {                                            // wait for slot jw (the producer's order is the MMA order)
      const int slot = jw % NW;
      mbar_wait(&B.wfull[slot], (jw / NW) & 1);
      tc_fence_after();
      return slot;
    };
    for (int T = 0; T < ntT; ++T) {
      mbar_wait(&B.gfull, T & 1);                                          // gated(T) in T_A
      if (T >= 1) mbar_wait(&B.ffree, (T - 1) & 1);                        // ffn(T - 1) read out of T_ATT
      tc_fence_after();
      // ---- att = gated Wo^T -> T_ATT
      for (int hf = 0; hf < 2; ++hf)
        for (int kb = 0; kb < 4; ++kb, ++jw) {
          const int slot = wnext();
          if (elect_one()) {
            const uint64_t dw = desc_k128(su + O_W + slot * WB);
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
              umma_ts_tf32(tmem + T_ATT + 64 * hf, tmem + T_A + (4 * kb + ks) * 8, dw + (uint64_t)(2 * ks), I_64, (kb | ks) ? 1u : 0u);
            tc_commit(&B.wempty[slot]);
            if (hf == 1 && kb == 3) tc_commit(&B.attfull);
          }
          __syncwarp();
        }
      mbar_wait(&B.yfull, T & 1);                                          // y(T) in T_A
      tc_fence_after();
      // ---- a | b of hidden chunk j -> T_AB + 64 (G & 1), G = 8 T + j
      auto ab = [&](int j) {
        const int G = 8 * T + j, jb = G & 1;
        if (!MMA_INORDER && G >= 2) { mbar_wait(&B.hfree[jb], ((G >> 1) - 1) & 1); tc_fence_after(); }   // ffn(G - 2) has read h
        for (int kb = 0; kb < 4; ++kb, ++jw) {
          const int slot = wnext();
          if (elect_one()) {
            const uint64_t dw = desc_k128(su + O_W + slot * WB);
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
              umma_ts_tf32(tmem + T_AB + 64 * jb, tmem + T_A + (4 * kb + ks) * 8, dw + (uint64_t)(2 * ks), I_64, (kb | ks) ? 1u : 0u);
            tc_commit(&B.wempty[slot]);
            if (kb == 3) tc_commit(&B.abfull[jb]);
          }
          __syncwarp();
        }
      };
      // ---- ffn (+)= h_j Wd[:, 32 j ..]^T -> T_ATT (over att, which the row threads have read before y)
      auto ffn = [&](int j) {
        const int G = 8 * T + j, jb = G & 1;
        mbar_wait(&B.hfull[jb], (G >> 1) & 1);
        for (int hf = 0; hf < 2; ++hf, ++jw) {
          const int slot = wnext();
          if (elect_one()) {
            const uint64_t dw = desc_k128(su + O_W + slot * WB);
#pragma unroll
            for (int ks = 0; ks < 4; ++ks)
              umma_ts_tf32(tmem + T_ATT + 64 * hf, tmem + T_AB + 64 * jb + 8 * ks, dw + (uint64_t)(2 * ks), I_64, (j | ks) ? 1u : 0u);
            tc_commit(&B.wempty[slot]);
            if (hf == 1) {
              tc_commit(&B.hfree[jb]);
              if (j == 7) tc_commit(&B.ffull);
            }
          }
          __syncwarp();
        }
      };
      ab(0);
      ab(1);
#pragma unroll 1
      for (int j = 0; j < 8; ++j) {
        ffn(j);
        if (j + 2 < 8) ab(j + 2);
      }
    }
  } else if (warp >= 4) {
    pdl_wait();                                                            // the modulation comes from earlier kernels; output stores
    // ------------------------------------------------------------------------------------------------ row threads
    const int wg = (warp - 4) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const int sp = (int)r / AT, at = (int)r % AT;
    const bool rowok = (int)r < SP * AT;
    float* red = reinterpret_cast<float*>(sm + O_RED);                     // [tile parity][warpgroup][128 rows]
    for (int T = 0; T < ntT; ++T) {
      int b, a0, s0; coords(T, b, a0, s0);
      const int n = (a0 + sp) * Bn + b, s = s0 + at;
      const bool ok = rowok && a0 + sp < A && s < S;
      const bool sv = save && ok;
      const size_t ro = ((size_t)n * S + s) * C;
      const int ia = NAK * T;
      auto aslot = [&](int k) { return su + O_A + ((ia + k) % NA) * KB; };
      auto await_ = [&](int k) { mbar_wait(&B.afull[(ia + k) % NA], ((ia + k) / NA) & 1); };
      auto arel = [&](int k) { if (lane == 0) mbar_arrive(&B.aempty[(ia + k) % NA]); };
      // the modulation k-block of ring position k, 16-B chunk q of this row's atom (row at of the [AT][32] box)
      auto mod4 = [&](int k, int q) { return u2f4(lds128(aslot(k) + sw128((uint32_t)at, (uint32_t)q))); };
      // ---- A: gated = sigmoid(g) o -> T_A (TF32), k-block kb, channels 32 kb + 16 wg .. + 15 (16-B chunks 4 wg .. 4 wg + 3)
      //      (T_A held y(T - 1): every a | b MMA of tile T - 1 has completed -- phase D waited on chunk 7's)
#pragma unroll 1
      for (int kb = 0; kb < 4; ++kb) {
        await_(2 * kb);
        await_(2 * kb + 1);
        uint32_t gr[16];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float4 g4 = u2f4(lds128(aslot(2 * kb) + sw128(r, 4 * wg + e)));
          const float4 o4 = u2f4(lds128(aslot(2 * kb + 1) + sw128(r, 4 * wg + e)));
          gr[4 * e + 0] = tf32r(sigm32(g4.x) * o4.x);
          gr[4 * e + 1] = tf32r(sigm32(g4.y) * o4.y);
          gr[4 * e + 2] = tf32r(sigm32(g4.z) * o4.z);
          gr[4 * e + 3] = tf32r(sigm32(g4.w) * o4.w);
        }
        tmem_st16(trow + T_A + 32 * kb + 16 * wg, gr);
        __syncwarp();
        arel(2 * kb);
        arel(2 * kb + 1);
      }
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.gfull);
      // ---- B: q1 = q + gate_a att -> T_Q1 (q1 / att saves), this warpgroup's channels 64 wg .. 64 wg + 63
      mbar_wait(&B.attfull, T & 1);
      tc_fence_after();
      await_(K_Q + 2 * wg);
      await_(K_Q + 1 + 2 * wg);
      await_(K_GA + 2 * wg);
      await_(K_GA + 1 + 2 * wg);
      float ss0 = 0.f, ss1 = 0.f;
#pragma unroll 1
      for (int u = 0; u < 4; ++u) {
        const int c0 = 64 * wg + 16 * u, kb = c0 >> 5, q0 = (c0 & 31) >> 2;
        uint32_t av[16], q1r[16];
        tmem_ld16(trow + T_ATT + c0, av);
        float4 qv[4], ga[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) { qv[e] = u2f4(lds128(aslot(K_Q + kb) + sw128(r, q0 + e))); ga[e] = mod4(K_GA + kb, q0 + e); }
        tmem_wait_ld();
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float x0 = fmaf(ga[e].x, __uint_as_float(av[4 * e + 0]), qv[e].x), x1 = fmaf(ga[e].y, __uint_as_float(av[4 * e + 1]), qv[e].y);
          const float x2 = fmaf(ga[e].z, __uint_as_float(av[4 * e + 2]), qv[e].z), x3 = fmaf(ga[e].w, __uint_as_float(av[4 * e + 3]), qv[e].w);
          ss0 = fmaf(x0, x0, fmaf(x1, x1, ss0));
          ss1 = fmaf(x2, x2, fmaf(x3, x3, ss1));
          q1r[4 * e + 0] = __float_as_uint(x0); q1r[4 * e + 1] = __float_as_uint(x1);
          q1r[4 * e + 2] = __float_as_uint(x2); q1r[4 * e + 3] = __float_as_uint(x3);
        }
        tmem_st16(trow + T_Q1 + c0, q1r);
        if (sv) { stg16w(ATTs + ro + c0, av); stg16w(Q1s + ro + c0, q1r); }
      }
      __syncwarp();
      // every warp releases all four q and gate_a k-blocks (the slots' count is 8): a warpgroup's arrival on the two of each it does
      // not read cannot land in the slots' previous phase, since attfull (all 8 warps past phase A) orders it after every release of
      // those slots' last use (G / O of this tile)
      for (int k = K_Q; k < K_SH; ++k) arel(k);
      float* rd = red + (T & 1) * 256;
      rd[wg * 128 + r] = ss0 + ss1;
      tmem_wait_st();
      named_bar_sync(1, 256);
      const float rstd = rsqrtf((rd[r] + rd[128 + r]) * (1.f / C) + eps);
      await_(K_SH + 2 * wg);
      await_(K_SH + 1 + 2 * wg);
      await_(K_SC + 2 * wg);
      await_(K_SC + 1 + 2 * wg);
      // ---- y = q1 rstd (1 + scale_f) + shift_f -> T_A (TF32; the att MMAs, gated's only readers, are done), the y save
#pragma unroll 1
      for (int u = 0; u < 4; ++u) {
        const int c0 = 64 * wg + 16 * u, kb = c0 >> 5, q0 = (c0 & 31) >> 2;
        uint32_t qv[16], yr[16];
        tmem_ld16(trow + T_Q1 + c0, qv);
        float4 sh[4], sc[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) { sh[e] = mod4(K_SH + kb, q0 + e); sc[e] = mod4(K_SC + kb, q0 + e); }
        tmem_wait_ld();
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float y0 = fmaf(__uint_as_float(qv[4 * e + 0]) * rstd, 1.f + sc[e].x, sh[e].x);
          const float y1 = fmaf(__uint_as_float(qv[4 * e + 1]) * rstd, 1.f + sc[e].y, sh[e].y);
          const float y2 = fmaf(__uint_as_float(qv[4 * e + 2]) * rstd, 1.f + sc[e].z, sh[e].z);
          const float y3 = fmaf(__uint_as_float(qv[4 * e + 3]) * rstd, 1.f + sc[e].w, sh[e].w);
          qv[4 * e + 0] = __float_as_uint(y0); qv[4 * e + 1] = __float_as_uint(y1);
          qv[4 * e + 2] = __float_as_uint(y2); qv[4 * e + 3] = __float_as_uint(y3);
          yr[4 * e + 0] = tf32r(y0); yr[4 * e + 1] = tf32r(y1); yr[4 * e + 2] = tf32r(y2); yr[4 * e + 3] = tf32r(y3);
        }
        tmem_st16(trow + T_A + c0, yr);
        if (sv) stg16w(Ys + ro + c0, qv);
      }
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.yfull);
      // shift_f / scale_f read: every warp releases all eight (their previous use, q / gate_a, was released by all 8 warps before the
      // named barrier above)
      for (int k = K_SH; k < K_GF; ++k) arel(k);
      // ---- D: h = silu(a) b per hidden chunk, hidden units 16 wg .. of the chunk, written over a (TF32)
#pragma unroll 1
      for (int j = 0; j < 8; ++j) {
        const int G = 8 * T + j, jb = G & 1;
        mbar_wait(&B.abfull[jb], (G >> 1) & 1);
        tc_fence_after();
        uint32_t av[16], bv[16];
        tmem_ld16(trow + T_AB + 64 * jb + 16 * wg, av);
        tmem_ld16(trow + T_AB + 64 * jb + 32 + 16 * wg, bv);
        tmem_wait_ld();
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          const float a = __uint_as_float(av[k]);
          av[k] = tf32r(a * sigm32(a) * __uint_as_float(bv[k]));
        }
        tmem_st16(trow + T_AB + 64 * jb + 16 * wg, av);
        tmem_wait_st();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.hfull[jb]);
      }
      // ---- E: out = q1 + gate_f ffn (the ffn save)
      await_(K_GF + 2 * wg);
      await_(K_GF + 1 + 2 * wg);
      mbar_wait(&B.ffull, T & 1);
      tc_fence_after();
#pragma unroll 1
      for (int u = 0; u < 4; ++u) {
        const int c0 = 64 * wg + 16 * u, kb = c0 >> 5, q0 = (c0 & 31) >> 2;
        uint32_t fv[16], qv[16];
        tmem_ld16(trow + T_ATT + c0, fv);
        tmem_ld16(trow + T_Q1 + c0, qv);
        float4 gf[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) gf[e] = mod4(K_GF + kb, q0 + e);
        tmem_wait_ld();
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          qv[4 * e + 0] = __float_as_uint(fmaf(gf[e].x, __uint_as_float(fv[4 * e + 0]), __uint_as_float(qv[4 * e + 0])));
          qv[4 * e + 1] = __float_as_uint(fmaf(gf[e].y, __uint_as_float(fv[4 * e + 1]), __uint_as_float(qv[4 * e + 1])));
          qv[4 * e + 2] = __float_as_uint(fmaf(gf[e].z, __uint_as_float(fv[4 * e + 2]), __uint_as_float(qv[4 * e + 2])));
          qv[4 * e + 3] = __float_as_uint(fmaf(gf[e].w, __uint_as_float(fv[4 * e + 3]), __uint_as_float(qv[4 * e + 3])));
        }
        if (ok) stg16w(OUT + ro + c0, qv);
        if (sv) stg16w(FFs + ro + c0, fv);
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.ffree);                               // T_ATT read out: tile T + 1's att may accumulate
      // gate_f read: every warp releases all four (their previous use, shift_f, was released by all 8 warps after phase y, and ffull
      // -- every hidden chunk's h from all 8 warps -- orders this after it)
      for (int k = K_GF; k < NAK; ++k) arel(k);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
