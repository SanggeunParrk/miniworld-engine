// bo_wgrad.cu -- the bias-only token DiT training backward's six weight gradients (and the cond-LN unfold) as ONE launch per block on
// sm_100a (integrations/bias_only_dit_train.py, MINIWORLD_BIAS_ONLY_DIT_BWD_FUSED): what six cuBLAS GEMMs and unfold did.
//
//   g  gradient        = grad^T act over the A L rows      out
//   0  squeeze         dz [M, 768]^T  h [M, 1536]           [768, 1536]
//   1  expand a | b    dab [M, 3072]^T xt [M, 768]          [3072, 768]
//   2  to_out          dy [M, 768]^T  og [M, DA]            [768, DA]
//   3  value | gate    dvg [M, 2 DA]^T xa [M, 768]          [2 DA, 768]
//   4  AdaLN proj.     dG [M, 3072]^T chat [M, 384]         unfold: dWraw = dWn w (w the cond-LN weight of the rows' block) ->
//                                                           DWU [3072, 384]; PW [192, 384] = per 16 rows sum of dWn o Wraw (finalize)
//   5  output gates    dGg [M, 1536]^T c2 [M, 384]          [1536, 384]
//
// ONE output tile per CTA over the whole K (all M rows), no split: the six outputs (7.1 M values at 768 attention channels) cut into
// 128-row bands (128 grad columns) x NC act columns, NC a multiple of 128 up to WMAX (the host's plan: 3 granules of 128 at 768
// channels = 144 tiles of [128][384] for 148 SMs; 4 granules where 3 would need more CTAs than SMs). Every CTA runs the same
// k-blocks in the same order at the same rate, so the CTAs walk the rows together and each 64-row slab of every operand comes from
// HBM about once (the L2 serves the other tiles of its band). No partials, no atomics: bit-identical reruns. (bwdf2 split K 4-way:
// 6.3 waves of items plus a 33-39 us reduction per tile; bwdf3's stream-K left the CTAs at different rows, so each read its operands
// from HBM: 854 us at L768.) Both operands MN-major ([M, *] rows are K): TMA boxes [64 columns][64 rows] (SW128), descriptors
// desc_mn128 with LBO = 8 KB between 64-column blocks, K steps of 16 rows = 2048 B; idesc a_mn = b_mn = 1; an NC of 384 / 512 is two
// MMAs per K step (N 256 + 128 / 256 + 256) into one accumulator of NC TMEM columns.
//
// SYNCHRONIZATION (one CTA, one tile):
//   operands   NST stages    producer (warp 0 lane 0) arrive.expect_tx ofull[s] + TMA (A 2 x 8 KB | B NC / 64 x 8 KB); MMA warp waits
//                            ofull[s], tcgen05.fence::after_thread_sync, MMAs, tcgen05.commit -> oempty[s]; the producer waits
//                            oempty[s] before refilling the stage
//   TMEM       NC columns    the MMA warp commits tfull after the last k-block; the epilogue (warps 4-11) waits tfull,
//                            tcgen05.fence::after_thread_sync, tcgen05.ld + wait::ld; before the teardown every thread
//                            tcgen05.fence::before_thread_sync, __syncthreads, then warp 2 deallocates
// A grid larger than the SMs (not planned for) runs extra tiles in later waves; a CTA never runs two tiles.
// -DTRACE: g_ev[CTA][16] %globaltimer: 0 start, 1 producer: last load issued, 2 MMA: first stage seen, 3 MMA: last commit, 4 epilogue:
// TMEM full, 5 done; 9 tile code, 10 k-blocks (bo_bwd_trace.py).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef DATT
#define DATT 768
#endif
#ifndef WMAX
#define WMAX 384                                                 // the widest tile of the plan (384 or 512 act columns)
#endif
static_assert(DATT == 768 || DATT == 1024, "768 or 1024 attention channels");
static_assert(WMAX == 384 || WMAX == 512, "tile width");

constexpr int QM = 128, KB = 64, ASZ = QM * 128, BSZ = WMAX * 128, STG = ASZ + BSZ;
constexpr int NST = WMAX == 384 ? 3 : 2;                        // 3 x 64 KB / 2 x 80 KB
constexpr int O_OP = 0, O_BAR = NST * STG, SMEM_BYTES = O_BAR + 1024;
static_assert(SMEM_BYTES <= 232448, "shared memory");
__host__ __device__ constexpr int act_cols(int g) { return g == 0 ? 1536 : g == 1 ? 768 : g == 2 ? DATT : g == 3 ? 768 : 384; }

#ifdef TRACE
__device__ unsigned long long g_ev[1024 * 16];
DEVI unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define EV(k) do { if (blockIdx.x < 1024) g_ev[blockIdx.x * 16 + (k)] = gtime(); } while (0)
#define EVV(k, v) do { if (blockIdx.x < 1024) g_ev[blockIdx.x * 16 + (k)] = (unsigned long long)(v); } while (0)
#else
#define EV(k) do { } while (0)
#define EVV(k, v) do { } while (0)
#endif

struct Bars {
  uint64_t ofull[NST], oempty[NST], tfull;
  uint32_t tmem;
};

DEVI uint4 pk8(const float* v) {
  return make_uint4(pack_bf16(v[0], v[1]), pack_bf16(v[2], v[3]), pack_bf16(v[4], v[5]), pack_bf16(v[6], v[7]));
}
DEVI void tmem_ld32w(uint32_t taddr, uint32_t (&r)[32]) { tmem_ld32(taddr, r); tmem_wait_ld(); }

extern "C" __global__ void __launch_bounds__(384, 1)
bo_wgrad_sm100(const __grid_constant__ CUtensorMap ma0, const __grid_constant__ CUtensorMap ma1, const __grid_constant__ CUtensorMap ma2,
               const __grid_constant__ CUtensorMap ma3, const __grid_constant__ CUtensorMap ma4, const __grid_constant__ CUtensorMap ma5,
               const __grid_constant__ CUtensorMap mb0, const __grid_constant__ CUtensorMap mb1, const __grid_constant__ CUtensorMap mb2,
               const __grid_constant__ CUtensorMap mb3, const __grid_constant__ CUtensorMap mb4, const __grid_constant__ CUtensorMap mb5,
               const int* __restrict__ TILES, void* __restrict__ OUT0, void* __restrict__ OUT1, void* __restrict__ OUT2,
               void* __restrict__ OUT3, void* __restrict__ DWU, void* __restrict__ OUT5, float* __restrict__ PW,
               const float* __restrict__ WRAW, const float* __restrict__ W1, const float* __restrict__ W2, int ODT, int NKB) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // tile code: g | mt << 3 | granule0 << 8 | ngranules << 12 (granules of 128 act columns)
  const int code = __ldg(TILES + blockIdx.x);
  const int g = code & 7, mt = (code >> 3) & 31, c0 = 128 * ((code >> 8) & 15), NC = 128 * ((code >> 12) & 7);

  if (tid == 0) {
    for (int i = 0; i < NST; ++i) { mbar_init(&B.ofull[i], 1); mbar_init(&B.oempty[i], 1); }
    mbar_init(&B.tfull, 1);
    fence_barrier_init();
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  if (tid == 0) { EV(0); EVV(9, code); EVV(10, NKB); }
  const CUtensorMap* ma = g == 0 ? &ma0 : g == 1 ? &ma1 : g == 2 ? &ma2 : g == 3 ? &ma3 : g == 4 ? &ma4 : &ma5;
  const CUtensorMap* mb = g == 0 ? &mb0 : g == 1 ? &mb1 : g == 2 ? &mb2 : g == 3 ? &mb3 : g == 4 ? &mb4 : &mb5;

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ producer
    if (lane == 0) {
      prefetch_map(ma); prefetch_map(mb);
      const uint64_t keep = pol_evict_last();                   // every slab is read by the other tiles of its band too
      for (int kb = 0; kb < NKB; ++kb) {
        const int st = kb % NST;
        if (kb >= NST) mbar_wait(&B.oempty[st], ((kb / NST) - 1) & 1);
        mbar_expect_tx(&B.ofull[st], (uint32_t)(ASZ + NC * 128));
        const uint32_t dst = su + O_OP + (uint32_t)st * STG;
#pragma unroll
        for (int a = 0; a < 2; ++a) tma_load_2d_h(dst + 8192 * a, ma, &B.ofull[st], QM * mt + 64 * a, KB * kb, keep);
        for (int b = 0; b < NC / 64; ++b) tma_load_2d_h(dst + ASZ + 8192 * b, mb, &B.ofull[st], c0 + 64 * b, KB * kb, keep);
      }
      EV(1);
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    const int n0 = NC > 256 ? 256 : NC, n1 = NC - n0;           // one or two MMAs per K step
    const uint32_t id0 = n0 == 256 ? idesc_bf16(QM, 256, 1, 1) : idesc_bf16(QM, 128, 1, 1);
    const uint32_t id1 = n1 == 256 ? idesc_bf16(QM, 256, 1, 1) : idesc_bf16(QM, 128, 1, 1);
    for (int kb = 0; kb < NKB; ++kb) {
      const int st = kb % NST;
      mbar_wait(&B.ofull[st], (kb / NST) & 1);
      tc_fence_after();
      if (lane == 0 && kb == 0) EV(2);
      if (elect_one()) {
        const uint32_t sa = su + O_OP + (uint32_t)st * STG, sb = sa + ASZ;
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
          const uint64_t da = desc_mn128(sa + 2048 * ks, 8192);
          umma_ss(tmem, da, desc_mn128(sb + 2048 * ks, 8192), id0, (kb | ks) ? 1u : 0u);
          if (n1) umma_ss(tmem + (uint32_t)n0, da, desc_mn128(sb + (uint32_t)(n0 / 64) * 8192 + 2048 * ks, 8192), id1, (kb | ks) ? 1u : 0u);
        }
        tc_commit(&B.oempty[st]);
        if (kb == NKB - 1) tc_commit(&B.tfull);
      }
      __syncwarp();
    }
    if (lane == 0) EV(3);
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------------ epilogue: the tile out
    const int ew = warp - 4, q4 = warp & 3, hh = ew >> 2;
    const uint32_t r = (uint32_t)(32 * q4 + lane), trow = tmem + ((uint32_t)(32 * q4) << 16);
    mbar_wait(&B.tfull, 0);
    tc_fence_after();
    if (tid == 128) EV(4);
    const int orow = QM * mt + (int)r, ld = act_cols(g);
    const bool f32 = (ODT >> g) & 1;
    // the two warp halves take alternate 32-column chunks: a thread's 32 columns are contiguous (64 B bf16 / 128 B fp32 stores, one
    // TMEM load per chunk; bwdf4's 16-column halves took 11 us mean, 41 us for the unfold tiles)
    for (int q = hh; q < NC / 32; q += 2) {
      uint32_t v[32];
      tmem_ld32w(trow + 32 * q, v);
      float t[32];
#pragma unroll
      for (int k = 0; k < 32; ++k) t[k] = __uint_as_float(v[k]);
      const int cc = c0 + 32 * q;
      if (g == 4) {
#pragma unroll
        for (int h2 = 0; h2 < 2; ++h2) {                          // two 16-column halves (registers)
          const int ch = cc + 16 * h2;
          const float* wv = (orow < 1536 ? W1 : W2) + ch;
          const float* wr = WRAW + (long)orow * 384 + ch;
          float4 w4[4], r4[4];                                   // every load of the half in flight at once
#pragma unroll
          for (int k = 0; k < 4; ++k) { w4[k] = __ldg(reinterpret_cast<const float4*>(wv) + k); r4[k] = __ldg(reinterpret_cast<const float4*>(wr) + k); }
          const float* th = t + 16 * h2;
          float o[16], pw[16];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            o[4 * k] = th[4 * k] * w4[k].x; o[4 * k + 1] = th[4 * k + 1] * w4[k].y; o[4 * k + 2] = th[4 * k + 2] * w4[k].z; o[4 * k + 3] = th[4 * k + 3] * w4[k].w;
            pw[4 * k] = th[4 * k] * r4[k].x; pw[4 * k + 1] = th[4 * k + 1] * r4[k].y; pw[4 * k + 2] = th[4 * k + 2] * r4[k].z; pw[4 * k + 3] = th[4 * k + 3] * r4[k].w;
          }
#pragma unroll
          for (int k = 0; k < 16; ++k) {                         // sum over the 16 rows of this half-warp (fixed tree)
#pragma unroll
            for (int o2 = 1; o2 < 16; o2 <<= 1) pw[k] += __shfl_xor_sync(0xffffffffu, pw[k], o2);
          }
          if ((lane & 15) == 0) {
            float* pr = PW + (long)(8 * mt + 2 * q4 + (lane >> 4)) * 384 + ch;
#pragma unroll
            for (int k = 0; k < 4; ++k) *reinterpret_cast<float4*>(pr + 4 * k) = make_float4(pw[4 * k], pw[4 * k + 1], pw[4 * k + 2], pw[4 * k + 3]);
          }
          if (f32) {
            float* po = reinterpret_cast<float*>(DWU) + (long)orow * 384 + ch;
#pragma unroll
            for (int k = 0; k < 4; ++k) *reinterpret_cast<float4*>(po + 4 * k) = make_float4(o[4 * k], o[4 * k + 1], o[4 * k + 2], o[4 * k + 3]);
          } else {
            __nv_bfloat16* po = reinterpret_cast<__nv_bfloat16*>(DWU) + (long)orow * 384 + ch;
            stg128(po, pk8(o)); stg128(po + 8, pk8(o + 8));
          }
        }
      } else {
        void* out = g == 0 ? OUT0 : g == 1 ? OUT1 : g == 2 ? OUT2 : g == 3 ? OUT3 : OUT5;
        if (f32) {
          float* po = reinterpret_cast<float*>(out) + (long)orow * ld + cc;
#pragma unroll
          for (int k = 0; k < 8; ++k) *reinterpret_cast<float4*>(po + 4 * k) = make_float4(t[4 * k], t[4 * k + 1], t[4 * k + 2], t[4 * k + 3]);
        } else {
          __nv_bfloat16* po = reinterpret_cast<__nv_bfloat16*>(out) + (long)orow * ld + cc;
#pragma unroll
          for (int k = 0; k < 4; ++k) stg128(po + 8 * k, pk8(t + 8 * k));
        }
      }
    }
    if (tid == 128) EV(5);
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
