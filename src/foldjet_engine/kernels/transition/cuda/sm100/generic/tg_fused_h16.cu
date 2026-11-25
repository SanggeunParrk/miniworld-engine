// tg_fused_h16.cu -- the whole fp32-activation Transition forward at d = 64 / 128 / 256 in one tcgen05 kernel (inference), with the products
// on the fp16 tensor cores (kind::f16, fp32 accumulation): fp16 has TF32's 10-bit mantissa and rounds to nearest, so xn, Wa, Wb, h and
// Ws are rounded as the TF32 path rounds them, at twice the TF32 tensor rate and half the operand bytes (the H100 f32io forward's scheme).
//
//   y = x + Ws (silu(Wa xn) * (Wb xn)),   xn = LN(x) gamma + beta
//
// Scales (device, ``sc``; fused_generic_sm100a._h16_forms): each operand is kept below fp16's range by a scale that is 1 unless a bound
// passes 30000: sc = [1 / s_x, s_x s_w, 1 / s_h, s_h s_ws]; Wa16 = Wa / s_w, Wb16 = Wb / s_w, Ws16 = Ws / s_ws.
//
// One CTA per SM, persistent over 128-row tiles. Roles (448 threads):
//   warp 0       TMA producer: per 64-unit hidden chunk c the expand pair [Wa_c ; Wb_c] (one 128-row B operand, 3-stage ring) and the
//                squeeze chunk Ws[:, c] (2-stage ring).
//   warp 1       MMA issuer (and TMEM owner): [a_c | b_c] = xn [Wa_c ; Wb_c]^T (A = xn from TENSOR memory, N = 128, double-buffered
//                accumulators), then y += h_{c-1} Ws_{c-1}^T (A = h from shared memory) one chunk behind.
//   warps 2-9    epilogue: h_c = fp16(silu(a_c) b_c / s_h) into a K-major, 128-B swizzled shared-memory buffer (double-buffered); at the
//                end of a tile y = acc s_h s_ws + x, staged in the idle h buffers and written by TMA.
//   warps 10-13  LayerNorm: one row per thread (its TMEM lane), fp32 statistics over the shared-memory x tile, xn / s_x written to tensor
//                memory as packed fp16 (tcgen05.st), double-buffered.
// TMEM columns: [a | b] x 2 at 0 / 128, y at 256 (D columns), xn x 2 (D / 2 columns each) after it.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
#include "../widths/tsx_store.cuh"
using namespace s100;

#ifndef D
#define D 128
#endif
// D = 256 ("XG"): the y accumulator takes 256 TMEM columns, so [a | b] and xn are single-buffered and the x tile is not staged in shared
// memory (the LayerNorm warps read their rows from global memory / L2); the weight ring has two stages
#define XG (D == 256)
// -DSAVE_BUILD=1 (round 3 part 7): the training build, which also writes xn (fp32 on the TF32 grid), mean and rstd when SAVE is set
#ifndef SAVE_BUILD
#define SAVE_BUILD 0
#endif
// -DHSIG: the SwiGLU sigmoid. 1 (round 3 part 20; default: the inference build): 0.5 tanh.approx(z / 2) + 0.5, one MUFU op a unit
// (the bf16 kernels' TR_SIG 1 form); 0: 1 / (1 + 2^(-z log2 e)), two MUFU ops (the training build: its saved h / gate stay as before);
// 9: timing probe only, no sigmoid (wrong numbers). Measured (random weights, fp64 reference): rms error 3.652e-4 / 3.879e-4 at
// [16 384, 256] / [65 536, 128] n 2 either way (the fp16 rounding of h dominates)
#ifndef HSIG
#define HSIG (SAVE_BUILD ? 0 : 1)
#endif
// timing probes only (wrong numbers; round 3 part 20): TBLN 1 / 2 the LayerNorm statistics (mean 0, rstd 1) / the whole LayerNorm
// skipped, TBW the weight rings loaded once and never refilled, TBEPI 1 / 2 / 3 no y epilogue / no residual loads / no y stores
#ifndef TBLN
#define TBLN 0
#endif
#ifndef TBW
#define TBW 0
#endif
#ifndef TBEPI
#define TBEPI 0
#endif
// -DMLANE=1 (round 3 part 10): the weight / x TMA boxes issued from a lane each (see the producer); default: the inference build only
// (the training build, SAVE_BUILD 1, keeps the one-lane producer unless asked for)
#ifndef MLANE
#define MLANE (!SAVE_BUILD)
#endif
// -DPROBE=1 (analysis builds only): per-CTA %globaltimer stamps into MEANO (as uint64 [grid][NSTAMP]); tile 0 phases + per-chunk events
#ifdef PROBE
constexpr int NSTAMP = 96;
#define STAMP(k) do { unsigned long long _t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_t)); \
  reinterpret_cast<unsigned long long*>(MEANO)[blockIdx.x * NSTAMP + (k)] = _t; } while (0)
#else
#define STAMP(k) do { } while (0)
#endif
constexpr int BN = 64;                           // hidden units per chunk (one 128-byte K-block of fp16 h)
constexpr int NSW = XG ? 2 : 3, NSS = 2, NH = 2; // stages: expand weight chunks, squeeze weight chunks, h buffers
constexpr int NAB = XG ? 1 : 2, NX = XG ? 1 : 2; // [a | b] accumulators, xn buffers (tensor memory)
constexpr int KBD = D / 64;                      // 128-byte K-blocks of a D-wide fp16 row
constexpr int XKB = D / 32;                      // 128-byte K-blocks of a D-wide fp32 row (the x tile)
constexpr int WKB = 2 * BN * 128;                // one K-block of a ring stage: Wa_c rows then Wb_c rows
constexpr int WST = KBD * WKB;                   // a ring stage: [Wa_c ; Wb_c] [2 BN][D]
constexpr int WS_BYTES = D * 128;                // the squeeze chunk Ws[:, c] as [D][BN]: one K-block
constexpr int H_BYTES = 128 * 128;               // h_c [128][64] fp16: one K-block
constexpr int O_W = 0, O_WS = O_W + NSW * WST, O_H = O_WS + NSS * WS_BYTES;
constexpr int O_XS = O_H + NH * H_BYTES;         // the x tile [128][D] fp32, TMA-loaded (128-B swizzled) for the LayerNorm warps
constexpr int XS_BYTES = XG ? 0 : XKB * 128 * 128;
constexpr int O_BAR = O_XS + XS_BYTES;
// -DLNS=1 (round 3 part 10; default: the inference build): gamma | beta staged in shared memory after the barriers (beside 224 KB of
// shared memory the SM has next to no L1: every __ldg of them was an L2 round trip per 32 columns of the store pass), at D 256 (x from
// global memory) the CTA's x tiles prefetched into L2 by a bulk prefetch, the statistics pass 16 loads deep and the store pass loading
// the next 32 columns while it converts the current ones. The same arithmetic: the same bits.
#ifndef LNS
#define LNS (!SAVE_BUILD)
#endif
// -DRXC=1 (round 3 part 10; default: the inference build): the epilogue's residual read coalesced through the staging slab (see there)
#ifndef RXC
#define RXC (!SAVE_BUILD)
#endif
// -DXLN=1 (round 3 part 10; default: the D 256 inference build): the LayerNorm reads its rows in the 16x256b fragment order (see there)
#ifndef XLN
#define XLN (XG && LNS && !SAVE_BUILD)
#endif
static_assert(!XLN || (XG && LNS && !SAVE_BUILD), "XLN: the D 256 inference build with LNS");
// -DYST=1 (round 3 part 10; default with RXC in the inference build): y written by coalesced stores from the staging slab (see there)
#ifndef YST
#define YST (RXC && !SAVE_BUILD)
#endif
static_assert(!YST || RXC, "YST needs RXC");
// -DRXT=1 (round 3 part 20; default: the inference build): a CTA's LAST tile takes its residual x from a TMA copy into the (then idle)
// weight ring and writes y by TMA stores from there (see the epilogue); the other tiles as before. The same arithmetic: the same bits.
// -DEP3=1 (S90; D 128 inference, M a multiple of 128): the y epilogue is split over BOTH warp groups and uses no staging (the h buffers the SwiGLU chunks need are not touched): K-blocks 0 / 1 (32 columns
// each) of tile i by the SwiGLU warps (half 0 / half 1) right after their chunks, K-blocks 2 / 3 by the LayerNorm warps after the LayerNorm of the next tile. The residual x is re-read from global
// memory in 128-B row segments, y written the same way from registers (store_t / load_t); every warp releases the accumulator (`yempty`, 12 arrivals) after its last TMEM load.
#ifndef EP3
#define EP3 0
#endif
#ifndef EP3_A
#define EP3_A 1   // K-blocks of the SwiGLU warps' half 0 / half 1; the LayerNorm warps take the rest of the four
#endif
#ifndef EP3_B
#define EP3_B 1
#endif
static_assert(!EP3 || (D == 128 && !SAVE_BUILD), "EP3: D 128 inference");
#ifndef RXT
#define RXT (MLANE && YST && !SAVE_BUILD && !EP3)
#endif
static_assert(!RXT || (MLANE && YST && !SAVE_BUILD), "RXT: the inference build with the lane-parallel producer");
// -DLNA=1 (round 3 part 11; default: the D 256 inference build): at a CTA's first tile the SwiGLU warps of half 0 (idle until the first
// expand) normalize the first 16-row group of each 32-row quarter, the LayerNorm warps the second (barrier xa0); the same bits.
#ifndef LNA
#define LNA (XLN)
#endif
static_assert(!LNA || XLN, "LNA: the XLN build");
// -DXS0=1 (round 3 part 11; default with LNA): the CTA's first x tile by TMA into shared memory (the second expand stage and the
// squeeze ring, 128 KB, unused until the first tile's LayerNorm is done: the producer holds those loads back until then), the first
// tile's LayerNorm reading it from there (one bulk read of x instead of two global passes; measured: [65 536, 256] n 2 84.3 -> 81.0 us
// back-to-back, one tile per CTA unchanged). The same arithmetic: the same bits.
#ifndef XS0
#define XS0 (LNA)
#endif
static_assert(!XS0 || (LNA && MLANE), "XS0: the LNA build with the lane-parallel producer");
DEVI void stg128(float* p, uint4 v) {
  asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
constexpr int O_GB = O_BAR + 256;
constexpr int SMEM_BYTES = (LNS && O_GB + 8 * D > O_BAR + 1024) ? O_GB + 8 * D : O_BAR + 1024;
static_assert(SMEM_BYTES <= 232448, "shared memory");
static_assert(D == 64 || D == 128 || D == 256, "width");
static_assert(NH * H_BYTES >= 8 * 4096, "the y staging reuses the h buffers");
constexpr int NEPI = 8, NLN = 4;
constexpr int HALF = BN / 2;                     // columns of a chunk per epilogue warp
constexpr int T_AB = XG ? 256 : 0, T_Y = XG ? 0 : 256, T_X = XG ? 384 : 256 + D;
static_assert(T_X + NX * (D / 2) <= 512 && T_AB + NAB * 2 * BN <= 512, "tensor memory");

// kind::f16 instruction descriptor: fp16 A / B, fp32 D, K-major both
__host__ __device__ constexpr uint32_t idesc_f16(int M, int N) {
  return (1u << 4) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}

DEVI float tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return __uint_as_float(r); }

// y blocks cb0 .. cb0 + ncb - 1 (32 fp32 columns each) of the 32 rows of this warp's TMEM lane quarter: residual rows from global memory, y to `Y` (both 128-B row segments through load_t / store_t)
DEVI void y_blocks(uint32_t tm, const float* X, float* Y, size_t off0, int cb0, int ncb, float s_y, int lane, uint64_t* yfull, uint32_t par, uint64_t* yempty) {
  uint4 xr[2][8];
  if (ncb == 0) {                                               // nothing to read: wait for the tile's accumulator, then count this warp out
    mbar_wait(yfull, par);
    __syncwarp();
    if (lane == 0) mbar_arrive(yempty);
    return;
  }
  load_t<32>(X + off0 + cb0 * 32, (size_t)D, xr[0], lane);
  mbar_wait(yfull, par);
  tc_fence_after();
  for (int j = 0; j < ncb; ++j) {
    uint32_t acc[32], o[32];
    tmem_ld32(tm + T_Y + (cb0 + j) * 32, acc);
    if (j + 1 < ncb) load_t<32>(X + off0 + (cb0 + j + 1) * 32, (size_t)D, xr[(j + 1) & 1], lane);
    tmem_wait_ld();
    if (j == ncb - 1) {
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(yempty);
    }
#pragma unroll
    for (int tt = 0; tt < 8; ++tt) {
      o[4 * tt + 0] = __float_as_uint(fmaf(__uint_as_float(acc[4 * tt + 0]), s_y, __uint_as_float(xr[j & 1][tt].x)));
      o[4 * tt + 1] = __float_as_uint(fmaf(__uint_as_float(acc[4 * tt + 1]), s_y, __uint_as_float(xr[j & 1][tt].y)));
      o[4 * tt + 2] = __float_as_uint(fmaf(__uint_as_float(acc[4 * tt + 2]), s_y, __uint_as_float(xr[j & 1][tt].z)));
      o[4 * tt + 3] = __float_as_uint(fmaf(__uint_as_float(acc[4 * tt + 3]), s_y, __uint_as_float(xr[j & 1][tt].w)));
    }
    store_t<32>(Y + off0 + (cb0 + j) * 32, (size_t)D, o, lane);
  }
}
struct Bars {
  uint64_t yfull, yempty, xa0;
  uint64_t xfull[2], xempty[2], xsfull, gbfull;
  uint64_t wfull[NSW], wempty[NSW], sfull[NSS], sempty[NSS];
  uint64_t abfull[2], abempty[2], hfull[NH], hempty[NH];
  uint64_t rxgo, rxfull;   // RXT: the weight ring free after the last expand / the last tile's residual landed
  uint32_t tmem;
};
static_assert(sizeof(Bars) <= 256, "barriers before O_GB");
// tcgen05.st 16x256b, 4 repetitions: lanes l0 + q / 4 (regs 4 i, 4 i + 1) and l0 + 8 + q / 4 (4 i + 2, 4 i + 3) of lane q, columns
// 8 i + 2 (q % 4) and + 1 (i = 0 .. 3)
DEVI void tmem_st16x256_x4(uint32_t a, const uint32_t (&v)[16]) {
  asm volatile("tcgen05.st.sync.aligned.16x256b.x4.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16};"
               :: "r"(a), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]), "r"(v[8]), "r"(v[9]),
                  "r"(v[10]), "r"(v[11]), "r"(v[12]), "r"(v[13]), "r"(v[14]), "r"(v[15]) : "memory");
}
DEVI void tmem_ld16x256_x4(uint32_t a, uint32_t (&v)[16]) {   // the load of the same fragment layout
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), "=r"(v[9]),
                 "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]) : "r"(a) : "memory");
}
DEVI void stg64(float* p, float a, float b) { asm volatile("st.global.v2.f32 [%0], {%1,%2};" :: "l"(p), "f"(a), "f"(b) : "memory"); }
DEVI void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint64_t* bar) {   // 16-byte aligned, a multiple of 16 bytes
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
DEVI void prefetch_l2_bulk(const void* p, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p), "r"(bytes) : "memory");
}

extern "C" __global__ void __launch_bounds__(64 + 32 * (NEPI + NLN), 1)
tg_fused_h16_sm100(const __grid_constant__ CUtensorMap mX, const __grid_constant__ CUtensorMap mWa, const __grid_constant__ CUtensorMap mWb, const __grid_constant__ CUtensorMap mWs,
                   const __grid_constant__ CUtensorMap mY, const float* __restrict__ X, const float* __restrict__ G, const float* __restrict__ Bt,
                   const float* __restrict__ SC, float eps, int M, int HID,
                   float* __restrict__ XN, float* __restrict__ MEANO, float* __restrict__ RSTDO, int SAVE) {   // SAVE: the training saves
  extern __shared__ __align__(1024) uint8_t sm[];
  if (threadIdx.x == 0) STAMP(1);
  const int ntiles = (M + 127) / 128;
  if (!XS0 && LNS && XG && threadIdx.x == 32 * (2 + NEPI) && (int)blockIdx.x < ntiles)   // the first x tile into L2 before the setup
    prefetch_l2_bulk(X + (long)blockIdx.x * 128 * D, (uint32_t)(min(128, M - (int)blockIdx.x * 128) * D * 4));
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int tiles = (M + 127) / 128, NC = HID / BN;

  if (threadIdx.x == 0) {
    mbar_init(&B.yfull, 1); mbar_init(&B.yempty, EP3 ? NEPI + NLN : NEPI); mbar_init(&B.xa0, 4); mbar_init(&B.xsfull, 1); mbar_init(&B.gbfull, 1);
    mbar_init(&B.rxgo, 1); mbar_init(&B.rxfull, 1);
    for (int s = 0; s < 2; ++s) { mbar_init(&B.xfull[s], NLN); mbar_init(&B.xempty[s], 1); }
    for (int s = 0; s < NSW; ++s) { mbar_init(&B.wfull[s], 1); mbar_init(&B.wempty[s], 1); }
    for (int s = 0; s < NSS; ++s) { mbar_init(&B.sfull[s], 1); mbar_init(&B.sempty[s], 1); }
    for (int s = 0; s < 2; ++s) { mbar_init(&B.abfull[s], 1); mbar_init(&B.abempty[s], NEPI); }
    for (int s = 0; s < NH; ++s) { mbar_init(&B.hfull[s], NEPI); mbar_init(&B.hempty[s], 1); }
    fence_barrier_init();
    prefetch_map(&mX); prefetch_map(&mWa); prefetch_map(&mWb); prefetch_map(&mWs); prefetch_map(&mY);
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  if (threadIdx.x == 0) STAMP(0);
#if XLN
  // (round 3 part 10) D 256: the LayerNorm reads its rows in the tcgen05 16x256b fragment order (a quad of lanes holds 16 consecutive
  // columns of two rows, r and r + 8, of a 16-row group): a load touches 8 rows (8 L1 wavefronts) instead of 32 with a row per lane, in
  // both passes. Statistics: each lane sums its quarter of the two rows about the row's first element, the quad adds them by shuffles
  // (summation order differs from the row-per-lane build: the same error to rounding). Group hf of the 32-row quarter wq of tile t.
  const float* gbs = reinterpret_cast<const float*>(sm + O_GB);   // gamma [D] then beta [D] (LNS)
  const float inv_sx_k = SC[0];
  // x of tile-relative row rt, 16-byte column group i: from global memory, or (sx, XS0) from the staged first tile (128-B swizzled
  // [128 rows][32 columns] boxes, 16 KB apart)
  constexpr uint32_t O_X0 = O_W + WST;
  static_assert(!XS0 || O_X0 + 128 * D * 4 <= O_H, "XS0: the staged x tile ends before the h buffers");
  auto xld = [&](bool sx, const float* xr, uint32_t rt, int i) -> uint4 {
    return sx ? lds128(su + O_X0 + (uint32_t)(i >> 3) * 16384u + sw128(rt, (uint32_t)(i & 7))) : ldg128(xr + 4 * i);
  };
  auto xln_stats = [&](long t, int wq, int hf, float (&mr)[4], bool sx) {
    const int qd = lane & 3, rl = lane >> 2;
    const long r0 = t * 128 + wq * 32 + hf * 16 + rl, r1 = r0 + 8;
    const uint32_t rt0 = (uint32_t)(wq * 32 + hf * 16 + rl), rt1 = rt0 + 8;
    const float* x0 = X + (r0 < M ? r0 : 0) * D;
    const float* x1 = X + (r1 < M ? r1 : 0) * D;
    const float c00 = sx ? __uint_as_float(lds32(su + O_X0 + sw128(rt0, 0))) : __ldg(x0);
    const float c01 = sx ? __uint_as_float(lds32(su + O_X0 + sw128(rt1, 0))) : __ldg(x1);
    float s0 = 0.f, q0 = 0.f, s1 = 0.f, q1 = 0.f;
#pragma unroll 8
    for (int j = 0; j < D / 16; ++j) {
      const uint4 wa = xld(sx, x0, rt0, 4 * j + qd), wb = xld(sx, x1, rt1, 4 * j + qd);
      const float a0 = __uint_as_float(wa.x) - c00, a1 = __uint_as_float(wa.y) - c00, a2 = __uint_as_float(wa.z) - c00, a3 = __uint_as_float(wa.w) - c00;
      const float b0 = __uint_as_float(wb.x) - c01, b1 = __uint_as_float(wb.y) - c01, b2 = __uint_as_float(wb.z) - c01, b3 = __uint_as_float(wb.w) - c01;
      s0 += (a0 + a1) + (a2 + a3); q0 += (a0 * a0 + a1 * a1) + (a2 * a2 + a3 * a3);
      s1 += (b0 + b1) + (b2 + b3); q1 += (b0 * b0 + b1 * b1) + (b2 * b2 + b3 * b3);
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      s0 += __shfl_xor_sync(~0u, s0, o); q0 += __shfl_xor_sync(~0u, q0, o);
      s1 += __shfl_xor_sync(~0u, s1, o); q1 += __shfl_xor_sync(~0u, q1, o);
    }
    const float m0 = s0 / (float)D, m1 = s1 / (float)D;
    mr[0] = c00 + m0; mr[2] = c01 + m1;
    mr[1] = rsqrtf(fmaxf(fmaf(-m0, m0, q0 / (float)D), 0.f) + eps); mr[3] = rsqrtf(fmaxf(fmaf(-m1, m1, q1 / (float)D), 0.f) + eps);
  };
  auto xln_store = [&](long t, int wq, int hf, int xb, const float (&mr)[4], bool sx) {
    const int qd = lane & 3, rl = lane >> 2;
    const uint32_t rt0 = (uint32_t)(wq * 32 + hf * 16 + rl), rt1 = rt0 + 8;
    const float4* gs4 = reinterpret_cast<const float4*>(gbs);
    const float mean0 = mr[0], rstd0 = mr[1], mean1 = mr[2], rstd1 = mr[3], inv_sx = inv_sx_k;
    const long r0 = t * 128 + wq * 32 + hf * 16 + rl, r1 = r0 + 8;
    const float* x0 = X + (r0 < M ? r0 : 0) * D;
    const float* x1 = X + (r1 < M ? r1 : 0) * D;
    const uint32_t ta = tmem + ((uint32_t)(wq * 32 + hf * 16) << 16) + T_X + xb * (D / 2);
      uint4 va[4], vb[4];                             // the next 64 columns load while the current ones convert
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) { va[jj] = xld(sx, x0, rt0, 4 * jj + qd); vb[jj] = xld(sx, x1, rt1, 4 * jj + qd); }
#pragma unroll 1
      for (int u = 0; u < D / 64; ++u) {             // 64 columns -> 32 packed fp16 TMEM columns (16x256b.x4)
        uint4 na[4], nb[4];
        if (u + 1 < D / 64) {
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) { na[jj] = xld(sx, x0, rt0, 16 * (u + 1) + 4 * jj + qd); nb[jj] = xld(sx, x1, rt1, 16 * (u + 1) + 4 * jj + qd); }
        }
        uint32_t o[16];
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          const float4 gg = gs4[16 * u + 4 * jj + qd], bb = gs4[D / 4 + 16 * u + 4 * jj + qd];
          const float4 a = make_float4(__uint_as_float(va[jj].x), __uint_as_float(va[jj].y), __uint_as_float(va[jj].z), __uint_as_float(va[jj].w));
          const float4 b = make_float4(__uint_as_float(vb[jj].x), __uint_as_float(vb[jj].y), __uint_as_float(vb[jj].z), __uint_as_float(vb[jj].w));
          o[4 * jj] = f2h2(((a.x - mean0) * rstd0 * gg.x + bb.x) * inv_sx, ((a.y - mean0) * rstd0 * gg.y + bb.y) * inv_sx);
          o[4 * jj + 1] = f2h2(((a.z - mean0) * rstd0 * gg.z + bb.z) * inv_sx, ((a.w - mean0) * rstd0 * gg.w + bb.w) * inv_sx);
          o[4 * jj + 2] = f2h2(((b.x - mean1) * rstd1 * gg.x + bb.x) * inv_sx, ((b.y - mean1) * rstd1 * gg.y + bb.y) * inv_sx);
          o[4 * jj + 3] = f2h2(((b.z - mean1) * rstd1 * gg.z + bb.z) * inv_sx, ((b.w - mean1) * rstd1 * gg.w + bb.w) * inv_sx);
        }
        tmem_st16x256_x4(ta + 32 * u, o);
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) { va[jj] = na[jj]; vb[jj] = nb[jj]; }
      }
  };
#endif

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ producer
#if MLANE
    // (round 3 part 10) every box of a chunk from a lane of its own: one lane issues ~one TMA load per 600 clk (BRIEF6), and a chunk is
    // 2 KBD expand boxes of 8 KB + one squeeze box. Measured (round 3 part 10): the same time as the one-lane producer at every width
    // (the chunk period is set by the MMA / SwiGLU pipeline, not by the issue rate); kept, the same bits.
    // Lanes 0 .. 2 KBD - 1: the expand boxes (lane 0 posts the stage's byte count first); lane 2 KBD: the squeeze chunk.
    constexpr int NWL = 2 * KBD;
    constexpr uint32_t WMASK = (1u << NWL) - 1u;
    if (lane < NWL) {
      const int k = lane >> 1;
      const CUtensorMap* mw = (lane & 1) ? &mWb : &mWa;
      int g = 0;
      for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
        for (int c = 0; c < NC; ++c, ++g) {
          const int sw = g % NSW;
          if (XS0 && g == 1) { mbar_wait(&B.xfull[0], 0); mbar_wait(&B.xa0, 0); }   // stage 1 held the first x tile
          mbar_wait(&B.wempty[sw], ((g / NSW) & 1) ^ 1);
          if (TBW && g >= NSW) { if (lane == 0) mbar_arrive(&B.wfull[sw]); __syncwarp(WMASK); continue; }   // timing probe
          if (lane == 0) mbar_expect_tx(&B.wfull[sw], WST);
          __syncwarp(WMASK);
          tma_load_2d(su + O_W + sw * WST + k * WKB + (lane & 1) * (BN * 128), mw, &B.wfull[sw], k * 64, c * BN);
          if (g == 0 && lane == 0) STAMP(2);
        }
      }
    } else if (lane == NWL) {
      int g = 0;
      for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
        for (int c = 0; c < NC; ++c, ++g) {
          const int ss = g % NSS;
          if (XS0 && g == 0) { mbar_wait(&B.xfull[0], 0); mbar_wait(&B.xa0, 0); }   // the squeeze ring held the first x tile
          mbar_wait(&B.sempty[ss], ((g / NSS) & 1) ^ 1);
          if (TBW && g >= NSS) { mbar_arrive(&B.sfull[ss]); continue; }   // timing probe
          mbar_expect_tx(&B.sfull[ss], WS_BYTES);
          tma_load_2d(su + O_WS + ss * WS_BYTES, &mWs, &B.sfull[ss], c * BN, 0);
          if (g == 0) STAMP(3);
        }
      }
    } else if (RXT && lane >= 16 && lane < 16 + XKB) {
      // (RXT) the residual x of the CTA's last tile into the weight ring once its last expand has completed (rxgo): XKB 16 KB boxes,
      // one lane each (lane 16 posts the byte count first)
      const int tl = (int)blockIdx.x + ((tiles - 1 - (int)blockIdx.x) / (int)gridDim.x) * (int)gridDim.x;
      mbar_wait(&B.rxgo, 0);
      if (lane == 16) mbar_expect_tx(&B.rxfull, XKB * 16384);
      __syncwarp(((1u << XKB) - 1u) << 16);
      tma_load_2d(su + O_W + (uint32_t)(lane - 16) * 16384u, &mX, &B.rxfull, (lane - 16) * 32, tl * 128);
    }
#else
    if (lane == 0) {
      int g = 0;
      for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
        for (int c = 0; c < NC; ++c, ++g) {
          const int sw = g % NSW, ss = g % NSS;
          mbar_wait(&B.wempty[sw], ((g / NSW) & 1) ^ 1);
          mbar_expect_tx(&B.wfull[sw], WST);
          const uint32_t wb = su + O_W + sw * WST;
          for (int k = 0; k < KBD; ++k) {
            tma_load_2d(wb + k * WKB, &mWa, &B.wfull[sw], k * 64, c * BN);
            tma_load_2d(wb + k * WKB + BN * 128, &mWb, &B.wfull[sw], k * 64, c * BN);
          }
          mbar_wait(&B.sempty[ss], ((g / NSS) & 1) ^ 1);
          mbar_expect_tx(&B.sfull[ss], WS_BYTES);
          tma_load_2d(su + O_WS + ss * WS_BYTES, &mWs, &B.sfull[ss], c * BN, 0);
        }
      }
    }
#endif
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    if (lane == 0) {
      constexpr uint32_t IDE = idesc_f16(128, 2 * BN), IDS = idesc_f16(128, D);
      int tc = 0, g = 0;
      for (int t = blockIdx.x; t < tiles; t += gridDim.x, ++tc) {
        const int xb = tc % NX;
        mbar_wait(&B.xfull[xb], (tc / NX) & 1);
        if (LNA && tc == 0) mbar_wait(&B.xa0, 0);
        if (tc == 0) STAMP(4); if (tc < 8) STAMP(68 + tc);
        tc_fence_after();
        const uint32_t tx = tmem + T_X + xb * (D / 2);
        for (int c = 0; c <= NC; ++c) {
          if (c < NC) {                                         // expand of chunk c
            const int gc = g + c, buf = gc % NAB, sw = gc % NSW;
            mbar_wait(&B.abempty[buf], ((gc / NAB) & 1) ^ 1);
            mbar_wait(&B.wfull[sw], (gc / NSW) & 1);
            if (tc == 0 && c < 16) STAMP(20 + c);
            tc_fence_after();
            const uint32_t ta = tmem + T_AB + buf * 2 * BN, wb = su + O_W + sw * WST;
            for (int k = 0; k < KBD; ++k) {
              const uint64_t dw = desc_k128(wb + k * WKB);
#pragma unroll
              for (int kk = 0; kk < 4; ++kk) umma_ts(ta, tx + k * 32 + kk * 8, dw + (uint64_t)(kk * 2), IDE, (k > 0 || kk > 0) ? 1u : 0u);
            }
            tc_commit(&B.wempty[sw]);
            tc_commit(&B.abfull[buf]);
            if (c == NC - 1) tc_commit(&B.xempty[xb]);
            if (RXT && c == NC - 1 && t + (int)gridDim.x >= tiles) tc_commit(&B.rxgo);   // the weight ring is free after this
          }
          if (c >= 1) {                                         // squeeze of chunk c - 1
            const int gc = g + c - 1, hb = gc % NH, ss = gc % NSS;
            if (c == 1) mbar_wait(&B.yempty, (tc & 1) ^ 1);
            mbar_wait(&B.hfull[hb], (gc / NH) & 1);
            mbar_wait(&B.sfull[ss], (gc / NSS) & 1);
            if (tc == 0 && c - 1 < 16) STAMP(36 + c - 1);
            tc_fence_after();
            const uint64_t dh = desc_k128(su + O_H + hb * H_BYTES), dw = desc_k128(su + O_WS + ss * WS_BYTES);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
              umma_ss(tmem + T_Y, dh + (uint64_t)(kk * 2), dw + (uint64_t)(kk * 2), IDS, (c > 1 || kk > 0) ? 1u : 0u);
            tc_commit(&B.sempty[ss]);
            tc_commit(&B.hempty[hb]);
            if (c == NC) tc_commit(&B.yfull);
            if (c == NC && tc == 0) STAMP(5);
          }
        }
        g += NC;
      }
    }
  } else if (warp >= 2 + NEPI) {
    // ------------------------------------------------------------------------------------------------ LayerNorm (warps 10-13)
    const uint32_t lrow = (uint32_t)((warp & 3) * 32) << 16;
    const uint32_t rr = (uint32_t)((warp & 3) * 32 + lane);
    const bool issuer = warp == 2 + NEPI && lane == 0;
    const uint32_t xs = su + O_XS;
    const float inv_sx = SC[0];
#if LNS
    float* gbs = reinterpret_cast<float*>(sm + O_GB);   // gamma [D] then beta [D]
    // two bulk copies on their own barrier (the LayerNorm threads do not stall on the first, cold, global loads of the kernel)
    if (issuer) {
      mbar_expect_tx(&B.gbfull, 8 * D);
      bulk_g2s(su + O_GB, G, 4 * D, &B.gbfull);
      bulk_g2s(su + O_GB + 4 * D, Bt, 4 * D, &B.gbfull);
    }
#endif
#if MLANE
    // the x tile as XKB 16 KB boxes, one lane each (lane 0 posts the byte count first)
    const bool xl = warp == 2 + NEPI && lane < XKB;
    if (!XG && xl && (int)blockIdx.x < tiles) {
      if (lane == 0) mbar_expect_tx(&B.xsfull, XS_BYTES);
      __syncwarp((1u << XKB) - 1u);
      tma_load_2d(xs + lane * 16384, &mX, &B.xsfull, lane * 32, blockIdx.x * 128);
    }
#else
    if (!XG && issuer && (int)blockIdx.x < tiles) {
      mbar_expect_tx(&B.xsfull, XS_BYTES);
      for (int k = 0; k < XKB; ++k) tma_load_2d(xs + k * 16384, &mX, &B.xsfull, k * 32, blockIdx.x * 128);
    }
#endif
#if XS0
    if (warp == 2 + NEPI && lane < XKB && (int)blockIdx.x < tiles) {   // the first x tile: XKB 16 KB boxes, one lane each
      if (lane == 0) mbar_expect_tx(&B.xsfull, 128 * D * 4);
      __syncwarp((1u << XKB) - 1u);
      tma_load_2d(su + O_X0 + lane * 16384, &mX, &B.xsfull, lane * 32, blockIdx.x * 128);
    }
#endif
    int tc = 0;
    for (int t = blockIdx.x; t < tiles; t += gridDim.x, ++tc) {
      const int xb = tc % NX;
      if (LNS && XG && issuer && t + (int)gridDim.x < tiles)   // the next x tile into L2
        prefetch_l2_bulk(X + (long)(t + gridDim.x) * 128 * D, (uint32_t)(min(128, M - (t + (int)gridDim.x) * 128) * D * 4));
      if (!XG) mbar_wait(&B.xsfull, tc & 1);
      if (warp == 2 + NEPI && lane == 0 && tc == 0) STAMP(6);
      const long xrow = (long)t * 128 + rr;
      const float* xg = X + (xrow < M ? xrow : 0) * D;
      auto ld4 = [&](int i) {                         // columns 4 i .. 4 i + 3 of the row
        const uint4 w = XG ? ldg128(xg + 4 * i) : lds128(xs + (uint32_t)(i >> 3) * 16384u + sw128(rr, (uint32_t)(i & 7)));
        return make_float4(__uint_as_float(w.x), __uint_as_float(w.y), __uint_as_float(w.z), __uint_as_float(w.w));
      };
#if XLN
      // (round 3 part 10) D 256: the rows read in the tcgen05 16x256b fragment order (a quad of lanes holds 16 consecutive columns of
      // two rows, r and r + 8, of a 16-row half): a load touches 8 rows (8 L1 wavefronts) instead of 32 with a row per lane, in both
      // passes. Statistics: each lane sums its quarter of the two rows about the row's first element, the quad adds them by shuffles
      // (summation order differs from the row-per-lane build: the same error to rounding)
      {
        // (round 3 part 11) both 16-row groups' statistics before the wait for the xn buffer (it had been group 0's statistics, then
        // after the wait group 0's store, group 1's statistics and store: three of four passes behind the previous tile's last expand).
        // At the CTA's first tile with LNA the SwiGLU warps of half 0 take group 0 (they are idle until the first expand). Same bits.
        const int h0 = (LNA && tc == 0) ? 1 : 0;
        if (XS0 && tc == 0) mbar_wait(&B.xsfull, 0);
        if (warp == 2 + NEPI && lane == 0 && tc == 0) STAMP(11);
        float mr[2][4];
#pragma unroll
        for (int hf = 0; hf < 2; ++hf)
          if (hf >= h0) { if (TBLN == 0) xln_stats(t, warp & 3, hf, mr[hf], XS0 && tc == 0); else { mr[hf][0] = mr[hf][2] = 0.f; mr[hf][1] = mr[hf][3] = 1.f; } }
        if (warp == 2 + NEPI && lane == 0 && tc == 0) STAMP(7);
        if (tc == 0) mbar_wait(&B.gbfull, 0);   // gamma | beta staged (the statistics pass ran under their copy)
        mbar_wait(&B.xempty[xb], ((tc / NX) & 1) ^ 1);
        tc_fence_after();
        if (warp == 2 + NEPI && lane == 0 && tc == 0) STAMP(12);
#pragma unroll
        for (int hf = 0; hf < 2; ++hf)
          if (hf >= h0 && TBLN < 2) xln_store(t, warp & 3, hf, xb, mr[hf], XS0 && tc == 0);
        if (warp == 2 + NEPI && lane == 0 && tc == 0) STAMP(13);
      }
#else
      float mean, rstd;
      if (XG) {
        // the row comes from global memory, and beside 224 KB of shared memory the SM has next to no L1 cache: every pass over it is an
        // L2 round trip per lane (the two statistics passes + the store pass were 31 % of the kernel). One statistics pass: sums of
        // (x - c0) and (x - c0)^2 about the row's first element c0, var = E[(x - c0)^2] - (mean - c0)^2
        const float c0 = ld4(0).x;
        float s = 0.f, q = 0.f;
#if LNS
#pragma unroll 16
#else
#pragma unroll 8
#endif
        for (int i = 0; i < D / 4; ++i) {
          const float4 v = ld4(i);
          const float a = v.x - c0, b = v.y - c0, c = v.z - c0, e = v.w - c0;
          s += (a + b) + (c + e);
          q += (a * a + b * b) + (c * c + e * e);
        }
        const float ms = s / (float)D;
        mean = c0 + ms;
        rstd = rsqrtf(fmaxf(fmaf(-ms, ms, q / (float)D), 0.f) + eps);
        if (warp == 2 + NEPI && lane == 0 && tc == 0) STAMP(7);
      } else if (TBLN) {
        mean = 0.f; rstd = 1.f;   // timing probe
      } else {
        float s = 0.f;
#pragma unroll 8
        for (int i = 0; i < D / 4; ++i) { const float4 v = ld4(i); s += (v.x + v.y) + (v.z + v.w); }
        mean = s / (float)D;
        float q = 0.f;
#pragma unroll 8
        for (int i = 0; i < D / 4; ++i) {
          const float4 v = ld4(i);
          const float a = v.x - mean, b = v.y - mean, c = v.z - mean, e = v.w - mean;
          q += (a * a + b * b) + (c * c + e * e);
        }
        rstd = rsqrtf(q / (float)D + eps);
        if (warp == 2 + NEPI && lane == 0 && tc == 0) STAMP(7);
      }
      const bool save = SAVE_BUILD && SAVE && xrow < M;   // the inference build (SAVE_BUILD 0) has no save code
      if (save) { MEANO[xrow] = mean; RSTDO[xrow] = rstd; }
      float* xnr = XN + (save ? xrow : 0) * D;
#if LNS
      const float4* gs4 = reinterpret_cast<const float4*>(gbs);
      float4 cv[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) cv[j] = XG ? ld4(j) : make_float4(0.f, 0.f, 0.f, 0.f);
      if (tc == 0) mbar_wait(&B.gbfull, 0);   // gamma | beta staged (the statistics pass ran under their copy)
      mbar_wait(&B.xempty[xb], ((tc / NX) & 1) ^ 1);
      tc_fence_after();
#pragma unroll 1
      for (int u = 0; u < (TBLN >= 2 ? 0 : D / 32); ++u) {   // 32 columns -> 16 packed fp16 TMEM columns
        float4 nv[8];
        if (XG && u + 1 < D / 32) {
#pragma unroll
          for (int j = 0; j < 8; ++j) nv[j] = ld4(8 * (u + 1) + j);
        }
        uint32_t o[16];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float4 v = XG ? cv[j] : ld4(8 * u + j);
          const float4 gg = gs4[8 * u + j], bb = gs4[D / 4 + 8 * u + j];
          o[2 * j] = f2h2(((v.x - mean) * rstd * gg.x + bb.x) * inv_sx, ((v.y - mean) * rstd * gg.y + bb.y) * inv_sx);
          o[2 * j + 1] = f2h2(((v.z - mean) * rstd * gg.z + bb.z) * inv_sx, ((v.w - mean) * rstd * gg.w + bb.w) * inv_sx);
          if (save)
            reinterpret_cast<float4*>(xnr)[8 * u + j] =
                make_float4(tf32r((v.x - mean) * rstd * gg.x + bb.x), tf32r((v.y - mean) * rstd * gg.y + bb.y),
                            tf32r((v.z - mean) * rstd * gg.z + bb.z), tf32r((v.w - mean) * rstd * gg.w + bb.w));
        }
        tmem_st16(tmem + lrow + T_X + xb * (D / 2) + 16 * u, o);
        if (XG) {
#pragma unroll
          for (int j = 0; j < 8; ++j) cv[j] = nv[j];
        }
      }
#else
      mbar_wait(&B.xempty[xb], ((tc / NX) & 1) ^ 1);
      tc_fence_after();
#pragma unroll 1
      for (int u = 0; u < D / 32; ++u) {                 // 32 columns -> 16 packed fp16 TMEM columns
        uint32_t o[16];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float4 v = ld4(8 * u + j);
          const float4 gg = __ldg(reinterpret_cast<const float4*>(G) + 8 * u + j), bb = __ldg(reinterpret_cast<const float4*>(Bt) + 8 * u + j);
          o[2 * j] = f2h2(((v.x - mean) * rstd * gg.x + bb.x) * inv_sx, ((v.y - mean) * rstd * gg.y + bb.y) * inv_sx);
          o[2 * j + 1] = f2h2(((v.z - mean) * rstd * gg.z + bb.z) * inv_sx, ((v.w - mean) * rstd * gg.w + bb.w) * inv_sx);
          if (save)   // (round 3 part 7) the backward's xn: fp32 on the TF32 grid (the generic backward's GATE / weight-gradient operand)
            reinterpret_cast<float4*>(xnr)[8 * u + j] =
                make_float4(tf32r((v.x - mean) * rstd * gg.x + bb.x), tf32r((v.y - mean) * rstd * gg.y + bb.y),
                            tf32r((v.z - mean) * rstd * gg.z + bb.z), tf32r((v.w - mean) * rstd * gg.w + bb.w));
        }
        tmem_st16(tmem + lrow + T_X + xb * (D / 2) + 16 * u, o);
      }
#endif
#endif  // XLN
      // XS has been read by all four warps: the next tile's x may land
      if (!XG) named_bar_sync(2, 32 * NLN);
#if MLANE
      if (!XG && xl && t + (int)gridDim.x < tiles) {
        fence_proxy_async();
        if (lane == 0) mbar_expect_tx(&B.xsfull, XS_BYTES);
        __syncwarp((1u << XKB) - 1u);
        tma_load_2d(xs + lane * 16384, &mX, &B.xsfull, lane * 32, (t + gridDim.x) * 128);
      }
#else
      if (!XG && issuer && t + (int)gridDim.x < tiles) {
        fence_proxy_async();
        mbar_expect_tx(&B.xsfull, XS_BYTES);
        for (int k = 0; k < XKB; ++k) tma_load_2d(xs + k * 16384, &mX, &B.xsfull, k * 32, (t + gridDim.x) * 128);
      }
#endif
      tmem_wait_st();
      tc_fence_before();
      if (XS0 && tc == 0) fence_proxy_async();   // the staged x tile's reads before the producer's TMA writes there
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.xfull[xb]);
      if (warp == 2 + NEPI && lane == 0) { if (tc == 0) STAMP(8); if (tc < 8) STAMP(76 + tc); }
#if EP3
      if (tc >= 1) {                                           // the previous tile's K-blocks 2, 3, after this tile's LayerNorm
        const int tp = t - (int)gridDim.x;
        y_blocks(tmem + lrow, X, XN, ((size_t)tp * 128 + (rr & ~7u)) * D + (size_t)(lane & 7) * 4, EP3_A + EP3_B, 4 - EP3_A - EP3_B, SC[3], lane, &B.yfull, (tc - 1) & 1, &B.yempty);
      }
#endif
    }
#if EP3
    if (tc >= 1) {
      const int tl = (int)blockIdx.x + (tc - 1) * (int)gridDim.x;
      y_blocks(tmem + lrow, X, XN, ((size_t)tl * 128 + (rr & ~7u)) * D + (size_t)(lane & 7) * 4, EP3_A + EP3_B, 4 - EP3_A - EP3_B, SC[3], lane, &B.yfull, (tc - 1) & 1, &B.yempty);
    }
#endif
  } else {
    // ------------------------------------------------------------------------------------------------ epilogue (warps 2-9)
    const int ew = warp - 2, q4 = warp & 3, half = ew >> 2;
    const uint32_t r = (uint32_t)(q4 * 32 + lane);           // row within the tile
    const uint32_t lrow = (uint32_t)(q4 * 32) << 16;
    const float s_ab = SC[1], inv_sh = SC[2], s_y = SC[3];
    const f2 K1 = mk2(-1.4426950408889634f * s_ab, -1.4426950408889634f * s_ab), K2 = mk2(s_ab * s_ab * inv_sh, s_ab * s_ab * inv_sh);
    const f2 KT = mk2(0.5f * s_ab, 0.5f * s_ab);
    (void)K1; (void)KT;
#if LNA
    if (half == 0 && (int)blockIdx.x < tiles) {   // the first tile's first 16-row group of quarter q4 (see LNA)
      float mr[4];
      if (XS0) mbar_wait(&B.xsfull, 0);
      if (ew == 0 && lane == 0) STAMP(14);
      if (TBLN == 0) xln_stats((long)blockIdx.x, q4, 0, mr, XS0); else { mr[0] = mr[2] = 0.f; mr[1] = mr[3] = 1.f; }
      if (ew == 0 && lane == 0) STAMP(15);
      mbar_wait(&B.gbfull, 0);
      mbar_wait(&B.xempty[0], 1);                 // a fresh barrier: passes
      tc_fence_after();
      if (TBLN < 2) xln_store((long)blockIdx.x, q4, 0, 0, mr, XS0);
      if (ew == 0 && lane == 0) STAMP(16);
      tmem_wait_st();
      tc_fence_before();
      if (XS0) fence_proxy_async();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.xa0);
    }
#endif
    int tc = 0, g = 0;
    for (int t = blockIdx.x; t < tiles; t += gridDim.x, ++tc) {
      for (int c = 0; c < NC; ++c, ++g) {
        const int buf = g % NAB, hb = g % NH;
        mbar_wait(&B.abfull[buf], (g / NAB) & 1);
        if (ew == 0 && lane == 0 && tc == 0 && c < 16) STAMP(52 + c);
        tc_fence_after();
        uint32_t hw[HALF / 2];
#pragma unroll
        for (int p = 0; p < HALF / 16; ++p) {
          uint32_t a[16], b[16];
          tmem_ld16(tmem + lrow + T_AB + buf * 2 * BN + half * HALF + 16 * p, a);
          tmem_ld16(tmem + lrow + T_AB + buf * 2 * BN + BN + half * HALF + 16 * p, b);
          tmem_wait_ld();
#pragma unroll
          for (int e = 0; e < 16; e += 2) {
            // the scales folded into two constants and the products packed (fp32x2): t = a k1 = -a s_ab log2 e,
            // h = (a / (1 + 2^t)) (b k2), k2 = s_ab^2 / s_h
#if HSIG == 1
            // (round 3 part 20) sigmoid(z) = 0.5 tanh(z / 2) + 0.5, z = a s_ab: one MUFU op per unit
            const f2 A = mk2u(a[e], a[e + 1]), T = mul2(A, KT);
            const f2 G = fma2(mk2(tanhf_approx(lo2(T)), tanhf_approx(hi2(T))), mk2(0.5f, 0.5f), mk2(0.5f, 0.5f));
#elif HSIG == 9
            const f2 A = mk2u(a[e], a[e + 1]);   // timing probe only (wrong numbers): no sigmoid
            const f2 G = mk2(1.f, 1.f);
#else
            const f2 A = mk2u(a[e], a[e + 1]), T = mul2(A, K1);
            const f2 G = mk2(rcpf(__fadd_rn(1.f, ex2f(lo2(T)))), rcpf(__fadd_rn(1.f, ex2f(hi2(T)))));
#endif
            const f2 H = mul2(mul2(A, G), mul2(mk2u(b[e], b[e + 1]), K2));
            hw[8 * p + e / 2] = f2h2(lo2(H), hi2(H));
          }
        }
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.abempty[buf]);
        mbar_wait(&B.hempty[hb], ((g / NH) & 1) ^ 1);
        const uint32_t hbase = su + O_H + hb * H_BYTES;
#pragma unroll
        for (int q = 0; q < HALF / 8; ++q)                   // 8 halves per 16-byte chunk
          sts128(hbase + sw128(r, half * (HALF / 8) + q), make_uint4(hw[4 * q], hw[4 * q + 1], hw[4 * q + 2], hw[4 * q + 3]));
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.hfull[hb]);
      }
#if EP3
      y_blocks(tmem + lrow, X, XN, ((size_t)t * 128 + (r & ~7u)) * D + (size_t)(lane & 7) * 4, half ? EP3_A : 0, half ? EP3_B : EP3_A, s_y, lane, &B.yfull, tc & 1, &B.yempty);
      continue;
#endif
#if RXT
      if (t + (int)gridDim.x >= tiles) {
        // (round 3 part 20, RXT) the CTA's last tile: its residual x is in the (idle) weight ring as XKB swizzled [128][32] boxes; each
        // lane adds its row's accumulator in place and the warp's [32 rows][32 columns] sub-box goes out by a TMA store (no global
        // residual loads, no slab round trip, no per-slab store wait). Same arithmetic: the same bits.
        mbar_wait(&B.yfull, tc & 1);
        if (ew == 0 && lane == 0 && tc == 0) STAMP(9);
        mbar_wait(&B.rxfull, 0);
        tc_fence_after();
#pragma unroll
        for (int u = 0; u < (TBEPI == 1 ? 0 : D / 64); ++u) {
          const int col = half * (D / 2) + u * 32;
          const uint32_t box = su + O_W + (uint32_t)(col / 32) * 16384u;
          uint32_t a0[16], a1[16];
          tmem_ld16(tmem + lrow + T_Y + col, a0);
          tmem_ld16(tmem + lrow + T_Y + col + 16, a1);
          tmem_wait_ld();
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            const uint32_t ad = box + sw128(r, (uint32_t)q);
            const uint4 w = lds128(ad);
            const uint32_t* av = q < 4 ? &a0[4 * q] : &a1[4 * (q - 4)];
            sts128(ad, make_uint4(__float_as_uint(fmaf(__uint_as_float(av[0]), s_y, __uint_as_float(w.x))),
                                  __float_as_uint(fmaf(__uint_as_float(av[1]), s_y, __uint_as_float(w.y))),
                                  __float_as_uint(fmaf(__uint_as_float(av[2]), s_y, __uint_as_float(w.z))),
                                  __float_as_uint(fmaf(__uint_as_float(av[3]), s_y, __uint_as_float(w.w)))));
          }
          fence_proxy_async();
          __syncwarp();
          if (lane == 0 && TBEPI != 3) { tma_store_2d(&mY, box + (uint32_t)q4 * 4096u, col, t * 128 + q4 * 32); tma_store_commit(); }
        }
        tc_fence_before();
        __syncwarp();
        if (lane == 0) { mbar_arrive(&B.yempty); tma_store_wait_read0(); }   // the stores' shared-memory reads done before the exit
        __syncwarp();
        if (ew == 0 && lane == 0 && tc < 8) STAMP(84 + tc);
        continue;
      }
#endif
      // y = acc s_h s_ws + x for this warp's 32 rows x D / 2 columns, staged 32 columns at a time in the idle h buffers (4 KB per warp,
      // 128-B swizzled) and written by TMA. The residual x of the first 32 columns is loaded before the wait for the accumulator and
      // each next slab's while the current one is staged (the x reload's latency had been ~25 % of the kernel at 10^6 rows)
      const long row = (long)t * 128 + r;
      const bool ok = row < M;
      const uint32_t slab = su + O_H + (uint32_t)ew * 4096u;
      const float* xrow = X + (ok ? row : 0) * D + half * (D / 2);
      uint4 cur[8];
#if RXC
      // (round 3 part 10) the residual slab [32 rows][32 columns] read coalesced (load q: rows 4 q .. 4 q + 3, 8 lanes a 128-byte row
      // segment) and passed through the staging slab, instead of a row per lane (32 L1 wavefronts a load, the same count again as the
      // LayerNorm's x pass); each lane then reads its own row back from shared memory. The same arithmetic: the same bits.
      const uint32_t rq0 = (uint32_t)(lane >> 3), cq = (uint32_t)(lane & 7);
      auto xres = [&](int u, int q) {
        const long rw = (long)t * 128 + q4 * 32 + 4 * q + rq0;
        return (TBEPI != 2 && rw < M) ? ldg128(X + rw * D + half * (D / 2) + 32 * u + 4 * cq) : make_uint4(0, 0, 0, 0);
      };
#pragma unroll
      for (int q = 0; q < 8; ++q) cur[q] = xres(0, q);
#else
#pragma unroll
      for (int q = 0; q < 8; ++q) cur[q] = ok ? ldg128(xrow + 4 * q) : make_uint4(0, 0, 0, 0);
#endif
      mbar_wait(&B.yfull, tc & 1);
      if (ew == 0 && lane == 0 && tc == 0) STAMP(9);
      tc_fence_after();
#pragma unroll
      for (int u = 0; u < (TBEPI == 1 ? 0 : D / 64); ++u) {
        const int col = half * (D / 2) + u * 32;
        uint4 nxt[8];
        if (u + 1 < D / 64) {
#pragma unroll
#if RXC
          for (int q = 0; q < 8; ++q) nxt[q] = xres(u + 1, q);
#else
          for (int q = 0; q < 8; ++q) nxt[q] = ok ? ldg128(xrow + 32 * (u + 1) + 4 * q) : make_uint4(0, 0, 0, 0);
#endif
        }
        if (!YST && u > 0) { if (lane == 0) tma_store_wait_read0(); __syncwarp(); }
#if RXC
#pragma unroll
        for (int q = 0; q < 8; ++q) sts128(slab + sw128(4u * q + rq0, cq), cur[q]);
        __syncwarp();
#endif
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          uint32_t acc[16];
          tmem_ld16(tmem + lrow + T_Y + col + 16 * hh, acc);
          tmem_wait_ld();
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const uint4 w = RXC ? lds128(slab + sw128((uint32_t)lane, 4 * hh + q)) : cur[4 * hh + q];
            sts128(slab + sw128((uint32_t)lane, 4 * hh + q),
                   make_uint4(__float_as_uint(fmaf(__uint_as_float(acc[4 * q]), s_y, __uint_as_float(w.x))),
                              __float_as_uint(fmaf(__uint_as_float(acc[4 * q + 1]), s_y, __uint_as_float(w.y))),
                              __float_as_uint(fmaf(__uint_as_float(acc[4 * q + 2]), s_y, __uint_as_float(w.z))),
                              __float_as_uint(fmaf(__uint_as_float(acc[4 * q + 3]), s_y, __uint_as_float(w.w)))));
          }
        }
#if YST
        // (round 3 part 10) the slab written by the warp's own coalesced stores (y's pointer in XN): no TMA store, no wait for its read
        // before the next slab (that wait had been the per-slab latency of the output pass)
        __syncwarp();
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const long rw = (long)t * 128 + q4 * 32 + 4 * q + rq0;
          const uint4 o = lds128(slab + sw128(4u * q + rq0, cq));
          if (TBEPI != 3 && rw < M) stg128(XN + rw * D + col + 4 * cq, o);
        }
        __syncwarp();
#else
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) { tma_store_2d(&mY, slab, col, t * 128 + q4 * 32); tma_store_commit(); }
#endif
        if (u + 1 < D / 64) {
#pragma unroll
          for (int q = 0; q < 8; ++q) cur[q] = nxt[q];
        }
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.yempty);
      // the h buffers are written again by the next tile's chunks: every warp's staged stores must have been read out
      if (!YST && lane == 0) tma_store_wait_read0();
      named_bar_sync(1, 32 * NEPI);
      if (ew == 0 && lane == 0 && tc < 8) STAMP(84 + tc);
    }
  }
  tc_fence_before();
  // (round 3 part 20) reconverge the role warps first: the producer / MMA warps leave their roles lane by lane, and bar.sync counts a
  // diverged warp's partial arrival as the whole warp (the stamps showed this barrier passed before the epilogue warps had finished,
  // so warp 1 could free the tensor memory under their last loads)
  __syncwarp();
  __syncthreads();
  if (threadIdx.x == 0) STAMP(10);
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}

extern "C" __device__ int transition_smem_bytes = SMEM_BYTES;
