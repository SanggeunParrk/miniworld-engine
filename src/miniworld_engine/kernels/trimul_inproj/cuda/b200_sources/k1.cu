// k1.cu -- TriMul K1 on B200 (tcgen05): input LayerNorm + four gated projections + mask -> channel-major bf16 planes.
//
// Fusion = the H100 K1 (Anthropic tmn k1_body, run unmodified by MiniWorld):
//   xn        = bf16(LN(x))                        fp32 two-pass statistics, rstd = rsqrt.approx.ftz, y = fma((x-mean)*rstd, g, b)
//   plane[c]  = bf16(sigmoid(g_c) * p_c * mask)     g_c = xn Wg_c^T, p_c = xn Wp_c^T on fp32 accumulators; ONE bf16 rounding
//   planes [512, L, L] (= [512, L*L]): [0:256) left (0:128 outgoing, 128:256 incoming), [256:512) right. Nothing is saved.
//   The pair mask is binary: a masked token's xn row is written as zeros, so g = p = 0 and the plane value is exactly 0.
//
// v3 mapping (shared-memory bandwidth is the binding resource on B200, so the A operand never lives in smem):
//   * tokens on M (128-token tile), xn is written by the LayerNorm warps straight into TENSOR MEMORY (tcgen05.st) and used as the
//     A operand of A-from-TMEM products; only the weights (B) are read from smem by the tensor core.
//   * chunk c (0..7) = plane channels 64c..64c+63: one M128 N128 K128 product, TMEM cols [0,64) gate, [64,128) proj.
//     Packed w1 [1024, 128]: chunk c rows [128c, 128c+64) gate rows, [128c+64, 128c+128) proj rows of those channels.
//   * epilogue reads the accumulator in the mma-fragment layout (tcgen05.ld.16x256b), forms the gate and transposes to
//     [channel][token] rows with stmatrix.trans into a 64-B-swizzled box, one TMA store per warp and chunk.
//
// 640 threads: warp 0 TMA x, warp 1 MMA, warp 2 TMEM alloc, warp 3 TMA weights; warps 4-11 LayerNorm (warp quarter q = lanes
// 32q.., half h = warp group); warps 12-15 / 16-19 epilogue for even / odd chunks (accumulator 0 / 1).
#include "sm100.cuh"
#include "tmap.h"

using namespace sm100;

namespace k1 {
#ifdef PROF
#define TK(ev) { if (blockIdx.x == 0 && (threadIdx.x & 31) == 0 && i == 3) g_prof[152 + ((ev) >> 6)][((ev) >> 3) & 7][(ev) & 7] = clock64(); }
#else
#define TK(ev)
#endif

constexpr int C = 128, NCHUNK = 8, TOK = 128;
constexpr int WSLOT = 128 * 128 * 2;         // one weight chunk: 2 K-blocks of [128 rows][64 k] SW128 = 32 KB
constexpr int NWS = 3;                       // weight ring depth (chunks)
constexpr int XT = TOK * C * 2;              // x tile: [128 tok][128 ch] as 2 K-blocks of 16 KB (SW128)
constexpr int OSTW = 64 * 32 * 2;            // per-warp output box: [64 ch][32 tok] bf16, 64-B swizzle = 4 KB
constexpr int O_W = 0, O_X = O_W + NWS * WSLOT, O_OUT = O_X + 2 * XT, O_RED = O_OUT + 8 * OSTW, O_GB = O_RED + 2048;
constexpr int O_BAR = O_GB + 1024;
constexpr int SMEM = O_BAR + 256 + 1024;
static_assert(SMEM <= 232448, "smem");
constexpr int NTHREADS = 512;
constexpr uint32_t T_ACC = 0, T_XN = 384;    // TMEM: three accumulators [0,384); xn ring of 2 tiles at [384,512)
constexpr int NXN = 2, NACC = 3;
static_assert(NACC == NWS, "weight slot s is released by the acc_full commit of the chunk that used it (same index)");
constexpr uint32_t IDESC = idesc_bf16(128, 128);
constexpr float LOG2E = 1.4426950408889634f;

struct Bars {
  uint64_t w_full[NWS], w_empty[NWS], x_full[2], x_empty[2], xn_full[NXN], xn_empty[NXN], acc_full[NACC], acc_empty[NACC];
  uint32_t tmem;
};

DEV float ex2_ftz(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float rcp_ftz(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float rsqrt_ftz(float x) { float y; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float tanh_approx(float x) { float y; asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
#ifndef K1_PAIRST
#define K1_PAIRST 0
#endif
#ifndef K1_WARPMMA
#define K1_WARPMMA 1
#endif
#ifndef K1_SIG
#define K1_SIG 1
#endif
// sigmoid of two fp32 accumulators. 0: the kit rcp.approx(1 + ex2.approx(-g log2e)) (two MUFU ops); 1: kit ex2 + Newton
// reciprocal on the FMA pipe (one MUFU op, rel. error < 1e-7); 2: 0.5 + 0.5 tanh.approx(g / 2) (one MUFU op).
DEV float2 sigmoid2(float2 g) {
#if K1_SIG == 0
  return make_float2(rcp_ftz(__fadd_rn(1.f, ex2_ftz(__fmul_rn(-LOG2E, g.x)))), rcp_ftz(__fadd_rn(1.f, ex2_ftz(__fmul_rn(-LOG2E, g.y)))));
#elif K1_SIG == 1
  const float2 t = __fmul2_rn(g, make_float2(-LOG2E, -LOG2E));
  const float2 d = __fadd2_rn(make_float2(ex2_ftz(fminf(t.x, 126.f)), ex2_ftz(fminf(t.y, 126.f))), make_float2(1.f, 1.f));
  float2 r = make_float2(__int_as_float(0x7EF311C3 - __float_as_int(d.x)), __int_as_float(0x7EF311C3 - __float_as_int(d.y)));
  const float2 two = make_float2(2.f, 2.f), nd = make_float2(-d.x, -d.y);
#pragma unroll
  for (int it = 0; it < 3; ++it) r = __fmul2_rn(r, __ffma2_rn(nd, r, two));
  return r;
#elif K1_SIG == 2
  const float2 h = __fmul2_rn(g, make_float2(0.5f, 0.5f));
  return __ffma2_rn(make_float2(tanh_approx(h.x), tanh_approx(h.y)), make_float2(0.5f, 0.5f), make_float2(0.5f, 0.5f));
#else
  // 3: one packed tanh.approx.f16x2 per pair (half the MUFU issue of 2; tanh.approx.f32 is itself only ~2^-11 accurate),
  // the 0.5 t + 0.5 affine stays fp32
  const float2 h = __fmul2_rn(g, make_float2(0.5f, 0.5f));
  uint32_t hx, tx;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(hx) : "f"(h.y), "f"(h.x));
  asm("tanh.approx.f16x2 %0, %1;" : "=r"(tx) : "r"(hx));
  float t0, t1;
  asm("{ .reg .f16 lo, hi; mov.b32 {lo, hi}, %2; cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }" : "=f"(t0), "=f"(t1) : "r"(tx));
  return __ffma2_rn(make_float2(t0, t1), make_float2(0.5f, 0.5f), make_float2(0.5f, 0.5f));
#endif
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
DEV void tmem_st8(uint32_t taddr, const uint32_t (&r)[8]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
               :: "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]) : "memory");
}
DEV void tmem_st32(uint32_t taddr, const uint32_t (&r)[32]) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
               "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};"
               :: "r"(taddr), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]),
                  "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(r[16]), "r"(r[17]),
                  "r"(r[18]), "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]),
                  "r"(r[27]), "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31]) : "memory");
}
// 16 TMEM lanes x (4 x 8) columns in the mma-fragment layout: thread t gets, for column group j, lane t/4 cols 8j+2(t%4)+{0,1}
// in r[4j], r[4j+1] and lane t/4+8 in r[4j+2], r[4j+3]
DEV void tmem_ld16x256_x4(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
                 "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
               : "r"(taddr) : "memory");
}
DEV void stmatrix_x4_trans(uint32_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(a), "r"(b), "r"(c), "r"(d)
               : "memory");
}

__global__ void __launch_bounds__(NTHREADS, 1)
    k1_kernel(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mw, const __grid_constant__ CUtensorMap mplane,
              const float* __restrict__ mask, const float* __restrict__ gamma, const float* __restrict__ beta, int tiles, float eps) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, G = gridDim.x;
  const int n_local = tiles > cta ? (tiles - cta + G - 1) / G : 0;

  if (tid == 0) {
    for (int s = 0; s < NWS; ++s) { mbar_init(&B.w_full[s], 1); mbar_init(&B.w_empty[s], 1); }
    for (int b = 0; b < 2; ++b) {
      mbar_init(&B.x_full[b], 1); mbar_init(&B.x_empty[b], 4);
    }
    for (int b = 0; b < NACC; ++b) { mbar_init(&B.acc_full[b], 1); mbar_init(&B.acc_empty[b], 4); }
    for (int b = 0; b < NXN; ++b) { mbar_init(&B.xn_full[b], 1); mbar_init(&B.xn_empty[b], 1); }
    fence_mbar_init();
    prefetch_tmap(&mx); prefetch_tmap(&mw); prefetch_tmap(&mplane);
  }
  if (tid < 128) {
    reinterpret_cast<float*>(sm + O_GB)[tid] = gamma[tid];
    reinterpret_cast<float*>(sm + O_GB)[128 + tid] = beta[tid];
  }
  if (warp == 2) { tmem_alloc(&B.tmem, 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  PROF_BEGIN

  if (warp < 4) {
    if (warp == 0 && lane == 0) {
      // -------------------------------------------------------------- x producer (buffer b free once LN of tile i-2 read it)
      for (int i = 0; i < n_local; ++i) {
        const int b = i & 1, row = (cta + i * G) * TOK;
        if (i >= 2) PW(0, mbar_wait(&B.x_empty[b], ((i >> 1) - 1) & 1));
        mbar_expect_tx(&B.x_full[b], XT);
        tma_load_2d(sm + O_X + b * XT, &mx, &B.x_full[b], 0, row, EVICT_FIRST);
        tma_load_2d(sm + O_X + b * XT + XT / 2, &mx, &B.x_full[b], 64, row, EVICT_FIRST);
      }
    } else if (warp == 3 && lane == 0) {
      // -------------------------------------------------------------- weight K-blocks: chunk c, K-block kb -> ring
      const int n = n_local * NCHUNK;
      for (int q = 0; q < n; ++q) {
        const int s = q % NWS, u = q / NWS, c = q & (NCHUNK - 1);
        if (q >= NWS) PW(0, mbar_wait(&B.acc_full[s], (u - 1) & 1));   // chunk q-3 (same slot, same accumulator) is done
#ifdef K1_NOW
        if (q >= NWS) { mbar_arrive(&B.w_full[s]); continue; }
#endif
        mbar_expect_tx(&B.w_full[s], WSLOT);
        tma_load_2d(sm + O_W + s * WSLOT, &mw, &B.w_full[s], 0, c * 128, EVICT_LAST);
        tma_load_2d(sm + O_W + s * WSLOT + WSLOT / 2, &mw, &B.w_full[s], 64, c * 128, EVICT_LAST);
      }
#if K1_WARPMMA
    } else if (warp == 1) {
      // whole warp runs the loop so ring index / descriptors stay warp-uniform (uniform registers, no per-MMA waterfall
      // ELECT/R2UR.BROADCAST loop that a lane-0-only branch forces); one elected lane issues the MMAs and commits
      const bool ldr = elect_one();
#else
    } else if (warp == 1 && lane == 0) {
      constexpr bool ldr = true;
#endif
      // -------------------------------------------------------------- MMA: acc[tok][0:64 gate | 64:128 proj] = xn (TMEM) . W_c^T
      // Lean loop (this lane shares its SMSP with three busy warps): ring index / phase counters advance incrementally (NACC == NWS,
      // so the accumulator and the weight slot of a chunk share one index), descriptors are a per-slot base plus constant offsets.
      const uint64_t w0 = desc_k_sw128(su + O_W);
      constexpr uint64_t WSD = WSLOT >> 4, KBD = (WSLOT / 2) >> 4;
      int s = 0;
      uint32_t ph = 0;                                      // phase parity of ring slot s's current use
      for (int i = 0; i < n_local; ++i) {
        const int b = i & (NXN - 1);
        TK(0);
        PW(0, mbar_wait(&B.xn_full[b], (i / NXN) & 1));
        TK(1);
        tc_fence_after();
        const uint32_t xa = tmem + T_XN + b * 64;
#pragma unroll 1
        for (int c = 0; c < NCHUNK; ++c) {
          PW(1, mbar_wait(&B.acc_empty[s], ph ^ 1));
          TK(8 + c);
          PW(2, mbar_wait(&B.w_full[s], ph));
          TK(16 + c);
          tc_fence_after();
          if (ldr) {
            const uint32_t d = tmem + T_ACC + s * 128;
            const uint64_t wd = w0 + (uint64_t)s * WSD;
            umma_ts(d, xa, wd, IDESC, 0);
            umma_ts(d, xa + 8, wd + 2, IDESC, 1);
            umma_ts(d, xa + 16, wd + 4, IDESC, 1);
            umma_ts(d, xa + 24, wd + 6, IDESC, 1);
            umma_ts(d, xa + 32, wd + KBD, IDESC, 1);
            umma_ts(d, xa + 40, wd + KBD + 2, IDESC, 1);
            umma_ts(d, xa + 48, wd + KBD + 4, IDESC, 1);
            umma_ts(d, xa + 56, wd + KBD + 6, IDESC, 1);
            umma_commit(&B.acc_full[s]);
          }
          TK(24 + c);
          if (K1_WARPMMA) __syncwarp();
          if (++s == NACC) { s = 0; ph ^= 1; }
        }
        if (ldr) umma_commit(&B.xn_empty[b]);
        if (K1_WARPMMA) __syncwarp();
      }
    }
  } else if (warp < 8) {
    // ------------------------------------------------------------------ LayerNorm: warps 4-7, every tile;
    // thread = one full row r = 32 (warp % 4) + lane, register-resident two-pass statistics, no cross-thread exchange
    const int lg = 0, r = (warp & 3) * 32 + lane;
    const uint32_t trow = tmem + ((uint32_t)((warp & 3) * 32) << 16);
    float mnext = lg < n_local ? mask[(cta + lg * G) * TOK + r] : 0.f;
    for (int i = 0; i < n_local; ++i) {
      const int b = i & 1, xb = i % NXN;
      const uint32_t kmask = mnext != 0.f ? 0xffffffffu : 0u;   // binary mask folded into xn
      if (i + 1 < n_local) mnext = mask[(cta + (i + 1) * G) * TOK + r];   // prefetch the next tile's mask
      PW(0, mbar_wait(&B.x_full[b], (i >> 1) & 1));
      const uint32_t xr = su + O_X + b * XT;
      auto ldx = [&](int q) { return lds128(xr + (q >> 3) * (XT / 2) + sw128(r, q & 7)); };
#ifdef PROF
      long long tp0 = clock64();
#endif
      // packed f32x2 statistics (per-lane IEEE fp32; only the summation order differs from a scalar loop)
      auto up = [](uint32_t w) { return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u)); };
      float2 a0 = make_float2(0.f, 0.f), a1 = a0, a2 = a0, a3 = a0;
#pragma unroll
      for (int q = 0; q < 16; ++q) {
        const uint4 t = ldx(q);
        a0 = __fadd2_rn(a0, up(t.x)); a1 = __fadd2_rn(a1, up(t.y)); a2 = __fadd2_rn(a2, up(t.z)); a3 = __fadd2_rn(a3, up(t.w));
      }
      a0 = __fadd2_rn(__fadd2_rn(a0, a1), __fadd2_rn(a2, a3));
      const float mean = __fmul_rn(__fadd_rn(a0.x, a0.y), 1.f / C);
#ifdef PROF
      pacc_[2] += clock64() - tp0; tp0 = clock64();
#endif
      const float2 nm = make_float2(-mean, -mean);
      a0 = a1 = a2 = a3 = make_float2(0.f, 0.f);
#pragma unroll
      for (int q = 0; q < 16; ++q) {
        const uint4 t = ldx(q);
        float2 d;
        d = __fadd2_rn(up(t.x), nm); a0 = __ffma2_rn(d, d, a0);
        d = __fadd2_rn(up(t.y), nm); a1 = __ffma2_rn(d, d, a1);
        d = __fadd2_rn(up(t.z), nm); a2 = __ffma2_rn(d, d, a2);
        d = __fadd2_rn(up(t.w), nm); a3 = __ffma2_rn(d, d, a3);
      }
      a0 = __fadd2_rn(__fadd2_rn(a0, a1), __fadd2_rn(a2, a3));
      const float rs = rsqrt_ftz(__fadd_rn(__fmul_rn(__fadd_rn(a0.x, a0.y), 1.f / C), eps));
#ifdef PROF
      pacc_[3] += clock64() - tp0;
#endif
      if (i >= NXN) PW(1, mbar_wait(&B.xn_empty[xb], ((i / NXN) - 1) & 1));   // tile i-3's products have read xn buffer xb
      tc_fence_after();
#ifdef PROF
      tp0 = clock64();
#endif
#pragma unroll
      for (int q4 = 0; q4 < 8; ++q4) {           // 16 columns (8 packed words) per TMEM store
        uint32_t o[8];
#pragma unroll
        for (int qq = 0; qq < 2; ++qq) {
          const int q = q4 * 2 + qq;
          const uint4 t = ldx(q);
          const int col = q * 8;
          const float4 ga = lds128f(su + O_GB + col * 4);
          const float4 gb4 = lds128f(su + O_GB + col * 4 + 16);
          const float4 ba = lds128f(su + O_GB + 512 + col * 4);
          const float4 bb = lds128f(su + O_GB + 512 + col * 4 + 16);
          const float2 rr = make_float2(rs, rs);
          // H100 order: fma((x - mean) * rstd, gamma, beta), each step rounded, two lanes at a time
          auto aff2 = [&](uint32_t w, float g0, float g1, float b0, float b1) {
            const float2 y = __ffma2_rn(__fmul2_rn(__fadd2_rn(up(w), nm), rr), make_float2(g0, g1), make_float2(b0, b1));
            return pack_bf16(y.x, y.y) & kmask;
          };
          o[4 * qq + 0] = aff2(t.x, ga.x, ga.y, ba.x, ba.y);
          o[4 * qq + 1] = aff2(t.y, ga.z, ga.w, ba.z, ba.w);
          o[4 * qq + 2] = aff2(t.z, gb4.x, gb4.y, bb.x, bb.y);
          o[4 * qq + 3] = aff2(t.w, gb4.z, gb4.w, bb.z, bb.w);
        }
#ifndef K1_NOLNST
        tmem_st8(trow + T_XN + xb * 64 + q4 * 8, o);
#endif
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.x_empty[b]);   // x tile fully read
#ifdef PROF
      pacc_[4] += clock64() - tp0; tp0 = clock64();
#endif
      tmem_wait_st();
#ifdef PROF
      pacc_[5] += clock64() - tp0;
#endif
      tc_fence_before();
      named_bar_sync(1 + lg, 128);
      if ((warp & 3) == 0 && lane == 0) mbar_arrive(&B.xn_full[xb]);
    }
  } else {
    // ------------------------------------------------------------------ gate epilogue: warp quarter q = tokens 32q..32q+31
    const int grp = (warp - 8) >> 2, q = warp & 3;
    const uint32_t stg = su + O_OUT + (warp - 8) * OSTW;
    int nst = 0;
    for (int i = 0; i < n_local; ++i) {
      const int row0 = (cta + i * G) * TOK;
      for (int c = grp; c < NCHUNK; c += 2) {
        const int ci = i * NCHUNK + c, acc = ci % NACC;
        PW(0, mbar_wait(&B.acc_full[acc], (ci / NACC) & 1));
        if (q == 0) { TK(32 + c); } else { TK(64 + (q - 1) * 8 + c); }
        tc_fence_after();
#ifdef K1_NOEPI
        tc_fence_before(); __syncwarp(); if (lane == 0) mbar_arrive(&B.acc_empty[acc]); continue;
#endif
#if K1_PAIRST
        // two warps (q, q ^ 1) fill one [64 ch][64 tok] SW128 box (128-byte rows) -> full-line plane writes
        const uint32_t pbuf = su + O_OUT + ((warp - 8) >> 1) * (2 * OSTW);
        if (lane == 0 && (q & 1) == 0) PW(1, bulk_wait_read<0>());
        named_bar_sync(8 + ((warp - 8) >> 1), 64);
#else
        if (lane == 0) PW(1, bulk_wait_read<0>());
        __syncwarp();
#endif
        const uint32_t ob = stg;
        // four load groups (token half hh, 32-channel quarter cq), software-pipelined: the next group's TMEM loads are in
        // flight while the current one is computed; the accumulator is released as soon as the last load has landed
        const uint32_t tq = tmem + T_ACC + acc * 128 + ((uint32_t)(q * 32) << 16);
        uint32_t ga[16], pa[16], gb[16], pb[16];
        auto ldg = [&](int grp4, uint32_t (&g)[16], uint32_t (&p)[16]) {
          const uint32_t ta = tq + ((uint32_t)((grp4 >> 1) * 16) << 16) + (grp4 & 1) * 32;
          tmem_ld16x256_x4(ta, g);
          tmem_ld16x256_x4(ta + 64, p);
        };
        auto work = [&](int grp4, const uint32_t (&g)[16], const uint32_t (&p)[16]) {
          const int hh = grp4 >> 1, cq = grp4 & 1;
          uint32_t m[8];
#ifdef K1_EPI_NOMATH
#pragma unroll
          for (int j = 0; j < 4; ++j) { m[2 * j] = g[4 * j] ^ p[4 * j + 1]; m[2 * j + 1] = g[4 * j + 2] ^ p[4 * j + 3]; }
          if (false)
#endif
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float2 s0 = sigmoid2(make_float2(__uint_as_float(g[4 * j]), __uint_as_float(g[4 * j + 1])));
            const float2 s1 = sigmoid2(make_float2(__uint_as_float(g[4 * j + 2]), __uint_as_float(g[4 * j + 3])));
            const float2 a0 = __fmul2_rn(s0, make_float2(__uint_as_float(p[4 * j]), __uint_as_float(p[4 * j + 1])));
            const float2 a1 = __fmul2_rn(s1, make_float2(__uint_as_float(p[4 * j + 2]), __uint_as_float(p[4 * j + 3])));
            m[2 * j] = pack_bf16(a0.x, a0.y);
            m[2 * j + 1] = pack_bf16(a1.x, a1.y);
          }
#pragma unroll
          for (int x = 0; x < 2; ++x) {
            const int k = lane >> 3, jr = lane & 7;
            const int chl = cq * 32 + (2 * x + (k >> 1)) * 8 + jr;
            const int qc = hh * 2 + (k & 1);
#ifdef K1_EPI_NOSTM
            if (m[4 * x] == 0x7f7f7f7fu && m[4 * x + 3] == 3u)
#endif
#if K1_PAIRST
            stmatrix_x4_trans(pbuf + chl * 128 + (((((q & 1) * 4) + qc) ^ (chl & 7)) << 4), m[4 * x], m[4 * x + 1], m[4 * x + 2], m[4 * x + 3]);
#else
            stmatrix_x4_trans(ob + chl * 64 + ((qc ^ ((chl >> 1) & 3)) << 4), m[4 * x], m[4 * x + 1], m[4 * x + 2], m[4 * x + 3]);
#endif
          }
        };
        ldg(0, ga, pa);
        tmem_wait_ld();
        ldg(1, gb, pb);
        work(0, ga, pa);
        tmem_wait_ld();
        ldg(2, ga, pa);
        work(1, gb, pb);
        tmem_wait_ld();
        ldg(3, gb, pb);
        work(2, ga, pa);
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.acc_empty[acc]);
        if (q == 0) TK(40 + c);
        if (q == 1) TK(48 + c);
        if (q == 2) TK(56 + c);
        if (q == 3) TK(88 + c);
        work(3, gb, pb);
        fence_async_smem();
#if K1_PAIRST
        named_bar_sync(8 + ((warp - 8) >> 1), 64);
#ifndef K1_NOSTORE
        if (lane == 0 && (q & 1) == 0) {
          tma_store_2d(&mplane, sm + O_OUT + ((warp - 8) >> 1) * (2 * OSTW), row0 + q * 32, c * 64);
          bulk_commit();
        }
#endif
#else
        __syncwarp();
#ifndef K1_NOSTORE
        if (lane == 0) {
          tma_store_2d(&mplane, sm + O_OUT + (warp - 8) * OSTW, row0 + q * 32, c * 64);
          bulk_commit();
        }
#endif
#endif
        ++nst;
      }
    }
    if (lane == 0) bulk_wait<0>();
  }
  if (lane == 0 && warp == 0) PROF_END(0);
  if (lane == 0 && warp == 1) PROF_END(1);
  if (lane == 0 && warp == 3) PROF_END(2);
  if (lane == 0 && warp == 4) PROF_END(3);
  if (lane == 0 && warp == 8) PROF_END(4);
  tc_fence_before();
  __syncthreads();
  if (warp == 2) tmem_dealloc(tmem, 512);
}

int num_sms() {
  static int n = 0;
  if (!n) { int d; cudaGetDevice(&d); cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d); }
  return n;
}

}  // namespace k1

std::vector<double> k1_prof() {
#ifndef PROF
  return {};
#else
  std::vector<unsigned long long> h(160 * 64);
  cudaMemcpyFromSymbol(h.data(), g_prof, sizeof(unsigned long long) * 160 * 64);
  int n = k1::num_sms();
  std::vector<double> m(64, 0.0);
  for (int b = 0; b < n; ++b) for (int k = 0; k < 64; ++k) m[k] += h[b * 64 + k] / (double)n;
  for (int k = 0; k < 128; ++k) m.push_back((double)((long long)h[152 * 64 + k] - (long long)h[152 * 64]));
  return m;
#endif
}

// x [L*L, 128] bf16, w1 [1024, 128] bf16 (packed, see header), mask [L*L] fp32 (0/1), gamma/beta [128] fp32, planes [512, L, L] bf16.
void k1_forward(torch::Tensor x, torch::Tensor w1, torch::Tensor mask, torch::Tensor gamma, torch::Tensor beta, torch::Tensor planes,
                double eps, int64_t grid) {
  using namespace k1;
  const int64_t M = x.size(0), L = planes.size(1);
  TORCH_CHECK(x.size(1) == C && M == L * L && M % TOK == 0 && w1.size(0) == 1024 && w1.size(1) == C && planes.size(0) == 512);
  TORCH_CHECK(x.is_contiguous() && w1.is_contiguous() && mask.is_contiguous() && planes.is_contiguous());
  auto mx = tmap::make(x.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, {(uint64_t)C, (uint64_t)M}, {(uint64_t)C * 2}, {64, TOK},
                       CU_TENSOR_MAP_SWIZZLE_128B);
  auto mw = tmap::make(w1.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, {(uint64_t)C, 1024}, {(uint64_t)C * 2}, {64, 128},
                       CU_TENSOR_MAP_SWIZZLE_128B);
  auto mp = tmap::make(planes.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, {(uint64_t)M, 512}, {(uint64_t)M * 2},
                       {K1_PAIRST ? 64u : 32u, 64}, K1_PAIRST ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B);
  const int tiles = (int)(M / TOK);
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(k1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM); attr = true; }
  int g = grid > 0 ? (int)grid : num_sms();
  g = std::min(g, tiles);
  k1_kernel<<<g, NTHREADS, SMEM, at::cuda::getCurrentCUDAStream()>>>(mx, mw, mp, mask.data_ptr<float>(), gamma.data_ptr<float>(),
                                                                     beta.data_ptr<float>(), tiles, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
