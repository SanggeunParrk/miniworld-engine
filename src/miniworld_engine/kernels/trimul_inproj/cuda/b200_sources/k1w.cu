// k1w.cu -- TriMul front on B200 (tcgen05) for any width D in {64, 128, 256, 384, 512}: input LayerNorm + gated projections +
// token mask -> channel-major bf16 planes. Same math and rounding points as k1.cu (the D128 kernel):
//   xn        = bf16(fma((x - mean) * rstd, g, b))    mean / rstd: fp32 two-pass statistics (k1w_stats), rstd = rsqrt.approx
//   plane[c]  = bf16(sigmoid(g_c) * p_c)              g_c = xn Wg_c^T, p_c = xn Wp_c^T on fp32 accumulators; ONE bf16 rounding
//   Pair (i, j) is masked unless tokens i and j both are kept (the pair mask is formed here from the [L] token mask); a masked
//   pair's xn row is zero, so its plane values are exactly 0.
// planes [P, L*L]: P = 4D (bidirectional: left out | left in | right out | right in) or 2D (one direction: left | right).
//
// Mapping. Tokens on M (128-token tiles, persistent CTAs). Unlike k1.cu the x tile does not stay in shared memory (at D512 it is
// 128 KB): x streams through a ring of 64-channel K-blocks, the transform warps normalise each K-block with precomputed statistics
// and write xn straight into TENSOR MEMORY (A operand, D/2 columns per tile). Weights stream as [128 rows][64 k] K-blocks through
// a second ring. Chunk c = plane channels 64c..64c+63 is one M128 N128 K=D product (TMEM cols [0,64) gate, [64,128) proj);
// the epilogue is k1.cu's (fragment-layout TMEM loads, sigmoid gate, stmatrix.trans into [64 ch][32 tok] boxes, TMA store).
// The weights are read in place (four row-major [P/2, D] matrices W_l, W_lg, W_r, W_rg): a chunk's K-block is its 64 gate rows
// and its 64 projection rows, two [64][64] TMA boxes into the two 8 KB halves of a slot (the same SW128 image as one packed
// [128][64] box), so no per-call weight pack is needed.
//
// 512 threads: warp 0 TMA x, warp 1 MMA, warp 2 TMEM alloc, warp 3 TMA weights; warps 4-7 transform (thread = token row);
// warps 8-11 / 12-15 epilogue for even / odd chunks.
#include "sm100.cuh"
#include "tmap.h"

using namespace sm100;

namespace k1w {

constexpr int TOK = 128, KBB = TOK * 64 * 2;   // one [128][64] bf16 K-block, SW128 = 16 KB
constexpr int OSTW = 64 * 32 * 2;              // per-warp output box: [64 ch][32 tok] bf16, 64-B swizzle = 4 KB
constexpr int NTHREADS = 512;
constexpr float LOG2E = 1.4426950408889634f;
constexpr uint32_t IDESC = idesc_bf16(128, 128);

template <int D>
struct Cfg {
  static constexpr int KB = D / 64;                       // K-blocks per token row
  static constexpr int XC = D / 2;                        // TMEM columns of one xn tile (bf16 pairs)
  static constexpr int NXN = (D <= 256) ? 2 : 1;          // xn tiles resident in TMEM
  static constexpr int NACC = (D <= 128) ? 3 : 2;         // 128-column accumulators
  static constexpr uint32_t T_ACC = 0, T_XN = NACC * 128;
  static_assert(T_XN + NXN * XC <= 512, "tmem");
  static constexpr int NWS = (D <= 128) ? 8 : 6;          // weight K-block ring
  static constexpr int NXS = (D <= 128) ? 4 : 4;          // x K-block ring
  static constexpr int O_W = 0, O_X = O_W + NWS * KBB, O_OUT = O_X + NXS * KBB, O_GB = O_OUT + 8 * OSTW;
  static constexpr int O_BAR = O_GB + 2 * D * 4;
  static constexpr int SMEM = O_BAR + 512 + 1024;
  static_assert(SMEM <= 232448, "smem");
};

struct WMaps { CUtensorMap m[4]; };   // W_l, W_lg, W_r, W_rg

struct Bars {
  uint64_t w_full[8], w_empty[8], x_full[4], x_empty[4], xn_full[2], xn_empty[2], acc_full[3], acc_empty[3];
  uint32_t tmem;
};

DEV float tanh_approx(float x) { float y; asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
DEV float rsqrt_ftz(float x) { float y; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
// k1.cu K1_SIG == 2: 0.5 + 0.5 tanh.approx(g / 2)
DEV float2 sigmoid2(float2 g) {
  const float2 h = __fmul2_rn(g, make_float2(0.5f, 0.5f));
  return __ffma2_rn(make_float2(tanh_approx(h.x), tanh_approx(h.y)), make_float2(0.5f, 0.5f), make_float2(0.5f, 0.5f));
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

template <int D, int P>
__global__ void __launch_bounds__(NTHREADS, 1)
    k1w_kernel(const __grid_constant__ CUtensorMap mx, const __grid_constant__ WMaps mw, const __grid_constant__ CUtensorMap mplane,
               const uint8_t* __restrict__ tokmask, int L, float* __restrict__ mean, float* __restrict__ rstd,
               const float* __restrict__ gamma, const float* __restrict__ beta, int tiles, float eps, int stats_out) {
  using G_ = Cfg<D>;
  constexpr int KB = G_::KB, XC = G_::XC, NXN = G_::NXN, NACC = G_::NACC, NWS = G_::NWS, NXS = G_::NXS;
  constexpr int NCH = P / 64;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + G_::O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x, G = gridDim.x;
  const int n_local = tiles > cta ? (tiles - cta + G - 1) / G : 0;

  if (tid == 0) {
    for (int s = 0; s < NWS; ++s) { mbar_init(&B.w_full[s], 1); mbar_init(&B.w_empty[s], 1); }
    for (int s = 0; s < NXS; ++s) { mbar_init(&B.x_full[s], 1); mbar_init(&B.x_empty[s], 4); }
    for (int b = 0; b < NACC; ++b) { mbar_init(&B.acc_full[b], 1); mbar_init(&B.acc_empty[b], 4); }
    for (int b = 0; b < NXN; ++b) { mbar_init(&B.xn_full[b], 1); mbar_init(&B.xn_empty[b], 1); }
    fence_mbar_init();
    prefetch_tmap(&mx); prefetch_tmap(&mplane);
    for (int k = 0; k < 4; ++k) prefetch_tmap(&mw.m[k]);
  }
  for (int k = tid; k < D; k += NTHREADS) {
    reinterpret_cast<float*>(sm + G_::O_GB)[k] = gamma[k];
    reinterpret_cast<float*>(sm + G_::O_GB)[D + k] = beta[k];
  }
  if (warp == 2) { tmem_alloc(&B.tmem, 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp < 4) {
    if (warp == 0 && lane == 0) {
      // ---------------------------------------------------------------- x K-blocks -> ring (slot free once the transform read it)
      int s = 0; uint32_t ph = 0;
      for (int i = 0; i < n_local; ++i) {
        const int row = (cta + i * G) * TOK;
        for (int kb = 0; kb < KB; ++kb) {
          mbar_wait(&B.x_empty[s], ph ^ 1);
          mbar_expect_tx(&B.x_full[s], KBB);
          tma_load_2d(sm + G_::O_X + s * KBB, &mx, &B.x_full[s], kb * 64, row, EVICT_FIRST);
          if (++s == NXS) { s = 0; ph ^= 1; }
        }
      }
    } else if (warp == 3 && lane == 0) {
      // ---------------------------------------------------------------- weight K-blocks: tile, chunk, K-block -> ring
      int s = 0; uint32_t ph = 0;
      for (int i = 0; i < n_local; ++i)
        for (int c = 0; c < NCH; ++c)
          for (int kb = 0; kb < KB; ++kb) {
            mbar_wait(&B.w_empty[s], ph ^ 1);
            mbar_expect_tx(&B.w_full[s], KBB);
            const int row = c * 64, side = row < P / 2 ? 0 : 1, r = side ? row - P / 2 : row;
            tma_load_2d(sm + G_::O_W + s * KBB, &mw.m[2 * side + 1], &B.w_full[s], kb * 64, r, EVICT_LAST);            // gate rows
            tma_load_2d(sm + G_::O_W + s * KBB + KBB / 2, &mw.m[2 * side], &B.w_full[s], kb * 64, r, EVICT_LAST);      // projection
            if (++s == NWS) { s = 0; ph ^= 1; }
          }
    } else if (warp == 1) {
      // ---------------------------------------------------------------- MMA: acc[tok][0:64 gate | 64:128 proj] = xn (TMEM) . W_c^T
      const bool ldr = elect_one();
      const uint64_t w0 = desc_k_sw128(su + G_::O_W);
      constexpr uint64_t WSD = KBB >> 4;
      int s = 0, a = 0;
      uint32_t ph = 0, pa = 0;
      for (int i = 0; i < n_local; ++i) {
        const int b = (NXN == 1) ? 0 : (i & 1);
        mbar_wait(&B.xn_full[b], (i / NXN) & 1);
        tc_fence_after();
        const uint32_t xa = tmem + G_::T_XN + b * XC;
#pragma unroll 1
        for (int c = 0; c < NCH; ++c) {
          mbar_wait(&B.acc_empty[a], pa ^ 1);
          tc_fence_after();
          const uint32_t d = tmem + G_::T_ACC + a * 128;
#pragma unroll 1
          for (int kb = 0; kb < KB; ++kb) {
            mbar_wait(&B.w_full[s], ph);
            tc_fence_after();
            if (ldr) {
              const uint64_t wd = w0 + (uint64_t)s * WSD;
              const uint32_t xk = xa + kb * 32;
              umma_ts(d, xk, wd, IDESC, kb > 0);
              umma_ts(d, xk + 8, wd + 2, IDESC, 1);
              umma_ts(d, xk + 16, wd + 4, IDESC, 1);
              umma_ts(d, xk + 24, wd + 6, IDESC, 1);
              umma_commit(&B.w_empty[s]);
            }
            __syncwarp();
            if (++s == NWS) { s = 0; ph ^= 1; }
          }
          if (ldr) umma_commit(&B.acc_full[a]);
          __syncwarp();
          if (++a == NACC) { a = 0; pa ^= 1; }
        }
        if (ldr) umma_commit(&B.xn_empty[b]);
        __syncwarp();
      }
    }
  } else if (warp < 8) {
    // ------------------------------------------------------------------ transform: thread = token row r, K-block by K-block
    const int r = (warp & 3) * 32 + lane;
    const uint32_t trow = tmem + ((uint32_t)((warp & 3) * 32) << 16);
    auto up = [](uint32_t w) { return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u)); };
    int s = 0; uint32_t ph = 0;
    for (int i = 0; i < n_local; ++i) {
      const int row = (cta + i * G) * TOK + r, b = (NXN == 1) ? 0 : (i & 1);
      // tokens are b-major: row = (sample * L + i) * L + j, the token mask is [B, L]
      const int ll = L * L, sb = row / ll, rin = row - sb * ll;
      const uint32_t kmask = (tokmask == nullptr || (tokmask[sb * L + rin / L] && tokmask[sb * L + rin % L])) ? 0xffffffffu : 0u;
      float mu, rs;
      if constexpr (D <= 128) {
        // whole row resident (KB <= 2 ring slots): two-pass statistics here, as k1.cu (packed f32x2 sums)
        int ss = s; uint32_t pp = ph;
        for (int kb = 0; kb < KB; ++kb) { mbar_wait(&B.x_full[ss], pp); if (++ss == NXS) { ss = 0; pp ^= 1; } }
        auto ldx = [&](int q) { const int kb = q >> 3; int sl = s + kb; if (sl >= NXS) sl -= NXS;
                                return lds128(su + G_::O_X + sl * KBB + sw128(r, q & 7)); };
        float2 a0 = make_float2(0.f, 0.f), a1 = a0, a2 = a0, a3 = a0;
#pragma unroll
        for (int q = 0; q < D / 8; ++q) {
          const uint4 t = ldx(q);
          a0 = __fadd2_rn(a0, up(t.x)); a1 = __fadd2_rn(a1, up(t.y)); a2 = __fadd2_rn(a2, up(t.z)); a3 = __fadd2_rn(a3, up(t.w));
        }
        a0 = __fadd2_rn(__fadd2_rn(a0, a1), __fadd2_rn(a2, a3));
        mu = __fmul_rn(__fadd_rn(a0.x, a0.y), 1.f / D);
        const float2 nm0 = make_float2(-mu, -mu);
        a0 = a1 = a2 = a3 = make_float2(0.f, 0.f);
#pragma unroll
        for (int q = 0; q < D / 8; ++q) {
          const uint4 t = ldx(q);
          float2 d;
          d = __fadd2_rn(up(t.x), nm0); a0 = __ffma2_rn(d, d, a0);
          d = __fadd2_rn(up(t.y), nm0); a1 = __ffma2_rn(d, d, a1);
          d = __fadd2_rn(up(t.z), nm0); a2 = __ffma2_rn(d, d, a2);
          d = __fadd2_rn(up(t.w), nm0); a3 = __ffma2_rn(d, d, a3);
        }
        a0 = __fadd2_rn(__fadd2_rn(a0, a1), __fadd2_rn(a2, a3));
        rs = rsqrt_ftz(__fadd_rn(__fmul_rn(__fadd_rn(a0.x, a0.y), 1.f / D), eps));
        if (stats_out) { mean[row] = mu; rstd[row] = rs; }
      } else {
        mu = mean[row]; rs = rstd[row];
      }
      const float2 nm = make_float2(-mu, -mu), rr = make_float2(rs, rs);
      if (i >= NXN) mbar_wait(&B.xn_empty[b], ((i / NXN) - 1) & 1);
      tc_fence_after();
      for (int kb = 0; kb < KB; ++kb) {
        if constexpr (D > 128) mbar_wait(&B.x_full[s], ph);
        const uint32_t xr = su + G_::O_X + s * KBB;
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {          // 16 channels (8 packed words) per TMEM store
          uint32_t o[8];
#pragma unroll
          for (int qq = 0; qq < 2; ++qq) {
            const int q = q4 * 2 + qq;
            const uint4 t = lds128(xr + sw128(r, q));
            const int col = kb * 64 + q * 8;
            const float4 ga = lds128f(su + G_::O_GB + col * 4);
            const float4 gb4 = lds128f(su + G_::O_GB + col * 4 + 16);
            const float4 ba = lds128f(su + G_::O_GB + D * 4 + col * 4);
            const float4 bb = lds128f(su + G_::O_GB + D * 4 + col * 4 + 16);
            auto aff2 = [&](uint32_t w, float g0, float g1, float b0, float b1) {
              const float2 y = __ffma2_rn(__fmul2_rn(__fadd2_rn(up(w), nm), rr), make_float2(g0, g1), make_float2(b0, b1));
              return pack_bf16(y.x, y.y) & kmask;
            };
            o[4 * qq + 0] = aff2(t.x, ga.x, ga.y, ba.x, ba.y);
            o[4 * qq + 1] = aff2(t.y, ga.z, ga.w, ba.z, ba.w);
            o[4 * qq + 2] = aff2(t.z, gb4.x, gb4.y, bb.x, bb.y);
            o[4 * qq + 3] = aff2(t.w, gb4.z, gb4.w, bb.z, bb.w);
          }
          tmem_st8(trow + G_::T_XN + b * XC + kb * 32 + q4 * 8, o);
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.x_empty[s]);
        if (++s == NXS) { s = 0; ph ^= 1; }
      }
      tmem_wait_st();
      tc_fence_before();
      named_bar_sync(1, 128);
      if (warp == 4 && lane == 0) mbar_arrive(&B.xn_full[b]);
    }
  } else {
    // ------------------------------------------------------------------ gate epilogue (k1.cu): warp quarter q = tokens 32q..32q+31
    const int grp = (warp - 8) >> 2, q = warp & 3;
    const uint32_t stg = su + G_::O_OUT + (warp - 8) * OSTW;
    for (int i = 0; i < n_local; ++i) {
      const int row0 = (cta + i * G) * TOK;
      for (int c = grp; c < NCH; c += 2) {
        const int ci = i * NCH + c, acc = ci % NACC;
        mbar_wait(&B.acc_full[acc], (ci / NACC) & 1);
        tc_fence_after();
        if (lane == 0) bulk_wait_read<0>();
        __syncwarp();
        const uint32_t tq = tmem + G_::T_ACC + acc * 128 + ((uint32_t)(q * 32) << 16);
        uint32_t ga[16], pa[16], gb[16], pb[16];
        auto ldg = [&](int grp4, uint32_t (&g)[16], uint32_t (&p)[16]) {
          const uint32_t ta = tq + ((uint32_t)((grp4 >> 1) * 16) << 16) + (grp4 & 1) * 32;
          tmem_ld16x256_x4(ta, g);
          tmem_ld16x256_x4(ta + 64, p);
        };
        auto work = [&](int grp4, const uint32_t (&g)[16], const uint32_t (&p)[16]) {
          const int hh = grp4 >> 1, cq = grp4 & 1;
          uint32_t m[8];
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
            stmatrix_x4_trans(stg + chl * 64 + ((qc ^ ((chl >> 1) & 3)) << 4), m[4 * x], m[4 * x + 1], m[4 * x + 2], m[4 * x + 3]);
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
        work(3, gb, pb);
        fence_async_smem();
        __syncwarp();
        if (lane == 0) {
          tma_store_2d(&mplane, sm + G_::O_OUT + (warp - 8) * OSTW, row0 + q * 32, c * 64);
          bulk_commit();
        }
      }
    }
    if (lane == 0) bulk_wait<0>();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) tmem_dealloc(tmem, 512);
}

// ---------------------------------------------------------------------- per-token LayerNorm statistics (warp per row, two-pass)
template <int D>
__global__ void k1w_stats_kernel(const __nv_bfloat16* __restrict__ x, float* __restrict__ mean, float* __restrict__ rstd,
                                 int M, float eps) {
  constexpr int NV = (D / 8 + 31) / 32;            // uint4 per lane
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
  if (row >= M) return;
  const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)row * D);
  uint4 v[NV];
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < NV; ++k) {
    const int q = lane + 32 * k;
    v[k] = q < D / 8 ? xr[q] : make_uint4(0, 0, 0, 0);
    const uint32_t w[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
#pragma unroll
    for (int j = 0; j < 4; ++j) s += bf16lo(w[j]) + bf16hi(w[j]);
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  const float mu = s * (1.f / D);
  float ss = 0.f;
#pragma unroll
  for (int k = 0; k < NV; ++k) {
    if (lane + 32 * k >= D / 8) continue;
    const uint32_t w[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float a = bf16lo(w[j]) - mu, b = bf16hi(w[j]) - mu;
      ss = fmaf(a, a, fmaf(b, b, ss));
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if (lane == 0) {
    mean[row] = mu;
    rstd[row] = rsqrt_ftz(ss * (1.f / D) + eps);
  }
}

int num_sms() {
  static int n = 0;
  if (!n) { int d; cudaGetDevice(&d); cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d); }
  return n;
}

template <int D, int P>
void launch(const CUtensorMap& mx, const WMaps& mw, const CUtensorMap& mp, const uint8_t* mask, int L, float* mean, float* rs,
            const float* g, const float* b, int tiles, int grid, cudaStream_t st, float eps, int stats_out) {
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(k1w_kernel<D, P>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<D>::SMEM); attr = true; }
  k1w_kernel<D, P><<<grid, NTHREADS, Cfg<D>::SMEM, st>>>(mx, mw, mp, mask, L, mean, rs, g, b, tiles, eps, stats_out);
}

template <int D>
void launch_stats(const torch::Tensor& x, torch::Tensor& mean, torch::Tensor& rstd, float eps, cudaStream_t st) {
  const int M = (int)x.size(0);
  k1w_stats_kernel<D><<<(M + 7) / 8, 256, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), mean.data_ptr<float>(),
                                                   rstd.data_ptr<float>(), M, eps);
}

}  // namespace k1w

// x [L*L, D] bf16 -> mean, rstd [L*L] fp32
void k1w_stats(torch::Tensor x, torch::Tensor mean, torch::Tensor rstd, double eps) {
  using namespace k1w;
  TORCH_CHECK(x.is_contiguous() && x.scalar_type() == torch::kBFloat16);
  auto st = at::cuda::getCurrentCUDAStream();
  switch (x.size(1)) {
    case 64: launch_stats<64>(x, mean, rstd, (float)eps, st); break;
    case 128: launch_stats<128>(x, mean, rstd, (float)eps, st); break;
    case 256: launch_stats<256>(x, mean, rstd, (float)eps, st); break;
    case 384: launch_stats<384>(x, mean, rstd, (float)eps, st); break;
    case 512: launch_stats<512>(x, mean, rstd, (float)eps, st); break;
    default: TORCH_CHECK(false, "k1w_stats: unsupported width ", x.size(1));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// x [L*L, D] bf16; wl, wlg, wr, wrg [P/2, D] bf16 with unit column stride (planes 0..P/2 are left, P/2..P right);
// tokmask [L] bool / uint8 (None: every token kept), mean / rstd [L*L] fp32, gamma / beta [D] fp32, planes [P, L, L] bf16.
// D <= 128: statistics are computed in the kernel and written to mean / rstd when given; D > 128: read from mean / rstd (k1w_stats).
void k1w_forward(torch::Tensor x, torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg,
                 c10::optional<torch::Tensor> tokmask, c10::optional<torch::Tensor> mean,
                 c10::optional<torch::Tensor> rstd, torch::Tensor gamma, torch::Tensor beta, torch::Tensor planes, int64_t grid,
                 double eps) {
  using namespace k1w;
  const int64_t M = x.size(0), D = x.size(1), P = planes.size(0);
  TORCH_CHECK(M % TOK == 0 && planes.numel() == P * M && x.is_contiguous() && planes.is_contiguous() && (planes.dim() == 3 || planes.dim() == 4));
  // planes [P, L, L] (one sample) or [P, B, L, L]; the tokens of x are b-major, M = B L L
  const int L = (int)planes.size(planes.dim() - 1);
  const int64_t nb = M / ((int64_t)L * L);
  TORCH_CHECK(nb * L * L == M && planes.size(planes.dim() - 2) == L);
  TORCH_CHECK(!tokmask.has_value() || (tokmask->numel() == nb * L && tokmask->element_size() == 1 && tokmask->is_contiguous()));
  const auto bf = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  auto mx = tmap::make(x.data_ptr(), bf, {(uint64_t)D, (uint64_t)M}, {(uint64_t)D * 2}, {64, TOK}, CU_TENSOR_MAP_SWIZZLE_128B);
  WMaps mw;
  const torch::Tensor* ws[4] = {&wl, &wlg, &wr, &wrg};
  for (int k = 0; k < 4; ++k) {
    const torch::Tensor& w = *ws[k];
    TORCH_CHECK(w.size(0) == P / 2 && w.size(1) == D && w.stride(1) == 1 && (w.stride(0) * 2) % 16 == 0,
                "k1w_forward: front weights must be [P/2, D] with unit column stride");
    mw.m[k] = tmap::make(w.data_ptr(), bf, {(uint64_t)D, (uint64_t)(P / 2)}, {(uint64_t)w.stride(0) * 2}, {64, 64},
                         CU_TENSOR_MAP_SWIZZLE_128B);
  }
  auto mp = tmap::make(planes.data_ptr(), bf, {(uint64_t)M, (uint64_t)P}, {(uint64_t)M * 2}, {32u, 64}, CU_TENSOR_MAP_SWIZZLE_64B);
  const int tiles = (int)(M / TOK);
  int g = grid > 0 ? (int)grid : num_sms();
  g = std::min(g, tiles);
  auto st = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(D <= 128 || (mean.has_value() && rstd.has_value()), "k1w_forward: D > 128 needs precomputed statistics");
  const uint8_t* mk = tokmask.has_value() ? reinterpret_cast<const uint8_t*>(tokmask->data_ptr()) : nullptr;
  const float *ga = gamma.data_ptr<float>(), *be = beta.data_ptr<float>();
  float* mu = mean.has_value() ? mean->data_ptr<float>() : nullptr;
  float* rs = rstd.has_value() ? rstd->data_ptr<float>() : nullptr;
  const int so = (D <= 128 && mu != nullptr) ? 1 : 0;
#define K1W_CASE(DD, PP) \
  if (D == DD && P == PP) { launch<DD, PP>(mx, mw, mp, mk, L, mu, rs, ga, be, tiles, g, st, (float)eps, so); C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
  K1W_CASE(64, 256) K1W_CASE(64, 128) K1W_CASE(128, 512) K1W_CASE(128, 256) K1W_CASE(256, 1024) K1W_CASE(256, 512)
  K1W_CASE(384, 1536) K1W_CASE(384, 768) K1W_CASE(512, 2048) K1W_CASE(512, 1024)
#undef K1W_CASE
  TORCH_CHECK(false, "k1w_forward: unsupported (D, P) = (", D, ", ", P, ")");
}

// ---------------------------------------------------------------------------------------------------------------- prep
// Per-call operands the kernels cannot read in place, in one launch: row-major copies of the four front matrices when they are
// stored column-major (the D128 bidirectional module) -- 32 x 32 tiles through shared memory, read along the contiguous
// dimension -- and k3g's output projection with its columns in the TMEM K order (wpp[n][16b + 4q + 2r + u] = wp[n][16b + 8r + 2q + u]).
namespace k1w {
struct W4 { const __nv_bfloat16* p[4]; long long s0[4], s1[4]; };
__global__ void __launch_bounds__(256) prep_kernel(W4 w, int R, int D, int copy_tiles, __nv_bfloat16* __restrict__ wc,
                                                   const __nv_bfloat16* __restrict__ wp, long long p0, long long p1, int H,
                                                   __nv_bfloat16* __restrict__ wpp) {
  __shared__ __nv_bfloat16 tile[32][33];
  const int tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
  if ((int)blockIdx.x < copy_tiles) {
    const int per = (R / 32) * (D / 32), k = blockIdx.x / per, t = blockIdx.x % per;
    const int r0 = (t / (D / 32)) * 32, c0 = (t % (D / 32)) * 32;
    const __nv_bfloat16* src = w.p[k];
    const long long s0 = w.s0[k], s1 = w.s1[k];
    if (s1 == 1) {
      for (int y = ty; y < 32; y += 8) tile[y][tx] = src[(r0 + y) * s0 + c0 + tx];
    } else {
      for (int y = ty; y < 32; y += 8) tile[tx][y] = src[(r0 + tx) * s0 + (c0 + y) * s1];
    }
    __syncthreads();
    for (int y = ty; y < 32; y += 8) wc[((size_t)k * R + r0 + y) * D + c0 + tx] = tile[y][tx];
  } else {
    const int n = blockIdx.x - copy_tiles;
    for (int col = threadIdx.x; col < H; col += blockDim.x) {
      const int b = col >> 4, q = (col >> 2) & 3, rr = (col >> 1) & 1, u = col & 1;
      wpp[(size_t)n * H + col] = wp[n * p0 + (16 * b + 8 * rr + 2 * q + u) * p1];
    }
  }
}
}  // namespace k1w

// wcopy [4, P/2, D] (optional): row-major copies of wl, wlg, wr, wrg; wp [D, H] -> wpp [D, H] (optional, k3g).
void k1w_prep(torch::Tensor wl, torch::Tensor wlg, torch::Tensor wr, torch::Tensor wrg, c10::optional<torch::Tensor> wcopy,
              c10::optional<torch::Tensor> wp, c10::optional<torch::Tensor> wpp) {
  using namespace k1w;
  const int R = (int)wl.size(0), D = (int)wl.size(1);
  W4 w;
  const torch::Tensor* ts[4] = {&wl, &wlg, &wr, &wrg};
  for (int k = 0; k < 4; ++k) {
    w.p[k] = reinterpret_cast<const __nv_bfloat16*>(ts[k]->data_ptr());
    w.s0[k] = ts[k]->stride(0); w.s1[k] = ts[k]->stride(1);
  }
  int copy_tiles = 0, H = 0, blocks = 0;
  __nv_bfloat16* wc = nullptr;
  if (wcopy.has_value()) {
    TORCH_CHECK(wcopy->is_contiguous() && wcopy->numel() == 4LL * R * D && R % 32 == 0 && D % 32 == 0);
    wc = reinterpret_cast<__nv_bfloat16*>(wcopy->data_ptr());
    copy_tiles = 4 * (R / 32) * (D / 32);
  }
  const __nv_bfloat16* wpi = nullptr;
  __nv_bfloat16* wpo = nullptr;
  long long p0 = 0, p1 = 0;
  if (wp.has_value()) {
    TORCH_CHECK(wpp.has_value() && wpp->is_contiguous() && wp->size(0) == D);
    H = (int)wp->size(1);
    wpi = reinterpret_cast<const __nv_bfloat16*>(wp->data_ptr());
    wpo = reinterpret_cast<__nv_bfloat16*>(wpp->data_ptr());
    p0 = wp->stride(0); p1 = wp->stride(1);
    blocks += D;
  }
  blocks += copy_tiles;
  if (blocks == 0) return;
  prep_kernel<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(w, R, D, copy_tiles, wc, wpi, p0, p1, H, wpo);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
