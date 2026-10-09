// cond_tables_sm100.cu -- every conditioning table of the bias-only token DiT step in ONE kernel, sm_100a:
//
//   G[:, n] = sig_n( rstd_t (c W_n - mu_t colsum(W_n)) + b_n )   for n < n_g1  (AdaLN scale / shift of both halves on LN(c))
//   G[:, n] = sig_n( c W_n + b_n )                                for n >= n_g1 (the two output gates, on the raw conditioning)
//
// sig_n = sigmoid on the AdaLN scale and gate columns (every table the row passes read through a sigmoid), identity on the shifts:
// the consumers read the tables as they are. c [T, 384] bf16 (T = L, one conditioning shared by the samples, or S L), W [N, 384]
// bf16 (the cond-LayerNorm weight already folded into the first n_g1 rows), b [N] and colsum [N] fp32.
//
// The LayerNorm of the conditioning is folded in algebraically -- LN(c) W^T = rstd (c W^T - mu colsum(W)) -- so the GEMM reads
// the raw rows, and one A tile [128 rows x 384] (96 KB, six SW128 boxes) serves every n-tile of that row tile; the epilogue
// takes the row's mu / rstd from the resident tile. (The unfused path rounded LN(c) to bf16 before its GEMM; this one does not.)
// Items = (128-row tile, NT-column tile), m-major, dealt in contiguous ranges to persistent CTAs, so a CTA reloads A only when
// its range crosses a row tile. Accumulators double-buffered in TMEM (2 x NT columns): an item's epilogue runs under the next
// item's K loop. Each epilogue warp stores its 32 rows per 32-column piece through its own staging tiles.
// Warps: 0 TMA producer, 1 MMA, 2 TMEM allocator, 3 idle, 4-7 epilogue (one row per thread).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef NT
#define NT 256
#endif
constexpr int KC = 384, NKB = KC / 64, QM = 128;
constexpr int TA = QM * 128, STB = NT * 128, NST = 2;
constexpr int NEW = 8;                                       // epilogue warps: two per SMSP, the two warpgroups take alternate pieces
constexpr int SWB = 32 * 64, NSW = 3;                        // per-warp staging: bf16 [32 rows][32 cols] SW64, 2 KB, NSW of them
constexpr int PB = 2 * NT * 4;                               // an item's bias and colsum columns, fp32, double-buffered
constexpr int O_A = 0, O_B = O_A + NKB * TA, O_S = O_B + NST * STB, O_P = O_S + NEW * NSW * SWB, O_R = O_P + 2 * PB,
              O_BAR = O_R + 2 * 2 * QM * 4, SMEM_BYTES = O_BAR + 256;
static_assert(SMEM_BYTES <= 232448, "shared memory");
constexpr uint32_t IDESC = idesc_bf16(QM, NT);

#ifdef TRACE
// CTA TRACE_CTA: 0 B stage consumed by the MMA warp; 1 epilogue: [0] A in, [1] stats done, [2] acc in, [3 + ch] piece ch stored
__device__ long long g_tr[4][64];
#ifndef TRACE_CTA
#define TRACE_CTA 0
#endif
#define TR(ev, i) do { if (blockIdx.x == TRACE_CTA && (i) < 64) g_tr[ev][i] = clock64() - tclk; } while (0)
#else
#define TR(ev, i) do { } while (0)
#endif

struct Bars { uint64_t full[NST], empty[NST], a_full, a_free, acc_full[2], acc_empty[2]; uint32_t tmem; };

// 1 / (1 + 2^(-v log2 e)): the exponential on MUFU, the reciprocal by two Newton steps on the FMA pipe (relative error <= 2e-4,
// below the bf16 rounding of the table)
DEVI float sgm(float v) {
  const float d = fmin_nan(__fadd_rn(1.f, ex2f(__fmul_rn(-1.4426950408889634f, v))), RCP_SEED_MAX);   // seed valid below 2^126
  float r = __int_as_float(0x7EF311C3 - __float_as_int(d));
  r = r * fmaf(-d, r, 2.f);
  return r * fmaf(-d, r, 2.f);
}

// sigmoid of a pair, packed. Default: both exponentials in ONE MUFU op (ex2.approx.f16x2: relative error ~5e-4, below the bf16
// rounding of the table; an argument below -11 gives 2^t = inf in f16, clamped to a sigmoid of ~0, absolute error < 2e-5), the
// reciprocal by two Newton steps in f32 on the FMA pipe. SIG_F32: the exponentials in f32 (two MUFU ops); SIG_TANH: 0.5 tanh(a / 2)
// + 0.5 (one MUFU op per element, absolute error ~2.4e-4).
DEVI f2 sigmoid_pair(f2 a) {
#if defined(SIG_KIT)
  return mk2(sigmoid_kit(lo2(a)), sigmoid_kit(hi2(a)));
#elif defined(SIG_KITH2)
  const f2 t = mul2(a, mk2(-1.4426950408889634f, -1.4426950408889634f));
  uint32_t th, eh;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(th) : "f"(hi2(t)), "f"(lo2(t)));
  asm("ex2.approx.f16x2 %0, %1;" : "=r"(eh) : "r"(th));
  const float2 ef = h2f2(eh);
  return mk2(rcpf(ef.x + 1.f), rcpf(ef.y + 1.f));
#elif defined(SIG_TANH)
  const f2 h = mul2(a, mk2(0.5f, 0.5f));
  return fma2(mk2(tanhf_approx(lo2(h)), tanhf_approx(hi2(h))), mk2(0.5f, 0.5f), mk2(0.5f, 0.5f));
#else
  const f2 t = mul2(a, mk2(-1.4426950408889634f, -1.4426950408889634f));
#if defined(SIG_F32)
  const f2 e = mk2(ex2f(lo2(t)), ex2f(hi2(t)));
#else
  uint32_t th, eh;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(th) : "f"(hi2(t)), "f"(lo2(t)));
  asm("ex2.approx.f16x2 %0, %1;" : "=r"(eh) : "r"(th));
  const float2 ef = h2f2(eh);
  const f2 e = mk2(ef.x, ef.y);
#endif
  const f2 e1 = add2(e, mk2(1.f, 1.f));                     // f16 2^t is inf from t ~ 16 (a < -11): clamp for the seed
  const f2 d = mk2(fmin_nan(lo2(e1), RCP_SEED_MAX), fmin_nan(hi2(e1), RCP_SEED_MAX)), nd = neg2(d), two = mk2(2.f, 2.f);
  f2 r = mk2(__int_as_float(0x7EF311C3 - __float_as_int(lo2(d))), __int_as_float(0x7EF311C3 - __float_as_int(hi2(d))));
  r = mul2(r, fma2(nd, r, two));
  return mul2(r, fma2(nd, r, two));
#endif
}

extern "C" __global__ void __launch_bounds__(128 + 32 * NEW, 1)
bo_cond_tables_sm100(const __grid_constant__ CUtensorMap mc, const __grid_constant__ CUtensorMap mw,
                     const __grid_constant__ CUtensorMap mo, const float* __restrict__ bias, const float* __restrict__ colsum,
                     int T, int N, int n_g1, float eps) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int ntl = N / NT, items = (T / QM) * ntl;
  const int i0 = (int)((long)blockIdx.x * items / gridDim.x), i1 = (int)((long)(blockIdx.x + 1) * items / gridDim.x);

  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    mbar_init(&B.a_full, 1); mbar_init(&B.a_free, 2);           // a_free: the MMAs of the row tile and its row statistics
    for (int b = 0; b < 2; ++b) { mbar_init(&B.acc_full[b], 1); mbar_init(&B.acc_empty[b], NEW); }
    fence_barrier_init();
    prefetch_map(&mc); prefetch_map(&mw); prefetch_map(&mo);
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
#ifdef TRACE
  const long long tclk = clock64();
#endif

  if (warp == 0) {
    if (lane == 0) {
      int g = 0, am = -1, na = 0;
      for (int i = i0; i < i1; ++i) {
        const int m = i / ntl, n = i % ntl;
        if (m != am) {                                          // a new row tile: once the last one's MMAs and statistics are done
          if (na > 0) mbar_wait(&B.a_free, (na - 1) & 1);
          mbar_expect_tx(&B.a_full, NKB * TA);
          for (int kb = 0; kb < NKB; ++kb) tma_load_2d(su + O_A + kb * TA, &mc, &B.a_full, kb * 64, m * QM);
          am = m; ++na;
        }
        for (int kb = 0; kb < NKB; ++kb, ++g) {
          const int s = g % NST;
          if (g >= NST) mbar_wait(&B.empty[s], ((g / NST) - 1) & 1);
          mbar_expect_tx(&B.full[s], STB);
          tma_load_2d(su + O_B + s * STB, &mw, &B.full[s], kb * 64, n * NT);
        }
      }
    }
  } else if (warp == 1) {
    int g = 0, am = -1, na = 0, li = 0;
    for (int i = i0; i < i1; ++i, ++li) {
      const int m = i / ntl;
      if (m != am) { mbar_wait(&B.a_full, na & 1); am = m; ++na; }
      const bool last_of_m = (i + 1 == i1) || ((i + 1) / ntl != m);
      const int b = li & 1;
      if (li >= 2) mbar_wait(&B.acc_empty[b], ((li >> 1) - 1) & 1);
      tc_fence_after();
      for (int kb = 0; kb < NKB; ++kb, ++g) {
        const int s = g % NST;
        mbar_wait(&B.full[s], (g / NST) & 1);
        if (lane == 0) TR(0, g);
        tc_fence_after();
        const uint64_t da = desc_k128(su + O_A + kb * TA), dw = desc_k128(su + O_B + s * STB);
        if (elect_one()) {
#pragma unroll
          for (int ks = 0; ks < 4; ++ks) umma_ss(tmem + b * NT, da + (uint64_t)(ks * 2), dw + (uint64_t)(ks * 2), IDESC, (kb > 0 || ks > 0) ? 1u : 0u);
          tc_commit(&B.empty[s]);
          if (kb == NKB - 1) {
            tc_commit(&B.acc_full[b]);
            if (last_of_m) tc_commit(&B.a_free);
          }
        }
        __syncwarp();
      }
    }
  } else if (warp >= 4) {
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    const int wg = (warp - 4) >> 2, et = (warp - 4) * 32 + lane;   // warpgroup 0 / 1; epilogue thread 0..255
    const uint32_t stg = su + O_S + (uint32_t)(warp - 4) * NSW * SWB;
    float* red = reinterpret_cast<float*>(sm + O_R);            // [warpgroup][sum, sum of squares][row]
    int am = -1, na = 0, t = 0;
    float mu = 0.f, rs = 1.f;
    for (int i = i0, li = 0; i < i1; ++i, ++li) {
      const int m = i / ntl, n = i % ntl;
      if (m != am) {                                            // this row's LayerNorm statistics, from the resident A tile
        mbar_wait(&B.a_full, na & 1);
        if (r == 0 && li == 0) TR(1, 0);
        // one pass, packed pairs: var = E[c^2] - mu^2 (c is the bf16 conditioning, O(1); fp32 sums over 384 values)
        f2 s2[2] = {mk2(0.f, 0.f), mk2(0.f, 0.f)}, q2[2] = {mk2(0.f, 0.f), mk2(0.f, 0.f)};
#pragma unroll 1
        for (int kb = wg * (NKB / 2); kb < (wg + 1) * (NKB / 2); ++kb) {     // each warpgroup half of the row
          uint4 v[8];
#pragma unroll
          for (int c = 0; c < 8; ++c) v[c] = lds128(su + O_A + kb * TA + sw128(r, c));
#pragma unroll
          for (int c = 0; c < 8; ++c) {
            const uint32_t w4[4] = {v[c].x, v[c].y, v[c].z, v[c].w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const f2 x = mk2(bf16lo(w4[e]), bf16hi(w4[e]));
              s2[e & 1] = add2(s2[e & 1], x);
              q2[e & 1] = fma2(x, x, q2[e & 1]);
            }
          }
        }
        const f2 st = add2(s2[0], s2[1]), qt = add2(q2[0], q2[1]);
        red[(wg * 2) * QM + r] = lo2(st) + hi2(st);
        red[(wg * 2 + 1) * QM + r] = lo2(qt) + hi2(qt);
        named_bar_sync(1, 32 * NEW);
        mu = (red[r] + red[2 * QM + r]) * (1.f / KC);
        rs = rsqrtf(fmaxf((red[QM + r] + red[3 * QM + r]) * (1.f / KC) - mu * mu, 0.f) + eps);
        named_bar_sync(1, 32 * NEW);                              // red is read before the next row tile's statistics
        if (r == 0 && li == 0) TR(1, 1);
        if (et == 0) mbar_arrive(&B.a_free);                     // one arrival: the barrier counts the MMAs and this
        am = m; ++na;
      }
      const int b = li & 1, c0 = n * NT;
      const bool g1 = c0 < n_g1;
      // the tables that pass through a sigmoid: the AdaLN scales (groups 0 and 2 of every block's four) and both gates
      const bool sig = g1 ? (((c0 / 768) & 1) == 0) : true;
      // this item's bias and colsum columns into shared memory (every thread one float4), read back as broadcasts
      const uint32_t pb = su + O_P + (uint32_t)b * PB;
      if (et < 2 * NT / 4) {
        const float* src = et < NT / 4 ? bias + c0 + 4 * et : colsum + c0 + 4 * (et - NT / 4);
        const float4 x = __ldg(reinterpret_cast<const float4*>(src));
        sts128(pb + et * 16, make_uint4(__float_as_uint(x.x), __float_as_uint(x.y), __float_as_uint(x.z), __float_as_uint(x.w)));
      }
      named_bar_sync(1, 32 * NEW);
      mbar_wait(&B.acc_full[b], (li >> 1) & 1);
      if (r == 0 && li == 0) TR(1, 2);
      tc_fence_after();
      const f2 nmu = mk2(-mu, -mu), rs2 = mk2(rs, rs);
      // one 32-column piece: v -> bf16 staged and stored by this warp
      auto piece = [&](const uint32_t (&v)[32], int ch) {
        const int col = c0 + ch * 32;
        const bool tr = warp == 4 && lane == 0 && li == 0 && ch < 3;
        if (tr) TR(2, ch * 8 + 0);
        uint32_t p[16];
#pragma unroll
        for (int q = 0; q < 8; ++q) {
#ifdef NOLDS
          const uint4 bq = make_uint4(0u, 0u, 0u, 0u), cq = bq;
#else
          const uint4 bq = lds128(pb + (ch * 32 + 4 * q) * 4);
          const uint4 cq = lds128(pb + NT * 4 + (ch * 32 + 4 * q) * 4);
#endif
          const uint32_t bw[4] = {bq.x, bq.y, bq.z, bq.w}, cw[4] = {cq.x, cq.y, cq.z, cq.w};
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int k = 2 * q + h;
            f2 a = mk2u(v[2 * k], v[2 * k + 1]);
            if (g1) a = mul2(fma2(nmu, mk2u(cw[2 * h], cw[2 * h + 1]), a), rs2);
            a = add2(a, mk2u(bw[2 * h], bw[2 * h + 1]));
#ifndef NOSIG
            if (sig) a = sigmoid_pair(a);
#endif
            p[k] = pack_bf16(lo2(a), hi2(a));
          }
        }
        if (tr) TR(2, ch * 8 + 1);
        const uint32_t sb = stg + (uint32_t)(t % NSW) * SWB;
#pragma unroll
        for (int q = 0; q < 4; ++q) sts128(sb + sw64(lane, q), make_uint4(p[4 * q], p[4 * q + 1], p[4 * q + 2], p[4 * q + 3]));
        if (tr) TR(2, ch * 8 + 2);
        fence_proxy_async();
        if (tr) TR(2, ch * 8 + 3);
        __syncwarp();
        if (lane == 0) {
          tma_store_2d(&mo, sb, col, m * QM + (int)lb);
          tma_store_commit();
          asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(NSW - 1) : "memory");   // the tile reused next is free
        }
        __syncwarp();
        if (tr) TR(2, ch * 8 + 4);
        ++t;
      };
      // the next piece's TMEM load is in flight while this one is computed and stored
      // warpgroup wg takes pieces wg, wg + 2, ...
      uint32_t va[32], vb[32];
      tmem_ld32(trow + b * NT + wg * 32, va);
#pragma unroll 1
      for (int ch = wg; ch < NT / 32; ch += 4) {
        tmem_wait_ld();
        tmem_ld32(trow + b * NT + (ch + 2) * 32, vb);
        piece(va, ch);
        tmem_wait_ld();
        if (ch + 4 < NT / 32) tmem_ld32(trow + b * NT + (ch + 4) * 32, va);
        piece(vb, ch + 2);
        if (r == 0 && li == 0) TR(1, 3 + ch);
      }
      tmem_wait_ld();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.acc_empty[b]);
    }
    if (lane == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
