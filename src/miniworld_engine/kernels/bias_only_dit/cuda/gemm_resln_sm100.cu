// gemm_resln_sm100.cu -- a width-768 GEMM with the residual, the output gate and the next AdaLN in its epilogue, sm_100a:
//
//   x      += gate[tok] * (A W^T)                            (fp32 residual, in place)
//   xa      = LN(x) * scale[tok] + shift[tok]                (bf16, the next GEMM's input)        -- or, with FINAL,
//   out     = x                                              (bf16, the block's output; x is not written back)
//
// gate = sigmoid(gl), scale = sigmoid(ms), shift = mb; with PRESIG the tables already hold the sigmoids.
// A [M, K] bf16 (the gated attention output, K = 768, or the SwiGLU output, K = 1536), W [768, K] bf16 (nn.Linear layout).
//
// The LayerNorm needs whole 768-wide rows, and one CTA per 128-row tile would leave 15 CTAs at M = 1920: a cluster of CL CTAs
// takes one row tile, CTA c owns output columns [c NC, c NC + NC) (NC = 768 / CL). Each CTA reduces its slice to (mean, centred
// sum of squares), and the cluster exchanges those once (Chan's combination: exact, no E[x^2] - mean^2 cancellation): remote
// each CTA bulk-copies its 128 (mean, M2) pairs (1 KB) into every CTA, completing the transaction on that CTA's exchange mbarrier
// (armed at start). No barrier.cluster: the input producer warp waits on the epilogue and cannot take part; release arrives per
// peer cost ~0.5 us each. A CTA leaves only after its own exchange barrier completed, i.e. after every peer's copy into it.
//
// K loop: 16 KB A boxes ([64 k][128 rows]) and NC x 128 B W boxes, SW128, into an NST-stage ring. Epilogue: the updated x stays
// in tensor memory (written back over the accumulator). Its inputs stream in 32-column pieces through NIN slots (x fp32 SW128 +
// gate bf16 SW64 for pass A, scale + shift for pass C), loaded by their own producer warp; a slot is freed as soon as the four
// warps have read it into registers. Outputs leave through NOUT staging tiles, one TMA store per piece, the store of piece t - 1
// waited for (read) only after piece t is issued.
// Warps: 0 K-loop TMA producer, 1 MMA, 2 TMEM allocator, 3 epilogue-input TMA producer, 4-7 epilogue (one row per thread).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef CL
#define CL 8
#endif
constexpr int D = 768, NC = D / CL, QM = 128, NQ = NC / 32;
static_assert(NC % 32 == 0 && NC <= 256, "column slice");
constexpr int TA = QM * 128, TB = NC * 128, STB = TA + TB;
constexpr int XF = QM * 128, XH = QM * 64;                   // fp32 [128][32] SW128: 16 KB; bf16 [128][32] SW64: 8 KB
constexpr int NIN = 3, ES = XF + XH, NOUT = 3;               // input slot: x + gate (pass A) or scale + shift (pass C)
constexpr int O_IN = 0, O_OUT = O_IN + NIN * ES, O_ST = O_OUT + NOUT * XF;
constexpr int O_RED = 232448 - 1024 - 2 * (CL + 1) * QM * 4;   // float red[CL][2][128] and this CTA's own [2][128], then barriers
constexpr int NST = (O_RED - O_ST) / STB;
constexpr int O_MINE = O_RED + 2 * CL * QM * 4, O_BAR = O_MINE + 2 * QM * 4, SMEM_BYTES = O_BAR + 256;
static_assert(NST >= 2 && SMEM_BYTES <= 232448, "shared memory");
static_assert(O_OUT % 1024 == 0 && O_ST % 1024 == 0, "SW128 tiles need 1 KB alignment");
constexpr uint32_t IDESC = idesc_bf16(QM, NC), TCOLS = NC <= 128 ? 128 : 256;

#ifdef PRESIG
DEVI float sg(float v) { return v; }
#else
DEVI float sg(float v) { return rcp_nr(__fadd_rn(1.f, ex2f(__fmul_rn(-1.4426950408889634f, v)))); }   // one MUFU op
#endif

#ifdef TRACE
// CTA TRACE_CTA: 0 K block consumed (per block); 1 epilogue: [0] acc in, [1] pass A, [2] local stats, [3] exchanged, [4] pass C,
// [5] stored
__device__ long long g_tr[4][64];
#ifndef TRACE_CTA
#define TRACE_CTA 0
#endif
#define TR(ev, i) do { if (blockIdx.x == TRACE_CTA && (i) < 64) g_tr[ev][i] = clock64() - tclk; } while (0)
#else
#define TR(ev, i) do { } while (0)
#endif

#ifdef DBG
// a wait that gives up: prints where it hung and traps (debug builds only)
#define WAIT(bar, par, id) do { long long _n = 0; while (!mbar_try_wait((bar), (par))) { if (++_n == (1ll << 24)) { \
  printf("hang: cta %d warp %d lane %d wait %d parity %d\n", (int)blockIdx.x, warp, lane, (id), (int)(par)); __trap(); } } } while (0)
#else
#define WAIT(bar, par, id) mbar_wait((bar), (par))
#endif

struct Bars { uint64_t full[NST], empty[NST], efull[NIN], eempty[NIN], acc, xch; uint32_t tmem; };

DEVI void cluster_sync_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned; barrier.cluster.wait.aligned;" ::: "memory"); }
DEVI void st_cluster_f32(uint32_t local_addr, uint32_t rank, float v) {
  uint32_t ra;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(local_addr), "r"(rank));
  asm volatile("st.shared::cluster.f32 [%0], %1;" :: "r"(ra), "f"(v) : "memory");
}
DEVI void fence_acq_rel_cluster() { asm volatile("fence.acq_rel.cluster;" ::: "memory"); }
DEVI void tma_store_wait_read1() { asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory"); }
DEVI void tmem_st32(uint32_t taddr, const uint32_t (&r)[32]) {
  tmem_st16(taddr, *reinterpret_cast<const uint32_t(*)[16]>(r));
  tmem_st16(taddr + 16, *reinterpret_cast<const uint32_t(*)[16]>(r + 16));
}

extern "C" __global__ void __launch_bounds__(256, 1)
bo_gemm_resln_sm100(const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mw,
                    const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mgl,
                    const __grid_constant__ CUtensorMap mms, const __grid_constant__ CUtensorMap mmb,
                    const __grid_constant__ CUtensorMap mo, const __grid_constant__ CUtensorMap mxs,
                    const __grid_constant__ CUtensorMap mos, int K, int T, float eps) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t rank = cluster_rank();
  const int m0 = (int)(blockIdx.x / CL) * QM, n0 = (int)rank * NC, trow0 = m0 % T, nkb = K / 64;
#ifdef FINAL
  constexpr int NE = NQ;                                      // epilogue input pieces: pass A only
#else
  constexpr int NE = 2 * NQ;                                  // pass A, then pass C
#endif

  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    for (int e = 0; e < NIN; ++e) { mbar_init(&B.efull[e], 1); mbar_init(&B.eempty[e], 4); }
    mbar_init(&B.acc, 1);
    mbar_init(&B.xch, 1);
#ifndef FINAL
    mbar_expect_tx(&B.xch, CL * 2 * QM * 4);                  // armed now: every CTA's (mean, M2) rows arrive by bulk copy
#endif
    fence_barrier_init();
    prefetch_map(&ma); prefetch_map(&mw); prefetch_map(&mx); prefetch_map(&mgl); prefetch_map(&mo);
  }
  if (warp == 2) { tmem_alloc(smem_u32(&B.tmem), TCOLS); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync_relaxed();                                     // every CTA of the cluster is running before any remote access
  tc_fence_after();
  const uint32_t tmem = B.tmem;
#ifdef TRACE
  const long long tclk = clock64();
#endif

  if (warp == 0) {
    if (lane == 0) {
      for (int kb = 0; kb < nkb; ++kb) {
        const int s = kb % NST;
        if (kb >= NST) WAIT(&B.empty[s], ((kb / NST) - 1) & 1, 1);
        mbar_expect_tx(&B.full[s], STB);
        tma_load_2d(su + O_ST + s * STB, &ma, &B.full[s], kb * 64, m0);
        tma_load_2d(su + O_ST + s * STB + TA, &mw, &B.full[s], kb * 64, n0);
      }
    }
  } else if (warp == 1) {
    for (int kb = 0; kb < nkb; ++kb) {
      const int s = kb % NST;
      WAIT(&B.full[s], (kb / NST) & 1, 2);
      if (lane == 0) TR(0, kb);
      tc_fence_after();
      const uint64_t da = desc_k128(su + O_ST + s * STB), dw = desc_k128(su + O_ST + s * STB + TA);
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) umma_ss(tmem, da + (uint64_t)(ks * 2), dw + (uint64_t)(ks * 2), IDESC, (kb > 0 || ks > 0) ? 1u : 0u);
        tc_commit(&B.empty[s]);
        if (kb == nkb - 1) tc_commit(&B.acc);
      }
      __syncwarp();
    }
  } else if (warp == 3) {
    if (lane == 0) {
      for (int e = 0; e < NE; ++e) {
        const int slot = e % NIN, q = e < NQ ? e : e - NQ;
        const uint32_t sb = su + O_IN + slot * ES;
        if (e >= NIN) WAIT(&B.eempty[slot], ((e / NIN) - 1) & 1, 3);
        if (e < NQ) {
          mbar_expect_tx(&B.efull[slot], QM * 32 * 4 + QM * 32 * 2);
          tma_load_2d(sb, &mx, &B.efull[slot], n0 + q * 32, m0);
          tma_load_2d(sb + XF, &mgl, &B.efull[slot], n0 + q * 32, trow0);
        } else {
          mbar_expect_tx(&B.efull[slot], 2 * QM * 32 * 2);
          tma_load_2d(sb, &mms, &B.efull[slot], n0 + q * 32, trow0);
          tma_load_2d(sb + XH, &mmb, &B.efull[slot], n0 + q * 32, trow0);
        }
      }
    }
  } else if (warp >= 4) {
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane, trow = tmem + (lb << 16);
    float* red = reinterpret_cast<float*>(sm + O_RED);      // [CL][2][128]: (mean, centred sum of squares) of every CTA's slice
    // input piece e: wait, read into registers (by the caller), then free the slot
    auto wait_in = [&](int e) { WAIT(&B.efull[e % NIN], (e / NIN) & 1, 4); return su + O_IN + (e % NIN) * ES; };
    auto free_in = [&](int e) { __syncwarp(); if (lane == 0) mbar_arrive(&B.eempty[e % NIN]); };
    // output piece t from staging tile t % NOUT: each warp stores its own 32 rows (no cross-warp barrier), then waits for its
    // store of piece t - 1 to have read that tile. rowb = bytes per staged row (128 fp32 SW128, 64 bf16 SW64).
    auto store_out = [&](int t, const CUtensorMap* m, uint32_t rowb) {
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) {
        tma_store_2d(m, su + O_OUT + (t % NOUT) * XF + lb * rowb, n0 + (t % NQ) * 32, m0 + (int)lb);
        tma_store_commit();
        asm volatile("cp.async.bulk.wait_group.read %0;" :: "n"(NOUT - 1) : "memory");   // the tile reused next is free
      }
      __syncwarp();
    };
    // the slice's (count, mean, centred sum of squares), merged piece by piece (Chan)
    float n_ = 0.f, mean_ = 0.f, m2_ = 0.f;
    auto merge = [&](const uint32_t (&v)[32]) {
      float s4[4] = {0.f, 0.f, 0.f, 0.f}, q4[4] = {0.f, 0.f, 0.f, 0.f};      // four chains: the sums are latency-bound
#pragma unroll
      for (int k = 0; k < 32; ++k) s4[k & 3] += __uint_as_float(v[k]);
      const float pm = ((s4[0] + s4[1]) + (s4[2] + s4[3])) * (1.f / 32);
#pragma unroll
      for (int k = 0; k < 32; ++k) { const float d = __uint_as_float(v[k]) - pm; q4[k & 3] += d * d; }
      const float pq = (q4[0] + q4[1]) + (q4[2] + q4[3]);
      const float nn = n_ + 32.f, d = pm - mean_;
      mean_ += d * (32.f / nn);
      m2_ += pq + d * d * (n_ * 32.f / nn);
      n_ = nn;
    };
    WAIT(&B.acc, 0, 5);
    if (r == 0) TR(1, 0);
    tc_fence_after();
    // pass A: x += gate * acc, back into tensor memory and out; the slice's statistics
#pragma unroll 1
    for (int q = 0; q < NQ; ++q) {
      uint32_t acc[32];
      tmem_ld32(trow + q * 32, acc);
      const uint32_t xb = wait_in(q), gb = xb + XF;
      if (r == 0) TR(2, q * 6 + 0);
      uint4 x4[8], g4[4];
#pragma unroll
      for (int c = 0; c < 8; ++c) x4[c] = lds128(xb + sw128(r, c));
#pragma unroll
      for (int c = 0; c < 4; ++c) g4[c] = lds128(gb + sw64(r, c));
      free_in(q);
      tmem_wait_ld();
      if (r == 0) TR(2, q * 6 + 1);
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const uint32_t g0 = (c & 1) ? g4[c >> 1].z : g4[c >> 1].x, g1 = (c & 1) ? g4[c >> 1].w : g4[c >> 1].y;
        const float v0 = __uint_as_float(x4[c].x) + sg(bf16lo(g0)) * __uint_as_float(acc[4 * c + 0]);
        const float v1 = __uint_as_float(x4[c].y) + sg(bf16hi(g0)) * __uint_as_float(acc[4 * c + 1]);
        const float v2 = __uint_as_float(x4[c].z) + sg(bf16lo(g1)) * __uint_as_float(acc[4 * c + 2]);
        const float v3 = __uint_as_float(x4[c].w) + sg(bf16hi(g1)) * __uint_as_float(acc[4 * c + 3]);
        acc[4 * c + 0] = __float_as_uint(v0); acc[4 * c + 1] = __float_as_uint(v1);
        acc[4 * c + 2] = __float_as_uint(v2); acc[4 * c + 3] = __float_as_uint(v3);
      }
#ifndef FINAL
      merge(acc);
#endif
      if (r == 0) TR(2, q * 6 + 2);
      const uint32_t ob = su + O_OUT + (q % NOUT) * XF;
#ifdef FINAL
#pragma unroll
      for (int c = 0; c < 4; ++c)
        sts128(ob + sw64(r, c), make_uint4(pack_bf16(__uint_as_float(acc[8 * c + 0]), __uint_as_float(acc[8 * c + 1])),
                                           pack_bf16(__uint_as_float(acc[8 * c + 2]), __uint_as_float(acc[8 * c + 3])),
                                           pack_bf16(__uint_as_float(acc[8 * c + 4]), __uint_as_float(acc[8 * c + 5])),
                                           pack_bf16(__uint_as_float(acc[8 * c + 6]), __uint_as_float(acc[8 * c + 7]))));
      store_out(q, &mos, 64);
#else
      tmem_st32(trow + q * 32, acc);
#pragma unroll
      for (int c = 0; c < 8; ++c) sts128(ob + sw128(r, c), make_uint4(acc[4 * c], acc[4 * c + 1], acc[4 * c + 2], acc[4 * c + 3]));
      if (r == 0) TR(2, q * 6 + 3);
      store_out(q, &mxs, 128);
      if (r == 0) TR(2, q * 6 + 4);
#endif
    }
    if (r == 0) TR(1, 1);
#ifndef FINAL
    tmem_wait_st();
    const float lmean = mean_, m2 = m2_;
    if (r == 0) TR(1, 2);
    // this CTA's rows go to every CTA's red[rank] by one 1 KB bulk copy each, completing on that CTA's exchange barrier
    float* own = reinterpret_cast<float*>(sm + O_MINE);
    own[r] = lmean;
    own[QM + r] = m2;
    fence_proxy_async();
    named_bar_sync(1, 128);
    if (r == 0) {
      const uint32_t src = su + O_MINE, dst = su + O_RED + (uint32_t)rank * 2 * QM * 4, bar = smem_u32(&B.xch);
      for (int c = 0; c < CL; ++c) {
        uint32_t rd, rb;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rd) : "r"(dst), "r"(c));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rb) : "r"(bar), "r"(c));
        asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(rd), "r"(src), "n"(2 * QM * 4), "r"(rb) : "memory");
      }
    }
    WAIT(&B.xch, 0, 6);
    float mean = 0.f;
#pragma unroll
    for (int c = 0; c < CL; ++c) mean += red[(c * 2) * QM + r];
    mean *= 1.f / CL;
    float M2 = 0.f;
#pragma unroll
    for (int c = 0; c < CL; ++c) { const float d = red[(c * 2) * QM + r] - mean; M2 += red[(c * 2 + 1) * QM + r] + NC * d * d; }
    const float rstd = rsqrtf(M2 * (1.f / D) + eps);
    if (r == 0) TR(1, 3);
    // pass C: xa = (x - mean) rstd scale + shift, bf16
#pragma unroll 1
    for (int q = 0; q < NQ; ++q) {
      uint32_t v[32];
      tmem_ld32(trow + q * 32, v);
      const uint32_t sb = wait_in(NQ + q), bb = sb + XH;
      uint4 s4[4], b4[4];
#pragma unroll
      for (int c = 0; c < 4; ++c) { s4[c] = lds128(sb + sw64(r, c)); b4[c] = lds128(bb + sw64(r, c)); }
      free_in(NQ + q);
      tmem_wait_ld();
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const uint32_t sw[4] = {s4[c].x, s4[c].y, s4[c].z, s4[c].w}, bw[4] = {b4[c].x, b4[c].y, b4[c].z, b4[c].w};
        uint32_t p[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float x0 = __uint_as_float(v[8 * c + 2 * k]), x1 = __uint_as_float(v[8 * c + 2 * k + 1]);
          p[k] = pack_bf16((x0 - mean) * rstd * sg(bf16lo(sw[k])) + bf16lo(bw[k]), (x1 - mean) * rstd * sg(bf16hi(sw[k])) + bf16hi(bw[k]));
        }
        s4[c] = make_uint4(p[0], p[1], p[2], p[3]);
      }
      const uint32_t ob = su + O_OUT + ((NQ + q) % NOUT) * XF;
#pragma unroll
      for (int c = 0; c < 4; ++c) sts128(ob + sw64(r, c), s4[c]);
      store_out(NQ + q, &mos, 64);
    }
    if (r == 0) TR(1, 4);
#endif
    if (r == 0) { tma_store_wait0(); TR(1, 5); }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 2) { tc_fence_after(); tmem_dealloc(tmem, TCOLS); }
}
