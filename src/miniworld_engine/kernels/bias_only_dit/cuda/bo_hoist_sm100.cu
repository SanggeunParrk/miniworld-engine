// bo_hoist_sm100.cu -- the hoisted pair bias of the bias-only token DiT on B200 (sm_100a), bf16 and fp32: every block's bias from ONE pass
// over the pair (forward) and the whole backward in ONE more pass (``integrations/bias_only_dit_hoist.py``; shared pair [1, L, L, 128]
// or a pair per sample [A, L, L, 128] -- R = A L^2 pair rows either way).
//
//   forward    bias_all^T [N, R] = W' LN0(x)^T,  W' = [W_b diag(g_b)]_b [N, 128] bf16, LN0 the LayerNorm without affine
//   backward   d LN0 = dbias_all^T W'  ->  dx = LN0 backward (no affine),  dW' += dbias_all LN0
//
// Forward (bo_hoist_fwd_sm100). The projection runs on the RAW pair rows: LN0(x) W'^T = rstd (x W'^T - mean s), s_n = sum_k W'[n, k]
// (the bf16 W' the product uses, summed in fp32), so the pair tile goes from TMA straight into the tcgen05 product (M 128 rows x
// N NP x K 128, K-major x and W') and the LayerNorm is two numbers per row applied in the epilogue: the normalised pair never exists.
// (The products are exact in fp32 and the row is not rounded to bf16 after normalising: closer to the exact value than LN0 rounded
// to bf16 and multiplied; the cancellation of x W'^T against mean s costs ~2^-24 |mean| / std, nothing for a pair activation.)
//
// Backward (bo_hoist_bwd_sm100). Per 128-row tile: d LN0 = dbias^T W' on the tensor core (M 128 rows x N 128 channels x K NP: the
// dbias tile [NP][128 rows] is the MN-major A operand, W' [NP][128] the MN-major B), drained by the thread of each row (TMEM lane =
// row): one pass over the row and d LN0 for the row statistics (shifted one-pass sums, the forward's arithmetic), mean(dy) and
// mean(dy xhat); a second for dx = rstd (dy - mean(dy) - xhat mean(dy xhat)), staged per warp and stored by TMA. The weight
// gradient never forms xhat: with d'[n, r] = dbias[n, r] rstd_r,
//     dW'[n, k] = sum_r d'[n, r] (x[r, k] - mean_r) = S[n, k] - mean_k S[n, :],   S = d' x
// because every LN0 row sums to zero (sum_k S[n, k] = 128 sum_r d'[n, r] mean_r). So S^T [128 channels][NP] accumulates in TMEM over
// the CTA's tiles from the RAW pair tile (MN-major A, as TMA left it) and d' as the two-term bf16 split [hi | lo] (K-major B, written
// by the row threads: NP values each, hi + lo = the fp32 product to ~2^-17) -- the bf16 dbias against the exact pair and rstd to fp32
// accumulation; the host subtracts the channel mean of the summed partials. S is fp32: the subtraction costs ~2^-24 |mean| / std.
// Each CTA leaves S as an fp32 partial [NP][128], summed by the host in a fixed order: deterministic.
//
// Warps (both kernels): 0 TMA producer, 1 MMA issuer and TMEM allocator, 2-9 two epilogue groups of four (group g takes the
// CTA's tiles g, g + 2, ..., each with its own accumulator: one group drains while the other's product runs). Persistent grid over
// the R / 128 tiles. -DNP: the padded weight rows (a multiple of 16; rows N .. NP - 1 are TMA zero fill and never stored).
//
// More weight rows and fp32 (2026-10-29): the forward takes -DNCH chunks of NP rows (W' resident; four TMEM accumulators, two per
// epilogue group) and the host splits larger N into row ranges per launch; -DTF32 is its fp32 form (the tile normalised in place,
// then kind::tf32). The backward past one chunk, and in fp32, is bo_hoist_bwd2_sm100 (-DBWD2, at the end of this file).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef NP
#define NP 16
#endif
static_assert(NP % 16 == 0 && NP >= 16 && NP <= 128, "weight rows: a multiple of 16, at most 128");
constexpr int TM = 128, DP = 128;                 // rows per tile, pair channels
constexpr int XT = TM * DP * 2;                   // pair tile [2 boxes][128 rows][64] bf16 SW128: 32 KB
constexpr int XB = TM * 64 * 2;                   // one box: 16 KB
constexpr int WT = NP * DP * 2;                   // W' [2 boxes][NP][64] SW128
constexpr int WB = NP * 64 * 2;
constexpr int DBT = NP * TM * 2;                  // dbias tile [2 boxes][NP][64 rows] SW128
constexpr int DBB = NP * 64 * 2;
constexpr int THREADS = 320;
constexpr int SMEM_MAX = 232448;
// forward: NCH chunks of NP weight rows (N <= NCH NP per launch; the host splits larger N over launches), W' resident
#ifndef NCH
#define NCH 1
#endif
#ifdef TF32
constexpr int ES = 4;                             // fp32 pair / W' / bias
#else
constexpr int ES = 2;
#endif
constexpr int F_NBOX = DP * ES / 128;             // 128-byte K boxes per pair row: 2 (bf16) / 4 (fp32)
constexpr int F_XT = TM * DP * ES, F_XB = TM * 128;              // pair tile [F_NBOX][128 rows][128 B] SW128: 32 / 64 KB
constexpr int F_WB = NP * 128, F_WT = F_NBOX * F_WB;             // W' chunk [F_NBOX][NP rows][128 B] SW128
constexpr int F_KSTEPS = DP * ES / 32;            // K steps of 32 bytes: 8 (bf16, K16) / 16 (tf32, K8)
#ifdef TF32
constexpr int F_SN = 0;                           // no s_n: the tile is normalised in place before its products
#else
constexpr int F_SN = NCH * NP * 4;
#endif
constexpr int F_O_W = 0, F_O_ST = F_O_W + ((NCH * F_WT + 1023) / 1024) * 1024;
constexpr int F_FIT = (SMEM_MAX - 1024 - F_O_ST - F_SN - 512) / F_XT;
constexpr int F_NST = F_FIT > 8 ? 8 : F_FIT;
constexpr int F_O_S = F_O_ST + F_NST * F_XT, F_O_BAR = F_O_S + F_SN, F_SMEM = 1024 + F_O_BAR + 512;
static_assert(F_NST >= 2 && F_SMEM <= SMEM_MAX, "forward shared memory");
static_assert(NCH >= 1 && NCH * NP <= 1024, "weight-row chunks");
#ifdef TF32
constexpr uint32_t F_IDESC = idesc_tf32(TM, NP);
#else
constexpr uint32_t F_IDESC = idesc_bf16(TM, NP);
#endif
constexpr uint32_t F_TCOLS = 4 * NP <= 32 ? 32 : 4 * NP <= 64 ? 64 : 4 * NP <= 128 ? 128 : 4 * NP <= 256 ? 256 : 512;
#if NP <= 64 && NCH == 1 && !defined(TF32)  // ------------------------------------------- backward (bf16, NP <= 64)
constexpr int B_STAGE = XT + DBT;
constexpr int DXS = 32 * 64, B_DX = 8 * 2 * DXS;          // dx staging: per warp two [32 rows][32 channels] bf16 SW64 slots
constexpr int B_O_W = 0, B_O_DS = B_O_W + ((WT + 1023) / 1024) * 1024;   // W' | d' [2 groups][hi, lo][NP][128 rows] | dx staging
constexpr int B_O_DX = B_O_DS + 2 * 2 * DBT, B_O_ST = B_O_DX + B_DX;
constexpr int B_NST = (SMEM_MAX - 1024 - B_O_ST - 512) / B_STAGE > 6 ? 6 : (SMEM_MAX - 1024 - B_O_ST - 512) / B_STAGE;
constexpr int B_O_BAR = B_O_ST + B_NST * B_STAGE, B_SMEM = 1024 + B_O_BAR + 256;
static_assert(B_NST >= 2 && B_SMEM <= SMEM_MAX, "backward shared memory");
static_assert(B_STAGE % 1024 == 0 && XT % 1024 == 0 && DBT % 1024 == 0, "SW128 operands need 1 KB alignment");
constexpr uint32_t B_ID_DLN = idesc_bf16(TM, DP, 1, 1);   // A = dbias^T (dbias tile MN-major), B = W' (MN-major)
constexpr uint32_t B_ID_S = idesc_bf16(DP, NP, 1, 0);     // A = x^T (pair tile MN-major), B = d' (K-major)
constexpr int B_COL_S = 2 * DP;                           // TMEM: d LN0 [2 groups][128], S^T [NP]
#endif

// bf16 pair -> (lo, hi) fp32
DEVI f2 unpk(uint32_t w) { return mk2u(w << 16, w & 0xffff0000u); }
// a fresh read of shared memory (ld.volatile: never merged with an earlier read of the same address). The row is read again in each
// pass instead of held in registers across passes: ptxas merges the passes' unpacks of a held row and then keeps all 128 unpacked
// values live (spills)
DEVI uint4 lds128v(uint32_t a) {
  uint4 v;
  asm volatile("ld.volatile.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory");
  return v;
}
DEVI uint32_t pk(f2 v) { return pack_bf16(lo2(v), hi2(v)); }
DEVI f2 dup2(float a) { return mk2(a, a); }
// byte offset of 16-byte chunk q (0..15) of row r in a [2 boxes][128 rows][64] bf16 SW128 tile
DEVI uint32_t xoff(uint32_t r, uint32_t q) { return (q >> 3) * XB + sw128(r, q & 7); }
// the row's first value: the shift of its one-pass statistics (sums of x - k and (x - k)^2: no cancellation beyond |mean - k| / std)
DEVI float row_shift(uint32_t tile, uint32_t r) { return bf16lo(lds128v(tile + xoff(r, 0)).x); }
// accumulate the shifted sums of one 16-byte chunk (8 values)
DEVI void acc_stats(const uint4& u, f2 nk, f2& s1, f2& s2) {
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const f2 t = add2(unpk(w[k]), nk);
    s1 = add2(s1, t);
    s2 = fma2(t, t, s2);
  }
}
// mean and 1 / sqrt(var + eps) from the shifted sums: the forward's and the backward's arithmetic (bit-identical statistics)
DEVI void finish_stats(f2 s1, f2 s2, float k, float eps, float& mean, float& rstd, float& m1) {
  m1 = (lo2(s1) + hi2(s1)) * (1.f / DP);                    // mean - k
  const float var = fmaxf((lo2(s2) + hi2(s2)) * (1.f / DP) - m1 * m1, 0.f);
  mean = k + m1;
  rstd = rsqrtf(var + eps);
}

// a per-tile opaque copy of a loop-invariant value: the shared-memory addresses derived from it are recomputed per tile (a few
// integer instructions) instead of hoisted out of the tile loop into ~30 registers
DEVI uint32_t opaque(uint32_t v) { asm volatile("" : "+r"(v)); return v; }

DEVI void stg32(void* p, uint32_t v) { asm volatile("st.global.b32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }

// ================================================================================================ forward
// Items: the CTA's tiles i (persistent grid), each with NCH chunks of NP weight rows. Epilogue group g = i & 1 drains tile i's chunks
// in order from its two accumulators (2 g + (u & 1), u = (i >> 1) NCH + j its running chunk count), so the products of the next
// chunk run while one drains. The MMA warp walks the tiles in pairs and, within a pair, the chunks: chunk j of both groups' tiles
// before chunk j + 1, so neither group waits behind the other's drain.
// bf16: the products read the raw pair tile; the drain applies the LayerNorm, bias = rstd acc - mean rstd s_n.
// -DTF32: the group's row threads normalise their rows IN PLACE first (LN0 rounded to the nearest tf32, cvt.rna, as cuBLAS rounds
// the GEMM operand of the LayerNorm-rows path), then arrive on normed[s]; the products read LN0 and the drain stores acc as it is
// -- the arithmetic of LN0 rows + a TF32 GEMM, with the normalised pair never leaving shared memory. (The raw-row fold would put the
// tf32 rounding of x, ~2^-11 |x|, against std: ~2^-11 |mean| / std after the cancellation.)
// Barriers: full[s] (TMA), empty[s] (bf16: the last chunk's MMA commit + the group's 4 warps after their statistics; tf32: the MMA
// commit), normed[s] (tf32: the group's 4 warps), acc[a] (MMA commit) / acc_empty[a] (4 warps), w (W', once).
struct FBars { uint64_t full[F_NST], empty[F_NST], normed[F_NST], acc[4], acc_empty[4], w; uint32_t tmem; };

#ifdef TF32
DEVI uint32_t tf32_rna(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
#endif

// mx: pair rows [R][128], box (128 B, 128 rows) SW128; mw: W' [N][128] (this launch's rows), box (128 B, NP) SW128 (rows past N
// zero-filled); out: bias_all^T [N][R] (bf16, or fp32 with TF32), rows of R; wf: W' (bf16: the row sums s_n); R a multiple of 128
extern "C" __global__ void __launch_bounds__(THREADS, 1)
bo_hoist_fwd_sm100(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mw, void* __restrict__ outv,
                   const void* __restrict__ wfv, int R, int N, int eps_bits) {
  extern __shared__ __align__(1024) uint8_t smraw[];
  uint8_t* sm = smraw + ((1024u - (smem_u32(smraw) & 1023u)) & 1023u);
  const uint32_t su = smem_u32(sm);
  FBars& B = *reinterpret_cast<FBars*>(sm + F_O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int T = R / TM, G = gridDim.x, my = (T > (int)blockIdx.x) ? (T - (int)blockIdx.x + G - 1) / G : 0;
  const float eps = __int_as_float(eps_bits);

  if (tid == 0) {
    for (int s = 0; s < F_NST; ++s) {
#ifdef TF32
      mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); mbar_init(&B.normed[s], 4);
#else
      mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 5);   // empty: the MMA + 4 drain warps
#endif
    }
    for (int a = 0; a < 4; ++a) { mbar_init(&B.acc[a], 1); mbar_init(&B.acc_empty[a], 4); }
    mbar_init(&B.w, 1);
    fence_barrier_init();
    prefetch_map(&mx); prefetch_map(&mw);
  }
#ifndef TF32
  float* s_n = reinterpret_cast<float*>(sm + F_O_S);
  const uint32_t sn_base = su + F_O_S;
  {                                                          // s_n = sum_k W'[n, k] (the bf16 values), 0 for the padding rows
    const __nv_bfloat16* wf = reinterpret_cast<const __nv_bfloat16*>(wfv);
    for (int n = tid; n < NCH * NP; n += THREADS) {
      float s = 0.f;
      if (n < N)
        for (int k = 0; k < DP; ++k) s += __bfloat162float(wf[(long)n * DP + k]);
      s_n[n] = s;
    }
  }
#endif
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), F_TCOLS); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    if (lane == 0) {
      mbar_expect_tx(&B.w, NCH * F_WT);
      for (int j = 0; j < NCH; ++j)
        for (int b = 0; b < F_NBOX; ++b) tma_load_2d(su + F_O_W + j * F_WT + b * F_WB, &mw, &B.w, b * (128 / ES), j * NP);
      for (int i = 0; i < my; ++i) {
        const int s = i % F_NST, r0 = ((int)blockIdx.x + i * G) * TM;
        if (i >= F_NST) mbar_wait(&B.empty[s], ((i / F_NST) - 1) & 1);
        const uint32_t st = su + F_O_ST + s * F_XT;
        mbar_expect_tx(&B.full[s], F_XT);
        for (int b = 0; b < F_NBOX; ++b) tma_load_2d(st + b * F_XB, &mx, &B.full[s], b * (128 / ES), r0);
      }
    }
  } else if (warp == 1) {
    mbar_wait(&B.w, 0);
    for (int p = 0; p < my; p += 2)
      for (int j = 0; j < NCH; ++j)
        for (int t = p; t < p + 2 && t < my; ++t) {
          const int s = t % F_NST, g = t & 1, u = (t >> 1) * NCH + j, a = 2 * g + (u & 1);
#ifdef TF32
          if (j == 0) mbar_wait(&B.normed[s], (t / F_NST) & 1);   // the tile normalised in place
#else
          if (j == 0) mbar_wait(&B.full[s], (t / F_NST) & 1);
#endif
          if (u >= 2) mbar_wait(&B.acc_empty[a], ((u >> 1) - 1) & 1);
          tc_fence_after();
          const uint32_t st = su + F_O_ST + s * F_XT, wj = su + F_O_W + j * F_WT;
          if (elect_one()) {
#pragma unroll
            for (int ks = 0; ks < F_KSTEPS; ++ks) {          // K step: 32 bytes within a 128-byte row (16 bf16 / 8 tf32 channels)
              const uint32_t bx = (uint32_t)(ks >> 2), ko = (uint32_t)(ks & 3) * 2;
#ifdef TF32
              umma_ss_tf32(tmem + a * NP, desc_k128(st + bx * F_XB) + ko, desc_k128(wj + bx * F_WB) + ko, F_IDESC, ks ? 1u : 0u);
#else
              umma_ss(tmem + a * NP, desc_k128(st + bx * F_XB) + ko, desc_k128(wj + bx * F_WB) + ko, F_IDESC, ks ? 1u : 0u);
#endif
            }
            tc_commit(&B.acc[a]);
            if (j == NCH - 1) tc_commit(&B.empty[s]);
          }
          __syncwarp();
        }
  } else {
    const int g = (warp - 2) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane;
    for (int i = g; i < my; i += 2) {
      const int s = i % F_NST, r0 = ((int)blockIdx.x + i * G) * TM;
      // (tile i - F_NST, the stage's previous use, may be the other group's when F_NST is odd: its full phase has completed all the
      // same -- this group drained tile i - 2, so the MMA warp, which takes the tiles in order, has waited for every earlier tile)
      mbar_wait(&B.full[s], (i / F_NST) & 1);
      const uint32_t xt = su + F_O_ST + s * F_XT, rr = opaque(r);
#ifdef TF32
      // shifted one-pass statistics of the fp32 row (32 chunks of 4), then the row rewritten as LN0 (tf32-rounded) in place
      const float k0 = __uint_as_float(lds128v(xt + xoff(rr, 0)).x);
      f2 s1 = mk2(0.f, 0.f), s2 = s1, s3 = s1, s4 = s1;
      const f2 nk = dup2(-k0);
#pragma unroll 4
      for (int q = 0; q < 32; ++q) {
        const uint4 u = lds128v(xt + xoff(rr, q));
        const f2 t0 = add2(mk2u(u.x, u.y), nk), t1 = add2(mk2u(u.z, u.w), nk);
        s1 = add2(s1, t0); s2 = fma2(t0, t0, s2); s3 = add2(s3, t1); s4 = fma2(t1, t1, s4);
      }
      float mean, rstd, m1;
      finish_stats(add2(s1, s3), add2(s2, s4), k0, eps, mean, rstd, m1);
      const f2 rs = dup2(rstd), nmr = dup2(-mean * rstd);
#pragma unroll 4
      for (int q = 0; q < 32; ++q) {
        const uint32_t ad = xt + xoff(rr, q);
        const uint4 u = lds128v(ad);
        const f2 y0 = fma2(mk2u(u.x, u.y), rs, nmr), y1 = fma2(mk2u(u.z, u.w), rs, nmr);
        sts128(ad, make_uint4(tf32_rna(lo2(y0)), tf32_rna(hi2(y0)), tf32_rna(lo2(y1)), tf32_rna(hi2(y1))));
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.normed[s]);
      float* ob = reinterpret_cast<float*>(outv) + r0 + (int)r;
      for (int j = 0; j < NCH; ++j) {
        const int u = (i >> 1) * NCH + j, a = 2 * g + (u & 1);
        mbar_wait(&B.acc[a], (u >> 1) & 1);
        tc_fence_after();
#pragma unroll 1
        for (int c = 0; c < NP / 16; ++c) {
          uint32_t v[16];
          tmem_ld16(tmem + (lb << 16) + a * NP + c * 16, v);
          tmem_wait_ld();
#pragma unroll
          for (int k = 0; k < 16; ++k) {
            const int row = j * NP + c * 16 + k;
            if (row < N) stg32(ob + (long)row * R, v[k]);
          }
        }
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.acc_empty[a]);
      }
#else
      const bool odd = lane & 1;
      const float k0 = row_shift(xt, rr);
      f2 s1 = mk2(0.f, 0.f), s2 = s1, s3 = s1, s4 = s1;
#pragma unroll
      for (int q = 0; q < 16; q += 2) {
        acc_stats(lds128v(xt + xoff(rr, q)), dup2(-k0), s1, s2);
        acc_stats(lds128v(xt + xoff(rr, q + 1)), dup2(-k0), s3, s4);
      }
      float mean, rstd, m1;
      finish_stats(add2(s1, s3), add2(s2, s4), k0, eps, mean, rstd, m1);
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.empty[s]);
      const f2 rs = dup2(rstd), nmr = dup2(-mean * rstd);
      // bias[n][r0 + r] = rstd acc[r][n] - mean rstd s_n; two lanes trade one value per column pair so that each stores a 32-bit
      // word: the even lane (n, r | r + 1), the odd lane (n + 1, r - 1 | r)
      __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(outv) + r0 + (int)lb + (lane & ~1);
      for (int j = 0; j < NCH; ++j) {
        const int u = (i >> 1) * NCH + j, a = 2 * g + (u & 1);
        mbar_wait(&B.acc[a], (u >> 1) & 1);
        tc_fence_after();
#pragma unroll 1
        for (int c = 0; c < NP / 16; ++c) {
          uint32_t v[16];
          tmem_ld16(tmem + (lb << 16) + a * NP + c * 16, v);
          tmem_wait_ld();
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            const int n = j * NP + c * 16 + 2 * k;
            float2 snf;                                     // ld.volatile: plain loads of s_n get hoisted into NP registers
            asm volatile("ld.volatile.shared.v2.f32 {%0,%1}, [%2];" : "=f"(snf.x), "=f"(snf.y) : "r"(sn_base + n * 4) : "memory");
            const f2 sn = mk2(snf.x, snf.y);
            const f2 o = fma2(mk2u(v[2 * k], v[2 * k + 1]), rs, mul2(nmr, sn));
            const float send = odd ? lo2(o) : hi2(o);
            const float recv = __shfl_xor_sync(0xffffffffu, send, 1);
            const uint32_t w = odd ? pack_bf16(recv, hi2(o)) : pack_bf16(lo2(o), recv);
            const int row = n + (odd ? 1 : 0);
            if (row < N) stg32(ob + (long)row * R, w);
          }
        }
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.acc_empty[a]);
      }
#endif
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, F_TCOLS); }
}

// ================================================================================================ backward
// (built for bf16, NP <= 64 only: the d' buffers and two stages need the shared memory)
#if NP <= 64 && NCH == 1 && !defined(TF32)
struct BBars { uint64_t full[B_NST], empty[B_NST], acc[2], acc_empty[2], dfull[2], dempty[2], w, done; uint32_t tmem; };

// byte offset of element (n, r) in a [2 boxes][NP][64 rows] bf16 SW128 tile (the dbias tile and the d' tiles)
DEVI uint32_t doff(uint32_t n, uint32_t r) { return (r >> 6) * DBB + n * 128u + (((((r & 63u) >> 3) ^ (n & 7u))) << 4) + (r & 7u) * 2u; }
DEVI uint32_t lds16(uint32_t a) { unsigned short v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a) : "memory"); return v; }
DEVI void sts16(uint32_t a, uint32_t v) { asm volatile("st.shared.u16 [%0], %1;" :: "r"(a), "h"((unsigned short)v) : "memory"); }

// mx: pair rows (as forward); mdb: dbias_all [N][R] bf16, box (64, NP) SW128 (rows past N zero-filled); mw: W' (as forward);
// mdx: d pair [R][128] bf16, box (32, 32) SW64; part: S partials [grid][NP][128] fp32
extern "C" __global__ void __launch_bounds__(THREADS, 1)
bo_hoist_bwd_sm100(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mdb,
                   const __grid_constant__ CUtensorMap mw, const __grid_constant__ CUtensorMap mdx, float* __restrict__ part, int R,
                   int eps_bits) {
  extern __shared__ __align__(1024) uint8_t smraw[];
  uint8_t* sm = smraw + ((1024u - (smem_u32(smraw) & 1023u)) & 1023u);
  const uint32_t su = smem_u32(sm);
  BBars& B = *reinterpret_cast<BBars*>(sm + B_O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int T = R / TM, G = gridDim.x, my = (T > (int)blockIdx.x) ? (T - (int)blockIdx.x + G - 1) / G : 0;
  const float eps = __int_as_float(eps_bits);

  if (tid == 0) {
    for (int s = 0; s < B_NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 5); }   // empty: the S product + 4 warps
    for (int g = 0; g < 2; ++g) {
      mbar_init(&B.acc[g], 1); mbar_init(&B.acc_empty[g], 4); mbar_init(&B.dfull[g], 4); mbar_init(&B.dempty[g], 1);
    }
    mbar_init(&B.w, 1); mbar_init(&B.done, 1);
    fence_barrier_init();
    prefetch_map(&mx); prefetch_map(&mdb); prefetch_map(&mw); prefetch_map(&mdx);
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    if (lane == 0) {
      mbar_expect_tx(&B.w, WT);
      tma_load_2d(su + B_O_W, &mw, &B.w, 0, 0);
      tma_load_2d(su + B_O_W + WB, &mw, &B.w, 64, 0);
      for (int i = 0; i < my; ++i) {
        const int s = i % B_NST, r0 = ((int)blockIdx.x + i * G) * TM;
        if (i >= B_NST) mbar_wait(&B.empty[s], ((i / B_NST) - 1) & 1);
        const uint32_t st = su + B_O_ST + s * B_STAGE;
        mbar_expect_tx(&B.full[s], B_STAGE);
        tma_load_2d(st, &mx, &B.full[s], 0, r0);
        tma_load_2d(st + XB, &mx, &B.full[s], 64, r0);
        tma_load_2d(st + XT, &mdb, &B.full[s], r0, 0);
        tma_load_2d(st + XT + DBB, &mdb, &B.full[s], r0 + 64, 0);
      }
    }
  } else if (warp == 1) {
    // d LN0 of tile i (into group i & 1's accumulator) is issued before S of tile i - 1, which waits for that tile's d'
    auto dln = [&](int i) {
      const int s = i % B_NST, g = i & 1;
      mbar_wait(&B.full[s], (i / B_NST) & 1);
      if (i >= 2) mbar_wait(&B.acc_empty[g], ((i >> 1) - 1) & 1);
      tc_fence_after();
      const uint32_t db = su + B_O_ST + s * B_STAGE + XT;
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < NP / 16; ++ks)                 // K step: 16 weight rows = 2048 bytes of the MN-major tiles
          umma_ss(tmem + g * DP, desc_mn128(db + ks * 2048, DBB), desc_mn128(su + B_O_W + ks * 2048, WB), B_ID_DLN, ks ? 1u : 0u);
        tc_commit(&B.acc[g]);
      }
      __syncwarp();
    };
    auto sprod = [&](int i) {
      const int s = i % B_NST, g = i & 1;
      mbar_wait(&B.dfull[g], (i >> 1) & 1);
      tc_fence_after();
      const uint32_t st = su + B_O_ST + s * B_STAGE;
      if (elect_one()) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {                        // d' = hi + lo
          const uint32_t dp = su + B_O_DS + (uint32_t)(g * 2 + h) * DBT;
#pragma unroll
          for (int ks = 0; ks < TM / 16; ++ks)               // K step: 16 rows; d' K-major (32 bytes within a 128-byte row)
            umma_ss(tmem + B_COL_S, desc_mn128(st + ks * 2048, XB), desc_k128(dp + (ks >> 2) * DBB) + (uint64_t)((ks & 3) * 2),
                    B_ID_S, (i | h | ks) ? 1u : 0u);
        }
        tc_commit(&B.empty[s]);
        tc_commit(&B.dempty[g]);
      }
      __syncwarp();
    };
    mbar_wait(&B.w, 0);
    if (my > 0) dln(0);
    for (int i = 0; i < my; ++i) {
      if (i + 1 < my) dln(i + 1);
      sprod(i);
    }
    if (elect_one()) tc_commit(&B.done);
    __syncwarp();
  } else {
    const int g = (warp - 2) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane;
    const uint32_t dsh = su + B_O_DS + (uint32_t)(g * 2) * DBT, dxs = su + B_O_DX + (uint32_t)(warp - 2) * 2 * DXS;
    int nst = 0;                                             // staged dx stores issued by this warp
    for (int i = g; i < my; i += 2) {
      const int s = i % B_NST, r0 = ((int)blockIdx.x + i * G) * TM;
      const uint32_t st = su + B_O_ST + s * B_STAGE, rr = opaque(r);
      mbar_wait(&B.full[s], (i / B_NST) & 1);
      const float k0 = row_shift(st, rr);
      const f2 nk = dup2(-k0);
      mbar_wait(&B.acc[g], (i >> 1) & 1);
      tc_fence_after();
      const uint32_t ta = tmem + (lb << 16) + g * DP;
      // pass 1, one read of the row and of d LN0: the shifted statistics, sum(dy) and sum(dy (x - k))
      f2 s1 = mk2(0.f, 0.f), s2 = s1, sd = s1, sdx = s1, u1 = s1, u2 = s1, ud = s1, udx = s1;   // two chains each
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        uint32_t v[32];
        tmem_ld32(ta + c * 32, v);
        uint4 xq[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) xq[j] = lds128v(st + xoff(rr, c * 4 + j));
        tmem_wait_ld();
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t xw[4] = {xq[j].x, xq[j].y, xq[j].z, xq[j].w};
#pragma unroll
          for (int k = 0; k < 4; k += 2) {
            const f2 t = add2(unpk(xw[k]), nk), d = mk2u(v[8 * j + 2 * k], v[8 * j + 2 * k + 1]);
            const f2 t2 = add2(unpk(xw[k + 1]), nk), d2 = mk2u(v[8 * j + 2 * k + 2], v[8 * j + 2 * k + 3]);
            s1 = add2(s1, t); s2 = fma2(t, t, s2); sd = add2(sd, d); sdx = fma2(d, t, sdx);
            u1 = add2(u1, t2); u2 = fma2(t2, t2, u2); ud = add2(ud, d2); udx = fma2(d2, t2, udx);
          }
        }
      }
      sd = add2(sd, ud); sdx = add2(sdx, udx);
      float mean, rstd, m1;
      finish_stats(add2(s1, u1), add2(s2, u2), k0, eps, mean, rstd, m1);
      const float mdy = (lo2(sd) + hi2(sd)) * (1.f / DP);
      // mean(dy xhat) = rstd (mean(dy (x - k)) - (mean - k) mean(dy))
      const float mdyx = rstd * fmaf(-m1, mdy, (lo2(sdx) + hi2(sdx)) * (1.f / DP));
      const f2 rs = dup2(rstd), nmr = dup2(-mean * rstd), nmdy = dup2(-mdy), nmdyx = dup2(-mdyx);
      // d' = dbias rstd for this row (column r of the dbias tile) as hi + lo: the S product's B operand
      if (i >= 2) mbar_wait(&B.dempty[g], ((i >> 1) - 1) & 1);   // the group's d' buffers: read by its previous S product
      {
        const uint32_t db = st + XT;
#pragma unroll 8
        for (int n = 0; n < NP; ++n) {
          const uint32_t o = doff((uint32_t)n, rr);
          const float d = __uint_as_float(lds16(db + o) << 16) * rstd;
          const uint32_t hb = pack_bf16(d, 0.f) & 0xffffu;
          sts16(dsh + o, hb);
          sts16(dsh + DBT + o, pack_bf16(d - __uint_as_float(hb << 16), 0.f) & 0xffffu);
        }
      }
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.dfull[g]);
      // pass 2: dx = rstd (dy - mean(dy) - xhat mean(dy xhat)), per 32 channels through the warp's staging slot and a TMA store
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        uint32_t v[32];
        tmem_ld32(ta + c * 32, v);
        uint4 xq[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) xq[j] = lds128v(st + xoff(rr, c * 4 + j));
        tmem_wait_ld();
        uint4 od[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t xw[4] = {xq[j].x, xq[j].y, xq[j].z, xq[j].w};
          uint32_t o[4];
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const f2 xh = fma2(unpk(xw[k]), rs, nmr);
            const f2 d = mk2u(v[8 * j + 2 * k], v[8 * j + 2 * k + 1]);
            o[k] = pk(mul2(fma2(xh, nmdyx, add2(d, nmdy)), rs));
          }
          od[j] = make_uint4(o[0], o[1], o[2], o[3]);
        }
        const uint32_t sb = dxs + (uint32_t)(nst & 1) * DXS;
        if (nst >= 2) {
          if (lane == 0) asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");   // the slot's previous store has read it
          __syncwarp();
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) sts128(sb + sw64(lane, j), od[j]);
        fence_proxy_async();
        __syncwarp();
        if (lane == 0) { tma_store_2d(&mdx, sb, c * 32, r0 + (int)lb); tma_store_commit(); }
        ++nst;
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) { mbar_arrive(&B.acc_empty[g]); mbar_arrive(&B.empty[s]); }
    }
    if (lane == 0) tma_store_wait0();
    if (g == 0) {                                            // this CTA's S partial: TMEM lane = channel, column = weight row
      mbar_wait(&B.done, 0);
      tc_fence_after();
      float* pp = part + (long)blockIdx.x * NP * DP + r;
#pragma unroll
      for (int c = 0; c < NP / 16; ++c) {
        uint32_t v[16];
        tmem_ld16(tmem + (lb << 16) + B_COL_S + c * 16, v);
        tmem_wait_ld();
#pragma unroll
        for (int k = 0; k < 16; ++k) pp[(c * 16 + k) * DP] = my > 0 ? __uint_as_float(v[k]) : 0.f;
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
#endif  // NP <= 64 && NCH == 1 && !TF32

// ================================================================================================ backward over weight-row chunks
// -DBWD2 (bo_hoist_bwd2_sm100): the backward for any number of weight rows (bf16 past 64, every fp32 / -DTF32 hoist), the weight rows
// in chunks of NP (bf16 64, tf32 32) streamed with the tile instead of resident:
//   d LN0 = sum_j dbias_j^T W'_j      (TMEM, accumulated over the chunks: one [128 rows][128] accumulator per epilogue group)
//   S_j  += x^T d'_j                   (TMEM, columns 256 + (j - s0) NP for the launch's S chunks [s0, s1): at most 256 columns)
// A launch walks chunks [c0, c1) of every tile; ``dx`` launches take every chunk for d LN0 (and the LayerNorm backward, d pair) and
// the first <= 256 weight rows of S; the rest of S takes S-only launches over their chunks (``hoist.py``: a weight-row range per
// launch, the pair read again).
// bf16 is the one-chunk kernel's arithmetic (S against the RAW pair and d' = dbias rstd as hi + lo; the host subtracts the channel
// mean). TF32: the group's row threads overwrite their rows with xhat (fp32) after the statistics, so S = xhat^T dbias (dW' itself, the
// arithmetic of the LN0-rows path: LN0 against the TF32 dbias, with no cancellation of a mean term), d = the dbias chunk copied to the
// K-major layout the S product's B operand needs (the chunk arrives MN-major for d LN0: an MN-major kind::tf32 operand sits in the
// 128-B swizzle with 32-B atoms); dx = rstd (dy - mean(dy) - xhat mean(dy xhat)) is written over xhat and stored by TMA from there.
// Rings: X (the pair tile, XST slots), D (a chunk: the dbias tile, and W' in dx launches; DST slots). Barriers: xfull / xempty (the
// MMA's commit after the tile's last product + the group's 4 warps), dfull_ / dempty_ (TMA; the MMA commit + the group's 4 warps),
// acc / acc_empty (d LN0, per group), wfull / wempty (the group's d' buffer: 4 warps / the S product's commit), done.
#ifdef BWD2
constexpr int X2T = TM * DP * ES, X2B = TM * 128;                 // pair tile [DP*ES/128 boxes][128 rows][128 B]: 32 / 64 KB
constexpr int X2_NBOX = DP * ES / 128;
constexpr int D2B = NP * 128;                     // one MN box of a chunk [NP][128 B]: dbias (rows) or W' (channels)
constexpr int D2T = (TM * ES / 128) * D2B;        // the dbias chunk [TM*ES/128 boxes][NP][128 B]
constexpr int W2T = X2_NBOX * D2B;                // the W' chunk [DP*ES/128 boxes][NP][128 B]
constexpr int D2S = D2T + W2T;                    // a D slot
#ifdef TF32
constexpr int WD2 = (TM * 4 / 128) * D2B;         // d: the dbias chunk K-major [4 boxes][NP][128 B]: 16 KB at NP 32
constexpr int NWD = 1;                            // one term
constexpr int DX2 = 0;                            // dx is staged in the pair tile itself
#else
constexpr int WD2 = D2T;                          // d' hi / lo: [2 boxes][NP][64 rows] each (the dbias tile's layout)
constexpr int NWD = 2;
constexpr int DX2 = 8 * 2 * (32 * 64);            // dx staging: per warp two [32 rows][32 channels] bf16 SW64 slots
#endif
constexpr int XST = 2, DST = 2;
constexpr int O2_X = 0, O2_D = O2_X + XST * X2T, O2_W = O2_D + DST * D2S, O2_DX = O2_W + 2 * NWD * WD2, O2_BAR = O2_DX + DX2;
constexpr int B2_SMEM = 1024 + O2_BAR + 512;
static_assert(B2_SMEM <= SMEM_MAX, "backward (chunks) shared memory");
static_assert(X2T % 1024 == 0 && D2S % 1024 == 0 && WD2 % 1024 == 0 && D2B % 1024 == 0, "1 KB alignment of the swizzled tiles");
constexpr int S2_COLS = 256;                      // S columns per launch
#ifdef TF32
static_assert(NP % 8 == 0 && NP <= 32, "tf32 chunk: at most 32 weight rows (shared memory)");
constexpr uint32_t ID2_DLN = idesc_tf32(TM, DP, 1, 1);   // A = dbias chunk (MN-major), B = W' chunk (MN-major)
constexpr uint32_t ID2_S = idesc_tf32(DP, NP, 1, 0);     // A = xhat^T (pair tile MN-major), B = d (K-major)
constexpr int K2_DLN = NP / 8, K2_S = TM / 8;            // K steps (8 rows / 8 weight rows = 1 KB of an MN operand)
#else
static_assert(NP % 16 == 0 && NP <= 64, "bf16 chunk: at most 64 weight rows (shared memory)");
constexpr uint32_t ID2_DLN = idesc_bf16(TM, DP, 1, 1);
constexpr uint32_t ID2_S = idesc_bf16(DP, NP, 1, 0);
constexpr int K2_DLN = NP / 16, K2_S = TM / 16;          // K steps (16 rows = 2 KB of an MN operand)
#endif
constexpr int K2_STEP = ES == 2 ? 2048 : 1024;           // bytes per K step of an MN operand: 16 bf16 / 8 tf32 rows of 128 B

struct B2Bars {
  uint64_t xfull[XST], xempty[XST], dfull_[DST], dempty_[DST], acc[2], acc_empty[2], wfull[2], wempty[2], done;
  uint32_t tmem;
};

#ifdef TF32
// byte offset of fp32 (r, k) -- k = 4 c .. 4 c + 3 as one 16-byte chunk c (0..31) -- in an MN tile of 32-element atoms [box][rows][128 B]
// in the 128-B swizzle with 32-B atomicity: 32-B granule j of 128-B row r sits at granule j ^ (r & 3)
DEVI uint32_t x32off(uint32_t r, uint32_t c) { return (c >> 3) * X2B + r * 128u + ((((c & 7u) >> 1) ^ (r & 3u)) << 5) + (c & 1u) * 16u; }
// (weight row n, pair row r) in the dbias chunk [4 boxes of 32 rows][NP][128 B], 32-B atoms
DEVI uint32_t db32off(uint32_t n, uint32_t r) { return (r >> 5) * D2B + n * 128u + ((((r & 31u) >> 3) ^ (n & 3u)) << 5) + (r & 7u) * 4u; }
// (n, r) in d [4 boxes of 32 rows][NP][128 B], K-major SW128
DEVI uint32_t dk32off(uint32_t n, uint32_t r) { return (r >> 5) * D2B + n * 128u + ((((r & 31u) >> 2) ^ (n & 7u))) * 16u + (r & 3u) * 4u; }
DEVI float lds32f(uint32_t a) { float v; asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(a) : "memory"); return v; }
DEVI void sts32f(uint32_t a, float v) { asm volatile("st.shared.f32 [%0], %1;" :: "r"(a), "f"(v) : "memory"); }
#endif

DEVI void umma2(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t acc) {
#ifdef TF32
  umma_ss_tf32(d, a, b, id, acc);
#else
  umma_ss(d, a, b, id, acc);
#endif
}
DEVI uint64_t desc_mn2(uint32_t saddr, uint32_t lbo) {
#ifdef TF32
  return desc_mn32b(saddr, lbo);
#else
  return desc_mn128(saddr, lbo);
#endif
}

// mx: pair rows (bf16: box (64, 128) SW128; tf32: box (32, 128) 128-B swizzle with 32-B atoms); mdb: dbias_all [N][R], box (bf16 64 /
// tf32 32 rows, NP) (rows past N zero-filled); mw: W' [N][128], box (64 / 32 channels, NP); mdx: d pair [R][128] (bf16: box (32, 32)
// SW64 from the staging slots; tf32: box (32, 32) 32-B-atom swizzle from the pair tile); part: S partials [grid][(s1 - s0) NP][128]
// fp32. Chunks [c0, c1) of NP weight rows; S for chunks [s0, s1) (within [c0, c1)); dx != 0: d LN0 over the chunks and d pair.
extern "C" __global__ void __launch_bounds__(THREADS, 1)
bo_hoist_bwd2_sm100(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mdb, const __grid_constant__ CUtensorMap mw,
                    const __grid_constant__ CUtensorMap mdx, float* __restrict__ part, int R, int c0, int c1, int s0, int s1, int dx,
                    int eps_bits) {
  extern __shared__ __align__(1024) uint8_t smraw[];
  uint8_t* sm = smraw + ((1024u - (smem_u32(smraw) & 1023u)) & 1023u);
  const uint32_t su = smem_u32(sm);
  B2Bars& B = *reinterpret_cast<B2Bars*>(sm + O2_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int T = R / TM, G = gridDim.x, my = (T > (int)blockIdx.x) ? (T - (int)blockIdx.x + G - 1) / G : 0;
  const int nc = c1 - c0, ns = s1 - s0;
  const float eps = __int_as_float(eps_bits);

  if (tid == 0) {
    for (int s = 0; s < XST; ++s) { mbar_init(&B.xfull[s], 1); mbar_init(&B.xempty[s], 5); }
    for (int s = 0; s < DST; ++s) { mbar_init(&B.dfull_[s], 1); mbar_init(&B.dempty_[s], 5); }
    for (int g = 0; g < 2; ++g) {
      mbar_init(&B.acc[g], 1); mbar_init(&B.acc_empty[g], 4); mbar_init(&B.wfull[g], 4); mbar_init(&B.wempty[g], 1);
    }
    mbar_init(&B.done, 1);
    fence_barrier_init();
    prefetch_map(&mx); prefetch_map(&mdb); prefetch_map(&mw); prefetch_map(&mdx);
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    if (lane == 0) {
      int q = 0;                                             // running D-ring index
      for (int i = 0; i < my; ++i) {
        const int xs = i % XST, r0 = ((int)blockIdx.x + i * G) * TM;
        if (i >= XST) mbar_wait(&B.xempty[xs], ((i / XST) - 1) & 1);
        const uint32_t xt = su + O2_X + xs * X2T;
        mbar_expect_tx(&B.xfull[xs], X2T);
        for (int b = 0; b < X2_NBOX; ++b) tma_load_2d(xt + b * X2B, &mx, &B.xfull[xs], b * (128 / ES), r0);
        for (int j = c0; j < c1; ++j, ++q) {
          const int ds = q % DST;
          if (q >= DST) mbar_wait(&B.dempty_[ds], ((q / DST) - 1) & 1);
          const uint32_t dt = su + O2_D + ds * D2S;
          mbar_expect_tx(&B.dfull_[ds], D2T + (dx ? W2T : 0));
          for (int b = 0; b < TM * ES / 128; ++b) tma_load_2d(dt + b * D2B, &mdb, &B.dfull_[ds], r0 + b * (128 / ES), j * NP);
          if (dx)
            for (int b = 0; b < X2_NBOX; ++b) tma_load_2d(dt + D2T + b * D2B, &mw, &B.dfull_[ds], b * (128 / ES), j * NP);
        }
      }
    }
  } else if (warp == 1) {
    int q = 0;
    for (int i = 0; i < my; ++i) {
      const int g = i & 1, k = i >> 1, xs = i % XST;
      const uint32_t xt = su + O2_X + xs * X2T;
      mbar_wait(&B.xfull[xs], (i / XST) & 1);
      if (dx && k >= 1) mbar_wait(&B.acc_empty[g], (k - 1) & 1);
      for (int j = c0; j < c1; ++j, ++q) {
        const int ds = q % DST;
        mbar_wait(&B.dfull_[ds], (q / DST) & 1);
        tc_fence_after();
        const uint32_t dt = su + O2_D + ds * D2S;
        if (dx) {
          if (elect_one()) {
#pragma unroll
            for (int ks = 0; ks < K2_DLN; ++ks)
              umma2(tmem + g * DP, desc_mn2(dt + ks * K2_STEP, D2B), desc_mn2(dt + D2T + ks * K2_STEP, D2B), ID2_DLN, (j > c0 || ks) ? 1u : 0u);
          }
          __syncwarp();
        }
        if (j >= s0 && j < s1) {
          const int v = k * ns + (j - s0);                   // the group's running S chunk
          mbar_wait(&B.wfull[g], v & 1);
          tc_fence_after();
          if (elect_one()) {
#pragma unroll
            for (int h = 0; h < NWD; ++h) {
              const uint32_t dp = su + O2_W + (uint32_t)(g * NWD + h) * WD2;
#pragma unroll
              for (int ks = 0; ks < K2_S; ++ks)              // K step: 16 (8) rows; d' / d K-major (32 bytes within a 128-byte row)
                umma2(tmem + 2 * DP + (j - s0) * NP, desc_mn2(xt + ks * K2_STEP, X2B), desc_k128(dp + (ks >> 2) * D2B) + (uint64_t)((ks & 3) * 2),
                      ID2_S, (i | h | ks) ? 1u : 0u);
            }
            tc_commit(&B.wempty[g]);
          }
          __syncwarp();
        }
        if (elect_one()) tc_commit(&B.dempty_[ds]);
        __syncwarp();
      }
      if (elect_one()) {
        if (dx) tc_commit(&B.acc[g]);
        tc_commit(&B.xempty[xs]);
      }
      __syncwarp();
    }
    if (elect_one()) tc_commit(&B.done);
    __syncwarp();
  } else {
    const int g = (warp - 2) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane;
    const uint32_t wd = su + O2_W + (uint32_t)(g * NWD) * WD2;
#ifndef TF32
    const uint32_t dxs = su + O2_DX + (uint32_t)(warp - 2) * 2 * (32 * 64);
    int nst = 0;                                             // staged dx stores issued by this warp
#endif
    for (int i = g; i < my; i += 2) {
      const int k = i >> 1, xs = i % XST, r0 = ((int)blockIdx.x + i * G) * TM;
      const uint32_t xt = su + O2_X + xs * X2T, rr = opaque(r);
      mbar_wait(&B.xfull[xs], (i / XST) & 1);
      // ---- the row statistics (shifted one-pass sums: the forward's arithmetic)
#ifdef TF32
      const float k0 = __uint_as_float(lds128v(xt + x32off(rr, 0)).x);
      const f2 nk = dup2(-k0);
      f2 q1 = mk2(0.f, 0.f), q2 = q1, q3 = q1, q4 = q1;
#pragma unroll 4
      for (int c = 0; c < 32; ++c) {
        const uint4 u = lds128v(xt + x32off(rr, c));
        const f2 t0 = add2(mk2u(u.x, u.y), nk), t1 = add2(mk2u(u.z, u.w), nk);
        q1 = add2(q1, t0); q2 = fma2(t0, t0, q2); q3 = add2(q3, t1); q4 = fma2(t1, t1, q4);
      }
      float mean, rstd, m1;
      finish_stats(add2(q1, q3), add2(q2, q4), k0, eps, mean, rstd, m1);
      const f2 rs = dup2(rstd), nmr = dup2(-mean * rstd);
#pragma unroll 4
      for (int c = 0; c < 32; ++c) {                          // the row becomes xhat rounded to the nearest tf32 (cvt.rna): S's operand
        const uint32_t ad = xt + x32off(rr, c);              // and the dx pass's input (the MMA would otherwise drop the low bits:
        const uint4 u = lds128v(ad);                          // a one-sided error that S's sum over every pair row accumulates)
        const f2 y0 = fma2(mk2u(u.x, u.y), rs, nmr), y1 = fma2(mk2u(u.z, u.w), rs, nmr);
        sts128(ad, make_uint4(tf32_rna(lo2(y0)), tf32_rna(hi2(y0)), tf32_rna(lo2(y1)), tf32_rna(hi2(y1))));
      }
#else
      const float k0 = row_shift(xt, rr);
      const f2 nk = dup2(-k0);
      f2 q1 = mk2(0.f, 0.f), q2 = q1, q3 = q1, q4 = q1;
#pragma unroll
      for (int q = 0; q < 16; q += 2) {
        acc_stats(lds128v(xt + xoff(rr, q)), nk, q1, q2);
        acc_stats(lds128v(xt + xoff(rr, q + 1)), nk, q3, q4);
      }
      float mean, rstd, m1;
      finish_stats(add2(q1, q3), add2(q2, q4), k0, eps, mean, rstd, m1);
      const f2 rs = dup2(rstd), nmr = dup2(-mean * rstd);
#endif
      // ---- the chunks: d' (bf16: dbias rstd as hi + lo) / d (tf32: the dbias copy) for the S product; release the D slot
      for (int j = c0; j < c1; ++j) {
        const int q = i * nc + (j - c0), ds = q % DST;
        // The slot's previous use (q - DST) may be the other group's chunk: wait for its release before the parity wait on dfull_,
        // which would otherwise pass on that use's phase while its fill is still in flight (a whole ring ahead of the fills)
        if (q >= DST) mbar_wait(&B.dempty_[ds], ((q / DST) - 1) & 1);
        mbar_wait(&B.dfull_[ds], (q / DST) & 1);
        if (j >= s0 && j < s1) {
          const int v = k * ns + (j - s0);
          if (v >= 1) mbar_wait(&B.wempty[g], (v - 1) & 1);  // the group's previous S product has read d
          const uint32_t db = su + O2_D + ds * D2S;
#ifdef TF32
#pragma unroll 8
          for (int n = 0; n < NP; ++n)                        // rounded to the nearest tf32, as xhat
            sts32f(wd + dk32off((uint32_t)n, rr), __uint_as_float(tf32_rna(lds32f(db + db32off((uint32_t)n, rr)))));
#else
#pragma unroll 8
          for (int n = 0; n < NP; ++n) {
            const uint32_t o = doff((uint32_t)n, rr);
            const float d = __uint_as_float(lds16(db + o) << 16) * rstd;
            const uint32_t hb = pack_bf16(d, 0.f) & 0xffffu;
            sts16(wd + o, hb);
            sts16(wd + WD2 + o, pack_bf16(d - __uint_as_float(hb << 16), 0.f) & 0xffffu);
          }
#endif
          fence_proxy_async();
          __syncwarp();
          if (lane == 0) mbar_arrive(&B.wfull[g]);
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.dempty_[ds]);
      }
      if (dx) {
        // ---- the LayerNorm backward from d LN0 (TMEM, every chunk summed)
        mbar_wait(&B.acc[g], k & 1);
        tc_fence_after();
        const uint32_t ta = tmem + (lb << 16) + g * DP;
        f2 sd = mk2(0.f, 0.f), sdx = sd, ud = sd, udx = sd;
#pragma unroll
        for (int c = 0; c < 4; ++c) {                         // pass 1: sum(dy), sum(dy t) (t = xhat for tf32, x - k for bf16)
          uint32_t vv[32];
          tmem_ld32(ta + c * 32, vv);
          uint4 xq[8];
#ifdef TF32
#pragma unroll
          for (int e = 0; e < 8; ++e) xq[e] = lds128v(xt + x32off(rr, c * 8 + e));
          tmem_wait_ld();
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const f2 t0 = mk2u(xq[e].x, xq[e].y), t1 = mk2u(xq[e].z, xq[e].w);
            const f2 d0 = mk2u(vv[4 * e], vv[4 * e + 1]), d1 = mk2u(vv[4 * e + 2], vv[4 * e + 3]);
            sd = add2(sd, d0); sdx = fma2(d0, t0, sdx); ud = add2(ud, d1); udx = fma2(d1, t1, udx);
          }
#else
#pragma unroll
          for (int e = 0; e < 4; ++e) xq[e] = lds128v(xt + xoff(rr, c * 4 + e));
          tmem_wait_ld();
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const uint32_t xw[4] = {xq[e].x, xq[e].y, xq[e].z, xq[e].w};
#pragma unroll
            for (int kk = 0; kk < 4; kk += 2) {
              const f2 t = add2(unpk(xw[kk]), nk), d = mk2u(vv[8 * e + 2 * kk], vv[8 * e + 2 * kk + 1]);
              const f2 t2 = add2(unpk(xw[kk + 1]), nk), d2 = mk2u(vv[8 * e + 2 * kk + 2], vv[8 * e + 2 * kk + 3]);
              sd = add2(sd, d); sdx = fma2(d, t, sdx); ud = add2(ud, d2); udx = fma2(d2, t2, udx);
            }
          }
#endif
        }
        sd = add2(sd, ud); sdx = add2(sdx, udx);
        const float mdy = (lo2(sd) + hi2(sd)) * (1.f / DP);
#ifdef TF32
        const float mdyx = (lo2(sdx) + hi2(sdx)) * (1.f / DP);                 // mean(dy xhat)
#else
        const float mdyx = rstd * fmaf(-m1, mdy, (lo2(sdx) + hi2(sdx)) * (1.f / DP));   // rstd (mean(dy (x - k)) - (mean - k) mean(dy))
#endif
        const f2 nmdy = dup2(-mdy), nmdyx = dup2(-mdyx);
        // pass 2: dx = rstd (dy - mean(dy) - xhat mean(dy xhat))
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          uint32_t vv[32];
          tmem_ld32(ta + c * 32, vv);
#ifdef TF32
          uint4 xq[8];
#pragma unroll
          for (int e = 0; e < 8; ++e) xq[e] = lds128v(xt + x32off(rr, c * 8 + e));
          tmem_wait_ld();
#pragma unroll
          for (int e = 0; e < 8; ++e) {                     // written over xhat (every MMA reading this tile has completed: acc)
            const f2 o0 = mul2(fma2(mk2u(xq[e].x, xq[e].y), nmdyx, add2(mk2u(vv[4 * e], vv[4 * e + 1]), nmdy)), rs);
            const f2 o1 = mul2(fma2(mk2u(xq[e].z, xq[e].w), nmdyx, add2(mk2u(vv[4 * e + 2], vv[4 * e + 3]), nmdy)), rs);
            sts128(xt + x32off(rr, c * 8 + e), make_uint4(__float_as_uint(lo2(o0)), __float_as_uint(hi2(o0)), __float_as_uint(lo2(o1)), __float_as_uint(hi2(o1))));
          }
#else
          uint4 xq[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) xq[e] = lds128v(xt + xoff(rr, c * 4 + e));
          tmem_wait_ld();
          uint4 od[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const uint32_t xw[4] = {xq[e].x, xq[e].y, xq[e].z, xq[e].w};
            uint32_t o[4];
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
              const f2 xh = fma2(unpk(xw[kk]), rs, nmr);
              const f2 d = mk2u(vv[8 * e + 2 * kk], vv[8 * e + 2 * kk + 1]);
              o[kk] = pk(mul2(fma2(xh, nmdyx, add2(d, nmdy)), rs));
            }
            od[e] = make_uint4(o[0], o[1], o[2], o[3]);
          }
          const uint32_t sb = dxs + (uint32_t)(nst & 1) * (32 * 64);
          if (nst >= 2) {
            if (lane == 0) asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");   // the slot's previous store has read it
            __syncwarp();
          }
#pragma unroll
          for (int e = 0; e < 4; ++e) sts128(sb + sw64(lane, e), od[e]);
          fence_proxy_async();
          __syncwarp();
          if (lane == 0) { tma_store_2d(&mdx, sb, c * 32, r0 + (int)lb); tma_store_commit(); }
          ++nst;
#endif
        }
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.acc_empty[g]);
#ifdef TF32
        fence_proxy_async();                                  // dx in the pair tile -> the TMA stores of this warp's 32 rows
        __syncwarp();
        if (lane == 0) {
          for (int b = 0; b < 4; ++b) tma_store_2d(&mdx, xt + b * X2B + lb * 128, b * 32, r0 + (int)lb);
          tma_store_commit();
          tma_store_wait_read0();                             // the X slot is free once the stores have read it
        }
#endif
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.xempty[xs]);
    }
    if (lane == 0) tma_store_wait0();
    if (g == 0) {                                            // this CTA's S partial: TMEM lane = channel, column = weight row
      mbar_wait(&B.done, 0);
      tc_fence_after();
      float* pp = part + (long)blockIdx.x * ns * NP * DP + r;
      for (int c = 0; c < ns * NP / 16; ++c) {
        uint32_t v[16];
        tmem_ld16(tmem + (lb << 16) + 2 * DP + c * 16, v);
        tmem_wait_ld();
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) pp[(c * 16 + kk) * DP] = my > 0 ? __uint_as_float(v[kk]) : 0.f;
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
#endif  // BWD2
