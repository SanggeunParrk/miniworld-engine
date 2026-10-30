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
// More weight rows and fp32 (2026-10-29): -DTF32 is the forward's fp32 form up to 64 weight rows (the tile normalised in place, then
// kind::tf32); fp32 past 64 rows takes bo_hoist_fwdx_sm100 (-DFWDX: CTA pairs, W' resident split between them, staged TMA stores);
// bf16 past 32 weight rows bo_hoist_fwdt_sm100 (-DFWDT: the weight rows on M, staged TMA stores); the backward past 64 bf16 rows, and
// every fp32 backward, bo_hoist_bwd3_sm100 (-DBWD3: the tile normalised once, no per-chunk work of the row threads).
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
// forward: one chunk of NP weight rows per launch, W' resident (NCH: the loops below are written per chunk)
constexpr int NCH = 1;
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
#ifdef TF32
constexpr uint32_t F_IDESC = idesc_tf32(TM, NP);
#else
constexpr uint32_t F_IDESC = idesc_bf16(TM, NP);
#endif
constexpr uint32_t F_TCOLS = 4 * NP <= 32 ? 32 : 4 * NP <= 64 ? 64 : 4 * NP <= 128 ? 128 : 4 * NP <= 256 ? 256 : 512;
#if NP <= 64 && !defined(TF32)  // ------------------------------------------- backward (bf16, NP <= 64)
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
#if NP <= 64 && !defined(TF32)
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
#endif  // NP <= 64 && !TF32


// ================================================================================================ forward, weight rows on M (bf16)
// -DFWDT=<chunks> (bo_hoist_fwdt_sm100): the bf16 forward for more than 64 weight rows. Per tile and chunk j of 128 weight rows the
// product is the TRANSPOSED tile bias^T_j [128 weight rows][128 pair rows] = W'_j x^T (A = the resident W' chunk, B = the raw pair
// tile, both K-major), so the drain thread is a weight row n (TMEM lane) holding 32 consecutive pair rows per load: the bias row it
// writes is contiguous in [N][R]. The LayerNorm stays in the epilogue, bias[n][r] = rstd_r acc[n][r] + (-mean_r rstd_r) s_n (the
// arithmetic of bo_hoist_fwd_sm100, element for element): the group's row threads write (rstd, -mean rstd) of their rows to shared
// memory, the drain reads them as broadcasts, and each warp stages its [32 weight rows][64 pair rows] bf16 in a 128-B-swizzled slot
// (two per warp) for a TMA store: whole 128-B lines per weight row (bo_hoist_fwd_sm100 stores a 2-row strip of 64-B pieces per warp
// instruction; 64-B pieces through staging were still 1.8x slower at R = 7.1 M rows, where the bias leaves L2 for DRAM).
// s_n (the row sums of the bf16 W', summed in the order bo_hoist_fwd_sm100 sums them) lives in registers: one per chunk per thread.
// Warps as above; MMA order as bo_hoist_fwd_sm100 (tile pairs, chunk-major within a pair); four TMEM accumulators of 128 columns.
// Shared memory is used to the last KB (W' FWDT x 32 KB): the dynamic base must be 1024-aligned (it is with no static shared
// memory; trapped otherwise).
#ifdef FWDT
constexpr int T_NCH = FWDT;
static_assert(T_NCH >= 1 && T_NCH <= 3, "chunks of 128 weight rows per launch");
constexpr int T_XT = TM * DP * 2, T_XB = TM * 128;          // pair tile [2 boxes][128 rows][128 B] SW128
constexpr int T_WT = 128 * DP * 2, T_WB = 128 * 128;        // W' chunk [2 boxes][128 weight rows][128 B] SW128
constexpr int T_STG = 32 * 64 * 2;                          // a staging slot: [32 weight rows][64 pair rows] bf16, SW128 (4 KB)
constexpr int T_O_W = 0, T_O_SN = T_O_W + T_NCH * T_WT;     // W' | stats [2 groups][rstd, -mean rstd][128] | staging [8 warps][2]
constexpr int T_O_STG = T_O_SN + 2 * 2 * 128 * 4, T_O_ST = T_O_STG + 8 * 2 * T_STG;
constexpr int T_FIT = (SMEM_MAX - T_O_ST - 512) / T_XT;
constexpr int T_NST = T_FIT > 6 ? 6 : T_FIT;
constexpr int T_O_BAR = T_O_ST + T_NST * T_XT, T_SMEM = T_O_BAR + 512;
static_assert(T_NST >= 2 && T_SMEM <= SMEM_MAX && T_O_ST % 1024 == 0, "forward (weight rows on M) shared memory");
constexpr uint32_t T_IDESC = idesc_bf16(128, TM);           // M = 128 weight rows, N = 128 pair rows

struct TBars { uint64_t full[T_NST], empty[T_NST], acc[4], acc_empty[4], w; uint32_t tmem; };

// mx: pair rows [R][128], box (64, 128) SW128; mw: W' [N][128] (this launch's rows), box (64, 128) SW128 (rows past N zero-filled);
// mo: bias_all^T [N][R] bf16, box (32 pair rows, 32 weight rows) SW64 (rows past N clipped); wf: W' (s_n); R a multiple of 128
extern "C" __global__ void __launch_bounds__(THREADS, 1)
bo_hoist_fwdt_sm100(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mw, const __grid_constant__ CUtensorMap mo,
                    const void* __restrict__ wfv, int R, int N, int eps_bits) {
  extern __shared__ __align__(1024) uint8_t smraw[];
  if (smem_u32(smraw) & 1023u) __trap();                     // the layout uses the whole 227 KB: no room to realign
  uint8_t* sm = smraw;
  const uint32_t su = smem_u32(sm);
  TBars& B = *reinterpret_cast<TBars*>(sm + T_O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int T = R / TM, G = gridDim.x, my = (T > (int)blockIdx.x) ? (T - (int)blockIdx.x + G - 1) / G : 0;
  const float eps = __int_as_float(eps_bits);

  if (tid == 0) {
    for (int s = 0; s < T_NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 5); }   // empty: the MMA + the 4 warps
    for (int a = 0; a < 4; ++a) { mbar_init(&B.acc[a], 1); mbar_init(&B.acc_empty[a], 4); }
    mbar_init(&B.w, 1);
    fence_barrier_init();
    prefetch_map(&mx); prefetch_map(&mw); prefetch_map(&mo);
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    if (lane == 0) {
      mbar_expect_tx(&B.w, T_NCH * T_WT);
      for (int j = 0; j < T_NCH; ++j)
        for (int b = 0; b < 2; ++b) tma_load_2d(su + T_O_W + j * T_WT + b * T_WB, &mw, &B.w, b * 64, j * 128);
      for (int i = 0; i < my; ++i) {
        const int s = i % T_NST, r0 = ((int)blockIdx.x + i * G) * TM;
        if (i >= T_NST) mbar_wait(&B.empty[s], ((i / T_NST) - 1) & 1);
        const uint32_t st = su + T_O_ST + s * T_XT;
        mbar_expect_tx(&B.full[s], T_XT);
        tma_load_2d(st, &mx, &B.full[s], 0, r0);
        tma_load_2d(st + T_XB, &mx, &B.full[s], 64, r0);
      }
    }
  } else if (warp == 1) {
    mbar_wait(&B.w, 0);
    for (int p = 0; p < my; p += 2)
      for (int j = 0; j < T_NCH; ++j)
        for (int t = p; t < p + 2 && t < my; ++t) {
          const int s = t % T_NST, g = t & 1, u = (t >> 1) * T_NCH + j, a = 2 * g + (u & 1);
          if (j == 0) mbar_wait(&B.full[s], (t / T_NST) & 1);
          if (u >= 2) mbar_wait(&B.acc_empty[a], ((u >> 1) - 1) & 1);
          tc_fence_after();
          const uint32_t st = su + T_O_ST + s * T_XT, wj = su + T_O_W + j * T_WT;
          if (elect_one()) {
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) {                 // K step: 32 bytes within a 128-byte row (16 channels)
              const uint32_t bx = (uint32_t)(ks >> 2), ko = (uint32_t)(ks & 3) * 2;
              umma_ss(tmem + a * 128, desc_k128(wj + bx * T_WB) + ko, desc_k128(st + bx * T_XB) + ko, T_IDESC, ks ? 1u : 0u);
            }
            tc_commit(&B.acc[a]);
            if (j == T_NCH - 1) tc_commit(&B.empty[s]);
          }
          __syncwarp();
        }
  } else {
    const int g = (warp - 2) >> 2, we = warp - 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane;
    float* rsg = reinterpret_cast<float*>(sm + T_O_SN) + g * 256;   // [rstd | -mean rstd] of the group's current tile
    const uint32_t rs_a = su + T_O_SN + (uint32_t)g * 1024, stg = su + T_O_STG + (uint32_t)we * 2 * T_STG;
    // s_n for this thread's weight row of each chunk, summed as bo_hoist_fwd_sm100 sums it (0 past N)
    float snr[T_NCH];
    {
      const uint4* wf = reinterpret_cast<const uint4*>(wfv);
#pragma unroll
      for (int j = 0; j < T_NCH; ++j) {
        const int n = j * 128 + (int)r;
        float s = 0.f;
        if (n < N) {
          uint4 q[16];
#pragma unroll
          for (int e = 0; e < 16; ++e) q[e] = __ldg(wf + (long)n * 16 + e);
#pragma unroll
          for (int e = 0; e < 16; ++e) {
            const uint32_t w4[4] = {q[e].x, q[e].y, q[e].z, q[e].w};
#pragma unroll
            for (int h = 0; h < 4; ++h) { s += bf16lo(w4[h]); s += bf16hi(w4[h]); }
          }
        }
        snr[j] = s;
      }
    }
    int nst = 0;                                             // TMA stores issued by this warp
    for (int i = g; i < my; i += 2) {
      const int s = i % T_NST, r0 = ((int)blockIdx.x + i * G) * TM;
      mbar_wait(&B.full[s], (i / T_NST) & 1);
      const uint32_t xt = su + T_O_ST + s * T_XT, rr = opaque(r);
      const float k0 = row_shift(xt, rr);
      f2 s1 = mk2(0.f, 0.f), s2 = s1, s3 = s1, s4 = s1;
#pragma unroll
      for (int q = 0; q < 16; q += 2) {
        acc_stats(lds128v(xt + xoff(rr, q)), dup2(-k0), s1, s2);
        acc_stats(lds128v(xt + xoff(rr, q + 1)), dup2(-k0), s3, s4);
      }
      float mean, rstd, m1;
      finish_stats(add2(s1, s3), add2(s2, s4), k0, eps, mean, rstd, m1);
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.empty[s]);
      named_bar_sync(1 + g, 128);                            // the group's previous tile is drained: its statistics are free
      rsg[rr] = rstd;
      rsg[128 + rr] = -mean * rstd;
      named_bar_sync(1 + g, 128);                            // every row's statistics written
      for (int j = 0; j < T_NCH; ++j) {
        const int u = (i >> 1) * T_NCH + j, a = 2 * g + (u & 1);
        mbar_wait(&B.acc[a], (u >> 1) & 1);
        tc_fence_after();
        const f2 sn = dup2(snr[j]);
#pragma unroll 1
        for (int c = 0; c < 4 && j * 128 + (int)lb < N; ++c) {   // 32 pair rows per load (a warp of padding rows skips)
          uint32_t v[32];
          tmem_ld32(tmem + (lb << 16) + a * 128 + c * 32, v);
          tmem_wait_ld();
          uint32_t o[16];
#pragma unroll
          for (int p = 0; p < 8; ++p) {                      // rows c * 32 + 4 p .. + 3: (rstd, -mean rstd) as broadcasts
            const uint4 rs4 = lds128(rs_a + (uint32_t)(c * 32 + 4 * p) * 4), nm4 = lds128(rs_a + 512 + (uint32_t)(c * 32 + 4 * p) * 4);
            o[2 * p] = pk(fma2(mk2u(v[4 * p], v[4 * p + 1]), mk2u(rs4.x, rs4.y), mul2(mk2u(nm4.x, nm4.y), sn)));
            o[2 * p + 1] = pk(fma2(mk2u(v[4 * p + 2], v[4 * p + 3]), mk2u(rs4.z, rs4.w), mul2(mk2u(nm4.z, nm4.w), sn)));
          }
          // stage the 32 rows into their half of the slot; every second load the slot ([32 weight rows][64 pair rows], 128 B per
          // weight row: whole lines -- 64-B row pieces halved the DRAM write rate at large R) leaves by one TMA store
          const uint32_t sb = stg + (uint32_t)(nst & 1) * T_STG;
          if ((c & 1) == 0 && nst >= 2) {
            if (lane == 0) asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");   // the slot's previous store has read it
            __syncwarp();
          }
#pragma unroll
          for (int q = 0; q < 4; ++q) sts128(sb + sw128(lane, (c & 1) * 4 + q), make_uint4(o[4 * q], o[4 * q + 1], o[4 * q + 2], o[4 * q + 3]));
          if (c & 1) {
            fence_proxy_async();
            __syncwarp();
            if (lane == 0) { tma_store_2d(&mo, sb, r0 + (c - 1) * 32, j * 128 + (int)lb); tma_store_commit(); }
            ++nst;
          }
        }
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.acc_empty[a]);
      }
    }
    if (lane == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}
#endif  // FWDT

// ================================================================================================ forward, fp32, CTA pairs
// -DFWDX -DTF32 (bo_hoist_fwdx_sm100): the fp32 (TF32) forward for more than 128 weight rows in ONE pass over the pair. The fp32 W'
// (512 B per weight row) does not fit beside the 64 KB fp32 tiles of one CTA (bo_hoist_fwd_sm100 -DTF32 -DNCH read the pair again per
// 128-row launch), and streaming it from L2 costs each SM's TMA intake more than the pair itself (measured: 30 GB/s per SM shared by
// the tiles and W', 3.3 ms at N384 R 7.08 M). Here a cluster of two CTAs keeps W' RESIDENT, split between them: products of M 256
// (the two CTAs' tiles) x N 64 (a chunk of 64 weight rows) on cta_group::2, whose B operand is split by N -- each CTA holds 32 rows of
// every chunk (16 KB fp32), so 384 rows take 96 KB per CTA beside two tile slots, 640 rows 160 KB beside one.
// The tile is the A operand in TENSOR memory: the normaliser warps (a thread = a pair row = a TMEM lane) compute the row statistics
// (shifted one-pass sums: bo_hoist_fwd_sm100's arithmetic), write LN0 rounded to the nearest tf32 (cvt.rna, as the LayerNorm rows +
// TF32 GEMM path rounds its operand) into A[i & 1] by tcgen05.st, and load the next tile into the shared-memory slot they have read.
// Pair rows on M: the drain thread is a pair row, each warp store one 128-B line of bias_all^T [N][R] (32 consecutive pair rows of
// one weight row); TMEM columns 0-31 of an accumulator are the leader's W' rows of the chunk, 32-63 the peer's.
// Items: the cluster's tile pairs k = cluster + i clusters (tiles 2 k and 2 k + 1, one per CTA; an odd last tile pairs with zero
// rows that are not stored). Warps (448 threads): 0 TMA (W', once), 1 MMA (the leader) and TMEM, 2-5 the normaliser, 6-9 / 10-13 two
// drain groups (chunk u = i nch + j of the CTA's sequence goes to group u & 1; four 64-column accumulators, a = u & 3).
// Barriers: xfull[s] (TMA, each CTA), w (TMA, each CTA) / wpeer (leader: the peer's W' landed), aready[b] (leader, 8 = 4 warps x 2
// CTAs, release / acquire at cluster scope) / aempty[b] (multicast commit), acc[a] (multicast commit) / acc_empty[a] (leader, 8).
#ifdef FWDX
static_assert(ES == 4, "FWDX is the fp32 (TF32) forward: build with -DTF32");
constexpr int XST = 1;                                      // tile slots (the normaliser reads a tile into TMEM at once)
constexpr int X_THREADS = 448;
constexpr int X_WC = 64, X_WH = 32;                         // weight rows per chunk (the products' N) / per CTA
constexpr int X_XB = TM * 128, X_XT = 4 * X_XB;             // tile [4 boxes of 32 channels][128 rows][128 B] SW128: 64 KB
constexpr int X_WB = X_WH * 128, X_WT = 4 * X_WB;           // this CTA's half of a W' chunk [4 boxes][32 rows][128 B] SW128: 16 KB
constexpr int X_ST = 32 * TM * 4;                           // a staging slot [32 weight rows][128 pair rows] fp32: 16 KB, two per group
constexpr int X_O_BAR = 0, X_O_X = 1024, X_O_S = X_O_X + XST * X_XT, X_O_W = X_O_S + 4 * X_ST;
constexpr int X_MAXCH = (SMEM_MAX - 1024 - X_O_W) / X_WT;   // W' chunks that fit
static_assert(X_MAXCH >= 1, "forward (CTA pairs) shared memory");
constexpr uint32_t X_COL_ACC = 2 * DP;                      // TMEM: A[2] | accumulators [4][64]
constexpr uint32_t X_IDESC = idesc_tf32(2 * TM, X_WC);

struct XBars { uint64_t xfull[XST], w, wpeer, aready[2], aempty[2], acc[4], acc_empty[4]; uint32_t tmem; };
static_assert(sizeof(XBars) <= 1024, "barriers");

DEVI void umma_ts2_tf32(uint32_t d, uint32_t a_tmem, uint64_t b, uint32_t idesc, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::2.kind::tf32 [%0], [%1], %2, %3, p; }"
               :: "r"(d), "r"(a_tmem), "l"(b), "r"(idesc), "r"(acc) : "memory");
}

// mx: pair rows [R][128] fp32, box (32, 128) SW128; mw: W' [N][128] fp32 (this launch's rows), box (32, 32) SW128 (rows past N zero-
// filled); out: bias_all^T [N][R] fp32 (this launch's rows); R a multiple of 128, N <= 64 X_MAXCH; grid: clusters of 2
extern "C" __global__ void __launch_bounds__(X_THREADS, 1)
bo_hoist_fwdx_sm100(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mw, const __grid_constant__ CUtensorMap mo,
                    int R, int N, int eps_bits) {
  extern __shared__ __align__(1024) uint8_t smraw[];
  uint8_t* sm = smraw + ((1024u - (smem_u32(smraw) & 1023u)) & 1023u);
  const uint32_t su = smem_u32(sm);
  XBars& B = *reinterpret_cast<XBars*>(sm + X_O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int crank = (int)cluster_rank();
  const bool leader = crank == 0;
  const int T = R / TM, TP = (T + 1) / 2, cl = (int)blockIdx.x >> 1, ncl = (int)gridDim.x >> 1;
  const int my = (TP > cl) ? (TP - cl + ncl - 1) / ncl : 0;
  const int nch = (N + X_WC - 1) / X_WC;
  const float eps = __int_as_float(eps_bits);
  if (nch > X_MAXCH) __trap();

  if (tid == 0) {
    for (int s = 0; s < XST; ++s) mbar_init(&B.xfull[s], 1);
    mbar_init(&B.w, 1); mbar_init(&B.wpeer, 1);
    for (int b = 0; b < 2; ++b) { mbar_init(&B.aready[b], 8); mbar_init(&B.aempty[b], 1); }
    for (int a = 0; a < 4; ++a) { mbar_init(&B.acc[a], 1); mbar_init(&B.acc_empty[a], 8); }
    fence_barrier_init();
    prefetch_map(&mx); prefetch_map(&mw); prefetch_map(&mo);
  }
  if (warp == 1) { tmem_alloc2(smem_u32(&B.tmem), 512); tmem_relinquish2(); }
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    if (lane == 0) {                                         // this CTA's half of every W' chunk, once
      mbar_expect_tx(&B.w, nch * X_WT);
      for (int j = 0; j < nch; ++j)
        for (int b = 0; b < 4; ++b) tma_load_2d(su + X_O_W + j * X_WT + b * X_WB, &mw, &B.w, b * 32, j * X_WC + crank * X_WH);
      if (!leader) { mbar_wait(&B.w, 0); mbar_arrive_remote(&B.wpeer, 0); }
    }
  } else if (warp == 1) {
    if (leader) {
      mbar_wait(&B.w, 0);
      mbar_wait_cl(&B.wpeer, 0);
      int u = 0;
      for (int i = 0; i < my; ++i) {
        const int b = i & 1;
        mbar_wait_cl(&B.aready[b], (i >> 1) & 1);           // both CTAs' tiles i in A[b]
        for (int j = 0; j < nch; ++j, ++u) {
          const int a = u & 3;
          if (u >= 4) mbar_wait_cl(&B.acc_empty[a], ((u >> 2) - 1) & 1);
          tc_fence_after();
          const uint32_t wj = su + X_O_W + j * X_WT;
          if (elect_one()) {
#pragma unroll
            for (int ks = 0; ks < 16; ++ks) {                // K step: 8 channels (8 TMEM columns of A; 32 bytes of a W' row)
              const uint32_t bx = (uint32_t)(ks >> 2), ko = (uint32_t)(ks & 3) * 2;
              umma_ts2_tf32(tmem + X_COL_ACC + a * X_WC, tmem + b * DP + ks * 8, desc_k128(wj + bx * X_WB) + ko, X_IDESC, ks ? 1u : 0u);
            }
            tc_commit2_mc(&B.acc[a], 3);
            if (j == nch - 1) tc_commit2_mc(&B.aempty[b], 3);
          }
          __syncwarp();
        }
      }
    }
  } else if (warp < 6) {                                     // the normaliser: tile i from slot i % XST into A[i & 1]
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane;
    auto load_x = [&](int i) {
      const int s = i % XST, r0 = (2 * (cl + i * ncl) + crank) * TM;   // past R (an odd last tile's partner): zero fill
      mbar_expect_tx(&B.xfull[s], X_XT);
      for (int b = 0; b < 4; ++b) tma_load_2d(su + X_O_X + s * X_XT + b * X_XB, &mx, &B.xfull[s], b * 32, r0);
    };
    if (warp == 2 && lane == 0)
      for (int i = 0; i < XST && i < my; ++i) load_x(i);
    for (int i = 0; i < my; ++i) {
      const int s = i % XST, b = i & 1;
      const uint32_t xs = su + X_O_X + s * X_XT, rr = opaque(r);
      mbar_wait(&B.xfull[s], (i / XST) & 1);
      const float k0 = __uint_as_float(lds128v(xs + xoff(rr, 0)).x);
      f2 s1 = mk2(0.f, 0.f), s2 = s1, s3 = s1, s4 = s1;
      const f2 nk = dup2(-k0);
#pragma unroll 4
      for (int q = 0; q < 32; ++q) {
        const uint4 u = lds128v(xs + xoff(rr, q));
        const f2 t0 = add2(mk2u(u.x, u.y), nk), t1 = add2(mk2u(u.z, u.w), nk);
        s1 = add2(s1, t0); s2 = fma2(t0, t0, s2); s3 = add2(s3, t1); s4 = fma2(t1, t1, s4);
      }
      float mean, rstd, m1;
      finish_stats(add2(s1, s3), add2(s2, s4), k0, eps, mean, rstd, m1);
      const f2 rs = dup2(rstd), nmr = dup2(-mean * rstd);
      if (i >= 2) mbar_wait(&B.aempty[b], ((i >> 1) - 1) & 1);   // A[b]'s previous tile multiplied
      tc_fence_after();
      const uint32_t ta = tmem + (lb << 16) + (uint32_t)b * DP;
#pragma unroll 1
      for (int c = 0; c < 8; ++c) {                          // 16 channels per TMEM store
        uint32_t o[16];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const uint4 u = lds128v(xs + xoff(rr, c * 4 + e));
          const f2 y0 = fma2(mk2u(u.x, u.y), rs, nmr), y1 = fma2(mk2u(u.z, u.w), rs, nmr);
          o[4 * e] = tf32_rna(lo2(y0)); o[4 * e + 1] = tf32_rna(hi2(y0)); o[4 * e + 2] = tf32_rna(lo2(y1)); o[4 * e + 3] = tf32_rna(hi2(y1));
        }
        tmem_st16(ta + c * 16, o);
      }
      tmem_wait_st();
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive_remote_relaxed(&B.aready[b], 0);
      named_bar_sync(1, 128);                                // the slot read by the normaliser's 4 warps: the next tile into it
      if (warp == 2 && lane == 0 && i + XST < my) load_x(i + XST);
    }
  } else {                                                   // drain group g: the CTA's chunks u = g mod 2
    const int g = (warp - 6) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane;
    const bool gl = warp == 6 + 4 * g && lane == 0;          // the group's TMA store issuer
    const uint32_t stg = su + X_O_S + (uint32_t)g * 2 * X_ST;
    int u = 0, nst = 0;
    for (int i = 0; i < my; ++i) {
      const int r0 = (2 * (cl + i * ncl) + crank) * TM;
      for (int j = 0; j < nch; ++j, ++u) {
        if ((u & 1) != g) continue;
        const int a = u & 3;
        mbar_wait(&B.acc[a], (u >> 2) & 1);
        tc_fence_after();
        uint32_t v[64];
        tmem_ld32(tmem + (lb << 16) + X_COL_ACC + a * X_WC, *reinterpret_cast<uint32_t(*)[32]>(v));
        tmem_ld32(tmem + (lb << 16) + X_COL_ACC + a * X_WC + 32, *reinterpret_cast<uint32_t(*)[32]>(v + 32));
        tmem_wait_ld();
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive_remote_relaxed(&B.acc_empty[a], 0);
#pragma unroll
        for (int h = 0; h < 2; ++h, ++nst) {                 // weight rows j 64 + 32 h ..: [32 weight rows][128 pair rows] by TMA
          const uint32_t sb = stg + (uint32_t)(nst & 1) * X_ST;
          if (gl && nst >= 2) asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");   // the slot's previous store read it
          named_bar_sync(2 + g, 128);
#pragma unroll
          for (int k = 0; k < 32; ++k) sts32(sb + (uint32_t)k * 512u + r * 4u, v[32 * h + k]);
          fence_proxy_async();
          named_bar_sync(2 + g, 128);
          if (gl) { tma_store_2d(&mo, sb, r0, j * X_WC + 32 * h); tma_store_commit(); }
        }
      }
    }
    if (gl) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  cluster_sync();
  if (warp == 1) { tc_fence_after(); tmem_dealloc2(tmem, 512); }
}
#endif  // FWDX

// ================================================================================================ backward, normalised tile
// -DBWD3 (bo_hoist_bwd3_sm100; bf16, and -DTF32 for fp32): the backward for any number of weight rows. The row threads turn their tile
// into xhat = ((x - k) - (mean - k)) rstd once per tile -- bf16: two bf16 terms, hi over the raw row in place and lo into the slot's
// second tile (hi + lo = xhat to ~2^-17); tf32: in place, rounded to the nearest tf32 (cvt.rna) -- so every product of the tile reads
// operands as TMA left them:
//   d LN0 = sum_j dbias_j^T W'_j          (A = the dbias chunk MN-major, B = the W' chunk MN-major; TMEM rows x channels)
//   S_j  += xhat^T dbias_j^T               (A = the tile MN-major, B = the dbias chunk K-major)
// S = dbias xhat is dW' itself (no channel mean to subtract): bf16 the d bias against xhat to ~2^-17, tf32 both operands rounded to
// the nearest tf32 (the arithmetic of the LN0-rows path), fp32 accumulation. Then the LayerNorm backward from d LN0: mean(dy),
// mean(dy xhat), dx = rstd (dy - mean(dy) - xhat mean(dy xhat)), written over the tile and stored by TMA from there. The row threads
// have no work per weight-row chunk: the weight rows cost tensor-core work and the dbias / W' streams only.
// Chunks: bf16 dbias 128 weight rows ([2 boxes of 64 pair rows][128][128 B] SW128: S's K-major B and d LN0's MN-major A at once), W'
// 64 rows; tf32 32 rows each, the dbias chunk loaded twice (K-major SW128 for S, MN-major with 32-B atoms for d LN0: an MN-major
// kind::tf32 operand must sit in that layout, a K-major one cannot).
// Rings: the tile slot of group g (slot g, loaded by the group itself once its previous tile's dx is stored), the dbias ring and the
// W' ring (the MMA warp's order of use). TMEM: d LN0 accumulators -- one per group while S takes <= 256 columns, else one shared
// (the MMA runs the next tile's first S products while the drain empties it) -- and S [128 channels][weight rows].
// A launch walks the dbias chunks [jb, je) of every tile: ``dx`` launches all of them (d LN0, d pair) with S for [s0, s1) (<= 384
// weight rows); S-only launches (dx = 0) [s0, s1) (<= 512 weight rows), the pair read again (``hoist.bwd3_plan``).
#ifdef BWD3
#ifdef TF32
constexpr int X3B = TM * 128, X3T = 4 * X3B;              // the tile [4 boxes of 32 channels][128 rows][128 B], 32-B atoms (raw -> xhat -> dx)
constexpr int NTERM = 1;                                  // xhat in one term
#ifdef SMALLN
// -DSMALLN (N <= 16): chunks of 16 weight rows (no zero-filled half), W' resident, one dbias stage, THREE tile slots
constexpr int DC = 16, WC = 16;
#else
constexpr int DC = 32, WC = 32;                           // weight rows per dbias / W' chunk
#endif
constexpr int DKB = DC * 128;                             // a box of the dbias chunk [32 weight rows][32 pair rows]: 4 KB
constexpr int D3K = 4 * DKB, D3T = 2 * D3K;               // the dbias chunk K-major SW128 | MN-major 32-B atoms
constexpr int W3B = WC * 128, W3T = 4 * W3B;              // W' chunk [4 boxes of 32 channels][32 weight rows][128 B], 32-B atoms
constexpr int KS_X = 16, KST_X = 1024;                    // S: K steps of 8 pair rows (1 KB of the MN tile)
constexpr int KS_W = WC / 8, KST_W = 1024;                // d LN0: K steps of 8 weight rows per W' chunk
constexpr uint32_t ID3_DLN = idesc_tf32(TM, DP, 1, 1);
DEVI uint64_t dmn3(uint32_t a, uint32_t lbo) { return desc_mn32b(a, lbo); }
DEVI void mma3(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t acc) { umma_ss_tf32(d, a, b, id, acc); }
DEVI uint32_t idesc3_s(int n) { return idesc_tf32(DP, n, 1, 0); }
// byte offset of fp32 (r, 4 c .. 4 c + 3) -- chunk c (0..31) -- in the tile: 32-B granule j of 128-B row r at granule j ^ (r & 3)
DEVI uint32_t x3off(uint32_t r, uint32_t c) { return (c >> 3) * X3B + r * 128u + ((((c & 7u) >> 1) ^ (r & 3u)) << 5) + (c & 1u) * 16u; }
#else
constexpr int X3B = TM * 128, X3T = 2 * X3B;              // the tile [2 boxes of 64 channels][128 rows][128 B] SW128 (raw -> hi -> dx)
constexpr int NTERM = 2;                                  // xhat as hi | lo
constexpr int DC = 128, WC = 64;
constexpr int DKB = DC * 128;                             // a box of the dbias chunk [128 weight rows][64 pair rows]: 16 KB
constexpr int D3K = 2 * DKB, D3T = D3K;                   // one copy serves both products
constexpr int W3B = WC * 128, W3T = 2 * W3B;              // W' chunk [2 boxes of 64 channels][64 weight rows][128 B] SW128
constexpr int KS_X = 8, KST_X = 2048;                     // S: K steps of 16 pair rows
constexpr int KS_W = WC / 16, KST_W = 2048;               // d LN0: K steps of 16 weight rows
constexpr uint32_t ID3_DLN = idesc_bf16(TM, DP, 1, 1);
DEVI uint64_t dmn3(uint32_t a, uint32_t lbo) { return desc_mn128(a, lbo); }
DEVI void mma3(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t acc) { umma_ss(d, a, b, id, acc); }
DEVI uint32_t idesc3_s(int n) { return idesc_bf16(DP, n, 1, 0); }
#endif
constexpr int X3S = NTERM * X3T;                          // a slot
#ifdef SMALLN
static_assert(NTERM == 1, "SMALLN: tf32");
constexpr int D3ST = 1, W3ST = 1, NSLOT = 3;              // W' (one chunk) loaded once
constexpr bool WRES = true;
#else
constexpr int D3ST = 2, W3ST = 2, NSLOT = 2;              // tile slot i % NSLOT: the group's own with two
constexpr bool WRES = false;
#endif
constexpr int O3_X = 0, O3_D = O3_X + NSLOT * X3S, O3_W = O3_D + D3ST * D3T, O3_BAR = O3_W + W3ST * W3T;
constexpr int B3_SMEM = 1024 + O3_BAR + 512;
static_assert(B3_SMEM <= SMEM_MAX, "backward (normalised tile) shared memory");
static_assert(DC % WC == 0 && X3T % 1024 == 0 && D3K % 1024 == 0 && W3T % 1024 == 0, "chunks; 1 KB alignment");

struct B3Bars {
  uint64_t xfull[NSLOT], xready[2], xmma[2], dfull[D3ST], dempty[D3ST], wfull[W3ST], wempty[W3ST], acc[2], acc_empty[2], done;
  uint32_t tmem;
};

// mx: pair rows [R][128] (bf16: box (64, 128) SW128; tf32: box (32, 128) 32-B atoms); mdb: dbias_all [N][R], box (bf16 64 / tf32 32
// pair rows, DC weight rows) SW128 (rows past N zero-filled); mdbm (tf32): the same box, 32-B atoms; mw: W' [N][128], box (64 / 32
// channels, WC rows) (bf16 SW128, tf32 32-B atoms; zero-filled past N); mdx: d pair [R][128] (bf16 box (64, 32) SW128, tf32 (32, 32)
// 32-B atoms); part: S partials [grid][(s1 - s0) DC][128] fp32 (weight row, channel). dx != 0: d LN0 over every chunk and d pair.
extern "C" __global__ void __launch_bounds__(THREADS, 1)
bo_hoist_bwd3_sm100(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mdb, const __grid_constant__ CUtensorMap mdbm,
                    const __grid_constant__ CUtensorMap mw, const __grid_constant__ CUtensorMap mdx, float* __restrict__ part, int R, int N,
                    int s0, int s1, int dx, int eps_bits) {
  extern __shared__ __align__(1024) uint8_t smraw[];
  uint8_t* sm = smraw + ((1024u - (smem_u32(smraw) & 1023u)) & 1023u);
  const uint32_t su = smem_u32(sm);
  B3Bars& B = *reinterpret_cast<B3Bars*>(sm + O3_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int T = R / TM, G = gridDim.x, my = (T > (int)blockIdx.x) ? (T - (int)blockIdx.x + G - 1) / G : 0;
  const float eps = __int_as_float(eps_bits);
  const int nd = (N + DC - 1) / DC, nw = (N + WC - 1) / WC;
  const int jb = dx ? 0 : s0, nj = (dx ? nd : s1) - jb, la = nj < D3ST ? nj : D3ST;
  const int ns = s1 - s0, na = dx ? (ns * DC <= 256 ? 2 : 1) : 0, scol = na * DP;

  if (tid == 0) {
    for (int g = 0; g < 2; ++g) {
      mbar_init(&B.xready[g], 4); mbar_init(&B.xmma[g], 1);
      mbar_init(&B.acc[g], 1); mbar_init(&B.acc_empty[g], 4);
    }
    for (int s = 0; s < NSLOT; ++s) mbar_init(&B.xfull[s], 1);
    for (int s = 0; s < D3ST; ++s) { mbar_init(&B.dfull[s], 1); mbar_init(&B.dempty[s], 1); }
    for (int s = 0; s < W3ST; ++s) { mbar_init(&B.wfull[s], 1); mbar_init(&B.wempty[s], 1); }
    mbar_init(&B.done, 1);
    fence_barrier_init();
    prefetch_map(&mx); prefetch_map(&mdb); prefetch_map(&mw); prefetch_map(&mdx);
#ifdef TF32
    prefetch_map(&mdbm);
#endif
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    if (lane == 0) {                                         // the dbias and W' rings, in the MMA warp's order of use
      int qd = 0, qw = 0;
      if (WRES && dx && my > 0) {                            // W' (one chunk) resident: loaded once
        mbar_expect_tx(&B.wfull[0], W3T);
        for (int b = 0; b < DP * ES / 128; ++b) tma_load_2d(su + O3_W + b * W3B, &mw, &B.wfull[0], b * (128 / ES), 0);
      }
      for (int i = 0; i < my; ++i) {
        const int r0 = ((int)blockIdx.x + i * G) * TM;
        auto load_d = [&](int J) {
          const int ds = qd % D3ST;
          if (qd >= D3ST) mbar_wait(&B.dempty[ds], ((qd / D3ST) - 1) & 1);
          const uint32_t dt = su + O3_D + ds * D3T;
          mbar_expect_tx(&B.dfull[ds], D3T);
          for (int b = 0; b < TM * ES / 128; ++b) {
            tma_load_2d(dt + b * DKB, &mdb, &B.dfull[ds], r0 + b * (128 / ES), J * DC);
#ifdef TF32
            tma_load_2d(dt + D3K + b * DKB, &mdbm, &B.dfull[ds], r0 + b * (128 / ES), J * DC);
#endif
          }
          ++qd;
        };
        for (int t = 0; t < la; ++t) load_d(jb + t);
        for (int t = 0; t < nj; ++t) {
          const int J = jb + t;
          if (t >= la) load_d(J);
          if (dx && !WRES)
            for (int w = J * (DC / WC); w < (J + 1) * (DC / WC) && w < nw; ++w, ++qw) {
              const int ws = qw % W3ST;
              if (qw >= W3ST) mbar_wait(&B.wempty[ws], ((qw / W3ST) - 1) & 1);
              const uint32_t wt = su + O3_W + ws * W3T;
              mbar_expect_tx(&B.wfull[ws], W3T);
              for (int b = 0; b < DP * ES / 128; ++b) tma_load_2d(wt + b * W3B, &mw, &B.wfull[ws], b * (128 / ES), w * WC);
            }
        }
      }
    }
  } else if (warp == 1) {
    int qd = 0, qw = 0;
    for (int i = 0; i < my; ++i) {
      const int g = i & 1, a = na == 2 ? g : 0;               // a: the TMEM accumulator; the barriers acc / acc_empty are per group
      const uint32_t xs = su + O3_X + (uint32_t)(i % NSLOT) * X3S;
      mbar_wait(&B.xready[g], (i >> 1) & 1);                 // the group's tile is xhat
      auto sprod = [&](int J, uint32_t dt) {
        if (J < s0 || J >= s1) return;
        const int nJ = min(DC, ((N - J * DC) + 15) & ~15);
        const uint32_t id = idesc3_s(nJ), col = tmem + scol + (uint32_t)(J - s0) * DC;
        if (elect_one()) {
#pragma unroll
          for (int h = 0; h < NTERM; ++h)
#pragma unroll
            for (int ks = 0; ks < KS_X; ++ks)                // K step: 32 bytes of a pair row
              mma3(col, dmn3(xs + h * X3T + ks * KST_X, X3B), desc_k128(dt + (ks >> 2) * DKB) + (uint64_t)((ks & 3) * 2), id,
                   (i | h | ks) ? 1u : 0u);
        }
        __syncwarp();
      };
      tc_fence_after();
      for (int t = 0; t < la; ++t) {                         // the first chunks' S products: before d LN0 waits for its accumulator
        const int ds = (qd + t) % D3ST;
        mbar_wait(&B.dfull[ds], ((qd + t) / D3ST) & 1);
        tc_fence_after();
        sprod(jb + t, su + O3_D + ds * D3T);
      }
      // the accumulator's previous tile drained: the group's own previous tile (two accumulators) or the other group's (one; a
      // barrier pair per group, never one barrier whose alternate phases two groups wait for: a waiter a phase early would pass)
      if (dx && na == 2 && i >= 2) mbar_wait(&B.acc_empty[g], ((i >> 1) - 1) & 1);
      if (dx && na == 1 && i >= 1) mbar_wait(&B.acc_empty[g ^ 1], ((i - 1) >> 1) & 1);
      tc_fence_after();
      bool first = true;
      for (int t = 0; t < nj; ++t) {
        const int J = jb + t, ds = (qd + t) % D3ST;
        const uint32_t dt = su + O3_D + ds * D3T;
        if (t >= la) {
          mbar_wait(&B.dfull[ds], ((qd + t) / D3ST) & 1);
          tc_fence_after();
          sprod(J, dt);
        }
        if (dx)
          for (int w = J * (DC / WC); w < (J + 1) * (DC / WC) && w < nw; ++w, ++qw) {
            const int ws = WRES ? 0 : qw % W3ST;
            mbar_wait(&B.wfull[ws], WRES ? 0 : (qw / W3ST) & 1);   // resident: phase 0, completed once
            tc_fence_after();
#ifdef TF32
            const uint32_t da = dt + D3K;                    // the MN-major copy
#else
            const uint32_t da = dt + (uint32_t)(w & 1) * (WC * 128);   // this W' chunk's 64 weight rows of the dbias chunk
#endif
            const uint32_t wt = su + O3_W + ws * W3T;
            if (elect_one()) {
#pragma unroll
              for (int ks = 0; ks < KS_W; ++ks)              // K step: 16 (bf16) / 8 (tf32) weight rows
                mma3(tmem + a * DP, dmn3(da + ks * KST_W, DKB), dmn3(wt + ks * KST_W, W3B), ID3_DLN, (first && ks == 0) ? 0u : 1u);
              if (!WRES) tc_commit(&B.wempty[ws]);
            }
            __syncwarp();
            first = false;
          }
        if (elect_one()) tc_commit(&B.dempty[ds]);
        __syncwarp();
      }
      qd += nj;
      if (elect_one()) {
        if (dx) tc_commit(&B.acc[g]);
        tc_commit(&B.xmma[g]);
      }
      __syncwarp();
    }
    if (elect_one()) tc_commit(&B.done);
    __syncwarp();
  } else {
    const int g = (warp - 2) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane;
    const bool leader = warp == 2 + 4 * g && lane == 0;     // issues tile loads: its group's first, then into each slot it frees
    auto load_x = [&](int i) {                               // tile i into slot i % NSLOT
      const int r0 = ((int)blockIdx.x + i * G) * TM, sl = i % NSLOT;
      const uint32_t xd = su + O3_X + (uint32_t)sl * X3S;
      mbar_expect_tx(&B.xfull[sl], X3T);
      for (int b = 0; b < DP * ES / 128; ++b) tma_load_2d(xd + b * X3B, &mx, &B.xfull[sl], b * (128 / ES), r0);
    };
    if (leader)
      for (int i = g; i < NSLOT && i < my; i += 2) load_x(i);
    for (int i = g; i < my; i += 2) {
      const int k = i >> 1, r0 = ((int)blockIdx.x + i * G) * TM;
      const uint32_t rr = opaque(r), xs = su + O3_X + (uint32_t)(i % NSLOT) * X3S;
      mbar_wait(&B.xfull[i % NSLOT], (i / NSLOT) & 1);
      // ---- the row statistics (shifted one-pass sums: the forward's arithmetic), then xhat
      float mean, rstd, m1, k0;
#ifdef TF32
      {
        k0 = __uint_as_float(lds128v(xs + x3off(rr, 0)).x);
        const f2 nk = dup2(-k0);
        f2 q1 = mk2(0.f, 0.f), q2 = q1, q3 = q1, q4 = q1;
#pragma unroll 4
        for (int c = 0; c < 32; ++c) {
          const uint4 u = lds128v(xs + x3off(rr, c));
          const f2 t0 = add2(mk2u(u.x, u.y), nk), t1 = add2(mk2u(u.z, u.w), nk);
          q1 = add2(q1, t0); q2 = fma2(t0, t0, q2); q3 = add2(q3, t1); q4 = fma2(t1, t1, q4);
        }
        finish_stats(add2(q1, q3), add2(q2, q4), k0, eps, mean, rstd, m1);
        const f2 rs = dup2(rstd), nm1r = dup2(-m1 * rstd);
#pragma unroll 4
        for (int c = 0; c < 32; ++c) {                        // xhat rounded to the nearest tf32, in place
          const uint32_t ad = xs + x3off(rr, c);
          const uint4 u = lds128v(ad);
          const f2 y0 = fma2(add2(mk2u(u.x, u.y), nk), rs, nm1r), y1 = fma2(add2(mk2u(u.z, u.w), nk), rs, nm1r);
          sts128(ad, make_uint4(tf32_rna(lo2(y0)), tf32_rna(hi2(y0)), tf32_rna(lo2(y1)), tf32_rna(hi2(y1))));
        }
      }
#else
      {
        k0 = row_shift(xs, rr);
        const f2 nk = dup2(-k0);
        f2 q1 = mk2(0.f, 0.f), q2 = q1, q3 = q1, q4 = q1;
#pragma unroll
        for (int q = 0; q < 16; q += 2) {
          acc_stats(lds128v(xs + xoff(rr, q)), nk, q1, q2);
          acc_stats(lds128v(xs + xoff(rr, q + 1)), nk, q3, q4);
        }
        finish_stats(add2(q1, q3), add2(q2, q4), k0, eps, mean, rstd, m1);
        const f2 rs = dup2(rstd), nm1r = dup2(-m1 * rstd), mone = dup2(-1.f);
#pragma unroll 4
        for (int q = 0; q < 16; ++q) {                        // xhat as hi (in place) + lo
          const uint32_t ad = xs + xoff(rr, q);
          const uint4 u = lds128v(ad);
          const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
          uint32_t hi[4], lo[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const f2 xh = fma2(add2(unpk(w4[e]), nk), rs, nm1r);    // (x - k) rstd - (mean - k) rstd
            hi[e] = pk(xh);
            lo[e] = pk(fma2(unpk(hi[e]), mone, xh));
          }
          sts128(ad, make_uint4(hi[0], hi[1], hi[2], hi[3]));
          sts128(xs + X3T + xoff(rr, q), make_uint4(lo[0], lo[1], lo[2], lo[3]));
        }
      }
#endif
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.xready[g]);
      if (dx) {
        const int a = na == 2 ? g : 0;
        mbar_wait(&B.acc[g], k & 1);                         // d LN0 summed over every chunk; every product of this tile done
        tc_fence_after();
        const uint32_t ta = tmem + (lb << 16) + a * DP;
        const f2 rs = dup2(rstd);
        f2 sd = mk2(0.f, 0.f), sdx = sd, ud = sd, udx = sd;
        // xhat of 32 channels (8 f2 pairs per 16-byte chunk group): tf32 the tile, bf16 hi + lo
#pragma unroll
        for (int c = 0; c < 4; ++c) {                        // pass 1: sum(dy), sum(dy xhat)
          uint32_t v[32];
          tmem_ld32(ta + c * 32, v);
          uint4 hq[8];
#ifdef TF32
#pragma unroll
          for (int e = 0; e < 8; ++e) hq[e] = lds128v(xs + x3off(rr, c * 8 + e));
#else
#pragma unroll
          for (int e = 0; e < 4; ++e) { hq[e] = lds128v(xs + xoff(rr, c * 4 + e)); hq[4 + e] = lds128v(xs + X3T + xoff(rr, c * 4 + e)); }
#endif
          tmem_wait_ld();
#pragma unroll
          for (int p = 0; p < 16; p += 2) {
            f2 x0, x1;
#ifdef TF32
            x0 = mk2u(hq[p >> 1].x, hq[p >> 1].y);            // channels 2 p, 2 p + 1 (p even: the chunk's first half)
            x1 = mk2u(hq[p >> 1].z, hq[p >> 1].w);
#else
            const uint32_t h0 = (&hq[p >> 2].x)[p & 3], l0 = (&hq[4 + (p >> 2)].x)[p & 3];
            const uint32_t h1 = (&hq[(p + 1) >> 2].x)[(p + 1) & 3], l1 = (&hq[4 + ((p + 1) >> 2)].x)[(p + 1) & 3];
            x0 = add2(unpk(h0), unpk(l0));
            x1 = add2(unpk(h1), unpk(l1));
#endif
            const f2 d0 = mk2u(v[2 * p], v[2 * p + 1]), d1 = mk2u(v[2 * p + 2], v[2 * p + 3]);
            sd = add2(sd, d0); sdx = fma2(d0, x0, sdx); ud = add2(ud, d1); udx = fma2(d1, x1, udx);
          }
        }
        sd = add2(sd, ud); sdx = add2(sdx, udx);
        const f2 nmdy = dup2(-(lo2(sd) + hi2(sd)) * (1.f / DP)), nmdyx = dup2(-(lo2(sdx) + hi2(sdx)) * (1.f / DP));
#pragma unroll
        for (int c = 0; c < 4; ++c) {                        // pass 2: dx = rstd (dy - mean(dy) - xhat mean(dy xhat)), over the tile
          uint32_t v[32];
          tmem_ld32(ta + c * 32, v);
          uint4 hq[8];
#ifdef TF32
#pragma unroll
          for (int e = 0; e < 8; ++e) hq[e] = lds128v(xs + x3off(rr, c * 8 + e));
#else
#pragma unroll
          for (int e = 0; e < 4; ++e) { hq[e] = lds128v(xs + xoff(rr, c * 4 + e)); hq[4 + e] = lds128v(xs + X3T + xoff(rr, c * 4 + e)); }
#endif
          tmem_wait_ld();
#ifdef TF32
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const f2 o0 = mul2(fma2(mk2u(hq[e].x, hq[e].y), nmdyx, add2(mk2u(v[4 * e], v[4 * e + 1]), nmdy)), rs);
            const f2 o1 = mul2(fma2(mk2u(hq[e].z, hq[e].w), nmdyx, add2(mk2u(v[4 * e + 2], v[4 * e + 3]), nmdy)), rs);
            sts128(xs + x3off(rr, c * 8 + e), make_uint4(__float_as_uint(lo2(o0)), __float_as_uint(hi2(o0)), __float_as_uint(lo2(o1)), __float_as_uint(hi2(o1))));
          }
#else
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const uint32_t hw[4] = {hq[e].x, hq[e].y, hq[e].z, hq[e].w}, lw[4] = {hq[4 + e].x, hq[4 + e].y, hq[4 + e].z, hq[4 + e].w};
            uint32_t o[4];
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
              const f2 xh = add2(unpk(hw[kk]), unpk(lw[kk])), d = mk2u(v[8 * e + 2 * kk], v[8 * e + 2 * kk + 1]);
              o[kk] = pk(mul2(fma2(xh, nmdyx, add2(d, nmdy)), rs));
            }
            sts128(xs + xoff(rr, c * 4 + e), make_uint4(o[0], o[1], o[2], o[3]));
          }
#endif
        }
        tc_fence_before();
        __syncwarp();
        if (lane == 0) mbar_arrive(&B.acc_empty[g]);
        fence_proxy_async();                                 // dx in the tile -> this warp's TMA stores of its 32 rows
        __syncwarp();
        if (lane == 0) {
          for (int b = 0; b < DP * ES / 128; ++b) tma_store_2d(&mdx, xs + b * X3B + lb * 128, b * (128 / ES), r0 + (int)lb);
          tma_store_commit();
          tma_store_wait_read0();                            // the slot is free once the stores have read it
        }
      } else {
        mbar_wait(&B.xmma[g], k & 1);                        // the tile's S products done
      }
      named_bar_sync(1 + g, 128);                            // the group's 4 warps are done with the slot
      if (leader && i + NSLOT < my) load_x(i + NSLOT);
    }
    if (lane == 0) tma_store_wait0();
    if (g == 0) {                                            // this CTA's S partial: TMEM lane = channel, column = weight row
      mbar_wait(&B.done, 0);
      tc_fence_after();
      float* pp = part + (long)blockIdx.x * ns * DC * DP + r;
      for (int c = 0; c < ns * DC / 16; ++c) {
        uint32_t v[16];
        tmem_ld16(tmem + (lb << 16) + scol + c * 16, v);
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
#endif  // BWD3
