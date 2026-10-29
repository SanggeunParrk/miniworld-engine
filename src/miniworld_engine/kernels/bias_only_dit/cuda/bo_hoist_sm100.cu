// bo_hoist_sm100.cu -- the hoisted pair bias of the bias-only token DiT on B200 (sm_100a), bf16: every block's bias from ONE pass
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
// forward
constexpr int F_O_W = 0, F_O_ST = F_O_W + ((WT + 1023) / 1024) * 1024;
constexpr int F_NST = (SMEM_MAX - 1024 - F_O_ST - 1024) / XT > 8 ? 8 : (SMEM_MAX - 1024 - F_O_ST - 1024) / XT;
constexpr int F_O_S = F_O_ST + F_NST * XT, F_O_BAR = F_O_S + NP * 4, F_SMEM = 1024 + F_O_BAR + 256;
static_assert(F_NST >= 2 && F_SMEM <= SMEM_MAX, "forward shared memory");
constexpr uint32_t F_IDESC = idesc_bf16(TM, NP);
constexpr uint32_t F_TCOLS = 2 * NP <= 32 ? 32 : 2 * NP <= 64 ? 64 : 2 * NP <= 128 ? 128 : 256;
#if NP <= 64  // ---------------------------------------------------------------------------------------- backward (NP <= 64)
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
struct FBars { uint64_t full[F_NST], empty[F_NST], acc[2], acc_empty[2], w; uint32_t tmem; };

// mx: pair rows [R][128] bf16, box (64, 128) SW128; mw: W' [N][128] bf16, box (64, NP) SW128 (rows past N zero-filled);
// out: bias_all^T [N][R] bf16; wf: W' (the row sums s_n); R a multiple of 128
extern "C" __global__ void __launch_bounds__(THREADS, 1)
bo_hoist_fwd_sm100(const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mw,
                   __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ wf, int R, int N, int eps_bits) {
  extern __shared__ __align__(1024) uint8_t smraw[];
  uint8_t* sm = smraw + ((1024u - (smem_u32(smraw) & 1023u)) & 1023u);
  const uint32_t su = smem_u32(sm);
  FBars& B = *reinterpret_cast<FBars*>(sm + F_O_BAR);
  float* s_n = reinterpret_cast<float*>(sm + F_O_S);
  const uint32_t sn_base = su + F_O_S;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int T = R / TM, G = gridDim.x, my = (T > (int)blockIdx.x) ? (T - (int)blockIdx.x + G - 1) / G : 0;
  const float eps = __int_as_float(eps_bits);

  if (tid == 0) {
    for (int s = 0; s < F_NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 5); }   // empty: the MMA + 4 drain warps
    for (int g = 0; g < 2; ++g) { mbar_init(&B.acc[g], 1); mbar_init(&B.acc_empty[g], 4); }
    mbar_init(&B.w, 1);
    fence_barrier_init();
    prefetch_map(&mx); prefetch_map(&mw);
  }
  if (tid < NP) {                                            // s_n = sum_k W'[n, k] (the bf16 values), 0 for the padding rows
    float s = 0.f;
    if (tid < N)
      for (int k = 0; k < DP; ++k) s += __bfloat162float(wf[(long)tid * DP + k]);
    s_n[tid] = s;
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), F_TCOLS); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    if (lane == 0) {
      mbar_expect_tx(&B.w, WT);
      tma_load_2d(su + F_O_W, &mw, &B.w, 0, 0);
      tma_load_2d(su + F_O_W + WB, &mw, &B.w, 64, 0);
      for (int i = 0; i < my; ++i) {
        const int s = i % F_NST, r0 = ((int)blockIdx.x + i * G) * TM;
        if (i >= F_NST) mbar_wait(&B.empty[s], ((i / F_NST) - 1) & 1);
        const uint32_t st = su + F_O_ST + s * XT;
        mbar_expect_tx(&B.full[s], XT);
        tma_load_2d(st, &mx, &B.full[s], 0, r0);
        tma_load_2d(st + XB, &mx, &B.full[s], 64, r0);
      }
    }
  } else if (warp == 1) {
    mbar_wait(&B.w, 0);
    for (int i = 0; i < my; ++i) {
      const int s = i % F_NST, g = i & 1;
      mbar_wait(&B.full[s], (i / F_NST) & 1);
      if (i >= 2) mbar_wait(&B.acc_empty[g], ((i >> 1) - 1) & 1);
      tc_fence_after();
      const uint32_t st = su + F_O_ST + s * XT;
      if (elect_one()) {
#pragma unroll
        for (int ks = 0; ks < DP / 16; ++ks) {               // K step 16 channels = 32 bytes within a 128-byte row
          const uint32_t bx = (uint32_t)(ks >> 2), ko = (uint32_t)(ks & 3) * 2;
          umma_ss(tmem + g * NP, desc_k128(st + bx * XB) + ko, desc_k128(su + F_O_W + bx * WB) + ko, F_IDESC, ks ? 1u : 0u);
        }
        tc_commit(&B.empty[s]);
        tc_commit(&B.acc[g]);
      }
      __syncwarp();
    }
  } else {
    const int g = (warp - 2) >> 2;
    const uint32_t lb = (uint32_t)(warp & 3) * 32, r = lb + lane;
    const bool odd = lane & 1;
    for (int i = g; i < my; i += 2) {
      const int s = i % F_NST, r0 = ((int)blockIdx.x + i * G) * TM;
      mbar_wait(&B.full[s], (i / F_NST) & 1);
      const uint32_t xt = su + F_O_ST + s * XT, rr = opaque(r);
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
      mbar_wait(&B.acc[g], (i >> 1) & 1);
      tc_fence_after();
      // bias[n][r0 + r] = rstd acc[r][n] - mean rstd s_n; two lanes trade one value per column pair so that each stores a 32-bit
      // word: the even lane (n, r | r + 1), the odd lane (n + 1, r - 1 | r)
      __nv_bfloat16* ob = out + r0 + (int)lb + (lane & ~1);
#pragma unroll 1
      for (int c = 0; c < NP / 16; ++c) {
        uint32_t v[16];
        tmem_ld16(tmem + (lb << 16) + g * NP + c * 16, v);
        tmem_wait_ld();
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int n = c * 16 + 2 * k;
          float2 snf;                                       // ld.volatile: plain loads of s_n get hoisted into NP registers
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
      if (lane == 0) mbar_arrive(&B.acc_empty[g]);
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, F_TCOLS); }
}

// ================================================================================================ backward
// (built for NP <= 64 only: the d' buffers and two stages need the shared memory)
#if NP <= 64
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
#endif  // NP <= 64
