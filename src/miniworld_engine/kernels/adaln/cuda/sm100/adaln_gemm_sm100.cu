// adaln_gemm_sm100.cu -- the B200 (sm_100a) adaptive LayerNorm without a backward, after its one row pass (``ln_cond_xstats`` of
// adaln_rows.cu: cn = LN(cond) * wc and the statistics of x):
//
//   y = sigmoid(cn Ws^T + bs) * (x rstd - mean rstd) + cn Wb^T
//
// one persistent, warp-specialised tcgen05 GEMM with the gate and the LayerNorm of x in its epilogue: the [scale | bias] pre-activations
// never reach HBM (the row kernels + cuBLAS route wrote and re-read them, and took three launches more). The structure is the generic
// sm_100a Transition GEMM's (kernels/transition/cuda/sm100/generic/tg_gemm.cu, its SWIGLU operand layout: one A operand, two B tiles per
// ring stage, two accumulators), with this op's epilogue:
//
//   roles (320 threads, one CTA per SM): warp 0 TMA producer, warp 1 MMA issuer (and TMEM owner), warps 2-9 the epilogue -- warp w reads
//   TMEM lanes 32 (w % 4) .. + 31 (rows 32 (w % 4) .. of the tile); the two warps of a lane quarter split the 32-column units and
//   TMA-store 32-row slabs from a swizzled staging buffer. The two accumulators (acc 0 = cn Wb^T, acc 1 = cn Ws^T) are double-buffered in
//   tensor memory (2 x 2 x BN columns), so the epilogue of tile i overlaps the main loop of tile i + 1.
//
// -DF32=1: fp32 activations on the TF32 tensor cores (kind::tf32; cn and the weights arrive on the TF32 grid, rounded to nearest; ``rna``
// rounds y to it too, for a consumer whose MMA truncates), else bf16 (kind::f16). Any M (TMA zero-fills the tail rows of the last 128-row
// tile and clips them on the store), N a multiple of the output group (64 bf16 / 32 fp32 columns), K a multiple of 64 (bf16) / 32 (fp32).
// Launched as a programmatic dependent of the row pass: everything up to the first read of cn overlaps that kernel's tail.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef F32
#define F32 0
#endif
#ifndef BN
#define BN 128
#endif

constexpr int NACC = 2;                              // acc 0 = cn Wb^T (the shift), acc 1 = cn Ws^T (the gate's pre-activation)
constexpr int ESZ = F32 ? 4 : 2;                     // activation / operand element bytes
constexpr int BKE = 128 / ESZ;                       // elements per 128-byte K-block row
constexpr int OCG = F32 ? 32 : 64;                   // output columns per 128-byte staging row
constexpr int UPG = OCG / 32;                        // 32-column TMEM loads per output group
#ifndef STGP
#define STGP 2
#endif
constexpr int STG_PAR = STGP;                        // staging buffers per warp (2: double-buffered stores; 1: room for a deeper operand ring)
constexpr int A_BYTES = 128 * 128, B_BYTES = BN * 128, STAGE = A_BYTES + NACC * B_BYTES;
constexpr int SLAB = 32 * 128;                       // one warp's staging slab: 32 rows x 128 B
constexpr int NEPI = 8;                              // epilogue warps: two per TMEM lane quarter, splitting the columns
constexpr int STG_BYTES = NEPI * STG_PAR * SLAB;
constexpr int BUDGET = 232448 - 1024;                // dynamic shared memory minus the barriers' kilobyte
constexpr int BSM_BYTES = 2 * BN * 4;                // the gate bias of the current and the previous tile's columns (fp32)
constexpr int NST_FIT = (BUDGET - STG_BYTES - BSM_BYTES) / STAGE;
constexpr int NST = NST_FIT > 8 ? 8 : NST_FIT;
constexpr int O_ST = 0, O_STG = NST * STAGE, O_BSM = O_STG + STG_BYTES, O_BAR = O_BSM + BSM_BYTES;
constexpr int SMEM_BYTES = O_BAR + 1024;
constexpr int TCOLS = NACC * BN;                     // TMEM columns of one accumulator buffer
static_assert(NST >= 2, "ring depth");
static_assert(2 * TCOLS <= 512, "two accumulator buffers must fit tensor memory");
static_assert(BN % 32 == 0 && BN % OCG == 0 && BN >= 32 && BN <= 128, "tile width");
static_assert(STAGE % 1024 == 0 && SLAB % 1024 == 0, "128-B swizzle atoms");

#if F32
// kind::tf32 instruction descriptor: TF32 A / B (format 2), fp32 D, K-major both, N, M
constexpr uint32_t IDESC = (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)(BN >> 3) << 17) | ((uint32_t)(128 >> 4) << 24);
DEVI void umma(uint32_t d, uint64_t a, uint64_t b, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d), "l"(a), "l"(b), "r"(IDESC), "r"(acc) : "memory");
}
#else
constexpr uint32_t IDESC = idesc_bf16(128, BN);
DEVI void umma(uint32_t d, uint64_t a, uint64_t b, uint32_t acc) { umma_ss(d, a, b, IDESC, acc); }
#endif

DEVI float tf32r(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return __uint_as_float(r); }
DEVI float sigm(float a) { return sigmoid_kit(a); }   // rcp.approx(1 + ex2.approx(-a log2 e)), as the tuned kernels
DEVI void tma_store_wait_read1() { asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory"); }

struct Bars {
  uint64_t full[NST], empty[NST], accf[2], acce[2];
  uint32_t tmem;
};

// 32 output values of one row -> 32 columns of its 128-B staging row (unit u of the group): bf16 64 B (4 chunks), fp32 128 B (8 chunks)
DEVI void stage32(uint32_t slab, uint32_t r, int u, const float (&v)[32]) {
#if F32
#pragma unroll
  for (int q = 0; q < 8; ++q)
    sts128(slab + sw128(r, q), make_uint4(__float_as_uint(v[4 * q]), __float_as_uint(v[4 * q + 1]), __float_as_uint(v[4 * q + 2]),
                                          __float_as_uint(v[4 * q + 3])));
#else
#pragma unroll
  for (int q = 0; q < 4; ++q)
    sts128(slab + sw128(r, u * 4 + q), make_uint4(pack_bf16(v[8 * q], v[8 * q + 1]), pack_bf16(v[8 * q + 2], v[8 * q + 3]),
                                                  pack_bf16(v[8 * q + 4], v[8 * q + 5]), pack_bf16(v[8 * q + 6], v[8 * q + 7])));
#endif
}

// 32 activation-dtype values of a row as fp32
DEVI void load32(const void* src, long off, bool ok, float (&v)[32]) {
#if F32
  const float* p = reinterpret_cast<const float*>(src) + off;
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const uint4 w = ok ? ldg128(p + 4 * q) : make_uint4(0, 0, 0, 0);
    v[4 * q] = __uint_as_float(w.x); v[4 * q + 1] = __uint_as_float(w.y); v[4 * q + 2] = __uint_as_float(w.z); v[4 * q + 3] = __uint_as_float(w.w);
  }
#else
  const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(src) + off;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const uint4 w = ok ? ldg128(p + 8 * q) : make_uint4(0, 0, 0, 0);
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) { v[8 * q + 2 * e] = bf16lo(ww[e]); v[8 * q + 2 * e + 1] = bf16hi(ww[e]); }
  }
#endif
}

// maps: A cn [M][K] (box BKE x 128), B0 Wb [N][K], B1 Ws [N][K] (box BKE x BN), O y [M][N] (box OCG x 32).
// X [M][ldx] the activation, ST [M] (rstd, mean rstd) of its rows, BS [N] the gate bias (fp32).
extern "C" __global__ void __launch_bounds__(64 + 32 * NEPI, 1)
adaln_gemm_sm100(const __grid_constant__ CUtensorMap mA, const __grid_constant__ CUtensorMap mB0, const __grid_constant__ CUtensorMap mB1,
                 const __grid_constant__ CUtensorMap mO, const void* __restrict__ X, const float2* __restrict__ ST,
                 const float* __restrict__ BS, int M, int N, int K, int ldx, int rna) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int NT = (N + BN - 1) / BN, tiles = ((M + 127) / 128) * NT;

  if (threadIdx.x == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&B.full[s], 1); mbar_init(&B.empty[s], 1); }
    for (int s = 0; s < 2; ++s) { mbar_init(&B.accf[s], 1); mbar_init(&B.acce[s], NEPI); }
    fence_barrier_init();
    prefetch_map(&mA); prefetch_map(&mB0); prefetch_map(&mB1); prefetch_map(&mO);
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  const int nk = K / BKE;
  // the weight tiles of the first ring stages do not depend on the row pass: issued before waiting for it (the row pass triggers this
  // launch at its start, so they and the prologue above run under it)
  const int npre = (int)blockIdx.x < tiles ? (nk < NST ? nk : NST) : 0;
  if (warp == 0 && lane == 0) {
    const int n0 = ((int)blockIdx.x % NT) * BN;
    for (int k = 0; k < npre; ++k) {
      mbar_expect_tx(&B.full[k], A_BYTES + NACC * B_BYTES);
      const uint32_t base = su + O_ST + k * STAGE;
      tma_load_2d(base + A_BYTES, &mB0, &B.full[k], k * BKE, n0);
      tma_load_2d(base + A_BYTES + B_BYTES, &mB1, &B.full[k], k * BKE, n0);
    }
  }
  pdl_wait();   // cn and the statistics come from the row pass this launch depends on

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ producer
    if (lane == 0) {
      int it = 0;
      for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
        const int m0 = (t / NT) * 128, n0 = (t % NT) * BN;
        for (int k = 0; k < nk; ++k, ++it) {
          const int s = it % NST;
          const uint32_t base = su + O_ST + s * STAGE;
          if (it < npre) {   // this stage's weight tiles are in flight already (above)
            tma_load_2d(base, &mA, &B.full[s], k * BKE, m0);
            continue;
          }
          mbar_wait(&B.empty[s], ((it / NST) & 1) ^ 1);
          mbar_expect_tx(&B.full[s], A_BYTES + NACC * B_BYTES);
          tma_load_2d(base, &mA, &B.full[s], k * BKE, m0);
          tma_load_2d(base + A_BYTES, &mB0, &B.full[s], k * BKE, n0);
          tma_load_2d(base + A_BYTES + B_BYTES, &mB1, &B.full[s], k * BKE, n0);
        }
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    if (lane == 0) {
      int it = 0, tc = 0;
      const int nk = K / BKE;
      for (int t = blockIdx.x; t < tiles; t += gridDim.x, ++tc) {
        const int buf = tc & 1;
        mbar_wait(&B.acce[buf], ((tc >> 1) & 1) ^ 1);
        tc_fence_after();
        const uint32_t dbase = tmem + buf * TCOLS;
        for (int k = 0; k < nk; ++k, ++it) {
          const int s = it % NST;
          mbar_wait(&B.full[s], (it / NST) & 1);
          tc_fence_after();
          const uint32_t base = su + O_ST + s * STAGE;
          const uint64_t da = desc_k128(base);
#pragma unroll
          for (int j = 0; j < NACC; ++j) {
            const uint64_t db = desc_k128(base + A_BYTES + j * B_BYTES);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
              umma(dbase + j * BN, da + (uint64_t)(kk * 2), db + (uint64_t)(kk * 2), (k > 0 || kk > 0) ? 1u : 0u);
          }
          tc_commit(&B.empty[s]);
        }
        tc_commit(&B.accf[buf]);
      }
    }
  } else {
    // ------------------------------------------------------------------------------------------------ epilogue (warps 2-9)
    const int ew = warp - 2, q4 = warp & 3, half = ew >> 2;  // lane quarter, column half
    const uint32_t r = (uint32_t)lane;                       // row within the quarter's 32-row slab
    // UPG == 1 (fp32 out): each warp owns whole groups (g % 2 == half) and its own slabs. UPG == 2 (bf16 out): the two warps of a
    // quarter write the two 64-B halves of one slab (the half-0 warp's), joined by named barrier 1 + q4, and the half-0 warp stores it.
    const int owner = UPG == 1 ? ew : q4;
    const uint32_t stg = su + O_STG + (uint32_t)(owner * STG_PAR) * SLAB;
    const bool storer = UPG == 1 || half == 0;
    int tc = 0, grp = 0;
    for (int t = blockIdx.x; t < tiles; t += gridDim.x, ++tc) {
      const int buf = tc & 1;
      const int m0 = (t / NT) * 128, n0 = (t % NT) * BN;
      const int row = m0 + q4 * 32 + lane;
      const bool rok = row < M;
      const float2 st = rok ? ST[row] : make_float2(0.f, 0.f);   // (rstd, mean rstd): x_hat = x rstd - mean rstd
      // the gate bias of the tile's columns into shared memory (double-buffered by tile parity), loaded before the accumulator wait by the
      // eight epilogue warps, which meet at named barrier 5: in the group loop it was a dependent global load per element (a DRAM round
      // trip on a cold cache) between the MMA and the first store. (Registers instead spilled at two tiles a CTA: bf16 c384 L1024 24 -> 28 us.)
      constexpr int NGW = UPG == 1 ? (BN / OCG + 1) / 2 : BN / OCG;   // groups this warp handles
      float* bsm = reinterpret_cast<float*>(sm + O_BSM) + (tc & 1) * BN;
      for (int i = ew * 32 + lane; i < BN; i += 32 * NEPI) bsm[i] = n0 + i < N ? __ldg(BS + n0 + i) : 0.f;
      named_bar_sync(5, 32 * NEPI);
      float xpre[32];                                        // x of the first group, also in flight under the accumulator wait
      {
        const int g0 = UPG == 1 ? half : 0;
        if (g0 < BN / OCG && n0 + g0 * OCG < N) load32(X, (long)row * ldx + n0 + g0 * OCG + (UPG == 1 ? 0 : half) * 32, rok, xpre);
      }
      mbar_wait(&B.accf[buf], (tc >> 1) & 1);
      tc_fence_after();
      const uint32_t tb = tmem + ((uint32_t)(q4 * 32) << 16) + buf * TCOLS;
#pragma unroll
      for (int jg = 0; jg < NGW; ++jg, ++grp) {              // this warp's groups (UPG == 1: g % 2 == half)
        const int g = UPG == 1 ? half + 2 * jg : jg;
        const int c0 = n0 + g * OCG;                         // first output column of the group
        if (g >= BN / OCG || c0 >= N) break;
        const int par = STG_PAR == 2 ? (grp & 1) : 0;
        if (storer && lane == 0) { if (STG_PAR == 2) tma_store_wait_read1(); else tma_store_wait_read0(); }
        if (UPG == 2) named_bar_sync(1 + q4, 64); else __syncwarp();
        {
          const int u = UPG == 1 ? 0 : half;
          const int col = g * OCG + u * 32;                  // column within the tile
          float o[32];
          if (jg == 0) {
#pragma unroll
            for (int e = 0; e < 32; ++e) o[e] = xpre[e];
          } else {
            load32(X, (long)row * ldx + n0 + col, rok, o);   // x, issued before the tensor-memory loads
          }
          uint32_t acc[NACC][32];
#pragma unroll
          for (int j = 0; j < NACC; ++j) tmem_ld32(tb + j * BN + col, acc[j]);
          tmem_wait_ld();
#pragma unroll
          for (int e = 0; e < 32; ++e) {
            const float sp = __uint_as_float(acc[1][e]) + bsm[col + e];
            const float y = sigm(sp) * (o[e] * st.x - st.y) + __uint_as_float(acc[0][e]);
            o[e] = (F32 && rna) ? tf32r(y) : y;
          }
          stage32(stg + (uint32_t)par * SLAB, r, u, o);
        }
        fence_proxy_async();
        if (UPG == 2) named_bar_sync(1 + q4, 64); else __syncwarp();
        if (storer && lane == 0) {
          tma_store_2d(&mO, stg + (uint32_t)par * SLAB, c0, m0 + q4 * 32);
          tma_store_commit();
        }
      }
      tc_fence_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.acce[buf]);
    }
    if (storer && lane == 0) tma_store_wait0();
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}

extern "C" __device__ int transition_smem_bytes = SMEM_BYTES;
