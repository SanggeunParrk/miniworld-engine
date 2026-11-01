// adaln_fused_sm100.cu -- the B200 (sm_100a) adaptive LayerNorm without a backward at a narrow conditioning width (the atom transformer:
// cond 128), in ONE kernel:
//
//   y = sigmoid(cn Ws^T + bs) * LN(x) + cn Wb^T,      cn = LN(cond) * wc
//
// At the atom width the two-launch route (``ln_cond_xstats`` -> ``adaln_gemm_sm100``) moves cn through HBM and back, which is as many bytes
// as the op's own input and output; here cn never leaves shared memory. Persistent CTAs over 128-row tiles of one output-column tile each
// (the CTA's column tile is fixed: grid % NT == 0), roles (320 threads):
//
//   warp 0       TMA: this CTA's [Wb ; Ws] column tile once (resident), then per row tile the RAW cond rows straight into the A buffer (a
//                128-B swizzled K-major box is the layout the MMA reads) and, bf16, the x rows into a staging buffer
//   warp 1       MMA issuer and TMEM owner: per row tile two accumulators (cn Wb^T, cn Ws^T), double-buffered in tensor memory
//   warps 2-5    one thread per row: LN(cond) * wc IN PLACE in the A buffer (bf16, or fp32 on the TF32 grid) from shared memory (16-B
//                chunks, conflict-free under the swizzle; bf16 keeps the row in registers between the statistics and the write) and x's
//                (rstd, mean rstd) -- two-pass statistics (mean, then squared deviations from it), fp32, safe for a large mean
//   warps 6-9    the epilogue: warp w reads TMEM lanes 32 (w % 4) .., one row per thread; y = sigmoid(acc1 + bs) (x rstd - mean rstd) + acc0,
//                stored straight to global memory (64 / 128 contiguous bytes per thread)
//
// bf16: A and the x staging double-buffered (the loads of tile i + 1 run under the MMA and epilogue of tile i; the epilogue reads x from the
// staging). fp32: the whole 128-column weight tile is resident (LN(cond) built once per row tile, not once per column tile), which leaves
// room for one A buffer and no staging: the producers read x's rows from global memory (issued before the wait for the cond tile) and the
// epilogue re-reads them (an L2 hit). Measured (B200, CUDA graph, kernel alone, us): bf16 [40960, 128] 16.3, Triton's fused kernel 23.7,
// the two-launch route 38.7; fp32 36.9 / 32.9 (Triton, kernel alone) / 46.0.
//
// -DF32=1: fp32 activations on the TF32 tensor cores (kind::tf32; ``rna`` rounds y to the TF32 grid for a consumer whose MMA truncates).
// -DKC: the conditioning width (a multiple of 128), -DDX: x's width (a multiple of 128), -DBN: the output-column tile. Any M.
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

#ifndef F32
#define F32 0
#endif
#ifndef BN
#define BN 128
#endif
#ifndef KC
#define KC 128
#endif
#ifndef DX
#define DX 128
#endif

constexpr int NACC = 2;
constexpr int ESZ = F32 ? 4 : 2;
constexpr int BKE = 128 / ESZ;                       // elements per 128-byte K-block row
constexpr int NKB = KC / BKE;                        // K-blocks
constexpr int KB_A = 128 * 128;                      // one K-block of the A tile: 128 rows x 128 B
constexpr int KB_B = BN * 128;                       // one K-block of a B tile
constexpr int NKX = DX / BKE;                        // K-blocks of an x row
constexpr int A_BYTES = NKB * KB_A, B_BYTES = NKB * KB_B, X_BYTES = NKX * KB_A;
// bf16: A and the x staging double-buffered (the epilogue reads x there). fp32: the whole weight tile resident needs the room (BN up to
// 128: LN(cond) is built once per row tile, not once per column tile), so one A buffer and no x staging -- the producers read x's rows
// from global memory for the statistics and the epilogue re-reads them (an L2 hit)
constexpr int NA = F32 ? 1 : 2;
constexpr int NXS = F32 ? 0 : 2;
// fp32 (no x staging): one 32-row x 128-B slab per epilogue warp, through which x is read and y written as whole 128-B row segments
// (the thread-per-row global accesses were one cache line per lane and instruction: 6.8 us of an epilogue per tile at [20480, 128])
constexpr int SLAB = 32 * 128, NSLAB = NXS == 0 ? 4 : 0;
constexpr int O_B = 0, O_A = NACC * B_BYTES, O_X = O_A + NA * A_BYTES, O_SL = O_X + NXS * X_BYTES, O_ST = O_SL + NSLAB * SLAB,
              O_WC = O_ST + 2 * 128 * 8, O_BS = O_WC + KC * 4, O_BAR = O_BS + BN * 4;
constexpr int SMEM_BYTES = O_BAR + 256;
constexpr int TCOLS = NACC * BN;
static_assert(KC % 128 == 0 && DX % 128 == 0 && KC <= 256 && DX <= 256, "widths");
static_assert(2 * TCOLS <= 512 && BN % 32 == 0 && BN <= 128, "tile width");
static_assert(SMEM_BYTES <= 232448, "shared memory");

#if F32
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
DEVI void store32(void* dst, long off, const float (&v)[32]) {
#if F32
  float* p = reinterpret_cast<float*>(dst) + off;
#pragma unroll
  for (int q = 0; q < 8; ++q)
    stg128(p + 4 * q, make_uint4(__float_as_uint(v[4 * q]), __float_as_uint(v[4 * q + 1]), __float_as_uint(v[4 * q + 2]), __float_as_uint(v[4 * q + 3])));
#else
  __nv_bfloat16* p = reinterpret_cast<__nv_bfloat16*>(dst) + off;
#pragma unroll
  for (int q = 0; q < 4; ++q)
    stg128(p + 8 * q, make_uint4(pack_bf16(v[8 * q], v[8 * q + 1]), pack_bf16(v[8 * q + 2], v[8 * q + 3]), pack_bf16(v[8 * q + 4], v[8 * q + 5]),
                                 pack_bf16(v[8 * q + 6], v[8 * q + 7])));
#endif
}

// one 16-B shared-memory chunk as fp32 values (8 bf16 / 4 fp32), and back (bf16 rounded to nearest; fp32 to the TF32 grid)
DEVI void unpack16(const uint4 u, float (&v)[16 / ESZ]) {
#if F32
  v[0] = __uint_as_float(u.x); v[1] = __uint_as_float(u.y); v[2] = __uint_as_float(u.z); v[3] = __uint_as_float(u.w);
#else
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int e = 0; e < 4; ++e) { v[2 * e] = bf16lo(w[e]); v[2 * e + 1] = bf16hi(w[e]); }
#endif
}
DEVI void rd16(uint32_t a, float (&v)[16 / ESZ]) { unpack16(lds128(a), v); }
DEVI void wr16(uint32_t a, const float (&v)[16 / ESZ]) {
#if F32
  sts128(a, make_uint4(__float_as_uint(tf32r(v[0])), __float_as_uint(tf32r(v[1])), __float_as_uint(tf32r(v[2])), __float_as_uint(tf32r(v[3]))));
#else
  sts128(a, make_uint4(pack_bf16(v[0], v[1]), pack_bf16(v[2], v[3]), pack_bf16(v[4], v[5]), pack_bf16(v[6], v[7])));
#endif
}

struct Bars {
  uint64_t bfull, rawfull[2], afull[2], aempty[2], sempty[2], accf[2], acce[2];
  uint32_t tmem;
};

// maps: B0 Wb [D][KC], B1 Ws [D][KC] (box BKE x BN), C cond [M][KC] and XM x [M][DX] (box BKE x 128). WC the LayerNorm weight of cond (fp32),
// X / Y [M][DX] (pointers), BS [DX] fp32.
extern "C" __global__ void __launch_bounds__(320, 1)
adaln_fused_sm100(const __grid_constant__ CUtensorMap mB0, const __grid_constant__ CUtensorMap mB1, const __grid_constant__ CUtensorMap mC,
                  const __grid_constant__ CUtensorMap mX,
                  const float* __restrict__ WC, const void* __restrict__ X, void* __restrict__ Y, const float* __restrict__ BS, int M,
                  float eps_c, float eps_x, int rna) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  float2* ST = reinterpret_cast<float2*>(sm + O_ST);   // [2][128] (rstd, mean rstd) of x's rows
  float* WCs = reinterpret_cast<float*>(sm + O_WC);    // [KC] the LayerNorm weight of cond
  float* BSs = reinterpret_cast<float*>(sm + O_BS);    // [BN] the gate bias of this CTA's columns
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  constexpr int NT = DX / BN;
  const int MT = (M + 127) / 128, tiles = MT * NT;
  const int n0 = (blockIdx.x % NT) * BN;               // fixed per CTA: the grid is a multiple of NT

  if (threadIdx.x == 0) {
    mbar_init(&B.bfull, 1);
    for (int s = 0; s < 2; ++s) {
      mbar_init(&B.rawfull[s], 1);
      mbar_init(&B.afull[s], 4); mbar_init(&B.aempty[s], 1); mbar_init(&B.sempty[s], 4);
      mbar_init(&B.accf[s], 1); mbar_init(&B.acce[s], 4);
    }
    fence_barrier_init();
    prefetch_map(&mB0); prefetch_map(&mB1); prefetch_map(&mC); prefetch_map(&mX);
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 512); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------------ the resident weight tile
    if (lane == 0) {
      mbar_expect_tx(&B.bfull, NACC * B_BYTES);
#pragma unroll
      for (int kb = 0; kb < NKB; ++kb) {
        tma_load_2d(su + O_B + kb * KB_B, &mB0, &B.bfull, kb * BKE, n0);
        tma_load_2d(su + O_B + B_BYTES + kb * KB_B, &mB1, &B.bfull, kb * BKE, n0);
      }
      int tc = 0;
      for (int t = blockIdx.x; t < tiles; t += gridDim.x, ++tc) {
        const int buf = tc & 1;
        const int m0 = (t / NT) * 128;
        const int ab = NA == 2 ? buf : 0;
        const uint32_t aph = NA == 2 ? (uint32_t)((tc >> 1) & 1) : (uint32_t)(tc & 1);
        mbar_wait(&B.aempty[ab], aph ^ 1);                 // the MMA of the last tile in this A buffer has read it
        if (NXS == 2) mbar_wait(&B.sempty[buf], ((tc >> 1) & 1) ^ 1);  // the epilogue of tile tc - 2 has read x staging buffer buf
        mbar_expect_tx(&B.rawfull[ab], A_BYTES + NXS / 2 * X_BYTES);
#pragma unroll
        for (int kb = 0; kb < NKB; ++kb) tma_load_2d(su + O_A + ab * A_BYTES + kb * KB_A, &mC, &B.rawfull[ab], kb * BKE, m0);
        if (NXS == 2) {
#pragma unroll
          for (int kb = 0; kb < NKX; ++kb) tma_load_2d(su + O_X + buf * X_BYTES + kb * KB_A, &mX, &B.rawfull[ab], kb * BKE, m0);
        }
      }
    }
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------------ MMA issuer
    if (lane == 0) {
      mbar_wait(&B.bfull, 0);
      int tc = 0;
      for (int t = blockIdx.x; t < tiles; t += gridDim.x, ++tc) {
        const int buf = tc & 1;
        const uint32_t ph = (tc >> 1) & 1;
        const int ab = NA == 2 ? buf : 0;
        mbar_wait(&B.acce[buf], ph ^ 1);
        mbar_wait(&B.afull[ab], NA == 2 ? ph : (uint32_t)(tc & 1));
        tc_fence_after();
        const uint32_t dbase = tmem + buf * TCOLS;
#pragma unroll
        for (int kb = 0; kb < NKB; ++kb) {
          const uint64_t da = desc_k128(su + O_A + ab * A_BYTES + kb * KB_A);
#pragma unroll
          for (int j = 0; j < NACC; ++j) {
            const uint64_t db = desc_k128(su + O_B + j * B_BYTES + kb * KB_B);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
              umma(dbase + j * BN, da + (uint64_t)(kk * 2), db + (uint64_t)(kk * 2), (kb > 0 || kk > 0) ? 1u : 0u);
          }
        }
        tc_commit(&B.aempty[ab]);
        tc_commit(&B.accf[buf]);
      }
    }
  } else if (warp < 6) {
    // ------------------------------------------------------------------------------------------------ A producers (warps 2-5)
    const uint32_t r = (uint32_t)((warp - 2) * 32 + lane);   // this thread's row of the tile
    // wc into shared memory (read by every row below): issued here, under the first tile's TMA, instead of as global loads inside the
    // normalisation (a DRAM round trip on a cold cache, on the path to the MMA); the four producer warps meet at named barrier 1
    for (int i = (int)r; i < KC; i += 128) WCs[i] = __ldg(WC + i);
    named_bar_sync(1, 128);
    int tc = 0;
    for (int t = blockIdx.x; t < tiles; t += gridDim.x, ++tc) {
      const int buf = tc & 1;
      const uint32_t ph = (tc >> 1) & 1;
      const int ab = NA == 2 ? buf : 0;
      mbar_wait(&B.sempty[buf], ph ^ 1);                // the epilogue of tile tc - 2 has read these statistics
      // x's statistics, two passes (the mean, then the sum of squared deviations from it, fp32): from the staging (bf16) or global memory
      // (fp32: the first pass issued before the wait for cond, so its loads overlap that TMA; the second re-reads the row, an L1 hit)
      const long xrow = (long)(t / NT) * 128 + r;
      const float* xr = reinterpret_cast<const float*>(X) + xrow * DX;
      const uint32_t xbase = su + O_X + (NXS == 2 ? buf : 0) * X_BYTES;
      float s = 0.f;
      if (NXS == 0) {
#pragma unroll
        for (int c = 0; c < DX / 4; ++c) {
          const float4 v = xrow < M ? __ldg(reinterpret_cast<const float4*>(xr) + c) : make_float4(0.f, 0.f, 0.f, 0.f);
          s += (v.x + v.y) + (v.z + v.w);
        }
      }
      mbar_wait(&B.rawfull[ab], NA == 2 ? ph : (uint32_t)(tc & 1));   // raw cond (and bf16 x) rows of this tile landed
      if (NXS == 2) {
#pragma unroll
        for (int c = 0; c < DX * ESZ / 16; ++c) {
          float v[16 / ESZ];
          rd16(xbase + (c / 8) * KB_A + sw128(r, c % 8), v);
#pragma unroll
          for (int e = 0; e < 16 / ESZ; ++e) s += v[e];
        }
      }
      const float mx = s * (1.f / DX);
      float q2 = 0.f;
      if (NXS == 0) {
#pragma unroll
        for (int c = 0; c < DX / 4; ++c) {
          const float4 v = xrow < M ? __ldg(reinterpret_cast<const float4*>(xr) + c) : make_float4(mx, mx, mx, mx);
          q2 += ((v.x - mx) * (v.x - mx) + (v.y - mx) * (v.y - mx)) + ((v.z - mx) * (v.z - mx) + (v.w - mx) * (v.w - mx));
        }
      } else {
#pragma unroll
        for (int c = 0; c < DX * ESZ / 16; ++c) {
          float v[16 / ESZ];
          rd16(xbase + (c / 8) * KB_A + sw128(r, c % 8), v);
#pragma unroll
          for (int e = 0; e < 16 / ESZ; ++e) q2 += (v[e] - mx) * (v[e] - mx);
        }
      }
      const float rx = rsqrtf(q2 * (1.f / DX) + eps_x);
      ST[buf * 128 + r] = make_float2(rx, mx * rx);
      // LN(cond) * wc in place, two-pass statistics: bf16 keeps the row in registers (read once), fp32 re-reads it from shared memory
      const uint32_t abase = su + O_A + ab * A_BYTES;
      constexpr int NCH = KC * ESZ / 16;
      constexpr bool KEEP = !F32;
      uint4 raw[KEEP ? NCH : 1];
      s = 0.f;
#pragma unroll
      for (int c = 0; c < NCH; ++c) {
        const uint4 u = lds128(abase + (c / 8) * KB_A + sw128(r, c % 8));
        if (KEEP) raw[KEEP ? c : 0] = u;
        float v[16 / ESZ];
        unpack16(u, v);
#pragma unroll
        for (int e = 0; e < 16 / ESZ; ++e) s += v[e];
      }
      const float mc = s * (1.f / KC);
      q2 = 0.f;
#pragma unroll
      for (int c = 0; c < NCH; ++c) {
        float v[16 / ESZ];
        unpack16(KEEP ? raw[KEEP ? c : 0] : lds128(abase + (c / 8) * KB_A + sw128(r, c % 8)), v);
#pragma unroll
        for (int e = 0; e < 16 / ESZ; ++e) q2 += (v[e] - mc) * (v[e] - mc);
      }
      const float rc = rsqrtf(q2 * (1.f / KC) + eps_c);
#pragma unroll
      for (int c = 0; c < NCH; ++c) {
        const uint32_t a = abase + (c / 8) * KB_A + sw128(r, c % 8);
        float v[16 / ESZ];
        unpack16(KEEP ? raw[KEEP ? c : 0] : lds128(a), v);
        const float4* wp = reinterpret_cast<const float4*>(WCs + c * (16 / ESZ));
#pragma unroll
        for (int e4 = 0; e4 < 16 / ESZ / 4; ++e4) {
          const float4 w = wp[e4];
          v[4 * e4] = (v[4 * e4] - mc) * rc * w.x;
          v[4 * e4 + 1] = (v[4 * e4 + 1] - mc) * rc * w.y;
          v[4 * e4 + 2] = (v[4 * e4 + 2] - mc) * rc * w.z;
          v[4 * e4 + 3] = (v[4 * e4 + 3] - mc) * rc * w.w;
        }
        wr16(a, v);
      }
      fence_proxy_async();                              // the generic-proxy stores of A, visible to the MMA (async proxy)
      __syncwarp();
      if (lane == 0) mbar_arrive(&B.afull[ab]);
    }
  } else {
    // ------------------------------------------------------------------------------------------------ epilogue (warps 6-9)
    const int q4 = warp & 3;
    const int rl = q4 * 32 + lane;
    // the gate bias of this CTA's columns into shared memory (it was a global load per element in the loop below): the four epilogue
    // warps meet at named barrier 2, long before the first accumulator is ready
    for (int i = (warp - 6) * 32 + lane; i < BN; i += 128) BSs[i] = __ldg(BS + n0 + i);
    named_bar_sync(2, 128);
    int tc = 0;
    for (int t = blockIdx.x; t < tiles; t += gridDim.x, ++tc) {
      const int buf = tc & 1;
      const uint32_t ph = (tc >> 1) & 1;
#if F32
      // x read and y written through this warp's slab as whole 128-B row segments: the 32 x 32 block (rows m0 .. m0 + 31, columns
      // n0 + 32 c ..) is eight 16-B loads a lane, lane l of load i taking row 4 i + l / 8, chunk l % 8 (each instruction four full row
      // segments); in the slab (128-B swizzle) every lane then reads and rewrites its own row, and the block leaves the same way.
      // The thread-per-row global accesses this replaces were one cache line per lane and instruction (6.8 us of epilogue a tile at [20480, 128]).
      const long m0 = (long)(t / NT) * 128 + q4 * 32;
      const uint32_t slab = su + O_SL + (uint32_t)(warp - 6) * SLAB;
      uint4 xq[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {                     // the first block, in flight under the accumulator wait
        const long rr = m0 + 4 * i + (lane >> 3);
        xq[i] = rr < M ? ldg128(reinterpret_cast<const float*>(X) + rr * DX + n0 + 4 * (lane & 7)) : make_uint4(0, 0, 0, 0);
      }
      mbar_wait(&B.accf[buf], ph);
      tc_fence_after();
      const float2 st = ST[buf * 128 + rl];
      const uint32_t tb = tmem + ((uint32_t)(q4 * 32) << 16) + buf * TCOLS;
#pragma unroll 1
      for (int c = 0; c < BN / 32; ++c) {
#pragma unroll
        for (int i = 0; i < 8; ++i) sts128(slab + sw128((uint32_t)(4 * i + (lane >> 3)), (uint32_t)(lane & 7)), xq[i]);
        if (c + 1 < BN / 32) {                          // the next block's x in flight under this block's math
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const long rr = m0 + 4 * i + (lane >> 3);
            xq[i] = rr < M ? ldg128(reinterpret_cast<const float*>(X) + rr * DX + n0 + (c + 1) * 32 + 4 * (lane & 7)) : make_uint4(0, 0, 0, 0);
          }
        }
        uint32_t acc[NACC][32];
        tmem_ld32(tb + c * 32, acc[0]);
        tmem_ld32(tb + BN + c * 32, acc[1]);
        __syncwarp();
        float o[32];
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const uint4 v = lds128(slab + sw128((uint32_t)lane, (uint32_t)q));
          o[4 * q] = __uint_as_float(v.x); o[4 * q + 1] = __uint_as_float(v.y); o[4 * q + 2] = __uint_as_float(v.z); o[4 * q + 3] = __uint_as_float(v.w);
        }
        tmem_wait_ld();
#pragma unroll
        for (int e = 0; e < 32; ++e) {
          const float sp = __uint_as_float(acc[1][e]) + BSs[c * 32 + e];
          const float y = sigmoid_kit(sp) * (o[e] * st.x - st.y) + __uint_as_float(acc[0][e]);
          o[e] = rna ? tf32r(y) : y;
        }
#pragma unroll
        for (int q = 0; q < 8; ++q)
          sts128(slab + sw128((uint32_t)lane, (uint32_t)q), make_uint4(__float_as_uint(o[4 * q]), __float_as_uint(o[4 * q + 1]),
                                                                      __float_as_uint(o[4 * q + 2]), __float_as_uint(o[4 * q + 3])));
        __syncwarp();
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int rr = 4 * i + (lane >> 3);
          const uint4 v = lds128(slab + sw128((uint32_t)rr, (uint32_t)(lane & 7)));
          if (m0 + rr < M) stg128(reinterpret_cast<float*>(Y) + (m0 + rr) * DX + n0 + c * 32 + 4 * (lane & 7), v);
        }
        __syncwarp();                                   // the slab is rewritten by the next block
      }
#else
      const long row = (long)(t / NT) * 128 + rl;
      const bool ok = row < M;
      const uint32_t xbase = su + O_X + buf * X_BYTES;  // this tile's x rows, staged by the TMA warp
      mbar_wait(&B.accf[buf], ph);
      tc_fence_after();
      const float2 st = ST[buf * 128 + rl];
      const uint32_t tb = tmem + ((uint32_t)(q4 * 32) << 16) + buf * TCOLS;
#pragma unroll 1
      for (int c = 0; c < BN / 32; ++c) {
        float o[32];
#pragma unroll
        for (int g = 0; g < 32 * ESZ / 16; ++g) {       // the 32 columns n0 + 32 c .. as 16-B chunks of the swizzled staging
          const int k = n0 + c * 32 + g * (16 / ESZ);
          float v[16 / ESZ];
          rd16(xbase + (k / BKE) * KB_A + sw128((uint32_t)rl, (uint32_t)((k % BKE) * ESZ / 16)), v);
#pragma unroll
          for (int e = 0; e < 16 / ESZ; ++e) o[g * (16 / ESZ) + e] = v[e];
        }
        uint32_t acc[NACC][32];
        tmem_ld32(tb + c * 32, acc[0]);
        tmem_ld32(tb + BN + c * 32, acc[1]);
        tmem_wait_ld();
#pragma unroll
        for (int e = 0; e < 32; ++e) {
          const float sp = __uint_as_float(acc[1][e]) + BSs[c * 32 + e];
          o[e] = sigmoid_kit(sp) * (o[e] * st.x - st.y) + __uint_as_float(acc[0][e]);
        }
        if (ok) store32(Y, row * DX + n0 + c * 32, o);
      }
#endif
      tc_fence_before();
      __syncwarp();
      if (lane == 0) { mbar_arrive(&B.acce[buf]); mbar_arrive(&B.sempty[buf]); }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 512); }
}

extern "C" __device__ int transition_smem_bytes = SMEM_BYTES;
