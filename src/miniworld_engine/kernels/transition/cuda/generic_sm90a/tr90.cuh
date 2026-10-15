// tr90.cuh -- generic sm_90a Transition kernels: any channel width D (multiple of 64), any hidden width (multiple of 128),
// bf16 (wgmma k16) or fp32 activations on TF32 tensor cores (wgmma k8), any row count (TMA zero-fills / clips the ragged tail,
// the row kernels and direct stores guard on the row index).
// SPDX-License-Identifier: Apache-2.0
//
//   forward    ln_fwd          x -> xn (T), rstd, c1                      (warp per row)
//              gemm<SWIGLU>    h = silu(xn Wa^T) * (xn Wb^T)    [M, HID]
//              gemm<RES>       y = h Ws^T + x                              (residual in the epilogue)
//   backward   gate_kernel     dh = dy Ws, [a|b] = xn [Wa|Wb]^T, SwiGLU backward -> h [M,HID], dAB [M,2 HID]  (one pass)
//              gemm<F32>       dxn = dAB [Wa;Wb]  (fp32 out)
//              ln_bwd          dxn, x, stats -> dx (+ dy residual), per-CTA dgamma / dbeta partials
//              (dW = cuBLAS GEMMs with fp32 output, issued by the caller)
//
// GEMM structure (gemm_kernel / gate_kernel): persistent grid, one producer warpgroup (TMA, 128B-swizzled K-major slabs through an
// NSTAGE mbarrier ring) and two consumer warpgroups (64 rows each, wgmma SS).  Every operand is K-major, which is what TF32 wgmma
// requires; the callers hand it transposed weight copies where a GEMM contracts over a weight's row index.
#pragma once
#include "tmn_ptx.cuh"
#include "wgmma_ops.cuh"

namespace tr90 {
using namespace tmn;

template <typename T> struct Tt;
template <> struct Tt<__nv_bfloat16> {
  static constexpr int ES = 2, BK = 64;
  static __device__ __forceinline__ float2 ld2(const __nv_bfloat16* p) {
    uint32_t v = *reinterpret_cast<const uint32_t*>(p);
    return make_float2(bf16lo(v), bf16hi(v));
  }
  static __device__ __forceinline__ void st2(__nv_bfloat16* p, float a, float b) { *reinterpret_cast<uint32_t*>(p) = pack_bf16(a, b); }
  static __device__ __forceinline__ void st2m(__nv_bfloat16* p, float a, float b) { st2(p, a, b); }          // operand of a following MMA
  static __device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
  static __device__ __forceinline__ __nv_bfloat16 from_f(float v) { return __float2bfloat16_rn(v); }
  static __device__ __forceinline__ float rnaf(float v) { return v; }
};
template <> struct Tt<float> {
  static constexpr int ES = 4, BK = 32;
  static __device__ __forceinline__ float2 ld2(const float* p) { return *reinterpret_cast<const float2*>(p); }
  static __device__ __forceinline__ void st2(float* p, float a, float b) { *reinterpret_cast<float2*>(p) = make_float2(a, b); }
  // An fp32 tensor that a TF32 MMA reads next is stored rounded to nearest TF32 (the tensor core would truncate it: a one-sided
  // error that adds up to 4x the rounding error of cuBLAS TF32).
  static __device__ __forceinline__ float rna(float x) { uint32_t u; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(u) : "f"(x)); return __uint_as_float(u); }
  static __device__ __forceinline__ void st2m(float* p, float a, float b) { st2(p, rna(a), rna(b)); }
  static __device__ __forceinline__ float to_f(float v) { return v; }
  static __device__ __forceinline__ float from_f(float v) { return v; }
  static __device__ __forceinline__ float rnaf(float v) { return rna(v); }
};

enum Mode { MODE_RES = 0, MODE_OUT = 1, MODE_SWIGLU = 2, MODE_SWIGLU_SAVE = 3, MODE_GATE = 4 };

constexpr int BM = 128;
constexpr int SMEM_LIMIT = 227 * 1024;
// Two CTAs per SM (MINB = 2): one CTA's epilogue overlaps the other's mainloop.  ~113 KB of shared memory and 96 consumer registers each.
constexpr int smem_limit(int minb) { return minb == 2 ? 113 * 1024 : SMEM_LIMIT; }

__device__ __forceinline__ uint64_t dsc(uint32_t addr) { return smem_desc(addr, 16, 1024, 1); }
__device__ __forceinline__ float sigmoidf_(float x) { return rcpf(1.f + ex2f(-1.4426950408889634f * x)); }


// bf16 outputs are staged through shared memory and written by TMA: a thread's wgmma fragment holds 2 contiguous bf16 (4 B), a quad of
// threads 16 B of a row -- half a DRAM sector per store, which costs 5x the kernel on the gate stage.  A [64 rows][64 cols] bf16 block is
// 8 KB in the layout of a {64, 64} TMA box with SWIZZLE_128B (16-byte chunk ^= row & 7, conflict-free for the fragment writes).  fp32 / TF32
// outputs write 8 B per thread = a whole 32 B sector per quad, and are stored directly.
__device__ __forceinline__ uint32_t stg_addr(uint32_t base, int blk, int r, int c) {   // c: even column inside the 64-wide block
  const uint32_t cb = (uint32_t)c * 2;
  return base + blk * 8192 + r * 128 + ((((cb >> 4) ^ (r & 7)) << 4) | (cb & 15));
}
__device__ __forceinline__ uint32_t lds32_(uint32_t a) { uint32_t v; asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ void tma_store_2d(const CUtensorMap* map, uint32_t src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];" :: "l"(map), "r"(src), "r"(c0), "r"(c1) : "memory");
}

// ---- programmatic dependent launch: a kernel launched with the serialization attribute may start (barrier init, descriptors) while its
// predecessor finishes; `pdl_wait` blocks until the predecessor has completed and its writes are visible (a no-op in an ordinary launch).
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }

// ---- thread-block clusters (2 CTAs = two adjacent 128-row tiles of the same output columns share the B tile) ----
__device__ __forceinline__ uint32_t cluster_ctarank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;\n" ::: "memory");
}
// TMA load whose bytes land at the same smem offset of every CTA in `mask` and complete_tx on the barrier at that offset in each.
__device__ __forceinline__ void tma_load_2d_mc(uint32_t dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;\n"
               :: "r"(dst), "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "h"(mask) : "memory");
}
__device__ __forceinline__ void mbar_arrive_cta(uint64_t* bar, uint32_t cta) {      // arrive on the barrier at the same offset in CTA `cta`
  asm volatile("{\n .reg .b32 ra;\n mapa.shared::cluster.u32 ra, %0, %1;\n mbarrier.arrive.shared::cluster.b64 _, [ra];\n}\n"
               :: "r"(smem_u32(bar)), "r"(cta) : "memory");
}

template <int BN, int ES, int MODE, int LIM = SMEM_LIMIT, int ARES = 0> struct GemmCfg {
  static constexpr bool SW = MODE == MODE_SWIGLU || MODE == MODE_SWIGLU_SAVE;
  static constexpr int NW = SW ? BN / 2 : BN;                                  // output columns per tile
  static constexpr bool STAGED = ES == 2;
  static_assert(STAGED || (MODE != MODE_SWIGLU_SAVE && MODE != MODE_GATE), "saved-a/b modes are bf16 only");
  static constexpr int NSLOT = (MODE == MODE_GATE || MODE == MODE_SWIGLU_SAVE) ? 2 : 1;   // GATE: a, b (then dA, dB in place)
  static constexpr int STG = STAGED ? NSLOT * 64 * NW * 2 : 0;                 // staging bytes per consumer warpgroup
  // ARES > 0 (LayerNorm-fused expand): the CTA's whole [BM][D] x tile is resident (ARES bytes at the start of shared memory, as
  // D / 64 slabs of [BM][64]); the stage ring then carries B tiles only.
  static constexpr int A_BYTES = ARES ? 0 : BM * 128, B_BYTES = BN * 128, STAGE = A_BYTES + B_BYTES;
  static constexpr int SB = ARES;                                              // base of the stage ring
  static constexpr int NS0 = (LIM - 1024 - ARES - 2 * STG - 256) / STAGE;
  static constexpr int NSTAGE = NS0 > 6 ? 6 : NS0;
  static constexpr int F_STG = SB + NSTAGE * STAGE, F_BAR = F_STG + 2 * STG;
  static constexpr int SMEM = F_BAR + (2 * NSTAGE + 4) * 8 + 64;
  static_assert(SMEM <= 231424, "shared memory budget");
};

// Arguments of the LayerNorm-fused expand (DD > 0): gamma / beta fp32 [D], the activation tensors the training forward saves.
struct LnArgs {
  const float* gamma;
  const float* beta;
  __nv_bfloat16* xn;       // [M][D] normalized x (written when save)
  float* rstd;             // [M]
  float* c1;               // [M]  mean * rstd
  float eps;
  int save;
};

// LayerNorm of this warpgroup's 64 rows of the resident x tile (KBA slabs of [128 rows][64 cols], 128 B swizzled), IN PLACE; the
// rows are the A operand of the wgmma that follows.  Warp w owns rows 16 w .. 16 w + 15, eight rows at a time (independent shuffle
// chains); statistics in fp32, two-pass, one bf16 rounding of x_hat -- the arithmetic of ln_fwd_kernel.
template <int KBA>
__device__ __forceinline__ void ln_resident(uint32_t abase, int cw, int warp, int lane, const float (&g)[KBA][2], const float (&bt)[KBA][2],
                                            const LnArgs& ln, int m0, int M) {
  constexpr float INV_D = 1.f / (KBA * 64);
  const uint32_t xcol = (uint32_t)((4 * lane) & 15), gran = (uint32_t)(4 * lane) >> 4;
#pragma unroll 1
  for (int half = 0; half < 2; ++half) {
    uint32_t v[8][KBA];
    float s8[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const uint32_t r = (uint32_t)(64 * cw + 16 * warp + 8 * half + u);
      float acc = 0.f;
#pragma unroll
      for (int k = 0; k < KBA; ++k) {
        v[u][k] = lds32_(abase + k * 16384 + r * 128 + (((gran ^ (r & 7u)) << 4) | xcol));
        acc += bf16lo(v[u][k]) + bf16hi(v[u][k]);
      }
      s8[u] = acc;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1)
#pragma unroll
      for (int u = 0; u < 8; ++u) s8[u] += __shfl_xor_sync(0xffffffffu, s8[u], o);
    float mean8[8], q8[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      mean8[u] = s8[u] * INV_D;
      float q = 0.f;
#pragma unroll
      for (int k = 0; k < KBA; ++k) {
        const float d0 = bf16lo(v[u][k]) - mean8[u], d1 = bf16hi(v[u][k]) - mean8[u];
        q += d0 * d0 + d1 * d1;
      }
      q8[u] = q;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1)
#pragma unroll
      for (int u = 0; u < 8; ++u) q8[u] += __shfl_xor_sync(0xffffffffu, q8[u], o);
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const uint32_t r = (uint32_t)(64 * cw + 16 * warp + 8 * half + u);
      const float mean = mean8[u], rs = rsqrtf(q8[u] * INV_D + ln.eps);
      const int gr = m0 + (int)r;
#pragma unroll
      for (int k = 0; k < KBA; ++k) {
        const uint32_t o = pack_bf16((bf16lo(v[u][k]) - mean) * rs * g[k][0] + bt[k][0],
                                     (bf16hi(v[u][k]) - mean) * rs * g[k][1] + bt[k][1]);
        sts32(abase + k * 16384 + r * 128 + (((gran ^ (r & 7u)) << 4) | xcol), o);
        if (ln.save && gr < M) *reinterpret_cast<uint32_t*>(ln.xn + (size_t)gr * (KBA * 64) + k * 64 + 2 * lane) = o;
      }
      if (ln.save && gr < M && lane == 0) { ln.rstd[gr] = rs; ln.c1[gr] = mean * rs; }
    }
  }
}

// ------------------------------------------------------------------------------------------------------------------------------
// out = A B^T (+ epilogue).  A [M][K] (row-major, K-major), B [N][K] (nn.Linear weight layout).
//   MODE_RES    : O (T) = acc + X (T)                 B tile = rows n0 .. n0+BN of mB
//   MODE_OUT    : O (T) = acc
//   MODE_SWIGLU : O (T) [M][N] = silu(a) * b          B tile = rows j0 .. j0+BN/2 of mB (a) then of mB2 (b); N = hidden width
//   MODE_SWIGLU_SAVE : as SWIGLU, and mO2 stores [a | b] ([M][2N], a in columns 0..N) for the backward
//   BT (bf16, MODE_OUT / MODE_GATE): the B operand is the weight as stored, [K][N] with N contiguous (MN-major wgmma operand, tnspB = 1):
//                 no transposed weight copy.  MODE_OUT reads K = 2 HID rows as mB (first half) then mB2 (second half).
//   MODE_GATE   : acc = dh = dy Ws; a, b are loaded from mX ([M][2N] as saved), the SwiGLU backward stores dA / dB to mO ([M][2N])
//                 (h is not needed: the forward saved it from its fp32 accumulators)
// mX / mO: TMA maps of X and O (bf16 staged epilogue only).
template <typename T, int BN, int MODE, bool BT = false, int MINB = 1, bool CL = false, int DD = 0>
__global__ void __launch_bounds__(384, MINB)
gemm_kernel(const __grid_constant__ CUtensorMap mA, const __grid_constant__ CUtensorMap mB, const __grid_constant__ CUtensorMap mB2,
            const __grid_constant__ CUtensorMap mX, const __grid_constant__ CUtensorMap mO, const __grid_constant__ CUtensorMap mO2,
            int M, int N, int K, const T* __restrict__ X, void* __restrict__ O, const LnArgs ln) {
  using C = GemmCfg<BN, Tt<T>::ES, MODE, smem_limit(MINB), DD * BM * 128>;   // DD = D / 64 slabs of [BM][64] when LayerNorm-fused
  static_assert(DD == 0 || (MODE == MODE_SWIGLU || MODE == MODE_SWIGLU_SAVE), "LayerNorm fusion is the expand GEMM's");
  static_assert(DD == 0 || (!CL && !BT && Tt<T>::ES == 2), "LayerNorm fusion: bf16, no cluster, K-major B");
  constexpr int BK = Tt<T>::BK, NSTAGE = C::NSTAGE, NACC = BN / 2, NW = C::NW;
  constexpr bool STAGED = C::STAGED;
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + C::F_BAR);
  uint64_t* empty = full + NSTAGE;
  uint64_t* xfull = empty + NSTAGE;
  uint64_t* xa_full = xfull + 2;                                      // LayerNorm fusion: resident x tile arrived
  uint64_t* a_free = xfull + 3;                                       //   ... and released by both consumer warpgroups
  const int NT = N / NW, mtiles = (M + BM - 1) / BM, KB = K / BK;
  // cluster mode: CTA pair (rank 0, 1) = tiles (2 p, 2 p + 1) of the same output columns; one work unit per pair
  const uint32_t rank = CL ? cluster_ctarank() : 0u;
  const int units = CL ? ((mtiles + 1) / 2) * NT : mtiles * NT;
  // tile range of this CTA: strided (round robin) normally; contiguous in tile order (all the n tiles of an x tile together) when
  // LayerNorm-fused, so the resident x tile is loaded and normalized once per m tile and CTA
  int first = CL ? (int)(blockIdx.x >> 1) : (int)blockIdx.x, step = CL ? (int)(gridDim.x >> 1) : (int)gridDim.x;
  int last = units;
  if (DD > 0) {
    first = (int)(((long long)blockIdx.x * units) / gridDim.x);
    last = (int)(((long long)(blockIdx.x + 1) * units) / gridDim.x);
    step = 1;
  }
  if (tid == 0) {
    for (int s = 0; s < NSTAGE; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, CL ? 4 : 2); }
    mbar_init(xfull, 1); mbar_init(xfull + 1, 1);
    mbar_init(xa_full, 1); mbar_init(a_free, 2);
    fence_barrier_init();
  }
  if (CL) cluster_sync_all(); else __syncthreads();
  pdl_trigger();
  pdl_wait();                                                          // everything below reads the predecessor's output
  if (wg == 0) {                                                       // ---------------------------------------- producer
    setmaxnreg_dec<40>();
    if (tid != 0) {
      if (CL) cluster_sync_all();                                      // keep the CTA resident until the peer is done with our smem
      return;
    }
    uint32_t it = 0, groups = 0;
    int cur_m = -1;
    for (int t = first; t < last; t += step) {
      const int m0 = CL ? ((t / NT) * 2 + (int)rank) * BM : (t / NT) * BM, n0 = (t % NT) * NW;
      if (DD > 0 && t / NT != cur_m) {                                 // a new x tile: bring it in once for all its n tiles
        if (cur_m >= 0) mbar_wait(a_free, (groups - 1) & 1u);          // both consumer warpgroups are done with the previous one
        mbar_arrive_expect_tx(xa_full, DD * 16384);
        for (int k = 0; k < DD; ++k) tma_load_2d(sm + k * 16384, &mA, xa_full, k * 64, m0);
        cur_m = t / NT;
        ++groups;
      }
      for (int kb = 0; kb < KB; ++kb, ++it) {
        const int s = (int)(it % NSTAGE);
        if (it >= NSTAGE) mbar_wait(empty + s, ((it / NSTAGE) - 1) & 1u);
        uint8_t* st = sm + C::SB + s * C::STAGE;
        mbar_arrive_expect_tx(full + s, C::STAGE);
        if (DD == 0) tma_load_2d(st, &mA, full + s, kb * BK, m0);
        if (CL) {                                                      // B tile: each CTA loads its half and multicasts it to both
          const uint32_t b_dst = su + C::SB + s * C::STAGE + C::A_BYTES;
          if (C::SW) {
            tma_load_2d_mc(b_dst + rank * (BN / 2) * 128, rank ? &mB2 : &mB, full + s, kb * BK, n0, 3);
          } else if (BT) {
            const bool second = MODE == MODE_OUT && kb >= KB / 2;
            const int kr = (second ? kb - KB / 2 : kb) * BK;
            for (int q = (int)rank; q < BN / 64 || (q == 0 && rank == 0); q += 2)
              tma_load_2d_mc(b_dst + q * 8192, second ? &mB2 : &mB, full + s, n0 + 64 * q, kr, 3);
          } else {
            tma_load_2d_mc(b_dst + rank * (BN / 2) * 128, &mB, full + s, kb * BK, n0 + (int)rank * (BN / 2), 3);
          }
        } else if (C::SW) {
          tma_load_2d(st + C::A_BYTES, &mB, full + s, kb * BK, n0);
          tma_load_2d(st + C::A_BYTES + (BN / 2) * 128, &mB2, full + s, kb * BK, n0);
        } else if (BT) {
          const bool second = MODE == MODE_OUT && kb >= KB / 2;
          const int kr = (second ? kb - KB / 2 : kb) * BK;
          for (int q = 0; q < BN / 64; ++q) tma_load_2d(st + C::A_BYTES + q * 8192, second ? &mB2 : &mB, full + s, n0 + 64 * q, kr);
        } else {
          tma_load_2d(st + C::A_BYTES, &mB, full + s, kb * BK, n0);
        }
      }
    }
    if (CL) cluster_sync_all();
    return;
  }
  setmaxnreg_inc<MINB == 2 ? 96 : 232>();                                               // ---------------------------------------- consumers
  const int cw = wg - 1;
  const uint32_t stg = su + C::F_STG + cw * C::STG;
  uint32_t it = 0, tl = 0, cg = 0;
  int cur_m = -1;
  float gam[DD > 0 ? DD : 1][2], bet[DD > 0 ? DD : 1][2];            // this lane's LayerNorm affine, per 64-column slab
  if constexpr (DD > 0) {
#pragma unroll
    for (int k = 0; k < DD; ++k) {
      const float2 g2 = *reinterpret_cast<const float2*>(ln.gamma + k * 64 + 2 * lane);
      const float2 b2 = *reinterpret_cast<const float2*>(ln.beta + k * 64 + 2 * lane);
      gam[k][0] = g2.x; gam[k][1] = g2.y; bet[k][0] = b2.x; bet[k][1] = b2.y;
    }
  }
  for (int t = first; t < last; t += step, ++tl) {
    const int m0 = CL ? ((t / NT) * 2 + (int)rank) * BM : (t / NT) * BM, n0 = (t % NT) * NW;
    if (STAGED && MODE == MODE_RES && wtid == 0) {                     // residual tile -> this warpgroup's staging, during the mainloop
      if (tl > 0) tma_store_wait_read<0>();
      mbar_arrive_expect_tx(xfull + cw, C::STG);
      for (int q = 0; q < NW / 64; ++q) tma_load_2d(sm + C::F_STG + cw * C::STG + q * 8192, &mX, xfull + cw, n0 + 64 * q, m0 + 64 * cw);
    }
    if (MODE == MODE_GATE && wtid == 0) {                              // saved a | b tiles -> staging slots 0 / 1, during the mainloop
      if (tl > 0) tma_store_wait_read<0>();
      mbar_arrive_expect_tx(xfull + cw, 2 * (NW / 64) * 8192);
      for (int q = 0; q < NW / 64; ++q) {
        tma_load_2d(sm + C::F_STG + cw * C::STG + q * 8192, &mX, xfull + cw, n0 + 64 * q, m0 + 64 * cw);
        tma_load_2d(sm + C::F_STG + cw * C::STG + (NW / 64 + q) * 8192, &mX, xfull + cw, N + n0 + 64 * q, m0 + 64 * cw);
      }
    }
    if constexpr (DD > 0) {
      if (t / NT != cur_m) {                                           // LayerNorm of this x tile, in place (once per m tile)
        mbar_wait(xa_full, cg & 1u);
        ++cg;
        cur_m = t / NT;
        ln_resident<DD>(su, cw, warp, lane, gam, bet, ln, m0, M);
        fence_proxy_async();                                           // the generic-proxy stores are the wgmma's operand
        named_bar_sync(1 + cw, 128);
      }
    }
    float acc[NACC];
#pragma unroll
    for (int e = 0; e < NACC; ++e) acc[e] = 0.f;
    int prev = -1;
    for (int kb = 0; kb < KB; ++kb, ++it) {
      const int s = (int)(it % NSTAGE);
      mbar_wait(full + s, (it / NSTAGE) & 1u);
      const uint32_t a0 = (DD > 0 ? su + kb * 16384 : su + C::SB + s * C::STAGE) + cw * 8192, b0 = su + C::SB + s * C::STAGE + C::A_BYTES;
      fence_regs(acc); wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        if (BT) mmaT<BN>(acc, dsc(a0 + ks * 32), smem_desc(b0 + ks * 2048, 8192, 1024, 1));   // MN-major B: LBO = 64-column chunk, SBO = 8 K rows
        else mma<T, BN>(acc, dsc(a0 + ks * 32), dsc(b0 + ks * 32));
      }
      wgmma_commit();
      if (prev >= 0) {
        wgmma_wait<1>(); fence_regs(acc);
        if (wtid == 0) { if (CL) { mbar_arrive_cta(empty + prev, 0); mbar_arrive_cta(empty + prev, 1); } else mbar_arrive(empty + prev); }
      }
      prev = s;
    }
    wgmma_wait<0>(); fence_regs(acc);
    if (wtid == 0) { if (CL) { mbar_arrive_cta(empty + prev, 0); mbar_arrive_cta(empty + prev, 1); } else mbar_arrive(empty + prev); }
    if (DD > 0 && wtid == 0 && (t + 1 >= last || (t + 1) / NT != t / NT)) mbar_arrive(a_free);   // last n tile of this x tile
    // ------------------------------------------------------------------------------------------------ epilogue
    const int rl = 16 * warp + (lane >> 2), c0 = 2 * (lane & 3), r0 = m0 + 64 * cw + rl;
    if (STAGED) {
      if (MODE == MODE_RES) {
        mbar_wait(xfull + cw, tl & 1u);
#pragma unroll
        for (int g = 0; g < BN / 8; ++g)
#pragma unroll
          for (int hf = 0; hf < 2; ++hf) {
            const uint32_t ad = stg_addr(stg, g >> 3, rl + 8 * hf, (8 * g + c0) & 63);
            const uint32_t xr = lds32_(ad);
            sts32(ad, pack_bf16(bf16lo(xr) + acc[4 * g + 2 * hf], bf16hi(xr) + acc[4 * g + 2 * hf + 1]));
          }
      } else if (MODE == MODE_SWIGLU_SAVE) {
        constexpr uint32_t SLOT = (NW / 64) * 8192;
        if (wtid == 0 && tl > 0) tma_store_wait_read<0>();
        named_bar_sync(1 + cw, 128);
#pragma unroll
        for (int g = 0; g < BN / 16; ++g)
#pragma unroll
          for (int hf = 0; hf < 2; ++hf) {
            const uint32_t ad = stg_addr(stg, g >> 3, rl + 8 * hf, (8 * g + c0) & 63);
            sts32(ad, pack_bf16(acc[4 * g + 2 * hf], acc[4 * g + 2 * hf + 1]));
            sts32(ad + SLOT, pack_bf16(acc[4 * (g + BN / 16) + 2 * hf], acc[4 * (g + BN / 16) + 2 * hf + 1]));
          }
        fence_proxy_async();
        named_bar_sync(1 + cw, 128);
        if (wtid == 0) {
          for (int q = 0; q < NW / 64; ++q) {
            tma_store_2d(&mO2, stg + q * 8192, n0 + 64 * q, m0 + 64 * cw);
            tma_store_2d(&mO2, stg + SLOT + q * 8192, N + n0 + 64 * q, m0 + 64 * cw);
          }
          tma_store_commit();
          tma_store_wait_read<0>();
        }
        named_bar_sync(1 + cw, 128);
#pragma unroll
        for (int g = 0; g < BN / 16; ++g)
#pragma unroll
          for (int hf = 0; hf < 2; ++hf) {
            const float a0_ = acc[4 * g + 2 * hf], a1_ = acc[4 * g + 2 * hf + 1];
            const float b0_ = acc[4 * (g + BN / 16) + 2 * hf], b1_ = acc[4 * (g + BN / 16) + 2 * hf + 1];
            sts32(stg_addr(stg, g >> 3, rl + 8 * hf, (8 * g + c0) & 63),
                  pack_bf16(a0_ * sigmoidf_(a0_) * b0_, a1_ * sigmoidf_(a1_) * b1_));
          }
      } else if (MODE == MODE_GATE) {
        constexpr uint32_t SLOT = (NW / 64) * 8192;
        mbar_wait(xfull + cw, tl & 1u);
#pragma unroll
        for (int g = 0; g < NW / 8; ++g)
#pragma unroll
          for (int hf = 0; hf < 2; ++hf) {
            const uint32_t ad = stg_addr(stg, g >> 3, rl + 8 * hf, (8 * g + c0) & 63);
            const uint32_t av = lds32_(ad), bv = lds32_(ad + SLOT);
            const float a2[2] = {bf16lo(av), bf16hi(av)}, b2[2] = {bf16lo(bv), bf16hi(bv)};
            float da[2], db[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
              const float a = a2[i], b = b2[i], d = acc[4 * g + 2 * hf + i];
              const float s = sigmoidf_(a);
              da[i] = d * b * s * (1.f + a * (1.f - s));
              db[i] = d * a * s;
            }
            sts32(ad, pack_bf16(da[0], da[1]));
            sts32(ad + SLOT, pack_bf16(db[0], db[1]));
          }
      } else {
        if (wtid == 0 && tl > 0) tma_store_wait_read<0>();
        named_bar_sync(1 + cw, 128);
        if (MODE == MODE_SWIGLU) {
#pragma unroll
          for (int g = 0; g < BN / 16; ++g)
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
              const float a0_ = acc[4 * g + 2 * hf], a1_ = acc[4 * g + 2 * hf + 1];
              const float b0_ = acc[4 * (g + BN / 16) + 2 * hf], b1_ = acc[4 * (g + BN / 16) + 2 * hf + 1];
              sts32(stg_addr(stg, g >> 3, rl + 8 * hf, (8 * g + c0) & 63),
                    pack_bf16(a0_ * sigmoidf_(a0_) * b0_, a1_ * sigmoidf_(a1_) * b1_));
            }
        } else {                                                       // OUT
#pragma unroll
          for (int g = 0; g < BN / 8; ++g)
#pragma unroll
            for (int hf = 0; hf < 2; ++hf)
              sts32(stg_addr(stg, g >> 3, rl + 8 * hf, (8 * g + c0) & 63), pack_bf16(acc[4 * g + 2 * hf], acc[4 * g + 2 * hf + 1]));
        }
      }
      fence_proxy_async();
      named_bar_sync(1 + cw, 128);
      if (wtid == 0) {
        if (MODE == MODE_GATE) {
          constexpr uint32_t SLOT = (NW / 64) * 8192;
          for (int q = 0; q < NW / 64; ++q) {
            tma_store_2d(&mO, stg + q * 8192, n0 + 64 * q, m0 + 64 * cw);
            tma_store_2d(&mO, stg + SLOT + q * 8192, N + n0 + 64 * q, m0 + 64 * cw);
          }
        } else {
          for (int q = 0; q < NW / 64; ++q) tma_store_2d(&mO, stg + q * 8192, n0 + 64 * q, m0 + 64 * cw);
        }
        tma_store_commit();
      }
    } else if (MODE == MODE_SWIGLU) {
      T* H = reinterpret_cast<T*>(O);
#pragma unroll
      for (int g = 0; g < BN / 16; ++g)
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const int r = r0 + 8 * hf;
          const float a0_ = acc[4 * g + 2 * hf], a1_ = acc[4 * g + 2 * hf + 1];
          const float b0_ = acc[4 * (g + BN / 16) + 2 * hf], b1_ = acc[4 * (g + BN / 16) + 2 * hf + 1];
          if (r < M) Tt<T>::st2m(H + (size_t)r * N + n0 + 8 * g + c0, a0_ * sigmoidf_(a0_) * b0_, a1_ * sigmoidf_(a1_) * b1_);
        }
    } else {
#pragma unroll
      for (int g = 0; g < BN / 8; ++g)
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const int r = r0 + 8 * hf, c = n0 + 8 * g + c0;
          const float v0 = acc[4 * g + 2 * hf], v1 = acc[4 * g + 2 * hf + 1];
          if (r < M) {
            if (MODE == MODE_RES) {
              const float2 xr = Tt<T>::ld2(X + (size_t)r * N + c);
              Tt<T>::st2(reinterpret_cast<T*>(O) + (size_t)r * N + c, v0 + xr.x, v1 + xr.y);
            } else {
              Tt<T>::st2(reinterpret_cast<T*>(O) + (size_t)r * N + c, v0, v1);
            }
          }
        }
    }
  }
  if (STAGED && wtid == 0) tma_store_wait_all();
  if (CL) cluster_sync_all();
}

// ------------------------------------------------------------------------------------------------------------------------------
// Backward gate: for a 128-row x 64-hidden tile, dh = dy Ws (Wst [HID][D] = Ws^T, K-major), a = xn Wa^T, b = xn Wb^T over K = D,
// then the SwiGLU backward.  Writes h [M][HID] and dAB [M][2 HID] (dA then dB).
template <typename T> struct GateCfg {
  static constexpr bool STAGED = Tt<T>::ES == 2;
  static constexpr int STG = STAGED ? 3 * 8192 : 0;                            // h, dA, dB tiles [64][64] bf16 per consumer warpgroup
  static constexpr int A_BYTES = BM * 128, W_BYTES = 64 * 128, STAGE = 2 * A_BYTES + 3 * W_BYTES;
  static constexpr int NS0 = (SMEM_LIMIT - 1024 - 2 * STG - 256) / STAGE;
  static constexpr int NSTAGE = NS0 > 4 ? 4 : NS0;
  static constexpr int F_STG = NSTAGE * STAGE, F_BAR = F_STG + 2 * STG;
  static constexpr int SMEM = F_BAR + 2 * NSTAGE * 8 + 64;
  static_assert(SMEM <= 231424, "shared memory budget");
};

template <typename T>
__global__ void __launch_bounds__(384, 1)
gate_kernel(const __grid_constant__ CUtensorMap mXn, const __grid_constant__ CUtensorMap mDy, const __grid_constant__ CUtensorMap mWa,
            const __grid_constant__ CUtensorMap mWb, const __grid_constant__ CUtensorMap mWst, const __grid_constant__ CUtensorMap mH,
            const __grid_constant__ CUtensorMap mDAB, int M, int D, int HID, T* __restrict__ Hout, T* __restrict__ DAB) {
  using C = GateCfg<T>;
  constexpr int BK = Tt<T>::BK, NSTAGE = C::NSTAGE;
  constexpr bool STAGED = C::STAGED;
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, warp = wtid >> 5, lane = tid & 31;
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + C::F_BAR);
  uint64_t* empty = full + NSTAGE;
  const int NT = HID / 64, tiles = ((M + BM - 1) / BM) * NT, KB = D / BK;
  if (tid == 0) {
    for (int s = 0; s < NSTAGE; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, 2); }
    fence_barrier_init();
  }
  __syncthreads();
  if (wg == 0) {
    setmaxnreg_dec<40>();
    if (tid != 0) return;
    uint32_t it = 0;
    for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
      const int m0 = (t / NT) * BM, j0 = (t % NT) * 64;
      for (int kb = 0; kb < KB; ++kb, ++it) {
        const int s = (int)(it % NSTAGE);
        if (it >= NSTAGE) mbar_wait(empty + s, ((it / NSTAGE) - 1) & 1u);
        uint8_t* st = sm + s * C::STAGE;
        mbar_arrive_expect_tx(full + s, C::STAGE);
        tma_load_2d(st, &mXn, full + s, kb * BK, m0);
        tma_load_2d(st + C::A_BYTES, &mDy, full + s, kb * BK, m0);
        tma_load_2d(st + 2 * C::A_BYTES, &mWa, full + s, kb * BK, j0);
        tma_load_2d(st + 2 * C::A_BYTES + C::W_BYTES, &mWb, full + s, kb * BK, j0);
        tma_load_2d(st + 2 * C::A_BYTES + 2 * C::W_BYTES, &mWst, full + s, kb * BK, j0);
      }
    }
    return;
  }
  setmaxnreg_inc<232>();
  const int cw = wg - 1;
  const uint32_t stg = su + C::F_STG + cw * C::STG;
  uint32_t it = 0, tl = 0;
  for (int t = blockIdx.x; t < tiles; t += gridDim.x, ++tl) {
    const int m0 = (t / NT) * BM, j0 = (t % NT) * 64;
    float aa[32], bb[32], dh[32];
#pragma unroll
    for (int e = 0; e < 32; ++e) { aa[e] = 0.f; bb[e] = 0.f; dh[e] = 0.f; }
    int prev = -1;
    for (int kb = 0; kb < KB; ++kb, ++it) {
      const int s = (int)(it % NSTAGE);
      mbar_wait(full + s, (it / NSTAGE) & 1u);
      const uint32_t base = su + s * C::STAGE;
      const uint32_t xa = base + cw * 8192, ya = base + C::A_BYTES + cw * 8192;
      const uint32_t wa = base + 2 * C::A_BYTES, wb = wa + C::W_BYTES, ws = wb + C::W_BYTES;
      fence_regs(aa); fence_regs(bb); fence_regs(dh); wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        mma<T, 64>(aa, dsc(xa + ks * 32), dsc(wa + ks * 32));
        mma<T, 64>(bb, dsc(xa + ks * 32), dsc(wb + ks * 32));
        mma<T, 64>(dh, dsc(ya + ks * 32), dsc(ws + ks * 32));
      }
      wgmma_commit();
      if (prev >= 0) { wgmma_wait<1>(); fence_regs(aa); fence_regs(bb); fence_regs(dh); if (wtid == 0) mbar_arrive(empty + prev); }
      prev = s;
    }
    wgmma_wait<0>(); fence_regs(aa); fence_regs(bb); fence_regs(dh);
    if (wtid == 0) mbar_arrive(empty + prev);
    const int rl = 16 * warp + (lane >> 2), c0 = 2 * (lane & 3), r0 = m0 + 64 * cw + rl;
    if (STAGED) {
      if (wtid == 0 && tl > 0) tma_store_wait_read<0>();
      named_bar_sync(1 + cw, 128);
    }
#pragma unroll
    for (int g = 0; g < 8; ++g) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        float h2[2], da[2], db[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const float a = aa[4 * g + 2 * hf + i], b = bb[4 * g + 2 * hf + i], d = dh[4 * g + 2 * hf + i];
          const float s = sigmoidf_(a), sw = a * s;
          h2[i] = sw * b;
          da[i] = d * b * s * (1.f + a * (1.f - s));
          db[i] = d * sw;
        }
        if (STAGED) {
          const uint32_t o = stg_addr(0, 0, rl + 8 * hf, 8 * g + c0);
          sts32(stg + o, pack_bf16(h2[0], h2[1]));
          sts32(stg + 8192 + o, pack_bf16(da[0], da[1]));
          sts32(stg + 16384 + o, pack_bf16(db[0], db[1]));
        } else {
          const int r = r0 + 8 * hf, c = j0 + 8 * g + c0;
          if (r < M) {
            Tt<T>::st2(Hout + (size_t)r * HID + c, h2[0], h2[1]);
            Tt<T>::st2m(DAB + (size_t)r * 2 * HID + c, da[0], da[1]);
            Tt<T>::st2m(DAB + (size_t)r * 2 * HID + HID + c, db[0], db[1]);
          }
        }
      }
    }
    if (STAGED) {
      fence_proxy_async();
      named_bar_sync(1 + cw, 128);
      if (wtid == 0) {
        tma_store_2d(&mH, stg, j0, m0 + 64 * cw);
        tma_store_2d(&mDAB, stg + 8192, j0, m0 + 64 * cw);
        tma_store_2d(&mDAB, stg + 16384, HID + j0, m0 + 64 * cw);
        tma_store_commit();
      }
    }
  }
  if (STAGED && wtid == 0) tma_store_wait_all();
}

// ------------------------------------------------------------------------------------------------------------------------------
// LayerNorm forward, warp per row.  Lane owns the column pairs (2 (lane + 32 i), +1), i < PER / 2.
template <typename T, int PER>
__device__ __forceinline__ void ln_fwd_body(int blk, const T* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ beta,
                                            T* __restrict__ xn, float* __restrict__ rstd_out, float* __restrict__ c1_out, int M, int D,
                                            float eps, int save_stats) {
  const int row = blk * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= M) return;
  float v[PER];
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < PER / 2; ++i) {
    const float2 p = Tt<T>::ld2(x + (size_t)row * D + 2 * (lane + 32 * i));
    v[2 * i] = p.x; v[2 * i + 1] = p.y; s += p.x + p.y;
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  const float mean = s / D;
  float q = 0.f;
#pragma unroll
  for (int i = 0; i < PER; ++i) { const float d = v[i] - mean; q += d * d; }
#pragma unroll
  for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
  const float rs = rsqrtf(q / D + eps);
#pragma unroll
  for (int i = 0; i < PER / 2; ++i) {
    const int c = 2 * (lane + 32 * i);
    const float2 g = *reinterpret_cast<const float2*>(gamma + c), b = *reinterpret_cast<const float2*>(beta + c);
    Tt<T>::st2m(xn + (size_t)row * D + c, (v[2 * i] - mean) * rs * g.x + b.x, (v[2 * i + 1] - mean) * rs * g.y + b.y);
  }
  if (save_stats && lane == 0) { rstd_out[row] = rs; c1_out[row] = mean * rs; }
}

template <typename T, int PER>
__global__ void __launch_bounds__(256)
ln_fwd_kernel(const T* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ beta, T* __restrict__ xn,
              float* __restrict__ rstd_out, float* __restrict__ c1_out, int M, int D, float eps, int save_stats) {
  pdl_trigger();
  ln_fwd_body<T, PER>(blockIdx.x, x, gamma, beta, xn, rstd_out, c1_out, M, D, eps, save_stats);
}

// LayerNorm backward + residual from the saved statistics: xhat = x rstd - c1, wdy = gamma dxn,
//   dx = rstd (wdy - mean(wdy) - xhat mean(wdy xhat)) + dy;  dgamma += dxn xhat;  dbeta += dxn.
// dxn is [M][D] in the activation dtype.  Persistent blocks of 8 warps; per-block partial dgamma / dbeta [gridDim.x][D] (deterministic, summed by the caller).
template <typename T, int PER>
__global__ void __launch_bounds__(256)
ln_bwd_kernel(const T* __restrict__ dxn, const T* __restrict__ x, const T* __restrict__ dy, const float* __restrict__ rstd,
              const float* __restrict__ c1, const float* __restrict__ gamma, T* __restrict__ dx, float* __restrict__ pdg,
              float* __restrict__ pdb, int M, int D) {
  __shared__ float red[8][PER * 32];
  pdl_trigger();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  float g[PER], dg[PER], db[PER];
#pragma unroll
  for (int i = 0; i < PER / 2; ++i) {
    const float2 p = *reinterpret_cast<const float2*>(gamma + 2 * (lane + 32 * i));
    g[2 * i] = p.x; g[2 * i + 1] = p.y;
  }
#pragma unroll
  for (int i = 0; i < PER; ++i) { dg[i] = 0.f; db[i] = 0.f; }
  for (int row = blockIdx.x * 8 + warp; row < M; row += gridDim.x * 8) {
    const float rs = rstd[row], cc = c1[row];
    float xh[PER], wd[PER], dn[PER];
    float sa = 0.f, sb = 0.f;
#pragma unroll
    for (int i = 0; i < PER / 2; ++i) {
      const int c = 2 * (lane + 32 * i);
      const float2 d2 = Tt<T>::ld2(dxn + (size_t)row * D + c);
      const float2 x2 = Tt<T>::ld2(x + (size_t)row * D + c);
      dn[2 * i] = d2.x; dn[2 * i + 1] = d2.y;
      xh[2 * i] = x2.x * rs - cc; xh[2 * i + 1] = x2.y * rs - cc;
      wd[2 * i] = g[2 * i] * d2.x; wd[2 * i + 1] = g[2 * i + 1] * d2.y;
      sa += wd[2 * i] * xh[2 * i] + wd[2 * i + 1] * xh[2 * i + 1];
      sb += wd[2 * i] + wd[2 * i + 1];
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) { sa += __shfl_xor_sync(0xffffffffu, sa, o); sb += __shfl_xor_sync(0xffffffffu, sb, o); }
    sa /= D; sb /= D;
#pragma unroll
    for (int i = 0; i < PER / 2; ++i) {
      const int c = 2 * (lane + 32 * i);
      const float2 y2 = Tt<T>::ld2(dy + (size_t)row * D + c);
      Tt<T>::st2(dx + (size_t)row * D + c, (wd[2 * i] - xh[2 * i] * sa - sb) * rs + y2.x,
                 (wd[2 * i + 1] - xh[2 * i + 1] * sa - sb) * rs + y2.y);
      dg[2 * i] += dn[2 * i] * xh[2 * i]; dg[2 * i + 1] += dn[2 * i + 1] * xh[2 * i + 1];
      db[2 * i] += dn[2 * i]; db[2 * i + 1] += dn[2 * i + 1];
    }
  }
  // block reduction of the 8 warps (fixed order: deterministic)
#pragma unroll
  for (int i = 0; i < PER / 2; ++i) *reinterpret_cast<float2*>(&red[warp][2 * (lane + 32 * i)]) = make_float2(dg[2 * i], dg[2 * i + 1]);
  __syncthreads();
  for (int c = threadIdx.x; c < D; c += 256) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) s += red[w][c];
    pdg[(size_t)blockIdx.x * D + c] = s;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < PER / 2; ++i) *reinterpret_cast<float2*>(&red[warp][2 * (lane + 32 * i)]) = make_float2(db[2 * i], db[2 * i + 1]);
  __syncthreads();
  for (int c = threadIdx.x; c < D; c += 256) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) s += red[w][c];
    pdb[(size_t)blockIdx.x * D + c] = s;
  }
}

}  // namespace tr90

namespace tr90 {

// ------------------------------------------------------------------------------------------------------------------------------
// Backward weight prep, one launch: wst = Ws^T [HID][D], wabt = [Wa;Wb]^T [D][2 HID] (the K-major B operands of the gate and d_xn
// GEMMs) and, for fp32 activations, TF32-rounded copies of Wa / Wb (the gate's B operands; every output is rounded to nearest TF32).
// 32x32 tiles through shared memory, blockIdx.z picks the job (bit z of `mask` enables it): 0 wst, 1 wabt left half (Wa), 2 wabt right half (Wb),
// 3 / 4 / 5 copies of Wa / Wb / Ws (rounded when `round`).
template <typename T>
__global__ void __launch_bounds__(256)
prep_bwd_kernel(const T* __restrict__ wa, const T* __restrict__ wb, const T* __restrict__ ws, T* __restrict__ wst, T* __restrict__ wabt,
                T* __restrict__ wa_r, T* __restrict__ wb_r, T* __restrict__ ws_r, int D, int HID, int mask, int round) {
  __shared__ float tile[32][33];
  const int job = blockIdx.z;
  if (!((mask >> job) & 1)) return;
  const T* src = (job == 0 || job == 5) ? ws : (job == 1 || job == 3) ? wa : wb;
  const bool transpose = job <= 2;
  const int R = (job == 0 || job == 5) ? D : HID, C = (job == 0 || job == 5) ? HID : D;   // src [R][C]
  const int bx = blockIdx.x, by = blockIdx.y;                                      // tile = src rows by*32.., cols bx*32..
  if (bx * 32 >= C || by * 32 >= R) return;
  for (int i = threadIdx.y; i < 32; i += 8) {
    float v = Tt<T>::to_f(src[(size_t)(by * 32 + i) * C + bx * 32 + threadIdx.x]);
    tile[i][threadIdx.x] = (round && Tt<T>::ES == 4) ? Tt<T>::rnaf(v) : v;
  }
  __syncthreads();
  if (transpose) {
    T* dst = job == 0 ? wst : wabt;
    const size_t ld = job == 0 ? D : 2 * (size_t)HID;
    const size_t col0 = job == 2 ? HID : 0;
    for (int i = threadIdx.y; i < 32; i += 8)                                       // dst row = src col, dst col = src row
      dst[(size_t)(bx * 32 + i) * ld + col0 + by * 32 + threadIdx.x] = Tt<T>::from_f(tile[threadIdx.x][i]);
  } else {
    T* dst = job == 3 ? wa_r : job == 4 ? wb_r : ws_r;
    for (int i = threadIdx.y; i < 32; i += 8)
      dst[(size_t)(by * 32 + i) * C + bx * 32 + threadIdx.x] = Tt<T>::from_f(tile[i][threadIdx.x]);
  }
}

// SwiGLU backward from a cuBLAS dh (bf16 only): h = silu(a) b, dA = dh b silu'(a), dB = dh silu(a); ab = [a | b] [M][2 HID], dh [M][HID].
// Writes dab [M][2 HID] (dA then dB); 8 hidden units (16 B) per thread.
__global__ void __launch_bounds__(256)
swiglu_bwd_kernel(const __nv_bfloat16* __restrict__ dh, const __nv_bfloat16* __restrict__ ab, __nv_bfloat16* __restrict__ dab,
                  long long nvec, int HID) {
  const int per_row = HID / 8;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < nvec; i += (long long)gridDim.x * blockDim.x) {
    const long long row = i / per_row;
    const int c = (int)(i - row * per_row) * 8;
    const uint4 dv = *reinterpret_cast<const uint4*>(dh + row * HID + c);
    const uint4 av = *reinterpret_cast<const uint4*>(ab + row * 2 * HID + c);
    const uint4 bv = *reinterpret_cast<const uint4*>(ab + row * 2 * HID + HID + c);
    const uint32_t* d4 = reinterpret_cast<const uint32_t*>(&dv);
    const uint32_t* a4 = reinterpret_cast<const uint32_t*>(&av);
    const uint32_t* b4 = reinterpret_cast<const uint32_t*>(&bv);
    uint32_t dao[4], dbo[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float dd[2] = {bf16lo(d4[k]), bf16hi(d4[k])}, aa[2] = {bf16lo(a4[k]), bf16hi(a4[k])}, bb[2] = {bf16lo(b4[k]), bf16hi(b4[k])};
      float da[2], db[2];
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const float sg = sigmoidf_(aa[e]);
        da[e] = dd[e] * bb[e] * sg * (1.f + aa[e] * (1.f - sg));
        db[e] = dd[e] * aa[e] * sg;
      }
      dao[k] = pack_bf16(da[0], da[1]); dbo[k] = pack_bf16(db[0], db[1]);
    }
    *reinterpret_cast<uint4*>(dab + row * 2 * HID + c) = make_uint4(dao[0], dao[1], dao[2], dao[3]);
    *reinterpret_cast<uint4*>(dab + row * 2 * HID + HID + c) = make_uint4(dbo[0], dbo[1], dbo[2], dbo[3]);
  }
}

// Sum the per-CTA dgamma / dbeta partials [G][D] x 2 in a fixed order: out_g, out_b [D].
__global__ void __launch_bounds__(1024)
reduce_cols_kernel(const float* __restrict__ pdg, const float* __restrict__ pdb, float* __restrict__ out_g, float* __restrict__ out_b, int G, int D) {
  __shared__ float red[2][32][32];
  pdl_wait();
  const int c = blockIdx.x * 32 + (threadIdx.x & 31), g = threadIdx.x >> 5;               // 32 row groups x 32 columns
  float a = 0.f, b = 0.f;
  if (c < D) {
#pragma unroll 4
    for (int r = g; r < G; r += 32) { a += pdg[(size_t)r * D + c]; b += pdb[(size_t)r * D + c]; }
  }
  red[0][g][threadIdx.x & 31] = a; red[1][g][threadIdx.x & 31] = b;
  __syncthreads();
  if (threadIdx.x < 64 && c < D) {
    const int which = threadIdx.x >> 5;
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < 32; ++w) s += red[which][w][threadIdx.x & 31];
    (which ? out_b : out_g)[blockIdx.x * 32 + (threadIdx.x & 31)] = s;
  }
}

}  // namespace tr90
