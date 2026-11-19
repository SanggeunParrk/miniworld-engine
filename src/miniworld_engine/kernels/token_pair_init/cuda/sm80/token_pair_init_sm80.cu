// token_pair_init_sm80.cu -- the A100 (sm_80) build of sm90/token_pair_init.cu: the backward, reduce and prep kernels and every host entry point are
// the H100 source compiled unchanged (that file stays byte-identical); only the forward kernel differs. Its H100 form fetches the class-row table
// by one bulk (TMA) copy completing on an mbarrier (sm_90 only); here the CTA's threads fetch it by cp.async, overlapping the same packing of the
// rows' bins, and the sums are the H100 kernel's in the same order. How: the H100 kernel's definition is renamed (tpi_fwd_h100, its inline asm
// emptied: it is never launched), and the host code's `tpi_fwd<<<...>>>` / cudaFuncSetAttribute(tpi_fwd, ...) bind to the kernel declared here.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>
#include <vector>

#ifdef MWE_SM120
#define TPI_FWD_WARPS 16   // the sm90 source's NWARP of this build (one CTA per SM on the RTX 5090: the 72 KB table)
#else
#define TPI_FWD_WARPS 8
#endif

__global__ void __launch_bounds__(32 * TPI_FWD_WARPS)
tpi_fwd(const float* __restrict__ left, const float* __restrict__ right, const float* __restrict__ tbl, const int* __restrict__ ids,
        const uint8_t* __restrict__ bond, float* __restrict__ out, int B, int L, int rmax, int smax, int R);

#ifdef MWE_SM120
// RTX 5090: a persistent grid (one CTA per SM, the class-row table fetched once per CTA) over work units of TPI_UNIT_COLS columns of one row of z
#define TPI_UNIT_COLS 64
// Short calls (B L rows <= the SM count: L 128 is 128 CTAs on 170 SMs) keep the one-row CTA of the previous kernel (one unit = a whole row, no second unit and
// no re-pack per CTA: the 64-column split cost 2 us at L 128); longer ones the persistent grid.
static inline unsigned tpi_grid_sm120(int B, int L) {
  const long long sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  if ((long long)B * L <= sms) return (unsigned)((long long)B * L);
  const long long units = (long long)B * L * ((L + TPI_UNIT_COLS - 1) / TPI_UNIT_COLS);
  return (unsigned)std::min<long long>(units, sms);
}
#define TPI_GRID(B, tiles, L) tpi_grid_sm120((B), (L))
#endif
#define tpi_fwd(...) tpi_fwd_h100(__VA_ARGS__)   // the definition only: the launches name it without a following '('
#define volatile(...) volatile("")               // the H100 forward's mbarrier / bulk-copy statements (the only inline asm of the file)
#include "../sm90/token_pair_init.cu"
#undef volatile
#undef tpi_fwd
static_assert(TPI_FWD_WARPS == NWARP, "the forward launch is 32 * NWARP threads");

#ifdef MWE_SM120
// Persistent forward (R = 1 rows per tile, the 5090's setting): each CTA fetches the table once, then walks the work units (row i, 64 columns) of the whole call in a
// grid-stride loop -- 384 one-row CTAs on 170 SMs are 2.3 waves of 3; units of 64 columns keep every SM busy to the end.  The sum order per element is the one-row kernel's.
__global__ void __launch_bounds__(32 * TPI_FWD_WARPS)
tpi_fwd(const float* __restrict__ left, const float* __restrict__ right, const float* __restrict__ tbl, const int* __restrict__ ids,
        const uint8_t* __restrict__ bond, float* __restrict__ out, int B, int L, int rmax, int smax, int R) {
  extern __shared__ __align__(128) float sm[];
  const int nrel = 2 * (2 * rmax + 2) + (2 * smax + 2) + 1, nbin = nrel + 2;
  uint32_t* pk = (uint32_t*)(sm + nbin * P);                        // [L]
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, BL = B * L;
  for (int c = threadIdx.x; c < nbin * P / 4; c += blockDim.x) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(sm + 4 * c);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(tbl + 4 * c));
  }
  asm volatile("cp.async.commit_group;\n" ::);
  const bool whole = (long long)B * L == (long long)gridDim.x;     // short call: one CTA a row (tpi_grid_sm120)
  const int ucols = whole ? L : TPI_UNIT_COLS;
  const int chunks = (L + ucols - 1) / ucols;
  const long long units = (long long)B * L * chunks;
  bool first = true;
  for (long long u = blockIdx.x; u < units; u += gridDim.x) {
    const int jc = (int)(u % chunks), bi = (int)(u / chunks), b = bi / L, i = bi % L, j0 = jc * ucols;
    const int nj = min(ucols, L - j0);
    if (threadIdx.x < nj) {
      const int ri = b * L + i, j = j0 + threadIdx.x;
      pk[threadIdx.x] = pack_pair(load_row(ids, BL, ri), load_row(ids, BL, b * L + j), bond[(size_t)ri * L + j], rmax, smax);
    }
    const float4 lf = ((const float4*)(left + (size_t)bi * P))[lane];
    if (first) { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); first = false; }
    __syncthreads();
    const float4* T = (const float4*)sm;
    float4* obase = (float4*)(out + ((size_t)bi * L + j0) * P);
    for (int jj = warp * UNR; jj < nj; jj += TPI_FWD_WARPS * UNR) {
      float4 rt[UNR];
#pragma unroll
      for (int q = 0; q < UNR; q++) rt[q] = ((const float4*)(right + ((size_t)b * L + j0 + min(jj + q, nj - 1)) * P))[lane];
      float4 o[UNR];
#pragma unroll
      for (int q = 0; q < UNR; q++) {
        const uint32_t w = pk[min(jj + q, nj - 1)];
        o[q] = lf + rt[q];
        o[q] += T[(w & 255) * 32 + lane];
        o[q] += T[((w >> 8) & 255) * 32 + lane];
        o[q] += T[((w >> 16) & 255) * 32 + lane];
        if ((w >> 24) & 1) o[q] += T[(nrel - 1) * 32 + lane];
        o[q] += T[(nrel + ((w >> 25) & 1)) * 32 + lane];
      }
#pragma unroll
      for (int q = 0; q < UNR; q++)
        if (jj + q < nj) obase[(size_t)(jj + q) * 32 + lane] = o[q];
    }
    __syncthreads();                                                // pk is rewritten by the next unit
  }
}
#else
__global__ void __launch_bounds__(32 * TPI_FWD_WARPS)
tpi_fwd(const float* __restrict__ left, const float* __restrict__ right, const float* __restrict__ tbl, const int* __restrict__ ids,
        const uint8_t* __restrict__ bond, float* __restrict__ out, int B, int L, int rmax, int smax, int R) {
  extern __shared__ __align__(128) float sm[];
  const int nrel = 2 * (2 * rmax + 2) + (2 * smax + 2) + 1, nbin = nrel + 2;
  uint32_t* pk = (uint32_t*)(sm + nbin * P);                        // [R][L]
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, BL = B * L;
  const int tiles = (L + R - 1) / R;
  const int b = blockIdx.x / tiles, i0 = (blockIdx.x % tiles) * R;
  const int nrows = min(R, L - i0);
  for (int c = threadIdx.x; c < nbin * P / 4; c += blockDim.x) {     // the table, 16 bytes per copy, in flight during the packing
    const unsigned d = (unsigned)__cvta_generic_to_shared(sm + 4 * c);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(tbl + 4 * c));
  }
  asm volatile("cp.async.commit_group;\n" ::);
  pack_rows(pk, ids, bond, BL, L, b, i0, nrows, rmax, smax);
  float4 lf[RMAX];
#pragma unroll
  for (int r = 0; r < RMAX; r++) lf[r] = r < nrows ? ((const float4*)(left + ((size_t)b * L + i0 + r) * P))[lane] : make_float4(0, 0, 0, 0);
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  __syncthreads();
  const float4* T = (const float4*)sm;
  float4* obase = (float4*)(out + ((size_t)b * L + i0) * L * P);
  for (int j0 = warp * UNR; j0 < L; j0 += TPI_FWD_WARPS * UNR) {
    float4 rt[UNR];
#pragma unroll
    for (int u = 0; u < UNR; u++) rt[u] = ((const float4*)(right + ((size_t)b * L + min(j0 + u, L - 1)) * P))[lane];
#pragma unroll
    for (int r = 0; r < RMAX; r++) if (r < nrows) {
      float4 o[UNR];
#pragma unroll
      for (int u = 0; u < UNR; u++) {
        const int j = min(j0 + u, L - 1);
        const uint32_t w = pk[r * L + j];
        o[u] = lf[r] + rt[u];
        o[u] += T[(w & 255) * 32 + lane];
        o[u] += T[((w >> 8) & 255) * 32 + lane];
        o[u] += T[((w >> 16) & 255) * 32 + lane];
        if ((w >> 24) & 1) o[u] += T[(nrel - 1) * 32 + lane];
        o[u] += T[(nrel + ((w >> 25) & 1)) * 32 + lane];
      }
#pragma unroll
      for (int u = 0; u < UNR; u++)
        if (j0 + u < L) __stcs(obase + ((size_t)r * L + j0 + u) * 32 + lane, o[u]);
    }
  }
}
#endif
