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

#define tpi_fwd(...) tpi_fwd_h100(__VA_ARGS__)   // the definition only: the launches name it without a following '('
#define volatile(...) volatile("")               // the H100 forward's mbarrier / bulk-copy statements (the only inline asm of the file)
#include "../sm90/token_pair_init.cu"
#undef volatile
#undef tpi_fwd
static_assert(TPI_FWD_WARPS == NWARP, "the forward launch is 32 * NWARP threads");

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
  for (int r = 0; r < RMAX; r++) lf[r] = r < nrows ? ((const float4*)(left + (size_t)(b * L + i0 + r) * P))[lane] : make_float4(0, 0, 0, 0);
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  __syncthreads();
  const float4* T = (const float4*)sm;
  float4* obase = (float4*)(out + ((size_t)(b * L + i0) * L) * P);
  for (int j0 = warp * UNR; j0 < L; j0 += TPI_FWD_WARPS * UNR) {
    float4 rt[UNR];
#pragma unroll
    for (int u = 0; u < UNR; u++) rt[u] = ((const float4*)(right + (size_t)(b * L + min(j0 + u, L - 1)) * P))[lane];
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
