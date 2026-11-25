// scatter.cuh -- the two reductions of G1 (the gradient of the first product's output) of the edge tail backward on the A100 (sm_80):
//
//   grad_query[g]      = sum of the K consecutive rows of group g                         (the query block is broadcast over the neighbour axis)
//   grad_neighbor[n]   = sum of the rows r with idx[r] == n                               (the gathered table's gradient: a scatter-add)
//
// The scatter-add is a segmented reduction over the reverse graph (CSR: counts -> scan -> fill), one warp per node, instead of one fp32 atomic per element (about 50 G adds / s on this
// card whatever the contention).  The order of a node's rows inside its segment is the order the fill kernel's counter hands out (not fixed from run to run), so the fp32 sums are
// not bitwise reproducible; they are rounded to bf16 once at the end.
#pragma once
#include "common.cuh"

namespace me80 {

__global__ void csr_count_kernel(const int64_t* __restrict__ idx, int* __restrict__ cnt, int rows) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < rows) atomicAdd(&cnt[(int)idx[r]], 1);
}

// exclusive scan of cnt[0, nodes) into offs[0, nodes], one block of 1024 threads, each owning a contiguous chunk; warp shuffles plus 128 bytes of shared memory: the kernel runs beside the
// weight-gradient kernel, whose CTAs hold 97 of the SM's 100 KB shared-memory carve-out, so a CTA that needs more than a few KB (the block-wide scan through 4 KB of shared memory did) cannot
// start until they are gone
__global__ void __launch_bounds__(1024) csr_scan_kernel(const int* __restrict__ cnt, int* __restrict__ offs, int nodes) {
  __shared__ int wsum[32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, per = (nodes + 1023) / 1024, lo = tid * per, hi = min(nodes, lo + per);
  int s = 0;
  for (int i = lo; i < hi; ++i) s += cnt[i];
  int v = s;                                                                  // inclusive scan inside the warp
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int u = __shfl_up_sync(0xffffffffu, v, d);
    if (lane >= d) v += u;
  }
  if (lane == 31) wsum[warp] = v;
  __syncthreads();
  if (warp == 0) {                                                            // inclusive scan of the 32 warp sums
    int w = wsum[lane];
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int u = __shfl_up_sync(0xffffffffu, w, d);
      if (lane >= d) w += u;
    }
    wsum[lane] = w;
  }
  __syncthreads();
  int base = v - s + (warp > 0 ? wsum[warp - 1] : 0);                         // exclusive prefix of this thread's chunk
  for (int i = lo; i < hi; ++i) { offs[i] = base; base += cnt[i]; }
  if (tid == 1023) offs[nodes] = wsum[31];
}

__global__ void csr_fill_kernel(const int64_t* __restrict__ idx, const int* __restrict__ offs, int* __restrict__ cursor, int* __restrict__ perm, int rows) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const int n = (int)idx[r];
  perm[offs[n] + atomicAdd(&cursor[n], 1)] = r;
}

DEVI uint2 ldg64(const void* p) { return __ldg(reinterpret_cast<const uint2*>(p)); }

DEVI void accum4(float (&acc)[4], uint2 v) {
  acc[0] += bf16lo(v.x); acc[1] += bf16hi(v.x); acc[2] += bf16lo(v.y); acc[3] += bf16hi(v.y);
}

// one warp per node, 4 warps per CTA (128 threads: the CTA is small enough to share an SM with a weight-gradient CTA, which these kernels run beside): lane l owns the 4 channels 4 l .. 4 l + 3
// (an 8-byte load per row, 256 contiguous bytes per warp).  The warp reads the node's row list 32 entries at a time (one per
// lane) and takes the rows by shuffle with 16 row loads in flight, so a node costs two dependent memory latencies per 32 rows (the list, then the rows) instead of two per 4.
__global__ void __launch_bounds__(128) nbr_reduce_kernel(const __nv_bfloat16* __restrict__ g1, const int* __restrict__ offs, const int* __restrict__ perm,
                                                          __nv_bfloat16* __restrict__ out, int nodes) {
  const int node = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (node >= nodes) return;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  const int e0 = offs[node], e1 = offs[node + 1];
  for (int base = e0; base < e1; base += 32) {
    const int n = min(32, e1 - base);
    const int mine = lane < n ? perm[base + lane] : 0;
    for (int k = 0; k < n; k += 16) {
      uint2 v[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int r = __shfl_sync(0xffffffffu, mine, (k + j) & 31);
        v[j] = (k + j < n) ? ldg64(g1 + (size_t)r * D + 4 * lane) : make_uint2(0u, 0u);
      }
#pragma unroll
      for (int j = 0; j < 16; ++j) accum4(acc, v[j]);
    }
  }
  *reinterpret_cast<uint2*>(out + (size_t)node * D + 4 * lane) = make_uint2(pack_bf16(acc[0], acc[1]), pack_bf16(acc[2], acc[3]));
}

// one warp per group of K consecutive rows
__global__ void __launch_bounds__(128) query_reduce_kernel(const __nv_bfloat16* __restrict__ g1, __nv_bfloat16* __restrict__ out, int groups, int K) {
  const int grp = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (grp >= groups) return;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  const __nv_bfloat16* base = g1 + (size_t)grp * K * D + 4 * lane;
  int k = 0;
  for (; k + 4 <= K; k += 4) {
    const uint2 v0 = ldg64(base + (size_t)k * D), v1 = ldg64(base + (size_t)(k + 1) * D), v2 = ldg64(base + (size_t)(k + 2) * D), v3 = ldg64(base + (size_t)(k + 3) * D);
    accum4(acc, v0); accum4(acc, v1); accum4(acc, v2); accum4(acc, v3);
  }
  for (; k < K; ++k) accum4(acc, ldg64(base + (size_t)k * D));
  *reinterpret_cast<uint2*>(out + (size_t)grp * D + 4 * lane) = make_uint2(pack_bf16(acc[0], acc[1]), pack_bf16(acc[2], acc[3]));
}

}  // namespace me80
