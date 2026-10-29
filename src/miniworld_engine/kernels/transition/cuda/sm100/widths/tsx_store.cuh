// tsx_store.cuh -- row-segment stores for the small-M Transition kernels (tsx_*.cu), sm_100a.
// The epilogues hold one row per thread (tcgen05.ld 32x32b); a warp storing its rows' 16-byte chunks directly writes 32 rows x 16 B
// per instruction, which keeps the LSU busy for ~1 us per 32 KB tile (measured on the gate, round sx3). store_t first transposes the
// chunks inside groups of C = 2 / 4 / 8 lanes (lane k ends up with chunk k of the group's C rows; tbwd_d384.cu's quad transpose,
// generalised), so each store writes 16 rows x 32 B, 8 x 64 B or 4 x 128 B.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "sm100.cuh"

namespace s100 {
DEVI void xchg16(uint4& keep, uint4& give, bool up, int m) {
  uint4 t = up ? keep : give;
  t.x = __shfl_xor_sync(0xffffffffu, t.x, m); t.y = __shfl_xor_sync(0xffffffffu, t.y, m);
  t.z = __shfl_xor_sync(0xffffffffu, t.z, m); t.w = __shfl_xor_sync(0xffffffffu, t.w, m);
  if (up) keep = t; else give = t;
}
// v: this thread's row segment, N 32-bit words (N = 8, 16, 32: 32, 64, 128 B). p: the group's first row at this lane's chunk, i.e.
// row (r & ~(C - 1)), element offset (lane & (C - 1)) * 16 B; ld: the row stride in elements of T.
template <int N, typename T> DEVI void store_t(T* p, size_t ld, const uint32_t (&v)[N], int lane) {
  static_assert(N == 8 || N == 16 || N == 32, "32-, 64- or 128-byte row segments");
  constexpr int C = N / 4;
  uint4 x[C];
#pragma unroll
  for (int c = 0; c < C; ++c) x[c] = make_uint4(v[4 * c], v[4 * c + 1], v[4 * c + 2], v[4 * c + 3]);
  const int k = lane & (C - 1);
#pragma unroll
  for (int m = 1; m < C; m <<= 1)                              // swap bit m of (lane, chunk): a butterfly transpose
#pragma unroll
    for (int c = 0; c < C; ++c)
      if (!(c & m)) xchg16(x[c], x[c | m], (k & m) != 0, m);
#pragma unroll
  for (int t = 0; t < C; ++t) stg128(p + (size_t)t * ld, x[t]);
}
// the element offset of this lane's first store: (group's first row) * ld + its chunk
template <int N> DEVI size_t store_t_off(int r, int lane, size_t ld, int elem_bytes) {
  constexpr int C = N / 4;
  return (size_t)(r & ~(C - 1)) * ld + (size_t)((lane & (C - 1)) * 16 / elem_bytes);
}

// distributed shared memory stores that complete on the destination CTA's mbarrier (complete_tx): the receiver waits on its own
// barrier, the sender needs no release (the cluster-scope releases of barrier.cluster / mbarrier.arrive.release cost ~1 us each in
// these kernels, measured with globaltimer stamps). addr / bar: this CTA's shared addresses, mapped to CTA `rank`.
DEVI uint32_t dsm_map(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
DEVI void st_async_v4(uint32_t addr, uint32_t bar, uint32_t rank, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
               :: "r"(dsm_map(addr, rank)), "r"(a), "r"(b), "r"(c), "r"(d), "r"(dsm_map(bar, rank)) : "memory");
}
DEVI void st_async_v2f(uint32_t addr, uint32_t bar, uint32_t rank, float a, float b) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];"
               :: "r"(dsm_map(addr, rank)), "f"(a), "f"(b), "r"(dsm_map(bar, rank)) : "memory");
}
// byte offset of 16-byte chunk L of a row-major buffer, swizzled within 128-byte lines (chunk position ^ line & 7) so that a warp
// accessing the same chunk of 32 consecutive rows spreads over the banks
DEVI uint32_t swz16(uint32_t L) { return ((L >> 3) << 7) | (((L & 7u) ^ ((L >> 3) & 7u)) << 4); }
}  // namespace s100
